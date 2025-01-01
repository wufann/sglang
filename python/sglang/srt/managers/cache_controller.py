import torch
import threading
import time
import queue
from queue import PriorityQueue, Queue
from typing import Optional
import logging


from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPoolHost,
    MemoryStateInt,
)

logger = logging.getLogger(__name__)


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.data = None
        self.node_id = [node_id]

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        self.priority = priority if priority is not None else self.id

    def merge(self, other: "CacheOperation") -> None:
        self.host_indices = torch.cat([self.host_indices, other.host_indices])
        self.device_indices = torch.cat([self.device_indices, other.device_indices])
        self.priority = min(self.priority, other.priority)
        self.node_id.extend(other.node_id)

    def __lt__(self, other: "CacheOperation"):
        return self.priority < other.priority


class TransferBuffer:
    def __init__(self, buffer_count: int = 2, buffer_size: int = 1000) -> None:
        self.buffers = Queue()
        self.buffer_count = buffer_count
        # todo: adjust the buffer size based on throughput profile of the system
        self.buffer_size = buffer_size
        self.timeout = 1

        # todo: alternative fixed size buffer implementation

    def full(self) -> bool:
        return self.buffers.qsize() >= self.buffer_count

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, buffer) -> None:
        self.buffers.put(buffer)

    def get(self) -> Optional[CacheOperation]:
        try:
            return self.buffers.get(timeout=self.timeout)
        except queue.Empty:
            return None


class RevokableQueue(PriorityQueue):
    def revoke(self, host_indices):
        num_tokens = len(host_indices)
        with self.mutex:
            for i, item in enumerate(self.queue):
                if torch.equal(item.host_indices[-num_tokens:], host_indices):
                    if len(item.host_indices) == num_tokens:
                        del self.queue[i]
                        break
                    else:
                        self.queue[i].host_indices = item.host_indices[:-num_tokens]
                        self.queue[i].device_indices = item.device_indices[:-num_tokens]
                        break
            else:
                return False
        return True


class HiCacheController:

    def __init__(
        self,
        mem_pool_device: MHATokenToKVPool,
        mem_pool_host: MLATokenToKVPoolHost,
        write_policy: str = "write_through_selective",
    ):

        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy

        if write_policy in ["write_through", "write_through_selective", "write_back"]:
            self.write_queue = PriorityQueue()
        elif write_policy == "write_through_revokable":
            # todo: deprecate
            self.write_queue = RevokableQueue()
        else:
            raise ValueError(f"Invalid write policy: {write_policy}")
        self.load_queue = PriorityQueue()
        self.ack_write_queue = Queue()
        self.ack_load_queue = Queue()

        self.write_buffer = TransferBuffer()
        self.load_buffer = TransferBuffer()

        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

        self.write_thread = threading.Thread(
            target=self.write_thread_func_buffer, daemon=True
        )
        self.load_thread = threading.Thread(
            target=self.load_thread_func_buffer, daemon=True
        )
        self.write_thread.start()
        self.load_thread.start()

    def write_through(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
    ):
        """
        Optimistically write through calculated KV caches to host memory.
        Certain KV caches may not be written through due to runtime memory requirements.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.put(
            CacheOperation(host_indices, device_indices, priority, node_id)
        )
        self.mem_pool_host.protect_write(host_indices)
        return host_indices

    def write_back(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
    ):
        """
        Write back KV caches from device memory to host memory.
        """
        # todo: deduplicate
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.put(
            CacheOperation(host_indices, device_indices, priority, node_id)
        )
        self.mem_pool_host.protect_write(host_indices)
        return host_indices

    def load_back(self, host_indices, priority: Optional[int] = None, node_id: int = 0):
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.put(
            CacheOperation(host_indices, device_indices, priority, node_id)
        )
        self.mem_pool_host.protect_load(host_indices)
        return device_indices

    def write_thread_func_direct(self):
        """
        Directly write through KV caches to host memory without buffering.
        """
        with torch.cuda.stream(self.write_stream):
            while True:
                if self.write_queue.empty():
                    time.sleep(0.1)
                try:
                    operation = self.write_queue.get(timeout=1)
                    operation.data = self.mem_pool_device.get_flat_data(
                        operation.device_indices
                    )
                    self.mem_pool_host.transfer(operation.host_indices, operation.data)
                    self.mem_pool_host.complete_io(operation.host_indices)
                    for node_id in operation.node_id:
                        self.ack_write_queue.put(node_id)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(e)

    def load_thread_func_direct(self):
        """
        Directly load KV caches from host memory to device memory without buffering.
        """
        with torch.cuda.stream(self.load_stream):
            while True:
                if self.load_queue.empty():
                    time.sleep(0.1)
                try:
                    operation = self.load_queue.get(timeout=1)
                    operation.data = self.mem_pool_host.get_flat_data(
                        operation.host_indices
                    )
                    self.mem_pool_device.transfer(
                        operation.device_indices, operation.data
                    )
                    self.mem_pool_host.complete_io(operation.host_indices)
                    for node_id in operation.node_id:
                        self.ack_load_queue.put(node_id)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(e)

    def write_aux_func(self, no_wait=False):
        buffer = None
        while True:
            if self.write_queue.empty() or self.write_buffer.full():
                time.sleep(0.1)
            try:
                operation = self.write_queue.get(timeout=1)
                if buffer is None:
                    buffer = operation
                else:
                    buffer.merge(operation)
                if (
                    no_wait
                    or len(buffer.host_indices) >= self.write_buffer.buffer_size
                    or self.write_queue.empty()
                    or self.write_buffer.empty()
                ):
                    if len(buffer.host_indices) == 0:
                        buffer = None
                        continue
                    buffer.device_indices.to(
                        device=self.mem_pool_device.device, copy=True
                    )
                    buffer.data = self.mem_pool_device.get_flat_data(
                        buffer.device_indices
                    ).contiguous()
                    self.write_buffer.put(buffer)
                    buffer = None
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(e)

    def load_aux_func(self):
        buffer = None
        while True:
            if self.load_queue.empty() or self.load_buffer.full():
                time.sleep(0.1)
            try:
                operation = self.load_queue.get(timeout=1)
                if buffer is None:
                    buffer = operation
                else:
                    buffer.merge(operation)
                if (
                    len(buffer.host_indices) >= self.load_buffer.buffer_size
                    or self.load_queue.empty()
                    or self.load_buffer.empty()
                ):
                    buffer.data = (
                        self.mem_pool_host.get_flat_data(buffer.host_indices)
                        .contiguous()
                        .pin_memory()
                    )
                    buffer.device_indices.to(
                        device=self.mem_pool_device.device, copy=True
                    )
                    self.load_buffer.put(buffer)
                    buffer = None
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(e)

    def write_thread_func_(self):
        while True:
            operation = self.write_buffer.get()
            if operation is None:
                continue
            self.mem_pool_host.transfer(operation.host_indices, operation.data)
            self.mem_pool_host.complete_io(operation.host_indices)
            for node_id in operation.node_id:
                self.ack_write_queue.put(node_id)

    def load_thread_func_(self):
        while True:
            operation = self.load_buffer.get()
            if operation is None:
                continue
            self.mem_pool_device.transfer(operation.device_indices, operation.data)
            self.mem_pool_host.complete_io(operation.host_indices)
            for node_id in operation.node_id:
                self.ack_load_queue.put(node_id)

    def write_thread_func_buffer(self):
        aux_thread = threading.Thread(target=self.write_aux_func, daemon=True)
        aux_thread.start()
        with torch.cuda.stream(self.write_stream):
            self.write_thread_func_()

    def load_thread_func_buffer(self):
        aux_thread = threading.Thread(target=self.load_aux_func, daemon=True)
        aux_thread.start()
        with torch.cuda.stream(self.load_stream):
            self.load_thread_func_()

    def evict_device(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor
    ) -> int:
        with self.mem_pool_host.lock:
            host_mem_state = self.mem_pool_host.get_state(host_indices)

            if host_mem_state == MemoryStateInt.PROTECTED:
                assert (
                    self.write_policy == "write_through_revokable"
                ), "Inconsistent state."
                return 0
            elif host_mem_state == MemoryStateInt.RESERVED:
                assert (
                    self.write_policy == "write_through_revokable"
                ), "Inconsistent state."
                # pending write through, revoke to free device memory
                if self.write_queue.revoke(host_indices):
                    self.mem_pool_host.free(host_indices)
                    self.mem_pool_device.free(device_indices)
                else:
                    # failed to revoke, wait for the write operation to complete instead
                    return 0
                return len(device_indices)
            elif host_mem_state == MemoryStateInt.IDLE:
                raise ValueError("Inconsistent states.")
                # self.mem_pool_device.free(device_indices)
                # return len(device_indices)
            elif host_mem_state == MemoryStateInt.BACKUP:
                raise ValueError("Inconsistent states.")
            elif host_mem_state == MemoryStateInt.SYNCED:
                self.mem_pool_device.free(device_indices)
                self.mem_pool_host.update_backup(host_indices)
                return len(device_indices)
            else:
                raise ValueError(f"Invalid state: {host_mem_state}")

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        with self.mem_pool_host.lock:
            host_mem_state = self.mem_pool_host.get_state(host_indices)

            if backup_only and host_mem_state != MemoryStateInt.BACKUP:
                raise ValueError("Inconsistent states.")
                # it is recommended to evict host-only KV caches
                return 0

            # todo: a more versatile protocol

            if host_mem_state == MemoryStateInt.BACKUP:
                self.mem_pool_host.free(host_indices)
                return len(host_indices)
            elif host_mem_state == MemoryStateInt.IDLE:
                raise ValueError("Double free detected.")
            elif host_mem_state == MemoryStateInt.RESERVED:
                # not supposed to compete with pending write through
                raise ValueError("Inconsistent states.")
            elif host_mem_state == MemoryStateInt.PROTECTED:
                raise ValueError("Inconsistent states.")
            elif host_mem_state == MemoryStateInt.SYNCED:
                # todo: different behavior for different write policies
                raise ValueError("Inconsistent states.")
                self.mem_pool_host.free(host_indices)
                return len(host_indices)
            else:
                raise ValueError(f"Invalid state: {host_mem_state}")


if __name__ == "__main__":
    pass

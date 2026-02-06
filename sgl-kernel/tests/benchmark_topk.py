"""
Benchmark script for top-k kernels with comprehensive performance measurements.

Usage:
    python benchmark_topk.py [--mode {all,decode,extend,target_verify}] [--warmup N] [--iterations N]
"""

import argparse
import time
from typing import Optional, Tuple

import torch
from sgl_kernel import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)

# Constants
MAX_SEQ_LEN = 128000 
TOPK = 2048


def benchmark_kernel(
    kernel_fn,
    kernel_name: str,
    args: dict,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
) -> Tuple[float, float, float]:
    """
    Benchmark a kernel function.
    
    Returns:
        Tuple of (mean_time_ms, std_time_ms, throughput_tokens_per_sec)
    """
    # Warmup
    for _ in range(warmup_iterations):
        kernel_fn(**args)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(benchmark_iterations):
        start_event.record()
        kernel_fn(**args)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # Calculate throughput (tokens/sec)
    # For topk kernels, we process batch_size * seq_len scores
    if "score" in args:
        batch_size = args["score"].shape[0]
        seq_len = args["lengths"][0].item() if "lengths" in args else args["score"].shape[1]
        total_elements = batch_size * seq_len
        throughput = (total_elements / (mean_time / 1000.0))  # tokens per second
    else:
        throughput = 0.0
    
    return mean_time, std_time, throughput


def benchmark_topk_v2(
    bs: int,
    seq_len: int,
    has_row_starts: bool,
    warmup: int = 10,
    iterations: int = 100,
):
    """Benchmark fast_topk_v2 kernel."""
    torch.manual_seed(42)
    
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    
    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None
    
    args = {
        "score": score,
        "lengths": lengths,
        "topk": TOPK,
        "row_starts": row_starts,
    }
    
    mean_time, std_time, throughput = benchmark_kernel(
        fast_topk_v2,
        "fast_topk_v2",
        args,
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
    )
    
    return {
        "kernel": "fast_topk_v2",
        "bs": bs,
        "seq_len": seq_len,
        "has_row_starts": has_row_starts,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "throughput_Mtokens_per_sec": throughput / 1e6,
    }


def benchmark_topk_transform(
    bs: int,
    seq_len: int,
    mode: str,
    warmup: int = 10,
    iterations: int = 1000,
):
    """Benchmark fast_topk_transform_fused kernel."""
    torch.manual_seed(42)
    
    if mode == "decode":
        step = 1
    else:
        step = 4 if bs % 4 == 0 else 1
    
    num_tokens = bs
    actual_bs = bs // step
    
    if mode == "extend":
        row_starts = torch.randint(0, 2048, (actual_bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None
    
    score = torch.randn(actual_bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((actual_bs,), seq_len, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(
        0, num_tokens + 1, step=step, dtype=torch.int32, device="cuda"
    )
    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
    src_page_table = src_page_table.unsqueeze(0).expand(actual_bs, -1)
    
    args = {
        "score": score,
        "lengths": lengths,
        "page_table_size_1": src_page_table,
        "cu_seqlens_q": cu_seqlens_q,
        "topk": TOPK,
        "row_starts": row_starts,
    }
    
    mean_time, std_time, throughput = benchmark_kernel(
        fast_topk_transform_fused,
        "fast_topk_transform_fused",
        args,
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
    )
    
    return {
        "kernel": "fast_topk_transform_fused",
        "bs": bs,
        "actual_bs": actual_bs,
        "seq_len": seq_len,
        "mode": mode,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "throughput_Mtokens_per_sec": throughput / 1e6,
    }


def benchmark_topk_transform_ragged(
    bs: int,
    seq_len: int,
    has_row_starts: bool,
    warmup: int = 10,
    iterations: int = 100,
):
    """Benchmark fast_topk_transform_ragged_fused kernel."""
    torch.manual_seed(42)
    
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    
    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None
    
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device="cuda"
    )
    
    args = {
        "score": score,
        "lengths": lengths,
        "topk_indices_offset": topk_indices_offset,
        "topk": TOPK,
        "row_starts": row_starts,
    }
    
    mean_time, std_time, throughput = benchmark_kernel(
        fast_topk_transform_ragged_fused,
        "fast_topk_transform_ragged_fused",
        args,
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
    )
    
    return {
        "kernel": "fast_topk_transform_ragged_fused",
        "bs": bs,
        "seq_len": seq_len,
        "has_row_starts": has_row_starts,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "throughput_Mtokens_per_sec": throughput / 1e6,
    }


def print_results_table(results: list):
    """Print benchmark results in a formatted table."""
    if not results:
        return
    
    # Group by kernel type
    kernels = {}
    for result in results:
        kernel_name = result["kernel"]
        if kernel_name not in kernels:
            kernels[kernel_name] = []
        kernels[kernel_name].append(result)
    
    for kernel_name, kernel_results in kernels.items():
        print(f"\n{'='*100}")
        print(f"Kernel: {kernel_name}")
        print('='*100)
        
        if kernel_name == "fast_topk_v2":
            print(f"{'BS':<8} {'SeqLen':<10} {'RowStarts':<12} {'Time(ms)':<12} {'Std(ms)':<12} {'Throughput(M tok/s)':<20}")
            print('-'*100)
            for r in kernel_results:
                print(f"{r['bs']:<8} {r['seq_len']:<10} {str(r['has_row_starts']):<12} "
                      f"{r['mean_time_ms']:<12.3f} {r['std_time_ms']:<12.3f} {r['throughput_Mtokens_per_sec']:<20.2f}")
        
        elif kernel_name == "fast_topk_transform_fused":
            print(f"{'BS':<8} {'ActualBS':<10} {'SeqLen':<10} {'Mode':<15} {'Time(ms)':<12} {'Std(ms)':<12} {'Throughput(M tok/s)':<20}")
            print('-'*100)
            for r in kernel_results:
                print(f"{r['bs']:<8} {r['actual_bs']:<10} {r['seq_len']:<10} {r['mode']:<15} "
                      f"{r['mean_time_ms']:<12.3f} {r['std_time_ms']:<12.3f} {r['throughput_Mtokens_per_sec']:<20.2f}")
        
        elif kernel_name == "fast_topk_transform_ragged_fused":
            print(f"{'BS':<8} {'SeqLen':<10} {'RowStarts':<12} {'Time(ms)':<12} {'Std(ms)':<12} {'Throughput(M tok/s)':<20}")
            print('-'*100)
            for r in kernel_results:
                print(f"{r['bs']:<8} {r['seq_len']:<10} {str(r['has_row_starts']):<12} "
                      f"{r['mean_time_ms']:<12.3f} {r['std_time_ms']:<12.3f} {r['throughput_Mtokens_per_sec']:<20.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark top-k kernels")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "topk_v2", "transform_decode", "transform_extend", "transform_target_verify", "transform_ragged"],
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 24],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[8, 128, 1024, 3072, 4096, 8192, 16384, 32768, 65536,90000,128000],
        help="Sequence lengths to test",
    )
    
    args = parser.parse_args()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"\nBenchmark configuration:")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Benchmark iterations: {args.iterations}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Sequence lengths: {args.seq_lens}")
    print(f"  Top-k: {TOPK}")
    
    results = []
    
    # Benchmark fast_topk_v2
    if args.mode in ["all", "topk_v2"]:
        print("\n" + "="*100)
        print("Benchmarking fast_topk_v2...")
        print("="*100)
        for bs in args.batch_sizes:
            for seq_len in args.seq_lens:
                for has_row_starts in [False, True]:
                    print(f"Running: bs={bs}, seq_len={seq_len}, has_row_starts={has_row_starts}")
                    result = benchmark_topk_v2(
                        bs, seq_len, has_row_starts, args.warmup, args.iterations
                    )
                    results.append(result)
    
    # Benchmark fast_topk_transform_fused (decode mode)
    if args.mode in ["all", "transform_decode"]:
        print("\n" + "="*100)
        print("Benchmarking fast_topk_transform_fused (decode mode)...")
        print("="*100)
        for bs in args.batch_sizes:
            for seq_len in args.seq_lens:
                print(f"Running: bs={bs}, seq_len={seq_len}, mode=decode")
                result = benchmark_topk_transform(
                    bs, seq_len, "decode", args.warmup, args.iterations
                )
                results.append(result)
    
    # # Benchmark fast_topk_transform_fused (extend mode)
    # if args.mode in ["all", "transform_extend"]:
    #     print("\n" + "="*100)
    #     print("Benchmarking fast_topk_transform_fused (extend mode)...")
    #     print("="*100)
    #     for bs in args.batch_sizes:
    #         for seq_len in args.seq_lens:
    #             print(f"Running: bs={bs}, seq_len={seq_len}, mode=extend")
    #             result = benchmark_topk_transform(
    #                 bs, seq_len, "extend", args.warmup, args.iterations
    #             )
    #             results.append(result)
    
    # # Benchmark fast_topk_transform_fused (target_verify mode)
    # if args.mode in ["all", "transform_target_verify"]:
    #     print("\n" + "="*100)
    #     print("Benchmarking fast_topk_transform_fused (target_verify mode)...")
    #     print("="*100)
    #     for bs in args.batch_sizes:
    #         for seq_len in args.seq_lens:
    #             print(f"Running: bs={bs}, seq_len={seq_len}, mode=target_verify")
    #             result = benchmark_topk_transform(
    #                 bs, seq_len, "target_verify", args.warmup, args.iterations
    #             )
    #             results.append(result)
    
    # # Benchmark fast_topk_transform_ragged_fused
    # if args.mode in ["all", "transform_ragged"]:
    #     print("\n" + "="*100)
    #     print("Benchmarking fast_topk_transform_ragged_fused...")
    #     print("="*100)
    #     for bs in args.batch_sizes:
    #         for seq_len in args.seq_lens:
    #             for has_row_starts in [False, True]:
    #                 print(f"Running: bs={bs}, seq_len={seq_len}, has_row_starts={has_row_starts}")
    #                 result = benchmark_topk_transform_ragged(
    #                     bs, seq_len, has_row_starts, args.warmup, args.iterations
    #                 )
    #                 results.append(result)
    
    # Print all results
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print_results_table(results)
    
    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    for kernel_name in set(r["kernel"] for r in results):
        kernel_results = [r for r in results if r["kernel"] == kernel_name]
        mean_times = [r["mean_time_ms"] for r in kernel_results]
        throughputs = [r["throughput_Mtokens_per_sec"] for r in kernel_results]
        
        print(f"\n{kernel_name}:")
        print(f"  Min time: {min(mean_times):.3f} ms")
        print(f"  Max time: {max(mean_times):.3f} ms")
        print(f"  Avg time: {sum(mean_times)/len(mean_times):.3f} ms")
        print(f"  Min throughput: {min(throughputs):.2f} M tokens/s")
        print(f"  Max throughput: {max(throughputs):.2f} M tokens/s")
        print(f"  Avg throughput: {sum(throughputs)/len(throughputs):.2f} M tokens/s")


if __name__ == "__main__":
    main()

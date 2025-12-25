from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from sglang.srt.utils import is_hip

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
}

BF16 = "bfloat16"
FP8 = "float8_e4m3"
FP32 = "float32"

_is_hip = is_hip()

@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def fp8_index_kernel(h: int, d: int, clear_accum=True):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 512
    blk_n2 = 128

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
            q_smem = T.alloc_shared((h, d), FP8)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), FP8)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=clear_accum,
                )
                # if i_b == 0 and i_m == 0 and i1_n == 0 and i2_n == 0:
                #     T.print(logits)

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return fp8_index_kernel_


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.

    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    if _is_hip:
        return fp8_index_kernel(q.shape[2], q.shape[3], False)(q, q_s, k, k_s)
    else:
        return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)

q_fp8_partial = (
    torch.randn(1, 3919, 64, 128, device="cuda", dtype=torch.bfloat16)
    .to(torch.float8_e4m3fn)
)

weights_partial = torch.randn(1, 3919, 64, device="cuda", dtype=torch.float32)
k_fp8 = (
    torch.randn(1, 3919, 128, device="cuda", dtype=torch.bfloat16)
    .to(torch.float8_e4m3fn)
)

k_scale = torch.randn(1, 3919, device="cuda", dtype=torch.float32)

index_score = fp8_index(
    q_fp8_partial,
    weights_partial,
    k_fp8,
    k_scale,
)
print(index_score)
print(index_score.shape)
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, cast

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import (
    dispatch_w8a8_block_fp8_linear,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.utils import get_bool_env_var, is_hip

__all__ = ["QuarkW8A8BlockFp8"]

_is_fp8_fnuz = is_fp8_fnuz()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight

    from sglang.srt.layers.quantization.fp8_utils import (
        _use_aiter_bpreshuffle_gfx95,
        aiter_w8a8_block_fp8_linear,
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )


class QuarkW8A8BlockFp8(QuarkLinearScheme):
    """Block-scale (e.g. 128x128) FP8 linear scheme for Quark mixed-quant models.

    Weight is static block-scale FP8; activation is dynamic per-1x(block_k)
    FP8 quantized at runtime. Mirrors the native fp8 Fp8LinearMethod block path
    and reuses w8a8_block_fp8_linear so the aiter/triton block GEMM is shared.
    """

    def __init__(
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        block_size = weight_config.get("block_size")
        if block_size is None:
            group_size = weight_config.get("group_size", 128)
            block_size = [group_size, group_size]
        self.weight_block_size = list(block_size)
        self.out_dtype = torch.get_default_dtype()
        self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.orig_dtype = params_dtype

        block_n, block_k = self.weight_block_size[0], self.weight_block_size[1]

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        scale = BlockQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition + block_n - 1) // block_n,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale_inv", scale)

        layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_fp8_fnuz:
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
            )
        else:
            weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data

        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = Parameter(weight_scale, requires_grad=False)
        layer.input_scale = None

        if (
            _use_aiter
            and _use_aiter_bpreshuffle_gfx95
            and self.w8a8_block_fp8_linear is aiter_w8a8_block_fp8_linear
        ):
            n, k = layer.weight.shape
            if not use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k):
                t = shuffle_weight(layer.weight, (16, 16))
                layer.weight = Parameter(t, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # With fused RMSNorm+quant, x is a (fp8_input, scale) tuple already
        # quantized upstream; otherwise it's a bf16/fp16 tensor quantized here.
        if isinstance(x, tuple):
            return self.w8a8_block_fp8_linear(
                input=x[0],
                weight=layer.weight,
                block_size=self.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=x[1],
                bias=bias,
            )
        return self.w8a8_block_fp8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
            bias=bias,
        )

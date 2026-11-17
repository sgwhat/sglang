"""Load-time block-FP8 quantization of a BF16 DeepSeek-V4 ``wo_a``.

The FP8 ``wo_a`` absorb GEMM (``SGLANG_OPT_FP8_WO_A_GEMM``) takes a 128x128
block-FP8 weight. A checkpoint that keeps ``wo_a`` unquantized gets one from
``quantize_bf16_wo_a_streaming`` under ``SGLANG_OPT_FP8_WO_A_ONLINE_QUANT``.
"""

import logging
from typing import Iterable, Tuple

import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config

logger = logging.getLogger(__name__)

_BLOCK = 128


def wo_a_online_quant_config() -> Fp8Config:
    """The block-FP8 config ``wo_a`` is built with when the checkpoint's
    quantization config leaves it unquantized."""
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[_BLOCK, _BLOCK],
    )


def quant_fp8_block128(
    weight: torch.Tensor, *, pow2_scale: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[N, K]`` weight to e4m3 plus ``[N/128, K/128]`` FP32 dequant scales."""
    n, k = weight.shape
    assert n % _BLOCK == 0 and k % _BLOCK == 0, f"{weight.shape} is not 128-aligned"
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    blocks = weight.float().view(n // _BLOCK, _BLOCK, k // _BLOCK, _BLOCK)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-10) / fp8_max
    if pow2_scale:
        # UE8M0 scales must be powers of two; round up so nothing saturates.
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (blocks / scale[:, None, :, None]).clamp(-fp8_max, fp8_max)
    return quantized.to(torch.float8_e4m3fn).view(n, k), scale.contiguous()


def quantize_bf16_wo_a_streaming(
    weights: Iterable[Tuple[str, torch.Tensor]], *, pow2_scale: bool
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Replace each BF16 checkpoint ``*.wo_a.weight`` with its FP8 weight and
    the ``*.wo_a.scale`` the FP8 checkpoint layout carries; pass the rest."""
    num_quantized = 0
    for name, tensor in weights:
        if name.endswith(".wo_a.weight") and tensor.dtype == torch.bfloat16:
            weight_fp8, scale = quant_fp8_block128(tensor, pow2_scale=pow2_scale)
            num_quantized += 1
            yield name, weight_fp8
            yield name.removesuffix(".weight") + ".scale", scale
            continue
        yield name, tensor
    if num_quantized:
        logger.info("Quantized %d BF16 wo_a weights to block FP8", num_quantized)

"""Load-time block-FP8 quantization of a BF16 DeepSeek-V4 ``wo_a``.

The FP8 ``wo_a`` absorb GEMM (``SGLANG_OPT_FP8_WO_A_GEMM``) takes a 128x128
block-FP8 weight. A checkpoint that keeps ``wo_a`` unquantized gets one from
``quantize_bf16_wo_a_streaming`` under ``SGLANG_OPT_FP8_WO_A_ONLINE_QUANT``.
"""

import logging
from typing import Iterable, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.utils import is_layer_skipped

logger = logging.getLogger(__name__)

_BLOCK = 128


def is_wo_a_skipped_in_quant_config(
    quant_config: Optional[QuantizationConfig],
    wo_a_prefix: str = "model.layers.0.self_attn.wo_a",
) -> bool:
    """Whether ``quant_config`` leaves ``wo_a`` unquantized via ``ignored_layers``."""
    if not isinstance(quant_config, Fp8Config) or not quant_config.ignored_layers:
        return False
    return is_layer_skipped(
        wo_a_prefix,
        quant_config.ignored_layers,
        fused_mapping=quant_config.packed_modules_mapping,
    )


def validate_uniform_wo_a_quant_config(
    quant_config: Optional[QuantizationConfig],
    num_hidden_layers: int,
    *,
    model_prefix: str = "model",
) -> None:
    """``DeepseekV4ForCausalLM`` tracks a single model-wide ``wo_a_fp8`` flag across
    weight loading and post-load scale setup, so mixed per-layer ``wo_a`` skipping is
    unsupported unless online quantization normalizes every BF16 ``wo_a`` to FP8."""
    if (
        not isinstance(quant_config, Fp8Config)
        or not quant_config.ignored_layers
        or num_hidden_layers <= 1
        or envs.SGLANG_OPT_FP8_WO_A_ONLINE_QUANT.get()
    ):
        return
    skipped = [
        is_wo_a_skipped_in_quant_config(
            quant_config, f"{model_prefix}.layers.{i}.self_attn.wo_a"
        )
        for i in range(num_hidden_layers)
    ]
    if any(skipped) and not all(skipped):
        raise ValueError(
            "DeepSeek V4 requires self_attn.wo_a quantization to be uniform across "
            "all decoder layers unless SGLANG_OPT_FP8_WO_A_ONLINE_QUANT=1."
        )


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

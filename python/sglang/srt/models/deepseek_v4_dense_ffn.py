"""Dense FFN layers on the DeepSeek-V4 runtime (``config.n_dense_layers``).

With ``--moe-dense-tp-size 1`` every rank holds the whole dense FFN and computes
only its own rows, with no collective, which is what attention DP and the MoE
all-to-all backends need. Otherwise the FFN is sharded over the TP group and
the layer's stage boundary completes its sum.
"""

from typing import List, Optional

from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.layer_boundary.layout import is_dense_ffn_fully_dp
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
from sglang.srt.runtime_context import get_parallel


def is_dense_ffn_layer(config, layer_id: int, *, is_nextn: bool) -> bool:
    # Not first_k_dense_replace: the V3 config this schema extends defaults it to 3.
    return not is_nextn and layer_id < config.n_dense_layers


def _unsupported_dense_ffn_reasons(config, *, is_cuda: bool) -> List[str]:
    parallel = get_parallel()
    fully_dp = is_dense_ffn_fully_dp()
    return [
        reason
        for bad, reason in (
            (not is_cuda, "a non-CUDA platform"),
            (
                config.hc_pre_from_prev_sublayer,
                "hyper-connections that pre-mix from the previous sublayer",
            ),
            (
                is_dsa_enable_prefill_cp() or parallel.attn_cp_size > 1,
                "attention context parallelism",
            ),
            (
                parallel.enable_attn_tp_input_scattered,
                "input-scattered attention TP",
            ),
            # mHC streams stay on the attention's rows; a local FFN would
            # compute on this rank's attention-TP slice of them.
            (
                fully_dp and parallel.attn_tp_size > 1,
                "--moe-dense-tp-size 1 with an attention TP size above 1",
            ),
            (
                not fully_dp
                and (parallel.attn_dp_size > 1 or not get_moe_a2a_backend().is_none()),
                "DP attention or a MoE all-to-all backend without "
                "--moe-dense-tp-size 1",
            ),
        )
        if bad
    ]


def build_dense_mlp(
    config,
    quant_config: Optional[QuantizationConfig],
    prefix: str,
    *,
    is_cuda: bool,
) -> DeepseekV2MLP:
    unsupported = _unsupported_dense_ffn_reasons(config, is_cuda=is_cuda)
    if unsupported:
        raise NotImplementedError(
            "DeepSeek V4 dense FFN layers (n_dense_layers > 0) do not support "
            + ", ".join(unsupported)
        )
    replicated = is_dense_ffn_fully_dp()
    return DeepseekV2MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        quant_config=quant_config,
        reduce_results=False,
        prefix=prefix,
        swiglu_limit=getattr(config, "swiglu_limit", None),
        # A TP shard leaves its sum to the boundary, so it keeps the unfused down.
        allow_fused_down=replicated,
        parallel_group="replicated" if replicated else "tp",
    )

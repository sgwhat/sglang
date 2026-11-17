# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Identity mHC (``hc_residual_mode=identity``) for the DeepSeek-V4 runtime.

The streams keep the Sinkhorn variant's pre/post gates but are never mixed with
each other (comb is the identity), so ``hc_*_fn`` holds only the ``[pre; post]``
rows and two scales. The eager functions define the semantics; the Triton
kernels in ``sglang.kernels.ops.layernorm.ihc`` (the default backend) replace
them op by op. Kernel errors propagate.
"""

from typing import Callable, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs

# Same multiplier as the Sinkhorn path's post gate (MHC_POST_MULT_VALUE).
IDENTITY_POST_MULT = 2.0
# DModel adds no eps to the post gate; Hunyuan-V4 adds hc_eps there.
IDENTITY_POST_EPS = 0.0

_RESIDUAL_MODES = ("sinkhorn", "identity")
_BACKENDS = ("triton", "eager")


def hc_pre_identity_torch(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-mix of streams ``x [T, hc, H]`` with gate math in fp32; returns the
    collapsed sublayer input ``[T, H]`` and the post gates ``[T, hc]``."""
    hc_mult = x.shape[-2]
    x_flat = x.flatten(-2).float()
    mixes = F.linear(x_flat, hc_fn.float()) * torch.rsqrt(
        x_flat.square().mean(-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_eps
    post = IDENTITY_POST_MULT * torch.sigmoid(
        mixes[:, hc_mult:] * hc_scale[1] + hc_base[hc_mult:]
    )
    y = (pre.unsqueeze(-1) * x.float()).sum(dim=-2)
    return y.to(x.dtype), post


def hc_post_identity_torch(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    """``R'[i] = R[i] + post[i] * x``, accumulated in fp32."""
    out = residual.float() + post.unsqueeze(-1) * x.float().unsqueeze(-2)
    return out.to(residual.dtype)


class IdentityMhcConfig(msgspec.Struct, frozen=True):
    backend: str
    # Rows from which the Triton pre runs the tiled stage 1; None never does.
    pre_tiled_min_tokens: Optional[int] = None
    pre_bf16x3: bool = False

    @property
    def fused(self) -> bool:
        return self.backend == "triton"


def is_identity_residual(config) -> bool:
    mode = config.hc_residual_mode
    if mode not in _RESIDUAL_MODES:
        raise ValueError(
            f"hc_residual_mode must be one of {_RESIDUAL_MODES}, got {mode!r}"
        )
    return mode == "identity"


def check_identity_residual_supported(config, *, is_cuda: bool, is_nextn: bool):
    unsupported = [
        reason
        for bad, reason in (
            (not is_cuda, "a non-CUDA platform"),
            (is_nextn, "draft layers (NextN, DSpark)"),
            (
                config.hc_pre_from_prev_sublayer,
                "hyper-connections that pre-mix from the previous sublayer",
            ),
        )
        if bad
    ]
    if unsupported:
        raise NotImplementedError(
            "DeepSeek V4 identity hyper-connections (hc_residual_mode=identity) "
            "do not support " + ", ".join(unsupported)
        )


def load_identity_mhc_config() -> IdentityMhcConfig:
    backend = envs.SGLANG_OPT_IHC_BACKEND.get()
    if backend not in _BACKENDS:
        raise ValueError(
            f"SGLANG_OPT_IHC_BACKEND must be one of {_BACKENDS}, got {backend!r}"
        )
    return IdentityMhcConfig(backend=backend)


def identity_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm: Optional[torch.nn.Module],
    rms_eps: float,
    hc_eps: float,
    config: IdentityMhcConfig,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Returns ``(y, post, norm_fused)``; ``y`` has gone through ``norm`` iff
    ``norm_fused``. The Triton path folds the norm after the readout's rounding."""
    if not config.fused or x.shape[0] == 0:
        y, post = hc_pre_identity_torch(
            x, hc_fn, hc_scale, hc_base, rms_eps=rms_eps, hc_eps=hc_eps
        )
        return y, post, False
    from sglang.kernels.ops.layernorm.ihc import ihc_pre

    y, post = ihc_pre(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        post_mult=IDENTITY_POST_MULT,
        post_eps=IDENTITY_POST_EPS,
        norm_weight=norm.weight if norm is not None else None,
        norm_eps=norm.variance_epsilon if norm is not None else 0.0,
        tiled_min_tokens=config.pre_tiled_min_tokens,
        bf16x3=config.pre_bf16x3,
    )
    return y, post, norm is not None


def identity_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    *,
    config: IdentityMhcConfig,
) -> torch.Tensor:
    if not config.fused or x.shape[0] == 0:
        return hc_post_identity_torch(x, residual, post)
    from sglang.kernels.ops.layernorm.ihc import ihc_post

    return ihc_post(x, residual, post)


def identity_head_and_norm(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm: torch.nn.Module,
    rms_eps: float,
    hc_eps: float,
    config: IdentityMhcConfig,
    head: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """The final stream collapse and norm; ``head`` is the model's own head."""
    if not config.fused or x.shape[0] == 0:
        return norm(head(x, hc_fn, hc_scale, hc_base))
    from sglang.kernels.ops.layernorm.ihc import ihc_head

    return ihc_head(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        norm_weight=norm.weight,
        norm_eps=norm.variance_epsilon,
        tiled_min_tokens=config.pre_tiled_min_tokens,
        bf16x3=config.pre_bf16x3,
    )

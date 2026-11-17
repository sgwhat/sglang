"""Triton kernels for identity hyper-connections (iHC / identity mHC).

Derived from the Hunyuan-V4 iHC kernels in ``hy4_ihc.py``.

Reference math, ``x`` is the ``[T, hc, H]`` residual stream (gate math in fp32):

    x_flat = x.flatten(-2).float()                               # [T, hc*H]
    r      = rsqrt(mean(x_flat ** 2) + rms_eps)
    mixes  = (x_flat @ fn.T) * r                                 # [T, 2*hc]
    pre    = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_eps
    post   = post_mult * sigmoid(mixes[:, hc:] * scale[1] + base[hc:]) + post_eps
    y      = sum_m pre[:, m] * x[:, m]                           # -> x.dtype

    post:  R'[:, m] = R[:, m] + post[:, m] * out

Hunyuan-V4 adds ``hc_eps`` to the post gate and DModel DeepSeek-V4 does not, so
``post_eps`` is a separate argument.

Coverage: CUDA, fp32 gate weights, any ``hc`` and hidden size; the tiled stage 1
needs SM90+ tensor cores (TF32x3, or BF16x3 for a BF16 stream).

``pre`` runs split-K in two kernels: per (token, K split) partial sums of squares
and gate projections, then a per-token fold of the splits in ascending order,
the gates and the weighted stream sum. Kernels compile with
``enable_fp_fusion=False`` so a product is rounded before its sum, as in eager
PyTorch; ``post`` is therefore bit-exact against eager. ``pre`` is not: eager
``F.linear`` sums the projection in cuBLAS order.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_BLOCK_K = 1024
_BLOCK_D = 1024


@triton.jit(do_not_specialize=["NUM_TOKENS"])
def _ihc_pre_stage1_tiled(
    x_ptr,
    fn_ptr,
    part_ptr,
    NUM_TOKENS,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PART_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INNER_K: tl.constexpr = 128,
    HAS_POST: tl.constexpr = True,
):
    # The per-token CTA reads the same gate tile separately for each token.
    # Share it across BLOCK_M rows and use TF32x3 for the FP32 projection;
    # weights, partials, squared sums and final gates remain FP32.
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    split = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    # Keep operands small enough to avoid register spills from a full
    # [16, 1024] FP32 dot. The external split layout remains unchanged.
    fragment = tl.arange(0, INNER_K)
    projection = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    sumsq = tl.zeros((BLOCK_M,), tl.float32)
    for block in range(tl.cdiv(BLOCK_K, INNER_K)):
        ks = split * BLOCK_K + block * INNER_K + fragment
        x = tl.load(
            x_ptr + rows[:, None] * K_TOTAL + ks[None, :],
            mask=(rows[:, None] < NUM_TOKENS) & (ks[None, :] < K_TOTAL),
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(
            fn_ptr + cols[None, :] * K_TOTAL + ks[:, None],
            mask=(cols[None, :] < (2 if HAS_POST else 1) * HC_MULT)
            & (ks[:, None] < K_TOTAL),
            other=0.0,
        ).to(tl.float32)
        projection = tl.dot(x, weights, projection, input_precision="tf32x3")
        sumsq += tl.sum(x * x, axis=1)
    base = part_ptr + (rows * NSPLIT + split) * PART_STRIDE
    tl.store(base, sumsq, mask=rows < NUM_TOKENS)
    offsets = tl.where(cols < HC_MULT, cols, HC_POW2 + cols - HC_MULT)
    tl.store(
        base[:, None] + 1 + offsets[None, :],
        projection,
        mask=(rows[:, None] < NUM_TOKENS) & (cols[None, :] < 2 * HC_MULT),
    )


@triton.jit(do_not_specialize=["NUM_TOKENS"])
def _ihc_pre_stage1_bf16x3(
    x_ptr,
    fn_ptr,
    part_ptr,
    NUM_TOKENS,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PART_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INNER_K: tl.constexpr,
    HAS_POST: tl.constexpr = True,
):
    # Same partial layout as _ihc_pre_stage1_tiled. The BF16 stream is fed to
    # the tensor cores as is; the FP32 gate weights are split into three BF16
    # parts, so every product is exact and only the FP32 accumulation rounds.
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    split = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    fragment = tl.arange(0, INNER_K)
    row_mask = rows[:, None] < NUM_TOKENS
    col_mask = cols[None, :] < (2 if HAS_POST else 1) * HC_MULT
    high = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    correction = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    squares = tl.zeros((BLOCK_M, INNER_K), tl.float32)
    for block in range(tl.cdiv(BLOCK_K, INNER_K)):
        ks = split * BLOCK_K + block * INNER_K + fragment
        x = tl.load(
            x_ptr + rows[:, None] * K_TOTAL + ks[None, :],
            mask=row_mask & (ks[None, :] < K_TOTAL),
            other=0.0,
        )
        weights = tl.load(
            fn_ptr + cols[None, :] * K_TOTAL + ks[:, None],
            mask=col_mask & (ks[:, None] < K_TOTAL),
            other=0.0,
        )
        w_high = weights.to(tl.bfloat16)
        residual = weights - w_high.to(tl.float32)
        w_middle = residual.to(tl.bfloat16)
        w_low = (residual - w_middle.to(tl.float32)).to(tl.bfloat16)
        high = tl.dot(x, w_high, high)
        correction = tl.dot(x, w_low, correction)
        correction = tl.dot(x, w_middle, correction)
        x_f32 = x.to(tl.float32)
        squares += x_f32 * x_f32
    sumsq = tl.sum(squares, axis=1)
    projection = high + correction
    base = part_ptr + (rows * NSPLIT + split) * PART_STRIDE
    tl.store(base, sumsq, mask=rows < NUM_TOKENS)
    offsets = tl.where(cols < HC_MULT, cols, HC_POW2 + cols - HC_MULT)
    tl.store(
        base[:, None] + 1 + offsets[None, :],
        projection,
        mask=row_mask & (cols[None, :] < 2 * HC_MULT),
    )


@triton.jit
def _ihc_pre_stage1(
    x_ptr,
    fn_ptr,
    part_ptr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PART_STRIDE: tl.constexpr,
    HAS_POST: tl.constexpr = True,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_s = tl.program_id(1)
    m_idx = tl.arange(0, HC_POW2)
    m_mask = m_idx < HC_MULT

    k_offs = pid_s * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K_TOTAL

    x_tile = tl.load(x_ptr + pid_t * K_TOTAL + k_offs, mask=k_mask, other=0.0).to(
        tl.float32
    )
    sumsq = tl.sum(x_tile * x_tile, axis=0)

    fn_offs = m_idx[:, None] * K_TOTAL + k_offs[None, :]
    fn_mask = m_mask[:, None] & k_mask[None, :]
    mix_pre = tl.sum(
        tl.load(fn_ptr + fn_offs, mask=fn_mask, other=0.0) * x_tile[None, :], axis=1
    )
    mix_post = tl.zeros((HC_POW2,), dtype=tl.float32)
    if HAS_POST:
        mix_post = tl.sum(
            tl.load(fn_ptr + HC_MULT * K_TOTAL + fn_offs, mask=fn_mask, other=0.0)
            * x_tile[None, :],
            axis=1,
        )

    base = part_ptr + (pid_t * NSPLIT + pid_s) * PART_STRIDE
    tl.store(base, sumsq)
    tl.store(base + 1 + m_idx, mix_pre, mask=m_mask)
    tl.store(base + 1 + HC_POW2 + m_idx, mix_post, mask=m_mask)


@triton.jit
def _ihc_pre_stage2(
    x_ptr,
    part_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    post_ptr,
    norm_weight_ptr,
    hidden_size: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NSPLIT: tl.constexpr,
    PART_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    post_mult: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    post_eps: tl.constexpr,
    FUSE_NORM: tl.constexpr = False,
    OUT_NORM_EPS: tl.constexpr = 0.0,
    HAS_POST: tl.constexpr = True,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    m_idx = tl.arange(0, HC_POW2)
    m_mask = m_idx < HC_MULT

    # Fold the splits in ascending order, as a single-CTA accumulation would.
    row = part_ptr + pid_t * NSPLIT * PART_STRIDE
    sumsq = tl.zeros((), dtype=tl.float32)
    mix_pre = tl.zeros((HC_POW2,), dtype=tl.float32)
    mix_post = tl.zeros((HC_POW2,), dtype=tl.float32)
    for s in tl.static_range(NSPLIT):
        b = row + s * PART_STRIDE
        sumsq += tl.load(b)
        mix_pre += tl.load(b + 1 + m_idx, mask=m_mask, other=0.0)
        mix_post += tl.load(b + 1 + HC_POW2 + m_idx, mask=m_mask, other=0.0)

    rsqrt = tl.rsqrt(sumsq / K_TOTAL + rms_eps)
    scale_pre = tl.load(scale_ptr)
    base_pre = tl.load(base_ptr + m_idx, mask=m_mask, other=0.0)

    pre = tl.sigmoid(mix_pre * rsqrt * scale_pre + base_pre) + hc_eps
    if HAS_POST:
        if pid_d == 0:
            scale_post = tl.load(scale_ptr + 1)
            base_post = tl.load(base_ptr + HC_MULT + m_idx, mask=m_mask, other=0.0)
            post = (
                post_mult * tl.sigmoid(mix_post * rsqrt * scale_post + base_post)
                + post_eps
            )
            tl.store(post_ptr + pid_t * HC_MULT + m_idx, post, mask=m_mask)

    x_row = x_ptr + pid_t * K_TOTAL
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < hidden_size
    y_block = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for m in tl.static_range(HC_MULT):
        x_m = tl.load(x_row + m * hidden_size + d_offs, mask=d_mask, other=0.0)
        pre_m = tl.sum(tl.where(m_idx == m, pre, 0.0), axis=0)
        y_block += pre_m * x_m.to(tl.float32)
    if FUSE_NORM:
        # Preserve the sublayer boundary: collapse rounds to BF16 before RMSNorm.
        y_block = y_block.to(y_ptr.dtype.element_ty).to(tl.float32)
        variance = (
            tl.sum(tl.where(d_mask, y_block * y_block, 0.0), axis=0) / hidden_size
        )
        weight = tl.load(norm_weight_ptr + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        y_block = (y_block * tl.rsqrt(variance + OUT_NORM_EPS)) * weight
    tl.store(
        y_ptr + pid_t * hidden_size + d_offs,
        y_block.to(y_ptr.dtype.element_ty),
        mask=d_mask,
    )


@triton.jit
def _ihc_post_kernel(
    out_ptr,
    res_ptr,
    post_ptr,
    y_ptr,
    hidden_size: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    m_idx = tl.arange(0, HC_POW2)
    m_mask = m_idx < HC_MULT
    post = tl.load(post_ptr + pid_t * HC_MULT + m_idx, mask=m_mask, other=0.0)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < hidden_size
    out_block = tl.load(out_ptr + pid_t * hidden_size + d_offs, mask=d_mask, other=0.0)
    out_block = out_block.to(tl.float32)

    res_row = res_ptr + pid_t * HC_MULT * hidden_size
    y_row = y_ptr + pid_t * HC_MULT * hidden_size
    for m in tl.static_range(HC_MULT):
        res_block = tl.load(res_row + m * hidden_size + d_offs, mask=d_mask, other=0.0)
        post_m = tl.sum(tl.where(m_idx == m, post, 0.0), axis=0)
        y_block = post_m * out_block + res_block.to(tl.float32)
        tl.store(
            y_row + m * hidden_size + d_offs,
            y_block.to(y_ptr.dtype.element_ty),
            mask=d_mask,
        )


def _launch_pre_stage1(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    part: torch.Tensor,
    *,
    k_total: int,
    hc_mult: int,
    hc_pow2: int,
    nsplit: int,
    part_stride: int,
    has_post: bool,
    tiled: bool,
    bf16x3: bool,
) -> None:
    num_tokens = x.shape[0]
    if tiled and bf16x3 and x.dtype == torch.bfloat16:
        # Untuned: 128-token tiles once there are enough CTAs to fill the GPU.
        block_m = 128 if num_tokens >= 4096 else 64
        stage1 = _ihc_pre_stage1_bf16x3
        grid = (triton.cdiv(num_tokens, block_m), nsplit)
        stage1_args = dict(
            NUM_TOKENS=num_tokens,
            BLOCK_M=block_m,
            BLOCK_N=max(16, 2 * hc_pow2),
            INNER_K=64,
            num_warps=8 if block_m == 128 else 4,
            num_stages=4,
        )
    elif tiled:
        stage1 = _ihc_pre_stage1_tiled
        grid = (triton.cdiv(num_tokens, 64), nsplit)
        stage1_args = dict(
            NUM_TOKENS=num_tokens,
            BLOCK_M=64,
            BLOCK_N=max(16, 2 * hc_pow2),
            INNER_K=64,
            num_warps=4,
        )
    else:
        stage1 = _ihc_pre_stage1
        grid = (num_tokens, nsplit)
        stage1_args = dict(num_warps=8)
    stage1[grid](
        x,
        hc_fn,
        part,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NSPLIT=nsplit,
        BLOCK_K=_BLOCK_K,
        PART_STRIDE=part_stride,
        HAS_POST=has_post,
        enable_fp_fusion=False,
        **stage1_args,
    )


def ihc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    post_eps: float = 0.0,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 0.0,
    tiled_min_tokens: Optional[int] = None,
    bf16x3: bool = False,
    _has_post: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collapse ``x [T, hc, H]`` to the sublayer input ``[T, H]``: ``(y, post)``.

    ``norm_weight`` folds the following RMSNorm after the readout's rounding to
    ``x.dtype``. Batches of at least ``tiled_min_tokens`` rows (None: never) run
    the tiled stage 1; ``bf16x3`` makes it the BF16x3 variant for a BF16 stream.
    """
    assert x.dim() == 3, f"x must be 3D (T, hc_mult, hidden_size), got {x.shape}"
    assert hc_fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32 and hc_base.dtype == torch.float32

    x = x.contiguous()
    hc_fn = hc_fn.contiguous()
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    num_gates = 2 if _has_post else 1
    assert hc_fn.shape == (num_gates * hc_mult, k_total)
    assert hc_base.shape == (num_gates * hc_mult,)
    assert hc_scale.shape == (num_gates,)
    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        norm_weight = norm_weight.contiguous()

    y = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    if num_tokens == 0:
        return y, post

    hc_pow2 = triton.next_power_of_2(hc_mult)
    # One BLOCK_K tile per CTA, so the partials replay a single-CTA tile order.
    nsplit = triton.cdiv(k_total, _BLOCK_K)
    part_stride = 1 + 2 * hc_pow2
    part = torch.empty(
        (num_tokens, nsplit, part_stride), dtype=torch.float32, device=x.device
    )
    _launch_pre_stage1(
        x,
        hc_fn,
        part,
        k_total=k_total,
        hc_mult=hc_mult,
        hc_pow2=hc_pow2,
        nsplit=nsplit,
        part_stride=part_stride,
        has_post=_has_post,
        tiled=tiled_min_tokens is not None and num_tokens >= tiled_min_tokens,
        bf16x3=bf16x3,
    )
    block_d = (
        triton.next_power_of_2(hidden_size) if norm_weight is not None else _BLOCK_D
    )
    _ihc_pre_stage2[(num_tokens, triton.cdiv(hidden_size, block_d))](
        x,
        part,
        hc_scale,
        hc_base,
        y,
        post,
        norm_weight,
        hidden_size=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NSPLIT=nsplit,
        PART_STRIDE=part_stride,
        BLOCK_D=block_d,
        post_mult=post_mult,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        post_eps=post_eps,
        FUSE_NORM=norm_weight is not None,
        OUT_NORM_EPS=norm_eps,
        HAS_POST=_has_post,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y, post


def ihc_head(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 0.0,
    tiled_min_tokens: Optional[int] = None,
    bf16x3: bool = False,
) -> torch.Tensor:
    """Terminal readout ``[T, hc, H] -> [T, H]`` with pre gates only, sharing
    ``ihc_pre``'s split-K projection and fused norm."""
    return ihc_pre(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        post_mult=0.0,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        tiled_min_tokens=tiled_min_tokens,
        bf16x3=bf16x3,
        _has_post=False,
    )[0]


def ihc_post(
    out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """``residual[:, m] + post[:, m] * out`` for every stream, bit-exact vs eager."""
    assert out.dim() == 2, f"out must be 2D (T, hidden_size), got {out.shape}"
    assert post.dtype == torch.float32

    out = out.contiguous()
    residual = residual.contiguous()
    num_tokens, hidden_size = out.shape
    hc_mult = post.shape[-1]
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    assert post.shape == (num_tokens, hc_mult)

    y = torch.empty_like(residual)
    if num_tokens == 0:
        return y
    _ihc_post_kernel[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        out,
        residual,
        post.contiguous(),
        y,
        hidden_size=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=triton.next_power_of_2(hc_mult),
        BLOCK_D=_BLOCK_D,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y

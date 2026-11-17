"""FP32 router projection from a BF16 input with three BF16 products.

Keep the checkpoint router in FP32. Convert input tiles in registers instead of
materializing ``hidden_states.float()``. Split FP32 weight tiles into three BF16
components in registers; retain separate FP32 high/correction accumulators.
The checkpoint weight and output stay FP32; numerical routing gates still apply.
The ``_bf16x3`` variants take the three BF16 parts pre-split at load time.

Coverage: CUDA SM90+, ``[M, K]`` row-major BF16 input against an ``[N, K]`` FP32
weight, FP32 output; the pre-split variants need ``K % 512 == 0``. Same contract
as ``router_gemv.router_gemm_bf16x3_cublas``, with a compensated K accumulation.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _fp32_router_gemm(
    x_ptr,
    weight_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    X_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    accum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    compensation = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for block in range(tl.cdiv(K, BLOCK_K)):
        k = block * BLOCK_K + ks
        x = tl.load(
            x_ptr + rows[:, None] * X_STRIDE + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + cols[None, :] * W_STRIDE + k[:, None],
            mask=(cols[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        w_high = weight.to(tl.bfloat16)
        residual = weight - w_high.to(tl.float32)
        w_middle = residual.to(tl.bfloat16)
        w_low = (residual - w_middle.to(tl.float32)).to(tl.bfloat16)
        high = tl.dot(x, w_high)
        correction = tl.dot(x, w_low)
        correction = tl.dot(x, w_middle, correction)
        # Fold short dot products with compensation instead of keeping one
        # long BF16 tensor-core accumulator across all of K.
        partial = high + correction
        adjusted = partial - compensation
        updated = accum + adjusted
        compensation = (updated - accum) - adjusted
        accum = updated
    tl.store(
        output_ptr + rows[:, None] * N + cols[None, :],
        accum,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def fp32_router_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.ndim == weight.ndim == 2 and x.shape[1] == weight.shape[1]
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.float32
    assert x.is_cuda and weight.is_cuda and x.device == weight.device
    assert x.stride(1) == weight.stride(1) == 1
    m, k = x.shape
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if m:
        _fp32_router_gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            x,
            weight,
            output,
            m,
            n,
            k,
            x.stride(0),
            weight.stride(0),
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
        )
    return output


@triton.jit(do_not_specialize=["M"])
def _fp32_router_gemm_bf16x3(
    x_ptr,
    weight_bf16x3_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    X_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    # weight_bf16x3 is [3, N, K]: the high, middle and low BF16 parts of the
    # FP32 weight, as split in _fp32_router_gemm. Chain the tensor-core
    # accumulators over CHUNK_K and fold chunks with compensation.
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    x_rows = x_ptr + rows[:, None] * X_STRIDE
    w_high_cols = weight_bf16x3_ptr + cols[None, :] * K
    w_middle_cols = w_high_cols + N * K
    w_low_cols = w_middle_cols + N * K
    accum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    compensation = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for chunk in range(K // CHUNK_K):
        high = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        correction = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for block in range(CHUNK_K // BLOCK_K):
            k = chunk * CHUNK_K + block * BLOCK_K + ks
            x = tl.load(x_rows + k[None, :], mask=row_mask, other=0.0)
            w_high = tl.load(w_high_cols + k[:, None], mask=col_mask, other=0.0)
            w_middle = tl.load(w_middle_cols + k[:, None], mask=col_mask, other=0.0)
            w_low = tl.load(w_low_cols + k[:, None], mask=col_mask, other=0.0)
            high = tl.dot(x, w_high, high)
            correction = tl.dot(x, w_low, correction)
            correction = tl.dot(x, w_middle, correction)
        partial = high + correction
        adjusted = partial - compensation
        updated = accum + adjusted
        compensation = (updated - accum) - adjusted
        accum = updated
    tl.store(
        output_ptr + rows[:, None] * N + cols[None, :],
        accum,
        mask=row_mask & col_mask,
    )


@triton.jit(do_not_specialize=["M"])
def _fp32_router_gemm_bf16x3_split_k(
    x_ptr,
    weight_bf16x3_ptr,
    part_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    X_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Small-M (decode) variant: each CTA covers K // SPLIT_K and writes FP32
    # partials [SPLIT_K, M, N]; _fp32_router_reduce_splits folds them in order.
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(2)
    ks = tl.arange(0, BLOCK_K)
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    x_rows = x_ptr + rows[:, None] * X_STRIDE
    w_high_cols = weight_bf16x3_ptr + cols[None, :] * K
    w_middle_cols = w_high_cols + N * K
    w_low_cols = w_middle_cols + N * K
    high = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    correction = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for block in range(K // SPLIT_K // BLOCK_K):
        k = split * (K // SPLIT_K) + block * BLOCK_K + ks
        x = tl.load(x_rows + k[None, :], mask=row_mask, other=0.0)
        w_high = tl.load(w_high_cols + k[:, None], mask=col_mask, other=0.0)
        w_middle = tl.load(w_middle_cols + k[:, None], mask=col_mask, other=0.0)
        w_low = tl.load(w_low_cols + k[:, None], mask=col_mask, other=0.0)
        high = tl.dot(x, w_high, high)
        correction = tl.dot(x, w_low, correction)
        correction = tl.dot(x, w_middle, correction)
    tl.store(
        part_ptr + (split * M + rows[:, None]) * N + cols[None, :],
        high + correction,
        mask=row_mask & col_mask,
    )


@triton.jit
def _fp32_router_reduce_splits(
    part_ptr,
    output_ptr,
    NUM_ELEMENTS,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUM_ELEMENTS
    accum = tl.zeros((BLOCK,), tl.float32)
    compensation = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLIT_K):
        partial = tl.load(part_ptr + split * NUM_ELEMENTS + offsets, mask=mask)
        adjusted = partial - compensation
        updated = accum + adjusted
        compensation = (updated - accum) - adjusted
        accum = updated
    tl.store(output_ptr + offsets, accum, mask=mask)


_BF16X3_BLOCK_K = 32
_BF16X3_CHUNK_K = 512
# Arbitrary: below this many rows the M x N grid cannot fill an H200.
_BF16X3_SPLIT_K_MAX_M = 256
_BF16X3_SPLIT_K = 16


def split_fp32_weight_bf16x3(weight: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` FP32 weight to its ``[3, N, K]`` high/middle/low BF16 parts."""
    assert weight.dtype == torch.float32 and weight.ndim == 2
    weight = weight.contiguous()
    high = weight.to(torch.bfloat16)
    residual = weight - high.float()
    middle = residual.to(torch.bfloat16)
    low = (residual - middle.float()).to(torch.bfloat16)
    return torch.stack((high, middle, low)).contiguous()


def supports_fp32_router_bf16x3(k: int) -> bool:
    return k % _BF16X3_CHUNK_K == 0


def fp32_router_linear_bf16x3(
    x: torch.Tensor, weight_bf16x3: torch.Tensor
) -> torch.Tensor:
    """``fp32_router_linear`` with a pre-split weight and larger tiles."""
    assert x.ndim == 2 and weight_bf16x3.ndim == 3
    assert weight_bf16x3.shape[0] == 3 and x.shape[1] == weight_bf16x3.shape[2]
    assert x.dtype == weight_bf16x3.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight_bf16x3.device
    assert x.stride(1) == 1 and weight_bf16x3.is_contiguous()
    m, k = x.shape
    n = weight_bf16x3.shape[1]
    assert supports_fp32_router_bf16x3(k)
    output = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if 0 < m <= _BF16X3_SPLIT_K_MAX_M:
        _fp32_router_linear_bf16x3_split_k(x, weight_bf16x3, output)
    elif m:
        # Untuned; 3 stages of these tiles stay well inside SM90 shared memory.
        block_m, block_n = 64, 128
        _fp32_router_gemm_bf16x3[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
            x,
            weight_bf16x3,
            output,
            m,
            n,
            k,
            x.stride(0),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=_BF16X3_BLOCK_K,
            CHUNK_K=_BF16X3_CHUNK_K,
            num_warps=8,
            num_stages=3,
        )
    return output


def _fp32_router_linear_bf16x3_split_k(
    x: torch.Tensor, weight_bf16x3: torch.Tensor, output: torch.Tensor
) -> None:
    m, k = x.shape
    n = weight_bf16x3.shape[1]
    block_m, block_n, split_k = 64, 64, _BF16X3_SPLIT_K
    assert k % (split_k * _BF16X3_BLOCK_K) == 0
    part = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
    _fp32_router_gemm_bf16x3_split_k[
        (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
    ](
        x,
        weight_bf16x3,
        part,
        m,
        n,
        k,
        x.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_BF16X3_BLOCK_K,
        SPLIT_K=split_k,
        num_warps=4,
        num_stages=3,
    )
    block = 1024
    _fp32_router_reduce_splits[(triton.cdiv(m * n, block),)](
        part,
        output,
        m * n,
        SPLIT_K=split_k,
        BLOCK=block,
        num_warps=4,
    )

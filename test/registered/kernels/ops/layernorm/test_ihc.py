"""Triton identity-mHC kernels against the eager reference, including the tiled
and BF16x3 stage 1, partial tiles, the fused norm and CUDA graph replay."""

import unittest

import torch

from sglang.kernels.ops.layernorm.ihc import ihc_head, ihc_post, ihc_pre
from sglang.srt.models.deepseek_common.identity_mhc import (
    hc_post_identity_torch,
    hc_pre_identity_torch,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_EPS = 1e-6


def _inputs(tokens, hc, hidden, dtype, seed=1234):
    torch.manual_seed(seed)
    x = torch.randn(tokens, hc, hidden, device="cuda", dtype=dtype)
    fn = torch.randn(2 * hc, hc * hidden, device="cuda", dtype=torch.float32) * 0.01
    scale = torch.tensor([0.3, 0.5], device="cuda", dtype=torch.float32)
    base = torch.randn(2 * hc, device="cuda", dtype=torch.float32) * 0.1
    return x, fn, scale, base


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "SM90+ required",
)
class TestIdentityMhcKernels(CustomTestCase):
    def test_pre_matches_eager(self):
        variants = (
            ("per_token", dict()),
            ("tiled", dict(tiled_min_tokens=1)),
            ("bf16x3", dict(tiled_min_tokens=1, bf16x3=True)),
        )
        for tokens, hc, hidden in [
            (0, 4, 4096),
            (1, 3, 128),
            (17, 4, 4096),
            (128, 4, 4096),
            (1024, 4, 4096),
            (4097, 4, 4096),
        ]:
            for dtype in [torch.bfloat16, torch.float32]:
                for name, kwargs in variants:
                    with self.subTest(
                        tokens=tokens, hc=hc, hidden=hidden, dtype=dtype, variant=name
                    ):
                        self._check_pre(tokens, hc, hidden, dtype, kwargs)

    def _check_pre(self, tokens, hc, hidden, dtype, kwargs):
        x, fn, scale, base = _inputs(tokens, hc, hidden, dtype)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            expected_y, expected_post = hc_pre_identity_torch(
                x, fn, scale, base, rms_eps=_EPS, hc_eps=_EPS
            )
            y, post = ihc_pre(
                x, fn, scale, base, rms_eps=_EPS, hc_eps=_EPS, post_mult=2.0, **kwargs
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        torch.testing.assert_close(post, expected_post, rtol=2e-5, atol=2e-6)
        if dtype == torch.bfloat16:
            torch.testing.assert_close(y, expected_y, rtol=0.008, atol=0.02)
        else:
            torch.testing.assert_close(y, expected_y, rtol=4e-5, atol=2e-5)

    def test_post_is_bit_exact(self):
        for tokens in [0, 1, 129]:
            with self.subTest(tokens=tokens):
                x, fn, scale, base = _inputs(tokens, 4, 4096, torch.bfloat16)
                _, post = hc_pre_identity_torch(
                    x, fn, scale, base, rms_eps=_EPS, hc_eps=_EPS
                )
                out = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
                torch.testing.assert_close(
                    ihc_post(out, x, post),
                    hc_post_identity_torch(out, x, post),
                    rtol=0,
                    atol=0,
                )

    def test_tiled_head_with_fused_norm_replays_new_input(self):
        torch.manual_seed(43)
        for tokens in [1, 128, 129]:
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
                fn = torch.randn(4, 16384, device="cuda", dtype=torch.float32) * 0.01
                scale = torch.tensor([0.3], device="cuda")
                base = torch.randn(4, device="cuda") * 0.1
                weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
                kwargs = dict(
                    rms_eps=_EPS, hc_eps=_EPS, norm_weight=weight, norm_eps=_EPS
                )
                reference = ihc_head(x, fn, scale, base, **kwargs)
                tiled = dict(kwargs, tiled_min_tokens=1)
                actual = ihc_head(x, fn, scale, base, **tiled)
                torch.testing.assert_close(actual, reference, rtol=0.012, atol=0.03)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        ihc_head(x, fn, scale, base, **tiled)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = ihc_head(x, fn, scale, base, **tiled)
                x.add_(0.25)
                expected = ihc_head(x, fn, scale, base, **tiled)
                graph.replay()
                torch.testing.assert_close(captured, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

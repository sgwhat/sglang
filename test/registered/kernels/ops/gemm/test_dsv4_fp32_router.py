"""The tensor-core FP32 router keeps FP32 logits and expert choices, with and
without the load-time BF16 split of its weight."""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.dsv4_fp32_router import (
    fp32_router_linear,
    fp32_router_linear_bf16x3,
    split_fp32_weight_bf16x3,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "SM90+ required",
)
class TestDsv4Fp32Router(CustomTestCase):
    def test_values_and_top6(self):
        # 17 and 256 take the pre-split split-K route, 4097 its tiled route.
        for tokens in [0, 1, 17, 64, 128, 256, 1024, 4097]:
            for presplit in [False, True]:
                with self.subTest(tokens=tokens, presplit=presplit):
                    self._check(tokens, presplit)

    def _check(self, tokens, presplit):
        torch.manual_seed(1234)
        x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(256, 4096, device="cuda", dtype=torch.float32) * 0.02
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            expected = F.linear(x.float(), weight)
            if presplit:
                actual = fp32_router_linear_bf16x3(
                    x, split_fp32_weight_bf16x3(weight)
                )
            else:
                actual = fp32_router_linear(x, weight)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
        reference_ids = expected.topk(6).indices.sort().values
        actual_ids = actual.topk(6).indices.sort().values
        changed = (reference_ids != actual_ids).any(dim=-1)
        if changed.any():
            # Near ties go to a higher-precision oracle instead of requiring
            # cuBLAS FP32's reduction order to be reproduced exactly.
            oracle = F.linear(x[changed].double(), weight.double())
            torch.testing.assert_close(
                actual_ids[changed],
                oracle.topk(6).indices.sort().values,
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()

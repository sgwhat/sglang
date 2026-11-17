"""A prefill TBO child's indexer schedule must be planned under the DeepGEMM SM
budget its layers run with, and every other batch under the full budget."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import _tbo_child_planning_scope
from sglang.srt.layers.attention.dsa.utils import fp8_mqa_logits_make_fused_kv
from sglang.srt.layers.deep_gemm_wrapper import configure_deep_gemm_num_sms
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_BUDGET = (
    "sglang.srt.batch_overlap.operations_strategy."
    "deepseek_v4_tbo_deep_gemm_num_sms"
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "Hopper regression",
)
class TestDsv4TboSmBudget(CustomTestCase):
    def setUp(self):
        import deep_gemm

        self.deep_gemm = deep_gemm
        self.full_sms = deep_gemm.get_num_sms()
        self.lengths = torch.full((4,), 256, device="cuda", dtype=torch.int32)
        self.page_table = torch.arange(4, device="cuda", dtype=torch.int32).repeat(4, 1)

    def tearDown(self):
        self.deep_gemm.set_num_sms(self.full_sms)

    def _plan(self, *, child, mode, reserved_sms):
        batch = SimpleNamespace(tbo_parent_token_range=(0, 4) if child else None)
        with patch(_BUDGET, return_value=self.full_sms - reserved_sms):
            with _tbo_child_planning_scope(batch, mode):
                planned_sms = self.deep_gemm.get_num_sms()
                schedule = self.deep_gemm.get_paged_mqa_logits_metadata(
                    self.lengths[:, None], 256, planned_sms
                )
        self.assertEqual(self.deep_gemm.get_num_sms(), self.full_sms)
        return planned_sms, schedule

    def _run(self, schedule, num_sms):
        kv = torch.zeros((4, 64, 128), device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.ones((4, 64), device="cuda", dtype=torch.float32)
        packed = fp8_mqa_logits_make_fused_kv(kv, scales, 64, 128).view(4, 64, 1, 132)
        q = torch.zeros((4, 1, 64, 128), device="cuda", dtype=torch.float8_e4m3fn)
        weights = torch.ones((4, 64), device="cuda", dtype=torch.float32)
        with configure_deep_gemm_num_sms(num_sms):
            logits = self.deep_gemm.fp8_paged_mqa_logits(
                q,
                packed,
                weights,
                self.lengths[:, None],
                self.page_table,
                schedule,
                256,
                False,
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(logits, torch.zeros_like(logits), rtol=0, atol=0)

    def test_child_prefill_plans_under_the_reserved_budget(self):
        for reserved_sms in (48, 32):
            with self.subTest(reserved_sms=reserved_sms):
                planned, schedule = self._plan(
                    child=True, mode=ForwardMode.EXTEND, reserved_sms=reserved_sms
                )
                self.assertEqual(planned, self.full_sms - reserved_sms)
                self._run(schedule, planned)

    def test_other_batches_keep_the_full_budget(self):
        for child, mode in (
            (False, ForwardMode.EXTEND),
            (True, ForwardMode.DECODE),
            (True, ForwardMode.TARGET_VERIFY),
        ):
            with self.subTest(child=child, mode=mode):
                planned, _ = self._plan(child=child, mode=mode, reserved_sms=48)
                self.assertEqual(planned, self.full_sms)

    def test_planning_failure_restores_budget(self):
        batch = SimpleNamespace(tbo_parent_token_range=(0, 4))
        with (
            patch(_BUDGET, return_value=self.full_sms - 48),
            self.assertRaisesRegex(RuntimeError, "planner failure"),
        ):
            with _tbo_child_planning_scope(batch, ForwardMode.EXTEND):
                raise RuntimeError("planner failure")
        self.assertEqual(self.deep_gemm.get_num_sms(), self.full_sms)


if __name__ == "__main__":
    unittest.main()

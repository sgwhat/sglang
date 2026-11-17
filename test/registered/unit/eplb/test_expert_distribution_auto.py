"""DeepEP auto mode counts experts at TopK, as normal mode does."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAutoExpertDistribution(CustomTestCase):
    def test_auto_mode_counts_selected_physical_experts(self):
        """Recording with --moe-a2a-backend deepep --deepep-mode auto used to
        raise NotImplementedError when the gatherer was built."""
        metadata = SimpleNamespace(
            num_layers=2, num_physical_experts=16, num_local_physical_experts=2
        )
        ids = torch.tensor([[0, 1, 2], [0, -1, 15]])
        expected = torch.zeros((2, 16), dtype=torch.int32)
        expected[0, 0] = 2
        expected[0, [1, 2, 15]] = 1
        platform = override_platform(is_npu=False)
        platform.install()
        self.addCleanup(platform.restore)
        for mode in ["normal", "auto"]:
            execution = SimpleNamespace(
                moe=SimpleNamespace(
                    expert_distribution_recorder_mode="stat",
                    moe_a2a_backend="deepep",
                    deepep_mode=mode,
                    elastic_ep_backend=None,
                )
            )
            with (
                self.subTest(mode=mode),
                patch.object(distribution, "get_exec", return_value=execution),
                patch.object(distribution, "get_device", return_value="cpu"),
            ):
                gatherer = distribution._SinglePassGatherer.init_new(metadata, 0)
                self.assertIsInstance(
                    gatherer, distribution._SelectExpertsSinglePassGatherer
                )
                gatherer.on_select_experts(0, ids)
                torch.testing.assert_close(
                    gatherer.collect()["global_physical_count"], expected
                )


if __name__ == "__main__":
    unittest.main()

"""DeepSeek-V4 dense FFN layers (config.n_dense_layers): supported layouts and
the stage facts other pipeline ranks derive from the config alone."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.layer_boundary.contracts import ExitRows
from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_dense_ffn as dense_ffn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _a2a(none: bool):
    return SimpleNamespace(is_none=lambda: none)


def _parallel(*, attn_dp: int, attn_tp: int, attn_cp: int = 1):
    return SimpleNamespace(
        attn_dp_size=attn_dp,
        attn_tp_size=attn_tp,
        attn_cp_size=attn_cp,
        enable_attn_tp_input_scattered=False,
    )


class TestDenseFfnLayouts(CustomTestCase):
    def _reasons(self, *, parallel, fully_dp, a2a_none, is_cuda=True, cp=False):
        config = SimpleNamespace(hc_pre_from_prev_sublayer=False)
        with (
            patch.object(dense_ffn, "get_parallel", return_value=parallel),
            patch.object(dense_ffn, "is_dense_ffn_fully_dp", return_value=fully_dp),
            patch.object(dense_ffn, "get_moe_a2a_backend", return_value=_a2a(a2a_none)),
            patch.object(dense_ffn, "is_dsa_enable_prefill_cp", return_value=cp),
        ):
            return dense_ffn._unsupported_dense_ffn_reasons(config, is_cuda=is_cuda)

    def test_supported_layouts(self):
        """TP8, and DP8 attention with a2a MoE when every rank holds the FFN."""
        self.assertEqual(
            self._reasons(
                parallel=_parallel(attn_dp=1, attn_tp=8), fully_dp=False, a2a_none=True
            ),
            [],
        )
        for a2a_none in (True, False):
            with self.subTest(a2a_none=a2a_none):
                self.assertEqual(
                    self._reasons(
                        parallel=_parallel(attn_dp=8, attn_tp=1),
                        fully_dp=True,
                        a2a_none=a2a_none,
                    ),
                    [],
                )

    def test_layouts_that_would_mix_rows_raise(self):
        """A TP-sharded FFN would sum unrelated DP rows, and a local one under
        attention TP would compute on a slice of the mHC streams' rows."""
        cases = (
            (dict(parallel=_parallel(attn_dp=8, attn_tp=1), fully_dp=False), True),
            (dict(parallel=_parallel(attn_dp=1, attn_tp=8), fully_dp=False), False),
            (dict(parallel=_parallel(attn_dp=1, attn_tp=8), fully_dp=True), True),
            (dict(parallel=_parallel(attn_dp=2, attn_tp=4), fully_dp=True), True),
            (
                dict(
                    parallel=_parallel(attn_dp=1, attn_tp=1, attn_cp=8),
                    fully_dp=True,
                    cp=True,
                ),
                True,
            ),
            (
                dict(
                    parallel=_parallel(attn_dp=8, attn_tp=1),
                    fully_dp=True,
                    is_cuda=False,
                ),
                True,
            ),
        )
        for kwargs, a2a_none in cases:
            with self.subTest(**{k: v for k, v in kwargs.items() if k != "parallel"}):
                self.assertNotEqual(self._reasons(a2a_none=a2a_none, **kwargs), [])

    def test_stage_facts_declare_dense_layers_without_moe_transport(self):
        """Dense layers declare no operand transport, and the last of them marks
        where a TBO split meets the first sparse layer."""
        config = SimpleNamespace(
            hc_pre_from_prev_sublayer=False,
            hc_residual_mode="sinkhorn",
            n_dense_layers=3,
        )
        parallel = _parallel(attn_dp=8, attn_tp=1)
        with (
            patch.object(boundary, "get_moe_a2a_backend", return_value=_a2a(False)),
            patch.object(boundary, "get_parallel", return_value=parallel),
            patch(
                "sglang.srt.layers.layer_boundary.layout.get_parallel",
                return_value=parallel,
            ),
        ):
            ffn = {
                layer_id: boundary.stages(config, layer_id)[1]
                for layer_id in (1, 2, 3)
            }
        for layer_id in (1, 2):
            with self.subTest(layer_id=layer_id):
                self.assertFalse(ffn[layer_id].sparse)
                self.assertIsNone(ffn[layer_id].row_transport)
        self.assertIsNone(ffn[1].exit_rows)
        self.assertIs(ffn[2].exit_rows, ExitRows.TBO_SPLIT)
        self.assertTrue(ffn[3].sparse)
        self.assertIsNotNone(ffn[3].row_transport)

    def test_draft_layers_never_run_dense(self):
        config = SimpleNamespace(n_dense_layers=3)
        self.assertTrue(dense_ffn.is_dense_ffn_layer(config, 2, is_nextn=False))
        self.assertFalse(dense_ffn.is_dense_ffn_layer(config, 3, is_nextn=False))
        self.assertFalse(dense_ffn.is_dense_ffn_layer(config, 0, is_nextn=True))


if __name__ == "__main__":
    unittest.main()

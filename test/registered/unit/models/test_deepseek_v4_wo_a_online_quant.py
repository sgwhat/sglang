"""DeepSeek-V4 ``wo_a`` FP8 gating with ``ignored_layers`` and opt-in online quant."""

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.deepseek_common.wo_a_online_quant import (
    is_wo_a_skipped_in_quant_config,
    quantize_bf16_wo_a_streaming,
    validate_uniform_wo_a_quant_config,
    wo_a_online_quant_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fp8_config(*, ignored_layers=None, block=128) -> Fp8Config:
    return Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [block, block],
            "ignored_layers": ignored_layers or [],
        }
    )


class TestDeepseekV4WoAIgnoredLayers(CustomTestCase):
    def test_is_wo_a_skipped_matches_hf_and_model_prefixes(self):
        skipped_cfg = _fp8_config(
            ignored_layers=[
                "lm_head",
                "model.layers.0.self_attn.wo_a",
                "layers.1.self_attn.wo_a",
            ]
        )
        self.assertTrue(
            is_wo_a_skipped_in_quant_config(
                skipped_cfg, "model.layers.0.self_attn.wo_a"
            )
        )
        self.assertTrue(
            is_wo_a_skipped_in_quant_config(
                skipped_cfg, "model.layers.1.self_attn.wo_a"
            )
        )
        self.assertFalse(
            is_wo_a_skipped_in_quant_config(
                skipped_cfg, "model.layers.2.self_attn.wo_a"
            )
        )
        stock_cfg = _fp8_config(ignored_layers=["lm_head", "model.layers.0.mlp"])
        self.assertFalse(
            is_wo_a_skipped_in_quant_config(stock_cfg, "model.layers.0.self_attn.wo_a")
        )

    def test_validate_uniform_wo_a_quant_config(self):
        all_skipped = _fp8_config(
            ignored_layers=[f"model.layers.{i}.self_attn.wo_a" for i in range(3)]
        )
        none_skipped = _fp8_config(ignored_layers=["lm_head"])
        mixed = _fp8_config(ignored_layers=["model.layers.0.self_attn.wo_a"])

        validate_uniform_wo_a_quant_config(all_skipped, 3)
        validate_uniform_wo_a_quant_config(none_skipped, 3)

        with (
            envs.SGLANG_OPT_FP8_WO_A_ONLINE_QUANT.override(False),
            self.assertRaisesRegex(ValueError, "uniform across all decoder layers"),
        ):
            validate_uniform_wo_a_quant_config(mixed, 3)

        # Online quant converts any BF16 wo_a to block-128 FP8 at load.
        with envs.SGLANG_OPT_FP8_WO_A_ONLINE_QUANT.override(True):
            validate_uniform_wo_a_quant_config(mixed, 3)

    def test_wo_a_fp8_gemm_enabled_respects_ignored_layers_and_online_quant(self):
        from sglang.srt.models import deepseek_v4 as dsv4

        skipped_cfg = _fp8_config(ignored_layers=["model.layers.0.self_attn.wo_a"])
        stock_cfg = _fp8_config(ignored_layers=["lm_head"])

        with mock.patch.object(dsv4, "_FP8_WO_A_GEMM", True):
            with envs.SGLANG_OPT_FP8_WO_A_ONLINE_QUANT.override(False):
                self.assertTrue(
                    dsv4.wo_a_fp8_gemm_enabled(
                        stock_cfg, "model.layers.0.self_attn.wo_a"
                    )
                )
                self.assertFalse(
                    dsv4.wo_a_fp8_gemm_enabled(
                        skipped_cfg, "model.layers.0.self_attn.wo_a"
                    )
                )
            with envs.SGLANG_OPT_FP8_WO_A_ONLINE_QUANT.override(True):
                self.assertTrue(
                    dsv4.wo_a_fp8_gemm_enabled(
                        skipped_cfg, "model.layers.0.self_attn.wo_a"
                    )
                )

        online_cfg = wo_a_online_quant_config()
        self.assertEqual(online_cfg.weight_block_size, [128, 128])
        self.assertFalse(
            is_wo_a_skipped_in_quant_config(online_cfg, "model.layers.0.self_attn.wo_a")
        )

    def test_streaming_dequant_passes_bf16_wo_a_and_dequants_fp8_wo_a(self):
        from sglang.srt.models import deepseek_v4 as dsv4

        bf16_w = torch.randn(128, 128, dtype=torch.bfloat16)
        other_w = torch.randn(16, 16, dtype=torch.bfloat16)
        streamed = list(
            dsv4._dequant_fp8_wo_a_streaming(
                [
                    ("layers.0.attn.wo_a.weight", bf16_w),
                    ("layers.0.attn.q_norm.weight", other_w),
                ]
            )
        )
        self.assertEqual([name for name, _ in streamed], [
            "layers.0.attn.wo_a.weight",
            "layers.0.attn.q_norm.weight",
        ])
        self.assertIs(streamed[0][1], bf16_w)

        # Round-trip through online quant + streaming dequant.
        quant_stream = list(
            quantize_bf16_wo_a_streaming(
                [("layers.0.attn.wo_a.weight", bf16_w)],
                pow2_scale=False,
            )
        )
        self.assertEqual(
            [name for name, _ in quant_stream],
            ["layers.0.attn.wo_a.weight", "layers.0.attn.wo_a.scale"],
        )
        self.assertEqual(quant_stream[0][1].dtype, torch.float8_e4m3fn)
        self.assertEqual(quant_stream[1][1].shape, (1, 1))

        dequant_stream = list(dsv4._dequant_fp8_wo_a_streaming(quant_stream))
        self.assertEqual(len(dequant_stream), 1)
        self.assertEqual(dequant_stream[0][0], "layers.0.attn.wo_a.weight")
        self.assertEqual(dequant_stream[0][1].dtype, torch.bfloat16)
        torch.testing.assert_close(dequant_stream[0][1], bf16_w, rtol=0.1, atol=0.2)


if __name__ == "__main__":
    unittest.main()

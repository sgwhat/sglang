"""DModel DeepSeek-V4-Flash config normalization onto the V4 runtime schema."""

import copy
import unittest

from sglang.srt.configs import dmodel_deepseek_v4 as dmodel
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4]

# The public deepseek-ai/DeepSeek-V4-Flash-Base config.json.
PUBLIC_V4_FLASH_BASE = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "expert_dtype": "fp8",
    "hc_eps": 1e-06,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 512,
    "initializer_range": 0.02,
    "max_position_embeddings": 1048576,
    "model_type": "deepseek_v4",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 43,
    "num_hash_layers": 3,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 1,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 129280,
    "compress_rope_theta": 160000,
    "compress_ratios": _COMPRESS_RATIOS + [0],
}

# A DModel DeepSeek-V4-Flash checkpoint config.json.
DMODEL_V4_FLASH = {
    **{
        k: v
        for k, v in PUBLIC_V4_FLASH_BASE.items()
        if k not in ("quantization_config", "expert_dtype")
    },
    "bos_token_id": None,
    "eos_token_id": None,
    "model_type": "dmodel_deepseek_v4_flash",
    "num_nextn_predict_layers": 0,
    "num_hash_layers": 0,
    "scoring_func": "softmax",
    "compress_rope_theta": 160000.0,
    "compress_ratios": _COMPRESS_RATIOS,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "main": {
            "type": "default",
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.125,
        },
        "compress": {
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
            "rope_theta": 160000.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "partial_rotary_factor": 0.125,
            "attention_factor": 1.0,
        },
    },
    "intermediate_size": 14336,
    "first_k_dense_replace": 3,
    "n_dense_layers": 3,
    "dense_inter_dim": 14336,
    "dense_intermediate_size": 14336,
    "dmodel_router_dtype": "float32",
    "auto_map": {
        "AutoConfig": "configuration_dmodel_deepseek_v4_flash.DModelDeepSeekV4FlashConfig"
    },
    "enable_hyper_connection": True,
    "hc_residual_mode": "identity",
    "enable_attnres": False,
    "csa_stage_mode": "sparse",
    "dsa_indexer_enabled": True,
    "dsa_indexer_use_for_attention": True,
    "layer_types": ["sliding_attention"] * 2
    + ["compressed_sparse_attention", "heavily_compressed_attention"] * 20
    + ["compressed_sparse_attention"],
    "compress_rates": {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    },
}

# Keys where the DModel checkpoint intentionally differs from the public release.
_EXPECTED_DELTAS = {
    "model_type": "dmodel_deepseek_v4_flash",
    "num_hash_layers": 0,
    "num_nextn_predict_layers": 0,
    "scoring_func": "softmax",
}

# Runtime fields the DSV4 model reads directly from every config it serves.
_RUNTIME_FIELDS = {"hc_residual_mode": "sinkhorn", "n_dense_layers": 0}


def _normalized(**overrides):
    values = copy.deepcopy(DMODEL_V4_FLASH)
    values.update(overrides)
    return dmodel.normalize_dmodel_deepseek_v4_config(values)


class TestDModelDeepseekV4Config(CustomTestCase):
    def test_matches_public_v4_schema_except_documented_deltas(self):
        normalized = _normalized()
        for key, public_value in PUBLIC_V4_FLASH_BASE.items():
            if key in ("quantization_config", "compress_ratios", "expert_dtype"):
                continue
            with self.subTest(key=key):
                self.assertEqual(
                    normalized[key], _EXPECTED_DELTAS.get(key, public_value)
                )
        # The public list also carries the MTP layer's ratio.
        self.assertEqual(
            normalized["compress_ratios"], PUBLIC_V4_FLASH_BASE["compress_ratios"][:43]
        )

    def test_dmodel_runtime_fields(self):
        normalized = _normalized()
        self.assertEqual(normalized["n_dense_layers"], 3)
        self.assertEqual(normalized["first_k_dense_replace"], 3)
        self.assertEqual(normalized["hc_residual_mode"], "identity")
        self.assertIs(normalized["router_fp32"], True)
        for key in ("layer_types", "compress_rates", "auto_map"):
            self.assertNotIn(key, normalized)

    def test_rope_matches_public_v4(self):
        class PublicV4(dmodel.DeepseekV3Config):
            model_type = "deepseek_v4"

        public = PublicV4(**copy.deepcopy(PUBLIC_V4_FLASH_BASE))
        config = dmodel.DModelDeepseekV4FlashConfig(**copy.deepcopy(DMODEL_V4_FLASH))
        self.assertEqual(config.rope_parameters, public.rope_parameters)
        self.assertEqual(config.compress_rope_theta, public.compress_rope_theta)
        self.assertEqual(config.model_type, "dmodel_deepseek_v4_flash")
        self.assertEqual(config.to_dict()["model_type"], "dmodel_deepseek_v4_flash")

    def test_default_instance_keeps_stock_runtime_fields(self):
        config = dmodel.DModelDeepseekV4FlashConfig()
        for name, value in _RUNTIME_FIELDS.items():
            self.assertEqual(getattr(config, name), value)
        dmodel_config = dmodel.DModelDeepseekV4FlashConfig(
            **copy.deepcopy(DMODEL_V4_FLASH)
        )
        self.assertEqual(dmodel_config.to_diff_dict()["n_dense_layers"], 3)

    def test_rejects_unsupported_variants(self):
        compress = DMODEL_V4_FLASH["rope_scaling"]["compress"]
        for overrides in (
            {"csa_stage_mode": "dense"},
            {"dsa_indexer_use_for_attention": False},
            {"enable_attnres": True},
            {"enable_hyper_connection": False},
            {"hc_residual_mode": "attnres"},
            {"head_dim": 128},
            {"compress_ratios": [0, 0] + [4, 64] * 20 + [4]},
            {"compress_ratios": [0, 0, 4]},
            {"num_nextn_predict_layers": 1},
            {"num_hash_layers": 3},
            {"scoring_func": "sigmoid"},
            {"first_k_dense_replace": 2},
            {"dense_intermediate_size": 12288},
            {"dmodel_router_dtype": "float16"},
            {
                "rope_scaling": {
                    "main": {"rope_type": "yarn", "factor": 4.0},
                    "compress": compress,
                }
            },
            {
                "rope_scaling": {
                    "main": DMODEL_V4_FLASH["rope_scaling"]["main"],
                    "compress": {**compress, "type": "default", "rope_type": "default"},
                }
            },
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "Unsupported dmodel_deepseek_v4"),
            ):
                _normalized(**overrides)


class TestDeepseekV4RuntimeFields(CustomTestCase):
    def test_every_v4_runtime_config_carries_the_stock_defaults(self):
        """The DSV4 model reads these fields directly, so a config class that
        reaches it without them fails at construction."""
        from sglang.srt.hf_transformers.common import _CONFIG_REGISTRY

        alias = _CONFIG_REGISTRY["deepseek_v4"]
        for config_cls in (alias, DeepseekV41Config):
            for name, value in _RUNTIME_FIELDS.items():
                with self.subTest(config=config_cls.__name__, field=name):
                    self.assertEqual(getattr(config_cls, name), value)
        schema = DeepSeekV4Config.__dataclass_fields__
        for name, value in _RUNTIME_FIELDS.items():
            self.assertEqual(schema[name].default, value)

    def test_dmodel_config_mirrors_the_v4_alias(self):
        """The DModel class subclasses the V3 config, not the V4 alias, so every
        field the alias adds must be repeated on it."""
        from sglang.srt.hf_transformers.common import _CONFIG_REGISTRY

        config_cls = _CONFIG_REGISTRY["dmodel_deepseek_v4_flash"]
        self.assertIs(config_cls, dmodel.DModelDeepseekV4FlashConfig)
        alias = _CONFIG_REGISTRY["deepseek_v4"]
        for name, value in vars(alias).items():
            if name.startswith("_") or name == "model_type" or callable(value):
                continue
            with self.subTest(field=name):
                self.assertEqual(getattr(config_cls, name), value)


if __name__ == "__main__":
    unittest.main()

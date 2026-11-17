"""Identity-mHC dispatch: Triton op by op or the eager reference, never a
silent fallback. The Triton kernels are replaced by a fake module, so it runs
on CPU."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common import identity_mhc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

HC, HIDDEN, TOKENS, EPS = 4, 128, 3, 1e-6
_KERNELS = "sglang.kernels.ops.layernorm.ihc"


def _inputs():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(TOKENS, HC, HIDDEN, generator=gen).to(torch.bfloat16)
    out = torch.randn(TOKENS, HIDDEN, generator=gen).to(torch.bfloat16)
    fn = torch.randn(2 * HC, HC * HIDDEN, generator=gen) * 0.01
    scale = torch.rand(2, generator=gen) + 0.5
    base = torch.randn(2 * HC, generator=gen) * 0.1
    return x, out, fn, scale, base


def _eager_pre(x, fn, scale, base):
    return identity_mhc.hc_pre_identity_torch(
        x, fn, scale, base, rms_eps=EPS, hc_eps=EPS
    )


class TestIdentityMhcDispatch(CustomTestCase):
    def setUp(self):
        self.calls = []
        kernels = types.ModuleType(_KERNELS)
        kernels.ihc_pre = self._record(
            "pre", lambda x, fn, scale, base, **kw: _eager_pre(x, fn, scale, base)
        )
        kernels.ihc_post = self._record("post", identity_mhc.hc_post_identity_torch)
        patch = mock.patch.dict(sys.modules, {_KERNELS: kernels})
        patch.start()
        self.addCleanup(patch.stop)

    def _record(self, op, impl):
        def run(*args, **kwargs):
            self.calls.append((op, kwargs))
            return impl(*args, **kwargs)

        return run

    def test_eager_backend_never_calls_the_kernels(self):
        x, out, fn, scale, base = _inputs()
        norm = SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=EPS)
        with envs.SGLANG_OPT_IHC_BACKEND.override("eager"):
            config = identity_mhc.load_identity_mhc_config()
        y, post, norm_fused = identity_mhc.identity_pre(
            x, fn, scale, base, norm=norm, rms_eps=EPS, hc_eps=EPS, config=config
        )
        identity_mhc.identity_post(out, x, post, config=config)
        self.assertEqual(self.calls, [])
        self.assertFalse(norm_fused)
        ref_y, ref_post = _eager_pre(x, fn, scale, base)
        torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
        torch.testing.assert_close(post, ref_post, rtol=0, atol=0)

    def test_triton_backend_fuses_the_norm_and_passes_the_tiling(self):
        x, out, fn, scale, base = _inputs()
        weight = torch.ones(HIDDEN)
        norm = SimpleNamespace(weight=weight, variance_epsilon=EPS)
        config = identity_mhc.IdentityMhcConfig(
            backend="triton", pre_tiled_min_tokens=1024, pre_bf16x3=True
        )
        _, post, norm_fused = identity_mhc.identity_pre(
            x, fn, scale, base, norm=norm, rms_eps=EPS, hc_eps=EPS, config=config
        )
        identity_mhc.identity_post(out, x, post, config=config)
        self.assertTrue(norm_fused)
        (op, pre_kwargs), (post_op, _) = self.calls
        self.assertEqual((op, post_op), ("pre", "post"))
        self.assertIs(pre_kwargs["norm_weight"], weight)
        self.assertEqual(pre_kwargs["tiled_min_tokens"], 1024)
        self.assertTrue(pre_kwargs["bf16x3"])
        # DModel's post gate carries no eps, unlike Hunyuan-V4's.
        self.assertEqual(pre_kwargs["post_eps"], 0.0)
        self.assertEqual(pre_kwargs["post_mult"], 2.0)

    def test_empty_batches_take_the_reference(self):
        x, out, fn, scale, base = _inputs()
        config = identity_mhc.IdentityMhcConfig(backend="triton")
        y, post, norm_fused = identity_mhc.identity_pre(
            x[:0], fn, scale, base, norm=None, rms_eps=EPS, hc_eps=EPS, config=config
        )
        identity_mhc.identity_post(out[:0], x[:0], post, config=config)
        self.assertEqual(self.calls, [])
        self.assertEqual((tuple(y.shape), norm_fused), ((0, HIDDEN), False))

    def test_kernel_errors_propagate(self):
        """A failing kernel must not degrade the process to another backend."""
        x, out, fn, scale, base = _inputs()
        sys.modules[_KERNELS].ihc_post = mock.Mock(side_effect=RuntimeError("boom"))
        config = identity_mhc.IdentityMhcConfig(backend="triton")
        _, post = _eager_pre(x, fn, scale, base)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                identity_mhc.identity_post(out, x, post, config=config)

    def test_load_config_reads_the_flags(self):
        with (
            envs.SGLANG_OPT_IHC_PRE_TILED.override(False),
            envs.SGLANG_OPT_IHC_PRE_TILED_MIN_TOKENS.override(64),
        ):
            self.assertIsNone(
                identity_mhc.load_identity_mhc_config().pre_tiled_min_tokens
            )
        with (
            envs.SGLANG_OPT_IHC_PRE_TILED.override(True),
            envs.SGLANG_OPT_IHC_PRE_TILED_MIN_TOKENS.override(64),
        ):
            self.assertEqual(
                identity_mhc.load_identity_mhc_config().pre_tiled_min_tokens, 64
            )
        with (
            envs.SGLANG_OPT_IHC_BACKEND.override("hpc"),
            self.assertRaisesRegex(ValueError, "SGLANG_OPT_IHC_BACKEND"),
        ):
            identity_mhc.load_identity_mhc_config()


class TestIdentityMhcSemantics(CustomTestCase):
    def test_post_with_zero_output_keeps_every_stream(self):
        """The comb is the identity: a zero sublayer output leaves the streams."""
        x, out, fn, scale, base = _inputs()
        _, post = _eager_pre(x, fn, scale, base)
        torch.testing.assert_close(
            identity_mhc.hc_post_identity_torch(torch.zeros_like(out), x, post),
            x,
            rtol=0,
            atol=0,
        )

    def test_mixing_params_follow_the_checkpoint_layout(self):
        """Identity checkpoints carry [pre; post] rows and two scales only."""
        from sglang.srt.models.deepseek_v4 import make_hc_mixing_params

        for identity, rows, scales in ((True, 2 * HC, 2), (False, (2 + HC) * HC, 3)):
            attn_fn, ffn_fn, attn_base, ffn_base, attn_scale, ffn_scale = (
                make_hc_mixing_params(HC, HIDDEN, identity=identity)
            )
            with self.subTest(identity=identity):
                for fn in (attn_fn, ffn_fn):
                    self.assertEqual(tuple(fn.shape), (rows, HC * HIDDEN))
                for base in (attn_base, ffn_base):
                    self.assertEqual(tuple(base.shape), (rows,))
                for scale in (attn_scale, ffn_scale):
                    self.assertEqual(tuple(scale.shape), (scales,))

    def test_unsupported_layouts_raise(self):
        config = SimpleNamespace(hc_pre_from_prev_sublayer=False)
        identity_mhc.check_identity_residual_supported(
            config, is_cuda=True, is_nextn=False
        )
        for kwargs, message in (
            (dict(is_cuda=False, is_nextn=False), "non-CUDA"),
            (dict(is_cuda=True, is_nextn=True), "draft layers"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(NotImplementedError, message),
            ):
                identity_mhc.check_identity_residual_supported(config, **kwargs)
        with self.assertRaisesRegex(NotImplementedError, "pre-mix"):
            identity_mhc.check_identity_residual_supported(
                SimpleNamespace(hc_pre_from_prev_sublayer=True),
                is_cuda=True,
                is_nextn=False,
            )
        with self.assertRaisesRegex(ValueError, "hc_residual_mode"):
            identity_mhc.is_identity_residual(
                SimpleNamespace(hc_residual_mode="attnres")
            )


if __name__ == "__main__":
    unittest.main()

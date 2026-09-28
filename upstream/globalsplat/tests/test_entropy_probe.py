"""CPU diagnostics tests; also runnable without pytest via unittest discover."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec
from globalsplat.compression.entropy_probe import Table, HistogramFit, EntropyProbe, context_bins


class EntropyProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rans_tail_accounting_and_roundtrip(self):
        table = Table([[0, 16384, 49152, 65536]], [4], [0])
        symbols = np.array([-2, -1, 0, 1, 2, 3, 19, 4096])
        indexes = np.zeros_like(symbols)
        costs = table.costs(symbols, indexes)
        self.assertEqual(costs["escape_symbols"], 6)
        self.assertEqual(costs["bypass_bits"], 8 + 8 + 4 + 8 + 12 + 20)
        payload = table.encode(symbols, indexes)
        np.testing.assert_array_equal(table.decode(payload, indexes), symbols)
        gap = len(payload) * 8 - costs["cdf_bits"] - costs["bypass_bits"]
        self.assertGreaterEqual(gap, 0)
        self.assertLess(gap, 96)

    def test_conditional_fit_generalizes_and_refuses_test_updates(self):
        base = Table([[0, 12000, 53536, 65535, 65536]], [5], [-1])
        fit = HistogramFit(base, bins=2, prior=8)
        symbols = np.tile([-1, 1], 2000)
        bins = np.tile([0, 1], 2000)
        indexes = np.zeros_like(symbols)
        fit.add(symbols, indexes, bins)
        fit.freeze()
        test_symbols, test_bins = symbols[:800], bins[:800]
        marginal, mi = fit.select("marginal", indexes[:800], test_bins)
        conditional, ci = fit.select("context", indexes[:800], test_bins)
        marginal_bytes = marginal.encode(test_symbols, mi)
        context_bytes = conditional.encode(test_symbols, ci)
        self.assertLess(len(context_bytes), len(marginal_bytes) / 2)
        np.testing.assert_array_equal(conditional.decode(context_bytes, ci), test_symbols)
        with self.assertRaises(RuntimeError):
            fit.add(symbols, indexes, bins)

    def test_empty_fit_keeps_parent_cdf(self):
        base = Table([[0, 16384, 49152, 65536]], [4], [0])
        fit = HistogramFit(base, bins=5)
        fit.freeze()
        self.assertEqual(fit.tables["marginal"].cdfs, base.cdfs)
        self.assertEqual(fit.tables["context"].cdfs, base.cdfs * 5)

    def test_residual_context_candidate_is_exactly_marginal(self):
        base = Table([[0, 16384, 49152, 65536]], [4], [0])
        fit = HistogramFit(base, bins=1, prior=16)
        symbols = np.zeros(128, dtype=np.int64)
        fit.add(symbols, symbols, symbols)
        fit.freeze()
        self.assertNotEqual(fit.tables["marginal"].cdfs, base.cdfs)
        self.assertEqual(fit.tables["context"].export(), fit.tables["marginal"].export())

    def test_context_uses_even_anchors_and_handles_final_odd(self):
        step = torch.ones(1, 1, 1, 1)
        base = torch.zeros(1, 1, 1, 6)
        anchors = torch.tensor([[[[0., 2., 8.]]]])
        bins = context_bins(step, (1, 1, 1, 3), anchors, base)
        np.testing.assert_array_equal(bins, [1, 4, 4])
        # Undecoded odd base values are not read by this context function.
        base[..., 1::2] = 999
        np.testing.assert_array_equal(context_bins(step, (1, 1, 1, 3), anchors, base), bins)

    @staticmethod
    def codec():
        torch.manual_seed(91)
        cfg = CodecConfig(texture_channels=4, geometry_channels=4,
                          geometry_observable_channels=4, rank=4,
                          residual_n=4, residual_m=6, adapter_hidden=6,
                          transform="nonlinear", transform_hidden=4,
                          score_mean_condition=True, score_channel_context=True,
                          score_spatial_context=True, score_spatial_entropy="split",
                          score_slice_channels=2, score_context_hidden=4)
        codec = ObservableLowRank1DCodec(cfg).eval()
        codec.update(force=True, update_quantiles=False)
        return codec

    def test_full_feature_roundtrip_frozen_model_and_wrapper_restoration(self):
        codec = self.codec()
        original_state = {k: v.clone() for k, v in codec.state_dict().items()}
        inputs = tuple(torch.randn(1, 32, c) for c in (4, 4, 3))
        with torch.no_grad():
            original = codec.compress(*inputs).data
            with EntropyProbe(codec, prior=32) as probe:
                for _ in range(3):
                    probe.begin()
                    codec.compress(*(torch.randn(1, 32, c) for c in (4, 4, 3)))
                    probe.end()
                    probe.fit_scene()
                probe.freeze()
                probe.begin()
                captured = codec.compress(*inputs).data
                probe.end()
                self.assertEqual(original, captured)
                for name, record in probe.records.items():
                    native = probe.fits[name].base.encode(record["symbols"], record["indexes"])
                    self.assertEqual(native, record["native"], name)
                    costs = record["stats"]
                    gap = costs["actual_bits"] - costs["cdf_bits"] - costs["bypass_bits"]
                    self.assertGreaterEqual(gap, -1e-5, name)
                    self.assertLess(gap, 96, name)
                for variant in ("marginal", "context"):
                    payload, _ = probe.recode(captured, variant)
                    self.assertTrue(probe.verify(captured, payload, variant))
            self.assertEqual(codec.compress(*inputs).data, original)
        for name, value in codec.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[name]), name)
        self.assertNotIn("_scene_parameters", codec.score_context.__dict__)

    def test_exception_restores_methods(self):
        codec = self.codec()
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with EntropyProbe(codec):
                raise RuntimeError("test exception")
        self.assertNotIn("_scene_parameters", codec.score_context.__dict__)
        self.assertNotIn("compress", codec.residual_codec.gaussian_conditional.__dict__)

    def test_nonzero_context_odd_length_and_offline_cache(self):
        codec = self.codec()
        with torch.no_grad():
            ctx = codec.score_context
            ctx.mean_conditioner[-1].weight.normal_(std=0.15)
            ctx.mean_conditioner[-1].bias.normal_(std=0.3)
            for predictor in (*ctx.channel_predictors, *ctx.spatial_predictors):
                predictor.weight.normal_(std=0.1)
                predictor.bias.normal_(std=0.2)
            for entropy in (*ctx.group_entropies, *ctx.spatial_odd_entropies):
                entropy.entropy_bottleneck.quantiles[:, :, 1].fill_(0.375)
            codec.update(force=True, update_quantiles=False)
            with EntropyProbe(codec, prior=16) as probe, TemporaryDirectory() as tmp:
                probe.begin()
                original = codec.compress(*(torch.randn(1, 31, c) for c in (4, 4, 3))).data
                probe.end()
                probe.fit_scene()
                probe.freeze()
                probe.begin()
                original = codec.compress(*(torch.randn(1, 31, c) for c in (4, 4, 3))).data
                probe.end()
                for variant in ("marginal", "context"):
                    recoded, _ = probe.recode(original, variant)
                    self.assertTrue(probe.verify(original, recoded, variant))
                path = Path(tmp) / "scene.npz"
                probe.save_scene_cache(path, original)
                with np.load(path, allow_pickle=False) as cache:
                    self.assertEqual(cache["native_scene_bytes"].tobytes(), original)
                    for name, record in probe.records.items():
                        symbols, indexes = cache[name + "__symbols"], cache[name + "__indexes"]
                        np.testing.assert_array_equal(cache[name + "__shape"], record["shape"])
                        native = Table(**probe.export_tables()[name]["native"])
                        self.assertEqual(native.encode(symbols, indexes), record["native"])
                with self.assertRaises(FileExistsError):
                    probe.save_scene_cache(path, original)

    def test_invalid_histogram_prior_is_rejected(self):
        table = Table([[0, 16384, 49152, 65536]], [4], [0])
        for prior in (0, -1, float("nan"), float("inf")):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                HistogramFit(table, bins=5, prior=prior)


if __name__ == "__main__":
    unittest.main()

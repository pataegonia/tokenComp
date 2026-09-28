"""Compare the main codec against the unmodified pre-cleanup implementation."""

from dataclasses import replace
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec
from globalsplat.compression.bitstream import SceneBitstream, ScoreContextBitstream
from globalsplat.compression.checkpoint import (
    infer_config,
    load_feature_codec_checkpoint,
    reset_score_mean_offset_head,
    validate_feature_codec_checkpoint,
    validate_score_mean_offset_mode,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "archive/codec_experiments_20260915/globalsplat/compression"
spec = importlib.util.spec_from_file_location(
    "nfcgs_pre_cleanup_reference",
    REFERENCE / "__init__.py",
    submodule_search_locations=[str(REFERENCE)],
)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)


def small_config():
    return CodecConfig(
        texture_channels=8,
        geometry_channels=8,
        geometry_observable_channels=4,
        rank=7,
        residual_n=6,
        residual_m=8,
        adapter_hidden=6,
        transform_hidden=5,
        score_slice_channels=3,
        score_context_hidden=5,
    )


def reference_pair():
    config = small_config()
    torch.manual_seed(101)
    old = reference.ObservableLowRank1DCodec(reference.CodecConfig(**config.to_dict()))
    torch.manual_seed(101)
    new = ObservableLowRank1DCodec(config)
    # Exercise trained/nonzero predictors, not just their zero initialization.
    with torch.no_grad():
        for module in (
            old.analysis_mlp[-1],
            old.synthesis_mlp[-1],
            old.score_context.mean_conditioner[-1],
            *old.score_context.channel_predictors,
            *old.score_context.spatial_predictors,
        ):
            module.weight.normal_(0, 0.04)
            if module.bias is not None:
                module.bias.normal_(0, 0.02)
        old.score_log_scale.uniform_(-0.3, 0.3)
        old.residual_mean.normal_(0, 0.02)
        old.residual_std.uniform_(0.8, 1.2)
    new.load_state_dict(old.state_dict(), strict=True)
    return old, new


class MainCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_config_is_main_and_retired_modes_fail_explicitly(self):
        config = CodecConfig()
        self.assertEqual((config.rank, config.transform_hidden), (56, 32))
        for key, value in {
            "transform": "linear",
            "use_residual": False,
            "use_morton": False,
            "score_mean_condition": False,
            "score_channel_context": False,
            "score_spatial_context": False,
            "score_spatial_predictor": "residual7",
            "score_spatial_entropy": "shared",
        }.items():
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "main codec requires"),
            ):
                replace(config, **{key: value})
        for mode in ("gaussian", "conditional_scale"):
            with self.assertRaises(ValueError):
                replace(config, score_spatial_entropy=mode)

    def test_initial_weights_and_parameter_order_match_reference(self):
        config = small_config()
        torch.manual_seed(107)
        old = reference.ObservableLowRank1DCodec(
            reference.CodecConfig(**config.to_dict())
        )
        torch.manual_seed(107)
        new = ObservableLowRank1DCodec(config)
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        for scope in ("all", "score_probability"):
            old.set_trainable_scope(scope)
            new.set_trainable_scope(scope)
            self.assertEqual(
                [n for n, p in old.named_parameters() if p.requires_grad],
                [n for n, p in new.named_parameters() if p.requires_grad],
            )

    def test_bitstreams_features_and_likelihoods_are_unchanged(self):
        old, new = reference_pair()
        old.eval()
        new.eval()
        old.update(force=True)
        new.update(force=True)
        for points in (32, 33, 64):
            inputs = tuple(torch.randn(1, points, channels) for channels in (8, 8, 3))
            with self.subTest(points=points), torch.no_grad():
                before = old(*inputs, restore_original_order=False)
                after = new(*inputs, restore_original_order=False)
                torch.testing.assert_close(
                    before.reconstruction_sorted,
                    after.reconstruction_sorted,
                    rtol=0,
                    atol=0,
                )
                for name in before.likelihoods:
                    torch.testing.assert_close(
                        before.likelihoods[name],
                        after.likelihoods[name],
                        rtol=0,
                        atol=0,
                    )
                payload = old.compress(*inputs).data
                self.assertEqual(payload, new.compress(*inputs).data)
                old_features = torch.cat(old.decompress(payload), -1)
                new_features = torch.cat(new.decompress(payload), -1)
                torch.testing.assert_close(old_features, new_features, rtol=0, atol=0)
                torch.testing.assert_close(
                    new_features, after.reconstruction_sorted, rtol=1e-5, atol=2e-6
                )
                scene = SceneBitstream.unpack(payload)
                self.assertEqual(sum(scene.bytes_by_stream.values()), len(payload))
                self.assertEqual(
                    len(ScoreContextBitstream.unpack(scene.score).strings), 6
                )

    def test_training_loss_gradients_and_optimizer_resume_match(self):
        old, new = reference_pair()
        old.train()
        new.train()
        inputs = tuple(torch.randn(2, 32, c) for c in (8, 8, 3))
        for model in (old, new):
            torch.manual_seed(113)
            output = model(*inputs)
            (
                output.reconstruction_sorted.square().mean()
                + 1e-4 * output.estimated_bits
            ).backward()
        old_parameters = dict(old.named_parameters())
        for name, parameter in new.named_parameters():
            expected = old_parameters[name].grad
            if expected is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
        old_opt = torch.optim.Adam(
            [p for p in old.parameters() if p.requires_grad], lr=1e-4
        )
        old_opt.step()
        new_opt = torch.optim.Adam(
            [p for p in new.parameters() if p.requires_grad], lr=1e-4
        )
        new_opt.load_state_dict(old_opt.state_dict())
        self.assertEqual(len(new_opt.state), len(old_opt.state))

    def test_bfloat16_sender_and_receiver_match_reference(self):
        old, new = reference_pair()
        old.eval()
        new.eval()
        old.update(force=True)
        new.update(force=True)
        inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            payload = old.compress(*inputs).data
            self.assertEqual(payload, new.compress(*inputs).data)
            torch.testing.assert_close(
                torch.cat(old.decompress(payload), -1),
                torch.cat(new.decompress(payload), -1),
                rtol=0,
                atol=0,
            )

    def test_score_context_b_diagnostics_match_actual_coding(self):
        _, codec = reference_pair()
        codec.eval()
        codec.update(force=True)
        inputs = tuple(torch.randn(1, 33, channels) for channels in (8, 8, 3))
        with torch.no_grad():
            baseline = codec.compress(*inputs).data
            self.assertIsNone(codec.score_context.last_compress_diagnostics)
            codec.score_context.capture_diagnostics = True
            self.assertEqual(codec.compress(*inputs).data, baseline)
            stats = codec.score_context.last_compress_diagnostics
        self.assertEqual(len(stats["mean_offset_b"]), codec.config.rank)
        self.assertEqual(len(stats["quantization_step_d"]), codec.config.rank)
        for parity, count in (("even", 17), ("odd", 16)):
            group = stats[f"effective_b_{parity}"]
            self.assertEqual(group["count"], [count] * codec.config.rank)
            for key in ("first_token_b", "sum", "sum_squares", "sum_abs", "min", "max"):
                self.assertEqual(len(group[key]), codec.config.rank)
            for name in (
                "score", "mean_centered_score", "centered_score", "entropy_input"
            ):
                measured = stats[f"{name}_{parity}"]
                self.assertEqual(measured["count"], [count] * codec.config.rank)
                for key in ("sum", "sum_squares", "sum_abs", "min", "max"):
                    self.assertEqual(len(measured[key]), codec.config.rank)
            for channel in range(codec.config.rank):
                target_sum = stats[f"score_{parity}"]["sum"][channel]
                centered_sum = stats[f"centered_score_{parity}"]["sum"][channel]
                base_sum = group["sum"][channel]
                self.assertAlmostEqual(target_sum - base_sum, centered_sum, places=4)
                step = stats["quantization_step_d"][channel]
                symbol_sum = stats[f"entropy_input_{parity}"]["sum"][channel]
                self.assertAlmostEqual(centered_sum / step, symbol_sum, places=4)
                if parity == "even" and channel < codec.score_context.group_sizes[0]:
                    mean_centered = stats["mean_centered_score_even"]["sum"][channel]
                    self.assertAlmostEqual(centered_sum, mean_centered, places=4)
        first_group = codec.score_context.group_sizes[0]
        self.assertEqual(
            stats["effective_b_even"]["first_token_b"][:first_group],
            stats["mean_offset_b"][:first_group],
        )

    def test_mean_offset_ablation_preserves_d_and_marks_bitstream(self):
        _, codec = reference_pair()
        codec.eval()
        codec.update(force=True)
        inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
        scene_mean = torch.randn(1, codec.config.observable_channels)
        with torch.no_grad():
            before_b, before_d = codec.score_context._scene_parameters(scene_mean)
            baseline = codec.compress(*inputs).data
            codec.score_context.mean_offset_enabled = False
            after_b, after_d = codec.score_context._scene_parameters(scene_mean)
            codec.score_context.capture_diagnostics = True
            ablated = codec.compress(*inputs).data
            stats = codec.score_context.last_compress_diagnostics
            decoded = codec.decompress(ablated)
        self.assertGreater(before_b.abs().max().item(), 0)
        self.assertEqual(after_b.count_nonzero().item(), 0)
        torch.testing.assert_close(after_d, before_d, rtol=0, atol=0)
        self.assertEqual(stats["mean_offset_b"], [0.0] * codec.config.rank)
        self.assertEqual(torch.cat(decoded, dim=-1).shape[-1], codec.config.observable_channels)
        self.assertNotEqual(baseline, ablated)
        base_flags = ScoreContextBitstream.unpack(SceneBitstream.unpack(baseline).score).flags
        ablated_flags = ScoreContextBitstream.unpack(SceneBitstream.unpack(ablated).score).flags
        self.assertEqual(
            ablated_flags, base_flags | codec.score_context.FLAG_NO_MEAN_OFFSET
        )
        codec.score_context.mean_offset_enabled = True
        with self.assertRaisesRegex(ValueError, "does not match codec configuration"):
            codec.decompress(ablated)
        validate_score_mean_offset_mode({}, codec)
        with self.assertRaisesRegex(ValueError, "score_mean_offset_enabled"):
            validate_score_mean_offset_mode({"score_mean_offset_enabled": False}, codec)
        validate_score_mean_offset_mode(
            {"score_mean_offset_enabled": False}, codec, allow_conversion=True
        )
        reset_score_mean_offset_head(codec)
        reset_b, reset_d = codec.score_context._scene_parameters(scene_mean)
        self.assertEqual(reset_b.count_nonzero().item(), 0)
        torch.testing.assert_close(reset_d, before_d, rtol=0, atol=0)
        codec.score_context.mean_offset_enabled = False
        codec.zero_grad(set_to_none=True)
        _, train_d = codec.score_context._scene_parameters(scene_mean)
        train_d.sum().backward()
        head_grad = codec.score_context.mean_conditioner[-1].weight.grad
        self.assertEqual(head_grad[: codec.config.rank].count_nonzero().item(), 0)
        self.assertGreater(head_grad[codec.config.rank :].abs().sum().item(), 0)
        codec.eval()
        control = copy.deepcopy(codec)
        control.score_context.mean_offset_enabled = True
        with torch.no_grad():
            control_output = control(*inputs)
            ablated_output = codec(*inputs)
        torch.testing.assert_close(
            control_output.reconstruction_sorted,
            ablated_output.reconstruction_sorted,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            control_output.estimated_bits,
            ablated_output.estimated_bits,
            rtol=0,
            atol=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no_mean_offset.ckpt"
            torch.save(
                {
                    "state_dict": {
                        f"model.feature_codec.{key}": value
                        for key, value in codec.state_dict().items()
                    },
                    "feature_codec_config": codec.config.to_dict(),
                    "score_mean_offset_enabled": False,
                },
                path,
            )
            loaded = load_feature_codec_checkpoint(path)
            self.assertFalse(loaded.codec.score_context.mean_offset_enabled)

    def test_probability_step_keeps_reconstruction_and_quantiles_fixed(self):
        _, model = reference_pair()
        model.set_trainable_scope("score_probability")
        inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
        model.eval()
        with torch.no_grad():
            before = model(*inputs).reconstruction_sorted.clone()
        quantiles = {
            n: p.clone() for n, p in model.named_parameters() if "quantiles" in n
        }
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        model.train()
        model(*inputs).estimated_bits.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model(*inputs).reconstruction_sorted, before, rtol=0, atol=0
            )
        for name, value in quantiles.items():
            torch.testing.assert_close(
                dict(model.named_parameters())[name], value, rtol=0, atol=0
            )

    def test_reference_checkpoint_strict_load_and_inference(self):
        old, _ = reference_pair()
        old.eval()
        old.update(force=True)
        checkpoint = {
            "state_dict": {
                f"model.feature_codec.{k}": v for k, v in old.state_dict().items()
            },
            "feature_codec_config": old.config.to_dict(),
            "global_step": 10000,
        }
        config = small_config()
        validate_feature_codec_checkpoint(checkpoint, config)
        self.assertEqual(infer_config(old.state_dict()), config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.ckpt"
            torch.save(checkpoint, path)
            loaded = load_feature_codec_checkpoint(path)
            self.assertEqual(loaded.config, config)
            inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
            self.assertEqual(
                old.compress(*inputs).data, loaded.codec.eval().compress(*inputs).data
            )
        corrupt = dict(checkpoint["state_dict"])
        del corrupt[
            "model.feature_codec.score_context.spatial_odd_entropies.0.entropy_bottleneck.quantiles"
        ]
        with self.assertRaisesRegex(ValueError, "state mismatch"):
            validate_feature_codec_checkpoint(
                {**checkpoint, "state_dict": corrupt}, config
            )

    def test_decoder_rejects_retired_bitstreams(self):
        _, codec = reference_pair()
        codec.eval()
        codec.update(force=True)
        inputs = tuple(torch.randn(1, 32, c) for c in (8, 8, 3))
        scene = SceneBitstream.unpack(codec.compress(*inputs).data)
        for flag in (0, scene.flags | 1, scene.flags | 2):
            with self.assertRaises(ValueError):
                codec.decompress(replace(scene, flags=flag).pack())
        score = ScoreContextBitstream.unpack(scene.score)
        with self.assertRaises(ValueError):
            codec.decompress(
                replace(scene, score=replace(score, flags=7).pack()).pack()
            )


if __name__ == "__main__":
    unittest.main()

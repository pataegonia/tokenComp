"""Score switches must agree in training, standalone decoding and checkpoints."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec
from globalsplat.compression.bitstream import SceneBitstream, ScoreContextBitstream
from globalsplat.compression.checkpoint import (
    load_feature_codec_checkpoint, validate_feature_codec_checkpoint,
)


def config(**kwargs):
    return CodecConfig(
        texture_channels=8, geometry_channels=8, geometry_observable_channels=4,
        rank=7, residual_n=6, residual_m=8, adapter_hidden=6,
        transform_hidden=5, score_slice_channels=3, score_context_hidden=5, **kwargs,
    )


MINIMAL = dict(use_centering=False, use_score_norm=False, score_mean_condition=False,
               score_channel_context=False, transform="linear")


class ScorePathAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_switches_round_trip_and_reject_wrong_decoder(self):
        cases = [dict(use_score_norm=False), dict(transform="linear"),
                 dict(score_mean_condition=False), dict(score_channel_context=False),
                 dict(use_centering=False, score_mean_condition=False), MINIMAL]
        for switches in cases:
            torch.manual_seed(193)
            full = ObservableLowRank1DCodec(config()).eval()
            with torch.no_grad():
                # Use trained-like values so bypasses are observable.
                for name, p in full.named_parameters():
                    if any(tag in name for tag in ("mlp.", "mean_conditioner", "channel_predictors", "spatial_predictors")):
                        p.normal_(0, 0.1)
                full.score_log_scale.fill_(0.4)
            codec = ObservableLowRank1DCodec(config(**switches)).eval()
            codec.load_state_dict(full.state_dict(), strict=True)
            codec.update(force=True)
            for points in (32, 33):
                inputs = tuple(torch.randn(1, points, n) for n in (8, 8, 3))
                with self.subTest(switches=switches, points=points), torch.no_grad():
                    predicted = codec(*inputs, restore_original_order=False)
                    payload = codec.compress(*inputs).data
                    receiver = copy.deepcopy(codec)
                    reconstructed = torch.cat(receiver.decompress(payload), -1)
                    torch.testing.assert_close(reconstructed, predicted.reconstruction_sorted, atol=3e-6, rtol=1e-5)
                    scene = SceneBitstream.unpack(payload)
                    self.assertEqual(len(scene.mean_fp16), 24 if codec.config.use_centering else 0)
                    self.assertTrue(scene.residual)
                    self.assertEqual(sum(scene.bytes_by_stream.values()), len(payload))
                    flags = ScoreContextBitstream.unpack(scene.score).flags
                    self.assertEqual(bool(flags & codec.score_context.FLAG_MEAN), codec.config.score_mean_condition)
                    self.assertEqual(bool(flags & codec.score_context.FLAG_CHANNEL), codec.config.score_channel_context)
                    self.assertTrue(flags & codec.score_context.FLAG_SPATIAL)
                    self.assertTrue(flags & codec.score_context.FLAG_SPATIAL_ENTROPY_SPLIT)
                    with self.assertRaisesRegex(ValueError, "does not match codec"):
                        full.decompress(payload)

    def test_disabled_parameters_never_execute_or_train_and_msh_is_retained(self):
        torch.manual_seed(211)
        full = ObservableLowRank1DCodec(config())
        torch.manual_seed(211)
        codec = ObservableLowRank1DCodec(config(**MINIMAL))
        for key, value in full.residual_codec.state_dict().items():
            torch.testing.assert_close(value, codec.residual_codec.state_dict()[key], rtol=0, atol=0)
        codec.set_trainable_scope("all")
        inactive = ("analysis_mlp.", "synthesis_mlp.", "score_log_scale",
                    "score_context.mean_conditioner.", "score_context.channel_predictors.")
        with torch.no_grad():
            for name, parameter in codec.named_parameters():
                if name.startswith(inactive):
                    self.assertFalse(parameter.requires_grad)
                    parameter.fill_(float("nan"))
        inputs = tuple(torch.randn(2, 33, c) + 2 for c in (8, 8, 3))
        output = codec(*inputs)
        loss = output.reconstruction_sorted.square().mean() + 1e-3 * output.estimated_bits + codec.aux_loss()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in codec.named_parameters():
            if name.startswith(inactive):
                self.assertIsNone(parameter.grad)
        for module in (codec.residual_codec, codec.score_context.spatial_predictors,
                       codec.score_context.group_entropies, codec.score_context.spatial_odd_entropies):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_no_mean_or_channel_information_leaks_into_spatial_only_context(self):
        codec = ObservableLowRank1DCodec(config(**MINIMAL)).eval()
        context = codec.score_context
        scores = torch.randn(1, 7, 1, 33)
        with torch.no_grad():
            a, pa = context(scores, torch.randn(1, 12), codec.score_entropy, training=False)
            b, pb = context(scores, torch.randn(1, 12) * 100, codec.score_entropy, training=False)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)
            changed = scores.clone()
            changed[:, :3] += 100
            c, pc = context(changed, torch.zeros(1, 12), codec.score_entropy, training=False)
            torch.testing.assert_close(a[:, 3:], c[:, 3:], rtol=0, atol=0)
            torch.testing.assert_close(pa[:, 3:], pc[:, 3:], rtol=0, atol=0)

    def test_minimal_bfloat16_round_trip_and_no_mean_payload(self):
        codec = ObservableLowRank1DCodec(config(**MINIMAL)).eval()
        codec.update(force=True)
        inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
        sender = {}
        synthesize = codec._synthesize_low_rank
        compress_residual = codec.residual_codec.compress

        def capture_low_rank(score):
            sender["low_rank"] = synthesize(score)
            return sender["low_rank"]

        def capture_residual(residual):
            payload, sender["residual"] = compress_residual(residual)
            return payload, sender["residual"]

        receiver = copy.deepcopy(codec)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            with patch.object(codec, "_synthesize_low_rank", side_effect=capture_low_rank), patch.object(
                codec.residual_codec, "compress", side_effect=capture_residual
            ):
                payload = codec.compress(*inputs).data
            residual = sender["residual"] * codec.residual_std + codec.residual_mean
            expected = sender["low_rank"] + residual.squeeze(2).transpose(1, 2)
            decoded = torch.cat(receiver.decompress(payload), -1)
        self.assertEqual(SceneBitstream.unpack(payload).mean_fp16, b"")
        # Compare the actual sender's decoded symbols with a separate receiver.
        # Approximate MSH forward under BF16 is not the arithmetic-coder oracle.
        torch.testing.assert_close(expected, decoded, rtol=0, atol=0)

    def test_checkpoint_round_trip_and_explicit_conversion(self):
        codec = ObservableLowRank1DCodec(config(**MINIMAL)).eval()
        codec.update(force=True)
        checkpoint = {"state_dict": {f"model.feature_codec.{k}": v for k, v in codec.state_dict().items()},
                      "feature_codec_config": codec.config.to_dict()}
        validate_feature_codec_checkpoint(checkpoint, codec.config)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            validate_feature_codec_checkpoint(checkpoint, config())
        validate_feature_codec_checkpoint(checkpoint, config(), allow_score_path_conversion=True)
        full_checkpoint = {**checkpoint, "feature_codec_config": config().to_dict()}
        validate_feature_codec_checkpoint(full_checkpoint, codec.config, allow_score_path_conversion=True)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            validate_feature_codec_checkpoint(full_checkpoint, codec.config)
        # A stripped-metadata checkpoint must not silently become an ablation.
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            validate_feature_codec_checkpoint({"state_dict": checkpoint["state_dict"]}, codec.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimal.ckpt"
            torch.save(checkpoint, path)
            loaded = load_feature_codec_checkpoint(path)
            self.assertEqual(loaded.config, codec.config)
            inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
            self.assertEqual(codec.compress(*inputs).data, loaded.codec.eval().compress(*inputs).data)

    def test_mean_payload_validation_and_invalid_mean_dependency(self):
        with self.assertRaisesRegex(ValueError, "no mean is transmitted"):
            config(use_centering=False)
        codec = ObservableLowRank1DCodec(config(**MINIMAL)).eval()
        codec.update(force=True)
        inputs = tuple(torch.randn(1, 32, c) for c in (8, 8, 3))
        scene = SceneBitstream.unpack(codec.compress(*inputs).data)
        with self.assertRaisesRegex(ValueError, "mean payload"):
            codec.decompress(replace(scene, mean_fp16=b"\0\0").pack())


if __name__ == "__main__":
    unittest.main()

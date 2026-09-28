from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfcgs_codec import (  # noqa: E402
    CodecConfig,
    ObservableLowRank1DCodec,
    SceneBitstream,
    load_feature_codec_checkpoint,
    morton_order_3d,
)


class MortonTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        positions = torch.tensor(
            [[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.2, 0.8]]]
        )
        values = torch.randn(1, 3, 7)
        order = morton_order_3d(positions)
        self.assertTrue(torch.equal(order.restore(order.apply(values)), values))


class BitstreamTests(unittest.TestCase):
    def test_scene_container_round_trip_and_authentication(self) -> None:
        original = SceneBitstream(4096, 736, 56, b"mean", b"score", b"residual")
        data = original.pack()
        self.assertEqual(len(data[:80]), 80)
        decoded = SceneBitstream.unpack(data)
        self.assertEqual(decoded, original)
        damaged = bytearray(data)
        damaged[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "authentication"):
            SceneBitstream.unpack(bytes(damaged))

    def test_archived_scene_header(self) -> None:
        relative = (
            "experiments/"
            "shared_lowrank_rank56_untied_nonlinear64_synthesis_multiscale1d/"
            "rate_points/lambda_0p0256/evaluations/ariel/"
            "rank_nonlinearity_followup_50k/remote_followup_16gpu_50k/"
            "offline_50k_export/final_50k/zero_update/bitstreams/"
            "bf3598d725ce7cf4/scene.bin"
        )
        path = str((ROOT / relative).resolve())
        if os.name == "nt":
            path = "\\\\?\\" + path
        if not os.path.exists(path):
            self.skipTest("archived scene bitstream is not available")
        with open(path, "rb") as stream:
            scene = SceneBitstream.unpack(stream.read())
        self.assertEqual((scene.points, scene.channels, scene.rank), (4096, 736, 56))
        self.assertEqual(len(scene.mean_fp16), 1472)
        self.assertEqual(len(scene.score), 47752)
        self.assertEqual(len(scene.residual), 2664)
        self.assertTrue(scene.residual.startswith(b"M3HPRANS"))


class CodecShapeTests(unittest.TestCase):
    def test_forward_shapes_and_parameter_layout(self) -> None:
        torch.manual_seed(7)
        config = CodecConfig(
            texture_channels=16,
            geometry_channels=12,
            geometry_observable_channels=8,
            rank=4,
            residual_n=6,
            residual_m=8,
            adapter_hidden=6,
            synthesis_mlp_hidden=5,
        )
        codec = ObservableLowRank1DCodec(config).eval()
        texture = torch.randn(2, 64, 16)
        geometry = torch.randn(2, 64, 12)
        positions = torch.randn(2, 64, 3)
        output = codec(texture, geometry, positions, training=False)
        self.assertEqual(output.texture.shape, texture.shape)
        self.assertEqual(output.geometry_observable.shape, (2, 64, 8))
        self.assertEqual(output.score_hat.shape, (2, 64, 4))
        self.assertEqual(output.reconstruction_sorted.shape, (2, 64, 24))
        self.assertTrue(torch.isfinite(output.estimated_bits))

    def test_entropy_round_trip(self) -> None:
        torch.manual_seed(11)
        config = CodecConfig(
            texture_channels=8,
            geometry_channels=6,
            geometry_observable_channels=4,
            rank=3,
            residual_n=6,
            residual_m=8,
            adapter_hidden=6,
            synthesis_mlp_hidden=5,
        )
        codec = ObservableLowRank1DCodec(config).eval()
        codec.update(force=True)
        texture = torch.randn(1, 64, 8)
        geometry = torch.randn(1, 64, 6)
        positions = torch.randn(1, 64, 3)
        compressed = codec.compress(texture, geometry, positions)
        texture_hat, geometry_hat = codec.decompress(
            compressed.data,
            inverse_permutation=compressed.order.inverse_permutation,
        )
        expected = compressed.order.restore(compressed.reconstruction_sorted)
        self.assertTrue(torch.allclose(texture_hat, expected[..., :8]))
        self.assertTrue(torch.allclose(geometry_hat, expected[..., 8:]))


class ArchivedCheckpointTests(unittest.TestCase):
    CHECKPOINT = ROOT / (
        "experiments/"
        "shared_lowrank_rank56_untied_nonlinear64_synthesis_multiscale1d/"
        "rate_points/lambda_0p0256/checkpoints/stages/step50000.ckpt"
    )

    @unittest.skipUnless(CHECKPOINT.exists(), "archived checkpoint is not available")
    def test_rank56_checkpoint_strict_load(self) -> None:
        loaded = load_feature_codec_checkpoint(self.CHECKPOINT)
        self.assertEqual(loaded.config.rank, 56)
        self.assertEqual(loaded.config.observable_channels, 736)
        self.assertEqual(loaded.config.basis_parameterization, "untied_synthesis")
        self.assertEqual(loaded.config.synthesis_mlp_hidden, 64)
        self.assertEqual(len(loaded.codec.state_dict()), 105)
        codec = loaded.codec.eval()
        texture = 0.01 * torch.randn(1, 64, 512)
        geometry = 0.01 * torch.randn(1, 64, 512)
        positions = torch.randn(1, 64, 3)
        compressed = codec.compress(texture, geometry, positions)
        texture_hat, geometry_hat = codec.decompress(
            compressed.data,
            inverse_permutation=compressed.order.inverse_permutation,
        )
        expected = compressed.order.restore(compressed.reconstruction_sorted)
        self.assertTrue(torch.allclose(texture_hat, expected[..., :512], atol=1e-6))
        self.assertTrue(torch.allclose(geometry_hat, expected[..., 512:], atol=1e-6))

    RANK40_CHECKPOINT = ROOT / (
        "experiments/shared_lowrank_rank40_multiscale1d/"
        "rate_points/lambda_0p1024/"
        "checkpoints/stages/step50000.ckpt"
    )

    @unittest.skipUnless(RANK40_CHECKPOINT.exists(), "rank40 checkpoint is not available")
    def test_rank40_tied_checkpoint_strict_load(self) -> None:
        loaded = load_feature_codec_checkpoint(self.RANK40_CHECKPOINT)
        self.assertEqual(loaded.config.rank, 40)
        self.assertEqual(loaded.config.basis_parameterization, "tied")
        self.assertIsNone(loaded.config.synthesis_mlp_hidden)
        self.assertEqual(len(loaded.codec.state_dict()), 102)


if __name__ == "__main__":
    unittest.main()

"""Real single-path entropy packets, gradients, and standalone receivers."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec
from globalsplat.compression.architecture import pilot_config, apply_shared_fresh_state
from globalsplat.compression.bitstream import SceneBitstream, ScoreContextBitstream
from globalsplat.compression.checkpoint import infer_config, load_feature_codec_checkpoint, validate_feature_codec_checkpoint


def small(mode):
    return CodecConfig(texture_channels=8, geometry_channels=8, geometry_observable_channels=4,
                       rank=0 if mode == "msh_only" else 4, residual_n=6, residual_m=8, adapter_hidden=6,
                       codec_mode=mode, use_residual=mode != "score_only",
                       score_spatial_context=mode != "msh_only",
                       score_context_schedule="dyadic4", score_spatial_stages=4, score_spatial_kernel=5,
                       score_slice_channels=2, token_order="hilbert")


class ArchitectureCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_packets_match_forward_with_only_active_branches(self):
        for mode in ("two_path", "score_only", "msh_only"):
            with self.subTest(mode=mode):
                codec = ObservableLowRank1DCodec(small(mode)).eval()
                codec.update(force=True)
                inputs = tuple(torch.randn(1, 37, c) for c in (8, 8, 3))
                with torch.no_grad():
                    expected = codec(*inputs, restore_original_order=False, training=False)
                    packet = codec.compress(*inputs).data
                    decoded = torch.cat(codec.decompress(packet), -1)
                scene = SceneBitstream.unpack(packet)
                self.assertEqual(sum(scene.bytes_by_stream.values()), len(packet))
                self.assertEqual(bool(scene.score), mode != "msh_only")
                self.assertEqual(bool(scene.residual), mode != "score_only")
                self.assertEqual(codec.residual_codec is None, mode == "score_only")
                self.assertEqual(codec.shared_basis is None, mode == "msh_only")
                self.assertEqual(codec.score_context is None, mode == "msh_only")
                torch.testing.assert_close(decoded, expected.reconstruction_sorted, atol=2e-6, rtol=1e-5)
                self.assertEqual(set(expected.likelihoods), {"residual_y", "residual_z"} if mode == "msh_only" else ({"score"} if mode == "score_only" else {"score", "residual_y", "residual_z"}))
                validate_feature_codec_checkpoint({"state_dict":codec.state_dict(), "feature_codec_config":codec.config.to_dict()}, codec.config)

    def test_missing_metadata_inactive_keys_and_wrong_packet_mode_are_rejected(self):
        score = ObservableLowRank1DCodec(small("score_only")).eval()
        score.update(force=True)
        with self.assertRaises(ValueError):
            infer_config(score.state_dict())
        state = dict(score.state_dict(), **{"residual_codec.unexpected": torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, "state mismatch"):
            infer_config(state, score.config.to_dict())
        packet = score.compress(*(torch.randn(1, 32, c) for c in (8, 8, 3))).data
        other = ObservableLowRank1DCodec(small("two_path"))
        with self.assertRaisesRegex(ValueError, "score-path"):
            other.decompress(packet)
        scene = SceneBitstream.unpack(packet)
        malformed = replace(scene, residual=b"unexpected")
        with self.assertRaisesRegex(ValueError, "presence"):
            score.decompress(malformed.pack())

    def test_gradients_reach_active_paths_and_projection(self):
        for mode in ("score_only", "msh_only"):
            codec = ObservableLowRank1DCodec(small(mode)).train()
            codec.set_trainable_scope("all")
            output = codec(*(torch.randn(2, 32, c) for c in (8, 8, 3)))
            loss = output.reconstruction_sorted.square().mean()+output.estimated_bits*1e-4+codec.aux_loss()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(codec.geometry_projection.weight.grad.abs().sum().item(), 0)
            weight = codec.shared_basis if mode == "score_only" else codec.residual_codec.g_a[0].weight
            self.assertGreater(weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in codec.parameters() if p.grad is not None))

    def test_standalone_receiver_uses_only_checkpoint_and_packet(self):
        for mode in ("score_only", "msh_only"):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                codec = ObservableLowRank1DCodec(small(mode)).eval()
                codec.update(force=True)
                packet = codec.compress(*(torch.randn(1, 32, c) for c in (8, 8, 3))).data
                expected = torch.cat(codec.decompress(packet), -1)
                torch.save({"state_dict":codec.state_dict(), "feature_codec_config":codec.config.to_dict()}, folder/"codec.pt")
                (folder/"scene.bin").write_bytes(packet)
                code = 'import sys,torch; from pathlib import Path; from globalsplat.compression.checkpoint import load_feature_codec_checkpoint; torch.set_num_threads(1); p=Path(sys.argv[1]); c=load_feature_codec_checkpoint(p/"codec.pt").codec.eval(); torch.save(torch.cat(c.decompress((p/"scene.bin").read_bytes()),-1),p/"out.pt")'
                result = subprocess.run([sys.executable,"-c",code,str(folder)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                torch.testing.assert_close(torch.load(folder/"out.pt",weights_only=True),expected,atol=0,rtol=0)

    def test_shared_initialization_and_full_width_groups(self):
        base = ObservableLowRank1DCodec(small("two_path"))
        for mode in ("score_only", "msh_only"):
            codec = ObservableLowRank1DCodec(small(mode))
            hashes = apply_shared_fresh_state(codec, base.state_dict())
            self.assertIn("geometry_projection.weight",hashes)
            for key in hashes:
                torch.testing.assert_close(codec.state_dict()[key],base.state_dict()[key],atol=0,rtol=0)
        full = ObservableLowRank1DCodec(pilot_config("score736_only"))
        self.assertEqual(full.score_context.group_sizes,(184,184,184,184))
        self.assertEqual(sum(p.numel() for p in full.score_context.anchor_predictors.parameters()),816960)
        full.eval().update(force=True)
        packet = full.compress(*(torch.randn(1, 32, c) for c in (512,512,3))).data
        self.assertEqual(len(ScoreContextBitstream.unpack(SceneBitstream.unpack(packet).score).strings),16)
        self.assertEqual(torch.cat(full.decompress(packet),-1).shape,(1,32,736))


if __name__ == "__main__":
    unittest.main()

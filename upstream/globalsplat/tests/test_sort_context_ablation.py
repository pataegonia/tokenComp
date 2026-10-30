"""Independent sorting/context ablations must survive coding and warm starts."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from globalsplat.compression import ObservableLowRank1DCodec
from globalsplat.compression.bitstream import SceneBitstream, ScoreContextBitstream
from globalsplat.compression.checkpoint import (
    convert_score_path_state, load_feature_codec_checkpoint,
    resize_registered_buffers, validate_feature_codec_checkpoint,
)
from codec_helpers import config


def ablation(order="none", context=True, stages=3, schedule="legacy"):
    if schedule != "legacy":
        stages = 2 if schedule == "quarter2" else 4
    return ObservableLowRank1DCodec(config(
        token_order=order, score_spatial_context=context,
        score_context_schedule=schedule,
        score_spatial_stages=stages, score_spatial_kernel=5,
    ))


class SortContextAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_slots_bypass_all_sorting_in_forward_and_actual_coding(self):
        inputs = tuple(torch.randn(1, 65, c) for c in (8, 8, 3))
        for schedule, enabled in (("legacy", True), ("quarter2", True),
                                  ("dyadic4", True), ("legacy", False)):
            with self.subTest(schedule=schedule, context=enabled):
                codec = ablation(context=enabled, schedule=schedule).eval()
                # Exercise nonzero anchor predictions, rather than only initialization.
                with torch.no_grad():
                    for name, parameter in codec.score_context.named_parameters():
                        if "predictors." in name:
                            parameter.normal_(0, .03)
                codec.update(force=True)
                with patch("globalsplat.compression.codec.morton_order_3d", side_effect=AssertionError("sorted")), \
                     patch("globalsplat.compression.codec.hilbert_order_3d", side_effect=AssertionError("sorted")), \
                     patch("globalsplat.compression.codec.greedy_neighbor_order", side_effect=AssertionError("sorted")):
                    output = codec(*inputs, training=False)
                    compressed = codec.compress(*inputs)
                self.assertEqual(output.order.permutation.tolist(), [list(range(65))])
                torch.testing.assert_close(output.order.apply(inputs[0]), inputs[0], rtol=0, atol=0)
                decoded = torch.cat(codec.decompress(compressed.data), -1)
                torch.testing.assert_close(decoded, output.reconstruction_sorted, rtol=1e-5, atol=3e-6)
                self.assertFalse(codec.config.use_morton)
        self.assertEqual(config(use_morton=False).token_order, "none")

    def test_context_off_is_independent_of_other_tokens_and_bypasses_predictors(self):
        for order, stages in (("hilbert", 2), ("hilbert", 3), ("none", 3)):
            with self.subTest(order=order, stages=stages):
                codec = ablation(order, False, stages)
                context = codec.score_context
                predictors = [parameter for name, parameter in context.named_parameters()
                              if "predictors." in name]
                with torch.no_grad():
                    for parameter in predictors:
                        parameter.fill_(3)
                scores = torch.randn(1, codec.config.rank, 1, 65, requires_grad=True)
                altered = scores.detach().clone()
                altered[..., 0] += 100
                # Hooks fail if even a frozen predictor is accidentally evaluated.
                def fail(*args):
                    raise AssertionError("disabled spatial predictor ran")
                handles = [module.register_forward_pre_hook(fail)
                           for name in ("spatial_predictors", "split_even_predictors")
                           for module in getattr(context, name, ())]
                try:
                    reconstructed, likelihood = codec._score_forward(scores, training=False)
                    other, other_likelihood = codec._score_forward(altered, training=False)
                finally:
                    for handle in handles:
                        handle.remove()
                torch.testing.assert_close(reconstructed[..., 1:], other[..., 1:], rtol=0, atol=0)
                torch.testing.assert_close(likelihood[..., 1:], other_likelihood[..., 1:], rtol=0, atol=0)
                loss = -likelihood.clamp_min(1e-9).log().sum()
                loss.backward()
                self.assertIsNotNone(scores.grad)
                self.assertTrue(all(parameter.grad is None for parameter in predictors))
                self.assertTrue(any(parameter.grad is not None for entropy in context.active_entropies()
                                    for parameter in entropy.parameters() if parameter.requires_grad))
                codec.set_trainable_scope("all")
                self.assertTrue(all(not parameter.requires_grad for parameter in predictors))

    def test_hilbert_only_actual_stream_and_decoder_reject_context_mismatch(self):
        codec = ablation("hilbert", False).eval()
        codec.update(force=True)
        inputs = tuple(torch.randn(1, 65, c) for c in (8, 8, 3))
        compressed = codec.compress(*inputs)
        output = codec(*inputs, training=False)
        decoded = torch.cat(codec.decompress(compressed.data), -1)
        torch.testing.assert_close(decoded, output.reconstruction_sorted, rtol=1e-5, atol=3e-6)
        packed = ScoreContextBitstream.unpack(SceneBitstream.unpack(compressed.data).score)
        self.assertFalse(packed.flags & codec.score_context.FLAG_SPATIAL)
        self.assertEqual(len(packed.strings), 3 * len(codec.score_context.group_sizes))
        self.assertTrue(SceneBitstream.unpack(compressed.data).residual)
        wrong = ablation("hilbert", True).eval()
        resize_registered_buffers(wrong, codec.state_dict())
        wrong.load_state_dict(codec.state_dict())
        with self.assertRaisesRegex(ValueError, "bitstream does not match codec configuration"):
            wrong.decompress(compressed.data)

    def test_both_off_keeps_msh_split_streams_and_trainable_low_rank_path(self):
        codec = ablation("none", False).eval()
        codec.update(force=True)
        inputs = tuple(torch.randn(1, 65, c) for c in (8, 8, 3))
        payload = codec.compress(*inputs).data
        scene = SceneBitstream.unpack(payload)
        score = ScoreContextBitstream.unpack(scene.score)
        self.assertEqual(scene.mean_fp16, b"")
        self.assertTrue(scene.residual)
        self.assertFalse(score.flags & codec.score_context.FLAG_SPATIAL)
        self.assertEqual(len(score.strings), 3 * len(codec.score_context.group_sizes))
        output = codec(*inputs, training=False)
        decoded = torch.cat(codec.decompress(payload), -1)
        torch.testing.assert_close(decoded, output.reconstruction_sorted, rtol=1e-5, atol=3e-6)
        codec.train().set_trainable_scope("all")
        trained = codec(*inputs, training=True)
        (trained.reconstruction_sorted.square().mean() + trained.estimated_bits / 10000).backward()
        self.assertIsNotNone(codec.shared_basis.grad)
        self.assertIsNotNone(codec.shared_synthesis_basis.grad)
        self.assertTrue(torch.isfinite(codec.shared_basis.grad).all())
        self.assertTrue(any(p.grad is not None for p in codec.residual_codec.parameters()))
        self.assertTrue(all(p.grad is None for name, p in codec.score_context.named_parameters()
                            if "predictors." in name))

    def test_weights_only_conversion_preserves_msh_and_metadata_is_required_for_strict_load(self):
        parent = ObservableLowRank1DCodec(config()).eval()
        parent.update(force=True)
        checkpoint = {"state_dict": parent.state_dict(), "feature_codec_config": parent.config.to_dict()}
        for order, enabled, schedule in (("none", True, "legacy"), ("hilbert", False, "legacy"),
                                         ("none", False, "legacy"),
                                         ("none", True, "quarter2"), ("none", True, "dyadic4")):
            with self.subTest(order=order, schedule=schedule):
                codec = ablation(order, enabled, schedule=schedule).eval()
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    validate_feature_codec_checkpoint(checkpoint, codec.config)
                validate_feature_codec_checkpoint(checkpoint, codec.config, allow_score_path_conversion=True)
                converted = convert_score_path_state(checkpoint["state_dict"], codec)
                for name, value in parent.state_dict().items():
                    if not name.startswith("score_context."):
                        torch.testing.assert_close(converted[name], value, rtol=0, atol=0)
                # CDF arrays are resized by the production loader below.
                saved = {"state_dict": converted, "feature_codec_config": codec.config.to_dict()}
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "ablated.ckpt"
                    torch.save(saved, path)
                    loaded = load_feature_codec_checkpoint(path)
                self.assertEqual(loaded.config, codec.config)
                validate_feature_codec_checkpoint(saved, codec.config)
                changed = replace(codec.config, score_spatial_context=not enabled)
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    validate_feature_codec_checkpoint(saved, changed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_bf16_training_keeps_active_codec_gradients(self):
        for order, enabled, schedule in (("none", True, "legacy"), ("hilbert", False, "legacy"),
                                         ("none", False, "legacy"),
                                         ("none", True, "quarter2"), ("none", True, "dyadic4")):
            with self.subTest(order=order, schedule=schedule):
                codec = ablation(order, enabled, schedule=schedule).cuda().train()
                codec.set_trainable_scope("all")
                inputs = tuple(torch.randn(2, 129, c, device="cuda") for c in (8, 8, 3))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = codec(*inputs)
                    loss = output.reconstruction_sorted.square().mean() + output.estimated_bits / 10000
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(codec.shared_basis.grad)
                self.assertTrue(torch.isfinite(codec.shared_basis.grad).all())
                self.assertTrue(any(parameter.grad is not None for parameter in codec.residual_codec.parameters()))
                if schedule != "legacy":
                    self.assertTrue(any(parameter.grad is not None for parameter in codec.score_context.anchor_predictors.parameters()))
                if not enabled:
                    self.assertTrue(all(parameter.grad is None for name, parameter in codec.score_context.named_parameters()
                                        if "predictors." in name))


if __name__ == "__main__":
    unittest.main()

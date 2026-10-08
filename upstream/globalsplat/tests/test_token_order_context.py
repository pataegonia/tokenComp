"""Ordering, causal anchors, actual entropy round trips and old checkpoints."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from globalsplat.compression import ObservableLowRank1DCodec
from globalsplat.compression.bitstream import ScoreContextBitstream
from globalsplat.compression.checkpoint import (
    convert_score_path_state, infer_config, load_feature_codec_checkpoint,
    resize_registered_buffers, validate_feature_codec_checkpoint,
)
from globalsplat.compression.token_order import hilbert_codes_3d, greedy_neighbor_order
from test_score_path_ablation import config, MINIMAL


def make_codec(schedule="legacy", order="morton", **kwargs):
    return ObservableLowRank1DCodec(config(
        **MINIMAL, token_order=order, score_context_schedule=schedule,
        score_spatial_stages=4 if schedule == "dyadic4" else 2,
        score_context_quantization="ste", **kwargs,
    ))


def trained_anchors(codec):
    with torch.no_grad():
        for name, parameter in codec.score_context.named_parameters():
            if name.startswith("anchor_predictors."):
                parameter.normal_(0, .1)
            if name.endswith("quantiles"):
                parameter[:, 0, 1].fill_(.27)


class TokenOrderContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hilbert_dense_grid_bijection_and_adjacent_cells(self):
        for bits in (1, 2, 3):
            n = 1 << bits
            grid = torch.stack(torch.meshgrid(*[torch.arange(n)] * 3, indexing="ij"), -1).reshape(1, -1, 3)
            codes = hilbert_codes_3d(grid, bits)
            self.assertEqual(codes.unique().numel(), n ** 3)
            self.assertEqual(codes.max().item(), n ** 3 - 1)
            ordered = grid[:, codes[0].argsort()]
            self.assertEqual((ordered[:, 1:] - ordered[:, :-1]).abs().sum(-1).max().item(), 1)
        same = torch.zeros(2, 9, 3)
        self.assertTrue(torch.equal(make_codec(order="hilbert")._make_order(same).permutation,
                                    torch.arange(9).expand(2, -1)))

    def test_greedy_tour_expected_neighbors_ties_and_inverse(self):
        values = torch.tensor([[[0.], [2.], [1.], [4.]]])
        positions = torch.zeros(1, 4, 3)
        order = greedy_neighbor_order(values, positions)
        self.assertEqual(order.permutation.tolist(), [[0, 2, 1, 3]])
        torch.testing.assert_close(order.restore(order.apply(values)), values, rtol=0, atol=0)
        tied = greedy_neighbor_order(torch.zeros_like(values), positions)
        self.assertEqual(tied.permutation.tolist(), [[0, 1, 2, 3]])

    def test_new_partitions_keep_all_tokens(self):
        expected = {
            "quarter2": [[0, 4, 8, 12, 16], [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]],
            "dyadic4": [[0, 8, 16], [4, 12], [2, 6, 10, 14], list(range(1, 17, 2))],
        }
        for schedule, groups in expected.items():
            actual = [indices.tolist() for _, indices in make_codec(schedule).score_context.stage_indices(17, "cpu")]
            self.assertEqual(actual, groups)
            self.assertEqual(sorted(sum(actual, [])), list(range(17)))

    def test_gather_uses_distance_four_and_quarter_phase_and_boundary(self):
        dyadic = make_codec("dyadic4").score_context.anchor_predictors[0][0]
        quarter = make_codec("quarter2").score_context.anchor_predictors[0][0]
        with torch.no_grad():
            dyadic.weight[0, 0, 0, 1] = 1  # target4 sees right anchor8, beyond kernel5
            quarter.weight[:, 0, 0, 0] = torch.tensor([1., 2., 3.])
            quarter.edge_bias[:, 0] = 10
        canvas = torch.zeros(1, 3, 1, 9)
        canvas[..., 8] = 7
        self.assertEqual(dyadic(canvas, torch.tensor([4]))[0, 0, 0, 0].item(), 7)
        canvas[..., 0] = 2
        out = quarter(canvas, torch.tensor([1, 2, 3]))
        self.assertEqual(out[0, 0, 0].tolist(), [2, 4, 6])
        out = quarter(canvas[..., :8], torch.tensor([5, 6, 7]))
        self.assertEqual(out[0, 0, 0].tolist(), [10, 10, 10])

    def test_only_decoded_prior_stages_are_visible(self):
        for schedule in ("quarter2", "dyadic4"):
            codec = make_codec(schedule).eval()
            trained_anchors(codec)
            context = codec.score_context
            scores = torch.randn(1, 7, 1, 33)
            captured = {}
            handles = []
            for stage, groups in enumerate(context.anchor_predictors, 1):
                for group, predictor in enumerate(groups):
                    def capture(module, args, key=(stage, group)):
                        captured[key] = args[0].detach().clone()
                    handles.append(predictor.register_forward_pre_hook(capture))
            output, likelihood = context(scores, torch.zeros(1, 12), codec.score_entropy, training=False)
            for handle in handles:
                handle.remove()
            visible = torch.zeros(33, dtype=torch.bool)
            partitions = context.stage_indices(33, "cpu")
            for stage in range(1, len(partitions)):
                visible[partitions[stage - 1][1]] = True
                start = 0
                for group, width in enumerate(context.group_sizes):
                    expected = output[:, start:start + width].clone()
                    expected[..., ~visible] = 0
                    torch.testing.assert_close(captured[stage, group], expected, rtol=0, atol=0)
                    start += width
            changed = scores.clone()
            changed[..., partitions[-1][1]] += 1000
            other, other_likelihood = context(changed, torch.zeros(1, 12), codec.score_entropy, training=False)
            earlier = torch.cat([indices for _, indices in partitions[:-1]])
            torch.testing.assert_close(output[..., earlier], other[..., earlier], rtol=0, atol=0)
            torch.testing.assert_close(likelihood[..., earlier], other_likelihood[..., earlier], rtol=0, atol=0)

    def test_short_streams_ste_and_standalone_context_decoder(self):
        for schedule in ("quarter2", "dyadic4"):
            codec = make_codec(schedule).train()
            trained_anchors(codec)
            codec.update(force=True)
            context = codec.score_context
            for points in (1, 2, 3, 4, 5, 7, 9, 17, 33):
                scores = torch.randn(1, 7, 1, points, requires_grad=True)
                mean = torch.zeros(1, 12)
                expected, likelihood = context(scores, mean, codec.score_entropy, training=True)
                payload, sender = context.compress(scores.detach(), mean, codec.score_entropy)
                receiver = copy.deepcopy(context).decompress(payload, mean, points, codec.score_entropy)
                torch.testing.assert_close(sender, receiver, rtol=0, atol=0)
                torch.testing.assert_close(expected, sender, rtol=0, atol=1e-6)
                self.assertEqual(sum(context.last_compress_stage_bytes.values()),
                                 sum(map(len, ScoreContextBitstream.unpack(payload).strings)))
                if points == 33:
                    (expected.square().mean() - torch.log2(likelihood).mean()).backward()
                    self.assertGreater(scores.grad.abs().sum().item(), 0)
                    for name, parameter in context.named_parameters():
                        if name.startswith("anchor_predictors."):
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_all_order_and_context_combinations_actual_scene_and_checkpoint(self):
        inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
        for order in ("morton", "hilbert", "nn_xyz", "nn_score"):
            for schedule in ("legacy", "quarter2", "dyadic4"):
                with self.subTest(order=order, schedule=schedule):
                    codec = make_codec(schedule, order).eval()
                    trained_anchors(codec)
                    codec.update(force=True)
                    expected = codec(*inputs, training=False)
                    compressed = codec.compress(*inputs)
                    torch.testing.assert_close(compressed.order.permutation, expected.order.permutation)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "codec.ckpt"
                        torch.save({"state_dict": codec.state_dict(), "feature_codec_config": codec.config.to_dict()}, path)
                        receiver = load_feature_codec_checkpoint(path).codec.eval()
                        actual = torch.cat(receiver.decompress(compressed.data), -1)
                        torch.testing.assert_close(actual, expected.reconstruction_sorted, rtol=1e-5, atol=3e-6)
                        self.assertEqual(receiver.config, codec.config)
                    wrong = make_codec("quarter2" if schedule != "quarter2" else "legacy", order).eval()
                    wrong.update(force=True)
                    with self.assertRaisesRegex(ValueError, "configuration"):
                        wrong.decompress(compressed.data)

    def test_old_metadata_and_explicit_warm_start_preserve_MSH(self):
        source = ObservableLowRank1DCodec(config()).eval()
        source.update(force=True)
        old_metadata = source.config.to_dict()
        for key in ("token_order", "score_order_scale", "score_context_schedule"):
            old_metadata.pop(key)
        self.assertEqual(infer_config(source.state_dict(), old_metadata), source.config)
        checkpoint = {"state_dict": source.state_dict(), "feature_codec_config": old_metadata}
        for schedule in ("quarter2", "dyadic4"):
            codec = make_codec(schedule, "hilbert")
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                validate_feature_codec_checkpoint(checkpoint, codec.config)
            validate_feature_codec_checkpoint(checkpoint, codec.config, allow_score_path_conversion=True)
            state = convert_score_path_state(source.state_dict(), codec)
            resize_registered_buffers(codec, state)
            codec.load_state_dict(state, strict=True)
            for name, tensor in source.state_dict().items():
                if not name.startswith("score_context."):
                    torch.testing.assert_close(codec.state_dict()[name], tensor, rtol=0, atol=0)
            self.assertEqual(infer_config(state, codec.config.to_dict()), codec.config)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                infer_config(state, replace(codec.config, score_context_schedule="legacy",
                    score_spatial_kernel=5).to_dict())

    def test_value_order_keeps_mean_once_and_gradients_through_both_paths(self):
        codec = make_codec("quarter2", "nn_score", score_order_scale=(1, 2, 3, 4, 5, 6, 7)).train()
        trained_anchors(codec)
        inputs = tuple(torch.randn(2, 65, c, requires_grad=True) for c in (8, 8, 3))
        output = codec(*inputs, training=True)
        (output.reconstruction_sorted.square().mean() + .0001 * output.estimated_bits).backward()
        for parameter in (codec.shared_basis, codec.residual_codec.g_a[0].weight,
                          codec.score_context.anchor_predictors[0][0].weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        # Full centering uses one transmitted mean, independent of sorted reduction.
        centered = ObservableLowRank1DCodec(config(token_order="nn_score")).eval()
        centered.update(force=True)
        inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
        torch.testing.assert_close(torch.cat(centered.decompress(centered.compress(*inputs).data), -1),
                                   centered(*inputs, training=False).reconstruction_sorted, rtol=1e-5, atol=3e-6)

    def test_new_schedules_batch_4096_and_mixed_precision_sender(self):
        for schedule in ("quarter2", "dyadic4"):
            codec = make_codec(schedule, "hilbert").train()
            trained_anchors(codec)
            inputs = tuple(torch.randn(2, 4096, c) for c in (8, 8, 3))
            output = codec(*inputs, training=True)
            (output.reconstruction_sorted.square().mean() + .0001 * output.estimated_bits).backward()
            self.assertGreater(codec.shared_basis.grad.abs().sum().item(), 0)
            self.assertGreater(codec.residual_codec.g_a[0].weight.grad.abs().sum().item(), 0)
            self.assertGreater(codec.score_context.anchor_predictors[0][0].weight.grad.abs().sum().item(), 0)
            codec.eval().update(force=True)
            small = tuple(value[:1, :33] for value in inputs)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                payload = codec.compress(*small).data
            receiver = copy.deepcopy(codec)
            expected = torch.cat(codec.decompress(payload), -1)
            torch.testing.assert_close(torch.cat(receiver.decompress(payload), -1), expected, rtol=0, atol=0)

    def test_kernel7_reaches_distance_three_even_and_round_trips(self):
        codec = make_codec(score_spatial_kernel=7).eval()
        context = codec.score_context
        with torch.no_grad():
            context.spatial_predictors[0].weight[0, 0, 0, 6] = 1
        scores = torch.zeros(1, 7, 1, 17)
        scores[0, 0, 0, 4] = 2  # odd1 reads even4, at distance+3.
        observed = []
        hook = context.spatial_predictors[0].register_forward_hook(
            lambda module, args, output: observed.append(output.detach().clone()))
        expected, _ = context(scores, torch.zeros(1, 12), codec.score_entropy, training=False)
        hook.remove()
        self.assertEqual(observed[0][0, 0, 0, 1].item(), 2)
        codec.update(force=True)
        payload, sender = context.compress(scores, torch.zeros(1, 12), codec.score_entropy)
        decoded = context.decompress(payload, torch.zeros(1, 12), 17, codec.score_entropy)
        torch.testing.assert_close(sender, decoded, rtol=0, atol=0)
        torch.testing.assert_close(sender, expected, rtol=0, atol=0)
        self.assertEqual(infer_config(codec.state_dict(), codec.config.to_dict()), codec.config)


if __name__ == "__main__":
    unittest.main()

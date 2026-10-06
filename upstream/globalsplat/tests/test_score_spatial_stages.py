"""Decoder causality, quantized training context and spatial warm starts."""
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
from test_score_path_ablation import config, MINIMAL


def staged_codec(stages):
    return ObservableLowRank1DCodec(config(
        **MINIMAL, score_spatial_stages=stages, score_spatial_kernel=5,
        score_context_quantization="ste",
    ))


def trained_predictors(codec):
    with torch.no_grad():
        for name, parameter in codec.score_context.named_parameters():
            if "predictors" in name and "channel" not in name:
                parameter.normal_(0, 0.1)
            if name.endswith("quantiles"):
                parameter[:, 0, 1].fill_(0.27)


class SpatialStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_partition_keeps_every_token_and_matches_order(self):
        for stages, expected in ((2, [[0, 2, 4, 6], [1, 3, 5]]),
                                 (3, [[0, 4], [2, 6], [1, 3, 5]]),
                                 (4, [[0, 4], [2, 6], [1, 5], [3]])):
            context = staged_codec(stages).score_context
            indices = [idx.tolist() for _, idx in context.stage_indices(7, "cpu")]
            self.assertEqual(indices, expected)
            self.assertEqual(sorted(sum(indices, [])), list(range(7)))

    def test_predictor_sees_only_previous_reconstructed_stages(self):
        for stages in (3, 4):
            codec = staged_codec(stages).eval()
            trained_predictors(codec)
            context = codec.score_context
            scores = torch.randn(1, 7, 1, 17)
            mean = torch.zeros(1, 12)
            inputs = {}
            handles = []
            def capture(key):
                def hook(module, args):
                    inputs[key] = args[0].detach().clone()
                return hook
            for name, modules in (("B", context.split_even_predictors),
                                  ("C", context.spatial_predictors),
                                  ("D", getattr(context, "split_odd_predictors", []))):
                for group, module in enumerate(modules):
                    handles.append(module.register_forward_pre_hook(capture((name, group))))
            reconstruction, likelihood = context(scores, mean, codec.score_entropy, training=False)
            for handle in handles:
                handle.remove()
            previous = {"B": [0], "C": [0, 2], "D": [0, 2, 1]}
            for name in previous:
                if name == "D" and stages == 3:
                    continue
                mask = torch.tensor([i % 4 in previous[name] for i in range(17)])
                start = 0
                for group, width in enumerate(context.group_sizes):
                    expected = reconstruction[:, start:start + width].clone()
                    expected[..., ~mask] = 0
                    torch.testing.assert_close(inputs[name, group], expected, rtol=0, atol=0)
                    start += width
            changed = scores.clone()
            changed[..., 1::2] += 1000  # C/D cannot affect A/B values or rates.
            other, other_likelihood = context(changed, mean, codec.score_entropy, training=False)
            torch.testing.assert_close(reconstruction[..., 0::2], other[..., 0::2], rtol=0, atol=0)
            torch.testing.assert_close(likelihood[..., 0::2], other_likelihood[..., 0::2], rtol=0, atol=0)

    def test_distance_two_neighbors_reach_B_and_D(self):
        codec = staged_codec(4).eval()
        context = codec.score_context
        with torch.no_grad():
            # Output channel zero reads its +2 neighbor on the original sequence.
            context.split_even_predictors[0].weight[0, 0, 0, 4] = 1
            context.split_odd_predictors[0].weight[0, 0, 0, 4] = 1
        observed = {}
        def save(name):
            def hook(module, args, output):
                observed[name] = output.detach().clone()
            return hook
        hb = context.split_even_predictors[0].register_forward_hook(save("B"))
        hd = context.split_odd_predictors[0].register_forward_hook(save("D"))
        scores = torch.zeros(1, 7, 1, 9)
        scores[0, 0, 0, 4] = 2  # A4 is visible to B2.
        scores[0, 0, 0, 5] = 3  # C5 is visible to D3.
        context(scores, torch.zeros(1, 12), codec.score_entropy, training=False)
        hb.remove(); hd.remove()
        self.assertEqual(observed["B"][0, 0, 0, 2].item(), 2)
        self.assertEqual(observed["D"][0, 0, 0, 3].item(), 3)

    def test_ste_training_matches_decoded_lattice_and_trains_every_stage(self):
        for stages in (2, 3, 4):
            codec = staged_codec(stages).train()
            trained_predictors(codec)
            codec.update(force=True)
            context = codec.score_context
            scores = torch.randn(1, 7, 1, 35, requires_grad=True)
            mean = torch.zeros(1, 12)
            reconstruction, likelihood = context(scores, mean, codec.score_entropy, training=True)
            payload, encoded = context.compress(scores.detach(), mean, codec.score_entropy)
            decoded = context.decompress(payload, mean, 35, codec.score_entropy)
            torch.testing.assert_close(reconstruction, encoded, rtol=0, atol=1e-6)
            torch.testing.assert_close(encoded, decoded, rtol=0, atol=0)
            (reconstruction.square().mean() * .01 - torch.log2(likelihood).mean()).backward()
            self.assertTrue(torch.isfinite(scores.grad).all())
            self.assertGreater(scores.grad.abs().sum().item(), 0)
            for name, parameter in context.named_parameters():
                if "predictors" in name and "channel" not in name:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            for entropy in context.active_entropies(codec.score_entropy):
                gradients = [p.grad for n, p in entropy.named_parameters() if not n.endswith("quantiles")]
                self.assertGreater(sum(g.abs().sum().item() for g in gradients if g is not None), 0)

    def test_short_streams_and_uneven_counts_round_trip(self):
        for stages in (3, 4):
            codec = staged_codec(stages).eval()
            trained_predictors(codec)
            codec.update(force=True)
            context = codec.score_context
            for points in (1, 2, 3, 4, 5, 17, 33):
                scores = torch.randn(1, 7, 1, points)
                mean = torch.zeros(1, 12)
                expected, _ = context(scores, mean, codec.score_entropy, training=False)
                payload, sender = context.compress(scores, mean, codec.score_entropy)
                receiver = copy.deepcopy(context).decompress(payload, mean, points, codec.score_entropy)
                torch.testing.assert_close(expected, sender, rtol=0, atol=0)
                torch.testing.assert_close(sender, receiver, rtol=0, atol=0)
                packed = ScoreContextBitstream.unpack(payload)
                self.assertEqual(sum(context.last_compress_stage_bytes.values()), sum(map(len, packed.strings)))
                self.assertEqual(len(packed.strings), stages * len(context.group_sizes))

    def test_batched_4096_token_codec_trains_with_MSH(self):
        for stages in (3, 4):
            codec = staged_codec(stages).train()
            inputs = tuple(torch.randn(2, 4096, c) for c in (8, 8, 3))
            output = codec(*inputs, training=True)
            self.assertEqual(output.likelihoods["score"].shape, (2, 7, 1, 4096))
            loss = output.reconstruction_sorted.square().mean() + .0001 * output.estimated_bits
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(codec.residual_codec.g_a[0].weight.grad.abs().sum().item(), 0)
            self.assertGreater(codec.shared_basis.grad.abs().sum().item(), 0)
            self.assertGreater(codec.score_context.split_even_predictors[0].weight.grad.abs().sum().item(), 0)

    def test_explicit_warm_start_preserves_MSH_and_initial_score_reconstruction(self):
        source = ObservableLowRank1DCodec(config()).eval()
        trained_predictors(source)
        source.update(force=True)
        source_state = {"model.feature_codec." + k: v for k, v in source.state_dict().items()}
        source_state["model.unrelated"] = torch.tensor([123.])
        checkpoint = {"state_dict": source_state, "feature_codec_config": source.config.to_dict()}
        control = None
        inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
        for stages in (2, 3, 4):
            codec = staged_codec(stages).eval()
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                validate_feature_codec_checkpoint(checkpoint, codec.config)
            validate_feature_codec_checkpoint(checkpoint, codec.config, allow_score_path_conversion=True)
            converted = convert_score_path_state(source_state, codec)
            self.assertIs(converted["model.unrelated"], source_state["model.unrelated"])
            state = {k[len("model.feature_codec."):]: v for k, v in converted.items()
                     if k.startswith("model.feature_codec.")}
            resize_registered_buffers(codec, state)
            codec.load_state_dict(state, strict=True)
            for key, tensor in source.state_dict().items():
                if not key.startswith("score_context."):
                    torch.testing.assert_close(codec.state_dict()[key], tensor, rtol=0, atol=0)
            weights = codec.score_context.spatial_predictors[0].weight
            torch.testing.assert_close(weights[..., 1:4], source.score_context.spatial_predictors[0].weight)
            self.assertEqual(weights[..., (0, 4)].abs().sum().item(), 0)
            output = codec(*inputs, training=False).reconstruction_sorted
            if control is None:
                control = output
            else:
                torch.testing.assert_close(output, control, rtol=0, atol=0)
            self.assertEqual(infer_config(state, codec.config.to_dict()), codec.config)
            if stages > 2:
                # Metadata cannot mislabel the tensor architecture.
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    infer_config(state, replace(codec.config, score_spatial_stages=2).to_dict())

    def test_standalone_bfloat16_receiver_and_checkpoint_metadata(self):
        for stages in (3, 4):
            codec = staged_codec(stages).eval()
            trained_predictors(codec)
            codec.update(force=True)
            inputs = tuple(torch.randn(1, 33, c) for c in (8, 8, 3))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                compressed = codec.compress(*inputs)
            expected = torch.cat(codec.decompress(compressed.data), -1)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "stages.ckpt"
                torch.save({"state_dict": codec.state_dict(), "feature_codec_config": codec.config.to_dict()}, path)
                receiver = load_feature_codec_checkpoint(path).codec.eval()
                actual = torch.cat(receiver.decompress(compressed.data), -1)
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            wrong = staged_codec(4 if stages == 3 else 3).eval()
            wrong.update(force=True)
            with self.assertRaisesRegex(ValueError, "configuration"):
                wrong.decompress(compressed.data)

    def test_malformed_stream_count_and_empty_stage_are_rejected(self):
        codec = staged_codec(4).eval()
        codec.update(force=True)
        context = codec.score_context
        mean = torch.zeros(1, 12)
        payload, _ = context.compress(torch.zeros(1, 7, 1, 1), mean, codec.score_entropy)
        packed = ScoreContextBitstream.unpack(payload)
        with self.assertRaisesRegex(ValueError, "stream count"):
            context.decompress(replace(packed, strings=packed.strings[:-1]).pack(), mean, 1, codec.score_entropy)
        bad = list(packed.strings)
        bad[len(context.group_sizes)] = b"bad"
        with self.assertRaisesRegex(ValueError, "empty token stage"):
            context.decompress(replace(packed, strings=tuple(bad)).pack(), mean, 1, codec.score_entropy)


if __name__ == "__main__":
    unittest.main()

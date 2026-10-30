"""Offline traces must describe the exact bitstream without changing coding."""
import unittest

import torch

from globalsplat.compression import ObservableLowRank1DCodec
from globalsplat.compression.mechanism import (
    StageRecorder, fixed_score_relations, observe_stages, payload_components,
    references, schedule_references,
)
from test_score_path_ablation import MINIMAL, config


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, schedule="legacy", enabled=True):
        codec = ObservableLowRank1DCodec(config(**MINIMAL, token_order="none",
            score_spatial_context=enabled, score_context_schedule=schedule,
            score_spatial_stages=2 if schedule == "quarter2" else 4 if schedule == "dyadic4" else 3,
            score_spatial_kernel=5, score_context_quantization="ste")).eval()
        with torch.no_grad():
            for name, p in codec.score_context.named_parameters():
                if "predictors." in name:
                    p.normal_(0, .025)
                if name.endswith("quantiles"):
                    p[:, 0, 1].fill_(.27)
        codec.update(force=True)
        return codec

    def test_observer_keeps_identical_bytes_state_and_actual_decoded_anchors(self):
        for schedule in ("legacy", "quarter2", "dyadic4"):
            with self.subTest(schedule=schedule), torch.no_grad():
                codec = self.make(schedule)
                context = codec.score_context
                score, xyz = torch.randn(1, 7, 1, 65), torch.randn(1, 65, 3)
                mean = torch.zeros(1, codec.config.observable_channels)
                expected, expected_hat = context.compress(score, mean, codec.score_entropy)
                before = {k: v.clone() for k, v in codec.state_dict().items()}
                recorder = StageRecorder(score, xyz, keep_tensors=True)
                with observe_stages(context, recorder):
                    packet, actual = context.compress(score, mean, codec.score_entropy)
                self.assertEqual(packet, expected)
                torch.testing.assert_close(actual, expected_hat, rtol=0, atol=0)
                received = context.decompress(packet, mean, 65, codec.score_entropy)
                torch.testing.assert_close(received, actual, rtol=0, atol=0)
                torch.testing.assert_close(recorder.reconstruction, received[0, :, 0].T, rtol=0, atol=0)
                stages = recorder.summary()
                self.assertEqual(sum(s["tokens"] for s in stages), 65)
                self.assertEqual(sum(s["raw_bytes"] for s in stages), sum(context.last_compress_stage_bytes.values()))
                for k, v in codec.state_dict().items():
                    torch.testing.assert_close(v, before[k], rtol=0, atol=0)
                for item in recorder.tensors:
                    start, width = item["start"], item["target"].shape[-1]
                    valid, ref = item["reference_valid"], item["references"]
                    if valid.any():
                        expected_anchor = received[0, start:start+width, 0].T[ref[valid]]
                        torch.testing.assert_close(item["reference_inputs"][valid], expected_anchor, rtol=0, atol=0)
                self.assertIsNone(context.stage_observer)

    def test_true_legacy_kernel_support_uses_only_visible_sites(self):
        codec = self.make()
        context = codec.score_context
        visible = torch.zeros(15, dtype=torch.bool)
        visible[::4] = True
        _, predictor = context._stage_modules(1, 0)
        edge = references(predictor, torch.tensor([2, 6, 10, 14]), visible, 15)
        self.assertEqual(edge["indices"][0][edge["valid"][0]].tolist(), [0, 4])
        self.assertEqual(edge["indices"][-1][edge["valid"][-1]].tolist(), [12])
        self.assertEqual(edge["kind"], "masked_canvas_convolution")
        visible[2::4] = True
        _, predictor = context._stage_modules(2, 0)
        edge = references(predictor, torch.tensor([1, 3]), visible, 15)
        self.assertEqual(edge["indices"][0][edge["valid"][0]].tolist(), [0, 2])

    def test_recorded_anchor_distances_and_residual_statistics_match_raw_tensors(self):
        codec = self.make()
        score, xyz = torch.randn(1, 7, 1, 65) * 3, torch.randn(1, 65, 3)
        recorder = StageRecorder(score, xyz, keep_tensors=True)
        with observe_stages(codec.score_context, recorder), torch.no_grad():
            codec.score_context.compress(score, torch.zeros(1, codec.config.observable_channels), codec.score_entropy)
        for group, trace in zip(recorder.groups, recorder.tensors):
            width, start = group["width"], group["start"]
            scale = recorder.scale[start:start+width]
            residual = trace["target"] - trace["prediction"]
            baseline = trace["target"] - trace["baseline"]
            torch.testing.assert_close(torch.tensor(group["residual_raw_mae_channels"]), residual.abs().mean(0))
            self.assertAlmostEqual(group["normalized_sq_error_sum"], float((residual / scale).square().sum()), places=4)
            self.assertAlmostEqual(group["baseline_raw_abs_error_sum"], float(baseline.abs().sum()), places=4)
            valid = trace["reference_valid"]
            if valid.any():
                target = trace["target"][:, None].expand_as(trace["reference_inputs"])[valid]
                delta = target - trace["reference_inputs"][valid]
                relation = group["anchor_relation"]
                self.assertAlmostEqual(relation["raw_l1"], float(delta.abs().mean()), places=5)
                self.assertAlmostEqual(relation["normalized_mse"], float((delta/scale).square().mean()), places=5)
        for stage in recorder.summary():
            groups = [t for t in recorder.tensors if t["stage"] == stage["stage"]]
            residual = torch.cat([g["target"] - g["prediction"] for g in groups], 1)
            self.assertAlmostEqual(stage["raw_mae"], float(residual.abs().mean()), places=5)
            self.assertAlmostEqual(stage["raw_mse"], float(residual.square().mean()), places=4)

    def test_anchor_boundary_duplicate_and_schedule_partition(self):
        codec = self.make("dyadic4")
        _, predictor = codec.score_context._stage_modules(1, 0)
        visible = torch.zeros(13, dtype=torch.bool)
        visible[::8] = True
        edge = references(predictor, torch.tensor([4, 12]), visible, 13)
        self.assertEqual(edge["indices"].tolist(), [[0, 8], [8, 8]])
        self.assertEqual(edge["boundary"].tolist(), [False, True])
        for schedule in ("legacy3", "quarter2", "dyadic4"):
            stages = list(schedule_references(schedule, 13))
            self.assertEqual(sorted(torch.cat([s[1] for s in stages]).tolist()), list(range(13)))

    def test_prediction_off_is_reversible_even_after_an_exception(self):
        codec = self.make()
        recorder = StageRecorder(torch.randn(1, 7, 1, 65), torch.randn(1, 65, 3))
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with observe_stages(codec.score_context, recorder, prediction_off=True):
                self.assertFalse(codec.score_context.spatial_context)
                raise RuntimeError("stop")
        self.assertTrue(codec.score_context.spatial_context)
        self.assertIsNone(codec.score_context.stage_observer)

    def test_packet_components_and_stage_bytes_do_not_double_count_headers(self):
        codec = self.make("dyadic4")
        inputs = tuple(torch.randn(1, 65, c) for c in (8, 8, 3))
        data = codec.compress(*inputs).data
        counted = payload_components(data)
        self.assertEqual(sum(counted["components"].values()), len(data))
        self.assertEqual(sum(codec.score_context.last_compress_stage_bytes.values()), counted["components"]["score_raw"])
        self.assertGreater(counted["components"]["headers"], 0)
        self.assertGreater(counted["score_total_bytes"], counted["components"]["score_raw"])

    def test_fixed_score_orders_share_values_and_scale_and_random_is_repeatable(self):
        score, xyz = torch.randn(33, 7), torch.randn(33, 3)
        score[:, -1] = 2
        before, xyz_before = score.clone(), xyz.clone()
        a = fixed_score_relations(score, xyz, scene="scene", backend="row", distance_device="cpu")
        b = fixed_score_relations(score, xyz, scene="scene", backend="row", distance_device="cpu")
        self.assertEqual(a, b)
        self.assertEqual(set(a), {"none", "morton", "hilbert", "nn_score", "random"})
        self.assertEqual(a["none"]["lags"]["1"]["correlation"]["valid_channels"], 6)
        torch.testing.assert_close(score, before, rtol=0, atol=0)
        torch.testing.assert_close(xyz, xyz_before, rtol=0, atol=0)
        for order in a.values():
            for stages in order["topology"].values():
                self.assertEqual(sum(stage["target_tokens"] for stage in stages), 33)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the read-only context/probability overhead profiler."""

import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from globalsplat.compression.bitstream import ScoreContextBitstream
from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load_script("profile_nfcgs_overhead_test", "profile_nfcgs_overhead.py")
summary = load_script("summarize_nfcgs_overhead_test", "summarize_nfcgs_overhead.py")


def distribution(value: float):
    return {"mean": value, "median": value, "p90": value, "stdev": 0.0, "min": value, "max": value}


def fake_report(suite: str, label: str, score_bytes: float, model_bytes: int):
    names = (
        "score_encode_ms",
        "score_decode_ms",
        "feature_codec_encode_ms",
        "feature_codec_decode_ms",
        "score_encode_peak_cuda_bytes",
        "score_decode_peak_cuda_bytes",
        "feature_codec_encode_peak_cuda_bytes",
        "feature_codec_decode_peak_cuda_bytes",
        "total_bytes",
        "score_payload_bytes",
        "score_entropy_string_bytes",
        "score_wrapper_bytes",
        "residual_bytes",
        "mean_bytes",
        "outer_and_residual_wrapper_bytes",
    )
    values = {name: distribution(10.0) for name in names}
    values["score_payload_bytes"] = distribution(score_bytes)
    values["score_entropy_string_bytes"] = distribution(score_bytes - 8.0)
    values["score_wrapper_bytes"] = distribution(8.0)
    values["total_bytes"] = distribution(score_bytes + 100.0)
    values["score_stream_count"] = 2
    values["score_roundtrip_exact_all"] = True
    footprint = {
        "parameter_count": model_bytes // 4,
        "parameter_bytes": model_bytes,
        "buffer_count": 0,
        "buffer_bytes": 0,
        "tensor_bytes": model_bytes,
        "serialized_state_dict_bytes": model_bytes + 100,
    }
    return {
        "schema_version": 1,
        "suite": suite,
        "label": label,
        "lambda_tag": "0p0064",
        "measure_scene_ids": ["a", "b"],
        "warmup_scene_ids": ["w"],
        "environment": {"gpu": "test", "torch": "test", "compressai": "test", "device": "cuda"},
        "footprints": {
            "score_path": footprint,
            "context_predictor": footprint,
            "probability_model": footprint,
        },
        "metrics": values,
    }


class OverheadProfileTests(unittest.TestCase):
    def test_footprint_and_context_wrapper_accounting(self):
        module = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        result = profile.module_footprint([("module", module)])
        self.assertEqual(result["parameter_count"], sum(p.numel() for p in module.parameters()))
        self.assertEqual(result["tensor_bytes"], result["parameter_bytes"] + result["buffer_bytes"])
        self.assertGreater(result["serialized_state_dict_bytes"], result["tensor_bytes"])
        self.assertEqual(profile.module_footprint([])["serialized_state_dict_bytes"], 0)

        strings = (b"abc", b"12345")
        payload = ScoreContextBitstream(rank=4, slice_channels=2, flags=0, strings=strings).pack()
        parts = profile.score_payload_parts(payload, contextual=True)
        self.assertEqual(parts["score_entropy_string_bytes"], 8)
        self.assertEqual(parts["score_wrapper_bytes"], 24 + 8 * 2)
        self.assertEqual(parts["score_stream_count"], 2)
        self.assertEqual(
            parts["score_payload_bytes"],
            parts["score_entropy_string_bytes"] + parts["score_wrapper_bytes"],
        )

    def test_paired_baselines_deltas_and_report(self):
        reports = [
            fake_report("context", "factorized", 1000.0, 4000),
            fake_report("context", "full", 900.0, 5000),
            fake_report("probability", "shared", 800.0, 6000),
            fake_report("probability", "split", 750.0, 6500),
        ]
        aggregate = summary.aggregate_reports(reports)
        rows = {(row["suite"], row["label"]): row for row in aggregate["rows"]}
        full = rows[("context", "full")]
        self.assertEqual(full["baseline"], "factorized")
        self.assertEqual(full["score_payload_delta_bytes"], -100.0)
        self.assertEqual(full["added_score_path_tensor_bytes"], 1000)
        self.assertEqual(full["model_payload_break_even_scenes"], 10.0)
        split = rows[("probability", "split")]
        self.assertEqual(split["baseline"], "shared")
        self.assertEqual(split["model_payload_break_even_scenes"], 10.0)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "REPORT.md"
            summary.write_markdown(path, aggregate)
            text = path.read_text(encoding="utf-8")
            self.assertIn("no-context Factorized", text)
            self.assertIn("Full+Shared", text)
            self.assertIn("| full | 2 | 892.0 | 8.0 | 900.0 (-100.00 B)", text)

    def test_scene_mismatch_is_rejected(self):
        baseline = fake_report("context", "factorized", 1000.0, 4000)
        candidate = fake_report("context", "full", 900.0, 5000)
        candidate["measure_scene_ids"] = ["different"]
        with self.assertRaisesRegex(ValueError, "scene mismatch"):
            summary.aggregate_reports([baseline, candidate])

    def test_small_context_codec_profile_is_read_only_and_verified(self):
        torch.manual_seed(7)
        codec = ObservableLowRank1DCodec(
            CodecConfig(
                texture_channels=4,
                geometry_channels=4,
                geometry_observable_channels=3,
                rank=4,
                residual_n=4,
                residual_m=6,
                adapter_hidden=6,
                score_mean_condition=True,
                score_channel_context=True,
                score_spatial_context=True,
                score_slice_channels=2,
                score_context_hidden=4,
            )
        ).eval()
        codec.update(force=True)
        model = SimpleNamespace(
            feature_codec=codec,
            gaussian_decoder=SimpleNamespace(decoded_token_centers=lambda value: value),
            encode_scene_tokens=lambda inputs: (inputs["texture"], inputs["geometry"]),
        )

        def batch(name):
            return {
                "scene_info": {"scene": [name]},
                "inputs": {
                    "texture": torch.randn(1, 32, 4),
                    "geometry": torch.randn(1, 32, 4),
                },
            }

        data = SimpleNamespace(test_dataloader=lambda: iter([batch("warm"), batch("a"), batch("b")]))
        footprint = profile.module_footprint([("feature_codec", codec)])
        footprints = {
            "codec": footprint,
            "score_path": footprint,
            "context_predictor": profile.module_footprint([]),
            "probability_model": footprint,
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "test").mkdir()
            (root / "test" / "index.json").write_text("{}", encoding="utf-8")
            checkpoint = root / "model.ckpt"
            checkpoint.touch()
            args = SimpleNamespace(
                checkpoint=checkpoint,
                label="full",
                suite="context",
                lambda_tag="0p0064",
                dataset_root=root,
                output_dir=root / "out",
                warmup_scenes=1,
                measure_scenes=2,
                num_workers=0,
                threads=1,
                seed=0,
                device="cpu",
                precision="fp32",
                expected_step=1,
            )
            metadata = {
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": 0,
                "checkpoint_step": 1,
                "codec": codec.config.to_dict(),
                "dataset": {},
            }
            with patch.object(profile, "parse_args", return_value=args), patch.object(
                profile, "load_model_and_data", return_value=(model, data, metadata, footprints)
            ), redirect_stdout(io.StringIO()):
                profile.main()
            report = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["warmup_scene_ids"], ["warm"])
            self.assertEqual(report["measure_scene_ids"], ["a", "b"])
            self.assertTrue(report["metrics"]["score_roundtrip_exact_all"])
            self.assertEqual(report["metrics"]["score_stream_count"], 4)
            self.assertEqual(report["metrics"]["score_wrapper_bytes"]["mean"], 24 + 8 * 4)


if __name__ == "__main__":
    unittest.main()

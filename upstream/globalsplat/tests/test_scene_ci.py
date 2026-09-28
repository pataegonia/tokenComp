"""Tests for paired full-scene NFCGS confidence intervals."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_nfcgs_scene_ci_test", ROOT / "scripts/analyze_nfcgs_scene_ci.py"
)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class SceneCITests(unittest.TestCase):
    def test_two_point_bd_rate_known_scale(self):
        value = ci.two_point_bd_rate(
            [100.0, 50.0], [20.0, 30.0], [90.0, 45.0], [20.0, 30.0]
        )
        self.assertAlmostEqual(float(value), -10.0)

    def test_full_grid_is_paired_and_bootstrapped_by_scene(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            scene_ids = ["scene_c", "scene_a", "scene_b"]
            for model_index, model in enumerate(ci.MODEL_ORDER):
                for lambda_index, lambda_tag in enumerate(ci.LAMBDA_ORDER):
                    result_dir = root / model / lambda_tag
                    result_dir.mkdir(parents=True)
                    records = []
                    for scene_index, scene in enumerate(scene_ids):
                        records.append(
                            {
                                "scene": scene,
                                "context_frame_ids": list(range(12)),
                                "target_frame_ids": list(range(12, 20)),
                                "actual_bytes": 1000.0 - 50.0 * model_index - 400.0 * lambda_index + scene_index,
                                "psnr": 24.0 + 0.02 * model_index - 0.8 * lambda_index + 0.01 * scene_index,
                                "ssim": 0.75 + 0.001 * model_index + 0.0001 * scene_index,
                                "lpips": 0.25 - 0.001 * model_index + 0.0001 * scene_index,
                            }
                        )
                    (result_dir / "actual_rate_per_scene.json").write_text(
                        json.dumps(records), encoding="utf-8"
                    )
                    averages = {
                        metric: float(np.mean([record[metric] for record in records]))
                        for metric in ci.METRICS
                    }
                    (result_dir / "scores_all_avg.json").write_text(
                        json.dumps(averages), encoding="utf-8"
                    )
                    entries.append(
                        {
                            "model": model,
                            "display_name": model,
                            "lambda_tag": lambda_tag,
                            "results_dir": str(result_dir),
                        }
                    )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "entries": entries}), encoding="utf-8"
            )
            values, scenes, frames, names, sources = ci.load_paired_data(manifest)
            self.assertEqual(scenes, sorted(scene_ids))
            self.assertEqual(values.shape, (3, 4, 2, 4))
            self.assertEqual(len(sources), 8)
            self.assertEqual(frames["scene_a"]["target_frame_ids"], list(range(12, 20)))
            boot_a = ci.bootstrap_means(values, draws=1000, seed=42, batch_size=17)
            boot_b = ci.bootstrap_means(values, draws=1000, seed=42, batch_size=64)
            np.testing.assert_array_equal(boot_a, boot_b)
            report = ci.build_report(values, boot_a, confidence=0.95, display_names=names)
            comparison = next(
                row
                for row in report["pairwise"]
                if row["reference"] == "linear_factorized"
                and row["candidate"] == "nonlinear_split"
                and row["lambda_tag"] == "0p0064"
            )
            self.assertGreater(comparison["rate_saving_pct"]["estimate"], 0)
            self.assertGreater(comparison["psnr_delta_db"]["estimate"], 0)
            self.assertEqual(comparison["scene_win_fraction"]["lower_rate"], 1.0)
            payload = {
                "scene_count": len(scenes),
                "bootstrap": {"draws": 1000, "confidence": 0.95},
                "analysis": report,
            }
            markdown = root / "REPORT.md"
            ci.write_markdown(markdown, payload)
            text = markdown.read_text(encoding="utf-8")
            self.assertIn("resamples complete scenes", text)
            self.assertIn("Approximate two-point BD-rate", text)

    def test_frame_mismatch_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for model in ci.MODEL_ORDER:
                for lambda_tag in ci.LAMBDA_ORDER:
                    result_dir = root / model / lambda_tag
                    result_dir.mkdir(parents=True)
                    context = [0, 1] if model == ci.MODEL_ORDER[0] else [0, 2]
                    records = [
                        {
                            "scene": "scene",
                            "context_frame_ids": context,
                            "target_frame_ids": [3],
                            "actual_bytes": 100.0,
                            "psnr": 20.0,
                            "ssim": 0.5,
                            "lpips": 0.4,
                        }
                    ]
                    (result_dir / "actual_rate_per_scene.json").write_text(
                        json.dumps(records), encoding="utf-8"
                    )
                    (result_dir / "scores_all_avg.json").write_text(
                        json.dumps({metric: records[0][metric] for metric in ci.METRICS}),
                        encoding="utf-8",
                    )
                    entries.append(
                        {
                            "model": model,
                            "display_name": model,
                            "lambda_tag": lambda_tag,
                            "results_dir": str(result_dir),
                        }
                    )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "entries": entries}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "context frame mismatch"):
                ci.load_paired_data(manifest)


if __name__ == "__main__":
    unittest.main()

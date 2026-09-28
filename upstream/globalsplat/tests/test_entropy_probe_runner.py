"""Exercise probe sampling, recoding, audit reports and caches on a small codec."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_runner_test", ROOT / "scripts/probe_nfcgs_entropy.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProbeRunnerTests(unittest.TestCase):
    def test_verified_pipeline_reports_additive_savings_and_separate_caches(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        codec = ObservableLowRank1DCodec(CodecConfig(
            texture_channels=4, geometry_channels=4, geometry_observable_channels=4,
            rank=4, residual_n=4, residual_m=6, adapter_hidden=6,
            score_mean_condition=True, score_channel_context=True, score_spatial_context=True,
            score_spatial_entropy="split", score_slice_channels=2, score_context_hidden=4)).eval()
        codec.update(force=True)
        model = SimpleNamespace(feature_codec=codec,
            compress_scene=lambda inputs: codec.compress(inputs["texture"], inputs["geometry"], inputs["positions"]))

        def batch(name):
            return dict(scene_info={"scene": [name]}, inputs=dict(
                texture=torch.randn(1, 32, 4), geometry=torch.randn(1, 32, 4),
                positions=torch.randn(1, 32, 3), frame_ids=torch.arange(12).unsqueeze(0)))

        a, b, c, d = (batch(name) for name in ("train_a", "train_b", "test_c", "test_d"))
        data = SimpleNamespace(train_dataloader=lambda: iter([a, a, b]),
                               test_dataloader=lambda: iter([c, c, d]))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "test"):
                (root / split).mkdir()
                (root / split / "index.json").write_text("{}", encoding="utf-8")
            checkpoint = root / "parent.ckpt"
            checkpoint.touch()
            args = SimpleNamespace(checkpoint=checkpoint, dataset_root=root, output_dir=root / "results",
                fit_scenes=2, eval_scenes=2, threads=1, prior_strength=16.0, num_workers=0,
                seed=0, device="cpu", precision="fp32", expected_step=10000, cache_scenes_per_split=1)
            metadata = dict(checkpoint=str(checkpoint), checkpoint_step=10000, codec=codec.config.to_dict())
            with patch.object(runner, "parse_args", return_value=args), \
                    patch.object(runner, "load_model_and_data", return_value=(model, data, metadata)), \
                    redirect_stdout(io.StringIO()):
                runner.main()
            report = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["fit_scene_ids"], ["train_a", "train_b"])
            self.assertEqual(report["eval_scene_ids"], ["test_c", "test_d"])
            metrics = report["metrics"]
            self.assertTrue(metrics["exact_feature_roundtrip_all"])
            self.assertEqual(metrics["overhead"]["score_container_bytes"], 24 + 8 * 4)
            self.assertEqual(metrics["variants"]["marginal"]["residual_bytes"],
                             metrics["variants"]["context"]["residual_bytes"])
            for result in metrics["variants"].values():
                self.assertAlmostEqual(result["total_saving_pct"],
                    result["score_only_saving_total_pct"] + result["residual_only_saving_total_pct"])
                self.assertEqual(len(result["score_minus_residual_saving_total_pp_bootstrap95"]), 2)
            paths = list((args.output_dir / "symbol_cache").glob("*/*.npz"))
            self.assertEqual(len(paths), 2)
            for path in paths:
                with np.load(path, allow_pickle=False) as cache:
                    self.assertIn("score_g0_even__symbols", cache)
            markdown = (args.output_dir / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Score-only saving", markdown)
            self.assertIn("Integer CDF", markdown)
            self.assertIn("not the globally best", markdown)
            rows = [json.loads(line) for line in (args.output_dir / "scenes.jsonl").read_text().splitlines()]
            rows[0]["exact_feature_roundtrip"] = False
            with self.assertRaisesRegex(ValueError, "verified scenes"):
                runner.summarize(rows, seed=0)
            # An overlapping TEST scene must not silently enter a held-out result.
            args.output_dir = root / "overlap_results"
            data.test_dataloader = lambda: iter([a])
            with patch.object(runner, "parse_args", return_value=args), \
                    patch.object(runner, "load_model_and_data", return_value=(model, data, metadata)), \
                    redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "overlap"):
                runner.main()


if __name__ == "__main__":
    unittest.main()

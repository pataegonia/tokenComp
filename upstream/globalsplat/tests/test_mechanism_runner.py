"""Matched inputs, partial resume, exact byte accounting and report outputs."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_complexity_profile import TinyModel
import test_mechanism as mechanism_tests

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


analysis = module("mechanism_runner", ROOT / "scripts/analyze_nfcgs_mechanism.py")
reporting = module("mechanism_reporting", ROOT / "scripts/report_nfcgs_mechanism.py")


class MechanismRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, folder, focused=False):
        root = Path(folder)
        source = root / "source.ckpt"
        source.write_bytes(b"fixture")
        specs, models = [], {}
        for label, schedule, order in (
                ("context_only_0p0128", "legacy", "none"),
                ("morton_legacy3_0p0128", "legacy", "morton"),
                ("hilbert_legacy3_0p0128", "legacy", "hilbert"),
                ("nn_score_0p0128", "legacy", "nn_score"),
                ("hilbert_quarter2_0p0128", "quarter2", "hilbert"),
                ("hilbert_dyadic4_0p0128", "dyadic4", "hilbert")):
            codec = mechanism_tests.MechanismTests.make(self, schedule)
            from dataclasses import replace
            codec.config = replace(codec.config, token_order=order, use_morton=order != "none",
                                   nn_order_backend="row" if order == "nn_score" else "legacy")
            model = TinyModel(codec)
            model.register_buffer("counter", torch.tensor(0))  # scalar state must hash correctly
            models[label] = model
            specs.append(dict(label=label, family=label, checkpoint=str(source), config=codec.config.to_dict()))
        manifest = dict(version=1, reference="hilbert_legacy3_0p0128", points=specs)
        args = SimpleNamespace(device="cpu", precision="fp32", scenes=2, seed=0, tokens=65, threads=1,
            num_workers=0, warmup=0, trace_scenes=1, fixed_nn_backend="row", prediction_ablation=True,
            dataset_root=root / "data", output=root / "run", resume=False)
        if focused:
            manifest["points"] = specs[:4]
            manifest["reference"] = "context_only_0p0128"
            args.scope = "legacy3-anchors"
            args.prediction_ablation = False
        cache = [dict(scene=f"scene_{i}", inputs=dict(texture=torch.randn(1, 65, 8), geometry=torch.randn(1, 65, 8)))
                 for i in range(2)]
        def load(spec, _):
            return copy.deepcopy(models[spec["label"]]), None, {}
        return manifest, args, cache, load

    def test_four_legacy3_models_share_inputs_and_only_observe_trained_context(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, args, cache, load = self.fixture(folder, focused=True)
            original = analysis.analyze_scene
            calls = 0
            def interrupted(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupted")
                return original(*a, **kw)
            with patch.object(analysis.profile, "load_model", side_effect=load), \
                 patch.object(analysis.profile, "cache_scenes", return_value=cache), \
                 patch.object(analysis, "fixed_score_relations", side_effect=AssertionError("out of scope")), \
                 patch.object(analysis, "analyze_scene", side_effect=interrupted), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    analysis.run(manifest, args)
            first = args.output / "points/context_only_0p0128/scene_0000.json"
            before = first.read_bytes()
            args.resume = True
            with patch.object(analysis.profile, "load_model", side_effect=load), \
                 patch.object(analysis.profile, "cache_scenes", side_effect=AssertionError("recached")), \
                 patch.object(analysis, "fixed_score_relations", side_effect=AssertionError("out of scope")), \
                 patch.object(analysis, "analyze_scene", wraps=original) as analyzed, \
                 contextlib.redirect_stdout(io.StringIO()):
                analysis.run(manifest, args)
            self.assertEqual(analyzed.call_count, 7)
            self.assertEqual(before, first.read_bytes())
            self.assertEqual(len(list((args.output / "points").glob("*/scene_*.json"))), 8)
            metadata, results = reporting.read_run(args.output)
            self.assertEqual(metadata["identity"]["scope"], "legacy3-anchors")
            self.assertIsNone(metadata["identity"]["fixed_nn_source"])
            for rows in results.values():
                self.assertEqual([row["scene"] for row in rows], [c["scene"] for c in cache])
                self.assertNotIn("fixed_score", rows[0])
                self.assertEqual(set(rows[0]["measurements"]), {"trained"})
            output = args.output / "report"
            summary = reporting.report(args.output, output)
            self.assertEqual(len(summary["models"]), 4)
            self.assertEqual(len(summary["candidate_minus_reference"]), 3)
            for stages in summary["models"].values():
                self.assertIsNone(stages[0]["anchor_normalized_l1"]["mean"])
                self.assertEqual(stages[0]["prediction_mse_ratio"]["mean"], 1.)
                self.assertEqual(sum(s["tokens"] for s in stages), 65)
                self.assertEqual(stages[1]["normalized_mae"]["scenes"], 2)
            for stem in ("01_target_anchor_score_distance", "02_prediction_residuals"):
                for suffix in ("png", "pdf"):
                    self.assertGreater((output / f"{stem}.{suffix}").stat().st_size, 1000)
            self.assertFalse((output / "01_fixed_score_relations.png").exists())
            with self.assertRaises(FileExistsError):
                reporting.report(args.output, output)
            args.prediction_ablation = True
            with self.assertRaisesRegex(ValueError, "disable prediction ablation"):
                analysis.run(manifest, args)

    def test_focused_manifest_accepts_only_the_four_legacy3_completed_models(self):
        folder = ROOT.parents[1] / "artifacts"
        manifest = analysis.load_manifest(folder / "nfcgs_legacy3_anchors_manifest_2026-10-16.json", "legacy3-anchors")
        self.assertEqual(len(manifest["points"]), 4)
        self.assertEqual({p["config"]["token_order"] for p in manifest["points"]},
                         {"none", "morton", "hilbert", "nn_score"})
        self.assertTrue(all(p["eval_complete"] for p in manifest["points"]))
        with self.assertRaisesRegex(ValueError, "four legacy3"):
            analysis.load_manifest(folder / "nfcgs_mechanism_manifest_2026-10-16.json", "legacy3-anchors")

    def test_partial_resume_preserves_completed_scenes_and_report_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, args, cache, load = self.fixture(folder)
            original = analysis.analyze_scene
            calls = 0
            def stop_after_one(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupted")
                return original(*a, **kw)
            with patch.object(analysis.profile, "load_model", side_effect=load), \
                 patch.object(analysis.profile, "cache_scenes", return_value=cache), \
                 patch.object(analysis, "analyze_scene", side_effect=stop_after_one), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    analysis.run(manifest, args)
            first = args.output / "points/hilbert_legacy3_0p0128/scene_0000.json"
            before = first.read_bytes()
            self.assertFalse((args.output / ".analysis-lock").exists())
            args.resume = True
            with patch.object(analysis.profile, "load_model", side_effect=load), \
                 patch.object(analysis.profile, "cache_scenes", side_effect=AssertionError("inputs were recached")), \
                 patch.object(analysis, "analyze_scene", wraps=original) as analyzed, \
                 contextlib.redirect_stdout(io.StringIO()):
                analysis.run(manifest, args)
            self.assertEqual(analyzed.call_count, 11)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(len(list((args.output / "points").glob("*/scene_*.json"))), 12)
            metadata, results = reporting.read_run(args.output)
            summary = reporting.summarize(metadata, results)
            self.assertEqual(len(summary["models"]), 6)
            self.assertEqual(len(summary["prediction_bypass_minus_trained"]), 6)
            self.assertEqual(len(summary["fixed_score"]), 5)
            output = args.output / "report"
            reporting.report(args.output, output)
            for name in ("01_fixed_score_relations", "02_causal_context_and_bits", "03_actual_byte_decomposition"):
                for extension in ("png", "pdf"):
                    self.assertGreater((output / f"{name}.{extension}").stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                reporting.report(args.output, output)
            args.seed = 5
            with self.assertRaisesRegex(ValueError, "resume protocol"), contextlib.redirect_stdout(io.StringIO()):
                analysis.run(manifest, args)
            self.assertEqual(first.read_bytes(), before)

    def test_scene_identity_or_byte_accounting_mismatch_rejects_report(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, args, cache, load = self.fixture(folder)
            with patch.object(analysis.profile, "load_model", side_effect=load), \
                 patch.object(analysis.profile, "cache_scenes", return_value=cache), \
                 contextlib.redirect_stdout(io.StringIO()):
                analysis.run(manifest, args)
            path = args.output / "points/context_only_0p0128/scene_0000.json"
            value = json.loads(path.read_text())
            value["scene"] = "wrong_scene"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "scene pairing"):
                reporting.read_run(args.output)
            value["scene"] = cache[0]["scene"]
            value["measurements"]["trained"]["total_bytes"] += 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "byte decomposition"):
                reporting.read_run(args.output)

    def test_manifest_selects_six_completed_matched_checkpoints(self):
        path = ROOT.parents[1] / "artifacts/nfcgs_mechanism_manifest_2026-10-16.json"
        manifest = analysis.load_manifest(path)
        self.assertEqual(len(manifest["points"]), 6)
        self.assertTrue(all(p["eval_complete"] for p in manifest["points"]))

    def test_scene_bootstrap_uses_paired_differences_not_token_counts(self):
        a, b = [100, 200, 300], [90, 190, 290]
        described = reporting.describe([x-y for x, y in zip(a, b)])
        self.assertEqual(described, dict(mean=10., ci95=[10., 10.], scenes=3))

    def test_slurm_lf_shebang_and_syntax(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.exists():
            self.skipTest("Git Bash unavailable")
        for name in ("analyze_nfcgs_mechanism.slurm", "analyze_nfcgs_legacy3_anchors.slurm",
                     "analyze_nfcgs_hilbert_schedule_anchors.slurm"):
            path = ROOT / "scripts/slurm" / name
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"#!/usr/bin/env bash\n"))
            self.assertNotIn(b"\r", data)
            result = subprocess.run([str(bash), "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_slurm_plan_and_run_keep_protocol_and_create_job_tmpdir(self):
        self.check_slurm_protocol(focused=False)

    def test_focused_slurm_runs_four_model_scope_without_prediction_bypass(self):
        self.check_slurm_protocol(focused=True)

    def test_schedule_slurm_reuses_source_run_without_prediction_bypass(self):
        self.check_slurm_protocol(focused=False, schedules=True)

    def check_slurm_protocol(self, focused, schedules=False):
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.exists():
            self.skipTest("Git Bash unavailable")
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            name = "analyze_nfcgs_legacy3_anchors.slurm" if focused else "analyze_nfcgs_mechanism.slurm"
            prefix = "ANCHOR" if focused else "MECHANISM"
            tmp_name = "l3_anchors" if focused else "mechanism"
            if schedules:
                name, prefix, tmp_name = "analyze_nfcgs_hilbert_schedule_anchors.slurm", "SCHEDULE", "hsched_anchors"
            for relative in (f"scripts/slurm/{name}", "globalsplat/compression/mechanism.py",
                    "globalsplat/compression/score_context.py", "scripts/profile_nfcgs_complexity.py",
                    "scripts/analyze_nfcgs_mechanism.py", "scripts/report_nfcgs_mechanism.py"):
                destination = repo / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            (repo / "manifest.json").write_text("{}")
            profile = repo / "fake_conda/etc/profile.d"
            profile.mkdir(parents=True)
            (profile / "conda.sh").write_text("conda() { return 0; }\n", newline="\n")
            fake = repo / "fake_bin"
            fake.mkdir()
            (fake / "conda").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n', newline="\n")
            (fake / "srun").write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n', newline="\n")
            (fake / "python").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_LOG"\n'
                'printf "%s\\n" "${TMPDIR:-}" >> "$FAKE_TMP_LOG"\n', newline="\n")
            for fake_tool in ("conda", "srun", "python"):
                (fake / fake_tool).chmod(0o755)
            env = dict(os.environ, REPO_DIR=repo.as_posix(),
                FAKE_CONDA_BASE=(repo/"fake_conda").as_posix(), FAKE_LOG=(repo/"calls.txt").as_posix(),
                FAKE_TMP_LOG=(repo/"tmp.txt").as_posix(), NFCGS_TMP_ROOT="checkpoint_tmp")
            env[f"{prefix}_MANIFEST"] = (repo/"manifest.json").as_posix()
            for key in (f"{prefix}_SCENES", f"{prefix}_TRACE_SCENES", f"{prefix}_RESUME", f"{prefix}_OUTPUT",
                        f"{prefix}_PREDICTION_ABLATION", "SLURM_JOB_ID"):
                env.pop(key, None)
            def shell(option):
                return subprocess.run([str(bash), "-c", 'export PATH="$(pwd)/fake_bin:$PATH"; '
                    f'bash scripts/slurm/{name} ' + option],
                    cwd=repo, env=env, capture_output=True, text=True)
            planned = shell("plan")
            self.assertEqual(planned.returncode, 0, planned.stdout + planned.stderr)
            self.assertFalse((repo / "checkpoint_tmp").exists())
            planned_call = (repo / "calls.txt").read_text()
            for flag in ("--scenes 256", "--trace-scenes 8", "--plan"):
                self.assertIn(flag, planned_call)
            if schedules:
                self.assertIn("--scope hilbert-schedule-anchors",planned_call)
                self.assertIn("--input-run",planned_call)
                self.assertIn("job_454995",planned_call)
                self.assertIn("--expected-input-fingerprint 069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29",planned_call)
                self.assertNotIn("--prediction-ablation",planned_call)
            elif focused:
                self.assertIn("--scope legacy3-anchors", planned_call)
                self.assertNotIn("--prediction-ablation", planned_call)
            else:
                self.assertIn("--prediction-ablation", planned_call)
            env.update(SLURM_JOB_ID="12345")
            env.update({f"{prefix}_SCENES": "3", f"{prefix}_TRACE_SCENES": "2"})
            ran = shell("")
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertTrue((repo / f"checkpoint_tmp/{tmp_name}_12345").is_dir())
            calls = (repo / "calls.txt").read_text().splitlines()
            self.assertEqual(len(calls), 3)
            self.assertIn(f"--scenes {256 if schedules else 3} --trace-scenes 2", calls[1])
            self.assertIn("scripts/report_nfcgs_mechanism.py", calls[2])
            self.assertIn("report_12345", calls[2])
            self.assertEqual((repo / "tmp.txt").read_text().splitlines()[-2:],
                             [f"checkpoint_tmp/{tmp_name}_12345"] * 2)


if __name__ == "__main__":
    unittest.main()

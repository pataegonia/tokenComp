"""Recovery preserves protected trainings and persists every accepted Slurm ID."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from globalsplat.compression import ObservableLowRank1DCodec
from test_score_path_ablation import config, MINIMAL

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sort_context_recovery", ROOT / "scripts/recover_nfcgs_sort_context_ablation.py")
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(recovery)


def small_config(point):
    return config(**MINIMAL, token_order=point.order, score_spatial_context=point.context,
        score_context_schedule=point.schedule, score_spatial_stages=point.stages,
        score_spatial_kernel=5, score_context_quantization="ste",
        nn_order_backend="knn" if point.protected else "legacy",
        nn_order_distance_device="input" if point.protected else "cpu")


def checkpoint(codec_config, step):
    codec = ObservableLowRank1DCodec(codec_config)
    value = dict(state_dict={"model.feature_codec." + name: value for name, value in codec.state_dict().items()},
        feature_codec_config=codec_config.to_dict(), score_mean_offset_enabled=True, global_step=step,
        optimizer_states=[{"state": {0: {"step": torch.tensor(step)}}}],
        lr_schedulers=[{"last_epoch": step}], loops={"fit_loop": {"epoch_progress": 1}})
    value["pytorch-lightning_version"] = "2.4.0"
    return value


class Cluster:
    def __init__(self, states):
        self.states = states
        self.calls = []
        self.fail_call = None

    def command(self, args, **kwargs):
        if args[0] == "sacct":
            return "\n".join(f"{job}|{state}|test|" for job, state in self.states.items())
        if args[0] == "squeue":
            return "\n".join(f"{job}|{state}" for job, state in self.states.items() if state in recovery.ACTIVE)
        if args[0] == "sbatch":
            self.calls.append((args, kwargs["env"].copy()))
            if self.fail_call == len(self.calls):
                raise subprocess.CalledProcessError(1, args, stderr="QOSMaxSubmitJobPerUserLimit")
            job = str(90000 + len(self.calls))
            self.states[job] = "PENDING"
            return job + ";fake"
        raise AssertionError(args)


class SortContextRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, directory):
        repo = Path(directory).resolve()
        for relative in ("scripts/run_nfcgs.py", "scripts/prepare_nfcgs_nn_score_resume.py",
                         "scripts/recover_nfcgs_sort_context_ablation.py", "scripts/slurm/nfcgs_order_context_settings.sh",
                         "scripts/slurm/train_nfcgs_order_context_recovery.slurm", "scripts/slurm/eval_nfcgs_order_context.slurm"):
            destination = repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        parent = repo / "parent.ckpt"
        torch.save(checkpoint(config(), 10000), parent)
        old_dir = repo / "logs/slurm/sort_context_ablation_rd_20261013"
        old_dir.mkdir(parents=True)
        states = {}
        for index, point in enumerate(recovery.points()):
            if point.protected:
                train = {"0.0256": "450934", "0.0064": "450937", "0.0128": "450939"}[point.rate]
                evaluation = str(int(train) + 1)
                if point.rate == "0.0256":
                    path = repo / "logs/slurm/nn_score_resume_retry_450904.state"
                else:
                    path = repo / "logs/slurm/nn_score_rd_gpu_20261013" / f"nn_score_legacy3_{point.rate.replace('.', 'p')}.state"
                signature = "a" * 64
                states[train] = "RUNNING"
            else:
                if index >= 13:  # dyadic4 .0128/.0256 were never submitted.
                    continue
                train = str(450977 + (index - 3) * 2)
                evaluation = str(int(train) + 1) if index < 12 else "-"
                path = old_dir / f"{point.key}.state"
                values = ["order-context-v1", str(repo), str(parent), point.order, point.schedule,
                    str(point.stages), "5", point.rate, "50000", "5000", "", "train-time-limit", "6-00:00:00"]
                if not point.context:
                    values += ["spatial-context", "false"]
                signature = hashlib.sha256(b"".join(value.encode() + b"\0" for value in values)).hexdigest()
                states[train] = "CANCELLED"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"v1\t{signature}\t{train}\t{evaluation}\n")
            if evaluation != "-":
                states[evaluation] = "CANCELLED"
        return repo, old_dir, parent, repo / "logs/slurm/recovery.json", Cluster(states)

    def run_recovery(self, fixture, **kwargs):
        repo, old_dir, parent, state, cluster = fixture
        with patch.object(recovery.Point, "config", small_config), patch.object(recovery, "command", cluster.command), \
             patch.dict(os.environ, {"USER": "test"}), contextlib.redirect_stdout(io.StringIO()):
            return recovery.recover(repo, old_dir, state, parent, **kwargs)

    def test_preview_is_read_only_and_complete_recovery_reuses_all_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            repo, old_dir, parent, state, cluster = fixture
            original = {p: p.read_bytes() for p in old_dir.glob("*.state")}
            self.run_recovery(fixture, dry_run=True)
            self.assertFalse(state.exists())
            self.assertFalse(cluster.calls)
            manifest = self.run_recovery(fixture)
            self.assertEqual(len(cluster.calls), 27)  # 3 protected evals + 12 train/eval pairs.
            self.assertEqual(len(manifest["points"]), 15)
            trains = [call for call in cluster.calls if "train_nfcgs_order_context_recovery.slurm" in call[0][-1]]
            self.assertEqual(len(trains), 12)
            self.assertTrue(all(call[1]["TOKEN_ORDER"] in ("none", "hilbert") for call in trains))
            self.assertTrue(all(call[1]["RECOVERY_MODE"] == "restart" for call in trains))
            for point in recovery.points()[:3]:
                record = manifest["points"][point.key]
                self.assertIn(record["train_job"], recovery.PROTECTED)
                call = next(call for call in cluster.calls if f"--dependency=afterok:{record['train_job']}" in call[0])
                self.assertEqual(call[1]["EVAL_RUN_DIR"], record["run_dir"])
            self.run_recovery(fixture)
            self.assertEqual(len(cluster.calls), 27)
            for path, data in original.items():
                self.assertEqual(path.read_bytes(), data)

    def test_partial_eval_rejection_keeps_new_training_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            repo, old_dir, parent, state, cluster = fixture
            cluster.fail_call = 5  # after three NN evals and the first new training.
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_recovery(fixture)
            saved = json.loads(state.read_text())
            self.assertEqual(saved["points"]["context_only_0p0064"]["train_job"], "90004")
            self.assertIsNone(saved["points"]["context_only_0p0064"]["eval_job"])
            cluster.fail_call = None
            self.run_recovery(fixture)
            self.assertEqual(len(cluster.calls), 28)
            trains = [call for call in cluster.calls if "train_nfcgs_order_context_recovery.slurm" in call[0][-1]]
            self.assertEqual(len(trains), 12)
            self.assertTrue(any("--dependency=afterok:90004" in call[0] for call in cluster.calls[5:]))

    def test_existing_checkpoint_resumes_and_copy_preserves_optimizer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            repo, old_dir, parent, state, cluster = fixture
            point = next(p for p in recovery.points() if p.key == "context_quarter2_0p0064")
            source = point.run_root(repo) / "job_450989/checkpoints/nfcgs_main/version_0/step000005000.ckpt"
            source.parent.mkdir(parents=True)
            torch.save(checkpoint(small_config(point), 5000), source)
            before = source.read_bytes()
            manifest = self.run_recovery(fixture)
            action = manifest["points"][point.key]
            self.assertEqual(action["step"], 5000)
            call = next(call for call in cluster.calls if call[1].get("RECOVERY_CHECKPOINT") == str(source))
            self.assertEqual(call[1]["RECOVERY_MODE"], "resume")
            destination = Path(action["run_dir"]) / "checkpoints/nfcgs_main/version_0/resume.ckpt"
            with contextlib.redirect_stdout(io.StringIO()):
                recovery.main(["--copy-checkpoint", str(source), "--copy-to", str(destination),
                               "--expected-config", json.dumps(small_config(point).to_dict())])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(destination.read_bytes(), before)
            loaded = torch.load(destination, weights_only=False)
            self.assertEqual(loaded["global_step"], 5000)
            self.assertEqual(loaded["optimizer_states"][0]["state"][0]["step"].item(), 5000)
            self.assertEqual(loaded["lr_schedulers"][0]["last_epoch"], 5000)

    def test_cancelled_training_with_final_checkpoint_needs_only_one_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            repo, old_dir, parent, state, cluster = fixture
            point = next(p for p in recovery.points() if p.key == "context_only_0p0064")
            source = point.run_root(repo) / "job_450977/checkpoints/nfcgs_main/version_0/step000050000.ckpt"
            source.parent.mkdir(parents=True)
            torch.save(checkpoint(small_config(point), 50000), source)
            manifest = self.run_recovery(fixture)
            self.assertEqual(manifest["points"][point.key]["mode"], "eval_only")
            self.assertEqual(len(cluster.calls), 26)
            self.run_recovery(fixture)
            self.assertEqual(len(cluster.calls), 26)

    def test_unknown_status_or_terminated_protected_job_fails_before_submission(self):
        for variant in ("unknown", "protected_cancelled", "dangling_eval", "bad_state"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                repo, old_dir, parent, state, cluster = fixture
                if variant == "unknown":
                    cluster.states.pop("450989")
                elif variant == "protected_cancelled":
                    cluster.states["450934"] = "CANCELLED"
                elif variant == "dangling_eval":
                    cluster.states["450978"] = "PENDING"
                else:
                    (old_dir / "context_only_0p0064.state").write_text("invalid\n")
                with self.assertRaises(ValueError):
                    self.run_recovery(fixture)
                self.assertFalse(cluster.calls)
                self.assertFalse(state.exists())

    def test_nn_eval_only_does_not_submit_any_training(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            manifest = self.run_recovery(fixture, evals_only=True)
            cluster = fixture[-1]
            self.assertEqual(len(cluster.calls), 3)
            self.assertTrue(all(call[0][-1].endswith("eval_nfcgs_order_context.slurm") for call in cluster.calls))
            self.assertEqual({entry["train_job"] for entry in manifest["points"].values()}, set(recovery.PROTECTED))

    def test_completed_protected_training_evaluates_without_an_expired_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            repo, old_dir, parent, state, cluster = fixture
            point = recovery.points()[0]
            cluster.states["450934"] = "COMPLETED"
            source = point.run_root(repo) / "resume_job_450934/checkpoints/nfcgs_main/version_0/step000050000.ckpt"
            source.parent.mkdir(parents=True)
            torch.save(checkpoint(small_config(point), 50000), source)
            manifest = self.run_recovery(fixture, evals_only=True)
            self.assertEqual(manifest["points"][point.key]["train_job"], "450934")
            call = cluster.calls[0]
            self.assertFalse(any(arg.startswith("--dependency=") for arg in call[0]))
            self.assertEqual(call[1]["TRAIN_JOB_ID"], "450934")
            self.assertEqual(len(cluster.calls), 3)

    def test_recovery_slurm_has_valid_lf_shebang_and_syntax(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.exists():
            self.skipTest("Git Bash unavailable")
        path = ROOT / "scripts/slurm/train_nfcgs_order_context_recovery.slurm"
        data = path.read_bytes()
        self.assertTrue(data.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", data)
        result = subprocess.run([str(bash), "-n", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_slurm_executes_resume_or_weights_only_with_the_same_output_path(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.exists():
            self.skipTest("Git Bash unavailable")
        for mode in ("restart", "resume"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                repo, old_dir, parent, state, cluster = self.fixture(directory)
                fake = repo / "fake_bin"
                fake.mkdir()
                base = repo / "fake_conda"
                profile = base / "etc/profile.d"
                profile.mkdir(parents=True)
                (profile / "conda.sh").write_text("conda() { return 0; }\n", newline="\n")
                (fake / "conda").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n', newline="\n")
                (fake / "python").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_RECOVERY_LOG"\nprintf "%s\\n" "$TMPDIR" >> "$FAKE_RECOVERY_TMP_LOG"\n', newline="\n")
                point = next(p for p in recovery.points() if p.key == "context_quarter2_0p0064")
                env = recovery.environment(point, repo)
                log = repo / "runner.txt"
                tmp_log = repo / "tmpdir.txt"
                tmp_root = repo / "checkpoint_tmp"
                env.update(RECOVERY_MODE=mode, RECOVERY_CHECKPOINT=parent.as_posix(),
                    RECOVERY_CODEC_CONFIG=json.dumps(small_config(point).to_dict()), SLURM_JOB_ID="12345",
                    FAKE_CONDA_BASE=base.as_posix(), FAKE_RECOVERY_LOG=log.as_posix(),
                    FAKE_RECOVERY_TMP_LOG=tmp_log.as_posix(), NFCGS_TMP_ROOT="checkpoint_tmp")
                result = subprocess.run([str(bash), "-c", 'export PATH="$(pwd)/fake_bin:$PATH"; '
                    'bash scripts/slurm/train_nfcgs_order_context_recovery.slurm'], cwd=repo, env=env,
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = log.read_text().splitlines()
                expected_tmp = tmp_root / "job_12345"
                self.assertTrue(expected_tmp.is_dir())
                self.assertEqual(tmp_log.read_text().splitlines(), ["checkpoint_tmp/job_12345"] * len(calls))
                self.assertIn("CHECKPOINT_TMPDIR=checkpoint_tmp/job_12345", result.stdout)
                train = calls[-1]
                self.assertIn("scripts/run_nfcgs.py train --devices 2 --launcher srun", train)
                self.assertIn("--token-order none", train)
                self.assertIn("--score-context-schedule quarter2 --score-spatial-stages 2", train)
                self.assertIn("--batch-size 2 --accumulate 2", train)
                output = point.run_root(repo).as_posix() + "/recovery_job_12345"
                self.assertIn("--output " + output, train)
                if mode == "resume":
                    self.assertEqual(len(calls), 2)
                    self.assertIn("--copy-checkpoint " + parent.as_posix(), calls[0])
                    self.assertIn("--checkpoint " + output + "/checkpoints/nfcgs_main/version_0/resume.ckpt --resume", train)
                    self.assertNotIn("--allow-score-path-conversion", train)
                else:
                    self.assertEqual(len(calls), 1)
                    self.assertIn("--checkpoint " + parent.as_posix() + " --allow-score-path-conversion", train)
                    self.assertNotIn("--resume", train)


if __name__ == "__main__":
    unittest.main()

"""Fake Slurm verifies preflight, replacement order and resume dependencies."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")
FILES = ("submit_nfcgs_nn_score_resume.sh", "train_nfcgs_nn_score_resume.slurm",
         "nfcgs_order_context_settings.sh", "eval_nfcgs_order_context.slurm")


@unittest.skipUnless(BASH, "bash unavailable")
class NNScoreResumeSlurmTests(unittest.TestCase):
    def fixture(self, directory):
        repo = Path(directory)
        scripts = repo / "scripts/slurm"
        scripts.mkdir(parents=True)
        for name in FILES:
            shutil.copyfile(ROOT / "scripts/slurm" / name, scripts / name)
        (repo / "scripts/run_nfcgs.py").write_text("# fake runner\n")
        (repo / "scripts/prepare_nfcgs_nn_score_resume.py").write_text("# fake validator\n")
        checkpoint_dir = repo / "source/job_448650/checkpoints/nfcgs_main/version_0"
        checkpoint_dir.mkdir(parents=True)
        for step in (5000, 10000):
            (checkpoint_dir / f"step{step:09d}.ckpt").write_bytes(b"checkpoint")
        (repo / "queue.txt").write_text("448650\n448651\n448652\n", newline="\n")
        (repo / "counter.txt").write_text("0\n", newline="\n")
        fake = repo / "fake_bin"
        fake.mkdir()
        stubs = {
            "python": '#!/usr/bin/env bash\n'
                'printf "PYTHON %s\\n" "$*" >> events.txt\n'
                'if [[ "$*" == *prepare_nfcgs_nn_score_resume.py* && "${FAKE_BAD_CKPT:-0}" == 1 ]]; then exit 9; fi\n',
            "squeue": '#!/usr/bin/env bash\n'
                '[[ "${FAKE_QUEUE_FAIL:-0}" != 1 ]] || exit 9\ncat queue.txt\n',
            "scancel": '#!/usr/bin/env bash\n'
                'printf "CANCEL %s\\n" "$*" >> events.txt\n'
                'while read -r job; do [[ ",$1," == *",${job},"* ]] || printf "%s\\n" "$job"; done < queue.txt > queue.tmp\n'
                'mv queue.tmp queue.txt\n',
            "sbatch": '#!/usr/bin/env bash\n'
                'count=$(cat counter.txt); count=$((count+1)); echo "$count" > counter.txt\n'
                'printf "SBATCH %s ORDER=%s BACKEND=%s DISTANCE=%s EVAL_DIR=%s\\n" "$*" '
                '"$TOKEN_ORDER" "$NN_ORDER_BACKEND" "$NN_ORDER_DISTANCE_DEVICE" "${EVAL_RUN_DIR:-}" >> events.txt\n'
                'if [[ "$*" == *--dependency=* && "${FAKE_FAIL_EVAL:-0}" == 1 ]]; then exit 9; fi\n'
                'echo "$((90000+count));fake"\n',
        }
        for name, text in stubs.items():
            (fake / name).write_text(text, encoding="utf-8", newline="\n")
            (fake / name).chmod(0o755)
        env = os.environ.copy()
        for key in ("RESUME_CHECKPOINT", "NN_SCORE_RESUME_STATE_FILE", "NN_SCORE_TARGET_STEPS",
                    "FAKE_BAD_CKPT", "FAKE_QUEUE_FAIL", "FAKE_FAIL_EVAL", "EVAL_RUN_DIR",
                    "NN_ORDER_DISTANCE_DEVICE"):
            env.pop(key, None)
        env.update(REPO_DIR=repo.as_posix(), NN_SCORE_SOURCE_RUN="source/job_448650", USER="test")
        return repo, env

    def run_script(self, repo, env, flags="--replace-jobs 448650,448651"):
        return subprocess.run([BASH, "-c", 'export PATH="$(pwd)/fake_bin:$PATH"; '
            'bash scripts/slurm/submit_nfcgs_nn_score_resume.sh ' + flags],
            cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8")

    def test_validate_then_cancel_then_resume_with_afterok_and_idempotent_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env = self.fixture(directory)
            # Dedicated settings must override stale unrelated experiment knobs.
            env.update(TOKEN_ORDER="nn_xyz", RATE_LAMBDA="0.0064")
            result = self.run_script(repo, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            events = (repo / "events.txt").read_text().splitlines()
            validation = next(i for i, e in enumerate(events) if "prepare_nfcgs_nn_score_resume.py" in e)
            cancel = next(i for i, e in enumerate(events) if e.startswith("CANCEL"))
            submissions = [(i, e) for i, e in enumerate(events) if e.startswith("SBATCH")]
            self.assertLess(validation, cancel)
            self.assertLess(cancel, submissions[0][0])
            self.assertEqual(events[cancel], "CANCEL 448650,448651")
            self.assertEqual((repo / "queue.txt").read_text(), "448652\n")
            self.assertIn("step000010000.ckpt", events[validation])
            self.assertTrue(any("--resume" in e and "--devices 2 --batch-size 2 --accumulate 2" in e for e in events))
            self.assertEqual(len(submissions), 2)
            self.assertIn("--dependency=afterok:90001", submissions[1][1])
            self.assertIn("resume_job_90001", submissions[1][1])
            for _, row in submissions:
                self.assertIn("ORDER=nn_score BACKEND=knn DISTANCE=cpu", row)
            rerun = self.run_script(repo, env)
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            final = (repo / "events.txt").read_text().splitlines()
            self.assertEqual(sum(e.startswith("SBATCH") for e in final), 2)
            self.assertEqual(sum(e.startswith("CANCEL") for e in final), 1)

    def test_preview_bad_checkpoint_and_failed_queue_never_change_jobs(self):
        for variant in ("preview", "bad", "queue_fail", "active_without_replace"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                repo, env = self.fixture(directory)
                flags = "--replace-jobs 448650,448651"
                if variant == "preview":
                    flags += " --dry-run"
                elif variant == "bad":
                    env["FAKE_BAD_CKPT"] = "1"
                elif variant == "queue_fail":
                    env["FAKE_QUEUE_FAIL"] = "1"
                else:
                    flags = ""
                result = self.run_script(repo, env, flags)
                self.assertEqual(result.returncode == 0, variant == "preview", result.stderr)
                events = (repo / "events.txt").read_text()
                self.assertNotIn("CANCEL", events)
                self.assertNotIn("SBATCH", events)
                self.assertEqual((repo / "queue.txt").read_text(), "448650\n448651\n448652\n")

    def test_partial_eval_failure_reuses_training_and_does_not_cancel_again(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env = self.fixture(directory)
            env["FAKE_FAIL_EVAL"] = "1"
            first = self.run_script(repo, env)
            self.assertNotEqual(first.returncode, 0)
            self.assertIn("eval failed", first.stderr)
            env.pop("FAKE_FAIL_EVAL")
            retry = self.run_script(repo, env)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            events = (repo / "events.txt").read_text().splitlines()
            self.assertEqual(sum(e.startswith("CANCEL") for e in events), 1)
            self.assertEqual(sum(e.startswith("SBATCH") for e in events), 3)
            self.assertIn("REUSING TRAIN_JOB=90001", retry.stdout)
            self.assertIn("--dependency=afterok:90001", events[-1])

    def test_gpu_distance_option_reaches_both_jobs_and_has_its_own_state(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env = self.fixture(directory)
            env["NN_ORDER_DISTANCE_DEVICE"] = "input"
            result = self.run_script(repo, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            events = (repo / "events.txt").read_text().splitlines()
            self.assertTrue(any("--nn-order-distance-device input" in e for e in events))
            submissions = [e for e in events if e.startswith("SBATCH")]
            self.assertEqual(len(submissions), 2)
            self.assertTrue(all("DISTANCE=input" in e for e in submissions))
            self.assertIn("nn_knn_input_k64", submissions[1])
            self.assertTrue((repo / "logs/slurm/nn_score_resume_448650_input_knn.state").exists())
            self.assertFalse((repo / "logs/slurm/nn_score_resume_448650_cpu_knn.state").exists())
            # An explicitly reused CPU/GPU state file must fail before actions.
            env.update(NN_ORDER_DISTANCE_DEVICE="cpu", NN_SCORE_RESUME_STATE_FILE=
                "logs/slurm/nn_score_resume_448650_input_knn.state")
            changed = self.run_script(repo, env)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("mismatched resume state", changed.stderr)
            final = (repo / "events.txt").read_text().splitlines()
            self.assertEqual(sum(e.startswith("SBATCH") for e in final), 2)
            self.assertEqual(sum(e.startswith("CANCEL") for e in final), 1)

    def test_retry_state_replaces_failed_pair_without_reusing_old_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env = self.fixture(directory)
            env["NN_ORDER_DISTANCE_DEVICE"] = "input"
            first = self.run_script(repo, env)
            self.assertEqual(first.returncode, 0, first.stderr)
            old_state = repo / "logs/slurm/nn_score_resume_448650_input_knn.state"
            old_bytes = old_state.read_bytes()
            # Training 90001 failed; only its dependent eval remains active.
            (repo / "queue.txt").write_text("90002\n448652\n", newline="\n")
            env["NN_SCORE_RESUME_STATE_FILE"] = "logs/slurm/retry_failed.state"
            retry = self.run_script(repo, env, "--replace-jobs 90001,90002")
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertIn("TRAIN_JOB=90003 EVAL_JOB=90004", retry.stdout)
            self.assertEqual(old_state.read_bytes(), old_bytes)
            self.assertEqual((repo / "queue.txt").read_text(), "448652\n")
            events = (repo / "events.txt").read_text().splitlines()
            self.assertIn("CANCEL 90002", events)
            self.assertIn("--dependency=afterok:90003", events[-1])
            rerun = self.run_script(repo, env, "--replace-jobs 90001,90002")
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            final = (repo / "events.txt").read_text().splitlines()
            self.assertEqual(sum(e.startswith("SBATCH") for e in final), 4)
            self.assertEqual(sum(e.startswith("CANCEL") for e in final), 2)

    def test_scripts_are_lf_and_resumed_output_matches_eval_override(self):
        for name in FILES:
            path = ROOT / "scripts/slurm" / name
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"#!/usr/bin/env bash\n"))
            self.assertNotIn(b"\r", data)
            result = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        train = (ROOT / "scripts/slurm/train_nfcgs_nn_score_resume.slurm").read_text()
        self.assertIn("--gres=gpu:normal:2", train)
        self.assertIn("--devices 2", train)
        self.assertIn("--batch-size 2 --accumulate 2", train)
        self.assertIn("--checkpoint \"${COPY}\" --resume", train)
        self.assertNotIn("allow-score-path-conversion", train)


if __name__ == "__main__":
    unittest.main()

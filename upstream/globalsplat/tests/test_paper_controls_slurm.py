"""Check costly experiment submissions using fake sbatch and the real CLI."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import test_token_order_context_slurm as existing


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "submit_nfcgs_paper_controls.sh"
STATE = "logs/slurm/paper_controls_20261018"


@unittest.skipUnless(existing.BASH, "bash unavailable")
class PaperControlsSlurmTests(unittest.TestCase):
    def fixture(self, directory):
        helper = existing.OrderContextSlurmTests()
        repo, env, log = helper.rd_fixture(directory)
        env.pop("PAPER_CONTROLS_STATE_DIR", None)
        shutil.copyfile(ROOT / "scripts/slurm" / SCRIPT, repo / "scripts/slurm" / SCRIPT)
        shutil.copyfile(ROOT / "scripts/run_nfcgs.py", repo / "scripts/run_nfcgs.py")
        fake = repo / "fake_bin/sbatch"
        text = fake.read_text().replace("EVAL_DIR=%s\\n", "EVAL_DIR=%s SLICE=%s\\n")
        text = text.replace('"${EVAL_RUN_DIR:-}" >>', '"${EVAL_RUN_DIR:-}" "${SCORE_SLICE_CHANNELS:-16}" >>')
        fake.write_text(text, newline="\n")
        return helper, repo, env, log

    def test_exact_planned_points_dependencies_and_idempotent_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            helper, repo, env, log = self.fixture(directory)
            env.update(TOKEN_ORDER="nn_score", CONTEXT_SCHEDULE="single", STAGES="1", KERNEL="7",
                       SPATIAL_CONTEXT="true", SCORE_SLICE_CHANNELS="56", MAX_STEPS="10", CHECKPOINT_EVERY="1",
                       NN_ORDER_BACKEND="knn", NN_ORDER_DISTANCE_DEVICE="input", SCORE_ORDER_SCALE="wrong.json",
                       EVAL_RUN_DIR="wrong-eval", RD_LAMBDAS="0.0064", EXTREMES_STATE_DIR="wrong-state")
            result = helper.rd_run(repo, env, script=SCRIPT)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            submissions = log.read_text().splitlines()
            records = [r.split("\t") for r in (repo / STATE / "submissions.tsv").read_text().splitlines()[1:]]
            plan = json.loads((ROOT / "config/plans/nfcgs_paper_experiments_20261018.json").read_text(encoding="utf-8"))
            controls = [r for r in plan["future_runs"] if r["kind"] == "existing_codec_control"]
            expected = {(r["codec"]["token_order"], r["codec"]["score_context_schedule"],
                         str(r["codec"]["score_spatial_stages"]), r["proposed_rate_lambda"],
                         r["codec"]["score_spatial_context"]) for r in controls}
            contexts = (repo / "contexts.txt").read_text().splitlines()
            observed = set()
            self.assertEqual(len(submissions), 14)
            self.assertEqual(len(records), 7)
            self.assertEqual(len({r[4] for r in records}), 7)
            self.assertEqual(len({r[6] for r in records}), 1)
            for i, record in enumerate(records):
                spatial = contexts[2 * i] == "SPATIAL_CONTEXT=true"
                observed.add((*record[:4], spatial))
                self.assertEqual(contexts[2 * i:2 * i + 2], [f"SPATIAL_CONTEXT={str(spatial).lower()}"] * 2)
                self.assertIn(f"--dependency=afterok:{record[4]}", submissions[2 * i + 1])
                self.assertIn("--time=6-00:00:00", submissions[2 * i])
                self.assertEqual("_context_off/" in record[7], not spatial)
                for line in submissions[2 * i:2 * i + 2]:
                    self.assertIn("SLICE=16", line)
                    self.assertIn("MAX_STEPS=50000 KERNEL=5 SCALE=", line)
                    self.assertIn("NN_BACKEND=legacy NN_DEVICE=cpu", line)
                    self.assertNotIn("wrong", line)
            self.assertEqual(observed, expected)
            self.assertIn("SCORE_STRINGS=L3:12,D4:16", result.stdout)
            rerun = helper.rd_run(repo, env, script=SCRIPT)
            self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
            self.assertEqual(log.read_text().splitlines(), submissions)
            self.assertEqual(rerun.stdout.count("REUSING TRAIN_JOB="), 7)
            self.assertFalse((repo / STATE / ".submit-lock").exists())
            changed_parent = repo / "other.ckpt"
            changed_parent.write_bytes(b"different parent")
            changed = helper.rd_run(repo, dict(env, SOURCE_CHECKPOINT=changed_parent.as_posix()), script=SCRIPT)
            self.assertNotEqual(changed.returncode, 0)
            self.assertEqual(log.read_text().splitlines(), submissions)

    def test_dry_run_and_qos_rejected_last_eval_preserve_train_id(self):
        with tempfile.TemporaryDirectory() as directory:
            helper, repo, env, log = self.fixture(directory)
            preview = helper.rd_run(repo, env, "--dry-run", script=SCRIPT)
            self.assertEqual(preview.returncode, 0, preview.stdout + preview.stderr)
            self.assertEqual(preview.stdout.count("PREVIEW ORDER="), 7)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "logs").exists())
            env["FAKE_FAIL_CALL"] = "14"
            failed = helper.rd_run(repo, env, script=SCRIPT)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("eval submission failed", failed.stderr)
            state = repo / STATE / "hilbert_dyadic4_off16_0p0128.state"
            self.assertEqual(state.read_text().split()[-2:], ["90013", "-"])
            self.assertFalse((repo / STATE / ".submit-lock").exists())
            env.pop("FAKE_FAIL_CALL")
            retry = helper.rd_run(repo, env, script=SCRIPT)
            self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
            lines = log.read_text().splitlines()
            self.assertEqual(len(lines), 15)  # 14 accepted and one rejected eval
            self.assertEqual(sum("--dependency=" not in line for line in lines), 7)
            self.assertIn("--dependency=afterok:90013", lines[-1])
            self.assertEqual(state.read_text().split()[-2:], ["90013", "90015"])
            self.assertEqual(len((repo / STATE / "submissions.tsv").read_text().splitlines()), 8)

    def test_invalid_last_state_missing_parent_and_lock_prevent_submissions(self):
        with tempfile.TemporaryDirectory() as directory:
            helper, repo, env, log = self.fixture(directory)
            state_dir = repo / STATE
            state_dir.mkdir(parents=True)
            last = state_dir / "hilbert_dyadic4_off16_0p0128.state"
            last.write_text("corrupt\n")
            invalid = helper.rd_run(repo, env, script=SCRIPT)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertFalse(log.exists())
            last.unlink()
            (state_dir / ".submit-lock").mkdir()
            locked = helper.rd_run(repo, env, script=SCRIPT)
            self.assertNotEqual(locked.returncode, 0)
            self.assertIn("Submission lock exists", locked.stderr)
            self.assertFalse(log.exists())
            (state_dir / ".submit-lock").rmdir()
            env.pop("SOURCE_CHECKPOINT")
            missing = helper.rd_run(repo, env, script=SCRIPT)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("lambda0p0256/residual_on/checkpoints/", missing.stderr)
            self.assertFalse(log.exists())

    def test_script_is_lf_and_valid_bash(self):
        path = ROOT / "scripts/slurm" / SCRIPT
        self.assertNotIn(b"\r", path.read_bytes())
        result = subprocess.run([existing.BASH, "-n", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

"""Check the requested 13-point RD extension and interrupted submission recovery."""
from pathlib import Path
import shutil
import tempfile
import unittest

import test_token_order_context_slurm as existing


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "submit_nfcgs_order_context_extremes.sh"
STATE = "logs/slurm/order_context_extremes_20261017"


@unittest.skipUnless(existing.BASH, "bash unavailable")
class ExtremesSlurmTests(unittest.TestCase):
    def fixture(self, directory):
        helper = existing.OrderContextSlurmTests()
        repo, env, log = helper.rd_fixture(directory)
        for name in ("score_context.py", "checkpoint.py"):
            destination = repo / "globalsplat/compression" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/"globalsplat/compression"/name, destination)
        # Exercise actual CLI preflight too, while only sbatch is mocked.
        shutil.copyfile(ROOT/"scripts/run_nfcgs.py", repo/"scripts/run_nfcgs.py")
        fake = repo/"fake_bin/sbatch"
        text = fake.read_text().replace('EVAL_DIR=%s\\n', 'EVAL_DIR=%s SLICE=%s\\n')
        text = text.replace('"${EVAL_RUN_DIR:-}" >>', '"${EVAL_RUN_DIR:-}" "${SCORE_SLICE_CHANNELS:-16}" >>')
        fake.write_text(text, newline="\n")
        return helper, repo, env, log

    def test_thirteen_exact_points_matching_dependencies_and_retry_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            helper, repo, env, log = self.fixture(directory)
            env.update(TOKEN_ORDER="nn_score", CONTEXT_SCHEDULE="single", STAGES="1", KERNEL="7",
                       SPATIAL_CONTEXT="false", SCORE_SLICE_CHANNELS="56", MAX_STEPS="10", CHECKPOINT_EVERY="1",
                       NN_ORDER_BACKEND="knn", NN_ORDER_DISTANCE_DEVICE="input", SCORE_ORDER_SCALE="missing.json",
                       RD_LAMBDAS="0.0256", RD_CONFIGS="wrong", EVAL_RUN_DIR="wrong-dir")
            result = helper.rd_run(repo, env, script=SCRIPT)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            submissions = log.read_text().splitlines()
            records = [r.split("\t") for r in (repo/STATE/"submissions.tsv").read_text().splitlines()[1:]]
            self.assertEqual(len(records), 13)
            self.assertEqual(len(submissions), 26)
            expected = {(order, schedule, str(stages), rate) for order, schedule, stages in (
                ("hilbert", "dyadic4", 4), ("morton", "dyadic4", 4), ("hilbert", "legacy", 3),
                ("morton", "legacy", 3), ("none", "dyadic4", 4), ("hilbert", "single", 1))
                for rate in ("0.0032", "0.0512")}
            expected.add(("hilbert", "dyadic4", "4", "0"))
            self.assertEqual({tuple(r[:4]) for r in records}, expected)
            self.assertEqual(tuple(records[0][:4]), ("hilbert", "dyadic4", "4", "0"))
            self.assertEqual(len({r[4] for r in records}), 13)
            self.assertEqual(len({r[6] for r in records}), 1)
            contexts = (repo/"contexts.txt").read_text().splitlines()
            for i, record in enumerate(records):
                context_off = record[1] == "single"
                for line in submissions[2*i:2*i+2]:
                    self.assertIn("SLICE=" + ("56" if context_off else "16"), line)
                    self.assertIn("MAX_STEPS=50000 KERNEL=5 SCALE=", line)
                    self.assertIn("NN_BACKEND=legacy NN_DEVICE=cpu", line)
                    self.assertNotIn("wrong-dir", line)
                self.assertIn(f"--dependency=afterok:{record[4]}", submissions[2*i+1])
                self.assertIn("--time=6-00:00:00", submissions[2*i])
                self.assertEqual(contexts[2*i:2*i+2], [f"SPATIAL_CONTEXT={str(not context_off).lower()}"]*2)
                self.assertEqual("_context_off_c56/" in record[7], context_off)
            self.assertIn("/lambda0/job_", records[0][7])
            rerun = helper.rd_run(repo, env, script=SCRIPT)
            self.assertEqual(rerun.returncode, 0, rerun.stdout+rerun.stderr)
            self.assertEqual(log.read_text().splitlines(), submissions)
            self.assertEqual(rerun.stdout.count("REUSING TRAIN_JOB="), 13)

    def test_dry_run_and_partial_qos_failure_preserve_accepted_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            helper, repo, env, log = self.fixture(directory)
            preview = helper.rd_run(repo, env, "--dry-run", script=SCRIPT)
            self.assertEqual(preview.returncode, 0, preview.stdout+preview.stderr)
            self.assertEqual(preview.stdout.count("PREVIEW ORDER="), 13)
            self.assertFalse(log.exists())
            self.assertFalse((repo/"logs").exists())
            env["FAKE_FAIL_CALL"] = "20"
            failed = helper.rd_run(repo, env, script=SCRIPT)
            self.assertNotEqual(failed.returncode, 0)
            states = list((repo/STATE).glob("*.state"))
            self.assertEqual(len(states), 10)
            self.assertEqual(sum(p.read_text().split()[-1] == "-" for p in states), 1)
            env.pop("FAKE_FAIL_CALL")
            rerun = helper.rd_run(repo, env, script=SCRIPT)
            self.assertEqual(rerun.returncode, 0, rerun.stdout+rerun.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 27)  # 26 accepted + one rejected eval
            self.assertEqual(sum("--dependency=" not in r for r in rows), 13)
            self.assertIn("--dependency=afterok:90019", rows[20])

    def test_last_point_state_parent_and_lock_preflight_before_first_sbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            helper, repo, env, log = self.fixture(directory)
            state = repo/STATE
            state.mkdir(parents=True)
            last = state/"hilbert_context_off_single_0p0512.state"
            last.write_text("corrupt\n")
            result = helper.rd_run(repo, env, script=SCRIPT)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(log.exists())
            last.unlink()
            (state/".submit-lock").mkdir()
            locked = helper.rd_run(repo, env, script=SCRIPT)
            self.assertNotEqual(locked.returncode, 0)
            self.assertIn("Submission lock exists", locked.stderr)
            self.assertFalse(log.exists())
            (state/".submit-lock").rmdir()
            env["SOURCE_CHECKPOINT"] = (repo/"missing.ckpt").as_posix()
            missing = helper.rd_run(repo, env, script=SCRIPT)
            self.assertNotEqual(missing.returncode, 0)
            self.assertFalse(log.exists())

    def test_new_rates_use_existing_fullsplit_parent_in_generic_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            helper, repo, env, log = self.fixture(directory)
            env.pop("SOURCE_CHECKPOINT")
            for rate in ("0", "0.0032", "0.0512"):
                result = helper.rd_run(repo, dict(env, RATE_LAMBDA=rate), script="submit_nfcgs_order_context.sh")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("lambda0p0256/residual_on/checkpoints/", result.stderr)
                self.assertNotIn(f"lambda{rate.replace('.', 'p')}/", result.stderr)
            self.assertFalse(log.exists())


if __name__ == "__main__":
    unittest.main()

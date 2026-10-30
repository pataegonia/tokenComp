"""Smoke-test submission/settings with a local fake sbatch, never a real job."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILES = ("nfcgs_order_context_settings.sh", "train_nfcgs_order_context.slurm",
         "eval_nfcgs_order_context.slurm", "submit_nfcgs_order_context.sh", "submit_nfcgs_order_context_rd.sh",
         "submit_nfcgs_hilbert_context_rd.sh", "submit_nfcgs_order_context_followup.sh",
         "submit_nfcgs_legacy2_rd.sh", "submit_nfcgs_nn_score_rd.sh", "submit_nfcgs_sort_context_ablation_rd.sh",
         "submit_nfcgs_sort_context_both_off_rd.sh", "submit_nfcgs_context_off_blocks_rd.sh",
         "submit_nfcgs_order_context_extremes.sh")
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")


@unittest.skipUnless(BASH, "bash unavailable")
class OrderContextSlurmTests(unittest.TestCase):
    def test_scripts_have_lf_shebang_and_valid_syntax(self):
        paths = [ROOT / "scripts/slurm" / name for name in (*FILES, "benchmark_nfcgs_nn_order.slurm")]
        for path in paths:
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"#!/usr/bin/env bash\n"))
            self.assertNotIn(b"\r", data)
            result = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        for name in FILES[1:3]:
            text = (ROOT / "scripts/slurm" / name).read_text()
            self.assertRegex(text, r"(?m)^#SBATCH --exclude=.*\bariel-v7\b")
            self.assertIn("--gres=gpu:normal:", text)

    def test_each_mode_exports_identical_settings_and_afterok_dependency(self):
        for order, schedule, stages in (("morton", "legacy", 3), ("hilbert", "legacy", 3),
                ("morton", "quarter2", 2), ("hilbert", "dyadic4", 4), ("nn_score", "legacy", 3)):
            with self.subTest(order=order, schedule=schedule), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                scripts = repo / "scripts/slurm"
                scripts.mkdir(parents=True)
                for name in FILES:
                    shutil.copyfile(ROOT / "scripts/slurm" / name, scripts / name)
                (repo / "scripts/run_nfcgs.py").write_text("# repository marker\n")
                fake_bin = repo / "fake_bin"
                fake_bin.mkdir()
                stub = fake_bin / "sbatch"
                stub.write_text('#!/usr/bin/env bash\n'
                    'printf "%s ORDER=%s SCHEDULE=%s STAGES=%s KERNEL=%s\\n" "$*" '
                    '"$TOKEN_ORDER" "$CONTEXT_SCHEDULE" "$STAGES" "$KERNEL" >> "$FAKE_SBATCH_LOG"\n'
                    'if [[ "$*" == *--dependency=* ]]; then echo "98766;fake"; else echo "98765;fake"; fi\n',
                    encoding="utf-8", newline="\n")
                stub.chmod(0o755)
                checkpoint = repo / "parent.ckpt"
                checkpoint.write_bytes(b"probe")
                log = repo / "submissions.txt"
                env = os.environ.copy()
                for key in ("STAGES", "KERNEL", "RATE_LAMBDA", "MAX_STEPS", "CHECKPOINT_EVERY", "SCORE_ORDER_SCALE"):
                    env.pop(key, None)
                env.update(REPO_DIR=repo.as_posix(), SOURCE_CHECKPOINT=checkpoint.as_posix(),
                    TOKEN_ORDER=order, CONTEXT_SCHEDULE=schedule, FAKE_SBATCH_LOG=log.as_posix())
                result = subprocess.run([BASH, "-c",
                    'export PATH="$(pwd)/fake_bin:$PATH"; bash scripts/slurm/submit_nfcgs_order_context.sh'],
                    cwd=repo, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = log.read_text().splitlines()
                self.assertEqual(len(rows), 2)
                self.assertIn("--dependency=afterok:98765", rows[1])
                for row in rows:
                    self.assertIn(f"ORDER={order} SCHEDULE={schedule} STAGES={stages} KERNEL=5", row)
                self.assertIn("job_98765", result.stdout)

    def test_relative_repository_root_is_canonicalized_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            scripts = repo / "scripts/slurm"
            scripts.mkdir(parents=True)
            for name in FILES:
                shutil.copyfile(ROOT / "scripts/slurm" / name, scripts / name)
            (repo / "scripts/run_nfcgs.py").write_text("# repository marker\n")
            (repo / "parent.ckpt").write_bytes(b"probe")
            stub = repo / "sbatch"
            stub.write_text('#!/usr/bin/env bash\n'
                '[[ "$REPO_DIR" == /* && "$SOURCE_CHECKPOINT" == /* ]] || exit 9\n'
                'echo 98765\n', encoding="utf-8", newline="\n")
            stub.chmod(0o755)
            env = os.environ.copy()
            for key in ("STAGES", "KERNEL", "RATE_LAMBDA", "MAX_STEPS", "CHECKPOINT_EVERY", "SCORE_ORDER_SCALE"):
                env.pop(key, None)
            env.update(REPO_DIR=".", SOURCE_CHECKPOINT="parent.ckpt", TOKEN_ORDER="hilbert", CONTEXT_SCHEDULE="legacy")
            result = subprocess.run([BASH, "-c",
                'export PATH="$(pwd):$PATH"; bash scripts/slurm/submit_nfcgs_order_context.sh'],
                cwd=repo, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("TRAIN_JOB=98765", result.stdout)

    def test_bad_root_or_missing_checkpoint_stops_before_sbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            scripts = repo / "scripts/slurm"
            scripts.mkdir(parents=True)
            for name in FILES:
                shutil.copyfile(ROOT / "scripts/slurm" / name, scripts / name)
            (repo / "scripts/run_nfcgs.py").write_text("# repository marker\n")
            stub = repo / "sbatch"
            stub.write_text('#!/usr/bin/env bash\ntouch submitted\necho 98765\n', encoding="utf-8", newline="\n")
            stub.chmod(0o755)
            for root in ("./missing-repo", "."):
                env = os.environ.copy()
                for key in ("STAGES", "KERNEL", "RATE_LAMBDA", "MAX_STEPS", "CHECKPOINT_EVERY", "SCORE_ORDER_SCALE"):
                    env.pop(key, None)
                env.update(REPO_DIR=root, SOURCE_CHECKPOINT="missing.ckpt", TOKEN_ORDER="hilbert", CONTEXT_SCHEDULE="legacy")
                result = subprocess.run([BASH, "-c",
                    'export PATH="$(pwd):$PATH"; bash scripts/slurm/submit_nfcgs_order_context.sh'],
                    cwd=repo, env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((repo / "submitted").exists())
                self.assertNotIn("TRAIN_JOB=", result.stdout)

    def test_rd_sweep_submits_six_matched_points_with_individual_eval_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            result = self.rd_run(repo, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 12)
            records = list((repo / "logs/slurm").glob("order_context_rd_*.tsv"))
            self.assertEqual(len(records), 1)
            fields = [row.split("\t") for row in records[0].read_text().splitlines()[1:]]
            self.assertEqual(len(fields), 6)
            points = {(row[0], row[1], row[2], row[3]) for row in fields}
            expected = {(order, schedule, stage, rate) for order, schedule, stage in (
                ("morton", "legacy", "3"), ("hilbert", "legacy", "3"), ("morton", "dyadic4", "4"))
                for rate in ("0.0064", "0.0128")}
            self.assertEqual(points, expected)
            self.assertEqual(len({row[6] for row in fields}), 1)  # identical warm start
            for index, record in enumerate(fields):
                self.assertIn(f"--dependency=afterok:{record[4]}", rows[index * 2 + 1])
                self.assertIn("--job-name=gs-oc-", rows[index * 2])
                for row in rows[index * 2:index * 2 + 2]:
                    self.assertIn(f"LAMBDA={record[3]}", row)
                    self.assertIn("MAX_STEPS=50000", row)
            self.assertIn("6 training jobs", result.stdout)

    def test_rd_preview_and_bad_last_point_submit_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            result = self.rd_run(repo, env, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PREVIEW ONLY", result.stdout)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "logs").exists())
            for rates in ("0.0064 invalid", "0.0064 0.0064"):
                with self.subTest(rates=rates):
                    result = self.rd_run(repo, dict(env, RD_LAMBDAS=rates))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(log.exists())

    def test_hilbert_anchor_sweep_submits_both_contexts_at_all_three_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            # Stale values from the previous sweep must not change this dedicated matrix.
            env.update(RD_CONFIGS="morton_legacy3", RD_LAMBDAS="0.0064")
            result = self.rd_run(repo, env, script="submit_nfcgs_hilbert_context_rd.sh")
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 12)
            record = next((repo / "logs/slurm").glob("order_context_rd_*.tsv"))
            fields = [row.split("\t") for row in record.read_text().splitlines()[1:]]
            self.assertEqual(len(fields), 6)
            self.assertEqual({(row[0], row[1], row[2], row[3]) for row in fields}, {
                ("hilbert", schedule, stage, rate) for schedule, stage in (("quarter2", "2"), ("dyadic4", "4"))
                for rate in ("0.0064", "0.0128", "0.0256")})
            self.assertEqual(len({row[6] for row in fields}), 1)
            for index, record in enumerate(fields):
                label = "hq2" if record[1] == "quarter2" else "hd4"
                self.assertIn(f"--job-name=gs-oc-{label}-{record[3].replace('.', 'p')}", rows[index * 2])
                self.assertIn(f"--dependency=afterok:{record[4]}", rows[index * 2 + 1])
                for row in rows[index * 2:index * 2 + 2]:
                    self.assertIn(f"ORDER=hilbert SCHEDULE={record[1]} STAGES={record[2]} LAMBDA={record[3]}", row)
                    self.assertIn("MAX_STEPS=50000", row)
                self.assertIn(f"/hilbert/{record[1]}_stages{record[2]}_k5_ste/lambda{record[3].replace('.', 'p')}/", record[7])

    def test_hilbert_anchor_sweep_preview_submits_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            result = self.rd_run(repo, env, "--dry-run", script="submit_nfcgs_hilbert_context_rd.sh")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("PREVIEW ORDER=hilbert"), 6)
            self.assertIn("SCHEDULE=quarter2 STAGES=2 LAMBDA=0.0256", result.stdout)
            self.assertIn("SCHEDULE=dyadic4 STAGES=4 LAMBDA=0.0256", result.stdout)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "logs").exists())

    def test_missing_middle_rate_parent_reports_highrate_warm_start(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            env.pop("SOURCE_CHECKPOINT")
            relative = ("outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/"
                "lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_"
                "lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt")
            env.update(RATE_LAMBDA="0.0128", DRY_RUN="1")
            result = subprocess.run([BASH, "scripts/slurm/submit_nfcgs_order_context.sh"],
                cwd=repo, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(relative, result.stderr)
            self.assertNotIn("lambda0p0128/", result.stderr)
            self.assertFalse(log.exists())

    def test_followup_submits_four_points_without_xyz_and_reuses_submissions(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            # The dedicated recipe must override unrelated settings left in a shell.
            env.update(TOKEN_ORDER="hilbert", CONTEXT_SCHEDULE="dyadic4", STAGES="4",
                KERNEL="7", MAX_STEPS="20", CHECKPOINT_EVERY="10", SCORE_ORDER_SCALE="missing.json",
                RD_CONFIGS="hilbert_dyadic4", RD_LAMBDAS="0.0064")
            result = self.rd_run(repo, env, script="submit_nfcgs_order_context_followup.sh")
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 8)
            state_dir = repo / "logs/slurm/order_context_followup_20261011"
            fields = [row.split("\t") for row in (state_dir / "submissions.tsv").read_text().splitlines()[1:]]
            self.assertEqual(len(fields), 4)
            expected = {("morton", "quarter2", "2", rate) for rate in ("0.0064", "0.0128", "0.0256")}
            expected.add(("nn_score", "legacy", "3", "0.0256"))
            self.assertEqual({tuple(row[:4]) for row in fields}, expected)
            self.assertEqual(len({row[6] for row in fields}), 1)
            for index, record in enumerate(fields):
                label = {"morton": "mq2", "nn_score": "ns3"}[record[0]]
                self.assertIn(f"--job-name=gs-oc-{label}-{record[3].replace('.', 'p')}", rows[index * 2])
                self.assertIn(f"--dependency=afterok:{record[4]}", rows[index * 2 + 1])
                for row in rows[index * 2:index * 2 + 2]:
                    self.assertIn("MAX_STEPS=50000 KERNEL=5 SCALE=", row)
            self.assertEqual(len(list(state_dir.glob("*.state"))), 4)
            self.assertFalse((state_dir / ".submit-lock").exists())
            rerun = self.rd_run(repo, env, script="submit_nfcgs_order_context_followup.sh")
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertEqual(log.read_text().splitlines(), rows)  # no second sbatch
            self.assertEqual(rerun.stdout.count("REUSING TRAIN_JOB="), 4)
            self.assertEqual(rerun.stdout.count("REUSING EVAL_JOB="), 4)
            self.assertEqual(len((state_dir / "submissions.tsv").read_text().splitlines()), 5)

    def test_followup_retries_only_eval_after_partial_submission_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            env["FAKE_FAIL_CALL"] = "2"  # first train succeeds, its eval is rejected
            result = self.rd_run(repo, env, script="submit_nfcgs_order_context_followup.sh")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("eval submission failed", result.stderr)
            state_dir = repo / "logs/slurm/order_context_followup_20261011"
            states = list(state_dir.glob("*.state"))
            self.assertEqual(len(states), 1)
            self.assertEqual(states[0].read_text().split()[-2:], ["90001", "-"])
            self.assertFalse((state_dir / ".submit-lock").exists())
            env.pop("FAKE_FAIL_CALL")
            rerun = self.rd_run(repo, env, script="submit_nfcgs_order_context_followup.sh")
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 9)  # eight accepted submissions + one rejected eval
            self.assertIn("--dependency=afterok:90001", rows[2])
            self.assertIn("REUSING TRAIN_JOB=90001", rerun.stdout)
            fields = [row.split("\t") for row in (state_dir / "submissions.tsv").read_text().splitlines()[1:]]
            self.assertEqual(fields[0][4:6], ["90001", "90003"])
            self.assertEqual(len(fields), 4)
            self.assertEqual(len({row[4] for row in fields}), 4)

    def test_followup_preview_submits_nothing_and_writes_no_state(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            result = self.rd_run(repo, env, "--dry-run", script="submit_nfcgs_order_context_followup.sh")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("PREVIEW ORDER="), 4)
            self.assertIn("ORDER=nn_score SCHEDULE=legacy STAGES=3 LAMBDA=0.0256", result.stdout)
            self.assertNotIn("ORDER=nn_xyz", result.stdout)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "logs").exists())

    def test_followup_changed_parent_or_corrupt_state_submits_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            script = "submit_nfcgs_order_context_followup.sh"
            result = self.rd_run(repo, env, script=script)
            self.assertEqual(result.returncode, 0, result.stderr)
            original_log = log.read_text()
            new_parent = repo / "other_parent.ckpt"
            new_parent.write_bytes(b"other parent")
            result = self.rd_run(repo, dict(env, SOURCE_CHECKPOINT=new_parent.as_posix()), script=script)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatched submission state", result.stderr)
            self.assertEqual(log.read_text(), original_log)
            state = repo / "logs/slurm/order_context_followup_20261011/morton_quarter2_0p0064.state"
            state.write_text("corrupt\n")
            result = self.rd_run(repo, env, script=script)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(log.read_text(), original_log)

    def test_followup_existing_lock_stops_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            lock = repo / "logs/slurm/order_context_followup_20261011/.submit-lock"
            lock.mkdir(parents=True)
            result = self.rd_run(repo, env, script="submit_nfcgs_order_context_followup.sh")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Another submission is active", result.stderr)
            self.assertFalse(log.exists())
            self.assertTrue(lock.exists())

    def test_nn_backend_exports_to_both_jobs_and_changes_output_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            env.update(TOKEN_ORDER="nn_score", CONTEXT_SCHEDULE="legacy", STAGES="3",
                NN_ORDER_BACKEND="knn", NN_ORDER_DISTANCE_DEVICE="input", NN_ORDER_CANDIDATES="65",
                SUBMISSION_STATE_FILE="nn.state")
            script = "submit_nfcgs_order_context.sh"
            result = self.rd_run(repo, env, script=script)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertIn("NN_BACKEND=knn NN_DEVICE=input NN_CANDIDATES=65", row)
            self.assertIn("--dependency=afterok:90001", rows[1])
            self.assertIn("legacy_stages3_k5_ste_nn_knn_input_k65", result.stdout)
            rerun = self.rd_run(repo, env, script=script)
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertEqual(log.read_text().splitlines(), rows)
            changed = self.rd_run(repo, dict(env, NN_ORDER_CANDIDATES="64"), script=script)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("mismatched submission state", changed.stderr)
            self.assertEqual(log.read_text().splitlines(), rows)

    def test_nn_settings_preview_and_invalid_backend_submit_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            env.update(TOKEN_ORDER="nn_xyz", CONTEXT_SCHEDULE="legacy", STAGES="3",
                       NN_ORDER_BACKEND="row", NN_ORDER_DISTANCE_DEVICE="input", DRY_RUN="1")
            result = self.rd_run(repo, env, script="submit_nfcgs_order_context.sh")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("NN_BACKEND=row NN_DISTANCE_DEVICE=input NN_CANDIDATES=64", result.stdout)
            self.assertFalse(log.exists())
            for change in (dict(NN_ORDER_CANDIDATES="0"), dict(NN_ORDER_BACKEND="legacy"),
                           dict(TOKEN_ORDER="hilbert"), dict(NN_ORDER_DISTANCE_DEVICE="cuda")):
                invalid = self.rd_run(repo, dict(env, **change), script="submit_nfcgs_order_context.sh")
                self.assertNotEqual(invalid.returncode, 0)
                self.assertFalse(log.exists())

    def test_legacy2_pair_pins_rates_and_default_parent_and_reuses_job_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            custom_parent = env.pop("SOURCE_CHECKPOINT")
            relative = ("outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/"
                "lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_"
                "lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt")
            script = "submit_nfcgs_legacy2_rd.sh"
            # Default-parent lookup must use .0256 even for the first .0064 point.
            # Verify the lookup without creating a >260-character Windows path.
            missing = self.rd_run(repo, env, "--dry-run", script=script)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn(relative, missing.stderr)
            self.assertNotIn("lambda0p0064/", missing.stderr)
            self.assertFalse(log.exists())
            env["SOURCE_CHECKPOINT"] = custom_parent
            env.update(TOKEN_ORDER="hilbert", CONTEXT_SCHEDULE="dyadic4", STAGES="4",
                RATE_LAMBDA="0.0256", KERNEL="7", MAX_STEPS="20", CHECKPOINT_EVERY="10",
                SCORE_ORDER_SCALE="missing.json", FOLLOWUP_STATE_DIR=str(repo / "wrong-state"))
            result = self.rd_run(repo, env, script=script)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 4)
            state_dir = repo / "logs/slurm/order_context_legacy2_rd_20261012"
            records = [row.split("\t") for row in (state_dir / "submissions.tsv").read_text().splitlines()[1:]]
            self.assertEqual({tuple(row[:4]) for row in records}, {
                ("morton", "legacy", "2", "0.0064"), ("morton", "legacy", "2", "0.0128")})
            self.assertEqual(len(list(state_dir.glob("*.state"))), 2)
            self.assertEqual(len({record[6] for record in records}), 1)
            for index, record in enumerate(records):
                self.assertTrue(record[6].endswith("/parent.ckpt"))
                self.assertIn(f"--job-name=gs-oc-m2-{record[3].replace('.', 'p')}", rows[index * 2])
                self.assertIn(f"--dependency=afterok:{record[4]}", rows[index * 2 + 1])
                for row in rows[index * 2:index * 2 + 2]:
                    self.assertIn(f"ORDER=morton SCHEDULE=legacy STAGES=2 LAMBDA={record[3]}", row)
                    self.assertIn("MAX_STEPS=50000 KERNEL=5 SCALE=", row)
                self.assertIn(f"/morton/legacy_stages2_k5_ste/lambda{record[3].replace('.', 'p')}/", record[7])
            self.assertFalse((repo / "wrong-state").exists())
            self.assertFalse((state_dir / ".submit-lock").exists())
            rerun = self.rd_run(repo, env, script=script)
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertEqual(log.read_text().splitlines(), rows)
            self.assertEqual(rerun.stdout.count("REUSING TRAIN_JOB="), 2)
            self.assertEqual(rerun.stdout.count("REUSING EVAL_JOB="), 2)

    def test_legacy2_preview_and_partial_eval_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            script = "submit_nfcgs_legacy2_rd.sh"
            preview = self.rd_run(repo, env, "--dry-run", script=script)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertEqual(preview.stdout.count("PREVIEW ORDER=morton SCHEDULE=legacy STAGES=2"), 2)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "logs").exists())
            env["FAKE_FAIL_CALL"] = "2"
            result = self.rd_run(repo, env, script=script)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("eval submission failed", result.stderr)
            state_dir = repo / "logs/slurm/order_context_legacy2_rd_20261012"
            self.assertEqual((state_dir / "morton_legacy2_0p0064.state").read_text().split()[-2:], ["90001", "-"])
            self.assertFalse((state_dir / ".submit-lock").exists())
            env.pop("FAKE_FAIL_CALL")
            rerun = self.rd_run(repo, env, script=script)
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 5)  # four accepted jobs and the rejected eval
            self.assertIn("--dependency=afterok:90001", rows[2])
            records = [row.split("\t") for row in (state_dir / "submissions.tsv").read_text().splitlines()[1:]]
            self.assertEqual(records[0][4:6], ["90001", "90003"])
            self.assertEqual(len({row[4] for row in records}), 2)

    def test_nn_score_rd_adds_two_gpu_rates_from_one_parent_and_reuses_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            env.update(TOKEN_ORDER="nn_xyz", CONTEXT_SCHEDULE="dyadic4", STAGES="4", KERNEL="7",
                RATE_LAMBDA="0.0256", MAX_STEPS="20", SCORE_ORDER_SCALE="missing.json",
                NN_ORDER_BACKEND="legacy", NN_ORDER_DISTANCE_DEVICE="cpu", NN_ORDER_CANDIDATES="1",
                TRAIN_TIME_LIMIT="00:10:00", EVAL_RUN_DIR="wrong-old-resume-dir")
            script = "submit_nfcgs_nn_score_rd.sh"
            result = self.rd_run(repo, env, script=script)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 4)
            state_dir = repo / "logs/slurm/nn_score_rd_gpu_20261013"
            records = [row.split("\t") for row in (state_dir / "submissions.tsv").read_text().splitlines()[1:]]
            self.assertEqual({tuple(row[:4]) for row in records}, {
                ("nn_score", "legacy", "3", "0.0064"), ("nn_score", "legacy", "3", "0.0128")})
            self.assertEqual(len({row[6] for row in records}), 1)
            for index, record in enumerate(records):
                self.assertIn(f"--job-name=gs-ns-{record[3].replace('.', 'p')}", rows[index * 2])
                self.assertIn("--time=6-00:00:00", rows[index * 2])
                self.assertIn(f"--dependency=afterok:{record[4]}", rows[index * 2 + 1])
                for row in rows[index * 2:index * 2 + 2]:
                    self.assertIn("NN_BACKEND=knn NN_DEVICE=input NN_CANDIDATES=64", row)
                    self.assertIn("MAX_STEPS=50000 KERNEL=5 SCALE=", row)
                    self.assertIn("EVAL_DIR=", row)
                    self.assertNotIn("wrong-old-resume-dir", row)
                self.assertIn("legacy_stages3_k5_ste_nn_knn_input_k64", record[7])
            retry = self.rd_run(repo, env, script=script)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(log.read_text().splitlines(), rows)
            self.assertEqual(retry.stdout.count("REUSING TRAIN_JOB="), 2)

    def test_nn_score_rd_preview_and_partial_eval_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            script = "submit_nfcgs_nn_score_rd.sh"
            preview = self.rd_run(repo, env, "--dry-run", script=script)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertEqual(preview.stdout.count("PREVIEW ORDER=nn_score"), 2)
            self.assertIn("NN_DISTANCE_DEVICE=input", preview.stdout)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "logs").exists())
            env["FAKE_FAIL_CALL"] = "2"
            first = self.rd_run(repo, env, script=script)
            self.assertNotEqual(first.returncode, 0)
            self.assertIn("eval submission failed", first.stderr)
            env.pop("FAKE_FAIL_CALL")
            retry = self.rd_run(repo, env, script=script)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 5)
            self.assertIn("--dependency=afterok:90001", rows[2])
            self.assertIn("REUSING TRAIN_JOB=90001", retry.stdout)

    def test_sort_context_ablation_submits_twelve_matched_points_and_reuses_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            env.update(TOKEN_ORDER="nn_score", SPATIAL_CONTEXT="false", CONTEXT_SCHEDULE="dyadic4",
                NN_ORDER_BACKEND="knn", NN_ORDER_DISTANCE_DEVICE="input", STAGES="4", KERNEL="7",
                MAX_STEPS="20", SCORE_ORDER_SCALE="missing.json", EVAL_RUN_DIR="wrong-resume-dir")
            script = "submit_nfcgs_sort_context_ablation_rd.sh"
            result = self.rd_run(repo, env, script=script)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 24)
            state_dir = repo / "logs/slurm/sort_context_ablation_rd_20261013"
            records = [row.split("\t") for row in (state_dir / "submissions.tsv").read_text().splitlines()[1:]]
            self.assertEqual({tuple(row[:4]) for row in records}, {
                (order, schedule, stages, rate)
                for order, schedule, stages in (("none", "legacy", "3"), ("hilbert", "legacy", "3"),
                                               ("none", "quarter2", "2"), ("none", "dyadic4", "4"))
                for rate in ("0.0064", "0.0128", "0.0256")})
            self.assertEqual(len({row[6] for row in records}), 1)
            self.assertEqual(len({row[4] for row in records}), 12)
            contexts = (repo / "contexts.txt").read_text().splitlines()
            for index, record in enumerate(records):
                tag = "cq2" if record[1] == "quarter2" else "cd4" if record[1] == "dyadic4" else "co" if record[0] == "none" else "ho"
                self.assertIn(f"--job-name=gs-{tag}-{record[3].replace('.', 'p')}", rows[index * 2])
                self.assertIn("--time=6-00:00:00", rows[index * 2])
                self.assertIn(f"--dependency=afterok:{record[4]}", rows[index * 2 + 1])
                expected = "SPATIAL_CONTEXT=" + ("true" if record[0] == "none" else "false")
                self.assertEqual(contexts[index * 2:index * 2 + 2], [expected, expected])
                for row in rows[index * 2:index * 2 + 2]:
                    self.assertIn("NN_BACKEND=legacy NN_DEVICE=cpu NN_CANDIDATES=64", row)
                    self.assertIn("MAX_STEPS=50000 KERNEL=5 SCALE=", row)
                    self.assertNotIn("wrong-resume-dir", row)
                self.assertEqual("context_off" in record[7], record[0] == "hilbert")
            retry = self.rd_run(repo, env, script=script)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(log.read_text().splitlines(), rows)
            self.assertEqual(retry.stdout.count("REUSING TRAIN_JOB="), 12)

    def test_sort_context_ablation_preview_and_partial_eval_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            script = "submit_nfcgs_sort_context_ablation_rd.sh"
            preview = self.rd_run(repo, env, "--dry-run", script=script)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertEqual(preview.stdout.count("PREVIEW ORDER="), 12)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "logs").exists())
            env["FAKE_FAIL_CALL"] = "2"
            failed = self.rd_run(repo, env, script=script)
            self.assertNotEqual(failed.returncode, 0)
            env.pop("FAKE_FAIL_CALL")
            retry = self.rd_run(repo, env, script=script)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 25)
            self.assertEqual(sum("--dependency=" not in row for row in rows), 12)
            self.assertIn("--dependency=afterok:90001", rows[2])

    def test_context_off_changes_signature_and_train_eval_cli_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            env.update(TOKEN_ORDER="hilbert", SUBMISSION_STATE_FILE="same.state", SPATIAL_CONTEXT="true")
            result = self.rd_run(repo, env, script="submit_nfcgs_order_context.sh")
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = log.read_text()
            env["SPATIAL_CONTEXT"] = "false"
            changed = self.rd_run(repo, env, script="submit_nfcgs_order_context.sh")
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("mismatched submission state", changed.stderr)
            self.assertEqual(log.read_text(), rows)
            cli = subprocess.run([BASH, "-c", 'source scripts/slurm/nfcgs_order_context_settings.sh; '
                                  'printf "%s\\n" "${CODEC_ARGS[@]}"'], cwd=repo, env=env,
                                 capture_output=True, text=True)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertIn("--no-spatial-context", cli.stdout)

    def test_expanded_ablation_reuses_the_original_six_training_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            script = "submit_nfcgs_sort_context_ablation_rd.sh"
            path = repo / "scripts/slurm" / script
            expanded = path.read_text(encoding="utf-8")
            path.write_text(expanded.replace(
                "ARMS=(context_only hilbert_only context_quarter2 context_dyadic4)",
                "ARMS=(context_only hilbert_only)"), encoding="utf-8", newline="\n")
            first = self.rd_run(repo, env, script=script)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = log.read_text().splitlines()
            self.assertEqual(len(original), 12)
            state_dir = repo / "logs/slurm/sort_context_ablation_rd_20261013"
            states = {p.name: p.read_bytes() for p in state_dir.glob("*.state")}
            path.write_text(expanded, encoding="utf-8", newline="\n")
            extended = self.rd_run(repo, env, script=script)
            self.assertEqual(extended.returncode, 0, extended.stderr)
            self.assertEqual(extended.stdout.count("REUSING TRAIN_JOB="), 6)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 24)
            self.assertEqual(rows[:12], original)
            for name, data in states.items():
                self.assertEqual((state_dir / name).read_bytes(), data)
            self.assertTrue(all("SCHEDULE=quarter2" in row for row in rows[12:18]))
            self.assertTrue(all("SCHEDULE=dyadic4" in row for row in rows[18:24]))

    def test_expanded_ablation_bad_final_state_submits_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            state_dir = repo / "logs/slurm/sort_context_ablation_rd_20261013"
            state_dir.mkdir(parents=True)
            (state_dir / "context_dyadic4_0p0256.state").write_text("invalid\n")
            result = self.rd_run(repo, env, script="submit_nfcgs_sort_context_ablation_rd.sh")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatched submission state", result.stderr)
            self.assertFalse(log.exists())

    def test_both_off_submits_only_three_matched_points_and_reuses_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            env.update(TOKEN_ORDER="nn_score", SPATIAL_CONTEXT="true", CONTEXT_SCHEDULE="dyadic4",
                STAGES="4", KERNEL="7", MAX_STEPS="20", CHECKPOINT_EVERY="10",
                NN_ORDER_BACKEND="knn", NN_ORDER_DISTANCE_DEVICE="input", NN_ORDER_CANDIDATES="8",
                SCORE_ORDER_SCALE="missing.json", TRAIN_TIME_LIMIT="00:01:00", EVAL_RUN_DIR="wrong-dir",
                SORT_CONTEXT_ABLATION_STATE_DIR="wrong-state")
            script = "submit_nfcgs_sort_context_both_off_rd.sh"
            result = self.rd_run(repo, env, script=script)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 6)
            state_dir = repo / "logs/slurm/sort_context_both_off_rd_20261016"
            records = [row.split("\t") for row in (state_dir / "submissions.tsv").read_text().splitlines()[1:]]
            self.assertEqual({tuple(row[:4]) for row in records}, {
                ("none", "legacy", "3", rate) for rate in ("0.0064", "0.0128", "0.0256")})
            self.assertEqual(len({row[6] for row in records}), 1)
            self.assertEqual((repo / "contexts.txt").read_text().splitlines(), ["SPATIAL_CONTEXT=false"] * 6)
            self.assertEqual(len(list(state_dir.glob("both_off_*.state"))), 3)
            self.assertFalse((repo / "wrong-state").exists())
            self.assertFalse((state_dir / ".submit-lock").exists())
            for index, record in enumerate(records):
                self.assertIn(f"--job-name=gs-bo-{record[3].replace('.', 'p')}", rows[index * 2])
                self.assertIn("--time=6-00:00:00", rows[index * 2])
                self.assertIn(f"--dependency=afterok:{record[4]}", rows[index * 2 + 1])
                self.assertIn(f"/none/legacy_stages3_k5_ste_context_off/lambda{record[3].replace('.', 'p')}/", record[7])
                for row in rows[index * 2:index * 2 + 2]:
                    self.assertIn("MAX_STEPS=50000 KERNEL=5 SCALE= NN_BACKEND=legacy NN_DEVICE=cpu NN_CANDIDATES=64 EVAL_DIR=", row)
                    self.assertNotIn("wrong-dir", row)
            retry = self.rd_run(repo, env, script=script)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(log.read_text().splitlines(), rows)
            self.assertEqual(retry.stdout.count("REUSING TRAIN_JOB="), 3)
            self.assertEqual(retry.stdout.count("REUSING EVAL_JOB="), 3)
            self.assertEqual(len((state_dir / "submissions.tsv").read_text().splitlines()), 4)

    def test_both_off_preview_and_partial_eval_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            script = "submit_nfcgs_sort_context_both_off_rd.sh"
            preview = self.rd_run(repo, env, "--dry-run", script=script)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertEqual(preview.stdout.count("PREVIEW ORDER=none SCHEDULE=legacy STAGES=3"), 3)
            self.assertEqual(preview.stdout.count("PREVIEW SPATIAL_CONTEXT=false SCORE_PATH=minimal"), 3)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "logs").exists())
            env["FAKE_FAIL_CALL"] = "2"
            failed = self.rd_run(repo, env, script=script)
            self.assertNotEqual(failed.returncode, 0)
            state_dir = repo / "logs/slurm/sort_context_both_off_rd_20261016"
            self.assertEqual((state_dir / "both_off_0p0064.state").read_text().split()[-2:], ["90001", "-"])
            self.assertFalse((state_dir / ".submit-lock").exists())
            env.pop("FAKE_FAIL_CALL")
            retry = self.rd_run(repo, env, script=script)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            rows = log.read_text().splitlines()
            self.assertEqual(len(rows), 7)  # six accepted, one rejected eval
            self.assertEqual(sum("--dependency=" not in row for row in rows), 3)
            self.assertIn("--dependency=afterok:90001", rows[2])

    def test_both_off_default_parent_is_highrate_even_for_first_lowrate_point(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.rd_fixture(directory)
            env.pop("SOURCE_CHECKPOINT")
            relative = ("outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/"
                "lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_"
                "lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt")
            # Verify default lookup without creating a >260-character Windows path.
            result = self.rd_run(repo, env, "--dry-run", script="submit_nfcgs_sort_context_both_off_rd.sh")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(relative, result.stderr)
            self.assertNotIn("lambda0p0064/", result.stderr)
            self.assertFalse(log.exists())

    def test_both_off_invalid_last_state_or_existing_lock_submits_nothing(self):
        for variant in ("bad_state", "lock", "old_launcher"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.rd_fixture(directory)
                state_dir = repo / "logs/slurm/sort_context_both_off_rd_20261016"
                if variant == "bad_state":
                    state_dir.mkdir(parents=True)
                    (state_dir / "both_off_0p0256.state").write_text("invalid\n")
                elif variant == "lock":
                    (state_dir / ".submit-lock").mkdir(parents=True)
                else:
                    path = repo / "scripts/slurm/train_nfcgs_order_context.slurm"
                    path.write_text("#!/usr/bin/env bash\n# old launcher\n", newline="\n")
                result = self.rd_run(repo, env, script="submit_nfcgs_sort_context_both_off_rd.sh")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(log.exists())

    def test_both_off_training_stages_checkpoint_on_ceph_before_python(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, _ = self.rd_fixture(directory)
            profile = repo / "fake_conda/etc/profile.d"
            profile.mkdir(parents=True)
            (profile / "conda.sh").write_text("conda() { return 0; }\n", newline="\n")
            fake = repo / "fake_bin"
            (fake / "conda").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n', newline="\n")
            (fake / "python").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_TRAIN_LOG"\n'
                'printf "%s\\n" "$TMPDIR" >> "$FAKE_TRAIN_TMP_LOG"\n', newline="\n")
            for name in ("conda", "python"):
                (fake / name).chmod(0o755)
            log, tmp_log = repo / "train.txt", repo / "tmp.txt"
            env.update(TOKEN_ORDER="none", SPATIAL_CONTEXT="false", CONTEXT_SCHEDULE="legacy", STAGES="3",
                RATE_LAMBDA="0.0128", SLURM_JOB_ID="12345", NFCGS_TMP_ROOT="checkpoint_tmp",
                FAKE_CONDA_BASE=(repo / "fake_conda").as_posix(), FAKE_TRAIN_LOG=log.as_posix(),
                FAKE_TRAIN_TMP_LOG=tmp_log.as_posix())
            result = self.rd_run(repo, env, script="train_nfcgs_order_context.slurm")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((repo / "checkpoint_tmp/job_12345").is_dir())
            self.assertEqual(tmp_log.read_text().splitlines(), ["checkpoint_tmp/job_12345"])
            self.assertIn("CHECKPOINT_TMPDIR=checkpoint_tmp/job_12345", result.stdout)
            call = log.read_text()
            for flag in ("train --devices 2 --launcher srun", "--token-order none", "--no-spatial-context",
                         "--score-path minimal", "--score-context-schedule legacy", "--score-spatial-stages 3",
                         "--score-spatial-kernel 5", "--score-context-quantization ste", "--allow-score-path-conversion",
                         "--scope all --lr 0.0001 --batch-size 2 --accumulate 2", "--max-steps 50000"):
                self.assertIn(flag, call)
            self.assertNotIn("--resume", call)

    def rd_fixture(self, directory):
        repo = Path(directory)
        scripts = repo / "scripts/slurm"
        scripts.mkdir(parents=True)
        for name in FILES:
            shutil.copyfile(ROOT / "scripts/slurm" / name, scripts / name)
        (repo / "scripts/run_nfcgs.py").write_text("# fake runner accepts dry-run preflight\n")
        checkpoint = repo / "parent.ckpt"
        checkpoint.write_bytes(b"parent")
        fake_bin = repo / "fake_bin"
        fake_bin.mkdir()
        (fake_bin / "sbatch").write_text('#!/usr/bin/env bash\n'
            'count=$(cat "$FAKE_COUNTER"); count=$((count+1)); echo "$count" > "$FAKE_COUNTER"\n'
            'printf "%s ORDER=%s SCHEDULE=%s STAGES=%s LAMBDA=%s MAX_STEPS=%s KERNEL=%s SCALE=%s NN_BACKEND=%s NN_DEVICE=%s NN_CANDIDATES=%s EVAL_DIR=%s\\n" "$*" '
            '"$TOKEN_ORDER" "$CONTEXT_SCHEDULE" "$STAGES" "$RATE_LAMBDA" "$MAX_STEPS" "$KERNEL" "$SCORE_ORDER_SCALE" '
            '"$NN_ORDER_BACKEND" "$NN_ORDER_DISTANCE_DEVICE" "$NN_ORDER_CANDIDATES" "${EVAL_RUN_DIR:-}" >> "$FAKE_SBATCH_LOG"\n'
            'printf "SPATIAL_CONTEXT=%s\\n" "${SPATIAL_CONTEXT:-true}" >> "$FAKE_CONTEXT_LOG"\n'
            'if [[ "${FAKE_FAIL_CALL:-}" == "$count" ]]; then echo "mock rejection" >&2; exit 1; fi\n'
            'echo "$((90000+count));fake"\n', encoding="utf-8", newline="\n")
        (fake_bin / "sbatch").chmod(0o755)
        counter = repo / "counter.txt"
        counter.write_text("0\n")
        log = repo / "submissions.txt"
        env = os.environ.copy()
        for key in ("RD_CONFIGS", "RD_LAMBDAS", "RD_SUBMISSION_RECORD", "SUBMISSION_RECORD", "STAGES",
                "KERNEL", "RATE_LAMBDA", "MAX_STEPS", "CHECKPOINT_EVERY", "SCORE_ORDER_SCALE", "DRY_RUN",
                "SUBMISSION_STATE_FILE", "FOLLOWUP_STATE_DIR", "LEGACY2_STATE_DIR", "FAKE_FAIL_CALL",
                "NN_ORDER_BACKEND", "NN_ORDER_DISTANCE_DEVICE", "NN_ORDER_CANDIDATES",
                "NN_SCORE_RD_STATE_DIR", "TRAIN_TIME_LIMIT", "EVAL_RUN_DIR", "SPATIAL_CONTEXT",
                "SORT_CONTEXT_ABLATION_STATE_DIR", "SORT_CONTEXT_BOTH_OFF_STATE_DIR", "SCORE_SLICE_CHANNELS",
                "CONTEXT_OFF_ORDERS", "CONTEXT_OFF_BLOCKS_STATE_DIR", "EXTREMES_STATE_DIR"):
            env.pop(key, None)
        env.update(REPO_DIR=repo.as_posix(), SOURCE_CHECKPOINT=checkpoint.as_posix(),
            FAKE_COUNTER=counter.as_posix(), FAKE_SBATCH_LOG=log.as_posix(),
            FAKE_CONTEXT_LOG=(repo / "contexts.txt").as_posix())
        return repo, env, log

    def rd_run(self, repo, env, option="", script="submit_nfcgs_order_context_rd.sh"):
        return subprocess.run([BASH, "-c",
            f'export PATH="$(pwd)/fake_bin:$PATH"; bash scripts/slurm/{script} ' + option],
            cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

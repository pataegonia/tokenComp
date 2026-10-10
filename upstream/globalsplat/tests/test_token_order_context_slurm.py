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
         "submit_nfcgs_hilbert_context_rd.sh")
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")


@unittest.skipUnless(BASH, "bash unavailable")
class OrderContextSlurmTests(unittest.TestCase):
    def test_scripts_have_lf_shebang_and_valid_syntax(self):
        paths = [ROOT / "scripts/slurm" / name for name in FILES]
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
            'printf "%s ORDER=%s SCHEDULE=%s STAGES=%s LAMBDA=%s MAX_STEPS=%s\\n" "$*" '
            '"$TOKEN_ORDER" "$CONTEXT_SCHEDULE" "$STAGES" "$RATE_LAMBDA" "$MAX_STEPS" >> "$FAKE_SBATCH_LOG"\n'
            'echo "$((90000+count));fake"\n', encoding="utf-8", newline="\n")
        (fake_bin / "sbatch").chmod(0o755)
        counter = repo / "counter.txt"
        counter.write_text("0\n")
        log = repo / "submissions.txt"
        env = os.environ.copy()
        for key in ("RD_CONFIGS", "RD_LAMBDAS", "RD_SUBMISSION_RECORD", "SUBMISSION_RECORD", "STAGES",
                "KERNEL", "RATE_LAMBDA", "MAX_STEPS", "CHECKPOINT_EVERY", "SCORE_ORDER_SCALE", "DRY_RUN"):
            env.pop(key, None)
        env.update(REPO_DIR=repo.as_posix(), SOURCE_CHECKPOINT=checkpoint.as_posix(),
            FAKE_COUNTER=counter.as_posix(), FAKE_SBATCH_LOG=log.as_posix())
        return repo, env, log

    def rd_run(self, repo, env, option="", script="submit_nfcgs_order_context_rd.sh"):
        return subprocess.run([BASH, "-c",
            f'export PATH="$(pwd)/fake_bin:$PATH"; bash scripts/slurm/{script} ' + option],
            cwd=repo, env=env, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()

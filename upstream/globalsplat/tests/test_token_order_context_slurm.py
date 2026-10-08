"""Smoke-test submission/settings with a local fake sbatch, never a real job."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILES = ("nfcgs_order_context_settings.sh", "train_nfcgs_order_context.slurm",
         "eval_nfcgs_order_context.slurm", "submit_nfcgs_order_context.sh")
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
        result = subprocess.run([BASH, "-n", *map(str, paths)], capture_output=True, text=True)
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


if __name__ == "__main__":
    unittest.main()

"""The cleanup inventory is read-only and distinguishes unknown live inputs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("checkpoint_audit", ROOT / "scripts/audit_nfcgs_checkpoints.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CheckpointAuditTests(unittest.TestCase):
    def test_inventory_handles_checkpoint_types_and_keeps_scope_inside_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "outputs/run/checkpoints/version_0"
            folder.mkdir(parents=True)
            for name in ("last.ckpt", "step000050000.ckpt", "model.pth", "source.pt"):
                (folder / name).write_bytes(b"probe")
            (folder / "metrics.json").write_text("{}")
            rows, errors = audit.checkpoint_files(root, ["outputs", "../outside"])
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["bytes"] == 5 for row in rows))
            self.assertEqual(len(errors), 1)
            self.assertTrue(all(Path(row["path"]).is_relative_to(root) for row in rows))

    def test_literal_checkpoint_refs_and_unresolved_templates_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs = audit.checkpoint_references(
                'SOURCE_CHECKPOINT=outputs/parent.ckpt\nCHECKPOINT="${ROOT}/step${STEP_TAG}.ckpt"\n', root)
            self.assertEqual(refs, [str((root / "outputs/parent.ckpt").resolve())])

    def test_live_logs_find_parent_and_run_directory_pending_job_is_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "train.out"
            source = root / "outputs/parent.ckpt"
            run = root / "outputs/current/job_445516"
            log.write_text(f"SOURCE_CHECKPOINT={source.as_posix()} TRAIN_OUTPUT={run.as_posix()}\n")
            def query(command):
                if command[0] == "squeue":
                    return "445516|gs-order-context|RUNNING|ariel-v8\n445518|gs-order-context|PENDING|QOSMaxGRESPerUser\n", None
                if command[1:3] == ["show", "job"]:
                    job_id = command[-1]
                    output = log.as_posix() if job_id == "445516" else (root / "pending.out").as_posix()
                    return f"JobId={job_id} JobName=gs-order-context WorkDir={root.as_posix()} StdOut={output} StdErr=/dev/null Dependency=(null)", None
                return 'SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}"', None
            with patch.object(audit, "run_query", side_effect=query):
                jobs, errors, complete = audit.live_jobs("clue9986")
            self.assertTrue(complete)
            self.assertFalse(errors)
            self.assertEqual(jobs[0]["checkpoint_references"], [str(source.resolve())])
            self.assertEqual(jobs[0]["output_paths"], [str(run.resolve())])
            self.assertFalse(jobs[0]["requires_manual_check"])
            self.assertTrue(jobs[1]["requires_manual_check"])

    def test_main_writes_only_audit_and_protects_reported_live_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "outputs/current"
            folder.mkdir(parents=True)
            ckpt = folder / "last.ckpt"
            ckpt.write_bytes(b"fake weights: must not load")
            output = root / "audit.json"
            jobs = [{"checkpoint_references": [str(ckpt.resolve())], "output_paths": [str(folder.resolve())]}]
            with patch.object(audit, "live_jobs", return_value=(jobs, [], True)):
                audit.main(["--repo", str(root), "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["checkpoints"][0]["referenced_by_live_job"])
            self.assertTrue(report["checkpoints"][0]["inside_reported_live_output"])
            self.assertEqual(ckpt.read_bytes(), b"fake weights: must not load")
            self.assertEqual(len(list(folder.iterdir())), 1)

    def test_missing_slurm_query_is_explicitly_incomplete(self):
        with patch.object(audit, "run_query", return_value=("", "squeue unavailable")):
            jobs, errors, complete = audit.live_jobs("clue9986")
        self.assertFalse(complete)
        self.assertFalse(jobs)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()

"""Reviewed cleanup never crosses scope or deletes changed/live checkpoints."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("checkpoint_prune", ROOT / "scripts/prune_nfcgs_checkpoints.py")
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)


def item(path, root):
    stat = path.stat()
    return {"relative_path": path.relative_to(root).as_posix(), "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), "inode": stat.st_ino}


def fixture(root):
    (root / "scripts").mkdir()
    (root / "scripts/run_nfcgs.py").write_text("# marker")
    (root / "outputs/run").mkdir(parents=True)
    keep, delete = root / "outputs/run/final.ckpt", root / "outputs/run/old.ckpt"
    keep.write_bytes(b"final")
    delete.write_bytes(b"old")
    return {"repo": str(root), "user": "clue9986", "keep": [item(keep, root)],
            "delete": [item(delete, root)], "audit_jobs": [], "confirmed_pending_sources": {}}, keep, delete


class CheckpointPruneTests(unittest.TestCase):
    def test_preview_does_not_delete_or_query_slurm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, keep, delete = fixture(root)
            with patch.object(prune, "live_jobs") as query, patch.object(prune.subprocess, "run") as rm:
                prune.main([], embedded_plan=plan)
            query.assert_not_called()
            rm.assert_not_called()
            self.assertTrue(keep.exists() and delete.exists())

    def test_apply_calls_rm_only_for_exact_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, keep, delete = fixture(root)
            with patch.object(prune, "live_jobs", return_value=([], [], True)), patch.object(prune.subprocess, "run") as rm:
                prune.main(["--apply"], embedded_plan=plan)
            rm.assert_called_once_with(["rm", "--", str(delete)], check=True)
            self.assertTrue(keep.exists())

    def test_changed_candidate_or_missing_keeper_stops_before_any_rm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, keep, delete = fixture(root)
            delete.write_bytes(b"new checkpoint")
            with patch.object(prune.subprocess, "run") as rm, self.assertRaisesRegex(ValueError, "changed since audit"):
                prune.main(["--apply"], embedded_plan=plan)
            rm.assert_not_called()
            plan["delete"] = [item(delete, root)]
            keep.unlink()
            with self.assertRaisesRegex(ValueError, "required checkpoint"):
                prune.validate_snapshot(root, plan)

    def test_outside_path_keep_overlap_and_non_checkpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, keep, delete = fixture(root)
            for relative in ("../outside.ckpt", "/outside.ckpt", "outputs/run/old.json", "checkpoints/parent.ckpt"):
                with self.assertRaisesRegex(ValueError, "invalid deletion path"):
                    prune.target_path(root, {"relative_path": relative})
            plan["delete"] = plan["keep"]
            with self.assertRaisesRegex(ValueError, "overlap"):
                prune.validate_snapshot(root, plan)

    def test_future_output_tree_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, keep, delete = fixture(root)
            plan["protected_output_prefixes"] = [str(delete.parent)]
            with self.assertRaisesRegex(ValueError, "protected output tree"):
                prune.validate_snapshot(root, plan)

    def test_new_jobs_unresolved_pending_input_and_live_output_block_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, keep, delete = fixture(root)
            candidates, _ = prune.validate_snapshot(root, plan)
            job = {"job_id": "445516", "requires_manual_check": True,
                   "details": {"WorkDir": str(root)}, "checkpoint_references": [], "output_paths": []}
            with self.assertRaisesRegex(ValueError, "new jobs"):
                prune.validate_live_jobs(plan, [job], candidates)
            plan["audit_jobs"] = [job]
            with self.assertRaisesRegex(ValueError, "unresolved input"):
                prune.validate_live_jobs(plan, [job], candidates)
            plan["confirmed_pending_sources"] = {"445516": [str(keep)]}
            prune.validate_live_jobs(plan, [job], candidates)
            job["output_paths"] = [str(delete.parent)]
            with self.assertRaisesRegex(ValueError, "protects deletion target"):
                prune.validate_live_jobs(plan, [job], candidates)


if __name__ == "__main__":
    unittest.main()

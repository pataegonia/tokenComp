"""Exercise real seed-pair submission/state logic with a simulated scheduler."""
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import submit_nfcgs_context_seed_pair as submit


class ContextSeedSubmissionTests(unittest.TestCase):
    def fixture(self, root):
        for name in submit.FILES:
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, path)
        parent = root/"parent.ckpt"; parent.write_bytes(b"same parent")
        return {"REPO_DIR": str(root), "SOURCE_CHECKPOINT": str(parent),
                "DATASET_ROOT": str(root/"compute_only_dataset"), "CONTEXT_SEED_STATE_DIR": str(root/"state"),
                "CONTEXT_SEED_OUTPUT_ROOT": str(root/"outputs"),
                # Unrelated inherited settings must not alter this fixed recipe.
                "RATE_LAMBDA": "0", "TOKEN_ORDER": "nn_score", "MAX_STEPS": "10",
                "MODEL_SEED": "1", "DATA_LOADER_SEED": "2"}

    def test_missing_login_dataset_preview_and_qos_retry_preserve_ids_and_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); env = self.fixture(root); calls = []; fail = {"call": 2}
            def sbatch(command, **kwargs):
                calls.append((command, dict(kwargs["env"])))
                if len(calls) == fail.get("call"):
                    raise subprocess.CalledProcessError(1, command, stderr="QOSMaxSubmitJobPerUserLimit")
                return SimpleNamespace(stdout=f"{82000+len(calls)};cluster\n")
            with patch.dict("os.environ", env), patch.object(submit.subprocess, "run", side_effect=sbatch), \
                 patch("sys.stdout", new_callable=StringIO), patch("sys.stderr", new_callable=StringIO):
                submit.main(["--dry-run"])
                self.assertFalse(calls); self.assertFalse((root/"state").exists())
                with self.assertRaises(SystemExit):
                    submit.main([])
                saved = json.loads((root/"state/state.json").read_text())
                self.assertEqual(saved["arms"]["on16"], {"train": "82001"})
                self.assertFalse((root/"state/.submit-lock").exists())
                fail.clear(); submit.main([])
                saved = json.loads((root/"state/state.json").read_text())
                self.assertEqual(saved["arms"]["on16"], {"train": "82001", "eval": "82003"})
                self.assertEqual(saved["arms"]["off16"], {"train": "82004", "eval": "82005"})
                self.assertIn("--dependency=afterok:82001", calls[2][0])
                self.assertIn("--dependency=afterok:82004", calls[4][0])
                self.assertEqual(sum("train_nfcgs" in c[-1] for c, _ in calls), 2)
                self.assertEqual([e["CONTEXT_SEED_ARM"] for c, e in calls if "train_nfcgs" in c[-1]], ["on16", "off16"])
                self.assertEqual(calls[2][1]["TRAIN_JOB_ID"], "82001")
                self.assertEqual(calls[4][1]["TRAIN_JOB_ID"], "82004")
                self.assertEqual({e["SOURCE_CHECKPOINT"] for _, e in calls}, {str((root/"parent.ckpt").resolve())})
                before = len(calls); submit.main([])
                self.assertEqual(len(calls), before)

    def test_missing_files_parent_and_lock_stop_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); env = self.fixture(root)
            with patch.dict("os.environ", env), patch.object(submit.subprocess, "run") as sbatch, \
                 patch("sys.stdout", new_callable=StringIO):
                missing = root/submit.FILES[-1]; content = missing.read_bytes(); missing.unlink()
                with self.assertRaises(FileNotFoundError):
                    submit.main([])
                missing.write_bytes(content)
                parent = root/"parent.ckpt"; parent.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "FullSplit parent"):
                    submit.main([])
                parent.write_bytes(b"same parent")
                (root/"state/.submit-lock").mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    submit.main([])
                sbatch.assert_not_called()

    def test_runtime_or_parent_change_and_orphan_eval_state_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); env = self.fixture(root); calls = []
            def sbatch(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(stdout=f"{83000+len(calls)}\n")
            with patch.dict("os.environ", env), patch.object(submit.subprocess, "run", side_effect=sbatch), \
                 patch("sys.stdout", new_callable=StringIO):
                submit.main([]); self.assertEqual(len(calls), 4)
                runtime = root/submit.FILES[0]; original = runtime.read_bytes()
                runtime.write_bytes(original+b"\n# changed\n")
                with self.assertRaisesRegex(ValueError, "differs from inputs/code"):
                    submit.main(["--dry-run"])
                runtime.write_bytes(original)
                parent = root/"parent.ckpt"; parent.write_bytes(b"other parent")
                with self.assertRaisesRegex(ValueError, "differs from inputs/code"):
                    submit.main([])
                parent.write_bytes(b"same parent")
                state = root/"state/state.json"; saved = json.loads(state.read_text())
                saved["arms"]["off16"].pop("train"); state.write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, "invalid dependencies"):
                    submit.main([])
                self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()

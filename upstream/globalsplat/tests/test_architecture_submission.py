"""Mock Slurm submissions while retaining actual dependency/state logic."""
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import subprocess

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("arch_submit",ROOT/"scripts/submit_nfcgs_architecture_pilots.py")
submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)


class ArchitectureSubmissionTests(unittest.TestCase):
    def fixture(self,root):
        for name in submit.FILES:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("runtime file\n")
            if name in submit.MARKERS:
                p.write_text(submit.MARKERS[name]+"\n")
        vanilla=root/"vanilla.ckpt";vanilla.write_bytes(b"checkpoint")
        for split in ("train","test"):
            p=root/f"data/{split}/index.json";p.parent.mkdir(parents=True);p.write_text("{}")
        return {"REPO_DIR":str(root),"VANILLA_CHECKPOINT":str(vanilla),"DATASET_ROOT":str(root/"data"),
                "ARCH_PILOTS_STATE_DIR":str(root/"state"),"ARCH_PILOTS_PREPARATION":str(root/"prep"),
                "ARCH_PILOTS_OUTPUT_ROOT":str(root/"outputs")}

    def test_fullwidth_extension_adds_only_two_jobs_and_reuses_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=self.fixture(root);calls=[]
            def sbatch(command,**kwargs):
                calls.append((command,dict(kwargs["env"])))
                return SimpleNamespace(stdout=f"{84000+len(calls)};cluster\n")
            with patch.dict("os.environ",env),patch.object(submit.subprocess,"run",side_effect=sbatch),patch("sys.stdout",new_callable=StringIO) as output:
                submit.main([])
                self.assertEqual(len(calls),7)
                state_file=root/"state/state.json"
                original=state_file.read_bytes()
                saved=json.loads(original)
                submit.main(["--include-fullwidth","--dry-run"])
                self.assertEqual(len(calls),7)
                self.assertEqual(state_file.read_bytes(),original)
                self.assertIn("PLAN score736_only TRAIN=new VALIDATION=new TEST=deferred",output.getvalue())
                submit.main(["--include-fullwidth"])
                self.assertEqual(len(calls),9)
                train,validation=calls[-2:]
                self.assertIn("--dependency=afterok:84001",train[0])
                self.assertIn("--dependency=afterok:84008",validation[0])
                self.assertEqual(train[1]["PILOT_ARM"],"score736_only")
                self.assertEqual(validation[1]["PILOT_ARM"],"score736_only")
                self.assertEqual(validation[1]["PILOT_EVAL_MODE"],"validation")
                self.assertEqual(validation[1]["TRAIN_JOB_ID"],"84008")
                current=json.loads(state_file.read_text())
                self.assertEqual(current["prepare"],saved["prepare"])
                for arm,jobs in saved["arms"].items():
                    self.assertEqual(current["arms"][arm],jobs)
                self.assertEqual(current["arms"]["score736_only"],{"train":"84008","validation":"84009"})
                self.assertEqual(sum("prepare_nfcgs_architecture" in command[-1] for command,_ in calls),1)
                submit.main(["--include-fullwidth"])
                self.assertEqual(len(calls),9)

    def test_compute_node_dataset_need_not_exist_on_login_node(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=self.fixture(root)
            for split in ("train","test"):
                (root/f"data/{split}/index.json").unlink()
            calls=[]
            def sbatch(command,**kwargs):
                calls.append((command,dict(kwargs["env"])))
                return SimpleNamespace(stdout=f"{81000+len(calls)};cluster\n")
            with patch.dict("os.environ",env),patch.object(submit.subprocess,"run",side_effect=sbatch),patch("sys.stdout",new_callable=StringIO) as output:
                submit.main(["--dry-run"])
                self.assertFalse(calls)
                self.assertFalse((root/"state").exists())
                self.assertIn("DATASET_INDEX_CHECK=prepare_job_on_compute_node",output.getvalue())
                submit.main([])
                self.assertEqual(len(calls),7)
                self.assertTrue(calls[0][0][-1].endswith("prepare_nfcgs_architecture_pilots.slurm"))
                self.assertEqual(calls[0][1]["DATASET_ROOT"],str((root/"data").resolve()))
                self.assertTrue(all("--dependency=afterok:81001" in command for command,_ in calls if "train_nfcgs_architecture" in command[-1]))
                Path(env["VANILLA_CHECKPOINT"]).unlink()
                with self.assertRaisesRegex(FileNotFoundError,"required existing input missing"):
                    submit.main(["--dry-run"])
                self.assertEqual(len(calls),7)

    def test_qos_partial_recovery_optional_fullwidth_and_test_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=self.fixture(root);calls=[];fail={"call":3}
            def sbatch(command,**kwargs):
                calls.append((command,dict(kwargs["env"])))
                number=len(calls)
                if number==fail.get("call"):
                    raise subprocess.CalledProcessError(1,command,stderr="QOSMaxSubmitJobPerUserLimit")
                return SimpleNamespace(stdout=f"{80000+number};cluster\n")
            with patch.dict("os.environ",env),patch.object(submit.subprocess,"run",side_effect=sbatch):
                submit.main(["--dry-run"])
                self.assertFalse((root/"state").exists())
                with self.assertRaises(SystemExit):submit.main([])
                saved=json.loads((root/"state/state.json").read_text())
                self.assertEqual(saved["prepare"],"80001")
                self.assertEqual(saved["arms"]["score56_only"]["train"],"80002")
                self.assertNotIn("validation",saved["arms"]["score56_only"])
                fail.clear();submit.main([])
                accepted=[c for i,c in enumerate(calls,1) if i!=3]
                self.assertEqual(len(accepted),7)
                self.assertEqual(sum("prepare_nfcgs_architecture" in c[0][-1] for c in accepted),1)
                self.assertEqual(sum("train_nfcgs_architecture" in c[0][-1] for c in accepted),3)
                before=len(calls);submit.main([]);self.assertEqual(len(calls),before)
                submit.main(["--include-fullwidth","--with-test"])
                saved=json.loads((root/"state/state.json").read_text())
                self.assertEqual(len(saved["arms"]),4)
                self.assertTrue(all("test" in arm for arm in saved["arms"].values()))
                for command,export in calls:
                    if "train_nfcgs_architecture" in command[-1]:
                        self.assertIn("--dependency=afterok:80001",command)
                    if "eval_nfcgs_architecture" in command[-1] and export["PILOT_EVAL_MODE"]=="test":
                        validation=saved["arms"][export["PILOT_ARM"]]["validation"]
                        self.assertIn(f"--dependency=afterok:{validation}",command)
                self.assertFalse((root/"state/.submit-lock").exists())

    def test_missing_files_locks_and_changed_runtime_stop_before_new_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=self.fixture(root)
            with patch.dict("os.environ",env),patch.object(submit.subprocess,"run",return_value=SimpleNamespace(stdout="90001")) as run:
                (root/"state/.submit-lock").mkdir(parents=True)
                with self.assertRaises(FileExistsError):submit.main([])
                run.assert_not_called()
                (root/"state/.submit-lock").rmdir()
                submit.main([])
                before=run.call_count
                with (root/submit.FILES[0]).open("a") as handle:
                    handle.write("changed architecture\n")
                with self.assertRaisesRegex(ValueError,"differs from inputs/code"):
                    submit.main([])
                self.assertEqual(run.call_count,before)


if __name__=="__main__":
    unittest.main()

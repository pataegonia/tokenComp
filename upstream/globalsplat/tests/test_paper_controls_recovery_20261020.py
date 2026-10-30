"""Only two full-state resumes; existing eval IDs and accepted IDs survive retries."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_sort_context_recovery import checkpoint, small_config

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
spec=importlib.util.spec_from_file_location("paper_control_recovery",ROOT/"scripts/recover_nfcgs_paper_controls_20261020.py")
recovery=importlib.util.module_from_spec(spec);sys.modules[spec.name]=recovery;spec.loader.exec_module(recovery)


class Cluster:
    def __init__(self):
        self.states={"456951":"FAILED","457099":"FAILED","457098":"PENDING","457100":"PENDING"}
        self.deps={"457098":"afterok:456951(failed)","457100":"afterok:457099(failed)"}
        self.submissions=[];self.updates=[];self.fail_submit=None;self.fail_update=False

    def command(self,args,**kwargs):
        if args[0]=="sacct":
            return "\n".join(f"{job}|{state}|test|" for job,state in self.states.items())
        if args[0]=="squeue":
            if "%E" in args[-1]:
                return "\n".join(f"{job}|{self.states[job]}|{dep}" for job,dep in self.deps.items() if self.states[job]=="PENDING")
            return "\n".join(f"{job}|{state}" for job,state in self.states.items() if state in recovery.base.ACTIVE)
        if args[0]=="sbatch":
            self.submissions.append((args,kwargs["env"].copy()))
            if len(self.submissions)==self.fail_submit:
                raise subprocess.CalledProcessError(1,args,stderr="QOSMaxSubmitJobPerUserLimit")
            job=str(90000+len(self.submissions));self.states[job]="PENDING";return job+";cluster"
        if args[0]=="scontrol":
            if self.fail_update:raise subprocess.CalledProcessError(1,args,stderr="temporary controller failure")
            self.updates.append(args)
            job=args[2].split("=",1)[1];self.deps[job]=args[3].split("=",1)[1]
            return ""
        raise AssertionError(args)


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self,directory,steps=(25000,15000)):
        repo=Path(directory).resolve()
        for relative in recovery.FILES:
            dest=repo/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
        folder=repo/"logs/slurm/paper_controls_20261018";folder.mkdir(parents=True)
        rows=[];sources=[]
        for target,step in zip(recovery.TARGETS,steps):
            point=target.point;run_dir=point.run_root(repo)/f"job_{target.train}"
            path=run_dir/f"checkpoints/nfcgs_main/version_0/step{step:09d}.ckpt"
            path.parent.mkdir(parents=True);torch.save(checkpoint(small_config(point),step),path);sources.append(path)
            rows.append(dict(order=point.order,schedule=point.schedule,stages=str(point.stages),
                **{"lambda":point.rate},train_job=target.train,eval_job=target.evaluation,
                source_checkpoint="unused_parent",output=str(run_dir)))
        with (folder/"submissions.tsv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t");writer.writeheader();writer.writerows(rows)
        return repo,Cluster(),sources

    def run_recovery(self,repo,cluster,**kwargs):
        with patch.object(recovery.base.Point,"config",small_config), \
             patch.object(recovery.base,"command",side_effect=cluster.command), \
             patch.dict(os.environ,{"USER":"test"}),contextlib.redirect_stdout(io.StringIO()):
            return recovery.recover(repo,**kwargs)

    def test_preview_readonly_then_two_resumes_reuse_pending_evals_and_existing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,cluster,sources=self.fixture(directory)
            original=[path.read_bytes() for path in sources]
            self.run_recovery(repo,cluster)
            self.assertFalse(cluster.submissions);self.assertFalse(cluster.updates)
            self.assertFalse((repo/"logs/slurm/paper_controls_recovery_20261020").exists())
            result=self.run_recovery(repo,cluster,apply=True)
            self.assertEqual(len(cluster.submissions),2);self.assertEqual(len(cluster.updates),2)
            self.assertEqual(cluster.deps,{"457098":"afterok:90001","457100":"afterok:90002"})
            for i,target in enumerate(recovery.TARGETS):
                args,env=cluster.submissions[i]
                self.assertEqual(env["RESUME_EXPECTED_STEP"],str((25000,15000)[i]))
                self.assertEqual(env["SCORE_SLICE_CHANNELS"],"16")
                self.assertEqual(env["RESUME_RUN_DIR"],str(sources[i].parents[3]))
                self.assertEqual(result["points"][target.point.key]["eval_job"],target.evaluation)
                self.assertIn("resume_nfcgs_paper_controls_20261020.slurm",args[-1])
                self.assertEqual(sources[i].read_bytes(),original[i])
            self.run_recovery(repo,cluster,apply=True)
            self.assertEqual(len(cluster.submissions),2);self.assertEqual(len(cluster.updates),2)

    def test_dependency_update_failure_keeps_accepted_training_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,cluster,_=self.fixture(directory);cluster.fail_update=True
            with self.assertRaises(subprocess.CalledProcessError):self.run_recovery(repo,cluster,apply=True)
            state=json.loads((repo/"logs/slurm/paper_controls_recovery_20261020/state.json").read_text())
            self.assertEqual(state["points"][recovery.TARGETS[0].point.key]["train_job"],"90001")
            cluster.fail_update=False
            self.run_recovery(repo,cluster,apply=True)
            self.assertEqual(len(cluster.submissions),2)

    def test_second_submission_rejection_reuses_first_training_and_eval_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,cluster,_=self.fixture(directory);cluster.fail_submit=2
            with self.assertRaises(subprocess.CalledProcessError):self.run_recovery(repo,cluster,apply=True)
            cluster.fail_submit=None
            self.run_recovery(repo,cluster,apply=True)
            self.assertEqual(len(cluster.submissions),3)
            self.assertEqual(cluster.deps["457098"],"afterok:90001")
            self.assertEqual(cluster.deps["457100"],"afterok:90003")

    def test_full_state_snapshot_preserves_optimizer_scheduler_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            _,_,sources=self.fixture(directory);source=sources[0]
            target=source.with_name("resume_job_90001.ckpt")
            raw=source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                recovery.copy_for_resume(source,target,25000,recovery.digest(source),small_config(recovery.TARGETS[0].point))
            self.assertEqual(source.read_bytes(),raw);self.assertEqual(target.read_bytes(),raw)
            saved=torch.load(target,weights_only=False)
            self.assertEqual(saved["global_step"],25000)
            self.assertEqual(saved["optimizer_states"][0]["state"][0]["step"].item(),25000)
            self.assertEqual(saved["lr_schedulers"][0]["last_epoch"],25000)
            with self.assertRaises(ValueError):
                recovery.copy_for_resume(source,target,25000,"wrong",small_config(recovery.TARGETS[0].point))

    def test_missing_or_weights_only_checkpoint_never_falls_back_to_parent(self):
        for kind in ("missing","weights_only"):
            with tempfile.TemporaryDirectory() as directory:
                repo,cluster,sources=self.fixture(directory)
                if kind=="missing":sources[1].unlink()
                else:
                    state=torch.load(sources[1],weights_only=False);state.pop("optimizer_states");torch.save(state,sources[1])
                with self.assertRaises(ValueError):self.run_recovery(repo,cluster,apply=True)
                self.assertFalse(cluster.submissions);self.assertFalse(cluster.updates)

    def test_active_original_job_or_changed_eval_dependency_prevents_all_mutations(self):
        for kind in ("active","dependency"):
            with tempfile.TemporaryDirectory() as directory:
                repo,cluster,_=self.fixture(directory)
                if kind=="active":cluster.states["457099"]="RUNNING"
                else:cluster.deps["457100"]="afterok:12345"
                with self.assertRaises(ValueError):self.run_recovery(repo,cluster,apply=True)
                self.assertFalse(cluster.submissions);self.assertFalse(cluster.updates)

    def test_final_checkpoint_needs_only_existing_eval_dependency_update(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,cluster,_=self.fixture(directory,steps=(50000,50000))
            self.run_recovery(repo,cluster,apply=True)
            self.assertFalse(cluster.submissions)
            self.assertEqual(cluster.deps,{"457098":"afterany:456951","457100":"afterany:457099"})

    def test_resume_runner_uses_absolute50000_and_full_state_in_original_version(self):
        import run_nfcgs
        from globalsplat.checkpointing import resolve_run
        with tempfile.TemporaryDirectory() as directory:
            repo,_,sources=self.fixture(directory)
            target=recovery.TARGETS[1]
            args=run_nfcgs.parse_args(["train","--resume","--checkpoint",str(sources[1]),
                "--rate-lambda",target.point.rate,"--score-path","minimal","--token-order",target.point.order,
                "--score-context-schedule",target.point.schedule,"--score-spatial-stages","4",
                "--score-spatial-kernel","5","--score-context-quantization","ste","--no-spatial-context",
                "--devices","2","--batch-size","2","--accumulate","2","--workers","8","--max-steps","50000",
                "--output",str(sources[1].parents[3])])
            command,_,_=run_nfcgs.build_command(args)
            self.assertIn("checkpointing.resume=true",command);self.assertIn("trainer.max_steps=50000",command)
            self.assertNotIn("checkpointing.allow_score_path_conversion=true",command)
            resolved=resolve_run(load=str(sources[1]),resume=True,auto_resume=False,exp_ckpt_base="unused")
            self.assertEqual(Path(resolved.ckpt_dir),sources[1].parent)
            self.assertIsNone(resolved.weights_only)

    def test_slurm_shell_syntax_lf_and_fullstate_flags(self):
        bash=Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.is_file():
            bash=shutil.which("bash")
            if not bash:self.skipTest("bash unavailable")
        script=ROOT/"scripts/slurm/resume_nfcgs_paper_controls_20261020.slurm"
        raw=script.read_bytes();self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"));self.assertNotIn(b"\r",raw)
        result=subprocess.run([str(bash),"-n",str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn(b"--resume",raw);self.assertIn(b"--max-steps 50000",raw)
        self.assertNotIn(b"--allow-score-path-conversion",raw)

    def test_slurm_executes_snapshot_then_resume_with_original_output_and_off16(self):
        bash=Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.is_file():
            bash=shutil.which("bash")
            if not bash:self.skipTest("bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            repo,_,sources=self.fixture(directory)
            conda=repo/"fake_conda/etc/profile.d";conda.mkdir(parents=True)
            (conda/"conda.sh").write_text("conda() { return 0; }\n",newline="\n")
            fake=repo/"fake_bin";fake.mkdir()
            for name,content in {
                "conda":'#!/usr/bin/env bash\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n',
                "python":'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_LOG"\n',
            }.items():
                (fake/name).write_text(content,newline="\n");(fake/name).chmod(0o755)
            env=recovery.base.environment(recovery.TARGETS[1].point,repo)
            env.update(SLURM_JOB_ID="90001",RESUME_ORIGINAL_TRAIN="457099",RESUME_RUN_DIR=sources[1].parents[3].as_posix(),
                RESUME_CHECKPOINT=sources[1].as_posix(),RESUME_EXPECTED_STEP="15000",RESUME_EXPECTED_SHA256="expected",
                RESUME_EXPECTED_CONFIG="{}",SCORE_SLICE_CHANNELS="16",NFCGS_TMP_ROOT="tmp_job",
                FAKE_CONDA_BASE=(repo/"fake_conda").as_posix(),FAKE_LOG=(repo/"calls.txt").as_posix())
            result=subprocess.run([str(bash),"-c",'export PATH="$(pwd)/fake_bin:$PATH"; '
                'bash scripts/slurm/resume_nfcgs_paper_controls_20261020.slurm'],
                cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            calls=(repo/"calls.txt").read_text().splitlines();self.assertEqual(len(calls),2)
            self.assertIn("--copy-source",calls[0]);self.assertIn("resume_job_90001.ckpt",calls[0])
            for flag in ("--resume","--score-context-schedule dyadic4","--score-spatial-stages 4",
                         "--score-slice-channels 16","--no-spatial-context","--max-steps 50000",
                         "--devices 2","--batch-size 2","--accumulate 2"):
                self.assertIn(flag,calls[1])
            self.assertIn("--output "+sources[1].parents[3].as_posix(),calls[1])
            self.assertNotIn("--allow-score-path-conversion",calls[1])
            self.assertTrue((repo/"tmp_job/job_90001").is_dir())


if __name__=="__main__":unittest.main()

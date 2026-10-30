"""Pre-submit Stage B without leaking test results or replacing frozen Stage A."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import submit_nfcgs_fresh_rd as submit
import run_nfcgs_fresh_rd as runner
from globalsplat.fresh_rd import VERSION,atomic_json,checkpoint_path,file_hash,read_json,run_key,wave_runs


def setup(tmp_path):
    args=argparse.Namespace(root=tmp_path/"out",state=tmp_path/"logs/state.json",dataset_root=tmp_path/"data",
        preparation=tmp_path/"prep",native_checkpoint=tmp_path/"native.ckpt",dry_run=False,require_gate0=False,
        budget=100000,provisional=True,restart_stage_b_from_60k=False,reason=None,wave=1)
    state=dict(version=VERSION,inputs=dict(root=str(args.root.resolve()),dataset=str(args.dataset_root.resolve()),
        preparation=str(args.preparation.resolve()),native_checkpoint=str(args.native_checkpoint.resolve()),
        code={n:file_hash(ROOT/n) for n in submit.FILES}),jobs={},runs={},lanes={})
    for lane,(cell,rate,seed) in enumerate(wave_runs(1)):
        key=run_key(cell,rate,seed);job=str(5000+lane)
        run=dict(key=key,cell=cell,rate_lambda=rate,seed_index=seed,wave=1,preparation=str(args.preparation),
            run_dir=str(args.root/key/f"job_{job}"))
        state["runs"][key]=run
        state["jobs"][key+"/A"]=dict(id=job,dependencies=[],history=[],lane=lane,run=copy.deepcopy(run))
        state["lanes"][str(lane)]=job
    state["jobs"]["monitor/1/A"]=dict(id="5999",dependencies=[],history=[],lane=15,run=None)
    state["lanes"]["15"]="5999"
    return args,state


def responses(start=8000):
    counter=iter(range(start,start+200))
    return lambda *a,**kw:subprocess.CompletedProcess(a[0],0,str(next(counter))+"\n","")


def provision(args,state):
    with patch.object(submit.subprocess,"run",side_effect=responses()) as call:
        submit.submit_stage_b(submit.Submission(args,state),args)
    assert call.call_count==15


def gate(args,state,recommendation=100000):
    changes=[]
    for run in state["runs"].values():
        folder=Path(run["run_dir"])
        atomic_json(folder/"run.json",dict(key=run["key"],protocol="frozen runtime"))
        for step in range(10000,60001,10000):
            p=checkpoint_path(folder,step);p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(f"{run['key']} immutable Stage A {step}".encode())
        atomic_json(folder/"stage_A_complete.json",dict(step=60000,checkpoint_sha256=file_hash(checkpoint_path(folder,60000))))
        for step in (40000,60000):
            out=folder/"validation"/f"step{step:09d}"
            rows=[dict(scene=str(i),mse=.01,psnr=20.,ssim=.7,lpips=.2,actual_bytes=1000,
                actual_score_bytes=200,actual_residual_y_bytes=600,actual_residual_z_bytes=100,
                actual_mean_bytes=0,actual_container_bytes=100,context_frame_ids=list(range(12)),
                target_frame_ids=list(range(12,20))) for i in range(128)]
            path=out/"nfcgs_main/scene_metrics.json";atomic_json(path,rows)
            atomic_json(out/"complete.json",dict(split="validation",mode="validation",scene_count=128,
                checkpoint_sha256=file_hash(checkpoint_path(folder,step)),scene_metrics_sha256=file_hash(path)))
        (folder/"checkpoints/nfcgs_main/version_0/last.ckpt").write_bytes(b"provisional last checkpoint")
        checkpoint_path(folder,70000).write_bytes(b"provisional 70k checkpoint")
        atomic_json(folder/"stage_B_complete.json",dict(step=100000,kind="preserved provisional marker"))
        changes.append(dict(cell=run["cell"],**{"lambda":run["rate_lambda"]},seed=run["seed_index"]))
    atomic_json(args.root/"convergence/gate1.json",dict(test_used=False,runs=changes,
        recommended_budget=recommendation,consider_150k=False))


def test_provisional_only_trains_after_own_stage_a_and_is_idempotent(tmp_path):
    args,state=setup(tmp_path);before=copy.deepcopy(state)
    provision(args,state)
    assert len(state["jobs"])==31 and len(state["runs"])==15
    assert state.get("budget") is None
    assert state["provisional_stage_b"]["submitted_before_gate1"]
    assert not any(k.endswith("/test") or k=="monitor/1/final" for k in state["jobs"])
    for lane,(c,r,s) in enumerate(wave_runs(1)):
        key=run_key(c,r,s);job=state["jobs"][key+"/B"]
        assert job["dependencies"]==[before["jobs"][key+"/A"]["id"]]
        assert job["lane"]==lane
        assert job["env"]["FRESH_BUDGET"]=="100000" and job["env"]["FRESH_RESUME"]=="false"
        assert job["env"]["FRESH_RUN_DIR"]==before["runs"][key]["run_dir"]
    assert state["jobs"]["monitor/1/A"]==before["jobs"]["monitor/1/A"]
    with patch.object(submit.subprocess,"run") as call:
        submit.submit_stage_b(submit.Submission(args,state),args)
        call.assert_not_called()


@pytest.mark.parametrize("accepted_count",[3,14])
def test_partial_provisional_submission_reuses_accepted_ids(tmp_path,capsys,accepted_count):
    args,state=setup(tmp_path)
    answers=[subprocess.CompletedProcess([],0,str(i)+"\n","") for i in range(8000,8000+accepted_count)]
    answers.append(subprocess.CalledProcessError(1,["sbatch"],stderr="sbatch: error: fixture scheduler rejection\n"))
    with patch.object(submit.subprocess,"run",side_effect=answers):
        with pytest.raises(subprocess.CalledProcessError):submit.submit_stage_b(submit.Submission(args,state),args)
    assert "sbatch: error: fixture scheduler rejection" in capsys.readouterr().err
    persisted=read_json(args.state)
    assert "pending_submission" not in persisted
    assert persisted["submission_failures"][-1]["stderr"]=="sbatch: error: fixture scheduler rejection\n"
    accepted={k:v["id"] for k,v in state["jobs"].items() if k.endswith("/B")}
    with patch.object(submit.subprocess,"run",side_effect=responses(9000)) as call:
        submit.submit_stage_b(submit.Submission(args,state),args)
    assert call.call_count==15-accepted_count
    assert all(state["jobs"][k]["id"]==v for k,v in accepted.items())
    if accepted_count==14:
        assert call.call_args.kwargs["env"]["CELL"]=="msh"
        assert "--dependency=afterok:5014" in call.call_args.args[0]


def test_confirmation_waits_for_gate1_and_attaches_test_without_retraining(tmp_path):
    args,state=setup(tmp_path);provision(args,state)
    ids={k:v["id"] for k,v in state["jobs"].items() if k.endswith("/B")}
    args.provisional=False
    with patch.object(submit.subprocess,"run") as call:
        with pytest.raises(FileNotFoundError):submit.submit_stage_b(submit.Submission(args,state),args)
        call.assert_not_called()
    gate(args,state)
    with patch.object(submit.subprocess,"run",side_effect=responses(9000)) as call:
        submit.submit_stage_b(submit.Submission(args,state),args)
    assert call.call_count==16
    assert all(c.args[0][-1]!=submit.SLURM+"train_nfcgs_fresh_rd.slurm" for c in call.call_args_list)
    assert all(state["jobs"][k]["id"]==v for k,v in ids.items())
    assert state["budget"]==100000 and state["provisional_stage_b"]["status"]=="confirmed"
    assert state["provisional_stage_b"]["decision"]["test_used"] is False
    with patch.object(submit.subprocess,"run") as call:
        submit.submit_stage_b(submit.Submission(args,state),args)
        call.assert_not_called()


def test_extension_needs_explicit_restart_and_holdout_reason_for_override(tmp_path):
    args,state=setup(tmp_path);provision(args,state);gate(args,state)
    args.provisional=False;args.budget=120000
    with pytest.raises(ValueError,match="explicit --restart"):
        submit.submit_stage_b(submit.Submission(args,state),args)
    args.restart_stage_b_from_60k=True
    with patch.object(submit.subprocess,"run") as call:
        with pytest.raises(ValueError,match="holdout-based --reason"):
            submit.submit_stage_b(submit.Submission(args,state),args)
        call.assert_not_called()


def test_extension_forks_same_60k_and_preserves_all_original_outputs(tmp_path):
    args,state=setup(tmp_path);provision(args,state);gate(args,state,120000)
    before=copy.deepcopy(state);args.provisional=False;args.budget=120000;args.restart_stage_b_from_60k=True
    stopped=set();commands=[];new_id=iter(range(9000,9200))
    def run(command,**kw):
        commands.append(command)
        if command[0]=="scancel":
            stopped.add(command[1]);return subprocess.CompletedProcess(command,0)
        assert command[0]=="sbatch"
        return subprocess.CompletedProcess(command,0,str(next(new_id))+"\n","")
    def status(job):return "CANCELLED" if job in stopped else "RUNNING"
    with patch.object(submit,"job_state",side_effect=status),patch.object(submit,"verify_provisional_job"), \
            patch.object(submit.subprocess,"run",side_effect=run):
        submit.submit_stage_b(submit.Submission(args,state),args)
        count=len(commands)
        submit.submit_stage_b(submit.Submission(args,state),args)
        assert len(commands)==count
    original_b={v["id"] for k,v in before["jobs"].items() if k.endswith("/B")}
    assert stopped==original_b
    assert state["runs"]==before["runs"] and state["jobs"]==before["jobs"] and state["lanes"]==before["lanes"]
    branch=read_json(args.state.with_name("state_B120000.json"));root=args.root/"branches/B120000"
    assert branch["budget"]==120000 and state["provisional_stage_b"]["status"]=="branched"
    assert len([c for c in commands if c[0]=="sbatch"])==31
    for key,source in before["runs"].items():
        old=Path(source["run_dir"]);new=Path(branch["runs"][key]["run_dir"])
        assert file_hash(old/"run.json")==file_hash(new/"run.json")
        assert file_hash(checkpoint_path(old,60000))==file_hash(checkpoint_path(new,60000))
        assert checkpoint_path(old,70000).read_bytes()==b"provisional 70k checkpoint"
        assert (old/"checkpoints/nfcgs_main/version_0/last.ckpt").read_bytes()==b"provisional last checkpoint"
        assert not checkpoint_path(new,70000).exists()
        assert not (new/"checkpoints/nfcgs_main/version_0/last.ckpt").exists()
        assert not (new/"stage_B_complete.json").exists()
        train=branch["jobs"][key+"/B"]
        assert train["dependencies"]==[]  # Parent files are verified; historical A IDs can expire.
        assert train["env"]["FRESH_RESUME"]=="false" and train["env"]["FRESH_BUDGET"]=="120000"
        assert not original_b.intersection(branch["jobs"][key+"/test"]["dependencies"])
        a=runner.parse_args(["train","--cell",source["cell"],"--rate-lambda",source["rate_lambda"],
            "--stage","B","--budget","120000","--run-dir",str(new)])
        command,checkpoint=runner.build_command(a)
        assert checkpoint==checkpoint_path(new,60000)
        assert "optimizer.lr_milestones=[84000,108000]" in command
        assert "checkpointing.override_lr_milestones=true" in command
    assert (root/"branch_provenance.json").exists()


def test_invalid_stage_a_stops_extension_before_any_job_is_cancelled(tmp_path):
    args,state=setup(tmp_path);provision(args,state);gate(args,state,120000)
    args.provisional=False;args.budget=120000;args.restart_stage_b_from_60k=True
    source=next(iter(state["runs"].values()))
    checkpoint_path(source["run_dir"],60000).write_bytes(b"unexpected replacement")
    with patch.object(submit.subprocess,"run") as call:
        with pytest.raises(ValueError,match="60k checkpoint differs"):
            submit.submit_stage_b(submit.Submission(args,state),args)
        call.assert_not_called()


def test_interrupted_budget_rollback_cannot_be_confirmed_as_100k(tmp_path):
    args,state=setup(tmp_path);provision(args,state);gate(args,state)
    state["provisional_stage_b"]["rollback"]={"budget":120000}
    with pytest.raises(ValueError,match="rollback is already recorded"):
        submit.submit_stage_b(submit.Submission(args,state),args)
    args.provisional=False
    with patch.object(submit.subprocess,"run") as call:
        with pytest.raises(ValueError,match="rollback is already recorded"):
            submit.submit_stage_b(submit.Submission(args,state),args)
        call.assert_not_called()


def test_controller_only_migration_preserves_runtime_and_job_ids(tmp_path):
    args,state=setup(tmp_path);inputs=copy.deepcopy(state["inputs"])
    state["inputs"]["code"]["scripts/submit_nfcgs_fresh_rd.py"]="old controller"
    before=copy.deepcopy(state["jobs"])
    submit.adopt_inputs(state,copy.deepcopy(inputs),allow_stage_b_controller_update=True)
    assert state["jobs"]==before and state["inputs"]==inputs
    state["inputs"]["code"]["scripts/run_nfcgs_fresh_rd.py"]="changed runtime"
    with pytest.raises(ValueError,match="runtime changed"):
        submit.adopt_inputs(state,inputs,allow_stage_b_controller_update=True)

"""Placement recovery never restarts begun runs or duplicates accepted jobs."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import recover_nfcgs_fresh_start_20261021 as recovery
import submit_nfcgs_fresh_rd as submit
from globalsplat.fresh_rd import VERSION, file_hash, run_key, wave_runs


def fixture(tmp_path):
    args=argparse.Namespace(root=tmp_path/"outputs/fresh",state=tmp_path/"state/state.json",log_root=tmp_path/"logs",
        dataset_root=tmp_path/"data",preparation=tmp_path/"prep",native_checkpoint=tmp_path/"native",
        budget=100000,dry_run=False,restart_monitor=False)
    args.log_root.mkdir()
    code={name:file_hash(ROOT/name) for name in submit.FILES}
    code[recovery.PLACEMENT_FILES[0]]="old-scheduler-placement"
    state=dict(version=VERSION,inputs=dict(code=code,root=str(args.root),dataset=str(args.dataset_root),
        preparation=str(args.preparation),native_checkpoint=str(args.native_checkpoint)),runs={},jobs={},lanes={})
    for lane,(cell,rate,seed) in enumerate(wave_runs(1)):
        key=run_key(cell,rate,seed);job=str(1000+lane)
        run=dict(key=key,cell=cell,rate_lambda=rate,seed_index=seed,wave=1,preparation=str(args.preparation),run_dir=str(args.root/key/f"job_{job}"))
        state["runs"][key]=run
        state["jobs"][key+"/A"]=dict(id=job,lane=lane,history=[],dependencies=[])
        state["lanes"][str(lane)]=job
        (args.log_root/f"slurm-gs-fresh-train-{job}.err").write_text(
            f"validate_preparation(args)\nFileNotFoundError: {args.dataset_root}/train/index.json\n",encoding="utf-8")
    state["jobs"]["monitor/1/A"]=dict(id="1100",lane=15,history=[],dependencies=[])
    state["lanes"]["15"]="1100"
    return args,state


def test_recover_all_prestart_failures_and_reuse_ids(tmp_path):
    args,state=fixture(tmp_path)
    sub=submit.Submission(args,state)
    answers=[subprocess.CompletedProcess([],0,str(2000+i)+"\n","") for i in range(15)]
    with patch.object(recovery,"allocation_info",return_value=dict(state="FAILED",nodes="ariel-v7")), \
            patch.object(submit.subprocess,"run",side_effect=answers) as call:
        recovery.recover(sub,args)
        assert call.call_count==15
        for invocation in call.call_args_list:
            assert "--exclude=ariel-v7" in invocation.args[0]
            assert not any(s.startswith("--dependency") for s in invocation.args[0])
            assert invocation.kwargs["env"]["FRESH_RESUME"]=="false"
        recovery.recover(sub,args)
        assert call.call_count==15
    assert all(state["jobs"][key+"/A"]["history"] for key in state["runs"])
    assert state["jobs"]["monitor/1/A"]["id"]=="1100"


def test_begun_run_is_never_reinitialized(tmp_path):
    args,state=fixture(tmp_path)
    run=next(iter(state["runs"].values()))
    folder=Path(run["run_dir"]);folder.mkdir(parents=True)
    (folder/"run.json").write_text("{}")
    with patch.object(submit.subprocess,"run") as call:
        with pytest.raises(ValueError,match="training artifacts"):
            recovery.recover(submit.Submission(args,state),args)
        call.assert_not_called()


def test_active_job_is_not_replaced(tmp_path):
    args,state=fixture(tmp_path)
    with patch.object(recovery,"allocation_info",return_value=dict(state="RUNNING",nodes="ariel-v7")), \
            patch.object(submit.subprocess,"run") as call:
        with pytest.raises(ValueError,match="terminal failure"):
            recovery.recover(submit.Submission(args,state),args)
        call.assert_not_called()


def test_monitor_restart_is_explicit_and_idempotent(tmp_path):
    args,state=fixture(tmp_path);args.restart_monitor=True
    convergence=args.root/"convergence";convergence.mkdir(parents=True)
    lock=convergence/".monitor_1_A.lock";lock.write_text("pid=123\n")
    marker=dict(excluded_nodes=["ariel-v7"])
    sub=submit.Submission(args,state)
    with patch.object(submit,"job_state",side_effect=["RUNNING","CANCELLED"]), \
            patch.object(submit.subprocess,"run",side_effect=[subprocess.CompletedProcess([],0),subprocess.CompletedProcess([],0,"9000\n","")]) as call:
        recovery.recover_monitor(sub,args,marker)
        assert call.call_args_list[0].args[0]==["scancel","1100"]
    assert not lock.exists()
    assert state["jobs"]["monitor/1/A"]["id"]=="9000"
    with patch.object(submit.subprocess,"run") as call:
        recovery.recover_monitor(sub,args,marker)
        call.assert_not_called()


def monitor_fixture(tmp_path):
    args,state=fixture(tmp_path)
    state["inputs"]["code"]={name:file_hash(ROOT/name) for name in submit.FILES}
    monitor=state["jobs"]["monitor/1/A"]
    monitor.update(run=None,env=dict(submit.base_env(args),MONITOR_PHASE="A",MONITOR_WAVE="1"),
        command=["sbatch",submit.SLURM+"monitor_nfcgs_fresh_rd.slurm"])
    (args.log_root/"slurm-gs-fresh-monitor-1100.err").write_text(
        "FileExistsError: outputs/convergence/.monitor_1_A.lock\n",encoding="utf-8")
    convergence=args.root/"convergence";convergence.mkdir(parents=True)
    lock=convergence/".monitor_1_A.lock";lock.write_text("pid=123\n")
    for run in state["runs"].values():
        folder=Path(run["run_dir"]);folder.mkdir(parents=True)
        (folder/"run.json").write_text("already training")
    return args,state,lock


def test_monitor_only_preserves_begun_learners_and_reuses_accepted_job(tmp_path):
    args,state,lock=monitor_fixture(tmp_path);before=copy.deepcopy(state)
    sub=submit.Submission(args,state)
    answers=[subprocess.CompletedProcess([],0),subprocess.CompletedProcess([],0,"9000\n","")]
    with patch.object(submit,"job_state",side_effect=["FAILED","RUNNING","CANCELLED"]), \
            patch.object(recovery,"active_monitor_jobs",side_effect=[{"460326"},set()]), \
            patch.object(recovery,"verify_previous_monitor") as verify, \
            patch.object(submit.subprocess,"run",side_effect=answers) as call:
        recovery.recover_monitor_only(sub,args)
    assert verify.call_count==2
    assert call.call_args_list[0].args[0]==["scancel","460326"]
    command=call.call_args_list[1].args[0]
    assert command[-1]==submit.SLURM+"monitor_nfcgs_fresh_rd.slurm"
    assert not any(x.startswith("--dependency") for x in command)
    assert not lock.exists()
    assert state["jobs"]["monitor/1/A"]["id"]=="9000"
    assert state["jobs"]["monitor/1/A"]["history"]==["1100"]
    assert state["runs"]==before["runs"] and state["inputs"]==before["inputs"]
    assert {k:v for k,v in state["jobs"].items() if k!="monitor/1/A"}=={k:v for k,v in before["jobs"].items() if k!="monitor/1/A"}
    assert {k:v for k,v in state["lanes"].items() if k!="15"}=={k:v for k,v in before["lanes"].items() if k!="15"}
    with patch.object(submit,"job_state",return_value="RUNNING"),patch.object(submit.subprocess,"run") as call:
        recovery.recover_monitor_only(sub,args)
        call.assert_not_called()


def test_monitor_only_preserves_locks_if_unknown_monitor_active(tmp_path):
    args,state,lock=monitor_fixture(tmp_path);before=copy.deepcopy(state)
    with patch.object(submit,"job_state",return_value="FAILED"), \
            patch.object(recovery,"active_monitor_jobs",return_value={"9999"}), \
            patch.object(submit.subprocess,"run") as call:
        with pytest.raises(ValueError,match="other active monitors"):
            recovery.recover_monitor_only(submit.Submission(args,state),args)
        call.assert_not_called()
    assert lock.exists() and state==before


def test_monitor_only_refuses_changed_lock(tmp_path):
    args,state,lock=monitor_fixture(tmp_path)
    def active():
        lock.write_text("pid=456\n")
        return set()
    with patch.object(submit,"job_state",side_effect=["FAILED","COMPLETED"]), \
            patch.object(recovery,"active_monitor_jobs",side_effect=active), \
            patch.object(submit.subprocess,"run") as call:
        with pytest.raises(ValueError,match="lock changed"):
            recovery.recover_monitor_only(submit.Submission(args,state),args)
        call.assert_not_called()
    assert lock.read_text()=="pid=456\n"


def test_monitor_only_requires_matching_failure_and_frozen_runtime(tmp_path):
    args,state,lock=monitor_fixture(tmp_path)
    state["inputs"]["code"]["scripts/submit_nfcgs_fresh_rd.py"]="different"
    with pytest.raises(ValueError,match="unchanged sweep code"):
        recovery.recover_monitor_only(submit.Submission(args,state),args)
    state["inputs"]["code"]["scripts/submit_nfcgs_fresh_rd.py"]=file_hash(ROOT/"scripts/submit_nfcgs_fresh_rd.py")
    (args.log_root/"slurm-gs-fresh-monitor-1100.err").write_text("some other failure")
    with patch.object(submit,"job_state",return_value="FAILED"),patch.object(submit.subprocess,"run") as call:
        with pytest.raises(ValueError,match="recorded monitor-lock failure"):
            recovery.recover_monitor_only(submit.Submission(args,state),args)
        call.assert_not_called()
    assert lock.exists()


def test_monitor_only_sbatch_rejection_leaves_retryable_state(tmp_path):
    args,state,lock=monitor_fixture(tmp_path)
    sub=submit.Submission(args,state)
    with patch.object(submit,"job_state",side_effect=["FAILED","COMPLETED"]), \
            patch.object(recovery,"active_monitor_jobs",return_value=set()), \
            patch.object(submit.subprocess,"run",side_effect=subprocess.CalledProcessError(1,["sbatch"])):
        with pytest.raises(subprocess.CalledProcessError):
            recovery.recover_monitor_only(sub,args)
    assert state["lanes"]["15"]=="1100"
    assert state["jobs"]["monitor/1/A"]["id"]=="1100"
    assert "pending_submission" not in state


def test_previous_monitor_verification_rejects_learner(tmp_path):
    result=subprocess.CompletedProcess([],0,
        f"JobId=460326 JobName=gs-fresh-monitor WorkDir={ROOT} Command={ROOT}/scripts/slurm/train_nfcgs_fresh_rd.slurm","")
    with patch.object(recovery.subprocess,"run",return_value=result):
        with pytest.raises(ValueError,match="not a verified monitor"):
            recovery.verify_previous_monitor("460326")

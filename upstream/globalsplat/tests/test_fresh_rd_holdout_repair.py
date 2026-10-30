"""A late Stage B resave needs real holdout evaluation, never SHA relabeling."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import repair_nfcgs_stage_a_holdout as repair
import run_nfcgs_fresh_rd as runner
import submit_nfcgs_fresh_rd as submit
import monitor_nfcgs_fresh_rd as monitor
import test_fresh_rd_120k as fixtures
from globalsplat.fresh_rd import atomic_json, checkpoint_path, file_hash, read_json


def setup(tmp_path):
    args, state, _, _ = fixtures.repair_sweep(tmp_path)
    manifest = dict(arms={arm:{"sha256":"initializer"} for arm in ("two_path56","score56_only","msh_only")},
        holdout_sha256="holdout", validation_index_sha256="index", vanilla_sha256="vanilla")
    atomic_json(args.preparation/"preparation.json", manifest)
    target = None
    for run in state["runs"].values():
        folder = Path(run["run_dir"])
        a = runner.parse_args(repair.eval_args(run, args.dataset_root, folder))
        protocol = runner.identity(a)
        for name in runner.RUNTIME_REPAIR_FILES:
            protocol["code_sha256"][name] = "legacy-runtime"
        atomic_json(folder/"run.json", protocol)
        path = checkpoint_path(folder, 60000)
        saved = torch.load(path, weights_only=False)
        saved["fresh_rd_protocol"] = protocol
        saved["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["processed"] = 239979
        torch.save(saved, path)
        original = file_hash(path)
        atomic_json(folder/"stage_A_complete.json", dict(step=60000, checkpoint_sha256=original))
        fixtures.set_evaluation(folder, 60000, .009)
        for archive in (folder/"validation").glob("*.superseded_*"):
            done = read_json(archive/"complete.json")
            done["checkpoint_sha256"] = original
            atomic_json(archive/"complete.json", done)
        if run["key"] == "h_on/lambda0/s0":
            target = folder
            saved["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["processed"] = 239980
            torch.save(saved, path)  # Metrics remain bound to original bytes.
        for step in (10000,20000,30000,50000):
            output = folder/"validation"/f"step{step:09d}"
            rows = read_json(folder/"validation/step000040000/nfcgs_main/scene_metrics.json")
            atomic_json(output/"nfcgs_main/scene_metrics.json", rows)
            atomic_json(output/"complete.json", dict(split="validation",mode="validation",scene_count=128,
                checkpoint_sha256=file_hash(checkpoint_path(folder,step)),
                scene_metrics_sha256=file_hash(output/"nfcgs_main/scene_metrics.json")))
    atomic_json(args.state, state)
    return argparse.Namespace(root=args.root.resolve(),state=args.state.resolve(),submit=False,dry_run=False), state, target


def evaluate(command, log):
    args = runner.parse_args(command[2:])
    # Exercise actual scratch-manifest reconstruction and restricted migration.
    assert runner.identity(args) == read_json(args.run_dir/"run.json")
    assert args.checkpoint != checkpoint_path(args.run_dir, 60000)
    assert (args.output/"complete.json").exists() is False
    archive = next(args.output.parent.glob("step000060000.superseded_*"))
    rows = read_json(archive/"nfcgs_main/scene_metrics.json")
    atomic_json(args.output/"nfcgs_main/scene_metrics.json", rows)
    atomic_json(args.output/"complete.json", dict(split="validation",mode="validation",scene_count=128,
        checkpoint_sha256=file_hash(args.checkpoint),
        scene_metrics_sha256=file_hash(args.output/"nfcgs_main/scene_metrics.json")))
    return True


def test_late_lambda0_resave_gets_real_evaluation_then_new_gate_without_touching_learners(tmp_path):
    args,state,target = setup(tmp_path)
    state_bytes = args.state.read_bytes()
    source = target/"run.json"
    original_manifest = source.read_bytes()
    checkpoint = checkpoint_path(target,60000)
    sha = file_hash(checkpoint)
    old_complete = read_json(target/"validation/step000060000/complete.json")
    with patch.object(monitor,"evaluate",side_effect=evaluate) as call, patch.object(submit.subprocess,"run") as scheduler:
        repair.repair(args)
    call.assert_called_once()
    scheduler.assert_not_called()
    assert args.state.read_bytes() == state_bytes and source.read_bytes() == original_manifest
    assert file_hash(checkpoint) == sha
    archive = next((target/"validation").glob("*.superseded_*"))
    assert read_json(archive/"complete.json") == old_complete
    assert read_json(target/"validation/step000060000/complete.json")["checkpoint_sha256"] == sha
    assert len(read_json(args.root/"convergence/curve_wave1_A.json")) == 90
    assert read_json(args.root/"convergence/wave1_A_complete.json")["run_count"] == 15
    assert read_json(args.root/"convergence/gate1.json")["test_used"] is False
    assert repair.plan_repair(args)["repairs"] == []
    # Same production 120k preflight must now accept all 15 parents.
    a = copy.copy(args);a.accept_resaved_60k=True
    assert len(submit.stage_a_branch_sources(submit.Submission(a,state),a)) == 15


def test_dry_run_reports_lambda0_only_without_any_writes_or_submission(tmp_path,capsys):
    args,_,target = setup(tmp_path)
    before = {p:p.read_bytes() for p in (args.state,args.root/"convergence/gate1.json",target/"validation/step000060000/complete.json")}
    with patch.object(submit.subprocess,"run") as call:
        repair.main(["--state",str(args.state),"--root",str(args.root),"--dry-run"])
    call.assert_not_called()
    assert "reevaluate=1" in capsys.readouterr().out
    assert all(p.read_bytes()==body for p,body in before.items())
    assert not (args.root/"convergence/holdout_repairs").exists()


@pytest.mark.parametrize("failure",["wrong_processed","changed_model","damaged_metrics","wrong_old_sha"])
def test_preflight_rejects_corruption_before_archive_or_invalidation(tmp_path,failure):
    args,_,target = setup(tmp_path)
    if failure == "wrong_processed":
        p=checkpoint_path(target,60000);saved=torch.load(p,weights_only=False)
        saved["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["processed"]+=1;torch.save(saved,p)
    elif failure == "changed_model":
        p=target/"run.json";protocol=read_json(p);protocol["code_sha256"]["globalsplat/compression/codec.py"]="other-model"
        atomic_json(p,protocol);p=checkpoint_path(target,60000);saved=torch.load(p,weights_only=False)
        saved["fresh_rd_protocol"]=protocol;torch.save(saved,p)
    elif failure == "damaged_metrics":
        (target/"validation/step000060000/nfcgs_main/scene_metrics.json").write_text("[]")
    else:
        p=target/"validation/step000060000/complete.json";done=read_json(p);done["checkpoint_sha256"]="unknown"
        atomic_json(p,done)
    gate = (args.root/"convergence/gate1.json").read_bytes()
    with patch.object(monitor,"evaluate") as call:
        with pytest.raises(ValueError):repair.repair(args)
    call.assert_not_called()
    assert (args.root/"convergence/gate1.json").read_bytes() == gate
    assert not list((target/"validation").glob("*.superseded_*"))


def test_evaluation_failure_can_resume_but_metric_difference_keeps_gate_invalid(tmp_path):
    args,_,target = setup(tmp_path)
    with patch.object(monitor,"evaluate",return_value=False):
        with pytest.raises(RuntimeError):repair.repair(args)
    assert not (args.root/"convergence/gate1.json").exists()
    assert len(repair.plan_repair(args)["repairs"]) == 1
    def different(command,log):
        evaluate(command,log)
        output=runner.parse_args(command[2:]).output
        rows=read_json(output/"nfcgs_main/scene_metrics.json");rows[0]["mse"]+=.001
        atomic_json(output/"nfcgs_main/scene_metrics.json",rows)
        done=read_json(output/"complete.json");done["scene_metrics_sha256"]=file_hash(output/"nfcgs_main/scene_metrics.json")
        atomic_json(output/"complete.json",done)
        return True
    with patch.object(monitor,"evaluate",side_effect=different):
        with pytest.raises(ValueError,match="differs"):repair.repair(args)
    assert not (args.root/"convergence/gate1.json").exists()
    with pytest.raises(ValueError,match="differs"):repair.plan_repair(args)


def test_submit_uses_one_gpu_bash_and_keeps_sweep_state_intact(tmp_path,capsys):
    args,_,_=setup(tmp_path);plan=repair.plan_repair(args);before=args.state.read_bytes()
    with patch.object(repair.subprocess,"run",return_value=subprocess.CompletedProcess([],0,"12345\n","")) as call:
        repair.submit_repair(args,plan)
    command=call.call_args.args[0]
    assert "--gres=gpu:normal:1" in command and "--exclude=ariel-v7" in command
    assert command[-1].startswith("exec bash -lc ") and not any("dependency" in v for v in command)
    assert call.call_args.kwargs["env"]["DATASET_ROOT"] == plan["dataset"]
    assert "HOLDOUT_REPAIR_JOB=12345" in capsys.readouterr().out
    assert args.state.read_bytes()==before


def test_login_node_cannot_start_gpu_evaluation_directly(tmp_path):
    with patch.dict(os.environ,{},clear=True):
        with pytest.raises(SystemExit):repair.main(["--state",str(tmp_path/"missing.json")])

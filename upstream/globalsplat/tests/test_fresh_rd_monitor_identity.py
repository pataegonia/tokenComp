"""Checkpoint replacement requires new evaluation; original metrics stay intact."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import monitor_nfcgs_fresh_rd as monitor
import recover_nfcgs_monitor_identity_20261023 as repair
import submit_nfcgs_fresh_rd as submit
from globalsplat.fresh_rd import VERSION,atomic_json,checkpoint_path,file_hash,read_json,run_key,wave_runs


def evaluation(output,checkpoint):
    rows=[dict(scene=str(i),mse=.01,psnr=20.,ssim=.7,lpips=.2,actual_bytes=1000,
        actual_score_bytes=200,actual_residual_y_bytes=600,actual_residual_z_bytes=100,
        actual_mean_bytes=0,actual_container_bytes=100,context_frame_ids=list(range(12)),
        target_frame_ids=list(range(12,20))) for i in range(128)]
    path=output/"nfcgs_main/scene_metrics.json";atomic_json(path,rows)
    atomic_json(output/"complete.json",dict(split="validation",mode="validation",scene_count=128,
        checkpoint_sha256=file_hash(checkpoint),scene_metrics_sha256=file_hash(path)))


def test_replaced_checkpoint_requires_reevaluation_and_preserves_old_metrics(tmp_path):
    checkpoint=tmp_path/"step000060000.ckpt";checkpoint.write_bytes(b"original")
    output=tmp_path/"validation/step000060000";evaluation(output,checkpoint)
    digest=file_hash(output/"nfcgs_main/scene_metrics.json")
    checkpoint.write_bytes(b"rewritten at resume")
    with pytest.raises(monitor.CheckpointIdentityChanged):monitor.completed_rows(output,checkpoint)
    assert monitor.supersede_evaluation(output,checkpoint)
    archive=list(output.parent.glob("step*.superseded_*"))[0]
    assert not output.exists() and file_hash(archive/"nfcgs_main/scene_metrics.json")==digest
    assert read_json(archive/"superseded.json")["evaluated_sha256"]!=file_hash(checkpoint)
    evaluation(output,checkpoint)
    assert len(monitor.completed_rows(output,checkpoint))==128


def test_other_corruption_is_not_reclassified_as_checkpoint_replacement(tmp_path):
    checkpoint=tmp_path/"model.ckpt";checkpoint.write_bytes(b"original")
    output=tmp_path/"validation/step000060000";evaluation(output,checkpoint)
    (output/"nfcgs_main/scene_metrics.json").write_text("modified")
    with pytest.raises(ValueError) as error:monitor.completed_rows(output,checkpoint)
    assert not isinstance(error.value,monitor.CheckpointIdentityChanged)
    assert output.exists()


def test_active_output_is_not_moved_and_unexpected_paths_are_refused(tmp_path):
    checkpoint=tmp_path/"model.ckpt";checkpoint.write_bytes(b"original")
    output=tmp_path/"validation/step000060000";evaluation(output,checkpoint)
    (output/".lock").write_text("pid=1")
    checkpoint.write_bytes(b"changed")
    assert not monitor.supersede_evaluation(output,checkpoint)
    assert output.exists()
    with pytest.raises(ValueError,match="unexpected"):
        monitor.supersede_evaluation(tmp_path/"other_directory",checkpoint)


def test_gate1_is_invalidated_when_its_frozen_input_is_missing(tmp_path):
    out=tmp_path/"convergence";atomic_json(out/"gate1.json",dict(test_used=False))
    monitor.refresh_gate1(out,[dict(run_dir=str(tmp_path/"run"))])
    assert not (out/"gate1.json").exists()
    assert len(list(out.glob("gate1.superseded_*.json")))==1


def repair_fixture(tmp_path):
    args=argparse.Namespace(root=tmp_path/"outputs/fresh",state=tmp_path/"state.json",log_root=tmp_path/"logs",dry_run=False)
    args.log_root.mkdir()
    state=dict(version=VERSION,inputs=dict(code={n:file_hash(ROOT/n) for n in submit.FILES}),runs={},jobs={},lanes={})
    state["inputs"]["code"][repair.MONITOR]="previous monitor"
    for lane,(cell,rate,seed) in enumerate(wave_runs(1)):
        key=run_key(cell,rate,seed)
        state["runs"][key]=dict(key=key,cell=cell,rate_lambda=rate,seed_index=seed,wave=1,
            run_dir=str(args.root/key/f"job_{5000+lane}"))
        state["jobs"][key+"/A"]=dict(id=str(5000+lane))
        state["jobs"][key+"/B"]=dict(id=str(6000+lane))
        state["lanes"][str(lane)]=str(6000+lane)
    state["jobs"]["monitor/1/A"]=dict(id="7000",run=None,history=[],lane=15,
        command=["sbatch",submit.SLURM+"monitor_nfcgs_fresh_rd.slurm"],
        env=dict(REPO_DIR=str(ROOT),FRESH_STATE=str(args.state),FRESH_OUTPUT_ROOT=str(args.root),MONITOR_PHASE="A",MONITOR_WAVE="1"))
    state["lanes"]["15"]="7000"
    (args.log_root/"slurm-gs-fresh-monitor-7000.err").write_text("ValueError: checkpoint identity changed after evaluation")
    return args,state


def test_recovery_keeps_all_30_learner_records_and_does_not_cancel_them(tmp_path):
    args,state=repair_fixture(tmp_path);before=copy.deepcopy(state)
    with patch.object(submit,"job_state",return_value="FAILED"),patch.object(repair,"active_monitor_jobs",return_value=set()), \
            patch.object(submit.subprocess,"run",return_value=subprocess.CompletedProcess([],0,"8000\n","")) as call:
        repair.recover(args,state)
    call.assert_called_once()
    command=call.call_args.args[0]
    assert command[0]=="sbatch" and command[-1]==submit.SLURM+"monitor_nfcgs_fresh_rd.slurm"
    assert not any(x.startswith("--dependency") for x in command)
    assert state["runs"]==before["runs"]
    assert {k:v for k,v in state["jobs"].items() if k!="monitor/1/A"}=={k:v for k,v in before["jobs"].items() if k!="monitor/1/A"}
    assert state["jobs"]["monitor/1/A"]["id"]=="8000"
    assert state["inputs"]["code"][repair.MONITOR]==file_hash(ROOT/repair.MONITOR)


def test_audit_finds_original_backup_without_restoring_or_changing_current_file(tmp_path):
    args,state=repair_fixture(tmp_path)
    run=next(iter(state["runs"].values()));folder=Path(run["run_dir"])
    checkpoint=checkpoint_path(folder,60000);checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"original Stage A")
    digest=file_hash(checkpoint)
    backup=checkpoint.with_name("step000060000-v1.ckpt");backup.write_bytes(checkpoint.read_bytes())
    atomic_json(folder/"stage_A_complete.json",dict(step=60000,checkpoint_sha256=digest))
    checkpoint.write_bytes(b"reserialized Stage B boundary")
    current=checkpoint.read_bytes()
    result=repair.audit_checkpoints(state,args.root,tmp_path/"audit.json")
    row=next(r for r in result["runs"] if r["key"]==run["key"])
    assert row["exact_stage_a_file_preserved"] is False and row["matching_backups"]==[str(backup)]
    assert checkpoint.read_bytes()==current

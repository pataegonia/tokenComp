"""Checkpoint replacement requires new evaluation; original metrics stay intact."""
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import monitor_nfcgs_fresh_rd as monitor
from globalsplat.fresh_rd import atomic_json,file_hash,read_json


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

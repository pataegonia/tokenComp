"""Audit completed Score736 validation and the remaining D4 OFF16 resume."""
from collections import Counter
import copy
import importlib.util
import json
import math
from pathlib import Path
import re

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[2]
WORKSPACE=REPO.parents[1]
LOGS=WORKSPACE/"slurm"
PRIOR=REPO/"docs/analysis/nfcgs_logs_2026-10-20_1940"
spec=importlib.util.spec_from_file_location("audit1940",PRIOR/"audit_snapshot.py")
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
audit,base,digest=prior.audit,prior.base,prior.digest


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def completed_architecture(row):
    # The downloaded TRAIN stdout/err can be older than the newly downloaded eval.
    train=audit.read(LOGS/f"slurm-gs-arch-train-{row['train_job']}.out")
    err=audit.read(LOGS/f"slurm-gs-arch-train-{row['train_job']}.err")
    path=LOGS/f"slurm-gs-arch-eval-{row['eval_job']}.out"
    text=audit.read(path);eval_err=audit.read(path.with_suffix(".err"))
    assert not any(audit.FATAL.search(line) for line in (train+err+text+eval_err).splitlines())
    cfg=audit.command_config(text)
    assert not audit.check_overrides(cfg,{"loss.rate_lambda":row["lambda"],"model.feature_codec.codec_mode":"score_only",
        "model.feature_codec.rank":736,"model.feature_codec.use_residual":"false","model.feature_codec.token_order":"hilbert",
        "model.feature_codec.score_context_schedule":"dyadic4","model.feature_codec.score_spatial_stages":4,
        "model.feature_codec.score_slice_channels":184,"model.feature_codec.score_spatial_context":"true",
        "architecture_entropy_determinism":"true","dataset.eval_all_scenes":"false","test.actual_bitstream":"true",
        "seed":0,"trainer.precision":"bf16-mixed","trainer.devices":1,"optimizer.batch_size":1})
    assert cfg["checkpointing.load"].endswith(f"/job_{row['train_job']}/checkpoints/nfcgs_main/version_0/step000050000.ckpt")
    for name in ("dataset.eval_index_path","dataset.eval_train_holdout_manifest"):
        assert "initialization_20261019/" in cfg[name]
    metrics={k:float(v) for k,v in re.findall(r"(?m)^("+"|".join(audit.METRICS)+r"): ([-+\deE.]+)$",text)}
    assert set(metrics)==set(audit.METRICS) and all(math.isfinite(v) for v in metrics.values())
    progress=re.findall(r"Testing DataLoader 0:.*? (\d+)/(\d+) \[",text)
    assert tuple(map(int,progress[-1]))==(128,128)
    error=metrics["actual_bytes"]-sum(metrics[k] for k in ("actual_score_bytes","actual_residual_y_bytes",
        "actual_residual_z_bytes","actual_mean_bytes","actual_container_bytes"))
    assert abs(error)<1e-5
    assert metrics["actual_residual_y_bytes"]==metrics["actual_residual_z_bytes"]==metrics["actual_mean_bytes"]==0
    # This runner validates actual saved global_step/config/provenance before launching
    # the process that produced the successful 128-scene evaluation, not just a filename.
    runner=(REPO/"scripts/run_nfcgs_architecture_pilot.py").read_text(encoding="utf-8")
    assert 'elif saved.get("global_step")!=50000:' in runner
    row.update(train_status="complete_50k",eval_status="complete_holdout128",metrics=metrics,
        evaluated_scenes=128,byte_accounting_error=error,checkpoint=cfg["checkpointing.load"],
        eval_source_log=path.relative_to(WORKSPACE).as_posix(),eval_source_sha256=digest(path),
        declared_codec_overrides={k.removeprefix("model.feature_codec."):v for k,v in cfg.items() if k.startswith("model.feature_codec.")},
        train_completion_footer_downloaded="`max_steps=50000` reached" in err,
        last_downloaded_train_progress=base.last_progress(train),estimated_optimizer_step=50000,
        train_completion_evidence="Successful final50k validation via unchanged runner's saved-step/config/provenance guard; TRAIN log itself is older.",
        checkpoint_evidence="Evaluation runner checks loaded global_step50000 and pilot config/provenance; raw checkpoint not downloaded.",
        fatal_matches=[])
    return row


def main():
    before=json.loads((PRIOR/"review.json").read_text(encoding="utf-8"))
    inventory=json.loads((PRIOR/"inventory.json").read_text(encoding="utf-8"))
    changed=[]
    for name,info in before["sources"].items():
        path=WORKSPACE/name;sha=digest(path)
        audit.SOURCES[name]=dict(bytes=path.stat().st_size,sha256=sha)
        if sha!=info["sha256"]:changed.append(name)
    for point in inventory["results"]+inventory["lambda_zero_points"]+inventory["seed_repeat_points"]:
        assert digest(WORKSPACE/point["source_log"])==point["source_sha256"]
    hash_checks={}
    for folder in ("architecture_pilots_20261019","context_seed_pair_20261019","paper_controls_recovery_20261020"):
        state=json.loads(audit.read(LOGS/folder/"state.json"))
        mismatch=[name for name,sha in state["inputs"]["code_sha256"].items() if digest(REPO/name)!=sha]
        assert not mismatch
        hash_checks[folder]=dict(files=len(state["inputs"]["code_sha256"]),mismatches=mismatch)
    recovery=json.loads(audit.read(LOGS/"paper_controls_recovery_20261020/state.json"))
    rows=[]
    for previous in before["experiments"]:
        row=copy.deepcopy(previous)
        if row["eval_status"]=="complete_holdout128":
            assert digest(WORKSPACE/row["eval_source_log"])==row["eval_source_sha256"]
        if row["train_job"]=="457828":row=completed_architecture(row)
        if row.get("resume_job")=="458783":row=prior.evaluation(prior.resumed(row,recovery))
        if row.get("progress"):
            row.update(progress_epoch=row["progress"]["epoch"],progress_epoch_batch=row["progress"]["batch"],
                       progress_nominal_epoch_batches=row["progress"]["epoch_batches"])
        row.update(effective_estimated_step=row["estimated_optimizer_step"],
                   estimated_training_percent=row["estimated_optimizer_step"]/500,
                   remaining_training_hours_at_logged_speed=0.0 if row["train_status"]=="complete_50k" else None)
        rows.append(row)
    arms={r["arm"]:r for r in rows if r["group"]=="architecture"}
    assert all(r["eval_status"]=="complete_holdout128" for r in arms.values())
    two,score,full,msh=(arms[k]["metrics"] for k in ("two_path56","score56_only","score736_only","msh_only"))
    comparison=dict(two_path_minus_score736={k:two[k]-full[k] for k in ("psnr","ssim","lpips")},
        two_path_byte_reduction_vs_score736_pct=100*(1-two["actual_bytes"]/full["actual_bytes"]),
        score736_minus_score56={k:full[k]-score[k] for k in ("psnr","ssim","lpips")},
        score736_byte_increase_vs_score56_pct=100*(full["actual_bytes"]/score["actual_bytes"]-1),
        score736_bytes_div_msh=full["actual_bytes"]/msh["actual_bytes"],
        scope="Same pilot holdout128, lambda.0128, fresh common initialization, one seed; point dominance, not architecture-wide RD dominance.")
    result=dict(before,snapshot="2026-10-20 21:59 Asia/Seoul downloaded logs",experiments=rows,
        training_status_counts=dict(Counter(r["train_status"] for r in rows)),
        full_test_complete=sum(r["eval_status"]=="complete_fulltest" for r in rows),
        holdout128_complete=sum(r["eval_status"]=="complete_holdout128" for r in rows),
        architecture_comparison=comparison,runtime_hash_checks=hash_checks,sources=audit.SOURCES,
        changed_previous_sources=changed,reverified_source_names=sorted(audit.SOURCES),
        audit_scope="All previous recorded source files rehashed; previous completed70 evals unchanged; new Score736 holdout checked; remaining D4 OFF16 resume reread.",
        newly_complete_since_1940=dict(train="457828",eval="457829",scope="holdout128"),
        new_followup_state_downloaded=(LOGS/"followup_20261020/state.json").exists(),
        new_followup_logs=[p.name for p in sorted(LOGS.glob("slurm-gs-followup20-*"))])
    inventory.update(snapshot=result["snapshot"],architecture_pilot_points=[dict(arm=r["arm"],train_job=r["train_job"],
        eval_job=r["eval_job"],metrics=r["metrics"],scenes=128,source_log=r["eval_source_log"],source_sha256=r["eval_source_sha256"])
        for r in arms.values()])
    assert result["training_status_counts"]=={"complete_50k":25,"resumed_progress_logged":1}
    assert result["full_test_complete"]==21 and result["holdout128_complete"]==4
    save("review.json",result);save("inventory.json",inventory);save("architecture_comparison.json",comparison)
    print(json.dumps(dict(counts=result["training_status_counts"],fulltest=result["full_test_complete"],
        holdout=result["holdout128_complete"],changes=changed,comparison=comparison,
        score736_kib=full["actual_bytes"]/1024,remaining=[dict(train=r.get("resume_job") or r["train_job"],
            step=r["estimated_optimizer_step"]) for r in rows if r["train_status"]!="complete_50k"])))


if __name__=="__main__":main()

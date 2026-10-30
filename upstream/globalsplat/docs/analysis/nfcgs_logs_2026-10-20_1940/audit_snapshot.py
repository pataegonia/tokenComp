"""Audit the 19:40 download, including the final L3 factorial RD endpoint."""
from collections import Counter
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.interpolate import PchipInterpolator

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[2]
WORKSPACE=REPO.parents[1]
LOGS=WORKSPACE/"slurm"
PRIOR=REPO/"docs/analysis/nfcgs_logs_2026-10-20_1647"
spec=importlib.util.spec_from_file_location("prior1647",PRIOR/"audit_snapshot.py")
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
audit,base=prior.audit,prior.base


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):value.update(block)
    return value.hexdigest()


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def resumed(row,state):
    saved=next(v for v in state["points"].values() if v["original_train"]==row["train_job"])
    job=saved["train_job"]
    text=audit.read(LOGS/f"slurm-gs-paper-control-resume-{job}.out")
    err=audit.read(LOGS/f"slurm-gs-paper-control-resume-{job}.err")
    cfg=audit.command_config(text)
    assert cfg["checkpointing.resume"]=="true" and cfg["trainer.max_steps"]=="50000"
    assert cfg["output_dir"]==saved["run_dir"]+"/checkpoints"
    assert cfg["checkpointing.load"].endswith(f"/resume_job_{job}.ckpt")
    assert f"RESTORED_STEP={saved['step']} TARGET_STEP=50000" in text
    assert "Restored all states from the checkpoint" in err
    assert saved["dependency_bound"] and saved["dependency"]==f"afterok:{job}"
    fatal=[line for line in (text+err).splitlines() if audit.FATAL.search(line) or "Killed" in line]
    assert not fatal,fatal
    done="`max_steps=50000` reached" in err
    progress=base.last_progress(text)
    row.update(resume_job=job,restored_step=saved["step"],progress=progress,
        train_status="complete_50k" if done else "resumed_progress_logged",fatal_matches=fatal,
        estimated_optimizer_step=50000 if done else progress["estimated_step"],
        train_completion_evidence="max_steps=50000 footer" if done else "last complete progress line")
    return row


def evaluation(row):
    path=LOGS/f"slurm-gs-order-context-eval-{row['eval_job']}.out"
    text=audit.read(path);err=audit.read(path.with_suffix(".err"))
    assert not any(audit.FATAL.search(line) for line in (text+err).splitlines())
    metrics={k:float(v) for k,v in re.findall(r"(?m)^("+"|".join(audit.METRICS)+r"): ([-+\deE.]+)$",text)}
    if len(metrics)!=len(audit.METRICS):
        row.update(eval_status="log_without_final_metrics" if text else "submitted_no_start_log",metrics=metrics)
        return row
    assert row["train_status"]=="complete_50k" and all(math.isfinite(v) for v in metrics.values())
    headers=re.findall(r"checkpoint codec=(\{[^\n]+\}) global_step=(\d+)",text)
    assert headers and all(int(step)==50000 for _,step in headers)
    declared=ast.literal_eval(headers[-1][0]);cfg=audit.command_config(text)
    expected={"token_order":row["order"],"score_context_schedule":row["schedule"],
        "score_spatial_stages":row["stages"],"score_slice_channels":row["slice_channels"],
        "score_spatial_context":row["spatial_context"],"use_residual":True,"rank":56,
        "transform":"linear","use_centering":False,"use_score_norm":False,
        "score_channel_context":False,"score_mean_condition":False}
    assert all(declared[k]==v for k,v in expected.items())
    assert not audit.check_overrides(cfg,{"loss.rate_lambda":row["lambda"],"dataset":"re10k_eval_all_ctx12",
        "test.actual_bitstream":"true","test.max_scenes":"null","seed":"0", "optimizer.batch_size":"1",
        "trainer.devices":"1","trainer.precision":"bf16-mixed"})
    assert f"/job_{row['train_job']}/" in cfg["checkpointing.load"]
    assert cfg["checkpointing.load"].endswith("/step000050000.ckpt")
    progress=re.findall(r"Testing DataLoader 0:.*? (\d+)/(\d+) \[",text)
    assert tuple(map(int,progress[-1]))==(6991,7286)
    error=metrics["actual_bytes"]-sum(metrics[k] for k in ("actual_score_bytes","actual_residual_y_bytes",
        "actual_residual_z_bytes","actual_mean_bytes","actual_container_bytes"))
    assert abs(error)<1e-5
    row.update(eval_status="complete_fulltest",metrics=metrics,evaluated_scenes=6991,
        nominal_dataloader_scene_count=7286,declared_codec_config=declared,checkpoint=cfg["checkpointing.load"],
        byte_accounting_error=error,eval_source_log=path.relative_to(WORKSPACE).as_posix(),
        eval_source_sha256=digest(path),eval_log_downloaded=True)
    row.pop("last_eval_progress",None)
    return row


def l3_comparison(points):
    families=("both_off","context_only","hilbert_only","hilbert_legacy3")
    curves={name:sorted([r for r in points if r["family"]==name],key=lambda r:r["rate_lambda"]) for name in families}
    assert all([r["rate_lambda"] for r in rows]==[.0032,.0064,.0128,.0256,.0512] for rows in curves.values())
    lo=max(min(r["metrics"]["psnr"] for r in rows) for rows in curves.values())
    hi=min(max(r["metrics"]["psnr"] for r in rows) for rows in curves.values())
    assert hi>lo
    fitted={}
    for name,rows in curves.items():
        order=sorted(rows,key=lambda r:r["metrics"]["psnr"])
        quality=np.array([r["metrics"]["psnr"] for r in order])
        rate=np.array([r["metrics"]["actual_bytes"] for r in order])
        assert np.all(np.diff(quality)>0) and np.all(np.diff(rate)>0)
        fitted[name]=float(PchipInterpolator(quality,np.log(rate),extrapolate=False).integrate(lo,hi)/(hi-lo))
    pairs={"context_without_sorting":("context_only","both_off"),
           "context_after_hilbert":("hilbert_legacy3","hilbert_only"),
           "sorting_without_context":("hilbert_only","both_off"),
           "sorting_with_context":("hilbert_legacy3","context_only"),
           "combined":("hilbert_legacy3","both_off")}
    bd={name:100*math.expm1(fitted[candidate]-fitted[baseline]) for name,(candidate,baseline) in pairs.items()}
    same_lambda=[]
    for i,rate in enumerate((.0032,.0064,.0128,.0256,.0512)):
        values={name:curves[name][i]["metrics"] for name in families}
        interaction=math.log(values["hilbert_legacy3"]["actual_bytes"])-math.log(values["hilbert_only"]["actual_bytes"])-math.log(values["context_only"]["actual_bytes"])+math.log(values["both_off"]["actual_bytes"])
        same_lambda.append(dict(rate_lambda=rate,log_rate_interaction=interaction,
            context_none_byte_reduction_pct=100*(1-values["context_only"]["actual_bytes"]/values["both_off"]["actual_bytes"]),
            context_hilbert_byte_reduction_pct=100*(1-values["hilbert_legacy3"]["actual_bytes"]/values["hilbert_only"]["actual_bytes"]),
            context_none_psnr_delta=values["context_only"]["psnr"]-values["both_off"]["psnr"],
            context_hilbert_psnr_delta=values["hilbert_legacy3"]["psnr"]-values["hilbert_only"]["psnr"]))
    return dict(families=families,points=curves,common_psnr_interval=[lo,hi],pchip_bd_rate_pct=bd,
        integrated_log_rate_interaction=fitted["hilbert_legacy3"]-fitted["hilbert_only"]-fitted["context_only"]+fitted["both_off"],
        same_lambda=same_lambda,
        scope="Aggregate rounded log metrics, one primary seed, common four-curve PSNR band, PCHIP log-rate, no extrapolation; no per-scene/frame or seed CI.",
        recipe_exception="Historical Hilbert+L3 lambda.0256 used GPU4; retain this known recipe exception when interpreting curves.")


def main():
    before=json.loads((PRIOR/"review.json").read_text(encoding="utf-8"))
    inventory=json.loads((PRIOR/"inventory.json").read_text(encoding="utf-8"))
    changed=[]
    for name,info in before["sources"].items():
        path=WORKSPACE/name;sha=digest(path)
        audit.SOURCES[name]=dict(bytes=path.stat().st_size,sha256=sha)
        if sha!=info["sha256"]:changed.append(name)
    completed=inventory["results"]+inventory["lambda_zero_points"]+inventory["seed_repeat_points"]
    for point in completed:
        assert digest(WORKSPACE/point["source_log"])==point["source_sha256"]
    for row in before["experiments"]:
        if row["eval_status"]=="complete_holdout128":
            assert digest(WORKSPACE/row["eval_source_log"])==row["eval_source_sha256"]
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
        if row["train_job"]=="457828":row=base.architecture(row)
        if row.get("resume_job"):row=evaluation(resumed(row,recovery))
        if row.get("progress"):
            row.update(progress_epoch=row["progress"]["epoch"],progress_epoch_batch=row["progress"]["batch"],
                progress_nominal_epoch_batches=row["progress"]["epoch_batches"])
        row["effective_estimated_step"]=row["estimated_optimizer_step"]
        row["estimated_training_percent"]=row["estimated_optimizer_step"]/500
        row["remaining_training_hours_at_logged_speed"]=0.0 if row["train_status"]=="complete_50k" else None
        rows.append(row)
    new=next(r for r in rows if r["train_job"]=="456951")
    assert new["train_status"]=="complete_50k" and new["eval_status"]=="complete_fulltest"
    inventory["results"].append(dict(label="both_off_0p0512",family="both_off",rate_lambda=.0512,
        source_log=new["eval_source_log"],source_sha256=new["eval_source_sha256"],evaluated_scenes=6991,
        metrics=new["metrics"],byte_accounting_error=new["byte_accounting_error"],
        declared_config=new["declared_codec_config"],checkpoint=new["checkpoint"]))
    assert len(inventory["results"])==64
    assert len({(r["family"],r["rate_lambda"]) for r in inventory["results"]})==64
    comparison=l3_comparison(inventory["results"])
    result=dict(before,snapshot="2026-10-20 19:40 Asia/Seoul downloaded logs",live_slurm_query=False,
        experiments=rows,training_status_counts=dict(Counter(r["train_status"] for r in rows)),
        full_test_complete=sum(r["eval_status"]=="complete_fulltest" for r in rows),
        holdout128_complete=sum(r["eval_status"]=="complete_holdout128" for r in rows),
        primary_rd_points=64,changed_previous_sources=changed,runtime_hash_checks=hash_checks,
        sources=audit.SOURCES,reverified_source_names=sorted(audit.SOURCES),
        audit_scope="All previous recorded source files rehashed, previous completed69 evaluations unchanged, two resumes and Score736 reread, new full-test endpoint checked.",
        l3_comparison=comparison,newly_complete=dict(train="458782",original_train="456951",eval="457098"),
        new_followup_state_downloaded=(LOGS/"followup_20261020/state.json").exists(),
        new_followup_logs=[p.name for p in sorted(LOGS.glob("slurm-gs-followup20-*"))])
    inventory["snapshot"]=result["snapshot"]
    save("review.json",result);save("inventory.json",inventory);save("l3_comparison.json",comparison)
    print(json.dumps(dict(counts=result["training_status_counts"],fulltest=result["full_test_complete"],
        holdout=result["holdout128_complete"],changed=changed,new_metrics=new["metrics"],
        remaining=[dict(train=r.get("resume_job") or r["train_job"],step=r["estimated_optimizer_step"])
                   for r in rows if r["train_status"]!="complete_50k"],bd_rate=comparison["pchip_bd_rate_pct"])))


if __name__=="__main__":main()

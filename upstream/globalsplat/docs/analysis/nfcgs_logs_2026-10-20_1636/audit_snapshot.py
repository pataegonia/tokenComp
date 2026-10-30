"""Audit the 16:36 download without modifying earlier snapshots or runtime code."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

import numpy as np

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[2]
WORKSPACE=REPO.parents[1]
LOGS=WORKSPACE/"slurm"
PRIOR=REPO/"docs/analysis/nfcgs_logs_2026-10-20_1051"
spec=importlib.util.spec_from_file_location("snapshot1051",PRIOR/"audit_snapshot.py")
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
audit=base.audit


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def interval(values):
    values=np.asarray(values,dtype=float)
    generator=np.random.default_rng(0)
    samples=values[generator.integers(0,len(values),(1000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())


def main():
    before=json.loads((PRIOR/"review.json").read_text(encoding="utf-8"))
    inventory=json.loads((PRIOR/"inventory.json").read_text(encoding="utf-8"))
    changed=[]
    for name,info in before["sources"].items():
        path=WORKSPACE/name
        with path.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
        audit.SOURCES[name]=dict(bytes=path.stat().st_size,sha256=digest)
        if digest!=info["sha256"]:changed.append(name)
    hash_checks={}
    for folder in ("architecture_pilots_20261019","context_seed_pair_20261019","paper_controls_recovery_20261020"):
        state=json.loads(audit.read(LOGS/folder/"state.json"))
        mismatch=[name for name,sha in state["inputs"]["code_sha256"].items()
                  if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=sha]
        assert not mismatch
        hash_checks[folder]=dict(files=len(state["inputs"]["code_sha256"]),mismatches=mismatch)
    state=json.loads(audit.read(LOGS/"paper_controls_recovery_20261020/state.json"))
    resumed={value["original_train"]:value for value in state["points"].values()}
    rows=[]
    for previous in before["experiments"]:
        row=base.architecture(previous) if previous["group"]=="architecture" else audit.inspect(dict(previous))
        if previous["train_job"] in resumed:
            saved=resumed[previous["train_job"]];job=saved["train_job"]
            text=audit.read(LOGS/f"slurm-gs-paper-control-resume-{job}.out")
            err=audit.read(LOGS/f"slurm-gs-paper-control-resume-{job}.err")
            command=audit.command_config(text)
            assert command["checkpointing.resume"]=="true" and command["trainer.max_steps"]=="50000"
            assert command["checkpointing.load"].endswith(f"/resume_job_{job}.ckpt")
            assert command["output_dir"]==saved["run_dir"]+"/checkpoints"
            assert f"RESTORED_STEP={saved['step']} TARGET_STEP=50000" in text
            assert "Restored all states from the checkpoint" in err
            assert saved["dependency_bound"] and saved["dependency"]==f"afterok:{job}"
            fatal=[line for line in (text+err).splitlines() if audit.FATAL.search(line) or "Killed" in line]
            assert not fatal
            done="`max_steps=50000` reached" in err
            progress=base.last_progress(text)
            row.update(train_status="complete_50k" if done else "resumed_progress_logged",resume_job=job,
                restored_step=saved["step"],progress=progress,estimated_optimizer_step=50000 if done else progress["estimated_step"],
                estimated_training_percent=100 if done else progress["estimated_step"]/500,
                eval_dependency_at_saved_state=saved["dependency"],fatal_matches=fatal)
        if row["eval_status"]=="log_without_final_metrics":
            stem="slurm-gs-seed-pair-eval" if row["group"]=="seed_pair" else "slurm-gs-order-context-eval"
            text=audit.read(LOGS/f"{stem}-{row['eval_job']}.out")
            progress=re.findall(r"Testing DataLoader 0:.*? (\d+)/(\d+) \[",text)
            row["last_eval_progress"]=dict(processed=int(progress[-1][0]),nominal=int(progress[-1][1])) if progress else None
        if row["group"]=="architecture" or row.get("resume_job"):
            row.pop("last_complete_progress_line",None)
            if row.get("progress"):
                row.update(progress_epoch=row["progress"]["epoch"],progress_epoch_batch=row["progress"]["batch"],
                           progress_nominal_epoch_batches=row["progress"]["epoch_batches"])
            row["effective_estimated_step"]=row["estimated_optimizer_step"]
        row["remaining_training_hours_at_logged_speed"]=0.0 if row["train_status"]=="complete_50k" else None
        rows.append(row)
    for point in inventory["results"]+inventory["lambda_zero_points"]:
        assert audit.SOURCES[point["source_log"]]["sha256"]==point["source_sha256"]
    seeds=[]
    for row in rows:
        if row["group"]=="seed_pair" and row["eval_status"]=="complete_fulltest":
            seeds.append(dict(arm=row["arm"],train_job=row["train_job"],eval_job=row["eval_job"],
                model_seed=111124,data_seed=404,rate_lambda=.0128,scenes=row["evaluated_scenes"],metrics=row["metrics"],
                declared_config=row["declared_codec_config"],checkpoint=row["checkpoint"],
                source_log=row["eval_source_log"],source_sha256=row["eval_source_sha256"],
                scope="separate seed repeat; not inserted into primary-seed RD curves"))
    path=LOGS/"slurm-gs-fixed-order-458818.out"
    text=audit.read(path);err=audit.read(path.with_suffix(".err"))
    assert "FIXED_ORDER_COMPLETE scenes=256 orders=13" in text
    assert not any(audit.FATAL.search(line) for line in err.splitlines())
    pairs=[dict(ordinal=int(m[1]),scene=m[2],hilbert_bytes=int(m[3]),slot_bytes=int(m[4]))
           for m in re.finditer(r"FIXED_ORDER_SCENE=(\d+)/256 scene=(\S+) hilbert_bytes=(\d+) slot_bytes=(\d+)",text)]
    assert len(pairs)==256 and len({row["scene"] for row in pairs})==256
    assert [row["ordinal"] for row in pairs]==list(range(1,257))
    original=json.loads(audit.read(LOGS/"nfcgs_legacy3_anchors_454995/inputs.json"))
    assert [row["scene"] for row in pairs]==original["scenes"]
    hilbert=np.array([row["hilbert_bytes"] for row in pairs]);slot=np.array([row["slot_bytes"] for row in pairs])
    t06=dict(job="458818",scene_count=256,orders=13,complete_footer=True,
        hilbert_mean_bytes=float(hilbert.mean()),slot_mean_bytes=float(slot.mean()),
        hilbert_mean_kib=float(hilbert.mean()/1024),slot_mean_kib=float(slot.mean()/1024),
        hilbert_byte_reduction_vs_slot_pct=float((1-hilbert.mean()/slot.mean())*100),
        hilbert_smaller_scenes=int((hilbert<slot).sum()),paired_hilbert_minus_slot_bytes=interval(hilbert-slot),
        scope="frozen Hilbert-trained model, non-Hilbert orders are distribution shifts; no matched-quality/retrained RD claim",
        unavailable="remote REPORT/summary/scene JSON: Morton/random/stage/anchor/residual/feature-error statistics not downloaded")
    architectures={row["arm"]:row for row in rows if row["group"]=="architecture" and row["eval_status"]=="complete_holdout128"}
    two=architectures["two_path56"]["metrics"];score=architectures["score56_only"]["metrics"];msh=architectures["msh_only"]["metrics"]
    comparison=dict(two_path_minus_score56_psnr=two["psnr"]-score["psnr"],
        two_path_byte_reduction_vs_score56_pct=(1-two["actual_bytes"]/score["actual_bytes"])*100,
        two_path_minus_score56_ssim=two["ssim"]-score["ssim"],two_path_minus_score56_lpips=two["lpips"]-score["lpips"],
        two_path_minus_msh_psnr=two["psnr"]-msh["psnr"],two_path_bytes_div_msh_bytes=two["actual_bytes"]/msh["actual_bytes"],
        scope="common pilot recipe/holdout128/lambda.0128; one seed and one point, not complete RD curves or statistical seed significance")
    result=dict(snapshot="2026-10-20 16:36 Asia/Seoul downloaded logs",live_slurm_query=False,
        training_status_counts=dict(Counter(row["train_status"] for row in rows)),
        full_test_complete=sum(row["eval_status"]=="complete_fulltest" for row in rows),
        holdout128_complete=sum(row["eval_status"]=="complete_holdout128" for row in rows),
        experiments=rows,architecture_comparison=comparison,t06=t06,seed_repeat_points=seeds,
        changed_previous_sources=changed,runtime_hash_checks=hash_checks,sources=audit.SOURCES)
    inventory.update(snapshot=result["snapshot"],seed_repeat_points=seeds,
        seed_repeat_scope="stored separately from primary-seed RD63 and lambda0")
    save("review.json",result);save("inventory.json",inventory);save("t06_stdout.json",pairs)
    print(json.dumps(dict(counts=result["training_status_counts"],fulltest=result["full_test_complete"],
        holdout=result["holdout128_complete"],t06=t06,architecture=comparison,sources=len(audit.SOURCES))))
    for row in rows:
        if row["group"] in ("architecture","seed_pair") or row.get("resume_job"):
            print(json.dumps({k:row.get(k) for k in ("train_job","eval_job","arm","train_status","eval_status",
                "resume_job","estimated_optimizer_step","last_eval_progress","metrics")}))


if __name__=="__main__":main()

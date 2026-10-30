"""Incremental audit: new seed ON evaluation and three unfinished training logs.

Completed evaluation logs are all rehashed. Other historical training records
are carried from the 16:36 audit, rather than parsing unchanged multi-GB logs.
"""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[2]
WORKSPACE=REPO.parents[1]
LOGS=WORKSPACE/"slurm"
PRIOR=REPO/"docs/analysis/nfcgs_logs_2026-10-20_1636"
spec=importlib.util.spec_from_file_location("audit1636",PRIOR/"audit_snapshot.py")
prior_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior_module)
base=prior_module.base
audit=prior_module.audit


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def main():
    previous=json.loads((PRIOR/"review.json").read_text(encoding="utf-8"))
    inventory=json.loads((PRIOR/"inventory.json").read_text(encoding="utf-8"))
    runtime={}
    for folder in ("architecture_pilots_20261019","context_seed_pair_20261019","paper_controls_recovery_20261020"):
        state=json.loads(audit.read(LOGS/folder/"state.json"))
        mismatches=[name for name,sha in state["inputs"]["code_sha256"].items()
                    if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=sha]
        assert not mismatches
        runtime[folder]=dict(files=len(state["inputs"]["code_sha256"]),mismatches=mismatches)
    completed=inventory["results"]+inventory["lambda_zero_points"]+inventory.get("seed_repeat_points",[])
    for point in completed:
        name=point["source_log"];path=WORKSPACE/name
        with path.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
        assert digest==point["source_sha256"]
        audit.SOURCES[name]=dict(bytes=path.stat().st_size,sha256=digest)
    rows=[]
    for original in previous["experiments"]:
        row=copy.deepcopy(original)
        if row["train_job"]=="457786":
            row=audit.inspect(row)
            assert row["train_status"]=="complete_50k" and row["eval_status"]=="complete_fulltest"
            row.pop("last_eval_progress",None)
        elif row["train_job"]=="457828":
            row=base.architecture(row)
            row.pop("last_complete_progress_line",None)
        elif row.get("resume_job"):
            job=row["resume_job"]
            text=audit.read(LOGS/f"slurm-gs-paper-control-resume-{job}.out")
            err=audit.read(LOGS/f"slurm-gs-paper-control-resume-{job}.err")
            command=audit.command_config(text)
            assert command["checkpointing.resume"]=="true" and command["trainer.max_steps"]=="50000"
            assert f"/job_{row['train_job']}/checkpoints" in command["output_dir"]
            assert "Restored all states from the checkpoint" in err
            fatal=[line for line in (text+err).splitlines() if audit.FATAL.search(line) or "Killed" in line]
            assert not fatal
            done="`max_steps=50000` reached" in err
            row.update(train_status="complete_50k" if done else "resumed_progress_logged",
                progress=base.last_progress(text),fatal_matches=fatal)
            row["estimated_optimizer_step"]=50000 if done else row["progress"]["estimated_step"]
        if row.get("progress"):
            row.update(progress_epoch=row["progress"]["epoch"],progress_epoch_batch=row["progress"]["batch"],
                       progress_nominal_epoch_batches=row["progress"]["epoch_batches"])
        row["effective_estimated_step"]=row["estimated_optimizer_step"]
        row["estimated_training_percent"]=row["estimated_optimizer_step"]/500
        row["remaining_training_hours_at_logged_speed"]=0.0 if row["train_status"]=="complete_50k" else None
        rows.append(row)
    seeds=[]
    for row in rows:
        if row["group"]=="seed_pair" and row["eval_status"]=="complete_fulltest":
            seeds.append(dict(arm=row["arm"],train_job=row["train_job"],eval_job=row["eval_job"],
                model_seed=111124,data_seed=404,rate_lambda=.0128,scenes=row["evaluated_scenes"],metrics=row["metrics"],
                declared_config=row["declared_codec_config"],checkpoint=row["checkpoint"],
                source_log=row["eval_source_log"],source_sha256=row["eval_source_sha256"],
                scope="separate seed repeat; not inserted into primary-seed RD curves"))
    arms={point["arm"]:point for point in seeds}
    assert set(arms)=={"on16","off16"}
    for point in arms.values():
        assert point["scenes"]==6991
        metrics=point["metrics"]
        assert abs(metrics["actual_bytes"]-sum(metrics[name] for name in ("actual_score_bytes", "actual_residual_y_bytes",
            "actual_residual_z_bytes","actual_mean_bytes","actual_container_bytes")))<1e-5
        config=point["declared_config"]
        assert config["score_context_schedule"]=="dyadic4" and config["score_spatial_stages"]==4 and config["score_slice_channels"]==16
        assert config["use_residual"] and config["token_order"]=="hilbert"
    assert arms["on16"]["declared_config"]["score_spatial_context"] is True
    assert arms["off16"]["declared_config"]["score_spatial_context"] is False
    left=arms["on16"]["declared_config"];right=arms["off16"]["declared_config"]
    assert [name for name in left if left[name]!=right[name]]==["score_spatial_context"]
    on=arms["on16"]["metrics"];off=arms["off16"]["metrics"]
    comparison=dict(model_seed=111124,data_seed=404,rate_lambda=.0128,scenes=6991,
        on_kib=on["actual_bytes"]/1024,off_kib=off["actual_bytes"]/1024,
        on_minus_off={key:on[key]-off[key] for key in on},
        on_byte_reduction_pct=100*(1-on["actual_bytes"]/off["actual_bytes"]),
        score_byte_reduction_pct=100*(1-on["actual_score_bytes"]/off["actual_score_bytes"]),
        raw_score_fraction_of_total_saved_pct=100*(off["actual_score_bytes"]-on["actual_score_bytes"])/(off["actual_bytes"]-on["actual_bytes"]),
        scope="same-lambda extra-seed full-test means; not equal-quality BD-rate or seed significance; base-seed OFF16 unfinished")
    sources=dict(previous["sources"]);sources.update(audit.SOURCES)
    changed=[name for name,info in audit.SOURCES.items() if previous["sources"].get(name,{}).get("sha256")!=info["sha256"]]
    result=dict(previous,snapshot="2026-10-20 16:47 Asia/Seoul downloaded logs",experiments=rows,
        training_status_counts=dict(Counter(row["train_status"] for row in rows)),
        full_test_complete=sum(row["eval_status"]=="complete_fulltest" for row in rows),
        holdout128_complete=sum(row["eval_status"]=="complete_holdout128" for row in rows),
        seed_repeat_points=seeds,seed_pair_comparison=comparison,runtime_hash_checks=runtime,
        changed_previous_sources=changed,reverified_source_names=sorted(audit.SOURCES),sources=sources,
        audit_scope="incremental: all completed evaluation logs rehashed, seed ON rerun through config/count checks, three unfinished training logs reread; other historical training observations carried from16:36")
    inventory.update(snapshot=result["snapshot"],seed_repeat_points=seeds)
    save("review.json",result);save("inventory.json",inventory);save("seed_pair_comparison.json",comparison)
    print(json.dumps(dict(comparison=comparison,counts=result["training_status_counts"],fulltest=result["full_test_complete"],
        holdout=result["holdout128_complete"],changed=changed,reverified_sources=len(audit.SOURCES))))


if __name__=="__main__":main()

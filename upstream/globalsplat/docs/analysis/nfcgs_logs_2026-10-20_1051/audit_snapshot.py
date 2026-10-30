"""Audit downloaded logs, including holdout evaluations and resumed controls."""
from collections import Counter
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
WORKSPACE = REPO.parents[1]
LOGS = WORKSPACE / "slurm"
PRIOR = REPO / "docs/analysis/nfcgs_logs_2026-10-20_0911"
spec = importlib.util.spec_from_file_location("prior_audit", REPO / "docs/analysis/nfcgs_logs_2026-10-19_2020/audit_snapshot.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def last_progress(text):
    matches = [m for line in text.splitlines() if (m := audit.PROGRESS.match(line))]
    if not matches:
        return None
    m = matches[-1]
    return dict(epoch=int(m[1]), batch=int(m[2]), epoch_batches=int(m[3]),
                estimated_step=int(m[1])*((int(m[3])+1)//2)+int(m[2])//2)


def architecture(row):
    train = audit.read(LOGS/f"slurm-gs-arch-train-{row['train_job']}.out")
    err = audit.read(LOGS/f"slurm-gs-arch-train-{row['train_job']}.err")
    evaluation_path = LOGS/f"slurm-gs-arch-eval-{row['eval_job']}.out"
    evaluation = audit.read(evaluation_path)
    eval_err = audit.read(evaluation_path.with_suffix(".err"))
    result = dict(row, train_status="complete_50k" if "`max_steps=50000` reached" in err else "progress_logged",
        progress=last_progress(train), eval_status="submitted_no_start_log" if not evaluation else "log_without_final_metrics",
        fatal_matches=[line for line in (train+err+evaluation+eval_err).splitlines() if audit.FATAL.search(line)])
    result["estimated_optimizer_step"] = 50000 if result["train_status"]=="complete_50k" else result["progress"]["estimated_step"]
    result["estimated_training_percent"] = result["estimated_optimizer_step"]/500
    command = audit.command_config(train)
    expected = {"model.feature_codec.codec_mode":row["codec_mode"], "model.feature_codec.rank":row["rank"],
        "model.feature_codec.use_residual":str(row["use_residual"]).lower(), "trainer.max_steps":50000,
        "trainer.devices":2, "trainer.accumulate_grad_batches":2, "optimizer.batch_size":2,
        "architecture_entropy_determinism":"true", "trainer.limit_val_batches":0,
        "checkpointing.resume":"false", "model.feature_codec.token_order":"hilbert"}
    assert not audit.check_overrides(command,expected)
    assert command["dataset.train_holdout_manifest"].endswith("/holdout.json")
    metrics = {k:float(v) for k,v in re.findall(r"(?m)^("+"|".join(audit.METRICS)+r"): ([-+\deE.]+)$",evaluation)}
    result["metrics"] = metrics
    if len(metrics) == len(audit.METRICS):
        assert result["train_status"] == "complete_50k"
        command = audit.command_config(evaluation)
        declared = {k.removeprefix("model.feature_codec."):v for k,v in command.items() if k.startswith("model.feature_codec.")}
        assert declared["codec_mode"]==row["codec_mode"] and int(declared["rank"])==row["rank"]
        assert declared["use_residual"]==str(row["use_residual"]).lower() and declared["token_order"]=="hilbert"
        for key in ("dataset.eval_train_holdout_manifest","dataset.eval_index_path"):
            assert "initialization_20261019/" in command[key]
        assert command["dataset.eval_all_scenes"]=="false" and command["test.actual_bitstream"]=="true"
        assert f"/job_{row['train_job']}/" in command["checkpointing.load"]
        assert command["checkpointing.load"].endswith("/step000050000.ckpt")
        progress = re.findall(r"Testing DataLoader 0:.*? (\d+)/(\d+) \[",evaluation)
        assert tuple(map(int,progress[-1]))==(128,128)
        error = metrics["actual_bytes"]-sum(metrics[k] for k in ("actual_score_bytes","actual_residual_y_bytes",
            "actual_residual_z_bytes","actual_mean_bytes","actual_container_bytes"))
        assert abs(error)<1e-5
        result.update(eval_status="complete_holdout128",evaluated_scenes=128,byte_accounting_error=error,
            declared_codec_overrides=declared, checkpoint=command["checkpointing.load"],
            checkpoint_evidence="unchanged pilot runner validates saved step50000/config before the printed command; raw checkpoint not downloaded",
            eval_source_log=evaluation_path.relative_to(WORKSPACE).as_posix(),
            eval_source_sha256=audit.SOURCES[evaluation_path.relative_to(WORKSPACE).as_posix()]["sha256"])
    assert not result["fatal_matches"]
    return result


def main():
    before = json.loads((PRIOR/"review.json").read_text(encoding="utf-8"))
    inventory = json.loads((PRIOR/"inventory.json").read_text(encoding="utf-8"))
    changed = []
    for name, info in before["sources"].items():
        path = WORKSPACE/name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream,"sha256").hexdigest()
        audit.SOURCES[name] = dict(bytes=path.stat().st_size,sha256=digest)
        if audit.SOURCES[name]["sha256"] != info["sha256"]:
            changed.append(name)
    hashes = {}
    for folder in ("architecture_pilots_20261019","context_seed_pair_20261019","paper_controls_recovery_20261020"):
        state = json.loads(audit.read(LOGS/folder/"state.json"))
        mismatches = [name for name,sha in state["inputs"]["code_sha256"].items()
                      if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=sha]
        assert not mismatches
        hashes[folder]=dict(runtime_files=len(state["inputs"]["code_sha256"]),mismatches=mismatches)
    recovery_state = json.loads(audit.read(LOGS/"paper_controls_recovery_20261020/state.json"))
    resumed = {r["original_train"]:r for r in recovery_state["points"].values()}
    rows = []
    for original in before["experiments"]:
        row = architecture(original) if original["group"]=="architecture" else audit.inspect(dict(original))
        if original["train_job"] in resumed:
            saved = resumed[original["train_job"]]
            job = saved["train_job"]
            text = audit.read(LOGS/f"slurm-gs-paper-control-resume-{job}.out")
            err = audit.read(LOGS/f"slurm-gs-paper-control-resume-{job}.err")
            command = audit.command_config(text)
            assert command["checkpointing.resume"]=="true" and command["trainer.max_steps"]=="50000"
            assert command["checkpointing.load"].endswith(f"/resume_job_{job}.ckpt")
            assert command["output_dir"]==saved["run_dir"]+"/checkpoints"
            assert f"RESTORED_STEP={saved['step']} TARGET_STEP=50000" in text
            assert "Restored all states from the checkpoint" in err
            assert saved["dependency_bound"] and saved["dependency"]==f"afterok:{job}"
            fatal = [line for line in (text+err).splitlines() if audit.FATAL.search(line)]
            assert not fatal and "Killed" not in err
            row.update(train_status="resumed_progress_logged", original_train_status="failed_signal9",
                original_failed_last_logged_step=row["estimated_optimizer_step"],
                resume_job=job, restored_step=saved["step"], progress=last_progress(text),
                effective_estimated_step=last_progress(text)["estimated_step"],
                estimated_optimizer_step=last_progress(text)["estimated_step"],
                estimated_training_percent=last_progress(text)["estimated_step"]/500,
                eval_dependency_at_saved_state=saved["dependency"], fatal_matches=fatal,
                resume_scope="optimizer/scheduler/loop state restored; fresh-shuffle loader, not exact next-sample order")
        rows.append(row)
    for point in inventory["results"]+inventory["lambda_zero_points"]:
        name = point["source_log"]
        assert audit.SOURCES[name]["sha256"]==point["source_sha256"]
    out = audit.read(LOGS/"slurm-gs-independent-receiver-458784.out")
    err = audit.read(LOGS/"slurm-gs-independent-receiver-458784.err")
    rx = re.compile(r"INDEPENDENT_RECEIVER_SCENE=(\d+)/(\d+) scene=(\S+) psnr=([-\d.]+) delta=([+\-\d.]+)")
    received = [dict(ordinal=int(m[1]),total=int(m[2]),scene=m[3],psnr=float(m[4]),delta=float(m[5])) for m in rx.finditer(out)]
    assert len(received)==256 and len({r["scene"] for r in received})==256
    assert [r["ordinal"] for r in received]==list(range(1,257)) and all(r["total"]==256 for r in received)
    assert "INDEPENDENT_RECEIVER_COMPLETE scenes=256" in out
    assert not any(audit.FATAL.search(line) for line in err.splitlines())
    t05 = audit.read(LOGS/"slurm-gs-path-diagnostics-458004.out")
    references = {m[1]:float(m[2]) for m in re.finditer(r"PATH_DIAGNOSTIC_SCENE=\d+/256 scene=(\S+) b1_psnr=[-\d.]+ actual_psnr=([-\d.]+)",t05)}
    assert set(references)=={r["scene"] for r in received}
    assert all(references[r["scene"]]==r["psnr"] for r in received)
    t15 = dict(job="458784",scene_count=256,complete_footer=True,
        max_abs_printed_psnr_delta=max(abs(r["delta"]) for r in received),
        all_5decimal_psnr_match_t05=True,
        scope="stdout completion implies checks passed; remote REPORT/summary/metadata/scene JSON not downloaded. "
              "No historical pixel/feature bit-exact claim; exact SSIM/LPIPS deltas not in stdout.")
    result = dict(snapshot="2026-10-20 10:51 Asia/Seoul downloaded logs",live_slurm_query=False,
        counts=dict(Counter(r["train_status"] for r in rows)),
        full_test_complete=sum(r["eval_status"]=="complete_fulltest" for r in rows),
        holdout128_complete=sum(r["eval_status"]=="complete_holdout128" for r in rows),
        experiments=rows,t15=t15,runtime_hash_checks=hashes,changed_previous_sources=changed,sources=audit.SOURCES)
    inventory["snapshot"]=result["snapshot"]
    save("review.json",result);save("inventory.json",inventory);save("t15_stdout.json",received)
    print(json.dumps(dict(counts=result["counts"],fulltest=result["full_test_complete"],
        holdout=result["holdout128_complete"],t15=t15,source_count=len(audit.SOURCES))))
    for row in rows:
        if row["train_status"]!="complete_50k" or row.get("group")=="architecture":
            print(json.dumps({k:row.get(k) for k in ("train_job","arm","train_status","eval_status","resume_job",
                "restored_step","effective_estimated_step","progress","estimated_optimizer_step","metrics")}))


if __name__=="__main__":main()

"""Audit downloaded 2026-10-20 logs; keep old snapshots and runtime code intact."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
WORKSPACE = REPO.parents[1]
PRIOR = REPO / "docs/analysis/nfcgs_logs_2026-10-19_2020"
SNAPSHOT = "2026-10-20 09:11 Asia/Seoul downloaded logs"


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def diagnostic_logs(audit):
    b0_path = WORKSPACE / "slurm/slurm-gs-b0-matched-458024.out"
    t05_path = WORKSPACE / "slurm/slurm-gs-path-diagnostics-458004.out"
    b0, t05 = audit.read(b0_path), audit.read(t05_path)
    errors = {}
    for path in (b0_path, t05_path):
        error = audit.read(path.with_suffix(".err"))
        errors[path.name] = [line for line in error.splitlines() if audit.FATAL.search(line)]
        assert not errors[path.name]
    bp = re.compile(r"B0_MATCHED_SCENE=(\d+)/(\d+) scene=(\S+) b0=([-\d.]+) b1=([-\d.]+)")
    tp = re.compile(r"PATH_DIAGNOSTIC_SCENE=(\d+)/(\d+) scene=(\S+) b1_psnr=([-\d.]+) actual_psnr=([-\d.]+)")
    b = [dict(ordinal=int(m[1]), total=int(m[2]), scene=m[3], b0=float(m[4]), b1=float(m[5])) for m in bp.finditer(b0)]
    t = [dict(ordinal=int(m[1]), total=int(m[2]), scene=m[3], b1=float(m[4]), actual=float(m[5])) for m in tp.finditer(t05)]
    for rows in (b, t):
        assert len(rows) == 256 and len({row["scene"] for row in rows}) == 256
        assert [row["ordinal"] for row in rows] == list(range(1, 257))
        assert all(row["total"] == 256 for row in rows)
    assert "B0_MATCHED_COMPLETE scenes=256" in b0
    assert "PATH_DIAGNOSTICS_COMPLETE scenes=256 modes=7" in t05
    source_inputs_path = WORKSPACE / "slurm/nfcgs_legacy3_anchors_454995/inputs.json"
    source = json.loads(audit.read(source_inputs_path))
    assert source["fingerprint"] == "069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29"
    assert set(source["scenes"]) == {row["scene"] for row in b} == {row["scene"] for row in t}
    tb = {row["scene"]: row for row in t}
    paired = [dict(scene=row["scene"], b0=row["b0"], b1_b0_job=row["b1"],
                   b1_t05_job=tb[row["scene"]]["b1"], actual=tb[row["scene"]]["actual"]) for row in b]
    summary = dict(
        b0_job="458024", t05_job="458004", scenes=256,
        b0_complete_footer=True, t05_complete_footer=True,
        source_scene_set_matches=True,
        source_scene_sequence_matches_b0=source["scenes"] == [row["scene"] for row in b],
        source_scene_sequence_matches_t05=source["scenes"] == [row["scene"] for row in t],
        mean_b0_psnr=statistics.fmean(row["b0"] for row in paired),
        mean_b1_psnr=statistics.fmean(row["b1_b0_job"] for row in paired),
        mean_actual_psnr=statistics.fmean(row["actual"] for row in paired),
        mean_actual_minus_b0_psnr=statistics.fmean(row["actual"]-row["b0"] for row in paired),
        mean_actual_minus_b1_psnr=statistics.fmean(row["actual"]-row["b1_b0_job"] for row in paired),
        max_abs_b1_bridge_printed_psnr_delta=max(abs(row["b1_b0_job"]-row["b1_t05_job"]) for row in paired),
        actual_greater_than_b1_scenes=sum(row["actual"] > row["b1_b0_job"] for row in paired),
        b0_greater_than_actual_scenes=sum(row["b0"] > row["actual"] for row in paired),
        precision="Means reconstructed from 5-decimal per-scene stdout, not summary.json full precision",
        unverified="Remote summary/metadata/per-scene JSON, frame/camera/target tensor fingerprints, "
                   "checkpoint SHA-256, backend/GPU identity, renderer-repeat errors and five other T05 modes "
                   "were not downloaded. No pixel/tensor bit-exact claim.",
        fatal_matches=errors,
    )
    return dict(summary=summary, paired_scenes=paired)


def signature(config):
    return tuple(config.get(key) for key in (
        "token_order", "score_context_schedule", "score_spatial_stages", "score_slice_channels", "score_spatial_context"))


def main():
    spec = importlib.util.spec_from_file_location("prior_log_audit", PRIOR / "audit_snapshot.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    prior = json.loads((PRIOR / "review.json").read_text(encoding="utf-8"))
    before = json.loads((PRIOR / "inventory.json").read_text(encoding="utf-8"))
    changed = []
    for relative, expected in prior["sources"].items():
        audit.read(WORKSPACE / relative)
        if audit.SOURCES.get(relative, {}).get("sha256") != expected["sha256"]:
            changed.append(relative)
    states = {}
    for folder in ("architecture_pilots_20261019", "context_seed_pair_20261019"):
        state = json.loads(audit.read(WORKSPACE / "slurm" / folder / "state.json"))
        mismatches = [name for name, digest in state["inputs"]["code_sha256"].items()
                      if not (REPO/name).is_file() or hashlib.sha256((REPO/name).read_bytes()).hexdigest() != digest]
        assert not mismatches
        states[folder] = dict(runtime_files=len(state["inputs"]["code_sha256"]), mismatches=mismatches)
    rows = []
    for original in prior["experiments"]:
        row = audit.inspect(dict(original))
        if row["group"] in ("controls", "rd_extensions"):
            stem = "slurm-gs-order-context"
        elif row["group"] == "architecture":
            stem = "slurm-gs-arch-train"
        else:
            stem = "slurm-gs-seed-pair-train"
        err = audit.read(WORKSPACE / "slurm" / f"{stem}-{row['train_job']}.err")
        if "returned non-zero exit status 137" in err:
            assert row["train_status"] != "complete_50k"
            row["train_status"] = "terminated_exit137"
            row["remaining_training_hours_at_logged_speed"] = None
            row["termination"] = dict(exit_status=137, root_cause="unknown; request sacct/cluster accounting",
                evidence=[line for line in err.splitlines() if line.startswith("srun:")])
        rows.append(row)
        print("AUDITED", row["train_job"], row["train_status"], row["eval_status"], flush=True)
    points = list(before["results"])
    zero = list(before["lambda_zero_points"])
    for point in points + zero:
        audit.read(WORKSPACE / point["source_log"])
        assert audit.SOURCES[point["source_log"]]["sha256"] == point["source_sha256"]
    families = {signature(row["declared_config"]): row["family"] for row in points}
    evaluated_before = {row["eval_job"] for row in prior["experiments"] if row["eval_status"] == "complete_fulltest"}
    additions = []
    for row in rows:
        if row["eval_status"] != "complete_fulltest" or row["eval_job"] in evaluated_before:
            continue
        family = families[signature(row["declared_codec_config"])]
        point = dict(label=f"{family}_{row['lambda'].replace('.', 'p')}", family=family,
            rate_lambda=float(row["lambda"]), source_log=row["eval_source_log"], source_sha256=row["eval_source_sha256"],
            evaluated_scenes=row["evaluated_scenes"], metrics=row["metrics"], declared_config=row["declared_codec_config"],
            checkpoint=row["checkpoint"], train_job=row["train_job"], eval_job=row["eval_job"],
            byte_accounting_error=row["byte_accounting_error"])
        additions.append(point)
        (points if point["rate_lambda"] > 0 else zero).append(point)
    assert len({(row["family"], row["rate_lambda"]) for row in points}) == len(points)
    diagnostic = diagnostic_logs(audit)
    counts = dict(Counter(row["train_status"] for row in rows))
    report = dict(snapshot=SNAPSHOT, scope="downloaded files only; no live Slurm query or remote artifact access",
        recent_training_points=len(rows), training_status_counts=counts,
        completed_recent_fulltest_evaluations=sum(row["eval_status"] == "complete_fulltest" for row in rows),
        newly_completed_fulltest_points=additions, completed_positive_rate_points=len(points), lambda_zero_points=len(zero),
        changed_previous_sources=changed, runtime_hash_checks=states, experiments=rows,
        diagnostic_log_summary=diagnostic["summary"],
        diagnostics_remote_json_available=False, sources=audit.SOURCES,
        progress_scope="last complete progress; epoch*ceil(nominal_epoch_batches/2)+floor(batch/2). "
                       "All earlier complete epochs in the six progressing logs end at nominal 33017/33017.",
        eta_scope="logged-speed estimate only; not current wall-clock ETA; no ETA assigned to killed jobs")
    inventory = dict(version=1, snapshot=SNAPSHOT, scope="completed actual full-test results, each 6991 scenes",
        completed_positive_rate_points=len(points), families=len({row["family"] for row in points}),
        lambda_zero_points=zero, results=points)
    save("review.json", report)
    save("inventory.json", inventory)
    save("diagnostic_stdout.json", diagnostic)
    print("SUMMARY", json.dumps(dict(counts=counts, positive_points=len(points), lambda_zero=len(zero),
                                     additions=len(additions), sources=len(audit.SOURCES))), flush=True)
    print("DIAGNOSTICS", json.dumps(diagnostic["summary"]), flush=True)


if __name__ == "__main__":
    main()

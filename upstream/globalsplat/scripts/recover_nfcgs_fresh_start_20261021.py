#!/usr/bin/env python3
"""Replace fresh jobs that failed before creating a run due to missing node-local data.

Restores the established ariel-v7 exclusion. No dataset path, initialization,
training schedule, or accepted successful run is changed. No preflight GPU job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import submit_nfcgs_fresh_rd as submit
from globalsplat.fresh_rd import VERSION, atomic_json, directory_lock, file_hash, read_json, run_key, wave_runs

PLACEMENT_FILES = (
    "scripts/slurm/train_nfcgs_fresh_rd.slurm", "scripts/slurm/eval_nfcgs_fresh_rd.slurm",
    "scripts/slurm/monitor_nfcgs_fresh_rd.slurm", "scripts/slurm/fresh_rd_environment.sh",
    "scripts/slurm/prepare_nfcgs_architecture_pilots.slurm",
    "scripts/submit_nfcgs_fresh_rd.py",
)
TAG = "missing_node_data_recovery_20261021"
MONITOR_TAG = "monitor_lock_recovery_20261021"


def allocation_info(job):
    result = subprocess.run(["sacct", "-j", str(job), "-X", "--noheader", "--parsable2", "--format=JobIDRaw,State,NodeList"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields[0] == str(job):
            return dict(state=fields[1].split()[0].rstrip("+"), nodes=fields[2])
    raise ValueError(f"allocation state unavailable for {job}")


def node_options(recovery):
    return ["--exclude=" + ",".join(sorted(set(recovery["excluded_nodes"])))]


def validate_state(state):
    if state.get("version") != VERSION or state.get("pending_submission"):
        raise ValueError("unsupported state or unresolved sbatch intent; reconcile it first")
    current = {name:file_hash(ROOT/name) for name in submit.FILES}
    old = state["inputs"]["code"]
    changed = sorted(k for k in set(old)|set(current) if old.get(k) != current.get(k))
    if set(changed) - set(PLACEMENT_FILES):
        raise ValueError(f"training/runtime changes outside this placement repair: {changed}")
    expected = {run_key(c,r,s) for c,r,s in wave_runs(1)}
    if set(state.get("runs", {})) != expected:
        raise ValueError("recovery requires exactly the fifteen wave-1 runs")
    for key in expected:
        if key+"/A" not in state["jobs"] or any(key+suffix in state["jobs"] for suffix in ("/B", "/test")):
            raise ValueError("stage B/test already submitted or stage A missing")
    return current, changed


def initial_data_failure(run, record, log_root, dataset_root):
    """Never turn a real interrupted training run into a new step-zero run."""
    folder = Path(run["run_dir"])
    if (folder/"run.json").exists() or list(folder.glob("**/*.ckpt")) or list(folder.glob("attempts/**/invocation.json")):
        raise ValueError(f"run has training artifacts; use full-state resume instead: {folder}")
    log = Path(log_root)/f"slurm-gs-fresh-train-{record['id']}.err"
    text = log.read_text(encoding="utf-8", errors="replace").replace("\\", "/")
    missing_index = str(dataset_root).replace("\\", "/").rstrip("/")+"/train/index.json"
    python_failure = "FileNotFoundError" in text and "validate_preparation" in text and missing_index in text
    shell_failure = "ERROR: missing RE10K index " + missing_index in text
    if not (python_failure or shell_failure):
        raise ValueError(f"not a proven pre-training data-index failure: {log}")
    return dict(stderr=str(log), stderr_sha256=file_hash(log), run_manifest_absent=True, checkpoints_absent=True)


def recover_monitor(sub, args, recovery):
    key = "monitor/1/A"
    marker = recovery.setdefault("monitor", dict(old_id=sub.state["jobs"][key]["id"]))
    current = sub.state["jobs"][key]
    if current["id"] != marker["old_id"]:
        if marker["old_id"] not in current.get("history", []):
            raise ValueError("monitor replacement history differs")
        marker["new_id"] = current["id"]
        sub.save()
        print(f"REUSING_MONITOR={current['id']}", flush=True)
        return
    if not args.restart_monitor:
        print(f"KEEPING_MONITOR={current['id']} (placement unchanged)", flush=True)
        return
    sub.save()  # restart intent precedes cancellation/replacement
    convergence = args.root/"convergence"
    locks = [convergence/".monitor_1_A.lock", convergence/".csv.lock"]
    before = {str(p):file_hash(p) for p in locks if p.exists()}
    if args.dry_run:
        print(f"WOULD_REPLACE_MONITOR={current['id']} on data-capable allocation", flush=True)
    else:
        status = submit.job_state(current["id"])
        if status in ("RUNNING", "PENDING", "CONFIGURING", "SUSPENDED", "COMPLETING", "REQUEUED"):
            subprocess.run(["scancel", current["id"]], cwd=ROOT, check=True)
            for _ in range(60):
                if submit.job_state(current["id"]) in submit.TERMINAL_FAILURES | {"COMPLETED"}:
                    break
                time.sleep(1)
            else:
                raise ValueError("old monitor has not stopped; learner jobs have not been replaced")
        # Only the two exact locks from the old, now inactive monitor can be removed.
        for lock in locks:
            if not lock.exists():
                continue
            if not lock.resolve().is_relative_to(convergence.resolve()) or before.get(str(lock)) != file_hash(lock):
                raise ValueError("monitor lock identity changed; inspect it before retrying")
            lock.unlink()
    if sub.state["lanes"].get("15") == current["id"]:
        del sub.state["lanes"]["15"]
    env = dict(submit.base_env(args), MONITOR_PHASE="A", MONITOR_WAVE="1")
    new = sub.add(key, submit.SLURM+"monitor_nfcgs_fresh_rd.slurm", env, lane=15, replace_failed=True,
        sbatch_options=node_options(recovery))
    marker["new_id"] = new
    sub.save()


def recover(sub, args):
    state = sub.state
    current, changed = validate_state(state)
    recovery = state.setdefault(TAG, dict(runs={}, created_utc=datetime.now(timezone.utc).isoformat()))
    excluded = set(recovery.get("excluded_nodes", ["ariel-v7"]))
    plans = []
    for key, run in state["runs"].items():
        if not Path(run["run_dir"]).resolve().is_relative_to(args.root.resolve()):
            raise ValueError("run output is outside the registered sweep root")
        record = state["jobs"][key+"/A"]
        entry = recovery["runs"].get(key)
        if entry and record["id"] != entry["old_id"]:
            if entry["old_id"] not in record.get("history", []):
                raise ValueError("replacement job has no matching history")
            entry["new_id"] = record["id"]
            print(f"REUSING {key}={record['id']}", flush=True)
            continue
        proof = initial_data_failure(run, record, args.log_root, args.dataset_root)
        if not args.dry_run:
            allocation = allocation_info(record["id"])
            if allocation["state"] not in submit.TERMINAL_FAILURES:
                raise ValueError(f"job is not in a terminal failure state: {record['id']}")
            if not re.fullmatch(r"[A-Za-z0-9_.\[\],-]+", allocation["nodes"]) or allocation["nodes"] in ("None", "Unknown"):
                raise ValueError("failed node is unknown; inspect scheduler placement before replacement")
            excluded.add(allocation["nodes"])
            proof["allocation"] = allocation
        recovery["runs"][key] = dict(old_id=record["id"], **proof)
        plans.append((key, run, record))
    recovery["excluded_nodes"] = sorted(excluded)
    if changed:
        state.setdefault("input_migrations", []).append(dict(previous_inputs=state["inputs"].copy(), changed_files=changed,
            reason="restore node exclusion before replacing proven step-zero data failures", utc=datetime.now(timezone.utc).isoformat()))
        state["inputs"] = dict(state["inputs"], code=current)
    recovery["recovery_script_sha256"] = file_hash(Path(__file__))
    sub.save()
    recover_monitor(sub, args, recovery)
    for key, run, old in plans:
        lane = old.get("lane")
        if lane is None:
            raise ValueError("original learner has no resource lane")
        if state["lanes"].get(str(lane)) == old["id"]:
            del state["lanes"][str(lane)]
        new = sub.add(key+"/A", submit.SLURM+"train_nfcgs_fresh_rd.slurm",
            submit.run_env(args, run, "A", False), lane=lane, run=run, replace_failed=True,
            sbatch_options=node_options(recovery))
        recovery["runs"][key]["new_id"] = new
        sub.save()
    print(f"RECOVERY_REPLACED={len(plans)} same_initializers=True fresh_step_zero=True no_gate0_dependency=True", flush=True)


def active_monitor_jobs():
    """Find candidate monitor allocations without touching learner jobs."""
    result = subprocess.run(["squeue", "--me", "--noheader", "--format=%A|%j"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    names = submit.stage_a_monitor_names()
    jobs = set()
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", 1)
        if len(fields) == 2 and fields[1].strip() in names:
            if not re.fullmatch(r"[1-9][0-9]*", fields[0].strip()):
                raise ValueError("unrecognized monitor allocation in squeue")
            jobs.add(fields[0].strip())
    return jobs


def verify_previous_monitor(job):
    """Only the known previous monitor in this repository may be stopped."""
    result = subprocess.run(["scontrol", "show", "job", str(job), "-o"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", result.stdout))
    if (fields.get("JobId") != str(job) or fields.get("JobName") not in submit.stage_a_monitor_names()
            or Path(fields.get("WorkDir", "")).resolve() != ROOT.resolve()
            or Path(fields.get("Command", "")).resolve() != (ROOT/submit.SLURM/"monitor_nfcgs_fresh_rd.slurm").resolve()):
        raise ValueError(f"previous job {job} is not a verified monitor in this repository")


def recover_monitor_only(sub, args):
    """Repair this incident's stale monitor lock; never recreate a learner."""
    state = sub.state
    _, changed = validate_state(state)
    if changed:
        raise ValueError(f"monitor-only recovery requires unchanged sweep code: {changed}")
    key = "monitor/1/A"
    current = state["jobs"][key]
    if (current.get("run") is not None or current.get("lane") != 15
            or state["lanes"].get("15") != current["id"]):
        raise ValueError("monitor lane differs; no jobs were changed")
    env = current["env"].copy()
    expected = dict(FRESH_OUTPUT_ROOT=str(args.root), FRESH_STATE=str(args.state),
        REPO_DIR=str(ROOT), MONITOR_PHASE="A", MONITOR_WAVE="1")
    if any(env.get(k) != value for k, value in expected.items()):
        raise ValueError("registered monitor environment differs from this sweep")
    script = submit.SLURM+"monitor_nfcgs_fresh_rd.slurm"
    if current["command"][-1] != script:
        raise ValueError("registered job is not the expected monitor")
    prior = {"460326"} | set(current.get("history", []))
    prior.discard(current["id"])
    if any(not re.fullmatch(r"[1-9][0-9]*", job) for job in prior | {current["id"]}):
        raise ValueError("invalid recorded monitor job ID")

    if not args.dry_run:
        status = submit.job_state(current["id"])
        if status not in submit.TERMINAL_FAILURES:
            if status in {"RUNNING", "PENDING", "CONFIGURING", "SUSPENDED", "COMPLETING", "REQUEUED", "COMPLETED"}:
                print(f"REUSING_MONITOR={current['id']} state={status} learner_jobs_unchanged=True", flush=True)
                return
            raise ValueError(f"monitor is not in a known terminal failure state: {status}")
    stderr = args.log_root/f"slurm-gs-fresh-monitor-{current['id']}.err"
    error = stderr.read_text(encoding="utf-8", errors="replace")
    lock = args.root/"convergence/.monitor_1_A.lock"
    if "FileExistsError" not in error or ".monitor_1_A.lock" not in error:
        raise ValueError("monitor-only repair requires the recorded monitor-lock failure")
    # CSV writers in other phases must not be affected by this Stage A repair.
    csv_lock = args.root/"convergence/.csv.lock"
    snapshots = {p: file_hash(p) for p in (lock, csv_lock) if p.exists()}
    if args.dry_run:
        print(f"WOULD_CHECK_MONITORS current={current['id']} previous={sorted(prior)}", flush=True)
        print("WOULD_STOP_VERIFIED_PREVIOUS_MONITORS_AND_REMOVE_UNCHANGED_MONITOR_LOCKS", flush=True)
    else:
        active = active_monitor_jobs()
        unknown = active - prior - {current["id"]}
        if unknown:
            raise ValueError(f"other active monitors exist; locks preserved: {sorted(unknown)}")
        # Validate every active candidate before stopping any of them.
        for job in sorted(prior & active):
            verify_previous_monitor(job)
        for job in sorted(prior):
            status = submit.job_state(job)
            if status not in submit.TERMINAL_FAILURES | {"COMPLETED"}:
                verify_previous_monitor(job)
                subprocess.run(["scancel", job], cwd=ROOT, check=True)
                for _ in range(60):
                    if submit.job_state(job) in submit.TERMINAL_FAILURES | {"COMPLETED"}:
                        break
                    time.sleep(1)
                else:
                    raise ValueError(f"previous monitor {job} has not stopped; locks preserved")
        if active_monitor_jobs():
            raise ValueError("a monitor is still active; locks preserved")
        convergence = (args.root/"convergence").resolve()
        for path in (lock, csv_lock):
            if not path.exists():
                continue
            if not path.resolve().is_relative_to(convergence) or snapshots.get(path) != file_hash(path):
                raise ValueError(f"monitor lock changed while inspecting jobs; retry recovery: {path}")
        for path in (lock, csv_lock):
            path.unlink(missing_ok=True)

    entry = dict(old_id=current["id"], previous_monitor_ids=sorted(prior),
        stderr_sha256=file_hash(stderr), script_sha256=file_hash(Path(__file__)),
        utc=datetime.now(timezone.utc).isoformat())
    state.setdefault(MONITOR_TAG, []).append(entry)
    del state["lanes"]["15"]  # Do not inherit afterok from the failed monitor.
    try:
        new = sub.add(key, script, env, lane=15, replace_failed=True)
    except subprocess.CalledProcessError:
        # sbatch rejected the submission; leave a retryable lane and accepted ID.
        state["lanes"]["15"] = current["id"]
        sub.save()
        raise
    entry["new_id"] = new
    sub.save()
    print(f"MONITOR_ONLY_REPLACED={new} learner_jobs_unchanged=True checkpoints_unchanged=True", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, default=ROOT/"logs/slurm/fresh_rd/state.json")
    p.add_argument("--log-root", type=Path, default=ROOT/"logs/slurm")
    p.add_argument("--restart-monitor", action="store_true", help="stop the old monitor and replace it on a data-capable node")
    p.add_argument("--monitor-only", action="store_true", help="repair the failed Stage A monitor only; preserve all learner jobs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    def work():
        state = read_json(args.state)
        args.root = Path(state["inputs"]["root"])
        args.dataset_root = Path(state["inputs"]["dataset"])
        args.preparation = Path(state["inputs"]["preparation"])
        args.native_checkpoint = Path(state["inputs"]["native_checkpoint"])
        args.budget = state.get("budget") or 100000
        if not args.dry_run and (not args.root.resolve().is_relative_to((ROOT/"outputs").resolve())):
            raise ValueError("recovery output must stay inside this repository's outputs")
        sub = submit.Submission(args, state)
        if args.monitor_only:
            recover_monitor_only(sub, args)
        else:
            recover(sub, args)
    if args.dry_run:
        work()
    else:
        with directory_lock(args.state.parent, ".submit.lock"):
            work()


if __name__ == "__main__":
    main()

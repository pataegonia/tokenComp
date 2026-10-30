#!/usr/bin/env python3
"""Re-evaluate replaced Wave 1 60k files without changing learners or lineage.

Submit with --submit, or inspect with --dry-run. Direct evaluation requires a
Slurm allocation. Original metrics are archived; matching hashes are never forged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import monitor_nfcgs_fresh_rd as monitor
import run_nfcgs_fresh_rd as runner
import submit_nfcgs_fresh_rd as submit
from globalsplat.fresh_rd import (VERSION, atomic_json, checkpoint_path, directory_lock,
    file_hash, read_json)
from globalsplat.fresh_rd_runtime import checkpoint_protocol_matches


def eval_args(run, dataset, run_dir, output=None):
    command = ["validation", "--cell", run["cell"], "--rate-lambda", run["rate_lambda"],
        "--seed-index", str(run["seed_index"]), "--step", "60000", "--budget", "100000",
        "--run-dir", str(run_dir), "--checkpoint", str(checkpoint_path(run["run_dir"], 60000)),
        "--preparation", run["preparation"], "--dataset-root", str(dataset)]
    if output is not None:
        command += ["--output", str(output)]
    return command


def plan_repair(args):
    state_sha = file_hash(args.state)
    state = read_json(args.state)
    provisional = state.get("provisional_stage_b", {})
    if (state.get("version") != VERSION or state.get("budget") is not None
            or provisional.get("status") != "provisional" or provisional.get("rollback")
            or state.get("pending_submission")):
        raise ValueError("repair requires the undecided original provisional sweep before rollback")
    if Path(state["inputs"]["root"]).resolve() != args.root:
        raise ValueError("state belongs to another output root")
    dataset = Path(state["inputs"]["dataset"]).resolve()
    repairs = []
    runs = submit.wave1_records(state)
    for run in runs:
        folder = Path(run["run_dir"])
        if not folder.resolve().is_relative_to(args.root) or (folder/"resume_parent.json").exists():
            raise ValueError("repair accepts only original registered Stage A runs")
        path = checkpoint_path(folder, 60000)
        done = read_json(folder/"stage_A_complete.json")
        if done["step"] != 60000:
            raise ValueError(f"{run['key']}: Stage A did not finish at 60k")
        current = file_hash(path)
        original = done["checkpoint_sha256"]
        output = folder/"validation/step000060000"
        monitor.completed_rows(folder/"validation/step000040000", checkpoint_path(folder, 40000))
        if (output/".lock").exists():
            raise ValueError(f"{run['key']}: evaluation is locked; leave the lock intact")
        if (output/"complete.json").exists():
            try:
                rows = monitor.completed_rows(output, path)
            except monitor.CheckpointIdentityChanged:
                monitor.completed_rows(output)  # Check old metrics before archiving.
                if read_json(output/"complete.json")["checkpoint_sha256"] != original:
                    raise ValueError(f"{run['key']}: stale evaluation is not the Stage A original")
            else:
                if current != original:
                    submit.check_superseded_60k(folder, original, rows)
                continue
        checks = ["stage_a_original_sha_matches"]
        mode = "stage_a_original"
        protocol = read_json(folder/"run.json")
        if current != original:
            checks = submit.check_resaved_60k(path, original, protocol)
            mode = "resaved_after_first_microbatch"
        current_protocol = runner.identity(runner.parse_args(eval_args(run, dataset, folder)))
        changed = sorted(name for name in set(protocol.get("code_sha256", {}))
            | set(current_protocol["code_sha256"])
            if protocol.get("code_sha256", {}).get(name) != current_protocol["code_sha256"].get(name))
        parent = dict(protocol=protocol, changed_runtime_files=changed, source_60k_mode=mode,
                      checkpoints_sha256={"60000": current})
        current_protocol["resume_parent"] = parent
        if not checkpoint_protocol_matches(dict(global_step=60000, fresh_rd_protocol=protocol), current_protocol):
            raise ValueError(f"{run['key']}: runtime/recipe changed beyond the declared 120k repair")
        if file_hash(path) != current:
            raise ValueError(f"{run['key']}: checkpoint changed during preflight")
        repairs.append(dict(run=run, checkpoint=str(path), checkpoint_sha256=current,
            original_stage_a_sha256=original, run_manifest_sha256=file_hash(folder/"run.json"),
            protocol=current_protocol, parent=parent, passed_checks=checks))
        print(f"HOLDOUT_REPAIR_NEEDED={run['key']} mode={mode} sha256={current}", flush=True)
    if file_hash(args.state) != state_sha:
        raise ValueError("sweep state changed during preflight")
    print(f"HOLDOUT_REPAIR_PLAN runs={len(runs)} reevaluate={len(repairs)} learner_jobs_changed=False", flush=True)
    return dict(state_sha256=state_sha, dataset=str(dataset), runs=runs, repairs=repairs)


def unchanged(args, plan):
    if file_hash(args.state) != plan["state_sha256"]:
        raise ValueError("sweep state changed during repair; no learners changed")
    for item in plan["repairs"]:
        folder = Path(item["run"]["run_dir"])
        if (file_hash(item["checkpoint"]) != item["checkpoint_sha256"]
                or file_hash(folder/"run.json") != item["run_manifest_sha256"]):
            raise ValueError("checkpoint/manifest changed during repair; no learners changed")


def publish_curves(root, runs):
    table = []
    complete = True
    for run in runs:
        for step in range(10000, 60001, 10000):
            folder = Path(run["run_dir"])/"validation"/f"step{step:09d}"
            try:
                rows = monitor.completed_rows(folder, checkpoint_path(run["run_dir"], step))
            except FileNotFoundError:
                complete = False
                continue
            table.append(monitor.summary_row(run, step, rows))
    with directory_lock(root, ".csv.lock"):
        atomic_json(root/"curve_wave1_A.json", table)
        merged = {}
        for path in sorted(root.glob("curve_wave*.json")):
            for row in read_json(path):
                merged[(row["cell"], row["rate_lambda"], row["seed"], row["step"])] = row
        monitor.atomic_csv(root/"holdout_curve.csv", [merged[k] for k in sorted(merged)])
    if complete:
        atomic_json(root/"wave1_A_complete.json", dict(wave=1, phase="A", budget=100000,
            run_count=len(runs), split="validation", test_used=False))


def repair(args):
    convergence = args.root/"convergence"
    with directory_lock(convergence, ".monitor_1_A.lock"):
        plan = plan_repair(args)
        unchanged(args, plan)
        session = convergence/"holdout_repairs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+f"_{time.time_ns()}")
        atomic_json(session/"plan.json", plan)
        monitor.invalidate_gate1(convergence)
        complete = convergence/"wave1_A_complete.json"
        if complete.exists():
            complete.rename(complete.with_name(f"wave1_A_complete.superseded_{time.time_ns()}.json"))
        for item in plan["repairs"]:
            unchanged(args, plan)
            run = item["run"]
            folder = Path(run["run_dir"])
            output = folder/"validation/step000060000"
            audit_run = session/run["key"]
            # Evaluate through a separate, narrowly migrated manifest. Never edit
            # the original run.json or load an old checkpoint under a new identity.
            atomic_json(audit_run/"resume_parent.json", item["parent"])
            atomic_json(audit_run/"run.json", item["protocol"])
            if (output/"complete.json").exists():
                if not monitor.supersede_evaluation(output, item["checkpoint"]):
                    raise ValueError("evaluation changed or became locked before archiving")
            command = [sys.executable, str(ROOT/"scripts/run_nfcgs_fresh_rd.py"),
                *eval_args(run, plan["dataset"], audit_run, output)]
            if not monitor.evaluate(command, output/"monitor.log"):
                raise RuntimeError(f"holdout evaluation failed; inspect {output/'monitor.log'}; learners untouched")
            unchanged(args, plan)
            rows = monitor.completed_rows(output, item["checkpoint"])
            compared = submit.check_superseded_60k(folder, item["original_stage_a_sha256"], rows)
            atomic_json(audit_run/"verified.json", dict(checkpoint_sha256=item["checkpoint_sha256"],
                superseded_comparisons=compared, source_manifest_unchanged=True, learner_jobs_changed=False))
            print(f"HOLDOUT_REPAIR_VERIFIED={run['key']} original_metrics_equal=True", flush=True)
        unchanged(args, plan)
        # Recheck every run, including archive equality after an interrupted repair.
        remaining = plan_repair(args)
        if remaining["repairs"]:
            raise ValueError("some holdout identities are still stale; Gate 1 not published")
        publish_curves(convergence, plan["runs"])
        monitor.refresh_gate1(convergence, plan["runs"])
        gate = read_json(convergence/"gate1.json")
        atomic_json(session/"complete.json", dict(gate1_sha256=file_hash(convergence/"gate1.json"),
            recommended_budget=gate["recommended_budget"], consider_150k=gate["consider_150k"],
            learner_jobs_changed=False, checkpoint_files_changed=False))
        print(f"HOLDOUT_REPAIR_COMPLETE reevaluated={len(plan['repairs'])} recommended_budget={gate['recommended_budget']} consider_150k={gate['consider_150k']} learners_untouched=True", flush=True)


def submit_repair(args, plan):
    # This allocation is only the missing holdout evaluations; no learning or
    # scheduler dependencies on expired Stage A IDs are involved.
    command = [sys.executable, str(ROOT/"scripts/repair_nfcgs_stage_a_holdout.py"),
        "--root", str(args.root), "--state", str(args.state)]
    body = "set -euo pipefail\nsource "+shlex.quote(str(ROOT/"scripts/slurm/fresh_rd_environment.sh"))+"\n"+shlex.join(command)
    command = ["sbatch", "--parsable", "--export=ALL", "--job-name=gs-fr-repair-w1-holdout60k",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--gres=gpu:normal:1", "--mem-per-gpu=48G",
        "--partition=batch_ugrad", "--exclude=ariel-v7", "--time=24:00:00",
        "--output=logs/slurm/slurm-gs-fresh-holdout-repair-%j.out",
        "--error=logs/slurm/slurm-gs-fresh-holdout-repair-%j.err", "--wrap", "exec bash -lc "+shlex.quote(body)]
    env = dict(os.environ, REPO_DIR=str(ROOT), FRESH_OUTPUT_ROOT=str(args.root),
        DATASET_ROOT=plan["dataset"], PREPARATION_DIR=plan["runs"][0]["preparation"])
    if args.dry_run:
        print(shlex.join(command))
        return
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        print(result.stderr, file=sys.stderr, end="")
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    job = result.stdout.strip().split(";", 1)[0]
    print(f"HOLDOUT_REPAIR_JOB={job} learner_jobs_changed=False", flush=True)
    print(f"tail -n 40 logs/slurm/slurm-gs-fresh-holdout-repair-{job}.out")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT/"outputs/nfcgs_fresh_rd")
    parser.add_argument("--state", type=Path, default=ROOT/"logs/slurm/fresh_rd/state.json")
    parser.add_argument("--submit", action="store_true", help="submit one GPU allocation for stale holdout evaluations")
    parser.add_argument("--dry-run", action="store_true", help="read-only preflight; no jobs or files changed")
    args = parser.parse_args(argv)
    args.root, args.state = args.root.resolve(), args.state.resolve()
    if args.submit or args.dry_run:
        plan = plan_repair(args)
        if args.submit:
            submit_repair(args, plan)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU evaluation requires a Slurm allocation; use --submit from the login node")
    repair(args)


if __name__ == "__main__":
    main()

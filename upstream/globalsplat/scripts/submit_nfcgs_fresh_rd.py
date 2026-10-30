#!/usr/bin/env python3
"""Submit fresh RD jobs; diagnostics are optional and never gate training by default."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_nfcgs_fresh_rd as runner
from globalsplat.fresh_rd import (VERSION, atomic_json, checkpoint_path, directory_lock, file_hash, read_json, run_key, wave_runs)

SLURM = "scripts/slurm/"
FILES = runner.RUNTIME_FILES + ("scripts/submit_nfcgs_fresh_rd.py", "scripts/monitor_nfcgs_fresh_rd.py",
    "scripts/check_nfcgs_fresh_rd_gate0.py", "scripts/prepare_nfcgs_architecture_pilots.py",
    SLURM+"train_nfcgs_fresh_rd.slurm", SLURM+"eval_nfcgs_fresh_rd.slurm",
    SLURM+"monitor_nfcgs_fresh_rd.slurm", SLURM+"gate0_nfcgs_fresh_rd.slurm",
    SLURM+"prepare_gate0_nfcgs_fresh_rd.slurm", SLURM+"fresh_rd_environment.sh",
    SLURM+"submit_nfcgs_fresh_rd.sh", SLURM+"prepare_nfcgs_architecture_pilots.slurm")
TERMINAL_FAILURES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
RESTART_REPAIR_FILES = runner.RUNTIME_REPAIR_FILES | {"scripts/submit_nfcgs_fresh_rd.py",
    "scripts/monitor_nfcgs_fresh_rd.py", SLURM+"train_nfcgs_fresh_rd.slurm"}


def job_name(key, env=None):
    """Expose the model, lambda, seed and stage in scheduler listings."""
    env = env or {}
    parts = key.split("/")
    budget = int(env.get("FRESH_BUDGET") or 100000)
    steps = f"{budget//1000}k" if budget % 1000 == 0 else str(budget)
    if len(parts) == 4 and parts[1].startswith("lambda"):
        cell, rate, seed, stage = parts
        label = {"h_on":"H-on", "h_off":"H-off", "n_on":"N-on", "n_off":"N-off",
                 "m_on":"M-on", "h_l3":"H-L3", "s56":"S56", "msh":"MSH"}.get(cell, cell)
        phase = "A60k" if stage == "A" else f"{stage}{steps}"
        return f"gs-fr-{label}-l{rate[6:]}-{seed}-{phase}"
    if len(parts) == 3 and parts[0] == "monitor":
        phase = "A60k" if parts[2] == "A" else parts[2]+steps
        return f"gs-fr-monitor-w{parts[1]}-{phase}"
    return "gs-fr-" + re.sub(r"[^A-Za-z0-9_-]+", "-", key).strip("-")


def stage_a_monitor_names():
    # Existing jobs keep their historical names until explicitly renamed.
    return {job_name("monitor/1/A"), "gs-fresh-monitor",
            "gs-fr-" + hashlib.sha256(b"monitor/1/A").hexdigest()[:10]}


def job_state(job):
    result = subprocess.run(["sacct", "-j", str(job), "-X", "--noheader", "--parsable2", "--format=JobIDRaw,State"],
        capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields[0] == str(job):
            return fields[1].split()[0].rstrip("+")
    raise ValueError(f"scheduler state unavailable for {job}; no replacement submitted")


class Submission:
    def __init__(self, args, state):
        self.args, self.state = args, state
        self._dry_sequence = len(state["jobs"])

    def save(self):
        if not self.args.dry_run:
            atomic_json(self.args.state, self.state)

    def add(self, key, script, env, dependencies=(), lane=None, run=None, replace_failed=False, sbatch_options=()):
        jobs = self.state["jobs"]
        if key in jobs and not replace_failed:
            print(f"REUSING {key}={jobs[key]['id']}", flush=True)
            return jobs[key]["id"]
        deps = list(dict.fromkeys(str(d) for d in dependencies if d))
        if lane is not None and self.state["lanes"].get(str(lane)):
            deps.append(self.state["lanes"][str(lane)])
        deps = list(dict.fromkeys(deps))
        token = "fresh-rd-" + uuid.uuid4().hex
        command = ["sbatch", "--parsable", "--export=ALL", f"--comment={token}",
            f"--job-name={job_name(key, env)}"]
        command += list(sbatch_options)
        if deps:
            command += ["--dependency=afterok:" + ":".join(deps)]
        command += [script]
        payload = dict(command=command, env=env, dependencies=deps, lane=lane, run=run)
        if self.args.dry_run:
            self._dry_sequence += 1
            job = f"DRY_{self._dry_sequence}"
            print(f"{key}: " + shlex.join(command), flush=True)
        else:
            if self.state.get("pending_submission"):
                raise ValueError("an interrupted sbatch intent remains; reconcile its accepted ID before retrying")
            self.state["pending_submission"] = dict(key=key, token=token, payload=payload)
            self.save()  # journal BEFORE sbatch, so an interruption cannot cause a silent duplicate
            try:
                result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, **env), capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as error:
                print(f"SBATCH_FAILED={key} exit={error.returncode} accepted_ids_preserved=True",file=sys.stderr,flush=True)
                if error.stderr:
                    print(error.stderr.rstrip(),file=sys.stderr,flush=True)
                if error.stdout:
                    print("SBATCH_STDOUT="+error.stdout.rstrip(),file=sys.stderr,flush=True)
                self.state.setdefault("submission_failures",[]).append(dict(key=key,command=command,
                    returncode=error.returncode,stderr=error.stderr or "",stdout=error.stdout or "",
                    utc=datetime.now(timezone.utc).isoformat()))
                self.state.pop("pending_submission", None)
                self.save()
                raise
            job = result.stdout.strip().split(";", 1)[0]
            if not re.fullmatch(r"[1-9][0-9]*", job):
                raise ValueError("unrecognized sbatch response; intent retained for scheduler reconciliation")
        self.accept(key, job, payload)
        print(f"ACCEPTED {key}={job}", flush=True)
        return job

    def accept(self, key, job, payload):
        old = self.state["jobs"].get(key)
        history = list(old.get("history", [])) + [old["id"]] if old else []
        self.state["jobs"][key] = dict(id=job, **payload, history=history)
        lane, run = payload["lane"], payload["run"]
        if lane is not None:
            self.state["lanes"][str(lane)] = job
        if run:
            run = dict(run)
            if not run.get("run_dir"):
                run["run_dir"] = str(self.args.root / run["key"] / f"job_{job}")
            self.state["runs"][run["key"]] = run
        self.state.pop("pending_submission", None)
        self.save()


def base_env(args):
    return dict(REPO_DIR=str(ROOT), DATASET_ROOT=str(args.dataset_root), FRESH_OUTPUT_ROOT=str(args.root),
        FRESH_STATE=str(args.state), PREPARATION_DIR=str(args.preparation),
        FRESH_NATIVE_CHECKPOINT=str(args.native_checkpoint), OMP_NUM_THREADS="8")


def rename_active_jobs(sub):
    """Rename only live, owned, registered jobs after verifying submission identity."""
    queued = subprocess.run(["squeue", "--me", "--noheader", "--format=%A"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    live = set(queued.stdout.split())
    plans = []
    for key, record in sub.state["jobs"].items():
        if record["id"] not in live:
            continue
        info = subprocess.run(["scontrol", "show", "job", record["id"], "-o"],
            cwd=ROOT, capture_output=True, text=True, check=True)
        fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", info.stdout))
        token = next((v.split("=",1)[1] for v in record["command"] if v.startswith("--comment=")), None)
        if (not token or fields.get("JobId") != record["id"] or fields.get("Comment") != token
                or Path(fields.get("WorkDir", "")).resolve() != ROOT.resolve()
                or Path(fields.get("Command", "")).resolve() != (ROOT/record["command"][-1]).resolve()):
            raise ValueError(f"refusing to rename an unverified job: {key}={record['id']}")
        name = job_name(key, record["env"])
        if fields.get("JobName") != name:
            plans.append((key, record, fields.get("JobName"), name))
    sub.save()  # Persist the controller migration even if a later rename fails.
    for key, record, old_name, name in plans:
        command = ["scontrol", "update", f"JobId={record['id']}", f"JobName={name}"]
        if sub.args.dry_run:
            print(shlex.join(command), flush=True)
            continue
        subprocess.run(command, cwd=ROOT, check=True)
        record.setdefault("name_history", []).append(dict(previous=old_name, name=name,
            utc=datetime.now(timezone.utc).isoformat()))
        record["display_name"] = name
        sub.save()
        print(f"RENAMED {record['id']} {key} -> {name}", flush=True)
    print(f"READABLE_JOB_NAMES changed={len(plans)} resubmitted=0", flush=True)


def run_env(args, run, stage="A", resume=False):
    return dict(base_env(args), CELL=run["cell"], RATE_LAMBDA=run["rate_lambda"], SEED_INDEX=str(run["seed_index"]),
        PREPARATION_DIR=run["preparation"], FRESH_RUN_DIR=run.get("run_dir", ""), FRESH_STAGE=stage,
        FRESH_BUDGET=str(args.budget or 100000), FRESH_RESUME=str(resume).lower(), FRESH_EVAL_MODE="test",
        FRESH_TEST_AFTER_TRAIN=str(stage=="full").lower())


def require_gate0(args):
    if args.dry_run or not getattr(args, "require_gate0", False):
        return
    gate = read_json(args.root / "gate0/complete.json")
    if gate.get("passed") is not True or gate["code_sha256"] != {n:file_hash(ROOT/n) for n in runner.RUNTIME_FILES}:
        raise ValueError("Gate 0 did not pass for this runtime")
    if (gate["preparation_sha256"] != file_hash(args.preparation/"preparation.json")
            or gate["holdout_sha256"] != file_hash(args.preparation/"holdout.json")
            or gate["native_sha256"] != file_hash(args.native_checkpoint)
            or gate["dataset_root"] != str(args.dataset_root.resolve())):
        raise ValueError("Gate 0 inputs differ from this sweep")


def require_gate1(args):
    if not args.dry_run or args.state.exists():
        gate = read_json(args.root / "convergence/gate1.json")
        if gate.get("test_used") is not False or len(gate.get("runs", [])) != 15:
            raise ValueError("complete holdout-only Gate 1 evidence required")


def require_gate2(args, state):
    if args.dry_run:
        return
    decision = read_json(args.root / "gate2_decision.json")
    source = args.root / "convergence/curve_wave1_final.json"
    if (decision["decision"] != "proceed" or decision["budget"] != args.budget
            or decision["holdout_results_sha256"] != file_hash(source)):
        raise ValueError("Gate 2 proceed decision does not match the completed holdout evidence")


def submit_gate0(sub, args):
    jobs = []
    for i, task in enumerate(("initialization", "boundary", "forward", "native", "resume")):
        env = dict(base_env(args), GATE0_TASK=task)
        # The initializer check is CPU-only; the other four checks have separate GPU chains.
        script = SLURM + "gate0_nfcgs_fresh_rd.slurm"
        if task == "initialization":
            script = SLURM + "prepare_gate0_nfcgs_fresh_rd.slurm"
        jobs.append(sub.add("gate0/"+task, script, env))
    sub.add("gate0/collect", SLURM+"prepare_gate0_nfcgs_fresh_rd.slurm", dict(base_env(args), GATE0_TASK="collect"), dependencies=jobs)


def submit_wave1(sub, args):
    require_gate0(args)
    detach_diagnostic_lanes(sub)
    for lane, (cell, rate, seed) in enumerate(wave_runs(1)):
        key = run_key(cell, rate, seed)
        run = dict(key=key, cell=cell, rate_lambda=rate, seed_index=seed, wave=1, preparation=str(args.preparation))
        sub.add(key+"/A", SLURM+"train_nfcgs_fresh_rd.slurm", run_env(args, run), lane=lane, run=run)
    # Polling monitor starts only after every run path is durably present in state.
    sub.add("monitor/1/A", SLURM+"monitor_nfcgs_fresh_rd.slurm", dict(base_env(args), MONITOR_PHASE="A", MONITOR_WAVE="1"), lane=15)


def detach_diagnostic_lanes(sub):
    """Keep accepted diagnostic IDs, without making learners depend on them."""
    diagnostic_ids = {r["id"] for key,r in sub.state["jobs"].items() if key.startswith("gate0/")}
    for lane,job in list(sub.state["lanes"].items()):
        if job in diagnostic_ids:
            sub.state.setdefault("diagnostic_lanes", {})[lane] = job
            del sub.state["lanes"][lane]
    sub.save()


def adopt_inputs(state, inputs, allow_stage_b_controller_update=False):
    """An existing diagnostic-only submission may adopt the fixed launcher.

    Already accepted IDs remain immutable. Active sweeps may explicitly adopt
    this submission controller alone; training/evaluation runtime stays frozen.
    """
    if state.get("version") != VERSION:
        raise ValueError("unsupported submission state version")
    old = state.get("inputs", {})
    if old == inputs:
        return
    if {k:v for k,v in old.items() if k != "code"} != {k:v for k,v in inputs.items() if k != "code"}:
        raise ValueError("submission data/output/preparation paths differ")
    changes = sorted(k for k in set(old.get("code", {})) | set(inputs["code"])
                     if old.get("code", {}).get(k) != inputs["code"].get(k))
    if (allow_stage_b_controller_update and changes == ["scripts/submit_nfcgs_fresh_rd.py"]
            and not state.get("pending_submission")):
        state.setdefault("input_migrations", []).append(dict(previous_inputs=old,
            changed_files=changes, reason="submission controller update; training and evaluation runtime unchanged",
            utc=datetime.now(timezone.utc).isoformat()))
        state["inputs"] = inputs
        print("ADOPTING_STAGE_B_CONTROLLER learner_and_monitor_ids_preserved=True", flush=True)
        return
    if state.get("runs") or any(not key.startswith("gate0/") for key in state.get("jobs", {})) or state.get("pending_submission"):
        raise ValueError("runtime changed after learner submission; preserve this state and inspect the change")
    state.setdefault("input_migrations", []).append(dict(previous_inputs=old,
        changed_files=changes, reason="diagnostics decoupled before first learner submission",
        utc=datetime.now(timezone.utc).isoformat()))
    state["inputs"] = inputs
    print(f"ADOPTING_DIAGNOSTIC_ONLY_STATE preserved_ids={len(state.get('jobs', {}))}", flush=True)


def wave1_records(state):
    keys = [run_key(c,r,s) for c,r,s in wave_runs(1)]
    if any(key not in state["runs"] or key+"/A" not in state["jobs"] for key in keys):
        raise ValueError("wave 1 has not been fully submitted")
    return [state["runs"][key] for key in keys]


def gate1_evidence(args):
    require_gate1(args)
    if args.dry_run and not args.state.exists():
        return dict(recommended_budget=args.budget, evidence="dry-run; no decision made")
    path = args.root/"convergence/gate1.json"
    gate = read_json(path)
    expected = {(c,r,s) for c,r,s in wave_runs(1)}
    actual = {(r["cell"],runner.rate_text(r["lambda"]),r["seed"]) for r in gate["runs"]}
    if actual != expected or gate.get("recommended_budget") not in (100000,120000):
        raise ValueError("Gate 1 inventory or recommendation differs from this sweep")
    if args.budget != gate["recommended_budget"] and not args.reason:
        raise ValueError("a holdout-based --reason is required to override the Gate 1 recommendation")
    reason = args.reason or ("Gate 1 recommends 120k; 150k flag considered, choose 120k for the schedule"
                            if args.budget == 120000 and gate.get("consider_150k") else "follow holdout-only Gate 1 recommendation")
    return dict(gate1_sha256=file_hash(path), recommended_budget=gate["recommended_budget"],
        consider_150k=gate.get("consider_150k",False), chosen_budget=args.budget,
        reason=reason, test_used=False,
        utc=datetime.now(timezone.utc).isoformat())


def submit_provisional_stage_b(sub, args):
    if args.budget != 100000:
        raise ValueError("provisional Stage B is restricted to B=100000")
    require_gate0(args)
    if sub.state.get("budget") is not None:
        raise ValueError("the final budget is already frozen")
    runs = wave1_records(sub.state)
    provisional = sub.state.get("provisional_stage_b")
    if provisional and provisional.get("rollback"):
        raise ValueError("a budget rollback is already recorded; resume --restart-stage-b-from-60k")
    if provisional and provisional["status"] != "provisional":
        raise ValueError("provisional Stage B has already been decided")
    if any(key.endswith("/test") or key.startswith("monitor/1/final") for key in sub.state["jobs"]):
        raise ValueError("final evaluations already submitted; cannot mark Stage B provisional")
    if provisional is None:
        if any(r["key"]+"/B" in sub.state["jobs"] for r in runs):
            raise ValueError("existing Stage B jobs were not registered as provisional")
        if (args.root/"convergence/gate1.json").exists() and not args.dry_run:
            raise ValueError("Gate 1 is already available; submit ordinary Stage B with its decision")
        provisional = sub.state["provisional_stage_b"] = dict(status="provisional", budget=100000,
            source_step=60000, submitted_before_gate1=True, final_test_submitted=False,
            created_utc=datetime.now(timezone.utc).isoformat(), controller_sha256=file_hash(Path(__file__)))
        sub.save()
    for lane,run in enumerate(runs):
        sub.add(run["key"]+"/B", SLURM+"train_nfcgs_fresh_rd.slurm", run_env(args,run,"B"),
            dependencies=[sub.state["jobs"][run["key"]+"/A"]["id"]], lane=lane, run=run)
    sub.save()
    print("PROVISIONAL_STAGE_B=100000 learners=15 gate1_still_required=True final_test_submitted=False", flush=True)


def submit_stage_b(sub, args):
    if getattr(args,"provisional",False):
        return submit_provisional_stage_b(sub,args)
    require_gate0(args); require_gate1(args)
    if not args.budget:
        raise ValueError("an explicit --budget is required after Gate 1")
    if not sub.state.get("runs") and args.dry_run:
        submit_wave1(sub, args)
    runs = wave1_records(sub.state)
    provisional = sub.state.get("provisional_stage_b")
    if getattr(args,"restart_stage_b_from_60k",False):
        return restart_provisional_stage_b(sub,args)
    if provisional and provisional.get("rollback"):
        raise ValueError("a budget rollback is already recorded; resume --restart-stage-b-from-60k")
    if provisional and provisional["status"] == "provisional":
        if args.budget != 100000:
            raise ValueError("extension requires explicit --restart-stage-b-from-60k; provisional results are preserved")
        evidence = gate1_evidence(args)
        provisional.update(status="confirmed", decision=evidence)
        if not args.dry_run:
            atomic_json(args.root/"convergence/stage_b_budget_decision.json", evidence)
    elif provisional and provisional["status"] != "confirmed":
        raise ValueError("this provisional sweep was branched; use the recorded branch state/root")
    if sub.state.get("budget") not in (None, args.budget):
        raise ValueError("budget already frozen for this sweep; late extension needs a separate branch from 60k")
    sub.state["budget"] = args.budget
    sub.save()
    # The monitor has its own lane while fifteen train->test chains run.
    for lane, run in enumerate(runs):
        key = run["key"]
        train = sub.add(key+"/B", SLURM+"train_nfcgs_fresh_rd.slurm", run_env(args, run, "B"),
            dependencies=() if sub.state.get("stage_b_origin") else [sub.state["jobs"][key+"/A"]["id"]], lane=lane, run=run)
        sub.add(key+"/test", SLURM+"eval_nfcgs_fresh_rd.slurm", run_env(args, run, "B"), dependencies=[train], lane=lane)
    sub.add("monitor/1/final", SLURM+"monitor_nfcgs_fresh_rd.slurm",
        dict(base_env(args), MONITOR_PHASE="final", MONITOR_WAVE="1", FRESH_BUDGET=str(args.budget)), lane=15)


def verify_provisional_job(record):
    return verify_registered_job(record, SLURM+"train_nfcgs_fresh_rd.slurm")


def verify_registered_job(record, script):
    info = subprocess.run(["scontrol","show","job",record["id"],"-o"],
        cwd=ROOT,capture_output=True,text=True,check=True)
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)",info.stdout))
    token = next((v.split("=",1)[1] for v in record["command"] if v.startswith("--comment=")),None)
    if (not token or fields.get("JobId") != record["id"] or fields.get("Comment") != token
            or Path(fields.get("WorkDir","")).resolve() != ROOT.resolve()
            or Path(fields.get("Command","")).resolve() != (ROOT/script).resolve()):
        raise ValueError("refusing to change an unverified registered job")


def frozen_link(source, destination):
    """Fork an immutable Stage A artifact without moving any original output."""
    source,destination = Path(source),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if os.path.samefile(source,destination) or file_hash(source)==file_hash(destination):
            return
        raise ValueError(f"branch destination differs from the frozen source: {destination}")
    try:
        os.link(source,destination)
    except OSError:
        temporary=destination.with_name(destination.name+".forking_"+uuid.uuid4().hex)
        try:
            shutil.copy2(source,temporary)
            os.replace(temporary,destination)
        finally:
            temporary.unlink(missing_ok=True)


def restart_inputs(state, inputs):
    """Keep the running sweep's hashes frozen; approve only a new branch runtime."""
    old = state["inputs"]
    if {k:v for k,v in old.items() if k != "code"} != {k:v for k,v in inputs.items() if k != "code"}:
        raise ValueError("restart data/output/preparation paths differ")
    changed = sorted(k for k in set(old["code"]) | set(inputs["code"])
                     if old["code"].get(k) != inputs["code"].get(k))
    if set(changed) - RESTART_REPAIR_FILES:
        raise ValueError(f"runtime changes outside the 120k repair: {changed}")
    if state.get("pending_submission"):
        raise ValueError("resolve the outstanding submission before a runtime repair")
    return changed


def check_resaved_60k(path, original_sha, protocol):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    checks = []
    def require(condition, label):
        if not condition:
            raise ValueError(label)
        checks.append(label)
    require(saved.get("global_step") == 60000 and saved.get("epoch") == 7, "step_60000_epoch_7")
    schedulers = saved.get("lr_schedulers", [])
    require(len(schedulers) == 1, "one_full_scheduler")
    scheduler = schedulers[0]
    require(scheduler.get("last_epoch") == 60000, "scheduler_last_epoch_60000")
    require(Counter(scheduler.get("milestones", {})) == Counter([70000,90000]), "milestones_70000_90000")
    optimizers = saved.get("optimizer_states", [])
    require(len(optimizers) == 1, "one_full_optimizer")
    groups = optimizers[0].get("param_groups", [])
    bases = scheduler.get("base_lrs", [])
    require(bool(groups) and len(groups) == len(bases) and all(
        abs(float(group.get("lr", -1)) - .0001) < 1e-12 and abs(float(base) - .0001) < 1e-12
        for group,base in zip(groups,bases)), "all_optimizer_lrs_base_1e_4")
    fit = saved.get("loops", {}).get("fit_loop", {})
    processed = fit.get("epoch_loop.batch_progress", {}).get("total", {}).get("processed")
    require(processed == 239980, "processed_239980_exactly_original_plus_one")
    require(bool(fit.get("state_dict", {}).get("combined_loader")), "full_loader_state_present")
    require(saved.get("fresh_rd_protocol") == protocol, "saved_parent_protocol_unchanged")
    require(saved.get("fresh_rd_quantile_step") == 60000, "quantile_update_step_restored_60000")
    return checks


def check_superseded_60k(folder, original_sha, current):
    from monitor_nfcgs_fresh_rd import completed_rows
    rows_by_scene = {r["scene"]:r for r in current}
    compared = []
    for archive in sorted((folder/"validation").glob("step000060000.superseded_*")):
        done = read_json(archive/"complete.json")
        if done["checkpoint_sha256"] != original_sha:
            raise ValueError("superseded evaluation is not bound to the Stage A original")
        old_rows = completed_rows(archive)
        if {r["scene"] for r in old_rows} != set(rows_by_scene):
            raise ValueError("superseded evaluation scene cohort differs")
        for old in old_rows:
            new = rows_by_scene[old["scene"]]
            for field in ("context_frame_ids", "target_frame_ids", "mse", "lpips", "actual_bytes"):
                if old[field] != new[field]:
                    raise ValueError(f"superseded evaluation differs: {old['scene']} {field}")
        compared.append(dict(path=str(archive),scene_metrics_sha256=done["scene_metrics_sha256"],
                             max_mse_difference=0, max_lpips_difference=0, max_bytes_difference=0))
    return compared


def stage_a_branch_sources(sub,args):
    from monitor_nfcgs_fresh_rd import completed_rows
    from globalsplat.fresh_rd import checkpoint_path, paired_change
    sources=[]
    recalculated=[]
    inspect_existing = not args.dry_run or args.state.exists()
    gate = read_json(args.root/"convergence/gate1.json") if inspect_existing else None
    accept = getattr(args,"accept_resaved_60k",False)
    for run in wave1_records(sub.state):
        folder=Path(run["run_dir"])
        mode="stage_a_original";checks=[];archives=[];known={}
        if inspect_existing:
            if not folder.resolve().is_relative_to(args.root.resolve()):
                raise ValueError("Stage A source is outside the registered sweep")
            done=read_json(folder/"stage_A_complete.json")
            path=checkpoint_path(folder,60000)
            actual=file_hash(path)
            try:
                if done["step"]!=60000:
                    raise ValueError("Stage A completion step is not 60000")
                if actual != done["checkpoint_sha256"]:
                    if not accept:
                        raise ValueError("Stage A 60k checkpoint differs from its completion record; --accept-resaved-60k required")
                    checks=check_resaved_60k(path,done["checkpoint_sha256"],read_json(folder/"run.json"))
                    mode="resaved_after_first_microbatch"
                    known=dict(skipped_microbatches=1,skipped_scenes=2,first_resumed_optimizer_microbatches=3,
                               original_processed=239979,resaved_processed=239980)
                else:
                    checks.append("stage_a_original_sha_matches")
                before=completed_rows(folder/"validation/step000040000",checkpoint_path(folder,40000))
                after=completed_rows(folder/"validation/step000060000",path)
                checks.append("holdout_40k_and_60k_sha_and_metrics_verified")
                if mode=="resaved_after_first_microbatch":
                    archives=check_superseded_60k(folder,done["checkpoint_sha256"],after)
                    checks.append("superseded_metrics_equal" if archives else "no_superseded_evaluation_present")
                if accept:
                    recorded=next(r for r in gate["runs"] if (r["cell"],runner.rate_text(r["lambda"]),r["seed"])==
                                  (run["cell"],run["rate_lambda"],run["seed_index"]))
                    calculated=paired_change(before,after,run["rate_lambda"])
                    recalculated.append(dict(cell=run["cell"],**{"lambda":run["rate_lambda"]},**calculated))
                    for field,value in calculated.items():
                        if isinstance(value,float):
                            equal=math.isclose(recorded.get(field,float("nan")),value,rel_tol=1e-10,abs_tol=1e-12)
                        else:
                            equal=recorded.get(field)==value
                        if not equal:
                            raise ValueError(f"Gate 1 evidence differs from current paired scene metrics: {field}")
                    checks.append("gate1_recomputed_from_current_scene_metrics")
                checkpoint_hashes={str(step):file_hash(checkpoint_path(folder,step)) for step in range(10000,60001,10000)}
                digest=file_hash(folder/"run.json")
                old_runtime=read_json(folder/"run.json").get("code_sha256",{})
                if old_runtime:
                    runtime_changes={name for name in set(old_runtime)|set(runner.RUNTIME_FILES)
                                     if old_runtime.get(name)!=file_hash(ROOT/name)}
                    if runtime_changes-runner.RUNTIME_REPAIR_FILES:
                        raise ValueError(f"source run changed outside the declared runtime repair: {sorted(runtime_changes)}")
            except Exception as error:
                raise ValueError(f"{run['key']}: {error}; stop without automatic substitution. Inspect the original 50k checkpoint for a separate continuation plan.") from error
        else:
            done=dict(checkpoint_sha256="dry-run")
            actual="dry-run";digest="dry-run";checkpoint_hashes={}
        source=dict(run=copy.deepcopy(run),checkpoint_60k_sha256=actual,
            original_stage_a_sha256=done["checkpoint_sha256"],source_60k_mode=mode,
            passed_checks=checks,superseded_comparisons=archives,known_differences=known,
            checkpoints_sha256=checkpoint_hashes,run_manifest_sha256=digest,
            gate1_sha256=file_hash(args.root/"convergence/gate1.json") if inspect_existing else "dry-run")
        sources.append(source)
        print(f"SOURCE_60K={run['key']} mode={mode} sha256={actual} original_sha256={done['checkpoint_sha256']} checks={','.join(checks)} known_differences={json.dumps(known,sort_keys=True)}",flush=True)
    if recalculated:
        from globalsplat.fresh_rd import gate1_decision
        decision=gate1_decision(recalculated)
        for key in ("recommended_budget","consider_150k","largest_significant_delta_db_j"):
            if decision[key]!=gate.get(key):
                raise ValueError(f"Gate 1 decision differs from current metrics: {key}; no learners stopped")
    return sources


def restart_provisional_stage_b(sub,args):
    provisional=sub.state.get("provisional_stage_b")
    if not provisional or provisional["status"] not in ("provisional","branched") or args.budget not in (120000,150000):
        raise ValueError("only undecided provisional 100k jobs may restart from 60k at B=120k/150k")
    if sub.state.get("budget") is not None or any(k.endswith("/test") for k in sub.state["jobs"]):
        raise ValueError("a confirmed/tested budget cannot be replaced by this procedure")
    evidence=gate1_evidence(args)
    print(f"GATE1_CHOICE budget={args.budget} recommended={evidence.get('recommended_budget')} consider_150k={evidence.get('consider_150k',False)} reason={evidence.get('reason','synthetic dry-run')}",flush=True)
    sources=stage_a_branch_sources(sub,args)
    current_inputs=getattr(args,"current_inputs",dict(sub.state["inputs"],code={n:file_hash(ROOT/n) for n in FILES}))
    changed_files=restart_inputs(sub.state,current_inputs)
    branch_root=args.root/"branches"/f"B{args.budget}"
    branch_path=args.state.with_name(f"{args.state.stem}_B{args.budget}{args.state.suffix}")
    if not branch_root.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("budget branch escaped the sweep root")
    rollback=provisional.get("rollback")
    if rollback:
        if rollback["budget"]!=args.budget or rollback["root"]!=str(branch_root) or rollback["state"]!=str(branch_path):
            raise ValueError("another budget rollback is already recorded")
        if rollback["decision"].get("gate1_sha256")!=evidence.get("gate1_sha256"):
            raise ValueError("Gate 1 evidence changed during rollback")
        if rollback["source_runs"]!=sources:
            raise ValueError("frozen Stage A lineage changed during rollback")
    else:
        if branch_path.exists() or branch_root.exists():
            raise ValueError("unregistered budget branch exists; refusing to overwrite it")
        rollback=provisional["rollback"]=dict(budget=args.budget,root=str(branch_root),state=str(branch_path),
            decision=evidence,source_runs=sources,job_states={},status="preparing",
            branch_inputs=dict(current_inputs,root=str(branch_root.resolve())),
            changed_files=changed_files,
            utc=datetime.now(timezone.utc).isoformat())
        sub.save()  # rollback intent is durable before any cancellation or copying
    if rollback.get("branch_inputs",{}).get("code",current_inputs["code"]) != current_inputs["code"]:
        raise ValueError("branch runtime changed during the recorded rollback")
    records=[]
    for source in sources:
        key=source["run"]["key"]+"/B"
        if key not in sub.state["jobs"]:
            continue  # a partially submitted provisional wave may also be decided
        record=sub.state["jobs"][key]
        env=record["env"]
        if (record["command"][-1]!=SLURM+"train_nfcgs_fresh_rd.slurm" or env.get("FRESH_STAGE")!="B"
                or env.get("FRESH_BUDGET")!="100000" or env.get("FRESH_RESUME") not in ("false","true")
                or env.get("FRESH_RUN_DIR")!=source["run"]["run_dir"]):
            raise ValueError("job is not an original provisional 100k Stage B allocation")
        inspect_existing=not args.dry_run or args.state.exists()
        status=job_state(record["id"]) if inspect_existing else "DRY_RUN"
        if inspect_existing and status not in TERMINAL_FAILURES|{"COMPLETED"}:
            verify_provisional_job(record)
        records.append((key,record,status))
    if not args.dry_run or args.state.exists():
        for source in sources:
            if (file_hash(checkpoint_path(source["run"]["run_dir"],60000)) != source["checkpoint_60k_sha256"]
                    or file_hash(Path(source["run"]["run_dir"])/"run.json")!=source["run_manifest_sha256"]):
                raise ValueError("60k checkpoint changed during preflight; no learner stopped")
    for key,record,status in records:
        journal=rollback["job_states"].setdefault(key,dict(id=record["id"],before=status))
        sub.save()
        if args.dry_run:
            action="WOULD_KEEP_COMPLETED" if status in TERMINAL_FAILURES|{"COMPLETED"} else "WOULD_STOP_PROVISIONAL_ONLY"
            print(f"{action}={record['id']} state={status} preserve_original_outputs=True",flush=True)
        elif status not in TERMINAL_FAILURES|{"COMPLETED"}:
            status=job_state(record["id"])
            if status not in TERMINAL_FAILURES|{"COMPLETED"}:
                try:
                    subprocess.run(["scancel",record["id"]],cwd=ROOT,check=True)
                except subprocess.CalledProcessError:
                    if job_state(record["id"]) not in TERMINAL_FAILURES|{"COMPLETED"}:
                        raise
            for _ in range(60):
                status=job_state(record["id"])
                if status in TERMINAL_FAILURES|{"COMPLETED"}:
                    break
                time.sleep(1)
            else:
                raise ValueError("provisional Stage B has not stopped; original outputs remain intact")
        journal["after"]=status
        sub.save()

    branch_args=copy.copy(args)
    branch_args.root,branch_args.state=branch_root,branch_path
    branch_args.provisional=False;branch_args.restart_stage_b_from_60k=False
    branch_args.accept_resaved_60k=False
    provenance=dict(source_state=str(args.state),source_root=str(args.root),budget=args.budget,
        decision=rollback["decision"],source_runs=sources,provisional_outputs_preserved=True)
    for source in sources:
        folder=Path(source["run"]["run_dir"])
        target=branch_root/source["run"]["key"]/folder.name
        for step in range(10000,60001,10000):
            print(f"{'WOULD_LINK' if args.dry_run else 'FORK_FILE'}={checkpoint_path(folder,step)} -> {checkpoint_path(target,step)}",flush=True)
            if args.dry_run:
                for relative in ("complete.json","invocation.json","nfcgs_main/scene_metrics.json"):
                    path=folder/"validation"/f"step{step:09d}"/relative
                    if path.exists():
                        print(f"WOULD_LINK={path} -> {target/'validation'/f'step{step:09d}'/relative}",flush=True)
        print(f"BRANCH_PROTOCOL={target/'run.json'} original_manifest_preserved={target/'source_run.json'}",flush=True)
    if not args.dry_run:
        marker=branch_root/"branch_provenance.json"
        if marker.exists() and read_json(marker)!=provenance:
            raise ValueError("budget branch provenance differs from the registered rollback")
        atomic_json(marker,provenance)
        for source in sources:
            run=source["run"];folder=Path(run["run_dir"])
            target=branch_root/run["key"]/folder.name
            if (file_hash(checkpoint_path(folder,60000))!=source["checkpoint_60k_sha256"]
                    or file_hash(folder/"run.json")!=source["run_manifest_sha256"]):
                raise ValueError("60k source changed after stopping provisional learners; no branch learner submitted")
            original_protocol=read_json(folder/"run.json")
            new_runtime={name:file_hash(ROOT/name) for name in runner.RUNTIME_FILES}
            old_runtime=original_protocol.get("code_sha256",{})
            differences=sorted(name for name in set(old_runtime)|set(new_runtime)
                               if old_runtime.get(name)!=new_runtime.get(name)) if old_runtime else []
            if set(differences)-runner.RUNTIME_REPAIR_FILES:
                raise ValueError(f"run runtime differs outside the repair: {differences}")
            if differences:
                frozen_link(folder/"run.json",target/"source_run.json")
                parent=dict(protocol=original_protocol,checkpoints_sha256=source["checkpoints_sha256"],
                    changed_runtime_files=differences,source_60k_mode=source["source_60k_mode"],
                    original_stage_a_sha256=source["original_stage_a_sha256"],known_differences=source["known_differences"])
                new_protocol=dict(original_protocol,code_sha256=new_runtime,resume_parent=parent)
                for name,body in (("resume_parent.json",parent),("run.json",new_protocol)):
                    path=target/name
                    if path.exists() and read_json(path)!=body:
                        raise ValueError("branch runtime protocol differs from the recorded parent")
                    if not path.exists():atomic_json(path,body)
            else:
                frozen_link(folder/"run.json",target/"run.json")
            for step in range(10000,60001,10000):
                path=checkpoint_path(folder,step)
                if path.exists():
                    frozen_link(path,checkpoint_path(target,step))
                for relative in ("complete.json","invocation.json","nfcgs_main/scene_metrics.json"):
                    path=folder/"validation"/f"step{step:09d}"/relative
                    if path.exists():
                        print(f"FORK_FILE={path} -> {target/'validation'/f'step{step:09d}'/relative}",flush=True)
                        frozen_link(path,target/"validation"/f"step{step:09d}"/relative)
            # Never fork last.ckpt, provisional 70k+ files, or Stage B completion markers.
            protected=runner.freeze_stage_a_checkpoint(checkpoint_path(target,60000))
            atomic_json(target/"stage_a_lineage.json",source)
            atomic_json(target/"stage_A_complete.json",dict(step=60000,budget=100000,
                checkpoint=str(checkpoint_path(target,60000)),checkpoint_sha256=source["checkpoint_60k_sha256"],
                source_60k_mode=source["source_60k_mode"],**protected))
        for name in ("gate1.json","curve_wave1_A.json","wave1_A_complete.json"):
            path=args.root/"convergence"/name
            if path.exists():
                frozen_link(path,branch_root/"convergence"/name)
        cohort=args.root/"test_frame_cohort.json"
        if cohort.exists():
            frozen_link(cohort,branch_root/cohort.name)
        atomic_json(branch_root/"convergence/stage_b_budget_decision.json",rollback["decision"])
    if branch_path.exists():
        branch=read_json(branch_path)
        if branch.get("stage_b_origin")!=provenance or branch.get("pending_submission"):
            raise ValueError(f"branch provenance/pending intent requires inspection: {branch_path}")
    else:
        keys={r["run"]["key"]+"/A" for r in sources}|{"monitor/1/A"}
        branch=dict(version=VERSION,inputs=dict(current_inputs,root=str(branch_root.resolve())),
            budget=args.budget,jobs={k:copy.deepcopy(v) for k,v in sub.state["jobs"].items() if k in keys},
            runs={},lanes={},stage_b_origin=provenance)
        # All Stage A files were validated above. Do not introduce new Slurm
        # dependencies on historical A/monitor IDs which may exceed MinJobAge.
        for source in sources:
            run=copy.deepcopy(source["run"])
            run["run_dir"]=str(branch_root/run["key"]/Path(run["run_dir"]).name)
            branch["runs"][run["key"]]=run
    branch_sub=Submission(branch_args,branch)
    branch_sub.save()
    submit_stage_b(branch_sub,branch_args)
    rollback["status"]="submitted"
    provisional["status"]="branched"
    sub.state["active_budget_branch"]=dict(root=str(branch_root),state=str(branch_path),budget=args.budget)
    sub.save()
    print(f"ACTIVE_BUDGET_BRANCH state={branch_path} root={branch_root} budget={args.budget}",flush=True)


def submit_wave2(sub, args):
    if not args.budget:
        raise ValueError("wave 2 requires --budget")
    require_gate0(args); require_gate2(args, sub.state)
    if sub.state.get("budget") not in (None, args.budget):
        raise ValueError("all core runs must use the same frozen budget")
    choices = list(map(runner.rate_text, args.msh_lambdas))
    if "msh_lambdas" in sub.state and (sub.state["msh_lambdas"] != choices or sub.state["omit_seeds"] != args.omit_seeds):
        raise ValueError("wave-2 choices already frozen")
    sub.state.update(budget=args.budget, msh_lambdas=choices, omit_seeds=args.omit_seeds)
    sub.save()
    if args.budget == 120000:
        return submit_wave2_continuous(sub,args,choices)
    for i, (cell, rate, seed) in enumerate(wave_runs(2, choices, args.omit_seeds)):
        key = run_key(cell, rate, seed)
        prep = args.preparation if seed == 0 else runner.preparation_for(seed)
        if not args.dry_run and not (prep/"preparation.json").is_file():
            raise ValueError(f"seed preparation missing: {prep}")
        run = dict(key=key, cell=cell, rate_lambda=rate, seed_index=seed, wave=2, preparation=str(prep))
        a = sub.add(key+"/A", SLURM+"train_nfcgs_fresh_rd.slurm", run_env(args, run), lane=i%16, run=run)
        run = sub.state["runs"][key]
        b = sub.add(key+"/B", SLURM+"train_nfcgs_fresh_rd.slurm", run_env(args, run, "B"), dependencies=[a], lane=i%16, run=run)
        sub.add(key+"/test", SLURM+"eval_nfcgs_fresh_rd.slurm", run_env(args, run, "B"), dependencies=[b], lane=i%16)


def submit_wave2_continuous(sub,args,choices):
    """Twenty allocations, each training through 120k then evaluating in place."""
    rows=wave_runs(2,choices,args.omit_seeds)
    core=[r for r in rows if r[0] in ("h_on","h_off","n_on","n_off")]
    optional=[r for r in rows if r[0]=="msh"]+[r for r in rows if r[0] in ("m_on","h_l3")]
    rows=core+optional
    if len(core)!=16 or len(rows)!=20:
        raise ValueError("the 120k continuous Wave 2 plan requires sixteen core and four optional runs")
    if not args.dry_run:
        for _,_,seed in rows:
            prep=args.preparation if seed==0 else runner.preparation_for(seed)
            if not (prep/"preparation.json").is_file():
                raise ValueError(f"seed preparation missing: {prep}")
    if not args.dry_run or args.state.exists():
        historical={r["id"] for key,r in sub.state["jobs"].items()
                    if key.startswith("monitor/1/") or (r.get("run") or {}).get("wave")==1}
        for lane,job in list(sub.state["lanes"].items()):
            if job in historical and job_state(job) in TERMINAL_FAILURES|{"COMPLETED"}:
                sub.state.setdefault("finished_wave1_lanes",{})[lane]=job
                del sub.state["lanes"][lane]
    execution=dict(kind="single_job_train_then_test",budget=120000,
        main_keys=[run_key(*r) for r in core],optional_keys=[run_key(*r) for r in optional],
        optional_after="all sixteen core train+test allocations complete",priority_order=[run_key(*r) for r in rows])
    if "wave2_execution" in sub.state:
        if any(sub.state["wave2_execution"].get(k)!=v for k,v in execution.items()):
            raise ValueError("the registered Wave 2 execution plan differs")
    else:
        sub.state["wave2_execution"]=execution
    sub.save()
    front=[];all_jobs=[]
    for i,(cell,rate,seed) in enumerate(rows):
        key=run_key(cell,rate,seed)
        if key+"/A" in sub.state["jobs"] or key+"/B" in sub.state["jobs"]:
            raise ValueError("existing split Wave 2 jobs cannot be silently replaced by continuous jobs")
        prep=args.preparation if seed==0 else runner.preparation_for(seed)
        if not args.dry_run and not (prep/"preparation.json").is_file():
            raise ValueError(f"seed preparation missing: {prep}")
        run=dict(key=key,cell=cell,rate_lambda=rate,seed_index=seed,wave=2,preparation=str(prep),
                 execution="continuous_0_to_120k_then_test",optional=i>=16)
        env=dict(run_env(args,run,"full"),FRESH_TEST_AFTER_TRAIN="true")
        job=sub.add(key+"/full",SLURM+"train_nfcgs_fresh_rd.slurm",env,
                    dependencies=front if i>=16 else (),lane=i%16,run=run)
        all_jobs.append(job)
        if i<16:front.append(job)
    dependencies=all_jobs
    if not args.dry_run or args.state.exists():
        # The monitor may be the last quota-deferred submission. Successful
        # parents already recorded by sacct need no new afterok reference.
        dependencies=[job for job in all_jobs if job_state(job)!="COMPLETED"]
    sub.add("monitor/2/final",SLURM+"monitor_nfcgs_fresh_rd.slurm",
        dict(base_env(args),MONITOR_PHASE="final",MONITOR_WAVE="2",FRESH_BUDGET="120000"),
        dependencies=dependencies,lane=None)
    print("WAVE2_CONTINUOUS trainings=20 bundled_tests=20 jobs=21 optional_after_core=4",flush=True)


def cancel_wave2_optional(sub,args):
    plan=sub.state.get("wave2_execution",{})
    if plan.get("kind")!="single_job_train_then_test" or plan.get("budget")!=120000:
        raise ValueError("optional cancellation requires the continuous 120k Wave 2 plan")
    core=[sub.state["jobs"][key+"/full"] for key in plan["main_keys"]]
    optional=[sub.state["jobs"][key+"/full"] for key in plan["optional_keys"]]
    if len(core)!=16 or len(optional)!=4:
        raise ValueError("unexpected Wave 2 core/optional inventory")
    if any(job_state(r["id"])!="COMPLETED" for r in core):
        raise ValueError("wait for all sixteen core train+test jobs before dropping the optional four")
    monitor=sub.state["jobs"]["monitor/2/final"]
    resuming=(plan.get("optional_cancellation",{}).get("monitor_id")==monitor["id"]
              and set(plan.get("optional_cancelled_keys",[]))==set(plan["optional_keys"]))
    monitor_status=job_state(monitor["id"])
    if monitor_status!="PENDING" and not resuming:
        raise ValueError("final monitor must still be pending for an optional-cohort change")
    if monitor_status=="PENDING":verify_registered_job(monitor,SLURM+"monitor_nfcgs_fresh_rd.slurm")
    statuses=[]
    for record in optional:
        if record["env"].get("FRESH_STAGE")!="full" or record["env"].get("FRESH_BUDGET")!="120000":
            raise ValueError("optional allocation does not belong to the 120k plan")
        status=job_state(record["id"])
        if status not in TERMINAL_FAILURES|{"COMPLETED"}:verify_provisional_job(record)
        statuses.append(status)
    if args.dry_run:
        for record,status in zip(optional,statuses):
            print(f"WOULD_STOP_OPTIONAL={record['id']} state={status}",flush=True)
        print("WOULD_SET_FINAL_MONITOR_COHORT=16 dependencies=none core_completed="+":".join(r["id"] for r in core),flush=True)
        return
    plan.setdefault("optional_cancellation",dict(ids=[r["id"] for r in optional],
        monitor_id=monitor["id"],core_completed_ids=[r["id"] for r in core],
        reason=args.reason or "schedule cutoff after sixteen core Wave 2 jobs",utc=datetime.now(timezone.utc).isoformat()))
    plan["optional_cancelled_keys"]=list(plan["optional_keys"])
    sub.save()
    # First remove the cancelled parents from the monitor's dependency. Some
    # sites kill invalid dependent jobs automatically when a parent is cancelled.
    # The sixteen required allocations were all verified COMPLETED above, so no
    # new afterok references to potentially expired IDs are needed.
    if monitor_status=="PENDING":
        subprocess.run(["scontrol","update",f"JobId={monitor['id']}","Dependency="],cwd=ROOT,check=True)
    monitor["dependencies"]=[];plan["optional_cancellation"]["monitor_dependency_cleared"]=True;sub.save()
    for record,status in zip(optional,statuses):
        if status not in TERMINAL_FAILURES|{"COMPLETED"}:
            subprocess.run(["scancel",record["id"]],cwd=ROOT,check=True)
            for _ in range(60):
                if job_state(record["id"]) in TERMINAL_FAILURES|{"COMPLETED"}:break
                time.sleep(1)
            else:raise ValueError("optional job has not stopped; retry this command")
    plan["optional_cancellation"]["complete"]=True;sub.save()
    print("WAVE2_OPTIONAL_CANCELLED=4 core_jobs_preserved=16 final_monitor_cohort=16",flush=True)


def record_gate2(args, state):
    if not args.budget or not args.reason:
        raise ValueError("Gate 2 needs --budget and a brief --reason based on holdout")
    complete = read_json(args.root / "convergence/wave1_final_complete.json")
    if complete["budget"] != args.budget or complete["run_count"] != 15 or complete["test_used"]:
        raise ValueError("wave-1 final holdout evaluation is incomplete")
    source = args.root / "convergence/curve_wave1_final.json"
    rows = read_json(source)
    if len(rows) != 15 or any(r["scene_count"] != 128 for r in rows):
        raise ValueError("fifteen final holdout rows required")
    atomic_json(args.root / "gate2_decision.json", dict(decision=args.gate2, budget=args.budget,
        reason=args.reason, holdout_results_sha256=file_hash(source), test_used=False))


def submit_monitor(sub, args):
    if not args.wave or not args.budget:
        raise ValueError("monitor submission requires --wave and --budget")
    excluded=set(sub.state.get("wave2_execution",{}).get("optional_cancelled_keys",[])) if args.wave==2 else set()
    dependencies = [v["id"] for k,v in sub.state["jobs"].items()
        if k.endswith(("/test","/full")) and k.rsplit("/",1)[0] not in excluded
        and sub.state["runs"][k.rsplit("/",1)[0]]["wave"] == args.wave]
    if not dependencies and not args.dry_run:
        raise ValueError("requested wave has no submitted test chains")
    sub.add(f"monitor/{args.wave}/{args.monitor}", SLURM+"monitor_nfcgs_fresh_rd.slurm",
        dict(base_env(args), MONITOR_PHASE=args.monitor, MONITOR_WAVE=str(args.wave), FRESH_BUDGET=str(args.budget)),
        dependencies=dependencies, lane=15)


def resume_run(sub, args):
    key = args.resume_run
    if key not in sub.state["runs"] or args.stage not in ("A", "B", "full"):
        raise ValueError("resume requires an existing run key and --stage A|B")
    run = sub.state["runs"][key]
    if args.budget is None:
        args.budget = sub.state.get("budget", 100000)
    job_key = key+"/"+args.stage
    previous = sub.state["jobs"][job_key]
    if not args.dry_run and job_state(previous["id"]) not in TERMINAL_FAILURES:
        raise ValueError("resume only replaces a failed/inactive training job")
    # A retry replaces its own failed dependency, so it must not depend on the lane tail.
    new = sub.add(job_key, SLURM+"train_nfcgs_fresh_rd.slurm", run_env(args, run, args.stage, True), run=run, replace_failed=True)
    repairs = sub.state.setdefault("dependency_repairs", [])
    for child_key, child in list(sub.state["jobs"].items()):
        if child_key == job_key or previous["id"] not in child["dependencies"]:
            continue
        dependencies = [new if d == previous["id"] else d for d in child["dependencies"]]
        command = ["scontrol", "update", f"JobId={child['id']}", "Dependency=afterok:"+":".join(dependencies)]
        repairs.append(dict(child=child_key, old=previous["id"], new=new, command=command, complete=False))
    sub.save()
    repair_dependencies(sub)


def repair_dependencies(sub):
    for item in sub.state.get("dependency_repairs", []):
        if item["complete"]:
            continue
        if sub.args.dry_run:
            print(shlex.join(item["command"]))
        else:
            state = job_state(sub.state["jobs"][item["child"]]["id"])
            if state != "PENDING":
                raise ValueError("dependent job is not pending; inspect it before explicit replacement")
            subprocess.run(item["command"], check=True, cwd=ROOT)
        child = sub.state["jobs"][item["child"]]
        child["dependencies"] = [item["new"] if d == item["old"] else d for d in child["dependencies"]]
        item["complete"] = True
        sub.save()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate0", action="store_true")
    p.add_argument("--require-gate0", action="store_true", help="optional strict preflight; training otherwise starts independently")
    p.add_argument("--wave", type=int, choices=(1, 2))
    p.add_argument("--stage-b", action="store_true")
    p.add_argument("--provisional", action="store_true", help="pre-submit B=100k after each Stage A; Gate 1 still decides the final budget")
    p.add_argument("--restart-stage-b-from-60k", action="store_true", help="after Gate 1, preserve provisional outputs and fork B=120k/150k from 60k")
    p.add_argument("--accept-resaved-60k", action="store_true", help="for 120k rollback only: strictly validate the first-microbatch resave and record its known resume difference")
    p.add_argument("--monitor", choices=("A", "final", "appendix"))
    p.add_argument("--budget", type=int, choices=(100000, 120000, 150000))
    p.add_argument("--msh-lambdas", nargs=2, default=(".0064", ".0128"))
    p.add_argument("--omit-seeds", action="store_true", help="explicitly reduce the 150k plan to 27 runs")
    p.add_argument("--cancel-wave2-optional", action="store_true", help="after sixteen core jobs finish, cancel only the optional four and reconnect the final monitor")
    p.add_argument("--gate2", choices=("proceed", "hold"))
    p.add_argument("--reason")
    p.add_argument("--resume-run", help="run key, e.g. h_on/lambda0p0128/s0")
    p.add_argument("--stage", choices=("A", "B", "full"))
    p.add_argument("--repair-dependencies", action="store_true")
    p.add_argument("--rename-active-jobs", action="store_true", help="give registered live jobs readable names without resubmitting them")
    p.add_argument("--record-accepted", metavar="JOB_ID", help="recover the single outstanding sbatch intent")
    p.add_argument("--root", type=Path, default=Path(os.environ.get("FRESH_RD_ROOT", ROOT/"outputs/nfcgs_fresh_rd")))
    p.add_argument("--state", type=Path, default=ROOT/"logs/slurm/fresh_rd/state.json")
    p.add_argument("--preparation", type=Path, default=runner.preparation_for(0))
    p.add_argument("--native-checkpoint", type=Path, default=ROOT/"checkpoints/pretrained/globalsplat-re10k-32k.ckpt")
    p.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("DATASET_ROOT", "/data3/local_datasets/re10k")))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.stage_b and args.wave != 1:
        p.error("--stage-b requires --wave 1")
    if (args.provisional or args.restart_stage_b_from_60k) and not args.stage_b:
        p.error("provisional/restart options require --stage-b --wave 1")
    if args.provisional and args.restart_stage_b_from_60k:
        p.error("choose provisional submission or holdout-based restart, not both")
    if args.provisional and args.budget != 100000:
        p.error("--provisional requires --budget 100000")
    if args.restart_stage_b_from_60k and args.budget not in (120000,150000):
        p.error("--restart-stage-b-from-60k requires --budget 120000 or 150000")
    if args.accept_resaved_60k and (not args.restart_stage_b_from_60k or args.budget!=120000):
        p.error("--accept-resaved-60k requires --restart-stage-b-from-60k --budget 120000")
    if args.omit_seeds and args.budget != 150000:
        p.error("seed omission is explicit and only supported for the 150k plan")
    for name in ("root","state","preparation","native_checkpoint","dataset_root"):
        setattr(args,name,getattr(args,name).resolve())
    inputs = dict(root=str(args.root.resolve()), dataset=str(args.dataset_root.resolve()), preparation=str(args.preparation.resolve()),
        native_checkpoint=str(args.native_checkpoint.resolve()), code={n:file_hash(ROOT/n) for n in FILES})
    args.current_inputs=inputs
    def work():
        state = read_json(args.state) if args.state.exists() else dict(version=VERSION, inputs=inputs, jobs={}, runs={}, lanes={})
        if args.restart_stage_b_from_60k:
            restart_inputs(state,inputs)
        else:
            adopt_inputs(state, inputs, allow_stage_b_controller_update=args.stage_b or args.rename_active_jobs)
        sub = Submission(args, state)
        if state.get("provisional_stage_b",{}).get("rollback") and not args.restart_stage_b_from_60k:
            branch=state["provisional_stage_b"]["rollback"]
            raise ValueError(f"budget rollback recorded; resume it or use --state {branch['state']} --root {branch['root']}")
        if state.get("active_budget_branch") and not args.restart_stage_b_from_60k:
            branch=state["active_budget_branch"]
            raise ValueError(f"continue the decided branch with --state {branch['state']} --root {branch['root']}")
        if args.record_accepted:
            intent = state.get("pending_submission")
            if not intent or not re.fullmatch(r"[1-9][0-9]*", args.record_accepted):
                raise ValueError("no recoverable intent or invalid ID")
            info = subprocess.run(["scontrol", "show", "job", args.record_accepted, "-o"], capture_output=True, text=True, check=True)
            if "Comment="+intent["token"] not in info.stdout:
                raise ValueError("scheduler comment does not match the outstanding submission intent")
            sub.accept(intent["key"], args.record_accepted, intent["payload"])
        elif state.get("pending_submission"):
            raise ValueError("unfinished submission intent: inspect Slurm comment and recover using --record-accepted ID")
        elif args.rename_active_jobs:
            rename_active_jobs(sub)
        elif args.cancel_wave2_optional:
            cancel_wave2_optional(sub,args)
        elif args.repair_dependencies:
            repair_dependencies(sub)
        elif args.resume_run:
            resume_run(sub, args)
        elif args.gate2:
            if not args.dry_run:
                record_gate2(args, state)
        elif args.gate0:
            submit_gate0(sub, args)
        elif args.monitor:
            submit_monitor(sub, args)
        elif args.wave == 1:
            submit_stage_b(sub, args) if args.stage_b else submit_wave1(sub, args)
        elif args.wave == 2:
            submit_wave2(sub, args)
        elif args.dry_run:
            print("PLAN: 15 wave-1 + 20 wave-2 trainings, each A -> B -> test; Gate 2 precedes wave 2.")
            submit_gate0(sub, args); submit_wave1(sub, args)
            args.budget = args.budget or 100000
            submit_stage_b(sub, args); submit_wave2(sub, args)
        else:
            p.error("choose --gate0, --wave, --stage-b, --gate2 or --resume-run")
    if args.dry_run:
        work()
    else:
        with directory_lock(args.state.parent, ".submit.lock"):
            work()


if __name__ == "__main__":
    main()

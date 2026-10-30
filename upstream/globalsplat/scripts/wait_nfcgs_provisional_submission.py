#!/usr/bin/env python3
"""Retry provisional 100k or explicit 120k rollback submission on Slurm quota errors.

Runs on the login node, not as a Slurm allocation. The existing controller
retains its journal, locking and accepted-ID reuse on every attempt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
QUOTA_ERROR="QOSMaxSubmitJobPerUserLimit"


def submission_command(state_path, restart_120k=False):
    state_path=Path(state_path).resolve()
    state=json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("version")!="nfcgs_fresh_rd_v1":
        raise ValueError("unsupported fresh RD state")
    rollback=state.get("provisional_stage_b",{}).get("rollback",{})
    if restart_120k and rollback and rollback.get("budget")!=120000:
        raise ValueError("another extended budget was already selected")
    if not restart_120k and (state.get("active_budget_branch") or rollback):
        raise ValueError("the sweep already selected an extended budget branch")
    inputs=state["inputs"]
    options=(["--budget","120000","--restart-stage-b-from-60k","--accept-resaved-60k"]
             if restart_120k else ["--budget","100000","--provisional"])
    return [sys.executable,str(ROOT/"scripts/submit_nfcgs_fresh_rd.py"),
        "--stage-b","--wave","1",*options,
        "--state",str(state_path),"--root",inputs["root"],"--dataset-root",inputs["dataset"],
        "--preparation",inputs["preparation"],"--native-checkpoint",inputs["native_checkpoint"]]


def wait_for_submission(state_path, interval, restart_120k=False):
    if interval<=0:
        raise ValueError("retry interval must be positive")
    attempt=0
    while True:
        # The child reloads the durable state and releases .submit.lock before
        # this supervisor sleeps, allowing other submission/recovery operations.
        command=submission_command(state_path,restart_120k)
        attempt+=1
        label="RESTART_120K_SUBMIT_ATTEMPT" if restart_120k else "PROVISIONAL_SUBMIT_ATTEMPT"
        print(f"{label}={attempt} utc={datetime.now(timezone.utc).isoformat()}",flush=True)
        print(shlex.join(command),flush=True)
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        if result.stdout:
            print(result.stdout,end="" if result.stdout.endswith("\n") else "\n",flush=True)
        if result.stderr:
            print(result.stderr,end="" if result.stderr.endswith("\n") else "\n",file=sys.stderr,flush=True)
        if result.returncode==0:
            print(("RESTART_120K_SUBMISSION_COMPLETE" if restart_120k else "PROVISIONAL_SUBMISSION_COMPLETE")+" existing_ids_reused=True",flush=True)
            return 0
        if QUOTA_ERROR not in result.stderr:
            print("SUBMISSION_STOPPED non_quota_error=True inspect_output_above=True",file=sys.stderr,flush=True)
            return result.returncode if result.returncode>0 else 1
        print(f"WAITING_FOR_SUBMIT_SLOT retry_in_seconds={interval:g} mode={'restart120k' if restart_120k else 'provisional'}",flush=True)
        time.sleep(interval)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state",type=Path,default=ROOT/"logs/slurm/fresh_rd/state.json")
    parser.add_argument("--interval",type=float,default=600)
    parser.add_argument("--restart-120k",action="store_true",help="retry the authorized 120k/resaved-60k branch submission only on QOS submit-quota errors")
    args=parser.parse_args(argv)
    if args.interval<=0:
        parser.error("--interval must be positive")
    return wait_for_submission(args.state,args.interval,args.restart_120k)


if __name__=="__main__":
    raise SystemExit(main())

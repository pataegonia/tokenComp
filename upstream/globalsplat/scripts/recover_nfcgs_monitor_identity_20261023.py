#!/usr/bin/env python3
"""Audit changed Stage A checkpoints and replace only the failed monitor.

Does not modify checkpoints, stop learners, or waive full-state lineage checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
import submit_nfcgs_fresh_rd as submit
from recover_nfcgs_fresh_start_20261021 import active_monitor_jobs
from globalsplat.fresh_rd import VERSION,atomic_json,checkpoint_path,directory_lock,file_hash,read_json

MONITOR="scripts/monitor_nfcgs_fresh_rd.py"


def audit_checkpoints(state,root,output):
    rows=[]
    for key,run in sorted(state["runs"].items()):
        if run["wave"]!=1:continue
        folder=Path(run["run_dir"])
        if not folder.resolve().is_relative_to(root.resolve()):
            raise ValueError("registered Stage A run escaped this sweep")
        done_path=folder/"stage_A_complete.json"
        checkpoint=checkpoint_path(folder,60000)
        row=dict(key=key,checkpoint=str(checkpoint),stage_a_completion_exists=done_path.exists())
        if done_path.exists() and checkpoint.exists():
            done=read_json(done_path)
            expected=done["checkpoint_sha256"];actual=file_hash(checkpoint)
            row.update(expected_stage_a_sha256=expected,current_sha256=actual,
                exact_stage_a_file_preserved=expected==actual,matching_backups=[])
            if expected!=actual:
                for candidate in checkpoint.parent.glob("step000060000*.ckpt"):
                    if candidate!=checkpoint and file_hash(candidate)==expected:
                        row["matching_backups"].append(str(candidate))
            print(f"CHECKPOINT_AUDIT={key} exact_original={row['exact_stage_a_file_preserved']} backups={len(row['matching_backups'])}",flush=True)
        else:
            print(f"CHECKPOINT_AUDIT={key} Stage_A_not_yet_complete=True",flush=True)
        rows.append(row)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),runs=rows,
        exact_full_state_restore_not_assumed=True,checkpoints_modified=False)
    atomic_json(output,result)
    return result


def recover(args,state):
    if state.get("version")!=VERSION or state.get("pending_submission"):
        raise ValueError("unsupported state or unresolved sbatch intent")
    key="monitor/1/A";record=state["jobs"][key]
    current={name:file_hash(ROOT/name) for name in submit.FILES}
    old=state["inputs"]["code"]
    changed=sorted(k for k in set(old)|set(current) if old.get(k)!=current.get(k))
    if set(changed)-{MONITOR,"scripts/submit_nfcgs_fresh_rd.py"}:
        raise ValueError(f"changes outside the monitor-only repair: {changed}")
    if state["lanes"].get("15")!=record["id"] or record.get("run") is not None:
        raise ValueError("monitor lane changed; learner jobs preserved")
    env=record["env"]
    if (record["command"][-1]!=submit.SLURM+"monitor_nfcgs_fresh_rd.slurm"
            or env.get("REPO_DIR")!=str(ROOT) or env.get("FRESH_STATE")!=str(args.state)
            or env.get("FRESH_OUTPUT_ROOT")!=str(args.root)
            or env.get("MONITOR_PHASE")!="A" or env.get("MONITOR_WAVE")!="1"):
        raise ValueError("registered monitor does not belong to this sweep")
    status=submit.job_state(record["id"])
    if status not in submit.TERMINAL_FAILURES:
        if MONITOR not in changed:
            if changed:
                submit.adopt_inputs(state,dict(state["inputs"],code=current),allow_stage_b_controller_update=True)
                atomic_json(args.state,state)
            print(f"REUSING_MONITOR={record['id']} state={status}",flush=True)
            return
        raise ValueError("the old monitor is not in a terminal failure state; no jobs changed")
    error=args.log_root/f"slurm-gs-fresh-monitor-{record['id']}.err"
    if "checkpoint identity changed after evaluation" not in error.read_text(encoding="utf-8",errors="replace"):
        raise ValueError("not the expected checkpoint-identity monitor failure")
    if active_monitor_jobs():
        raise ValueError("another monitor is active; no locks changed")
    audit=args.log_root/"fresh_rd"/("checkpoint_identity_audit_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
    audit_checkpoints(state,args.root,audit)
    convergence=args.root/"convergence"
    locks=[convergence/".monitor_1_A.lock",convergence/".csv.lock"]
    snapshots={p:file_hash(p) for p in locks if p.exists()}
    if active_monitor_jobs():
        raise ValueError("monitor appeared during audit; locks preserved")
    for path in locks:
        if path.exists() and (path.resolve().parent!=convergence.resolve() or snapshots.get(path)!=file_hash(path)):
            raise ValueError("monitor lock identity changed")
    for path in locks:path.unlink(missing_ok=True)
    # Only the monitoring/submission controllers change. The 30 model/training/evaluation
    # runtime hashes and every learner record stay exactly as submitted.
    if changed:
        state.setdefault("input_migrations",[]).append(dict(previous_inputs=state["inputs"],
            changed_files=changed,reason="re-evaluate replaced checkpoint files instead of accepting old metrics",
            utc=datetime.now(timezone.utc).isoformat()))
        state["inputs"]=dict(state["inputs"],code=current)
    sub=submit.Submission(args,state)
    entry=dict(old_id=record["id"],audit=str(audit),stderr_sha256=file_hash(error),
        utc=datetime.now(timezone.utc).isoformat())
    state.setdefault("monitor_identity_recovery_20261023",[]).append(entry)
    del state["lanes"]["15"]
    try:
        new=sub.add(key,submit.SLURM+"monitor_nfcgs_fresh_rd.slurm",record["env"],lane=15,replace_failed=True)
    except Exception:
        if not state.get("pending_submission"):
            state["lanes"]["15"]=record["id"];sub.save()
        raise
    entry["new_id"]=new;sub.save()
    print(f"MONITOR_RECOVERED={new} audit={audit} learner_jobs_unchanged=True",flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state",type=Path,default=ROOT/"logs/slurm/fresh_rd/state.json")
    parser.add_argument("--log-root",type=Path,default=ROOT/"logs/slurm")
    parser.add_argument("--audit-only",action="store_true",help="write a checkpoint identity report without any Slurm actions")
    args=parser.parse_args(argv);args.state=args.state.resolve();args.log_root=args.log_root.resolve();args.dry_run=False
    with directory_lock(args.state.parent,".submit.lock"):
        state=read_json(args.state);args.root=Path(state["inputs"]["root"])
        if not args.root.resolve().is_relative_to((ROOT/"outputs").resolve()):
            raise ValueError("sweep is outside this repository's outputs")
        if args.audit_only:
            output=args.log_root/"fresh_rd"/("checkpoint_identity_audit_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
            audit_checkpoints(state,args.root,output)
            print(f"AUDIT_ONLY={output}",flush=True)
        else:
            recover(args,state)


if __name__=="__main__":main()

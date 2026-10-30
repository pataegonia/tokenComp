#!/usr/bin/env python3
"""Submit the independent follow-ups, preserving every accepted job ID.

One None+D4 OFF16 50k train + afterok eval, one CPU boundary audit,
one native full-test eval, one saved-T06 packet render. Existing jobs unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

import run_nfcgs
import run_nfcgs_followup_20261020 as follow

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("boundary","native","fixed-render","none-off-train","none-off-eval")
FILES = ("scripts/submit_nfcgs_followup_20261020.py","scripts/run_nfcgs_followup_20261020.py",
    "scripts/slurm/run_nfcgs_followup_20261020.slurm","scripts/slurm/train_nfcgs_order_context.slurm",
    "scripts/slurm/eval_nfcgs_order_context.slurm","scripts/slurm/nfcgs_order_context_settings.sh",
    "scripts/run_nfcgs.py","scripts/check_nfcgs_checkpoint.py","scripts/eval_nfcgs_b0_matched.py",
    "scripts/eval_nfcgs_independent_receiver.py","scripts/analyze_nfcgs_fixed_order.py",
    "scripts/eval_nfcgs_path_diagnostics.py","scripts/profile_nfcgs_complexity.py")


def atomic_state(path, state):
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,encoding="utf-8",delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(state,stream,indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def none_environment(root, parent, state_dir):
    env = os.environ.copy()
    # Do not inherit another sweep's architecture or output-directory selectors.
    for key in ("EVAL_RUN_DIR","SUBMISSION_STATE_FILE","SUBMISSION_RECORD","FOLLOWUP_OUTPUT","FOLLOWUP_RESUME"):
        env.pop(key,None)
    env.update(REPO_DIR=str(root),SOURCE_CHECKPOINT=str(parent),TOKEN_ORDER="none",
        CONTEXT_SCHEDULE="dyadic4",STAGES="4",KERNEL="5",SPATIAL_CONTEXT="false",SCORE_SLICE_CHANNELS="16",
        RATE_LAMBDA="0.0128",MAX_STEPS="50000",CHECKPOINT_EVERY="5000",SCORE_ORDER_SCALE="",
        NN_ORDER_BACKEND="legacy",NN_ORDER_CANDIDATES="64",NN_ORDER_DISTANCE_DEVICE="cpu",
        FOLLOWUP_STATE_DIR=str(state_dir),OMP_NUM_THREADS="8")
    return env


def validate_recipe(parent):
    args = run_nfcgs.parse_args(["train","--checkpoint",str(parent),"--rate-lambda","0.0128",
        "--score-path","minimal","--token-order","none","--score-context-schedule","dyadic4",
        "--score-spatial-stages","4","--score-slice-channels","16","--score-spatial-kernel","5",
        "--score-context-quantization","ste","--no-spatial-context","--allow-score-path-conversion",
        "--devices","2","--launcher","srun","--scope","all","--lr","0.0001",
        "--batch-size","2","--accumulate","2","--workers","8","--max-steps","50000","--checkpoint-every","5000"])
    command,_,_ = run_nfcgs.build_command(args)
    return command


def task_command(task, train=None):
    command = ["sbatch","--parsable","--export=ALL",f"--job-name=gs-f20-{task}"]
    if task.startswith("none-off"):
        if task == "none-off-train":
            command += ["--time=6-00:00:00","scripts/slurm/train_nfcgs_order_context.slurm"]
        else:
            if train is None:
                raise ValueError("eval requires the accepted training ID")
            command += [f"--dependency=afterok:{train}","scripts/slurm/eval_nfcgs_order_context.slurm"]
    else:
        if task == "boundary":
            command += ["--mem=4G","--cpus-per-task=2","--time=00:30:00"]
        else:
            command += ["--gres=gpu:normal:1","--mem=48G","--cpus-per-task=8",
                        "--time="+("1-00:00:00" if task == "native" else "08:00:00")]
        command += ["scripts/slurm/run_nfcgs_followup_20261020.slurm",task]
    return command


def load_state(path, signature, inputs):
    state = follow.read(path) if path.exists() else dict(version=1,signature=signature,inputs=inputs,jobs={})
    if state.get("version") != 1 or state.get("signature") != signature or state.get("inputs") != inputs:
        raise ValueError("saved follow-up state differs from code/inputs; no jobs submitted")
    jobs = state.get("jobs",{})
    if set(jobs)-set(TASKS) or any(not re.fullmatch(r"[1-9][0-9]*",str(v)) for v in jobs.values()):
        raise ValueError("invalid saved Slurm IDs")
    if "none-off-eval" in jobs and "none-off-train" not in jobs:
        raise ValueError("saved eval has no training dependency")
    return state


def submit_missing(root,env,state,state_file):
    for task in TASKS:
        if task in state["jobs"]:
            print(f"REUSING {task}={state['jobs'][task]}",flush=True)
            continue
        train = state["jobs"].get("none-off-train")
        command = task_command(task,train)
        task_env = dict(env)
        if task == "none-off-eval":
            task_env["TRAIN_JOB_ID"] = train
        result = subprocess.run(command,cwd=root,env=task_env,capture_output=True,text=True,check=True)
        job = result.stdout.strip().split(";",1)[0]
        if not re.fullmatch(r"[1-9][0-9]*",job):
            raise ValueError(f"invalid sbatch response; inspect scheduler before retrying: {result.stdout!r}")
        print(f"ACCEPTED {task}={job}",flush=True)
        state["jobs"][task] = job
        atomic_state(state_file,state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args(argv)
    root = Path(os.environ.get("REPO_DIR",ROOT)).resolve()
    state_dir = Path(os.environ.get("FOLLOWUP_STATE_DIR",root/"logs/slurm/followup_20261020")).resolve()
    parent = Path(os.environ.get("SOURCE_CHECKPOINT",root/"outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt")).resolve()
    env = none_environment(root,parent,state_dir)
    defaults = follow.parse_args(["plan"])
    path_env = dict(b0_run="FOLLOWUP_B0_RUN",path_run="FOLLOWUP_PATH_RUN",fixed_run="FOLLOWUP_FIXED_RUN",
        bundle="FOLLOWUP_TARGET_BUNDLE",checkpoint="FOLLOWUP_CHECKPOINT",native_checkpoint="FOLLOWUP_NATIVE_CHECKPOINT",
        reference="FOLLOWUP_REFERENCE")
    for name,key in path_env.items():
        setattr(defaults,name,Path(os.environ.get(key,getattr(defaults,name))).resolve())
        env[key] = str(getattr(defaults,name))
    required = set(FILES)
    required.update(p.relative_to(root).as_posix() for p in (root/"globalsplat").rglob("*.py"))
    required.update(p.relative_to(root).as_posix() for p in (root/"config").rglob("*.yaml"))
    for name in sorted(required):
        path = root/name
        if not path.is_file():
            raise FileNotFoundError(f"upload dependency: {name}")
        if path.suffix in (".sh",".slurm") and (b"\r" in path.read_bytes() or not path.read_bytes().startswith(b"#!/usr/bin/env bash\n")):
            raise ValueError(f"upload with LF, no BOM: {name}")
    if "CHECKPOINT_TMPDIR=" not in (root/"scripts/slurm/train_nfcgs_order_context.slurm").read_text():
        raise ValueError("required Ceph checkpoint staging fix missing")
    if not parent.is_file() or parent.stat().st_size == 0:
        raise FileNotFoundError(parent)
    recipe = validate_recipe(parent)
    follow.preflight(defaults,("boundary","native","fixed-render"))
    inputs = dict(repo=str(root),parent=str(parent),parent_sha256=follow.digest(parent),
        artifact_paths={name:str(getattr(defaults,name)) for name in path_env},
        reference_sha256=follow.digest(defaults.reference),
        artifact_completion_sha256={str(folder/"complete.json"):follow.digest(folder/"complete.json")
            for folder in (defaults.b0_run,defaults.path_run,defaults.fixed_run,defaults.bundle)},
        dataset_root=env.get("DATASET_ROOT","/data3/local_datasets/re10k"),
        code_sha256={name:follow.digest(root/name) for name in sorted(required)},
        protocol="None_D4_OFF16_lambda0p0128_50k_seed111123_data403_plus_N01_N02_N03_v1")
    signature = hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    state_file = state_dir/"state.json"
    state = load_state(state_file,signature,inputs)
    print("NONE_OFF_RECIPE="+shlex.join(recipe),flush=True)
    for task in TASKS:
        command = task_command(task,state["jobs"].get("none-off-train","TRAIN_JOB"))
        print(f"PLAN {task} JOB={state['jobs'].get(task,'new')} "+shlex.join(command),flush=True)
    if args.dry_run:
        print("PREVIEW ONLY: 1 train + 1 afterok full eval + 2 GPU evals + 1 CPU audit; no submission",flush=True)
        return
    state_dir.mkdir(parents=True,exist_ok=True)
    (root/"logs/slurm").mkdir(parents=True,exist_ok=True)
    lock = state_dir/".submit-lock"; lock.mkdir()
    try:
        if load_state(state_file,signature,inputs) != state:
            raise ValueError("state changed while acquiring lock")
        atomic_state(state_file,state)
        submit_missing(root,env,state,state_file)
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error),file=sys.stderr)
        print(f"Accepted IDs are saved in {state_file}; rerun the same command to retry only missing jobs.",file=sys.stderr)
        raise SystemExit(1)
    finally:
        lock.rmdir()
    print(f"STATE={state_file}",flush=True)
    print("squeue -j "+",".join(state["jobs"].values())+" -o '%.18i %.28j %.10T %.30R'",flush=True)
    print("Existing 457828/457829 and all recovery dependencies are unchanged. No architecture-lambda jobs submitted.",flush=True)


if __name__ == "__main__":
    main()

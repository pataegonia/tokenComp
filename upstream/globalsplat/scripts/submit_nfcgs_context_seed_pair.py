#!/usr/bin/env python3
"""Submit two seed-repeat training jobs and their afterok full-test evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import run_nfcgs_context_seed_pair as runner

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/run_nfcgs_context_seed_pair.py", "scripts/run_nfcgs.py", "scripts/check_nfcgs_checkpoint.py",
    "scripts/slurm/train_nfcgs_context_seed_pair.slurm", "scripts/slurm/eval_nfcgs_context_seed_pair.slurm",
    "globalsplat/main.py", "globalsplat/model/model_wrapper.py", "globalsplat/compression/config.py",
    "globalsplat/compression/codec.py", "globalsplat/compression/checkpoint.py",
    "globalsplat/compression/score_context.py", "globalsplat/dataset/data_module.py",
    "globalsplat/dataset/re10k.py", "config/experiment/re10k_32k_nfcgs.yaml",
    "config/dataset/re10k_nfcgs_paper_train.yaml", "config/dataset/re10k_eval_all_ctx12.yaml",
    "config/model/globalsplat_nfcgs_rank56.yaml",
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_state(path, state):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, indent=2)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(os.environ.get("REPO_DIR", ROOT)).resolve()
    parent = Path(os.environ.get("SOURCE_CHECKPOINT", root/
        "outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/"
        "nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/"
        "version_0/step000010000.ckpt")).resolve()
    dataset = Path(os.environ.get("DATASET_ROOT", "/data3/local_datasets/re10k")).resolve()
    outputs = Path(os.environ.get("CONTEXT_SEED_OUTPUT_ROOT", root/
        "outputs/nfcgs_context_seed_pair/seed111124_data404")).resolve()
    state_dir = Path(os.environ.get("CONTEXT_SEED_STATE_DIR", root/"logs/slurm/context_seed_pair_20261019")).resolve()
    for relative in FILES:
        path = root/relative
        if not path.is_file():
            raise FileNotFoundError(f"upload required server file: {relative}")
        if relative.endswith(".slurm") and (
            not path.read_bytes().startswith(b"#!/usr/bin/env bash\n") or b"\r" in path.read_bytes()
        ):
            raise ValueError(f"Slurm file must have LF line endings and a valid shebang: {relative}")
    if not parent.is_file() or not parent.stat().st_size:
        raise FileNotFoundError(f"required FullSplit parent missing: {parent}")
    # Dataset indexes are node-local; the GPU jobs check them on their compute nodes.
    inputs = {
        "repo": str(root), "source_checkpoint": str(parent), "source_sha256": digest(parent),
        "dataset_root": str(dataset), "output_root": str(outputs),
        "protocol": "T11_hilbert_dyadic4_ON_OFF16_lambda0p0128_50k_model111124_data404_fulltest_C12_T8_v1",
        "code_sha256": {name: digest(root/name) for name in FILES},
    }
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    state_file = state_dir/"state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        "version": 1, "signature": signature, "inputs": inputs, "arms": {},
    }
    if state.get("version") != 1 or state.get("signature") != signature:
        raise ValueError("saved seed-pair state differs from inputs/code; no jobs submitted")
    if set(state.get("arms", {}))-set(runner.ARMS):
        raise ValueError("unexpected arm in saved seed-pair state; no jobs submitted")
    for jobs in state.get("arms", {}).values():
        if set(jobs)-{"train", "eval"} or (jobs.get("eval") and not jobs.get("train")):
            raise ValueError("invalid dependencies in saved seed-pair state; no jobs submitted")
        for value in jobs.values():
            if value is not None and not re.fullmatch(r"[1-9][0-9]*", str(value)):
                raise ValueError("invalid Slurm ID in saved seed-pair state; no jobs submitted")
    print(f"SOURCE_CHECKPOINT={parent} SHA256={inputs['source_sha256']}", flush=True)
    print(f"MODEL_SEED=111124 DATA_LOADER_SEED=404 DATASET_ROOT={dataset} DATASET_CHECK=compute_jobs", flush=True)
    for arm in runner.ARMS:
        # Validate both configurations before any sbatch call.
        runner.build_command(runner.parse_args([
            "train", "--arm", arm, "--checkpoint", str(parent), "--run-dir", str(outputs/arm/"job_PREVIEW"),
            "--dataset-root", str(dataset), "--launcher", "srun",
        ]))
        jobs = state["arms"].get(arm, {})
        print(f"PLAN {arm} TRAIN={jobs.get('train') or 'new'} EVAL={jobs.get('eval') or 'new'} "
              "HILBERT D4 SCORE_STRINGS=16 MSH=on LAMBDA=0.0128 MAX_STEPS=50000", flush=True)
    if args.dry_run:
        print("PREVIEW ONLY: 2 training jobs + 2 afterok full-test eval jobs; no prepare job", flush=True)
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = state_dir/".submit-lock"
    lock.mkdir()
    try:
        if state_file.exists():
            current = json.loads(state_file.read_text())
            if current != state:
                raise ValueError("submission state changed while obtaining lock; retry")
        (root/"logs/slurm").mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(REPO_DIR=str(root), SOURCE_CHECKPOINT=str(parent), DATASET_ROOT=str(dataset),
                   CONTEXT_SEED_OUTPUT_ROOT=str(outputs))
        def submit(arm, role, dependency=None):
            jobs = state["arms"].setdefault(arm, {})
            if jobs.get(role):
                print(f"REUSING {arm} {role.upper()}={jobs[role]}", flush=True)
                return jobs[role]
            command = ["sbatch", "--parsable", f"--job-name=gs-s2-{arm}-{role}", "--export=ALL"]
            if dependency:
                command.append(f"--dependency=afterok:{dependency}")
            command.append(f"scripts/slurm/{role}_nfcgs_context_seed_pair.slurm")
            result = subprocess.run(command, cwd=root, env=env, check=True, capture_output=True, text=True)
            job = result.stdout.strip().split(";", 1)[0]
            if not re.fullmatch(r"[1-9][0-9]*", job):
                raise ValueError(f"invalid sbatch response: {result.stdout}")
            jobs[role] = job
            atomic_state(state_file, state)
            print(f"SUBMITTED {arm} {role.upper()}={job}", flush=True)
            return job
        for arm in runner.ARMS:
            env["CONTEXT_SEED_ARM"] = arm
            train = submit(arm, "train")
            env["TRAIN_JOB_ID"] = train
            submit(arm, "eval", train)
        print(f"STATE={state_file}; rerun to reuse accepted IDs and retry missing submissions", flush=True)
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        print(f"Accepted IDs are saved in {state_file}; retry when submission capacity is available.", file=sys.stderr)
        raise SystemExit(1)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    main()

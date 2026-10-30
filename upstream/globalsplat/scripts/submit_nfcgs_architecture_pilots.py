#!/usr/bin/env python3
"""Submit fresh architecture pilots; persist every accepted Slurm ID."""
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

ROOT=Path(__file__).resolve().parents[1]
FILES=("globalsplat/compression/config.py","globalsplat/compression/codec.py",
       "globalsplat/compression/architecture.py","globalsplat/compression/checkpoint.py",
       "globalsplat/dataset/holdout.py","globalsplat/dataset/data_module.py","globalsplat/dataset/re10k.py",
       "globalsplat/main.py","globalsplat/model/model_wrapper.py",
       "globalsplat/compression/initialization.py","globalsplat/compression/bitstream.py",
       "globalsplat/model/globalsplat.py","scripts/check_nfcgs_checkpoint.py",
       "scripts/prepare_nfcgs_architecture_pilots.py","scripts/run_nfcgs_architecture_pilot.py",
       "scripts/smoke_nfcgs_architecture_pilot.py","scripts/run_nfcgs.py",
       "scripts/slurm/prepare_nfcgs_architecture_pilots.slurm",
       "scripts/slurm/train_nfcgs_architecture_pilot.slurm","scripts/slurm/eval_nfcgs_architecture_pilot.slurm")
MARKERS={"globalsplat/compression/config.py":"codec_mode",
         "globalsplat/compression/codec.py":"FLAG_MSH_ONLY",
         "globalsplat/compression/checkpoint.py":"_infer_single_path_config",
         "globalsplat/dataset/data_module.py":"_train_holdout_manifest",
         "globalsplat/dataset/re10k.py":"eval_train_holdout_manifest",
         "globalsplat/main.py":"architecture_pilot_initialization",
         "globalsplat/model/model_wrapper.py":"codec.score_context is not None else True",
         "scripts/check_nfcgs_checkpoint.py":"model.feature_codec.geometry_projection.weight"}


def digest(path):
    sha=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""):
            sha.update(chunk)
    return sha.hexdigest()


def atomic_state(path, data):
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False,encoding="utf-8") as handle:
        tmp=Path(handle.name)
        json.dump(data,handle,indent=2);handle.write("\n");handle.flush();os.fsync(handle.fileno())
    try:
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--include-fullwidth",action="store_true")
    parser.add_argument("--with-test",action="store_true",help="also submit full C12/T8 test after successful holdout evaluation")
    args=parser.parse_args(argv)
    root=Path(os.environ.get("REPO_DIR",ROOT)).resolve()
    vanilla=Path(os.environ.get("VANILLA_CHECKPOINT",root/"checkpoints/pretrained/globalsplat-re10k-32k.ckpt")).resolve()
    dataset=Path(os.environ.get("DATASET_ROOT","/data3/local_datasets/re10k")).resolve()
    state_dir=Path(os.environ.get("ARCH_PILOTS_STATE_DIR",root/"logs/slurm/architecture_pilots_20261019")).resolve()
    prep=Path(os.environ.get("ARCH_PILOTS_PREPARATION",root/"outputs/nfcgs_architecture_pilots/initialization_20261019")).resolve()
    outputs=Path(os.environ.get("ARCH_PILOTS_OUTPUT_ROOT",root/"outputs/nfcgs_architecture_pilots/lambda0p0128")).resolve()
    for relative in FILES:
        if not (root/relative).is_file():
            raise FileNotFoundError(f"upload required server file: {relative}")
    for relative,marker in MARKERS.items():
        if marker not in (root/relative).read_text(encoding="utf-8"):
            raise ValueError(f"upload the complete architecture-pilot bundle; stale server file: {relative}")
    if not vanilla.is_file() or not vanilla.stat().st_size:
        raise FileNotFoundError(f"required existing input missing: {vanilla}")
    # /data3 is compute-node-local and need not be mounted on the login node.
    # create_holdout in the preparation job checks both dataset indexes/chunks;
    # all training jobs depend on that preparation succeeding.
    payload={"repo":str(root),"vanilla":str(vanilla),"vanilla_sha256":digest(vanilla),
             "dataset_root":str(dataset),"preparation":str(prep),"outputs":str(outputs),
             "protocol":"fresh_codec_50k_seed111123_train_holdout128_lambda0p0128_v1",
             "code_sha256":{name:digest(root/name) for name in FILES}}
    signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    state_file=state_dir/"state.json"
    state=json.loads(state_file.read_text()) if state_file.exists() else {"version":1,"signature":signature,"inputs":payload,"prepare":None,"arms":{}}
    if state.get("version")!=1 or state.get("signature")!=signature:
        raise ValueError("saved submission state differs from inputs/code; no jobs submitted")
    for job in [state.get("prepare"),*[v for arm in state.get("arms",{}).values() for v in arm.values()]]:
        if job is not None and not re.fullmatch(r"[1-9][0-9]*",str(job)):
            raise ValueError("invalid saved Slurm ID; no jobs submitted")
    arms=["score56_only","msh_only","two_path56"]+(["score736_only"] if args.include_fullwidth else [])
    print(f"VANILLA_CHECKPOINT={vanilla} SHA256={payload['vanilla_sha256']}",flush=True)
    print(f"PREPARATION_DIR={prep} DATASET_ROOT={dataset} HOLDOUT=128 LAMBDA=0.0128 MAX_STEPS=50000",flush=True)
    print("DATASET_INDEX_CHECK=prepare_job_on_compute_node",flush=True)
    for arm in arms:
        saved=state["arms"].get(arm,{})
        print(f"PLAN {arm} TRAIN={saved.get('train') or 'new'} VALIDATION={saved.get('validation') or 'new'} TEST={saved.get('test') or ('new' if args.with_test else 'deferred')}",flush=True)
    if args.dry_run:
        print(f"PREVIEW ONLY: prepare + {len(arms)} train + {len(arms)} validation"+(f" + {len(arms)} test" if args.with_test else ""))
        return
    state_dir.mkdir(parents=True,exist_ok=True)
    lock=state_dir/".submit-lock"
    lock.mkdir()  # existing locks are never removed by a competing invocation
    try:
        # Re-read under the lock to protect against an intervening submitter.
        if state_file.exists():
            state=json.loads(state_file.read_text())
            if state.get("signature")!=signature:
                raise ValueError("submission state changed while obtaining lock")
        env=os.environ.copy()
        env.update(REPO_DIR=str(root),VANILLA_CHECKPOINT=str(vanilla),DATASET_ROOT=str(dataset),
                   PREPARATION_DIR=str(prep),PILOT_OUTPUT_ROOT=str(outputs))
        (root/"logs/slurm").mkdir(parents=True,exist_ok=True)
        def submit(role,script,job_name,dependency=None,arm=None):
            target=state if arm is None else state["arms"].setdefault(arm,{})
            if target.get(role):
                print(f"REUSING {arm or 'common'} {role.upper()}={target[role]}",flush=True)
                return target[role]
            options=["sbatch","--parsable",f"--job-name={job_name}","--export=ALL"]
            if dependency:
                options.append(f"--dependency=afterok:{dependency}")
            result=subprocess.run([*options,f"scripts/slurm/{script}"],cwd=root,env=env,check=True,capture_output=True,text=True)
            job=result.stdout.strip().split(";",1)[0]
            if not re.fullmatch(r"[1-9][0-9]*",job):
                raise ValueError(f"invalid sbatch response: {result.stdout}")
            target[role]=job
            atomic_state(state_file,state)
            print(f"SUBMITTED {arm or 'common'} {role.upper()}={job}",flush=True)
            return job
        prepare_id=submit("prepare","prepare_nfcgs_architecture_pilots.slurm","gs-a-prepare")
        for arm in arms:
            env["PILOT_ARM"]=arm
            train=submit("train","train_nfcgs_architecture_pilot.slurm",f"gs-a-{arm}",prepare_id,arm)
            env.update(TRAIN_JOB_ID=train,PILOT_EVAL_MODE="validation")
            validation=submit("validation","eval_nfcgs_architecture_pilot.slurm",f"gs-a-v-{arm}",train,arm)
            if args.with_test:
                env["PILOT_EVAL_MODE"]="test"
                submit("test","eval_nfcgs_architecture_pilot.slurm",f"gs-a-t-{arm}",validation,arm)
        print(f"STATE={state_file}; rerun to reuse accepted IDs and retry missing submissions",flush=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc),file=sys.stderr)
        print(f"Accepted IDs are saved in {state_file}; retry after submission capacity is available.",file=sys.stderr)
        raise SystemExit(1)
    finally:
        lock.rmdir()


if __name__=="__main__":
    main()

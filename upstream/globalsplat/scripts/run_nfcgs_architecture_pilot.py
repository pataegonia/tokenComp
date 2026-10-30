#!/usr/bin/env python3
"""Run a matching fresh architecture checkpoint with held-out rate selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import run_nfcgs as base

ROOT=Path(__file__).resolve().parents[1]
ARMS=("score56_only","msh_only","two_path56","score736_only")


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("train","validation","test"))
    parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--preparation",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--dataset-root",type=Path,default=Path(os.environ.get("DATASET_ROOT","/data3/local_datasets/re10k")))
    parser.add_argument("--launcher",choices=("python","srun"),default="python")
    parser.add_argument("--dry-run",action="store_true")
    return parser.parse_args(argv)


def build_command(args):
    preparation=args.preparation.resolve()
    checkpoint=(args.checkpoint or preparation/f"{args.arm}.ckpt").resolve()
    if args.mode != "train" and args.checkpoint is None:
        raise ValueError("evaluation requires the trained --checkpoint")
    train=args.mode=="train"
    rate="0.0128"
    argv=["train" if train else "eval","--checkpoint",str(checkpoint),"--output",str(args.output),
          "--dataset-root",str(args.dataset_root),"--rate-lambda",rate,"--score-path","minimal",
          "--token-order","hilbert","--score-context-schedule","dyadic4","--score-spatial-stages","4",
          "--score-spatial-kernel","5","--score-context-quantization","ste","--score-slice-channels","16"]
    if args.arm=="msh_only":
        argv.append("--no-spatial-context")
    if train:
        argv += ["--devices","2","--launcher",args.launcher,"--scope","all","--lr","0.0001",
                 "--batch-size","2","--accumulate","2","--workers","8","--max-steps","50000","--checkpoint-every","5000"]
    command,_,_=base.build_command(base.parse_args(argv))
    # New settings are opt-in; the historical runner/Slurm recipe is unchanged.
    mode="msh_only" if args.arm=="msh_only" else "score_only" if args.arm.startswith("score") else "two_path"
    command += ["++architecture_entropy_determinism=true", f"++model.feature_codec.codec_mode={mode}",
                f"model.feature_codec.rank={0 if mode=='msh_only' else 736 if args.arm=='score736_only' else 56}",
                f"model.feature_codec.use_residual={str(mode!='score_only').lower()}"]
    if args.arm=="score736_only":
        command=[item for item in command if not item.startswith("model.feature_codec.score_slice_channels=")]
        command.append("model.feature_codec.score_slice_channels=184")
    if train:
        command += [f"++dataset.train_holdout_manifest={(preparation/'holdout.json').as_posix()}",
                    "trainer.limit_val_batches=0"]
    elif args.mode=="validation":
        command += [f"dataset.eval_index_path={(preparation/'validation_index.json').as_posix()}",
                    "dataset.eval_all_scenes=false",
                    f"++dataset.eval_train_holdout_manifest={(preparation/'holdout.json').as_posix()}"]
    return command,checkpoint


def main():
    args=parse_args()
    command,checkpoint=build_command(args)
    print(shlex.join(command),flush=True)
    if args.dry_run:
        return
    import torch
    from globalsplat.compression.architecture import pilot_config
    from globalsplat.compression.checkpoint import validate_feature_codec_checkpoint
    from globalsplat.dataset.holdout import read_holdout,file_hash
    prep=args.preparation.resolve()
    manifest=json.loads((prep/"preparation.json").read_text())
    if manifest.get("kind")!="nfcgs_architecture_preparation_v1" or manifest.get("shared_initialization_checked") is not True:
        raise ValueError("preparation is incomplete or unsupported")
    read_holdout(prep/"holdout.json",args.dataset_root)
    if file_hash(prep/"holdout.json")!=manifest["holdout_sha256"] or file_hash(prep/"validation_index.json")!=manifest["validation_index_sha256"]:
        raise ValueError("frozen validation files changed after preparation")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    provenance=saved.get("architecture_pilot_initialization",{}).get("signature",{})
    if provenance.get("arm")!=args.arm or provenance.get("holdout_sha256")!=manifest["holdout_sha256"] or provenance.get("vanilla_sha256")!=manifest["vanilla_sha256"]:
        raise ValueError("checkpoint architecture/common initialization provenance mismatch")
    validate_feature_codec_checkpoint(saved,pilot_config(args.arm))
    if args.mode=="train":
        if saved.get("global_step",0)!=0 or any(key in saved for key in ("optimizer_states","lr_schedulers")):
            raise ValueError("pilot training must start from a fresh initialization checkpoint")
        if file_hash(checkpoint)!=manifest["arms"][args.arm]["sha256"]:
            raise ValueError("initializer checkpoint changed after preparation")
    elif saved.get("global_step")!=50000:
        raise ValueError("pilot evaluation requires the final 50k checkpoint")
    if args.launcher=="srun" and (int(os.environ.get("SLURM_NTASKS","0"))!=2 or int(os.environ.get("SLURM_JOB_NUM_NODES","0"))!=1):
        raise RuntimeError("training allocation must have one node and two tasks")
    del saved
    subprocess.run(command,cwd=ROOT,check=True)
    if args.mode=="validation":
        summary=json.loads((args.output/"nfcgs_main/scores_all_avg.json").read_text())
        expected=read_holdout(prep/"holdout.json",args.dataset_root)["scene_count"]
        if summary.get("scene_count")!=expected:
            raise RuntimeError(f"validation incomplete: {summary.get('scene_count')} scenes, expected {expected}")


if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Repeat the matched Hilbert D4 ON/OFF16 comparison with a second seed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import run_nfcgs as base

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("on16", "off16")
MODEL_SEED = 111124
DATA_SEED = 404
STEPS = 50000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path(os.environ.get("DATASET_ROOT", "/data3/local_datasets/re10k")))
    parser.add_argument("--launcher", choices=("python", "srun"), default="python")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "train" and args.checkpoint is None:
        parser.error("training requires the shared FullSplit --checkpoint")
    return args


def build_command(args):
    run = args.run_dir.resolve()
    checkpoint = args.checkpoint or run/"checkpoints/nfcgs_main/version_0/step000050000.ckpt"
    argv = [args.mode, "--checkpoint", str(checkpoint), "--rate-lambda", "0.0128",
            "--dataset-root", str(args.dataset_root), "--token-order", "hilbert", "--score-path", "minimal",
            "--score-context-schedule", "dyadic4", "--score-spatial-stages", "4",
            "--score-spatial-kernel", "5", "--score-context-quantization", "ste",
            "--score-slice-channels", "16", "--launcher", args.launcher,
            "--output", str(run if args.mode == "train" else run/"eval")]
    if args.arm == "off16":
        argv.append("--no-spatial-context")
    if args.mode == "train":
        argv += ["--allow-score-path-conversion", "--devices", "2", "--scope", "all",
                 "--lr", "0.0001", "--batch-size", "2", "--accumulate", "2", "--workers", "8",
                 "--max-steps", str(STEPS), "--checkpoint-every", "5000"]
    command, checkpoint, dataset = base.build_command(base.parse_args(argv))
    if args.mode == "train":
        command += [f"seed={MODEL_SEED}", f"dataset.data_loader_seed={DATA_SEED}"]
    return command, checkpoint, dataset


def validate_training_config(config, arm):
    expected = {
        "seed": MODEL_SEED, "dataset.data_loader_seed": DATA_SEED,
        "loss.rate_lambda": .0128, "trainer.max_steps": STEPS,
        "trainer.devices": 2, "trainer.accumulate_grad_batches": 2,
        "optimizer.batch_size": 2, "model.freeze_globalsplat": True,
        "checkpointing.resume": False, "model.feature_codec_train_scope": "all",
        "model.feature_codec.token_order": "hilbert", "model.feature_codec.rank": 56,
        "model.feature_codec.use_residual": True, "model.feature_codec.transform": "linear",
        "model.feature_codec.score_context_schedule": "dyadic4",
        "model.feature_codec.score_spatial_stages": 4, "model.feature_codec.score_slice_channels": 16,
        "model.feature_codec.score_spatial_context": arm == "on16",
        "model.feature_codec.score_spatial_kernel": 5, "model.feature_codec.score_context_quantization": "ste",
        "model.feature_codec.use_centering": False, "model.feature_codec.use_score_norm": False,
        "model.feature_codec.score_mean_condition": False, "model.feature_codec.score_channel_context": False,
    }
    for name, value in expected.items():
        actual = config
        for part in name.split("."):
            actual = actual[part]
        if actual != value:
            raise ValueError(f"training configuration mismatch: {name}={actual!r}, expected {value!r}")
    if config["dataset"].get("train_holdout_manifest") is not None:
        raise ValueError("seed repeat must retain the original training split")
    if config["model"]["feature_codec"].get("codec_mode", "two_path") != "two_path":
        raise ValueError("seed repeat requires the existing two-path codec")


def main(argv=None):
    args = parse_args(argv)
    command, checkpoint, dataset = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if args.launcher == "srun" and (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_NTASKS", "0")) != (2 if args.mode == "train" else 1)
        or int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1
    ):
        raise RuntimeError("SLURM allocation must have one node and one task per requested GPU")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    split = "train" if args.mode == "train" else "test"
    if not (dataset/split/"index.json").is_file():
        raise FileNotFoundError(f"missing compute-node RE10K {split} index under {dataset}")
    if args.mode == "eval":
        from omegaconf import OmegaConf
        saved_config = OmegaConf.load(args.run_dir/"hydra/.hydra/config.yaml")
        validate_training_config(OmegaConf.to_container(saved_config, resolve=True), args.arm)
    checker = [sys.executable, str(ROOT/"scripts/check_nfcgs_checkpoint.py"), str(checkpoint)]
    if args.mode == "eval":
        checker += ["--step", str(STEPS)]
    subprocess.run(checker, cwd=ROOT, check=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if args.mode == "eval":
        path = args.run_dir/"eval/nfcgs_main/scores_all_avg.json"
        count = json.loads(path.read_text())["scene_count"]
        if count != 6991:
            raise RuntimeError(f"full-test evaluation incomplete: {count} scenes, expected 6991")


if __name__ == "__main__":
    main()

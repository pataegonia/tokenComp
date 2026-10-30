#!/usr/bin/env python3
"""Train/evaluate the main codec; --dry-run prints the exact command."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def baseline_checkpoint(rate: str) -> Path:
    tag = rate.replace(".", "p")
    name = f"nfcgs_score_probability10k_e1_split_rank56_lambda{tag}_residual_on_morton_on_m1c1s1_split"
    return (
        REPO
        / "outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56"
        / f"lambda{tag}/residual_on/checkpoints"
        / name
        / "version_0/step000010000.ckpt"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--rate-lambda", choices=("0", "0.0032", "0.0064", "0.0128", "0.0256", "0.0512"), default="0.0256",
                        help="coefficient of the rate term; 0 retains quantization/coding without the rate penalty")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="randomly initialize GlobalSplat and codec and train both (no checkpoint)",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="train GlobalSplat and codec together; implied by --from-scratch",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("DATASET_ROOT", "/data3/local_datasets/re10k")),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=1, help="GPUs on one node")
    parser.add_argument(
        "--launcher", choices=("python", "srun"), default="python",
        help="use srun inside a matching single-node SLURM allocation",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--scope", choices=("all", "score_probability"), default="all")
    parser.add_argument("--spatial-context", action=argparse.BooleanOptionalAction, default=True,
                        help="disable spatial prediction while retaining independent Split entropy models")
    parser.add_argument("--token-order", choices=("none", "morton", "hilbert", "nn_xyz", "nn_score"), default="hilbert")
    parser.add_argument("--nn-order-backend", choices=("legacy", "row", "knn", "graph", "graph_gemm"),
                        default="legacy",
                        help="NN implementation only; knn requires numba and uses exact full-row fallback")
    parser.add_argument("--nn-order-candidates", type=int, default=64)
    parser.add_argument("--nn-order-distance-device", choices=("cpu", "input"), default="cpu",
                        help="input computes distances on the token device, in FP32; legacy requires cpu")
    parser.add_argument("--score-order-metric", choices=("l1", "l2"), default="l1",
                        help="nn_score distance between normalized Scores")
    parser.add_argument("--score-context-schedule", choices=("legacy", "quarter2", "dyadic4", "single"), default="dyadic4")
    parser.add_argument("--score-spatial-stages", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--score-slice-channels", type=int, default=16,
                        help="channels per entropy stream; 56 encodes all Score channels together")
    parser.add_argument("--score-spatial-kernel", type=int, choices=(3, 5, 7))
    parser.add_argument(
        "--allow-score-path-conversion", action="store_true",
        help="allow score/order switch changes only for a weights-only training warm start",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="save every N optimizer steps (default: joint 10000, codec-only 5000)",
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulate", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.score_slice_channels <= 56:
        parser.error("--score-slice-channels must be between 1 and 56")
    if args.nn_order_candidates < 1:
        parser.error("--nn-order-candidates must be positive")
    if args.nn_order_backend == "legacy" and args.nn_order_distance_device != "cpu":
        parser.error("legacy NN backend requires --nn-order-distance-device cpu")
    if args.token_order not in ("nn_xyz", "nn_score") and args.nn_order_backend != "legacy":
        parser.error("optimized NN backend requires --token-order nn_xyz or nn_score")
    if args.allow_score_path_conversion and (
        args.mode != "train" or args.checkpoint is None or args.resume or args.from_scratch
    ):
        parser.error("--allow-score-path-conversion requires weights-only training from --checkpoint")
    if args.score_spatial_stages is None:
        args.score_spatial_stages = {"single": 1, "dyadic4": 4}.get(args.score_context_schedule, 2)
    if args.score_spatial_stages == 1 and args.score_context_schedule != "single":
        parser.error("one token stage requires --score-context-schedule single")
    if args.score_context_schedule == "single" and args.spatial_context:
        parser.error("single requires --no-spatial-context")
    if args.score_context_schedule != "legacy":
        expected_stages = {"single": 1, "quarter2": 2, "dyadic4": 4}[args.score_context_schedule]
        if args.score_spatial_stages != expected_stages:
            parser.error(f"{args.score_context_schedule} requires {expected_stages} spatial stages")
    if args.score_order_metric != "l1" and args.token_order != "nn_score":
        parser.error("--score-order-metric requires --token-order nn_score")
    if args.score_spatial_kernel is None:
        args.score_spatial_kernel = 5 if args.score_spatial_stages > 2 else 3
    if args.score_spatial_stages > 2 and args.score_context_schedule == "legacy":
        if args.score_spatial_kernel not in (5, 7):
            parser.error("3/4 token stages require kernel 5 or 7")
    if args.from_scratch and (args.mode != "train" or args.checkpoint or args.resume):
        parser.error("--from-scratch requires train without --checkpoint or --resume")
    args.joint = args.joint or args.from_scratch
    if args.joint and (args.mode != "train" or args.scope != "all"):
        parser.error("joint training requires train with --scope all")
    if args.mode == "eval" and args.resume:
        parser.error("--resume is only valid for train")
    if args.mode == "eval" and args.rate_lambda not in ("0.0064", "0.0256") and args.checkpoint is None:
        parser.error(f"lambda {args.rate_lambda} has no pretrained FullSplit parent; eval requires --checkpoint")
    if args.mode == "eval" and args.devices != 1:
        parser.error("actual-bitstream evaluation requires --devices 1")
    if args.mode == "train" and args.checkpoint is None and not args.from_scratch:
        parser.error("train requires --checkpoint or --from-scratch")
    args.lr = args.lr if args.lr is not None else (5e-4 if args.joint else 1e-4)
    args.batch_size = (
        args.batch_size if args.batch_size is not None else (1 if args.joint else 2)
    )
    args.accumulate = (
        args.accumulate if args.accumulate is not None else (8 if args.joint else 4)
    )
    args.checkpoint_every = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else (10000 if args.joint else 5000)
    )
    positive_integer_args = (
        "devices",
        "batch_size",
        "accumulate",
        "max_steps",
        "max_scenes",
        "checkpoint_every",
    )
    for name in positive_integer_args:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    return args


def build_command(args):
    checkpoint = (
        None
        if args.from_scratch
        else (args.checkpoint or baseline_checkpoint(args.rate_lambda)).resolve()
    )
    output = (
        args.output
        or REPO
        / "outputs/nfcgs_main"
        / ("joint" if args.joint else args.mode)
        / f"lambda{args.rate_lambda.replace('.', 'p')}"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    dataset = args.dataset_root.resolve()
    command = [
        sys.executable,
        "-m",
        "globalsplat.main",
        "+experiment=" + ("re10k_32k_nfcgs_joint" if args.joint else "re10k_32k_nfcgs"),
        f"checkpointing.load={checkpoint.as_posix() if checkpoint else 'null'}",
        "checkpointing.auto_resume=false",
        f"loss.rate_lambda={args.rate_lambda}",
        f"dataset.dataset_roots=[{dataset.as_posix()}]",
        f"dataset.mvsplat_root={(REPO / 'third_party/ZPressor/mvsplat').as_posix()}",
        f"trainer.precision={args.precision}",
        f"trainer.devices={args.devices}",
        "trainer.num_nodes=1",
        "trainer.strategy=" + (
            "ddp_find_unused_parameters_true" if args.devices > 1 else "auto"
        ),
        f"hydra.run.dir={(output / 'hydra').as_posix()}",
        f"model.feature_codec.score_spatial_context={str(args.spatial_context).lower()}",
        f"model.feature_codec.score_spatial_stages={args.score_spatial_stages}",
        f"model.feature_codec.score_slice_channels={args.score_slice_channels}",
        f"model.feature_codec.score_spatial_kernel={args.score_spatial_kernel}",
        f"model.feature_codec.token_order={args.token_order}",
        f"model.feature_codec.use_morton={str(args.token_order != 'none').lower()}",
        f"model.feature_codec.nn_order_backend={args.nn_order_backend}",
        f"model.feature_codec.nn_order_candidates={args.nn_order_candidates}",
        f"model.feature_codec.nn_order_distance_device={args.nn_order_distance_device}",
        f"model.feature_codec.score_context_schedule={args.score_context_schedule}",
    ]
    if args.score_order_metric != "l1":
        command.append(f"model.feature_codec.score_order_metric={args.score_order_metric}")
    if args.mode == "eval":
        command += [
            "mode=test",
            "dataset=re10k_eval_all_ctx12",
            "dataset.augment=false",
            "optimizer.batch_size=1",
            f"optimizer.num_workers={args.workers if args.workers is not None else 4}",
            "test.actual_bitstream=true",
            "test.compute_scores=true",
            "test.save_image=false",
            "test.save_gt_image=false",
            "test.save_input_images=false",
            "test.save_video=false",
            f"test.max_scenes={args.max_scenes or 'null'}",
            f"test.output_path={output.as_posix()}",
            "seed=0",
        ]
    else:
        if args.allow_score_path_conversion:
            command.append("checkpointing.allow_score_path_conversion=true")
        probability_only = args.scope == "score_probability"
        steps = args.max_steps or (
            500000 if args.joint else (10000 if probability_only else 50000)
        )
        milestones = (
            "[]" if args.joint else ("[7000]" if probability_only else "[35000,45000]")
        )
        command += [
            "mode=train",
            f"checkpointing.resume={str(args.resume).lower()}",
            f"model.feature_codec_train_scope={args.scope}",
            f"optimizer.lr={args.lr}",
            f"optimizer.lr_milestones={milestones}",
            f"optimizer.batch_size={args.batch_size}",
            f"optimizer.num_workers={args.workers if args.workers is not None else 8}",
            f"trainer.accumulate_grad_batches={args.accumulate}",
            f"trainer.max_steps={steps}",
            f"checkpointing.every_n_train_steps={args.checkpoint_every}",
            f"loss.quantile_update_interval={0 if probability_only else 500}",
            f"output_dir={(output / 'checkpoints').as_posix()}",
            f"log_dir={(output / 'tensorboard').as_posix()}",
        ]
    if args.launcher == "srun":
        # Construct the command/output timestamp ONCE before starting DDP ranks.
        command = [
            "srun", "--nodes=1", f"--ntasks={args.devices}",
            f"--ntasks-per-node={args.devices}", "--kill-on-bad-exit=1",
            "--gpu-bind=none",
        ] + command
    return command, checkpoint, dataset


def main(argv=None):
    args = parse_args(argv)
    command, checkpoint, dataset = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if args.launcher == "srun":
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("--launcher srun requires a SLURM allocation")
        if (
            int(os.environ.get("SLURM_NTASKS", "0")) != args.devices
            or int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1
        ):
            raise RuntimeError("SLURM allocation must have one node and one task per requested GPU")
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"missing main codec checkpoint: {checkpoint}")
    split = "test" if args.mode == "eval" else "train"
    if not (dataset / split / "index.json").is_file():
        raise FileNotFoundError(f"missing RE10K {split} index under {dataset}")
    if checkpoint is not None:
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/check_nfcgs_checkpoint.py"),
                str(checkpoint),
            ],
            cwd=REPO,
            check=True,
        )
    subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()

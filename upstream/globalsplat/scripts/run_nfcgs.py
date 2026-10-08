#!/usr/bin/env python3
"""Train/evaluate the main codec; --dry-run prints the exact command."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--rate-lambda", choices=("0.0064", "0.0256"), default="0.0256")
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
    parser.add_argument(
        "--score-path", choices=("full", "minimal"), default="full",
        help="minimal disables centering, score norm, mean/channel context and MLPs; keeps MSH/spatial Split",
    )
    for flag in ("centering", "score-norm", "mean-context", "channel-context", "nonlinear"):
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--token-order", choices=("morton", "hilbert", "nn_xyz", "nn_score"), default="morton")
    parser.add_argument("--score-order-scale", type=Path,
                        help="JSON list of rank positive scales in quantization units for nn_score; default: scene mean absolute values")
    parser.add_argument("--score-context-schedule", choices=("legacy", "quarter2", "dyadic4"), default="legacy")
    parser.add_argument("--score-spatial-stages", type=int, choices=(2, 3, 4))
    parser.add_argument("--score-spatial-kernel", type=int, choices=(3, 5, 7))
    parser.add_argument(
        "--score-context-quantization", choices=("noise", "ste"),
        help="ste uses reconstructed quantized values in training; likelihoods retain the noise relaxation",
    )
    parser.add_argument(
        "--allow-score-path-conversion", action="store_true",
        help="allow score/order switch changes only for a weights-only training warm start",
    )
    parser.add_argument(
        "--no-score-mean-offset",
        action="store_true",
        help="ablate b_mean while retaining mean-conditioned quantization step d",
    )
    parser.add_argument(
        "--reset-score-mean-offset",
        action="store_true",
        help="zero the b_mean head after a weights-only load in both ablation arms",
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
    parser.add_argument(
        "--dump-score-context",
        action="store_true",
        help="during eval, save score, mean-only/full-context residual, and b statistics",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_score_path_conversion and (
        args.mode != "train" or args.checkpoint is None or args.resume or args.from_scratch
    ):
        parser.error("--allow-score-path-conversion requires weights-only training from --checkpoint")
    default_enabled = args.score_path == "full"
    requested_mean_context = args.mean_context
    requested_channel_context = args.channel_context
    for name in ("centering", "score_norm", "mean_context", "channel_context", "nonlinear"):
        if getattr(args, name) is None:
            setattr(args, name, default_enabled)
    if not args.centering:
        # No scene mean can condition the decoder without a transmitted mean.
        if requested_mean_context is True:
            parser.error("--mean-context requires --centering")
        args.mean_context = False
    if args.score_spatial_stages is None:
        args.score_spatial_stages = 4 if args.score_context_schedule == "dyadic4" else 2
    if args.score_context_schedule != "legacy":
        expected_stages = 2 if args.score_context_schedule == "quarter2" else 4
        if args.score_spatial_stages != expected_stages:
            parser.error(f"{args.score_context_schedule} requires {expected_stages} spatial stages")
        if requested_channel_context is True:
            parser.error("anchor schedules require --no-channel-context")
        args.channel_context = False
    args.score_order_scale_values = None
    if args.score_order_scale is not None:
        if args.token_order != "nn_score":
            parser.error("--score-order-scale requires --token-order nn_score")
        try:
            from math import isfinite
            values = json.loads(args.score_order_scale.read_text(encoding="utf-8"))
            if not isinstance(values, list) or len(values) != 56 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0 for v in values
            ):
                raise ValueError("expected 56 finite positive numbers")
            args.score_order_scale_values = values
        except (OSError, ValueError) as error:
            parser.error(f"invalid --score-order-scale: {error}")
    if args.score_spatial_kernel is None:
        args.score_spatial_kernel = 5 if args.score_spatial_stages > 2 else 3
    if args.score_context_quantization is None:
        args.score_context_quantization = "ste" if args.score_spatial_stages > 2 or args.score_context_schedule != "legacy" else "noise"
    if args.score_spatial_stages > 2 and args.score_context_schedule == "legacy":
        if args.channel_context or args.score_spatial_kernel not in (5, 7):
            parser.error("3/4 token stages require --no-channel-context (or --score-path minimal) and kernel 5 or 7")
    if args.score_spatial_stages > 2 or args.score_context_schedule != "legacy":
        if args.dump_score_context:
            parser.error("staged/anchor contexts save stage bytes; even/odd b diagnostics require legacy 2 stages")
    if args.from_scratch and (args.mode != "train" or args.checkpoint or args.resume):
        parser.error("--from-scratch requires train without --checkpoint or --resume")
    args.joint = args.joint or args.from_scratch
    if args.joint and (args.mode != "train" or args.scope != "all"):
        parser.error("joint training requires train with --scope all")
    if args.mode == "eval" and args.resume:
        parser.error("--resume is only valid for train")
    if args.mode != "eval" and args.dump_score_context:
        parser.error("--dump-score-context is only valid for eval")
    if args.reset_score_mean_offset and (
        args.mode != "train" or args.checkpoint is None or args.resume
    ):
        parser.error("--reset-score-mean-offset requires weights-only training")
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
        f"model.score_mean_offset_enabled={str(not args.no_score_mean_offset).lower()}",
        f"dataset.dataset_roots=[{dataset.as_posix()}]",
        f"dataset.mvsplat_root={(REPO / 'third_party/ZPressor/mvsplat').as_posix()}",
        f"trainer.precision={args.precision}",
        f"trainer.devices={args.devices}",
        "trainer.num_nodes=1",
        "trainer.strategy=" + (
            "ddp_find_unused_parameters_true" if args.devices > 1 else "auto"
        ),
        f"hydra.run.dir={(output / 'hydra').as_posix()}",
        f"model.feature_codec.use_centering={str(args.centering).lower()}",
        f"model.feature_codec.use_score_norm={str(args.score_norm).lower()}",
        f"model.feature_codec.score_mean_condition={str(args.mean_context).lower()}",
        f"model.feature_codec.score_channel_context={str(args.channel_context).lower()}",
        f"model.feature_codec.transform={'nonlinear' if args.nonlinear else 'linear'}",
        f"model.feature_codec.score_spatial_stages={args.score_spatial_stages}",
        f"model.feature_codec.score_spatial_kernel={args.score_spatial_kernel}",
        f"model.feature_codec.score_context_quantization={args.score_context_quantization}",
        f"model.feature_codec.token_order={args.token_order}",
        f"model.feature_codec.score_context_schedule={args.score_context_schedule}",
        "model.feature_codec.score_order_scale=" + (json.dumps(args.score_order_scale_values, separators=(",", ":"))
                                                    if args.score_order_scale_values is not None else "null"),
    ]
    if args.mode == "eval":
        command += [
            "mode=test",
            "dataset=re10k_eval_all_ctx12",
            "dataset.augment=false",
            "optimizer.batch_size=1",
            f"optimizer.num_workers={args.workers if args.workers is not None else 4}",
            "test.actual_bitstream=true",
            f"test.score_context_diagnostics={str(args.dump_score_context).lower()}",
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
        if args.no_score_mean_offset and checkpoint is not None and not args.resume:
            command.append("checkpointing.allow_score_mean_offset_conversion=true")
        if args.reset_score_mean_offset:
            command.append("checkpointing.reset_score_mean_offset=true")
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

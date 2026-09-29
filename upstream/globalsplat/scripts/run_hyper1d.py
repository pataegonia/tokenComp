#!/usr/bin/env python3
"""Run a single-GPU Hyper1D pilot or actual-bitstream evaluation."""

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--vanilla-checkpoint", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("DATASET_ROOT", "/data3/local_datasets/re10k")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lambda", dest="rate_lambda", type=float, default=0.0256)
    parser.add_argument("--max-steps", type=int, default=16000, help="absolute optimizer step limit, including resumed steps")
    parser.add_argument("--train-hours", type=float, default=11.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=2000, help="updates; converted to training batches")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--max-scenes", type=int, default=128)
    parser.add_argument("--sample-test", action="store_true", help="sample test scene IDs with a fixed seed, excluding the run's validation scenes")
    parser.add_argument("--sample-seed", type=int, default=111123)
    parser.add_argument("--exclude-validation", type=Path, help="validation manifest.json; otherwise inferred from the checkpoint's run directory")
    parser.add_argument("--save-images", action="store_true", help="eval: save rendered target images and ground truth")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--strides", choices=("4x", "2x"), default="4x", help="initialization only; checkpoint architecture is restored automatically")
    parser.add_argument("--morton", action="store_true", help="initialization only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "eval" and (args.vanilla_checkpoint or args.resume):
        parser.error("eval requires --checkpoint without --resume")
    if args.resume and not args.checkpoint:
        parser.error("--resume requires a full training --checkpoint")
    if args.mode != "eval" and (args.sample_test or args.exclude_validation or args.save_images):
        parser.error("sample-test, exclude-validation and save-images are eval-only options")
    if args.exclude_validation and not args.sample_test:
        parser.error("--exclude-validation requires --sample-test")
    for name in ("max_steps", "warmup_steps", "batch_size", "accumulate", "validate_every", "checkpoint_every", "max_scenes"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.workers < 0 or not math.isfinite(args.train_hours) or args.train_hours <= 0 or not math.isfinite(args.rate_lambda) or args.rate_lambda <= 0:
        parser.error("invalid workers, train-hours or lambda")
    if args.warmup_steps >= args.max_steps:
        parser.error("warmup-steps must be smaller than max-steps")
    return args


def time_limit(hours):
    seconds = int(hours * 3600)
    if seconds < 1:
        raise ValueError("train-hours must allow at least one second")
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_command(args, output, checkpoint, codec_config=None):
    from globalsplat.compression import Hyper1DConfig
    config = codec_config or Hyper1DConfig(strides=(2, 2) if args.strides == "4x" else (2, 1), use_morton=args.morton)
    cmd = [sys.executable, "-m", "globalsplat.main", "+experiment=re10k_hyper1d_12h",
           f"checkpointing.load={checkpoint.resolve().as_posix()}", "checkpointing.auto_resume=false",
           f"checkpointing.resume={str(args.resume).lower()}", f"loss.rate_lambda={args.rate_lambda}",
           f"dataset.dataset_roots=[{args.dataset_root.resolve().as_posix()}]",
           f"dataset.mvsplat_root={(REPO / 'third_party/ZPressor/mvsplat').as_posix()}",
           f"trainer.precision={args.precision}", f"optimizer.num_workers={args.workers}",
           f"output_dir={(output / 'checkpoints').as_posix()}", f"log_dir={(output / 'logs').as_posix()}",
           f"hydra.run.dir={(output / 'hydra').as_posix()}",
           f"validation.output_path={(output / 'validation').as_posix()}"]
    for name, value in config.to_dict().items():
        if name in ("texture_channels", "geometry_channels"):
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, tuple):
            value = "[" + ",".join(map(str, value)) + "]"
        cmd.append(f"model.feature_codec.{name}={value}")
    if args.mode == "train":
        cmd += [f"trainer.max_steps={args.max_steps}", f"trainer.max_time={time_limit(args.train_hours)}",
                f"optimizer.warmup_pct={args.warmup_steps / args.max_steps}",
                f"optimizer.batch_size={args.batch_size}", f"trainer.accumulate_grad_batches={args.accumulate}",
                f"trainer.val_check_interval={args.validate_every * args.accumulate}",
                f"trainer.limit_val_batches={args.max_scenes}",
                f"checkpointing.every_n_train_steps={args.checkpoint_every}"]
    else:
        cmd += ["mode=test", "dataset=re10k_eval_all_ctx12", "optimizer.batch_size=1",
                "test.actual_bitstream=true", "test.score_context_diagnostics=false",
                f"test.save_image={str(args.save_images).lower()}",
                f"test.save_gt_image={str(args.save_images).lower()}",
                f"test.max_scenes={args.max_scenes}", f"test.output_path={(output / 'evaluation').as_posix()}"]
        if args.sample_test:
            cmd.append(f"+test.scene_subset_path={(output / 'test_sample.json').as_posix()}")
    return cmd


def main(argv=None):
    args = parse_args(argv)
    output = (args.output or REPO / "outputs/hyper1d" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    checkpoint = args.checkpoint or output / "hyper1d_initial.ckpt"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    initialize = None
    if args.vanilla_checkpoint:
        initialize = [sys.executable, str(REPO / "scripts/initialize_hyper1d_from_vanilla.py"),
                      "--vanilla", str(args.vanilla_checkpoint.resolve()), "--output", str(checkpoint),
                      "--strides", args.strides] + (["--morton"] if args.morton else [])
    if args.dry_run:
        config = None
        if args.checkpoint and args.checkpoint.is_file():
            from globalsplat.compression import Hyper1DConfig, load_feature_codec_checkpoint
            loaded = load_feature_codec_checkpoint(args.checkpoint)
            if not isinstance(loaded.config, Hyper1DConfig):
                raise ValueError("--checkpoint must contain a Hyper1D codec")
            config = loaded.config
        if initialize:
            print(shlex.join(initialize))
        print(shlex.join(build_command(args, output, checkpoint, config)))
        return
    for split in (("train", "test") if args.mode == "train" else ("test",)):
        if not (args.dataset_root / split / "index.json").is_file():
            raise FileNotFoundError(f"missing RE10K {split}/index.json in {args.dataset_root}")
    if initialize:
        print(shlex.join(initialize), flush=True)
        subprocess.run(initialize, cwd=REPO, env=env, check=True)
    from globalsplat.compression import Hyper1DConfig, load_feature_codec_checkpoint
    loaded = load_feature_codec_checkpoint(checkpoint)
    if not isinstance(loaded.config, Hyper1DConfig):
        raise ValueError("--checkpoint must contain a Hyper1D codec")
    config = loaded.config
    del loaded
    if args.sample_test:
        from globalsplat.dataset.test_subset import prepare_test_subset
        manifest = args.exclude_validation
        if manifest is None:
            manifest = next((parent / "validation/manifest.json" for parent in checkpoint.resolve().parents
                             if (parent / "validation/manifest.json").is_file()), None)
        if manifest is None:
            raise FileNotFoundError("validation manifest not found; pass --exclude-validation /path/to/manifest.json")
        subset = prepare_test_subset(args.dataset_root, output / "test_sample.json",
                                     args.max_scenes, args.sample_seed, manifest)
        print(f"Sampled {args.max_scenes} test scenes (seed={args.sample_seed}), excluding validation; saved {subset}", flush=True)
    if args.resume:
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not state.get("optimizer_states") or not state.get("lr_schedulers"):
            raise ValueError("--resume requires optimizer and scheduler state")
        if int(state.get("global_step", 0)) >= args.max_steps:
            raise ValueError("max-steps must exceed the checkpoint global_step")
        del state
    command = build_command(args, output, checkpoint, config)
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()

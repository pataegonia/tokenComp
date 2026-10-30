#!/usr/bin/env python3
"""Plan §13.4: 20k-step decoder-readout finetune and its control from fresh H-ON λ0.

Both variants load the same 120k weights only (no optimizer state) and keep the
fresh recipe (views, holdout exclusion, seed s0) with λ=0. ``finetune`` trains
the four Gaussian readouts at 1e-5 and the codec at 1e-6; ``control`` trains
only the codec at 1e-6 for the same steps. The full test reuses the fresh test
command and must reproduce the frozen fresh test frame cohort.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_nfcgs as base
import run_nfcgs_fresh_rd as runner
from globalsplat.fresh_rd import (SEEDS, atomic_json, canonical_overrides, checkpoint_path, directory_lock,
    file_hash, read_json, validate_cohort, validate_rows)

SOURCE = ("outputs/nfcgs_fresh_rd/branches/B120000/h_on/lambda0/s0/job_460346/"
          "checkpoints/nfcgs_main/version_0/step000120000.ckpt")
STEPS = 20000
CODEC_LR = 1e-6
DECODER_LR = 1e-5


def train_command(args):
    argv = ["train", "--checkpoint", str(args.source), "--output", str(args.run_dir),
        "--dataset-root", str(args.dataset_root), "--rate-lambda", "0",
        "--token-order", "hilbert", "--score-context-schedule", "dyadic4", "--score-spatial-stages", "4",
        "--score-spatial-kernel", "5", "--score-slice-channels", "16",
        "--devices", "1", "--launcher", "python", "--scope", "all", "--lr", str(CODEC_LR), "--batch-size", "2",
        "--accumulate", "4", "--workers", "8", "--max-steps", str(STEPS), "--checkpoint-every", "10000"]
    command, _, _ = base.build_command(base.parse_args(argv))
    changes = ["loss.rate_lambda=0", "++model.feature_codec.codec_mode=two_path", "model.feature_codec.rank=56",
        "model.feature_codec.use_residual=true", "test.record_scene_metrics=true", f"trainer.max_steps={STEPS}",
        # A milestone past the end keeps the learning rates constant.
        "optimizer.lr_milestones=[1000000000]", "checkpointing.exact_resume=true", "checkpointing.record_lr=true",
        "trainer.log_every_n_steps=1", "checkpointing.override_lr_milestones=false",
        f"seed={SEEDS[0][0]}", f"dataset.data_loader_seed={SEEDS[0][1]}",
        f"++dataset.train_holdout_manifest={(args.preparation / 'holdout.json').as_posix()}",
        "trainer.limit_val_batches=0", "trainer.num_sanity_val_steps=0",
        f"hydra.run.dir={(args.run_dir / 'hydra_train').as_posix()}"]
    if args.variant == "finetune":
        changes += ["++model.train_decoder_readouts=true", f"++optimizer.decoder_lr={DECODER_LR}"]
    return canonical_overrides(command, changes)


def test_command(args, checkpoint):
    test = runner.parse_args(["test", "--cell", "h_on", "--rate-lambda", "0", "--run-dir", str(args.run_dir),
        "--checkpoint", str(checkpoint), "--preparation", str(args.preparation),
        "--dataset-root", str(args.dataset_root)])
    return runner.build_command(test)[0]


def validate_train_config(cfg, args):
    from omegaconf import OmegaConf
    expected = {"optimizer.lr": CODEC_LR, "trainer.max_steps": STEPS, "loss.rate_lambda": 0.,
        "model.freeze_globalsplat": True, "checkpointing.resume": False, "trainer.devices": 1,
        "optimizer.batch_size": 2, "trainer.accumulate_grad_batches": 4, "dataset.num_context_views": 24,
        "dataset.num_target_views": 12, "seed": SEEDS[0][0], "dataset.data_loader_seed": SEEDS[0][1],
        "model.feature_codec.token_order": "hilbert", "model.feature_codec.score_context_schedule": "dyadic4",
        "model.train_decoder_readouts": True if args.variant == "finetune" else None,
        "optimizer.decoder_lr": DECODER_LR if args.variant == "finetune" else None}
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"decoder finetune config mismatch: {key}")
    if cfg.checkpointing.get("allow_score_path_conversion", False):
        raise ValueError("the H-ON source must load without score-path conversion")


def run(args):
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("use a fresh run directory")
    digest = file_hash(args.source)
    if args.source_sha256 and digest != args.source_sha256:
        raise ValueError("source checkpoint differs from the given fresh H-ON λ0 120k checkpoint hash")
    args.source_sha256 = digest
    command = train_command(args)
    validate_train_config(runner.compose_command(command), args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with directory_lock(args.run_dir):
        atomic_json(args.run_dir / "run.json", dict(variant=args.variant, source=str(args.source),
            source_sha256=args.source_sha256, steps=STEPS, codec_lr=CODEC_LR,
            decoder_lr=DECODER_LR if args.variant == "finetune" else None, train_command=command,
            code_sha256={name: file_hash(ROOT / name) for name in (*runner.RUNTIME_FILES,
                "scripts/run_nfcgs_decoder_finetune.py")}))
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        final = checkpoint_path(args.run_dir, STEPS)
        import torch
        if torch.load(final, map_location="cpu", weights_only=False).get("global_step") != STEPS:
            raise ValueError("training did not reach the requested step")
        digest = file_hash(final)
        command = test_command(args, final)
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        path = args.run_dir / "eval/nfcgs_main/scene_metrics.json"
        rows = read_json(path)
        validate_rows(rows, 6991, rate_required=True)
        cohort = read_json(args.output_root / "test_frame_cohort.json")["scenes"]
        cohort_sha256 = validate_cohort(rows, cohort)
        if file_hash(final) != digest:
            raise ValueError("checkpoint changed during evaluation")
        atomic_json(args.run_dir / "complete.json", dict(checkpoint=str(final), checkpoint_sha256=digest,
            scene_metrics_sha256=file_hash(path), frame_cohort_sha256=cohort_sha256, scene_count=6991))
    print(f"DECODER_FINETUNE_COMPLETE variant={args.variant} run_dir={args.run_dir}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("finetune", "control"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, default=ROOT / SOURCE)
    p.add_argument("--source-sha256", help="optional expected SHA-256 of the source; recorded either way")
    p.add_argument("--preparation", type=Path, default=runner.preparation_for(0))
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/nfcgs_fresh_rd/branches/B120000",
                   help="fresh root holding the frozen test frame cohort")
    p.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.run_dir, args.source, args.preparation = args.run_dir.resolve(), args.source.resolve(), args.preparation.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        command = train_command(args)
        validate_train_config(runner.compose_command(command), args)
        print(shlex.join(command))
        print(shlex.join(test_command(args, checkpoint_path(args.run_dir, STEPS))))
        return
    run(args)


if __name__ == "__main__":
    main()

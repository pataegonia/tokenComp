#!/usr/bin/env python3
"""Plan §14 redesign (2026-10-30): Top-K fine-tune from fresh H-ON 120k and its control.

Fresh Top-K runs collapsed to the K floor because a randomly initialized codec
gives no distortion cost for dropping tokens. Here both variants load the
trained H-ON s0 120k weights of the same λ only (no optimizer state; the new
router and scorer keep their own initialization, κ≈95%). Codec, router and
scorer train at a constant 1e-5 and the four Gaussian readouts at 1e-5 for
20k steps. ``topk`` adds the ``tk_h`` selector; ``control`` is identical
without it. The full test reuses the fresh test command of that cell and must
reproduce the frozen fresh test frame cohort.
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
    file_hash, rate_text, read_json, run_key, selection_overrides, validate_cohort, validate_rows)

LAMBDAS = ("0.0032", "0.0128", "0.0512")
CELLS = dict(topk="tk_h", control="h_on")
SOURCE_STEP = 120000
STEPS = 20000
LR = 1e-5
DECODER_LR = 1e-5


def default_source(output_root, rate):
    """The single H-ON s0 120k checkpoint of this λ in the fresh root."""
    run = Path(output_root) / run_key("h_on", rate, 0)
    found = sorted(run.glob(f"job_*/{checkpoint_path('', SOURCE_STEP).as_posix()}"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one H-ON s0 {SOURCE_STEP} checkpoint under {run}, "
                                f"found {len(found)}; pass --source")
    return found[0]


def train_command(args):
    argv = ["train", "--checkpoint", str(args.source), "--output", str(args.run_dir),
        "--dataset-root", str(args.dataset_root), "--rate-lambda", args.rate_lambda,
        "--token-order", "hilbert", "--score-context-schedule", "dyadic4", "--score-spatial-stages", "4",
        "--score-spatial-kernel", "5", "--score-slice-channels", "16",
        "--devices", "1", "--launcher", "python", "--scope", "all", "--lr", str(LR), "--batch-size", "2",
        "--accumulate", "4", "--workers", "8", "--max-steps", str(STEPS), "--checkpoint-every", "10000"]
    command, _, _ = base.build_command(base.parse_args(argv))
    changes = [f"loss.rate_lambda={args.rate_lambda}", "++model.feature_codec.codec_mode=two_path",
        "model.feature_codec.rank=56", "model.feature_codec.use_residual=true", "test.record_scene_metrics=true",
        f"trainer.max_steps={STEPS}",
        # A milestone past the end keeps the learning rates constant.
        "optimizer.lr_milestones=[1000000000]", "checkpointing.exact_resume=true", "checkpointing.record_lr=true",
        "trainer.log_every_n_steps=1", "checkpointing.override_lr_milestones=false",
        f"seed={SEEDS[0][0]}", f"dataset.data_loader_seed={SEEDS[0][1]}",
        f"++dataset.train_holdout_manifest={(args.preparation / 'holdout.json').as_posix()}",
        "trainer.limit_val_batches=0", "trainer.num_sanity_val_steps=0",
        f"hydra.run.dir={(args.run_dir / 'hydra_train').as_posix()}",
        "++model.train_decoder_readouts=true", f"++optimizer.decoder_lr={DECODER_LR}",
        *selection_overrides(CELLS[args.variant])]
    return canonical_overrides(command, changes)


def test_command(args, checkpoint):
    test = runner.parse_args(["test", "--cell", CELLS[args.variant], "--rate-lambda", args.rate_lambda,
        "--run-dir", str(args.run_dir), "--checkpoint", str(checkpoint), "--preparation", str(args.preparation),
        "--dataset-root", str(args.dataset_root)])
    return runner.build_command(test)[0]


def validate_train_config(cfg, args):
    from omegaconf import OmegaConf
    topk = args.variant == "topk"
    expected = {"optimizer.lr": LR, "trainer.max_steps": STEPS, "loss.rate_lambda": float(args.rate_lambda),
        "model.freeze_globalsplat": True, "checkpointing.resume": False, "trainer.devices": 1,
        "optimizer.batch_size": 2, "trainer.accumulate_grad_batches": 4, "dataset.num_context_views": 24,
        "dataset.num_target_views": 12, "seed": SEEDS[0][0], "dataset.data_loader_seed": SEEDS[0][1],
        "model.feature_codec.token_order": "hilbert", "model.feature_codec.score_context_schedule": "dyadic4",
        "model.train_decoder_readouts": True, "optimizer.decoder_lr": DECODER_LR,
        "model.token_selection.order": "slot" if topk else None,
        "model.token_selection.forced_count": 0 if topk else None}
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"Top-K finetune config mismatch: {key}")
    if cfg.checkpointing.get("allow_score_path_conversion", False):
        raise ValueError("the H-ON source must load without score-path conversion")


def run(args):
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("use a fresh run directory")
    digest = file_hash(args.source)
    if args.source_sha256 and digest != args.source_sha256:
        raise ValueError("source checkpoint differs from the given H-ON 120k checkpoint hash")
    args.source_sha256 = digest
    command = train_command(args)
    validate_train_config(runner.compose_command(command), args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with directory_lock(args.run_dir):
        atomic_json(args.run_dir / "run.json", dict(variant=args.variant, cell=CELLS[args.variant],
            rate_lambda=args.rate_lambda, source=str(args.source), source_sha256=args.source_sha256, steps=STEPS,
            lr=LR, decoder_lr=DECODER_LR, token_selection=selection_overrides(CELLS[args.variant]),
            train_command=command, code_sha256={name: file_hash(ROOT / name) for name in (*runner.RUNTIME_FILES,
                "globalsplat/compression/token_selection.py", "scripts/run_nfcgs_topk_finetune.py")}))
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
    print(f"TOPK_FINETUNE_COMPLETE variant={args.variant} lambda={args.rate_lambda} run_dir={args.run_dir}",
          flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=tuple(CELLS), required=True)
    p.add_argument("--rate-lambda", required=True, help="one of " + ", ".join(LAMBDAS))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, help="default: the single H-ON s0 120k checkpoint of this λ")
    p.add_argument("--source-sha256", help="optional expected SHA-256 of the source; recorded either way")
    p.add_argument("--preparation", type=Path, default=runner.preparation_for(0))
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/nfcgs_fresh_rd/branches/B120000",
                   help="fresh root holding the H-ON sources and the frozen test frame cohort")
    p.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.rate_lambda = rate_text(args.rate_lambda)
    if args.rate_lambda not in LAMBDAS:
        p.error(f"--rate-lambda must be one of {', '.join(LAMBDAS)}")
    args.source = args.source or default_source(args.output_root, args.rate_lambda)
    args.run_dir, args.source, args.preparation = args.run_dir.resolve(), args.source.resolve(), args.preparation.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        command = train_command(args)
        validate_train_config(runner.compose_command(command), args)
        print(f"SOURCE={args.source}")
        print(shlex.join(command))
        print(shlex.join(test_command(args, checkpoint_path(args.run_dir, STEPS))))
        return
    run(args)


if __name__ == "__main__":
    main()

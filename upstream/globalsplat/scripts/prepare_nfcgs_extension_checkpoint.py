#!/usr/bin/env python3
"""Prepare a full-state checkpoint for a scheduled training extension.

The model, optimizer moments, global step, and data-module state are preserved.
Only the optimizer LR fields and Lightning scheduler state are replaced with a
fresh warmup/cosine schedule covering the additional steps.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def extension_scheduler_state(
    num_param_groups: int,
    *,
    additional_steps: int,
    warmup_steps: int,
    start_lr: float,
    peak_lr: float,
    final_lr: float,
) -> dict:
    if num_param_groups <= 0:
        raise ValueError("optimizer must have at least one parameter group")
    if additional_steps <= 0:
        raise ValueError("additional_steps must be positive")
    if not 0 < warmup_steps < additional_steps:
        raise ValueError("warmup_steps must be between 0 and additional_steps")
    if not 0 < start_lr <= peak_lr:
        raise ValueError("start_lr must be positive and no greater than peak_lr")
    if not 0 < final_lr <= peak_lr:
        raise ValueError("final_lr must be positive and no greater than peak_lr")

    parameters = [torch.nn.Parameter(torch.zeros(())) for _ in range(num_param_groups)]
    groups = [{"params": [parameter]} for parameter in parameters]
    optimizer = torch.optim.AdamW(groups, lr=peak_lr)
    warmup = LinearLR(
        optimizer,
        start_factor=start_lr / peak_lr,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=additional_steps - warmup_steps,
        eta_min=final_lr,
    )
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    return scheduler.state_dict()


def prepare_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_start_step: int,
    target_step: int,
    warmup_steps: int,
    start_lr: float,
    peak_lr: float,
    final_lr: float,
) -> dict:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    start_step = int(checkpoint.get("global_step", -1))
    if start_step != expected_start_step:
        raise ValueError(
            f"checkpoint global_step is {start_step}, expected {expected_start_step}"
        )
    if target_step <= start_step:
        raise ValueError("target_step must be greater than the checkpoint global_step")

    optimizer_states = checkpoint.get("optimizer_states")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1:
        raise ValueError("checkpoint must contain exactly one optimizer state")
    param_groups = optimizer_states[0].get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("checkpoint optimizer has no parameter groups")
    schedulers = checkpoint.get("lr_schedulers")
    if not isinstance(schedulers, list) or len(schedulers) != 1:
        raise ValueError("checkpoint must contain exactly one LR scheduler state")

    additional_steps = target_step - start_step
    scheduler_state = extension_scheduler_state(
        len(param_groups),
        additional_steps=additional_steps,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        peak_lr=peak_lr,
        final_lr=final_lr,
    )
    for group in param_groups:
        group["lr"] = float(start_lr)
        group["initial_lr"] = float(peak_lr)
    checkpoint["lr_schedulers"] = [scheduler_state]
    checkpoint["nfcgs_extension"] = {
        "start_step": start_step,
        "target_step": int(target_step),
        "warmup_steps": int(warmup_steps),
        "start_lr": float(start_lr),
        "peak_lr": float(peak_lr),
        "final_lr": float(final_lr),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination if destination.exists() else temporary)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return checkpoint["nfcgs_extension"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-start-step", type=int, default=220_000)
    parser.add_argument("--target-step", type=int, default=500_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--start-lr", type=float, default=1e-5)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--final-lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_checkpoint(
        args.source.resolve(),
        args.destination.resolve(),
        expected_start_step=args.expected_start_step,
        target_step=args.target_step,
        warmup_steps=args.warmup_steps,
        start_lr=args.start_lr,
        peak_lr=args.peak_lr,
        final_lr=args.final_lr,
    )
    print(f"EXTENSION_CHECKPOINT={args.destination.resolve()}")
    print(
        "EXTENSION_SCHEDULE="
        f"step {metadata['start_step']}->{metadata['target_step']}, "
        f"lr {metadata['start_lr']}->{metadata['peak_lr']}->{metadata['final_lr']}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate an exact Score-order resume and optionally copy its full state."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globalsplat.compression import CodecConfig  # noqa: E402
from globalsplat.compression.checkpoint import validate_feature_codec_checkpoint  # noqa: E402


def resume_config():
    return CodecConfig(token_order="nn_score", use_centering=False, use_score_norm=False,
        score_mean_condition=False, score_channel_context=False, transform="linear",
        score_context_schedule="legacy", score_spatial_stages=3, score_spatial_kernel=5,
        score_context_quantization="ste", nn_order_backend="knn", nn_order_candidates=64,
        nn_order_distance_device="cpu")


def prepare(source: Path, target_steps: int, destination: Path | None = None,
            *, expected_config: CodecConfig | None = None) -> int:
    source = source.resolve(strict=True)
    before = source.stat()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if "model.feature_codec.shared_basis" not in checkpoint.get("state_dict", {}):
        raise ValueError("resume requires an integrated GlobalSplat + codec checkpoint")
    validate_feature_codec_checkpoint(checkpoint, expected_config or resume_config())
    if checkpoint.get("score_mean_offset_enabled", True) is not True:
        raise ValueError("this resume recipe retains the original score mean-offset setting")
    for key in ("optimizer_states", "lr_schedulers"):
        if not checkpoint.get(key):
            raise ValueError(f"full-state resume requires {key}; weights-only checkpoints cannot resume")
    if not checkpoint.get("pytorch-lightning_version"):
        raise ValueError("full-state resume requires pytorch-lightning_version metadata")
    step = checkpoint.get("global_step")
    if type(step) is not int or not 0 <= step < target_steps:
        raise ValueError(f"checkpoint global_step={step!r} must be below target {target_steps}")
    numbered = re.fullmatch(r"step(\d{9})\.ckpt", source.name)
    if numbered and int(numbered.group(1)) != step:
        raise ValueError("checkpoint filename and saved global_step disagree")
    # Free the potentially large CPU state before copying the original bytes.
    del checkpoint
    if destination is not None:
        destination = destination.resolve()
        if destination == source or destination.exists():
            raise ValueError("resume destination must be a new file, separate from the source")
        if not re.fullmatch(r"version_\d+", destination.parent.name):
            raise ValueError("resume destination must live inside a version_<N> directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="resume_copy_", delete=False) as f:
                temporary = Path(f.name)
            shutil.copyfile(source, temporary)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("source checkpoint changed during validation/copy; retry with a stable checkpoint")
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--target-step", type=int, default=50000)
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args(argv)
    if args.target_step < 1:
        parser.error("--target-step must be positive")
    step = prepare(args.checkpoint, args.target_step, args.copy_to)
    print(f"RESUME_FROM_STEP={step} TARGET_STEP={args.target_step} REMAINING_STEPS={args.target_step - step}")
    if args.copy_to:
        print(f"RESUME_CHECKPOINT_COPY={args.copy_to.resolve()}")


if __name__ == "__main__":
    main()

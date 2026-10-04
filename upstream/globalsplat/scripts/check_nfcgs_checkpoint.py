#!/usr/bin/env python3
"""Validate an evaluation checkpoint before launching the renderer."""

import argparse

import torch

from globalsplat.compression.checkpoint import infer_config, validate_feature_codec_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--step", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Evaluation consumes the integrated GlobalSplat state, not a codec-only artifact.
    if "model.feature_codec.shared_basis" not in checkpoint.get(
        "state_dict", checkpoint
    ):
        raise ValueError(
            "evaluation requires an integrated GlobalSplat codec checkpoint"
        )
    raw_state = checkpoint.get("state_dict", checkpoint)
    feature_state = {
        key.removeprefix("model.feature_codec."): value
        for key, value in raw_state.items() if key.startswith("model.feature_codec.")
    }
    config = infer_config(feature_state, checkpoint.get("feature_codec_config"))
    validate_feature_codec_checkpoint(checkpoint, config)
    step = checkpoint.get("global_step")
    if args.step != "auto" and (step is None or int(step) != int(args.step)):
        raise ValueError(f"checkpoint global_step={step}, expected {args.step}")
    print(f"checkpoint codec={config.to_dict()} global_step={step}")


if __name__ == "__main__":
    main()

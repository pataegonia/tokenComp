#!/usr/bin/env python3
"""Validate an evaluation checkpoint before launching the renderer."""

import argparse

import torch

from globalsplat.compression.config import CodecConfig
from globalsplat.compression.checkpoint import validate_feature_codec_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--transform", choices=("linear", "nonlinear"), default="linear")
    parser.add_argument("--transform-hidden", type=int, default=32)
    parser.add_argument("--residual", choices=("true", "false"), required=True)
    parser.add_argument("--morton", choices=("true", "false"), required=True)
    parser.add_argument("--step", default="50000")
    parser.add_argument("--score-mean-condition", choices=("true", "false"), default="false")
    parser.add_argument("--score-channel-context", choices=("true", "false"), default="false")
    parser.add_argument("--score-spatial-context", choices=("true", "false"), default="false")
    parser.add_argument(
        "--score-spatial-predictor",
        choices=("linear", "residual3", "residual7"),
        default="linear",
    )
    parser.add_argument(
        "--score-spatial-entropy",
        choices=("shared", "split", "gaussian", "conditional_scale"),
        default="shared",
    )
    parser.add_argument("--score-spatial-hidden", type=int, default=32)
    parser.add_argument("--score-slice-channels", type=int, default=16)
    parser.add_argument("--score-context-hidden", type=int, default=64)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Evaluation consumes the integrated GlobalSplat state, not a codec-only artifact.
    if "model.feature_codec.shared_basis" not in checkpoint.get("state_dict", checkpoint):
        raise ValueError("evaluation requires an integrated GlobalSplat codec checkpoint")
    config = CodecConfig(
        rank=args.rank, transform=args.transform, transform_hidden=args.transform_hidden,
        use_residual=args.residual == "true", use_morton=args.morton == "true",
        score_mean_condition=args.score_mean_condition == "true",
        score_channel_context=args.score_channel_context == "true",
        score_spatial_context=args.score_spatial_context == "true",
        score_spatial_predictor=args.score_spatial_predictor,
        score_spatial_entropy=args.score_spatial_entropy,
        score_spatial_hidden=args.score_spatial_hidden,
        score_slice_channels=args.score_slice_channels,
        score_context_hidden=args.score_context_hidden,
    )
    validate_feature_codec_checkpoint(checkpoint, config)
    step = checkpoint.get("global_step")
    if args.step != "auto" and (step is None or int(step) != int(args.step)):
        raise ValueError(f"checkpoint global_step={step}, expected {args.step}")
    print(f"checkpoint codec={config.to_dict()} global_step={step}")


if __name__ == "__main__":
    main()

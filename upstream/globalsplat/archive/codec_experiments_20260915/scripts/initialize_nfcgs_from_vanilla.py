#!/usr/bin/env python3
"""Create a trainable NFC-GS checkpoint from released GlobalSplat."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import pytorch_lightning as pl

from globalsplat.compression import (
    initialize_observable_from_vanilla,
    load_codec_initialization,
)
from globalsplat.model.globalsplat import GlobalSplat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--codec-init",
        type=Path,
        default=None,
        help="Optional PCA/codec init artifact; only shape-compatible codec tensors are loaded.",
    )
    parser.add_argument("--seed", type=int, default=111123)
    parser.add_argument("--rank", type=int, default=56)
    parser.add_argument("--transform", choices=("linear", "nonlinear"), default="linear")
    parser.add_argument("--transform-hidden", type=int, default=32)
    parser.add_argument("--use-residual", choices=("true", "false"), default="true")
    parser.add_argument("--use-morton", choices=("true", "false"), default="true")
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
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Optional trained integrated checkpoint copied into matching tensors before adding score context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    codec_checkpoint = None
    artifact_projection = None
    artifact_manifest = None
    vanilla_sha256 = None
    if args.codec_init is not None:
        codec_checkpoint = torch.load(args.codec_init, map_location="cpu", weights_only=False)
        codec_state = codec_checkpoint.get("state_dict", codec_checkpoint)
        if isinstance(codec_state, dict):
            artifact_projection = codec_state.get("geometry_basis")
            if artifact_projection is None:
                for key in (
                    "model.feature_codec.geometry_projection.weight",
                    "feature_codec.geometry_projection.weight",
                    "geometry_projection.weight",
                ):
                    if key in codec_state:
                        artifact_projection = codec_state[key]
                        break
        if isinstance(codec_checkpoint, dict):
            artifact_manifest = codec_checkpoint.get("manifest")

        if isinstance(artifact_manifest, dict):
            artifact_rank = artifact_manifest.get("rank")
            if artifact_rank is not None and int(artifact_rank) != args.rank:
                raise ValueError(
                    f"codec artifact rank={artifact_rank}, requested rank={args.rank}"
                )
            expected_sha256 = artifact_manifest.get("official_checkpoint_sha256")
            if expected_sha256:
                digest = hashlib.sha256()
                with args.vanilla.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                vanilla_sha256 = digest.hexdigest()
                if vanilla_sha256 != expected_sha256:
                    raise ValueError(
                        "vanilla checkpoint SHA-256 does not match the codec artifact: "
                        f"actual={vanilla_sha256} expected={expected_sha256}"
                    )

    model = GlobalSplat(
        sh_degree=3,
        static_only=True,
        patch_size=8,
        latent_rep_token_amount=4096,
        dim_latents=512,
        dim_rays=256,
        dim_rgb_feat=512,
        rounds=4,
        slot_calib_layers_per_round=2,
        M_max=16,
        use_camera_diff_as_input=False,
        feature_codec={
            "geometry_observable_channels": 224,
            "rank": args.rank,
            "residual_N": 192,
            "residual_M": 320,
            "adapter_hidden": 96,
            "morton_bits": 10,
            "transform": args.transform,
            "transform_hidden": args.transform_hidden,
            "use_residual": args.use_residual == "true",
            "use_morton": args.use_morton == "true",
            "score_mean_condition": args.score_mean_condition == "true",
            "score_channel_context": args.score_channel_context == "true",
            "score_spatial_context": args.score_spatial_context == "true",
            "score_spatial_predictor": args.score_spatial_predictor,
            "score_spatial_entropy": args.score_spatial_entropy,
            "score_spatial_hidden": args.score_spatial_hidden,
            "score_slice_channels": args.score_slice_channels,
            "score_context_hidden": args.score_context_hidden,
        },
    )
    report = initialize_observable_from_vanilla(
        model,
        args.vanilla,
        observable_projection=artifact_projection,
    )
    codec_init = None
    if args.codec_init is not None:
        count, keys = load_codec_initialization(model.feature_codec, codec_checkpoint)
        codec_init = {
            "path": str(args.codec_init),
            "loaded_tensors": count,
            "keys": keys,
            "projection_source": (
                "codec_artifact.geometry_basis"
                if artifact_projection is not None
                else "vanilla_checkpoint.qr"
            ),
            "manifest": artifact_manifest,
        }

    warm_start = None
    if args.warm_start is not None:
        source_checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        source_config = source_checkpoint.get("feature_codec_config")
        raw_source = source_checkpoint.get("state_dict", source_checkpoint)
        if not isinstance(source_config, dict):
            source_config = {}
        basis = raw_source.get("model.feature_codec.shared_basis")
        if basis is None:
            basis = raw_source.get("feature_codec.shared_basis")
        inferred_rank = int(basis.shape[0]) if torch.is_tensor(basis) else None
        has_nonlinear = any(
            str(key).startswith((
                "model.feature_codec.analysis_mlp.",
                "feature_codec.analysis_mlp.",
            ))
            for key in raw_source
        )
        source_values = {
            "rank": source_config.get("rank", inferred_rank),
            "transform": source_config.get("transform", "nonlinear" if has_nonlinear else "linear"),
            # Legacy paper checkpoints did not always record ablation flags.
            "use_residual": source_config.get("use_residual", True),
            "use_morton": source_config.get("use_morton", True),
        }
        for key, expected in {
            "rank": args.rank,
            "transform": args.transform,
            "use_residual": args.use_residual == "true",
            "use_morton": args.use_morton == "true",
        }.items():
            if source_values[key] != expected:
                raise ValueError(
                    f"warm-start {key}={source_values[key]!r}, requested {expected!r}"
                )
        source_context = {
            "score_mean_condition": bool(source_config.get("score_mean_condition", False)),
            "score_channel_context": bool(source_config.get("score_channel_context", False)),
            "score_spatial_context": bool(source_config.get("score_spatial_context", False)),
        }
        target_context = {
            "score_mean_condition": args.score_mean_condition == "true",
            "score_channel_context": args.score_channel_context == "true",
            "score_spatial_context": args.score_spatial_context == "true",
        }
        # A factorized checkpoint may add context from scratch.  Once the source
        # already has context, require the same causal structure so a wrong
        # Full/Spatial parent cannot silently become a partial warm start.
        if any(source_context.values()):
            for key, expected in target_context.items():
                if source_context[key] != expected:
                    raise ValueError(
                        f"warm-start {key}={source_context[key]!r}, requested {expected!r}"
                    )
            for key, expected in {
                "score_slice_channels": args.score_slice_channels,
                "score_context_hidden": args.score_context_hidden,
            }.items():
                actual = int(source_config.get(key, expected))
                if actual != expected:
                    raise ValueError(
                        f"warm-start {key}={actual!r}, requested {expected!r}"
                    )
            source_predictor = str(
                source_config.get("score_spatial_predictor", "linear")
            )
            if source_predictor not in ("linear", args.score_spatial_predictor):
                raise ValueError(
                    "warm-start score_spatial_predictor="
                    f"{source_predictor!r} cannot initialize {args.score_spatial_predictor!r}"
                )
            if source_predictor != "linear":
                source_spatial_hidden = int(
                    source_config.get("score_spatial_hidden", args.score_spatial_hidden)
                )
                if source_spatial_hidden != args.score_spatial_hidden:
                    raise ValueError(
                        "warm-start score_spatial_hidden="
                        f"{source_spatial_hidden!r}, requested {args.score_spatial_hidden!r}"
                    )
            source_spatial_entropy = str(
                source_config.get("score_spatial_entropy", "shared")
            )
            if source_spatial_entropy not in ("shared", args.score_spatial_entropy):
                raise ValueError(
                    "warm-start score_spatial_entropy="
                    f"{source_spatial_entropy!r} cannot initialize "
                    f"{args.score_spatial_entropy!r}"
                )
        target = model.state_dict()
        copied = []
        normalized_source = {}
        for raw_key, value in raw_source.items():
            key = str(raw_key)
            if key.startswith("model."):
                key = key[len("model.") :]
            normalized_source[key] = value
            if key in target and torch.is_tensor(value) and target[key].shape == value.shape:
                target[key] = value.to(dtype=target[key].dtype)
                copied.append(key)
        # Only split a factorized parent's entropy parameters when channel
        # context is being introduced.  A contextual parent already has trained
        # group entropies, copied above, and those must not be overwritten by
        # its inactive legacy score_entropy branch.
        if (
            args.score_channel_context == "true"
            and not source_context["score_channel_context"]
        ):
            source_prefix = "feature_codec.score_entropy.entropy_bottleneck."
            start = 0
            group_index = 0
            while start < args.rank:
                width = min(args.score_slice_channels, args.rank - start)
                target_prefix = (
                    f"feature_codec.score_context.group_entropies.{group_index}."
                    "entropy_bottleneck."
                )
                for target_key, target_value in list(target.items()):
                    if not target_key.startswith(target_prefix):
                        continue
                    suffix = target_key[len(target_prefix) :]
                    # CDF buffers are rebuilt during training; copying their
                    # non-empty saved shapes into fresh modules would require
                    # resizing buffers before this initializer's strict load.
                    if suffix.startswith("_"):
                        continue
                    source_value = normalized_source.get(source_prefix + suffix)
                    if not torch.is_tensor(source_value):
                        continue
                    if source_value.shape == target_value.shape:
                        target[target_key] = source_value.to(dtype=target_value.dtype)
                        copied.append(target_key)
                    elif (
                        source_value.ndim > 0
                        and source_value.shape[0] == args.rank
                        and target_value.shape[0] == width
                        and source_value.shape[1:] == target_value.shape[1:]
                    ):
                        target[target_key] = source_value[start : start + width].to(
                            dtype=target_value.dtype
                        )
                        copied.append(target_key)
                start += width
                group_index += 1
        # A split model starts as an exact probability/reconstruction copy of
        # its shared parent: duplicate each trained even/shared entropy model
        # into the new odd-pass model.  CDF buffers are rebuilt from the copied
        # density parameters and fixed quantiles by codec.update().
        if args.score_spatial_entropy == "split" and str(
            source_config.get("score_spatial_entropy", "shared")
        ) == "shared":
            for group_index in range(
                (args.rank + args.score_slice_channels - 1)
                // args.score_slice_channels
                if args.score_channel_context == "true"
                else 1
            ):
                if args.score_channel_context == "true":
                    source_prefix = (
                        f"feature_codec.score_context.group_entropies.{group_index}."
                        "entropy_bottleneck."
                    )
                else:
                    source_prefix = "feature_codec.score_entropy.entropy_bottleneck."
                target_prefix = (
                    f"feature_codec.score_context.spatial_odd_entropies.{group_index}."
                    "entropy_bottleneck."
                )
                for target_key, target_value in list(target.items()):
                    if not target_key.startswith(target_prefix):
                        continue
                    suffix = target_key[len(target_prefix) :]
                    if suffix.startswith("_"):
                        continue
                    source_value = normalized_source.get(source_prefix + suffix)
                    if torch.is_tensor(source_value) and source_value.shape == target_value.shape:
                        target[target_key] = source_value.to(dtype=target_value.dtype)
                        copied.append(target_key)
        model.load_state_dict(target, strict=True)
        warm_start = {
            "path": str(args.warm_start),
            "source_global_step": source_checkpoint.get("global_step"),
            "copied_tensors": len(copied),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {f"model.{key}": value.cpu() for key, value in model.state_dict().items()},
            # A weights-only checkpoint still needs this key when passed to
            # Lightning's Trainer.test(..., ckpt_path=...). The released
            # vanilla checkpoint uses the same minimal state+version format.
            "pytorch-lightning_version": pl.__version__,
            "nfcgs_initialization": report.to_dict(),
            "codec_initialization": codec_init,
            "score_context_warm_start": warm_start,
            "feature_codec_config": model.feature_codec.config.to_dict(),
            "vanilla_checkpoint_sha256": vanilla_sha256,
        },
        args.output,
    )
    print(f"wrote {args.output}")
    print(report.to_dict())
    if codec_init is not None:
        print(f"codec init loaded_tensors={codec_init['loaded_tensors']}")
    if warm_start is not None:
        print(f"score context warm-start={warm_start}")


if __name__ == "__main__":
    main()

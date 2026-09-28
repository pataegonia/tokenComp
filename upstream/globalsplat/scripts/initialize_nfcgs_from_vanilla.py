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
        help="Optional PCA/codec init artifact; loads the rank56 geometry/PCA statistics.",
    )
    parser.add_argument("--seed", type=int, default=111123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    codec_checkpoint = None
    artifact_projection = None
    artifact_manifest = None
    vanilla_sha256 = None
    if args.codec_init is not None:
        codec_checkpoint = torch.load(
            args.codec_init, map_location="cpu", weights_only=False
        )
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
            if artifact_rank is not None and int(artifact_rank) != 56:
                raise ValueError(
                    f"codec artifact rank={artifact_rank}, requested rank={56}"
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
        feature_codec={},
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                f"model.{key}": value.cpu() for key, value in model.state_dict().items()
            },
            # A weights-only checkpoint still needs this key when passed to
            # Lightning's Trainer.test(..., ckpt_path=...). The released
            # vanilla checkpoint uses the same minimal state+version format.
            "pytorch-lightning_version": pl.__version__,
            "nfcgs_initialization": report.to_dict(),
            "codec_initialization": codec_init,
            "feature_codec_config": model.feature_codec.config.to_dict(),
            "vanilla_checkpoint_sha256": vanilla_sha256,
        },
        args.output,
    )
    print(f"wrote {args.output}")
    print(report.to_dict())
    if codec_init is not None:
        print(f"codec init loaded_tensors={codec_init['loaded_tensors']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a calibrated-input Hyper1D checkpoint from unique TRAIN scenes."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from globalsplat.compression import Hyper1DConfig, resize_registered_buffers
from globalsplat.compression.calibration import ChannelMoments
from globalsplat.compression.checkpoint import infer_feature_codec_config, _feature_state_dict
from globalsplat.main import build_model, build_datamodule
from initialize_hyper1d_from_vanilla import file_sha256


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="prefer an untrained QR-initialized Hyper1D checkpoint")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=111123)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.scenes <= 0:
        raise ValueError("scenes must be positive")
    if not (args.dataset_root / "train/index.json").is_file():
        raise FileNotFoundError("calibration requires RE10K train/index.json")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_state, _ = _feature_state_dict(checkpoint["state_dict"])
    config = infer_feature_codec_config(feature_state, checkpoint.get("feature_codec_config"))
    if not isinstance(config, Hyper1DConfig):
        raise ValueError("calibration requires a Hyper1D checkpoint")
    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(repo / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=["+experiment=re10k_hyper1d_12h",
            f"dataset.dataset_roots=[{args.dataset_root.resolve().as_posix()}]",
            f"dataset.mvsplat_root={(repo / 'third_party/ZPressor/mvsplat').as_posix()}",
            "dataset.augment=false", "optimizer.batch_size=1", "optimizer.num_workers=0",
            f"dataset.data_loader_seed={args.seed}"])
    cfg.model.feature_codec = OmegaConf.create(config.to_dict())
    model = build_model(cfg.model).eval()
    state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
    resize_registered_buffers(model, state)
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    dm, _ = build_datamodule(cfg)
    moments = ChannelMoments(config.observable_channels)
    seen, records = set(), []
    for batch in dm.train_dataloader():
        scene = str(batch["scene_info"]["scene"][0])
        if scene in seen:
            continue
        inputs = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in batch["inputs"].items()}
        texture, geometry = model.encode_scene_tokens(inputs)
        features = torch.cat((texture, model.feature_codec.project_geometry(geometry)), dim=-1)
        moments.update(features)
        seen.add(scene)
        records.append({"scene": scene, "context_frame_ids": inputs["frame_ids"].cpu().reshape(-1).tolist()})
        if len(records) == args.scenes:
            break
    if len(records) != args.scenes:
        raise ValueError(f"only {len(records)} unique usable TRAIN scenes; requested {args.scenes}")
    mean, std = moments.buffers()
    model.feature_codec.config = replace(config, input_norm="calibrated")
    model.feature_codec.f_mean.copy_(mean.to(args.device))
    model.feature_codec.f_std.copy_(std.to(args.device))
    model.feature_codec.validate_normalization()
    projection = model.feature_codec.geometry_projection.weight.detach().cpu().contiguous()
    manifest = {"split": "train", "scene_count": len(records), "token_count": moments.count,
                "seed": args.seed, "scenes": records, "source_checkpoint_sha256": file_sha256(args.checkpoint),
                "train_index_sha256": file_sha256(args.dataset_root / "train/index.json"),
                "geometry_projection_sha256": hashlib.sha256(projection.numpy().tobytes()).hexdigest(),
                "std_floor": 1e-6}
    provenance = dict(checkpoint.get("hyper1d_provenance", {}))
    provenance["calibration"] = manifest
    # A changes the input function. Drop optimizer/scheduler state deliberately;
    # this is a weights-only initialization for retraining or fine tuning.
    result = {"state_dict": {"model." + key: value.cpu() for key, value in model.state_dict().items()},
              "feature_codec_config": model.feature_codec.config.to_dict(), "hyper1d_provenance": provenance,
              "pytorch-lightning_version": checkpoint.get("pytorch-lightning_version", "2.4.0")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    args.output.with_suffix(".calibration.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.output}: {len(records)} TRAIN scenes, {moments.count} tokens")


if __name__ == "__main__":
    main()

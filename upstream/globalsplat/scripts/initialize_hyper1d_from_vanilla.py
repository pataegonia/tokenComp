#!/usr/bin/env python3
"""Initialize the QR geometry boundary and random Hyper1D transforms."""

import argparse
import hashlib
from pathlib import Path
import torch
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from globalsplat.compression import initialize_observable_from_vanilla
from globalsplat.model.globalsplat import GlobalSplat


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strides", choices=("4x", "2x"), default="4x")
    parser.add_argument("--morton", action="store_true")
    parser.add_argument("--seed", type=int, default=111123)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.manual_seed(args.seed)
    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(repo / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=["+experiment=re10k_hyper1d_12h"])
    values = OmegaConf.to_container(cfg.model, resolve=True)
    values.pop("name")
    values["feature_codec"]["strides"] = (2, 2) if args.strides == "4x" else (2, 1)
    values["feature_codec"]["use_morton"] = args.morton
    model = GlobalSplat(**values)
    source = torch.load(args.vanilla, map_location="cpu", weights_only=False)
    source_state = source.get("state_dict", source)
    source_tensors = {key.removeprefix("module.").removeprefix("model."): value
                      for key, value in source_state.items()}
    source_keys = set(source_tensors)
    if any(key.startswith("feature_codec.") for key in source_keys):
        raise ValueError("--vanilla requires a GlobalSplat checkpoint without a feature codec")
    missing = [key for key in model.state_dict() if not key.startswith("feature_codec.") and key not in source_keys]
    if missing:
        raise ValueError(f"vanilla checkpoint lacks backbone tensors: {missing}")
    readouts = {"gaussian_decoder.geo_pos_readout.weight", "gaussian_decoder.geo_param_readout.weight",
                "gaussian_decoder.gate_readout.weight"}
    wrong = [key for key, value in model.state_dict().items()
             if not key.startswith("feature_codec.") and key not in readouts
             and source_tensors[key].shape != value.shape]
    if wrong:
        raise ValueError(f"vanilla checkpoint backbone shapes do not match the configured 32K model: {wrong}")
    report = initialize_observable_from_vanilla(model, source)
    projection = model.feature_codec.geometry_projection.weight.detach().cpu().contiguous()
    provenance = {"vanilla_checkpoint": str(args.vanilla.resolve()),
                  "vanilla_checkpoint_sha256": file_sha256(args.vanilla),
                  "geometry_projection_sha256": hashlib.sha256(projection.numpy().tobytes()).hexdigest(),
                  "qr_initialization": report.to_dict(), "seed": args.seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {f"model.{key}": value.cpu() for key, value in model.state_dict().items()},
                "feature_codec_config": model.feature_codec.config.to_dict(),
                "pytorch-lightning_version": pl.__version__, "hyper1d_provenance": provenance}, args.output)
    print(f"wrote {args.output}\n{report.to_dict()}")


if __name__ == "__main__":
    main()

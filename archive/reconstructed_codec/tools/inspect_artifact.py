"""Inspect an NFC-GS checkpoint or scene bitstream."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfcgs_codec import SceneBitstream, load_feature_codec_checkpoint  # noqa: E402


def inspect_checkpoint(path: Path) -> None:
    loaded = load_feature_codec_checkpoint(path)
    report = {
        "path": str(path),
        "source_prefix": loaded.source_prefix,
        "config": loaded.config.to_dict(),
        "checkpoint_metadata": loaded.checkpoint_metadata,
        "feature_state_tensors": len(loaded.codec.state_dict()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in loaded.codec.parameters()
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def inspect_bitstream(path: Path) -> None:
    raw_path = str(path.resolve())
    if os.name == "nt" and not raw_path.startswith("\\\\?\\"):
        raw_path = "\\\\?\\" + raw_path
    with open(raw_path, "rb") as stream:
        scene = SceneBitstream.unpack(stream.read())
    report = {
        "path": str(path),
        "format": scene.MAGIC.decode("ascii"),
        "version": scene.version,
        "flags": scene.flags,
        "points": scene.points,
        "channels": scene.channels,
        "rank": scene.rank,
        "stream_bytes": {
            "mean_fp16": len(scene.mean_fp16),
            "score": len(scene.score),
            "residual": len(scene.residual),
        },
        "residual_magic": scene.residual[:8].decode("ascii", errors="replace"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    if args.artifact.suffix == ".ckpt":
        inspect_checkpoint(args.artifact)
    else:
        inspect_bitstream(args.artifact)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run NFC-GS sender and receiver as independent processes.

The ``encode`` command writes one authenticated E2EM0301 bitstream per scene and
then exits.  The ``decode`` command starts from those files, decodes and renders
them, and computes the full-scene evaluation metrics.  No in-memory encoder
output crosses the process boundary.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from globalsplat.compression.bitstream import SceneBitstream
from globalsplat.compression.checkpoint import (
    resize_registered_buffers,
    validate_feature_codec_checkpoint,
    validate_score_mean_offset_mode,
)
from globalsplat.compression.config import CodecConfig


SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("encode", "decode"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bitstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="required for decode")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bf16-mixed", "16-mixed", "32-true"),
        default="bf16-mixed",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-step", type=int, default=10000)
    parser.add_argument("--timing-warmup-scenes", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args(argv)
    if args.mode == "decode" and args.output is None:
        parser.error("decode requires --output")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.expected_step < 0:
        parser.error("--expected-step must be non-negative")
    if args.timing_warmup_scenes < 0:
        parser.error("--timing-warmup-scenes must be non-negative")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    return args


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


def scene_name(batch: dict[str, Any]) -> str:
    value = batch["scene_info"]["scene"]
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def frame_ids(section: dict[str, Any]) -> list[int]:
    return [int(value) for value in section["frame_ids"].detach().cpu().reshape(-1)]


def checkpoint_identity(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "global_step": checkpoint.get("global_step"),
        "score_mean_offset_enabled": checkpoint.get(
            "score_mean_offset_enabled", True
        ),
    }


def load_model_and_data(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from globalsplat.main import build_datamodule, build_model

    checkpoint_path = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not (dataset_root / "test" / "index.json").is_file():
        raise FileNotFoundError(f"missing test/index.json under {dataset_root}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") != args.expected_step:
        raise ValueError(
            f"wrong checkpoint step: expected {args.expected_step}, "
            f"got {checkpoint.get('global_step')}"
        )
    if checkpoint.get("score_mean_offset_enabled", True) is not True:
        raise ValueError(
            "this baseline sender/receiver evaluation requires the existing "
            "score mean-offset (b) setting"
        )
    if "feature_codec_config" not in checkpoint:
        raise ValueError("checkpoint has no feature_codec_config metadata")
    codec_config = CodecConfig.from_mapping(checkpoint["feature_codec_config"])
    validate_feature_codec_checkpoint(checkpoint, codec_config)

    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key[len("model.") :]: value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not model_state:
        raise ValueError("an integrated model.* checkpoint is required")

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "config")
    ):
        cfg = compose(
            config_name="main",
            overrides=[
                "+experiment=re10k_32k_nfcgs",
                "dataset=re10k_eval_all_ctx12",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.model.feature_codec = OmegaConf.create(codec_config.to_dict())
    cfg.model.dim_latents = codec_config.geometry_channels
    cfg.model.latent_rep_token_amount = int(model_state["scene_tokens"].shape[0])
    cfg.model.freeze_globalsplat = True
    cfg.model.feature_codec_train_scope = "all"
    cfg.model.score_mean_offset_enabled = True
    cfg.dataset.dataset_roots = [str(dataset_root)]
    cfg.dataset.mvsplat_root = str(REPO_ROOT / "third_party/ZPressor/mvsplat")
    cfg.dataset.augment = False
    cfg.dataset.data_loader_seed = args.seed
    cfg.optimizer.batch_size = 1
    cfg.optimizer.num_workers = args.workers
    cfg.seed = args.seed

    model = build_model(cfg.model)
    validate_score_mean_offset_mode(checkpoint, model.feature_codec)
    resize_registered_buffers(model, model_state)
    model.load_state_dict(model_state, strict=True)
    device = torch.device(args.device)
    model.requires_grad_(False).eval().to(device)
    model.set_stage(3, mix=1.0)
    model.feature_codec.update(force=True, update_quantiles=False)
    data, _ = build_datamodule(cfg)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": checkpoint_identity(checkpoint_path, checkpoint),
        "codec": codec_config.to_dict(),
        "dataset_root": str(dataset_root),
        "protocol": "re10k_eval_all_ctx12",
        "num_context_views": 12,
        "num_target_views": 8,
        "seed": args.seed,
        "precision": args.precision,
    }
    return model, data, metadata, device


def amp_context(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "32-true":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_stage(device: torch.device, operation):
    synchronize(device)
    start = perf_counter()
    result = operation()
    synchronize(device)
    return result, perf_counter() - start


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def timing_summary(
    rows: list[dict[str, Any]], fields: Iterable[str], warmup: int
) -> dict[str, Any]:
    used = rows[min(warmup, len(rows)) :]
    result: dict[str, Any] = {
        "warmup_scenes_excluded": len(rows) - len(used),
        "timed_scenes": len(used),
    }
    for field in fields:
        values = [float(row[field]) for row in used]
        if values:
            result[field] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "total": sum(values),
            }
    return result


@torch.no_grad()
def encode(args: argparse.Namespace) -> None:
    output = args.bitstream_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"bitstream directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    model, data, metadata, device = load_model_and_data(args)
    metadata["role"] = "sender"
    write_json(output / "metadata.json", metadata)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    manifest_partial = output / "manifest.jsonl.partial"
    with manifest_partial.open("x", encoding="utf-8") as manifest:
        for index, batch in enumerate(data.test_dataloader()):
            scene = scene_name(batch)
            if scene in seen:
                raise RuntimeError(f"duplicate scene in full evaluation: {scene}")
            seen.add(scene)
            inputs = to_device(batch["inputs"], device)

            encoded_patch, scene_encoder_seconds = elapsed_stage(
                device,
                lambda: _encode_scene_tokens(model, inputs, args.precision, device),
            )
            compressed, entropy_encode_seconds = elapsed_stage(
                device,
                lambda: _compress_scene_tokens(
                    model, encoded_patch, args.precision, device
                ),
            )
            del encoded_patch, inputs

            payload = compressed.data
            packed = SceneBitstream.unpack(payload)
            filename = f"{index:06d}.bin"
            write_start = perf_counter()
            write_bytes_atomic(output / filename, payload)
            write_seconds = perf_counter() - write_start
            streams = packed.bytes_by_stream
            row = {
                "index": index,
                "scene": scene,
                "file": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "points": packed.points,
                "context_frame_ids": frame_ids(batch["inputs"]),
                "target_frame_ids": frame_ids(batch["targets"]),
                "stream_bytes": streams,
                "scene_encoder_seconds": scene_encoder_seconds,
                "entropy_encode_seconds": entropy_encode_seconds,
                "encoder_seconds": scene_encoder_seconds + entropy_encode_seconds,
                "bitstream_write_seconds": write_seconds,
                "encoder_to_file_seconds": (
                    scene_encoder_seconds + entropy_encode_seconds + write_seconds
                ),
            }
            rows.append(row)
            manifest.write(json.dumps(row, allow_nan=False) + "\n")
            manifest.flush()
            if len(rows) % args.log_every == 0:
                print(
                    f"ENCODE scenes={len(rows)} last={scene} "
                    f"bytes={len(payload)}",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("full-scene test loader produced no scenes")
    manifest_partial.replace(output / "manifest.jsonl")
    summary = {
        **metadata,
        "scene_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "mean_bytes_per_scene": statistics.fmean(row["bytes"] for row in rows),
        "timing": timing_summary(
            rows,
            (
                "scene_encoder_seconds",
                "entropy_encode_seconds",
                "encoder_seconds",
                "bitstream_write_seconds",
                "encoder_to_file_seconds",
            ),
            args.timing_warmup_scenes,
        ),
    }
    write_json(output / "encoder_summary.json", summary)
    write_json(
        output / "COMPLETE.json",
        {"schema_version": SCHEMA_VERSION, "scene_count": len(rows)},
    )
    print(
        f"ENCODE_COMPLETE scenes={len(rows)} bitstreams={output} "
        f"bytes={summary['total_bytes']}",
        flush=True,
    )


def _encode_scene_tokens(model, inputs, precision: str, device: torch.device):
    with amp_context(precision, device):
        return model.encode_scene_tokens(inputs)


def _compress_scene_tokens(model, tokens, precision: str, device: torch.device):
    with amp_context(precision, device):
        return model.compress_scene_tokens(tokens)


def load_manifest(bitstream_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_path = bitstream_dir / "metadata.json"
    manifest_path = bitstream_dir / "manifest.jsonl"
    complete_path = bitstream_dir / "COMPLETE.json"
    for path in (metadata_path, manifest_path, complete_path):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete sender output; missing {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("scene_count") != len(rows):
        raise ValueError("sender completion marker and manifest scene counts differ")
    by_scene = {row["scene"]: row for row in rows}
    if len(by_scene) != len(rows):
        raise ValueError("sender manifest contains duplicate scenes")
    return metadata, rows


def validate_receiver_identity(
    sender: dict[str, Any], receiver: dict[str, Any]
) -> None:
    for key in (
        "schema_version",
        "checkpoint",
        "codec",
        "dataset_root",
        "protocol",
        "num_context_views",
        "num_target_views",
        "seed",
        "precision",
    ):
        if sender.get(key) != receiver.get(key):
            raise ValueError(f"sender/receiver mismatch for {key}")


def discard_encoder(model) -> None:
    """Release modules that the receiver must never call."""

    model.rgb_patch_embeds = None
    model.ray_patch_embeds = None
    model.view_pe = None
    model.slot_encoder = None
    model.scene_tokens = None
    model._resnet_mean = None
    model._resnet_std = None


@torch.no_grad()
def decode(args: argparse.Namespace) -> None:
    bitstream_dir = args.bitstream_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"decoder output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    sender_metadata, manifest_rows = load_manifest(bitstream_dir)
    manifest = {row["scene"]: row for row in manifest_rows}
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    model, data, receiver_metadata, device = load_model_and_data(args)
    validate_receiver_identity(sender_metadata, receiver_metadata)
    discard_encoder(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from globalsplat.dataset.data_module import add_repo_to_path
    from globalsplat.model.rendering import render_static_batched

    add_repo_to_path(
        REPO_ROOT / "third_party/ZPressor/mvsplat", repo_name="upstream-eval"
    )
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_scene_path = output / "per_scene.jsonl"
    with per_scene_path.open("x", encoding="utf-8") as handle:
        for batch in data.test_dataloader():
            scene = scene_name(batch)
            expected = manifest.get(scene)
            if expected is None:
                raise RuntimeError(f"decoder dataset scene is absent from manifest: {scene}")
            if scene in seen:
                raise RuntimeError(f"duplicate decoder scene: {scene}")
            if frame_ids(batch["inputs"]) != expected["context_frame_ids"]:
                raise RuntimeError(f"context frame identity mismatch for scene {scene}")
            if frame_ids(batch["targets"]) != expected["target_frame_ids"]:
                raise RuntimeError(f"target frame identity mismatch for scene {scene}")
            seen.add(scene)

            stream_path = bitstream_dir / expected["file"]
            read_start = perf_counter()
            payload = stream_path.read_bytes()
            bitstream_read_seconds = perf_counter() - read_start
            if len(payload) != expected["bytes"]:
                raise RuntimeError(f"bitstream length mismatch for scene {scene}")
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"bitstream SHA-256 mismatch for scene {scene}")
            SceneBitstream.unpack(payload)

            decoded_patch, entropy_decode_seconds = elapsed_stage(
                device,
                lambda: _decompress_scene_tokens(
                    model, payload, args.precision, device
                ),
            )
            gaussians, gaussian_decoder_seconds = elapsed_stage(
                device,
                lambda: _decode_scene_tokens(
                    model, decoded_patch, args.precision, device
                ),
            )
            del decoded_patch

            # Only target cameras/images move to the receiver GPU. Context images
            # remain on the CPU and are never passed to the model in this job.
            targets = to_device(batch["targets"], device)
            B, T, _, H, W = targets["images"].shape
            rendered, renderer_seconds = elapsed_stage(
                device,
                lambda: render_static_batched(
                    gaussians, targets, render_depth=False
                ),
            )
            pred = rendered["img"].view(B, T, 3, H, W)[0].clamp(0, 1)
            gt = targets["images"][0].float().clamp(0, 1)
            psnr = compute_psnr(gt, pred).mean().item()
            ssim = compute_ssim(gt, pred).mean().item()
            lpips = compute_lpips(gt, pred).mean().item()
            row = {
                "index": expected["index"],
                "scene": scene,
                "bytes": expected["bytes"],
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "bitstream_read_seconds": bitstream_read_seconds,
                "entropy_decode_seconds": entropy_decode_seconds,
                "gaussian_decoder_seconds": gaussian_decoder_seconds,
                "decoder_seconds": entropy_decode_seconds
                + gaussian_decoder_seconds,
                "renderer_seconds_per_scene": renderer_seconds,
                "renderer_seconds_per_view": renderer_seconds / T,
            }
            rows.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            if len(rows) % args.log_every == 0:
                print(
                    f"DECODE scenes={len(rows)}/{len(manifest_rows)} last={scene} "
                    f"psnr={psnr:.4f}",
                    flush=True,
                )
            del gaussians, rendered, pred, gt, targets

    missing = set(manifest) - seen
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(
            f"decoder loader ended before {len(missing)} manifest scenes; sample={sample}"
        )
    if len(rows) != len(manifest_rows):
        raise RuntimeError("decoder and sender scene counts differ")

    averages = {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in ("psnr", "ssim", "lpips", "bytes")
    }
    timing = timing_summary(
        rows,
        (
            "bitstream_read_seconds",
            "entropy_decode_seconds",
            "gaussian_decoder_seconds",
            "decoder_seconds",
            "renderer_seconds_per_scene",
            "renderer_seconds_per_view",
        ),
        args.timing_warmup_scenes,
    )
    summary = {
        **receiver_metadata,
        "role": "receiver",
        "source_bitstream_dir": str(bitstream_dir),
        "scene_count": len(rows),
        "averages": averages,
        "timing": timing,
    }
    write_json(output / "decoder_summary.json", summary)
    write_json(
        output / "scores_all_avg.json",
        {"scene_count": len(rows), **averages},
    )
    for metric in ("psnr", "ssim", "lpips"):
        write_json(output / f"scores_{metric}_all.json", [row[metric] for row in rows])
    print(
        "DECODE_COMPLETE "
        f"scenes={len(rows)} psnr={averages['psnr']:.4f} "
        f"ssim={averages['ssim']:.4f} lpips={averages['lpips']:.4f} "
        f"output={output}",
        flush=True,
    )


def _decompress_scene_tokens(model, payload, precision: str, device: torch.device):
    with amp_context(precision, device):
        return model.decompress_scene_tokens(payload)


def _decode_scene_tokens(model, tokens, precision: str, device: torch.device):
    with amp_context(precision, device):
        return model.decode_scene_tokens(tokens)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "encode":
        encode(args)
    else:
        decode(args)


if __name__ == "__main__":
    main()

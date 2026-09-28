#!/usr/bin/env python3
"""Measure deployment overhead of one integrated NFCGS checkpoint.

This is deliberately a read-only, no-training benchmark.  It times the score
entropy path on precomputed score tensors as well as the complete feature
codec, and separates entropy strings from per-scene containers.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

from globalsplat.compression.bitstream import SceneBitstream, ScoreContextBitstream
from globalsplat.compression.checkpoint import (
    resize_registered_buffers,
    validate_feature_codec_checkpoint,
)
from globalsplat.compression.config import CodecConfig


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


def scene_name(batch) -> str:
    value = batch["scene_info"]["scene"]
    return str(value[0] if isinstance(value, (list, tuple)) else value)


def _serialized_bytes(state: dict[str, torch.Tensor]) -> int:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.tell()


def module_footprint(named_modules: list[tuple[str, nn.Module]]) -> dict[str, int]:
    """Return deduplicated parameter/buffer and portable state-dict sizes."""

    parameters: dict[int, torch.Tensor] = {}
    buffers: dict[int, torch.Tensor] = {}
    state: dict[str, torch.Tensor] = {}
    for prefix, module in named_modules:
        for name, value in module.named_parameters(recurse=True):
            parameters.setdefault(id(value), value)
            state.setdefault(f"{prefix}.{name}", value.detach().cpu())
        for name, value in module.named_buffers(recurse=True):
            buffers.setdefault(id(value), value)
            state.setdefault(f"{prefix}.{name}", value.detach().cpu())
    return {
        "parameter_count": sum(value.numel() for value in parameters.values()),
        "parameter_bytes": sum(value.numel() * value.element_size() for value in parameters.values()),
        "buffer_count": sum(value.numel() for value in buffers.values()),
        "buffer_bytes": sum(value.numel() * value.element_size() for value in buffers.values()),
        "tensor_bytes": sum(value.numel() * value.element_size() for value in parameters.values())
        + sum(value.numel() * value.element_size() for value in buffers.values()),
        "serialized_state_dict_bytes": _serialized_bytes(state) if state else 0,
    }


def score_module_groups(codec) -> dict[str, list[tuple[str, nn.Module]]]:
    context = codec.score_context
    if context is None:
        base = [("score_entropy", codec.score_entropy)]
        return {"score_path": base, "context_predictor": [], "probability_model": base}

    score_path = [("score_context", context)]
    if not context.channel_context:
        # Mean-only and Mean+Spatial use the legacy base entropy bottleneck.
        score_path.append(("score_entropy", codec.score_entropy))
    predictors: list[tuple[str, nn.Module]] = []
    for name in (
        "mean_conditioner",
        "channel_predictors",
        "spatial_predictors",
        "spatial_corrections",
    ):
        module = getattr(context, name)
        if module is not None:
            predictors.append((name, module))
    probability: list[tuple[str, nn.Module]] = []
    for index, entropy in enumerate(context.active_entropies(codec.score_entropy)):
        probability.append((f"active_entropy_{index}", entropy))
    if context.spatial_gaussian is not None:
        probability.append(("spatial_gaussian", context.spatial_gaussian))
    if len(context.spatial_log_scales):
        probability.append(("spatial_log_scales", context.spatial_log_scales))
    if len(context.spatial_scale_predictors):
        probability.append(("spatial_scale_predictors", context.spatial_scale_predictors))
    return {
        "score_path": score_path,
        "context_predictor": predictors,
        "probability_model": probability,
    }


def load_model_and_data(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from globalsplat.main import build_datamodule, build_model

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "feature_codec_config" not in checkpoint:
        raise ValueError("profiler requires feature_codec_config checkpoint metadata")
    config = CodecConfig.from_mapping(checkpoint["feature_codec_config"])
    validate_feature_codec_checkpoint(checkpoint, config)
    if args.expected_step is not None and checkpoint.get("global_step") != args.expected_step:
        raise ValueError(
            f"wrong checkpoint step: expected {args.expected_step}, got {checkpoint.get('global_step')}"
        )
    model_state = {
        key[len("model.") :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    if not model_state:
        raise ValueError("an integrated model.* checkpoint is required")
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "config")):
        cfg = compose(
            config_name="main",
            overrides=["+experiment=re10k_32k_nfcgs_paper", "dataset=re10k_eval_all_ctx12"],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.model.feature_codec = OmegaConf.create(config.to_dict())
    cfg.model.dim_latents = config.geometry_channels
    cfg.model.latent_rep_token_amount = int(model_state["scene_tokens"].shape[0])
    cfg.model.freeze_globalsplat = True
    cfg.model.feature_codec_train_scope = "all"
    cfg.dataset.dataset_roots = [str(args.dataset_root.resolve())]
    cfg.dataset.mvsplat_root = str(REPO_ROOT / "third_party/ZPressor/mvsplat")
    cfg.dataset.augment = False
    cfg.dataset.data_loader_seed = args.seed
    cfg.optimizer.batch_size = 1
    cfg.optimizer.num_workers = args.num_workers
    cfg.seed = args.seed

    model = build_model(cfg.model)
    resize_registered_buffers(model, model_state)
    model.load_state_dict(model_state, strict=True)
    model.requires_grad_(False).eval()
    # Rebuild the exact tables that will be used by the benchmark before
    # measuring their deployment footprint.
    model.feature_codec.update(force=True, update_quantiles=False)
    footprints = {
        "codec": module_footprint([("feature_codec", model.feature_codec)]),
        **{
            name: module_footprint(modules)
            for name, modules in score_module_groups(model.feature_codec).items()
        },
    }
    model.to(args.device)
    model.set_stage(3, mix=1.0)
    data, _ = build_datamodule(cfg)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_step": checkpoint.get("global_step"),
        "codec": config.to_dict(),
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
    }
    return model, data, metadata, footprints


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def timed_call(function, device: str):
    """Time one call with device synchronization and CUDA peak-memory delta."""

    _sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device)
    else:
        allocated = 0
    started = time.perf_counter_ns()
    result = function()
    _sync(device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    peak_delta = (
        max(0, torch.cuda.max_memory_allocated(device) - allocated)
        if device.startswith("cuda")
        else 0
    )
    return result, elapsed_ms, peak_delta


def prepare_score(codec, texture, geometry, positions):
    features = codec._pack_features(texture, geometry)
    order = codec._make_order(positions)
    features = order.apply(features)
    mean = codec._quantize_mean(features.mean(dim=1))
    centered = features - mean[:, None, :]
    score = codec._analyze_low_rank(centered)
    normalized = score / codec.score_scale[None, None, :]
    return normalized.transpose(1, 2).unsqueeze(2), mean


def score_encode(codec, score_nchw, mean):
    if codec.score_context is None:
        strings = codec.score_entropy.compress(score_nchw)
        decoded = codec.score_entropy.decompress(strings, score_nchw.shape[-2:])
        return strings[0], decoded
    with torch.autocast(device_type=score_nchw.device.type, enabled=False):
        return codec.score_context.compress(score_nchw.float(), mean.float(), codec.score_entropy)


def score_decode(codec, payload: bytes, mean, points: int):
    if codec.score_context is None:
        return codec.score_entropy.decompress([payload], (1, points))
    with torch.autocast(device_type=mean.device.type, enabled=False):
        return codec.score_context.decompress(payload, mean.float(), points, codec.score_entropy)


def score_payload_parts(payload: bytes, contextual: bool) -> dict[str, int]:
    if not contextual:
        return {
            "score_payload_bytes": len(payload),
            "score_entropy_string_bytes": len(payload),
            "score_wrapper_bytes": 0,
            "score_stream_count": 1,
        }
    packed = ScoreContextBitstream.unpack(payload)
    raw = sum(len(value) for value in packed.strings)
    return {
        "score_payload_bytes": len(payload),
        "score_entropy_string_bytes": raw,
        "score_wrapper_bytes": len(payload) - raw,
        "score_stream_count": len(packed.strings),
    }


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "stdev": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty overhead rows")
    numeric = (
        "score_encode_ms",
        "score_decode_ms",
        "feature_codec_encode_ms",
        "feature_codec_decode_ms",
        "score_encode_peak_cuda_bytes",
        "score_decode_peak_cuda_bytes",
        "feature_codec_encode_peak_cuda_bytes",
        "feature_codec_decode_peak_cuda_bytes",
        "total_bytes",
        "score_payload_bytes",
        "score_entropy_string_bytes",
        "score_wrapper_bytes",
        "residual_bytes",
        "mean_bytes",
        "outer_and_residual_wrapper_bytes",
    )
    result = {key: distribution([float(row[key]) for row in rows]) for key in numeric}
    result["score_stream_count"] = rows[0]["score_stream_count"]
    if any(row["score_stream_count"] != result["score_stream_count"] for row in rows):
        raise RuntimeError("score stream count changed between scenes")
    result["score_roundtrip_exact_all"] = all(row["score_roundtrip_exact"] for row in rows)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--suite", required=True, choices=("context", "probability"))
    parser.add_argument("--lambda-tag", required=True, choices=("0p0064", "0p0256"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup-scenes", type=int, default=4)
    parser.add_argument("--measure-scenes", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--expected-step", type=int)
    args = parser.parse_args()
    if args.warmup_scenes < 0 or min(args.measure_scenes, args.threads) < 1:
        parser.error("warmup-scenes must be nonnegative; measure-scenes and threads must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if args.precision == "bf16" and not args.device.startswith("cuda"):
        parser.error("bf16 profiling requires CUDA; use fp32 for CPU")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if not (args.dataset_root / "test" / "index.json").is_file():
        raise FileNotFoundError(f"missing test/index.json under {args.dataset_root}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("use a fresh output directory; existing results are preserved")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    model, data, metadata, footprints = load_model_and_data(args)
    codec = model.feature_codec
    amp = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.precision == "bf16"
        else nullcontext
    )
    try:
        import compressai

        compressai_version = compressai.__version__
    except AttributeError:
        compressai_version = "unknown"
    environment = {
        "torch": torch.__version__,
        "compressai": compressai_version,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
    }
    rows: list[dict] = []
    scene_ids: list[str] = []
    seen: set[str] = set()
    target = args.warmup_scenes + args.measure_scenes
    print(
        f"OVERHEAD_START suite={args.suite} label={args.label} lambda={args.lambda_tag} "
        f"warmup={args.warmup_scenes} measure={args.measure_scenes}",
        flush=True,
    )
    loader = data.test_dataloader()
    for batch in loader:
        name = scene_name(batch)
        if name in seen:
            continue
        seen.add(name)
        inputs = to_device(batch["inputs"], args.device)
        with amp():
            texture, geometry = model.encode_scene_tokens(inputs)
            geometry_observable = codec.project_geometry(geometry)
            positions = model.gaussian_decoder.decoded_token_centers(geometry_observable)
            score_nchw, mean = prepare_score(codec, texture, geometry, positions)

        # Warm every exact operation before it can enter the measured set.
        with amp():
            warm_score_payload, warm_score_hat = score_encode(codec, score_nchw, mean)
            score_decode(codec, warm_score_payload, mean, score_nchw.shape[-1])
            warm_scene = codec.compress(texture, geometry, positions)
            codec.decompress(warm_scene.data)
        _sync(args.device)
        del warm_score_payload, warm_score_hat, warm_scene

        if len(scene_ids) < args.warmup_scenes:
            scene_ids.append(name)
            if len(scene_ids) == target:
                break
            continue

        with amp():
            (score_payload, score_hat), score_encode_ms, score_encode_mem = timed_call(
                lambda: score_encode(codec, score_nchw, mean), args.device
            )
            score_hat_rx, score_decode_ms, score_decode_mem = timed_call(
                lambda: score_decode(codec, score_payload, mean, score_nchw.shape[-1]),
                args.device,
            )
            compressed, codec_encode_ms, codec_encode_mem = timed_call(
                lambda: codec.compress(texture, geometry, positions), args.device
            )
            decoded, codec_decode_ms, codec_decode_mem = timed_call(
                lambda: codec.decompress(compressed.data), args.device
            )
        if not torch.equal(score_hat, score_hat_rx):
            raise RuntimeError(f"score sender/receiver mismatch for scene {name}")
        if decoded[0].shape[:2] != texture.shape[:2] or decoded[1].shape[:2] != geometry.shape[:2]:
            raise RuntimeError(f"feature codec decoded unexpected shape for scene {name}")
        scene = SceneBitstream.unpack(compressed.data)
        parts = scene.bytes_by_stream
        score_parts = score_payload_parts(scene.score, codec.score_context is not None)
        if score_payload != scene.score:
            raise RuntimeError(f"isolated and complete score payload differ for scene {name}")
        row = {
            "scene": name,
            "points": scene.points,
            "score_roundtrip_exact": True,
            "score_encode_ms": score_encode_ms,
            "score_decode_ms": score_decode_ms,
            "feature_codec_encode_ms": codec_encode_ms,
            "feature_codec_decode_ms": codec_decode_ms,
            "score_encode_peak_cuda_bytes": score_encode_mem,
            "score_decode_peak_cuda_bytes": score_decode_mem,
            "feature_codec_encode_peak_cuda_bytes": codec_encode_mem,
            "feature_codec_decode_peak_cuda_bytes": codec_decode_mem,
            "total_bytes": len(compressed.data),
            **score_parts,
            "residual_bytes": parts["residual_y"] + parts["residual_z"],
            "mean_bytes": parts["mean"],
            "outer_and_residual_wrapper_bytes": parts["container"],
        }
        rows.append(row)
        scene_ids.append(name)
        print(
            f"MEASURE scenes={len(rows)}/{args.measure_scenes} scene={name} "
            f"score_ms={score_encode_ms:.3f}/{score_decode_ms:.3f} "
            f"score_bytes={score_parts['score_payload_bytes']}",
            flush=True,
        )
        if len(scene_ids) == target:
            break
    del loader
    if len(rows) != args.measure_scenes:
        raise RuntimeError("test loader exhausted before requested unique scene count")

    report = {
        "schema_version": 1,
        "suite": args.suite,
        "label": args.label,
        "lambda_tag": args.lambda_tag,
        "seed": args.seed,
        "precision": args.precision,
        "warmup_scene_ids": scene_ids[: args.warmup_scenes],
        "measure_scene_ids": scene_ids[args.warmup_scenes :],
        "environment": environment,
        **metadata,
        "footprints": footprints,
        "metrics": summarize_rows(rows),
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        "OVERHEAD_COMPLETE "
        + json.dumps(
            {
                "score_encode_ms_median": report["metrics"]["score_encode_ms"]["median"],
                "score_decode_ms_median": report["metrics"]["score_decode_ms"]["median"],
                "score_payload_bytes_mean": report["metrics"]["score_payload_bytes"]["mean"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

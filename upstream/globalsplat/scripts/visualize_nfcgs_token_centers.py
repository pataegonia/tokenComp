#!/usr/bin/env python3
"""Collect and visualize NFC-GS token position candidates.

For a small number of dataset scenes, this script runs only the GlobalSplat
scene-token encoder and the geometry readout heads.  It saves, per scene:

* the ``M_max`` decoded candidate positions for every token;
* the candidate gate weights;
* the weighted representative used to construct the Morton order;
* the Morton code, permutation, and inverse permutation; and
* overview and per-token PNG visualizations.

No optimization or parameter update is performed.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect each scene token's decoded position candidates, gate weights, "
            "and Morton representative, then render diagnostic plots."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="RE10K root containing test/index.json and test/*.torch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/token_center_visualization"),
    )
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--tokens-per-scene", type=int, default=6)
    parser.add_argument(
        "--token-ids",
        type=str,
        default=None,
        help="Optional comma-separated original token indices, used for every scene.",
    )
    parser.add_argument(
        "--selection",
        choices=("spread-quantiles", "largest-spread", "uniform-index"),
        default="spread-quantiles",
        help=(
            "How to select per-token plots when --token-ids is omitted. "
            "spread-quantiles samples from compact through spatially diffuse tokens."
        ),
    )
    parser.add_argument(
        "--model-config",
        default="globalsplat_nfcgs_rank56",
        help="Hydra model config name.",
    )
    parser.add_argument(
        "--dataset-config",
        default="re10k_eval_all_ctx12",
        help="Hydra dataset config name.",
    )
    parser.add_argument(
        "--experiment-config",
        default="re10k_32k",
        help="Hydra experiment config name.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:1.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help=(
            "Encoder autocast precision. auto uses bfloat16 when supported, "
            "otherwise float16 on CUDA, and float32 on CPU."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if args.num_scenes <= 0:
        parser.error("--num-scenes must be positive")
    if args.tokens_per_scene <= 0:
        parser.error("--tokens-per-scene must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def _find_state_tensor(
    state: Mapping[str, torch.Tensor], suffix: str
) -> torch.Tensor | None:
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"checkpoint has multiple tensors ending in {suffix!r}")
    return matches[0]


def _compose_config(
    args: argparse.Namespace,
    state: Mapping[str, torch.Tensor],
):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = REPO_ROOT / "config"
    overrides = [
        f"model={args.model_config}",
        f"+experiment={args.experiment_config}",
        f"dataset={args.dataset_config}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="main", overrides=overrides)

    basis = _find_state_tensor(state, "feature_codec.shared_basis")
    projection = _find_state_tensor(state, "feature_codec.geometry_projection.weight")
    if basis is None or projection is None:
        raise ValueError(
            "checkpoint must contain an integrated model.feature_codec branch"
        )

    rank, observable_channels = map(int, basis.shape)
    geometry_observable, geometry_channels = map(int, projection.shape)
    texture_channels = observable_channels - geometry_observable
    if texture_channels != geometry_channels:
        raise ValueError(
            "this analysis expects equal appearance/geometry token widths; "
            f"got texture={texture_channels}, geometry={geometry_channels}"
        )

    cfg.model.dim_latents = geometry_channels
    cfg.model.feature_codec.rank = rank
    cfg.model.feature_codec.geometry_observable_channels = geometry_observable

    residual_n = _find_state_tensor(
        state, "feature_codec.residual_codec.g_a.0.weight"
    )
    residual_m = _find_state_tensor(
        state, "feature_codec.residual_codec.g_a.2.weight"
    )
    adapter = _find_state_tensor(
        state,
        "feature_codec.residual_codec.analysis_adapter.in_projection.weight",
    )
    if residual_n is not None:
        cfg.model.feature_codec.residual_N = int(residual_n.shape[0])
    if residual_m is not None:
        cfg.model.feature_codec.residual_M = int(residual_m.shape[0])
    if adapter is not None:
        cfg.model.feature_codec.adapter_hidden = int(adapter.shape[0])

    dataset_root = args.dataset_root.expanduser().resolve()
    cfg.dataset.dataset_roots = [str(dataset_root)]
    cfg.dataset.augment = False
    mvsplat_root = Path(str(cfg.dataset.mvsplat_root)).expanduser()
    if not mvsplat_root.is_absolute():
        mvsplat_root = REPO_ROOT / mvsplat_root
    cfg.dataset.mvsplat_root = str(mvsplat_root.resolve())
    cfg.optimizer.batch_size = 1
    cfg.optimizer.num_workers = args.num_workers
    cfg.seed = args.seed
    OmegaConf.resolve(cfg)
    return cfg


def _model_state_dict(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    expected = set(model.state_dict())
    result: dict[str, torch.Tensor] = {}
    prefixes = ("model.", "module.model.", "_forward_module.model.")

    for raw_key, value in state.items():
        candidates = [raw_key]
        candidates.extend(
            raw_key[len(prefix) :]
            for prefix in prefixes
            if raw_key.startswith(prefix)
        )
        for candidate in candidates:
            if candidate in expected:
                if candidate in result:
                    raise ValueError(f"duplicate checkpoint tensor for {candidate}")
                result[candidate] = value
                break

    missing = sorted(expected.difference(result))
    if missing:
        preview = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise RuntimeError(
            f"checkpoint is missing {len(missing)} GlobalSplat tensors: "
            f"{preview}{suffix}"
        )
    return result


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _autocast_factory(device: torch.device, precision: str):
    if precision == "auto":
        if device.type == "cuda":
            precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            precision = "float32"
    if precision == "float32":
        return nullcontext, torch.float32
    if device.type != "cuda":
        raise ValueError(f"{precision} autocast is only supported on CUDA by this script")
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return lambda: torch.autocast(device_type="cuda", dtype=dtype), dtype


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _scene_name(batch: Mapping[str, Any], batch_index: int) -> str:
    scene_info = batch.get("scene_info", {}) or {}
    value = scene_info.get("scene", f"scene_{batch_index:06d}")
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.flatten()[0].item()
    return str(value)


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return result or "scene"


@torch.inference_mode()
def collect_token_positions(
    model: torch.nn.Module,
    inputs: Mapping[str, Any],
    *,
    autocast_factory,
) -> dict[str, np.ndarray]:
    from globalsplat.compression.morton import morton_order_3d

    if model.feature_codec is None:
        raise RuntimeError("the configured model has no feature_codec")

    with autocast_factory():
        _, geometry = model.encode_scene_tokens(inputs)
        geometry_observable = model.feature_codec.project_geometry(geometry)
        decoder = model.gaussian_decoder
        batch, tokens, _ = geometry_observable.shape
        if batch != 1:
            raise ValueError(f"expected one scene per batch, got B={batch}")

        candidate_offsets = decoder.geo_pos_readout(geometry_observable).view(
            batch, tokens, decoder.M_max, 3
        )
        candidate_points = (
            candidate_offsets + decoder.patch_center_bias.unsqueeze(2)
        )
        gate_logits = decoder.gate_readout(geometry_observable).view(
            batch, tokens, decoder.M_max, 1
        )
        gate_weights = F.softmax(gate_logits / decoder.gate_tau, dim=2)
        representatives = decoder.decoded_token_centers(geometry_observable)

    # Morton quantization should see the same dtype/values used by the model,
    # while diagnostics are accumulated and serialized as float32.
    order = morton_order_3d(
        representatives, model.feature_codec.config.morton_bits
    )
    candidates_f32 = candidate_points.float()
    weights_f32 = gate_weights.float()
    representatives_f32 = representatives.float()
    squared_distance = (
        candidates_f32 - representatives_f32.unsqueeze(2)
    ).square().sum(dim=-1)
    spread = (weights_f32.squeeze(-1) * squared_distance).sum(dim=2).sqrt()
    entropy = -(
        weights_f32.squeeze(-1)
        * weights_f32.squeeze(-1).clamp_min(1e-12).log()
    ).sum(dim=2)
    normalized_entropy = entropy / np.log(decoder.M_max)
    argmax_index = weights_f32.squeeze(-1).argmax(dim=2)
    argmax_point = torch.gather(
        candidates_f32,
        2,
        argmax_index[..., None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)
    argmax_distance = (representatives_f32 - argmax_point).norm(dim=-1)

    def numpy(value: torch.Tensor) -> np.ndarray:
        return value[0].detach().cpu().numpy()

    return {
        "candidate_points": numpy(candidates_f32),
        "gate_weights": numpy(weights_f32.squeeze(-1)),
        "representative_points": numpy(representatives_f32),
        "spread": numpy(spread),
        "gate_entropy": numpy(entropy),
        "normalized_gate_entropy": numpy(normalized_entropy),
        "argmax_candidate": numpy(argmax_index),
        "argmax_distance": numpy(argmax_distance),
        "morton_codes": numpy(order.codes),
        "morton_permutation": numpy(order.permutation),
        "morton_inverse_permutation": numpy(order.inverse_permutation),
    }


def _parse_token_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        token_id = int(item)
        if token_id not in result:
            result.append(token_id)
    if not result:
        raise ValueError("--token-ids did not contain any integer token IDs")
    return result


def select_token_ids(
    data: Mapping[str, np.ndarray],
    count: int,
    selection: str,
    explicit: list[int] | None,
) -> list[int]:
    tokens = int(data["representative_points"].shape[0])
    if explicit is not None:
        invalid = [index for index in explicit if not 0 <= index < tokens]
        if invalid:
            raise ValueError(
                f"token IDs out of range [0,{tokens - 1}]: {invalid}"
            )
        return explicit

    count = min(count, tokens)
    if selection == "largest-spread":
        order = np.argsort(data["spread"])
        return [int(value) for value in order[-count:][::-1]]
    if selection == "uniform-index":
        return [
            int(value)
            for value in np.unique(np.rint(np.linspace(0, tokens - 1, count)))
        ]

    spread_order = np.argsort(data["spread"])
    ranks = np.unique(np.rint(np.linspace(0, tokens - 1, count)).astype(np.int64))
    return [int(spread_order[rank]) for rank in ranks]


def _set_equal_3d(ax, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float((maximum - minimum).max()) * 0.55, 1e-4)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _label_xyz(ax) -> None:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if hasattr(ax, "set_zlabel"):
        ax.set_zlabel("z")


def plot_scene_overview(
    data: Mapping[str, np.ndarray],
    selected: list[int],
    output: Path,
    *,
    scene_name: str,
    dpi: int,
) -> None:
    points = data["representative_points"]
    morton_rank = data["morton_inverse_permutation"]
    color = morton_rank / max(len(morton_rank) - 1, 1)

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    scatter = ax_3d.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=color, cmap="viridis", s=4, alpha=0.65, linewidths=0,
    )
    if selected:
        chosen = points[selected]
        ax_3d.scatter(
            chosen[:, 0], chosen[:, 1], chosen[:, 2],
            s=75, marker="o", facecolors="none", edgecolors="red", linewidths=1.4,
        )
    _set_equal_3d(ax_3d, points)
    _label_xyz(ax_3d)
    ax_3d.set_title("Representative points (color = Morton rank)")

    projections = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    for slot, (first, second, first_name, second_name) in enumerate(projections, start=2):
        ax = fig.add_subplot(2, 2, slot)
        ax.scatter(
            points[:, first], points[:, second],
            c=color, cmap="viridis", s=3, alpha=0.55, linewidths=0,
        )
        if selected:
            chosen = points[selected]
            ax.scatter(
                chosen[:, first], chosen[:, second],
                s=55, facecolors="none", edgecolors="red", linewidths=1.2,
            )
        ax.set_xlabel(first_name)
        ax.set_ylabel(second_name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{first_name.upper()}{second_name.upper()} projection")

    fig.colorbar(scatter, ax=fig.axes, shrink=0.65, label="normalized Morton rank")
    fig.suptitle(
        f"{scene_name} | {len(points)} token representatives | "
        f"selected={selected}",
        fontsize=13,
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_token(
    data: Mapping[str, np.ndarray],
    token_id: int,
    output: Path,
    *,
    scene_name: str,
    dpi: int,
) -> None:
    representatives = data["representative_points"]
    candidates = data["candidate_points"][token_id]
    weights = data["gate_weights"][token_id]
    center = representatives[token_id]
    morton_rank = int(data["morton_inverse_permutation"][token_id])
    morton_code = int(data["morton_codes"][token_id])

    fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)

    scene_ax = fig.add_subplot(1, 3, 1, projection="3d")
    scene_ax.scatter(
        representatives[:, 0], representatives[:, 1], representatives[:, 2],
        s=3, c="0.55", alpha=0.2, linewidths=0,
    )
    scene_ax.scatter(
        center[0], center[1], center[2],
        s=180, c="red", marker="*", edgecolors="black", linewidths=0.5,
    )
    _set_equal_3d(scene_ax, representatives)
    _label_xyz(scene_ax)
    scene_ax.set_title("Representative inside the scene")

    local_ax = fig.add_subplot(1, 3, 2, projection="3d")
    local_ax.scatter(
        candidates[:, 0], candidates[:, 1], candidates[:, 2],
        c=weights, cmap="viridis", vmin=0.0, vmax=max(float(weights.max()), 1e-6),
        s=30.0 + 650.0 * weights, edgecolors="black", linewidths=0.35,
    )
    for candidate_index, point in enumerate(candidates):
        local_ax.plot(
            [point[0], center[0]], [point[1], center[1]], [point[2], center[2]],
            color="0.55", alpha=0.25, linewidth=0.7,
        )
        local_ax.text(point[0], point[1], point[2], str(candidate_index), fontsize=7)
    local_ax.scatter(
        center[0], center[1], center[2],
        s=260, c="red", marker="*", edgecolors="black", linewidths=0.6,
        label="weighted representative",
    )
    _set_equal_3d(local_ax, np.concatenate((candidates, center[None]), axis=0))
    _label_xyz(local_ax)
    local_ax.legend(loc="best", fontsize=8)
    local_ax.set_title("Candidate points and weighted representative")

    weight_ax = fig.add_subplot(1, 3, 3)
    candidate_ids = np.arange(len(weights))
    bars = weight_ax.bar(candidate_ids, weights, color=plt.cm.viridis(weights / max(weights.max(), 1e-12)))
    argmax = int(data["argmax_candidate"][token_id])
    bars[argmax].set_edgecolor("red")
    bars[argmax].set_linewidth(2.0)
    weight_ax.set_xlabel("candidate index")
    weight_ax.set_ylabel("softmax gate weight")
    weight_ax.set_xticks(candidate_ids)
    weight_ax.set_ylim(0.0, max(float(weights.max()) * 1.18, 0.05))
    weight_ax.grid(axis="y", alpha=0.25)
    weight_ax.set_title(
        f"spread={float(data['spread'][token_id]):.5f}\n"
        f"normalized entropy={float(data['normalized_gate_entropy'][token_id]):.3f}\n"
        f"distance to argmax={float(data['argmax_distance'][token_id]):.5f}"
    )

    fig.suptitle(
        f"{scene_name} | original token {token_id} | Morton rank {morton_rank} | "
        f"code {morton_code}",
        fontsize=12,
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def save_scene(
    data: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    scene_name: str,
    batch_index: int,
    selected: list[int],
    dpi: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "token_positions.npz", **data)

    plot_scene_overview(
        data,
        selected,
        output_dir / "scene_overview.png",
        scene_name=scene_name,
        dpi=dpi,
    )
    for token_id in selected:
        plot_token(
            data,
            token_id,
            output_dir / f"token_{token_id:04d}.png",
            scene_name=scene_name,
            dpi=dpi,
        )

    metadata = {
        "scene": scene_name,
        "batch_index": batch_index,
        "tokens": int(data["representative_points"].shape[0]),
        "candidates_per_token": int(data["candidate_points"].shape[1]),
        "selected_token_ids": selected,
        "spread": _statistics(data["spread"]),
        "normalized_gate_entropy": _statistics(data["normalized_gate_entropy"]),
        "argmax_distance": _statistics(data["argmax_distance"]),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return metadata


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from globalsplat.compression import resize_registered_buffers
    from globalsplat.main import build_datamodule, build_model

    checkpoint_path = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if not (dataset_root / "test" / "index.json").is_file():
        raise FileNotFoundError(
            f"dataset root has no test/index.json: {dataset_root}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state_dict must be a mapping")

    device = _resolve_device(args.device)
    cfg = _compose_config(args, state)
    model = build_model(cfg.model)
    model_state = _model_state_dict(model, state)
    resize_registered_buffers(model, model_state)
    model.load_state_dict(model_state, strict=True)

    model = model.to(device).eval()
    model.set_stage(int(cfg.curriculum.final_stage), mix=1.0)
    autocast_factory, actual_dtype = _autocast_factory(device, args.precision)

    datamodule, _ = build_datamodule(cfg)
    loader = datamodule.test_dataloader()
    explicit_ids = _parse_token_ids(args.token_ids)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint={checkpoint_path}")
    print(f"dataset_root={dataset_root}")
    print(f"device={device} precision={actual_dtype}")
    print(f"output_dir={output_root}")

    scenes: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if len(scenes) >= args.num_scenes:
            break
        scene_name = _scene_name(batch, batch_index)
        inputs = _to_device(batch["inputs"], device)
        data = collect_token_positions(
            model,
            inputs,
            autocast_factory=autocast_factory,
        )
        selected = select_token_ids(
            data,
            args.tokens_per_scene,
            args.selection,
            explicit_ids,
        )
        scene_dir = output_root / f"{len(scenes):03d}_{_slug(scene_name)}"
        metadata = save_scene(
            data,
            scene_dir,
            scene_name=scene_name,
            batch_index=batch_index,
            selected=selected,
            dpi=args.dpi,
        )
        scenes.append(metadata)
        print(
            f"[{len(scenes)}/{args.num_scenes}] scene={scene_name} "
            f"tokens={metadata['tokens']} selected={selected} -> {scene_dir}"
        )

    if not scenes:
        raise RuntimeError("the test dataloader produced no scenes")
    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "model_config": args.model_config,
        "dataset_config": args.dataset_config,
        "experiment_config": args.experiment_config,
        "selection": "explicit" if explicit_ids is not None else args.selection,
        "device": str(device),
        "precision": str(actual_dtype),
        "scenes": scenes,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"wrote {len(scenes)} scenes and summary to {output_root}")


if __name__ == "__main__":
    main()

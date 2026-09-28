#!/usr/bin/env python3
"""Render a scene with all NFC-GS token ownership overlaid.

Unlike ``visualize_nfcgs_token_ownership.py`` (which postprocesses only saved
3-D positions), this script reruns one scene through the checkpoint so it has
the target cameras and the complete Gaussian attributes.  Every Gaussian made
by the same scene token receives the same diagnostic color and is rasterized
with its original scale, rotation, and opacity.

For every selected target view it writes the model render, ownership overlays
before/after Morton numbering, and a zoomable SVG containing a number for every
token center that projects inside the image.  A CSV retains all 4096 tokens,
including off-screen ones.
"""

from __future__ import annotations

import argparse
import base64
import csv
from contextlib import nullcontext
import html
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the checkpoint/config/dataset plumbing already verified by the token
# center collector.  Running a file in scripts/ places this directory on
# sys.path, so the sibling module is importable without making scripts a package.
from visualize_nfcgs_token_centers import (  # noqa: E402
    _autocast_factory,
    _compose_config,
    _model_state_dict,
    _resolve_device,
    _scene_name,
    _slug,
    _to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one scene and overlay ownership colors/numbers for all tokens "
            "before and after Morton sorting."
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
        default=Path("outputs/rendered_token_ownership"),
    )
    scene_group = parser.add_mutually_exclusive_group()
    scene_group.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Zero-based test-dataloader scene index (default: 0).",
    )
    scene_group.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Exact scene name to find instead of using --scene-index.",
    )
    parser.add_argument(
        "--view-indices",
        type=str,
        default="all",
        help="Comma-separated target-view indices, or 'all' (default).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Decoder stage; default is curriculum.final_stage (normally 3).",
    )
    parser.add_argument(
        "--mix",
        type=float,
        default=1.0,
        help="Coarse-to-fine stage mix in [0,1].",
    )
    parser.add_argument(
        "--overlay-strength",
        type=float,
        default=0.68,
        help="Ownership color strength over the RGB render in [0,1].",
    )
    parser.add_argument(
        "--svg-font-size",
        type=float,
        default=3.0,
        help="Font size, in image pixels, for every-token SVG labels.",
    )
    neighbor_group = parser.add_mutually_exclusive_group()
    neighbor_group.add_argument(
        "--neighbor-original-token",
        type=int,
        default=None,
        help=(
            "Center the one-token-per-frame sequence on this original token ID. "
            "If neither anchor option is given, use the visible token nearest "
            "the image center."
        ),
    )
    neighbor_group.add_argument(
        "--neighbor-morton-rank",
        type=int,
        default=None,
        help="Center the one-token-per-frame sequence on this Morton rank.",
    )
    parser.add_argument(
        "--neighbor-count",
        type=int,
        default=9,
        help="Number of consecutive Morton-order tokens to render separately; 0 disables.",
    )
    parser.add_argument(
        "--neighbor-view-index",
        type=int,
        default=0,
        help="Target view used for the one-token-per-frame sequence.",
    )
    parser.add_argument(
        "--neighbor-mask-gamma",
        type=float,
        default=0.45,
        help="Display gamma for each token's normalized contribution mask.",
    )
    parser.add_argument(
        "--neighbor-mask-threshold",
        type=float,
        default=1e-3,
        help="Hide raw per-pixel token contributions below this value.",
    )
    parser.add_argument(
        "--model-config", default="globalsplat_nfcgs_rank56"
    )
    parser.add_argument("--dataset-config", default="re10k_eval_all_ctx12")
    parser.add_argument("--experiment-config", default="re10k_32k")
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
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if args.scene_index is not None and args.scene_index < 0:
        parser.error("--scene-index must be non-negative")
    if not 0.0 <= args.mix <= 1.0:
        parser.error("--mix must be in [0,1]")
    if not 0.0 <= args.overlay_strength <= 1.0:
        parser.error("--overlay-strength must be in [0,1]")
    if args.svg_font_size <= 0:
        parser.error("--svg-font-size must be positive")
    if args.neighbor_count < 0:
        parser.error("--neighbor-count must be non-negative")
    if args.neighbor_count > 64:
        parser.error("--neighbor-count must be <= 64")
    if args.neighbor_view_index < 0:
        parser.error("--neighbor-view-index must be non-negative")
    if args.neighbor_mask_gamma <= 0:
        parser.error("--neighbor-mask-gamma must be positive")
    if not 0.0 <= args.neighbor_mask_threshold < 1.0:
        parser.error("--neighbor-mask-threshold must be in [0,1)")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def _parse_view_indices(value: str, target_views: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(target_views))
    try:
        indices = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    except ValueError as error:
        raise ValueError("--view-indices must be 'all' or comma-separated integers") from error
    indices = list(dict.fromkeys(indices))
    if not indices:
        raise ValueError("--view-indices selected no views")
    invalid = [index for index in indices if not 0 <= index < target_views]
    if invalid:
        raise ValueError(
            f"target view indices outside [0,{target_views - 1}]: {invalid}"
        )
    return indices


def _find_scene(loader, scene_index: int | None, scene_name: str | None):
    for batch_index, batch in enumerate(loader):
        current_name = _scene_name(batch, batch_index)
        matches = current_name == scene_name if scene_name is not None else batch_index == scene_index
        if matches:
            return batch_index, current_name, batch
    wanted = f"name={scene_name!r}" if scene_name is not None else f"index={scene_index}"
    raise RuntimeError(f"test dataloader ended before finding scene {wanted}")


def _turbo_table(count: int, device: torch.device) -> torch.Tensor:
    """Continuous sequence color: useful for seeing Morton locality."""

    values = np.linspace(0.0, 1.0, count, dtype=np.float32)
    rgb = plt.get_cmap("turbo")(values)[:, :3].astype(np.float32)
    return torch.from_numpy(rgb).to(device=device)


def _categorical_table(count: int, device: torch.device) -> torch.Tensor:
    """High-contrast deterministic colors for separating neighboring tokens.

    There cannot be 4096 perceptually unique colors, but hashing the sequence
    number before choosing hue/saturation/value prevents adjacent IDs from
    collapsing into the same smooth color band as they do with ``turbo``.
    """

    index = np.arange(count, dtype=np.uint64)
    hashed = (index * np.uint64(2654435761) + np.uint64(2246822519)) & np.uint64(
        0xFFFFFFFF
    )
    hue = (hashed & np.uint64(0xFFFF)).astype(np.float64) / 65536.0
    saturation = 0.72 + 0.26 * (
        ((hashed >> np.uint64(16)) & np.uint64(0xFF)).astype(np.float64) / 255.0
    )
    value = 0.78 + 0.21 * (
        ((hashed >> np.uint64(24)) & np.uint64(0xFF)).astype(np.float64) / 255.0
    )
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv).astype(np.float32)
    return torch.from_numpy(rgb).to(device=device)


@torch.inference_mode()
def _render_token_colors(
    gaussians,
    targets: Mapping[str, Any],
    colors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize arbitrary D-channel attributes with the Gaussian geometry."""

    from gsplat import rasterization
    from globalsplat.model.rendering import _rot_to_quat

    images = targets["images"]
    batch, views, _, height, width = images.shape
    device = gaussians.means.device
    means = gaussians.means.float().contiguous()
    scales = gaussians.scales.float().contiguous()
    quaternions = _rot_to_quat(gaussians.rotations.float().contiguous())
    opacities = gaussians.opacities.float()
    if opacities.ndim == 3 and opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)
    opacities = opacities.contiguous()
    colors = colors.to(device=device, dtype=torch.float32).contiguous()
    intrinsics = torch.as_tensor(targets["intrinsic"], device=device).float().contiguous()
    extrinsics = torch.as_tensor(targets["extrinsic"], device=device).float().contiguous()

    if colors.ndim != 3 or colors.shape[:2] != (batch, means.shape[1]):
        raise ValueError(
            "diagnostic attributes must have shape "
            f"[B,{means.shape[1]},D], got {tuple(colors.shape)}"
        )
    with torch.autocast(device_type=device.type, enabled=False):
        rendered, alpha, _ = rasterization(
            means=means,
            quats=quaternions,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=extrinsics,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=True,
            render_mode="RGB",
            sh_degree=None,
            backgrounds=None,
            camera_model="pinhole",
            eps2d=0.05,
            near_plane=1e-2,
        )
    expected = (batch, views, height, width, colors.shape[-1])
    if rendered.shape != expected:
        raise RuntimeError(f"unexpected token render shape: {tuple(rendered.shape)}")
    return rendered, alpha


def _as_rgb_uint8(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(image * 255.0).astype(np.uint8)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path)


def _blend_overlay(
    scene_btvchw: torch.Tensor,
    diagnostic_btbhwc: torch.Tensor,
    alpha_btbhw1: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    scene = scene_btvchw.permute(0, 1, 3, 4, 2).float().clamp(0, 1)
    alpha = alpha_btbhw1.float().clamp(0, 1)
    unpremultiplied = diagnostic_btbhwc.float() / alpha.clamp_min(1e-6)
    unpremultiplied = unpremultiplied.clamp(0, 1)
    amount = alpha * float(strength)
    return (scene * (1.0 - amount) + unpremultiplied * amount).clamp(0, 1)


def _token_centers_and_opacity(gaussians, token_count: int) -> torch.Tensor:
    batch, gaussian_count, _ = gaussians.means.shape
    if gaussian_count % token_count:
        raise ValueError(
            f"{gaussian_count} Gaussians cannot be grouped into {token_count} tokens"
        )
    groups = gaussian_count // token_count
    means = gaussians.means.float().reshape(batch, token_count, groups, 3)
    opacity = gaussians.opacities.float().reshape(batch, token_count, groups, -1)[..., 0]
    weight = opacity.clamp_min(1e-8)
    return (means * weight[..., None]).sum(dim=2) / weight.sum(dim=2, keepdim=True)


def _project_token_centers(
    centers_sorted: torch.Tensor,
    targets: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel coordinates, camera depth, and in-frame flags [T,N,...]."""

    centers = centers_sorted[0].float()
    intrinsics = torch.as_tensor(targets["intrinsic"], device=centers.device).float()[0]
    extrinsics = torch.as_tensor(targets["extrinsic"], device=centers.device).float()[0]
    _, target_views, _, height, width = targets["images"].shape
    homogeneous = torch.cat(
        [centers, torch.ones_like(centers[:, :1])], dim=-1
    )
    camera = torch.einsum("tij,nj->tni", extrinsics, homogeneous)[..., :3]
    projected = torch.einsum("tij,tnj->tni", intrinsics, camera)
    depth = camera[..., 2]
    uv = projected[..., :2] / projected[..., 2:3].clamp_min(1e-8)
    visible = (
        (depth > 1e-2)
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < width)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < height)
    )
    assert uv.shape[:2] == (target_views, centers.shape[0])
    return (
        uv.detach().cpu().numpy(),
        depth.detach().cpu().numpy(),
        visible.detach().cpu().numpy(),
    )


def _select_neighbor_ranks(
    *,
    args: argparse.Namespace,
    permutation: np.ndarray,
    inverse: np.ndarray,
    uv: np.ndarray,
    visible: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, int, str]:
    token_count = len(permutation)
    view_index = args.neighbor_view_index
    if not 0 <= view_index < uv.shape[0]:
        raise ValueError(
            f"--neighbor-view-index must be in [0,{uv.shape[0] - 1}]"
        )
    if args.neighbor_original_token is not None:
        token_id = args.neighbor_original_token
        if not 0 <= token_id < token_count:
            raise ValueError(
                f"--neighbor-original-token must be in [0,{token_count - 1}]"
            )
        anchor_rank = int(inverse[token_id])
        anchor_description = f"original token t{token_id}"
    elif args.neighbor_morton_rank is not None:
        anchor_rank = args.neighbor_morton_rank
        if not 0 <= anchor_rank < token_count:
            raise ValueError(
                f"--neighbor-morton-rank must be in [0,{token_count - 1}]"
            )
        anchor_description = f"Morton rank r{anchor_rank}"
    else:
        candidates = np.flatnonzero(visible[view_index])
        if len(candidates) == 0:
            raise RuntimeError(f"no token center projects inside target view {view_index}")
        height, width = image_size
        image_center = np.array([width / 2.0, height / 2.0])
        squared_distance = np.square(uv[view_index, candidates] - image_center).sum(axis=1)
        anchor_rank = int(candidates[np.argmin(squared_distance)])
        anchor_description = (
            f"automatic center token t{int(permutation[anchor_rank])} "
            f"(Morton rank r{anchor_rank})"
        )

    count = min(int(args.neighbor_count), token_count)
    start = anchor_rank - count // 2
    start = max(0, min(start, token_count - count))
    return np.arange(start, start + count, dtype=np.int64), anchor_rank, anchor_description


def _single_view_targets(
    targets: Mapping[str, Any], view_index: int
) -> dict[str, Any]:
    return {
        key: targets[key][:, view_index : view_index + 1]
        for key in ("images", "intrinsic", "extrinsic")
    }


def _token_contribution_visuals(
    scene_image: np.ndarray,
    raw_mask: np.ndarray,
    *,
    gamma: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create readable views while retaining the exact raw mask separately."""

    raw = np.asarray(raw_mask, dtype=np.float32).clip(0, 1)
    kept = raw >= float(threshold)
    peak = float(raw.max())
    if peak > 0:
        display = np.power(raw / peak, float(gamma), dtype=np.float32)
        display[~kept] = 0
    else:
        display = np.zeros_like(raw)

    scene = scene_image.astype(np.float32) / 255.0
    amount = display[..., None]
    isolated = scene * amount
    highlight = np.array([1.0, 0.12, 0.02], dtype=np.float32)
    highlighted = scene * 0.16 + amount * (scene * 0.62 + highlight * 0.38)
    highlighted = np.clip(highlighted, 0, 1)
    heat = plt.get_cmap("inferno")(display)[..., :3].astype(np.float32)
    heat[~kept] = 0

    def uint8(value: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)

    return uint8(isolated), uint8(highlighted), uint8(heat), display


def _annotate_gif_frame(image: np.ndarray, label: str) -> Image.Image:
    source = Image.fromarray(image, mode="RGB")
    bar_height = 22
    canvas = Image.new("RGB", (source.width, source.height + bar_height), "#0b1020")
    canvas.paste(source, (0, bar_height))
    ImageDraw.Draw(canvas).text((5, 5), label, fill="white")
    return canvas


@torch.inference_mode()
def _write_neighbor_sequence(
    *,
    output_dir: Path,
    gaussians,
    targets: Mapping[str, Any],
    rendered: torch.Tensor,
    ranks: np.ndarray,
    anchor_rank: int,
    anchor_description: str,
    permutation: np.ndarray,
    groups: int,
    view_index: int,
    frame_id: str,
    gamma: float,
    threshold: float,
    dpi: int,
) -> dict[str, Any]:
    """Render exact visible contribution masks for consecutive Morton tokens."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device = gaussians.means.device
    channels = len(ranks)
    attributes = torch.zeros(
        (1, gaussians.num_gaussians, channels),
        device=device,
        dtype=torch.float32,
    )
    for channel, rank_value in enumerate(ranks):
        rank = int(rank_value)
        attributes[:, rank * groups : (rank + 1) * groups, channel] = 1.0

    # Other tokens remain present with zero attributes.  They still consume
    # transmittance, so each output channel is this token's contribution in the
    # full composited render, rather than a misleading unoccluded solo render.
    contribution, _ = _render_token_colors(
        gaussians,
        _single_view_targets(targets, view_index),
        attributes,
    )
    masks = contribution[0, 0].detach().float().cpu().numpy()
    scene_image = _as_rgb_uint8(rendered[0, view_index])
    entries: list[dict[str, Any]] = []
    gif_frames: list[Image.Image] = []
    contact_images: list[np.ndarray] = []
    contact_titles: list[str] = []

    for sequence_index, rank_value in enumerate(ranks):
        rank = int(rank_value)
        original_id = int(permutation[rank])
        raw_mask = masks[..., sequence_index]
        isolated, overlay, heat, _display = _token_contribution_visuals(
            scene_image,
            raw_mask,
            gamma=gamma,
            threshold=threshold,
        )
        stem = f"{sequence_index:02d}_r{rank:04d}_t{original_id:04d}"
        np.save(output_dir / f"{stem}_raw_contribution.npy", raw_mask.astype(np.float32))
        _save_rgb(output_dir / f"{stem}_area_only.png", isolated)
        _save_rgb(output_dir / f"{stem}_overlay.png", overlay)
        _save_rgb(output_dir / f"{stem}_mask_heatmap.png", heat)
        label = f"step {sequence_index:02d} | Morton r{rank} | original t{original_id}"
        gif_frames.append(_annotate_gif_frame(overlay, label))
        contact_images.append(overlay)
        contact_titles.append(f"{sequence_index:02d}: r{rank} / t{original_id}")
        entries.append(
            {
                "sequence_index": sequence_index,
                "morton_rank": rank,
                "original_token_id": original_id,
                "is_anchor": rank == anchor_rank,
                "raw_contribution_max": float(raw_mask.max()),
                "raw_contribution_sum": float(raw_mask.sum()),
                "pixels_above_threshold": int((raw_mask >= threshold).sum()),
                "stem": stem,
            }
        )

    if gif_frames:
        gif_frames[0].save(
            output_dir / "morton_neighbor_tokens.gif",
            save_all=True,
            append_images=gif_frames[1:],
            duration=900,
            loop=0,
            optimize=False,
        )

    columns = min(3, max(1, len(contact_images)))
    rows = int(np.ceil(len(contact_images) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 4.2 * rows),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for index, (image_value, title) in enumerate(zip(contact_images, contact_titles)):
        axis = axes.flat[index]
        axis.imshow(image_value)
        axis.set_title(title, fontsize=9)
    fig.suptitle(
        f"One token per image · target view {view_index}, frame {frame_id}\n"
        f"consecutive Morton ranks around {anchor_description}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "morton_neighbor_tokens_contact_sheet.png", dpi=dpi)
    plt.close(fig)

    manifest = {
        "target_view_index": view_index,
        "frame_id": frame_id,
        "anchor_rank": anchor_rank,
        "anchor_description": anchor_description,
        "neighbor_definition": "consecutive Morton ranks, centered on anchor",
        "mask_definition": (
            "Exact alpha-compositing contribution of this token's Gaussians while "
            "all other Gaussians remain present as zero-valued occluders."
        ),
        "display": (
            "PNG masks are peak-normalized per token and gamma-adjusted for visibility; "
            "raw_contribution.npy stores the unnormalized contribution."
        ),
        "gamma": gamma,
        "threshold": threshold,
        "tokens": entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _png_data_uri(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _svg_token_groups(
    *,
    uv: np.ndarray,
    depth: np.ndarray,
    visible: np.ndarray,
    permutation: np.ndarray,
    inverse: np.ndarray,
    colors: np.ndarray,
    offset_x: float,
    offset_y: float,
    after_sort: bool,
    font_size: float,
) -> str:
    # Far labels are emitted first so nearer token labels remain on top.
    ranks = np.flatnonzero(visible)
    ranks = ranks[np.argsort(depth[ranks])[::-1]]
    pieces: list[str] = []
    circle_radius = max(0.65, font_size * 0.25)
    stroke_width = max(0.38, font_size * 0.18)
    for rank in ranks:
        original_id = int(permutation[rank])
        x = float(uv[rank, 0]) + offset_x
        y = float(uv[rank, 1]) + offset_y
        value = int(rank) if after_sort else original_id
        label = f"r{int(rank)}" if after_sort else f"t{original_id}"
        rgb = colors[value]
        color_hex = "#{:02x}{:02x}{:02x}".format(*rgb.tolist())
        tooltip = html.escape(
            f"original token t{original_id} | Morton rank r{int(inverse[original_id])} "
            f"| pixel ({uv[rank, 0]:.2f}, {uv[rank, 1]:.2f}) | depth {depth[rank]:.4f}"
        )
        pieces.append(
            f'<g class="token" data-token="{original_id}" data-rank="{int(rank)}">'
            f"<title>{tooltip}</title>"
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{circle_radius:.3f}" '
            f'fill="{color_hex}" stroke="#000" stroke-width="{stroke_width:.3f}"/>'
            f'<text x="{x + circle_radius + 0.4:.3f}" y="{y - 0.2:.3f}">{label}</text>'
            "</g>"
        )
    return "".join(pieces)


def _write_before_after_svg(
    output: Path,
    *,
    before_image: np.ndarray,
    after_image: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    visible: np.ndarray,
    permutation: np.ndarray,
    inverse: np.ndarray,
    color_table: np.ndarray,
    scene_name: str,
    view_index: int,
    frame_id: str,
    font_size: float,
) -> None:
    height, width, _ = before_image.shape
    title_height = 22.0
    gap = 8.0
    total_width = width * 2 + gap
    total_height = height + title_height
    before_uri = _png_data_uri(before_image)
    after_uri = _png_data_uri(after_image)
    before_groups = _svg_token_groups(
        uv=uv,
        depth=depth,
        visible=visible,
        permutation=permutation,
        inverse=inverse,
        colors=color_table,
        offset_x=0.0,
        offset_y=title_height,
        after_sort=False,
        font_size=font_size,
    )
    after_groups = _svg_token_groups(
        uv=uv,
        depth=depth,
        visible=visible,
        permutation=permutation,
        inverse=inverse,
        colors=color_table,
        offset_x=width + gap,
        offset_y=title_height,
        after_sort=True,
        font_size=font_size,
    )
    safe_scene = html.escape(scene_name)
    safe_frame = html.escape(frame_id)
    visible_count = int(visible.sum())
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_width * 3:.0f}" height="{total_height * 3:.0f}" viewBox="0 0 {total_width:.3f} {total_height:.3f}">
<title>Scene {safe_scene}, target view {view_index}, frame {safe_frame}: all projected token labels</title>
<style>
  text {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: {font_size}px; fill: #fff; stroke: #000; stroke-width: {max(.45, font_size * .2):.3f}px; paint-order: stroke fill; dominant-baseline: auto; }}
  .heading {{ font-family: system-ui, sans-serif; font-size: 7px; font-weight: 700; stroke-width: 1px; }}
  .token:hover circle {{ stroke: #fff; stroke-width: 1.2px; r: {max(1.5, font_size * .55):.3f}px; }}
  .token:hover text {{ fill: #ffff55; font-size: {font_size * 2:.3f}px; }}
</style>
<rect width="100%" height="100%" fill="#101522"/>
<text class="heading" x="3" y="10">BEFORE: original token ID (t#)</text>
<text class="heading" x="{width + gap + 3:.3f}" y="10">AFTER: Morton rank (r#)</text>
<text x="3" y="19" style="font-size:3px;stroke-width:.35px">{visible_count}/{len(permutation)} token centers project inside this view; hover a label for both IDs</text>
<image x="0" y="{title_height}" width="{width}" height="{height}" href="{before_uri}"/>
<image x="{width + gap}" y="{title_height}" width="{width}" height="{height}" href="{after_uri}"/>
{before_groups}
{after_groups}
</svg>
'''
    output.write_text(svg, encoding="utf-8")


def _write_projection_csv(
    output: Path,
    *,
    uv: np.ndarray,
    depth: np.ndarray,
    visible: np.ndarray,
    permutation: np.ndarray,
    centers_sorted: np.ndarray,
    selected_views: list[int],
    frame_ids: list[str],
) -> None:
    fields = [
        "target_view_index",
        "frame_id",
        "original_token_id",
        "morton_rank",
        "center_world_x",
        "center_world_y",
        "center_world_z",
        "pixel_u",
        "pixel_v",
        "camera_depth",
        "projects_inside_image",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for view_index in selected_views:
            for rank, original_id_value in enumerate(permutation):
                original_id = int(original_id_value)
                center = centers_sorted[rank]
                writer.writerow(
                    {
                        "target_view_index": view_index,
                        "frame_id": frame_ids[view_index],
                        "original_token_id": original_id,
                        "morton_rank": rank,
                        "center_world_x": f"{float(center[0]):.8g}",
                        "center_world_y": f"{float(center[1]):.8g}",
                        "center_world_z": f"{float(center[2]):.8g}",
                        "pixel_u": f"{float(uv[view_index, rank, 0]):.6g}",
                        "pixel_v": f"{float(uv[view_index, rank, 1]):.6g}",
                        "camera_depth": f"{float(depth[view_index, rank]):.6g}",
                        "projects_inside_image": int(bool(visible[view_index, rank])),
                    }
                )


def _save_contact_sheet(
    output: Path,
    *,
    scene_images: list[np.ndarray],
    before_images: list[np.ndarray],
    after_images: list[np.ndarray],
    view_indices: list[int],
    scene_name: str,
    dpi: int,
) -> None:
    rows = len(view_indices)
    fig, axes = plt.subplots(rows, 3, figsize=(13.5, 4.35 * rows), squeeze=False)
    columns = ("Model render", "BEFORE: original token ID", "AFTER: Morton rank")
    for row, view_index in enumerate(view_indices):
        for column, image in enumerate(
            (scene_images[row], before_images[row], after_images[row])
        ):
            ax = axes[row, column]
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(columns[column])
            if column == 0:
                ax.text(
                    0.01,
                    0.99,
                    f"target view {view_index}",
                    transform=ax.transAxes,
                    va="top",
                    color="white",
                    fontsize=9,
                    bbox={"fc": "black", "ec": "none", "alpha": 0.65, "pad": 2},
                )
    fig.suptitle(
        f"Scene {scene_name}: all-token ownership rendered with original Gaussian geometry\n"
        "Same scene/geometry in both ownership columns; only token color/numbering changes",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def _process_scene(
    *,
    model,
    batch: Mapping[str, Any],
    device: torch.device,
    autocast_factory,
    stage: int,
    mix: float,
    args: argparse.Namespace,
    scene_name: str,
    scene_output: Path,
) -> dict[str, Any]:
    from globalsplat.model.rendering import render_static_batched

    inputs = _to_device(batch["inputs"], device)
    targets = _to_device(batch["targets"], device)
    model.set_stage(stage, mix=mix)
    with autocast_factory():
        gaussians = model(inputs)
    codec_output = model.last_codec_output
    if codec_output is None:
        raise RuntimeError("model forward produced no codec output/Morton order")

    permutation_t = codec_output.order.permutation
    inverse_t = codec_output.order.inverse_permutation
    if permutation_t.shape[0] != 1:
        raise ValueError(f"expected batch size 1, got Morton order {tuple(permutation_t.shape)}")
    token_count = int(permutation_t.shape[1])
    gaussian_count = int(gaussians.num_gaussians)
    if gaussian_count % token_count:
        raise ValueError(
            f"Gaussian count {gaussian_count} is not divisible by token count {token_count}"
        )
    groups = gaussian_count // token_count
    if groups != 1 << stage:
        raise ValueError(f"stage {stage} implies {1 << stage} Gaussians/token, got {groups}")

    # Gaussian blocks are in Morton-rank order because GlobalSplat decodes the
    # codec reconstruction with restore_original_order=False.
    order_color_table_t = _turbo_table(token_count, device)
    categorical_table_t = _categorical_table(token_count, device)
    sorted_rank = torch.arange(token_count, device=device)
    original_id_by_rank = permutation_t[0].long()
    order_before_gaussian_colors = order_color_table_t[
        original_id_by_rank
    ].repeat_interleave(groups, dim=0)[None]
    order_after_gaussian_colors = order_color_table_t[sorted_rank].repeat_interleave(
        groups, dim=0
    )[None]
    categorical_before_gaussian_colors = categorical_table_t[
        original_id_by_rank
    ].repeat_interleave(groups, dim=0)[None]
    categorical_after_gaussian_colors = categorical_table_t[
        sorted_rank
    ].repeat_interleave(groups, dim=0)[None]

    rendered = render_static_batched(gaussians, targets, render_depth=False)["img"]
    batch_size, target_views, _, height, width = targets["images"].shape
    rendered = rendered.reshape(batch_size, target_views, 3, height, width)
    categorical_before_color, ownership_alpha = _render_token_colors(
        gaussians, targets, categorical_before_gaussian_colors
    )
    categorical_after_color, _ = _render_token_colors(
        gaussians, targets, categorical_after_gaussian_colors
    )
    order_before_color, _ = _render_token_colors(
        gaussians, targets, order_before_gaussian_colors
    )
    order_after_color, _ = _render_token_colors(
        gaussians, targets, order_after_gaussian_colors
    )
    categorical_before_overlay = _blend_overlay(
        rendered, categorical_before_color, ownership_alpha, args.overlay_strength
    )
    categorical_after_overlay = _blend_overlay(
        rendered, categorical_after_color, ownership_alpha, args.overlay_strength
    )
    order_before_overlay = _blend_overlay(
        rendered, order_before_color, ownership_alpha, args.overlay_strength
    )
    order_after_overlay = _blend_overlay(
        rendered, order_after_color, ownership_alpha, args.overlay_strength
    )

    centers_t = _token_centers_and_opacity(gaussians, token_count)
    uv, depth, visible = _project_token_centers(centers_t, targets)
    selected_views = _parse_view_indices(args.view_indices, target_views)
    permutation = permutation_t[0].detach().cpu().numpy().astype(np.int64)
    inverse = inverse_t[0].detach().cpu().numpy().astype(np.int64)
    centers = centers_t[0].detach().cpu().numpy()
    categorical_table = np.rint(
        categorical_table_t.detach().cpu().numpy() * 255
    ).astype(np.uint8)

    raw_frame_ids = targets.get("frame_ids")
    if isinstance(raw_frame_ids, torch.Tensor):
        frame_ids = [str(int(value)) for value in raw_frame_ids[0].detach().cpu().tolist()]
    else:
        frame_ids = [str(index) for index in range(target_views)]

    scene_output.mkdir(parents=True, exist_ok=True)
    neighbor_metadata: dict[str, Any] | None = None
    if args.neighbor_count > 0:
        neighbor_ranks, anchor_rank, anchor_description = _select_neighbor_ranks(
            args=args,
            permutation=permutation,
            inverse=inverse,
            uv=uv,
            visible=visible,
            image_size=(height, width),
        )
        neighbor_view = args.neighbor_view_index
        neighbor_dir = scene_output / f"neighbors_view_{neighbor_view:02d}"
        neighbor_metadata = _write_neighbor_sequence(
            output_dir=neighbor_dir,
            gaussians=gaussians,
            targets=targets,
            rendered=rendered,
            ranks=neighbor_ranks,
            anchor_rank=anchor_rank,
            anchor_description=anchor_description,
            permutation=permutation,
            groups=groups,
            view_index=neighbor_view,
            frame_id=frame_ids[neighbor_view],
            gamma=args.neighbor_mask_gamma,
            threshold=args.neighbor_mask_threshold,
            dpi=args.dpi,
        )
        neighbor_metadata["output_directory"] = str(neighbor_dir)

    contact_scene: list[np.ndarray] = []
    contact_before: list[np.ndarray] = []
    contact_after: list[np.ndarray] = []
    contact_order_before: list[np.ndarray] = []
    contact_order_after: list[np.ndarray] = []
    views_metadata: list[dict[str, Any]] = []
    for view_index in selected_views:
        prefix = f"view_{view_index:02d}_frame_{_slug(frame_ids[view_index])}"
        scene_image = _as_rgb_uint8(rendered[0, view_index])
        before_image = np.rint(
            categorical_before_overlay[0, view_index].detach().cpu().numpy() * 255
        ).astype(np.uint8)
        after_image = np.rint(
            categorical_after_overlay[0, view_index].detach().cpu().numpy() * 255
        ).astype(np.uint8)
        order_before_image = np.rint(
            order_before_overlay[0, view_index].detach().cpu().numpy() * 255
        ).astype(np.uint8)
        order_after_image = np.rint(
            order_after_overlay[0, view_index].detach().cpu().numpy() * 255
        ).astype(np.uint8)
        _save_rgb(scene_output / f"{prefix}_render.png", scene_image)
        _save_rgb(scene_output / f"{prefix}_overlay_before.png", before_image)
        _save_rgb(scene_output / f"{prefix}_overlay_after.png", after_image)
        _save_rgb(
            scene_output / f"{prefix}_overlay_order_gradient_before.png",
            order_before_image,
        )
        _save_rgb(
            scene_output / f"{prefix}_overlay_order_gradient_after.png",
            order_after_image,
        )
        _write_before_after_svg(
            scene_output / f"{prefix}_all_tokens_before_after.svg",
            before_image=before_image,
            after_image=after_image,
            uv=uv[view_index],
            depth=depth[view_index],
            visible=visible[view_index],
            permutation=permutation,
            inverse=inverse,
            color_table=categorical_table,
            scene_name=scene_name,
            view_index=view_index,
            frame_id=frame_ids[view_index],
            font_size=args.svg_font_size,
        )
        contact_scene.append(scene_image)
        contact_before.append(before_image)
        contact_after.append(after_image)
        contact_order_before.append(order_before_image)
        contact_order_after.append(order_after_image)
        views_metadata.append(
            {
                "target_view_index": view_index,
                "frame_id": frame_ids[view_index],
                "projected_token_centers_inside_image": int(visible[view_index].sum()),
                "total_tokens": token_count,
                "prefix": prefix,
            }
        )

    _write_projection_csv(
        scene_output / "all_token_projection.csv",
        uv=uv,
        depth=depth,
        visible=visible,
        permutation=permutation,
        centers_sorted=centers,
        selected_views=selected_views,
        frame_ids=frame_ids,
    )
    _save_contact_sheet(
        scene_output / "before_after_contact_sheet.png",
        scene_images=contact_scene,
        before_images=contact_before,
        after_images=contact_after,
        view_indices=selected_views,
        scene_name=scene_name,
        dpi=args.dpi,
    )
    _save_contact_sheet(
        scene_output / "order_gradient_contact_sheet.png",
        scene_images=contact_scene,
        before_images=contact_order_before,
        after_images=contact_order_after,
        view_indices=selected_views,
        scene_name=f"{scene_name} — continuous order-gradient diagnostic",
        dpi=args.dpi,
    )
    return {
        "scene": scene_name,
        "stage": stage,
        "mix": mix,
        "tokens": token_count,
        "gaussians": gaussian_count,
        "gaussians_per_token": groups,
        "image_size": [height, width],
        "overlay_strength": args.overlay_strength,
        "primary_color_mode": "hashed categorical colors for token separation",
        "secondary_color_mode": "continuous turbo gradient for sequence locality",
        "views": views_metadata,
        "neighbor_sequence": neighbor_metadata,
        "interpretation": {
            "before": "Colors and labels identify original pre-Morton token IDs.",
            "after": "Colors and labels identify positions in the Morton-sorted sequence.",
            "geometry": "Both use the same decoded Gaussian geometry; sorting does not move Gaussians.",
            "ownership": "All Gaussians emitted by one token share its diagnostic color and are alpha-composited with their learned geometry/opacity.",
            "colors": "overlay_before/after use hashed high-contrast categorical colors; overlay_order_gradient_* retains the continuous sequence colors used to inspect Morton locality.",
            "labels": "SVG numbers every token center projected inside the image; off-screen/behind-camera tokens remain in all_token_projection.csv.",
        },
    }


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
        raise FileNotFoundError(f"dataset root has no test/index.json: {dataset_root}")

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
    stage = int(cfg.curriculum.final_stage) if args.stage is None else args.stage
    if not 0 <= stage <= int(np.log2(model.gaussian_decoder.M_max)):
        raise ValueError(
            f"stage must be in [0,{int(np.log2(model.gaussian_decoder.M_max))}]"
        )
    autocast_factory, actual_dtype = _autocast_factory(device, args.precision)

    datamodule, _ = build_datamodule(cfg)
    loader = datamodule.test_dataloader()
    batch_index, scene_name, batch = _find_scene(
        loader, args.scene_index, args.scene_name
    )
    output_root = args.output_dir.expanduser().resolve()
    scene_output = output_root / f"{batch_index:03d}_{_slug(scene_name)}"
    print(f"checkpoint={checkpoint_path}")
    print(f"dataset_root={dataset_root}")
    print(f"scene={scene_name} batch_index={batch_index}")
    print(f"device={device} precision={actual_dtype} stage={stage} mix={args.mix:g}")

    metadata = _process_scene(
        model=model,
        batch=batch,
        device=device,
        autocast_factory=autocast_factory,
        stage=stage,
        mix=args.mix,
        args=args,
        scene_name=scene_name,
        scene_output=scene_output,
    )
    metadata.update(
        {
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(dataset_root),
            "batch_index": batch_index,
            "model_config": args.model_config,
            "dataset_config": args.dataset_config,
            "experiment_config": args.experiment_config,
        }
    )
    (scene_output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote rendered all-token ownership visualization to {scene_output}")
    for view in metadata["views"]:
        print(
            f"  view={view['target_view_index']} frame={view['frame_id']} "
            f"labels_in_frame={view['projected_token_centers_inside_image']}/"
            f"{view['total_tokens']}"
        )
    if metadata["neighbor_sequence"] is not None:
        neighbor = metadata["neighbor_sequence"]
        print(
            "  one-token neighbor sequence: "
            f"{neighbor['anchor_description']} -> {neighbor['output_directory']}"
        )


if __name__ == "__main__":
    main()

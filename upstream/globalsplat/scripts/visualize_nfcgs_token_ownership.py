#!/usr/bin/env python3
"""Visualize per-token Gaussian ownership before and after Morton sorting.

This is a zero-model-rerun postprocessor for ``token_positions.npz`` written by
``visualize_nfcgs_token_centers.py``.  It reconstructs the active position
groups of a coarse-to-fine decoder stage from the saved 16 candidates and gate
weights, then writes:

* ``token_mapping.csv`` -- original token ID <-> Morton rank mapping;
* ``before_after.png`` -- static XY/XZ/YZ comparison; and
* ``token_ownership.html`` -- interactive, self-contained 3-D viewer.

The active points identify which Gaussian centers are produced by each token.
They are not Gaussian influence volumes: the collection file does not contain
scale, rotation, opacity, or color.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show the Gaussian centers owned by each token and compare original "
            "token order with Morton-sorted order."
        )
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Scene directory containing token_positions.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <scene-dir>/ownership).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        help="Decoder stage to reconstruct: stage s has 2**s Gaussians/token.",
    )
    parser.add_argument(
        "--mix",
        type=float,
        default=1.0,
        help="Coarse-to-fine transition mix in [0,1] (normally 1.0 for evaluation).",
    )
    parser.add_argument(
        "--label-every",
        type=int,
        default=256,
        help="Label every K-th sequence element in the static and HTML views.",
    )
    parser.add_argument(
        "--token-ids",
        type=str,
        default="",
        help="Comma-separated original token IDs to highlight and always label.",
    )
    parser.add_argument(
        "--fit-percentile",
        type=float,
        default=99.5,
        help="Central percentage used for static axis limits and HTML 'Fit core'.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if args.label_every <= 0:
        parser.error("--label-every must be positive")
    if not 0.0 <= args.mix <= 1.0:
        parser.error("--mix must be in [0,1]")
    if not 90.0 <= args.fit_percentile <= 100.0:
        parser.error("--fit-percentile must be in [90,100]")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def _parse_token_ids(value: str, token_count: int) -> list[int]:
    if not value.strip():
        return []
    try:
        result = sorted({int(piece.strip()) for piece in value.split(",") if piece.strip()})
    except ValueError as error:
        raise ValueError("--token-ids must be comma-separated integers") from error
    invalid = [token_id for token_id in result if not 0 <= token_id < token_count]
    if invalid:
        raise ValueError(f"token IDs outside [0,{token_count - 1}]: {invalid}")
    return result


def _load_scene(scene_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = scene_dir / "token_positions.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing collection file: {npz_path}")

    required = {
        "candidate_points",
        "gate_weights",
        "representative_points",
        "morton_codes",
        "morton_permutation",
        "morton_inverse_permutation",
    }
    with np.load(npz_path) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"{npz_path} is missing arrays: {sorted(missing)}")
        data = {key: archive[key] for key in archive.files}

    metadata_path = scene_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    return data, metadata


def _validate(data: dict[str, np.ndarray], stage: int) -> tuple[int, int]:
    candidates = np.asarray(data["candidate_points"])
    weights = np.asarray(data["gate_weights"])
    representatives = np.asarray(data["representative_points"])
    if candidates.ndim != 3 or candidates.shape[-1] != 3:
        raise ValueError(f"candidate_points must be [N,M,3], got {candidates.shape}")
    token_count, max_candidates, _ = candidates.shape
    if weights.shape != (token_count, max_candidates):
        raise ValueError(
            f"gate_weights must be {(token_count, max_candidates)}, got {weights.shape}"
        )
    if representatives.shape != (token_count, 3):
        raise ValueError(
            f"representative_points must be {(token_count, 3)}, got {representatives.shape}"
        )
    if max_candidates <= 0 or max_candidates & (max_candidates - 1):
        raise ValueError("candidate count must be a positive power of two")
    max_stage = int(np.log2(max_candidates))
    if not 0 <= stage <= max_stage:
        raise ValueError(f"--stage must be in [0,{max_stage}] for M={max_candidates}")

    permutation = np.asarray(data["morton_permutation"], dtype=np.int64)
    inverse = np.asarray(data["morton_inverse_permutation"], dtype=np.int64)
    expected = np.arange(token_count, dtype=np.int64)
    if permutation.shape != (token_count,) or not np.array_equal(np.sort(permutation), expected):
        raise ValueError("morton_permutation is not a permutation of token IDs")
    if inverse.shape != (token_count,) or not np.array_equal(inverse[permutation], expected):
        raise ValueError("morton_inverse_permutation is inconsistent with permutation")
    return token_count, max_stage


def _block_reduce(
    candidates: np.ndarray, weights: np.ndarray, groups: int
) -> np.ndarray:
    token_count, max_candidates, _ = candidates.shape
    if groups == max_candidates:
        return candidates.astype(np.float32, copy=True)
    block_size = max_candidates // groups
    grouped_points = candidates.reshape(token_count, groups, block_size, 3)
    grouped_weights = weights.reshape(token_count, groups, block_size)
    denominator = grouped_weights.sum(axis=2, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError("gate weights must have a positive sum in every reduction block")
    normalized = grouped_weights / denominator
    return (grouped_points * normalized[..., None]).sum(axis=2).astype(np.float32)


def reconstruct_active_points(
    candidates: np.ndarray,
    weights: np.ndarray,
    stage: int,
    mix: float,
) -> np.ndarray:
    """Reproduce the decoder's gated position reduction for one stage."""

    groups = 1 << stage
    current = _block_reduce(candidates, weights, groups)
    if stage == 0:
        return current
    previous = _block_reduce(candidates, weights, groups >> 1)
    previous_upsampled = np.repeat(previous, 2, axis=1)
    return ((1.0 - mix) * previous_upsampled + mix * current).astype(np.float32)


def _central_bounds(points: np.ndarray, percentage: float) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tail = (100.0 - percentage) / 2.0
    low = np.percentile(flat, tail, axis=0)
    high = np.percentile(flat, 100.0 - tail, axis=0)
    span = np.maximum(high - low, 1e-6)
    padding = 0.045 * span
    return low - padding, high + padding


def _fit_record(points: np.ndarray, percentage: float) -> dict[str, list[float] | float]:
    low, high = _central_bounds(points, percentage)
    center = (low + high) / 2.0
    span = float(max(np.max(high - low), 1e-6))
    return {"center": center.round(7).tolist(), "span": span}


def _sequence_labels(
    token_count: int,
    permutation: np.ndarray,
    label_every: int,
    highlighted: list[int],
    after_sort: bool,
) -> list[tuple[int, str]]:
    if after_sort:
        ranks = set(range(0, token_count, label_every))
        inverse = np.empty(token_count, dtype=np.int64)
        inverse[permutation] = np.arange(token_count)
        ranks.update(int(inverse[token_id]) for token_id in highlighted)
        return [
            (int(permutation[rank]), f"r{rank}/t{int(permutation[rank])}")
            for rank in sorted(ranks)
        ]
    token_ids = set(range(0, token_count, label_every))
    token_ids.update(highlighted)
    return [(token_id, f"t{token_id}") for token_id in sorted(token_ids)]


def _write_csv(output: Path, data: dict[str, np.ndarray]) -> None:
    representatives = data["representative_points"]
    token_count = representatives.shape[0]
    inverse = data["morton_inverse_permutation"]
    optional = {
        "spread": data.get("spread", np.full(token_count, np.nan)),
        "gate_entropy": data.get("gate_entropy", np.full(token_count, np.nan)),
        "normalized_gate_entropy": data.get(
            "normalized_gate_entropy", np.full(token_count, np.nan)
        ),
        "argmax_candidate": data.get("argmax_candidate", np.full(token_count, -1)),
        "argmax_distance": data.get("argmax_distance", np.full(token_count, np.nan)),
    }
    fieldnames = [
        "original_token_id",
        "morton_rank",
        "morton_code",
        "representative_x",
        "representative_y",
        "representative_z",
        *optional.keys(),
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for token_id in range(token_count):
            point = representatives[token_id]
            row: dict[str, Any] = {
                "original_token_id": token_id,
                "morton_rank": int(inverse[token_id]),
                "morton_code": int(data["morton_codes"][token_id]),
                "representative_x": f"{float(point[0]):.8g}",
                "representative_y": f"{float(point[1]):.8g}",
                "representative_z": f"{float(point[2]):.8g}",
            }
            for key, values in optional.items():
                value = values[token_id]
                row[key] = int(value) if key == "argmax_candidate" else f"{float(value):.8g}"
            writer.writerow(row)


def _plot_static(
    output: Path,
    *,
    data: dict[str, np.ndarray],
    active: np.ndarray,
    scene_name: str,
    stage: int,
    mix: float,
    label_every: int,
    highlighted: list[int],
    fit_percentile: float,
    dpi: int,
) -> None:
    representatives = data["representative_points"]
    permutation = data["morton_permutation"].astype(np.int64)
    inverse = data["morton_inverse_permutation"].astype(np.int64)
    token_count, groups, _ = active.shape
    flat_active = active.reshape(-1, 3)
    token_for_active = np.repeat(np.arange(token_count), groups)
    low, high = _central_bounds(
        np.concatenate([flat_active, representatives], axis=0), fit_percentile
    )

    projections = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))
    cmap = plt.get_cmap("turbo")
    norm = colors.Normalize(vmin=0, vmax=max(token_count - 1, 1))
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

    for row, after_sort in enumerate((False, True)):
        scalar_by_token = inverse if after_sort else np.arange(token_count)
        path_order = permutation if after_sort else np.arange(token_count)
        labels = _sequence_labels(
            token_count, permutation, label_every, highlighted, after_sort
        )
        for column, (axis_a, axis_b, name_a, name_b) in enumerate(projections):
            ax = axes[row, column]
            path = representatives[path_order]
            ax.plot(
                path[:, axis_a],
                path[:, axis_b],
                color="#263238",
                linewidth=0.28,
                alpha=0.24,
                zorder=1,
            )
            active_scalars = scalar_by_token[token_for_active]
            ax.scatter(
                flat_active[:, axis_a],
                flat_active[:, axis_b],
                c=active_scalars,
                cmap=cmap,
                norm=norm,
                s=2.2,
                alpha=0.42,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
            ax.scatter(
                representatives[:, axis_a],
                representatives[:, axis_b],
                c=scalar_by_token,
                cmap=cmap,
                norm=norm,
                s=5.0,
                alpha=0.72,
                edgecolors="none",
                rasterized=True,
                zorder=3,
            )

            for token_id in highlighted:
                owned = active[token_id]
                center = representatives[token_id]
                for child in owned:
                    ax.plot(
                        [center[axis_a], child[axis_a]],
                        [center[axis_b], child[axis_b]],
                        color="black",
                        linewidth=0.75,
                        alpha=0.72,
                        zorder=4,
                    )
                ax.scatter(
                    owned[:, axis_a],
                    owned[:, axis_b],
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.8,
                    s=23,
                    zorder=5,
                )

            for token_id, label in labels:
                point = representatives[token_id]
                is_highlighted = token_id in highlighted
                ax.annotate(
                    label,
                    (point[axis_a], point[axis_b]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.2 if not is_highlighted else 8.0,
                    fontweight="normal" if not is_highlighted else "bold",
                    color="#111111",
                    bbox=(
                        {"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.62}
                        if is_highlighted
                        else None
                    ),
                    zorder=6,
                )

            ax.set_xlim(low[axis_a], high[axis_a])
            ax.set_ylim(low[axis_b], high[axis_b])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(name_a)
            ax.set_ylabel(name_b)
            ax.grid(alpha=0.16, linewidth=0.5)
            if column == 0:
                row_title = (
                    "AFTER — color/labels/path use Morton rank"
                    if after_sort
                    else "BEFORE — color/labels/path use original token ID"
                )
                ax.set_title(row_title, loc="left", fontsize=12, fontweight="bold")
            else:
                ax.set_title(f"{name_a}{name_b} projection", fontsize=10)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        label = "Morton rank" if after_sort else "Original token ID"
        colorbar = fig.colorbar(sm, ax=axes[row, :], location="right", shrink=0.83, pad=0.012)
        colorbar.set_label(label)

    fig.suptitle(
        f"Scene {scene_name}: token-owned Gaussian centers, before vs. Morton sort\n"
        f"stage={stage} ({groups} Gaussians/token), mix={mix:g}; "
        f"same 3-D points in both rows — only sequence/color/numbering changes\n"
        f"axes show central {fit_percentile:g}% (interactive HTML provides Fit all)",
        fontsize=14,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _json_array(array: np.ndarray, decimals: int | None = None) -> Any:
    value = np.asarray(array)
    if decimals is not None:
        value = np.round(value.astype(np.float64), decimals=decimals)
    return value.tolist()


def _write_interactive_html(
    output: Path,
    *,
    data: dict[str, np.ndarray],
    active: np.ndarray,
    scene_name: str,
    stage: int,
    mix: float,
    label_every: int,
    highlighted: list[int],
    fit_percentile: float,
) -> None:
    representatives = data["representative_points"].astype(np.float32)
    all_points = np.concatenate([active.reshape(-1, 3), representatives], axis=0)
    token_count, groups, _ = active.shape
    payload = {
        "scene": scene_name,
        "stage": stage,
        "mix": mix,
        "tokenCount": token_count,
        "groups": groups,
        "labelEvery": label_every,
        "highlighted": highlighted,
        "representatives": _json_array(representatives, 6),
        "active": _json_array(active, 6),
        "permutation": _json_array(data["morton_permutation"]),
        "inverse": _json_array(data["morton_inverse_permutation"]),
        "codes": [str(int(value)) for value in data["morton_codes"]],
        "spread": _json_array(
            data.get("spread", np.full(token_count, np.nan)), 6
        ),
        "entropy": _json_array(
            data.get("normalized_gate_entropy", np.full(token_count, np.nan)), 6
        ),
        "fitCore": _fit_record(all_points, fit_percentile),
        "fitAll": _fit_record(all_points, 100.0),
        "fitPercentile": fit_percentile,
    }
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    safe_scene = html.escape(scene_name)
    document = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token ownership — __SCENE__</title>
<style>
  :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0b1020; color: #e8edf7; }
  header { padding: 18px 22px 10px; }
  h1 { margin: 0 0 6px; font-size: 21px; }
  .sub { color: #aebbd0; font-size: 13px; line-height: 1.45; }
  .controls { margin: 4px 22px 14px; padding: 11px 13px; background: #141c30;
    border: 1px solid #26324b; border-radius: 10px; display: flex; gap: 12px;
    align-items: center; flex-wrap: wrap; font-size: 13px; }
  button, input { background: #202b44; color: #eef3ff; border: 1px solid #3b4a68;
    border-radius: 6px; padding: 6px 9px; }
  button { cursor: pointer; }
  button:hover { background: #2c3a59; }
  input[type=number] { width: 82px; }
  input[type=text] { width: 105px; }
  label { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 0 22px; }
  .panel { position: relative; background: #090e1a; border: 1px solid #26324b;
    border-radius: 10px; overflow: hidden; min-width: 0; }
  .panel-title { position: absolute; z-index: 2; top: 10px; left: 12px;
    background: rgba(10,15,27,.82); border: 1px solid #33405a; border-radius: 6px;
    padding: 6px 8px; font-size: 12px; pointer-events: none; }
  canvas { width: 100%; height: min(68vh, 720px); min-height: 460px; display: block;
    cursor: grab; touch-action: none; }
  canvas:active { cursor: grabbing; }
  .tooltip { position: absolute; z-index: 3; display: none; pointer-events: none;
    background: rgba(7,11,20,.94); border: 1px solid #536584; border-radius: 6px;
    padding: 6px 8px; font: 12px/1.35 ui-monospace, SFMono-Regular, monospace; }
  .details { margin: 12px 22px 22px; padding: 12px 14px; min-height: 48px;
    background: #141c30; border: 1px solid #26324b; border-radius: 10px;
    font: 12px/1.55 ui-monospace, SFMono-Regular, monospace; color: #cbd7ea; }
  .legend { display: inline-flex; width: 145px; height: 9px; border-radius: 5px;
    background: linear-gradient(90deg,hsl(240,85%,55%),hsl(180,85%,55%),hsl(120,85%,50%),hsl(60,90%,55%),hsl(0,90%,56%)); }
  .hint { margin-left: auto; color: #9cacbf; }
  @media (max-width: 900px) { .panels { grid-template-columns: 1fr; } canvas { height: 58vh; } }
</style>
</head>
<body>
<header>
  <h1>Scene __SCENE__: token-owned Gaussian centers</h1>
  <div class="sub">The two panels contain exactly the same 3-D points. Sorting changes sequence rank, color, labels, and path—not geometry. Each token owns __GROUPS__ active centers at stage __STAGE__. Drag to rotate, wheel to zoom, hover or click a representative center.</div>
</header>
<div class="controls">
  <button id="fitCore">Fit core (__FIT_PERCENTILE__%)</button><button id="fitAll">Fit all</button><button id="resetView">Reset rotation</button>
  <label><input id="showActive" type="checkbox" checked> owned centers</label>
  <label><input id="showReps" type="checkbox" checked> representatives</label>
  <label><input id="showPath" type="checkbox" checked> sequence path</label>
  <label><input id="showLabels" type="checkbox" checked> numbers</label>
  <label>label every <input id="labelEvery" type="number" min="1" step="1" value="__LABEL_EVERY__"></label>
  <label>original token <input id="tokenSearch" type="text" inputmode="numeric" placeholder="0…__MAX_TOKEN__"></label>
  <button id="findToken">Highlight</button>
  <span class="legend"></span><span>sequence start → end</span>
  <span class="hint">Click empty space to clear selection</span>
</div>
<div class="panels">
  <section class="panel"><div class="panel-title"><b>BEFORE</b> · original token ID/order</div><canvas id="before"></canvas><div class="tooltip"></div></section>
  <section class="panel"><div class="panel-title"><b>AFTER</b> · Morton rank/order</div><canvas id="after"></canvas><div class="tooltip"></div></section>
</div>
<div class="details" id="details">No token selected. Hover a representative point or search an original token ID.</div>
<script id="scene-data" type="application/json">__PAYLOAD__</script>
<script>
(() => {
  "use strict";
  const D = JSON.parse(document.getElementById("scene-data").textContent);
  const panels = [
    {canvas: document.getElementById("before"), after: false},
    {canvas: document.getElementById("after"), after: true}
  ];
  for (const p of panels) { p.ctx = p.canvas.getContext("2d"); p.tip = p.canvas.parentElement.querySelector(".tooltip"); }
  const state = { yaw: -0.72, pitch: 0.43, zoom: 1, fit: D.fitCore, selected: D.highlighted.length ? D.highlighted[0] : -1,
    dragging: false, dragX: 0, dragY: 0, moved: false };
  const el = id => document.getElementById(id);
  const checked = id => el(id).checked;
  const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
  const color = value => `hsl(${240 - 240 * value / Math.max(1,D.tokenCount-1)},88%,58%)`;
  function resize(p) {
    const rect = p.canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    const w = Math.round(rect.width * ratio), h = Math.round(rect.height * ratio);
    if (p.canvas.width !== w || p.canvas.height !== h) { p.canvas.width = w; p.canvas.height = h; }
    p.ctx.setTransform(ratio,0,0,ratio,0,0); p.width = rect.width; p.height = rect.height;
  }
  function project(point, p) {
    let x = point[0]-state.fit.center[0], y = point[1]-state.fit.center[1], z = point[2]-state.fit.center[2];
    const cy=Math.cos(state.yaw), sy=Math.sin(state.yaw), cp=Math.cos(state.pitch), sp=Math.sin(state.pitch);
    const xr=cy*x+sy*z, zr=-sy*x+cy*z, yr=cp*y-sp*zr, depth=sp*y+cp*zr;
    const scale = .86 * Math.min(p.width,p.height) / state.fit.span * state.zoom;
    return [p.width/2+xr*scale, p.height/2-yr*scale, depth];
  }
  function drawAxes(p) {
    const ctx=p.ctx, c=state.fit.center, len=state.fit.span*.12, axes=[[len,0,0,"X","#ff6b6b"],[0,len,0,"Y","#65d68a"],[0,0,len,"Z","#68a8ff"]];
    const o=project(c,p); ctx.save(); ctx.font="11px system-ui"; ctx.lineWidth=1.5;
    for (const a of axes) { const q=project([c[0]+a[0],c[1]+a[1],c[2]+a[2]],p); ctx.strokeStyle=a[4]; ctx.fillStyle=a[4]; ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.fillText(a[3],q[0]+3,q[1]-3); }
    ctx.restore();
  }
  function drawPanel(p) {
    resize(p); const ctx=p.ctx; ctx.clearRect(0,0,p.width,p.height); ctx.fillStyle="#090e1a";ctx.fillRect(0,0,p.width,p.height);
    const order = p.after ? D.permutation : null;
    if (checked("showPath")) { ctx.save();ctx.strokeStyle="rgba(205,218,239,.22)";ctx.lineWidth=.55;ctx.beginPath();
      for(let i=0;i<D.tokenCount;i++){const t=p.after?order[i]:i,q=project(D.representatives[t],p);if(i===0)ctx.moveTo(q[0],q[1]);else ctx.lineTo(q[0],q[1]);}ctx.stroke();ctx.restore(); }
    if (checked("showActive")) { ctx.globalAlpha=.48;
      for(let t=0;t<D.tokenCount;t++){const v=p.after?D.inverse[t]:t;ctx.fillStyle=color(v);for(const point of D.active[t]){const q=project(point,p);ctx.fillRect(q[0]-1,q[1]-1,2,2);}}
      ctx.globalAlpha=1; }
    if (checked("showReps")) { ctx.globalAlpha=.88;
      for(let t=0;t<D.tokenCount;t++){const v=p.after?D.inverse[t]:t,q=project(D.representatives[t],p);ctx.fillStyle=color(v);ctx.fillRect(q[0]-1.7,q[1]-1.7,3.4,3.4);}ctx.globalAlpha=1; }
    if (checked("showLabels")) { const stride=Math.max(1,parseInt(el("labelEvery").value,10)||1),tokens=new Set(D.highlighted);for(let pos=0;pos<D.tokenCount;pos+=stride)tokens.add(p.after?D.permutation[pos]:pos);ctx.save();ctx.font="10px ui-monospace,monospace";ctx.textBaseline="bottom";ctx.lineWidth=3;
      for(const t of tokens){const q=project(D.representatives[t],p),label=p.after?`r${D.inverse[t]}/t${t}`:`t${t}`;ctx.strokeStyle="rgba(7,10,17,.85)";ctx.strokeText(label,q[0]+3,q[1]-2);ctx.fillStyle="#f1f5ff";ctx.fillText(label,q[0]+3,q[1]-2);}ctx.restore(); }
    if (D.highlighted.length) { ctx.save();ctx.strokeStyle="rgba(255,255,255,.82)";ctx.lineWidth=1.15;for(const t of D.highlighted){const q=project(D.representatives[t],p);ctx.beginPath();ctx.arc(q[0],q[1],5,0,Math.PI*2);ctx.stroke();}ctx.restore(); }
    if (state.selected>=0) { const t=state.selected, center=project(D.representatives[t],p);ctx.save();ctx.strokeStyle="#ffffff";ctx.fillStyle="#ffffff";ctx.lineWidth=1.25;
      for(const point of D.active[t]){const q=project(point,p);ctx.beginPath();ctx.moveTo(center[0],center[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.beginPath();ctx.arc(q[0],q[1],4,0,Math.PI*2);ctx.stroke();}
      ctx.beginPath();ctx.arc(center[0],center[1],7,0,Math.PI*2);ctx.stroke();const label=p.after?`r${D.inverse[t]} / t${t}`:`t${t}`;ctx.font="bold 12px ui-monospace,monospace";ctx.lineWidth=4;ctx.strokeStyle="#080c16";ctx.strokeText(label,center[0]+9,center[1]-8);ctx.fillText(label,center[0]+9,center[1]-8);ctx.restore(); }
    drawAxes(p);
  }
  function render(){for(const p of panels)drawPanel(p);updateDetails();}
  function nearestToken(event,p){const rect=p.canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let best=-1,bestD=12*12;
    for(let t=0;t<D.tokenCount;t++){const q=project(D.representatives[t],p),dx=q[0]-x,dy=q[1]-y,d=dx*dx+dy*dy;if(d<bestD){bestD=d;best=t;}}return {token:best,x,y};}
  function tokenText(t){if(t<0)return "No token selected. Hover a representative point or search an original token ID.";const p=D.representatives[t];return `original token t${t}  |  Morton rank r${D.inverse[t]}  |  code ${D.codes[t]}  |  representative (${p.map(v=>Number(v).toFixed(5)).join(", ")})  |  spread ${Number(D.spread[t]).toFixed(5)}  |  normalized gate entropy ${Number(D.entropy[t]).toFixed(5)}  |  ${D.groups} owned centers`;}
  function updateDetails(){el("details").textContent=tokenText(state.selected);}
  function selectToken(t){state.selected=t;render();}
  for(const p of panels){
    p.canvas.addEventListener("pointerdown",e=>{state.dragging=true;state.dragX=e.clientX;state.dragY=e.clientY;state.moved=false;p.canvas.setPointerCapture(e.pointerId);});
    p.canvas.addEventListener("pointermove",e=>{if(state.dragging){const dx=e.clientX-state.dragX,dy=e.clientY-state.dragY;if(Math.abs(dx)+Math.abs(dy)>2)state.moved=true;state.yaw+=dx*.008;state.pitch=clamp(state.pitch+dy*.008,-1.5,1.5);state.dragX=e.clientX;state.dragY=e.clientY;render();return;}const hit=nearestToken(e,p);if(hit.token>=0){p.tip.style.display="block";p.tip.style.left=`${hit.x+12}px`;p.tip.style.top=`${hit.y+12}px`;p.tip.textContent=tokenText(hit.token);}else p.tip.style.display="none";});
    p.canvas.addEventListener("pointerup",e=>{if(!state.moved){const hit=nearestToken(e,p);selectToken(hit.token);}state.dragging=false;});
    p.canvas.addEventListener("pointerleave",()=>{p.tip.style.display="none";state.dragging=false;});
    p.canvas.addEventListener("wheel",e=>{e.preventDefault();state.zoom=clamp(state.zoom*Math.exp(-e.deltaY*.001),.15,15);render();},{passive:false});
  }
  el("fitCore").onclick=()=>{state.fit=D.fitCore;state.zoom=1;render();};
  el("fitAll").onclick=()=>{state.fit=D.fitAll;state.zoom=1;render();};
  el("resetView").onclick=()=>{state.yaw=-.72;state.pitch=.43;state.zoom=1;render();};
  for(const id of ["showActive","showReps","showPath","showLabels","labelEvery"])el(id).addEventListener("change",render);
  function findToken(){const t=parseInt(el("tokenSearch").value,10);if(Number.isInteger(t)&&t>=0&&t<D.tokenCount){selectToken(t);el("tokenSearch").setCustomValidity("");}else{el("tokenSearch").setCustomValidity(`Enter 0…${D.tokenCount-1}`);el("tokenSearch").reportValidity();}}
  el("findToken").onclick=findToken;el("tokenSearch").addEventListener("keydown",e=>{if(e.key==="Enter")findToken();});
  window.addEventListener("resize",render);render();
})();
</script>
</body>
</html>
'''
    replacements = {
        "__SCENE__": safe_scene,
        "__GROUPS__": str(groups),
        "__STAGE__": str(stage),
        "__FIT_PERCENTILE__": f"{fit_percentile:g}",
        "__LABEL_EVERY__": str(label_every),
        "__MAX_TOKEN__": str(token_count - 1),
        "__PAYLOAD__": payload_json.replace("</", "<\\/"),
    }
    for marker, value in replacements.items():
        document = document.replace(marker, value)
    output.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    data, metadata = _load_scene(scene_dir)
    token_count, max_stage = _validate(data, args.stage)
    highlighted = _parse_token_ids(args.token_ids, token_count)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else scene_dir / "ownership"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    active = reconstruct_active_points(
        data["candidate_points"].astype(np.float32),
        data["gate_weights"].astype(np.float32),
        args.stage,
        args.mix,
    )
    if args.stage == 0:
        maximum_error = float(
            np.max(np.abs(active[:, 0] - data["representative_points"]))
        )
        if maximum_error > 2e-5:
            raise ValueError(
                "stage-0 reconstruction disagrees with representative_points: "
                f"max_abs_error={maximum_error:.3g}"
            )

    scene_name = str(metadata.get("scene", scene_dir.name))
    _write_csv(output_dir / "token_mapping.csv", data)
    _plot_static(
        output_dir / "before_after.png",
        data=data,
        active=active,
        scene_name=scene_name,
        stage=args.stage,
        mix=args.mix,
        label_every=args.label_every,
        highlighted=highlighted,
        fit_percentile=args.fit_percentile,
        dpi=args.dpi,
    )
    _write_interactive_html(
        output_dir / "token_ownership.html",
        data=data,
        active=active,
        scene_name=scene_name,
        stage=args.stage,
        mix=args.mix,
        label_every=args.label_every,
        highlighted=highlighted,
        fit_percentile=args.fit_percentile,
    )

    run_metadata = {
        "scene": scene_name,
        "source": str(scene_dir / "token_positions.npz"),
        "tokens": token_count,
        "candidate_stage_max": max_stage,
        "visualized_stage": args.stage,
        "gaussians_per_token": int(active.shape[1]),
        "mix": args.mix,
        "highlighted_original_token_ids": highlighted,
        "static_label_every": args.label_every,
        "static_fit_percentile": args.fit_percentile,
        "interpretation": (
            "Active points are reconstructed Gaussian centers owned by each token. "
            "Sorting changes sequence order, rank labels, and colors but not positions. "
            "Gaussian influence volumes require scale/rotation/opacity, which are not in this NPZ."
        ),
    }
    (output_dir / "ownership_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(f"scene: {scene_name}")
    print(f"tokens: {token_count}; stage: {args.stage}; Gaussians/token: {active.shape[1]}")
    print(f"wrote: {output_dir / 'before_after.png'}")
    print(f"wrote: {output_dir / 'token_ownership.html'}")
    print(f"wrote: {output_dir / 'token_mapping.csv'}")


if __name__ == "__main__":
    main()

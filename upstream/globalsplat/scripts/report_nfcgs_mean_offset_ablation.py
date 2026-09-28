#!/usr/bin/env python3
"""Compare matched 32-scene evaluations of learned and disabled b_mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "psnr", "ssim", "lpips", "actual_bytes", "actual_bits_per_gaussian",
    "actual_score_bytes", "actual_residual_y_bytes", "actual_residual_z_bytes",
)
TIMING_METRICS = (
    "sender_mean_seconds_per_scene",
    "receiver_mean_seconds_per_scene",
    "renderer_mean_seconds_per_view",
    "end_to_end_mean_seconds_per_scene",
)


def load_arm(root: Path, variant: str) -> dict:
    output = root / f"{variant}_eval32" / "nfcgs_main"
    metrics = json.loads((output / "scores_all_avg.json").read_text())
    diagnostics = json.loads((output / "score_context_b_summary.json").read_text())
    timing = json.loads((output / "pipeline_timing.json").read_text())
    if metrics["scene_count"] != 32 or diagnostics["scene_count"] != 32:
        raise ValueError(f"{variant} did not evaluate exactly 32 scenes")
    return {
        "metrics": {name: float(metrics[name]) for name in METRICS},
        "mean_offset_b": diagnostics["mean_offset_b"],
        "prediction_gain": diagnostics["prediction_gain"],
        "timing": timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = load_arm(root, "baseline")
    ablated = load_arm(root, "no_mean_offset")
    if any(abs(value) > 1e-8 for value in ablated["mean_offset_b"]["mean"]):
        raise ValueError("no_mean_offset arm still has a nonzero mean offset")
    delta = {
        name: ablated["metrics"][name] - baseline["metrics"][name]
        for name in METRICS
    }
    timing_delta = {
        name: (
            ablated["timing"]["aggregates"][name]
            - baseline["timing"]["aggregates"][name]
        )
        for name in TIMING_METRICS
    }
    report = {
        "scene_count": 32,
        "baseline": baseline,
        "no_mean_offset": ablated,
        "delta_no_mean_offset_minus_baseline": delta,
        "timing_delta_no_mean_offset_minus_baseline": timing_delta,
    }
    path = root / "comparison.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"comparison: {path}")
    for name in METRICS:
        print(
            f"{name}: baseline={baseline['metrics'][name]:.6f} "
            f"no_mean_offset={ablated['metrics'][name]:.6f} "
            f"delta={delta[name]:+.6f}"
        )
    for name in TIMING_METRICS:
        base = baseline["timing"]["aggregates"][name]
        no_offset = ablated["timing"]["aggregates"][name]
        print(
            f"{name}: baseline={base:.6f} "
            f"no_mean_offset={no_offset:.6f} "
            f"delta={timing_delta[name]:+.6f}"
        )


if __name__ == "__main__":
    main()

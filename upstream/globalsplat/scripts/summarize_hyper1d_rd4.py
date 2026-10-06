#!/usr/bin/env python3
"""Collect the four-point Hyper1D full-test results into one RD CSV."""

import argparse
import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
VARIANTS = (
    ("single_off", 1, 0, "off"),
    ("single_on", 1, 0, "on"),
    ("dual_off", 2, 0, "off"),
    ("dual_on", 2, 0, "on"),
    ("lowrank_off", 2, 56, "off"),
    ("lowrank_on", 2, 56, "on"),
    ("plain4", 1, 0, "off"),
)
FIELDS = (
    "variant", "paths", "base_rank", "morton", "lambda", "scene_count",
    "psnr", "ssim", "lpips", "actual_bytes", "actual_kB", "actual_bpga",
    "summary_path",
)


def collect(repo: Path, eval_job_id: int, lambda4: str, include_plain4: bool):
    lambdas = (
        ("0.0256", "0.0128", "0.0064", "0.0032")
        if lambda4 == "0.0032" else
        ("0.0512", "0.0256", "0.0128", "0.0064")
    )
    rows = []
    for variant_index, (variant, paths, rank, morton) in enumerate(
        VARIANTS if include_plain4 else VARIANTS[:-1]
    ):
        for lambda_index, rate_lambda in enumerate(lambdas):
            task_id = 4 * variant_index + lambda_index
            tag = rate_lambda.replace(".", "p")
            summary = (
                repo / "outputs/hyper1d_rd4_50k/full_test"
                / f"job_{eval_job_id}_{task_id}" / variant / f"lambda{tag}"
                / "evaluation/hyper1d_12h/scores_all_avg.json"
            )
            if not summary.is_file():
                raise FileNotFoundError(f"missing RD point: {summary}")
            data = json.loads(summary.read_text(encoding="utf-8"))
            row = {
                "variant": variant, "paths": paths, "base_rank": rank,
                "morton": morton, "lambda": rate_lambda,
                "scene_count": int(data["scene_count"]),
                "psnr": float(data["psnr"]), "ssim": float(data["ssim"]),
                "lpips": float(data["lpips"]),
                "actual_bytes": float(data["actual_bytes"]),
                "actual_kB": float(data["actual_bytes"]) / 1000.0,
                "actual_bpga": float(data["actual_bpga"]),
                "summary_path": str(summary),
            }
            rows.append(row)
    scene_counts = {row["scene_count"] for row in rows}
    if len(scene_counts) != 1:
        raise ValueError(f"RD points used different scene counts: {sorted(scene_counts)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-job-id", type=int, required=True)
    parser.add_argument("--lambda4", choices=("0.0032", "0.0512"), default="0.0032")
    parser.add_argument("--include-plain4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = collect(args.repo, args.eval_job_id, args.lambda4, args.include_plain4)
    output = args.output or (
        args.repo / "outputs/hyper1d_rd4_50k/full_test" / f"rd4_job_{args.eval_job_id}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} RD points ({rows[0]['scene_count']} scenes each): {output}")


if __name__ == "__main__":
    main()

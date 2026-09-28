#!/usr/bin/env python3
"""Paired scene-bootstrap confidence intervals for existing NFCGS evals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = (
    "linear_factorized",
    "nonlinear_factorized",
    "linear_full",
    "nonlinear_split",
)
LAMBDA_ORDER = ("0p0064", "0p0256")
METRICS = ("actual_bytes", "psnr", "ssim", "lpips")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_results_dir(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Manifest entries are repository-relative and stable on workstation/server.
    return REPO_ROOT / path


def record_key(record: dict) -> str:
    scene = str(record["scene"])
    return scene


def load_entry(entry: dict, manifest_path: Path) -> tuple[dict[str, dict], dict]:
    results_dir = resolve_results_dir(entry["results_dir"], manifest_path)
    rate_path = results_dir / "actual_rate_per_scene.json"
    average_path = results_dir / "scores_all_avg.json"
    if not rate_path.is_file():
        raise FileNotFoundError(f"missing existing per-scene result: {rate_path}")
    if not average_path.is_file():
        raise FileNotFoundError(f"missing existing average result: {average_path}")
    raw = json.loads(rate_path.read_text(encoding="utf-8"))
    saved_average = json.loads(average_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"expected a nonempty list in {rate_path}")
    records: dict[str, dict] = {}
    for record in raw:
        missing = set(("scene", "context_frame_ids", "target_frame_ids", *METRICS)) - set(record)
        if missing:
            raise ValueError(f"record in {rate_path} is missing {sorted(missing)}")
        key = record_key(record)
        if key in records:
            raise ValueError(f"duplicate scene {key!r} in {rate_path}")
        for metric in METRICS:
            value = float(record[metric])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {metric} for scene {key!r} in {rate_path}")
        records[key] = record
    for metric in METRICS:
        observed = float(np.mean([float(record[metric]) for record in records.values()]))
        expected = float(saved_average[metric])
        if not np.isclose(observed, expected, rtol=1e-10, atol=1e-10):
            raise ValueError(
                f"{metric} mean mismatch in {results_dir}: records={observed}, saved={expected}"
            )
    metadata = {
        **entry,
        "results_dir": str(results_dir.resolve()),
        "actual_rate_per_scene": str(rate_path.resolve()),
        "actual_rate_per_scene_sha256": file_sha256(rate_path),
        "scores_all_avg": str(average_path.resolve()),
        "scores_all_avg_sha256": file_sha256(average_path),
        "scene_count": len(records),
    }
    return records, metadata


def load_paired_data(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema_version")
    entries = manifest.get("entries", [])
    expected = set(itertools.product(MODEL_ORDER, LAMBDA_ORDER))
    actual = {(entry["model"], entry["lambda_tag"]) for entry in entries}
    if len(entries) != len(actual):
        raise ValueError("manifest has duplicate model/lambda entries")
    if actual != expected:
        raise ValueError(f"manifest grid mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")

    loaded: dict[tuple[str, str], dict[str, dict]] = {}
    metadata = []
    display_names = {}
    for entry in entries:
        key = (entry["model"], entry["lambda_tag"])
        records, details = load_entry(entry, manifest_path)
        loaded[key] = records
        metadata.append(details)
        previous = display_names.setdefault(entry["model"], entry["display_name"])
        if previous != entry["display_name"]:
            raise ValueError(f"display name changes across lambdas for {entry['model']}")

    reference_key = (MODEL_ORDER[0], LAMBDA_ORDER[0])
    scene_ids = sorted(loaded[reference_key])
    reference_records = loaded[reference_key]
    for key, records in loaded.items():
        if sorted(records) != scene_ids:
            missing = sorted(set(scene_ids) - set(records))[:5]
            extra = sorted(set(records) - set(scene_ids))[:5]
            raise ValueError(f"scene set mismatch for {key}: missing={missing}, extra={extra}")
        for scene in scene_ids:
            reference = reference_records[scene]
            candidate = records[scene]
            if candidate["context_frame_ids"] != reference["context_frame_ids"]:
                raise ValueError(f"context frame mismatch for {key}, scene={scene}")
            if candidate["target_frame_ids"] != reference["target_frame_ids"]:
                raise ValueError(f"target frame mismatch for {key}, scene={scene}")

    values = np.empty(
        (len(scene_ids), len(MODEL_ORDER), len(LAMBDA_ORDER), len(METRICS)),
        dtype=np.float64,
    )
    for model_index, model in enumerate(MODEL_ORDER):
        for lambda_index, lambda_tag in enumerate(LAMBDA_ORDER):
            records = loaded[(model, lambda_tag)]
            for scene_index, scene in enumerate(scene_ids):
                values[scene_index, model_index, lambda_index] = [
                    float(records[scene][metric]) for metric in METRICS
                ]
    frames = {
        scene: {
            "context_frame_ids": reference_records[scene]["context_frame_ids"],
            "target_frame_ids": reference_records[scene]["target_frame_ids"],
        }
        for scene in scene_ids
    }
    return values, scene_ids, frames, display_names, metadata


def bootstrap_means(
    values: np.ndarray, *, draws: int, seed: int, batch_size: int
) -> np.ndarray:
    scenes = values.shape[0]
    result = np.empty((draws, *values.shape[1:]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, draws, batch_size):
        end = min(draws, start + batch_size)
        indices = rng.integers(0, scenes, size=(end - start, scenes))
        result[start:end] = values[indices].mean(axis=1)
    return result


def interval(samples: np.ndarray, confidence: float) -> list[float]:
    alpha = 1.0 - confidence
    return [float(value) for value in np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])]


def estimate(value: float, samples: np.ndarray, confidence: float) -> dict:
    return {
        "estimate": float(value),
        "confidence_interval": interval(samples, confidence),
        "bootstrap_standard_error": float(samples.std(ddof=1)),
    }


def two_point_bd_rate(rate_reference, psnr_reference, rate_candidate, psnr_candidate):
    """Two-point log-linear BD-rate; candidate vs reference, negative is better."""

    rate_reference = np.asarray(rate_reference, dtype=np.float64)
    psnr_reference = np.asarray(psnr_reference, dtype=np.float64)
    rate_candidate = np.asarray(rate_candidate, dtype=np.float64)
    psnr_candidate = np.asarray(psnr_candidate, dtype=np.float64)
    if rate_reference.shape[-1] != 2 or rate_candidate.shape[-1] != 2:
        raise ValueError("two-point BD-rate requires exactly two rate points")
    ref_low = np.minimum(psnr_reference[..., 0], psnr_reference[..., 1])
    ref_high = np.maximum(psnr_reference[..., 0], psnr_reference[..., 1])
    can_low = np.minimum(psnr_candidate[..., 0], psnr_candidate[..., 1])
    can_high = np.maximum(psnr_candidate[..., 0], psnr_candidate[..., 1])
    lower = np.maximum(ref_low, can_low)
    upper = np.minimum(ref_high, can_high)
    valid = (
        (upper > lower)
        & (rate_reference[..., 0] > 0)
        & (rate_reference[..., 1] > 0)
        & (rate_candidate[..., 0] > 0)
        & (rate_candidate[..., 1] > 0)
        & (psnr_reference[..., 0] != psnr_reference[..., 1])
        & (psnr_candidate[..., 0] != psnr_candidate[..., 1])
    )
    ref_slope = np.divide(
        np.log(rate_reference[..., 1]) - np.log(rate_reference[..., 0]),
        psnr_reference[..., 1] - psnr_reference[..., 0],
    )
    can_slope = np.divide(
        np.log(rate_candidate[..., 1]) - np.log(rate_candidate[..., 0]),
        psnr_candidate[..., 1] - psnr_candidate[..., 0],
    )
    ref_intercept = np.log(rate_reference[..., 0]) - ref_slope * psnr_reference[..., 0]
    can_intercept = np.log(rate_candidate[..., 0]) - can_slope * psnr_candidate[..., 0]
    average_log_difference = (
        0.5 * (can_slope - ref_slope) * (upper**2 - lower**2)
        + (can_intercept - ref_intercept) * (upper - lower)
    ) / (upper - lower)
    result = 100.0 * np.expm1(average_log_difference)
    return np.where(valid, result, np.nan)


def build_report(
    values: np.ndarray,
    boot: np.ndarray,
    *,
    confidence: float,
    display_names: dict[str, str],
) -> dict:
    point = values.mean(axis=0)
    metric_index = {name: index for index, name in enumerate(METRICS)}
    marginals = []
    for model_index, model in enumerate(MODEL_ORDER):
        for lambda_index, lambda_tag in enumerate(LAMBDA_ORDER):
            metrics = {}
            for name, index in metric_index.items():
                metrics[name] = estimate(
                    point[model_index, lambda_index, index],
                    boot[:, model_index, lambda_index, index],
                    confidence,
                )
            marginals.append(
                {
                    "model": model,
                    "display_name": display_names[model],
                    "lambda_tag": lambda_tag,
                    "metrics": metrics,
                }
            )

    pairwise = []
    for reference_index, candidate_index in itertools.combinations(range(len(MODEL_ORDER)), 2):
        reference = MODEL_ORDER[reference_index]
        candidate = MODEL_ORDER[candidate_index]
        for lambda_index, lambda_tag in enumerate(LAMBDA_ORDER):
            ref_point = point[reference_index, lambda_index]
            can_point = point[candidate_index, lambda_index]
            ref_boot = boot[:, reference_index, lambda_index]
            can_boot = boot[:, candidate_index, lambda_index]
            rate_point = 100.0 * (1.0 - can_point[metric_index["actual_bytes"]] / ref_point[metric_index["actual_bytes"]])
            rate_boot = 100.0 * (
                1.0
                - can_boot[:, metric_index["actual_bytes"]]
                / ref_boot[:, metric_index["actual_bytes"]]
            )
            psnr_point = can_point[metric_index["psnr"]] - ref_point[metric_index["psnr"]]
            psnr_boot = can_boot[:, metric_index["psnr"]] - ref_boot[:, metric_index["psnr"]]
            ssim_point = can_point[metric_index["ssim"]] - ref_point[metric_index["ssim"]]
            ssim_boot = can_boot[:, metric_index["ssim"]] - ref_boot[:, metric_index["ssim"]]
            # Positive is consistently defined as improvement.
            lpips_point = ref_point[metric_index["lpips"]] - can_point[metric_index["lpips"]]
            lpips_boot = ref_boot[:, metric_index["lpips"]] - can_boot[:, metric_index["lpips"]]
            scene_ref = values[:, reference_index, lambda_index]
            scene_can = values[:, candidate_index, lambda_index]
            rate_win = scene_can[:, metric_index["actual_bytes"]] < scene_ref[:, metric_index["actual_bytes"]]
            psnr_win = scene_can[:, metric_index["psnr"]] > scene_ref[:, metric_index["psnr"]]
            pairwise.append(
                {
                    "reference": reference,
                    "reference_display_name": display_names[reference],
                    "candidate": candidate,
                    "candidate_display_name": display_names[candidate],
                    "lambda_tag": lambda_tag,
                    "rate_saving_pct": estimate(rate_point, rate_boot, confidence),
                    "psnr_delta_db": estimate(psnr_point, psnr_boot, confidence),
                    "ssim_delta": estimate(ssim_point, ssim_boot, confidence),
                    "lpips_improvement": estimate(lpips_point, lpips_boot, confidence),
                    "scene_win_fraction": {
                        "lower_rate": float(rate_win.mean()),
                        "higher_psnr": float(psnr_win.mean()),
                        "lower_rate_and_higher_psnr": float((rate_win & psnr_win).mean()),
                    },
                }
            )

    bd_rates = []
    for reference_index, candidate_index in itertools.combinations(range(len(MODEL_ORDER)), 2):
        reference = MODEL_ORDER[reference_index]
        candidate = MODEL_ORDER[candidate_index]
        point_value = two_point_bd_rate(
            point[reference_index, :, metric_index["actual_bytes"]],
            point[reference_index, :, metric_index["psnr"]],
            point[candidate_index, :, metric_index["actual_bytes"]],
            point[candidate_index, :, metric_index["psnr"]],
        )
        samples = two_point_bd_rate(
            boot[:, reference_index, :, metric_index["actual_bytes"]],
            boot[:, reference_index, :, metric_index["psnr"]],
            boot[:, candidate_index, :, metric_index["actual_bytes"]],
            boot[:, candidate_index, :, metric_index["psnr"]],
        )
        valid_samples = samples[np.isfinite(samples)]
        if not np.isfinite(point_value) or len(valid_samples) < 0.99 * len(samples):
            raise RuntimeError(
                f"insufficient overlapping PSNR range for two-point BD-rate: {reference} vs {candidate}"
            )
        value = estimate(float(point_value), valid_samples, confidence)
        value["valid_bootstrap_draws"] = len(valid_samples)
        bd_rates.append(
            {
                "reference": reference,
                "reference_display_name": display_names[reference],
                "candidate": candidate,
                "candidate_display_name": display_names[candidate],
                "bd_rate_pct": value,
            }
        )
    return {"marginal": marginals, "pairwise": pairwise, "two_point_bd_rate": bd_rates}


def write_scene_csv(path: Path, values, scene_ids, frames):
    fieldnames = ["scene", "context_frame_ids", "target_frame_ids"]
    for model in MODEL_ORDER:
        for lambda_tag in LAMBDA_ORDER:
            fieldnames.extend(f"{model}__{lambda_tag}__{metric}" for metric in METRICS)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scene_index, scene in enumerate(scene_ids):
            row = {
                "scene": scene,
                "context_frame_ids": json.dumps(frames[scene]["context_frame_ids"], separators=(",", ":")),
                "target_frame_ids": json.dumps(frames[scene]["target_frame_ids"], separators=(",", ":")),
            }
            for model_index, model in enumerate(MODEL_ORDER):
                for lambda_index, lambda_tag in enumerate(LAMBDA_ORDER):
                    for metric_index, metric in enumerate(METRICS):
                        row[f"{model}__{lambda_tag}__{metric}"] = values[
                            scene_index, model_index, lambda_index, metric_index
                        ]
            writer.writerow(row)


def ci_text(value: dict, digits: int = 3, suffix: str = "") -> str:
    low, high = value["confidence_interval"]
    return f"{value['estimate']:+.{digits}f}{suffix} [{low:+.{digits}f}, {high:+.{digits}f}]"


def write_markdown(path: Path, report: dict):
    analysis = report["analysis"]
    lines = [
        "# NFCGS paired scene confidence intervals",
        "",
        f"Scenes: {report['scene_count']}; paired bootstrap draws: {report['bootstrap']['draws']:,}; "
        f"confidence level: {100*report['bootstrap']['confidence']:.1f}%.",
        "",
        "Every comparison resamples complete scenes and keeps all four models and both lambdas paired. "
        "Positive rate saving, PSNR delta, SSIM delta, and LPIPS improvement all favor the candidate.",
        "",
        "## Mean across scenes",
        "",
        "These intervals describe scene-to-scene sampling variation. KiB uses 1024 bytes.",
        "",
        "| Model | lambda | Mean KiB [CI] | PSNR dB [CI] | SSIM [CI] | LPIPS [CI] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["marginal"]:
        metrics = row["metrics"]
        size = metrics["actual_bytes"]
        size_ci = [value / 1024.0 for value in size["confidence_interval"]]
        lines.append(
            f"| {row['display_name']} | {row['lambda_tag'].replace('p', '.')} | "
            f"{size['estimate']/1024:.3f} [{size_ci[0]:.3f}, {size_ci[1]:.3f}] | "
            f"{metrics['psnr']['estimate']:.4f} [{metrics['psnr']['confidence_interval'][0]:.4f}, {metrics['psnr']['confidence_interval'][1]:.4f}] | "
            f"{metrics['ssim']['estimate']:.5f} [{metrics['ssim']['confidence_interval'][0]:.5f}, {metrics['ssim']['confidence_interval'][1]:.5f}] | "
            f"{metrics['lpips']['estimate']:.5f} [{metrics['lpips']['confidence_interval'][0]:.5f}, {metrics['lpips']['confidence_interval'][1]:.5f}] |"
        )
    for lambda_tag in LAMBDA_ORDER:
        lines += [
            "",
            f"## Paired differences: lambda {lambda_tag.replace('p', '.')}",
            "",
            "| Candidate vs reference | Rate saving % [CI] | PSNR delta dB [CI] | SSIM delta [CI] | LPIPS improvement [CI] | Rate/PSNR/joint scene wins |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in analysis["pairwise"]:
            if row["lambda_tag"] != lambda_tag:
                continue
            wins = row["scene_win_fraction"]
            lines.append(
                f"| {row['candidate_display_name']} vs {row['reference_display_name']} | "
                f"{ci_text(row['rate_saving_pct'], 3, '%')} | "
                f"{ci_text(row['psnr_delta_db'], 4)} | "
                f"{ci_text(row['ssim_delta'], 5)} | "
                f"{ci_text(row['lpips_improvement'], 5)} | "
                f"{100*wins['lower_rate']:.1f}% / {100*wins['higher_psnr']:.1f}% / "
                f"{100*wins['lower_rate_and_higher_psnr']:.1f}% |"
            )
    lines += [
        "",
        "## Approximate two-point BD-rate",
        "",
        "Only two lambda points are available, so this is a log-rate/PSNR linear interpolation screen, "
        "not a publication-grade four-point BD-rate. Negative values favor the candidate.",
        "",
        "| Candidate vs reference | Two-point BD-rate % [CI] |",
        "| --- | ---: |",
    ]
    for row in analysis["two_point_bd_rate"]:
        lines.append(
            f"| {row['candidate_display_name']} vs {row['reference_display_name']} | "
            f"{ci_text(row['bd_rate_pct'], 3, '%')} |"
        )
    lines += [
        "",
        "## Statistical scope",
        "",
        "- The resampling unit is a scene; the eight target frames already averaged inside each scene are not treated as independent samples.",
        "- These are paired percentile-bootstrap intervals for test-scene sampling uncertainty. They do not include training-seed, checkpoint-selection, dataset-shift, or hardware uncertainty.",
        "- Pairwise intervals are unadjusted for multiple comparisons. Use the JSON estimates rather than rounded Markdown values for downstream analysis.",
        "- Exact scene sets, context frames, and target frames were required to match across all eight eval result files before any interval was computed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "scripts/nfcgs_scene_ci_manifest_2026-09-15.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--bootstrap-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.bootstrap_draws < 1000:
        parser.error("bootstrap-draws must be at least 1000")
    if not 0.0 < args.confidence < 1.0:
        parser.error("confidence must be strictly between 0 and 1")
    if args.bootstrap_batch_size < 1:
        parser.error("bootstrap-batch-size must be positive")
    return args


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("use a fresh output directory; existing CI results are preserved")
    values, scene_ids, frames, display_names, sources = load_paired_data(manifest_path)
    print(
        f"SCENE_CI_START scenes={len(scene_ids)} draws={args.bootstrap_draws} "
        f"confidence={args.confidence}",
        flush=True,
    )
    boot = bootstrap_means(
        values,
        draws=args.bootstrap_draws,
        seed=args.seed,
        batch_size=args.bootstrap_batch_size,
    )
    report = {
        "schema_version": 1,
        "scene_count": len(scene_ids),
        "model_order": list(MODEL_ORDER),
        "lambda_order": list(LAMBDA_ORDER),
        "metrics": list(METRICS),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "sources": sources,
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap over scenes",
            "draws": args.bootstrap_draws,
            "confidence": args.confidence,
            "seed": args.seed,
            "batch_size": args.bootstrap_batch_size,
        },
        "analysis": build_report(
            values,
            boot,
            confidence=args.confidence,
            display_names=display_names,
        ),
    }
    write_scene_csv(
        args.output_dir / "paired_scene_metrics.csv", values, scene_ids, frames
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_markdown(args.output_dir / "REPORT.md", report)
    print(f"SCENE_CI_COMPLETE report={args.output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()

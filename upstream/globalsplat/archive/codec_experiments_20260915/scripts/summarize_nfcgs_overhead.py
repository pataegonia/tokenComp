#!/usr/bin/env python3
"""Aggregate controlled NFCGS context/probability overhead profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINE_LABEL = {"context": "factorized", "probability": "shared"}
LABEL_ORDER = {
    "context": ("factorized", "mean", "mean_channel", "mean_spatial", "full"),
    "probability": ("shared", "split", "gaussian", "conditional_scale"),
}


def _median(report: dict, name: str) -> float:
    return float(report["metrics"][name]["median"])


def _mean(report: dict, name: str) -> float:
    return float(report["metrics"][name]["mean"])


def _pct_delta(value: float, baseline: float) -> float | None:
    return 100.0 * (value / baseline - 1.0) if baseline else None


def comparison_row(report: dict, baseline: dict) -> dict:
    metrics = report["metrics"]
    score_bytes = _mean(report, "score_payload_bytes")
    baseline_score_bytes = _mean(baseline, "score_payload_bytes")
    score_path_bytes = report["footprints"]["score_path"]["tensor_bytes"]
    baseline_score_path_bytes = baseline["footprints"]["score_path"]["tensor_bytes"]
    added_model_bytes = score_path_bytes - baseline_score_path_bytes
    saving_per_scene = baseline_score_bytes - score_bytes
    break_even = (
        max(0, added_model_bytes) / saving_per_scene
        if added_model_bytes > 0 and saving_per_scene > 0
        else (0.0 if added_model_bytes <= 0 and saving_per_scene > 0 else None)
    )
    timings = {}
    for name in (
        "score_encode_ms",
        "score_decode_ms",
        "feature_codec_encode_ms",
        "feature_codec_decode_ms",
    ):
        value = _median(report, name)
        base = _median(baseline, name)
        timings[name] = {"median": value, "delta_ms": value - base, "delta_pct": _pct_delta(value, base)}
    memory = {}
    for name in (
        "score_encode_peak_cuda_bytes",
        "score_decode_peak_cuda_bytes",
        "feature_codec_encode_peak_cuda_bytes",
        "feature_codec_decode_peak_cuda_bytes",
    ):
        value = _median(report, name)
        base = _median(baseline, name)
        memory[name] = {"median": value, "delta_bytes": value - base}
    return {
        "suite": report["suite"],
        "lambda_tag": report["lambda_tag"],
        "label": report["label"],
        "baseline": baseline["label"],
        "scene_count": len(report["measure_scene_ids"]),
        "score_stream_count": metrics["score_stream_count"],
        "score_payload_bytes_mean": score_bytes,
        "score_entropy_string_bytes_mean": _mean(report, "score_entropy_string_bytes"),
        "score_wrapper_bytes_mean": _mean(report, "score_wrapper_bytes"),
        "score_payload_delta_bytes": score_bytes - baseline_score_bytes,
        "total_bytes_mean": _mean(report, "total_bytes"),
        "total_delta_bytes": _mean(report, "total_bytes") - _mean(baseline, "total_bytes"),
        "score_path_parameter_count": report["footprints"]["score_path"]["parameter_count"],
        "score_path_tensor_bytes": score_path_bytes,
        "score_path_serialized_bytes": report["footprints"]["score_path"][
            "serialized_state_dict_bytes"
        ],
        "context_predictor_parameter_count": report["footprints"]["context_predictor"][
            "parameter_count"
        ],
        "probability_model_parameter_count": report["footprints"]["probability_model"][
            "parameter_count"
        ],
        "added_score_path_tensor_bytes": added_model_bytes,
        "model_payload_break_even_scenes": break_even,
        "timings": timings,
        "peak_cuda_memory": memory,
    }


def aggregate_reports(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("no per-checkpoint summary.json files found")
    keys: set[tuple[str, str, str]] = set()
    by_key = {}
    reference_scenes = reports[0]["measure_scene_ids"]
    reference_warmup = reports[0]["warmup_scene_ids"]
    reference_environment = reports[0]["environment"]
    for report in reports:
        if report.get("schema_version") != 1:
            raise ValueError("unsupported or missing profile schema_version")
        key = (report["suite"], report["lambda_tag"], report["label"])
        if key in keys:
            raise ValueError(f"duplicate profile {key}")
        keys.add(key)
        by_key[key] = report
        if report["measure_scene_ids"] != reference_scenes or report["warmup_scene_ids"] != reference_warmup:
            raise ValueError(f"scene mismatch for {key}; overhead comparison is not paired")
        for field in ("gpu", "torch", "compressai", "device"):
            if report["environment"].get(field) != reference_environment.get(field):
                raise ValueError(f"environment field {field} differs for {key}")
        if not report["metrics"]["score_roundtrip_exact_all"]:
            raise ValueError(f"unverified score roundtrip in {key}")

    rows = []
    for suite in ("context", "probability"):
        present_lambdas = sorted({key[1] for key in keys if key[0] == suite})
        for lambda_tag in present_lambdas:
            baseline_key = (suite, lambda_tag, BASELINE_LABEL[suite])
            if baseline_key not in by_key:
                raise ValueError(f"missing baseline profile {baseline_key}")
            baseline = by_key[baseline_key]
            present_labels = {key[2] for key in keys if key[:2] == (suite, lambda_tag)}
            for label in LABEL_ORDER[suite]:
                if label in present_labels:
                    rows.append(comparison_row(by_key[(suite, lambda_tag, label)], baseline))
            unexpected = present_labels - set(LABEL_ORDER[suite])
            if unexpected:
                raise ValueError(f"unexpected labels for {suite}: {sorted(unexpected)}")
    return {
        "schema_version": 1,
        "paired_scene_ids": reference_scenes,
        "warmup_scene_ids": reference_warmup,
        "environment": reference_environment,
        "rows": rows,
    }


def _delta(value: float, suffix: str = "") -> str:
    return f"{value:+.2f}{suffix}"


def write_markdown(path: Path, aggregate: dict) -> None:
    lines = [
        "# NFCGS overhead profile",
        "",
        f"Paired TEST scenes: {len(aggregate['paired_scene_ids'])}; "
        f"warm-up scenes: {len(aggregate['warmup_scene_ids'])}.",
        "",
        "Context rows are compared with the same-lambda no-context Factorized continuation. "
        "Probability rows are compared with the same-lambda Full+Shared model.",
        "",
    ]
    for suite in ("context", "probability"):
        suite_rows = [row for row in aggregate["rows"] if row["suite"] == suite]
        if not suite_rows:
            continue
        lines += [f"## {suite.capitalize()}", ""]
        for lambda_tag in ("0p0064", "0p0256"):
            selected = [row for row in suite_rows if row["lambda_tag"] == lambda_tag]
            if not selected:
                continue
            lines += [f"### lambda {lambda_tag.replace('p', '.')}", ""]
            lines += [
                "| Model | Streams | Raw entropy B | Wrapper B | Score payload B | Total scene B | Break-even scenes |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in selected:
                break_even = row["model_payload_break_even_scenes"]
                break_even_text = f"{break_even:.1f}" if break_even is not None else "n/a"
                lines.append(
                    f"| {row['label']} | {row['score_stream_count']} | "
                    f"{row['score_entropy_string_bytes_mean']:.1f} | "
                    f"{row['score_wrapper_bytes_mean']:.1f} | "
                    f"{row['score_payload_bytes_mean']:.1f} "
                    f"({_delta(row['score_payload_delta_bytes'], ' B')}) | "
                    f"{row['total_bytes_mean']:.1f} ({_delta(row['total_delta_bytes'], ' B')}) | "
                    f"{break_even_text} |"
                )
            lines += [
                "",
                "| Model | Score-path params | Predictor params | Probability params | Score-path tensor KiB | Added tensor KiB | Serialized KiB |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in selected:
                lines.append(
                    f"| {row['label']} | {row['score_path_parameter_count']:,} | "
                    f"{row['context_predictor_parameter_count']:,} | "
                    f"{row['probability_model_parameter_count']:,} | "
                    f"{row['score_path_tensor_bytes']/1024:.2f} | "
                    f"{_delta(row['added_score_path_tensor_bytes']/1024, ' KiB')} | "
                    f"{row['score_path_serialized_bytes']/1024:.2f} |"
                )
            lines += [
                "",
                "| Model | Score encode ms | Score decode ms | Full codec encode ms | Full codec decode ms | Score peak enc/dec MiB |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in selected:
                timing = row["timings"]
                memory = row["peak_cuda_memory"]
                def timed(name):
                    item = timing[name]
                    return f"{item['median']:.2f} ({_delta(item['delta_pct'], '%')})"

                enc_mem = memory["score_encode_peak_cuda_bytes"]["median"] / 2**20
                dec_mem = memory["score_decode_peak_cuda_bytes"]["median"] / 2**20
                lines.append(
                    f"| {row['label']} | {timed('score_encode_ms')} | "
                    f"{timed('score_decode_ms')} | {timed('feature_codec_encode_ms')} | "
                    f"{timed('feature_codec_decode_ms')} | {enc_mem:.2f} / {dec_mem:.2f} |"
                )
            lines.append("")
    lines += [
        "## Interpretation notes",
        "",
        "- Score timing excludes the GlobalSplat image encoder, Morton ordering, low-rank analysis/synthesis, and residual coding. It includes the entropy coder plus the causal reconstruction work required by the sender.",
        "- Full codec timing starts from scene tokens, so it includes ordering, transform, score coding, residual coding, and bitstream packing, but not the image backbone or Gaussian rendering.",
        "- `Wrapper B` is only the SCCTX001 prefix and per-string lengths. The existing FP16 scene mean is not charged to context because the no-context codec already transmits it.",
        "- `Score path KiB` is exact parameter plus registered-buffer storage after checkpoint load. The full training checkpoint size is not a deployment-model measurement because it can contain optimizer state.",
        "- CUDA peak values are incremental peak-allocation deltas and omit native CPU/rANS allocations. Use medians and paired deltas; absolute latency is hardware/software specific.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_root
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        path
        for path in args.input_root.rglob("summary.json")
        if path.parent != output_dir or path.name != "aggregate.json"
    ]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]
    aggregate = aggregate_reports(reports)
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_markdown(output_dir / "REPORT.md", aggregate)
    print(f"OVERHEAD_AGGREGATE_COMPLETE profiles={len(reports)} report={output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()

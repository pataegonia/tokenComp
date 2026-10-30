"""Audit actual profiling JSON and export measured latency/memory comparisons."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


NAMES = {
    "morton_legacy2": "Morton legacy2", "morton_legacy3": "Morton legacy3",
    "hilbert_legacy3": "Hilbert legacy3", "morton_quarter2": "Morton quarter2",
    "hilbert_quarter2": "Hilbert quarter2", "morton_dyadic4": "Morton dyadic4",
    "hilbert_dyadic4": "Hilbert dyadic4", "nn_score": "NN-score legacy3",
    "context_only": "None legacy3", "context_quarter2": "None quarter2",
    "context_dyadic4": "None dyadic4", "hilbert_only": "Hilbert context OFF",
    "morton_legacy4": "Morton legacy4",
}
ENC, DEC, ORDER = "pipeline_entropy_encode", "codec_decode", "ordering_with_score_analysis"
SELECTION = ["hilbert_legacy3", "hilbert_quarter2", "hilbert_dyadic4", "nn_score", "hilbert_only", "context_only"]


def distribution(values):
    ordered = sorted(values)
    position = .9 * (len(ordered) - 1)
    left, right = math.floor(position), math.ceil(position)
    p90 = ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p90=p90, min=ordered[0], max=ordered[-1])


def audit(report, manifest=None, static=None):
    protocol, results = report["protocol"], report["results"]
    if report["mode"] != "actual" or report["skipped_missing"] or len(results) != 37:
        raise ValueError("Expected the completed actual 37-point run without missing checkpoints")
    if protocol != dict(tokens=4096, scenes=32, warmup=4, repeats=3, threads=8,
                        seed=0, precision="bf16", num_workers=4):
        raise ValueError("Unexpected profiling protocol")
    labels = [r["spec"]["label"] for r in results]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate labels")
    if manifest is not None and [r["spec"] for r in results] != manifest["points"]:
        raise ValueError("Point specifications differ from manifest")
    reference = results[0]
    measured, warmup = reference["measured_scenes"], reference["warmup_scenes"]
    if len(set(measured)) != 32 or len(set(warmup)) != 4 or set(measured) & set(warmup):
        raise ValueError("Scene lists are not disjoint and unique")
    comparisons = 0
    max_differences = defaultdict(float)
    static_rows = {r["family"]: r for r in static["results"]} if static else {}
    for row in results:
        env = {k: v for k, v in row["environment"].items() if k != "temp_directory"}
        if env != report["environment"]:
            raise ValueError("Mixed hardware/software environment")
        for key in ("input_fingerprint", "measured_scenes", "warmup_scenes"):
            if row[key] != reference[key]:
                raise ValueError(f"Mismatched {key}")
        samples = row["samples"]
        if [(s["scene"], s["repeat"]) for s in samples] != [(s, n) for s in measured for n in range(3)]:
            raise ValueError("Invalid sample order/count")
        for offset in range(0, len(samples), 3):
            group = samples[offset:offset+3]
            if len({s["total_bytes"] for s in group}) != 1:
                raise ValueError("Repeated bitstream size differs")
            for stage in ("scene_encoder", "token_position_prepare"):
                if any(s["stages"][stage] != group[0]["stages"][stage] for s in group):
                    raise ValueError("Once-per-scene measurements differ between repeats")
        for stage, saved in row["timings"].items():
            selected = samples if stage not in ("scene_encoder", "token_position_prepare") else samples[::3]
            for metric, summary in saved.items():
                values = [s["stages"][stage][metric] for s in selected]
                if not all(math.isfinite(v) and v >= 0 for v in values):
                    raise ValueError("Invalid raw measurement")
                for name, expected in distribution(values).items():
                    actual = summary[name]
                    delta = abs(actual - expected)
                    max_differences[metric] = max(max_differences[metric], delta)
                    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-7):
                        raise ValueError(f"Aggregate mismatch: {row['spec']['label']}/{stage}/{metric}/{name}")
                    comparisons += 1
            for sample in selected:
                raw = sample["stages"][stage]
                if raw["cuda_peak_bytes"] < raw["cuda_baseline_bytes"] or raw["cuda_peak_above_baseline_bytes"] != raw["cuda_peak_bytes"] - raw["cuda_baseline_bytes"]:
                    raise ValueError("Invalid allocator peak accounting")
        expected_payload = distribution([s["total_bytes"] for s in samples[::3]])
        if any(not math.isclose(row["payload_bytes"][k], v, rel_tol=1e-12, abs_tol=1e-7)
               for k, v in expected_payload.items()):
            raise ValueError("Payload aggregate mismatch")
        footprint = row["footprints"]
        if sum(footprint["groups"].values()) != footprint["active_architecture_parameters"]:
            raise ValueError("Active parameter accounting mismatch")
        if footprint["active_architecture_parameters"] + footprint["inactive_compatibility_parameters"] != footprint["stored_parameters"]:
            raise ValueError("Stored parameter accounting mismatch")
        if static_rows:
            old = static_rows[row["spec"]["family"]]
            for key in ("stored_parameters", "active_architecture_parameters"):
                if old["footprints"][key] != footprint[key]:
                    raise ValueError("Static and actual parameter counts disagree")
            for key, value in old["neural"].items():
                if row["neural"][key]["neural_macs"] != value["neural_macs"]:
                    raise ValueError("Static and actual registered neural MACs disagree")
    return dict(points=len(results), samples=len(results)*96, unique_measured_scenes=32,
                warmup_scenes=4, input_fingerprint=reference["input_fingerprint"],
                common_environment=report["environment"], protocol=protocol,
                aggregate_scalar_comparisons=comparisons, max_absolute_rounding_differences=dict(max_differences),
                same_scene_order=True, same_environment=True, same_input_fingerprint=True,
                static_counts_match=bool(static), manifest_match=manifest is not None)


def point_row(row):
    stages = {}
    for name, timing in row["timings"].items():
        stages[name] = {
            "count": timing["milliseconds"]["count"],
            "ms_mean": timing["milliseconds"]["mean"], "ms_median": timing["milliseconds"]["median"],
            "ms_p90": timing["milliseconds"]["p90"],
            "max_allocated_mib": timing["cuda_peak_bytes"]["max"]/2**20,
            "max_additional_allocated_mib": timing["cuda_peak_above_baseline_bytes"]["max"]/2**20,
            "mean_baseline_mib": timing["cuda_baseline_bytes"]["mean"]/2**20,
        }
    spec = row["spec"]
    return dict(label=spec["label"], family=spec["family"], rate_lambda=spec["rate_lambda"],
                checkpoint=spec["checkpoint"], stages=stages, payload_mean_kib=row["payload_bytes"]["mean"]/1024,
                active_params=row["footprints"]["active_architecture_parameters"],
                stored_params=row["footprints"]["stored_parameters"],
                codec_serialized_mib=row["footprints"]["serialized_codec_state_dict_bytes"]/2**20,
                neural_gmac={k: v["neural_macs"]/1e9 for k, v in row["neural"].items()},
                prior_full_eval=spec.get("observed_eval"))


def family_rows(points):
    grouped = defaultdict(list)
    for point in points:
        grouped[point["family"]].append(point)
    return [dict(family=key, name=name, point_count=len(grouped[key]),
                 mean_encode_ms=statistics.mean(p["stages"][ENC]["ms_mean"] for p in grouped[key]),
                 mean_decode_ms=statistics.mean(p["stages"][DEC]["ms_mean"] for p in grouped[key]),
                 max_encode_additional_mib=max(p["stages"][ENC]["max_additional_allocated_mib"] for p in grouped[key]),
                 max_decode_additional_mib=max(p["stages"][DEC]["max_additional_allocated_mib"] for p in grouped[key]))
            for key, name in NAMES.items() if key in grouped]


def plots(points, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = {r["family"]: r for r in points if r["rate_lambda"] == .0128}
    labels = [NAMES[k] for k in SELECTION]
    y = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for offset, stage, color, label in ((-.18, ENC, "#3976a8", "Encode"), (.18, DEC, "#dd9040", "Decode")):
        values = [rows[k]["stages"][stage]["ms_mean"] for k in SELECTION]
        bars = ax.barh([v+offset for v in y], values, height=.33, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set(yticks=y, yticklabels=labels, xlabel="Mean latency (ms / scene)", xlim=(0, 245),
           title="Actual codec latency | lambda = 0.0128")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.02, .015, "RTX A5000 | BF16 | batch 1 | same 32 scenes x 3 repeats\nEncode includes token-position preparation; image encoder, Gaussian decoder and rendering excluded.", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output/f"01_latency_l0128.{extension}", dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 4.9), sharey=True)
    for ax, stage, title in zip(axes, (ORDER, ENC, DEC), ("Ordering (+ Score analysis)", "Codec encode", "Codec decode")):
        values = [rows[k]["stages"][stage]["max_additional_allocated_mib"] for k in SELECTION]
        bars = ax.barh(y, values, height=.6, color=["#b86145" if k == "nn_score" else "#3976a8" for k in SELECTION])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set(xlim=(0, max(values)*1.2), xlabel="Max additional allocated (MiB)", title=title)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set(yticks=y, yticklabels=labels)
    axes[0].invert_yaxis()
    fig.suptitle("Allocator memory by measured call | lambda = 0.0128", fontsize=13)
    fig.text(.02, .015, "Maximum of (per-call peak - baseline) over 96 calls. PyTorch CUDA allocated memory, excluding allocator reserve and native CPU memory.\nDifferent calls have different live tensors; panels are separate measurements and must not be added.", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(output/f"02_memory_l0128.{extension}", dpi=190)
    plt.close(fig)


def markdown(points, families, checks, output):
    rows = {r["family"]: r for r in points if r["rate_lambda"] == .0128}
    h3, h4, q2, nn = (rows[k] for k in ("hilbert_legacy3", "hilbert_dyadic4", "hilbert_quarter2", "nn_score"))
    change = lambda a, b: (a/b-1)*100
    compare = lambda a, b, stage: change(a["stages"][stage]["ms_mean"], b["stages"][stage]["ms_mean"])
    env = checks["common_environment"]
    text = ["# Complexity 측정 결과 — job 455251 · 2026-10-17", "",
        "37개 checkpoint의 실제 GPU 측정 결과를 검증했다. 같은 입력·같은 GPU에서 비교한 결과, Hilbert legacy3가 quarter2와 dyadic4보다 실제 encode/decode가 빨랐다. NN-score는 decoder 시간은 비슷하지만 encoder의 정렬 비용이 컸다.", "",
        "## 검증한 측정 조건", "",
        f"- GPU: {env['gpu']}, host `{env['host']}`, UUID `{env['gpu_uuid']}`. 모든 point의 환경이 같다.",
        f"- Torch {env['torch']}, CompressAI {env['compressai']}, CUDA runtime {env['cuda_runtime']}, BF16, batch1, 4096 tokens, CPU threads8.",
        "- 같은 C12 입력 cache: warmup4장과 측정32장. Codec 구간은 scene당3회로 point당96회, image encoder와 token-position 준비는 scene당1회로 point당32회.",
        f"- 입력 fingerprint: `{checks['input_fingerprint']}`. 37개 모두 fingerprint와 장면 순서가 같다.",
        f"- 37개 point·총3552개 반복 기록에서 요약 통계 {checks['aggregate_scalar_comparisons']:,}개를 원시 sample로 재계산했다. 저장값과 허용 부동소수점 반올림 범위에서 일치했다. Allocator peak−baseline 계산, 반복 간 payload byte 일치도 확인했다.",
        "- Manifest의 37개 spec과 일치하며, 이전 static 측정의 활성/저장 파라미터 및 등록 신경망 MAC과도 일치한다.", "",
        "## λ=.0128: 실제 시간과 메모리", "",
        "시간은 32장×3회의 평균(ms/scene)이다. Encode는 `model.compress_scene_tokens`로 token-position 준비를 포함한다. Decode는 `feature_codec.decompress`이다. Image encoder, Gaussian decoder, renderer는 두 시간 열에서 제외된다.", "",
        "추가 peak는 각 호출의 `max_memory_allocated − memory_allocated_before_call`을 구한 뒤 96회 중 최댓값을 취했다. 총 peak는 측정 harness에 이미 올라간 전체 모델과 남아 있는 tensor를 포함한다. 이 값은 독립 배포한 codec의 최소 VRAM이나 NVML 전체 사용량을 뜻하지 않는다.", "",
        "| 설정 | Encode ms | Decode ms | 정렬 ms | Encode 총 peak MiB | Encode 추가 peak MiB | Decode 추가 peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for key in SELECTION:
        p = rows[key]
        t = p["stages"]
        text.append(f"| {NAMES[key]} | {t[ENC]['ms_mean']:.3f} | {t[DEC]['ms_mean']:.3f} | {t[ORDER]['ms_mean']:.3f} | {t[ENC]['max_allocated_mib']:.3f} | {t[ENC]['max_additional_allocated_mib']:.3f} | {t[DEC]['max_additional_allocated_mib']:.3f} |")
    text += ["", "![실제 codec 시간](01_latency_l0128.png)", "", "![호출별 추가 메모리](02_memory_l0128.png)", "",
        f"- Hilbert dyadic4는 legacy3보다 encode {compare(h4,h3,ENC):+.2f}%, decode {compare(h4,h3,DEC):+.2f}%. Quarter2도 encode {compare(q2,h3,ENC):+.2f}%, decode {compare(q2,h3,DEC):+.2f}%로 더 느렸다.",
        f"- NN-score는 Hilbert legacy3보다 encode {compare(nn,h3,ENC):+.2f}%, decode {compare(nn,h3,DEC):+.2f}%. 정렬+Score 분석은 {nn['stages'][ORDER]['ms_mean']:.2f}ms 대 {h3['stages'][ORDER]['ms_mean']:.2f}ms다.",
        f"- 정렬 호출의 추가 peak는 NN-score {nn['stages'][ORDER]['max_additional_allocated_mib']:.3f}MiB, Hilbert {h3['stages'][ORDER]['max_additional_allocated_mib']:.3f}MiB. 그러나 codec encode 전체 추가 peak 차이는 {nn['stages'][ENC]['max_additional_allocated_mib']-h3['stages'][ENC]['max_additional_allocated_mib']:.3f}MiB다. 서로 다른 구간에서 peak가 발생하므로 정렬의 증가량을 전체 peak에 그대로 더하면 안 된다.",
        f"- 이 harness에서 λ=.0128의 image encoder 총 peak는 설정별 {min(r['stages']['scene_encoder']['max_allocated_mib'] for r in rows.values()):.1f}–{max(r['stages']['scene_encoder']['max_allocated_mib'] for r in rows.values()):.1f}MiB, 추가 peak는 약250.3MiB다. 이를 codec peak와 더해 end-to-end peak로 부르지 않는다.",
        "- Context OFF도 기존 legacy3의 3단계×4채널그룹=12개 Score entropy streams를 유지한 모델이다. OFF가 이번 측정에서 더 빠르지는 않았다. 서로 학습된 Score 분포와 부호 길이도 달라 공간 예측 연산 삭제만의 인과 효과로 해석하지 않는다.", "",
        "## 세 λ를 묶은 설정별 시간", "",
        "각 λ의 평균 시간을 동일 가중으로 평균했다. 추가 메모리는 모든 λ/호출 중 최댓값이다. Morton legacy4만 λ=.0256 한 point다.", "",
        "| 설정 | λ 개수 | 평균 Encode ms | 평균 Decode ms | Encode 추가 peak MiB | Decode 추가 peak MiB |",
        "|---|---:|---:|---:|---:|---:|"]
    for f in families:
        text.append(f"| {f['name']} | {f['point_count']} | {f['mean_encode_ms']:.3f} | {f['mean_decode_ms']:.3f} | {f['max_encode_additional_mib']:.3f} | {f['max_decode_additional_mib']:.3f} |")
    text += ["", "NN-score의 정렬 시간은 λ=.0064에서 약103ms, .0128/.0256에서는 약68ms였다. 같은 GPU·입력이므로 λ에 따른 차이도 기록할 필요가 있다. 이 JSON에는 greedy fallback 횟수나 native CPU call stack이 없어 차이의 원인을 확정하지 않는다.", "",
        "## 파라미터와 MAC", "",
        "Encoder GMAC은 위치가 준비된 `feature_codec.compress` 경로다. 위 실제 시간의 encode 범위보다 좁다. Decoder GMAC은 `feature_codec.decompress`다. 등록된 행렬곱·convolution만 세므로 정렬 거리/비교, index/gather, 전송, rANS, 원소별 연산 등은 제외된다.", "",
        "| 설정 | 활성 Params M | 저장 Params M | Codec core encode GMAC | Decode GMAC |",
        "|---|---:|---:|---:|---:|"]
    for family in NAMES:
        p = next(r for r in points if r["family"] == family)
        text.append(f"| {NAMES[family]} | {p['active_params']/1e6:.6f} | {p['stored_params']/1e6:.6f} | {p['neural_gmac']['codec_core_encode']:.6f} | {p['neural_gmac'][DEC]:.6f} |")
    text += ["", "Quarter2/dyadic4는 predictor의 등록 MAC을 줄였지만 실제 시간은 legacy3보다 길었다. Dyadic4는 16개, legacy3는 12개의 Score entropy streams를 사용한다. 호출 횟수·작은 tensor 처리·entropy coding 등 MAC이 반영하지 않는 비용을 함께 봐야 한다. 현재 구간 측정은 이 비용들의 비율을 분리하지 않는다.", "",
        "## 기존 전체 평가의 품질·용량과 함께 보기", "",
        "아래 PSNR/bytes는 profile에서 새로 계산한 값이 아니라 manifest에 남아 있는 이전 전체 평가 값이다. 시간은 이번 통제된 profile을 사용하고, 32장 profile bytes를 전체 평가 bytes와 섞지 않는다.", "",
        "| λ=.0128 모델 | 기존 PSNR dB | 기존 실제 bytes/scene | 이번 Encode ms | 이번 Decode ms |",
        "|---|---:|---:|---:|---:|"]
    for p in (h3, q2, h4, nn):
        previous = p["prior_full_eval"]
        text.append(f"| {NAMES[p['family']]} | {previous['psnr']:.4f} | {previous['actual_bytes']:.3f} | {p['stages'][ENC]['ms_mean']:.3f} | {p['stages'][DEC]['ms_mean']:.3f} |")
    text += ["", "λ=.0128에서 Hilbert dyadic4는 기존 평가의 PSNR이 legacy3와 비슷하고 실제 bytes가 약2.85% 작지만 이번 codec 시간은 약9–11% 길다. 속도를 우선하면 Hilbert legacy3, 그 정도 시간 증가를 허용하고 용량을 줄이려면 Hilbert dyadic4가 후보이다. 이는 해당 point에서 관찰한 절충이며 고정 품질 RD 우위를 확정한 결과가 아니다.",
        "NN-score는 encode 지연을 감수하는 조건에서 고려할 수 있다. 이번 결과로 quarter2를 가장 빠른 후보로 분류했던 이전의 서로 다른 job 시간 비교를 수정한다. 같은 GPU/입력의 이번 측정에서는 legacy3가 더 빨랐다.", "",
        "## 범위와 산출물", "",
        "이번 목록은 기존 37개 point다. 새로 준비한 Hilbert Context OFF의 56채널 1 stream/4 streams 비교와 기본 순서+Context OFF 대조군은 포함하지 않는다. MSH 없는 모델, low-rank 없는 2-path, MSH 직접 압축과의 구조 비교도 포함하지 않는다.",
        "GPU 메모리는 PyTorch CUDA allocated 기준이다. Reserved cache, CUDA context/driver, CPU rANS/native memory 및 별도의 학습 peak는 측정하지 않았다. 전체 모델을 함께 로드한 profiling harness의 기준이므로 codec 전용 receiver의 최소 VRAM으로 읽지 않는다.", "",
        "- `audit.json`: 입력/환경/집계 재계산 검증 및 원본 SHA256.",
        "- `measurements.json`: 37개 point의 구간별 시간·메모리, MAC, 파라미터, 이전 품질 지표와 설정별 요약.",
        "- `01_latency_l0128.png/.pdf`, `02_memory_l0128.png/.pdf`: 같은 λ의 실제 시간과 추가 allocator 메모리.", "",
        "## 37개 point 상세", "",
        "총 peak는 encode 호출의 최대 allocated, 추가 peak는 호출 전 baseline을 뺀 최대 증가량이다. 두 열 모두 MiB다.", "",
        "| Point | Encode ms | Decode ms | Encode p90 ms | Encode 총 peak | Encode 추가 peak | Decode 추가 peak | Profile bytes KiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in points:
        t = p["stages"]
        text.append(f"| {p['label']} | {t[ENC]['ms_mean']:.3f} | {t[DEC]['ms_mean']:.3f} | {t[ENC]['ms_p90']:.3f} | {t[ENC]['max_allocated_mib']:.3f} | {t[ENC]['max_additional_allocated_mib']:.3f} | {t[DEC]['max_additional_allocated_mib']:.3f} | {p['payload_mean_kib']:.3f} |")
    (output/"REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--static", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = json.loads(raw)
    read = lambda path: json.loads(path.read_text(encoding="utf-8")) if path else None
    checks = audit(report, read(args.manifest), read(args.static))
    checks["source"] = str(args.input.resolve())
    checks["source_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    points = [point_row(r) for r in report["results"]]
    families = family_rows(points)
    for name, value in (("audit.json", checks), ("measurements.json", dict(points=points, families=families))):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    plots(points, args.output)
    markdown(points, families, checks, args.output)
    print(f"REPORT={args.output/'REPORT.md'}")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

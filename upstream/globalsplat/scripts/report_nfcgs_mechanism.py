#!/usr/bin/env python3
"""Create three scientific figures and paired scene-bootstrap summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABELS = dict(context_only_0p0128="None / L3", morton_legacy3_0p0128="Morton / L3",
    hilbert_legacy3_0p0128="Hilbert / L3", nn_score_0p0128="NN-score / L3",
    hilbert_quarter2_0p0128="Hilbert / Q2", hilbert_dyadic4_0p0128="Hilbert / D4")


def describe(values, repeats=2000):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return dict(mean=None, ci95=None, scenes=0)
    if not np.isfinite(values).all():
        raise ValueError("non-finite measurement")
    generator = np.random.default_rng(0)
    indices = generator.integers(0, len(values), (repeats, len(values)))
    interval = np.quantile(values[indices].mean(1), [.025, .975])
    return dict(mean=float(values.mean()), ci95=interval.tolist(), scenes=len(values))


def read_run(folder):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    inputs = json.loads((folder / "inputs.json").read_text(encoding="utf-8"))
    specs = metadata["identity"]["manifest"]["points"]
    if (set(complete["points"]) != {p["label"] for p in specs} or complete["scenes"] != len(inputs["scenes"])
            or complete["input_fingerprint"] != inputs["fingerprint"]):
        raise ValueError("incomplete analysis")
    results = {}
    for spec in specs:
        rows = [json.loads((folder / "points" / spec["label"] / f"scene_{i:04d}.json").read_text(encoding="utf-8"))
                for i in range(complete["scenes"])]
        if [r["scene"] for r in rows] != inputs["scenes"]:
            raise ValueError("scene pairing differs across checkpoints")
        for row in rows:
            if row["label"] != spec["label"]:
                raise ValueError("point label mismatch")
            for measurement in row["measurements"].values():
                if (sum(measurement["components"].values()) != measurement["total_bytes"] or
                    sum(s["raw_bytes"] for s in measurement["stages"]) != measurement["components"]["score_raw"]):
                    raise ValueError("inconsistent byte decomposition")
        results[spec["label"]] = rows
    return metadata, results


def anchor_metric(stage, metric="normalized_l1"):
    groups = [g for g in stage["groups"] if g["anchor_relation"]["pairs"]]
    weight = sum(g["width"] * g["anchor_relation"]["pairs"] for g in groups)
    return (sum(g["width"] * g["anchor_relation"]["pairs"] * g["anchor_relation"][metric]
                for g in groups) / weight) if weight else None


def anchor_l1(stage):
    return anchor_metric(stage)


ANCHOR_METRICS = ("raw_l1", "raw_mse", "normalized_l1", "normalized_mse")
RESIDUAL_METRICS = ("raw_mae", "raw_mse", "baseline_raw_mae", "baseline_raw_mse",
    "normalized_mae", "normalized_mse", "baseline_normalized_mae", "baseline_normalized_mse",
    "prediction_mse_ratio", "prediction_mae_ratio")


def anchor_scene_metrics(stage):
    result = {key: stage[key] for key in RESIDUAL_METRICS if key != "prediction_mae_ratio"}
    base = stage["baseline_normalized_mae"]
    result["prediction_mae_ratio"] = stage["normalized_mae"] / base if base else None
    result.update({f"anchor_{key}": anchor_metric(stage, key) for key in ANCHOR_METRICS})
    return result


def summarize_anchors(metadata, results):
    """Equal scene weighting; model differences paired before scene bootstrap."""
    reference = metadata["identity"]["manifest"]["reference"]
    metrics, models = {}, {}
    expected_tokens = metadata["identity"]["tokens"]
    for label, rows in results.items():
        if any(set(row["measurements"]) != {"trained"} or "fixed_score" in row for row in rows):
            raise ValueError("legacy3-anchors must contain only trained-model diagnostics")
        measured = [row["measurements"]["trained"]["stages"] for row in rows]
        first = measured[0]
        if len(first) != 3 or sum(s["tokens"] for s in first) != expected_tokens:
            raise ValueError("expected three legacy stages partitioning every token")
        for stages in measured:
            if [(s["stage"], s["name"], s["tokens"]) for s in stages] != [
                    (s["stage"], s["name"], s["tokens"]) for s in first]:
                raise ValueError("stage partition changed between scenes")
        metrics[label] = [[anchor_scene_metrics(s) for s in stages] for stages in measured]
        models[label] = [dict(stage=i, name=s["name"], tokens=s["tokens"],
            **{key: describe([row[i][key] for row in metrics[label]]) for key in metrics[label][0][i]})
            for i, s in enumerate(first)]
    paired = {}
    for label in results:
        if label == reference:
            continue
        paired[label] = [dict(stage=i, **{key: describe([
            a[i][key] - b[i][key] if a[i][key] is not None and b[i][key] is not None else None
            for a, b in zip(metrics[label], metrics[reference])]) for key in metrics[label][0][i]})
            for i in range(3)]
    return dict(scope="legacy3-anchors", scenes=len(next(iter(results.values()))), reference=reference,
        models=models, candidate_minus_reference=paired,
        interpretation="Distances compare original target Score with actual decoded anchor Score. "
            "MAE/MSE compare original target with the actual predictor output before entropy quantization. "
            "No-prediction baseline uses the same targets and stage partition, not another coding pass. "
            "Channels mix within predictor groups; same-channel distances describe inputs, not predictor capacity. "
            "Models have different learned bases/projections; differences are not isolated causal order effects. "
            "95% bootstrap intervals resample paired scenes, not tokens or training seeds. "
            "The first processable scenes are not a random dataset sample.")


def plots_anchors(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(summary["models"])
    colors = dict(zip(labels, ("#4c78a8", "#f58518", "#54a24b", "#b279a2")))

    def curves(ax, metric, ids, ylabel):
        for label in labels:
            values = [summary["models"][label][i][metric] for i in ids]
            y = np.array([v["mean"] for v in values])
            intervals = np.array([v["ci95"] for v in values]).T
            # Percentile intervals need not contain the finite-sample mean.
            ax.plot(ids, y, marker="o", color=colors[label], label=LABELS.get(label, label))
            ax.vlines(np.array(ids), intervals[0], intervals[1], color=colors[label], alpha=.6)
        ax.set(xticks=ids, xticklabels=[("A", "B", "C+D")[i] for i in ids],
               xlabel="Legacy3 decode stage", ylabel=ylabel)
        ax.grid(alpha=.2)

    def save(fig, stem):
        fig.tight_layout(rect=(0, 0, 1, .90))
        fig.savefig(output / f"{stem}.png", dpi=180)
        fig.savefig(output / f"{stem}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    curves(axes[0], "anchor_normalized_l1", [1, 2], "Target-anchor normalized L1")
    curves(axes[1], "anchor_normalized_mse", [1, 2], "Target-anchor normalized MSE")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Actual decoded anchor inputs; A has no spatial reference", y=.91, fontsize=11)
    save(fig, "01_target_anchor_score_distance")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, ids, title in (
            (axes[0, 0], "normalized_mae", [0, 1, 2], "Prediction residual normalized MAE"),
            (axes[0, 1], "normalized_mse", [0, 1, 2], "Prediction residual normalized MSE"),
            (axes[1, 0], "prediction_mae_ratio", [1, 2], "Residual MAE / baseline MAE"),
            (axes[1, 1], "prediction_mse_ratio", [1, 2], "Residual MSE / baseline MSE")):
        curves(ax, metric, ids, title)
        if "ratio" in metric:
            ax.axhline(1, color="gray", ls="--", lw=1)
    fig.legend(handles, names, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Per-scene channel STD normalization; bars show 95% scene-bootstrap intervals",
                 y=.94, fontsize=10)
    save(fig, "02_prediction_residuals")


def report_anchors(metadata, results, output):
    summary = summarize_anchors(metadata, results)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plots_anchors(summary, output)
    def number(value):
        return "—" if value is None else f"{value:.5f}"
    lines = ["# Legacy3 실제 anchor·예측 잔차 분석", "",
        f"λ=.0128, 동일 C12 입력 {summary['scenes']} scene, 모델4개. 입력 fingerprint는 상위 inputs.json에 있다.", "",
        "거리는 원래 target Score와 실제 양자화·복호된 anchor 사이에서 계산한다. 범위 밖 padding과 아직 복호되지 않은 zero canvas 위치는 anchor 통계에서 제외한다. A에는 공간 anchor가 없다. B는 ±2, C+D는 ±1의 유효 참조를 사용한다.", "",
        "잔차는 양자화 전 target−prediction이다. baseline은 동일 단계·동일 target의 target−base이며 현재 minimal 설정에서 base=0이다. 비율이1보다 작으면 해당 모델의 공간 예측이 이 기준보다 오차를 줄였다. Entropy median을 빼거나 다른 모델을 baseline으로 쓰지 않았다.", "",
        "| 모델 | 단계 | tokens | anchor 정규화 L1 | baseline MAE | 잔차 MAE | baseline MSE | 잔차 MSE | MSE 비율 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, stages in summary["models"].items():
        for stage in stages:
            values = [number(stage[key]["mean"]) for key in ("anchor_normalized_l1", "baseline_normalized_mae",
                "normalized_mae", "baseline_normalized_mse", "normalized_mse", "prediction_mse_ratio")]
            lines.append(f"| {LABELS.get(label,label)} | {stage['name']} | {stage['tokens']} | " + " | ".join(values) + " |")
    lines += ["", "![Anchor Score 거리](01_target_anchor_score_distance.png)", "",
        "![예측 잔차](02_prediction_residuals.png)", "",
        "정규화는 scene별·채널별 원래 Score의 population STD(floor=1e-6)를 사용하며 encoder 입력은 바꾸지 않는다. 단계 값은 모든 대상×채널 또는 유효 anchor 쌍×채널의 평균이고, 최종 값은 scene을 동일 가중치로 평균했다. 원래 Score 단위 MAE/MSE, 정규화 값, 예측 전 baseline, paired 차이와95% 구간은 summary.json에, group/채널별 값은 points/에 저장한다.", "",
        f"Paired 차이는 각 scene에서 candidate−`{summary['reference']}`를 먼저 계산한 뒤 scene 단위로 bootstrap했다. 구간에는 학습 seed 변동이 포함되지 않는다. 첫 processable scene 집합으로 전체 데이터의 무작위 표본은 아니다.", "",
        "학습된 basis·geometry projection·예측기가 모델마다 다르므로 이 결과는 학습된 네 모델의 동작을 설명한다. 정렬만 바꾼 인과 효과나 잔차 감소에 따른 전체 bit 감소를 단정하지 않는다. 예측기는 group 내 채널을 혼합하므로 같은 채널의 anchor 거리만으로 예측 성능을 설명할 수 없다. 재학습·렌더링·prediction bypass·다른 Context 스케줄 비교는 수행하지 않았다.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def summarize_schedule_anchors(metadata, results):
    reference = metadata["identity"]["manifest"]["reference"]
    specs = {p["label"]: p for p in metadata["identity"]["manifest"]["points"]}
    models, whole_values = {}, {}
    for label, rows in results.items():
        cfg = specs[label]["config"]
        stage_count = cfg["score_spatial_stages"]
        measured = [r["measurements"]["trained"] for r in rows]
        if any(set(r["measurements"]) != {"trained"} or "fixed_score" in r for r in rows):
            raise ValueError("schedule analysis must contain trained-model diagnostics only")
        first = measured[0]["stages"]
        if len(first) != stage_count or sum(s["tokens"] for s in first) != metadata["identity"]["tokens"]:
            raise ValueError("schedule stages do not partition the token set")
        stage_metrics = []
        for m in measured:
            if [(s["stage"],s["name"],s["tokens"]) for s in m["stages"]] != [
                    (s["stage"],s["name"],s["tokens"]) for s in first]:
                raise ValueError("schedule stage partition changed between scenes")
            stage_metrics.append([dict(anchor_scene_metrics(s), bits_per_token=s["bits_per_token"],
                raw_bytes=s["raw_bytes"], quantization_normalized_mse=s["quantization_normalized_mse"])
                for s in m["stages"]])
        whole_values[label] = []
        for m in measured:
            count = sum(s["symbols"] for s in m["stages"])
            values = {key: sum(s[key]*s["symbols"] for s in m["stages"])/count for key in (
                "raw_mae", "raw_mse", "normalized_mae", "normalized_mse", "baseline_normalized_mse",
                "quantization_normalized_mse")}
            values.update(total_bytes=m["total_bytes"], score_raw_bytes=m["components"]["score_raw"],
                msh_bytes=m["components"]["msh_y"]+m["components"]["msh_z"], headers_bytes=m["components"]["headers"])
            whole_values[label].append(values)
        models[label] = dict(schedule=cfg["score_context_schedule"],
            stages=[dict(stage=i,name=s["name"],tokens=s["tokens"],
                **{key:describe([r[i][key] for r in stage_metrics]) for key in stage_metrics[0][i]})
                for i,s in enumerate(first)],
            whole={key:describe([r[key] for r in whole_values[label]]) for key in whole_values[label][0]})
    paired = {label:{key:describe([a[key]-b[key] for a,b in zip(values,whole_values[reference])])
              for key in values[0]} for label,values in whole_values.items() if label != reference}
    checks = [r.get("source_reference_check") for r in results[reference]]
    if any(c is None for c in checks):
        raise ValueError("reference repeatability checks are missing")
    repeat = dict(scenes=len(checks), identical_packet_scenes=sum(c["packet_sha256_equal"] for c in checks),
        total_bytes_delta=describe([c["total_bytes_delta"] for c in checks]),
        score_raw_bytes_delta=describe([c["score_raw_bytes_delta"] for c in checks]),
        max_abs_stage_raw_mae_delta=max(abs(v) for c in checks for v in c["stage_raw_mae_delta"]))
    return dict(scope="hilbert-schedule-anchors",reference=reference,models=models,
        candidate_minus_reference_whole_scene=paired,reference_repeatability=repeat,
        interpretation="Hilbert ordering algorithm is fixed; learned geometry projections may change actual permutations. "
            "Learned bases and predictors also differ. "
            "Stage indices refer to different target sets across schedules; no ordinal-stage paired comparison. "
            "Whole-scene residuals weight each target-channel symbol equally, including unpredicted anchors. "
            "Distances use actual decoded anchor inputs; boundary duplicate slots remain duplicated. "
            "Scene-bootstrap intervals do not include training-seed variability.")


def plots_schedule_anchors(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(summary["models"])
    fig, axes = plt.subplots(len(labels), 4, figsize=(15, 3.1*len(labels)), layout="constrained", squeeze=False)
    for row,label in enumerate(labels):
        model=summary["models"][label]
        stages=model["stages"]
        for col,(metric,title) in enumerate((("anchor_normalized_l1","Decoded anchor-target normalized L1"),
                ("raw_mae","Residual MAE (Score units)"),("normalized_mae","Residual MAE / channel STD"),
                ("bits_per_token","Actual Score bits / stage target token"))):
            ax=axes[row,col]
            ids=[i for i,s in enumerate(stages) if s[metric]["mean"] is not None]
            vals=[stages[i][metric] for i in ids]
            ax.plot(ids,[v["mean"] for v in vals],marker="o")
            ax.vlines(ids,[v["ci95"][0] for v in vals],[v["ci95"][1] for v in vals],alpha=.5)
            ax.set(xticks=range(len(stages)),xticklabels=[f"{s['name']}\nn={s['tokens']}" for s in stages],
                   title=title,ylabel=LABELS.get(label,label) if col==0 else "")
            ax.tick_params(axis="x",labelsize=8)
            ax.grid(alpha=.2)
    fig.suptitle("Hilbert order / same cached scenes: each row has its own target partition",fontsize=12)
    for extension in ("png","pdf"):
        fig.savefig(output/f"01_schedule_stages.{extension}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for metric,name,color in (("score_raw_bytes","Score raw","#4c78a8"),("msh_bytes","MSH y+z","#f58518"),
                              ("headers_bytes","Headers","#bab0ac")):
        values=[summary["models"][label]["whole"][metric]["mean"]/1024 for label in labels]
        bottom=np.zeros(len(labels)) if metric=="score_raw_bytes" else bottom
        axes[0].bar(range(len(labels)),values,bottom=bottom,label=name,color=color)
        bottom+=np.array(values)
    axes[0].set(xticks=range(len(labels)),xticklabels=[LABELS.get(l,l) for l in labels],ylabel="Actual KiB / scene")
    axes[0].legend(fontsize=8,ncol=3)
    values=[summary["models"][label]["whole"]["raw_mae"] for label in labels]
    axes[1].bar(range(len(labels)),[v["mean"] for v in values],color="#54a24b")
    axes[1].vlines(range(len(labels)),[v["ci95"][0] for v in values],[v["ci95"][1] for v in values],color="black")
    axes[1].set(xticks=range(len(labels)),xticklabels=[LABELS.get(l,l) for l in labels],
                ylabel="Residual MAE (all 4096 targets, Score units)")
    for ax in axes:
        ax.grid(axis="y",alpha=.2)
    for extension in ("png","pdf"):
        fig.savefig(output/f"02_whole_scene_cost.{extension}",dpi=180)
    plt.close(fig)


def report_schedule_anchors(metadata, results, output):
    summary=summarize_schedule_anchors(metadata,results)
    output.mkdir(parents=True)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    plots_schedule_anchors(summary,output)
    repeat=summary["reference_repeatability"]
    lines=["# Hilbert 고정: legacy3·quarter2·dyadic4 실제 anchor와 잔차", "",
        f"λ=.0128, 동일 캐시 {len(next(iter(results.values())))} scene. 원래 입력 fingerprint: `{metadata['identity']['input_run']['inputs']['fingerprint']}`.", "",
        f"기존 Hilbert legacy3와 packet SHA256가 같은 scene은 {repeat['identical_packet_scenes']}/{repeat['scenes']}개다. 전체 byte 평균 차이는 {repeat['total_bytes_delta']['mean']:+.5f}B, 단계 raw MAE 차이의 최대 절댓값은 {repeat['max_abs_stage_raw_mae_delta']:.8g}다. 비교는 이번 job에서 다시 측정한 legacy3를 기준으로 한다.", "",
        "| 모델 | 단계 | 대상 수 | anchor 정규화 L1 | 예측 전 정규화 MAE | 잔차 정규화 MAE | 잔차 정규화 MSE | 잔차 원래 단위 MAE | bits/token |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def text(value):
        return "—" if value is None else f"{value:.5f}"
    for label,model in summary["models"].items():
        for s in model["stages"]:
            vals=[s[k]["mean"] for k in ("anchor_normalized_l1","baseline_normalized_mae","normalized_mae",
                  "normalized_mse","raw_mae","bits_per_token")]
            lines.append(f"| {LABELS.get(label,label)} | {s['name']} | {s['tokens']} | "+" | ".join(map(text,vals))+" |")
    lines += ["", "![단계별 anchor·잔차·bit](01_schedule_stages.png)", "",
        "legacy3: 0mod4→2mod4→홀수. Quarter2: 0mod4→나머지 phase1/2/3을 함께 복호한다. Dyadic4: 0mod8→4mod8→2mod4→홀수. Quarter2의 target은 자기4개 블록의 좌우0mod4 anchor를 참조하고, dyadic4의 간격은 단계별±4/±2/±1이다. 오른쪽 anchor가 없으면 왼쪽 anchor를 복제하며 두 입력 slot을 모두 통계에 포함한다.", "",
        "첫 단계에는 공간 anchor가 없다. 거리와 잔차는 실제 부호화 pass에서 관측한 값이며 앞 단계에서 실제 복호한 anchor를 사용한다. 현재 minimal 설정에서 base=0이다. 정규화는 scene·채널별 원래 Score의 population STD(floor1e-6)이며 encoder에는 적용하지 않는다.", "",
        "| 모델 | 전체4096 raw MAE | 전체4096 정규화 MAE | Score raw KiB | MSH KiB | 전체 KiB |", "|---|---:|---:|---:|---:|---:|"]
    for label,model in summary["models"].items():
        w=model["whole"]
        vals=[w[k]["mean"] for k in ("raw_mae","normalized_mae")]+[w[k]["mean"]/1024 for k in (
            "score_raw_bytes","msh_bytes","total_bytes")]
        lines.append(f"| {LABELS.get(label,label)} | "+" | ".join(map(text,vals))+" |")
    lines += ["", "![전체 비용](02_whole_scene_cost.png)", "",
        "단계별 대상 집합이 다르므로 같은 단계 번호끼리 차이를 계산하지 않는다. Whole-scene 잔차는 예측 없는 첫 단계까지 포함해 전체4096×56개 원소를 동일 가중치로 평균한다. 이 전체 지표와 byte 합에만 legacy3 대비 paired 차이와95% scene-bootstrap 구간을 summary.json에 기록한다. 원래 단위·정규화 MAE/MSE, group/채널별 값은 원본 points/에도 있다.", "",
        "Hilbert 정렬 알고리즘을 고정했지만 모델별 geometry projection 차이로 실제 permutation은 달라질 수 있다. 스케줄마다 학습된 basis·예측기·확률모델도 다르므로 학습된 모델의 동작 비교다. 잔차가 작다는 사실만으로 전체 RD 개선을 단정하지 않는다. Bootstrap 구간은 학습 seed 변동을 포함하지 않는다. 선택한 첫 processable256 scene은 전체 평가셋의 무작위 표본이 아니다.", ""]
    (output/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    return summary


def summarize(metadata, results):
    reference = metadata["identity"]["manifest"]["reference"]
    fixed_rows = [r["fixed_score"] for r in results[reference]]
    fixed, models = {}, {}
    for order in fixed_rows[0]:
        fixed[order] = dict(lags={}, topology={})
        for lag in fixed_rows[0][order]["lags"]:
            fixed[order]["lags"][lag] = {
                metric: describe([r[order]["lags"][lag][metric] for r in fixed_rows])
                for metric in ("normalized_l1", "xyz_distance", "xyz_distance_normalized")}
            fixed[order]["lags"][lag]["correlation"] = describe([
                r[order]["lags"][lag]["correlation"]["mean"] for r in fixed_rows])
        for schedule, stages in fixed_rows[0][order]["topology"].items():
            fixed[order]["topology"][schedule] = [dict(stage=i, tokens=s["target_tokens"],
                normalized_l1=describe([r[order]["topology"][schedule][i]["normalized_l1"] for r in fixed_rows]),
                correlation=describe([r[order]["topology"][schedule][i]["correlation"]["mean"] for r in fixed_rows]))
                for i, s in enumerate(stages)]
    for label, rows in results.items():
        models[label] = {}
        for variant in rows[0]["measurements"]:
            measured = [r["measurements"][variant] for r in rows]
            entry = dict(total_bytes=describe([m["total_bytes"] for m in measured]),
                feature_normalized_mse=describe([m["feature_normalized_mse"] for m in measured]),
                components={key: describe([m["components"][key] for m in measured]) for key in measured[0]["components"]},
                stages=[])
            for stage, first in enumerate(measured[0]["stages"]):
                stage_rows = [m["stages"][stage] for m in measured]
                if any(s["tokens"] != first["tokens"] or s["name"] != first["name"] for s in stage_rows):
                    raise ValueError("stage partition changed between scenes")
                entry["stages"].append(dict(name=first["name"], tokens=first["tokens"],
                    **{metric: describe([s[metric] for s in stage_rows]) for metric in (
                        "bits_per_token", "bits_per_symbol", "normalized_mae", "normalized_mse",
                        "prediction_mse_ratio", "quantization_normalized_mse", "coding_gap_bits_per_token")},
                    anchor_normalized_l1=describe([anchor_l1(s) for s in stage_rows])))
            models[label][variant] = entry
    # All comparisons use differences on the SAME scenes before bootstrapping.
    paired = []
    for label in results:
        if label == reference:
            continue
        a, b = results[label], results[reference]
        paired.append(dict(candidate=label, reference=reference, variant="trained",
            total_bytes_delta=describe([x["measurements"]["trained"]["total_bytes"]-
                                       y["measurements"]["trained"]["total_bytes"] for x, y in zip(a, b)])))
    prediction_pairs = {}
    for label, rows in results.items():
        if "prediction_bypassed" not in rows[0]["measurements"]:
            continue
        prediction_pairs[label] = {
            metric: describe([r["measurements"]["prediction_bypassed"][metric] -
                              r["measurements"]["trained"][metric] for r in rows])
            for metric in ("total_bytes", "feature_normalized_mse")}
        for component in ("score_raw", "msh_y", "msh_z", "headers"):
            prediction_pairs[label][component] = describe([
                r["measurements"]["prediction_bypassed"]["components"][component] -
                r["measurements"]["trained"]["components"][component] for r in rows])
    return dict(schema_version=1, metadata=metadata, fixed_score=fixed, models=models,
                paired_bytes_vs_hilbert_legacy3=paired, prediction_bypass_minus_trained=prediction_pairs,
                interpretation="Learned-model comparisons explain behavior, not isolated order interventions. "
                    "Prediction bypass is inference-only with unchanged learned densities/MSH, not a trained OFF control. "
                    "CIs resample scenes, not tokens; the first processable scene subset is not a random dataset sample. "
                    "No rendered PSNR or controlled latency was measured.")


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for order, stats in summary["fixed_score"].items():
        lags = sorted(stats["lags"], key=int)
        x = np.array([int(lag) for lag in lags])
        for ax, metric, title in zip(axes, ("normalized_l1", "correlation", "xyz_distance_normalized"),
                ("Score distance / channel STD", "Mean channel Pearson correlation", "XYZ distance / RMS scene radius")):
            values = [stats["lags"][lag][metric] for lag in lags]
            y = [v["mean"] for v in values]
            line, = ax.plot(x, y, marker="o", linestyle="--" if order == "random" else "-", label=order)
            ax.fill_between(x, [v["ci95"][0] for v in values], [v["ci95"][1] for v in values],
                            color=line.get_color(), alpha=.13)
            ax.set(xlabel="Sequence lag", title=title, xscale="log", xticks=[1, 2, 4, 8])
            ax.set_xticklabels([1, 2, 4, 8])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, "01_fixed_score_relations")
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    specs = summary["metadata"]["identity"]["manifest"]["points"]
    for column, selected in enumerate((
            [s for s in specs if s["config"]["score_context_schedule"] == "legacy"],
            [s for s in specs if s["config"]["token_order"] == "hilbert"])):
        for spec in selected:
            label = spec["label"]
            stages = summary["models"][label]["trained"]["stages"]
            x = np.arange(1, len(stages) + 1)
            for row, (metric, ylabel) in enumerate((
                    ("anchor_normalized_l1", "Target vs decoded input / channel STD"),
                    ("prediction_mse_ratio", "Residual NMSE / no-prediction NMSE"),
                    ("bits_per_token", "Actual Score bits / stage target token"))):
                y = [s[metric]["mean"] if s[metric]["mean"] is not None else np.nan for s in stages]
                axes[row, column].plot(x, y, marker="o", label=LABELS.get(label, label))
                axes[row, column].set(xlabel="Decode stage (schedule partitions differ)", ylabel=ylabel, xticks=[1,2,3,4])
                axes[row, column].grid(alpha=.2)
        axes[0, column].set_title("Trained legacy3 models / order comparison" if column == 0 else "Trained Hilbert models / schedule comparison")
        axes[0, column].legend(fontsize=8)
        axes[1, column].axhline(1, color="gray", linestyle="--", linewidth=1)
    save(fig, "02_causal_context_and_bits")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = list(summary["models"])
    bottom = np.zeros(len(labels))
    for component in ("score_raw", "msh_y", "msh_z", "headers", "mean"):
        heights = np.array([summary["models"][label]["trained"]["components"][component]["mean"] / 1024 for label in labels])
        ax.bar(np.arange(len(labels)), heights, bottom=bottom, label=component)
        bottom += heights
    ax.set(xticks=np.arange(len(labels)), xticklabels=[LABELS.get(label, label) for label in labels],
           ylabel="Actual KiB / scene", title="Actual packet decomposition: entropy strings and all headers")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.02), fontsize=8)
    ax.set_title("Actual packet decomposition: entropy strings and all headers", pad=38)
    ax.grid(axis="y", alpha=.2)
    save(fig, "03_actual_byte_decomposition")


def report(folder, output):
    metadata, results = read_run(folder)
    if output.exists():
        raise FileExistsError("use a fresh report directory")
    if metadata["identity"].get("scope") == "legacy3-anchors":
        return report_anchors(metadata, results, output)
    if metadata["identity"].get("scope") == "hilbert-schedule-anchors":
        return report_schedule_anchors(metadata, results, output)
    output.mkdir(parents=True)
    summary = summarize(metadata, results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plots(summary, output)
    lines = ["# 정렬·Context·실제 bit 분석", "",
        f"동일 scene {len(next(iter(results.values())))}개, λ=.0128. 고정 Score 기준: `{metadata['identity']['manifest']['reference']}`.", "",
        "단계별 bits/token은 해당 단계의 모든 Score 채널 raw entropy bytes × 8 / 대상 token 수다. Score·MSH·상위 container header는 용량 분해에서 중복 없이 계산했다.", "",
        "| 모델 | 실제 KiB/scene | Score raw KiB | MSH y+z KiB |", "|---|---:|---:|---:|"]
    for label, variants in summary["models"].items():
        value = variants["trained"]
        components = value["components"]
        lines.append(f"| {LABELS.get(label,label)} | {value['total_bytes']['mean']/1024:.3f} | {components['score_raw']['mean']/1024:.3f} | {(components['msh_y']['mean']+components['msh_z']['mean'])/1024:.3f} |")
    lines += ["", "| 모델 | 단계 | token 수 | 예측 NMSE/기준 NMSE | 실제 bits/token | 양자화 NMSE |", "|---|---|---:|---:|---:|---:|"]
    for label, variants in summary["models"].items():
        for stage in variants["trained"]["stages"]:
            ratio = stage["prediction_mse_ratio"]["mean"]
            ratio_text = f"{ratio:.4f}" if ratio is not None else "—"
            lines.append(f"| {LABELS.get(label,label)} | {stage['name']} | {stage['tokens']} | {ratio_text} | {stage['bits_per_token']['mean']:.3f} | {stage['quantization_normalized_mse']['mean']:.5f} |")
    lines += ["", "![고정 Score 관계](01_fixed_score_relations.png)", "",
              "![실제 causal Context와 bit](02_causal_context_and_bits.png)", "",
              "![실제 용량 분해](03_actual_byte_decomposition.png)", "",
              "고정 Score 분석은 순서만 바꾼 비교다. 학습된 모델 간 Score/basis 차이는 남아 있으므로 후자의 차이를 정렬만의 인과 효과로 해석하지 않는다. 첫 단계에는 공간 anchor가 없어서 anchor 관계 값이 없다. 스케줄별 단계는 서로 다른 token 집합을 뜻하며 위 표에 분모를 함께 제시했다.", "",
              "예측 bypass 결과가 있으면 summary.json에 같은 scene의 bypass−trained 차이와 95% scene-bootstrap 구간을 기록한다. 학습된 확률모델과 MSH를 그대로 쓰는 추론 진단이므로 재학습한 Context OFF 대조군과 구분한다. 예상 −log2 likelihood와 실제 부호 길이의 차이에는 정수 CDF, rANS 종료/escape 및 작은 배열 padding 등의 영향이 포함될 수 있다.", "",
              "선택한 첫 processable scene 집합에 대한 분석이며 전체 데이터의 무작위 표본은 아니다. Renderer/PSNR와 통제된 속도 측정은 수행하지 않았다.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.input, args.output)

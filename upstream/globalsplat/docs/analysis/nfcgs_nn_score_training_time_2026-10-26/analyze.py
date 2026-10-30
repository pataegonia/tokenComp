"""Read pre-fresh training progress clocks; never query or change Slurm."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
WORKSPACE = REPO.parents[1]
LOGS = WORKSPACE/"slurm"
RUNS = [
    ("NN-score L3", ".0064", "450937"), ("NN-score L3", ".0128", "450939"),
    ("None L3", ".0064", "451809"), ("None L3", ".0128", "451811"),
    ("Morton L3", ".0064", "447719"), ("Morton L3", ".0128", "447721"),
    ("Hilbert L3", ".0064", "447723"), ("Hilbert L3", ".0128", "447725"),
    ("Hilbert D4", ".0064", "447744"), ("Hilbert D4", ".0128", "447746"),
    ("Morton L2", ".0064", "449559"), ("Morton L2", ".0128", "449561"),
]
PROGRESS = re.compile(r"Epoch\s+(\d+):\s*\d+%[^\r\n]*?\s(\d+)/(\d+)\s*\[([\d:]+)")
KEYS = ["trainer.devices", "trainer.accumulate_grad_batches", "optimizer.batch_size",
    "trainer.max_steps", "trainer.precision", "checkpointing.resume", "checkpointing.load",
    "optimizer.lr", "model.feature_codec.token_order", "model.feature_codec.score_context_schedule",
    "model.feature_codec.score_spatial_stages", "model.feature_codec.score_spatial_kernel",
    "model.feature_codec.nn_order_backend", "model.feature_codec.nn_order_distance_device", "loss.rate_lambda"]


def seconds(stamp):
    result = 0
    for part in stamp.split(":"):
        result = 60*result+int(part)
    return result


def analyze(label, rate, job, completed=True):
    path, = [p for p in LOGS.glob("*.out") if p.stem.endswith("-"+job)]
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    error = path.with_suffix(".err").read_text(encoding="utf-8",errors="replace")
    config = {}
    for key in KEYS:
        found = re.search(r"(?:^|\s)"+re.escape(key)+r"=([^\s]+)", text)
        if found:
            config[key] = found.group(1)
    assert config["trainer.devices"] == "2"
    assert config["trainer.accumulate_grad_batches"] == config["optimizer.batch_size"] == "2"
    assert config["trainer.max_steps"] == "50000"
    assert abs(float(config["loss.rate_lambda"])-float(rate)) < 1e-12
    if completed:
        assert "`max_steps=50000` reached" in error
        assert not re.search(r"Traceback|CUDA out of memory|No space left",error)
    clocks = defaultdict(dict)
    reset_rows = 0
    for match in PROGRESS.finditer(text):
        epoch,batch,total = map(int,match.group(1,2,3))
        elapsed = seconds(match.group(4))
        # Lightning prints a reset-to-zero row for the old epoch at boundaries.
        # Keep the largest clock for each batch, never the final row in the file.
        if batch == 0 and clocks[epoch]:
            reset_rows += 1
        clocks[epoch][batch] = max(elapsed, clocks[epoch].get(batch,0))
        assert total == 33017
    epochs = []
    for epoch, values in sorted(clocks.items()):
        last_batch = max(values)
        last_sec = values[last_batch]
        assert last_sec == max(values.values())
        epochs.append(dict(epoch=epoch,last_batch=last_batch,elapsed_seconds=last_sec))
    elapsed = sum(e["elapsed_seconds"] for e in epochs)
    batch_total = sum(e["last_batch"] for e in epochs)
    resume = config.get("checkpointing.resume") == "true"
    if completed and not resume:
        assert [e["last_batch"] for e in epochs] == [33017,33017,33017,946]
        steps = 50000
    elif completed and resume:
        # This one source is the explicit step10k -> step50k NN-score continuation.
        assert job == "450934"
        assert "Restored all states" in error
        assert 20000 in clocks[0]
        steps = 40000
    else:
        steps = batch_total/2
    return dict(label=label,rate_lambda=float(rate),job=job,
        source_log=path.relative_to(WORKSPACE).as_posix(),source_sha256=hashlib.sha256(raw).hexdigest(),
        source_stderr_sha256=hashlib.sha256(path.with_suffix('.err').read_bytes()).hexdigest(),
        completed_50k=completed,resumed=resume,configuration=config,epochs=epochs,
        epoch_reset_rows_ignored=reset_rows,progress_elapsed_seconds=elapsed,
        progress_elapsed_hours=elapsed/3600,optimizer_steps_in_this_run=steps,
        seconds_per_optimizer_step=elapsed/steps,equivalent_50k_hours=elapsed/steps*50000/3600,
        gpu_model_confirmed=False,slurm_accounting_elapsed_available=False)


def main():
    rows = [analyze(*run) for run in RUNS]
    aux = [analyze("NN-score legacy initial", ".0256", "448650",False),
           analyze("NN-score optimized resumed", ".0256", "450934")]
    comparisons = []
    for rate in (.0064,.0128):
        nn, = [r for r in rows if r["label"] == "NN-score L3" and r["rate_lambda"] == rate]
        for base in [r for r in rows if r["label"] != "NN-score L3" and r["rate_lambda"] == rate]:
            ratio = nn["progress_elapsed_seconds"]/base["progress_elapsed_seconds"]
            comparisons.append(dict(rate_lambda=rate,nn_job=nn["job"],baseline_job=base["job"],
                baseline=base["label"],wall_time_ratio=ratio,extra_time_percent=100*(ratio-1),
                extra_hours=nn["progress_elapsed_hours"]-base["progress_elapsed_hours"]))
    source = rows[0]["configuration"]["checkpointing.load"]
    assert all(r["configuration"]["checkpointing.load"] == source for r in rows)
    for r in rows:
        assert r["configuration"].get("checkpointing.resume") == "false"
        assert r["configuration"]["model.feature_codec.score_spatial_kernel"] == "5"
        assert r["configuration"]["model.feature_codec.score_spatial_stages"] == ("4" if r["label"] == "Hilbert D4" else "2" if r["label"] == "Morton L2" else "3")
    result = dict(date="2026-10-26 Asia/Seoul",scope="pre-fresh warm-start 50k codec training",
        method="sum largest valid progress elapsed clock at each epoch's maximum batch; ignore old-epoch zero resets",
        matched_settings=dict(gpus=2,batch_per_gpu=2,accumulation=2,effective_batch=8,max_steps=50000,
            same_weights_only_parent=source,main_nn_backend="knn",main_nn_distance_device="input"),
        limitations=["Progress loop time, not sacct allocation elapsed; startup and epoch-transition overhead not fully captured.",
            "Includes work inside progress clocks (training/checkpoint/quantile work), excludes queue and test evaluation.",
            "GPU models/nodes/load not controlled by the available logs.",
            "Legacy initial 50k time is extrapolated; resumed .0256 runs only 40k optimizer steps."],
        runs=rows,auxiliary_runs=aux,comparisons=comparisons,server_jobs_changed=False)
    (OUT/"timing.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    report = ["""# Fresh 이전 NN-score 학습 시간 비교 — 2026-10-26

최적화된 NN-score L3는50k 학습 루프에 약28.2~28.6시간이 걸렸다. 같은λ·L3의 None/Morton/Hilbert는16.9~17.2시간으로, NN-score가 약1.66~1.70배(+66~70%), 약11~12시간 더 걸렸다. Hilbert D4와 비교하면 약1.53~1.55배(+53~55%), 약10시간 더 걸렸다.

## 비교 조건과 계산

- 새 fresh-init 실험을 제외한 기존 FullSplit .0256 step10k 부모의 weights-only 후속50k다.
- 주 비교12개 모두 실제 실행 명령의 GPU2 × microbatch2 × accumulation2 = 유효 batch8, 목표50k, kernel5를 확인했다. 같은λ .0064/.0128을 비교한다.
- NN-score는 `knn`·후보64개·거리장치 `input`의 최적화된 학습450937/450939다. L3 계열끼리의 비교를 우선하며 D4/L2는 schedule이 다른 참고 비교다.
- tqdm의 epoch별 마지막 batch에서 가장 큰 elapsed clock을 합했다. epoch 전환 때 출력되는 이전 epoch의0/33017·00:00 행은 제외했다. 주 비교 모두 batch 수33017+33017+33017+946, 최종50k 종료 footer를 확인했다. accumulation과 epoch 끝의 마지막 불완전 그룹 때문에 batch 수를 단순히2로 나누어 정확한 최종 global step을 추정하지 않는다.
- 아래 시간은 학습 progress 경과시간이다. Queue와 test eval은 포함하지 않으며, 초기화·epoch 전환 등의 일부 overhead가 빠져 Slurm allocation 전체 elapsed와 같지 않다. Clock 안의 checkpoint/quantile 등의 비용은 포함될 수 있다. GPU 모델·동일 노드·부하를 통제한 benchmark는 아니다.

## 완료한50k 학습 로그

| 설정 | λ=.0064 | λ=.0128 | Job(.0064 / .0128) |
|---|---:|---:|---|
"""]
    for label in ("None L3","Morton L3","Hilbert L3","Hilbert D4","Morton L2","NN-score L3"):
        pair = [next(r for r in rows if r["label"] == label and r["rate_lambda"] == rate) for rate in (.0064,.0128)]
        report.append(f"| {label} | {pair[0]['progress_elapsed_hours']:.2f}h | {pair[1]['progress_elapsed_hours']:.2f}h | {pair[0]['job']} / {pair[1]['job']} |\n")
    report.append("\n| NN-score / 기준 | λ | 시간 배수 | 추가 시간 | 증가율 |\n|---|---:|---:|---:|---:|\n")
    for c in comparisons:
        report.append(f"| {c['baseline']} | {c['rate_lambda']:g} | {c['wall_time_ratio']:.3f}× | {c['extra_hours']:.2f}h | +{c['extra_time_percent']:.1f}% |\n")
    report.append("\n## 초기 구현·재개 실행은 별도\n\n")
    old,resumed = aux
    report.append(f"초기 NN-score448650은 취소 시 약{old['optimizer_steps_in_this_run']/1000:.2f}k optimizer step까지{old['progress_elapsed_hours']:.2f}시간이 걸렸다. 그 속도를50k로 환산하면 약{old['equivalent_50k_hours']:.1f}시간({old['equivalent_50k_hours']/24:.2f}일)이지만, 완료한50k 실측 시간이 아니다.\n\n")
    report.append(f"최적화 후450934는 step10k에서50k까지40k를 재개해 약{resumed['progress_elapsed_hours']:.2f}시간이다.50k 환산은 약{resumed['equivalent_50k_hours']:.2f}시간으로, 새50k의 주 비교와 구분한다. 초기 취소 전 미저장분·실패450904·대기·재시도 시간을 완료본에 합치지 않았다.\n\n")
    report.append("## Step당 시간\n\n| 설정 | λ | 학습 루프 평균 s/optimizer-step |\n|---|---:|---:|\n")
    for r in rows:
        report.append(f"| {r['label']} | {r['rate_lambda']:g} | {r['seconds_per_optimizer_step']:.3f} |\n")
    report.append("\n## 근거\n\n[계산·로그 SHA](timing.json), [재계산 코드](analyze.py). 로그는 workspace 루트 `slurm/`의 다운로드본이다. 서버 조회·제출·취소·GPU 실험을 수행하지 않았다.\n\n")
    for r in rows+aux:
        report.append(f"- `{r['source_log']}` — SHA256 `{r['source_sha256']}`\n")
    (OUT/"REPORT.md").write_text("".join(report),encoding="utf-8")
    print(json.dumps(dict(main_hours=[(r['label'],r['rate_lambda'],round(r['progress_elapsed_hours'],3)) for r in rows],comparisons=comparisons,auxiliary=[dict(job=r['job'],hours=r['progress_elapsed_hours'],steps=r['optimizer_steps_in_this_run'],equivalent_50k=r['equivalent_50k_hours']) for r in aux])))


if __name__ == "__main__":
    main()

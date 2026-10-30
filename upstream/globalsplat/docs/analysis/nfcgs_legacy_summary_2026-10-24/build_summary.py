"""Consolidate pre-fresh experiments; preserve historical snapshots verbatim."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
WORKSPACE = REPO.parents[1]
DOCS = REPO/"docs"
MASTER = DOCS/"NFCGS_EXPERIMENTS_MASTER_2026-10-20.md"
TARGET = DOCS/"NFCGS_PREVIOUS_EXPERIMENTS_ALL_2026-10-24.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return path.read_text(encoding="utf-8")


def section(text, name):
    start = text.index(f'<a id="{name}"></a>')
    end = text.find('\n<a id="', start+1)
    return text[start:] if end < 0 else text[start:end]


def check_point(row, scenes):
    path = WORKSPACE/row["source_log"]
    assert digest(path) == row["source_sha256"], path
    text = read(path)
    actual = {k:float(v) for k,v in re.findall(r"(?m)^([a-z_]+): ([-+\deE.]+)\s*$", text)}
    for field in ("psnr","ssim","lpips","actual_bytes"):
        assert abs(actual[field]-row["metrics"][field]) < 1e-6, (path,field)
    progress = re.findall(r"Testing DataLoader 0:.*? (\d+)/(\d+) \[", text)
    assert int(progress[-1][0]) == scenes, path


def main():
    old = read(MASTER)
    old_sha = digest(MASTER)
    inventory_path = DOCS/"analysis/nfcgs_logs_2026-10-20_2159/inventory.json"
    inventory = json.loads(read(inventory_path))
    points = copy.deepcopy(inventory["results"])
    assert len(points) == 64  # Its old summary scalar still says 63; count actual rows.
    source = WORKSPACE/"slurm/slurm-gs-order-context-eval-457100.out"
    text = read(source)
    metrics = {k:float(v) for k,v in re.findall(r"(?m)^([a-z_]+): ([-+\deE.]+)\s*$", text)}
    checkpoint = re.search(r"checkpointing.load=([^\s]+)", text).group(1)
    points.append(dict(label="hilbert_dyadic4_off16_0p0128",family="hilbert_dyadic4_off16",rate_lambda=.0128,
        source_log=source.relative_to(WORKSPACE).as_posix(),source_sha256=digest(source),
        metrics=metrics,checkpoint=checkpoint,evaluated_scenes=6991,train_job="457099",
        resume_job="458783",eval_job="457100"))
    all_current = points+inventory["lambda_zero_points"]+inventory["seed_repeat_points"]+inventory["architecture_pilot_points"]
    for row in all_current:
        check_point(row, 128 if row in inventory["architecture_pilot_points"] else 6991)
    families = Counter(row["family"] for row in points)
    assert len(points) == 65 and len(families) == 17 and len(all_current) == 72
    assert len({(p["family"],p["rate_lambda"]) for p in points}) == 65

    labels = {}
    for line in section(old,"rd").splitlines():
        if line.startswith("| "):
            cells = [v.strip() for v in line.strip("|").split("|")]
            if cells[0] in families:
                labels[cells[0]] = cells[1]
    labels["hilbert_dyadic4_off16"] = "Hilbert + D4 OFF16"
    assert set(labels) == set(families)

    parts = ["""# GlobalSplat / NFC-GS 이전 실험 전체 종합 정리

통합일: **2026-10-24 (Asia/Seoul)**. 기존 완료 실험의 근거는10월20~21일 다운로드 로그와 보존된 과거 결과다. **이번에 진행 중인 새 fresh-init RD 설정과 그 Stage A/B·120k 연장·monitor 복구는 집계 대상에서 제외한다.** 이전의50k 구조 pilot4개는 종전 실험에 포함한다.

초기 rank·projection 탐색, linear/nonlinear·residual·context 학습, 정렬·entropy schedule RD, 대조군과 추가 seed, 구조 pilot, 경로·정렬·수신 진단, 복잡도·시간 측정, 실패·재개 이력을 이 문서 하나에 모았다. 원 종합본에서 미완료로 남아 있던 None+L3 OFF12 .0512, H+D4 OFF16 .0128, Score736-only의 완료 결과를 반영했다. 기존 snapshot 문서는 수정하지 않았다.

1. [완료 현황과 집계 범위](#status)
2. [공통 모델·평가 조건과 명칭](#protocol)
3. [전체 실험 계보](#lineage)
4. [기존26개 학습·평가 목록](#recent)
5. [주 seed RD 전체65점·17계열](#rd)
6. [λ0·두 seed D4 ON/OFF·구조 pilot](#special)
7. [비압축·경로·정렬·독립 수신 진단](#diagnostics)
8. [복잡도·시간·메모리](#complexity)
9. [초기 실험·과거 context48개](#legacy)
10. [기타 평가와 CleanToken 별도 분기](#other)
11. [실패·재개 이력](#failures)
12. [기존 후속 계획과 결과 미확인 항목](#pending)
13. [과거 Slurm 로그 색인](#ledger)
14. [출처와 검증 범위](#sources)

<a id="status"></a>

## 1. 완료 현황과 집계 범위

기존26개 묶음은 **26개 모두 최종50k 평가까지 완료**됐다. 이 묶음의 완료 평가는 **전체 test6991장22개 + train-derived holdout128장4개**다. 실패한 학습을 복구한 job은 같은 모델의 이력으로 기록하며 독립 실험으로 더하지 않는다.

| 범위 | 완료 기록 | 평가 집합과 해석 |
|---|---:|---|
| 정렬·schedule·context 주 seed, 양의λ | **65점·17계열** | 전체 test6991; H+D4 OFF16의 한 점을 별도 계열로 포함 |
| λ=0 |1점| 실제 양자화·entropy coding 유지; 비압축 기준 아님 |
| H+D4 추가 seed |2점| ON/OFF 각1점, λ=.0128; 같은 warm-start 부모 공유 |
| 구조 pilot |4점| 같은 train-derived holdout128; 전체 test와 분리 |
| 위 네 범위 합계 |72개 모델별 완료 평가 기록| 평가 집합·계보가 달라 단일 RD 곡선으로 합치지 않음 |
| 기존26개 제출 묶음 |26개 완료| 위72개와 겹치는 부분집합 |
| 복잡도 profile |37 checkpoint| 별도 측정; 학습37개를 추가했다는 뜻 아님 |
| 과거 context 보존 JSON |48개 평가 기록|9.3절에 전부 수록; 현재 원본 stdout가 없어 보존 자료 기준 |
| 초기 Linear 사용자 표 |12개 조건|9.2절; 원래 kB 단위를 유지 |
| 기타 현존 평가 |37개 기록|10절; subset·다른 계보·중복 checkpoint 포함 |

집계 범위끼리 겹치므로 위 숫자를 더해 독립 학습의 총수로 사용하지 않는다. 초기 탐색·분석 전용 작업·실패한 시도까지 아래 각 절에 포함하되 완료 모델과 구분한다.

"""]
    parts.append(section(old,"protocol"))
    parts.append("""<a id="lineage"></a>

## 3. 전체 실험 계보

```text
초기 rank / projection / residual / token 순서 탐색
  └─ Linear / Nonlinear factorized + context 추가학습
       ├─ P0 / P1 / P2 predictor
       ├─ 복원 고정 Shared / Split / Gaussian / Conditional probability-only
       ├─ low-LR 후속 / 고정-symbol entropy probe
       └─ Full P0+Split 부모에서 weights-only 후속50k
            └─ None / Morton / Hilbert / NN-score
                 × L2 / L3 / legacy4 / Q2 / D4 / spatial OFF
                 ├─ 주 seed 양의λ65점, λ0 한 점
                 ├─ stage/group을 맞춘 OFF12 / OFF16
                 ├─ D4 ON/OFF 추가 seed
                 └─ anchor / B0 / T05 / T06 / T15 / complexity 진단

공개 backbone + 공통 QR 경계 + 새 codec 초기화
  └─ 기존50k 구조 pilot4개, holdout128
       Score56-only / MSH-only / two-path56 / Score736-only

별도 CleanToken repository
  └─ Hyper1D single / dual / lowrank-base / Morton / λ 실험
```

앞의 warm-start RD와 기존 구조 pilot은 초기화와 평가 집합이 다르다. 추가 seed도 같은 부모 가중치를 공유하므로 독립 fresh-init 반복으로 해석하지 않는다. CleanToken 분기는 구현·압축 경계·계보가 달라 별도 표로 유지한다.

""")
    recent = section(old,"recent").replace("## 4. 최근 제출한 26개 전체 목록","## 4. 기존26개 학습·평가 전체 목록")
    recent = recent.replace("이 표는 현재 진행 추적용이다.","이 표는 기존26개의 최종 완료 기록이다.")
    replacements = {
        "학습 진행 44,762/50k; eval 시작 로그 없음":"완료 / test6991",
        "학습 진행 33,823/50k; eval 시작 로그 없음":"완료 / test6991",
        "학습 진행 41,294/50k; eval 시작 로그 없음":"완료 / holdout128",
    }
    for before,after in replacements.items():
        assert recent.count(before) == 1
        recent = recent.replace(before,after)
    parts.append(recent)

    rd = ["""<a id="rd"></a>

## 5. 주 seed RD 전체65점·17계열

양의λ는 .0032/.0064/.0128/.0256/.0512다. None/Hilbert×L3 OFF/ON 네 곡선은 각각5점으로 총20점이다. H+D4 OFF16은 λ=.0128의 한 점이므로 ON과의 단일 점 대조로만 사용한다. λ0와 추가 seed는6절에 분리한다.

| Family | 설정 | 완료λ | 점 수 |
|---|---|---|---:|
"""]
    for family in sorted(families):
        rates = sorted(p["rate_lambda"] for p in points if p["family"] == family)
        rd.append(f"| {family} | {labels[family]} | {', '.join(f'{r:g}' for r in rates)} | {len(rates)} |\n")
    rd.append("\n| 설정 | λ | PSNR | SSIM | LPIPS | KiB/scene | Checkpoint run | Eval |\n|---|---|---|---|---|---|---|---|\n")
    for p in sorted(points,key=lambda p:(p["family"],p["rate_lambda"])):
        m = p["metrics"]
        match = re.search(r"/(recovery_|resume_)?job_(\d+)/",p["checkpoint"])
        train = match.group(2)
        if match.group(1):
            train += " ("+match.group(1).rstrip("_")+")"
        evaluation = p.get("eval_job") or re.search(r"eval-([^/]+)\.out$",p["source_log"]).group(1)
        if p.get("resume_job"):
            train += " → "+p["resume_job"]
        rd.append(f"| {labels[p['family']]} | {p['rate_lambda']:g} | {m['psnr']:.4f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['actual_bytes']/1024:.5f} | {train} | {evaluation} |\n")
    rd.append("""
### L3 네 곡선의 공통 품질 구간 비교

공통 PSNR23.0360–23.7288dB에서 각5점의 PCHIP log-rate를 적분했고 외삽하지 않았다.

| 후보 / 기준 | BD-rate |
|---|---:|
| None ON / None OFF |−1.2648%|
| Hilbert ON / Hilbert OFF |−20.4325%|
| Hilbert OFF / None OFF |−14.7991%|
| Hilbert ON / None ON |−31.3393%|
| Hilbert ON / None OFF |−32.2077%|

이 값은 반올림된 집계 지표와 주 seed 한 개의 RD 보간에 근거한다. Scene별 paired CI나 training-seed 유의성은 확인하지 않았다. Hilbert+L3 .0256에는 GPU4 학습 예외도 있어 완전히 맞춘 학습 조건의 순수 인과 효과로 표현하지 않는다. [계산 근거](analysis/nfcgs_logs_2026-10-20_2159/REPORT.md).

""")
    parts.append("".join(rd))
    special = ["""<a id="special"></a>

## 6. λ0·두 seed D4 ON/OFF·구조 pilot

### 6.1 λ=0

| 설정 | Train / Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---:|---:|---:|---:|
| H+D4 λ0 |455327 /455328|24.0734|0.7493|0.2512|831.64219|

실제 양자화·entropy coding이 있는 모델이다. 비압축 또는 native 화질 상한으로 사용하지 않는다. H+D4 .0032 대비 PSNR+0.1769dB, bytes 약14.69배이며 용량 증가의 큰 부분은 MSH y다.

### 6.2 H+D4 ON16/OFF16 두 seed의 완료 비교

모두 λ=.0128, 전체 test6991장이다. OFF16은 공간 예측을 끄되 LR/MSH와 stage/group stream 구성을 유지한다.

| Seed(model/data) | Context | Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---|---:|---:|---:|---:|
|111123/403|ON16|447899|23.7021|0.7363|0.2673|37.178812|
|111123/403|OFF16|457100|23.6338|0.7338|0.2695|44.619252|
|111124/404|ON16|457787|23.7036|0.7362|0.2669|37.075100|
|111124/404|OFF16|457789|23.6310|0.7336|0.2692|44.768936|

| Seed | ON의 전송량 감소율 | ON−OFF PSNR |
|---|---:|---:|
|기본|16.6754%|+0.0683dB|
|추가|17.1857%|+0.0726dB|

두 seed 모두 같은λ에서 ON이 더 작은 actual bytes와 더 높은 PSNR을 보였다. 같은 warm-start 부모를 공유하는 반복이며 동일 품질 BD-rate 또는 독립 fresh-init의 seed 유의성 검정은 아니다. [완료 근거](analysis/nfcgs_d4_context_2026-10-21/REPORT.md).

### 6.3 기존50k 구조 pilot 네 arm

Hilbert·λ=.0128·model111123/data403, 공개 backbone/QR 경계에서 공통 codec 초기화 후50k를 학습했다. 같은 train-derived holdout128을 학습에서 제외했으며 일반 test를 읽는 validation loader는 학습 중 끈다. Warm-start 주 RD의 전체 test 결과와 분리한다.

| Arm | 활성 경로 | Train / Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---|---:|---:|---:|---:|
"""]
    arm_names = dict(score56_only="rank56 Score만",msh_only="F736 직접 MSH",two_path56="rank56+MSH residual",score736_only="736→736 Score만")
    for p in inventory["architecture_pilot_points"]:
        m = p["metrics"]
        special.append(f"| {p['arm']} | {arm_names[p['arm']]} | {p['train_job']} / {p['eval_job']} | {m['psnr']:.4f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['actual_bytes']/1024:.5f} |\n")
    special.append("""
Score736-only는 Score56-only보다 PSNR−0.0888dB·SSIM−0.0050·LPIPS+0.0020이고 bytes도3.4906% 많아 이번 한 점에서는 지배된다. Two-path56은 Score736-only보다 PSNR+1.3066dB이고 bytes16.3424% 적다. MSH-only는14.48KiB의 저율 점이므로 같은λ의 품질 차이만으로 제외하지 않는다.

Score736은 channel group184×4와 큰 predictor를 사용하므로 rank만의 순수 대조군은 아니다. Backbone/Gaussian decoder는 고정하고 codec과 geometry projection을 학습했다. 네 구조의 전체 test와 추가λ는 이 정리의 완료 결과에 없다. [구조 비교 근거](analysis/nfcgs_logs_2026-10-20_2159/REPORT.md).

""")
    parts.append("".join(special))
    for name in ("diagnostics","complexity","legacy","other"):
        parts.append(section(old,name))
    failures = section(old,"failures")
    failures = failures.replace("458782/458783으로 full-state 복구 진행","458782/458783으로 full-state 복구 후 최종50k 평가 완료")
    failures = failures.replace("현재 학습3개에 포함하지 않음","현재 진행 작업으로 세지 않음")
    parts.append(failures)
    parts.append("""<a id="pending"></a>

## 12. 기존 후속 계획과 결과 미확인 항목

아래는 과거에 제안·구현한 항목의 상태 기록이다. 이번 새 fresh-init RD 계획은 싣지 않는다. 이미 완료된457098·457100·457829는 이 대기 목록에서 제거했다.

| 항목 | 확인 가능한 범위 |
|---|---|
| None+D4 OFF16·N01·native 전체 test·T06 렌더 후속 묶음 |10월20일 실행 도구 구현; 현재 로컬에 `slurm-gs-followup20-*` 로그가 없어 접수·완료를 확정하지 않음 |
| 구조 pilot 전체6991장 test |네 arm 모두128장 평가 완료; 전체 test 완료 기록 없음 |
| 구조별 추가λ·bitrate 맞춤 |MSH-only .0032·two-path56 .0512 우선 탐색을 제안했으나 이 과거 묶음의 완료 결과 없음 |
| Morton L3 OFF12 .0128(T17) |과거 계획 deferred; 완료 결과 없음 |
| Hilbert L3 ON/OFF12 추가 seed |과거 계획 deferred; D4 seed 비교로 대체했다고 간주하지 않음 |
| B0/T05 전체6991장 확장 |기존 matched256 결과만 완료 |
| OFF/주요 구조 complexity 보완 |기존37-checkpoint profile 이후 추가 완료 자료 없음 |

결과 미확인은 서버에서 미접수라는 뜻이 아니다. 이 정리 작업에서는 기존 job을 재제출하거나 취소하지 않았다.

""")
    ledger = section(old,"ledger").replace("## 13. 내려받은 Slurm 로그 파일의 전체 job 목록","## 13. 과거 Slurm 로그 색인 — 10월20일16:47 보존본")
    ledger = ledger.replace("## 13. 과거 Slurm 로그 색인 — 10월20일16:47 보존본\n", "## 13. 과거 Slurm 로그 색인 — 10월20일16:47 보존본\n\n아래284개 stdout·49개 prefix의 목록은 원 snapshot의 이력이다. 현재 로컬 전체 파일 수가 아니며, 이번 새 fresh-init job은 추가하지 않았다. 이후 완료 평가457098·457100·457829는 앞의 결과 표에 반영했다.\n")
    parts.append(ledger)
    parts.append("""<a id="sources"></a>

## 14. 출처와 검증 범위

- [기존 전체 종합본](NFCGS_EXPERIMENTS_MASTER_2026-10-20.md): 초기 탐색·과거48개·기타37개·진단·profile·실패·로그 색인의 원본. 원 snapshot을 보존했다.
- [10월20일21:59 완료 검토](analysis/nfcgs_logs_2026-10-20_2159/REPORT.md), [원 inventory](analysis/nfcgs_logs_2026-10-20_2159/inventory.json): L3 마지막 점과 구조4개. 원 JSON의 `completed_positive_rate_points`는63으로 남아 있지만 실제 `results`는64개다. 이 문서는 실제 행 수를 세어64개를 읽고 최종 H+D4 OFF16 한 점을 추가했다.
- [H+D4 두 seed 최종 비교](analysis/nfcgs_d4_context_2026-10-21/REPORT.md): 기본 OFF 완료와 네 평가 원본 SHA.
- [통합 데이터·검증 결과](analysis/nfcgs_legacy_summary_2026-10-24/summary.json), [생성·검산 코드](analysis/nfcgs_legacy_summary_2026-10-24/build_summary.py).
- [기존 통합 기계 판독 자료](analysis/nfcgs_experiment_master_2026-10-20/all_experiments.json), [과거 context48개 지표](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json).
- [T06 수치 원문](analysis/nfcgs_logs_2026-10-20_1636/provided_t06_report.md), [복잡도37개](analysis/nfcgs_complexity_455251/REPORT.md), [초기 Linear 표](NFCGS_EXPERIMENT_RESULTS_2026-09-10.md).

이번 통합에서 주 RD65개·λ0 한 개·추가 seed2개·구조4개, 총72개 완료 평가의 stdout SHA·PSNR·SSIM·LPIPS·actual bytes·처리 장면 수를 다시 대조했다. 검증 범위는 위72개 평가 기록이며, 독립 학습의 전체 개수를 뜻하지 않는다. 과거48개 보존 자료·초기 사용자 표·기타37개와 진단/profile의 수치는 기존 종합본에서 범위와 한계를 유지해 옮겼다. 그 모든 원본을 이번에 새로 재검증한 것은 아니다.

서버의 실시간 상태나 raw checkpoint를 확인한 문서가 아니다. 다른 초기화·checkpoint 계보·장면 집합·seed·평가 프로토콜의 수치를 한 곡선에 섞지 않는다. 집계 지표 검증을 장면별 frame/tensor bit-exact 검증이나 training-seed 통계 검정으로 확대하지 않는다. 이 작업에서는 학습 코드·서버 job·dependency를 변경하지 않았다.
""")
    document = "".join(parts)
    assert "학습 진행 44,762" not in document and "학습 진행 33,823" not in document
    assert "학습 진행 41,294" not in document
    assert len(re.findall(r"^\| (?:rd_extensions|controls|architecture|seed_pair) \|",document,re.M)) == 26
    for name in ("complexity","legacy","other"):
        assert section(old,name) in document  # Preserve detailed historical numbers.
    for link in re.findall(r"\]\(([^)]+)\)",document):
        if link.startswith(("http:","https:","#")):
            continue
        local = (TARGET.parent/link.split("#",1)[0]).resolve()
        assert local.exists() or local == OUT/"summary.json", (link,local)
    TARGET.write_text(document,encoding="utf-8")
    assert digest(MASTER) == old_sha
    result = dict(compiled_date="2026-10-24 Asia/Seoul",scope="pre-current-fresh experiments only",
        document=TARGET.relative_to(REPO).as_posix(),source_master_sha256=old_sha,
        source_master_unchanged=True,positive_rate_points=len(points),families=len(families),
        lambda_zero_points=inventory["lambda_zero_points"],seed_repeat_points=inventory["seed_repeat_points"],
        architecture_pilot_points=inventory["architecture_pilot_points"],results=points,
        verified_completed_evaluation_sources=len(all_current),recent_batch_completed=26,
        recent_batch_full_test=22,recent_batch_holdout=4,new_setting_excluded=True,
        source_inventory_sha256=digest(inventory_path),document_sha256=digest(TARGET))
    (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(document=result["document"],positive_rate_points=65,families=17,
        verified_completed_evaluation_sources=72,recent_batch_completed=26,lines=len(document.splitlines())),ensure_ascii=True))


if __name__ == "__main__":
    main()

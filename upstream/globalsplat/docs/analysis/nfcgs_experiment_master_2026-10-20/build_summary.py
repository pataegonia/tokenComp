"""Build a comprehensive Markdown ledger from audited snapshots and old logs."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shlex

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
WORKSPACE=REPO.parents[1]
LOGS=WORKSPACE/"slurm"
OUTPUT=REPO/"docs/NFCGS_EXPERIMENTS_MASTER_2026-10-20.md"
SOURCES={}


def read(path):
    path=Path(path)
    raw=path.read_bytes()
    root=REPO if path.is_relative_to(REPO) else WORKSPACE
    SOURCES[path.relative_to(root).as_posix()]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return raw.decode("utf-8",errors="replace").replace("\r\n","\n")


def load(relative):return json.loads(read(REPO/relative))


def table(headers,rows):
    def clean(v):return str(v).replace("|","\\|").replace("\n"," ")
    return "\n".join(["| "+" | ".join(map(clean,headers))+" |",
        "|"+"|".join("---" for _ in headers)+"|",
        *("| "+" | ".join(map(clean,row))+" |" for row in rows)])+"\n"


def md_tables(text):
    return re.findall(r"(?m)^\|[^\n]+\n\|[-| :]+\n(?:\|[^\n]*\n)+",text)


def command(text):
    line=next((line for line in text.splitlines() if " -m globalsplat.main " in line),"")
    return {key.lstrip("+"):value for token in shlex.split(line) if "=" in token
            for key,value in [token.split("=",1)]}


def job_from_log(name):
    m=re.search(r"-(\d+(?:_\d+)?)\.out$",name)
    return m[1] if m else name.removeprefix("slurm-gs-").removesuffix(".out")


def run_from_checkpoint(path):
    matches=re.findall(r"/(job|resume_job|recovery_job)_(\d+(?:_\d+)?)/",path or "")
    if not matches:return "경로 참조"
    kind,job=matches[-1]
    return job+({"job":"","resume_job":" (resume)","recovery_job":" (recovery)"}[kind])


FAMILIES={
    "both_off":"None + L3 OFF12", "context_dyadic4":"None + D4 ON16",
    "context_only":"None + L3 ON12", "context_quarter2":"None + Q2 ON8",
    "hilbert_dyadic4":"Hilbert + D4 ON16", "hilbert_legacy3":"Hilbert + L3 ON12",
    "hilbert_off1":"Hilbert + OFF1", "hilbert_off4":"Hilbert + OFF4",
    "hilbert_only":"Hilbert + L3 OFF12", "hilbert_quarter2":"Hilbert + Q2 ON8",
    "morton_dyadic4":"Morton + D4 ON16", "morton_legacy2":"Morton + L2 ON8",
    "morton_legacy3":"Morton + L3 ON12", "morton_legacy4":"Morton + legacy4 ON16",
    "morton_quarter2":"Morton + Q2 ON8", "nn_score":"NN-score + L3 ON12",
}
LEGACY={
    "linear_factorized_continuation50k":"Linear factorized +50k",
    "nonlinear_transform_initial50k":"Nonlinear32 초기50k",
    "nonlinear_factorized_continuation50k":"Nonlinear factorized +50k",
    "nonlinear_score_context50k":"Nonlinear Spatial/Full +50k",
    "linear_score_context50k":"Linear mean/channel/spatial +50k",
    "score_probability10k":"복원 고정 probability-only +10k",
    "spatial_predictor25k":"P0/P1/P2 +25k",
    "full_lowlr10k":"Full low-LR +10k",
}


def case_name(row):
    if row["group"]=="architecture":return row["arm"]
    if row["group"]=="seed_pair":return "H+D4 "+row["arm"]+" 추가 seed"
    order={"none":"None","hilbert":"H","morton":"M"}[row["order"]]
    schedule={"legacy":f"L{row['stages']}","dyadic4":"D4","single":"OFF1"}[row["schedule"]]
    context="ON" if row["spatial_context"] else "OFF"
    if schedule=="OFF1":return f"{order}+OFF1"
    strings=4*row["stages"]
    return f"{order}+{schedule} {context}{strings}"


def extra_label(path,text,cfg):
    if "CleanToken/" in text:
        variant=re.search(r"variant=(\S+)",text)
        variant=variant[1] if variant else cfg.get("model.feature_codec.architecture","hyper1d")
        parts=[variant]
        for key,short in (("model.feature_codec.paths","paths"),("model.feature_codec.base_rank","base_rank"),
                          ("model.feature_codec.use_morton","Morton")):
            if key in cfg:parts.append(f"{short}={cfg[key]}")
        return "; ".join(parts),"CleanToken 별도 분기"
    labels={
        "slurm-gs-main-eval-426725.out":"Full nonlinear Split joint checkpoint220k",
        "slurm-gs-joint500k-eval-428944.out":"Full nonlinear Split joint500k",
        "slurm-gs-main-eval-428943.out":"기존 Split checkpoint 32장 재평가",
        "slurm-gs-score-pred-429783.out":"같은 Split checkpoint score 진단32장",
        "slurm-gs-score-minimal-eval-EEE441558.out":"초기 minimal L2/kernel3/noise",
        "slurm-gs-nonlinear50k-100-436236.out":"Nonlinear32 기존50k test100",
    }
    return labels.get(path.name,path.name),"tokencomp 과거 실행"


def main():
    inventory=load("docs/analysis/nfcgs_logs_2026-10-20_1647/inventory.json")
    review=load("docs/analysis/nfcgs_logs_2026-10-20_1647/review.json")
    legacy=load("docs/NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json")
    t05=load("docs/analysis/nfcgs_logs_2026-10-20_0911/provided_results.json")
    t06=load("docs/analysis/nfcgs_logs_2026-10-20_1636/provided_t06_results.json")
    early=read(REPO/"docs/ABLATION_PROGRESS_2026-09-02.md")
    original_linear=read(REPO/"docs/NFCGS_EXPERIMENT_RESULTS_2026-09-10.md")
    complexity=read(REPO/"docs/analysis/nfcgs_complexity_455251/REPORT.md")
    primary=inventory["results"]
    zero=inventory["lambda_zero_points"]
    seeds=inventory["seed_repeat_points"]
    pilots=[row for row in review["experiments"] if row["group"]=="architecture"]
    done_pilots=[row for row in pilots if row["eval_status"]=="complete_holdout128"]
    assert len(primary)==63 and len(zero)==1 and len(seeds)==2 and len(done_pilots)==3
    assert len({(row["family"],row["rate_lambda"]) for row in primary})==63
    assert len(legacy["evaluations"])==48
    known={Path(row["source_log"]).name:row for row in primary+zero+seeds}
    known.update({Path(row["eval_source_log"]).name:dict(metrics=row["metrics"]) for row in done_pilots})
    raw_records=[]
    for path in sorted(LOGS.glob("*.out")):
        if path.stat().st_size>2_500_000:continue
        text=path.read_text(encoding="utf-8",errors="replace")
        metrics={key:float(value) for key,value in re.findall(r"(?m)^(psnr|ssim|lpips|actual_bytes): ([-+\deE.]+)$",text)}
        if set(metrics)!={"psnr","ssim","lpips","actual_bytes"}:continue
        read(path)
        progress=re.findall(r"Testing DataLoader 0:.*? (\d+)/(\d+) \[",text)
        cfg=command(text)
        label,branch=extra_label(path,text,cfg)
        step_match=re.search(r"global_step=(\d+)",text)
        checkpoint=cfg.get("checkpointing.load","")
        filename_step=re.search(r"step(\d+)\.ckpt",checkpoint)
        stderr=path.with_suffix(".err")
        error=read(stderr) if stderr.exists() else ""
        fatal=[line for line in error.splitlines() if re.search(r"Traceback|(?:RuntimeError|ValueError|OSError):|No space left|CUDA out of memory",line)]
        row=dict(log=path.name,job=job_from_log(path.name),label=label,branch=branch,
            metrics=metrics,scenes=int(progress[-1][0]) if progress else None,
            rate_lambda=cfg.get("loss.rate_lambda","미기록"),checkpoint=checkpoint,
            step=int(step_match[1]) if step_match else int(filename_step[1]) if filename_step else None,
            step_evidence="saved metadata stdout" if step_match else "checkpoint filename" if filename_step else "unknown",
            command_overrides=cfg,fatal_matches=fatal,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if path.name in known:
            expected=known[path.name]["metrics"]
            assert all(abs(metrics[key]-expected[key])<1e-6 for key in metrics),path.name
        raw_records.append(row)
    # This older subset evaluator prints a JSON object instead of metric lines.
    path=LOGS/"slurm-gs-nonlinear50k-100-436236.out"
    text=read(path)
    data=json.loads(text[text.rfind("\n{")+1:])
    assert data["scene_count"]==100 and data["global_step"]==50000
    raw_records.append(dict(log=path.name,job="436236",label="Nonlinear32 기존50k test100",branch="tokencomp 과거 실행",
        metrics={key:data[key] for key in ("psnr","ssim","lpips","actual_bytes")},scenes=100,
        rate_lambda="0.0064",checkpoint=data["checkpoint"],step=50000,step_evidence="summary stdout",
        command_overrides={},fatal_matches=[],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(raw_records)==106 and len({row["log"] for row in raw_records})==106
    assert set(known)<={row["log"] for row in raw_records}
    extra=[row for row in raw_records if row["log"] not in known]
    assert len(extra)==37
    post_eval_errors=[row for row in extra if row["fatal_matches"]]
    assert {row["log"] for row in post_eval_errors}=={
        "slurm-gs-msh-fulltest-440685_0.out","slurm-gs-msh-fulltest-440685_1.out"}
    assert all(row["scenes"]==6991 and all("cannot create temp file for here-document: No space left on device" in line
        for line in row["fatal_matches"]) for row in post_eval_errors)
    assert all(row.get("evaluated_scenes",row.get("scenes"))==6991 for row in primary+zero+seeds)
    legacy_present=sum((LOGS/Path(row["log"]).name).exists() for row in legacy["evaluations"])
    assert legacy_present==0
    ledger=defaultdict(list)
    for path in sorted(LOGS.glob("*.out")):
        m=re.match(r"(.+?)-([A-Z]*\d+(?:_\d+)?)\.out$",path.name)
        ledger[m[1] if m else path.name].append(m[2] if m else "")
    assert sum(map(len,ledger.values()))==284
    content=[]
    def add(text):content.append(text.strip()+"\n")
    def section(anchor,title):add(f'<a id="{anchor}"></a>\n\n## {title}')

    add("""# GlobalSplat / NFC-GS 전체 실험 정리

작성일: **2026-10-20 (Asia/Seoul)**. 현재 상태는 **2026-10-20 16:47 다운로드 로그**를 기준으로 한다. 사용자가 전달한 T06 REPORT 수치도 반영했다. 실시간 `squeue/sacct` 조회 결과는 아니다.

완료한 학습·평가, 현재 남은 작업, 분석 전용 실험, 과거 실패·재개 기록과 별도 분기 결과를 한곳에 모았다. 완료 평가는 수치를 싣고, 현재 결과가 없는 작업은 상태와 job ID만 기록한다. 계획만 있는 실험은 수행한 목록과 구분한다.

### 목차

- [1. 현재 현황과 남은 3개 학습](#status)
- [2. 공통 모델·평가 조건과 명칭](#protocol)
- [3. 실험 계보와 집계 범위](#lineage)
- [4. 최근 제출한 26개 전체 목록](#recent)
- [5. 현행 주 seed RD: 16계열·63점](#rd)
- [6. λ=0·추가 seed·구조 pilot](#special)
- [7. 비압축·경로·정렬·독립 수신 진단](#diagnostics)
- [8. 복잡도·시간·메모리 측정](#complexity)
- [9. 9월 이전/초기 실험 전체 기록](#legacy)
- [10. 그 밖의 평가 수치와 CleanToken 별도 분기](#other)
- [11. 실패·중단·완료 여부 미확인 이력](#failures)
- [12. 아직 실행하지 않은 후속 작업](#pending)
- [13. 내려받은 Slurm 로그 파일의 전체 job 목록](#ledger)
- [14. 근거 파일과 검증 범위](#sources)
""")
    section("status","1. 현재 현황과 남은 3개 학습")
    add("""최근 26개 원래 실험 중 **23개는 목표 50k 학습 완료**다. 평가까지 끝난 것은 **전체 test 20개 + holdout128 3개**다. 남은 학습은 다음 세 개이며 연결 평가는 아직 시작 로그가 없다.

아래 ‘학습 진행’은 마지막 완전한 progress가 있다는 뜻이다. 현재 시각의 Slurm RUNNING을 직접 확인했다는 뜻은 아니다. Step은 progress 추정이며 실제 최신 저장 checkpoint step과 구분한다.""")
    remaining=[row for row in review["experiments"] if row["train_status"]!="complete_50k"]
    add(table(["실험","λ","원래 train","현재 학습 job","진행 / 목표","진행률","연결 eval"],[
        [case_name(row),row["lambda"],row["train_job"],row.get("resume_job") or row["train_job"],
         f"{row['estimated_optimizer_step']:,} / 50,000",f"{row['estimated_training_percent']:.3f}%",row["eval_job"]]
        for row in remaining]))
    add("""복구는 `456951 → 458782`, `457099 → 458783`이다. 각각 실제 저장된 **25,000 / 15,000 step** checkpoint에서 optimizer·scheduler·loop 상태를 복원했고, 원래 출력 디렉터리에서 총 50k까지 이어 간다. 기존 eval `457098 / 457100`의 dependency도 새 학습의 `afterok`로 연결했다. 새 평가를 중복 생성하지 않았다.

기본 loader는 재개 시 fresh shuffle을 사용한다. 전체 학습 상태 복원과 중단 없는 실행의 다음 sample 순서까지 일치하는 재현은 구분한다.

현재 판단: **구조 선택에 필요한 미완료 pilot은 Score736-only 하나**다. MSH-only는 저용량 후보, two-path56은 현재 완료한 pilot 중 주력 후보로 유지한다. 기본 seed의 OFF16이 끝나야 추가 seed와 함께 D4 ON/OFF 두 seed 비교가 완성된다.""")
    section("protocol","2. 공통 모델·평가 조건과 명칭")
    add(table(["항목","현행 조건"],[
        ["기반 모델","공개 GlobalSplat RE10K-32K; 4,096 scene tokens"],
        ["압축 경계","appearance512 + raw geometry512→observable geometry224 = F736"],
        ["기본 codec","rank56 LR/Score 복원 + MSH residual; MSH N192/M320/adapter96"],
        ["현행 minimal","centering·Score norm·mean conditioning·channel context OFF, linear transform; 공간 Context는 별도 스위치"],
        ["Gaussian","stage3, token당8개 = scene당32,768개; M_max16과 구분"],
        ["전체 test","RE10K C12/T8, 256×256, BF16, eval seed0, 실제 처리6,991장"],
        ["실제 용량","Score + MSH y/z + mean + 모든 container/header; 평균 B/scene ÷1,024 = KiB/scene"],
        ["공유 모델","checkpoint/신경망 weight는 scene payload에서 제외"],
        ["평가 중 최적화","현행 평가에서 per-scene fitting 또는 test-time 학습 없음"],
        ["후속50k의 일반 설정","GPU2 × batch/GPU2 × accumulation2 = 유효 batch8, LR1e-4, 35k/45k decay"],
        ["학습 초기화","기존 codec 대조군은 부모 weights warm start; 구조 pilot은 공개 backbone/QR 경계에서 codec fresh init"],
        ["학습되는 경계","codec 학습에는 geometry projection이 포함됨; backbone/Gaussian decoder는 고정"],
    ]))
    add("""전체 test loader의 표기상 분모는 7,286일 수 있지만 실제 유효 처리 수는 6,991이다. 시간 측정의 warmup5를 뺀 6,986회는 평가 장면 수가 아니다. 구조 pilot의 holdout128, 진단256, profile32, 옛 subset31/32/97/100 결과를 전체 test와 섞지 않는다.

`context_only`는 **기본 token 순서(None)+공간 Context ON인 two-path 모델**이다. Score-only를 뜻하지 않는다. `hilbert_only`는 **Hilbert+L3 OFF12**이며 MSH를 유지한다. 공간 Context OFF는 LR 또는 MSH 제거와 다르다.""")
    add(table(["표기","Token 단계","채널 그룹","Score strings","공간 예측"],[
        ["OFF1","전체4096 한 단계","전체56","1","OFF"],
        ["OFF4","D4 4단계","단계마다 전체56","4","OFF"],
        ["L3 OFF12","legacy 3단계","16/16/16/8","12","OFF"],
        ["D4 OFF16","D4 4단계","16/16/16/8","16","OFF"],
        ["L2 ON8 / Q2 ON8","각2단계; 참조 topology는 다름","16/16/16/8","8","ON"],
        ["L3 ON12","0mod4 → 2mod4 → 홀수","16/16/16/8","12","ON"],
        ["D4 ON16","0mod8 → 4mod8 → 2mod4 → 홀수","16/16/16/8","16","ON"],
    ]))
    add("""D4의 token 수는 512/512/1024/2048이며 실제 anchor 간격은 ±4/±2/±1이다. 오른쪽 anchor가 없으면 왼쪽을 복제한다. Legacy4와 D4는 다른 schedule이다. 정렬은 Score와 MSH 양쪽에 적용된다.

SCCTX001 Score wrapper는 `24+8K`B이며 K는 strings 수다. D4 ON16/OFF16은152B다. `actual_score_bytes`에 이 wrapper가 이미 포함되므로 순수 Score와 모든 header를 분리할 때만 이동한다.""")
    section("lineage","3. 실험 계보와 집계 범위")
    add("""```text
초기 rank/linear/nonlinear/residual/정렬 탐색
  └─ Linear / Nonlinear factorized 및 context 추가학습
       ├─ P0/P1/P2 predictor
       ├─ 복원 고정 Shared/Split/Gaussian/Conditional probability-only
       ├─ low-LR 후속·고정 symbol entropy probe
       └─ Full P0+Split 부모와 이후 joint/minimal/schedule 실험
            └─ None/Morton/Hilbert/NN-score × L2/L3/L4/Q2/D4/Context OFF
                 ├─ 현행 주 seed RD63점 + λ0
                 ├─ 맞춘 OFF12/OFF16 및 추가 seed
                 └─ B0/T05/T06/T15·anchor·복잡도 진단

공개 backbone + 공통 QR 경계 + 새 codec 초기화
  └─ Score56-only / MSH-only / two-path56 / Score736-only
       └─ train에서 제외한 같은 holdout128로 구조 pilot

별도 CleanToken repository
  └─ Hyper1D single/dual/lowrank-base/Morton/λ 실험 (현재 pilot과 분리)
```

다음 수는 서로 겹치므로 합쳐서 ‘학습 수’로 사용하지 않는다.""")
    add(table(["집계","수/상태","포함 범위"],[
        ["현행 주 seed 양의λ RD","63점·16계열 완료","전체 test; 최근26개 완료점 중 일부도 여기에 포함"],
        ["λ=0","별도1점 완료","전체 test; 비압축 기준이 아님"],
        ["추가 seed","별도2점 완료","기본 seed RD 곡선에 합치지 않음"],
        ["구조 pilot","4개 중3개 평가 완료,1개 학습 진행","holdout128; 전체 test 평가 아님"],
        ["최근26개 묶음","23개 학습·평가 완료,3개 학습 진행","20 full-test +3 holdout 완료"],
        ["최근 묶음 접수","prepare1 +train26 +eval26 =53 jobs","복구 train2와 이후 B0/T05/T15/T06 job4는 별도"],
        ["과거 context 결과 JSON","48개 완료 평가 기록","당시 원본 로그에서 추출; 현재 원본48개는 로컬 없음"],
        ["초기 Linear 사용자 표","12개 조건","원표 단위kB, 원본 bytes/jobID 미확인"],
        ["그 밖의 현존 평가 수치","37개 로그","CleanToken/이전joint/minimal/subset; 평가 후 shell 오류2개와 중복 checkpoint 포함"],
        ["현재 로컬 Slurm stdout","284파일·49개 prefix","job 재시도·배열·평가·다른 분기 포함; 독립 학습 수 아님"],
    ]))
    add("""평가 지표 기록은 **현존 four-metric 평가106개 + 과거 보존 JSON48개 =154개 source 기록**으로 정리했다. 여기에는 현재 주 실험69개(63+λ0+seed2+pilot3)와 추가37개가 포함된다. 추가37개 중2개는6,991장 평가 지표 출력 후 shell의 임시 파일 생성이 실패해 정상 job 종료와 구분한다. 같은 checkpoint를 다른 subset/진단에서 다시 평가한 기록도 있으므로154개를 독립 모델·학습 수로 해석하지 않는다. 초기 사용자 표12개와 초기 다중 모드 ablation, 독립 수신·profile은 이154개 집계와 별도로 수록한다.""")
    section("recent","4. 최근 제출한 26개 전체 목록")
    add("""이 표는 현재 진행 추적용이다. 여기서 완료된 전체 test 결과는 뒤의 RD·λ0·seed 표와 중복된다. `→`는 실패한 원래 train에서 새 full-state resume job으로 이어졌다는 뜻이다.""")
    def state_text(row):
        if row["eval_status"]=="complete_fulltest":return "완료 / test6991"
        if row["eval_status"]=="complete_holdout128":return "완료 / holdout128"
        return f"학습 진행 {row['estimated_optimizer_step']:,}/50k; eval 시작 로그 없음"
    add(table(["묶음","설정","λ","Train / 재개","Eval","최신 상태"],[
        [row["group"],case_name(row),row["lambda"],row["train_job"]+(" → "+row["resume_job"] if row.get("resume_job") else ""),
         row["eval_job"],state_text(row)] for row in review["experiments"]]))
    add("""구조 초기화 job457740은 공통 holdout128 선정과 네 arm 초기화를 완료했다. 구조 네 arm의 별도 전체6991장 test job은 아직 제출한 기록이 없다. 기존 seed·대조군의 full test와 혼동하지 않는다.""")
    section("rd","5. 현행 주 seed RD: 16계열·63점")
    families=defaultdict(list)
    for row in primary:families[row["family"]].append(row)
    assert sum(map(len,families.values()))==63 and len(families)==16
    add("""양의 λ는 .0032/.0064/.0128/.0256/.0512를 사용하며 계열마다 확보한 점 수가 다르다. 현행 최종 RD 후보는 Hilbert+D4이고, L3의 정렬×Context 2×2가 맞춘 대조군 분석 기준이다. 모두 같은λ 비교와 동일 화질 BD-rate를 구분한다.""")
    add(table(["JSON family","읽는 이름","완료 λ","점 수"],[
        [name,FAMILIES[name],", ".join(f"{r['rate_lambda']:g}" for r in sorted(rows,key=lambda r:r["rate_lambda"])),len(rows)]
        for name,rows in sorted(families.items())]))
    add("""미완료 핵심: None+L3 OFF12의 .0512가 끝나야 L3 네 곡선의 양의λ5점이 완성된다. 기본 seed H+D4 OFF16 .0128은 복구 중이며 아직 이63점에 없다. 아래 Run ID는 checkpoint 디렉터리에서 읽은 식별자다. 원래 job·실제 재개 job을 구분해야 하는 경우 최근 목록과 복구 이력을 함께 본다.""")
    rd_rows=[]
    for name,rows in sorted(families.items()):
        for row in sorted(rows,key=lambda r:r["rate_lambda"]):
            m=row["metrics"]
            rd_rows.append([FAMILIES[name],f"{row['rate_lambda']:g}",f"{m['psnr']:.4f}",f"{m['ssim']:.4f}",
                f"{m['lpips']:.4f}",f"{m['actual_bytes']/1024:.5f}",run_from_checkpoint(row["checkpoint"]),
                job_from_log(Path(row["source_log"]).name)])
    add(table(["설정","λ","PSNR","SSIM","LPIPS","KiB/scene","Checkpoint run","Eval"],rd_rows))
    section("special","6. λ=0·추가 seed·구조 pilot")
    add("### 6.1 λ=0: 여전히 실제 양자화·entropy coding이 있는 모델")
    add(table(["설정","Train / Eval","PSNR","SSIM","LPIPS","KiB/scene"],[
        ["H+D4 λ0",f"{r['train_job']} / {r['eval_job']}",f"{r['metrics']['psnr']:.4f}",f"{r['metrics']['ssim']:.4f}",
         f"{r['metrics']['lpips']:.4f}",f"{r['metrics']['actual_bytes']/1024:.5f}"] for r in zero]))
    add("""λ0를 비압축/화질 상한으로 사용하지 않는다. H+D4 .0032 대비 PSNR+0.1769dB에 bytes 약14.69배이며, 용량 증가의 큰 부분은 MSH y다. 역사적 non-comp24.7004dB나 현재 matched256 B0와 평가 집합을 섞어 손실을 계산하지 않는다.""")
    add("### 6.2 추가 seed: 같은 전체 train/test, model111124/data404")
    add(table(["설정","Train / Eval","PSNR","SSIM","LPIPS","KiB/scene"],[
        [r["arm"],f"{r['train_job']} / {r['eval_job']}",f"{r['metrics']['psnr']:.4f}",f"{r['metrics']['ssim']:.4f}",
         f"{r['metrics']['lpips']:.4f}",f"{r['metrics']['actual_bytes']/1024:.5f}"] for r in seeds]))
    add("""λ=.0128, D4 ON16/OFF16에서 ON은 bytes17.1857% 감소·PSNR+0.0726dB·SSIM+0.0026·LPIPS−0.0023이다. 같은λ의 추가 seed 한 쌍 관측이며 동일 화질 BD-rate나 seed 통계적 유의성이 아니다. 기본 seed OFF16이 아직 진행 중이므로 두 seed의 완성된 ON/OFF 반복 비교로 쓰지 않는다.""")
    add("### 6.3 구조 네 arm: 같은 train-derived holdout128")
    add(table(["Arm","활성 경로","Train / Eval","PSNR","SSIM","LPIPS","KiB/scene","상태"],[
        [r["arm"],{"score56_only":"rank56 Score만","msh_only":"F736 직접 MSH","two_path56":"rank56 +MSH residual","score736_only":"736→736 Score만"}[r["arm"]],
         f"{r['train_job']} / {r['eval_job']}",*(f"{r['metrics'][k]:.4f}" if r["metrics"] else "—" for k in ("psnr","ssim","lpips")),
         f"{r['metrics']['actual_bytes']/1024:.5f}" if r["metrics"] else "—",state_text(r)] for r in pilots]))
    add("""네 arm 모두 Hilbert·λ=.0128·새50k·model seed111123/data seed403 조건이다. **공개 backbone/QR 경계에서 codec을 공통 초기화하고 호환되는 tensor의 초기값도 맞춘 뒤 새로 학습**한다. Backbone/Gaussian decoder는 고정하고 codec과 geometry projection을 학습한다. Holdout128은 codec train에서 제외하며, 일반 test를 읽는 validation loader는 학습 중 사용하지 않는다. 과거 FullSplit 부모 후속 학습과 동일 조건이 아니다.

완료한 세 점에서 two-path56은 Score56-only보다 PSNR+1.2178dB·bytes13.42% 감소다. MSH-only 대비는 PSNR+2.3371dB와 bytes3.917배의 절충이므로 MSH-only를 탈락시키지 않는다. Score736은 channel group184×4로16 strings를 유지하지만 predictor 용량도 커져 rank만의 순수 대조군은 아니다. 모든 구조의 전체 test와 추가λ는 아직 평가/학습 완료 결과가 없다.""")
    section("diagnostics","7. 비압축·경로·정렬·독립 수신 진단")
    add(table(["실험","Job","대상/규모","확인 상태와 범위"],[
        ["L3 정렬별 실제 anchor·잔차","454995","None/Morton/Hilbert/NN-score,4모델×256장","ANALYSIS_COMPLETE; 현재 후속진단의 원본 C12 cache"],
        ["Hilbert schedule anchor·잔차","455166","L3/Q2/D4,3모델×같은256장","ANALYSIS_COMPLETE; 모델별 basis/projection이 달라 frozen-order 개입과 구분"],
        ["T04 native B0 및 B1 bridge","458024","native geometry512 vs current geometry224,256장","완료; 같은 encoder output·정렬·카메라"],
        ["T05 경로·양자화 우회","458004","현행 H+D4 .0128,7경로×256장","완료; actual만 packet rate 있음"],
        ["T15 현행 독립 수신","458784","T05 packet256개; 새 codec+Gaussian decoder 프로세스","완료; stdout PSNR delta256장 모두0, 검사 허용치 통과"],
        ["T06 frozen order 개입","458818","같은 F/P/basis/Score/XYZ,13순서×256장","완료; Hilbert packet은 T05와 SHA 일치; 재학습/PSNR 측정 아님"],
        ["과거 Full nonlinear Split 독립 수신","433727 /433728","기존 mean-offset 설정,전체6991장 encode→decode/render","두 프로세스 완료; PSNR24.3238/SSIM.7611/LPIPS.2371; 현재 T15와 다른 모델"],
    ]))
    add("""현재256장 공통 fingerprint는 `069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29`다. 주 checkpoint는 `outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/lambda0p0128/job_447746/checkpoints/nfcgs_main/version_0/step000050000.ckpt`다.

### 7.1 B0와 T05의 모든 경로

B0는 공개 native geometry512 decoder, B1은 현재 학습된 projection 이후 F736을 직접 넣는 우회다. B1은 현행 checkpoint의 화질 상한이 아니다.""")
    mode_names={"b1_direct":"B1 직접 F","lr_continuous":"연속 LR만","lr_decoded":"실제 복호 Score의 LR만", "score_quantization_bypass":"Score 양자화 우회",
        "msh_quantization_bypass":"MSH y/z 양자화 우회","both_quantization_bypass":"양쪽 양자화 우회","actual_packet":"실제 packet"}
    diagnostic_rows=[["B0 native", "24.90270","0.77651","0.24100","—"]]
    for key,row in t05["modes"].items():
        diagnostic_rows.append([mode_names[key],f"{row['psnr']:.5f}",f"{row['ssim']:.5f}",f"{row['lpips']:.5f}",
                                f"{row['KiB_per_scene']:.4f}" if row["KiB_per_scene"] is not None else "—"])
    add(table(["경로","PSNR","SSIM","LPIPS","KiB/scene"],diagnostic_rows))
    add("""Actual−B0는 이 matched256에서−1.02621dB다. MSH 양자화 우회 개선은+0.04677dB로 작지만 Hcont에 분석/합성 손실은 남는다. Score 우회는−0.60840dB로 내려가며, 이는 fixed checkpoint의 residual 분포·공동 적응을 포함한다. 양자화의 독립 손실이나 각 경로의 가산 기여로 해석하지 않는다. 재학습한 pilot 순위를 대신하지도 않는다.

Native slot-order 검사에서 Gaussian field 차이는0이고 평균 PSNR 차이는 약−2.61e−7dB지만 최대 pixel 차이는0.0124051이었다. 동일 Gaussian 반복 렌더에서는 B0/B1/actual의 pixel 차이가0이었다. 순열 불변 Gaussian 값과 GPU pixel bit-exact를 구분한다.

T15 수신기는 scene encoder·입력 영상·원래 순열 없이 packet과 공유 checkpoint로 Gaussian을 만든다. Target 카메라·영상 크기는 렌더, 실제 target RGB는 metric에만 쓴다. T05 원본 pixel/feature와의 bit-exact 증명은 원본 tensor가 저장되지 않아 주장하지 않는다.

### 7.2 T06 전체 aggregate 결과""")
    add(table(["순서","전체 KiB","Raw Score KiB","MSH y KiB","MSH z KiB","Header B","Anchor XYZ 거리","Score 잔차 정규화 MAE"],[
        [name,f"{r['total_kib']:.5f}",f"{r['score_raw_kib']:.5f}",f"{r['msh_y_kib']:.5f}",f"{r['msh_z_kib']:.5f}",
         f"{r['header_bytes']:.1f}",f"{r['anchor_xyz_distance']:.7f}",f"{r['normalized_score_mae']:.7f}"] for name,r in t06["rows"].items()]))
    add(table(["기준−Hilbert","평균 B/scene","95% paired scene bootstrap CI"],[
        [name,f"{r['mean']:+.3f}",f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"] for name,r in t06["paired_bytes_minus_hilbert"].items()]))
    add("""Slot→Hilbert는 전체−34.25%, raw Score−44.82%, anchor 거리−91.23%, 예측 잔차 MAE−8.75%다. 관찰된 전체 byte 차이의 약94.8%는 Score stream에서 나온다. MSH도 변하므로 이를 독립 인과 기여율로 읽지 않는다. Morton→Hilbert의 추가 전체 절감은 약1.73%다.

Random10은 scene 안에서 먼저 평균하고1000회 scene bootstrap을 사용했다. Training seed CI가 아니다. Anchor 거리는 유효 anchor pair의 가중 평균이고 MAE는 첫 단계까지 포함한 전체 symbol 가중 평균이라 분모가 다르다. 같은 Hilbert 학습 weight에 대한 분포 변화 진단이며 feature distortion도 달라질 수 있어 **동일 화질 RD 절감률이 아니다**.

주요 서버 출력: `outputs/nfcgs_fixed_order/job_458818/REPORT.md`, `outputs/nfcgs_independent_receiver/job_458784/receiver/REPORT.md`, `outputs/nfcgs_path_diagnostics/job_458004/REPORT.md`, `outputs/nfcgs_b0_matched/job_458024/REPORT.md`.""")
    section("complexity","8. 복잡도·시간·메모리 측정")
    add("""### 8.1 현행 정렬/context 37 checkpoint: job455251 완료

같은 RTX A5000(ariel-v10), BF16, threads8, 같은 C12 cache에서 warmup4+측정32장, codec 구간3회 반복이다.32장 profile bytes를 전체6991장 RD bytes와 섞지 않는다. Encode는 token-position 준비를 포함한 `compress_scene_tokens`, decode는 `feature_codec.decompress`이며 image encoder·Gaussian decoder·renderer는 제외한다.""")
    tables=md_tables(complexity)
    overview=next(block for block in tables if block.startswith("| 설정 | Encode ms | Decode ms | 정렬 ms"))
    add(overview)
    add("""이 표의 Hilbert Context OFF는 **L3 OFF12**다. D4 OFF16이 아니다. D4는 L3 대비 encode+8.71%, decode+10.70%였다. NN-score는 L3/Hilbert 대비 encoder가 약42.52% 느렸다. MAC만으로 이 시간 순위를 예측할 수 없다.

총 peak는 전체 모델이 올라간 harness의 allocated VRAM이며 codec-only 독립 수신기의 최소 메모리가 아니다. 추가 peak는 호출 직전 allocated를 뺀 증가량이다. CUDA context/driver·NVML·CPU rANS/native 메모리와 다르다.

37개 point의 실제 profile 목록:""")
    add(next(block for block in tables if block.startswith("| Point | Encode ms |")))
    add("설정별 parameter 수와 codec core 연산량도 함께 보존한다. 저장 parameter에는 비활성 모듈이 포함될 수 있으며, codec core GMAC는 CPU 정렬·entropy coder·renderer의 실행 시간을 나타내지 않는다.")
    add(next(block for block in tables if "| Codec core encode GMAC |" in block.splitlines()[0]))
    add("""### 8.2 과거 overhead·probe

- Overhead job423729: **18 profile 완료**. 기존 context/predictor/probability 계열, warmup4·측정32장. `OVERHEAD_AGGREGATE_COMPLETE profiles=18`과 `OVERHEAD_ALL_COMPLETE`를 확인했다.
- Fixed-symbol entropy probe array423055_0/1: λ=.0064/.0256, TRAIN128에서 table fit→TEST128 평가. 원본 symbol/feature 복원을 고정하고 context histogram이 전체 payload 약0.736%/0.759% 절감했다. 이는 새 full codec 학습이나 전체 test/render 결과가 아니다.
- Complexity job454488 및 broad mechanism job454797은 현재 stdout가 비어 있어 성공으로 집계하지 않는다. 실제 완료 근거는455251,454995,455166이다.

새 OFF1/OFF4/Both-OFF·OFF16·구조 모델의 빠진 profile 보완은 아직 별도 완료 결과가 없다. 새 할당에서는 H+L3/H+D4/NN-score .0128 reference를 함께 재측정하고 새 output으로 기록해야 한다.""")
    section("legacy","9. 9월 이전/초기 실험 전체 기록")
    add("""### 9.1 초기 rank·projection·residual·정렬 탐색

9월2일 보존 보고서 기준이다. 현재 H+D4 checkpoint의 B0/T05 결과와 다른 모델·초기화·시점이다. 당시의 `upper`는 역사적 vanilla 참고명이며 수학적 화질 상한이 아니다.

| 실험 | 수행 내용 | 당시 확인 상태 |
|---|---|---|
| Experiment17 | rank40/56/72/80에서 projection·quantization·full residual의 누적 우회 | 전체 test 결과 보존; 같은 full-trained checkpoint에서 경로를 끈 진단 |
| Experiment16 | 6,991장 covariance spectrum·채널/토큰 상관 | job409784 완료; covariance용64 tokens, lag통계32 channels |
| Experiment14 | Linear-no-residual / Nonlinear32-no-residual / Linear-residual의 새50k | 세 모델 완료; 기존 PCA artifact와 다른 초기화, 최초409785 실패 후 수정 |
| Experiment15 | no-residual 모델에서 none/Morton/random inference 순서 | eval array410949 완료 |
| Experiment15b | frozen Morton-trained residual 모델에서 순서 변경 | eval array411709 완료 |
| 초기 Full RD | λ=.0004/.0016/.0064/.0256와 codec-free | eval array411698 완료; 초기30장 sweep과 구분 |

Experiment17 전체 수치:""")
    early_tables=md_tables(early)
    condition_tables=[block for block in early_tables if block.startswith("| Condition | PSNR | SSIM | LPIPS | Bytes/scene")]
    add(condition_tables[0])
    add("""위 projection 경로들은 독립적으로 재학습한 모델이 아니라 같은 full-trained checkpoint의 누적 개입이다. 연속 경로에는 packet rate가 없다. 해당 손실을 현행 모델의 독립 양자화/경로 기여로 재사용하지 않는다.

Experiment16의 covariance energy:""")
    add(next(block for block in early_tables if block.startswith("| Stream | Mean abs. off-diagonal")))
    add("""Joint concat rank56의97.93% energy가 모든 지각적 appearance 정보를 보존한다는 뜻은 아니다. Appearance rank56은91.00%, observable geometry는99.62%여서 covariance scale의 영향도 고려해야 한다.

Experiment14의 새50k 구조 결과:""")
    add(condition_tables[1])
    add("""당시 구조 실험은 matched PCA 초기화 대조군이 아니므로 현재 fresh pilot과도 직접 합치지 않는다.

Experiment15 /15b의 inference 정렬 결과:""")
    add(next(block for block in early_tables if block.startswith("| Ordering | PSNR |")))
    add(next(block for block in early_tables if block.startswith("| Inference ordering |")))
    add("초기 protocol-matched Full RD:")
    add(next(block for block in early_tables if block.startswith("| Condition | Bytes/scene | bits/Gaussian")))
    add("""### 9.2 초기 Linear 사용자 표12개

초기50k의 사용자 제공 결과다. 원표의 크기 단위는 `kB`로 보존한다. 원본 bytes가 없어 KiB로 변환하지 않았고 job ID도 확정하지 않는다. 역사적 codec-OFF는24.7004/.7682/.2480이고 scene packet 크기는 없다.""")
    initial_section=original_linear.split("## 1. 초기 Linear — 사용자 제공 baseline 표",1)[1].split("## 2.",1)[0]
    add(md_tables(initial_section)[0])
    add("### 9.3 보존 JSON의 완료 평가48개")
    add("""9월15일에 원본 완료 eval 로그에서 재추출한 기록을 사용한다. **현재 다운로드 폴더에는 이48개 원본 stdout가 없으므로 이번에 다시 raw log를 검산한 결과는 아니다.** 정확한 bytes·checkpoint·당시 training lineage는 보존 JSON에 있다.

이전50k+추가50k, 추가25k/10k는 서로 다른 stage다. Stage step과 누적 lineage를 함께 표시한다. Weights-only continuation은 optimizer-state resume와 다르며, probability-only stage는 복원 관련 parameter를 고정했다.""")
    legacy_counts=Counter(row["family"] for row in legacy["evaluations"])
    add(table(["과거 묶음","완료 평가 수"],[[LEGACY[name],count] for name,count in legacy_counts.items()]))
    legacy_rows=[]
    for row in sorted(legacy["evaluations"],key=lambda r:(r["family"],r["lambda"],r["task"])):
        header=dict(re.findall(r"\b(PRIOR|ARM|RESIDUAL|ENTROPY)=([^\s]+)",row["header"]))
        detail=", ".join(f"{key}={value}" for key,value in header.items()) or f"task{row['task']}"
        legacy_rows.append([LEGACY[row["family"]]+" / "+detail,row["rank"],f"{row['lambda']:g}",
            f"{row['stage_optimizer_steps']:,}/{row['cumulative_optimizer_steps']:,}",f"{row['psnr']:.4f}",f"{row['ssim']:.4f}",
            f"{row['lpips']:.4f}",f"{row['actual_bytes']/1024:.5f}",job_from_log(Path(row["log"]).name)])
    add(table(["실험/arm","Rank","λ","Stage/누적 steps","PSNR","SSIM","LPIPS","KiB/scene","Eval"],legacy_rows))
    add("""주요 과거 판단: factorized 대비 decoder-causal context가 유효했고, 복원을 고정한 probability-only에서는 even/odd Split이 Shared 추가학습 대조군보다 실제 bytes를 줄였다. P1/P2의 추가 이득은 두λ에서 일관되게 크지 않았다. Full low-LR의 `.0064/P0`은 최종 저장 실패로 완료3개에 포함하지 않는다.

### 9.4 외부 스크리닝 기록

E82/E83 정렬·context 스크리닝은 보존 문서에 완료 결과가 있다. GlobalSplat16K·2048 tokens·train55/test12씬·고정 PCA 심볼·정적 histogram/교차 entropy 기반이다. 현행4096-token closed-loop 양자화·actual full-test codec과 다른 대리 진단이므로 위 RD 숫자에 합치지 않는다. 여기서 제안된 정렬 후보는 이후 실제 codec 학습/평가로 별도 검증했다.""")
    section("other","10. 그 밖의 평가 수치와 CleanToken 별도 분기")
    add("""현존 로그를 추가로 검색해 위 목록에 없는 **37개 평가 파일**을 찾았다. 아래는 기록 누락을 막기 위한 별도 목록이다. `CleanToken/` 경로는 현재 `tokencomp/`와 다른 repository·구조·훈련 recipe이며 현재 MSH-only/two-path pilot과 맞춘 대조군이 아니다.

Checkpoint step은 saved metadata가 출력된 경우 그 값을 쓰고, 그렇지 않으면 **파일명 근거**임을 표시했다. Eval Hydra의 `trainer.max_steps` 기본값으로 과거 학습 step을 추정하지 않았다.31/32/97/100장 subset은6,991장 전체 test와 구분한다. 일부32장 평가는 같은 checkpoint의 반복/진단이다.""")
    for branch in ("tokencomp 과거 실행","CleanToken 별도 분기"):
        add("### "+branch)
        add(table(["설정/변형","λ","평가 장면","Step 증거","PSNR","SSIM","LPIPS","KiB/scene","로그"],[
            [row["label"],row["rate_lambda"],row["scenes"],
             str(row["step"])+( " (파일명)" if row["step_evidence"]=="checkpoint filename" else ""),
             f"{row['metrics']['psnr']:.4f}",f"{row['metrics']['ssim']:.4f}",f"{row['metrics']['lpips']:.4f}",
             f"{row['metrics']['actual_bytes']/1024:.5f}",row["log"]]
            for row in extra if row["branch"]==branch]))
    add("""`slurm-gs-msh-fulltest-440685_0.out / _1.out`은6,991장과 최종 PSNR/SSIM/LPIPS/bytes를 모두 출력했다. 다만 stderr에는 후속 shell의 `cannot create temp file for here-document: No space left on device`가 있어 **평가 지표 산출 완료·후속 shell 오류**로 기록한다. 정상 job 종료를 확인한 것으로 쓰지 않는다.

예전 Hyper1D single/dual/lowrank-base의 수치나 joint500k 결과를 현재 fresh 구조 pilot의 우열 근거로 쓰지 않는다. 이 목록은 출력된 평가 지표를 보관한 것이며, 동일 초기화·학습 budget·scene/frame 정합까지 새로 맞춘 재실험이 아니다.""")
    section("failures","11. 실패·중단·완료 여부 미확인 이력")
    add(table(["작업","근거/상태","현재 취급"],[
        ["456951 /457099 원래 대조군","FAILED 및 step signal9; QOSMaxGRESPerUser만으로 강제종료 원인을 확정하지 않음","458782/458783으로 full-state 복구 진행"],
        ["451821 /451825 /451827","과거 ENOSPC 중단","453064/453066/453068 재개 후 완료점은 현재 RD에 포함; 옛 실패를 현재 진행으로 세지 않음"],
        ["NN-score 448650 계보 /450904 /450934","과거 재개 실패 및 재시도 기록","최종 eval450935의 완료 RD 점을 사용; 실패 job을 추가 모델로 세지 않음"],
        ["Mean-offset 433719_0/1","OSError28 No space left; 최종 성공 평가 없음","20k 설정의 시도 기록, 완료 ablation 결과로 사용하지 않음"],
        ["Mean-offset 재시도434543_0/1","/tmp/pymp ENOSPC·취소 오류; 시작 직후 진행만 있음","두 arm의 완성된 비교 결과 없음; 현재 학습3개에 포함하지 않음"],
        ["Nonlinear test100 job436182","11frames scene에서20개의 disjoint context/target을 요구해 ValueError","실패 subset 평가; 다른 λ의436236 test100 완료와 구분"],
        ["MSH fulltest440685_0/1","6,991장 지표 출력 후 shell here-document 임시 파일 ENOSPC","평가 수치는 보존; 정상 job 종료와 구분"],
        ["과거 Full low-LR .0064/P0","최종 checkpoint 저장 실패","완료48개 중 low-LR3개와 구분"],
        ["454797 broad mechanism /454488 complexity","현재 stdout 비어 있음","접수/완료를 추정하지 않음; 명시적 완료 후속 job을 사용"],
        ["그 밖의 옛 array/eval 시도","아래 전체 로그 목록에 포함; 완료 지표 없는 시도도 존재","오래된 progress만으로 지금 RUNNING/PENDING이라 하지 않음"],
    ]))
    add("""파일명의 array ID와 실제 stderr의 Slurm child job ID가 다를 수 있다. 실패 원인 확정이나 현재 취소 여부에는 해당 job의 accounting이 필요하다. 이번 정리는 옛 작업을 재제출하거나 취소하지 않는다.""")
    section("pending","12. 아직 실행하지 않은 후속 작업")
    add(table(["항목","현재 상태/의존성"],[
        ["Score736-only 평가457829","학습457828 완료 후 기존 연결 평가 사용"],
        ["L3 None OFF12 .0512 평가457098","복구458782의 최종50k 대기"],
        ["기본 seed D4 OFF16 평가457100","복구458783의 최종50k 대기"],
        ["구조 pilot 전체6991장 test","현재 네 arm은 holdout128 단계; 자동으로 전체 test를 추가하지 않음"],
        ["구조별 추가λ·bitrate 맞춤","Score736 결과와 현재 holdout의 화질/실제 용량을 보고 후보 선정"],
        ["Morton L3 OFF12 .0128 (T17)","계획상 deferred; 완료 결과 없음"],
        ["Hilbert L3 ON/OFF12 추가 seed","계획상 deferred; D4 추가 seed가 이를 대신하지 않음"],
        ["B0/T05 전체6991장 확장","현재 matched256만 완료; 별도 full-test RD로 표시하지 않음"],
        ["새 OFF/주요 구조 complexity 보완","기존37개 전체를 반복하지 않고 빠진 설정+동일할당 reference 중심"],
    ]))
    add("""다음 우선순위는 남은 세 학습과 평가 수집, 구조 후보/λ 결정, 기본 seed와 추가 seed의 ON/OFF 비교 완성이다. 현행 정렬의 동작 근거는 T06과 anchor 진단으로 보완됐으므로, 새 정렬 학습을 무조건 늘리기보다 남은 결과를 우선한다.""")
    section("ledger","13. 내려받은 Slurm 로그 파일의 전체 job 목록")
    add("""현재 로컬 `slurm/`의 stdout284개를 파일명 prefix별로 전부 나열했다. stderr도284개가 있다. 이 표는 **파일/시도 색인**이며 독립 모델 수나 현재 queue 상태가 아니다. CleanToken의 다른 분기, 재시도, 실패, 빈 로그도 포함된다. 오래된48개 평가 JSON의 원본 로그는 현재 폴더에 없으므로 이284개와 별도다.""")
    add(table(["로그 prefix","stdout 수","파일명에 나타난 job/task 식별자"],[
        [name,len(ids),", ".join(ids)] for name,ids in sorted(ledger.items())]))
    section("sources","14. 근거 파일과 검증 범위")
    add("""### 상태·전체 수치

- [최신 상태 검토](analysis/nfcgs_logs_2026-10-20_1647/REPORT.md), [review.json](analysis/nfcgs_logs_2026-10-20_1647/review.json), [inventory.json](analysis/nfcgs_logs_2026-10-20_1647/inventory.json).
- [이번 통합 목록의 기계 판독 원본](analysis/nfcgs_experiment_master_2026-10-20/all_experiments.json), [생성 코드](analysis/nfcgs_experiment_master_2026-10-20/build_summary.py).
- [공식 실험 계획과 조건](NFCGS_PAPER_EXPERIMENT_PLAN_2026-10-18.md). 계획 문서의 과거 status보다 최신 로그 snapshot을 우선한다.

### 진단·과거 기록

- [T06 사용자 REPORT 후속](analysis/nfcgs_logs_2026-10-20_1636/T06_FOLLOWUP.md), [T06 원문](analysis/nfcgs_logs_2026-10-20_1636/provided_t06_report.md).
- [B0/T05 사용자 REPORT·sacct 검토](analysis/nfcgs_logs_2026-10-20_0911/FOLLOWUP.md).
- [독립 수신 T15](NFCGS_INDEPENDENT_RECEIVER_2026-10-20.md), [복구 방법](NFCGS_PAPER_CONTROLS_RECOVERY_2026-10-20.md).
- [실제37개 complexity 결과](analysis/nfcgs_complexity_455251/REPORT.md).
- [9월 context 통합 기록](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14.md), [48개 원시 지표 JSON](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json).
- [초기9월2일 실험](ABLATION_PROGRESS_2026-09-02.md), [초기 Linear 사용자 표](NFCGS_EXPERIMENT_RESULTS_2026-09-10.md).
- [고정-symbol entropy probe](NFCGS_ENTROPY_PROBE_2026-09-14.md), [E82/E83 대리 스크리닝](E82_E83_SORT_CONTEXT_SCREEN_2026-10-07.md).
- [과거 overhead profile 방법](NFCGS_OVERHEAD_PROFILE_2026-09-15.md), [이전 전체 test 시간 기록](NFCGS_CONTEXT_TIMING_2026-10-12.md). 최신 동일 GPU profile의 비교에는 이 문서의8.1을 사용한다.

### 검증 수준

최신16:47 검토는 새 ON 평가와 미완료 세 학습을 다시 읽고 이전 완료 평가 로그65개를 재해시했다. 현재 통합 문서는 현존 four-metric 평가106개를 직접 재검색해 최신 inventory의69개와 수치를 대조했고, 추가37개를 별도 기록했다. 과거48개는 보존 JSON을 사용하며 현재 원본 로그 재검증은 불가능하다. 초기 사용자 표12개는 원표 수치·단위를 그대로 보존한다.

집계 지표와 protocol/scene-count 검증을 모든 장면의 frame/tensor bit-exact 검증으로 확대하지 않는다. 다른 초기화·checkpoint lineage·scope·scene 집합·training seed의 수치를 한 RD 곡선에 합치지 않는다. 진단 결과의 paired scene CI와 training seed CI도 구분한다.

문서 생성은 로컬 자료 정리만 수행했다. 모델/학습 runtime, 서버 job, dependency는 변경하지 않았다.""")
    document="\n".join(content)
    OUTPUT.write_text(document,encoding="utf-8",newline="\n")
    structured=dict(as_of="2026-10-20 16:47 Asia/Seoul",scope="documented experiments and available logs, not live queue",
        current_inventory=inventory,recent_experiments=review["experiments"],legacy48=legacy["evaluations"],
        additional37=extra,all_current_numeric106=raw_records,job_file_index=dict(ledger),
        counts=dict(primary_rd=63,lambda_zero=1,additional_seed=2,completed_pilot128=3,legacy_archived=48,
                    additional_numeric=37,numeric_source_records=154,initial_user_table=12,stdout_files=284),
        source_sha256=SOURCES)
    (HERE/"all_experiments.json").write_text(json.dumps(structured,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(output=str(OUTPUT),lines=len(document.splitlines()),bytes=len(document.encode()),
        modern_rows=len(rd_rows),legacy_rows=len(legacy_rows),extra_rows=len(extra),log_prefixes=len(ledger)),ensure_ascii=True))


if __name__=="__main__":main()

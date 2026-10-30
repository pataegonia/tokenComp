# 다음 세션 인수인계 — 2026년 10월 20일

## 0. 다음 세션에서 먼저 읽을 내용

현재 작업은 **GlobalSplat scene-token 압축 논문의 실험 수집·검증과 후속 실험 준비**다. 최신 근거는 **2026-10-20 21:59 KST 다운로드 로그**다. 실시간 Slurm 상태가 아니다.

**기존26개 학습 묶음은25개가 최종50k 평가까지 완료됐고, 남은 학습은 H+D4 OFF16 하나다.** 이 작업은 원래457099를 full-state 복구한 **458783**, 마지막 완전한 progress는 **48,946/50k, 97.892%**다. 연결 eval **457100**은 시작 로그가 아직 없다.

**Score736-only는 최종50k·holdout128 평가457829가 완료됐다.** 마지막 다운로드된 train 로그가47,614 step이라도 이를 미완료로 다시 세면 안 된다. 새 평가가 saved-step/config/provenance 검증을 통과한 완료 근거다.

현행 주 seed의 양의λ RD는 **64점·16계열**, 별도λ0 한 점과 추가 seed ON/OFF 두 점이 있다. **L3의 None/Hilbert×Context OFF/ON 네 곡선이 모두5λ, 총20점을 확보했다.**

사용자는 추가 제안까지 검토한 뒤, 736 결과와 독립적인 **None+D4 OFF16 + N01 + native 전체 test + T06 렌더**를 돌릴 수 있게 구현하라고 요청했다. 구현·검증·개별 파일과 명령 전달은 완료했다. 이는 **새 학습1 + 연결 평가1 + 진단/평가3 =5개 job 묶음**이다. 그러나 **새 묶음의 접수 state/로그는 아직 로컬에 없어 서버 접수 여부와 ID는 미확인**이다. 미접수라고 단정하거나 중복 제출하지 않는다.

736 결과를 받은 뒤의 구조 확장 우선순위는 **MSH-only λ=.0032와 two-path56 λ=.0512로 실제 rate 간격을 줄이는 탐색**이다. 이 추가λ runner는 아직 구현·제출하지 않았다. 12/17개 제안 전체가 접수됐다고 쓰지 않는다.

## 1. 사용자 작업 방식·경로

- 로컬 workspace: `C:/Users/a0103/Desktop/school/vml/global`.
- 로컬 repo: `C:/Users/a0103/Desktop/school/vml/global/upstream/globalsplat`.
- 다운로드한 서버 로그: **workspace 루트 `slurm/`**. repo의 `logs/slurm/` 또는 `slurm/`와 혼동하지 않는다.
- 서버 repo: `/ceph_data/clue9986/tokencomp/upstream/globalsplat`.
- 서버 환경: `conda activate globalsplat`.
- 사용자가 서버 명령을 실행하고 로그를 내려받는다. 이 세션에서 에이전트가 SSH·Slurm 제출·취소를 직접 수행하지 않았다.
- **새 코드·문서·산출물은 `upstream/globalsplat/` 안에 둔다.** 사용자에게 ZIP으로 전달하지 않는다. 목록 파일 링크만 주지 말고 **서버에 반영할 각 파일의 직접 클릭 링크와 실행 명령**을 채팅에 제공한다.
- 미커밋 변경·삭제·untracked 파일이 많다. 무관한 변경을 되돌리거나 `git reset`하지 않는다.
- 사용자는 짧은 실행 지시를 선호한다. 이미 승인한 범위를 반복 확인하지 않고 필요한 구현·검증을 진행한다. 준비·접수·시작·학습 완료·평가 완료를 구분한다.

## 2. 신뢰할 최신 자료와 보존 범위

repo 기준 상대 경로다.

| 자료 | 용도 |
|---|---|
| [21:59 REPORT](analysis/nfcgs_logs_2026-10-20_2159/REPORT.md) | 최신 상태·736 결과·마지막 L3 점·다음 판단 |
| [21:59 review.json](analysis/nfcgs_logs_2026-10-20_2159/review.json) | 기존26개 상태·출처 해시·설정·완료 근거 |
| [21:59 inventory.json](analysis/nfcgs_logs_2026-10-20_2159/inventory.json) | 주 seed RD64점·λ0·추가 seed2점·구조4점 |
| [구조 비교 계산](analysis/nfcgs_logs_2026-10-20_2159/architecture_comparison.json) | 736/Score56/two-path의 정확한 차이 |
| [19:40 L3 계산](analysis/nfcgs_logs_2026-10-20_1940/l3_comparison.json) | 네 곡선5점·같은λ·공통 구간 BD 비교 |
| [전체 실험 목록](NFCGS_EXPERIMENTS_MASTER_2026-10-20.md) | 초기 실험부터 과거 분기·실패·job 색인까지 전체 계보; 본문 수치는16:47 당시 |
| [추가 제안 원문](analysis/nfcgs_additional_experiment_review_2026-10-20/provided_proposal.md) | 사용자 첨부를 원문 그대로 보존; 계획이며 접수 기록이 아님 |
| [추가 제안 검토](NFCGS_ADDITIONAL_EXPERIMENT_REVIEW_2026-10-20.md) | E01~E20/N01~N10 우선순위·조건부 예산 |
| [독립 후속5개 실행 안내](NFCGS_INDEPENDENT_FOLLOWUPS_2026-10-20.md) | 새로 구현한 제출 묶음·파일·명령·입출력 |

21:59 검토는 이전 완료 평가70개의 stdout 해시가 유지되는지 확인하고, 새736 평가와 남은 복구 학습을 다시 읽었다. 기존 architecture/seed/recovery 제출 state의 runtime 코드 해시도 일치했다. Raw checkpoint·서버의 모든 scene JSON을 로컬에서 직접 확인한 결과로 확대하지 않는다.

**과거 snapshot은 덮어쓰지 않는다.** `audit_snapshot.py`와 master의 `build_summary.py`는 당시 점 수·현재 파일 집합을 가정하는 일회성 검산 코드다. 새 로그가 생긴 뒤 옛 코드를 그대로 실행하면 assert가 실패할 수 있다. 새 시각의 디렉터리에서 검산을 확장하고 최신 근거를 남긴다.

## 3. 현재 모델·프로토콜

| 항목 | 조건 |
|---|---|
| 기반 | 공개 GlobalSplat RE10K-32K,4096 scene tokens |
| 압축 대상 | appearance512 + geometry512→observable geometry224 = F736 |
| 주 모델 | rank56 LR/Score 복원 + MSH residual; N192/M320/adapter96 |
| Minimal | centering·Score norm·mean conditioning·channel Context OFF, linear transform |
| Gaussian | stage3, token당8개 = scene당32768개; M_max16과 구분 |
| 주 RD 학습 | FullSplit λ=.0256 부모 weights-only warm start,50k, GPU2×batch2×accumulation2, model111123/data403 |
| 학습 LR |1e-4,35k/45k decay |
| 전체 test | RE10K C12/T8,256×256,BF16,eval seed0,실제6991 scene |
| 구조 pilot | 공개 backbone/QR 경계에서 codec fresh init, 같은 train-derived holdout128 제외, 최종50k actual 평가 |
| 고정·학습 경계 | backbone/Gaussian decoder 고정, codec과 geometry projection 학습 |
| 용량 | 모든 scene packet/header 포함 B/scene÷1024=KiB; 공유 checkpoint 제외 |

주 RD의 기존 부모 후속 학습과 fresh 구조 pilot을 같은 곡선에 합치지 않는다. 구조 pilot은 학습에서 제외한128장이고 전체 test6991장과도 다르다. 기존 val이 test로 매핑되는 문제를 피하려고 구조 학습에서는 val loader를 끄고 별도 holdout을 썼다.

전체 test loader의 nominal 분모7286과 실제6991을 구분한다. Warmup5 이후의 timing6986회도 장면 수가 아니다. Pilot은128장, timing123회다. 평가 중 per-scene fitting/test-time 학습은 없다.

### 명칭·단계

- `context_only` = **None+L3 공간 Context ON인 two-path**. Score-only가 아니다.
- `hilbert_only` = **Hilbert+L3 OFF12인 two-path**. MSH는 유지한다.
- 공간 Context OFF는 LR/MSH 제거가 아니다. 정렬은 Score와 MSH 둘 다에 작용한다.
- OFF1: 전체 token×전체56을1 string. OFF4: D4 token4단계×전체56을4 strings.
- L3 OFF12/ON12:3단계×채널16/16/16/8. D4 OFF16/ON16:4단계×동일 그룹.
- L2/Q2는8 strings지만 다른 참조 topology다. Legacy4와 D4도 다르다.
- D4 순서:0mod8→4mod8→2mod4→홀수, token512/512/1024/2048. 실제 anchor 간격±4/±2/±1. 오른쪽이 없으면 왼쪽 복제. 참조는 실제 앞 단계 복호값이다.
- SCCTX001 wrapper는 `24+8K`B. D4 K16이면152B. `actual_score_bytes`에 이미 포함되어 있다. Raw Score를 분리할 때만 빼고 all headers에 더한다.

## 4. 기존26개 접수 묶음과 남은 작업

| 묶음 | 학습 수 | 학습·평가 완료 | 남은 학습 |
|---|---:|---:|---:|
| 양끝점·λ0 확장 |13|13|0|
| L3 대조군6 + H+D4 OFF16 |7|6|1|
| fresh 구조4개 |4|4|0|
| 추가 seed ON/OFF 쌍 |2|2|0|
| 합계 |**26**|**25**|**1**|

원래 접수 job은 train26+eval26+prepare1=53개였다. 복구 train2와 B0/T05/T15/T06 및 새로 준비한 후속5개는 별도다. 이를 모든 역사적 job 수로 쓰지 않는다.

### 학습이 남은 한 점

| 설정 | 원래 train | 현재 복구 train | 연결 eval | 마지막 추정 |
|---|---|---|---|---|
| H+D4 OFF16 .0128 |457099|458783|457100|48946/50000,97.892%|

실제15k 저장 checkpoint의 optimizer·scheduler·loop를 복구하고 원래 출력 디렉터리에서 총50k로 이어갔다. 기존 eval의 dependency도458783의 afterok로 연결했다. 새 eval을 만들지 않는다. 기본 loader는 fresh shuffle이므로 중단 없는 실행의 다음 sample 순서까지 같다는 보장은 없다.

None+L3 OFF12 .0512는 `456951→458782`의25k 복구를 거쳐 완료됐다. Eval457098은 **22.9042/.7114/.2980,30.14873KiB**다. 원래 실패 작업을 현재 진행으로 다시 세지 않는다. Signal9의 원인을 `sacct`의 QOSMaxGRESPerUser만으로 확정하지 않는다.

서버에서 필요할 때 사용자가 확인할 명령:

```bash
squeue -j 458783,457100 -o '%.18i %.28j %.10T %.30R %E'
sacct -j 458783,457100 --format=JobID,State,ExitCode,Reason,Elapsed,NodeList -P
```

로컬에서 먼저 읽을 파일은 `slurm-gs-paper-control-resume-458783.out/.err`, `slurm-gs-order-context-eval-457100.out/.err`다.

## 5. 구조4개 결과와 현재 판단

λ=.0128, fresh 공통 초기화, 같은 설계의 holdout128,50k다.

| Arm | Train / Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---:|---:|---:|---:|
| score56_only |457741 /457742|21.7889|.6653|.3295|65.53055|
| msh_only |457743 /457744|20.6696|.6123|.3700|14.48434|
| two_path56 |457745 /457746|23.0067|.7157|.2796|56.73486|
| score736_only |457828 /457829|21.7001|.6603|.3315|67.81793|

공통 prepare457740이 네 초기화와 holdout을 만들었다. 호환 tensor는 같은 초기값을 복사·해시 확인했다. T07/T18은 rank56 쪽, T08/T18은 MSH 쪽의 호환 초기 tensor를 공유한다. 특정 모델만 학습된 기존 codec을 재사용한 비교가 아니다.

- Two-path는 Score56보다 **+1.2178dB,bytes−13.42%**다.
- Two-path는 Score736보다 **+1.3066dB,bytes−16.3424%**다.
- Score736은 Score56보다 **−.0888dB,bytes+3.4906%**라 이번 점에서는 지배된다.
- MSH-only는 약3.92배 작은 rate이므로 낮은 화질만으로 탈락시키지 않는다.
- **한 점의 지배 관계를 구조 전체의 RD 지배·rank56의 보편적 최적성으로 확대하지 않는다.**

T09는 learned736→736 Score,184×4 채널 그룹으로16 strings를 유지한다. Slice16을 쓰면184 strings가 되는 다른 실험이다. Predictor와 symbol 수가 크게 달라 순수 rank 요인 비교가 아니다. Logged encode/decode597.4/348.0ms도 통제된 profile과 구분한다.

구조4개의 전체 test6991장 평가를 제출한 기록은 아직 없다. 현재 파일은 holdout 결과다.

## 6. 현행 RD와 추가 seed

### 주 seed RD64점·16계열

| Family | 읽는 이름 | 완료 점 수 |
|---|---|---:|
| both_off |None+L3 OFF12|5|
| context_only |None+L3 ON12|5|
| hilbert_only |H+L3 OFF12|5|
| hilbert_legacy3 |H+L3 ON12|5|
| context_dyadic4 |None+D4 ON16|5|
| hilbert_dyadic4 |H+D4 ON16|5|
| morton_dyadic4 |M+D4 ON16|5|
| morton_legacy3 |M+L3 ON12|5|
| hilbert_off1 |H+OFF1|5|
| hilbert_off4 |H+OFF4|3|
| hilbert_quarter2 |H+Q2 ON8|3|
| morton_quarter2 |M+Q2 ON8|3|
| context_quarter2 |None+Q2 ON8|3|
| morton_legacy2 |M+L2 ON8|3|
| nn_score |NN-score+L3 ON12|3|
| morton_legacy4 |M+legacy4 ON16|1|

5점 계열은 양의λ `.0032/.0064/.0128/.0256/.0512`,3점은 중앙 `.0064/.0128/.0256`다. M+legacy4 한 점은 `.0256`이며 D4로 읽지 않는다. 미완료 H+D4 OFF16과 새 None+D4 OFF16은64점에 없다.

주 RD 후보는 H+D4다. 기본 H+D4 .0128은 **23.7021/.7363/.2673,37.17881KiB**다. 진단256장의23.87649dB와 다른 표본이다.

λ0는 **24.0734dB,831.6422KiB**의 별도 모델이다. 양자화·entropy coding이 작동하므로 비압축 또는 화질 상한이 아니다. `.0032` 대비+.1769dB에 bytes14.69배다.

### L3 네 곡선의 공통 구간

공통 PSNR **23.0360–23.7288dB**에서5점 PCHIP log-rate 적분, 외삽 없음:

| 후보 / 기준 | BD-rate |
|---|---:|
| None ON / None OFF |−1.2648%|
| H ON / H OFF |−20.4325%|
| H OFF / None OFF |−14.7991%|
| H ON / None ON |−31.3393%|
| H ON / None OFF |−32.2077%|

반올림된 stdout 평균·주 seed 하나의 보간 결과다. 원본 scene/frame paired 검증·CI·seed 유의성까지 한 결과가 아니다. H+L3 .0256의 과거GPU4 예외도 남아 있으므로 완전히 맞춘 학습 조건의 순수 인과 효과로 표현하지 않는다.

### 추가 seed

Model111124/data404,같은 FullSplit 부모에서 새 후속50k,전체6991장:

| Arm | Train / Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---:|---:|---:|---:|
| H+D4 ON16 |457786 /457787|23.7036|.7362|.2669|37.07510|
| H+D4 OFF16 |457788 /457789|23.6310|.7336|.2692|44.76894|

ON은 같은λ에서 bytes17.1857% 감소·PSNR+.0726dB다. 기본 seed OFF16 평가가 끝나야 두 seed의 H ON/OFF 반복이 완성된다. 공통 부모 이후의 학습 랜덤성 반복이며 독립 fresh-init 일반성이나 D4 정렬×Context 2×2 전체의 반복이 아니다.

## 7. 완료된 진단과 해석 제한

| 실험 | Job | 상태·범위 |
|---|---|---|
| L3 네 정렬 실제 anchor/잔차 |454995|완료,4모델×256;원본 context cache |
| H L3/Q2/D4 anchor/잔차 |455166|완료,같은256장 |
| T04 native B0/B1 bridge |458024|완료,matched256 |
| T05 일곱 경로 |458004|완료,matched256 |
| T15 현행 packet-only receiver |458784|완료,T05 packet256개 |
| T06 frozen13순서 |458818|완료,256×slot/Morton/Hilbert/random10 |
| Complexity37 checkpoint |455251|완료,통제된 A5000 profile |

원본 cache fingerprint: `069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29`.

### B0/T05

| 경로 | PSNR |
|---|---:|
| Native B0 geometry512 |24.90270|
| B1 projected F 직접 |16.76771|
| 연속 LR만 |19.80692|
| 실제 복호 Score의 LR만 |18.56430|
| Score 양자화 우회 |23.26808|
| MSH 양자화 우회 |23.92326|
| 양쪽 양자화 우회 |23.31259|
| Actual packet |23.87649|

Actual packet은37.2592KiB다. Actual−native B0는 이256장에서−1.02621dB다. B1은 학습된 projection 이후 codec 복원 변환을 뺀 경로이며 현행 모델의 화질 상한이 아니다. B1 급락만으로 버그를 단정하지 않는다.

MSH 양자화 우회 개선은+.04677dB로 작지만 분석·합성 손실은 남는다. Score 우회는−.60840dB이며 residual을 다시 계산한 fixed checkpoint의 공동 적응을 포함한다. 차이를 더해 경로별 독립 양자화 손실로 만들지 않는다. 연속 우회에 실제 packet rate를 붙이지 않는다.

Native slot-order 검사의 Gaussian field 차이는0이지만 GPU max pixel 차이는.0124051이었다. 평균 PSNR 차이는 약−2.61e−7dB이고 동일 Gaussian 반복 렌더 차이는0이었다. Gaussian 순열 보존과 GPU pixel bit-exact를 구분한다.

### T06/T15

T06은 동일 F/P/basis/Score/XYZ와 Hilbert-trained predictor를 고정했다. Aggregate total KiB는 slot56.67001, Morton37.91431, Hilbert37.25922, random10평균56.72330이다. Hilbert의 slot 대비 전체 bytes 절감34.25%, raw Score 절감44.82%, anchor거리 감소91.23%, 정규화 residual MAE 감소8.75%다. Morton 대비 추가 전체 절감은약1.73%다.

H packet은 모든 장면에서T05와 SHA가 같다. 다른 순서는 분포 변화 진단이고 재학습한 RD가 아니다. 원래 T06에는 렌더가 없었다. Random10은 scene 안에서 평균 후1000회 scene bootstrap이며 training-seed CI가 아니다.

T15는 encoder·context images·원순열 없는 codec+Gaussian decoder 수신 프로세스다. Target camera/shape는 렌더,RGB는 metric에만 쓴다.256장 stdout PSNR delta가0이고 metric 허용치와 반복 feature/Gaussian 검사를 통과했다. 역사적 T05 tensor/pixel은 저장되지 않아 과거 값과의 bit-exact를 주장하지 않는다.

과거 다른 Full nonlinear Split 모델에서는433727/433728의 **전체6991장 독립 encode/decode**도 완료됐다. 이를 누락하거나 현행 H+D4의 전체 test 독립 수신으로 바꾸어 읽지 않는다.

### Complexity

37 checkpoint는 A5000 ariel-v10, BF16,threads8,warmup4+측정32장×3회다. H+L3 .0128 encode/decode150.583/84.279ms, H+D4 163.698/93.299ms, NN-score214.619/84.715ms다. D4는 L3보다8.71%/10.70% 느렸다.

전체 모델이 올라간 harness의 allocated peak는 독립 receiver의 최소 VRAM이 아니다. 새 OFF·구조 profile은 보완 전이며 기존37개 전부를 다시 돌릴 필요는 없다. 새 할당에서는 같은 cache와 reference도 함께 측정한다. [실제37점 보고서](analysis/nfcgs_complexity_455251/REPORT.md).

## 8. 이번 세션에서 새로 구현한 후속5개

### 구성·출처

1. **None+D4 OFF16 .0128**: 기본 seed,기존 FullSplit 부모에서 후속50k. 기존 None ON/H ON/H OFF와 기본 seed D4 2×2를 완성할 한 셀.
2. 위 새 모델의 최종50k **afterok 전체 actual eval**.
3. **N01 CPU audit**: 기존 B0/T05 metadata의 effective readout·feature error·B1 drift. 새 QR 복원·재학습이 아님.
4. **N03 native 전체 test**: H+D4 .0128의 `actual_rate_per_scene.json`6991개 scene과 C12/T8 frame을 맞춰 native만 렌더. 누락 시 다른 scene으로 채우지 않음.
5. **N02 T06 렌더**: 기존 slot/Morton/Hilbert packet256×3,기존T15 target bundle 재사용. 재인코딩·encoder 구성 없음.

N02는 H 품질이 T05 허용치 PSNR1e-3/SSIM1e-5/LPIPS1e-5를 통과해야 완료로 표시한다. 기존 GPU 종류·소프트웨어·수치 정책도 검사한다. Feature MSE는T06에서 재사용하며 전체 Gaussian 역순열 오차를 새로 측정한 것으로 쓰지 않는다.

### 개별 파일·명령

서버 반영 필수3개:

- [scripts/submit_nfcgs_followup_20261020.py](../scripts/submit_nfcgs_followup_20261020.py)
- [scripts/run_nfcgs_followup_20261020.py](../scripts/run_nfcgs_followup_20261020.py)
- [scripts/slurm/run_nfcgs_followup_20261020.slurm](../scripts/slurm/run_nfcgs_followup_20261020.slurm)

검증·안내: [tests/test_followup_20261020.py](../tests/test_followup_20261020.py), [실행 안내](NFCGS_INDEPENDENT_FOLLOWUPS_2026-10-20.md).

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
python scripts/submit_nfcgs_followup_20261020.py --dry-run
python scripts/submit_nfcgs_followup_20261020.py
```

원래26개 launcher를 재실행하는 명령이 아니다. 새 묶음 state는 **`logs/slurm/followup_20261020/state.json`**이다. 접수 ID를 즉시 저장하고 재실행은 저장된 ID를 재사용하며 미접수분만 시도한다. State를 삭제·다른 위치로 옮겨 중복 제출하지 않는다. 이미 접수한 실패 job을 자동 재시작하는 도구는 아니다.

새 stdout는 `logs/slurm/slurm-gs-followup20-<JOB>.out/.err`; 새 학습·연결 eval은 기존 order-context 이름이다. 새 분석 출력은:

```text
outputs/nfcgs_followup_20261020/boundary/job_<ID>/
outputs/nfcgs_followup_20261020/native/job_<ID>/
outputs/nfcgs_followup_20261020/fixed-render/job_<ID>/
```

각각 REPORT/metadata/summary/complete가 있고 GPU 평가는scenes/가 있다. **로컬에서 접수 자료가 없으므로 우선 서버 state와 새 로그를 받는다.**

### 필요한 기존 서버 산출물

```text
B0:      outputs/nfcgs_b0_matched/job_458024
T05:     outputs/nfcgs_path_diagnostics/job_458004
T06:     outputs/nfcgs_fixed_order/job_458818
Targets: outputs/nfcgs_independent_receiver/job_458784/bundle
Native:  checkpoints/pretrained/globalsplat-re10k-32k.ckpt
Reference:
  outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/lambda0p0128/job_447746/
  eval_all/nfcgs_main/actual_rate_per_scene.json
```

공개 native SHA256은 `90be09504fa046792dedfe2d1b17e00bb64a5bf95da477d71e233be01df72a42`다. Dataset index는 compute node의 `/data3/local_datasets/re10k`에서 검사한다. 로그인 노드에 이 index가 없다는 이유만으로 공유 경로를 임의 변경하지 않는다.

이 구현은 새 파일3개와 테스트를 추가한 것으로 기존 codec/model/loader/YAML을 수정하지 않았다. CPU 테스트11개, Bash 구문/LF, 실제 Hydra의 None+D4 OFF16·seed111123/data403·50k·유효 batch8 조합을 확인했다. 실제 entropy packet/최소 수신기/Gaussian decoder를 쓰되 테스트 렌더와 metric은 CPU fixture다. 실제 CUDA 본 평가를 로컬에서 수행했다고 쓰지 않는다.

## 9. 다음 실험 우선순위와 아직 없는 구현

### 먼저 할 일

1. **458783 학습·457100 평가 완료 확인.** 기본/추가 seed H ON/OFF 비교를 마무리한다.
2. **새 후속5개 접수 여부·결과 확인.** State/로그가 없으면 사용자 서버 확인 결과를 받되, 이미 받은 명령의 같은 state를 재사용한다.
3. N01 경계 metadata, native 전체 test, T06 품질을 받아 본문 해석을 보완한다.
4. 구조의 마지막 학습 곡선과 수렴을 확인한 뒤 **MSH-only .0032 / two-path56 .0512**를 먼저 확장한다. 두 점의 실제 rate가 겹친다고 보장하지 않는다. 다음λ는 holdout에서 적응적으로 고르고 test에서 고르지 않는다.

### E03 구현 시 주의

**현재 `scripts/run_nfcgs_architecture_pilot.py`는 λ=.0128을 하드코딩한다.** 단순히 환경변수 RATE_LAMBDA를 바꾸거나 기존 launcher를 재실행해 E03을 돌렸다고 하면 안 된다. 임의로 그 파일을 고쳐 기존 제출 state signature를 깨기보다, 명시적λ·출력·새 state를 갖는 확장 wrapper/도구를 준비하는 방향을 검토한다.

기존 `outputs/nfcgs_architecture_pilots/initialization_20261019/`의 공통 초기화·holdout·validation index와 geometry projection 정책을 유지해야 한다. 원래 fresh .0128의 학습 완료 checkpoint에서 추가λ를 이어 학습한 결과를 새50k fresh-from-common-init와 혼동하지 않는다. 새로운λ의 saved-step/config/provenance·실제 packet roundtrip·검증/전체 test 구분도 확장해야 한다.

### 추가 제안의 순서

검토에서 제안한 조건부 총17학습의 순서는 다음과 같았다. 현재 준비된 것은 이 중 기본 None OFF16 한 점과 독립 N작업뿐이다.

- E03 rate 중첩4 + E01 기본 None OFF1 + E04 fresh 2×2 나머지3 + E02 Morton OFF1 = 우선9개.
- Random decoded-anchor1을 더하면10개.
- E01의 나머지 seed6개를 더하면16개.
- Bias-only1을 더하면17개.

각 단계는 결과에 따른 예산이지 일괄 실행 확정이 아니다. Fresh 2×2를 세 번째 warm-start seed 전체 반복보다 먼저 볼 가치가 있다는 판단이다. Warm/fresh 차이에는 부모뿐 아니라 학습 이력·holdout 제외 정책도 있으므로 부모 초기화만의 순수 인과 효과로 단정하지 않는다.

E07 D4 OFF의 나머지λ, rank112, 재학습 random, 새 LIC/MSH context, identity F736, full-width 그룹 폭 변경은 조건부다.736의 이번 결과로 E09/E14의 우선순위를 낮췄다. 기존 T17 Morton+L3 OFF12와 L3 추가 seed는 제출된 것으로 세지 않는다.

외부 codec 한 개의 공통 subset 평가 가능성도 확인할 가치가 있다. LocoMoco에는 이미 Morton+공간/채널 Context가 있어 ‘정렬+Context 최초’ 주장은 피한다. Scene-token 경계·실제 복호 가능성·RD/비용·이웃 품질 통제에 논문의 근거를 둔다. [외부 평가 프로토콜 조사](NFCGS_EXTERNAL_EVAL_PROTOCOLS_2026-10-19.md)의 서로 다른 C/T·데이터·해상도 조건을 먼저 맞춘다.

사용자 일정의11월1일 새 구조/설정 동결·11월10일 제출 목표를 참고한다. 모든 선택 실험을 자동으로 추가하기보다 주장의 빈칸을 메우는 결과와 원고를 병행한다.

## 10. 과거 이력에서 누락하면 안 되는 것

- [전체 실험 목록](NFCGS_EXPERIMENTS_MASTER_2026-10-20.md)에 초기 rank/linear/nonlinear/residual/order,과거 context48평가,기타 CleanToken/Hyper1D 분기,실패·재개,파일별 job ID를 모았다. 이 목록 본문은16:47 당시 집계이며 새64점/736은 최신 inventory가 우선이다.
- 초기 Experiment17은 기존 full-trained 모델의 누적 경로 우회였고 독립 학습 구조 비교와 다르다. ‘과거 경로 진단이 전혀 없었다’고 하지 않는다.
- 과거48평가의 원본 stdout는 현재 로컬에 없다. 보존 JSON에서 가져온 값이며 이번에 raw log를 재검산한 것으로 쓰지 않는다. 초기 사용자 표12개는 원래 kB 단위를 보존했다.
- CleanToken/Hyper1D는 다른 repository·레시피다. 현재 fresh MSH-only/two-path pilot과 matched 비교로 합치지 않는다.
- Mean-offset433719_0/1과434543_0/1은 ENOSPC 등으로 완료 비교가 없다. 현재 학습 중으로 세지 않는다.
- 과거 ENOSPC451821/451825/451827은453064/453066/453068 후속 완료점과 연결한다. NN-score450904 실패와450934/450935 최종 완료도 독립 모델로 중복 세지 않는다.
- Model·scene/frame·λ·seed·native/pilot/full-test·평가 범위가 다른 수치를 한 RD 곡선에 섞지 않는다. Scene bootstrap CI는 training-seed CI가 아니다.

## 11. 구현 위치·로컬 작업 팁

- Codec: `globalsplat/compression/{architecture,config,codec,checkpoint,score_context}.py`.
- Fresh 공통 초기화: `scripts/prepare_nfcgs_architecture_pilots.py`; 실행/검사: `run_nfcgs_architecture_pilot.py`, `smoke_nfcgs_architecture_pilot.py`.
- Holdout: `globalsplat/dataset/holdout.py`, `data_module.py`, `re10k.py`.
- 복구: `scripts/recover_nfcgs_paper_controls_20261020.py`, `scripts/slurm/resume_nfcgs_paper_controls_20261020.slurm`.
- 진단: `scripts/eval_nfcgs_b0_matched.py`, `eval_nfcgs_path_diagnostics.py`, `eval_nfcgs_independent_receiver.py`, `analyze_nfcgs_fixed_order.py`.
- Slurm checkpoint/임시 파일은 `/ceph_data/<USER>/.tmp/nfcgs/` 아래를 사용한다. 기존 TMPDIR 수정은 유지한다.
- 서버 state의 로컬 다운로드는 루트 `slurm/architecture_pilots_20261019/state.json`, `context_seed_pair_20261019/state.json`, `paper_controls_recovery_20261020/state.json`에 있다. 새 `followup_20261020/state.json`은 아직 없다.
- PowerShell, 검사 Python은 `C:/Users/a0103/miniconda3/envs/nfcgs-local-profile/python.exe`; Torch CPU/Hydra/CompressAI가 있다. 기본 Python은 환경이 다를 수 있다.
- Bash는 `C:/Program Files/Git/bin/bash.exe`. 새 shell/Slurm은 LF·유효 shebang·BOM 없음을 유지한다.
- 한글 문서는 `apply_patch`로 쓴다. PowerShell here-string→Python stdin의 한글 literal은 깨질 수 있어 Python snippet은 ASCII를 사용한다.
- `rg`를 먼저 사용한다. 방대한 training progress는 `splitlines()`로 마지막 완전한 줄만 추출한다. Console cp949에서 progress block 문자를 출력하다 실패하면 UTF-8을 지정하거나 숫자/ASCII만 출력한다.
- 새 runtime 변경이 없으면 무거운 전체 회귀를 반복하지 않는다. 새 기능의 실제 entropy/인과성/접수 중단·재실행 등 필요한 검사를 수행한다.

## 12. 다음 세션의 첫 행동

1. 이 문서와21:59 REPORT·review·inventory를 읽는다.
2. 루트 `slurm/`에서21:59 이후 파일·내용이 갱신됐는지 확인한다. 특히458783/457100과새followup state를 우선한다.
3. 새 평가의 최종 checkpoint step·설정·scene count·payload 합계·오류를 확인하고 새 snapshot에 저장한다.
4. 기본/추가 seed D4 ON/OFF 비교를 갱신하고 새 N01/N02/N03 결과를 해석한다.
5. 사용자의 다음 실행 지시에 따라 E03 추가λ 도구를 구현한다. 기존 .0128 runner의 하드코딩과 공통 초기화/holdout을 먼저 확인한다.
6. 변경 파일을 개별 링크와 정확한 서버 명령으로 전달한다. 새 실험 준비를 이미 접수·완료한 실험으로 표시하지 않는다.

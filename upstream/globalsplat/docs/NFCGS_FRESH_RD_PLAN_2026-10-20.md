# Fresh-init 권장 재학습 계획 (1 GPU/job) — 2026-10-20

2026-10-21 실행 변경: 사용자 요청에 따라 Gate 0는 선택 진단으로 분리했다. 선행 테스트 없이 `--wave 1`로 본학습을 접수하며, 아래 원안의 Gate 0 선행 조건보다 [실행 안내](NFCGS_FRESH_RD_RUNBOOK_2026-10-20.md)의 최신 명령을 우선한다.

2026-10-22 실행 변경: 사용자 승인으로 Wave 1 Stage B를 `--budget 100000 --provisional`로 각 Stage A의 afterok에 미리 연결할 수 있다. Gate 1은 그대로 유지하며, 확정 전에는 test를 접수하지 않는다. 100k 결정 시 같은 job을 재사용하고, 연장 결정 시 임시 Stage B만 종료한 뒤 원본을 보존한 별도 budget 분기에서 동일60k full state로 재개한다. [실행 안내](NFCGS_FRESH_RD_RUNBOOK_2026-10-20.md)의 임시 Stage B 절을 우선한다.

2026-10-28 계획 추가: 경로별 Context·정렬 ablation, Score56-only 저rate 점, 정렬 축 분리(NN-xyz·NN-score), λ0 decoder finetune까지 학습 9개를 더한다(§13). 이어 Adaptive integer K(Top-K) 학습 6개를 더한다(§14). 35개와 합쳐 50개다.

논문 본문에 들어갈 결과를 **minimal codec을 λ마다 fresh init에서 독립 학습**하는 단일 프로토콜로 다시 만든다. GPU 16개, 1 GPU/job 기준이다.

진행 방식:

1. **짧은 Gate 0** — 경계·복호 정합, native 기준, 재개 동작을 먼저 확인한다(예상 2–3시간).
2. **Wave 1 Stage A** — 학습 15개를 60k까지 돌리고, 남은 GPU 1개로 중간 checkpoint를 holdout에서 평가한다.
3. **Gate 1** — 60k까지의 holdout 결과로 학습 길이(budget B)를 정한다.
4. **Stage B와 Wave 2** — wave 1을 60k에서 B까지 이어 학습하고, wave 2(20개)를 Stage A→B로 돌린다.

총 35개다. 실행 도구를 구현하고 로컬에서 검증했다. 서버 job은 제출하지 않았다. 실제 명령과 구현 범위는 [실행 안내](NFCGS_FRESH_RD_RUNBOOK_2026-10-20.md)를 따른다. Gate 1 뒤에는 wave 1 Stage B만 제출하고 wave 2는 Gate 2 최종 holdout 검토 뒤 제출한다. 아래 원안의 일정은 서버 처리량과 변경된 순서로 갱신해야 한다.

## 1. 다시 학습하는 이유

- 기존 warm-start 계보의 RD 점은 같은 부모에서 출발한다: `nfcgs_score_probability10k/.../lambda0p0256/.../step000010000.ckpt`. λ0·.0032·.0064·.0128·.0512 모두 이 λ=.0256 FullSplit 부모를 minimal로 변환한 뒤 50k를 학습했다(로컬 train 로그 39개 확인).
- 변환은 `score_context.*`만 다시 쓰고 centering·score norm·MLP 플래그를 끈다. basis, residual 통계, MSH, P는 부모 값 그대로다([checkpoint.py](../globalsplat/compression/checkpoint.py) `convert_score_path_state`).
- 고화질 끝점이 포화한다. H+D4 λ0는 831.6KiB를 쓰고도 24.07dB다. 같은 backbone·평가 프로토콜에서 9월 Nonlinear32 .0064(PCA init 50k)는 98KiB에 24.51dB였다. 다만 9월 모델과는 구조, rank-56 basis 초기화(PCA vs fresh의 random orthogonal), 학습량이 함께 다르므로 부모만의 효과로 특정하지 않는다.
- 부모가 Morton 순서로 학습됐으므로 None arm이 불리했을 수 있다는 가설이 있다. fresh 비교로 계보 의존성을 확인하되, 기존 결과가 불공정했다고 확정하지 않는다.
- 24.99dB를 낸 9월 λ.0004 checkpoint에서도 P가 62% drift했고 `W224·P`와 W512의 상대 오차가 29–64%였다. 따라서 P drift만으로 저화질을 설명할 수는 없다. 현행 경계가 정합하다는 검증은 아니며, Gate 0에서 따로 확인한다.

fresh 재학습은 계보 의존성을 줄여 본문 비교를 통일하는 방법이다. generator 대비 gap의 원인을 그 자체로 규명하는 실험은 아니다.

기존 결과(양의λ 64점, λ0, seed pair)는 **"warm-start 계보 예비 결과"**로 보관하고 최종 표·곡선에 섞지 않는다.

## 2. 논문 주장 범위

| 구분 | 위치 | 근거 |
|---|---|---|
| 정렬이 공간 Context의 효용을 높인다 | **핵심 주장** | fresh 2×2 RD(5λ) + .0128 seed 3개 |
| 왜 Hilbert와 D4를 골랐는가 | 설계 선택 설명 | 같은 프로토콜의 .0128에서 Morton+D4 ON, Hilbert+L3 ON과 비교. 용량·화질·시간을 함께 보인다 |
| Hilbert·D4가 전반적으로 더 우수하다 | 주장하지 않음 | 중앙 λ 한 점뿐이다 |
| Low-rank의 필요성·최적성 | 주장하지 않음 | Score736-only는 기존 fresh pilot(457828, 50k·2 GPU)으로 남긴다. 100k 최종 모델과 나란히 둘 때 조건을 분리해 표시한다 |

## 3. 공통 프로토콜

| 항목 | 설정 |
|---|---|
| Backbone | 공개 `globalsplat-re10k-32k.ckpt`, 고정 |
| 초기화 | architecture pilot의 공통 fresh 초기화: QR geometry + 새 codec(rank-56 basis는 random orthogonal). seed별로 따로 준비(§6.1) |
| Codec | minimal: centering·score norm·MLP·mean/channel context OFF, linear, rank56, MSH N192/M320, Score STE |
| 학습 데이터 | RE10K train − holdout128 (`initialization_20261019/holdout.json`, 구조 pilot과 같음) |
| View recipe | paper24: context pool 24 → 13-view branch 2개, target 12, subset consistency ON |
| GPU·batch | 1 GPU, micro 2 × accumulate 4 = effective 8, workers 8 |
| Optimizer | Adam 1e-4, weight decay 0, grad clip 0.5, bf16-mixed |
| Stage A | 0→60k. LR schedule은 처음부터 100k 기준 milestone 70k/90k로 둔다. 60k에서 끝나므로 Stage A 동안 LR은 1e-4로 일정하다 |
| Stage B | 60k checkpoint에서 full-state 재개해 B까지 학습. B=100k면 그대로 재개한다. B=120k(84k/108k)나 150k(105k/135k)면 저장된 scheduler의 milestone을 0.7B·0.9B로 바꿔 재개한다 |
| 데이터 순서 | `checkpointing.exact_resume=true`를 step 0부터 켠다. 재개해도 중단 없이 돈 것과 같은 샘플 순서를 따른다 |
| Quantile | deterministic update, 500 step 간격 |
| 로그 | `trainer.log_every_n_steps=1`(기존 20). LR도 TensorBoard에 기록 |
| Checkpoint | 10k마다 + last. 60k가 Stage A의 끝이자 Stage B의 시작점 |
| Seed | s0 = model 111123 / data 403, s1 = 111124 / 404, s2 = 111125 / 405. 초기화 seed도 같은 값을 쓴다 |
| 결정론 플래그 | 학습 OFF(기존 main 실험과 같음), 평가 ON(`++architecture_entropy_determinism=true`, 모든 arm 공통) |
| 최종 평가 | full test 6,991장, C12/T8, actual bitstream, eval seed 0. Stage B 성공에만 연결하고 보고용으로만 쓴다 |

- B는 모든 학습에 같게 적용한다. 예외는 rate penalty 없는 진단인 λ0 하나뿐이다(§8 Gate 1).
- 학습 길이와 진행 여부는 holdout128 결과로만 정한다. full test는 결정에 쓰지 않는다.
- 1-GPU 학습과 2-GPU 학습을 섞지 않는다.

## 4. 학습 목록 (35개)

Cell 정의 (L3를 뺀 나머지는 D4 = dyadic4·4단계·kernel 5·16 streams):

| Cell | 정렬 | Context | 초기화 파일 | 비고 |
|---|---|---|---|---|
| H-ON | Hilbert | D4 ON | `two_path56.ckpt` | 최종 모델 |
| H-OFF | Hilbert | D4 OFF (`--no-spatial-context`) | `two_path56.ckpt` | |
| N-ON | None | D4 ON | `two_path56.ckpt` | |
| N-OFF | None | D4 OFF | `two_path56.ckpt` | |
| M-ON | Morton | D4 ON | `two_path56.ckpt` | 설계 선택 비교 |
| H-L3 | Hilbert | L3 ON12 (legacy·3단계·kernel 5) | `two_path56.ckpt` | 설계 선택 비교 |
| S56 | Hilbert | D4 ON | `score56_only.ckpt` | Score56-only, MSH 없음 |
| MSH | Hilbert | 없음 | `msh_only.ckpt` | MSH-only, rank 0 |

two-path cell은 같은 `two_path56.ckpt`를 `--allow-score-path-conversion`으로 읽어 정렬·Context 플래그만 바꾼다. 같은 모양의 tensor는 그대로 복사되므로 공통 tensor가 동일하다. 구현 후 실제로 load해 hash를 대조한다. H-L3는 legacy 전용 모듈의 초기값 출처를 확인해 provenance에 남긴다(§6.1).

### Wave 1 — 학습 15개 + monitor (Gate 0 직후 Stage A 시작)

| # | Cell | λ | Seed |
|---:|---|---|---|
| 1–4 | H-ON, H-OFF, N-ON, N-OFF | .0032 | s0 |
| 5–8 | H-ON, H-OFF, N-ON, N-OFF | .0128 | s0 |
| 9–12 | H-ON, H-OFF, N-ON, N-OFF | .0512 | s0 |
| 13 | H-ON | 0 | s0 |
| 14 | S56 | .0128 | s0 |
| 15 | MSH | .0032 | s0 |
| GPU 16 | monitor | — | — |

S56과 MSH를 wave 1에 넣어 구조 arm의 수렴과 MSH의 rate 구간도 Gate 1에서 함께 본다. monitor는 학습이 아니라 중간 checkpoint를 holdout128에서 평가하는 job이다(§6.5).

### Wave 2 — 20개 (Gate 1에서 B를 정한 뒤 Stage A→B로 제출)

| # | Cell | λ | Seed | 비고 |
|---:|---|---|---|---|
| 16–17 | H-ON | .0064, .0256 | s0 | 최종 모델 5λ 완성 |
| 18–23 | H-OFF, N-ON, N-OFF | .0064, .0256 | s0 | 2×2 네 곡선 5λ 완성 |
| 24–27 | H-ON, H-OFF, N-ON, N-OFF | .0128 | s1 | seed 반복 |
| 28–31 | H-ON, H-OFF, N-ON, N-OFF | .0128 | s2 | seed 반복 |
| 32 | M-ON | .0128 | s0 | 설계 선택 비교 |
| 33 | H-L3 | .0128 | s0 | 설계 선택 비교 |
| 34–35 | MSH | 기본 .0064, .0128 | s0 | wave 1 MSH .0032의 holdout rate를 보고 조정 |

- MSH의 λ: 기존 pilot에서 MSH-only .0128은 14.5KiB로 two-path보다 훨씬 작았다. .0032의 holdout rate도 two-path 구간보다 낮으면, 34–35를 더 낮은 λ(.0016 등, 허용 목록 추가 필요)나 0으로 바꾼다.
- B = 150k로 정하면 seed 반복 8개(24–31)를 빼서 27개로 줄일지, Gate 1에서 다른 항목과 우선순위를 비교해 정한다.

## 5. 일정 (GPU 16개)

| 학습 1개 (1 GPU) | Stage A (0→60k) | Stage B, 100k | Stage B, 120k | Stage B, 150k |
|---|---:|---:|---:|---:|
| H-ON, N-ON, M-ON, S56 | 37.5h | 25.0h | 37.5h | 56.3h |
| H-OFF, N-OFF | 36.1h | 24.1h | 36.1h | 54.2h |
| H-L3 | 35.1h | 23.4h | 35.1h | 52.6h |
| MSH | 30.6h | 20.4h | 30.6h | 45.9h |

- 재시작에 job당 약 10분이 더 든다. full test 평가는 학습마다 약 0.9h, holdout128 평가는 checkpoint 하나에 약 5분이다.
- 근거: 2-GPU 50k 실측(D4 ON16 18.5h, OFF16 17.8h, L3 17.3h, MSH-only 15.1h)에 과거 1-GPU 로그의 속도비 1.69를 곱했다. 1-GPU 값은 다른 codec 설정에서 잰 것이라 ±15% 오차를 둔다. 첫 실제 처리량으로 갱신한다. 대기열·구현·장애·추가 진단 시간은 빠져 있다.

100k로 진행하는 경우 (Gate 0 시작 기준):

| 시점 | 상태 |
|---|---|
| 0 | Gate 0 (예상 2–3시간, 여러 GPU에서 병렬) |
| 약 0.1일 | wave 1 Stage A 15개 + monitor 시작, s1·s2 초기화(CPU job) |
| 약 1.4일 | MSH .0032 Stage A 종료 |
| 약 1.7일 | 나머지 Stage A 종료, 60k holdout 평가 → **Gate 1**: B 확정, wave 1 Stage B와 wave 2 제출 |
| 약 2.8일 | wave 1 최종 → holdout 평가 → **Gate 2** |
| 약 6.7일 | 전체 완료 (test 평가 포함) |

budget별 비교:

| B | 학습 수 | wave 1 최종 | 전체 완료 | 10월 22일 Gate 0 시작 시 |
|---|---:|---:|---:|---|
| 100k | 35 | 2.8일 | 6.7일 | 10월 29일 |
| 120k | 35 | 3.3일 | 7.8일 | 10월 30일 |
| 150k | 27 (seed 제외) | 4.1일 | 8.1일 | 10월 30일 |

- wave 2를 Gate 1 뒤에 시작하면 Stage A가 끝난 GPU가 Gate 1까지 몇 시간 쉰다. 미리 투입하면 전체가 약 0.2일 빨라지지만 wave 1 최종 결과가 약 1.5일 늦어지므로, 이 계획은 Gate 1 뒤 시작을 택한다.
- 한 job은 최대 약 2.3일(Stage B 150k)이라 slurm 한도 6일 안에 든다.
- 100k 35개의 학습 총량은 약 2,117 GPU-h다.

## 6. 제출 전에 구현할 것

현재 스크립트로는 바로 돌릴 수 없다. architecture pilot runner는 λ .0128·Hilbert·2 GPU·50k가 고정이고 학습에도 결정론 플래그를 켠다. `run_nfcgs.build_command`는 학습 milestone을 `[35000,45000]`으로 고정한다.

1. **초기화**
   - [prepare_nfcgs_architecture_pilots.py](../scripts/prepare_nfcgs_architecture_pilots.py)에 `--seed`를 추가한다. 공통 codec state, 모델 생성, provenance의 seed를 함께 바꾸고, holdout 선택 seed(0)는 유지한다.
   - s0: 기존 `outputs/nfcgs_architecture_pilots/initialization_20261019/`을 재사용한다.
   - s1·s2: `initialization_fresh_rd_s1/`, `..._s2/`를 새로 만든다. holdout.json은 s0와 sha256이 같아야 하고, 필요한 초기화 파일은 `two_path56.ckpt` 하나다.
   - [prepare_nfcgs_architecture_pilots.slurm](../scripts/slurm/prepare_nfcgs_architecture_pilots.slurm)이 `INIT_SEED`와 `ARMS` 환경변수를 받아 `--seed`, `--arms`로 넘기게 한다. 둘 다 없으면 지금과 똑같이 동작해야 한다.
   - M-ON·H-L3를 `two_path56.ckpt`에서 변환해 load할 수 있는지 확인한다. H-L3에서 legacy 전용 모듈이 checkpoint에 없으면 어디서 초기화되는지 기록하고, 필요하면 해당 arm의 초기화 파일을 따로 만든다.
2. **runner** `scripts/run_nfcgs_fresh_rd.py` (신규)
   - 인자: `--cell {h_on,h_off,n_on,n_off,m_on,h_l3,s56,msh}`, `--rate-lambda`, `--seed-index {0,1,2}`, `--stage {A,B}`, `--budget {100000,120000,150000}`, `--preparation`, `train|validation|test`, validation용 `--step`.
   - `run_nfcgs.build_command`로 명령을 만든 뒤 다음을 바꾼다. 중복 override는 기존 pilot runner처럼 **기존 항목을 지우고** 추가한다.
     - Stage A: `trainer.max_steps=60000`, `optimizer.lr_milestones=[70000,90000]`
     - Stage B: `step000060000.ckpt`에서 full-state 재개, `trainer.max_steps=B`. B≠100k면 `optimizer.lr_milestones=[0.7B,0.9B]`와 `checkpointing.override_lr_milestones=true`
     - 공통: `checkpointing.exact_resume=true`, `checkpointing.every_n_train_steps=10000`, `trainer.log_every_n_steps=1`, `seed=…`, `dataset.data_loader_seed=…`, `++dataset.train_holdout_manifest=…/holdout.json`, `trainer.limit_val_batches=0`
     - S56/MSH: `++model.feature_codec.codec_mode`, `model.feature_codec.rank`, `model.feature_codec.use_residual` (pilot runner와 같은 값)
     - 결정론 플래그는 평가 명령에만 넣는다.
   - two-path cell은 `--allow-score-path-conversion`을 붙이고 `two_path56.ckpt`를 읽는다.
   - Stage A 전 검사: 초기화 provenance와 sha256, `global_step == 0`, optimizer state 없음.
   - Stage B 전 검사: 같은 run의 60k checkpoint, 설정 일치, milestone 변경 여부.
   - 평가 전 검사: Hydra config 일치(seed pair runner의 `validate_training_config` 방식), `global_step`이 요청한 step(validation) 또는 B(test)와 같은지, 장면 수(test 6,991 / validation 128).
3. **LR 기록과 milestone 덮어쓰기**
   - 새 학습에서만 `LearningRateMonitor(logging_interval="step")`를 켠다. 설정 플래그로 두어 기존 실험 경로는 바꾸지 않는다.
   - `checkpointing.override_lr_milestones=true`: full-state 재개 때 `on_load_checkpoint`에서 **checkpoint에 저장된 scheduler state의 milestone 자체**를 설정값으로 바꾼다. 모델이나 scheduler 객체의 속성만 바꾸면, 복원 과정에서 checkpoint 값으로 덮어써진다. 재개 step이 새 첫 milestone보다 작을 때만 허용한다.
   - 이미 LR이 줄어든 scheduler에서 LR만 다시 올리는 방식은 쓰지 않는다. 연장은 항상 60k checkpoint에서 재개하는 방식으로 한다.
   - 테스트: 짧은 합성 학습으로 "60k에서 재개해 milestone 변경"을 축소 재현하고, optimizer LR과 scheduler 위치가 기대한 step에서 바뀌는지 확인한다.
4. **평가 출력** — 이미지별 MSE를 직접 집계해 기록한다. 평균 PSNR에서 역산하지 않는다. 장면별 실제 bits, PSNR, SSIM, LPIPS, MSE를 장면별 JSON으로 남겨 paired 비교에 쓴다.
5. **monitor** `scripts/monitor_nfcgs_fresh_rd.py` + `monitor_nfcgs_fresh_rd.slurm` (1 GPU, 신규)
   - 시작할 때 기준점을 holdout128에서 한 번 평가한다: native GlobalSplat, 계보 H+D4 ON16 5λ와 λ0.
   - wave 1 run 폴더를 주기적으로(예: 10분마다) 확인해, 새로 생긴 10k~60k checkpoint를 holdout128에서 평가한다.
   - 지킬 것: checkpoint 식별(경로·크기·mtime 또는 hash), load 실패 시 다음 주기에 재시도, 128장 완료 확인, 중복 평가 방지, CSV 원자적 갱신.
   - 결과를 `outputs/nfcgs_fresh_rd/convergence/holdout_curve.csv`에 누적한다: cell, λ, seed, step, KiB, PSNR, SSIM, LPIPS, MSE, J_h, 장면 수. Gate 1 판정값(§8)과 장면별 paired SE도 함께 계산한다.
   - wave 1의 60k 평가가 모두 끝나면 종료한다. 같은 스크립트를 Gate 2(최종 checkpoint)와 Gate 3(부록 곡선)에 다시 쓴다.
6. **Gate 0 검사** — §8 Gate 0의 네 항목을 실행하는 짧은 job들. 기존 B0·path diagnostics 스크립트에 checkpoint 지정 옵션을 더하고, eval forward 비교를 추가한다.
7. **Slurm과 제출**
   - `train_nfcgs_fresh_rd.slurm`: `--gres=gpu:normal:1`, `--ntasks=1`, `--cpus-per-task=8`, `--mem-per-gpu=48G`, `-t 6-00:00:00`, Ceph job별 `TMPDIR`(기존 ENOSPC 대응과 같음).
   - `eval_nfcgs_fresh_rd.slurm`: 1 GPU, `-t 24:00:00`, `test` 또는 `validation`.
   - `submit_nfcgs_fresh_rd.sh`
     - `--gate0`: Gate 0 job들을 제출한다.
     - `--wave 1`: Stage A 15개 + monitor를 제출한다.
     - `--stage-b --wave 1 --budget B`: wave 1 Stage B 15개를 제출하고 각각에 test eval을 `afterok`로 연결한다.
     - `--wave 2 --budget B [--msh-lambdas …]`: 20개를 Stage A → Stage B → test eval 순서로 `afterok` 연결해 제출한다.
     - job별 state 파일로 중복 제출을 막고, `--dry-run`을 지원한다.
8. **실패한 job 재개** — 같은 run 폴더의 `last.ckpt`에서 full state로 재개한다. 원래 다음 단계 job(Stage B 또는 eval)은 새 job에 `afterok`로 다시 연결한다. 재개 여부는 run 메타데이터에 기록한다.
9. **로컬 검사** — 조합별 Hydra dry-run 명령 생성, cell별 초기화 load와 hash 대조(CPU smoke, `smoke_nfcgs_architecture_pilot.py` 재사용), override가 한 번씩만 들어갔는지, milestone 덮어쓰기 테스트, 가짜 checkpoint로 monitor dry-run, submit 스크립트의 `bash -n`과 LF.

## 7. 서버 실행 순서 (구현 후)

아래는 원안의 실행 흐름이다. 구현된 실제 실행 순서와 파일 반영 목록은 [실행 안내](NFCGS_FRESH_RD_RUNBOOK_2026-10-20.md)를 따른다. Wave 2는 Gate 2 뒤에 제출한다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
mkdir -p logs/slurm

# 1) s1·s2 초기화 (CPU job, s0는 기존 디렉터리 재사용). wave 2 제출 전까지만 끝나면 된다.
for s in 1 2; do
  sbatch --export=ALL,REPO_DIR=$PWD,DATASET_ROOT=/data3/local_datasets/re10k,\
VANILLA_CHECKPOINT=$PWD/checkpoints/pretrained/globalsplat-re10k-32k.ckpt,\
PREPARATION_DIR=$PWD/outputs/nfcgs_architecture_pilots/initialization_fresh_rd_s$s,INIT_SEED=11112$((3+s)),ARMS=two_path56 \
    scripts/slurm/prepare_nfcgs_architecture_pilots.slurm
done

# 2) 계획 확인 (제출 없음)
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --dry-run

# 3) Gate 0 (예상 2–3시간). 이상이 없으면 바로 다음 단계
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --gate0

# 4) wave 1 Stage A + monitor
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --wave 1

# 5) 약 1.7일 뒤 수렴 표 확인 (Gate 1)
column -s, -t outputs/nfcgs_fresh_rd/convergence/holdout_curve.csv | less -S

# 6) B 확정 후 wave 1 Stage B 제출 (예: B=100k)
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --stage-b --wave 1 --budget 100000

# 7) wave 1 최종 holdout을 확인한 Gate 2 뒤 진행 결정 기록
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --gate2 proceed --budget 100000 \
  --reason '실제 holdout 결과를 검토한 진행 사유'
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --wave 2 --budget 100000
```

TensorBoard는 로그인 노드에서 `tensorboard --logdir outputs/nfcgs_fresh_rd --port 6006`을 띄우고, 로컬에서 `ssh -L 6006:localhost:6006 <서버>`로 접속해 본다.

출력: `outputs/nfcgs_fresh_rd/<cell>/lambda<tag>/s<k>/job_<TRAIN_ID>/`. 최종 checkpoint는 `checkpoints/nfcgs_main/version_0/step<B>.ckpt`, test 결과는 `eval/nfcgs_main/scores_all_avg.json`, TensorBoard는 `tensorboard/nfcgs_main/version_0/`이다.

## 8. 판단 지점

### Gate 0 — 짧은 사전 검사 (예상 2–3시간)

이상이 없으면 바로 wave 1을 시작한다. 이상이 발견될 때만 원인을 좁히고 고친 뒤 시작한다. 각 항목은 서로 다른 GPU에서 병렬로 돌린다.

| 항목 | 내용 | 기대 |
|---|---|---|
| G0-1 경계 | fresh 초기값(`two_path56.ckpt`, step 0)에서 codec을 거치지 않고 P·g를 decoder에 넣은 결과와 native를 같은 장면 소규모로 비교 | 거의 같음. 9월 exp17에서는 full test −0.063dB |
| G0-2 eval forward | 계보 H+D4 .0128 checkpoint와 실제 장면 몇 개로 eval forward(`training=False`)와 실제 compress→decompress를 단계별로 대조: Score, LR, residual 입력, MSH y·z와 복원, 최종 feature, Gaussian, PSNR | 수치 오차 수준의 차이 |
| G0-3 native 기준 | 현재 코드·BF16으로 native를 full test 1회, holdout128 1회 측정 | 논문 기준선과 Gate 1·2의 holdout 기준 |
| G0-4 재개 | 같은 설정의 짧은 연속 학습(예: 0→2k)과 Stage A/B 분할(0→1k, 1k→2k)을 비교 | 샘플 순서는 같음. 학습 수치의 완전 일치는 별도로 기록 |

- T05는 packet의 경로별 복호 합이 전체 복호와 같은지를 확인했을 뿐이다. eval forward와의 일치는 G0-2에서 처음 확인한다.
- MSH는 학습 때 additive noise, 실제 복호 때 rounding을 쓴다. learned compression의 표준 방식이지만, 이 차이가 화질에 주는 영향은 측정하지 않았다. T05의 MSH 양자화 우회(+0.047dB)는 고정 모델에서 추론 양자화를 뺀 효과라, 학습 방식의 효과와는 다른 비교다.

### Gate 1 — 수렴 판정 (60k, 약 1.7일)

모든 run이 60k에서 멈추므로 판단 마감은 없다. Stage B 제출이 늦어지는 만큼 GPU가 쉰다.

주 지표는 **holdout actual-RD 목적함수**다.

```text
J_h = MSE + 0.05·LPIPS + λ·bpga_actual
  MSE         : 이미지별 MSE의 평균 (PSNR에서 역산하지 않음)
  bpga_actual : 실제 bits / (32,768 Gaussians × 59)

ΔdB_J = −4.343 × (J_h,60k − J_h,40k) / MSE_40k
```

- ΔdB_J는 J_h의 변화가 MSE에서만 왔다고 할 때 같은 변화를 내는 PSNR 변화(dB)다.
- 장면별 차이의 표준편차로 paired SE를 계산한다. 2·SE보다 작은 변화는 개선으로 보지 않는다.
- J_h는 학습 총목적함수와 같지 않다. 학습 쪽에는 추정 rate, subset consistency, in-view 항이 있고 LPIPS 구현도 평가와 다를 수 있다. TensorBoard `train_loss`는 따로 기록해 보조 지표로 본다.
- PSNR, LPIPS, KiB를 각각 함께 본다. 실제 rate와 추정 rate(`train_codec_estimated_bpga`)의 차이도 확인한다.
- 계보 곡선 기울기 s로 환산한 ΔPSNR_eq = ΔPSNR − s·ln(KiB₆₀ₖ/KiB₄₀ₖ)는 참고용 근사로만 쓴다. s는 .0032 0.4, .0064 0.5, .0128 0.6, .0256 0.8, .0512 1.5다.

2×2 학습 중 가장 느린 run을 기준으로 판정한다.

| 40k→60k ΔdB_J | 결정 |
|---|---|
| 모두 0.05dB 미만 | B = 100k |
| 가장 큰 값이 0.05–0.1dB | B = 100k (마감 고려) |
| 0.1dB 이상인 run이 있음 | 연장: 기본은 B = 120k. 0.2dB 이상이면 150k와 seed 축소를 검토 |

- **λ0**: rate penalty 없는 진단이다. 양자화와 손실 변환이 남아 있고 학습 목적에 LPIPS도 있어서 순수 PSNR 상한은 아니다. λ0만 느리면 λ0만 연장하고 그 사실을 명시한다.
- **S56·MSH**: 이 둘만 느리면 B는 2×2 기준으로 정한다. 구조 비교는 "같은 budget에서의 비교"로 보고하고 해당 arm의 미수렴 가능성을 명시한다. 공통 budget의 공정성과 구조별 충분 수렴은 구분한다.
- **MSH λ**: MSH .0032의 60k rate를 two-path 60k rate와 비교해 wave 2의 MSH λ(34–35)를 정한다.
- **이상 확인**: 같은 holdout에서 native와 계보 checkpoint를 fresh 60k와 나란히 본다. 60k는 LR 감소 전이라 최종보다 낮은 게 정상이므로 발산이나 rate 이상만 확인한다. 계보 모델은 이 128장을 학습에 썼을 수 있다.
- 60k까지의 추세는 budget 결정을 돕지만 최종 수렴을 입증하지는 않는다. LR 감소 뒤의 곡선은 Gate 3에서 확인한다.

### Gate 2 — wave 1 최종 (100k 기준 약 2.8일)

wave 1 최종 checkpoint를 holdout128에서 평가해 같은 holdout의 계보 checkpoint와 비교한다. 결정은 holdout으로만 한다. 같은 시점에 나오는 full test 결과는 보고용이다.

| 결과 (holdout, 계보 대비) | 지지되는 해석 | 조치 |
|---|---|---|
| fresh 고화질 쪽이 더 좋음 | 새 초기화·학습 프로토콜이 개선에 기여했을 가능성. 부모만의 효과로 특정하지 않음 | 그대로 진행 |
| 비슷함 | 이 recipe·budget에서 비슷한 결과. minimal 구조 ceiling의 증명은 아님 | 그대로 진행 |
| 낮음 | 초기화·최적화·수렴·설정·rate 위치를 조사. 학습 부족 하나로 단정하지 않음 | 원인을 좁힌 뒤 wave 2 진행 여부 결정(§10) |

2×2는 각 λ에서 정렬 전후 Context의 bytes·화질 변화와 비례 rate 상호작용 `I_R = log R(H,ON) − log R(H,OFF) − log R(N,ON) + log R(N,OFF)`를 화질 변화와 함께 본다.

### Gate 3 — 전체 완료

- **핵심 주장:** 네 곡선의 5λ RD와 공통 화질 구간의 BD-rate. 공통 구간이 좁으면 측정점과 지배 관계를 우선 보고한다. .0128 2×2의 seed 3개는 seed별 점, 평균, 범위로 보고하고, seed 변동과 scene bootstrap을 섞지 않는다.
- **설계 선택:** .0128에서 H-ON, M-ON, H-L3의 용량·화질·encode/decode 시간. 결과가 비슷해도 선택 근거로 서술한다.
- **구조 비교:** two-path(H-ON 5λ), Score56-only(.0128), MSH-only(3λ). rate가 겹치는 구간에서 비교한다.
- **부록 수렴 그림:** 70k~B checkpoint까지 monitor로 holdout에서 평가해, step별 J_h·PSNR·KiB 곡선을 그린다. LR 감소 뒤 곡선이 평평해졌는지도 수렴 근거로 확인한다.

## 9. 저장 공간

- checkpoint 하나는 약 398MB이고, run당 B/10k + 1개(last)가 쌓인다.

| B | run당 | 전체 |
|---|---:|---:|
| 100k | 11개, 약 4.4GB | 35개, 약 153GB |
| 120k | 13개, 약 5.2GB | 35개, 약 181GB |
| 150k | 16개, 약 6.4GB | 27개, 약 172GB |

- 10월 15일 Ceph quota 기준 여유는 약 600GiB였다. 제출 전에 `ceph.quota`·`df`로 다시 확인한다.
- 부록 수렴 곡선 평가(Gate 3)가 끝난 뒤 60k·B·last만 남기고 나머지를 지운다. 기존 [prune_nfcgs_checkpoints.py](../scripts/prune_nfcgs_checkpoints.py)가 새 경로를 지원하는지 확인한다.

## 10. 대안

- **늦은 연장:** Stage B를 시작한 뒤에 연장이 필요해지면, 같은 run의 60k checkpoint로 돌아가 새 milestone으로 다시 재개한다. 버리는 것은 Stage B 진행분뿐이고 0부터 다시 할 필요는 없다.
- **초기 학습이 불안정할 때:** 중앙 λ의 2×2와 H-ON .0032·λ0 여섯 개를 먼저 확인한 뒤 나머지로 확장한다. 여섯 개는 35개에 포함된 학습이므로 중복으로 돌리지 않는다.
- **고화질 기반 fine-tune:** Gate 2에서 원인을 좁힌 결과 fresh 학습의 최적화 문제로 보이면, H-ON λ0 또는 .0032 fresh checkpoint를 공통 부모로 삼아 나머지 λ를 각각 fine-tune한다. 고rate에서 저rate로 내려가는 방향이라 지금 계보와 반대다. 정렬마다 MSH가 달라지므로 부모를 cell별로 따로 둘지 함께 결정한다.

## 11. 학습 없이 다시 할 평가 (전체 완료 후)

최종 checkpoint(H-ON .0128 s0, 필요하면 λ0)로 다시 실행한다. 지금은 [path diagnostics plan](../config/plans/nfcgs_path_diagnostics_20261019.json), [독립 수신 Slurm](../scripts/slurm/eval_nfcgs_independent_receiver.slurm), [10월 20일 follow-up runner](../scripts/run_nfcgs_followup_20261020.py)가 계보 checkpoint(job447746)를 지정하고 있으므로 경로를 새 checkpoint로 바꿔야 한다. native 기준은 Gate 0 측정값을 재사용한다.

| 기존 이름 | 내용 | 스크립트 |
|---|---|---|
| B0 matched / T05 | 같은 256장에서 native와 actual의 차이, 경로 분해 | `eval_nfcgs_b0_matched.py`, `eval_nfcgs_path_diagnostics.py` |
| T06 | 고정 가중치의 순서 개입 | `analyze_nfcgs_fixed_order.py` |
| T15 | packet과 공유 모델만으로 독립 수신 | `eval_nfcgs_independent_receiver.py` |
| complexity | encode/decode 시간, 메모리. 설계 선택 비교(H-ON, M-ON, H-L3)에도 사용 | `profile_nfcgs_complexity.py` |

T06의 "anchor XYZ"는 `decoded_token_centers(P·g)`로 계산한 송신측 정렬 좌표다. 실제 복호된 Gaussian 중심이 아니므로 논문에서는 정렬 좌표로 표기한다.

## 12. 기존 작업 처리

- 458782(None+L3 OFF12 .0512)는 학습·최종 평가가 완료됐다. 458783(H+D4 OFF16 .0128)은 최신 다운로드 기준 48,946/50k이며 실시간 상태는 새 로그로 확인한다. 예비 결과로 보존한다.
- 457828(Score736-only)은 fresh 초기화 pilot(50k·2 GPU·holdout 제외 학습)이며 최종 평가457829까지 완료됐다. §2의 참고 결과로 쓰고 조건을 분리해 표시한다.
- 이미 끝난 구조 pilot(Score56-only 457741, MSH-only 457743, two-path56 457745)도 같은 50k·2 GPU fresh pilot이다. 본문 비교는 이번 계획의 S56·MSH·H-ON 학습으로 대체하고, pilot 결과는 참고로만 남긴다.
- [추가 실험 제안](NFCGS_ADDITIONAL_EXPERIMENT_REVIEW_2026-10-20.md)의 E01(seed 2×2), E03(구조 rate 겹치기), E04(fresh 2×2)는 이 계획으로 대체한다. E07(D4 RD 확장)은 §4의 2×2 5λ로 대체한다.
- OFF1/OFF4, L2/Q2 schedule, Morton·L3의 다른 λ, NN-score는 예비 결과로 부록에 둔다.

## 13. 추가 학습 9개 (2026-10-28 결정)

35개와 같은 fresh 120k 프로토콜(§3)을 따른다. 아직 제출하지 않았다. `fresh_rd.py`·`token_order.py`가 Wave 2의 해시 잠금 대상이므로, 새 cell은 서버의 별도 저장소 사본에서 실행한다. 서버 제출은 사용자 확인 후 한다.

### 13.1 경로별 Context·정렬 ablation (λ 한 점)

two-path의 2×2를 다른 경로에서도 한 λ로 반복해, Context와 정렬의 효과가 경로마다 같은 방향인지 본다. 기존 점은 다시 학습하지 않는다.

| 경로 | λ | 기존 | 추가 | 초기화 |
|---|---|---|---|---|
| Score56-only | .0128 | S56 = Hilbert+D4 ON (wave 1) | Hilbert+D4 OFF, None+D4 ON, None+D4 OFF | `score56_only.ckpt` |
| MSH-only | .0032 | MSH = Hilbert (wave 1) | None | `msh_only.ckpt` |

MSH-only에는 Score Context가 없다. MSH는 1D mean/scale hyperprior이고 Context를 끄는 설정이 없으므로 정렬만 비교한다. 정렬은 MSH의 1D convolution 이웃을 바꾼다.

### 13.2 Score56-only 저rate 점

| 경로 | λ | 이유 |
|---|---|---|
| Score56-only, Hilbert+D4 ON | .0512 | S56 .0128(holdout 22.32dB/60.98KiB)은 H-ON .0128(23.46dB/50.58KiB)과 .0512(22.66dB/23.38KiB)에 모두 밀린다. Score-only가 이길 여지가 있는 저rate 구간을 H-ON .0512와 직접 비교한다 |

### 13.3 정렬 축 분리

NN-score와 Hilbert는 정렬 기준 신호(값/위치)와 비교 방법(greedy NN/공간 채움 곡선)이 함께 다르다. 위치를 보는 greedy NN을 추가해 두 축을 나눈다. 기존 NN-score는 warm-start 계보(L3, 50k)뿐이므로 fresh로 다시 학습한다.

| Cell | 정렬 | 나머지 | 비교 |
|---|---|---|---|
| X-ON | NN-xyz: 대표 xyz의 L2 greedy 최근접 | H-ON과 같음(D4 ON, two-path), .0128 s0 | Hilbert ↔ NN-xyz: 비교 방법. NN-xyz ↔ NN-score: 기준 신호 |
| S-ON | NN-score: Score의 채널별 가중 L1 greedy 최근접, knn backend | 같음 | |

NN 정렬은 학습이 Hilbert보다 약 1.5–1.7배 느리다(warm-start 측정). 120k에 약 5–5.5일로 예상하며, 이 둘이 전체 계획의 마지막 학습이다.

### 13.4 decoder finetune (λ0)

| Run | 시작점 | 학습 대상 | 길이 |
|---|---|---|---|
| finetune | H-ON λ0 120k | Gaussian decoder readout LR 1e-5, codec LR 1e-6 | 20k |
| 대조군 | 같음 | codec만 LR 1e-6, decoder 고정 | 20k |

λ0와 native의 차이 중 decoder 적응으로 메워지는 몫을 본다. 대조군은 추가 학습량만의 효과를 뺀다. 경로 진단(H-ON λ0·.0128) 결과를 함께 보고 해석한다.

### 13.5 평가

- §11의 complexity 측정에 N-ON, X-ON, S-ON을 더한다. NN 두 학습이 끝난 뒤 한 job으로 같은 GPU에서 측정한다. Morton의 정렬 시간도 따로 기록한다.
- 결과는 §8 Gate 3와 같은 방식으로 정리한다. 13.1은 경로별 단일 λ의 ON/OFF·H/N 차이로만 보고하고 BD-rate를 계산하지 않는다.

## 14. Adaptive integer K (Top-K) 학습 6개 (2026-10-28 결정)

TokenCodecGS의 [Adaptive integer K](../../../TokenCodecGS/docs/adaptive_integer_k_porting.md)를 H-ON 위에 얹는다. GlobalSplat encoder 출력 4096×1024에서 router가 장면별 K를, importance scorer가 남길 token을 고른다. 선택된 K개만 정렬·압축하고, 학습 중에는 4096개 전체 Gaussian의 opacity에 straight-through gate를 곱한다. Gaussian decoder가 token별 독립 readout이므로 dense gate와 compact 추론이 일치한다. 선택 index는 전송하지 않는다(수신 결과가 Gaussian 집합이고 정렬처럼 encoder 전용). K는 기존 header의 token 수로 전달된다.

정렬×Context 2×2 주장에는 섞지 않는 확장 실험이다. 비교 기준은 기존 H-ON 곡선이다.

| Run | 정렬 | K | λ | 비교 |
|---|---|---|---|---|
| TK-H | Hilbert | adaptive | .0032, .0128, .0512 | H-ON 곡선 대비 BD-rate |
| TK-fixed | Hilbert | 고정: TK-H .0128의 30k 시점 평균 K | .0128 | 장면별 K 조절 효과 분리 |
| TK-imp | importance score 내림차순 | adaptive | .0128 | 선택 token의 정렬 비교 |
| TK-NN | NN-score(선택 token만) | adaptive | .0128 | 같음 |

- 초기 κ는 약 95%(`init_logit`≈2.94)에서 시작한다. codec이 4096 token 기준으로 학습되기 때문이다.
- importance 정렬은 scorer가 Top-K로 학습될 때만 의미가 있다. 현재 D4·MSH는 앞부분만 복호하는 구조가 아니므로 중요도 순서의 점진 복호 이점은 평가하지 않고 압축 효율만 비교한다.
- 학습 전 smoke: K=4096 강제 시 H-ON .0128과 결과가 같아야 한다. 가변 K의 압축·복원 왕복, task loss만으로 router·scorer에 gradient가 가는지, dense gate와 compact 렌더의 일치를 확인한다.
- 학습 중에는 4096개를 모두 렌더하므로 학습 비용은 줄지 않는다. 추론의 codec·decoder·렌더 비용만 K에 비례해 준다. complexity 측정(§13.5)에 TK 모델을 포함한다.
- 구현(2026-10-28): [token_selection.py](../globalsplat/compression/token_selection.py), `GlobalSplat._forward_selected`, `fresh_rd.CELLS`의 `tk_h`·`tk_imp`·`tk_ns`(L2 graph)·`tk_ns1`(L1 graph_gemm), smoke [smoke_nfcgs_topk.slurm](../scripts/slurm/smoke_nfcgs_topk.slurm). 선택 token만 장면별로 codec에 넣고 원래 slot으로 되돌린 뒤 4096×G Gaussian 전체를 opacity×gate로 렌더한다. 시험에서 dense gate 렌더 입력과 K개 비트스트림 복호 결과가 일치했다. gsplat은 alpha<1/255 Gaussian을 건너뛰므로 선택되지 않은 token의 gate에는 실제로 gradient가 거의 가지 않는다. scorer는 선택된 token의 상대 순위와 κ 정규화 항으로 학습된다(TokenCodecGS 기준 구현과 같은 한계).
- scorer 기준(2026-10-29 결정): TokenCodecGS 원본과 같은 학습 전용 MLP를 쓴다. 원본 방식과 비교하는 실험이므로 opacity 같은 명시적 사전값은 넣지 않는다. scorer가 무엇을 골랐는지는 학습 뒤 학습된 점수와 token별 Gaussian opacity의 상관으로 분석한다.
- λ0 decoder finetune(§13.4) 구현: [run_nfcgs_decoder_finetune.py](../scripts/run_nfcgs_decoder_finetune.py), [finetune_nfcgs_decoder.slurm](../scripts/slurm/finetune_nfcgs_decoder.slurm). `++model.train_decoder_readouts=true`가 Gaussian readout 4개만 열고, `++optimizer.decoder_lr`가 그 그룹에만 1e-5를 준다.

## 15. 추가 학습 15개의 실행 순서 (2026-10-28 결정)

사용자 한도는 동시 실행 16개다. 평가 job은 자리가 나는 대로, 실행 가능한 것부터 바로 넣는다.

| 단계 | 학습 | 개수 | 제출 |
|---|---|---|---|
| 1 | §13.3 X-ON·S-ON, §13.4 finetune·대조군 | 4 | 10/29 오전 |
| 1 | §14 TK-H(.0032·.0128·.0512), TK-imp, TK-NN | 5 | Top-K smoke 통과 뒤(목표 10/30 오전) |
| 2 | §13.1 경로별 ablation 4개, §13.2 S56 .0512 | 5 | 1단계 제출 뒤 자리가 나는 대로 |
| 2 | §14 TK-fixed | 1 | TK-H .0128 30k 평균 K 확정 뒤, 자리가 나는 대로 |

1단계 Top-K 5개가 들어갈 자리를 남기기 위해, 2단계는 1단계 제출을 마친 뒤에 넣는다.

B4.1–3(TK-H·TK-imp·TK-NN)을 11/3 정오까지 끝내는 일정: Top-K smoke를 10/29 새벽에 마치고 10/29 정오에 제출한다. TK-NN은 아래 kNN-graph greedy로만 이 일정에 들어온다. TK-fixed는 이 마감과 무관하며 2단계에서 B1과 같은 순서로 자리가 나는 대로 넣는다.

제출 기록(2026-10-28 밤, `/ceph_data/clue9986/tokencomp_ext`, `scripts/slurm/submit_nfcgs_fresh_ext.sh`): 468866 X-ON(`nx_on`) .0128, 468867 S56 H-OFF(`s56_off`) .0128, 468868 S56 N-ON(`s56_n_on`) .0128, 468869 S56 N-OFF(`s56_n_off`) .0128, 468870 MSH-N(`msh_n`) .0032, 468871 S56 .0512. 측정 job: 468797 NN 정렬 GPU benchmark, 468798 인접 token 32장. Wave 2 추가 4개는 기본 16개가 모두 끝난 뒤 시작하므로 B1과 X-ON을 1단계보다 먼저 넣었다.

2026-10-29 오전: 469898 Top-K smoke(`tk_h tk_ns tk_ns1`, 300 step), 469899 decoder finetune, 469900 대조군, 469902 S-ON(`ns1_on`) .0128. NN-score 거리는 L1 graph_gemm으로 정했다. 실제 Score 32장에서 Score56 인접 차이가 random 대비 0.327(정확한 L1 0.323, L2 graph 0.332, Hilbert 0.467)이고 정확한 L1 tour와 연결 84%를 공유했으며, GPU 정렬 시간은 장면당 10.3ms(정확한 L1 kNN 99ms, L2 graph 8.8ms)였다. warm-start NN-score와 같은 L1 거리라 비교가 유지된다. Top-K의 NN 정렬도 `tk_ns1`을 쓴다.

Top-K smoke 469898 결과(ariel-v12, .0128, 300 step): 세 cell 모두 오류 없이 끝났고 κ와 실제 K가 일치했다. K는 세 cell 모두 같은 궤적으로 step 75–150 사이에 3904→약 700, step 300에 약 280까지 내려가 하한 256에 가까워졌다. 이 시점 codec은 인코딩당 token당 약 380 bit(H-ON .0128 holdout 약 101 bit의 약 4배)를 쓰므로 rate 항의 κ gradient(λ·R/K)가 과대평가된 상태다. 원본 TokenCodecGS 학습 결과의 평균 K는 약 780이었다([ordered_transform.md](../../../TokenCodecGS/docs/ordered_transform.md)). 원본에는 500-step rate ramp가 있고, 이식 가이드는 K를 점진적으로 줄이고 실제 K 이동량을 먼저 확인하라고 한다. step 시간은 K≈280에서 3.8초로, fresh S56 N-ON(ariel-v8) 약 2.65초보다 약 45% 길다(노드 차이 포함). K가 이대로면 120k에 약 5.3일이 걸리고 slurm 한도 6일 안에 든다. TK-fixed 후보(`tk_f2048`…`tk_f3840`)는 K가 이 범위로 돌아오지 않으면 맞는 값이 없다. λ 비교: rate loss는 `λ · estimated_bits / (4 × 32768 × 59)`다. 4는 인코딩 수(micro batch 2 × 13-view branch 2)이고 `estimated_bits`는 이 4개의 합이다. Gaussian은 32,768개(token 4096 × 8), 속성은 59개다. 로그의 rate_loss/estimated_bits(.0128에서 1.66e-9)는 4개 합 기준이다. 원본(장면당 평균 bit × λ)과 같은 기준인 인코딩당 bit 계수는 그 4배로, .0032 1.66e-9, .0128 6.6e-9, .0512 2.6e-8이다. 원본 λ 범위(1e-11–1e-8)의 높은 쪽 끝과 겹치거나 넘지만 task loss가 SSIM 대 MSE+0.05·LPIPS로 달라 같은 RD 균형은 아니다. 결정(10/29): 원본 방식 그대로 Top-K 5개를 제출하고 10k step에서 K를 확인한다. 그때도 K가 300 이하에 머물면 취소하고 router warmup 판으로 바꾼다. 10k 확인 때 TK-H 세 λ의 K를 비교한다. .0032만 높게 남으면 rate 압력(λ와 task loss 크기)이 원인이고, 셋 다 하한이면 codec이 무작위인 시작 시점에 router가 움직이는 것이 원인이다. router LR은 속도만 바꾼다(원본 4e-5면 급락이 step 약 400으로 늦어질 뿐 그때도 인코딩당 token당 약 330 bit). router만 LR을 따로 주는 설정은 없어 바꾸려면 새 사본이 필요하다.

정상 K 범위 보정(10/29, 2026-10-08 TokenCodecGS 발표 p.16–17. 같은 날 rate 정규화를 확인한 뒤 수정): integer K 모델의 평균 K는 네 데이터셋에서 λ 1e-8일 때 85–91, 3e-9일 때 258–358, 1e-9일 때 566–771, 5e-10일 때 840–1108, 1e-10일 때 1625–2187이다. 원본 rate는 장면당 평균 bit에 λ를 곱하고 token 수로 나누지 않는다(TokenCodecGS `codecs.py` `rate_loss`). 위 인코딩당 bit 계수로 로그 보간하면 .0032(1.66e-9)는 약 390–540, .0128(6.6e-9)은 약 125–145다. .0512(2.6e-8)는 λ 1e-8의 85–91보다 작다(외삽 약 30–40). 처음 보간할 때는 4개 합 기준 계수를 써서 값이 4배 어긋났다. 고친 값으로는 .0128과 .0512가 하한 256 근처에 붙는 것이 원본 기준으로 정상이거나 그보다 높다. 원본 token은 학습 후에도 token당 약 3700 bit(λ 1e-9, 0.347 MiB / K 778)를 쓴다. 우리 H-ON .0128은 holdout에서 장면당 50.58KiB(약 4.1e5 bit), 곧 token당 약 101 bit로 4096개를 모두 보낸다. smoke step 300의 Top-K는 인코딩당 약 1.0e5 bit(로그 4.1e5 ÷ 4)를 token 269개에 쓴다(token당 약 380 bit). 같은 λ에서 token당 비용이 낮아지면 K는 커져야 한다. 반대로 task loss 크기 차이(원본 0.8·L1 + 0.2·(1−SSIM)/2 대 MSE + 0.05·LPIPS)는 K를 낮추는 쪽이다. 그래서 위 환산은 대략값이다. 10k 확인 기준을 다시 보정한다. .0512와 .0128의 하한은 실패로 세지 않는다. 세 λ의 K가 모두 256–300에 붙어 λ에 반응하지 않으면, 균형점이 아니라 router가 갇힌 것으로 보고 취소한다. .0032의 K가 .0128보다 뚜렷이 높으면 rate 균형이 작동하는 것으로 보고 계속한다. smoke의 train PSNR은 K가 3904에서 약 280으로 내려가는 동안 10.2 → 11.6 → 11.9 dB로 오히려 올랐고, render loss도 약 0.10–0.11에서 거의 움직이지 않았다. 이 시점에는 codec이 덜 학습돼 token을 빼도 distortion 비용이 거의 없고, router에는 rate 항만 작용한다. 시작 시점이 원인이라는 가설과 맞는다.

### 학습 실패 모드와 40k 확인 (2026-10-29 결정)

Wave 2 test에서 .0128의 H-OFF s2(22.14dB/27.7KiB), N-ON s2(22.86dB/45.8KiB)와 Wave 1 N-ON s0(23.02dB/44.4KiB)가 같은 cell의 RD 곡선 아래에 있었다. 학습 곡선을 보면 s1과 20k까지 같고, 그 뒤 정상 run은 Score 경로 비트를 늘리며 PSNR이 오르는데 실패 run은 Score 비트가 줄고 40k train PSNR이 약 0.9dB 뒤처진 채 84k LR 감소 뒤 굳는다. 최종 train loss도 실패 run이 높다(H-OFF 0.0287 vs 0.0245, N-ON 0.0270 vs 0.0252). 다른 RD 지점이 아니라 최적화 실패다. .0128 실패는 H-ON 0/3, H-OFF 1/3, N-ON 2/3, N-OFF 0/3이다.

- 재학습하지 않는다. 실패 run도 seed 결과로 보고하고 실패율과 학습 곡선을 부록에 둔다. 정렬 효과(Context ON, 5λ)는 .0128 seed를 바꿔도 −32.4~−33.6%로 유지된다.
- 단일 seed 새 실험은 40k에서 [check_nfcgs_train_curve.py](../scripts/check_nfcgs_train_curve.py)로 확인한다. 같은 λ의 가장 가까운 정상 run 대비 train PSNR이 0.5dB 넘게 낮고 25k 대비 Score 비트가 줄었으면 seed를 하나 추가한다. 다른 cell을 기준으로 쓸 때는 cell 차이(정상 run끼리도 40k에서 최대 약 0.8dB)가 섞이므로 Score 비트 변화를 함께 본다.

모든 새 코드는 서버의 별도 저장소 사본에서만 실행한다. 원래 서버 저장소의 해시 잠금 파일은 Wave 2와 monitor가 끝날 때까지 바꾸지 않는다. 사본도 자기 run이 도는 동안에는 같은 이유(test 직전 code hash 재확인)로 잠긴다. 그래서 §13–15의 모든 cell(Top-K와 TK-fixed 후보 `tk_f2048`…`tk_f3840` 포함)을 10/28에 미리 정의하고, 사본 job이 시작하기 전에 한 번에 올린 뒤 고정한다. 이후 runtime 수정이 필요하면 새 사본을 만든다.

### NN 정렬 효율화: kNN-graph greedy

기존 측정에서 greedy 루프 자체는 장면당 약 1.9ms(CPU)였다. 비용은 전체 거리 행렬(N²·D, 장면당 64MiB), top-k, GPU→CPU 전체 행렬 복사와 그 동기화에서 나온다. 학습 step은 Hilbert D4 대비 +53~55%였다(2.03 vs 1.31 s/step, warm-start 50k).

- GPU에서 거리를 블록 단위로 계산하고 token별 top-64 이웃만 남긴다. 전체 행렬을 보관·복사하지 않는다(장면당 복사 약 3MiB).
- CPU greedy는 이웃 목록만 사용한다(O(N·k)). 후보가 모두 방문됐으면 전체 행 검색 대신 Hilbert(xyz) 순서에서 다음 미방문 token으로 이동한다.
- 정확한 greedy와는 fallback 지점에서만 순서가 다르다. 학습 전에 기존 16장 cache에서 정확한 greedy와 인접 Score 차이·fallback 횟수를 비교하고, GPU에서 정렬 시간을 잰다.
- S-ON·X-ON(§13.3)과 TK-NN(§14)에 같은 방식을 쓴다. warm-start NN-score(정확한 greedy)와는 정렬 구현이 다르다고 표시한다.

### Top-K fine-tune 재설계 (2026-10-30 결정)

fresh Top-K 5개(470017–470021)는 3.5k step에서 모두 K 하한 256에 붙었고, 같은 step의 H-ON .0128보다 train PSNR이 약 6.1dB 낮았다(12.6 vs 18.7dB). 정렬과 λ가 달라도 값이 같아 K=256에서 화질이 막힌 상태로 보고 취소한다. 10k 판정 기준(§15)을 기다리지 않은 이유는 6dB 손실에도 router가 K를 올리지 않았기 때문이다.

- 원인(원본 TokenCodecGS와 대조): 원본 LTCO Scalar-Q codec은 변환 없이 X/Δ를 반올림하므로 0 step부터 복원이 거의 무손실이다. LR은 router 4e-5·codec 4e-7의 OneCycle이라 초반이 낮다. 우리 fresh codec은 초기 PSNR이 10dB대라 token을 빼도 distortion 비용이 없어 rate 항만 작용했고, LR 1e-4가 0 step부터 적용됐다. 하한에서는 token당 Gaussian이 8개(원본 64개)라 화질 손실이 크다. 그런데 선택되지 않은 token은 opacity 0으로 렌더되지 않아 K를 올리는 gradient가 거의 없다.
- 설계: 같은 λ의 H-ON s0 120k weight에서 시작한다(router·scorer는 새로 초기화, κ≈95%). 20k step 동안 codec·router·scorer는 LR 1e-5 고정, Gaussian readout 4개도 LR 1e-5로 학습한다. TK-H .0032·.0128·.0512와 같은 조건에서 Top-K만 끈 대조군 3개, 총 6개다.
- TK-imp·TK-NN·TK-fixed는 뺀다. H-ON context 모델이 Hilbert 순서로 학습돼 다른 순서로 fine-tune하면 분포 이동이 섞이고, S-ON checkpoint는 11/3에 나온다.
- 비교: λ별 TK − 대조군을 Top-K 효과로 보고, 두 곡선의 BD-rate를 낸다. H-ON 곡선과의 차이에는 decoder 적응과 추가 20k 학습이 섞이므로 주장은 대조군 기준으로 한다. λ0 decoder finetune(469899·469900)은 decoder 적응 크기의 참고값이다.
- 위험: 학습된 우리 token은 token당 약 101 bit라 원본(약 3,700 bit)보다 token을 빼서 얻는 rate 이득이 작다. .0032에서 K가 4096 근처에 머물 수 있으며 그 자체를 결과로 보고한다.
- 구현: [run_nfcgs_topk_finetune.py](../scripts/run_nfcgs_topk_finetune.py), [finetune_nfcgs_topk.slurm](../scripts/slurm/finetune_nfcgs_topk.slurm), [test_topk_finetune.py](../tests/test_topk_finetune.py). 기존 `train_decoder_readouts`·`decoder_lr`·selector override만 써서 runtime 파일은 바꾸지 않았다. `tokencomp_ext`에 새 파일만 추가해 실행한다. 결과는 `outputs/nfcgs_topk_finetune/{topk,control}/lambda*/job_*`에 쌓인다.
- 제출(10/30 13:5x): TK 471552 .0032, 471551 .0128, 471553 .0512, 대조군 471554–471556(자리 대기). source는 H-ON s0 job_460334·460338·460342 120k이고, 세 TK 모두 `missing=50`(loss network 38 + selector 12), 학습 파라미터 5.3M(readout 포함)으로 시작했다. 약 1 it/s(step당 약 4초)다.
- 약 600 step 확인: K는 .0032 3901→3597, .0128 3900→3435, .0512 3900→3423이고, train PSNR은 23.54→23.74, 23.51→23.54, 22.78→22.86이다. fresh 판과 달리 화질이 H-ON 수준을 유지하고 K가 천천히 내려간다. 수렴 여부는 약 3k step에서 다시 본다.

### 10/30 확인 기록

- 40k 확인(§15): X-ON 468866은 H-ON .0128 대비 −0.31dB, Score bit −6%로 기준(0.5dB) 안이다. S-ON 469902는 10k에서 −0.31dB이고 40k는 10/31에 본다. S56 ablation 3개(22.30·22.35·22.38dB, Score bit −1%·+1%·+1%)와 MSH-N(20.85dB)은 처음 명령으로는 gap을 계산하지 못했다. S56 셋은 서로 0.08dB 안이고 Score bit가 줄지 않았다.
- 원인: Wave 1 run은 Stage A(0–60k)가 `outputs/nfcgs_fresh_rd/<cell>/<λ>/s0/job_*`에 있고, `branches/B120000/...`에는 60k 이후만 있다(S56 .0128 job_460347의 tensorboard 파일 하나, 805MB, 10/23 생성. Wave 2 120k 파일은 1.6GB). 그래서 `$B/...`를 Wave 1 기준으로 쓰면 40k 구간이 비어 `no reference`가 된다. X-ON의 −0.31dB도 H-ON s1·s2만 기준으로 계산된 값이다. Wave 1 s0를 40k 이전 기준으로 쓸 때는 `outputs/nfcgs_fresh_rd/...` 경로를 쓴다.
- 경로 진단 465355(fresh H-ON λ0 120k, matched256): actual 24.36dB(903KiB/scene)다. Score·MSH 양자화를 모두 우회하면 +0.07dB라 λ0 손실은 양자화가 아니다. B1(F 직접)은 12.90dB, `lr_continuous`·`lr_decoded`는 11.29·11.49dB로, fresh codec은 F를 복원하기보다 frozen decoder용 입력을 새로 학습한 것으로 보인다. fresh 모델에서는 B1을 비압축 상한으로 쓰지 않고 native B0와 비교한다. λ0 decoder finetune(§13.4)이 이 변환·decoder 정합 손실을 직접 겨냥한다.

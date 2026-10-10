# 정렬·spatial context의 3-point RD 실험

작성: 2026-10-10. 이 세션의 Morton/Hilbert 및 dyadic4 실험에만 해당한다.

## 비교할 곡선과 기존 point

각 곡선에서 λ=0.0064, 0.0128, 0.0256을 측정한다. λ=0.0256은 기존 실행을 재사용하고, 두 낮은 λ에 대한 학습을 추가한다.

| 곡선 | 정렬 | context | λ=.0256 학습 / eval | 내려받은 로그의 상태 |
|---|---|---|---|---|
| 대조군 | Morton | legacy 3단계, kernel5 | 441552_1 / 441553_1 | 완료 |
| 정렬 변경 | Hilbert | legacy 3단계, kernel5 | 445516 / 445517 | 완료 |
| context 변경 | Morton | dyadic4 | 445518 / 445519 | 학습 50k·전체 scene eval 완료 |

λ=.0256 전체 scene eval에서 Morton 3단계는 PSNR 23.4892 dB / 30.807 KiB, Hilbert 3단계는 23.5050 dB / 30.290 KiB, Morton dyadic4는 23.5089 dB / 29.984 KiB였다. 용량은 scene당 실제 bitstream의 평균이다.

### 완료된 λ=.0256 결과

| 설정 | PSNR (dB) | SSIM | LPIPS | 전체 KiB | Score KiB | MSH y+z KiB | entropy 복호 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Morton legacy3 | 23.4892 | 0.7294 | 0.2765 | 30.807 | 19.384 | 11.294 | 80.8 |
| Morton legacy4 | 23.4358 | 0.7271 | 0.2782 | 30.769 | 19.353 | 11.287 | 84.9 |
| Hilbert legacy3 | 23.5050 | 0.7295 | 0.2764 | 30.290 | 18.982 | 11.178 | 85.0 |
| Morton dyadic4 | 23.5089 | 0.7299 | 0.2753 | 29.984 | 18.430 | 11.425 | 84.8 |

Morton dyadic4의 새 결과는 `slurm/slurm-gs-order-context-eval-445519.out`에서 확인했다. step50k checkpoint와 dyadic4 설정을 로드했으며, 완료 요약과 실제 entropy 복호 시간을 출력했다. 대응 stderr에는 치명적 오류가 없다.

Morton legacy3 대비 전체 크기는 2.673%, Score 크기는 4.923% 감소하고 PSNR은 0.0197 dB 높았다. MSH y+z 크기는 1.158% 증가해 Score 절감의 일부를 상쇄했다. Hilbert legacy3 대비 전체 크기는 1.010% 감소하고 PSNR 차이는 +0.0039 dB였다. 한 λ에서 측정한 결과이므로 여러 λ의 RD 비교는 추가 실험 종료 후 수행한다.

새로 제출할 작업은 학습 6개와 eval 6개다. 각 eval에는 대응하는 학습 하나의 `afterok` dependency를 건다. 학습마다 2 normal GPU를 요청하므로, 전부 동시에 실행되면 학습 GPU는 12개다. 실제 동시 실행 수는 Slurm의 자원과 사용자 QOS 제한에 따라 결정된다.

## 초기화와 학습 조건

추가 point는 모두 기존 두 신규 실험과 같은 FullSplit λ=.0256, rank56, residual_on, step10k checkpoint에서 weights-only로 시작한다.

```text
outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt
```

- GlobalSplat 고정, codec 전체 학습. MSH residual을 포함한다.
- score path는 minimal: centering·score norm·mean/channel context·nonlinear 제거, spatial context 사용.
- 학습당 normal GPU 2개, GPU당 batch2, accumulation2, 유효 배치8.
- 50k optimizer step, LR1e-4, 35k/45k decay, checkpoint 5k마다, seed111123.
- 기존 Morton 대조군의 λ=.0256 실행은 GPU4 × batch2 × accumulation1이었다. 유효 배치는 추가 실험과 같은 8이다.
- full-scene eval은 기존 C12/T8 프로토콜과 실제 entropy bitstream을 사용한다.

λ=.0128의 FullSplit 부모는 없으므로 별도로 만든 경로를 찾지 않는다. 단일 point 제출에서 `.0128`을 지정하면 기본 부모는 `.0256`이며, RD sweep은 모든 추가 point의 부모를 `.0256`으로 명시한다. 중간 rate의 eval에는 해당 rate로 학습한 checkpoint를 반드시 지정한다.

## 서버에 반영할 파일

1. `scripts/run_nfcgs.py`
2. `scripts/slurm/nfcgs_order_context_settings.sh`
3. `scripts/slurm/submit_nfcgs_order_context.sh`
4. `scripts/slurm/submit_nfcgs_order_context_rd.sh`

## 제출

저장소 루트에서 실행한다.

```bash
# 모든 point의 부모·설정·launcher 지원을 확인한다. job은 제출하지 않는다.
bash scripts/slurm/submit_nfcgs_order_context_rd.sh --dry-run

# 학습 6개와 대응하는 전체 scene eval 6개 제출
bash scripts/slurm/submit_nfcgs_order_context_rd.sh
```

스크립트는 모든 point를 먼저 점검한 뒤 첫 job을 제출한다. 기존 λ=.0256을 기본 제출 대상에서 제외한다. 제출 후 job ID와 부모·출력 경로는 `logs/slurm/order_context_rd_<timestamp>_<pid>.tsv`에 저장한다. 반복 실행하면 추가 job을 다시 제출하므로 실제 제출은 한 번 수행한다.

| 설정 | 학습 job name 예시 | eval job name 예시 |
|---|---|---|
| Morton legacy3 | `gs-oc-m3-0p0064` | `gs-oc-e-m3-0p0064` |
| Hilbert legacy3 | `gs-oc-h3-0p0064` | `gs-oc-e-h3-0p0064` |
| Morton dyadic4 | `gs-oc-d4-0p0064` | `gs-oc-e-d4-0p0064` |

λ=.0128 point의 이름은 위 표에서 `0p0064` 대신 `0p0128`을 사용한다. 출력은 `outputs/nfcgs_order_context/<order>/<schedule>_stages<N>_k<K>_ste/lambda<tag>/job_<trainID>`이며 eval은 각 run의 `eval_all`에 저장한다. dyadic4는 anchor gather를 사용하므로 kernel5 표기는 호환용 설정이다.

## 445519 dependency

445519는 445518의 성공 종료를 기다리는 eval로 제출됐다. 초기에는 사용자가 dependency 대기를 확인했고, 이후 받은 로그에서 두 작업 모두 완료됐음을 확인했다. 현재 학습의 로그 파일을 내려받은 시점이 학습 완료 시점과 같지는 않다.

```bash
squeue -j 445518,445519 -o '%.18i %.24j %.12T %.40R'
sacct -j 445518,445519 --format=JobID,JobName%24,State,ExitCode,Elapsed -X
```

학습이 실패 종료하면 `afterok` 조건이 충족되지 않으므로 최종 checkpoint와 종료 상태를 확인한 뒤 eval을 처리해야 한다. 이번 대기 이유 확인만으로 실패 종료를 추정하지 않는다.

## 추가 실험: Hilbert + quarter2 / dyadic4

Hilbert 정렬을 고정하고 두 anchor context를 각각 λ=.0064, .0128, .0256에서 학습한다. 두 설정 모두 기존 결과가 없으므로 이번에는 .0256도 새로 제출한다. 총 학습 6개와 전체 scene eval 6개다.

| 설정 | 복호 순서 | 단계별 token 비율 | 학습 job name 예시 |
|---|---|---|---|
| Hilbert + quarter2 | `0::4 → 나머지` | 25%, 75% | `gs-oc-hq2-0p0064` |
| Hilbert + dyadic4 | `0::8 → 4::8 → 2::4 → 1::2` | 12.5%, 12.5%, 25%, 50% | `gs-oc-hd4-0p0064` |

두 context는 이전 단계의 복원된 anchor를 직접 gather한다. Kernel 표기는 호환용이며 gather의 참조 범위를 제한하지 않는다. GlobalSplat 고정, minimal/STE, MSH 유지, 같은 FullSplit .0256 부모, GPU2·유효 배치8·50k 학습 조건을 사용한다.

앞 절의 λ 지원 파일에 더해, 갱신한 `submit_nfcgs_order_context_rd.sh`와 새 `submit_nfcgs_hilbert_context_rd.sh`를 서버에 반영한다.

```bash
# 6개 point 확인, 제출 없음
bash scripts/slurm/submit_nfcgs_hilbert_context_rd.sh --dry-run

# 학습 6개 + 각 학습 성공 후 전체 scene eval 6개
bash scripts/slurm/submit_nfcgs_hilbert_context_rd.sh
```

전용 helper는 이전 환경의 RD_CONFIGS/RD_LAMBDAS 값에 관계없이 위의 2×3 matrix를 선택한다. 같은 방식으로 제출 ID와 출력 경로를 TSV에 기록한다. 원래 RD helper의 기본 3개 곡선·추가 2개 λ 동작도 유지한다.

## 추가 제출 현황 확인 — 2026-10-10

아래 진행 step은 받은 학습 로그에서 계산한 근사값이며, queue 상태는 사용자가 붙여준 squeue 결과다. 두 자료의 시점이 정확히 같다고 가정하지 않는다.

| 설정 | λ | 학습 job | eval job | 확인된 상태 |
|---|---|---|---|---|
| Morton legacy3 | .0064 | 447719 | 447720 | 학습 RUNNING, 로그 약 16.9k/50k; eval Dependency |
| Morton legacy3 | .0128 | 447721 | 447722 | 학습 RUNNING, 로그 약 17.0k/50k; eval Dependency |
| Hilbert legacy3 | .0064 | 447723 | 447724 | 학습 RUNNING, 로그 약 16.5k/50k; eval Dependency |
| Hilbert legacy3 | .0128 | 447725 | 447726 | 학습 RUNNING, 로그 약 13.0k/50k; eval Dependency |
| Morton dyadic4 | .0064 | 447727 | 447728 | 학습 RUNNING, 로그 약 12.4k/50k; eval Dependency |
| Morton dyadic4 | .0128 | 447729 | 447730 | 학습 QOSMaxGRESPerUser; eval Dependency |
| Hilbert quarter2 | .0064 | 447738 | 447739 | 학습 QOSMaxGRESPerUser; eval Dependency |
| Hilbert quarter2 | .0128 | 447740 | 447741 | 학습 QOSMaxGRESPerUser; eval Dependency |
| Hilbert quarter2 | .0256 | 447742 | 447743 | 학습 QOSMaxGRESPerUser; eval Dependency |
| Hilbert dyadic4 | .0064 | 447744 | 447745 | 학습 QOSMaxGRESPerUser; eval Dependency |
| Hilbert dyadic4 | .0128 | 447746 | 미확인 | 학습은 queue에 있음; 대응 eval 제출 미확인 |
| Hilbert dyadic4 | .0256 | 미확인 | 미확인 | 받은 제출 TSV와 queue에 없음, 확인 중 |

첫 sweep TSV는 학습·eval 쌍 6개를 기록했다. Hilbert anchor sweep TSV는 쌍 4개만 기록했고, queue에는 다섯 번째 학습 447746도 있다. 이후 사용자가 제공한 sacct에는 Hilbert dyadic4 학습 447744와 447746만 표시됐다. 현재 자료로는 .0128의 eval과 .0256의 train/eval 제출이 확인되지 않으며, 제출 종료 원인의 오류 메시지는 아직 제공되지 않았다.

현재 queue를 기준으로 빠진 작업만 추가하려면 아래처럼 기존 447746의 eval을 제출하고, .0256 point 하나를 제출한다. 기존 6-point helper 전체를 반복하는 대신 확인된 다섯 학습을 재사용한다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat

# 이미 등록된 .0128 학습 447746의 후속 eval만 제출
REPO_DIR="$PWD" TOKEN_ORDER=hilbert CONTEXT_SCHEDULE=dyadic4 STAGES=4 KERNEL=5 \
RATE_LAMBDA=0.0128 MAX_STEPS=50000 CHECKPOINT_EVERY=5000 TRAIN_JOB_ID=447746 \
sbatch --job-name=gs-oc-e-hd4-0p0128 --dependency=afterok:447746 --export=ALL \
  scripts/slurm/eval_nfcgs_order_context.slurm

# 아직 제출이 확인되지 않은 .0256 학습 + 후속 eval
TOKEN_ORDER=hilbert CONTEXT_SCHEDULE=dyadic4 STAGES=4 KERNEL=5 \
RATE_LAMBDA=0.0256 MAX_STEPS=50000 CHECKPOINT_EVERY=5000 \
TRAIN_JOB_NAME=gs-oc-hd4-0p0256 EVAL_JOB_NAME=gs-oc-e-hd4-0p0256 \
bash scripts/slurm/submit_nfcgs_order_context.sh
```

위 명령은 서버에서 실행하지 않았다. 재시도 시 sbatch가 거절되면 해당 오류와 성공적으로 생성된 job ID를 확인해 남은 부분만 이어서 제출한다.

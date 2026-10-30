# 정렬·스케줄 RD 범위 확장 — 2026-10-17

Hilbert+dyadic4의 λ=0 한 point와, 선택한 6개 방법의 λ=.0032/.0512 두 point씩을 추가한다. 총 학습13개와 학습 성공 후 실행할 전체 평가13개다. 사용자가 지정한 Hilbert Context OFF는 최근 실험과 같이 4096 token·56채널을 합친 실제 Score entropy stream 1개를 사용한다.

| 모델 | Token 단계 | 채널 그룹 | Score strings/scene | 공간 예측 | 추가 λ |
|---|---:|---|---:|---|---|
| Hilbert + dyadic4 | 4 | 16/16/16/8 | 16 | ON | 0, .0032, .0512 |
| Morton + dyadic4 | 4 | 16/16/16/8 | 16 | ON | .0032, .0512 |
| Hilbert + legacy3 | 3 | 16/16/16/8 | 12 | ON | .0032, .0512 |
| Morton + legacy3 | 3 | 16/16/16/8 | 12 | ON | .0032, .0512 |
| None + dyadic4 | 4 | 16/16/16/8 | 16 | ON | .0032, .0512 |
| Hilbert + Context OFF | 1 | 56 | 1 | OFF | .0032, .0512 |

MSH는 모든 조건에서 유지한다. Score strings 수는 Score payload 내부의 실제 rANS string 수이며 MSH y/z stream과 컨테이너는 별도다.

## 고정 조건

- 모든 point는 기존 정렬/context ablation과 같은 FullSplit λ=.0256 probability-only step10k 부모에서 시작한다. 부모 optimizer/scheduler/global step을 이어받지 않는 weights-only 새50k 학습이다.
- GlobalSplat 고정, codec 전체 학습, rank56 linear analysis/synthesis, MSH 유지.
- Minimal: scene centering/mean 전송, Score norm, mean/channel conditioner, 추가 nonlinear MLP OFF. 표의 첫5개 방법만 공간 예측 ON.
- Kernel5/STE, 유효배치8(GPU2 × batch2 × accumulate2), BF16, LR1e-4, LR 감소35k/45k, checkpoint 매5k, 학습 시간 제한6일.
- 기존 C24/T12 학습·subset consistency·seed111123/data-loader seed403 조건 유지.
- 최종50k checkpoint의 실제 encode→decode→render 전체 평가: C12/T8, batch1, seed0, BF16. 각 train의 `afterok`로 GPU1개 평가를 제출한다.
- Score와 MSH 모두 지정한 token 정렬을 사용한다. None은 원래 slot 순서다.

공통 부모 경로:

```text
outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt
```

## λ=0의 의미

현재 구현의 목적함수는 기존 rendering/consistency loss에 `λ × estimated BPGA`를 더한다. λ=0은 이 rate penalty만 0으로 만든다. 양자화, Score spatial context, MSH, entropy coding, 최종 실제 byte 측정은 유지한다. λ=0은 무양자화 모델이나 lossless codec을 뜻하지 않는다.

기존 deterministic quantile update(매500 steps)도 유지한다. 모든 λ에서 같은 loss/optimizer recipe를 사용하고 rate coefficient만 바꾼다. Runner는 0을 실제 `loss.rate_lambda=0`으로 전달하며 truthiness에 따른 기본 λ 치환을 하지 않는다.

## Context OFF 비교의 범위

사용자 지정에 따라 Context OFF는 single/stages1/channels56이다. 부모의 unpredicted 채널별 확률모델을 이어 붙여 초기화하는 기존 채널 통합 로더를 재사용한다. 이에 따라 Hilbert+dyadic4와 이 OFF 모델의 차이는 공간 예측과 함께 token/채널 분할 및 단계별 확률모델 수의 변화도 포함한다. 결과를 공간 예측만의 단독 효과로 설명하지 않는다.

이 곡선은 앞서 제출한 `single_stages1_k5_ste_context_off_c56`의 λ=.0064/.0128/.0256 결과와 연결한다. 과거 `legacy_stages3_k5_ste_context_off`의 12-stream Hilbert-only 곡선과는 구성이 다르다. 새 네 단계 full56 OFF(4 strings) 실험도 이번 13-point 목록에는 추가하지 않는다.

## 서버 반영 파일

다음 4개 파일은 모두 로컬 `upstream/globalsplat/` 안에 있으며, 서버 프로젝트 루트 아래의 같은 상대 경로에 반영한다.

1. `scripts/run_nfcgs.py`: 0/.0032/.0512 허용, 새 λ 평가에서 명시적 checkpoint 요구.
2. `scripts/slurm/nfcgs_order_context_settings.sh`: 새 λ와 출력 tag 허용.
3. `scripts/slurm/submit_nfcgs_order_context.sh`: 새 λ의 기본 부모를 기존 .0256 부모로 연결. 기존 기본 λ의 부모와 제출 signature는 유지한다.
4. `scripts/slurm/submit_nfcgs_order_context_extremes.sh`: 요청한 13개 point만 제출하는 전용 launcher.

이전 single/56채널 codec 변경과 Ceph TMPDIR 수정이 적용된 train/eval Slurm을 재사용한다. 전용 launcher는 단일 stream flag, 채널 통합 loader, Ceph TMPDIR 수정이 있는지 먼저 확인한다. 이전 1/4 stream 실험이 동작하는 서버에는 추가 codec/YAML 변경이 필요하지 않다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
if bash scripts/slurm/submit_nfcgs_order_context_extremes.sh --dry-run; then
    bash scripts/slurm/submit_nfcgs_order_context_extremes.sh
fi
```

전용 launcher는 `bash`로 실행한다. 먼저 모든 설정·부모·기존 state를 확인한다. 실제 제출 순서는 Hilbert+dyadic4 λ=0, 여섯 방법의 λ=.0032, 여섯 방법의 λ=.0512다. 이전 셸의 정렬, schedule, stage, channel group, NN 설정, 학습량, `EVAL_RUN_DIR`는 이 matrix의 값으로 고정한다. `SOURCE_CHECKPOINT`를 지정하면 모든 point에 같은 파일을 사용한다.

QOS 제한으로 중간에 제출이 거부되면 같은 명령을 다시 실행한다. Train ID를 eval 제출 전에 state에 기록하므로, 접수된 train을 재제출하지 않고 빠진 eval/이후 train을 이어간다. 이 기능은 제출 재시도이며 이미 실패·취소된 학습의 자동 resume 기능은 아니다. 다른 state 폴더는 별도 실험으로 취급한다.

## 출력과 상태

- State/요약: `logs/slurm/order_context_extremes_20261017/`의 point별 `.state` 및 `submissions.tsv`.
- Train name: `gs-x-<hd4/md4/hl3/ml3/nd4/h1off>-<λtag>`.
- Eval name: `gs-x-e-<arm>-<λtag>`.
- λ=0 tag는 `0`, 나머지는 `0p0032`/`0p0512`다.
- 공간 예측 ON: `outputs/nfcgs_order_context/<order>/<dyadic4_stages4/legacy_stages3>_k5_ste/lambda<λtag>/job_<TRAIN_ID>/`.
- Context OFF: `outputs/nfcgs_order_context/hilbert/single_stages1_k5_ste_context_off_c56/lambda<λtag>/job_<TRAIN_ID>/`.
- 최종 checkpoint: 각 출력의 `checkpoints/nfcgs_main/version_0/step000050000.ckpt`.
- 평가: 각 출력의 `eval_all/`.
- 로그: `logs/slurm/slurm-gs-order-context-<TRAIN_ID>.out/.err` 및 `slurm-gs-order-context-eval-<EVAL_ID>.out/.err`.

기존 job의 취소·재제출·체크포인트 변경은 수행하지 않는다. 서버 제출과 GPU 학습·평가는 사용자가 위 명령을 실행한 후 시작한다.

## 검증

로컬 관련 검사64개를 모두 통과했다: 새 제출 검사4개, Hydra/CLI18개, 기존 Slurm30개, 분산 runner12개다.

- 실제 Hydra 구성에서 13개 point × train/eval의 26개 조합을 확인했다. λ=0, minimal/MSH, stage/채널 수, frozen backbone, 새50k, 유효배치8, 전체 실제 bitstream 평가를 검사했다.
- 가짜 sbatch로 13개 train·13개 개별 afterok eval, 공통 부모, single 모델만 56채널, 반복 제출의 ID 재사용을 검사했다.
- 20번째 제출이 거부되는 상황에서 이미 접수된 train ID를 보존하고 중복 없이 나머지 제출을 이어가는지 확인했다.
- 마지막 point state 오류·기존 lock·부모 누락을 첫 제출 전에 탐지하는지 검사했다.
- 기존 λ의 CLI/Slurm/분산 runner 회귀 검사도 통과했다. 실제 서버 결과와 새 job ID는 아직 없다.

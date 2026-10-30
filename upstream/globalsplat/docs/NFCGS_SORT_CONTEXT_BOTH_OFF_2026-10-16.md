# Sorting OFF + Score 공간 Context OFF 대조군 — 2026-10-16

기존 rank56/minimal/legacy3 실험에서 빠진 기본 token 순서 + Score 공간 예측 OFF 조건을 λ=.0064/.0128/.0256에서 추가한다. 이 세 point로 정렬과 Score 공간 예측의 2×2 비교를 완성한다. Projection 없는 2-path로 구조를 바꾸는 실험은 이 비교와 별도로 진행한다.

| Token 순서 | Score 공간 Context OFF | Score 공간 Context ON |
|---|---|---|
| 원래 slot 순서 (`none`) | **이번 추가: both_off, 3개 λ** | 기존 context_only legacy3, 완료 |
| Hilbert | 기존 hilbert_only, 완료 | 기존 Hilbert legacy3, 완료 |

기존 결과는 [10월 16일 로그 검토](NFCGS_LOG_REVIEW_2026-10-16_1231.md)와 [기존 정렬/context 곡선](NFCGS_ORDER_CONTEXT_LOG_REVIEW_2026-10-13_1030.md)에 있다. 과거 factorized 모델은 학습 조건과 stream 구성이 다를 수 있으므로 이번 both_off 칸을 대신하지 않는다.

## 고정 조건과 OFF의 범위

- 세 λ 모두 기존 ablation과 같은 FullSplit λ=.0256, 마지막 probability-only 단계 step10k 부모의 가중치로 시작한다. 부모 optimizer/scheduler/global step을 이어받지 않고 각각 새50k optimizer steps를 학습한다.
- GlobalSplat 고정, rank56 linear analysis/synthesis 유지, codec 전체와 MSH 학습.
- `token_order=none`, `score_spatial_context=false`, `score_context_schedule=legacy`, stages3/kernel5/STE.
- Minimal: scene centering, FP16 mean 전송, Score norm, mean conditioner, channel context, 추가 nonlinear MLP 모두 OFF.
- Score의 채널별 학습 확률모델, median 기반 양자화, legacy3의 3단계 × 4개 channel group = 12개 entropy stream은 유지한다. 공간 예측 모듈은 실행하지 않고 동결한다.
- MSH의 transform, normalization, hyperprior, y/z stream은 유지한다. Score와 MSH 모두 원래 token slot 순서로 처리한다.
- 학습 C24/T12·subset consistency ON, seed111123/data-loader seed403, BF16, GPU2 × batch2 × accumulate2 = 유효배치8.
- LR1e-4, 35k/45k에서 감소, checkpoint 매5k, 학습 시간 제한6일.
- 학습 성공 후 최종50k checkpoint로 전체 test scene의 실제 bitstream encode→decode→render 평가. C12/T8·batch1·seed0·BF16, test-time optimization 없음.

기본 부모는 다음 경로다. 위치가 바뀐 동일 부모를 쓸 때만 `SOURCE_CHECKPOINT`를 지정한다.

```text
outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt
```

## 서버에 반영할 파일과 제출

이번에 반영할 파일은 두 개다. 앞서 서버에 적용한 codec/runner/settings/개별 제출 helper/eval Slurm 파일을 재사용한다.

1. [submit_nfcgs_sort_context_both_off_rd.sh](../scripts/slurm/submit_nfcgs_sort_context_both_off_rd.sh): 새 대조군 3개만 제출하는 전용 launcher.
2. [train_nfcgs_order_context.slurm](../scripts/slurm/train_nfcgs_order_context.slurm): recovery launcher와 같은 Ceph `TMPDIR`를 일반 학습 launcher에도 적용. Python 실행 전에 `/ceph_data/${USER}/.tmp/nfcgs/job_${SLURM_JOB_ID}`를 생성하고 `CHECKPOINT_TMPDIR=`를 출력한다. `NFCGS_TMP_ROOT`로 임시 저장 루트를 바꿀 수 있다.

전용 launcher는 일반 학습 파일의 `CHECKPOINT_TMPDIR` 수정 반영 여부를 확인한다. 수정이 없으면 제출 전에 중단한다. 기존 `/tmp` checkpoint staging 문제에 대한 진단과 근거는 [Ceph/ENOSPC 검토](NFCGS_CEPH_ENOSPC_DIAGNOSIS_2026-10-15.md)에 있다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
bash scripts/slurm/submit_nfcgs_sort_context_both_off_rd.sh --dry-run
bash scripts/slurm/submit_nfcgs_sort_context_both_off_rd.sh
```

총 학습3개(GPU2씩)와 `afterok` 평가3개(GPU1씩), 총6개 job이다. Launcher는 셸에 남은 다른 정렬/context/stage/NN/학습량 설정을 덮어써 이번 조건을 고정한다. 기본 부모, 세 설정, 저장된 제출 state를 모두 확인한 뒤 첫 job을 제출한다.

| λ | Train name | Eval name |
|---|---|---|
| .0064 | `gs-bo-0p0064` | `gs-bo-e-0p0064` |
| .0128 | `gs-bo-0p0128` | `gs-bo-e-0p0128` |
| .0256 | `gs-bo-0p0256` | `gs-bo-e-0p0256` |

출력: `outputs/nfcgs_order_context/none/legacy_stages3_k5_ste_context_off/lambda<λtag>/job_<TRAIN_ID>/`. 최종 checkpoint는 `checkpoints/nfcgs_main/version_0/step000050000.ckpt`, 평가는 `eval_all/`에 저장한다.

새 state 폴더는 `logs/slurm/sort_context_both_off_rd_20261016/`다. Point별 `both_off_<λtag>.state`에 학습 ID를 먼저 저장한 후 평가를 제출한다. `submissions.tsv`에 제출 요약을 기록한다. QOS 등으로 중간 제출이 거부되면 같은 명령을 다시 실행해 이미 접수된 ID를 재사용하고 빠진 제출을 이어간다. 이 기능은 제출 재시도이며 실패·취소된 학습의 checkpoint resume 기능은 아니다. 다른 state 폴더를 지정하면 별도 실험으로 취급하므로 기본 폴더를 유지한다.

기존 12-point ablation/recovery 명령 대신 위 전용 명령을 실행한다. 새 대조군은 기존 recovery의 15-point 목록에 아직 포함하지 않는다.

## 결과 해석

각 λ에서 actual bytes, PSNR/SSIM/LPIPS, Score 및 MSH stream별 bytes, encode/decode 시간을 비교한다. 정렬은 두 경로의 token 순서를 바꾸고, Context ON/OFF는 Score 공간 예측을 바꾼다. Score 복원값 변화는 MSH 입력 residual에도 영향을 주므로 총 bitstream과 경로별 크기를 함께 확인한다.

- Context OFF에서 Hilbert−기본 순서: 공간 Context 없이 정렬을 적용한 효과.
- 기본 순서에서 Context ON−OFF: 정렬 없이 공간 Context를 적용한 효과.
- Hilbert에서 Context ON−OFF와 기본 순서에서의 ON−OFF 차이: 정렬에 따라 Context 효과가 달라지는 정도.

같은 λ의 지표 차이를 고정 품질에서의 bitrate 절감으로 해석하지 않는다. 품질을 맞춘 RD 비교는 세 point 곡선의 공통 품질 구간에서 수행한다. 현재까지 준비·검증은 로컬에서 수행했으며 서버 제출 ID와 새 학습 결과는 아직 없다.

## 로컬 검증

- Codec 5개 테스트 통과: 두 옵션 OFF에서 identity 순서, 공간 예측 우회·동결, 다른 token에 대한 Score 복원/확률 독립성, 실제 bitstream 왕복, MSH payload 유지, low-rank/MSH gradient, 동일 부모의 weights-only 변환 및 strict checkpoint 검증을 확인했다. CUDA/BF16 학습 테스트 1개는 로컬 PyTorch가 CPU 빌드여서 생략했다.
- CLI/분산 runner 12개 테스트 통과. 새 대조군은 세 λ × train/eval의 6개 조합에서 minimal/legacy3/OFF 설정, weights-only 새50k, 유효배치8, LR schedule, 전체 실제 bitstream eval 설정을 검사했다.
- Slurm 제출·구문 30개 항목 검증 통과. 가짜 Slurm으로 새3개만 제출, 개별 afterok, 같은 부모 사용, 반복 제출 시 ID 재사용, 첫 eval 거부 후 이어서 제출, 잘못된 마지막 state/기존 lock/미수정 launcher에서 제출 전 중단을 확인했다. 가짜 Python/Conda로 실제 train shell을 실행해 Python 시작 전 job별 TMPDIR 생성·전달을 확인했다. 새 shell 파일의 LF·shebang·구문도 검사했다.
- 새 대조군 관련 Slurm 5개 테스트는 Windows 긴 경로를 만들지 않도록 테스트 fixture를 조정한 후 재실행해 모두 통과했다. 서버 GPU 학습·eval·실제 Ceph checkpoint 저장은 아직 실행하지 않았다.

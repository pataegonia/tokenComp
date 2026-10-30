# 정렬·context 실험 로그 확인 — 2026-10-12

10월 12일 08:44~08:46경 내려받은 Slurm 로그 기준이다. 다운로드 시각은 서버의 작업 종료 시각이 아니며, 아래 진행 상황은 실시간 squeue 조회 결과가 아니다. 이 세션의 Score 정렬·context 실험만 포함했다.

## 새로 완료된 학습·eval 5개

모든 학습에서 `Trainer.fit`의 `max_steps=50000` 도달 메시지를 확인했다. Eval은 50k checkpoint를 사용했고 최종 성능·실제 bitstream 크기·encode/decode 시간 요약이 있다. 각 eval의 시간 요약은 워밍업 5 scene을 제외한 6,986 scene이다. 실제 화질·용량 평가 수는 6,991 scene이다. IterableDataset의 progress bar 분모 7,286은 실제 평가 scene 수로 사용하지 않았다.

| 설정 | λ | 학습 job | eval job | PSNR dB | SSIM | LPIPS | 평균 실제 bitstream KiB |
|---|---:|---|---|---:|---:|---:|---:|
| Morton dyadic4 | .0128 | 447729 | 447730 | 23.7023 | .7362 | .2677 | 37.932 |
| Hilbert quarter2 | .0064 | 447738 | 447739 | 23.8168 | .7405 | .2626 | 48.269 |
| Hilbert quarter2 | .0128 | 447740 | 447741 | 23.7032 | .7363 | .2674 | 38.819 |
| Hilbert quarter2 | .0256 | 447742 | 447743 | 23.5015 | .7295 | .2765 | 30.617 |
| Hilbert dyadic4 | .0064 | 447744 | 447745 | 23.8170 | .7403 | .2623 | 46.741 |

평가 조건은 기존 C12/T8, 실제 bitstream encode/decode, 전체 처리 가능한 test scene이다. Score는 minimal/linear/STE/spatial-only, MSH residual은 유지했다. Eval stdout 첫 부분의 checkpoint codec 설정과 `global_step=50000`을 확인했다.

## 현재 완료된 RD point

각 셀은 **PSNR dB / scene당 평균 실제 bitstream KiB**다. Morton legacy3는 RD 비교 대조군이며, 구조상 가장 기본적인 even/odd 모델은 별도로 측정했던 legacy2다.

| 설정 | λ=.0064 | λ=.0128 | λ=.0256 |
|---|---|---|---|
| Morton legacy3 | 23.7910 / 47.525 | 23.6971 / 39.003 | 23.4892 / 30.807 |
| Hilbert legacy3 | 23.7994 / 46.894 | 23.7032 / 38.269 | 23.5050 / 30.290 |
| Morton dyadic4 | 23.7791 / 47.450 | 23.7023 / 37.932 | 23.5089 / 29.984 |
| Hilbert quarter2 | 23.8168 / 48.269 | 23.7032 / 38.819 | 23.5015 / 30.617 |
| Hilbert dyadic4 | 23.8170 / 46.741 | 학습 중 | 학습 중 |

처음 계획한 5개 곡선의 15 point 중 13 point의 eval이 완료됐다. 이후 제출한 Morton quarter2의 3 point와 nn_score/nn_xyz 각 1 point를 합친 20 point 계획에서는 13 point에 결과가 있다. 직전 로그 확인 이후 새로 나온 결과는 5 point다.

## 성능 해석

아래는 같은 λ끼리의 비교이며, 같은 PSNR로 보간한 BD-rate는 아니다.

- **Morton dyadic4 λ=.0128:** Morton legacy3 대비 전체 bitstream −2.746%, Score 부분 −2.996%, PSNR +0.0052 dB다. Entropy 복호는 86.9 → 91.5 ms다. λ=.0256도 전체 −2.673%로, 압축을 강하게 적용한 두 point에서 작은 개선이 있다. λ=.0064에서는 PSNR이 0.0119 dB 낮아, 모든 λ에서 일률적으로 우수하다고 볼 수 없다.
- **Hilbert quarter2:** Hilbert legacy3 대비 전체 bitstream이 λ=.0064/.0128/.0256에서 각각 +2.933%/+1.440%/+1.081%다. PSNR 차이는 +0.0174/0/−0.0035 dB다. 이번 같은 λ 비교에서는 압축 효율의 우위가 보이지 않는다. 다만 entropy 복호는 90.5/87.2/85.0 → 81.3/78.4/82.3 ms로 약 3~10% 짧다. 서로 다른 job에서 측정한 벽시계 시간이므로 속도 차이를 확정하려면 같은 GPU에서 반복 측정해야 한다.
- **Hilbert dyadic4 λ=.0064:** Hilbert legacy3 대비 전체 bitstream −0.326%, PSNR +0.0176 dB다. Morton legacy3와 비교하면 −1.650%, +0.0260 dB다. Entropy 복호는 94.2 ms로 Hilbert legacy3의 90.5 ms보다 길다. 개선 폭은 작으며, 남은 두 λ의 결과가 나온 뒤 전체 곡선을 비교한다.

## 학습이 진행 중인 로그

모두 GPU2, batch/GPU2, accumulation2, 유효 batch8, 목표 50k optimizer steps다. 진행 step은 마지막으로 완전하게 기록된 progress bar 행에서 추정했다. 1epoch 33,017 local batch, accumulation2이며 epoch 끝의 남은 batch update를 포함해 환산했다.

| 설정 | λ | 학습 job | 약 step / 50k | 학습 잔여 추정 |
|---|---:|---|---:|---:|
| Hilbert dyadic4 | .0128 | 447746 | 33,631 | 5.9시간 |
| Hilbert dyadic4 | .0256 | 447900 | 30,231 | 7.0시간 |
| Morton quarter2 | .0064 | 448644 | 33,703 | 5.4시간 |
| Morton quarter2 | .0128 | 448646 | 31,111 | 6.1시간 |
| Morton quarter2 | .0256 | 448648 | 24,999 | 8.3시간 |

잔여 시간은 받은 로그의 마지막 it/s로 계산한 **학습만의 추정값**이다. Checkpoint, validation, 속도 변동, 후속 eval, queue 대기는 포함하지 않는다. 제출 기록상 Morton quarter2의 eval은 448645/448647/448649다.

Hilbert dyadic4의 두 λ에 실제 학습 로그가 있어 이전의 제출·시작 미확인 상태는 해소됐다. 다만 이 두 job에 대응하는 eval ID는 이번 TSV에 추가되지 않아 로컬 자료만으로는 확인할 수 없다.

## 제출은 확인되지만 학습 로그가 없는 두 개

`slurm/order_context_followup_20261011/submissions.tsv`와 point별 state에 추가 학습 5개와 eval 5개가 기록돼 있다.

| 설정 | λ | 학습 job | eval job | 이번 로컬 자료 |
|---|---:|---|---|---|
| nn_score legacy3 | .0256 | 448650 | 448651 | 제출 기록 있음, 학습 stdout/stderr 없음 |
| nn_xyz legacy3 | .0256 | 448652 | 448653 | 제출 기록 있음, 학습 stdout/stderr 없음 |

이 두 개는 제출됐다. 로그가 없다는 사실만으로 현재 PENDING/RUNNING/FAILED 중 어떤 상태인지 판단할 수 없다.

이후 사용자가 2026-10-12에 nn_score 학습 448650과 nn_xyz 학습 448652가 queue에 있다고 확인했다. 두 작업은 제출돼 대기 중인 것으로 관리한다. 이는 위 다운로드 로그와 별도로 사용자가 제공한 최신 상태다.

## 오류 점검과 원본

받은 order/context 학습 17개와 eval 12개의 stdout/stderr에서 Traceback, CUDA OOM, RuntimeError 등, Slurm cancellation/time limit, NaN/Inf 표기를 찾지 못했다. Torchvision deprecated, IterableDataset length, DDP gradient stride 등은 경고이며 50k 완료 job에도 같은 종류의 경고가 있다.

- 원본 로그: `slurm/slurm-gs-order-context[-eval]-<job>.out`와 대응 `.err`
- Morton legacy3 λ=.0256의 기존 결과: `slurm/slurm-gs-score-stages-eval-441553_1.out`
- 분석·비교 수치: `artifacts/order_context_eval_2026-10-12.json`
- 분석 재실행: `python artifacts/review_order_context_logs_2026-10-12.py`
- 이전 확인: [2026-10-11](NFCGS_ORDER_CONTEXT_LOG_REVIEW_2026-10-11.md)

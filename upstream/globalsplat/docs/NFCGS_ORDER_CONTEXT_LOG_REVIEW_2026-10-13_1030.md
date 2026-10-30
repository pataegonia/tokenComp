# 정렬·context 로그 확인 — 2026-10-13 10:29 다운로드

10월 13일 10:29경 내려받은 새 Slurm 로그를 확인했다. 이전 결과와 진행률 snapshot은 별도로 보존했다. 아래 실행 상황은 실시간 squeue 조회 결과가 아니다.

## 기본 2단계 λ=.0128의 학습·eval 완료

학습 449561에서 `Trainer.fit`의 `max_steps=50000` 도달을 확인했다. 후속 eval 449562는 최종 `step000050000.ckpt`를 사용했고 성능·bitstream·시간 요약까지 출력했다. 설정은 Morton legacy2, kernel5/STE, Score minimal/linear/spatial-only, MSH 유지이며 C12/T8의 실제 bitstream 전체 scene eval이다.

| 항목 | 결과 |
|---|---:|
| PSNR | 23.6313 dB |
| SSIM | .7339 |
| LPIPS | .2703 |
| 평균 실제 bitstream | 40.835 KiB/scene |
| 평균 Score payload | 26,706.448 B/scene |
| 평균 MSH y / z payload | 14,859.973 / 116.743 B/scene |
| Codec encode / decode | 141.4 / 81.4 ms/scene |
| 전체 sender / receiver | 189.7 / 84.0 ms/scene |

평가 대상은 6,991 scene이다. 시간 평균은 워밍업 5 scene을 제외한 6,986 scene에서 계산했다. 최종 metric이 있는 새 eval은 이번 다운로드에서 한 개다.

## 완성된 RD 곡선 7개

각 셀은 **PSNR dB / 평균 실제 bitstream KiB/scene**이다. 기존 기본 2단계 λ=.0256 결과 442210_0을 재사용해 baseline 곡선을 완성했다. 전체 비교는 7개 설정 × 3λ = 21개 측정 point다.

| 설정 | λ=.0064 | λ=.0128 | λ=.0256 |
|---|---|---|---|
| Morton 기본 legacy2 | 23.7620 / 50.460 | 23.6313 / 40.835 | 23.4172 / 32.671 |
| Morton legacy3 | 23.7910 / 47.525 | 23.6971 / 39.003 | 23.4892 / 30.807 |
| Hilbert legacy3 | 23.7994 / 46.894 | 23.7032 / 38.269 | 23.5050 / 30.290 |
| Morton dyadic4 | 23.7791 / 47.450 | 23.7023 / 37.932 | 23.5089 / 29.984 |
| Hilbert dyadic4 | 23.8170 / 46.741 | 23.7021 / 37.179 | 23.5173 / 29.399 |
| Morton quarter2 | 23.7865 / 48.340 | 23.6325 / 39.010 | 23.4844 / 31.151 |
| Hilbert quarter2 | 23.8168 / 48.269 | 23.7032 / 38.819 | 23.5015 / 30.617 |

표의 Morton dyadic4는 계층형 anchor schedule이며, 기존 A→B→C→D의 legacy4와 구분한다. 기본 .0256은 GPU4·accumulation1, 새 두 λ는 GPU2·accumulation2였으며 유효배치8과 부모·학습 recipe는 유지했다.

## 기본 2단계에 대한 개선

아래는 같은 λ끼리 비교한 변화다. 같은 PSNR로 보간한 BD-rate나 반복 학습의 통계적 유의성을 측정한 값은 아니다.

| Hilbert dyadic4의 λ | 전체 크기 변화 | PSNR 변화 | Codec 복호 시간: 기본2 → dyadic4 |
|---:|---:|---:|---:|
| .0064 | −7.370% | +.0550 dB | 82.3 → 94.2 ms |
| .0128 | −8.954% | +.0708 dB | 81.4 → 89.9 ms |
| .0256 | −10.017% | +.1001 dB | 81.4 → 92.7 ms |

현재 비교에서는 Hilbert dyadic4가 기본 2단계보다 세 λ 모두 bitstream이 작고 PSNR이 높다. 시간은 더 길게 측정됐지만 다른 job/노드의 벽시계 시간이라 알고리즘만의 차이로 확정하지 않는다. Quarter2도 기본 2단계보다는 압축 효율이 개선됐으며, 기존에 우위가 보이지 않는다고 설명한 비교 대상은 같은 정렬의 legacy3였다.

## 남아 있는 NN 학습 2개

진행 step은 마지막 완전한 progress bar에서 1epoch 33,017 local batch와 accumulation2를 반영한 근사값이다. 잔여 시간은 마지막 it/s를 유지할 때의 학습 시간이며 checkpoint/validation, 후속 eval과 대기는 포함하지 않는다.

| 설정 | 학습 job | 후속 eval | 약 step / 50k | 속도 batch/s | 학습 잔여 추정 |
|---|---|---|---:|---:|---:|
| nn_score legacy3, λ=.0256 | 448650 | 448651 | 7,895 | .22 | 106.3시간 |
| nn_xyz legacy3, λ=.0256 | 448652 | 448653 | 16,289 | .46 | 40.7시간 |

nn_xyz progress bar의 99%는 첫 epoch의 진행률이며 전체 50k의 완료율은 약 32.6%다. 두 NN 학습 모두 50k 완료 메시지가 없고 후속 eval 로그도 아직 없다. 마지막으로 사용자가 확인한 nn_score의 실제 제한은 96시간, 종료 예정은 10월 16일 14:17:39였다. 이 제한이 유지됐다면 현재 속도에서도 50k까지 약 30시간이 부족하다. 서버에서 시간 제한을 새로 조회하거나 실행 중인 작업을 변경하지 않았다.

## 오류 점검과 저장 자료

받은 order/context 학습 21개와 eval 19개의 stdout/stderr에서 Traceback, CUDA OOM, RuntimeError, Slurm cancellation/time limit, NaN/Inf 표기를 찾지 못했다. 나머지 측정 두 point는 기존 score-stages eval 441553_1과 442210_0을 재사용했다.

- 새 원본: `slurm/slurm-gs-order-context-eval-449562.out`와 대응 `.err`.
- 분석 수치: `artifacts/order_context_eval_2026-10-13-1030.json`.
- 분석 재실행: `python artifacts/review_order_context_logs_2026-10-13-1030.py`.
- RD 그래프: `artifacts/order_context_rd_2026-10-13-1030.png`.
- 그래프 생성: `python artifacts/plot_order_context_rd_2026-10-13-1030.py`.
- 이전 기록: [10월 13일 오전](NFCGS_ORDER_CONTEXT_LOG_REVIEW_2026-10-13.md).

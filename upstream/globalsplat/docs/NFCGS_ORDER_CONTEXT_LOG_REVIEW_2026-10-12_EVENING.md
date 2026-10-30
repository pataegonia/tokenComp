# 정렬·context 실험 로그 확인 — 2026-10-12 저녁

10월 12일 18:24~18:25경 내려받은 로컬 Slurm 로그를 기준으로 정리했다. 실시간 squeue 조회 결과가 아니다. 오전 기록과 분석 JSON은 덮어쓰지 않고 별도로 보존했다. 다른 프로젝트의 Hyper1D 실험은 포함하지 않았다.

## 새로 완료된 학습·eval 5개

아래 학습 5개는 stderr의 `Trainer.fit`에서 `max_steps=50000` 도달을 확인했다. Eval은 `global_step=50000` checkpoint, C12/T8, `actual_bitstream=true`, `max_scenes=null`을 사용했다. 각 eval에 최종 성능·용량·파이프라인 시간 요약이 있다. 시간 평균은 워밍업 5 scene을 제외한 6,986 scene이며 실제 화질·용량 평가 수는 6,991 scene이다.

| 설정 | λ | 학습 job | eval job | PSNR dB | SSIM | LPIPS | 평균 실제 bitstream KiB |
|---|---:|---|---|---:|---:|---:|---:|
| Hilbert dyadic4 | .0128 | 447746 | 447899 | 23.7021 | .7363 | .2673 | 37.179 |
| Hilbert dyadic4 | .0256 | 447900 | 447901 | 23.5173 | .7298 | .2752 | 29.399 |
| Morton quarter2 | .0064 | 448644 | 448645 | 23.7865 | .7391 | .2634 | 48.340 |
| Morton quarter2 | .0128 | 448646 | 448647 | 23.6325 | .7335 | .2691 | 39.010 |
| Morton quarter2 | .0256 | 448648 | 448649 | 23.4844 | .7292 | .2771 | 31.151 |

Hilbert dyadic4의 eval ID는 이번 로그의 checkpoint 입력에서 확인했다. 해당 eval의 제출·실행이 확인돼 오전의 eval ID 미확인 상태는 해소됐다.

## 완성된 RD 곡선 6개

각 셀은 **PSNR dB / scene당 평균 실제 bitstream KiB**다. 모두 Score minimal/linear/STE/spatial-only이며 MSH를 유지한 50k 모델이다.

| 설정 | λ=.0064 | λ=.0128 | λ=.0256 |
|---|---|---|---|
| Morton legacy3 | 23.7910 / 47.525 | 23.6971 / 39.003 | 23.4892 / 30.807 |
| Hilbert legacy3 | 23.7994 / 46.894 | 23.7032 / 38.269 | 23.5050 / 30.290 |
| Morton dyadic4 | 23.7791 / 47.450 | 23.7023 / 37.932 | 23.5089 / 29.984 |
| Hilbert dyadic4 | 23.8170 / 46.741 | 23.7021 / 37.179 | 23.5173 / 29.399 |
| Morton quarter2 | 23.7865 / 48.340 | 23.6325 / 39.010 | 23.4844 / 31.151 |
| Hilbert quarter2 | 23.8168 / 48.269 | 23.7032 / 38.819 | 23.5015 / 30.617 |

현재 정렬·context 비교에서 18 point의 결과가 있다. 최근 추가한 기본 Morton legacy2의 두 point와 nn_score/nn_xyz 각 한 point는 아직 학습 중이다. 기존 Morton legacy2 kernel5/STE λ=.0256 결과 23.4172 dB / 32.671 KiB는 별도 실험 442210_0에 있으며, 기본 2단계 곡선은 아직 완성되지 않았다.

## 결과 해석

아래는 같은 λ끼리의 비교이며 같은 PSNR로 보간한 BD-rate는 아니다. 단일 학습 실행의 결과로 통계적 유의성은 측정하지 않았다.

- **Hilbert dyadic4:** Morton legacy3 대비 λ=.0064/.0128/.0256 전체 bitstream 변화는 −1.650%/−4.678%/−4.573%, PSNR 변화는 +0.0260/+0.0050/+0.0281 dB다. Hilbert legacy3 대비 크기는 −0.326%/−2.848%/−2.942%, PSNR은 +0.0176/−0.0011/+0.0123 dB다. 이번 여섯 곡선에서는 가장 유망한 압축 설정이다. 개선 폭은 수% 수준이며 화질의 큰 상승으로 해석하지 않는다.
- **Morton quarter2:** Morton legacy3 대비 크기는 +1.714%/+0.016%/+1.115%, PSNR은 −0.0045/−0.0646/−0.0048 dB다. 이번 세 point에서는 압축 효율의 우위가 보이지 않는다.
- **Quarter2의 정렬 변경:** 같은 quarter2에서 Hilbert는 Morton보다 세 λ 모두 PSNR이 높고 bitstream도 작다. λ=.0128은 +0.0707 dB다. 다만 Hilbert quarter2 자체는 Hilbert legacy3보다 용량이 더 크므로, 정렬 개선과 context 개선을 구분한다.
- **복호 시간:** 새 Hilbert dyadic4의 entropy_decode는 .0128에서 89.9 ms, .0256에서 92.7 ms로 Hilbert legacy3의 87.2/85.0 ms보다 길다. Morton quarter2는 83.8/82.4/82.6 ms다. 다른 job과 노드에서 측정한 시간이므로 알고리즘만의 속도 차이라고 단정하지 않는다.

## 학습이 시작된 4개

아래는 로그 마지막의 완전한 progress bar에서 계산한 근사 optimizer step이다. 1epoch 33,017 local batch와 accumulation2를 반영했다. 잔여 시간은 마지막 it/s 기준으로 추정한 학습 시간이며 checkpoint/validation, 속도 변동, 후속 eval, queue 대기는 포함하지 않는다.

| 설정 | λ | 학습 job | 후속 eval | 약 step / 50k | 속도 batch/s | 학습 잔여 추정 |
|---|---:|---|---|---:|---:|---:|
| nn_score legacy3 | .0256 | 448650 | 448651 | 1,572 | .22 | 122.3시간 |
| nn_xyz legacy3 | .0256 | 448652 | 448653 | 3,017 | .46 | 56.7시간 |
| Morton legacy2 | .0064 | 449559 | 449560 | 9,594 | 1.71 | 13.1시간 |
| Morton legacy2 | .0128 | 449561 | 449562 | 4,144 | 1.67 | 15.3시간 |

모두 GPU2, batch/GPU2, accumulation2, 유효 batch8, MSH ON, 목표 50k로 실행 중이다. 이전에 queue에 있던 nn_score/nn_xyz도 실제 학습을 시작했다. 이번 네 학습에는 50k 완료 메시지가 없고 후속 eval의 로컬 로그도 아직 없다.

### nn_score의 시간 제한

nn_score progress bar는 3,144 batch에 4시간 3분을 기록했다. 목표 50k까지 약 122시간, 전체 약 126시간이 필요할 전망이다. 이후 사용자가 제공한 실제 `scontrol show job 448650` 결과는 다음과 같다.

```text
JobState=RUNNING Reason=None Dependency=(null)
RunTime=04:12:17 TimeLimit=4-00:00:00
StartTime=2026-10-12T14:17:39
EndTime=2026-10-16T14:17:39
```

실제 시간 제한도 96시간이었다. 현재 속도가 이어지면 50k 도달에 필요한 총 시간이 제한보다 약 30시간 길다. 시간 제한을 연장하거나 저장된 checkpoint에서 학습을 이어야 한다. 사용자가 확인한 시점에 job에 남은 허용 시간은 약 91.8시간이며, 속도가 바뀌면 예상 학습 시간도 달라진다.

NN 정렬은 `token_order.py`에서 CPU로 N×N 거리 행렬과 순차 greedy 경로를 매 입력마다 계산한다. nn_score는 56차원 L1 거리를 사용해 추가 계산의 부담이 클 것으로 보인다. 다만 GPU/CPU별 profile을 측정하지 않아 속도 저하의 기여도를 확정하지 않았다. 실행 중인 job의 설정과 checkpoint는 변경하지 않았다.

## 오류 점검과 저장 자료

받은 order/context 학습 21개와 eval 17개의 stdout/stderr에서 Traceback, CUDA OOM, RuntimeError, Slurm cancellation/time limit, NaN/Inf 표기를 찾지 못했다. 완성된 곡선의 나머지 한 point는 기존 score-stages eval 441553_1에서 가져왔다. Torchvision deprecated, IterableDataset length, DDP gradient stride/find_unused_parameters 등의 경고는 있다.

- 분석 수치: `artifacts/order_context_eval_2026-10-12-evening.json`
- 분석 재실행: `python artifacts/review_order_context_logs_2026-10-12-evening.py`
- RD 그래프: `artifacts/order_context_rd_2026-10-12-evening.png`
- 그래프 생성: `python artifacts/plot_order_context_rd_2026-10-12-evening.py`
- 오전 기록: [2026-10-12](NFCGS_ORDER_CONTEXT_LOG_REVIEW_2026-10-12.md)

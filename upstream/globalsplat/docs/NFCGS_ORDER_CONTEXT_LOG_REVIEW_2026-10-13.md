# 정렬·context 실험 로그 확인 — 2026-10-13

10월 13일 09:00~09:02경 내려받은 로그를 확인했다. 아래 진행률은 실시간 squeue 조회 결과가 아니다. 이 세션의 Score 정렬·context 실험만 포함했다. 이전 snapshot JSON과 보고서는 별도로 보존했다.

## 새로 완료된 기본 2단계 λ=.0064

Morton legacy2, kernel5/STE, minimal, MSH 유지 설정의 학습 449559가 50k를 완료했다. 후속 eval 449560은 `global_step=50000` checkpoint, C12/T8, 실제 bitstream, 전체 처리 가능한 test scene을 사용했고 최종 성능·시간 요약까지 저장했다.

| 항목 | 결과 |
|---|---:|
| PSNR | 23.7620 dB |
| SSIM | .7386 |
| LPIPS | .2650 |
| 평균 실제 bitstream | 50.460 KiB/scene |
| Score payload | 32,634.074 B/scene |
| MSH y payload | 18,778.793 B/scene |
| MSH z payload | 126.469 B/scene |
| Codec encode | 142.7 ms/scene |
| Codec decode | 82.3 ms/scene |
| 전체 sender / receiver | 190.9 / 84.9 ms/scene |

화질·용량 평가는 6,991 scene이고 시간 평균은 워밍업 5 scene을 제외한 6,986 scene이다. 재구성 mean payload는 0 B이며 기존 minimal 조건과 일치한다.

## 같은 λ=.0064의 비교

| 설정 | PSNR dB | 평균 실제 bitstream KiB | Codec decode ms |
|---|---:|---:|---:|
| Morton 기본 2단계 | 23.7620 | 50.460 | 82.3 |
| Morton legacy3 | 23.7910 | 47.525 | 90.3 |
| Hilbert dyadic4 | 23.8170 | 46.741 | 94.2 |

Morton legacy3는 기본 2단계 대비 크기가 약 5.82% 작고 PSNR은 .0290 dB 높다. Hilbert dyadic4는 기본 2단계 대비 크기가 약 7.37% 작고 PSNR은 .0550 dB 높다. 단계가 추가된 모델에서 이번 point의 압축 개선은 보이지만 복호 시간은 길게 측정됐다. 서로 다른 job/노드의 평균 시간이며 같은 λ의 비교다. 같은 PSNR로 보간한 BD-rate나 통제된 하드웨어 속도 차이로 해석하지 않는다.

## 기본 2단계 RD 곡선의 상태

| λ | 학습 / eval job | 결과 또는 이번 로그 상태 |
|---:|---|---|
| .0064 | 449559 / 449560 | 완료: 23.7620 dB / 50.460 KiB |
| .0128 | 449561 / 449562 | 학습 약 48,955 / 50k; eval 로그 아직 없음 |
| .0256 | 442209_0 / 442210_0 | 기존 완료: 23.4172 dB / 32.671 KiB |

새로 제출한 두 λ 중 .0064가 완료돼 기존 .0256과 함께 두 point에 결과가 있다. .0128 결과가 나와야 기본 2단계의 세 point가 완성된다. 이번 정렬/context 분석 JSON에는 기존 18개와 새 .0064를 합친 19개 eval이 있으며, 별도 파일의 기존 기본 2단계 .0256은 위 표에서만 참고했다.

## 아직 학습 중인 로그

모두 GPU2, batch/GPU2, accumulation2, 유효 batch8, 목표 50k다. 진행 step은 마지막 완전한 progress bar 행에서 1epoch 33,017 local batch와 accumulation2를 반영해 계산했다. 표의 잔여 시간은 마지막 it/s를 유지했을 때의 학습 시간만이며 checkpoint/validation, 후속 eval, 대기 시간을 제외한다.

| 설정 | λ | 학습 job | 후속 eval | 약 step / 50k | 속도 batch/s | 학습 잔여 추정 |
|---|---:|---|---|---:|---:|---:|
| nn_score legacy3 | .0256 | 448650 | 448651 | 7,322 | .22 | 107.8시간 |
| nn_xyz legacy3 | .0256 | 448652 | 448653 | 15,046 | .46 | 42.2시간 |
| Morton legacy2 | .0128 | 449561 | 449562 | 48,955 | 1.73 | 약 20분 |

기본 2단계 .0128 progress bar의 현재 epoch 종료 ETA는 약 11분이지만, 다음 epoch의 남은 update까지 포함한 50k 도달 ETA는 약 20분이다. 이 학습의 후속 eval ID는 449562로 제출 TSV에서 확인했다.

## nn_score 시간 제한 재확인

이번 로그에서 nn_score의 속도는 이전과 같은 .22 batch/s다. Epoch0 진행에 약 18시간39분이 걸렸고 50k까지 약 108시간이 더 필요해 총 학습 예상은 여전히 약 126시간이다.

사용자가 10월 12일 확인해 준 마지막 실제 Slurm 설정은 다음과 같다. 오늘 서버에서 다시 조회한 값은 아니다.

```text
JobState=RUNNING
TimeLimit=4-00:00:00
StartTime=2026-10-12T14:17:39
EndTime=2026-10-16T14:17:39
```

이 96시간 제한이 유지됐다면 현재 속도로는 50k 도달에 약 30시간이 부족하다. 제한 연장 또는 저장 checkpoint에서 이어 학습하는 조치가 필요하다. 실행 중인 job과 checkpoint에는 변경하지 않았다.

## 오류 점검과 원본

받은 order/context 학습 21개와 eval 18개의 stdout/stderr에서 Traceback, CUDA OOM, RuntimeError, Slurm cancellation/time limit, NaN/Inf 표기를 찾지 못했다. DDP gradient stride/find_unused_parameters와 Torchvision 관련 표기는 기존 경고다.

- 새 eval 로그: 저장소 루트 `slurm/slurm-gs-order-context-eval-449560.out`와 대응 `.err`.
- 분석 수치: `artifacts/order_context_eval_2026-10-13.json`.
- 분석 재실행: `python artifacts/review_order_context_logs_2026-10-13.py`.
- 기본 2단계 제출 기록: `slurm/order_context_legacy2_rd_20261012/submissions.tsv`.
- 이전 결과: [10월 12일 저녁](NFCGS_ORDER_CONTEXT_LOG_REVIEW_2026-10-12_EVENING.md).

# 정렬·context 실험 로그 확인 — 2026-10-11

10월 11일 09:24~09:25경 내려받은 Slurm 로그를 기준으로 확인했다. 파일의 다운로드 시각은 서버에서 각 작업이 종료된 시각과 같지 않다. 최신 `squeue`를 조회한 결과가 아니므로 현재 서버 상태로 단정하지 않는다. 이 세션의 Morton/Hilbert 정렬·Score context 실험만 포함한다.

## 새로 완료된 작업

아래 학습 5개는 `Trainer.fit`의 `max_steps=50000` 도달 메시지를 확인했다. 각 후속 eval에는 최종 metric 요약과 entropy encode/decode 시간 요약이 있어 정상적으로 평가를 마쳤다.

| 설정 | λ | 학습 job | eval job |
|---|---:|---|---|
| Morton legacy3 | .0064 | 447719 | 447720 |
| Morton legacy3 | .0128 | 447721 | 447722 |
| Hilbert legacy3 | .0064 | 447723 | 447724 |
| Hilbert legacy3 | .0128 | 447725 | 447726 |
| Morton dyadic4 | .0064 | 447727 | 447728 |

기존 λ=.0256 결과 3개를 합하면 계획한 15개 point 중 8개에 학습·eval 결과가 있다. Morton legacy3와 Hilbert legacy3는 각각 3-point 곡선이 완성됐다. Morton dyadic4는 .0128 결과가 남아 있다. Hilbert quarter2/dyadic4의 eval 결과는 아직 받은 로그에 없다.

## 완료된 결과

각 셀은 **PSNR dB / scene당 평균 실제 bitstream KiB**다. 모든 point는 MSH residual을 유지하고, Score path의 centering·score norm·mean/channel context·nonlinear를 제거한 minimal 설정이다. Spatial context의 양자화는 STE다.

| 설정 | λ=.0064 | λ=.0128 | λ=.0256 |
|---|---|---|---|
| Morton legacy3 | 23.7910 / 47.525 | 23.6971 / 39.003 | 23.4892 / 30.807 |
| Hilbert legacy3 | 23.7994 / 46.894 | 23.7032 / 38.269 | 23.5050 / 30.290 |
| Morton dyadic4 | 23.7791 / 47.450 | 학습 중인 로그 | 23.5089 / 29.984 |

동일 λ에서 Hilbert legacy3를 Morton legacy3와 비교하면 아래와 같다. 이 용량 차이는 같은 λ끼리 비교한 값이며, 같은 PSNR로 보간한 BD-rate는 아니다.

| λ | 전체 bitstream 변화 | PSNR 변화 |
|---|---:|---:|
| .0064 | −1.328% | +0.0084 dB |
| .0128 | −1.884% | +0.0061 dB |
| .0256 | −1.680% | +0.0158 dB |

Hilbert 정렬은 이번 세 point에서 일관되게 작은 개선을 보였다. 단일 학습 실행의 결과이므로 반복 실험에 대한 통계적 유의성을 주장하지 않는다.

Morton dyadic4의 λ=.0064는 Morton legacy3 대비 전체 크기 −0.159%, PSNR −0.0119 dB였다. Score 크기는 오히려 0.855% 증가했고 MSH 감소가 이를 상쇄했다. Entropy 복호 시간은 90.3 ms에서 95.2 ms로 늘었다. 기존 λ=.0256에서는 전체 크기 −2.673%, PSNR +0.0197 dB였으므로, dyadic4의 이득이 모든 λ에서 같다고 결론 내리지 않는다.

## 아직 학습 중인 로그

진행 step은 progress bar의 epoch·batch 수와 accumulation2로 계산한 근사 optimizer step이다. 잔여 시간은 로그 마지막 속도에서 계산한 **학습만의 추정값**이다. 이후 속도 변화, checkpoint/validation 시간, 후속 eval, queue 대기를 포함하지 않는다.

| 설정 | λ | 학습 job | 진행 / 50k | 학습 잔여 추정 |
|---|---:|---|---:|---:|
| Morton dyadic4 | .0128 | 447729 | 약 16,295 | 약 12.1시간 |
| Hilbert quarter2 | .0064 | 447738 | 약 16,756 | 약 11.2시간 |
| Hilbert quarter2 | .0128 | 447740 | 약 17,074 | 약 11.0시간 |
| Hilbert quarter2 | .0256 | 447742 | 약 13,408 | 약 12.2시간 |
| Hilbert dyadic4 | .0064 | 447744 | 약 10,170 | 약 14.9시간 |

위 5개는 지난 queue에서 대기 중이었으나 이번 로그에서는 실제 학습을 시작한 것이 확인된다. 각 job의 후속 eval은 기존 제출 기록에 있다.

## 여전히 확인할 부분

- Hilbert dyadic4 λ=.0128 학습 447746: 이번 다운로드에 로그가 없다. 마지막으로 사용자가 제공한 queue에서는 GPU 사용자 한도로 대기 중이었다. 현재 상태는 미확인이다.
- 위 447746에 연결할 후속 eval: 보충 명령을 안내했지만 생성된 job ID는 아직 확인하지 못했다.
- Hilbert dyadic4 λ=.0256 학습·eval: 보충 명령 실행 후 job ID와 로그가 아직 없다. 미제출 또는 현재 대기로 단정하지 않는다.

## 오류 점검과 원본

이번 정렬·context 학습 및 eval의 받은 stdout/stderr에서 Traceback, CUDA OOM, Slurm cancellation/time limit, NaN/Inf 표기를 찾지 못했다. DDP gradient stride 및 `find_unused_parameters` 경고는 있으나, 완료한 학습에서도 같은 경고가 나타났으며 종료 오류는 아니다.

원본은 저장소 루트의 `slurm/slurm-gs-order-context[-eval]-<job>.out` 및 대응 `.err`다. Morton legacy3 λ=.0256은 `slurm/slurm-gs-score-stages-eval-441553_1.out`을 사용했다. 제출 기록은 `slurm/order_context_rd_20261010_102932_4064671.tsv`와 `slurm/order_context_rd_20261010_104409_698699.tsv`다.

- 수치 원본 정리: `artifacts/order_context_eval_2026-10-11.json`
- RD 그래프: `artifacts/order_context_rd_2026-10-11.png`
- 이전 계획·제출 현황: [NFCGS_ORDER_CONTEXT_RD_2026-10-10.md](NFCGS_ORDER_CONTEXT_RD_2026-10-10.md)

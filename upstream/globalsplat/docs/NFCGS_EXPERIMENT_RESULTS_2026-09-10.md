# GlobalSplat 압축 실험 및 결과 — 2026-09-10

**후속 업데이트:** [2026-09-11 Nonlinear+Spatial/Full 결과](NFCGS_NONLINEAR_CONTEXT_RESULTS_2026-09-11.md).
아래는 9월 10일 스냅샷이며, 당시 학습 중이던 4개가 모두 완료되어 최신 codec 결과 수는 40개다.

로컬에 받은 로그의 최신 복사 시각 약 17:28 KST 기준. 서버의 실시간 squeue 상태는 아니다.
이번 대화에서 다룬 paper24 계열 실험을 정리했다. 이전 초기 탐색 결과는
[2026-09-02 보고서](ABLATION_PROGRESS_2026-09-02.md)에 별도로 보존되어 있다.

**결과가 있는 codec 설정 36개**: 초기 Linear 12개(사용자 제공 표), Nonlinear32 8개,
Linear continuation 4개, Linear context 8개, Nonlinear continuation 4개.
여기에 공식 codec-OFF reference 1개가 있다. 후속 학습 4개는 아직 최종 eval 결과가 없다.

## 공통 조건과 숫자 해석

- GlobalSplat scene token을 압축하고 공유된 Gaussian decoder로 복원한다.
  공통 backbone 동결 정책을 유지하며 codec을 학습한다.
- 실측 eval은 RE10K all-test protocol, context 12 / target 8, batch 1,
  actual-bitstream encode/decode 경로를 사용한다.
- 원본 test index는 7,286개이지만 완료된 24개 eval 로그 모두 마지막 처리 수가
  6,991/7,286이고 최종 집계가 있다. 실제 평가 가능한 장면 수를 6,991로 보고한다.
  엄밀한 paired 비교에는 actual_rate_per_scene.json의 scene/frame ID 일치도 확인한다.
- 로그에서 추출한 크기는 scene당 actual_bytes 평균 / 1024, 즉 KiB이다.
  score + residual_y + residual_z + scene mean + container 비용을 포함한다.
  공유 codec weight는 scene payload에 포함하지 않는다.
- 초기 사용자 표의 크기는 원래 kB로 표기되었다. 원시 bytes를 이 로컬 로그 묶음에서
  확인하지 못했으므로 아래 첫 표는 값을 그대로 옮기고 단위를 구분한다.
  초기 표와 최신 표의 절감률을 계산하려면 초기 값도 bytes/1024였는지 확인해야 한다.
- 50k+50k는 원래 50k weight에서 시작해 optimizer/scheduler를 초기화하고 50k를 더 학습한 것.
  optimizer state까지 이어받은 uninterrupted 100k 학습과는 구분한다.
- continuation/context 실험은 micro-batch 2, accumulation 4, effective batch 8,
  paper24 subset consistency, bf16-mixed, seed 111123을 맞췄다.
  독립 GPU task 수는 모델별 batch/step 수를 바꾸지 않는다.
- 현재 context는 예측값과 scene별 양자화 간격도 변경한다. 결과는 joint training된
  context 구조 전체의 효과이며, 동일 symbol에 PMF만 교체한 효과는 아니다.
- 평균의 미세한 차이는 단일 실행의 관측값이다. 동일 λ의 크기 변화는 동일 화질
  절감률이나 BD-rate가 아니며, 작은 차이의 재현성은 paired 통계/반복 seed로 확인한다.

## 실험별 검증 질문

| 실험 | 검증하려는 것 | 대조 방법 | 현재 상태 |
|---|---|---|---|
| 초기 Linear rank/residual/Morton | rank 증가, residual, 공간 정렬의 실효성 | 같은 λ에서 한 옵션씩 비교 | 12개 결과 |
| Nonlinear32 | 선형 analysis/synthesis의 표현력 한계 | rank/λ/residual/Morton을 맞춘 Linear와 비교 | 8개 완료 |
| Linear factorized continuation | 추가 50k 학습 효과 및 Rank80의 수렴 부족 가능성 | context OFF로 동일하게 추가 50k | 4개 완료 |
| Linear score-context | mean/channel/spatial을 활용하면 RD가 좋아지는가 | 동일 50k+50k factorized control과 비교 | 8개 완료 |
| Nonlinear factorized continuation | nonlinear 결합 실험의 추가 학습 대조군 | nonlinear context OFF로 추가 50k | Rank56/80 총 4개 완료 |
| Nonlinear + Spatial/Full | transform과 context의 이득이 결합되는가 | nonlinear continuation 및 linear context와 비교 | 4개 학습 중 로그 |

## 1. 초기 Linear — 사용자 제공 baseline 표

초기 50k 기준. 공식 codec-OFF reference: **PSNR 24.7004 / SSIM .7682 / LPIPS .2480**.
Codec-OFF의 scene bitstream 크기는 제공되지 않았다.

| Rank | λ | Residual | Morton | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 크기(원표 kB 표기) ↓ |
|---:|---:|---|---|---:|---:|---:|---:|
| 56 | .0064 | ON | ON | 24.5021 | 0.7663 | 0.2342 | 102.68 |
| 56 | .0064 | OFF | ON | 24.2426 | 0.7582 | 0.2460 | 111.30 |
| 56 | .0064 | ON | OFF | 24.4920 | 0.7662 | 0.2356 | 105.45 |
| 56 | .0256 | ON | ON | 23.6659 | 0.7368 | 0.2701 | 48.53 |
| 56 | .0256 | OFF | ON | 23.4941 | 0.7311 | 0.2750 | 58.58 |
| 56 | .0256 | ON | OFF | 23.7240 | 0.7393 | 0.2668 | 53.08 |
| 80 | .0064 | ON | ON | 24.4776 | 0.7653 | 0.2371 | 103.84 |
| 80 | .0064 | OFF | ON | 24.2327 | 0.7577 | 0.2457 | 115.63 |
| 80 | .0064 | ON | OFF | 24.4845 | 0.7661 | 0.2370 | 110.21 |
| 80 | .0256 | ON | ON | 23.5391 | 0.7346 | 0.2764 | 48.12 |
| 80 | .0256 | OFF | ON | 23.4823 | 0.7313 | 0.2746 | 58.53 |
| 80 | .0256 | ON | OFF | 23.6440 | 0.7386 | 0.2690 | 54.54 |

관측:
- Residual ON은 각 rank/λ의 Morton-ON 조건에서 OFF보다 크기와 PSNR 모두 좋다.
- Morton ON은 residual-ON 모델의 크기를 줄이지만, 낮은 bitrate에서는 Morton OFF의
  PSNR/SSIM이 조금 더 높은 경우가 있다. Morton의 모든 지표 우세를 주장할 수는 없다.
- Rank80은 channel 수를 늘린 만큼 일관된 화질 이득을 만들지 못한다.

## 2. Nonlinear32 — 초기 50k

Morton ON. Factorized score, nonlinear analysis/synthesis hidden width 32.
Task 번호는 transform grid 기준이다.

| Task | Rank | λ | Residual | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KiB ↓ |
|---:|---:|---:|---|---:|---:|---:|---:|
| 8 | 56 | .0064 | ON | 24.5082 | 0.7665 | 0.2343 | 98.18 |
| 9 | 56 | .0064 | OFF | 24.2580 | 0.7587 | 0.2440 | 105.79 |
| 10 | 56 | .0256 | ON | 23.6943 | 0.7381 | 0.2673 | 46.69 |
| 11 | 56 | .0256 | OFF | 23.5011 | 0.7303 | 0.2738 | 55.34 |
| 12 | 80 | .0064 | ON | 24.4890 | 0.7659 | 0.2367 | 99.15 |
| 13 | 80 | .0064 | OFF | 24.2579 | 0.7584 | 0.2438 | 109.91 |
| 14 | 80 | .0256 | ON | 23.5833 | 0.7378 | 0.2699 | 47.31 |
| 15 | 80 | .0256 | OFF | 23.4970 | 0.7301 | 0.2728 | 55.41 |

Task 15도 완료되어 이제 8개 모두 결과가 있다. Rank56 residual-ON 기준으로
Nonlinear32는 Linear의 PSNR 24.5021→24.5082, 23.6659→23.6943을 기록했다.
두 transform의 초기 크기 비교는 앞서 설명한 초기 표 단위 확인을 전제로 한다.
Nonlinear에서도 Residual ON은 더 작은 payload와 더 높은 PSNR을 보인다.

## 3. Linear factorized continuation — 50k+50k

학습 420267_[0-3], eval 420718_[0-3], RUN=20260908_184640.
모든 학습에 max_steps=50000 reached가 있고, 네 eval 모두 최종 집계가 있다.
Residual ON, Morton ON, context OFF.

| Task | Rank | λ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KiB ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 56 | .0064 | 24.3024 | 0.7602 | 0.2374 | 90.80 |
| 1 | 56 | .0256 | 23.4875 | 0.7307 | 0.2729 | 42.26 |
| 2 | 80 | .0064 | 24.2805 | 0.7597 | 0.2399 | 90.94 |
| 3 | 80 | .0256 | 23.4740 | 0.7310 | 0.2758 | 42.64 |

Rank56은 두 λ에서 Rank80보다 PSNR/LPIPS/크기가 좋다.
단 λ=.0256의 SSIM은 Rank80 .7310 > Rank56 .7307로 아주 조금 높다.
따라서 Rank56의 모든 지표 우세라고 표현하지 않는다.

초기 50k와 비교하면 PSNR이 내려간 반면 크기도 작아졌다.
추가 학습이 더 낮은 rate의 operating point로 이동시킬 수 있음을 보여주므로,
context 성능을 초기 50k와만 비교해서 해석하면 안 된다.

## 4. Linear score-context — 50k+50k

학습 419615_[0-7], eval 419616_[0-7], RUN=20260908_092102.
Rank56, Residual ON, Morton ON. Mean은 전송된 scene mean 사용,
Channel은 먼저 복원한 channel slice 사용, Spatial은 Morton even/odd 문맥 사용.
Full은 Mean+Channel+Spatial이다.

| Task | λ | Context | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KiB ↓ |
|---:|---:|---|---:|---:|---:|---:|
| 0 | .0064 | Mean | 24.3179 | 0.7610 | 0.2371 | 89.82 |
| 2 | .0064 | Mean+Channel | 24.3151 | 0.7608 | 0.2371 | 88.25 |
| 4 | .0064 | Mean+Spatial | 24.3222 | 0.7611 | 0.2369 | 86.07 |
| 6 | .0064 | Full | 24.3232 | 0.7611 | 0.2368 | 84.91 |
| 1 | .0256 | Mean | 23.4950 | 0.7309 | 0.2723 | 41.61 |
| 3 | .0256 | Mean+Channel | 23.4943 | 0.7312 | 0.2727 | 41.49 |
| 5 | .0256 | Mean+Spatial | 23.5304 | 0.7319 | 0.2716 | 39.17 |
| 7 | .0256 | Full | 23.5352 | 0.7323 | 0.2716 | 39.23 |

대조군은 Linear continuation Rank56 task 0/1이다.
아래의 크기 변화는 반올림 전 actual_bytes로 계산했다.

| λ | Context | 크기 변화 | ΔPSNR | ΔSSIM | ΔLPIPS |
|---:|---|---:|---:|---:|---:|
| .0064 | Mean | -1.08% | 0.0155 | 0.0008 | -0.0003 |
| .0064 | Mean+Channel | -2.81% | 0.0127 | 0.0006 | -0.0003 |
| .0064 | Mean+Spatial | -5.21% | 0.0198 | 0.0009 | -0.0005 |
| .0064 | Full | -6.50% | 0.0208 | 0.0009 | -0.0006 |
| .0256 | Mean | -1.54% | 0.0075 | 0.0002 | -0.0006 |
| .0256 | Mean+Channel | -1.81% | 0.0068 | 0.0005 | -0.0002 |
| .0256 | Mean+Spatial | -7.30% | 0.0429 | 0.0012 | -0.0013 |
| .0256 | Full | -7.16% | 0.0477 | 0.0016 | -0.0013 |

같은 추가 학습량을 맞춘 이 실행들에서는 context가 더 작은 payload와 더 좋은
평균 화질을 보였다. Full은 .0064에서 6.50% 절감 / +.0208 dB,
.0256에서 7.16% 절감 / +.0477 dB이다. 추가 학습만으로 설명되는 결과는 아니다.

Spatial의 기여가 크다. Full 대비 Spatial은 .0064에서 약 1.16 KiB 더 크지만,
.0256에서는 Full보다 약 .06 KiB 작고 PSNR이 .0048 dB 낮다.
.0256의 두 설정은 tradeoff 관계이며 유일한 RD winner를 확정하지 않는다.

대표 조건의 전송량 구성:

| λ | 구조 | Score KiB | Residual y+z KiB | Mean+container KiB | 합계 KiB |
|---:|---|---:|---:|---:|---:|
| .0064 | Factorized | 76.01 | 13.23 | 1.57 | 90.80 |
| .0064 | Spatial | 71.71 | 12.80 | 1.57 | 86.07 |
| .0064 | Full | 70.47 | 12.87 | 1.57 | 84.91 |
| .0256 | Factorized | 34.33 | 6.36 | 1.57 | 42.26 |
| .0256 | Spatial | 31.42 | 6.19 | 1.57 | 39.17 |
| .0256 | Full | 31.41 | 6.25 | 1.57 | 39.23 |

각 항목 반올림 때문에 표시된 열의 합이 .01 KiB 정도 다를 수 있다.
크기 개선은 주로 score stream에서 나온다.

## 5. Nonlinear factorized continuation — 50k+50k

Rank56: 학습 420303_[0-1], eval 420304_[0-1], RUN=20260908_195856.
Rank80: 학습 420382_[10-11], eval 420383_[10-11], RUN=20260908_221227.
모두 Residual ON, Morton ON, context OFF.

| Task | Rank | λ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KiB ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 56 | .0064 | 24.3022 | 0.7602 | 0.2376 | 87.84 |
| 10 | 80 | .0064 | 24.2920 | 0.7601 | 0.2395 | 88.17 |
| 1 | 56 | .0256 | 23.5210 | 0.7328 | 0.2693 | 41.26 |
| 11 | 80 | .0256 | 23.4264 | 0.7319 | 0.2756 | 42.31 |

Rank56에서 같은 학습량의 Linear continuation 대비:

| λ | 크기 변화 | ΔPSNR | ΔSSIM | ΔLPIPS |
|---:|---:|---:|---:|---:|
| .0064 | -3.26% | -0.0002 | 0.0000 | 0.0002 |
| .0256 | -2.37% | 0.0335 | 0.0021 | -0.0036 |

.0064에서는 크기를 줄이고 PSNR/SSIM은 거의 같지만 LPIPS는 .0002 높다.
.0256에서는 nonlinear가 네 지표 모두 개선했다.
이는 nonlinear 자체의 유효성을 지지하지만, context와 결합한 성능을 보장하지는 않는다.

**17:28 업데이트: Rank80 추가 eval 2개 완료.** 두 로그 모두 nonlinear/rank80/
context-OFF인 global_step=50000 checkpoint를 검사하고, 6,991개 장면의 최종 집계를
출력했다. 오류 로그에는 일반적인 라이브러리 경고만 있다. 복사된 Rank80 학습 로그는
여전히 종료 전 지점에서 끊겨 있지만, 새 eval이 최종 50k checkpoint를 실제로 읽은 것이 확인된다.

같은 nonlinear continuation에서 Rank80 - Rank56 변화:

| λ | ΔKiB | 크기 변화 | ΔPSNR | ΔSSIM | ΔLPIPS |
|---:|---:|---:|---:|---:|---:|
| .0064 | +0.33 | +0.37% | -0.0102 | -0.0001 | +0.0019 |
| .0256 | +1.05 | +2.54% | -0.0946 | -0.0009 | +0.0063 |

현재 두 λ의 관측 결과에서 Rank56이 크기와 세 화질 지표 모두 우세하다.
따라서 nonlinear 표현력과 추가 50k 학습을 제공해도 Rank80의 이점은 나타나지 않았다.
모든 설정에서 Rank80이 불리하다는 일반화는 하지 않는다.

Rank80의 score는 Rank56보다 .0064에서 3.05 KiB, .0256에서 2.90 KiB 더 크다.
반면 residual y+z 절감은 각각 2.72 KiB, 1.85 KiB여서 score 증가를 메우지 못한다.
이는 현재 학습 결과의 bit allocation 관측이며 단독으로 화질 저하의 원인을 증명하지는 않는다.

참고로 Rank80 자체의 Linear→Nonlinear 비교에서는 .0064가 -3.05% / +.0115 dB,
.0256이 -.79% / -.0476 dB다. Nonlinear가 모든 rank/λ에서 모든 지표를 개선한 것은 아니다.

## 6. 복호 시간 — 기존 로그에서 관측한 값

entropy_decode는 decompress_scene 전체(복원 포함)의 평균이며 순수 entropy 연산만의 시간이 아니다.
아래는 기존 서로 다른 job의 참고 수치이며 통제된 지연시간 benchmark 결과가 아니다.

| λ | Linear factorized | Linear spatial | Linear full | Nonlinear factorized (Rank56) |
|---:|---:|---:|---:|---:|
| .0064 | 83.6 ms | 86.4 ms | 88.5 ms | 83.4 ms |
| .0256 | 80.7 ms | 82.2 ms | 84.3 ms | 74.4 ms |

로그상 Full은 Spatial보다 약 2.1 ms 더 걸렸고, 복호 단계 수 증가가 전체 시간의
4배 증가로 이어지지는 않았다. GPU/부하를 맞춘 재측정 전에는 정확한 속도 비율을 주장하지 않는다.
새 Nonlinear Rank80의 entropy_decode 평균은 .0064에서 94.7 ms, .0256에서 86.5 ms다.
각각 Rank56의 83.4 ms, 74.4 ms보다 높게 측정됐으며, 마찬가지로 서로 다른 job의 참고값이다.

## 7. 학습 중으로 기록된 후속 실험

| 학습 job/task | Rank | λ | Transform/prior | 받은 로그 기준 |
|---|---:|---:|---|---|
| 420916_6 | 56 | .0064 | Nonlinear+Mean+Spatial | Epoch1 약 53%, 종료 표식 없음 |
| 420916_7 | 56 | .0256 | Nonlinear+Mean+Spatial | Epoch1 약 55%, 종료 표식 없음 |
| 420916_8 | 56 | .0064 | Nonlinear+Full | Epoch1 약 47%, 종료 표식 없음 |
| 420916_9 | 56 | .0256 | Nonlinear+Full | Epoch1 약 48%, 종료 표식 없음 |

Epoch 진행률은 전체 50k optimizer-step 진행률이 아니다.
Nonlinear context RUN=20260910_092000. 기존에 설계한 task 6-9는 이미 제출되어
학습 로그가 있으므로 같은 네 개를 다시 제출할 필요가 없다.
현재 받은 이 네 task의 로그에서는 종료 표식/최종 eval 결과가 없고 치명적 에러도 확인되지 않았다.
서버에서 지금도 실행 중인지는 별도 실시간 queue 확인이 필요하다.

420303_10/11의 task id 0 through 9 오류는 과거 실패한 제출이다.
성공한 replacement 420382_10/11의 결과는 위 5절에 포함했고, 과거 실패한 제출은
별도 유효한 실험 결과로 세지 않는다.

## 현재 판단과 남은 검증

1. Rank56 + Residual ON이 현재의 실용적인 출발점이다. 새 Nonlinear continuation에서도
   Rank56이 Rank80보다 두 λ의 크기와 세 화질 지표 모두 우세했다.
2. Nonlinear32와 score-context 각각의 유효성은 matched continuation 결과로 지지된다.
3. Linear context에서는 Spatial이 가장 큰 추가 개선을 만들며, Full의 추가 이득은
   .0064에서 더 뚜렷하다.
4. 현재 모든 rate 영역에서 하나의 최종 모델이 이겼다는 결론은 아니다.
   50k Nonlinear Rank56은 98.18 KiB / 24.5082 dB이고,
   50k+50k Linear Full은 84.91 KiB / 24.3232 dB로 서로 다른 operating point다.
5. .0256에서 Nonlinear factorized의 .7328 SSIM / .2693 LPIPS는
   더 작은 Linear Full의 .7323 / .2716보다 좋다. PSNR·rate만으로 모든 목적의 최종 winner를 고르지 않는다.
6. 실행 중인 Nonlinear+Spatial/Full을 같은 nonlinear continuation과 먼저 비교하고,
   Linear context와도 비교해야 결합 이득을 판단할 수 있다.
7. Base-conditioned residual 2x2는 다음 구조 변경 설계이며 아직 구현/결과가 없다.
   구체안은 [다음 실험 설계](NFCGS_NEXT_EXPERIMENT_2026-09-10.md)에 있다.

## 로그 출처

아래 24개 완료 eval에서 값들을 추출했다. 지표의 소수 넷째 자리와
표의 KiB 소수 둘째 자리보다 작은 차이는 원시 JSON/bytes로 재계산한다.

- [slurm-gs-factorized-cont-eval-420718_0.out](../../../slurm/slurm-gs-factorized-cont-eval-420718_0.out)
- [slurm-gs-factorized-cont-eval-420718_1.out](../../../slurm/slurm-gs-factorized-cont-eval-420718_1.out)
- [slurm-gs-factorized-cont-eval-420718_2.out](../../../slurm/slurm-gs-factorized-cont-eval-420718_2.out)
- [slurm-gs-factorized-cont-eval-420718_3.out](../../../slurm/slurm-gs-factorized-cont-eval-420718_3.out)
- [slurm-gs-nfcgs-transform-eval-419630_10.out](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_10.out)
- [slurm-gs-nfcgs-transform-eval-419630_11.out](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_11.out)
- [slurm-gs-nfcgs-transform-eval-419630_12.out](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_12.out)
- [slurm-gs-nfcgs-transform-eval-419630_13.out](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_13.out)
- [slurm-gs-nfcgs-transform-eval-419630_14.out](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_14.out)
- [slurm-gs-nfcgs-transform-eval-419630_8.out](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_8.out)
- [slurm-gs-nfcgs-transform-eval-419630_9.out](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_9.out)
- [slurm-gs-nfcgs-transform-eval-420263_15.out](../../../slurm/slurm-gs-nfcgs-transform-eval-420263_15.out)
- [slurm-gs-nl-score-ctx-eval-420304_0.out](../../../slurm/slurm-gs-nl-score-ctx-eval-420304_0.out)
- [slurm-gs-nl-score-ctx-eval-420304_1.out](../../../slurm/slurm-gs-nl-score-ctx-eval-420304_1.out)
- [slurm-gs-nl-score-ctx-eval-420383_10.out](../../../slurm/slurm-gs-nl-score-ctx-eval-420383_10.out)
- [slurm-gs-nl-score-ctx-eval-420383_11.out](../../../slurm/slurm-gs-nl-score-ctx-eval-420383_11.out)
- [slurm-gs-score-ctx-eval-419616_0.out](../../../slurm/slurm-gs-score-ctx-eval-419616_0.out)
- [slurm-gs-score-ctx-eval-419616_1.out](../../../slurm/slurm-gs-score-ctx-eval-419616_1.out)
- [slurm-gs-score-ctx-eval-419616_2.out](../../../slurm/slurm-gs-score-ctx-eval-419616_2.out)
- [slurm-gs-score-ctx-eval-419616_3.out](../../../slurm/slurm-gs-score-ctx-eval-419616_3.out)
- [slurm-gs-score-ctx-eval-419616_4.out](../../../slurm/slurm-gs-score-ctx-eval-419616_4.out)
- [slurm-gs-score-ctx-eval-419616_5.out](../../../slurm/slurm-gs-score-ctx-eval-419616_5.out)
- [slurm-gs-score-ctx-eval-419616_6.out](../../../slurm/slurm-gs-score-ctx-eval-419616_6.out)
- [slurm-gs-score-ctx-eval-419616_7.out](../../../slurm/slurm-gs-score-ctx-eval-419616_7.out)

# Nonlinear32 + score-context 평가 결과 — 2026-09-11

로컬 로그 복사 시각 약 19:33 KST 기준. 서버의 실시간 queue 조회 결과는 아니다.
이 문서는 [2026-09-10 전체 실험 보고서](NFCGS_EXPERIMENT_RESULTS_2026-09-10.md)의
후속 업데이트다. 해당 보고서의 학습 중이던 네 조건이 이번에 모두 완료되었다.

**현재 결과가 있는 codec 설정은 40개**: 초기 사용자 제공 Linear 12개 +
로컬 최종 eval 로그 28개. 공식 codec-OFF reference는 별도다.
추적하던 Nonlinear context 네 조건 중 미완료 eval은 없다.

## 완료 확인

| Task | λ | Context | 학습 job | Eval job | 확인 |
|---:|---:|---|---|---|---|
| 6 | .0064 | Mean+Spatial | 420916_6 | 420919_6 | 추가 50k 도달, 최종 checkpoint 평가 완료 |
| 7 | .0256 | Mean+Spatial | 420916_7 | 420919_7 | 추가 50k 도달, 최종 checkpoint 평가 완료 |
| 8 | .0064 | Full | 420916_8 | 420919_8 | 추가 50k 도달, 최종 checkpoint 평가 완료 |
| 9 | .0256 | Full | 420916_9 | 420919_9 | 추가 50k 도달, 최종 checkpoint 평가 완료 |

- 네 학습 stderr 모두 Trainer.fit stopped: max_steps=50000 reached를 기록했다.
- 네 eval은 global_step=50000인 nonlinear/rank56/context 설정을 검사하고 정상적으로 최종 집계를 출력했다.
- 모두 6,991/7,286에서 평가가 종료되었다. 7,286은 원본 index 수이고 실제 처리 장면은 6,991개다.
- 새 eval stderr에서 traceback, OOM, 취소/시간 초과 등 치명적 오류는 발견되지 않았다.
- RUN=20260910_092000, Rank56, nonlinear hidden32, Residual ON, Morton ON,
  slice width16, context hidden64, paper24 subset consistency, micro-batch2 × accumulation4 = effective batch8.
- 해당 λ의 원래 nonlinear32 50k checkpoint에서 weight를 복사하고 추가 50k를 학습했다.
  원래 50k + 추가 50k이며 optimizer/scheduler는 continuation 시작 때 초기화한다.
- 평가는 기존 all-test C12/T8, bf16-mixed, 실제 bitstream encode/decode 경로다.
  크기는 scene당 actual_bytes 평균 / 1024 (KiB), 고정 mean/container까지 포함한다.

## 전체 직접 비교 — Rank56, 50k+50k

| λ | Transform | Prior | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KiB ↓ |
|---:|---|---|---:|---:|---:|---:|
| .0064 | Linear | Factorized control | 24.3024 | 0.7602 | 0.2374 | 90.80 |
| .0064 | Nonlinear32 | Factorized control | 24.3022 | 0.7602 | 0.2376 | 87.84 |
| .0064 | Linear | Mean+Spatial | 24.3222 | 0.7611 | 0.2369 | 86.07 |
| .0064 | Linear | Full | 24.3232 | 0.7611 | 0.2368 | 84.91 |
| .0064 | Nonlinear32 | Mean+Spatial | 24.3244 | 0.7612 | 0.2371 | 83.63 |
| .0064 | Nonlinear32 | Full | 24.3238 | 0.7611 | 0.2371 | 82.49 |
| .0256 | Linear | Factorized control | 23.4875 | 0.7307 | 0.2729 | 42.26 |
| .0256 | Nonlinear32 | Factorized control | 23.5210 | 0.7328 | 0.2693 | 41.26 |
| .0256 | Linear | Mean+Spatial | 23.5304 | 0.7319 | 0.2716 | 39.17 |
| .0256 | Linear | Full | 23.5352 | 0.7323 | 0.2716 | 39.23 |
| .0256 | Nonlinear32 | Mean+Spatial | 23.5867 | 0.7348 | 0.2678 | 38.24 |
| .0256 | Nonlinear32 | Full | 23.5707 | 0.7342 | 0.2683 | 37.96 |

## 같은 nonlinear 대조군 대비 context 효과

대조군은 420304_0(.0064), 420304_1(.0256)이다.
같은 원래 nonlinear checkpoint와 추가 학습량을 사용한 비교다.

| 새 Task | λ | Context | 크기 변화 | ΔPSNR | ΔSSIM | ΔLPIPS |
|---:|---:|---|---:|---:|---:|---:|
| 6 | .0064 | Mean+Spatial | -4.79% | 0.0222 | 0.0010 | -0.0005 |
| 8 | .0064 | Full | -6.09% | 0.0216 | 0.0009 | -0.0005 |
| 7 | .0256 | Mean+Spatial | -7.32% | 0.0657 | 0.0020 | -0.0015 |
| 9 | .0256 | Full | -7.99% | 0.0497 | 0.0014 | -0.0010 |

네 조건 모두 대조군보다 실제 bytes가 적고 PSNR/SSIM/LPIPS가 개선됐다.
따라서 이번 실행에서는 nonlinear transform을 사용한 뒤에도 context의 이득이 남는다.
이는 이득이 정확히 독립적이거나 개선율이 단순히 더해진다는 주장은 아니다.

## 같은 context에서 Linear → Nonlinear 효과

비교는 새 task6 ↔ Linear task4, 새 task7 ↔ Linear task5,
새 task8 ↔ Linear task6, 새 task9 ↔ Linear task7이다.

| λ | Context | 크기 변화 | ΔPSNR | ΔSSIM | ΔLPIPS |
|---:|---|---:|---:|---:|---:|
| .0064 | Mean+Spatial | -2.83% | 0.0022 | 0.0001 | 0.0002 |
| .0064 | Full | -2.84% | 0.0006 | 0.0000 | 0.0003 |
| .0256 | Mean+Spatial | -2.39% | 0.0563 | 0.0029 | -0.0038 |
| .0256 | Full | -3.25% | 0.0355 | 0.0019 | -0.0033 |

- .0064는 같은 context의 Linear 대비 약 2.84% 작아졌다. PSNR은 거의 같고
  LPIPS는 .0002~.0003 높으므로 모든 화질 지표 우세라고 표현하지 않는다.
- .0256는 Spatial/Full 모두 같은 context의 Linear보다 크기와 세 화질 지표를 개선했다.
  Nonlinear와 context를 결합할 근거가 특히 뚜렷한 영역이다.

참고로 전체 구조 변경을 Linear factorized continuation과 비교하면
Nonlinear Full .0064는 -9.16% / +.0214 dB,
Nonlinear Spatial .0256는 -9.52% / +.0992 dB,
Nonlinear Full .0256는 -10.18% / +.0832 dB다.

## Full 대 Spatial — 하나의 winner로 단정하지 않기

| λ | Full - Spatial KiB | Full 크기 변화 | ΔPSNR | ΔSSIM | ΔLPIPS |
|---:|---:|---:|---:|---:|---:|
| .0064 | -1.14 | -1.37% | -0.0006 | -0.0001 | 0.0000 |
| .0256 | -0.28 | -0.73% | -0.0160 | -0.0006 | 0.0005 |

.0064에서 Full은 Spatial보다 1.14 KiB 작으며 PSNR 차이는 -.0006 dB에 불과하다.
따라서 rate를 중요하게 보는 후속 실험의 출발점으로 Full(task8)을 추천한다.
다만 Spatial의 약간 높은 PSNR/SSIM, 짧은 복호 시간도 함께 기록한다.

.0256에서는 Full이 .28 KiB 작지만 Spatial이 PSNR +.0160 dB,
SSIM +.0006, LPIPS -.0005로 화질이 더 좋다.
화질을 중시하는 다음 predictor 개선의 출발점으로 Spatial(task7)을 추천하되,
최소 payload 후보 Full(task9)도 Pareto 후보로 보존한다.
이 두 점만으로 동일 bitrate의 최종 우열을 결정하지 않는다.

## 실제 stream 크기와 수신측 시간

| Task | λ | Context | Score KiB | Residual y+z KiB | Mean+container KiB | 전체 KiB | 수신측 평균 ms |
|---:|---:|---|---:|---:|---:|---:|---:|
| 6 | .0064 | Mean+Spatial | 70.39 | 11.68 | 1.57 | 83.63 | 85.3 |
| 8 | .0064 | Full | 69.29 | 11.63 | 1.57 | 82.49 | 89.2 |
| 7 | .0256 | Mean+Spatial | 30.63 | 6.04 | 1.57 | 38.24 | 83.6 |
| 9 | .0256 | Full | 30.43 | 5.97 | 1.57 | 37.96 | 84.3 |

score 비중은 .0064에서 약84%, .0256에서 약80%다.
대조군 대비 개선은 대부분 score에서 나오며, residual bytes도 소폭 감소했다.
표의 항목별 반올림 때문에 합계는 .01 KiB 수준에서 차이 날 수 있다.

entropy_decode는 decompress_scene 전체를 재며 순수 entropy 연산만의 시간은 아니다.
로그 평균으로 Full은 Spatial보다 .0064에서 3.9 ms, .0256에서 .7 ms 더 걸렸다.
job 간 GPU/동시 부하가 완전히 통제된 측정은 아니므로 정밀한 속도 우위는 재측정해야 한다.

## 판단 범위와 다음 단계

- 이번에는 집계 로그와 학습 설정을 확인했다. 서버의 actual_rate_per_scene.json은
  현재 로컬 outputs에 없어 scene/frame ID의 완전 일치 및 paired bootstrap은 수행하지 못했다.
- 미세한 평균 차이를 seed 변동을 넘어선 통계적 우위로 단정하지 않는다.
- 현재 context는 prediction과 scene별 quantization step까지 변경하며,
  결과는 joint-trained codec 전체의 효과다.
- 여기서의 비율은 같은 λ에서 실제 크기의 변화이며 동일 화질 절감률이나 BD-rate가 아니다.
- 초기 50k 고화질 모델과 50k+50k 저율 모델을 하나의 압도적 winner로 합치지 않는다.
- 기존에 기다리던 네 결과는 모두 확보됐다. 다음 공간 predictor 실험은
  P0 기존 / P1 비선형 kernel3 보정 / P2 비선형 kernel7 보정으로 설계할 수 있다.
  각 λ에서 같은 선택 checkpoint로 시작하고 P0까지 동일한 추가25k를 학습한다.
  비선형 보정의 마지막 층을 0 초기화해 출발 출력을 보존하고 두 단계 spatial 복호는 유지한다.
  Full을 부모로 쓰면 기존 channel slice 순서도 유지한다.
- 이번 작업은 결과 분석 및 문서 업데이트이며 모델 수정, 재학습, Slurm 제출은 하지 않았다.

## 원본 로그

- [학습 종료 6](../../../slurm/slurm-gs-nl-score-ctx-420916_6.err)
- [Eval 결과 6](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_6.out)
- [Eval stderr 6](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_6.err)
- [학습 종료 7](../../../slurm/slurm-gs-nl-score-ctx-420916_7.err)
- [Eval 결과 7](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_7.out)
- [Eval stderr 7](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_7.err)
- [학습 종료 8](../../../slurm/slurm-gs-nl-score-ctx-420916_8.err)
- [Eval 결과 8](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_8.out)
- [Eval stderr 8](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_8.err)
- [학습 종료 9](../../../slurm/slurm-gs-nl-score-ctx-420916_9.err)
- [Eval 결과 9](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_9.out)
- [Eval stderr 9](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_9.err)

이전 대조군과 Linear context 원본 로그는 [전체 보고서의 출처](NFCGS_EXPERIMENT_RESULTS_2026-09-10.md#로그-출처)에 있다.


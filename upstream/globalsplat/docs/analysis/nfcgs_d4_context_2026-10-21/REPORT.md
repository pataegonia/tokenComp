# H+D4 ON16/OFF16 완료 결과 — 2026-10-21

같은 λ=0.0128에서 spatial context ON은 두 seed 모두 OFF보다 실제 전송량이 작고 PSNR이 높았다. 기본 seed에서는 16.6754%·0.0683 dB, 추가 seed에서는 17.1857%·0.0726 dB의 차이를 확인했다.

## 비교 조건

- H는 Hilbert ordering, D4는 4개 entropy stage이며 channel group은 16/16/16/8이다. OFF16은 spatial prediction을 끄되 LR/MSH 및 stage/group stream 구성을 유지한다.
- 기존 공통 warm-start parent에서 weights-only로 시작해 50k를 추가 학습한 실험이다. 기본 seed는 model 111123/data 403, 추가 seed는 model 111124/data 404다.
- 추가 seed도 같은 parent를 공유하므로 독립 fresh-init 반복으로 해석하지 않는다. 새 fresh 실험 결과와 분리한다.
- RE10K full test의 실제 처리 수는 각 6,991 scene이다. C12/T8, 256×256, BF16, evaluation seed 0 조건이다. progress의 nominal 분모 7,286과 구분한다.
- 용량은 container를 포함한 scene당 actual bytes이며, KiB는 bytes/1024다.

## 완료 지표

| Seed | Context | Eval job | PSNR (dB) | SSIM | LPIPS | Bytes/scene | KiB/scene |
|---|---|---:|---:|---:|---:|---:|---:|
| 기본 111123/403 | ON16 | 447899 | 23.7021 | 0.7363 | 0.2673 | 38071.103419 | 37.178812 |
| 기본 111123/403 | OFF16 | 457100 | 23.6338 | 0.7338 | 0.2695 | 45690.114147 | 44.619252 |
| 추가 111124/404 | ON16 | 457787 | 23.7036 | 0.7362 | 0.2669 | 37964.902303 | 37.075100 |
| 추가 111124/404 | OFF16 | 457789 | 23.6310 | 0.7336 | 0.2692 | 45843.390073 | 44.768936 |

차이는 각 seed 안에서 계산했다. 전송량 감소율은 `(OFF bytes - ON bytes) / OFF bytes × 100`이다.

| Seed | ON 전송량 감소율 | PSNR ON−OFF | SSIM ON−OFF | LPIPS ON−OFF |
|---|---:|---:|---:|---:|
| 기본 | 16.6754% | +0.0683 dB | +0.0025 | −0.0022 |
| 추가 | 17.1857% | +0.0726 dB | +0.0026 | −0.0023 |

이 두 seed와 해당 λ에서는 context ON의 이점이 일관된다. 같은 λ의 개별 점 비교이며, 동일 품질에서의 BD-rate 또는 전체 RD 곡선의 우위를 계산한 결과는 아니다.

## 완료 확인 및 원본

기본 OFF16은 원 학습 457099 이후 full-state 복구 학습 458783에서 `max_steps=50000` 도달을 확인했다. 평가 457100은 최종 지표를 출력했고 치명적 평가 오류가 없다. 네 평가 stdout 모두 마지막 처리 수 `6991/7286`과 최종 품질·actual byte 지표를 다시 확인했다. 시간 통계는 warmup 5개를 제외한 6,986 scene이다.

2026-10-21 로컬에 다운로드된 원본 stdout을 재확인했다. 아래 경로는 workspace 루트를 기준으로 한다. 서버 실시간 조회나 작업 제출·취소는 수행하지 않았다.

| Eval job | 원본 stdout | SHA256 |
|---|---|---|
| 447899 | `slurm/slurm-gs-order-context-eval-447899.out` | `b6153a177a88ffeb7de844aef6e5dae8d2c8f815047610f67f4c8da3ee04188f` |
| 457100 | `slurm/slurm-gs-order-context-eval-457100.out` | `4049913e79f14b87891b0ecf06fcb3b10c3ba05d854a154b6740f91a0a60e799` |
| 457787 | `slurm/slurm-gs-seed-pair-eval-457787.out` | `1cbc7c160c17273cbb2831c1089d18075a98d7f7ec83dce39990f9f4cfd3eaa9` |
| 457789 | `slurm/slurm-gs-seed-pair-eval-457789.out` | `8c464d9104f8025c695b6a671f322aa2d3d44d7639c9261c4b0fbd6971cb09f9` |

기존 [기본 OFF 완료 기록](../nfcgs_logs_2026-10-21_fresh_unblocked/REPORT.md)과 [추가 seed inventory](../nfcgs_logs_2026-10-20_2159/inventory.json)를 함께 참조한다. 이 문서는 완료한 두 seed의 ON/OFF 비교를 모은 별도 기록이며, 이전 inventory snapshot을 수정하지 않는다.

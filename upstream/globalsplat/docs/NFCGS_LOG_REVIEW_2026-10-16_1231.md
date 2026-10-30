# 정렬·context ablation 로그 점검 — 2026-10-16 12:31

workspace `slurm/`에 12:30~12:31에 내려받은 stdout/stderr와 새 recovery JSON을 확인했다. 아래는 다운로드된 기록이며 실시간 Slurm 조회 결과가 아니다. 현재 state의 15개 point 중 13개가 학습 50k와 전체 scene eval을 완료했고, 2개는 학습 진행 기록이 있다. 이전 ENOSPC로 중단한 451821/451825/451827 대신 재제출한 453064/453066/453068을 현재 작업으로 사용했다.

## 재제출 및 오류 확인

재제출한 3개 로그 모두 `CHECKPOINT_TMPDIR=/ceph_data/clue9986/.tmp/nfcgs/job_<ID>`가 확인된다. Quarter2 .0064의 453064는 `RESUME_FROM_STEP=5000`과 `Restored all states from the checkpoint`가 있고 50k에 도달했다. Eval 453065도 새 실행 폴더의 step50k checkpoint를 읽어 최종 수치를 출력했으므로, 이 실행의 최종 checkpoint 저장·로드 성공을 확인했다.

453066/453068은 각각 약 33.1k/30.4k까지 진행됐다. 이 두 학습의 checkpoint 파일 자체는 로컬에 없지만, 주기적 저장 step을 넘어서 학습이 계속된 기록이 있다. 현재 15개 train/eval의 로그에서 새 traceback, CUDA OOM, ENOSPC, 취소 오류를 발견하지 못했다. 이전 3개 작업에 남은 오류와 현재 작업을 구분했다. `/tmp`가 실패 당시 실제로 가득 찼다는 직접 용량 측정은 받지 않았으므로, 수정 후 정상 진행과 과거 원인 확정을 구분한다.

453064에는 non-resumable DataLoader 경고가 남아 있다. 모델·optimizer·scheduler의 전체 상태 복원과 원래 DataLoader 입력 순서의 동일한 재현은 구분해야 한다.

## 현재 RD 결과

각 셀은 PSNR dB / 실제 bitstream KiB/scene이다. 완료한 13개 eval 모두 50k checkpoint, C12/T8, batch1, BF16, seed0, actual bitstream ON이다. 실제 평가는 6,991 scene이고 시간 평균은 warmup 5개를 뺀 6,986 scene이다. 진행 표시의 분모 7,286을 평가 scene 수로 사용하지 않았다.

| 설정 | λ=.0064 | λ=.0128 | λ=.0256 |
|---|---|---|---|
| NN-score + legacy3 | 23.8274 / 45.012 | 23.7431 / 36.951 | 23.5447 / 29.307 |
| Context-only legacy3 | 23.6369 / 55.882 | 23.5665 / 46.525 | 23.3521 / 38.264 |
| Hilbert-only (Score spatial context OFF) | 23.8130 / 55.110 | 23.6831 / 44.807 | 23.4072 / 35.783 |
| Context-only quarter2 | 23.6703 / 55.678 | 23.5874 / 46.800 | 학습 중 |
| Context-only dyadic4 | 학습 중 | 23.5838 / 46.737 | 23.3711 / 38.307 |
| 기존 Hilbert + legacy3 | 23.7994 / 46.894 | 23.7032 / 38.269 | 23.5050 / 30.290 |

Hilbert-only의 OFF는 Score spatial context OFF를 뜻한다. Score mean/channel context도 OFF이고, MSH residual codec과 legacy3의 entropy split은 유지한다. Quarter2/dyadic4와 entropy model 수까지 동일한 대조군은 아니다.

## 이번에 새로 완료된 eval 5개

| 설정 | λ | Train / eval | PSNR | SSIM | LPIPS ↓ | KiB/scene ↓ | Codec encode / decode ms |
|---|---:|---|---:|---:|---:|---:|---:|
| Hilbert-only (Score spatial context OFF) | 0.0256 | 451819 / 451820 | 23.4072 | 0.7264 | 0.2793 | 35.783 | 160.4 / 87.9 |
| Context-only quarter2 | 0.0064 | 453064 / 453065 | 23.6703 | 0.7358 | 0.2718 | 55.678 | 144.0 / 84.9 |
| Context-only quarter2 | 0.0128 | 451823 / 451824 | 23.5874 | 0.7327 | 0.2746 | 46.800 | 149.5 / 88.1 |
| Context-only dyadic4 | 0.0128 | 451868 / 451869 | 23.5838 | 0.7326 | 0.2748 | 46.737 | 155.8 / 92.1 |
| Context-only dyadic4 | 0.0256 | 451870 / 451871 | 23.3711 | 0.7255 | 0.2816 | 38.307 | 154.3 / 91.2 |

## 같은 λ에서의 비교

각 단독 설정을 동일 schedule의 기존 Hilbert+context 결과와 비교했다. 아래 양의 용량 변화는 단독 설정이 더 크다는 뜻이다. BD-rate나 동일 PSNR로 보간한 값이 아니며, seed 반복의 통계적 유의성도 검증하지 않았다.

| 단독 설정 | λ | 비교 기준 | 용량 변화 | PSNR 변화 dB |
|---|---:|---|---:|---:|
| Context-only legacy3 | 0.0064 | hilbert_legacy3 | +19.17% | -0.1625 |
| Context-only legacy3 | 0.0128 | hilbert_legacy3 | +21.57% | -0.1367 |
| Context-only legacy3 | 0.0256 | hilbert_legacy3 | +26.33% | -0.1529 |
| Hilbert-only (Score spatial context OFF) | 0.0064 | hilbert_legacy3 | +17.52% | +0.0136 |
| Hilbert-only (Score spatial context OFF) | 0.0128 | hilbert_legacy3 | +17.09% | -0.0201 |
| Hilbert-only (Score spatial context OFF) | 0.0256 | hilbert_legacy3 | +18.14% | -0.0978 |
| Context-only quarter2 | 0.0064 | hilbert_quarter2 | +15.35% | -0.1465 |
| Context-only quarter2 | 0.0128 | hilbert_quarter2 | +20.56% | -0.1158 |
| Context-only dyadic4 | 0.0128 | hilbert_dyadic4 | +25.71% | -0.1183 |
| Context-only dyadic4 | 0.0256 | hilbert_dyadic4 | +30.30% | -0.1462 |

정렬 없는 context-only는 현재 완료한 모든 point에서 동일 schedule의 Hilbert+context보다 bitstream이 더 크고 PSNR이 낮다. Hilbert-only도 context를 함께 쓴 legacy3보다 세 λ 모두 용량이 크다. 단, .0064의 PSNR은 Hilbert-only가 .0136dB 높고, .0128/.0256은 context를 함께 쓴 쪽이 높다. 이번 측정 범위에서는 sorting과 context를 함께 사용한 결과가 압축 효율에 유리하다. Context-only의 schedule별 최종 순위는 남은 두 point의 평가 후 확인한다.

Context-only quarter2는 legacy3 대비 .0064에서 PSNR +.0334dB/용량 약 -.37%, .0128에서 PSNR +.0209dB/용량 약 +.59%다. Dyadic4도 legacy3 대비 .0128/.0256에서 PSNR이 약 .017~.019dB 높지만 용량은 약 .11~.46% 크다. 같은 품질로 보간하기 전에는 schedule 승자를 단정하지 않는다.

NN-score legacy3의 기존 결과는 변하지 않았다. Hilbert legacy3 대비 같은 λ의 총용량이 약 3.24~4.01% 작고 PSNR은 약 .028~.040dB 높지만, codec encode는 약 258~262ms로 Hilbert legacy3의 약 153~163ms보다 길었다. 서로 다른 job/노드의 벽시계 시간이므로 정렬 알고리즘만의 시간 차로 귀속하지 않는다.

![RD comparison](../../../artifacts/nfcgs_ablation_rd_2026-10-16_1231.png)

선은 측정점을 연결하며 미완료 point를 보간·외삽하지 않았다.

## 남은 학습 2개

12:31 로그의 처리 속도를 유지한다고 가정한 50k 학습 종료 예상이다. Epoch 내 batch, 1epoch 33,017batch, accumulation2를 반영한 근사 step이며, checkpoint/I/O 변동·중단·후속 eval 대기는 제외한다.

| 설정 | Train / eval | 약step /50k | 전체진행률 | batch/s | 남은시간 | 예상완료(KST) |
|---|---|---:|---:|---:|---:|---|
| Context-only quarter2 | 453066 / 453067 | 33,120 | 66.2% | 1.65 | 5.68시간 | 10/16 18:12 |
| Context-only dyadic4 | 453068 / 453069 | 30,418 | 60.8% | 1.57 | 6.93시간 | 10/16 19:26 |

## 검증과 저장 자료

15개 학습의 실험 조건을 대조했다: GlobalSplat 고정, codec scope all/MSH 유지, C24/T12, subset consistency ON, GPU2·batch/GPU2=2·accumulation2·유효batch8, BF16, 목표50k/checkpoint5k. Score minimal의 centering/norm/mean conditioner/channel context OFF, linear/kernel5/STE와 각 arm의 정렬·schedule·단계 수가 일치한다.

13개 완료 eval은 로드한 checkpoint의 codec 설정을 학습 설정과 대조하고, 각 payload(Score/MSH y/MSH z/mean/container)의 합이 총bytes와 일치함을 확인했다. 모델이나 서버 job을 변경하지 않았다.

- [감사 JSON](../../../artifacts/nfcgs_log_review_2026-10-16_1231.json)
- [비교·ETA JSON](../../../artifacts/nfcgs_ablation_comparison_2026-10-16_1231.json)
- 원인 조사 및 재제출 기록: [ENOSPC 후속 확인](NFCGS_CEPH_ENOSPC_DIAGNOSIS_2026-10-15.md)
- 재생성: `python -X utf8 artifacts/review_nfcgs_logs_2026-10-14.py --snapshot 2026-10-16_1231`, 이어서 `python -X utf8 artifacts/report_nfcgs_logs_2026-10-16.py`.

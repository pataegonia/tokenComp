# 로그 검토 — 2026년 10월 20일 16:47 다운로드본

새로 완료된 결과는 추가 seed의 H+D4 ON16 전체 평가457787이다. OFF16 평가457789와 함께 최종50k·전체6991장·C12/T8·실제 bitstream의 같은λ 비교가 완성됐다. 이번 기록은 로컬 다운로드본 기준이며 실시간 Slurm 조회가 아니다.

## 추가 seed ON16/OFF16

Model seed111124, data seed404, λ=.0128이다. 두 학습은 같은 FullSplit 부모 checkpoint에서 시작하는 후속50k이며, 구조 pilot의 fresh initialization·holdout128과 구분한다.

| 설정 | Train / Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---:|---:|---:|---:|
| ON16 | 457786 / 457787 | 23.7036 | 0.7362 | 0.2669 | 37.07510 |
| OFF16 | 457788 / 457789 | 23.6310 | 0.7336 | 0.2692 | 44.76894 |

ON16−OFF16은 PSNR+0.0726dB, SSIM+0.0026, LPIPS−0.0023이다. 평균 packet은7,878.48777B/scene,17.1857% 감소했다. 같은λ의 이 seed 단일 지점에서 ON16은 화질 지표와 실제 용량 양쪽에서 유리하다. 동일 화질 BD-rate 절감률이나 seed 통계적 유의성으로 표현하지 않는다.

약90.01%의 전체 byte 차이가 Score stream 차이다. 두 설정의 Score wrapper가 같은16 strings이므로 wrapper 차이는0이며, Score 총량 차이는 raw Score 차이와 같다. MSH y도793.267915B 감소하고 MSH z는6.033472B 증가했다. 이 분해는 관찰된 stream byte 차이이며 독립적인 경로 인과 기여율은 아니다.

저장 config의 차이는 `score_spatial_context` 하나다. 둘 모두 D4·4단계·slice16·rank56·minimal linear·MSH residual 유지이며 actual bitstream을 쓴다. 학습 명령의 model/data seed, 평가의 saved step50000/config,6991장 처리 및 byte 구성 합계를 검사했다. 동일 평가 protocol과 설정은 확인했지만 per-scene frame/tensor 정합이나 paired CI는 raw JSON이 없어 이번에 계산하지 않았다.

기본 seed111123/data403의 OFF16은 아직 복구 학습 중이다. 따라서 이번 추가 seed의 완성된 쌍과 기본 seed까지 포함한 두 seed 비교의 완료를 구분한다.

## 남은 학습

| 실험 | Job | 마지막 완전한 progress의 추정 step |
|---|---|---:|
| Score736-only | 457828 | 41,294/50k,82.588% |
| None+L3 OFF12 λ=.0512 복구 | 458782 | 44,762/50k,89.524% |
| H+D4 OFF16 λ=.0128 복구 | 458783 | 33,823/50k,67.646% |

세 학습에서 새 치명 오류는 없고, 완료 footer나 연결 평가 시작 로그는 아직 없다. 이 step은 progress 기준이며 저장 checkpoint나 현재 queue 상태를 뜻하지 않는다. 새 학습이나 job 변경은 수행하지 않았다.

최근26개 원래 실험 중23개 학습 완료 상태는 그대로다. 평가 완료는 전체 test20개와 holdout128 세 개로 늘었다. 주 seed RD63점·16계열 및 별도λ0는 유지하고, 추가 seed ON/OFF 두 점은 `seed_repeat_points`에 따로 보관한다.

구조 pilot의 기존 판단도 유지한다. 같은 holdout128·λ=.0128에서 two-path56은 Score56-only보다 PSNR+1.2178dB·bytes13.42% 감소였으며, MSH-only는 저용량 후보로 남는다. 다음 구조 선택에 필요한 미완료 pilot은 Score736-only 하나다.

## 검산과 자료

이번에는 새 ON 평가와 미완료 세 학습을 다시 읽고, 기존 완료 평가 로그65개를 모두 재해시했다. 기존 평가 로그는 이전 SHA와 모두 일치했다. Architecture20개·seed17개·recovery21개 runtime hash도 일치한다. 원래185개 source 중 직접 재검증한 것은78개이며, 나머지 과거 학습 기록은16:36 snapshot에서 이월했다. 실제로 내용이 바뀐 source는 ON 평가 stdout과 세 미완료 학습 stdout 네 개다.

- [seed_pair_comparison.json](seed_pair_comparison.json):평균 지표·용량·stream별 차이.
- [review.json](review.json):상태·설정 검산·재검증 범위·source 해시.
- [inventory.json](inventory.json):주 seed RD와 추가 seed 두 점의 분리 보관.
- [audit_snapshot.py](audit_snapshot.py):증분 검토 재현 코드.

이후 제공된 T06 REPORT의 상세 해석은 [앞선 T06 후속 검토](../nfcgs_logs_2026-10-20_1636/T06_FOLLOWUP.md)에 보관했다.

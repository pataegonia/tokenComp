# 로그 검토 — 2026년 10월 20일 16:36 다운로드본

이후 사용자가 제공한 T06 REPORT의 Morton·random·anchor·잔차 수치 및 해석은 [T06 후속 검토](T06_FOLLOWUP.md)에 보관했다. 아래는 다운로드 로그만을 검토한 시점의 기록이다.

구조 pilot 중 Score56-only와 two-path56이50k 학습 및 holdout128 평가를 추가로 마쳤다. T06 고정 정렬 진단도256장·13개 순서로 완료됐다. 추가 seed OFF16은 전체6991장 평가가 끝났고 ON16은 평가 진행 로그가 있다. 이 문서는 다운로드한 로컬 `slurm/` 파일 기준이며 실시간 Slurm 상태가 아니다.

## 구조 비교: 같은 holdout128, λ=.0128

| 구조 | Train / Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---:|---:|---:|---:|
| Score56-only | 457741 / 457742 | 21.7889 | 0.6653 | 0.3295 | 65.53055 |
| MSH-only | 457743 / 457744 | 20.6696 | 0.6123 | 0.3700 | 14.48434 |
| two-path56 | 457745 / 457746 | 23.0067 | 0.7157 | 0.2796 | 56.73486 |

세 모델 모두 같은 공개 backbone/QR 초기화, 같은 train-derived holdout128, C12/T8, 새50k, 같은 model/data seed 및 λ=.0128의 pilot recipe다. 평가 명령의 holdout manifest·index 경로,128/128 처리, 실제 bitstream 및 최종50k 경로를 확인했다. runtime hash가 같은 pilot runner는 실행 전에 checkpoint step과 codec config를 검사한다. 이번 다운로드에는 raw checkpoint, 평가의 per-scene/frame JSON이 없으므로 로그·실행 경로 검증과 tensor-level paired 검증을 구분한다.

이 단일 지점에서 two-path56은 Score56-only보다 PSNR+1.2178dB, SSIM+0.0504, LPIPS−0.0499이며 bytes13.4223% 작다. 따라서 Score56-only의 추가λ 확장 우선순위는 낮아졌다. 이 관측을 모든 rate에서의 구조 우열이나 seed 통계적 유의성으로 확대하지 않는다.

MSH-only 대비 two-path56은 PSNR+2.3371dB인 동시에 bytes3.91698배다. MSH-only는 저용량 후보로 유지한다. 같은λ가 같은 bitrate를 뜻하지 않으므로 둘의 전체 RD 우열은 아직 판단하지 않는다.

Score736-only는 약40,824/50k,81.648%이며 평가 시작 로그는 없다. 이 모델은 Score predictor 채널 폭과 용량도 커지므로 단순한 rank 숫자만의 효과로 해석하지 않는다. 해당 결과를 받으면 다음 학습 후보 및 bitrate를 맞추기 위한 λ 조정을 결정한다.

## T06 고정 정렬 진단 완료

job458818의 완료 footer는 `scenes=256 orders=13`이다. `.err`는 비어 있다. 출력된256장 ID와 순서는 기존 원본 cache와 일치한다. 동일 Hilbert-trained 모델에서 순서만 바꾼 진단의 stdout으로 재계산한 전체 packet 용량은 다음과 같다.

| 순서 | 평균 B/scene | 평균 KiB/scene |
|---|---:|---:|
| Slot | 58,030.09375 | 56.67001 |
| Hilbert | 38,153.43750 | 37.25922 |

평균 용량의 비율로 Hilbert는 slot 대비34.2523% 작다.256장 모두 Hilbert가 작았고, paired Hilbert−slot은−19,876.65625B/scene이다.1000회 scene bootstrap95% 구간은[−20,176.60117,−19,598.98633]B다.

성공 footer는 코드상 모든 scene의 Hilbert packet이 T05 SHA-256 및 production compress와 정확히 일치하고, 실제 causal Score 복호·stage byte 합계·전체 byte 합계·고정 codec 상태 검사를 통과한 뒤 출력된다. 그러나 원격 metadata·scene JSON 자체는 이번 다운로드에 없어 해당 SHA와 tensor를 로컬에서 다시 읽은 것은 아니다.

비-Hilbert 순서는 Hilbert 학습 predictor의 분포를 바꾼 진단이다. 이34.25%를 재학습 정렬별 RD 개선, 동일 화질 절감률, 또는 Score Context만의 효과로 쓰지 않는다. 순서는 Score와 MSH 양쪽에 적용되며 실제 feature distortion도 달라질 수 있다. 이 진단은 렌더·PSNR을 측정하지 않는다.

Morton·random10, 실제 anchor 거리·예측 잔차·stage bits·feature 오차는 stdout에 없다. 다음 서버 보고서를 받아야 전체 해석을 완료할 수 있다.

```bash
cat outputs/nfcgs_fixed_order/job_458818/REPORT.md
```

원본 stdout의256개 paired bytes는 [t06_stdout.json](t06_stdout.json)에 보관했다.

## 추가 seed와 복구 학습

| 실험 | 최신 로그 |
|---|---|
| T11 ON16 train457786 / eval457787 | 50k 학습 완료, 평가6,295장 처리까지 확인; 최종 지표 없음 |
| T11 OFF16 train457788 / eval457789 | 50k 및 전체6991장 평가 완료 |
| None+L3 OFF12 λ=.0512, resume458782 | 약44,077/50k,88.154%; 기존 eval457098 시작 로그 없음 |
| H+D4 OFF16 λ=.0128, resume458783 | 약33,236/50k,66.472%; 기존 eval457100 시작 로그 없음 |

추가 seed OFF16은 PSNR23.6310dB, SSIM0.7336, LPIPS0.2692,44.76894KiB/scene이다. 최종50k/config·actual packet·6991장·byte 합계를 확인했다. ON16은 아직 최종 지표가 없으므로 이 seed의 ON/OFF 효과를 계산하지 않는다.

평가 loader progress의 nominal denominator는7286이지만 완료된 OFF 평가의 실제 처리 수는6991이며, 기존 production protocol의 예상 수와 일치한다. ON의6,295장은 최종 완료 수가 아니다.

두 복구 학습은 각각25k와15k의 full-state snapshot에서 재개했고 목표50k 및 원래 출력 경로를 유지한다. 위 step은 마지막 완전한 progress의 추정값이며 최신 저장 checkpoint를 직접 읽은 값이 아니다. 현재 미완료 세 학습과 평가 진행 로그에서 새 치명 오류는 찾지 못했다. 실시간 실행 상태·현재 시각 ETA를 의미하지 않는다.

## 집계와 근거

최근26개 원래 실험 기준23개가50k 학습 완료다. 전체 test 평가19개, holdout128 평가3개가 완료됐고 ON seed 평가1개는 진행 로그, 나머지 학습3개는 Score736-only와 복구2개다.

주 seed의 누적 RD는 기존 양의λ63점·16계열과 별도λ0 한 점으로 유지한다. 새 seed OFF16의 전체 test 한 점은 `seed_repeat_points`에 분리해 보관하며 주 seed 곡선에 섞지 않는다. 기존 inventory의 평가 로그 SHA-256은 모두 일치했다. architecture20개·seed17개·recovery21개 실행 파일 해시도 현재 로컬과 일치하며, 세 집합은 서로 겹친다.

- [review.json](review.json):185개 source의 크기·해시,26개 설정·상태, 비교 수치·T06 paired 구간.
- [inventory.json](inventory.json):주 seed RD63점·λ0와 별도 seed 반복 결과.
- [audit_snapshot.py](audit_snapshot.py):기존 검산 함수를 재사용하는 재현 코드.

이번 작업에서는 runtime 코드 수정이나 원격 job 제출·취소·dependency 변경을 수행하지 않았다.

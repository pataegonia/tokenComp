# 로그 검토 — 2026년 10월 20일 10:51 다운로드본

T15 독립 수신 job458784가256장 완료됐고, MSH-only 구조 pilot이50k 학습 및 holdout128 평가를 마쳤다. 두 중단 대조군은 각각25k·15k checkpoint에서 full-state resume에 성공했다. 이번 기록은 로컬 workspace의 `slurm/` 다운로드본에 근거하며 실시간 Slurm 조회가 아니다.

## T15 독립 수신 완료

- target export256장과 독립 수신256장의 완료 footer를 확인했다.
- 장면 ID256개가 T05 job458004와 일치한다. 출력된 PSNR은256장 모두 T05와 소수점5자리까지 같고, 출력된 PSNR delta도 모두0.0000000이다.
- 완료 footer는 수신 script의 지표 허용치 및 반복 feature/Gaussian 정합 검사를 통과한 뒤에만 출력된다. `.err`에는 torchvision/LPIPS 관련 경고만 있고 치명 오류는 없다.
- 서버의 `REPORT.md`, `summary.json`, `metadata.json`, scene JSON은 이번 다운로드에 없으므로 SSIM·LPIPS의 정확한 delta나 renderer 반복 pixel 오차를 직접 읽지는 못했다. 과거 T05의 pixel/feature와 bit-exact라고 표현하지 않는다.

per-scene stdout 재검산은 [t15_stdout.json](t15_stdout.json)에 있다.

## 두 대조군 복구

| 원래 학습 | 새 재개 학습 | 복원 step | 마지막 완전한 progress의 추정 step | 기존 평가 dependency |
|---|---|---:|---:|---|
| None+L3 OFF12 λ=.0512, 456951 | 458782 | 25,000 | 27,045/50k | 457098 → afterok:458782 |
| H+D4 OFF16 λ=.0128, 457099 | 458783 | 15,000 | 17,031/50k | 457100 → afterok:458783 |

두 `.err` 모두 `Restored all states from the checkpoint`를 확인했다. 실제 명령은 `checkpointing.resume=true`, 목표는 absolute50k, 출력은 원래 run/version0이다. 저장된 recovery state에서도 두 기존 평가의 dependency 변경을 확인했다. 이 state는 제출 당시 기록이며 현재 queue 상태 조회는 아니다.

비재개형 loader의 epoch 중간 재개 경고가 있다. 원래 설정의 fresh shuffle 정책에 따른 것이며 optimizer·scheduler 복원 실패로 읽지 않는다. 중단 없이 실행했을 때와 같은 다음 sample 순서까지 보장하지 않는다. 새 학습의 완료 footer나 연결 평가 시작 로그는 아직 없다.

## 구조 pilot

| 구조 | Train / Eval | 최신 로그 |
|---|---|---|
| T07 Score56-only | 457741 / 457742 | 약42,284/50k; 평가 시작 로그 없음 |
| T08 MSH-only | 457743 / 457744 | **50k 및 같은 holdout128 평가 완료** |
| T18 two-path56 | 457745 / 457746 | 약37,713/50k; 평가 시작 로그 없음 |
| T09 Score736-only | 457828 / 457829 | 약28,188/50k; 평가 시작 로그 없음 |

MSH-only의128장 결과는 PSNR20.6696 dB, SSIM0.6123, LPIPS0.3700, 평균14,831.96875B =14.48434KiB/scene이다. Score bytes0, MSH y9,208.5B, MSH z5,491.46875B, container132B로 byte 합계가 일치한다.

평가 명령의 train-derived holdout manifest·validation index,128/128 완료, final50k 경로 및 actual bitstream을 확인했다. hash가 같은 pilot runner는 실행 전에 저장 step50k와 codec config를 검사한다. 이번 다운로드에는 raw checkpoint나 평가 JSON이 없다. 이 결과를 T05의 test256 또는 기존 전체6991장 RD와 직접 순위 비교하지 않는다. 다른 세 구조가 끝나야 같은 holdout의 실제 화질·용량을 비교할 수 있다.

T11 seed111124/data404의 ON16·OFF16은 각각 약36,832/50k와36,998/50k다. 치명 오류와 연결 평가 시작 로그는 없다.

## 집계와 다음 실험

최근26개 원래 실험 기준으로19개가50k 학습 완료이며, 이 중18개는 전체 test,1개는 holdout128 평가를 완료했다. 나머지는 원래 학습5개와 재개 학습2개의 진행 로그가 있다. 전체 test 누적은 양의λ63점·16계열과 별도λ0 한 점으로 그대로다. T15와 MSH-only holdout은 여기에 합산하지 않는다.

기존 inventory의 평가 로그 SHA-256은 모두 일치했고, architecture20개·seed17개·recovery21개 실행 파일 해시도 현재 로컬과 일치했다. 이 파일 집합은 서로 겹친다. 상세 설정·상태·source 해시는 [review.json](review.json), 전체 test 목록은 [inventory.json](inventory.json), 재현 코드는 [audit_snapshot.py](audit_snapshot.py)에 있다.

새 구조 학습이나 추가λ 결정은 나머지 구조의 holdout 결과를 기다린다. 지금은 **T06 고정 H+D4 정렬 개입**을 진행한다. 같은256장과 하나의 checkpoint에서 F·projection·basis·Score·XYZ를 고정하고 slot/Morton/Hilbert/random10개의 실제 인과 anchor·예측 잔차·Score 및 MSH bytes를 연결한다. Hilbert packet이 T05와 정확히 재현돼야 나머지 순서를 분석한다.

이 진단의 비-Hilbert 순서는 Hilbert로 학습한 predictor에 대한 분포 변화다. 정렬별 재학습 RD나 구조 우열을 대신하지 않는다. 실행 파일·명령은 [T06 설명서](../../NFCGS_FIXED_ORDER_2026-10-20.md)에 있다. 이번 로컬 작업에서 원격 job을 직접 제출·취소·변경하지 않았다.

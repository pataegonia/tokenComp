# 로그 검토 — 2026년 10월 20일 09:11경 다운로드본

이후 사용자가 제공한 B0/T05 전체 보고서와 sacct 출력의 해석은 [후속 검토](FOLLOWUP.md)에 정리했다. 아래 내용은 최초 다운로드 로그를 검토한 시점의 기록이다.

B0 job458024와 T05 job458004는 모두 256장 완료 footer가 있다. 기존 26개 학습은 18개 학습·전체 평가 완료, 6개 진행 로그, 2개 `Killed / exit137` 종료 기록이다. 이번 검토는 로컬 workspace의 `slurm/` 파일 기준이며 실시간 Slurm 상태를 조회한 결과가 아니다.

## 1. B0 / T05: 완료와 예상 밖의 B1 우회 결과

| 경로 | 256장 평균 PSNR |
|---|---:|
| 공개 원본 B0 native geometry512 | 24.9027 dB |
| 현행 checkpoint의 projection 이후 B1 직접 우회 | 16.7677 dB |
| 현행 H+D4 λ=.0128 actual packet | 23.8765 dB |

위 평균은 `.out`에 소수점 다섯 자리로 출력된 장면별 PSNR을 집계했다. 서버 `summary.json`의 전체 정밀도 값을 읽은 것은 아니다. B0와 T05의 장면 ID 256개가 일대일로 일치하고, 기존 job454995의 장면 목록·순서와도 일치한다. 두 job의 B1 PSNR은 256장 모두 출력 정밀도에서 일치한다. 이를 tensor 또는 pixel bit-exact로 해석하지 않는다.

- actual−B0는 평균 −1.0262 dB다. 256장 중 255장에서 B0가 actual보다 높다. 이는 현재 맞춘 부분집합에서 native B0와 현행 방법의 전체 차이이며, 전부 양자화 손실로 볼 수 없다.
- actual−B1은 평균 +7.1088 dB다. 256장 모두 actual이 B1보다 높다. 따라서 B1 직접 우회 결과는 현행 모델의 화질 상한이 아니다.
- 학습된 projection과 codec의 복원 변환이 서로 보상하는 가능성을 우선 확인할 만하다. 직접 F를 넣는 우회가 학습된 변환을 제거해 decoder에 맞지 않는 표현을 줄 수도 있다. 이는 코드와 관측에 따른 가설이며, 현재 자료만으로 원인을 확정하거나 구현 오류를 배제하지 않는다.
- B0/T05의 `.err`에는 torchvision의 pretrained/weights deprecation 경고가 있으며, traceback·CUDA OOM·저장 오류는 없다.

현재 받은 파일에는 두 job의 `REPORT.md`, `summary.json`, `metadata.json`, `scenes/*.json`이 없다. 따라서 T05의 `lr_continuous`, `lr_decoded`, Score 양자화 우회, MSH 양자화 우회, 양쪽 양자화 우회 수치는 아직 읽지 못했다. target/frame tensor 정합, checkpoint SHA-256, backend/GPU 정보, effective readout 행렬 차이, renderer 반복 오차와 원래 slot 순서 검증 결과도 원본 JSON으로 확인하지 못했다. 전체 6991장 진단으로 확대해서 표시하지 않는다.

로그에서 재계산한 수치와 256장별 쌍은 [diagnostic_stdout.json](diagnostic_stdout.json)에 저장했다.

## 2. 종료된 학습 2개

| 실험 | Train / 연결 Eval | 마지막 완전한 progress 기준 step | 종료 기록 |
|---|---|---:|---|
| None+L3 OFF12, λ=.0512 | 456951 / 457098 | 약26,606 / 50k | ariel-v11, tasks0–1 Killed, exit137 |
| Hilbert+D4 OFF16, λ=.0128, T10 | 457099 / 457100 | 약15,336 / 50k | ariel-v9, tasks0–1 Killed, exit137 |

두 학습에는 50k 완료 메시지가 없고, 연결 평가 시작 로그도 없다. 위 step은 progress로 계산했으며 저장 checkpoint step을 확인한 값이 아니다. 로그에 CUDA OOM이나 Slurm의 명시적 종료 사유가 없어 메모리 부족·관리자 종료·취소 등을 확정하지 않는다. 재개 전에 accounting 상태, 연결 평가 dependency 상태와 실제 저장 checkpoint를 확인해야 한다. 이번 검토에서 job을 재제출·취소하거나 학습 코드를 수정하지 않았다.

```bash
sacct -j 456951,457099,457098,457100 \
  --format=JobID,State,ExitCode,Reason,Elapsed,Timelimit,ReqMem,MaxRSS,NodeList -P
```

## 3. 새 전체 평가 3개

세 평가 모두 actual bitstream encode/decode/render, 최종50k checkpoint, C12/T8, 6991장을 확인했다. byte 구성 합계는 로그 정밀도에서 일치한다.

| 모델 | λ | Train / Eval | PSNR | KiB/scene |
|---|---:|---|---:|---:|
| None+L3 OFF12 | .0032 | 456899 / 456900 | 23.7435 | 67.2904 |
| None+L3 ON | .0512 | 456901 / 456948 | 22.9142 | 29.4067 |
| Hilbert+L3 OFF12 | .0512 | 456949 / 456950 | 22.8816 | 25.4280 |

누적 전체 test 결과는 **양의 λ63점·16계열 + 별도 λ0 한 점**이다. 기존 61점의 원본 평가 로그 SHA-256은 이전 inventory와 모두 일치한다. 새 B0/T05 256장 진단은 이 전체-test 점 수에 합산하지 않았다.

새 matched 비교에서 주의할 점은 다음과 같다.

- λ=.0032, 기본 순서 None+L3에서는 Context ON이 OFF12보다 bytes0.5328% 크고 PSNR0.0147 dB 낮다. 같은 λ의 한 seed 관측이며, Context가 언제나 유리하다고 일반화하지 않는다.
- 같은 λ의 Hilbert+L3 ON은 OFF12보다 bytes9.3662% 작고 PSNR0.0169 dB 높다.
- λ=.0512의 Hilbert+L3 ON은 OFF12보다 bytes10.3781% 작고 PSNR0.1544 dB 높다.
- None+L3 OFF12의 λ=.0512 학습이 종료됐으므로 L3 2×2 네 곡선의 마지막 고율압축 점은 아직 비어 있다. T10 종료로 D4 ON16/OFF16 기본 seed 비교도 아직 미완료다.

계산값은 [new_l3_comparisons.json](new_l3_comparisons.json)에 있다. 모두 동일 λ 모델 비교이며 동일 화질 BD-rate·정렬만의 인과 효과·seed 통계 검증으로 확대하지 않는다.

## 4. 진행 로그가 있는 6개

| 실험 | Train / 연결 Eval | 마지막 추정 step | 전체50k 진행률 |
|---|---|---:|---:|
| T07 score56_only | 457741 / 457742 | 37,749 | 75.50% |
| T08 msh_only | 457743 / 457744 | 46,208 | 92.42% |
| T18 two_path56 | 457745 / 457746 | 33,502 | 67.00% |
| T09 score736_only | 457828 / 457829 | 24,528 | 49.06% |
| T11 D4 ON16, seed111124/data404 | 457786 / 457787 | 32,313 | 64.63% |
| T11 D4 OFF16, seed111124/data404 | 457788 / 457789 | 32,284 | 64.57% |

받은 로그에서는 치명 오류를 찾지 못했다. 실시간 실행 중이라는 뜻은 아니며 연결 평가 로그는 아직 없다. 구조4개는 같은 train-derived holdout128 검증이고, T11은 기존 전체 train/test 프로토콜이다. 아직 구조 우열을 판단하지 않는다.

진행률은 완전한 progress 줄의 `epoch×ceil(33017/2)+floor(batch/2)`로 계산했다. 여섯 학습 모두 앞선 epoch가 실제로33017/33017에 도달한 줄을 확인했다. review.json의 잔여 시간은 로그 속도에서 산출한 값이며 현재 시각 기준 ETA가 아니다. 종료된 두 job에는 ETA를 배정하지 않았다.

## 5. 우선 다음 확인

1. 서버의 T05/B0 `REPORT.md`·`summary.json`·`metadata.json`을 확보해 나머지5경로와 renderer 반복 오차, effective geometry readout 차이를 확인한다.
2. 두 종료 학습의 sacct와 저장 checkpoint를 확인해 복구 방식을 정한다. 중복 재제출이나 연결 평가의 자동 재개를 가정하지 않는다.
3. 나머지6개와 연결 평가를 수집한다. 추가 학습·전체 진단 확장 결정은 현재 경로 결과를 먼저 본다.

서버의 간단한 결과 표는 다음 명령으로 확인할 수 있다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
cat outputs/nfcgs_path_diagnostics/job_458004/REPORT.md
cat outputs/nfcgs_b0_matched/job_458024/REPORT.md
```

두 run의 모든 JSON이 서버에 있으면 기존 비교 기능으로 frame·target·정렬까지 검사할 수 있다. 출력 디렉터리는 새 경로를 사용한다.

```bash
python scripts/eval_nfcgs_b0_matched.py \
  --compare-only \
  --b0-run outputs/nfcgs_b0_matched/job_458024 \
  --path-run outputs/nfcgs_path_diagnostics/job_458004 \
  --output outputs/nfcgs_b0_t05_comparison/b0_458024_t05_458004_review_20261020
```

## 검증 자료

- [review.json](review.json): 26개 상태·설정 검산·종료 근거·이번 audit가 읽은166개 source의 크기/SHA-256. 이전149개 source 중 내용이 바뀐 파일은13개다.
- [inventory.json](inventory.json): 누적 양의 λ63점과 별도 λ0, checkpoint·config·원본 평가 로그.
- [diagnostic_stdout.json](diagnostic_stdout.json): 두 진단 로그의 256장 일대일 쌍과 출력 수치 집계.
- [audit_snapshot.py](audit_snapshot.py): 10월19일 audit의 검산 함수를 재사용하는 재현 코드. 이전 snapshot과 runtime 파일은 수정하지 않는다.

구조 state의 실행파일20개와 seed state의17개 해시는 현재 로컬 파일과 모두 일치했다. 두 집합에는 공통 파일이 있으므로 서로 다른37개 파일이라는 뜻은 아니다.

# 사용자 제공 REPORT 및 sacct 후속 검토 — 2026년 10월 20일

이 문서는 초기 [로그 검토](REPORT.md) 이후 받은 두 REPORT.md와 sacct 출력에 대한 후속 판단이다. 원문은 [provided_diagnostics_and_sacct.txt](provided_diagnostics_and_sacct.txt)에 보관했다. 서버의 원본 summary/metadata/per-scene JSON을 직접 읽은 것은 아니다.

## 관찰

같은 256장, H+D4 λ=.0128 최종50k checkpoint의 주요 결과는 다음과 같다. 두 양자화 우회 모드 모두 Score를 바꾸면 residual을 다시 계산하는 구현이다.

| Score | MSH y/z | PSNR | 실제 packet 대비 |
|---|---|---:|---:|
| 실제 복호 | 실제 복호 | 23.87649 | 기준 |
| 연속 우회 | 실제 복호 | 23.26808 | −0.60840 dB |
| 실제 복호 | 연속 우회 | 23.92326 | +0.04677 dB |
| 연속 우회 | 연속 우회 | 23.31259 | −0.56390 dB |

표의 차이는 사용자 REPORT가 출력한 값을 보존했다. 다섯 자리 평균을 다시 빼면 약1e−5 dB 차이가 날 수 있는데, 내부 전체 정밀도 평균을 별도로 반올림한 범위다. 우회 경로에는 actual packet rate가 없다.

1. **MSH y/z 양자화 우회의 순 개선 폭은 이 checkpoint에서 작다.** PSNR+0.04677 dB, SSIM0.74468→0.74670, LPIPS0.25992→0.25720이다. 이 관측은 남은 B0 차이를 전부 MSH 양자화 탓으로 설명하는 근거가 되지 않는다. 다만 Hcont에도 분석·합성 손실이 남으므로 MSH 전체 구조의 개선 여지가0.04677 dB라고 제한하는 것도 잘못이다.
2. **Score 양자화 우회는 전체 경로에서 오히려 낮은 화질을 보인다.** 양자화를 우회하면 decoded LR과 residual의 분포가 바뀐다. MSH를 다시 계산해도 그 변환은 실제 복호 Score를 사용하는 경로와 함께 학습된 가중치를 유지한다. 이 결과는 고정 checkpoint의 경로 상호작용을 보여주며, Score 양자화가 일반적으로 화질을 개선한다거나 Score 양자화가 불필요하다는 인과 결론을 지지하지 않는다.
3. **MSH residual을 제거하는 우회는 큰 차이를 만든다.** 실제 복호 Score만 사용하면18.56430 dB로, actual23.87649 dB보다5.31219 dB 낮다. 연속 LR만 사용하면19.80692 dB다. 따라서 LR 단독에서의 Score 양자화 차이1.24262 dB를 최종 packet의 독립 양자화 손실로 가져올 수 없다. 이 checkpoint에서 residual이 중요하다는 뜻이며, 독립적으로 재학습한 score-only/MSH-only 모델의 순위를 대신하지 않는다.
4. **B1 직접 우회는 화질 상한이 아니다.** B0=24.90270, B1=16.76771, actual=23.87649 dB다. B1에서는 원래 학습된 codec 복원 변환을 제거한다. native geometry512→학습된 projection 및 codec→geometry224 decoder의 결합을 확인해야 한다. 현재의 약1.02621 dB native B0 차이를 rank 축소·MSH 분석/합성·양자화에 독립적으로 나눠 배정하지 않는다.

native B0는 SSIM0.77651·LPIPS0.24100이고 actual은0.74468·0.25992다. 실제 packet 용량은 이 부분집합에서37.2592 KiB/scene이다. 전체6991장 결과나 역사적 non-comp 수치로 대체하지 않는다.

## 순열 및 renderer 확인

제공된 B0 REPORT에서 token당 Gaussian8개 묶음을 맞춘 means/scales/rotations/SH/opacities 최대 차이는 모두0이다. 원래 slot 순서와 공통 정렬의 평균 PSNR 차이는 약−2.61e−7 dB지만, 최대 pixel 절댓값 차이는0.0124051이다. 따라서 Gaussian 값의 순열 일치와 GPU pixel bit-exact는 구분해야 한다. 평균 PSNR 수준에서 이번 약1 dB native 차이를 정렬 순서의 renderer 차이로 설명할 근거는 없다.

동일 Gaussian·카메라의 반복 렌더에서는 B0, B1, actual 모두 보고된 pixel 최대 차이와 PSNR 차이가0이다. 이 확인은 해당 run의 반복 검사이며 모든 입력·GPU·순열에 대한 결정론 보장은 아니다.

## sacct 해석

| 학습 | 부모 job | 학습 step | 연결 평가 |
|---|---|---|---|
| 456951 | FAILED, 1:0 | 456951.0 CANCELLED, 0:9 | 457098 PENDING/Dependency |
| 457099 | FAILED, 1:0 | 457099.0 CANCELLED, 0:9 | 457100 PENDING/Dependency |

Slurm의 ExitCode는 종료값과 signal을 구분하므로 `0:9`는 signal9 종료다. 부모 Python wrapper의 exit1은 srun 실패와 일치한다. 다만 이 출력은 signal을 보낸 주체나 최초 원인을 알려주지 않는다.

`sacct Reason`은 마지막으로 Priority/Resources 이외의 사유로 스케줄링이 막힌 이유를 나타낸다. `QOSMaxGRESPerUser`는 사용자별 QOS GRES 한도와 관련된 대기 사유다. 이 값만 보고 이미 실행 중인 두 step이 GPU 한도 초과 때문에 강제 종료됐다고 결론 내리면 안 된다. [Slurm sacct 공식 문서](https://slurm.schedmd.com/sacct.html), [Job Reason Codes](https://slurm.schedmd.com/job_reason_codes.html).

부모 상태가 FAILED이고 step이 CANCELLED이며, 제공 출력에는 OUT_OF_MEMORY/TIMEOUT 상태가 없다. OOM 배제나 QOS 강제 종료 확정으로 확대하지 않는다. MaxRSS는 GPU VRAM 값이 아니며, 보고된 수치만으로 순간 메모리 피크나 노드 전체/cgroup 상황을 복원하지 않는다.

기존 연결 평가 두 개는 현재 제공 시점에 dependency로 대기한다. 원래 train job이 FAILED인 상태에서 성공 조건이 자동으로 충족된다고 가정하지 않는다. 복구 시 실제 저장 checkpoint·optimizer/scheduler 상태를 검증하고, 평가가 올바른 재개 run의 최종50k를 읽도록 경로와 dependency를 연결해야 한다. 기존 train wrapper를 그대로 다시 제출하면 weights-only warm start가 되므로 full-state resume와 같지 않다. 기본 loader 설정에서는 exact sample-order resume가 켜져 있다고 보장하지 않는다.

## 다음 순서

1. 중단된 None+L3 OFF12 λ=.0512와 H+D4 OFF16 λ=.0128의 저장 checkpoint를 확인해 복구한다. 현재 단계에서 새로운 λ나 구조를 늘리기보다 빠진 대조군을 완성한다.
2. 진행 중인 T07/T08/T18/T09 holdout128 평가를 기다려 재학습된 경로 비교로 판단한다. 이번 우회 결과를 그 순위로 대신하지 않는다.
3. 추가 GPU 학습 전에 이미 생성된 B0 metadata의 `model_pair.geometry_boundary_effective_readout_errors`와 scene feature 오차를 확인한다. projection/decoder 좌표의 변화가 B1 급락과 연결되는지 점검한다.
4. MSH 양자화 개선 학습은 현재 우선순위가 낮다. 모델 수정이 필요하면 고정 projection 대조군 등의 타당성을 검토하되, 지금 관측만으로 새 실험을 확정하지 않는다.

이번 후속 검토에서는 runtime 코드 수정, job 제출·취소·dependency 변경을 수행하지 않았다.

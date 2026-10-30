# Gate 1 전 임시 Stage B 및 60k 재분기 — 2026-10-22

사용자는 Stage A 완료 뒤 GPU 대기 시간을 줄이기 위해 임시 B=100k를 미리 afterok로 접수하고, 최종 budget은 기존 holdout Gate 1로 결정하는 절차를 승인했다. 구현은 제출 controller 한 파일에 한정했다. 30개 학습·평가 runtime 파일의 SHA-256은 다운로드 state와 모두 같다.

## 동작

- `--stage-b --wave 1 --budget 100000 --provisional`: 자기 Stage A afterok로 Stage B15개만 접수한다. 최종 budget은 미확정이며 test 및 최종 monitor는 아직 제출하지 않는다. 기존 Stage A15개·monitor ID·run 경로를 보존한다.
- Gate 1이100k를 권고한 뒤 같은 명령에서 `--provisional`을 빼면 기존 B15개를 재사용해 budget을 확정하고 test15개·최종 holdout monitor1개를 연결한다.
- 연장은 `--budget 120000 --restart-stage-b-from-60k` 또는150k와 holdout 근거인 `--reason`으로 명시한다. Gate 1 권고와 다른 선택에도 이유가 필요하다.
- 연장 전 source40k/60k의 실제 validation 완료 기록·hash와60k checkpoint의 완료 기록·hash를 확인한다. active 임시 B는 Slurm comment·job ID·작업 디렉터리·Slurm 스크립트 경로를 확인한 뒤 종료한다. Stage A와 monitor를 취소하지 않는다.
- 원래100k 실행의 checkpoint·last·로그·attempt는 원 위치에 보존한다. 새 budget root/state에서 동일 manifest 및 불변 Stage A checkpoint·완료 평가를 재사용한다. last와70k+ 및 임시 Stage B 완료 표시는 복사하지 않는다.
- 새 B는 동일60k full state에서120k(84k/108k milestone) 또는150k(105k/135k milestone)로 재개한다. 분기 안에서 이후 test·Gate 2·Wave 2도 같은 root/state를 사용한다.
- 접수 ID와 rollback 근거를 journal에 남기며 반복 실행은 ID를 재사용한다. rollback에 들어간 source는 다시100k로 확정할 수 없으며, 중간 실패 시 같은 rollback을 재개한다. 미확인 branch sbatch intent는 기존 ID 확인 절차로 처리한다.

60k→70k의 초기 LR은 같지만 버리는 임시 진행량은 각 job과 Gate 1 완료 시점에 따라 달라진다. 8~9시간을 고정된 비용으로 가정하지 않는다. 큐 접수도 즉시 GPU 할당을 보장하지 않는다.

## 검증

[다운로드 state의 오프라인 재현](downloaded_state_dry_run.txt)에서 기존16개 job과15개 run 경로 보존, Stage B15개만 추가, 각자의 Stage A만 dependency에 포함, test0개, 재실행 시 ID 재사용, 원본 state 불변을 확인했다. 실제 Slurm 제출은 하지 않았다. [hash와 검증값](verification.json).

추가 회귀 검사는 임시 접수·부분 접수 재개·100k 확정·명시적인 연장·원본 출력 보존·새 분기의60k 로드와 milestone·중복 제출 방지·rollback 중100k 재확정 금지·제출 controller만의 hash 갱신을 다룬다. Slurm 취소·접수 응답은 mock으로 검증한다.

```text
python -m pytest tests/test_fresh_rd_provisional.py tests/test_fresh_rd.py tests/test_fresh_rd_placement_recovery.py -q --disable-warnings
78 passed, 18 warnings in 23.53s
```

서버 반영 파일과 명령은 [runbook](../../NFCGS_FRESH_RD_RUNBOOK_2026-10-20.md)에 있다. 서버에서 실제 제출·취소·fork는 사용자 실행 시 이루어진다.

## 사용자 실행: 14개 접수 후 마지막 sbatch 거절

사용자가 붙인 터미널 출력에서 임시 B의14개 job461656~461669가 접수됐다. 마지막 `msh/lambda0p0032/s0/B`만 dependency460348로 sbatch exit1을 반환했다. 기존 출력에는 scheduler의 실제 stderr가 없어 거절 원인은 확정할 수 없다. 로컬에 내려받은 state는 이 접수보다 이전 것이므로 새14개 상태를 로컬 조회로 확인했다고 해석하지 않는다.

제출 controller가 실제 sbatch stderr/stdout을 콘솔과 `submission_failures`에 남기도록 수정했다. 미접수15번째만 재시도할 수 있으며 기존14개 ID는 그대로 재사용한다. 14개 접수 뒤 실패하는 경우를 회귀 검사에 추가해, 재실행 시 MSH1개만 자기 Stage A dependency로 접수하는지 확인한다. 기존 검증 JSON의 controller hash는 그 검증 당시 snapshot이며 이번 오류 출력 수정은 후속 revision이다.

후속 사용자 출력에서 실제 거절 사유는 `QOSMaxSubmitJobPerUserLimit`으로 확인됐다. 14개 ID는 모두 재사용됐고 MSH만 미접수다. 현재 실험 구성만30개 job이며 QoS의 정확한 숫자 설정은 조회하지 않았다.

`scripts/wait_nfcgs_provisional_submission.py`를 추가했다. 로그인 노드에서 현재 state의 입력 경로로 기존 controller를 실행하고, 해당 QoS 오류일 때만10분 뒤 재시도한다. 실행 중인 학습이나 대기 Stage B를 취소하지 않고 별도 Slurm/GPU job을 만들지 않는다. 잠은 child controller가 submit lock을 반납한 뒤 잔다. 다른 오류 또는 모든 접수 완료 시 종료한다. source controller·학습 runtime에는 이번 supervisor를 위해 추가 변경을 하지 않았다.

오류 출력/접수 복구 관련20개 검사는 통과했고, supervisor와 provisional 회귀 검사는 `13 passed in 7.94s`다. 실제 서버의 job 취소·조회·접수 또는 background process 실행은 agent가 수행하지 않았다.

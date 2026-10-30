# Fresh Stage A monitor 전용 복구 — 2026-10-21

18:27 다운로드에서 학습460334~460348은 모두 진행 중이지만 monitor460349는 `.monitor_1_A.lock` 때문에 `FileExistsError`로 종료됐다. state의 monitor history는 비어 있고 이전 monitor460326의 현재 상태는 다운로드만으로 알 수 없다.

`scripts/recover_nfcgs_fresh_start_20261021.py --monitor-only`를 추가했다. 학습15개가 이미 시작된 상태에서도 learner 항목을 유지하며 현재 monitor와 이전460326만 확인한다. 현재 monitor가 active/완료 상태면 재사용한다. lock 충돌로 실패한 경우 이전 active monitor를 scontrol로 확인한 뒤 종료하고, 다른 monitor 부재 및 lock 불변을 확인해 monitor만 재접수한다.

현재 sweep의 코드 hash가 바뀌었거나 예상하지 못한 active monitor가 있으면 복구를 중단한다. 원래 학습 복구 경로의 학습 전 데이터 누락 검사도 유지한다. 새 접수는 기존 Submission journal을 사용하고 실패 monitor의 dependency를 연결하지 않는다. sbatch가 거절되면 이전 lane을 복원한다.

서버 반영 파일은 기존 복구 Python 파일 하나다. 학습 및 monitor runtime 파일은 바꾸지 않아 이미 진행 중인 sweep의 코드 hash를 유지한다.

## 검증

```text
python -m pytest tests/test_fresh_rd_placement_recovery.py tests/test_fresh_rd.py -q
70 passed, 18 warnings in 23.80s
```

새 복구 검사에서는 실제 학습 artifact가 있는 15개 learner 항목·run 경로·lane·입력 hash를 유지하는지, 이전 monitor만 취소하는지, 중복 제출과 실패 dependency를 피하는지 확인했다. 다른 active monitor, lock 변경, runtime 변경, 다른 실패 원인에는 복구를 거부한다. sbatch 거절 시에도 원래 monitor ID와 lane을 유지한다.

Slurm 응답은 mock으로 검증했으며 서버 조회·제출·취소·lock 삭제는 실행하지 않았다. 서버 실행은 [runbook](../../NFCGS_FRESH_RD_RUNBOOK_2026-10-20.md)의 명령을 따른다.

## 20:27 다운로드에서 복구 후 진행 확인

사용자가 서버에서 실행한 복구는 state에 monitor460349→460774로 기록됐다. 새 monitor460774 stdout은 `evaluated=0→6→7→8`로 증가했고 stderr는 0 byte다. 기존460349의 lock 오류는 이전 작업의 로그다.

동시 다운로드한 학습15개는 모두 진행 중이며 치명적 오류 표식이 없다. 이 중8개는 약10.6k~14.2k optimizer step으로 10k를 넘었고, 나머지7개는 약4.3k~9.5k다. 새 monitor의 완료 평가8개는 현재 10k에 도달한 모델8개와 일치한다. `MONITOR_PENDING`은 남은 checkpoint를 기다린다는 의미로, 종료 오류가 아니다. 현 로그에는 monitor 재시작을 요구하는 증거가 없다. 실시간 Slurm 상태는 별도 조회하지 않았다.

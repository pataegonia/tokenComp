# Wave 1 Stage A 완료와 Gate 1 검증 — 2026-10-23 14:22

60k 학습 15개가 모두 완료됐고, monitor 462104도 `MONITOR_COMPLETE phase=A wave=1`을 출력했다. 사용자가 서버에서 출력한 Gate 1 JSON을 기존 판정 함수에 다시 입력한 결과 **120k 권고**가 일치한다. 현재 진행 중인 Stage B는 여전히 임시 100k이며 최종 budget은 state에 확정되지 않았다.

## 로그 확인

근거는 workspace `slurm/`의 14:21~14:22 다운로드본과 사용자가 붙여준 Gate 1 JSON이다. 실시간 Slurm 조회·접수·취소·checkpoint 변경은 실행하지 않았다.

- 등록된 Stage A 15개의 stderr에서 모두 `max_steps=60000` 종료를 확인했다. 가장 늦던 H-ON λ0, job 460346도 포함한다.
- 등록된 Stage A/B 30개의 다운로드된 stderr에 Traceback·CUDA OOM·Slurm 치명 오류 표식이 없다.
- monitor 462104는 정상 완료 표식을 남겼고 stderr는 0 byte다. 코드상 Stage A 15×6=90건의 checkpoint 평가가 모두 검증되어야 이 표식이 출력된다. 128장별 원 평가 파일은 이번 다운로드에 없다.
- 임시 B 15개의 시작 로그가 모두 존재한다. 14개는 progress 기준 약62.7k~74.6k다. λ0의 461668은 60k full state 복원 후 8,860 batch를 재생하는 단계로, stdout의 epoch7 batch0을 57,785 optimizer step으로 해석하면 안 된다. 재개 optimizer 진행은 아직 이 다운로드로 확인되지 않는다.

## Gate 1 집계 검증

[사용자 제공 값의 로컬 전사본](gate1_user_snapshot.json)은 서버 파일 원본과 byte hash가 같다고 주장하지 않는다. 15개 `(cell, lambda, seed)` 조합이 Wave 1 전체 inventory와 일치하며 전부 scenes=128, seed=0이다. 15개 모두 `delta_db_j > 2 * paired_se_db_j` 조건과 flag가 일치한다.

예산 선택에 쓰는 핵심 12개는 모두 유의한 개선을 보인다. 개선 지표 범위는0.202797~0.820431이며 최대값은 H-OFF λ0.0032의0.820431이다. λ0·S56·MSH는 핵심 최대값 계산에서 제외한다. 이 수치는 이미지 MSE·LPIPS·실제 bit를 합한 목적함수의 MSE 정규화 개선 지표다.

사전 정의한 규칙은 핵심 최대 유의 개선이0.1 이상이면120k,0.2 이상이면150k도 검토하도록 표시한다. 재계산 결과 `recommended_budget=120000`, `consider_150k=true`, `largest_significant_delta_db_j=0.8204312304757352`가 제공된 JSON과 일치한다. 150k 자동 선택은 하지 않는다. Gate JSON은 `test_used=false`와 train-derived holdout128을 명시한다. 원 장면별 metric이 없으므로 paired SE와 목적함수를 원본128장으로 재계산한 검증까지 수행한 것은 아니다.

## 120k 재시작에 남은 문제

다운로드된 실제 checkpoint 감사는09:09 KST에 완료된11개를 대상으로 했으며, **9개의 현재60k SHA가 Stage A completion의 원 SHA와 달랐다.** 그9개에서 같은 폴더의 `step000060000*.ckpt` 중 원 SHA와 일치하는 backup은 발견되지 않았다. 당시 나머지4개는 Stage A가 끝나지 않았다. 외부 backup이나 Ceph snapshot의 부재까지 증명한 것은 아니다.

이는 앞서 재현한 Stage B 재개 첫 microbatch의60k 파일 덮어쓰기 문제와 관련된다. 실제 서버의 원본과 현재 tensor·optimizer·RNG·loader가 같다고 검증한 것은 아니다. 작은 로컬 재현의 가중치 동일성을 실제 서버 full state 보존의 증거로 사용하지 않는다.

기존120k/150k rollback 절차는 Stage A completion hash를 먼저 확인한다. 이 불일치가 남아 있으면 B를 취소하기 전에 중단된다. 따라서 **Gate 1의120k 권고와, 원래60k full state에서120k를 시작할 수 있다는 판단은 별개**다. 현재 B 학습은 유지하며 최신15개 전체의 audit과 원본 backup 확인이 우선이다. hash 기준을 바꾸거나 불일치를 무시하지 않는다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
python scripts/recover_nfcgs_monitor_identity_20261023.py --audit-only
```

이 명령은 checkpoint와 learner job을 변경하지 않고 새 `logs/slurm/fresh_rd/checkpoint_identity_audit_<시각>.json`을 만든다. 이후 로그 다운로드에서 판정 파일도 함께 확보하려면:

```bash
mkdir -p logs/slurm/fresh_rd/convergence
cp outputs/nfcgs_fresh_rd/convergence/gate1.json \
   outputs/nfcgs_fresh_rd/convergence/wave1_A_complete.json \
   outputs/nfcgs_fresh_rd/convergence/curve_wave1_A.json \
   logs/slurm/fresh_rd/convergence/
```

monitor 종료 후 λ0 B의 첫 optimizer 이전 microbatch 저장으로60k가 바뀔 가능성도 있으므로, 연장 전에 실제 checkpoint와 평가 completion의 SHA를 다시 확인해야 한다. [검증 기록](verification.json).

# Fresh RD 구현 검증

상태: 로컬 구현·검증 완료. 서버 Slurm 접수와 실제 데이터 GPU Gate 0/학습/평가는 수행하지 않았다.

- 새 검사: `tests/test_fresh_rd.py` **57 passed**. Hydra 8셀×A/B budget, 추가 λ, 실제 CPU entropy encode/decode, 단계별 eval/coder 대조, 초기화 변환/hash, 직접 MSE/paired SE, 제출 중단·재실행, 의존성 복구, monitor 완료 검증을 포함한다.
- 기존 경로를 포함한 회귀 실행: **88 passed, 25 subtests passed**. 이 수에는 당시 새 검사도 포함되므로 57과 합산하지 않는다. 기존 resume/checkpoint, stateful loader, weights-only load, architecture pilot, seed pair, 10월 20일 follow-up을 함께 검사했다.
- 짧은 실제 Lightning CPU 학습에서 연속 실행과 분할 재개의 샘플·LR·최종 가중치가 일치했다. epoch를 넘는 재개와 저장된 MultiStepLR milestone 변경도 검사했다.
- 설치된 Lightning 2.3의 batch-end checkpoint에서 completed batch가 하나 뒤처지는 문제를 새 경로의 저장 hook으로 보정했다. quantile 갱신 후 checkpoint를 저장하고 RNG를 복원한다.
- shell/Slurm 8개 `bash -n`, LF, BOM 없음 확인. Python 문법 및 기존 추적 파일의 `git diff --check` 통과.
- 전체 dry-run: **113개 job 항목**. 35개 모델의 A/B 학습 70개 + test 35개 + Wave 1 monitor 2개 + Gate 0 6개. seed 초기화와 부록 monitor는 별도다. dry-run은 state를 생성하거나 sbatch를 호출하지 않는다.
- 새로운 GPU 작업은 16개의 afterok 체인으로 제한한다. 기존 다른 실험의 할당은 포함하지 않는다.

실행은 [실행 안내](../../NFCGS_FRESH_RD_RUNBOOK_2026-10-20.md)를 따른다. 실제 GPU 처리량·Gate 0 통과·35개 학습의 성능·수렴은 서버 결과로 확인해야 한다. 원안의 완료 날짜는 Gate 2 뒤 Wave 2를 시작하는 순서와 실측 처리량에 맞춰 재산정한다.

[파일 hash와 검증 요약](verification.json), [접수하지 않은 전체 dry-run](submission_dry_run.txt).

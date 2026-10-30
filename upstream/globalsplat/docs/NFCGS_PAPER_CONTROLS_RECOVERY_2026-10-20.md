# 456951 / 457099 두 대조군 복구

복구 대상은 None+L3 OFF12 λ=.0512의 train456951과 Hilbert+D4 OFF16 λ=.0128의 train457099다. 기존 연결 평가457098·457100은 유지하고, dependency를 새 재개 학습 ID로 변경한다. 다른 학습·평가는 선택하지 않는다.

## 반영할 새 파일

- [복구 controller](../scripts/recover_nfcgs_paper_controls_20261020.py)
- [full-state resume Slurm](../scripts/slurm/resume_nfcgs_paper_controls_20261020.slurm)

기존 `scripts/recover_nfcgs_sort_context_ablation.py`, `scripts/prepare_nfcgs_nn_score_resume.py`, `scripts/run_nfcgs.py`, 공통 Slurm 설정 및 원래 평가 script를 재사용한다. 기존 파일은 수정하지 않았다. 누락되면 어떤 파일이 필요한지 표시하고 제출 전에 종료한다.

## 서버 실행

두 새 파일을 동일한 repo 상대 경로에 반영한 뒤 실행한다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
python scripts/recover_nfcgs_paper_controls_20261020.py --submit
```

제출 없이 실제 checkpoint와 현재 Slurm 상태를 점검하려면 `--submit`을 빼면 된다. 별도의 확인 입력은 요구하지 않는다.

controller는 원래 `logs/slurm/paper_controls_20261018/submissions.tsv`의 두 행을 설정·경로·train/eval ID와 대조한다. sacct와 squeue로 본인 job의 현재 상태를 읽고, 원래 학습이 아직 active이면 재제출하지 않는다. 기존 평가가 pending일 때 dependency가 원래 train 또는 이 controller의 재개 train을 가리키는지도 검사한다. 두 실험 모두 preflight를 통과한 후 제출을 시작한다.

## checkpoint와 학습 상태

checkpoint는 다음 두 원래 run 디렉터리에서 찾는다.

```text
outputs/nfcgs_order_context/none/legacy_stages3_k5_ste_context_off/lambda0p0512/job_456951/
outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste_context_off/lambda0p0128/job_457099/
```

각 run의 가장 높은 numbered checkpoint와 last.ckpt를 실제로 읽고 global_step, codec 설정, optimizer, scheduler, Lightning version 및 loop state를 검사한다. 마지막 로그의26.6k·15.3k를 저장 step으로 가정하지 않는다. 유효한 checkpoint가 없거나 weights-only이면 중단하고, 10k 부모부터 새 학습하는 fallback은 사용하지 않는다.

재개 job은 선택 checkpoint의 step과 SHA-256을 다시 검사하고, 원래 `version_0` 안에 `resume_job_<NEW_JOB_ID>.ckpt`를 byte 단위로 복사한다. 이 snapshot에서 `--resume`으로 optimizer·scheduler·global_step·loop 상태를 복원한다. snapshot은 checkpoint pruning 대상의 원래 numbered 파일과 구분되므로 시작 상태를 보관할 수 있다.

총 목표는 **absolute global_step50,000**이다. 예를 들어 저장 step이25,000과15,000이면 각각 남은25,000과35,000을 학습한다. 50k를 추가로 학습하는 명령이 아니다.

같은 원래 run 디렉터리에서 계속 저장하므로, 기존 평가가 읽는 `checkpoints/nfcgs_main/version_0/step000050000.ckpt` 경로도 유지된다. GPU2·GPU당 batch2·누적2·effective batch8·BF16·model seed111123·data seed403·LR 및35k/45k scheduler·MSH 설정을 유지한다. 데이터셋 경로 검사는 계산 노드에서 수행하고 TMPDIR은 Ceph의 job별 경로를 사용한다.

기존 설정의 `exact_resume=false`는 바꾸지 않는다. 따라서 학습 상태는 복원하지만 끊기지 않은 실행과 동일한 다음 sample 순서까지 보장하지 않는다. 원래 loader의 재개 시 fresh shuffle 정책을 따르며, 결과 provenance에는 중단·재개 사실을 보존한다.

## 기존 평가 연결과 재실행

새 학습 접수 직후 ID를 `logs/slurm/paper_controls_recovery_20261020/state.json`에 atomic save한다. 그 다음 기존 평가에 다음 형태의 dependency를 적용한다.

```text
457098 → afterok:<456951의 새 resume job>
457100 → afterok:<457099의 새 resume job>
```

dependency 변경에는 `scontrol update JobId=... Dependency=...`를 사용한다. [Slurm scontrol 공식 문서](https://slurm.schedmd.com/scontrol.html).

새 학습·평가를 모두 다시 만드는 명령이나 scancel은 사용하지 않는다. 이미 최종50k checkpoint가 검증되면 학습을 추가하지 않고 기존 평가만 풀어 준다. 새 학습의 sbatch가 접수된 뒤 다른 제출이나 dependency 변경이 실패해도 ID를 먼저 저장하므로, 같은 `--submit` 명령을 재실행하면 접수된 학습을 재사용한다.

새 재개 학습 자체가 실패한 경우에는 자동으로 또 제출하지 않는다. 원인을 확인한 뒤 같은 controller에 `--submit --retry-failed`를 주면 최신 저장 상태에서 다음 재개 시도를 만들 수 있다. 수동으로 state 파일을 지워 다시 실행하거나 새 job을 `scontrol requeue`하는 방식 대신 이 기록을 유지한다.

상태 파일에는 원래 ID, 새 ID, 시도별 시작 step·checkpoint·SHA-256, 기존 평가 ID, 실제 dependency를 저장한다. 원래 paper-controls 제출 TSV와 state는 바꾸지 않는다. 이후 결과를 집계할 때는 새 resume 로그 및 이 state를 원래 run·평가 ID와 함께 읽어야 한다.

## 출력

```text
PLAN ... MODE=resume RESTORED_STEP=<실제 저장 step> ...
RESUME_SUBMITTED ORIGINAL=456951 NEW=<새 학습 ID> FROM_STEP=<저장 step>
EVAL_REBOUND ID=457098 DEPENDENCY=afterok:<새 학습 ID>
RESUME_SUBMITTED ORIGINAL=457099 NEW=<새 학습 ID> FROM_STEP=<저장 step>
EVAL_REBOUND ID=457100 DEPENDENCY=afterok:<새 학습 ID>
```

학습 로그는 `logs/slurm/slurm-gs-paper-control-resume-<NEW_JOB_ID>.out/.err`에 쌓인다. 평가는 기존 `slurm-gs-order-context-eval-457098`·`457100` 파일을 사용한다.

## 검증

[tests/test_paper_controls_recovery_20261020.py](../tests/test_paper_controls_recovery_20261020.py)에서 read-only preview, 두 job 제한, 기존 평가 재사용, partial submit/dependency 실패 후 중복 방지, optimizer·scheduler·byte 보존, 원래 version0 저장 경로, 최종50k 전용 평가, 잘못된 dependency/active train/weights-only 거부 및 Slurm 실행 명령을 검증한다.

```bash
python -m unittest discover -s tests -p test_paper_controls_recovery_20261020.py -v
```

로컬 검증은 작은 checkpoint fixture와 모의 Slurm을 사용한다. 실제 서버 checkpoint와 상태의 검증·제출은 위 controller를 서버에서 실행할 때 수행된다. 이번 구현 중 원격 job을 직접 제출하거나 변경하지 않았다.

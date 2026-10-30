# 취소된 정렬·context 실험 복구 — 2026-10-14

사용자는 NN-score 학습 450934/450937/450939를 유지하고 다른 job들을 잠시 취소했다. 이 기록은 10월 14일 14:02~14:03에 내려받은 로컬 로그와 state를 근거로 한다. 서버 상태를 직접 조회하거나 job을 제출하지 않았다.

## 로그에서 확인한 상태

| 유지할 학습 | 설정 | 마지막 로그의 진행 |
|---|---|---|
| 450934 | NN-score .0256, step10k 전체 상태 resume | epoch2, 약15,092/33,017 batch, 약 .94 it/s |
| 450937 | NN-score .0064 | epoch1, 약30,322/33,017 batch, 약 .97 it/s |
| 450939 | NN-score .0128 | epoch1, 약31,433/33,017 batch, 약 .99 it/s |

위 숫자는 progress bar의 batch 진행이며 optimizer global step으로 쓰지 않는다. 세 학습 모두 첫 batch 이후 진행 기록이 있어 이전 Numba import 오류를 넘어 학습했다. 450934는 모델·optimizer·scheduler 전체 상태 복원이 확인된다. 받은 로그의 끝에 새 실패 traceback이나 취소 메시지는 없다.

취소된 ablation 학습은 450977/450979/450981, 450983/450985/450987, 450989/450991/450993, 450995다. 13:46:51~13:47:23 사이 취소 메시지가 있다. `.out`에는 설정과 `srun` 실행 명령까지만 있으며, 모델 초기화·epoch·optimizer step·checkpoint 저장 기록이 없다. 따라서 로그만으로 이어받을 학습 진도를 확인할 수 없다. 서버의 실제 checkpoint 존재 여부는 복구 helper가 실행 시 검사한다.

| 실험 | .0064 이전 학습 / eval | .0128 이전 학습 / eval | .0256 이전 학습 / eval |
|---|---|---|---|
| 정렬 없는 legacy3 | 450977 / 450978 | 450979 / 450980 | 450981 / 450982 |
| Hilbert only | 450983 / 450984 | 450985 / 450986 | 450987 / 450988 |
| 정렬 없는 quarter2 | 450989 / 450990 | 450991 / 450992 | 450993 / 450994 |
| 정렬 없는 dyadic4 | 450995 / 미제출 | 미제출 / 미제출 | 미제출 / 미제출 |

기존 NN-score 후속 eval ID는 450935/450938/450940다. 사용자의 취소 범위상 이 eval들도 취소됐을 수 있으므로 helper는 현재 상태를 조회해 살아 있는 eval은 재사용하고 취소된 eval만 새로 연결한다.

## 복구 동작

[recover_nfcgs_sort_context_ablation.py](../scripts/recover_nfcgs_sort_context_ablation.py)는 사용자 소유 job의 `squeue`·`sacct` 상태와 원래 제출 state를 확인한다. 원래 ablation state 10개의 설정 서명이 이번 recipe와 일치함을 로컬에서 확인했다.

1. NN-score 학습 450934/450937/450939의 ID를 고정한다. 이 세 학습을 재제출·취소·재시작하지 않는다. 필요하면 각 학습에 대한 `afterok` eval만 복구한다.
2. 정렬 없는 legacy3·quarter2·dyadic4와 Hilbert only, 총 12개 학습 point를 복구한다. 기존 train이 살아 있으면 재사용한다.
3. 취소된 train의 실행 폴더에 checkpoint가 있으면 가장 높은 저장 step을 선택하고, codec 설정과 모델·optimizer·scheduler·Lightning metadata를 검증한다. 새 폴더에 원본 bytes를 복사해 전체 상태 resume로 총50k까지 이어 학습한다. 원본 checkpoint는 수정하지 않는다.
4. checkpoint가 없으면 공통 FullSplit .0256 부모의 가중치에서 새50k를 시작한다. 로그에 진도가 없던 이번 취소 건들은 이 경로가 될 가능성이 있지만 서버 파일 확인 전에는 확정하지 않는다.
5. 유효한 최종 step50k가 있으면 학습을 다시 제출하지 않고 eval만 연결한다.
6. 접수된 학습 ID를 eval 제출 전에 저장한다. QOS 한도로 중단돼도 같은 명령을 다시 실행하면 접수된 ID를 재사용하고 남은 제출만 재시도한다.

새 학습은 GPU2·batch/GPU2·accumulate2·유효배치8, scope all/MSH 유지, LR1e-4와 35k/45k decay, 5k checkpoint, BF16, 최대6일로 기존 조건을 유지한다. 예전 ablation state는 수정하지 않고 별도 복구 state를 사용한다. `scancel` 호출은 없으며 다른 프로젝트의 job은 복구 대상에 포함하지 않는다.

모델·optimizer·scheduler·global step은 checkpoint에서 복원한다. DataLoader의 입력 순서까지 동일하게 재현하는 `exact_resume`는 기존 기본값 false를 유지한다.

## 로컬 검증

11개 테스트와 13개 subtest가 통과했다. 가짜 Slurm 환경에서 보호 대상 세 학습을 유지한 채 취소된 eval만 복구하고, checkpoint가 없는 12개 point를 재시작하는 흐름을 확인했다. 실제 checkpoint의 byte 복사와 모델·optimizer·scheduler·global step 보존, 반복 실행의 ID 재사용, eval 제출 거부 직전에 접수된 학습 ID 저장과 부분 재시도, 완료된 step50k의 eval-only 처리도 통과했다. Git Bash에서는 resume와 weights-only 두 분기의 실행 인자·출력 경로 및 LF shebang·shell 구문을 검증했다.

다운로드된 원래 ablation state 10개의 설정 서명을 서버 경로 기준으로 검산했고 모두 일치했다. 나머지 두 point에는 원래 제출 state가 없다. 이 검증은 서버의 현재 job 상태나 실제 checkpoint 존재 여부를 대신하지 않으며, 해당 사항은 helper가 실행 시 확인한다.

## 서버 반영과 실행

서버 저장소의 같은 경로에 다음 두 파일을 추가한다.

- [scripts/recover_nfcgs_sort_context_ablation.py](../scripts/recover_nfcgs_sort_context_ablation.py)
- [scripts/slurm/train_nfcgs_order_context_recovery.slurm](../scripts/slurm/train_nfcgs_order_context_recovery.slurm)

이전에 사용한 `prepare_nfcgs_nn_score_resume.py`, `run_nfcgs.py`, `nfcgs_order_context_settings.sh`, `eval_nfcgs_order_context.slurm`과 ablation OFF 구현은 서버에 있어야 한다. 이번에는 이 기존 파일들을 변경하지 않았다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
python scripts/recover_nfcgs_sort_context_ablation.py --dry-run
python scripts/recover_nfcgs_sort_context_ablation.py
```

Preview에서 `MODE=restart`는 저장된 checkpoint가 없어 같은 부모에서 시작함을, `MODE=resume STEP=<N>`은 저장된 N step에서 전체 상태를 이어받음을 뜻한다. `MODE=reuse`는 현재 job을 그대로 사용하는 것이다.

세 NN-score의 eval만 먼저 연결하려면 다음 명령을 사용한다. 이후 전체 복구 명령을 실행해도 같은 복구 state에서 해당 eval ID를 재사용한다.

```bash
python scripts/recover_nfcgs_sort_context_ablation.py --nn-evals-only
```

기본 복구 state는 `logs/slurm/sort_context_recovery_20261014.json`이다. 새 학습 출력은 각 recipe의 `RUN_ROOT/recovery_job_<NEW_TRAIN_ID>/`이며 checkpoint와 `eval_all/`은 그 아래 저장된다. 반복 실행할 때 복구 state를 그대로 유지한다. 기존 제출 helper는 취소된 과거 ID를 재사용하므로 이번 복구에는 새 helper를 사용한다.

원래 ablation 12개가 모두 재제출 대상이고 NN-score eval 3개도 모두 취소됐다면 새 제출량은 학습12 + ablation eval12 + NN-score eval3 = 27개다. 유지하는 세 학습까지 포함하면 30개 job이다. 실제로 살아 있거나 완료된 job은 재사용하므로 현재 상태에 따라 새 제출 수는 줄어든다. 다른 job이 계정의 제출 한도를 사용 중이면 부분 제출 후 같은 복구 명령으로 이어서 제출한다.

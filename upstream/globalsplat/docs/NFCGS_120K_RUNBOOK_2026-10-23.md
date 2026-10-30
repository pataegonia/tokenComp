# 120k 분기와 Wave 2 실행 — 2026-10-23

Gate 1 권고120k를 적용한다. `consider_150k=true`는 기록에 남기되 일정 때문에120k를 선택한다. 기존 provisional100k 산출물은 원래 위치에 보존한다. 이 문서는 기존 fresh runbook의120k 연장·Wave 2 절차를 대체한다.

## 서버 반영 파일

[파일 목록](nfcgs_120k_20261023_files.txt)의9개 실행 파일을 같은 상대 경로로 반영한다. 이미 기존8개를 반영했다면 마지막 holdout 복구 파일만 추가한다. ZIP은 만들지 않는다.

- [fresh_rd_runtime.py](../globalsplat/fresh_rd_runtime.py)
- [model_wrapper.py](../globalsplat/model/model_wrapper.py)
- [run_nfcgs_fresh_rd.py](../scripts/run_nfcgs_fresh_rd.py)
- [submit_nfcgs_fresh_rd.py](../scripts/submit_nfcgs_fresh_rd.py)
- [monitor_nfcgs_fresh_rd.py](../scripts/monitor_nfcgs_fresh_rd.py)
- [train_nfcgs_fresh_rd.slurm](../scripts/slurm/train_nfcgs_fresh_rd.slurm)
- [wait_nfcgs_provisional_submission.py](../scripts/wait_nfcgs_provisional_submission.py)
- [verify_nfcgs_fresh_120k.py](../scripts/verify_nfcgs_fresh_120k.py)
- [repair_nfcgs_stage_a_holdout.py](../scripts/repair_nfcgs_stage_a_holdout.py)

기존 checkpoint, `run.json`, state JSON은 직접 수정하지 않는다. 원 sweep은 이전 runtime hash를 그대로 유지하며 새 분기에만 승인된 코드 변경을 기록한다. runtime 변경 허용 범위는 checkpoint callback, 해당 모델의 protocol 검사, fresh launcher의3개 파일이다. 나머지 모델·데이터·초기화·학습 recipe 변경은 거부한다. 제출·monitor·Slurm controller 변경은 별도로 기록한다.

## 수정과 허용 조건

`FreshRDCheckpoint.on_train_start`에서 `_last_global_step_saved`를 복원된 `trainer.global_step`으로 맞춘다. quantile step 저장·복원과 갱신 조건은 유지한다. Stage A 종료 후60k 파일을 `stage_a_final_step000060000.ckpt`로 hard link하고 모든 write bit를 제거한다. completion JSON에 원 경로·보호 경로·두 SHA를 남긴다. hard link는 같은 inode이므로 canonical60k도 읽기 전용이 된다.

Lightning2.3.0의 설치 소스를 확인하면 `_atomic_save`는 `fsspec.open(filepath, "wb")`에 직접 쓴다. 임시 파일 rename으로 원본이 보호된다고 가정하지 않는다. 경계 재저장 방지와 읽기 전용 link를 함께 사용하며 실제 재개 테스트에서 경계 파일의 SHA·mtime 유지와 이후 저장을 확인한다.

`--accept-resaved-60k`는 `--restart-stage-b-from-60k --budget 120000`과 함께만 허용한다. 원 SHA가 같으면 `stage_a_original`이다. 다르면 아래를 모두 확인한다.

1. step60000, epoch7; scheduler last_epoch60000, milestone70k/90k.
2. optimizer 모든 group와 base LR이1e-4; full optimizer/scheduler/loader state 존재.
3. total processed239980(원본239979에서 정확히+1), 저장된 quantile 갱신 step60000, 원 run protocol 일치.
4. 40k/60k 평가 completion의 checkpoint SHA·장면 metric SHA·128장 schema 확인.
5. superseded 평가가 있으면 원 Stage A SHA에 묶였는지 확인하고 장면별 frame ID·MSE·LPIPS·actual bytes가 현재 평가와 정확히 같은지 확인.
6. 현재 장면별 metric에서 paired 개선·SE와 Gate 1 결정을 다시 계산하여 JSON과 비교.

실패하면 해당 run과 검사 이유를 출력하고50k 원본의 별도 continuation을 검토하도록 안내한다. 자동 대체하지 않으며 모든 run의 preflight가 통과하기 전에는 B를 중단하지 않는다. 실제 dry-run도 같은 입력 검사를 한다.

재저장본은 원 full state와 같다고 기록하지 않는다. provenance에 원 SHA·현재 SHA·검사 목록·평가 비교·Gate 1 SHA/선택 사유를 저장하고, microbatch1개(장면2개)를 건너뛰며 첫 optimizer step은 microbatch3개라는 차이를 명시한다.

## Wave 1: 실제 입력 dry-run

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
python scripts/submit_nfcgs_fresh_rd.py \
  --stage-b --wave 1 --budget 120000 \
  --restart-stage-b-from-60k --accept-resaved-60k --dry-run
```

`SOURCE_60K`의 run별 판정, `WOULD_STOP_PROVISIONAL_ONLY`의 ID, `WOULD_LINK` 파일 목록을 확인한다. 실제 state가 있는 dry-run은 checkpoint와 평가를 검사하고 Slurm에는 조회만 한다. state·checkpoint·job을 변경하지 않는다. 원본 SHA가 같은 run과 재저장본 run을 구분한다.

### 모니터 종료 후60k가 재저장되어 평가 SHA가 다를 때

10월23일 실제 dry-run은 앞12개를 통과한 뒤 `h_on/lambda0/s0`에서 `checkpoint identity changed after evaluation`로 중단됐다. 뒤의S56/MSH는 이 출력만으로 통과 여부를 알 수 없다. 이 단계는 job을 중단하기 전이므로 기존 provisional B는 그대로 유지된다. λ0의 늦은 B 시작 시점과 오류는 모니터 완료 후 경계 재저장에 부합한다.

이 경우 현재60k를 실제 holdout128장으로 다시 평가한다. 위 복구 파일 하나를 서버의 `scripts/`에 추가하고 실행한다.

```bash
python scripts/repair_nfcgs_stage_a_holdout.py --submit
```

원 sweep의15개40k/60k 평가를 먼저 검사하며, 현재 SHA와 평가 SHA가 다른60k만 재평가한다. 재저장본의 step/epoch·scheduler·LR·processed+1·quantile·loader·원 protocol 조건을 먼저 검사한다. 변경된 runtime은 승인된3개 파일만 허용하며, 별도 audit run manifest를 사용하므로 원 `run.json`과 state·checkpoint를 수정하지 않는다. 실제 평가 전에는 기존 평가를 `.superseded_*`로 보관하고 Gate 1을 무효화한다. 새 평가를 기존 평가와 장면별 frame ID·MSE·LPIPS·actual bytes로 비교하여 모두 같을 때만 전체 Gate 1과 holdout curve를 갱신한다. 값이 다르면 Gate 1 없이 중단하며 다음 실행도 차이를 거부한다. 실패한 평가부터 같은 명령으로 재시도할 수 있다. 기존 monitor lock은 자동으로 지우지 않는다.

이 명령은 **재평가용1GPU job 하나**를 제출한다. login node에서 직접 GPU 평가를 시작하지 않으며 학습 job은 취소하지 않는다. `HOLDOUT_REPAIR_JOB=<ID>`의 `.out`·`.err`와 각 평가의 `monitor.log`를 확인한다.

```bash
tail -n 40 logs/slurm/slurm-gs-fresh-holdout-repair-<ID>.out
tail -n 40 logs/slurm/slurm-gs-fresh-holdout-repair-<ID>.err
```

`HOLDOUT_REPAIR_COMPLETE`를 확인한 뒤 위120k dry-run을 다시 실행한다. 복구는120k 학습을 자동 제출하지 않는다. 단순 SHA 덮어쓰기나50k 자동 대체는 하지 않는다. 추가 read-only 검사가 필요하면 복구 명령에 `--dry-run`을 사용한다.

## Wave 1: 제출

직접 실행은 같은 명령에서 `--dry-run`만 뺀다.

```bash
python scripts/submit_nfcgs_fresh_rd.py \
  --stage-b --wave 1 --budget 120000 \
  --restart-stage-b-from-60k --accept-resaved-60k
```

이 과정은 검증한 원 provisional B만 중단하고 `outputs/nfcgs_fresh_rd/branches/B120000/` 및 `logs/slurm/fresh_rd/state_B120000.json`을 만든다. 10k~60k와 평가를 연결하며70k 이상·`last.ckpt`·기존 B 완료 표식은 가져오지 않는다. 원 manifest는 `source_run.json`에 보존하고 새 `run.json`에는 현재 코드 및 `resume_parent.json`에 고정한 부모 lineage를 명시한다. launcher는 부모 파일 SHA를 검증하고 모델은 그60k 부모 protocol만 허용한다. 이후 저장되는 checkpoint에는 새 protocol이 들어간다.

부모 파일을 모두 검증한 새 분기는 과거 Stage A/monitor ID에 새 `afterok`를 걸지 않는다. 이미 끝난 ID는 Slurm `MinJobAge` 이후 새 의존성에 사용할 수 없기 때문이다. Wave 2도 완료된 Wave 1 lane을 해제하며, 실제 활성 predecessor가 있으면 유지한다. [Slurm 의존성 규칙](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

15개 B,15개 test, 최종 holdout monitor는31개라 기존30-job 제출 한도에 걸릴 수 있다. 이 환경에서는 위 직접 실행 대신 아래 supervisor를 쓰면 같은120k 명령을 실행하고 **QOSMaxSubmitJobPerUserLimit만**600초마다 재시도한다. 이미 받은 ID를 재사용하며 다른 오류는 즉시 중단한다. supervisor는 login node의 Python 프로세스이며 GPU job을 추가로 요청하지 않는다.

```bash
nohup python scripts/wait_nfcgs_provisional_submission.py --restart-120k --interval 600 \
  > logs/slurm/restart_B120k_submit.log 2>&1 < /dev/null &
tail -n 30 logs/slurm/restart_B120k_submit.log
```

`RESTART_120K_SUBMISSION_COMPLETE`가 나와야 전체 제출 완료다. CLI의 pending sbatch intent가 남았다면 기존 `--record-accepted` 절차로 해당 state의 ID를 먼저 복구한다. supervisor를 Slurm GPU job으로 제출하지 않는다.

## 시작 직후와70k 저장 후

```bash
python scripts/verify_nfcgs_fresh_120k.py \
  --state logs/slurm/fresh_rd/state_B120000.json
```

원 sweep·분기·보호 link의60k SHA를 승인한 부모 SHA와 비교한다.70k가 있으면 실제 optimizer LR1e-4, milestone84k/108k, 현재 protocol을 확인한다. TensorBoard의 LR event를 streaming으로 읽어84k 전 값이1e-4인지 검사한다. 체크포인트나 event가 아직 없으면 관측 대기로 기록하며, 관측된 불일치는 오류로 반환한다. 결과는 같은 state 폴더의 `verify_B120000_<UTC>.json`이다.70k 저장 전후에 각각 실행하면 된다. 이 검사는 CPU/파일 읽기만 사용한다.

## Gate 2와 Wave 2

Gate 2는 **분기** `convergence/curve_wave1_final.json`과 `wave1_final_complete.json`의 최종 holdout을 검토한다. test는 보고용이며 Gate 2 입력으로 사용하지 않는다. s1/s2 초기화 준비는 기존 fresh runbook 절차를 따른다. 필요한 seed preparation이 없으면 Wave 2는 첫 제출 전에 중단한다.

최종 holdout을 검토해 진행하기로 정한 뒤:

```bash
python scripts/submit_nfcgs_fresh_rd.py \
  --root outputs/nfcgs_fresh_rd/branches/B120000 \
  --state logs/slurm/fresh_rd/state_B120000.json \
  --gate2 proceed --budget 120000 --reason 'Reviewed final 120k holdout; test excluded'

python scripts/submit_nfcgs_fresh_rd.py \
  --root outputs/nfcgs_fresh_rd/branches/B120000 \
  --state logs/slurm/fresh_rd/state_B120000.json \
  --wave 2 --budget 120000
```

Wave 2는각각0→120k를 한 allocation에서 연속 학습한다. milestone84k/108k, exact resume,10k 저장,1GPU/micro2/accum4를 유지한다. 60k에서 GPU를 반납하지 않는다. 각 job에서 학습 후 full test까지 순차 실행하므로20개 train+test job과 최종 holdout monitor1개, 총21개를 제출한다.40개 별도 train/test job으로30-job 한도를 넘기지 않는다. Slurm walltime은6일이며 train+test 합계에 적용된다.

앞16개는 core, 뒤4개는 MSH 추가λ2개 → M-ON → H-L3 순이다. 뒤4개는 앞16개 train+test job 전체의 `afterok`에 묶인다. 논리 lane 제한은 자원 예약이나 다른 사용자보다 높은 우선순위를 뜻하지 않는다. Wave 1의60k 재개 및 재저장본에 따른 차이는 branch lineage와 Wave 2 `training_plan`에 남는다.

앞16개가 끝난 시점에 일정 때문에4개를 제외하려면 아래를 먼저 dry-run하고 같은 명령에서 `--dry-run`을 뺀다. 이 옵션은4개만 취소하고 final monitor의 의존성과 평가 inventory를16개로 함께 조정한다. 원 run 기록과 산출물은 보존한다.

실제 취소에서는16개 성공 완료를 확인한 뒤 선택 inventory를 먼저 기록하고 monitor의 기존 의존성을 해제한 다음4개를 취소한다. 자동 invalid-dependency 취소 정책에서 monitor까지 취소되는 것을 방지한다. 이미 끝난16개 ID를 새 `afterok`로 다시 연결하지 않으며 완료 확인 ID는 provenance에 남긴다. 중간에 끊겨도 같은 명령으로 이어서 처리할 수 있다.

```bash
python scripts/submit_nfcgs_fresh_rd.py \
  --root outputs/nfcgs_fresh_rd/branches/B120000 \
  --state logs/slurm/fresh_rd/state_B120000.json \
  --cancel-wave2-optional --reason 'Schedule cutoff after the sixteen core runs' --dry-run
```

사용자 계획의1.8일 +3.5일(+선택4개3.5일)은 queue·평가 시간을 제외한 학습 시간 가정이다.11월1일 일정은 실제 시작 시각과 진행 로그로 다시 확인한다.

## 로컬 검증

추가 검사: `tests/test_fresh_rd_120k.py`. 경계 저장 방지, hard link 보존, quantile 중복 갱신 없음, 재저장본 수락·거부 조건, 실제 입력 dry-run 무변경, 제한된 runtime migration과 모델 복원,20개 연속 job 및4개 취소/monitor 연결, TensorBoard LR event 검사를 포함한다. 기존 fresh 검사도 함께 실행한다.

Windows에서 `NFCGS_WINDOWS_REPLACE_RETRY_TESTS=1`을 지정한 검증은 해당 test의 writable 임시 파일에만 공유 위반 재시도를 적용한다. production IO나 읽기 전용 checkpoint 보호는 바꾸지 않는다. Linux/server에서는 적용되지 않는다. 최종 실행 결과와 SHA는 [검증 기록](analysis/nfcgs_120k_2026-10-23/REPORT.md)에 남긴다. 실제 서버 checkpoint와 Slurm을 대상으로 한 preflight는 업로드 후 위 dry-run에서 수행한다.

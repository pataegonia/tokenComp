# Fresh RD 실행 안내 — 2026-10-20

2026-10-23 120k 설계 적용: **재저장본 검증을 포함한 Wave 1 120k 분기와, Wave 2 단일0→120k job은 [새 실행 안내](NFCGS_120K_RUNBOOK_2026-10-23.md)를 따른다.** 아래의 기존 A→B Wave 2 설명은 이전 방식이다. 새 안내에는 보호 hard link, runtime 변경 provenance, 실제 입력 dry-run,30-job 한도 재시도, 시작 후 SHA·LR 확인과 선택4개 취소 절차가 있다.

2026-10-23 job 이름 개선: 새 접수는 `gs-fr-H-on-l0p0128-s0-A60k`, `gs-fr-H-on-l0p0128-s0-B100k`, `gs-fr-H-on-l0p0128-s0-test100k`처럼 모델·λ·seed·단계/목표 step을 표시한다. H/N/M은 Hilbert/None/Morton, on/off는 context, S56/MSH는 비교 모델이다. monitor는 `gs-fr-monitor-w1-A60k` 또는 `gs-fr-monitor-w1-final100k`다.

이미 실행·대기 중인 job도 새 `submit_nfcgs_fresh_rd.py`와 복구 스크립트 두 개를 반영한 뒤 아래 한 번으로 이름만 변경할 수 있다. state에 등록된 내 활성 job의 comment·작업 경로·실행 스크립트를 먼저 확인한다. job ID, 원 sbatch 명령 기록, 의존성, 학습 설정은 유지한다. monitor 복구 코드도 옛 hash 이름과 새 이름을 모두 인식한다. 아직 monitor 코드 변경을 state에 반영하지 않았다면 기존 identity 복구 명령을 먼저 실행한다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
python scripts/submit_nfcgs_fresh_rd.py --rename-active-jobs
squeue --me -o "%.18i %42j %.10T %.10M %R"
```

2026-10-21 수정: **Gate 0를 먼저 통과할 필요 없이 `--wave 1`로 본학습을 바로 접수한다.** 진단 job은 선택 사항이다. 기존 diagnostic-only state와 접수 ID는 보존한 채 새 코드로 갱신한다.

10:32 다운로드에서 처음 제출한 Wave 1의 15개가 모두 계산 노드의 RE10K index 누락으로 학습 시작 전에 실패했다. 이 상태는 아래 일반 Wave 1 명령 대신 [전용 복구 안내](analysis/nfcgs_logs_2026-10-21_1032/REPORT.md)를 따른다. 기존 노드 제외 설정을 복원했으며 복구 도구는 실제 실패 노드도 확인해 피한다.

본문 결과를 λ별 fresh 초기화·1 GPU·유효 batch 8로 다시 만든다. 공개 backbone·QR geometry 경계와 새 minimal codec에서 출발하며 backbone/Gaussian decoder는 고정한다. 같은 seed의 셀과 λ는 공통 초기 tensor를 공유하고 optimizer/scheduler는 각각 새로 시작한다.

## 모델과 진행 순서

| 목적 | 모델 수 |
|---|---:|
| Hilbert/None × D4 Context ON/OFF × 5λ | 20 |
| 중앙 λ의 추가 seed 2개 × 4셀 | 8 |
| Morton+D4 ON, Hilbert+L3 ON | 2 |
| Score56-only 1λ, MSH-only 3λ | 4 |
| H-ON λ0 진단 | 1 |
| 합계 | 35 |

Wave 1 Stage A(15개, 60k) → Gate 1 → Wave 1 Stage B(공통 B) → Gate 2 → Wave 2(20개, A→B) 순서다. Gate 0는 별도 진단이며 학습의 선행 의존성이 아니다. **Wave 2는 Gate 2의 최종 holdout 검토 후 제출한다.** 원안의 6.7일 완료 추정은 다시 산정해야 한다.

35개 모델은 A/B 학습 job 70개와 최종 test job 35개로 나뉜다. Gate 0과 monitor는 별도다. 새 묶음을 16개의 GPU 의존성 체인으로 묶는다. 기존 별도 실험이 쓰는 GPU는 이 상한에 포함되지 않는다.

## 서버에 반영할 파일

같은 상대 경로로 반영한다. ZIP은 만들지 않는다.

- `config/main.yaml`, `globalsplat/main.py`, `globalsplat/model/model_wrapper.py`
- `globalsplat/fresh_rd.py`, `globalsplat/fresh_rd_runtime.py`
- `scripts/prepare_nfcgs_architecture_pilots.py`, `scripts/slurm/prepare_nfcgs_architecture_pilots.slurm`
- `scripts/run_nfcgs_fresh_rd.py`, `scripts/monitor_nfcgs_fresh_rd.py`, `scripts/check_nfcgs_fresh_rd_gate0.py`, `scripts/submit_nfcgs_fresh_rd.py`
- `scripts/slurm/fresh_rd_environment.sh`, `train_nfcgs_fresh_rd.slurm`, `eval_nfcgs_fresh_rd.slurm`, `monitor_nfcgs_fresh_rd.slurm`, `gate0_nfcgs_fresh_rd.slurm`, `prepare_gate0_nfcgs_fresh_rd.slurm`, `submit_nfcgs_fresh_rd.sh`
- 검사·안내: `tests/test_fresh_rd.py`, 이 문서와 원 계획.

기존 기본 설정은 유지한다. 새 경로에서만 quantile 갱신 후 저장, RNG 복원, 저장된 batch 진행 상태 보정, LR 기록을 적용한다. Lightning 2.3의 batch-end checkpoint는 completed batch가 하나 뒤처질 수 있어 이 보정이 필요하다.

## 시작 명령

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
mkdir -p logs/slurm

# 계획만 표시. 접수·state 생성 없음.
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --dry-run

# Gate 0를 기다리지 않고 15개 본학습 + monitor 접수.
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --wave 1
```

60k 학습 15개와 holdout monitor 하나를 접수한다. 기존 Gate 0 ID가 lane에 남아 있어도 학습 afterok 의존성으로 넣지 않는다. 이미 접수된 459887~459892 등의 ID는 그대로 보존하며 진단 재접수·취소를 자동 수행하지 않는다. diagnostic-only state는 코드 변경 이력을 남겨 갱신하므로 state를 삭제할 필요가 없다. 학습이 이미 접수된 state의 runtime 변경은 계속 엄격히 확인한다.

선택 진단은 `--gate0`, 사용자가 선행 검사를 명시적으로 원할 때만 `--wave 1 --require-gate0`다. 기존 checkpoint/초기화/holdout의 간단한 입력 정합 검사는 학습 시작 시 유지한다. Gate 0 허용치를 바꿔 실패를 통과로 표시하지 않는다.

## s1·s2 초기화

Wave 2 전에 완료한다. s0는 기존 초기화를 재사용하며 새 두 초기화의 holdout bytes는 s0와 같아야 한다.

```bash
for s in 1 2; do
  seed=$((111123+s))
  sbatch --exclude=ariel-v7 --gres=gpu:normal:1 \
    --export=ALL,REPO_DIR=$PWD,DATASET_ROOT=/data3/local_datasets/re10k,\
VANILLA_CHECKPOINT=$PWD/checkpoints/pretrained/globalsplat-re10k-32k.ckpt,\
PREPARATION_DIR=$PWD/outputs/nfcgs_architecture_pilots/initialization_fresh_rd_s$s,\
INIT_SEED=$seed,ARMS=two_path56,\
HOLDOUT_REFERENCE=$PWD/outputs/nfcgs_architecture_pilots/initialization_20261019/holdout.json \
    scripts/slurm/prepare_nfcgs_architecture_pilots.slurm
done
```

이 두 prepare job은 sweep 제출 state와 별도다. 기존 파일이 있으면 동일 provenance인지 검증한다. 기본 seed·arms를 생략한 옛 prepare 명령의 동작은 유지한다.

노드 조건은 명령줄에 둔다. 제출 도구가 prepare·Gate 0 Slurm 파일의 SHA-256을 state에 고정하므로, 서버의 파일을 고치면 이후 Gate 2·Wave 2 제출이 거부된다. 준비 계산은 CPU만 사용하며, GPU 요청은 데이터셋이 있는 normal GPU 노드에 배치하기 위한 것이다.

## Gate 1과 Stage B

monitor는 10k~60k checkpoint를 holdout128에서 평가한다. 직접 측정한 이미지 MSE·LPIPS·header 포함 실제 bytes로 `J_h`를 계산하고 40k→60k의 장면별 paired SE를 구한다. 기본 monitor는 본학습 checkpoint만 평가한다. native 및 warm-start 기준 평가는 monitor의 선택 옵션 `--baselines`이며 기본 수렴 판정의 선행 조건이 아니다. warm-start 모델은 이 128장을 학습에 썼을 수 있으므로 fresh 일반화 비교와 구분한다.

```bash
cat outputs/nfcgs_fresh_rd/convergence/gate1.json
column -s, -t outputs/nfcgs_fresh_rd/convergence/holdout_curve.csv | less -S

# holdout 결과로 B를 정한 뒤. 예시는 100k.
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --stage-b --wave 1 --budget 100000
```

Stage B 15개, 각각 afterok test 15개, 최종 holdout monitor 1개를 접수한다. B=120k/150k는 같은 run의 60k full state에서 scheduler milestone을 84k/108k 또는 105k/135k로 바꾼다. 이미 LR이 감소한 checkpoint에서 LR을 올리지 않는다. B는 sweep에 고정한다. λ0만 별도 연장하거나 Stage B 진행 후 B를 바꾸는 탐색은 자동 실행하지 않으며, 기존 결과를 보존하는 별도 분기로 준비해야 한다.

### Gate 1 전 임시 100k 연결 — 2026-10-22 추가

Stage A가 끝날 때마다 대기 시간을 줄이기 위해 Stage B 학습15개를 미리 연결할 수 있다. 이 옵션은 **사용자가 승인한 임시 시작**이며 Gate 1을 생략하지 않는다. 서버 반영 파일은 `scripts/submit_nfcgs_fresh_rd.py` 하나다. 학습·평가 runtime과 현재 Stage A/monitor는 유지한다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --stage-b --wave 1 --budget 100000 --provisional
```

각 Stage B는 자기 Stage A의 `afterok`로만 연결한다. 60k checkpoint에서 full state를 읽어 총100k까지 학습하며 LR milestone은70k/90k다. 큐 접수는 GPU 자리를 보장하지 않는다. 상태에 `provisional_stage_b`와 선제출 시점을 기록하고 공통 최종 `budget`은 미확정으로 유지한다. 임시 접수에서는 test와 최종 monitor를 제출하지 않으므로 Gate 1 결정 전에 test 결과가 생기지 않는다. 같은 명령을 다시 실행하면 접수된 ID를 재사용한다.

일부만 접수되고 sbatch가 거절하면 state를 지우지 않고 같은 명령을 다시 실행한다. 이미 접수한 ID는 `REUSING`으로 유지하고 미접수 작업만 접수한다. 거절 시 실제 stderr/stdout을 콘솔과 state의 `submission_failures`에 기록한다. exit1만으로 quota·fairshare·dependency 등 구체적 사유를 단정하지 않는다.

실제 거절 사유가 `QOSMaxSubmitJobPerUserLimit`이면 실행 및 대기 중 job의 사용자별 접수 한도다. 현재 확인된 묶음은 Stage A15개+monitor1개+접수된 Stage B14개로30개이며 정확한 QoS 설정값은 별도 조회하지 않았다. 기존 작업을 취소하지 않고 빈 접수 자리를 기다리려면 추가 파일 `scripts/wait_nfcgs_provisional_submission.py`를 서버에 반영해 아래처럼 실행한다.

```bash
nohup python scripts/wait_nfcgs_provisional_submission.py --interval 600 \
  > logs/slurm/provisional_submit_wait.log 2>&1 < /dev/null &
tail -n 25 logs/slurm/provisional_submit_wait.log
```

이 프로세스는 로그인 노드에서10분마다 기존 제출 controller를 실행한다. 자체 Slurm/GPU job은 접수하지 않고 controller가 끝나면 submit lock을 반납한 상태로 대기한다. 기존 ID는 재사용하고 QoS 접수 한도 오류만 재시도한다. 다른 오류는 출력 후 종료한다. 남은 작업까지 접수되면 `PROVISIONAL_SUBMISSION_COMPLETE`를 출력하고 종료한다.

Gate 1이100k를 권고하면 아래 명령으로 확정한다. Stage B15개는 그대로 재사용하고, 각 Stage B 뒤의 test15개와 최종 holdout monitor만 추가한다.

```bash
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --stage-b --wave 1 --budget 100000
```

Gate 1이120k를 권고하면 아래 명령으로 공통 budget을 결정한다.

```bash
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --stage-b --wave 1 --budget 120000 --restart-stage-b-from-60k
```

150k를 선택하거나 Gate 1 권고와 다른 budget을 선택하려면 실제 holdout에 근거한 `--reason '선택 이유'`를 함께 지정한다. 완료한15개 holdout 비교와 그 SHA-256을 저장하며 test 결과는 판단에 사용하지 않는다.

연장 절차는 source의40k/60k validation과60k checkpoint를 검증한 뒤 임시 Stage B job만 확인·종료한다. Stage A와 기존 monitor에는 취소 명령을 보내지 않는다. 기존100k 분기의 모든 checkpoint·last·attempt·로그를 원래 위치에 보존한다. 새 분기에는 동일한 Stage A manifest와10k~60k의 불변 checkpoint·완료 평가를 hard link(불가 시 copy)하고, **임시70k+ checkpoint와 last는 가져오지 않는다**. 새 B는 반드시60k에서 full-state 재개하고 milestone을84k/108k 또는105k/135k로 바꾼다.

120k 분기는 다음 위치에 생성한다. 원 state에 `active_budget_branch`로 경로를 기록한다. 이후 Gate 2/Wave 2 명령에는 이 분기의 state와 root를 함께 지정한다.

```text
outputs/nfcgs_fresh_rd/branches/B120000/
logs/slurm/fresh_rd/state_B120000.json
```

```bash
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --gate2 proceed --budget 120000 \
  --state logs/slurm/fresh_rd/state_B120000.json \
  --root outputs/nfcgs_fresh_rd/branches/B120000 --reason '최종 holdout 검토 사유'
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --wave 2 --budget 120000 \
  --state logs/slurm/fresh_rd/state_B120000.json \
  --root outputs/nfcgs_fresh_rd/branches/B120000 --msh-lambdas 0.0064 0.0128
```

중단된 rollback은 같은 명령으로 재개한다. 접수된 새 ID는 재사용하고, 이미 기록한 budget/holdout 근거/Stage A 계보를 바꿀 수 없다. branch에 미확인 sbatch intent가 남으면 그 branch의 state/root를 사용해 기존 `--record-accepted` 절차로 ID를 확인한다. 100k로 이미 확정했거나 test를 접수한 뒤에는 이 rollback 옵션으로 budget을 바꾸지 않는다.

60k→70k에서 LR은100k·120k 설정 모두1e-4지만, 연장 시에는 그 구간도 동일60k에서 다시 학습한다. 버리는 임시 진행량은 Gate 1 완료 시점과 각 job의 실제 진행에 따라 달라진다. 원래 Stage B의 job ID·중단 전후 상태와 원본 출력은 기록으로 남긴다.

## Gate 2와 Wave 2

`convergence/wave1_final_complete.json` 및 `curve_wave1_final.json`을 검토한다. 같은 시점의 test는 보고용이다. 실제 holdout 관찰에 근거한 진행 이유를 기록한다.

```bash
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --gate2 proceed --budget 100000 \
  --reason '실제 holdout 결과를 검토한 진행 사유'

bash scripts/slurm/submit_nfcgs_fresh_rd.sh --wave 2 --budget 100000 \
  --msh-lambdas 0.0064 0.0128
```

보류는 `--gate2 hold --budget ... --reason ...`이다. proceed가 holdout 파일 hash 및 B와 일치해야 제출한다. MSH λ는 `.0016` 같은 추가 비음수 값도 지원한다. 두 값이 같거나 Wave 1 `.0032`와 중복되면 거부한다. 150k에서 seed 8개를 빼기로 명시적으로 결정한 경우에만 `--omit-seeds`를 추가한다. 이 경우 27개가 되며 3-seed 반복 근거가 사라진다.

## 부록 수렴 곡선

```bash
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --monitor appendix --wave 1 --budget 100000
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --monitor appendix --wave 2 --budget 100000
```

각 wave test chain 뒤에 70k~B holdout 평가를 접수한다. 같은 `holdout_curve.csv`에 원자적으로 병합한다. checkpoint 크기·mtime을 두 주기에 걸쳐 확인하고 SHA-256을 기록한다. load 실패·128장 미완료는 다음 주기에 재시도하며 완료로 세지 않는다.

## 저장물과 검증 범위

- state: `logs/slurm/fresh_rd/state.json`. 접수 ID·run 경로를 즉시 저장하고 재실행은 접수된 ID를 재사용한다. 삭제·이동해 중복 제출하지 않는다.
- run: `outputs/nfcgs_fresh_rd/<cell>/lambda<tag>/s<seed>/job_<A_ID>/`. B와 실패 재개도 같은 폴더다.
- 설정: `run.json`에 protocol·초기화/holdout/코드 hash, `attempts/<시각_PID>/`에 명령·실제 Hydra 설정. checkpoint에도 실제 학습 config와 protocol을 저장한다.
- checkpoint: `checkpoints/nfcgs_main/version_0/step000060000.ckpt`, 최종 `step000100000.ckpt` 등과 `last.ckpt`.
- 평가: `validation/step<9자리>/nfcgs_main/scene_metrics.json`, `eval/nfcgs_main/scene_metrics.json`. scene별 bits·PSNR·SSIM·LPIPS·직접 MSE·image별 MSE·frame ID를 보존한다.
- 전체 test는 같은 6991 scene/frame, holdout은 s0 manifest의 같은 128 scene/frame을 요구한다. full-test frame 기준은 기존 native 또는 warm-start 평가가 있으면 재사용하고, 없으면 최초 완료 fresh 평가에서 고정한다. `test_frame_cohort.json`에 출처를 저장하며 품질 수치는 기준 선정에 쓰지 않는다. native 진단을 기다리지 않는다.
- TensorBoard: `tensorboard/nfcgs_main/version_0/`. LR·train loss·estimated rate 기록.
- Slurm: `logs/slurm/slurm-gs-fresh-{train,eval,monitor,gate0,gate0-cpu}-<ID>.out/.err`.

Gate 0는 서버 실제 데이터·모델·GPU로 수행한다. 로컬 CPU 검사는 그 결과를 대신하지 않는다. 최종 fresh 모델의 T05/T06/T15/complexity는 후속 재평가 대상이며 기존 job447746 진단을 새 모델 결과로 재표기하지 않는다. 부록 평가 전에 중간 checkpoint를 지우지 않는다.

## 실패 복구

새 fresh 묶음의 실패한 학습 job에만 명시적으로 사용한다. 종료 상태인지 sacct로 확인하고 같은 run의 last에서 full state 재개한다.

```bash
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --resume-run h_on/lambda0p0128/s0 --stage A
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --resume-run h_on/lambda0p0128/s0 --stage B
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --repair-dependencies
```

pending 연결 job은 새 train ID의 afterok로 바꾼다. 이미 시작·종료한 연결 job은 자동 재제출하지 않고 상태 확인을 요구한다. 실패한 eval·Gate 0·monitor는 저장된 명령과 출력부터 확인해 별도 재개한다.

### Stage A monitor의 lock 충돌 복구 (2026-10-21)

monitor460349의 `.monitor_1_A.lock` 충돌에는 monitor 전용 모드를 사용한다. 서버에 갱신할 파일은 `scripts/recover_nfcgs_fresh_start_20261021.py` 하나다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
python scripts/recover_nfcgs_fresh_start_20261021.py --monitor-only
```

- 현재 monitor가 active 또는 완료 상태면 기존 ID를 재사용한다. 실패 상태일 때만 해당 stderr의 lock 충돌을 확인한다.
- 이전 monitor460326 및 monitor history의 job 상태를 확인한다. 종료가 필요한 작업은 `scontrol`의 job ID·이름·작업 디렉터리·monitor Slurm 경로를 확인한 monitor로 한정한다. 학습 job에는 취소·재접수를 하지 않는다.
- 다른 active monitor가 있거나 검사 중 lock 내용이 바뀌면 lock을 보존하고 중단한다. 이전 monitor의 종료와 다른 monitor 부재를 확인한 경우만 정확한 두 monitor lock을 정리한다.
- 같은 state의 monitor 항목과 lane15만 새 ID로 교체하며 실패한 monitor의 afterok를 연결하지 않는다. 같은 명령의 재실행은 이미 접수한 active monitor를 재사용한다.
- 저장된 10k~60k checkpoint와 이미 완료한 validation을 이어서 처리한다. 새 monitor는 파일 안정성을 두 번 확인하므로 첫 평가 시작까지 기본 polling 한 주기(10분) 이상 걸릴 수 있다.

서버에서 수행하는 선행 GPU 검사는 없다. 이 복구는 학습 runtime·초기화·step 설정·checkpoint를 변경하지 않는다. 로컬의 fresh 및 복구 검사 70개가 통과했으며 실제 서버 접수·취소는 사용자 실행 시 이루어진다.

접수 도중 프로세스가 끊겨 ID 저장 여부가 불명확하면 pending intent로 중복 접수를 차단한다. Slurm에서 intent의 `fresh-rd-...` comment와 일치하는 job을 확인한 뒤 `--record-accepted <ID>`로 복구한다. `scontrol show job`의 comment를 도구가 다시 확인한다. 미확인 intent를 삭제해 재접수하지 않는다.

## 현재 상태

2026-10-23 업데이트:08:53 다운로드에서 Stage A10개 완료·5개 진행, 임시 Stage B15개 접수 및9개 시작을 확인했다. MSH의 B ID는462078이다. monitor460774는74건 뒤 checkpoint identity 변경으로 종료됐다.10월21일 lock 오류와 다른 원인이므로 이전 `--monitor-only` 복구를 이 오류에 사용하지 않는다.

`scripts/monitor_nfcgs_fresh_rd.py`와 `scripts/recover_nfcgs_monitor_identity_20261023.py`를 서버에 반영한 뒤 실행한다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
python scripts/recover_nfcgs_monitor_identity_20261023.py
```

기존 평가를 보존하고 바뀐 checkpoint를 다시 평가하도록 monitor만 재접수한다.60k 원본과 hash 일치 backup의 audit은 `logs/slurm/fresh_rd/checkpoint_identity_audit_<시각>.json`에 남긴다. 학습과 checkpoint는 취소·수정하지 않는다. 원래 Stage A full state와 현재60k 파일이 동일하다고 가정하지 않으며,120k/150k 연장은 audit 결과와 기존 completion hash 검사까지 확인해야 한다. [상세 기록](analysis/nfcgs_logs_2026-10-23_0853/REPORT.md).

10월 21일 20:27 다운로드에서 Wave 1 학습460334~460348 모두 약4.3k~14.2k optimizer step의 진행을 확인했다. 사용자 실행으로 monitor460349가460774로 교체됐고, 새 monitor는 stderr0 byte·완료 평가8개로 진행 중이다. 10k를 넘은 학습8개와 평가 수가 일치하며 재시작할 근거는 없다. 18:27 속도로 추정한 60k 종료는 10월 23일 새벽~정오였으며 실시간 서버 상태는 조회하지 않았다. 이전 실패460311~460325 및 Gate 0 진단은 별도 기록으로 보존한다. 기존 H+D4 OFF16의 학습458783 및 최종 평가457100은 완료됐다.

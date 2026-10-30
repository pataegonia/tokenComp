# Score 정렬 최적화로 이어 학습 — GPU 2개

## 계산 방식과 eval 호환

이번 실행은 `nn_score`, 가중 L1 거리, Morton 최소 key 시작, 원래 slot 번호를 사용하는 동률 처리,
legacy 3단계·kernel5·STE·Score minimal·MSH 설정을 유지한다.
기본 거리 계산은 CPU FP32이며, exact kNN 후보 64개와 Numba greedy 루프를 사용한다.
후보 소진과 동률 경계에서는 전체 행을 검색한다. 근사 경로나 고정 정렬 cache를 추가하지 않는다.
같은 checkpoint·입력에서 기존 CPU 경로와 token 순서 및 실제 bitstream bytes가 같음을 검증했다.
decoder와 bitstream 형식은 같은 코드로 동작한다.

GPU는 학습에 2개를 사용한다. 기본 거리 장치는 CPU이며 `NN_ORDER_DISTANCE_DEVICE=input`으로 GPU 거리 계산도 선택할 수 있다.
GPU에서도 같은 가중 L1 거리와 exact greedy 규칙을 사용한다. CPU/GPU 반올림 차이로 순서가 달라질 수 있다.
최적화만 켜는 것으로 새로운 RD 모델을 만들거나 spatial context를 바꾸지 않는다.
checkpoint에서 이어 학습한 뒤 얻는 모델의 가중치는 학습에 따라 갱신되므로, 그 모델의 eval 성능은 따로 측정한다.

## 실행

서버에 아래 파일들을 반영한 뒤 저장소 root에서 실행한다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
python -m pip install numba

# checkpoint와 설정 확인만 한다. 기존 job 취소·새 제출 없음.
bash scripts/slurm/submit_nfcgs_nn_score_resume.sh --dry-run

# checkpoint 검증 후 기존 Score 학습/eval을 취소하고 이어 학습한다.
bash scripts/slurm/submit_nfcgs_nn_score_resume.sh --replace-jobs 448650,448651
```

helper는 `job_448650/checkpoints/nfcgs_main/version_0`의 가장 높은 번호 checkpoint를 선택한다.
번호 checkpoint가 없을 때만 `last.ckpt`를 확인한다.
optimizer·LR scheduler·step·모델 설정을 먼저 검증하며, 실패하면 기존 job을 취소하지 않는다.
취소 대상으로 지정한 자기 계정의 활성 job만 취소하고, 종료를 확인한 뒤 새 job을 제출한다.
새 학습은 원본 checkpoint를 새 실행 폴더로 복사해 full-state resume하므로 원래 checkpoint를 덮어쓰지 않는다.

마지막 저장 checkpoint부터 **총 50k optimizer step까지** 이어간다. 추가로 50k를 더 학습하는 설정은 아니다.
저장은 5k 간격이라 checkpoint 이후에 진행한 미저장 step은 다시 학습한다.
학습 GPU 2 × batch/GPU 2 × accumulation 2 = 유효배치 8이다.
subset-consistency의 두 입력을 처리하는 방식도 유지한다.
학습 시간 제한은 6일이며, `gpu:normal:2`, CPU 8/task, 메모리 48G/GPU, v7 제외다.

학습 job 이름은 `gs-ns-resume`이고, 후속 전체 scene eval은 `gs-ns-resume-eval`이다.
eval은 새 학습의 `afterok` dependency로 제출하며 GPU 1개를 사용한다.
eval 설정은 C12/T8·전체 scene·실제 bitstream이다.

기본 λ=.0256, LR1e-4 및 35k/45k decay recipe를 사용한다.
resume에서는 저장된 optimizer·scheduler 상태와 global step을 복원한다.
출력은 다음과 같다.

```text
outputs/nfcgs_order_context/nn_score/legacy_stages3_k5_ste_nn_knn_cpu_k64/lambda0p0256/resume_job_<TRAIN_ID>/
  checkpoints/nfcgs_main/version_0/resume.ckpt
  checkpoints/nfcgs_main/version_0/step000050000.ckpt
  eval_all/
```

제출 state는 `logs/slurm/nn_score_resume_448650_<cpu|input>_knn.state`에 저장한다.
같은 명령을 다시 실행하면 제출한 job ID를 재사용한다. 학습 제출 후 eval 제출이 실패했다면 eval만 재시도한다.
로그는 `slurm-gs-ns-resume-<TRAIN_ID>.out/.err` 및 기존 `slurm-gs-order-context-eval-<EVAL_ID>.out/.err` 규칙이다.
기존 job을 실제로 취소하거나 서버에 새 job을 제출한 상태는 아니다. 위 명령을 서버에서 실행해야 한다.

### GPU 거리 계산 선택

```bash
NN_ORDER_DISTANCE_DEVICE=input \
  bash scripts/slurm/submit_nfcgs_nn_score_resume.sh --dry-run
```

위 preview는 job을 취소하거나 제출하지 않는다. 최초 전환을 실행할 때에는 `--dry-run` 대신
`--replace-jobs <현재_학습_JOB>,<현재_eval_JOB>`을 지정한다.
앞의 CPU 명령을 이미 제출했다면 새 CPU job ID와 그 실행의 checkpoint를 기준으로 전환한다.
원래 `448650`의 checkpoint에서 다시 시작하는 것으로 CPU 최적화 job을 자동 교체하지는 않는다.

`input`은 학습·eval에서 CUDA tensor의 거리 계산과 top-k를 GPU에 맡긴다.
greedy 방문 루프와 후보 소진·동률 fallback은 CPU의 compiled 함수가 처리한다.
GPU 전체 정렬 시간은 아직 측정하지 않았다. CPU 실측의 1.89ms는 greedy 루프만의 시간이며
거리 계산·top-k·CPU 전달을 포함한 전체 GPU 정렬 시간이 아니다.
CPU 옵션은 `NN_ORDER_DISTANCE_DEVICE=cpu`다. 두 장치의 output 경로와 제출 state를 따로 사용한다.

특정 checkpoint를 지정하려면 다음처럼 실행한다.

```bash
RESUME_CHECKPOINT=/absolute/path/to/step000010000.ckpt \
  bash scripts/slurm/submit_nfcgs_nn_score_resume.sh --replace-jobs 448650,448651
```

checkpoint/target을 바꿔 새로운 실행을 만들 때는 새 `NN_SCORE_RESUME_STATE_FILE`을 지정한다.
기존 state와 조건이 다르면 중복 제출하지 않고 중단한다.

## 서버 반영 파일

| 파일 | 역할 |
|---|---|
| [token_order.py](../globalsplat/compression/token_order.py) | exact compiled greedy |
| [config.py](../globalsplat/compression/config.py) | backend 설정 |
| [codec.py](../globalsplat/compression/codec.py) | 정렬 호출 |
| [checkpoint.py](../globalsplat/compression/checkpoint.py) | 구현 옵션 변경을 허용하는 strict resume 검증 |
| [globalsplat_nfcgs_rank56.yaml](../config/model/globalsplat_nfcgs_rank56.yaml) | Hydra 설정 |
| [run_nfcgs.py](../scripts/run_nfcgs.py) | CLI 설정 |
| [nfcgs_order_context_settings.sh](../scripts/slurm/nfcgs_order_context_settings.sh) | 공통 codec 옵션 |
| [prepare_nfcgs_nn_score_resume.py](../scripts/prepare_nfcgs_nn_score_resume.py) | full-state 검사와 새 폴더 복사 |
| [submit_nfcgs_nn_score_resume.sh](../scripts/slurm/submit_nfcgs_nn_score_resume.sh) | 기존 job 교체·중복 없는 제출 |
| [train_nfcgs_nn_score_resume.slurm](../scripts/slurm/train_nfcgs_nn_score_resume.slurm) | GPU 2개 이어 학습 |
| [eval_nfcgs_order_context.slurm](../scripts/slurm/eval_nfcgs_order_context.slurm) | 새 resume 폴더에서 전체 scene eval |

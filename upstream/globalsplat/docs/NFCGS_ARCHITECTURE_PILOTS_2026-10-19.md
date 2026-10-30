# 다음 실험: 공통 초기화 구조 비교 pilot

기존 대조군 7개 다음으로 T07/T08/T18을 실행할 수 있게 준비했다. 기존 학습·평가는 취소하거나 재제출하지 않는다. 이번 기본 제출은 **CPU 준비 job 1개 + 학습 3개 + train holdout 평가 3개**다. 새 서버 job은 아직 제출하지 않았다.

## 모델과 학습량

| 계획 ID | Arm | 경로 | 기본 제출 |
|---|---|---|---|
| T07 | `score56_only` | F736 → learned rank56 → D4 Score Context → F736. MSH 없음 | 포함 |
| T08 | `msh_only` | F736 → direct MSH → F736. Score/rank56 경로 없음 | 포함 |
| T18 | `two_path56` | F736 → rank56 + MSH residual, D4 Context | 포함 |
| T09 | `score736_only` | F736 → learned 736→736 transform + D4 Context → F736. MSH 없음 | 선택 |

F는 appearance512와 observable geometry224를 합친 feature다. direct MSH에서 없애는 것은 rank56 경로이며, 공통 geometry512→224 projection은 유지한다.

모든 모델은 Hilbert, minimal, linear, MSH N192/M320/adapter96(해당 branch가 있을 때), 새50k step, GPU2×batch2×누적2, BF16, LR1e-4/35k·45k 감소, seed111123/data seed403이다. 공간 Context는 D4/kernel5/STE다. rank56은 16/16/16/8 그룹, full-width는 184×4 그룹으로 16개 Score strings를 유지한다.

초기 λ는 **.0128 한 점**이다. 같은 λ가 구조마다 같은 rate를 보장하지 않으므로 이번 결과는 선별용이다. 추가 λ의 학습은 이 제출에 포함하지 않는다. T18은 원래 권장안의 기준선이지만, 단일 경로와 공정하게 비교하려고 이번 세 모델에 함께 배치했다.

## 서버 반영과 기본 실행

로컬 `upstream/globalsplat/`의 수정·추가된 개별 파일을 서버의 `/ceph_data/clue9986/tokencomp/upstream/globalsplat/`에 같은 상대 경로로 반영한다. ZIP 업로드나 압축 해제 단계는 없다. 실행에 필요한 파일 22개는 [반영 파일 목록](nfcgs_architecture_pilots_20261019_files.txt)에 정리했다. 모델·checkpoint 로더·평가 hook·dataset 변경도 필요하므로 새 launcher만 단독으로 반영하면 안 된다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
if bash scripts/slurm/submit_nfcgs_architecture_pilots.sh --dry-run; then
    bash scripts/slurm/submit_nfcgs_architecture_pilots.sh
fi
```

기본 원본 가중치 경로는 `checkpoints/pretrained/globalsplat-re10k-32k.ckpt`, 데이터는 `/data3/local_datasets/re10k`다. 다른 위치에 있다면 제출 명령 앞에서 `VANILLA_CHECKPOINT`/`DATASET_ROOT`를 지정한다. 기존 FullSplit/학습된 codec checkpoint를 원본 가중치 대신 넣지 않는다.

`--dry-run`은 필수 파일·원본 checkpoint·저장된 state와 코드 signature를 확인하고 제출 계획만 출력한다. `/data3/local_datasets/re10k`는 계산 노드의 로컬 경로이므로 로그인 노드에서는 데이터 index의 존재를 검사하지 않는다. 실제 train/test index와 chunk는 **계산 노드의 CPU 준비 job**에서 검사한다. 준비에 실패하면 `afterok` 의존성으로 연결한 학습은 시작하지 않는다. 공통 초기화 checkpoint와 holdout JSON도 준비 job이 새로 생성하므로 미리 서버에 만들 필요가 없다.

## 자동 실행 흐름

1. CPU 준비 job: train에서 검증 장면128개와 C12/T8 프레임을 고정하고, 공통 초기화 checkpoint 네 종류를 준비한다. 네 번째 초기화 파일 생성은 GPU 학습 추가를 뜻하지 않는다.
2. 준비 job 성공 후 각 train 시작. 할당된 GPU에서 4096-token BF16 codec/독립 receiver/gradient smoke를 먼저 실행한다. 실패하면 50k 학습을 시작하지 않는다.
3. 각 train 성공 후 해당 최종50k checkpoint를 **train holdout128장**에서 actual encode→decode→render로 평가한다. 128장이 모두 평가됐는지 확인한다.

원본 GlobalSplat backbone과 QR로 축소한 GaussianDecoder는 고정하고, geometry projection은 모든 구조에서 동일하게 학습한다. codec은 모두 새로 초기화하며, 호환되는 rank56 transform·확률 모델 및 MSH tensor를 기준선과 정확히 공유한다. 초기 tensor 해시·원본 checkpoint 해시·검증 manifest 해시를 저장하고 이후 학습 checkpoint에도 provenance를 보존한다.

새 mode는 `two_path`/`score_only`/`msh_only`로 명시한다. Score-only에는 MSH 모듈·payload가 없고, MSH-only에는 Score/low-rank 모듈·payload가 없다. 별도 packet flag로 식별하고 잘못된 branch payload나 mode가 섞인 checkpoint를 거부한다. 기존 two-path의 default 가중치·bitstream·학습 동작은 회귀 검사에서 이전 구현과 동일했다.

## 검증 데이터와 전체 test

128장은 train index에서 test scene ID와 겹치지 않는 후보를 hash 순서로 고른다. 프레임 수·FOV·고정 프레임의 이미지 shape를 확인하고 C12/T8 카메라 위치 기반 FPS 프레임을 저장한다. 이 장면들은 새 구조 네 모델의 학습에서 view sampling 전에 제외한다. 기존 대조군·진행 중인 실험의 데이터 구성은 변경하지 않는다.

현재 기본 `val`은 test split을 읽으므로 이 pilot 학습에서는 비활성화한다. 별도 `validation` 명령은 명시된 **train chunk**와 고정 index를 읽는다. 합성 RE10K chunk로 학습 제외·실제 train chunk 평가·프레임 일치를 검사했다. 실제 서버 128장 manifest는 준비 job에서 생성될 예정이다.

기본 제출에는 전체 test가 들어 있지 않다. holdout 결과를 보고 현재 pilot을 최종 test로 평가하기로 정하면 다음 명령으로 기존 train/validation ID를 재사용하고 test job만 추가할 수 있다.

```bash
bash scripts/slurm/submit_nfcgs_architecture_pilots.sh --with-test
```

이는 현재 우리 프로토콜인 C12/T8 전체 test 평가다. 원본 GlobalSplat의 C3G C12/T3 프로토콜이나 외부 codec의 평가 설정으로 전환하는 명령은 아니다. λ를 변경한 새 학습이나 새로운 RD 곡선 확장은 별도 계획으로 준비한다.

## Full-width 선택 항목

T09까지 함께 실행하려면 `--include-fullwidth`를 사용한다. 학습1개와 holdout 평가1개가 추가된다. 이미 기본 세 모델을 제출한 뒤 같은 옵션을 붙여 재실행해도 세 모델을 중복 제출하지 않는다.

기존7개 접수 후 T09만 확장하는 절차와 확인 항목은 [T09 실행 안내](NFCGS_FULLWIDTH_CONTEXT_PILOT_2026-10-19.md)에 정리했다.

```bash
bash scripts/slurm/submit_nfcgs_architecture_pilots.sh --include-fullwidth
```

rank736 모델은 symbol 수·predictor parameter 수가 커진다. rank56과 연산·용량을 맞춘 순수 차원 비교로 해석하지 않는다. GPU에서 실제 학습·평가 시간과 메모리를 확인한다.

## QoS 제한, 상태와 출력

제출 제한으로 끊기면 용량이 생긴 뒤 같은 명령을 다시 실행한다. 모든 접수 ID는 다음 job 제출 전에 원자적으로 저장한다. 이미 접수된 CPU 준비·train·validation·test는 재사용한다. 이 기능은 제출 재시도이며 실패/취소 job의 자동 resume가 아니다. 이미 접수한 뒤 runtime 코드·원본 가중치·데이터 경로를 바꾸면 혼합을 막기 위해 제출을 중단한다.

```bash
bash scripts/slurm/submit_nfcgs_architecture_pilots.sh
```

| 항목 | 기본 위치 |
|---|---|
| 제출 ID/state | `logs/slurm/architecture_pilots_20261019/state.json` |
| 공통 초기화·holdout·프레임 목록 | `outputs/nfcgs_architecture_pilots/initialization_20261019/` |
| 학습 출력 | `outputs/nfcgs_architecture_pilots/lambda0p0128/<arm>/job_<TRAIN_ID>/` |
| 최종 가중치 | 위 출력의 `checkpoints/nfcgs_main/version_0/step000050000.ckpt` |
| 검증 지표·실제 bytes | 위 출력의 `eval_validation/nfcgs_main/` |
| 선택한 전체 test | 위 출력의 `eval_test/nfcgs_main/` |
| 준비 로그 | `logs/slurm/slurm-gs-arch-prepare-<JOB_ID>.out/.err` |
| 학습 로그 | `logs/slurm/slurm-gs-arch-train-<JOB_ID>.out/.err` |
| 검증/test 로그 | `logs/slurm/slurm-gs-arch-eval-<JOB_ID>.out/.err` |

## 확인한 범위

신규 구조/준비/제출 검사13개와 기존 codec·정렬·OFF·가중치 로드·CLI 회귀47개, 총60개를 통과했다. 네 모델의 train/validation/test Hydra 설정12개, BF16/4096-token의 기본 세 모델 실제 압축·복호·별도 프로세스 수신·backward(CPU), full-width의 16 strings, 공통 tensor 초기화와 train holdout 분리를 확인했다. Bash 네 파일의 문법과 LF도 검사했다. 로그인 노드에 데이터 index가 없는 경우와 기존7개 job 접수 후 T09 확장 시2개만 추가하는 회귀 검사를 포함해 제출 검사4개를 통과했다.

실제 공개 원본 checkpoint·서버 데이터128장으로의 준비, CUDA smoke, 전체 모델 GPU 학습·렌더 평가는 서버 실행으로 확인해야 한다. 로컬 준비 통합 검사는 실제 736채널 codec/224×512 QR 경계와 작은 가짜 backbone·합성 RE10K 데이터를 사용했다. 관련 파일 해시는 [검증 기록](analysis/nfcgs_architecture_pilots_20261019/validation.json)에 보존한다.

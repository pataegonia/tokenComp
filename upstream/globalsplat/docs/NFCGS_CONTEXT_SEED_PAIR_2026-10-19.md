# T11: Hilbert+dyadic4 Context ON/OFF16 추가 seed 비교

구조 pilot T07/T08/T18 다음으로 준비한 비교다. 공간 Context 효과가 다른 학습 seed에서도 유지되는지 확인한다. 기존 seed111123/data403의 ON16/OFF16과 새 seed111124/data404의 ON16/OFF16 차이를 비교한다. 기존 ON은 확보된 Hilbert+D4 결과를 쓰고, 기존 OFF16은 대조군 7개 묶음의 T10 결과가 필요하다. T10의 서버 제출·완료는 아직 확인하지 못했다.

## 실험과 제출 수

| 항목 | T11 ON16 | T11 OFF16 |
|---|---|---|
| 정렬 / token 단계 | Hilbert / dyadic4 | Hilbert / dyadic4 |
| Score 공간 예측 | ON | OFF |
| 채널 그룹 / Score entropy strings | 16/16/16/8 / 16개 | 16/16/16/8 / 16개 |
| rank56 / MSH residual | 모두 유지 | 모두 유지 |
| λ / 학습량 / GPU | .0128 / 새50k / 2개 | .0128 / 새50k / 2개 |
| model seed / data seed | 111124 / 404 | 111124 / 404 |

**총 4개 job: 학습 2개 + 각 학습 성공 후 전체 test 평가 2개.** 기존 FullSplit 부모가 있으므로 별도 prepare job은 없다. 두 학습은 서로 독립적으로 GPU 배정을 받는다. 평가 하나는 GPU 1개를 쓴다.

OFF16은 token 4단계와 채널 4그룹을 유지한다. OFF1 또는 OFF4로 바꾸지 않는다. 두 모델 모두 MSH를 유지하며 minimal/linear/kernel5/STE를 사용한다.

## 기존 결과와 맞추는 조건

- 부모는 기존 FullSplit λ=.0256 probability10k의 `step000010000.ckpt`다. 두 arm에 같은 부모 가중치를 로드하고 optimizer는 새로 시작한다. 특정 arm만 학습된 D4 모델에서 시작하지 않는다.
- backbone과 GaussianDecoder는 기존처럼 고정하고 codec 전체를 학습한다. GPU2×batch2×누적2=effective batch8, BF16, Adam LR1e-4, 35k/45k LR 감소, 5k마다 저장한다.
- 새 seed는 부모 학습 전체를 다시 수행한다는 뜻이 아니다. 같은 부모에서 시작하는 후속 학습의 model/data seed 반복이다. 공통 초기화를 새로 만드는 구조 pilot T18과 구분한다.
- 기존 대조군과 같은 전체 train split과 view sampling을 유지한다. 구조 pilot의 train holdout128 프로토콜은 적용하지 않는다.
- 최종50k checkpoint를 기존과 같은 C12/T8, eval seed0, actual encode→decode→render, 전체6991 scene으로 평가한다. 평가 전 저장된 실제 train Hydra config에서 seed·단계·그룹·공간 Context 설정을 확인하고, 최종 step과 평가 scene 수를 검사한다.
- 한 λ의 두 seed에서 ON/OFF 변화가 반복되는지 보는 실험이다. 이 결과만으로 전체 RD 곡선이나 seed 변화에 대한 신뢰구간을 주장하지 않는다. 기본 순서의 새 seed 쌍을 포함하지 않으므로 정렬×Context 상호작용 전체의 반복 검증도 아니다.

## 서버에 반영할 개별 파일

모두 로컬 `upstream/globalsplat/` 아래에 있다. 서버의 같은 상대 경로에 다음 신규 파일 5개를 반영한다.

```text
scripts/run_nfcgs_context_seed_pair.py
scripts/submit_nfcgs_context_seed_pair.py
scripts/slurm/submit_nfcgs_context_seed_pair.sh
scripts/slurm/train_nfcgs_context_seed_pair.slurm
scripts/slurm/eval_nfcgs_context_seed_pair.slurm
```

기존 공통 runner·codec·dataset 파일은 변경하지 않았다. 실행에 필요한 기존 파일이 빠졌으면 제출 전 검사에서 해당 경로를 출력한다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
if bash scripts/slurm/submit_nfcgs_context_seed_pair.sh --dry-run; then
    bash scripts/slurm/submit_nfcgs_context_seed_pair.sh
fi
```

기본 데이터 경로는 `/data3/local_datasets/re10k`다. 계산 노드의 로컬 경로이므로 로그인 노드에서는 index 존재 여부를 검사하지 않고, 각 GPU job에서 검사한다. 부모가 다른 위치에 있으면 `SOURCE_CHECKPOINT`를, 데이터 경로가 다르면 `DATASET_ROOT`를 지정한다.

## QoS 재시도와 결과

모든 접수 ID는 다음 제출 전에 저장한다. QoS 제출 제한으로 끊기면 같은 명령을 다시 실행해 미접수 job만 추가한다. 이미 접수한 뒤 부모·코드·경로가 달라지면 기존 state를 재사용하지 못하도록 중단한다. 실패하거나 취소된 학습을 자동 resume하는 기능은 아니다.

```bash
bash scripts/slurm/submit_nfcgs_context_seed_pair.sh
```

| 항목 | 경로 |
|---|---|
| 제출 ID/state | `logs/slurm/context_seed_pair_20261019/state.json` |
| 학습 출력 RUN | `outputs/nfcgs_context_seed_pair/seed111124_data404/<on16 또는 off16>/job_<TRAIN_ID>/` |
| 실제 train config | `RUN/hydra/.hydra/config.yaml` |
| 최종 checkpoint | `RUN/checkpoints/nfcgs_main/version_0/step000050000.ckpt` |
| 전체 test 결과 | `RUN/eval/nfcgs_main/` |
| 학습 로그 | `logs/slurm/slurm-gs-seed-pair-train-<ID>.out/.err` |
| 평가 로그 | `logs/slurm/slurm-gs-seed-pair-eval-<ID>.out/.err` |

이번 준비에서 원격 job을 제출하거나 취소하지 않았다. 로컬 검증 범위와 파일 해시는 `docs/analysis/nfcgs_context_seed_pair_20261019/validation.json`에 기록한다. 실제 서버 GPU 학습과 전체 평가는 실행 후 로그로 확인한다.

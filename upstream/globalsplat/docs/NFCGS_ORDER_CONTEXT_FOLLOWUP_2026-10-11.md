# 추가 학습 실험 — 2026-10-11 (10월 13일 XYZ 제외)

현재 제출 목록은 Morton quarter2의 λ 3개와 nn_score 정렬 탐색 1개다. 기존 Hilbert quarter2/dyadic4 작업과 출력 경로 및 job ID를 공유하지 않는다.
처음에는 nn_xyz도 포함한 5개였으나, 10월 13일 사용자 요청에 따라 후속 제출 목록에서 제외했다.
서버 취소 대상은 nn_xyz 학습 `448652`, 연결 eval `448653`이다. 이 문서 수정으로 서버 job이 취소되는 것은 아니다.

```bash
scancel 448652 448653
```

| 정렬 | context | 단계 | λ | 학습 job name | eval job name |
|---|---|---:|---:|---|---|
| Morton | quarter2 | 2 | .0064 | gs-oc-mq2-0p0064 | gs-oc-e-mq2-0p0064 |
| Morton | quarter2 | 2 | .0128 | gs-oc-mq2-0p0128 | gs-oc-e-mq2-0p0128 |
| Morton | quarter2 | 2 | .0256 | gs-oc-mq2-0p0256 | gs-oc-e-mq2-0p0256 |
| nn_score | legacy | 3 | .0256 | gs-oc-ns3-0p0256 | gs-oc-e-ns3-0p0256 |

## 목적과 조건

Morton quarter2는 이미 제출한 Hilbert quarter2와 같은 context에서 정렬 효과를 3-point로 비교한다. nn_score는 양자화 전 Score의 가중 L1 최근접 경로를 사용한다. nn_score는 λ=.0256 한 point로 탐색하며 완성된 RD 곡선으로 해석하지 않는다.

nn_score는 현재 구현의 기본 scene별 scale을 사용한다. TRAIN 고정 scale 파일은 지정하지 않는다. 정렬은 encoder에서 결정하고 Score와 MSH에 동일한 순열을 적용한다. Greedy NN 구현은 O(N²) 거리 행렬을 쓰므로 encoder 시간과 실제 학습 속도를 함께 확인한다.

- 모든 point의 부모: 기존 FullSplit rank56, residual_on, λ=.0256, step10k. .0064도 같은 부모에서 시작한다.
- GlobalSplat 고정, codec 전체 학습, MSH 유지.
- Score minimal: centering·score norm·mean/channel context·nonlinear 제거, spatial context 유지.
- STE, kernel5. Quarter2는 직접 anchor gather를 사용하므로 kernel5는 호환용 표기다.
- 학습당 normal GPU 2개, batch/GPU2, accumulation2, 유효배치8.
- 50k optimizer steps, LR1e-4, 35k/45k decay, checkpoint 5k마다.
- 현재 목록의 학습 4개는 서로 독립적이다. 4개 동시 학습 시 최대 8 GPU를 사용하며 Slurm QOS와 자원에 따라 대기할 수 있다.
- 각 학습의 성공 종료를 기다리는 `afterok` full-scene eval 1개씩, eval당 normal GPU 1개. 현재 목록을 새로 제출하면 학습 4개 + eval 4개 = 8 Slurm jobs다.
- 학습 및 eval의 노드 선택은 기존 `train_nfcgs_order_context.slurm`과 `eval_nfcgs_order_context.slurm` 설정을 사용한다.

## 서버에 반영할 파일

기존 정렬/context 코드와 Slurm 파일이 서버에 반영돼 있다는 기준으로 아래 두 파일만 갱신한다.

1. `scripts/slurm/submit_nfcgs_order_context.sh`
2. `scripts/slurm/submit_nfcgs_order_context_followup.sh`

## 실행

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat

# 현재 4개 point 설정과 부모 checkpoint를 확인한다. job 제출·상태 파일 생성 없음.
bash scripts/slurm/submit_nfcgs_order_context_followup.sh --dry-run

# 기록된 job ID를 재사용하고, 아직 제출되지 않은 job만 추가한다.
bash scripts/slurm/submit_nfcgs_order_context_followup.sh
```

부모의 경로가 변경됐다면 `SOURCE_CHECKPOINT=/absolute/parent.ckpt`를 앞에 지정한다. 전용 helper는 위의 4개 설정과 학습량을 고정하므로 이전 shell에서 남은 TOKEN_ORDER/STAGES/KERNEL/RD_CONFIGS 등의 값에 영향을 받지 않는다.

## 제출 중단 후 이어서 실행

동일한 명령을 다시 실행하면 저장된 job ID를 재사용하고 아직 제출되지 않은 학습·eval만 추가한다. 학습 직후 ID를 먼저 저장하므로 후속 eval 제출이 거절되어도 학습을 중복 제출하지 않는다. 같은 sweep의 동시 제출은 lock으로 막고, 독립 학습 사이에는 dependency를 걸지 않는다.

상태와 제출 요약은 저장소의 다음 경로에 저장한다.

```text
logs/slurm/order_context_followup_20261011/<order>_<schedule>_<lambda>.state
logs/slurm/order_context_followup_20261011/submissions.tsv
```

상태 파일은 설정·부모 경로의 fingerprint, 학습 job ID, eval job ID를 기록한다. 부모 또는 설정이 바뀌거나 상태 파일이 손상되면 job을 제출하지 않고 중단한다. 기존 상태가 있는 point는 완료/실패 여부와 관계없이 ID를 재사용한다. 실패한 학습을 재학습하려는 경우에는 실패 원인을 확인한 뒤 별도의 `FOLLOWUP_STATE_DIR`로 새 실험을 제출한다.

상태 디렉터리를 바꾸면 기존 제출을 재사용할 수 없으므로, 새 cycle을 의도할 때만 사용한다. 제출 프로세스가 강제 종료되어 `.submit-lock`이 남은 경우에는 해당 제출 프로세스가 종료됐는지 확인한 뒤 빈 lock 디렉터리를 제거하고 다시 실행한다.

출력은 기존 규칙을 따른다.

```text
outputs/nfcgs_order_context/<order>/<schedule>_stages<N>_k5_ste/lambda<tag>/job_<trainID>
```

최종 checkpoint는 `checkpoints/nfcgs_main/version_0/step000050000.ckpt`, eval 출력은 `eval_all`이다. 로그 파일 이름은 기존 `slurm-gs-order-context-<ID>.out/.err`와 `slurm-gs-order-context-eval-<ID>.out/.err`다.

로컬에서 파일과 제출 흐름을 준비했으며 서버에는 job을 제출하지 않았다.

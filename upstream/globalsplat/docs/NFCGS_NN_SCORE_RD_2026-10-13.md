# NN-score RD 추가 두 point — 2026-10-13

기존 λ=.0256에 λ=.0064와 .0128을 추가한다.
두 추가 학습은 기존 .0256 NN-score 실험을 시작했던 FullSplit .0256 step10k 가중치에서 독립적으로 시작한다.
현재 .0256 NN-score checkpoint의 optimizer나 학습 step을 새 λ로 이어받지는 않는다.

| λ | training job name | eval job name | 학습 |
|---|---|---|---|
| .0064 | `gs-ns-0p0064` | `gs-ns-e-0p0064` | 같은 FullSplit 부모에서 50k |
| .0128 | `gs-ns-0p0128` | `gs-ns-e-0p0128` | 같은 FullSplit 부모에서 50k |

## 공통 설정

- NN-score, 56채널 가중 L1, exact kNN 후보 64개, 후보 소진·동률 경계 전체 검색.
- CUDA에서 거리 계산·top-k, CPU에서 Numba greedy 루프.
- legacy 3단계, kernel5, STE, Score minimal, MSH 유지.
- GlobalSplat 고정, codec 전체 학습, LR1e-4, 35k/45k decay.
- 학습별 `gpu:normal:2`, batch/GPU2, accumulate2, 유효배치8.
- 50k optimizer steps, checkpoint 5k 간격, 시간 제한 6일.
- 각 학습의 성공 종료 후 `afterok` 전체 scene 실제 bitstream eval, C12/T8, GPU1개.
- 두 학습은 독립적으로 제출한다. 동시에 실행하면 학습 GPU는 총 4개를 사용한다.

CUDA와 CPU의 거리 누적 반올림으로 매우 가까운 이웃의 선택 순서가 달라질 수 있지만
거리 정의·greedy 규칙·context·복호 구조는 동일하다. GPU 거리 계산의 전체 속도는 아직 측정하지 않았다.

## 실행

기존 정렬 최적화 코드·모델 YAML·CLI·공통 Slurm 설정이 서버에 반영돼 있어야 한다.
이번 추가 제출용으로 아래 두 파일을 갱신한다.

- [submit_nfcgs_nn_score_rd.sh](../scripts/slurm/submit_nfcgs_nn_score_rd.sh)
- [submit_nfcgs_order_context.sh](../scripts/slurm/submit_nfcgs_order_context.sh)

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
python -m pip install numba
bash scripts/slurm/submit_nfcgs_nn_score_rd.sh
```

확인만 하려면 마지막 명령에 `--dry-run`을 붙인다. 두 point의 부모·설정·state를 모두 확인한 뒤 제출한다.
추가 제출량은 학습 2개 + 후속 eval 2개 = 총 4 Slurm jobs다.
.0256은 이 helper의 제출 목록에 포함되지 않는다.

기본 부모는 다음 파일이다.

```text
outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt
```

부모 경로가 변경됐다면 `SOURCE_CHECKPOINT=/absolute/path/to/parent.ckpt`를 지정한다.
세 λ의 초기화를 맞추려면 .0256 실험의 원래 FullSplit 부모와 같은 가중치를 사용한다.

출력과 상태는 다음 위치에 저장한다.

```text
outputs/nfcgs_order_context/nn_score/legacy_stages3_k5_ste_nn_knn_input_k64/lambda<λtag>/job_<TRAIN_ID>/
  checkpoints/nfcgs_main/version_0/step000050000.ckpt
  eval_all/

logs/slurm/nn_score_rd_gpu_20261013/
  nn_score_legacy3_0p0064.state
  nn_score_legacy3_0p0128.state
  submissions.tsv
```

다시 같은 명령을 실행하면 기존 ID를 재사용한다. 학습 제출 후 eval만 실패했다면 해당 eval만 재시도한다.
`.submit-lock`으로 같은 helper의 동시 제출을 막는다. 이 helper는 기존 job을 취소하지 않는다.
기존 resume job의 `EVAL_RUN_DIR`가 shell에 남아 있어도 새 λ들은 각자의 실행 폴더에서 평가한다.

로컬에서 제출·dependency·중복 방지 흐름을 검증했다. 서버 job을 실제로 제출한 상태는 아니다.

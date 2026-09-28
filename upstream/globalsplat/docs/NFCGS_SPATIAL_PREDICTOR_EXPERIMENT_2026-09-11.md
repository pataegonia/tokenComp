# Spatial predictor P0/P1/P2 experiment — 2026-09-11

2026-09-12 업데이트: 두 λ 모두 Full을 공통 기본 구조로 비교하기 위해
λ=.0256 Full P0/P1/P2를 task 6–8로 추가했다. 사용자는 기존 task 0–5를 실행 중이라고
알렸으며, 이번 작업에서 서버 queue를 조회하거나 새 job을 제출하지는 않았다.
기존 task 번호/경로는 보존하고 기본 제출 대상을 새 task 6–8로 변경했다.

## 검증 질문과 구조

| Arm | `score_spatial_predictor` | 보정 branch | 역할 |
|---|---|---|---|
| P0 | `linear` | 없음 | 동일 추가 학습 대조군 |
| P1 | `residual3` | Conv3 → GELU → Conv1×1 | 기존과 같은 ±1 anchor 범위의 비선형성 |
| P2 | `residual7` | Conv7 → GELU → Conv1×1 | ±1/±3 anchor까지 넓힌 Morton 문맥 |

P1/P2는 기존 `spatial_predictors.*` key와 출력을 유지한 채
`spatial_corrections.*`를 더한다. hidden width는 32이고 마지막 Conv의 weight와
bias를 0으로 초기화하므로 warm-start 직후 출력은 P0 부모와 같다. 입력은 복호된
even anchor와 decoder가 이미 계산한 base의 차이뿐이며, odd 위치는 0으로 둔다.
따라서 encoder-only feature나 미복호 odd score는 사용하지 않는다.

P0 bitstream flag는 기존과 동일하다. P1/P2는 `SCCTX001` flags에 predictor 종류를
기록하므로 다른 variant codec으로 복호하려 하면 실패한다. checkpoint metadata에는
variant와 hidden width가 저장되며, old checkpoint에서 두 key가 없으면 P0/32로
해석한다. predictor tensor의 누락·추가·shape 불일치도 eval 전 검사한다.

## 아홉 조건과 부모

기존 0/2/4는 λ=.0064의 완료된 Nonlinear Full task8, 1/3/5는 λ=.0256의 완료된
Nonlinear Spatial task7을 부모로 사용한다. 새 6/7/8은 λ=.0256의 완료된
Nonlinear Full task9을 부모로 사용한다. Parent task 번호는 nonlinear context grid의
번호이며, 이 표의 spatial predictor task 번호와 구분한다.

| Task | Arm | λ | Context | Parent task |
|---:|---|---:|---|---:|
| 0 | P0 | .0064 | Full | 8 |
| 1 | P0 | .0256 | Mean+Spatial | 7 |
| 2 | P1 | .0064 | Full | 8 |
| 3 | P1 | .0256 | Mean+Spatial | 7 |
| 4 | P2 | .0064 | Full | 8 |
| 5 | P2 | .0256 | Mean+Spatial | 7 |
| 6 | P0 | .0256 | Full | 9 |
| 7 | P1 | .0256 | Full | 9 |
| 8 | P2 | .0256 | Full | 9 |

Full 공통 구조의 비교는 `.0064: 0/2/4`, `.0256: 6/7/8`이다.
기존 `.0256 Spatial: 1/3/5`도 같은 λ의 Full과 비교할 수 있다.

기본 parent run은
`outputs/nfcgs_nonlinear_score_context/20260910_092000`이다. 각 조건은 해당
`step000050000.ckpt`의 모델 가중치를 복사하고 optimizer/scheduler는 새로 시작한다.
Full 부모의 학습된 group별 EntropyBottleneck parameter를 그대로 복사하며, 새로
channel context를 추가할 때만 factorized entropy를 group별로 나누는 기존 동작을
사용한다. CDF buffer 처리 방식은 기존 initializer와 같다.

새 task 6–8의 공통 서버 parent checkpoint:

```text
/ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_nonlinear_score_context_joint_rank56_lambda0p0256_residual_on_morton_on_m1c1s1/version_0/step000050000.ckpt
```

모든 arm은 추가 optimizer step 25,000, seed 111123, micro-batch2 × accumulate4,
paper24 subset consistency, codec-wide joint training, residual ON, Morton ON,
nonlinear32, bf16-mixed, worker8을 사용한다. task마다 GPU 1개이며 DDP로 batch를
키우지 않는다.

학습 recipe도 기존 0–5와 동일하게 유지했다. 현재 기본 LR은 1e-4이고 기존
milestone이 35k/45k라서 25k 동안 LR이 감소하지 않는다. 이번 Full 추가 작업에서는
스케줄을 변경하지 않았다. 저학습률 마무리는 학습 곡선을 확인한 뒤 비교할 후보에
같은 조건으로 적용할 별도 후속 작업이다.

## 로컬 검증

- P1/P2 zero-init 출력과 P0 출력의 exact equality
- 홀수·짝수 token 길이에서 actual entropy bitstream roundtrip
- P1/P2 bitstream variant mismatch 거부
- odd base를 바꿔도 prediction이 변하지 않는 decoder-causality 검사
- 0-init 마지막 층과 다음 optimizer step 뒤 첫 Conv의 gradient 검사
- checkpoint save/load, old P0 default, metadata/state mismatch 검사
- 아홉 task의 parent/context/λ/25k/output 경로 검사
- 새 Full task의 실제 공통 train/eval runner 경로 일치 검사
- Full/nonlinear P1/P2의 학습된 channel predictor를 포함한 홀수·짝수 길이 roundtrip
- mock sbatch로 기본 6–8만 제출하고 aftercorr로 연결하는지 검사
- 선택한 Full parent가 없을 때 다른 parent로 대체하거나 job을 제출하지 않는지 검사
- 새/공통 shell script `bash -n` 및 LF/BOM 검사

2026-09-12 검증: `tests/test_spatial_predictor_slurm.py`와
`tests/test_score_context_codec.py`에서 58개가 통과했다. 이전 전체 회귀 검증은
180 passed / 1 deselected였으며, 제외한 것은 기존 dirty 상태의 transform eval
노드 고정 설정과 테스트 기대값이 모순되는 1개다. 이번에는 변경 범위의 테스트를 실행했다.

## 서버에서 제출할 때

먼저 mapping만 확인한다.

```bash
bash scripts/slurm/train_nfcgs_spatial_predictor.slurm plan
bash scripts/slurm/eval_nfcgs_spatial_predictor.slurm plan
TASKS=6-8 bash scripts/slurm/submit_nfcgs_spatial_predictor.sh dry-run
```

train/eval의 `plan`은 전체 9조건을 표시한다. submitter의 `dry-run`은 선택한 task와
parent 경로만 표시하며 checkpoint 파일 확인이나 실제 제출을 하지 않는다.
실제 submitter는 선택한 task에 필요한 parent 파일만 확인한다.

아래 네 파일을 서버 repository의 같은 상대 경로로 반영한다. Full 모델 경로는 기존
P1/P2 구현이 이미 지원하므로, 앞서 모델 파일들을 반영했다면 추가 모델 변경은 없다.

- `scripts/slurm/nfcgs_spatial_predictor_grid.sh`
- `scripts/slurm/train_nfcgs_spatial_predictor.slurm`
- `scripts/slurm/eval_nfcgs_spatial_predictor.slurm`
- `scripts/slurm/submit_nfcgs_spatial_predictor.sh`

GlobalSplat repository root에서 다음 명령으로 새 Full 3개 training과 task별
`aftercorr` eval을 함께 제출한다. 기본 TASKS도 6–8이며, 기존 0–5는 재제출하지 않는다.

```bash
TASKS=6-8 bash scripts/slurm/submit_nfcgs_spatial_predictor.sh
```

submitter와 두 SBATCH header 모두 `ariel-v12`, `gpu:1`, array 6–8을 기본으로 사용한다.
submitter가 생성한 새 timestamp run에 저장한다. 같은 run tag를 명시하더라도
`.0256 Full/m1c1s1`과 기존 `.0256 Spatial/m1c0s1` 경로는 구분된다.

```text
outputs/nfcgs_spatial_predictor25k/<run>/{p0_linear,p1_residual3,p2_residual7}/...
outputs/nfcgs_spatial_predictor25k_eval/<run>/{p0_linear,p1_residual3,p2_residual7}/...
```

결과 비교는 각 λ에서 P0/P1/P2의 actual bytes, PSNR/SSIM/LPIPS, 수신측 시간을
사용한다. 접전이면 비교 대상 arm을 동일한 추가 seed로 함께 반복한다.

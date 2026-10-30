# Hilbert · Context OFF · Score 1 stream vs 4 streams — 2026-10-17

Score의 4096 token과 56채널을 하나의 entropy stream으로 부호화하는 조건과, dyadic4의 네 token 집합을 각각 하나의 stream으로 부호화하는 조건을 비교한다. 사용자 지정에 따라 Hilbert 정렬을 고정하고, 채널 그룹도 56채널 하나로 합친다. MSH는 두 조건 모두 유지한다.

| 조건 | Hilbert 순번의 token 집합 | 집합별 Score shape | 실제 Score rANS strings/scene |
|---|---|---|---:|
| `single`, stages1, channels56 | 전체 | `[4096, 56]` | 1 |
| `dyadic4`, stages4, channels56 | `0 mod 8`, `4 mod 8`, `2 mod 4`, 홀수 | `[512,56]`, `[512,56]`, `[1024,56]`, `[2048,56]` | 4 |

이 수는 Score payload 내부의 실제 entropy string 수다. MSH의 y/z stream과 bitstream 컨테이너는 별도로 유지된다. 한 stream 안에서도 확률분포는 채널별 factorized 분포이며, 채널 예측은 사용하지 않는다.

## 고정 조건

- λ=.0064/.0128/.0256 × 2조건: 학습6개와 `afterok` 평가6개.
- 기존 ablation과 같은 FullSplit λ=.0256 probability-only step10k 부모에서 weights-only로 시작한다. 부모 optimizer/scheduler/global step을 이어받지 않고 각 조건을 새50k optimizer steps 학습한다.
- GlobalSplat 고정, rank56 linear analysis/synthesis와 MSH 유지. Codec 전체를 학습한다.
- Minimal: scene centering, mean 전송, Score norm, mean conditioner, channel context, 추가 nonlinear MLP OFF. 두 조건 모두 Score spatial prediction OFF. 기존 확률모델의 채널별 median 양자화는 유지한다.
- Hilbert 순서를 Score와 MSH 양쪽에 적용한다. dyadic4에서도 복호된 anchor를 입력으로 쓰지 않는다.
- 학습 GPU2 × batch2 × accumulate2 = 유효배치8, BF16, LR1e-4, 35k/45k LR 감소, checkpoint 매5k. 학습 시간 제한6일.
- 기존 recipe의 학습 C24/T12, subset consistency ON, seed111123/data-loader seed403 유지.
- 최종50k checkpoint에서 전체 test split의 실제 encode→decode→render 평가. 기존 C12/T8, batch1, seed0, BF16 조건을 사용한다.

공통 부모:

```text
outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt
```

## 채널 통합 시 초기화와 해석

부모의 채널 그룹 `16/16/16/8`에 저장된 unpredicted Score entropy 모델을 채널축으로 이어 붙인다. 채널별 quantiles, matrices, biases, factors를 보존하고, 두 조건의 모든 활성 단계에 같은 분포를 복사한다. 정수 CDF 테이블은 새 채널 구성에 맞게 재생성한다. Basis, geometry projection, MSH 등 Score context 밖의 가중치는 그대로 로드한다.

이에 따라 두 조건은 같은 채널별 초기 확률분포와 양자화 median으로 시작한다. 단일 입력으로 검증한 초기 복원값도 같다. 비활성 predictor와 호환용 entropy 모듈은 동결한다. 학습/평가 checkpoint에는 1단계를 구분하는 marker와 bitstream flag가 저장되며, 일반 resume/eval은 metadata와 architecture를 엄격하게 검사한다.

4블록은 학습 중 네 단계의 확률분포를 독립적으로 최적화한다. 따라서 최종 차이는 rANS 종료·헤더 비용뿐 아니라 **블록별 확률모델 분리의 효과**도 포함한다. 추후 고정 checkpoint에서 분포를 공유하고 stream 경계만 바꾸는 측정을 추가하면 그 두 효과를 더 분리할 수 있다.

기존 Hilbert Context OFF는 legacy3 × 채널4그룹 = 12 strings였다. 이번 두 조건은 channels56으로 통일해 1 vs 4를 비교하며, 기존 실험과 출력 및 제출 signature를 구분한다. 기존 12-stream 결과와 비교할 때는 채널 통합과 단계 구성이 함께 달라졌다는 점을 반영한다.

## 서버 반영 파일

프로젝트 루트 `/ceph_data/clue9986/tokencomp/upstream/globalsplat` 아래의 같은 상대 경로에 다음 8개 파일을 반영한다.

1. `globalsplat/compression/config.py`
2. `globalsplat/compression/score_context.py`
3. `globalsplat/compression/codec.py`
4. `globalsplat/compression/checkpoint.py`
5. `scripts/run_nfcgs.py`
6. `scripts/slurm/nfcgs_order_context_settings.sh`
7. `scripts/slurm/submit_nfcgs_order_context.sh`
8. `scripts/slurm/submit_nfcgs_context_off_blocks_rd.sh`

서버의 기존 `train_nfcgs_order_context.slurm`과 `eval_nfcgs_order_context.slurm`을 재사용한다. 학습 Slurm의 Ceph `CHECKPOINT_TMPDIR=` 수정이 없으면 전용 launcher가 제출 전에 중단한다. YAML이나 `globalsplat/main.py`의 추가 수정은 필요하지 않다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
if bash scripts/slurm/submit_nfcgs_context_off_blocks_rd.sh --dry-run; then
    bash scripts/slurm/submit_nfcgs_context_off_blocks_rd.sh
fi
```

제출 launcher 자체는 `bash`로 실행한다. Dry-run은 6개 point의 설정·부모·저장 state를 확인하고 `Score streams=1/4`를 출력한다. 실제 제출에서는 각 학습에 GPU2개, 해당 학습 성공 후 평가에 GPU1개를 요청한다. 이미 제출된 ID는 state에 보존하며, QOS 등으로 일부 제출만 실패하면 같은 명령을 다시 실행해 빠진 제출을 이어간다. 실패·취소된 학습을 자동 resume하는 기능은 아니다.

기본값은 Hilbert만 제출한다. 별도 `CONTEXT_OFF_ORDERS` 설정이 없는 셸에서 실행한다. 기본 state 폴더를 바꾸면 별도 제출로 취급되므로 재시도 시에는 유지한다.

| 기록 | 경로 |
|---|---|
| 제출 state/요약 | `logs/slurm/context_off_blocks_rd_20261017/` 및 `submissions.tsv` |
| 1 stream 출력 | `outputs/nfcgs_order_context/hilbert/single_stages1_k5_ste_context_off_c56/lambda<λtag>/job_<TRAIN_ID>/` |
| 4 streams 출력 | `outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste_context_off_c56/lambda<λtag>/job_<TRAIN_ID>/` |
| 학습 로그 | `logs/slurm/slurm-gs-order-context-<TRAIN_ID>.out` / `.err` |
| 평가 로그 | `logs/slurm/slurm-gs-order-context-eval-<EVAL_ID>.out` / `.err` |

최종 checkpoint는 각 출력의 `checkpoints/nfcgs_main/version_0/step000050000.ckpt`, 평가는 `eval_all/`에 저장한다. 평가 기록에는 Score 단계별 raw bytes와 Score wrapper bytes, Score/MSH y/MSH z/컨테이너별 실제 bytes가 포함된다. 품질과 실제 전체 bytes, 단계별 bits/token, encode/decode 시간을 함께 비교한다.

## 로컬 검증 및 실행 상태

- 실제 rank56 × 4096 tokens에서 1/4개의 물리적 entropy strings, encode/decode 왕복, 예측기 미실행, token 간 독립성을 검증했다.
- 채널 통합 시 부모의 채널별 entropy 파라미터 보존, 두 조건의 초기 복원값 일치, MSH·basis 유지, 활성 경로 gradient, 비활성 파라미터 동결과 strict checkpoint 로딩을 검증했다.
- 실제 Hydra train/eval 구성, single 조건의 잘못된 spatial ON 거부, 기존 legacy/quarter2/dyadic4 및 정렬 실험의 회귀 검사를 통과했다.
- 가짜 Slurm으로 6개 학습과 6개 개별 afterok 평가, ID 재사용, eval 제출 실패 후 재시도, 16→56채널 변경 시 기존 제출 state 재사용 차단을 확인했다.
- 관련 고유 검사 83개 중 82개 통과, CUDA 검사는 로컬 CPU 환경으로 1개 생략했다. Python3.11 환경에서도 codec 핵심4개를 다시 통과했다.
- 서버 제출, GPU 학습/평가 및 새 실험 결과는 아직 없다.

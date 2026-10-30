# NN 정렬 구현 최적화 — 2026-10-13

## 적용한 변경

10월 13일 사용자 요청에 따라 새 거리 정렬 실험과 profiler는 `nn_score`만 대상으로 한다.
취소 대상은 XYZ 학습 `448652`와 eval `448653`이며, 서버에서 `scancel 448652 448653`을 실행해야 한다.
Score 학습 `448650`과 eval `448651`은 이 취소 대상에 포함되지 않는다.

4096-token 규모에서는 먼저 정확한 greedy 정렬의 구현 비용을 줄인다.
Score의 가중 L1 거리, XYZ의 L2 거리, Morton 최소 key에서 시작하는 규칙을 유지한다.
전체 token을 한 번씩 보내며, 같은 거리에서는 원래 slot 번호가 작은 token을 선택한다.
Score와 MSH는 같은 순서를 사용한다. 수신기는 정렬을 다시 계산하지 않는다.

| 옵션 | 거리 계산 | greedy 처리 | 추가 의존성 |
|---|---|---|---|
| `legacy` | CPU FP32 | 방문한 열 전체를 `inf`로 변경 | 없음 |
| `row` | CPU 또는 입력 장치 FP32 | 현재 행에만 방문 mask 적용 | 없음 |
| `knn` | CPU 또는 입력 장치 FP32 | exact top-k + Numba 루프 + 전체 행 fallback | Numba |

기본값은 `legacy`, `cpu`, 후보 64개다. `input`은 입력 token이 CUDA에 있으면 거리와 top-k를 CUDA에서 처리한다.
greedy 루프는 CPU에서 수행하고, 거리 행렬·후보 번호·후보 거리를 장면마다 CPU로 옮긴다.
token마다 CUDA 결과를 읽거나 동기화하지 않는다.

후보 64개는 자기 자신을 제외한 이웃이다. 사용자 자료의 65개 목록이 자기 자신을 포함했다면 같은 크기다.
후보가 전부 방문됐으면 전체 행을 검색한다. 선택할 거리가 후보 목록의 마지막 거리와 같으면
목록 밖에도 동률 token이 있을 수 있으므로 전체 행을 검색한다.
[PyTorch topk 문서](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)는 동률 index의 안정성을 보장하지 않는다.
이 fallback으로 **같은 거리 행렬에 대한** 원래 greedy 규칙을 유지한다.

CPU/GPU FP32 거리 누적의 반올림 차이로 매우 가까운 이웃의 선택이 달라질 수 있다.
CPU 최적화가 원래 CPU 정렬과 같은지, GPU 최적화가 같은 GPU 거리 행렬의 전체 검색과 같은지를 따로 검증한다.
benchmark JSON에는 GPU 경로와 원래 CPU 경로의 순서 일치 여부도 기록한다.

정렬 옵션은 checkpoint metadata에 저장된다. 구현 옵션만 바꾸면 모델 가중치나 optimizer 구조를 바꾸지 않아도 된다.
정렬 종류·context 종류를 바꾸는 기존 warm start 검증은 유지한다.
최적화 경로는 별도의 output 경로와 제출 state fingerprint를 사용한다.

## 로컬 실측

합성 입력, N=4096, batch 2, CPU thread 8, warmup 1회 후 3회 중앙값이다.
장면당 수치는 batch 시간을 2로 나눴다. Numba 초기 컴파일은 제외했다.
PyTorch 2.11.0+cpu, NumPy 1.26.4, Numba 0.68.0이다.

| 정렬 | 기존 전체 정렬 | row 전체 정렬 | knn 전체 정렬 | knn 속도 향상 |
|---|---:|---:|---:|---:|
| Score, 56차원 가중 L1 | 422.2 ms | 293.5 ms | 259.4 ms | 1.63배 |

첫 장면의 knn 단계별 중앙값:

| 정렬 | 거리 계산 | top-k | CPU 배열 준비 | greedy 루프 |
|---|---:|---:|---:|---:|
| Score | 234.17 ms | 24.59 ms | 0.032 ms | 1.89 ms |

단계별 합은 전체 batch 측정과 별도 실행한 값이다.
CPU 경로들은 두 장면 모두 기존 순서와 같았고, 실제 codec 테스트에서는 bitstream bytes도 같았다.
이 수치는 서버나 전체 학습 step의 속도 향상이 아니다. **GPU 수치는 아직 측정하지 않았다.**
[실측 JSON](../../../artifacts/nn_score_order_backends_cpu_2026-10-13.json)에 반복별 시간이 있다.

Score의 CPU 잔여 비용은 주로 가중 L1 거리 계산이다.
사용자 자료의 L2 행렬곱 시간을 현재 L1 정렬의 시간으로 대입하지 않는다.
[PyTorch cdist 문서](https://docs.pytorch.org/docs/2.14/generated/torch.cdist.html)의 행렬곱 옵션은 p=2 유클리드 거리에 대한 옵션이다.
이번 구현은 L1을 L2로 바꾸지 않으며, XYZ도 기존 direct L2 계산을 유지한다.

거리 계산은 O(N²D), 거리 행렬은 장면당 N²개의 FP32 값이다.
N=4096일 때 행렬 하나는 64 MiB이며, GPU 경로에는 CPU 복사본과 후보 배열도 필요하다.
knn greedy 루프의 비용은 O(NK+FN)이다. F는 전체 행 fallback 횟수이고, 최악에는 O(N²)이 된다.
전체 파이프라인을 O(N)으로 바꾼 구현은 아니다.

## 서버 측정과 실행

저장소 root에서 실행한다. 서버 코드 파일 목록은 아래 절에 있다.

```bash
conda activate globalsplat
python -m pip install numba
mkdir -p logs/slurm
sbatch scripts/slurm/benchmark_nfcgs_nn_order.slurm
```

측정 job 이름은 `gs-nn-order-bench`, `gpu:normal:1`, CPU 8개, 메모리 16G다. Score만 측정한다.
현재 프로젝트 Slurm 설정처럼 v7을 제외한다.
CUDA 거리·top-k, CPU 전달, compiled greedy 시간을 나눠 기록한다.
출력은 `outputs/nn_order_benchmark/job_<JOB_ID>.json`이다.
CPU/GPU 순서 차이, 후보 소진·동률 fallback 횟수, 전체 정렬 시간도 기록한다.

최적화를 켠 새 학습+전체 장면 eval 예시:

```bash
TOKEN_ORDER=nn_score CONTEXT_SCHEDULE=legacy STAGES=3 \
NN_ORDER_BACKEND=knn NN_ORDER_DISTANCE_DEVICE=input NN_ORDER_CANDIDATES=64 \
  bash scripts/slurm/submit_nfcgs_order_context.sh
```

Numba 없이 열 덮어쓰기 비용만 줄이려면 `NN_ORDER_BACKEND=row`를 사용한다.
원래 구현은 `NN_ORDER_BACKEND=legacy NN_ORDER_DISTANCE_DEVICE=cpu`다.
training은 GPU 2개, GPU당 batch 2, accumulate 2로 유효 batch 8이다.
eval은 training의 `afterok` dependency로 제출된다.
기본 λ는 .0256이고, 기존 `RATE_LAMBDA` 옵션을 사용할 수 있다.

최적화 output 예시:
`outputs/nfcgs_order_context/nn_score/legacy_stages3_k5_ste_nn_knn_input_k64/lambda0p0256/job_<TRAIN_ID>`.

CLI에서도 `--nn-order-backend knn --nn-order-distance-device input --nn-order-candidates 64`를 지정한다.
기존 NN checkpoint의 같은 모델·context 설정으로 resume할 때 구현 옵션만 바꾸는 것은 허용한다.
현재 실행 중인 프로세스가 파일 동기화만으로 새 backend로 전환되지는 않는다.

## 사용자 제공 실험을 반영한 후속 방향

아래 수치는 사용자가 전달한 외부 E87/E88/E91 결과다. 이 세션의 GlobalSplat 전체 장면 실측으로 기록하지 않는다.
실험 데이터·checkpoint·원본 측정 파일은 이 요약에 포함되지 않았다.

- 외부 E87은 초기 Score에서 정렬을 고정해도 Morton legacy2 대비 nn_score dyadic4의
  Score −18.9%, 특징 오차 −8.6%, J −13.5%를 보고했다.
  Hilbert dyadic4의 J −8.9%보다 좋아, 고정 reference 정렬을 검증할 근거가 된다.
- 외부 E88/E91은 정확한 값 공간 경로의 Score rate −42.0% 대비
  16채널 후보/56채널 선택 −41.9%, k-means64 −38.5%, Hilbert256 블록 −37.2%를 보고했다.
  큰 N에서 후보 탐색이나 블록 정렬을 비교할 근거가 된다.
- 외부 XYZ KD-tree는 순서 일치와 −20.7%를 보고했다. 최악 복잡도와 실제 fallback 비용은 별도로 측정해야 한다.

장면 ID만 사용하는 정렬 cache는 이번 구현에 넣지 않았다.
현재 `scope=all` 학습은 codec의 geometry projection과 Score basis를 갱신하고,
입력 view/augmentation도 바뀔 수 있으므로 같은 장면의 현재 Score 순서가 항상 같지는 않다.
고정 초기 정렬을 실험하려면 view·augmentation·token 수/slot 정체성이 같은 입력 key와
저장된 reference transform을 정의하고, eval에서도 그 reference를 사용해야 한다.
이때 학습과 eval이 같은 정렬 정책을 쓰도록 별도 옵션으로 구현한다.

## 서버에 반영할 파일

| 경로 | 용도 |
|---|---|
| [compression/token_order.py](../globalsplat/compression/token_order.py) | CPU/GPU 거리, exact kNN + compiled greedy |
| [compression/config.py](../globalsplat/compression/config.py) | backend/device/후보 수 설정 |
| [compression/codec.py](../globalsplat/compression/codec.py) | Score/XYZ 정렬 호출 연결 |
| [compression/checkpoint.py](../globalsplat/compression/checkpoint.py) | metadata·resume 호환 |
| [globalsplat_nfcgs_rank56.yaml](../config/model/globalsplat_nfcgs_rank56.yaml) | Hydra 기본 설정 |
| [run_nfcgs.py](../scripts/run_nfcgs.py) | CLI 옵션 |
| [nfcgs_order_context_settings.sh](../scripts/slurm/nfcgs_order_context_settings.sh) | train/eval 공통 설정과 output 경로 |
| [submit_nfcgs_order_context.sh](../scripts/slurm/submit_nfcgs_order_context.sh) | 제출 state 구분 |
| [submit_nfcgs_order_context_followup.sh](../scripts/slurm/submit_nfcgs_order_context_followup.sh) | 후속 목록에서 XYZ 제외 |
| [train_nfcgs_order_context.slurm](../scripts/slurm/train_nfcgs_order_context.slurm) | Numba 확인·설정 로그 |
| [eval_nfcgs_order_context.slurm](../scripts/slurm/eval_nfcgs_order_context.slurm) | 설정 로그 |
| [benchmark_nfcgs_nn_order.py](../scripts/benchmark_nfcgs_nn_order.py) | 시간·순서 검증 profiler |
| [benchmark_nfcgs_nn_order.slurm](../scripts/slurm/benchmark_nfcgs_nn_order.slurm) | normal GPU 1개 측정 job |

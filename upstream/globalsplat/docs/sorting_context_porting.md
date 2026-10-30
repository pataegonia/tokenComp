# Token 정렬 + Score context 모델을 다른 모델에 이식하는 가이드

기준: 2026-10-30, 정리된 NFC-GS 코덱(브랜치 `code-cleanup`, 커밋 609a781 이후).
이 문서는 **이식 가이드와 최소 코드 예제**이며, 대상 모델에 실제 통합한 구현은 아니다.
아래에서 ‘현재 구현’과 ‘새 모델에서 선택할 방법’을 구분한다.

## 1. 무엇을 떼어내는가

Token 정렬과 단계별 인과(causal) context 엔트로피 모델은 GlobalSplat encoder, Gaussian decoder,
MSH residual hyperprior와 독립인 모듈이다. 정렬만 가져오면 효과가 없고, 정렬된 순서를 이웃으로
쓰는 context 모델을 함께 가져와야 한다.

| 부분 | 입력 | 출력 | 역할 |
|---|---|---|---|
| Token 정렬 | token 위치 `P [B,N,3]` (또는 token 값) | 순열 `π [B,N]` | 순서상 이웃을 실제로 가까운 token으로 만든다 |
| (선택) 저차원 Score 변환 | token `X [B,N,C]` | Score `s [B,N,R]` | context가 다룰 채널 수를 줄이고 양자화 간격을 학습한다 |
| 단계별 context 엔트로피 모델 | 정렬된 Score `[B,R,1,N]` | 복원 Score, likelihood, bitstream | 먼저 복호한 이웃으로 예측하고 잔차만 부호화한다 |

```text
새 모델의 token producer → X [B,N,C], token 위치 P [B,N,3]
  ├─ 정렬: π = argsort(Hilbert(P))         ← encoder 전용, 전송하지 않음(§5)
  ├─ Score: s = X[π] · Aᵀ  [B,N,R]          ← 선택, 학습되는 선형 변환
  └─ context 엔트로피 모델 (dyadic4, 4단계)
        단계0  token 0, 8, 16, …   : 예측 없이 부호화
        단계1  4 mod 8             : 좌우 단계0 anchor로 예측
        단계2  2 mod 4             : 좌우의 이미 복호한 anchor로 예측
        단계3  홀수                : 좌우의 이미 복호한 anchor로 예측
     → ŝ, bits
  ŝ · Sᵀ → (후단이 원래 slot을 요구할 때만 π⁻¹로 복원) → 후단
```

현재 GlobalSplat 구현에서는 4096개 token, Score 56채널(16채널씩 4그룹), 정렬은 Hilbert,
schedule은 dyadic4다. Score로 설명하지 못한 나머지는 별도 MSH residual 경로로 보낸다(§9).

## 2. 가져올 소스

| 파일/함수 | 이식 시 용도 |
|---|---|
| [`token_order.py`](../globalsplat/compression/token_order.py)의 `hilbert_codes_3d`, `hilbert_order_3d` | 위치 기반 정렬(기본) |
| 같은 파일의 `greedy_neighbor_order`와 `_graph_candidates`, `_greedy_graph_order`, `graph_order_scene` | 값/위치 기반 greedy NN 정렬(선택) |
| [`morton.py`](../globalsplat/compression/morton.py)의 `MortonOrder`, `batched_gather`, `invert_permutation`, `_quantize_positions`, `morton_order_3d` | 순열 객체(`apply`/`restore`), 격자 양자화, Morton 정렬 |
| [`score_context.py`](../globalsplat/compression/score_context.py)의 `AnchorPairPredictor`, `ContextualScoreEntropy` | context 모델 전체(학습 forward, compress, decompress) |
| [`entropy.py`](../globalsplat/compression/entropy.py)의 `FactorizedScoreEntropy` | CompressAI `EntropyBottleneck` 래퍼 |
| [`bitstream.py`](../globalsplat/compression/bitstream.py)의 `ScoreContextBitstream` | context stream의 header와 길이 prefix |
| [`codec.py`](../globalsplat/compression/codec.py)의 `ObservableLowRank1DCodec` | 참고용. Score 분석·합성, MSH와 묶은 실제 통합 예 |

의존성은 PyTorch, `compressai==1.2.8`(EntropyBottleneck과 rANS), NumPy다. `numba`는 NN 정렬의
`knn`/`graph`/`graph_gemm` backend에만 필요하다. 위 5개 파일은 서로 상대 import(`from .morton import …`)를
쓰므로 **한 패키지로 함께 복사**한다. `bitstream.py`에서 `ScoreContextBitstream`만 쓰면 나머지
클래스는 지워도 된다.

추천 배치:

```text
your_project/
  sorted_context/
    __init__.py       # 빈 파일 또는 필요한 클래스만 export
    morton.py
    token_order.py
    entropy.py
    bitstream.py
    score_context.py
  model.py
```

저장소 라이선스(`pyproject.toml`: PolyForm Noncommercial 1.0.0과 `ADDITIONAL_TERMS.md`)와
CompressAI 라이선스 조건을 확인하고 보존한다.

## 3. 현재 구현의 수식

### 정렬: 장면별 격자 위의 Hilbert 곡선

```text
q = round( clamp((P − min_n P) / max(max_n P − min_n P, eps), 0, 1) × (2^b − 1) )   # 축마다, b = 10
key = Hilbert(q)          # Skilling axes-to-transpose, 3b비트 정수
π = argsort(key, stable)  # 같은 key는 원래 slot 순서
```

- 정규화는 **장면별, 축별** min–max다. 납작한 장면은 축마다 늘어나서 격자가 비등방이 된다.
- Morton(`morton_order_3d`)은 같은 격자에서 비트만 교차 배치한다. 우리 실험에서는 Hilbert가 조금 더 좋았다(§9).
- 정렬은 미분하지 않는다. `P`는 `detach`하고, gradient는 `gather`된 feature를 통해 흐른다.

### 정렬(선택): greedy NN tour

```text
u_c = s_c / max(mean_n |s_c|, 0.05)           # 채널별 장면 평균 절댓값으로 정규화
시작 = argmin_n Morton(P)
다음 = 아직 방문하지 않은 token 중 L1(u) 거리가 가장 가까운 것
```

`graph_gemm` backend는 L2 GEMM으로 token당 128개 후보를 뽑고, L1로 다시 정렬해 64개 이웃 목록을 만든다.
목록이 모두 방문됐으면 전체 검색 대신 Hilbert(`P`) 순서에서 다음 미방문 token으로 넘어간다.
`nn_xyz`는 같은 방식을 `P`의 L2 거리로 한다.

### dyadic4 단계와 anchor 예측

```text
단계0: i ≡ 0 (mod 8)   단계1: i ≡ 4 (mod 8)   단계2: i ≡ 2 (mod 4)   단계3: i 홀수
단계 k ≥ 1의 간격 d_k = 8, 4, 2
  왼쪽 anchor  a = ⌊i / d_k⌋ · d_k
  오른쪽 anchor b = a + d_k   (b ≥ N이면 b = a, edge = 1)
  예측  p_i = W_{k,0} ŝ_a + W_{k,1} ŝ_b + bias_k + edge · edge_bias_k   # 채널 그룹 안 선형 혼합
단계0: p_i = 0
부호화 기호  y_i = (s_i − p_i) / Δ,   Δ = 1
복원        ŝ_i = p_i + Δ · Q(y_i)
```

- Score 채널 `R`은 `slice_channels`(16)개씩 그룹으로 나뉜다. (단계, 그룹)마다 따로
  `FactorizedScoreEntropy`가 있어 stream은 `4 × ⌈R/16⌉`개다.
- 부호화는 **단계 우선**이다. 단계 k의 모든 그룹이 끝나야 단계 k+1로 간다.
- 예측기의 입력은 이전 단계에서 **복원된** 값만 담은 canvas다(`visible` mask). 현재·미래 단계의 원래
  값은 들어가지 않으므로 수신측이 같은 예측을 재현한다.
- `W`의 모양은 `[width, width, 2]`이고 0으로 초기화된다. 학습 초기에는 예측이 0이라 context가 없는
  모델과 같다.
- `Δ = 1`로 고정돼 있다. 실제 양자화 간격은 앞단의 학습되는 선형 변환 `A`의 크기가 정한다.
  변환 없이 raw feature를 넣는다면 채널별 간격을 따로 학습해야 한다.

다른 schedule도 같은 클래스에 있다.

| `context_schedule` | 단계 | 예측기 |
|---|---|---|
| `dyadic4` (기본) | 4 | 위 `AnchorPairPredictor` |
| `quarter2` | 2 (anchor `i≡0 mod 4`, 나머지) | anchor 쌍, 위치(phase)별 가중치 |
| `legacy`, 2·3·4단계 | 짝/홀 또는 A/B/C/D | 복원 anchor canvas 위의 1-D conv(kernel 3·5·7) |
| `single` | 1 | 예측 없음(Context OFF 대조군) |

### 학습 때의 양자화와 rate

```text
likelihood: EntropyBottleneck(training=True)    # uniform noise 근사
값:         v = y + stopgrad( round(y − m) + m − y )   # m = 학습된 channel median, STE
R = Σ −log2 likelihood
```

학습 중에도 예측기에는 STE로 양자화된 복원값이 들어간다. 그래서 학습과 실제 부호화의 context가 같다.

## 4. 최소 adapter 예제

아래 코드는 정리된 코드에서 CPU로 실행해 확인했다(§10 끝).

```python
import torch
from torch import nn
from sorted_context.token_order import hilbert_order_3d
from sorted_context.score_context import ContextualScoreEntropy


class SortedContextCodec(nn.Module):
    def __init__(self, width, rank, slice_channels=16):
        super().__init__()
        self.analysis = nn.Linear(width, rank, bias=False)     # s = x Aᵀ
        self.synthesis = nn.Linear(rank, width, bias=False)    # x̂ = ŝ Sᵀ
        self.context = ContextualScoreEntropy(
            rank=rank, slice_channels=slice_channels,
            spatial_stages=4, spatial_kernel=5, context_schedule="dyadic4")

    def forward(self, tokens, positions):                     # 학습
        order = hilbert_order_3d(positions.detach().float(), bits=10)
        score = self.analysis(order.apply(tokens))             # [B,N,R], 정렬 순서
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            score_hat, likelihood = self.context(score.float().transpose(1, 2).unsqueeze(2))
        recon = self.synthesis(score_hat.squeeze(2).transpose(1, 2))
        bits = -torch.log2(likelihood.clamp_min(1e-9)).sum()
        return order.restore(recon), bits                      # 원래 slot이 필요할 때만 restore

    @torch.no_grad()
    def compress(self, tokens, positions):                     # 장면 하나(B=1)
        order = hilbert_order_3d(positions.float(), bits=10)
        score = self.analysis(order.apply(tokens)).float().transpose(1, 2).unsqueeze(2)
        payload, score_hat = self.context.compress(score)      # score_hat = 수신측 복원과 같음
        return payload, order, score_hat

    @torch.no_grad()
    def decompress(self, payload, points):
        score_hat = self.context.decompress(payload, points)
        return self.synthesis(score_hat.squeeze(2).transpose(1, 2))   # 정렬 순서
```

학습과 평가:

```python
codec = SortedContextCodec(width=C, rank=R)
recon, bits = codec(tokens, positions)
loss = task_loss(recon) + lam * bits / normalizer        # §6
loss = loss + codec.context.aux_loss()                   # quantile 학습, §7
loss.backward()

codec.eval()
codec.context.update(force=True)                         # 현재 weight로 CDF 표를 다시 만든다
payload, order, sender = codec.compress(tokens[:1], positions[:1])
assert torch.equal(sender, codec.context.decompress(payload, tokens.shape[1]))
```

## 5. 모델 종류별 연결 방법

### A. token마다 독립적인 후단(Gaussian head 등, 출력이 집합)

현재 GlobalSplat가 이 경우다. Gaussian decoder가 token별로 독립이므로, 수신측은 **정렬된 순서 그대로**
복호한 token을 후단에 넣으면 된다. 순열을 전송하지 않는다. 학습 중 `restore`는 loss 계산 편의를 위한 것이다.

### B. 후단이 원래 slot 위치를 쓰는 모델(위치 embedding, grid, attention의 위치 bias)

순열이 필요하다. 두 방법 중 하나를 고른다.

1. 수신측이 먼저 받은 정보로 같은 순서를 다시 계산할 수 있으면 전송하지 않는다
   (예: 위치를 별도로 먼저 보내고 그 위치로 Hilbert를 계산).
2. 아니면 순열을 전송하고 그 비용을 rate에 넣는다. 4096개 token이면 대략 `N·log2 N ≈ 6 KiB`이고,
   우리 장면당 packet(수십 KiB) 대비 작지 않다.

### C. 3-D 위치가 없는 token

Hilbert·Morton은 3-D 위치가 필요하다. 위치가 없으면 값 기반 정렬(NN-score)이나 1-D/3-D PCA 좌표 위의
Hilbert를 쓴다. 정렬 기준이 encoder에서만 쓰이면 A의 조건에서 전송은 필요 없다.

### D. 이미 격자 구조인 latent(이미지·비디오 latent)

raster 순서가 이미 이웃을 보존한다. 이 모듈은 1-D 시퀀스용이므로, 이런 모델에는 2-D 마스크 context
(checkerboard 등)가 더 자연스럽다.

위치 `P`의 출처: 우리는 양자화 전 geometry feature로 계산한 token 중심(`decoded_token_centers`)을 쓴다.
token이 대표하는 3-D 점이면 무엇이든 된다(예: token에서 나오는 Gaussian들의 평균 위치).

## 6. Loss

```text
L = L_task + λ · R / normalizer + L_aux
```

- 우리 구현: `rate_loss = λ · Σbits / (인코딩 수 × Gaussian 수 × 59)`, 곧 bits per Gaussian attribute다.
  λ는 .0032–.0512를 썼다. 이 숫자는 정규화와 task loss(MSE + 0.05·LPIPS) 크기에 묶여 있으므로
  다른 모델에 그대로 옮기지 않는다.
- `L_aux = Σ EntropyBottleneck.loss()`(`context.aux_loss()`)는 quantile을 맞추는 보조항이다(§7).
- context의 효과는 rate 항에서 나온다. λ = 0이면 bit를 줄일 유인이 없어 예측기가 거의 학습되지 않는다.

## 7. 초기화, freeze, optimizer

1. 예측기(`AnchorPairPredictor`)는 0 초기화다. 처음에는 context 없는 모델과 같고, 학습하며 예측이 생긴다.
2. CompressAI 관례대로 `entropy_bottleneck.quantiles`는 주 optimizer에서 빼고 aux loss로 학습한다.
   평가 전에는 `update(force=True)`로 CDF 표를 다시 만든다.
3. 정렬은 미분하지 않는다(`detach`). gradient는 정렬된 feature와 예측기, 엔트로피 모델로 흐른다.
4. 혼합 정밀도(AMP)에서도 context 모델은 **FP32**로 돌린다(`autocast(enabled=False)`). 송신·수신 예측이
   bit 단위로 같아야 하기 때문이다.
5. 우리 학습은 codec만 학습(backbone 고정), Adam 1e-4, 120k step이었다. context ON/OFF, 정렬 유무
   네 조합을 같은 초기값에서 학습해 비교했다.
6. 다른 schedule에서 학습한 weight로 시작하려면 `checkpoint.convert_score_path_state`를 참고한다.
   kernel 3→5 확장, 단계 추가 시 entropy 모델 복사 규칙이 들어 있다.

## 8. Batch, inference, bitstream

- 학습 `forward`는 batch `B`를 받는다. `compress`/`decompress`는 장면 하나(`B = 1`)씩이다.
- 단계마다 token 수가 달라서, 8개 미만인 기호 배열은 rANS 버퍼를 위해 내부에서 0으로 padding한다.
  padding은 context에 들어가지 않는다.
- bitstream은 `ScoreContextBitstream(rank, slice_channels, flags, strings)`다. header(24B: magic,
  version, flags, rank, slice_channels, stream 수)와 stream마다 8B 길이 prefix가 붙는다. `flags`에 schedule·단계 수·kernel·spatial context가 기록되고, 수신측은
  자기 설정과 다르면 거부한다. 실제 rate에는 header와 prefix를 포함한다.
- 수신측 복원이 송신측과 같으려면 같은 연산 정책이 필요하다. 우리는 TF32 off, cuDNN deterministic으로
  같은 장치 종류에서 부호화·복호했다. CPU에서 부호화하고 GPU에서 복호하는 식의 장치 간 bit 일치는 확인하지 않았다.
- 복호는 단계 4번이 순서대로 돈다. 한 단계 안의 token은 모두 병렬이라 token별 자기회귀보다 훨씬 빠르다.
- 정렬 비용: Hilbert는 key 계산과 정렬 한 번이라 무시할 수준이다. NN 정렬은 §9 참고.

## 9. 기대할 수 있는 효과와 비용의 범위

아래는 **우리 모델(GlobalSplat 4096 token, Score 56채널)의 측정값**이다. 다른 모델에서 같은 크기를 기대하지 않는다.

| 항목 | 값 | 근거 |
|---|---|---|
| 정렬 효과(Context ON, 5λ RD) | 정렬 없음 대비 BD-rate 약 −33% (.0128 seed를 바꿔도 −32.4~−33.6%) | [계획 문서](NFCGS_FRESH_RD_PLAN_2026-10-20.md) §15 |
| Hilbert+D4 vs Morton+D4 / Hilbert+L3 | 같은 λ 양끝점에서 bytes 1.2–2.2% / 3.3–3.9% 작음 | [로그 검토](analysis/nfcgs_logs_2026-10-19_2020/REPORT.md) |
| 인접 token의 Score 차이(무작위 순서 = 1) | NN-score(L1) 0.32, Hilbert 0.47 | [계획 문서](NFCGS_FRESH_RD_PLAN_2026-10-20.md) §15 |
| NN 정렬 비용 | 정확한 greedy는 학습 step +53~55%, `graph_gemm`은 장면당 약 10ms(GPU) | 같은 문서 |

- 정렬은 **context가 이웃을 참조할 때** 의미가 있다. 정렬 효과는 Context ON 조건에서 측정했다.
- 이식 전에 대상 모델에서 [`analyze_nfcgs_token_adjacency.py`](../scripts/analyze_nfcgs_token_adjacency.py)처럼
  "정렬별 인접 token의 값 차이"를 먼저 재 보면 좋다. 학습 없이 몇 분이면 되고, 정렬로 이웃이 얼마나
  가까워지는지(곧 context가 얻을 여지)를 미리 알 수 있다.
- 우리 코덱은 Score(56채널, context)와 나머지 residual(MSH 1-D hyperprior, context 없음)의 두 경로를 쓴다.
  MSH도 1-D conv라 정렬의 영향을 받는다. 이 가이드의 모듈은 Score 경로에 해당한다.

## 10. 이식 후 필수 체크리스트

- [ ] 순열이 bijection이고 `order.restore(order.apply(x)) == x`인가?
- [ ] 예측기를 0으로 둔 상태에서 결과가 context 없는 모델(`single` schedule)과 같은가?
- [ ] 장면마다 송신측 `score_hat`과 `decompress` 결과가 `torch.equal`인가?
- [ ] 단계 k의 예측이 단계 k 이후 token 값의 변화에 영향받지 않는가? (미래 token을 바꿔도 예측 불변)
- [ ] AMP에서도 context가 FP32로 도는가?
- [ ] 평가 전에 `update(force=True)`를 했고, quantile이 주 optimizer에 들어가지 않는가?
- [ ] `N`이 8의 배수가 아니거나 아주 작을 때도 왕복이 맞는가?
- [ ] 추정 bit와 실제 byte 수(header 포함)를 함께 기록하는가?
- [ ] 분석 변환, 예측기, 엔트로피 모델에 gradient가 가는가?
- [ ] 후단이 원래 slot을 쓰는 모델이면, 순열 전송 비용을 rate에 넣었는가?

우리 저장소의 해당 테스트는 [`test_token_order_context.py`](../tests/test_token_order_context.py)(정렬·schedule
조합, 미래 단계 비가시성, 짧은 stream), [`test_score_spatial_stages.py`](../tests/test_score_spatial_stages.py),
[`test_context_off_blocks.py`](../tests/test_context_off_blocks.py)다. 이식한 코드에 맞게 옮겨 쓰면 된다.

### 이 문서 예제의 확인 범위

§4 예제를 정리된 코드(`code-cleanup`)에서 CPU로 실행했다. `B=2, N=1000, C=64, R=16`에서 학습 forward·backward
(분석 변환에 gradient 도달), `aux_loss`, `update(force=True)`, `compress→decompress`의 송신·수신 일치를 확인했다.
같은 Score 공간에서 `graph_gemm` NN 정렬이 bijection인 것도 확인했다. 실제 대상 모델에 통합하거나 RD 성능을
검증한 것은 아니며, 기존 학습 코드·weight·실행 job은 바꾸지 않았다.

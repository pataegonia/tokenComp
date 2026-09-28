# 심층 설명 — Context model / Nonlinear transform / Split

**대상 독자:** [EXPERIMENTS_OVERVIEW.md](EXPERIMENTS_OVERVIEW.md)를 읽고 세 핵심 구성요소를
코드 수준까지 이해하려는 사람. 구현을 수정하거나 새 변형을 설계할 사람.

**근거:** 아래 설명은 현재 활성 소스를 직접 읽고 쓴 것이다. 파일:줄 참조는 전부 실제 위치다.

| 참조 파일 | 역할 |
| --- | --- |
| [score_context.py](upstream/globalsplat/globalsplat/compression/score_context.py) | Context model 전체 + Split |
| [codec.py](upstream/globalsplat/globalsplat/compression/codec.py) | Nonlinear transform + 학습 scope + 송수신 |
| [entropy.py](upstream/globalsplat/globalsplat/compression/entropy.py) | `FactorizedScoreEntropy` 래퍼 |
| [archive/.../score_context.py](upstream/globalsplat/archive/codec_experiments_20260915/globalsplat/compression/score_context.py) | 은퇴한 변형(P1/P2, Shared/Gaussian/Conditional) 원본 |
| [archive/.../initialize_nfcgs_from_vanilla.py](upstream/globalsplat/archive/codec_experiments_20260915/scripts/initialize_nfcgs_from_vanilla.py) | Split 초기화(entropy 이식) 원본 |

**도해:** 아래 내용을 그림으로 옮긴 [NFC-GS 코덱 도해 다섯 장](https://claude.ai/artifact/VFLK2TcKisX6Uxc28YKt9T)
— 전체 파이프라인, 8-stream 스케줄, group 내부, Shared vs Split, Nonlinear 배치. 자세한 대응은 [§5.6](#56-실제로-그린-도해-2026-09-19).

---

## 0. 세 요소가 서로 어떻게 다른가 — 한 장 요약

이 셋은 **코덱의 서로 다른 층**을 건드린다. 혼동하면 실험 결과를 잘못 읽는다.

| | **Nonlinear transform** | **Context model** | **Split** |
| --- | --- | --- | --- |
| 건드리는 층 | 표현 (analysis/synthesis) | 양자화 격자 + 확률 모델 | **확률 모델만** |
| 복원값이 바뀌나 | **바뀐다** | **바뀐다** | **안 바뀐다** |
| 양자화 격자가 바뀌나 | 간접적 (score 분포가 바뀜) | **바뀐다** (`q` → `q·d`, 원점 `b` 이동) | 안 바뀐다 |
| score stream 수 | 1 (그대로) | 1 → **8** | 8 (그대로) |
| per-scene side info 추가 | 없음 | 없음 | 없음 |
| 추가 모델 크기 | MLP 2개 (작음) | +176~266 KiB | **+46~60 KiB** |
| 복호 속도 | 거의 동일 | +3~4% (전체 codec) | **−0.3% (측정 노이즈 수준)** |
| 측정된 이득 | −1.0~3.0 KiB (matched) | **−6.1~8.0%** (matched) | **−1.56~2.72%** |
| Break-even 장면 수 | — | 28~57 | **43~46** |
| 실험 격리 수준 | 낮음 (표현·rate 배분 동시 이동) | 낮음 | **높음** (화질 4자리까지 동일) |

**읽는 순서의 논리:**
Nonlinear은 "더 좋은 표현을 만든다", Context는 "같은 표현을 더 잘 예측한다",
Split은 "같은 예측을 더 잘 부호화한다". 세 층이 직교하므로 곱해서 쌓인다.

---

## 0.5 "mean"이 가리키는 다섯 가지 — 먼저 구분하고 들어가야 한다

이 코덱에서 "mean"이라는 말은 **서로 다른 다섯 개의 객체**를 가리킨다.
문서·코드·그림에서 전부 `mean`, `μ`, `median`으로 비슷하게 불려서 가장 자주 꼬이는 지점이다.

### ★ 먼저: "token mean"과 "channel mean"은 같은 것을 가리킨다

scene mean을 두고 문서마다 "per-scene token mean"(실험 16)과 "채널별 평균"이 섞여 쓰인다.
**둘은 반대말처럼 들리지만 같은 객체다.**

```python
mean = sorted_features.mean(dim=1)     # [B, 4096, 736] → [B, 736]
#                            ^^^^^ dim 1 = token 축
```

- **"token mean"** = *token 축으로* 평균냈다 → **무엇을 뭉갰는가**를 말한다
- **"channel mean"** = *채널마다* 하나씩 나왔다 → **결과가 무엇으로 인덱싱되는가**를 말한다

같은 연산의 두 측면이다. 혼동을 없애려면 **둘 중 하나만 쓰지 말고 둘 다 명시**한다:

> **토큰 축으로 평균낸 채널별 벡터** (`[B, 736]`, 4,096개 토큰이 공유)

### ★ 채널 축으로 평균내는 연산은 이 코덱에 **없다**

`globalsplat/compression/*.py` 전체에서 `.mean(` 호출은 두 곳뿐이고
([codec.py:185](upstream/globalsplat/globalsplat/compression/codec.py#L185),
[:242](upstream/globalsplat/globalsplat/compression/codec.py#L242)) 둘 다 `dim=1`(token 축)이다.

736개 채널을 뭉개서 토큰당 스칼라 하나를 만드는 연산은 **존재하지 않는다.**
"channel mean"을 "채널들의 평균"으로 읽으면 없는 것을 찾게 된다.

### 다섯 객체 전체 비교

| # | 객체 | 코드 | shape | 평균 축 | 무엇 당 하나 | 유효 범위 | 전송? | 학습? |
| ---: | --- | --- | --- | --- | --- | --- | :---: | :---: |
| 1 | **scene mean** `μ̂` | `features.mean(dim=1)` → `Q_FP16` | `[B, 736]` | **token 축** (4,096) | **채널** | **장면별** | ✅ **1,472 B** | ✗ (계산값) |
| 2 | **residual 정규화 통계** `m_E`, `s_E` | `register_buffer("residual_mean", …)` | `[1, 736, 1, 1]` | 데이터셋 전체 | 채널 | **전역** | ✗ | ✗ (buffer) |
| 3 | **entropy median** `m_g` | `EntropyBottleneck.quantiles[:, :, 1:2]` | `[r_g, 1, 1]` | **평균 아님** — 학습 밀도의 중앙값 | score 채널 | **전역** | ✗ | ✅ (scope=all일 때만) |
| 4 | **hyperprior 예측 평균** `μ_y` | `h_s(ẑ).chunk(2)` | `[B, 320, 1, 1024]` | **평균 아님** — 예측값 | **`y`의 원소 하나하나** | 장면별·**위치별** | ✗ (`ẑ`에서 재계산) | ✅ |
| 5 | **score offset** `b` | `f_μ(μ̂)` 앞 절반 | `[B, 56]` | **평균 아님** — 예측 offset | score 채널 | **장면별** | ✗ (`μ̂`에서 재계산) | ✅ |

### 하나씩

**① scene mean `μ̂` — 유일하게 실제로 전송되는 것**

[codec.py:174-186](upstream/globalsplat/globalsplat/compression/codec.py#L174-L186)

```python
mean = self._quantize_mean(sorted_features.mean(dim=1))   # [B, 736], FP16 + STE
centered = sorted_features - mean[:, None, :]             # broadcast over 4,096 tokens
```

- 토큰 축으로 평균 → 채널별 736개 값
- 장면마다 새로 계산한다
- **FP16으로 양자화**한 뒤 송신기도 그 복원값을 쓴다(안 그러면 수신기와 어긋남)
- `736 × 2 B = 1,472 B`를 실제로 보낸다 — payload의 1.77%
- centering에 쓰이고, **동시에 Mean conditioner의 유일한 입력**이다 → 여기서 `b`, `d`가 나온다

**② residual 정규화 통계 `m_E`, `s_E` — 전역 상수다**

[codec.py:61-62](upstream/globalsplat/globalsplat/compression/codec.py#L61-L62),
[initialization.py:227](upstream/globalsplat/globalsplat/compression/initialization.py#L227)

```python
self.register_buffer("residual_mean", torch.zeros(1, channels, 1, 1))
self.register_buffer("residual_std",  torch.ones(1, channels, 1, 1))
# PCA artifact에서 로드: source["residual_mean"].reshape(1, -1, 1, 1)
```

- 이름은 `residual_mean`이지만 **장면별 통계가 아니다.**
  PCA 초기화 artifact(`re10k_ctx12_s64_t512_rank56.pt`)가 채우는 **데이터셋 수준 상수**다.
- `register_buffer`라 **학습되지 않고**, checkpoint에 들어가며, **전송하지 않는다.**
- ①과 헷갈리기 가장 쉬운 항목이다. ①은 장면마다 다르고 전송되며, ②는 모든 장면에 같고 전송되지 않는다.

**③ entropy median `m_g` — 평균이 아니라 분위수다**

CompressAI `EntropyBottleneck`은 채널당 3개의 quantile을 갖고, `_get_medians()`가 가운데(index 1)를 뽑는다.

```python
def _get_medians(self):
    return self.quantiles[:, :, 1:2]     # [C, 1, 1]
```

- **데이터의 평균이 아니라 학습된 확률 밀도의 중앙값**이다.
- `K = round(V − m_g)`의 기준점 — 즉 **복원 lattice를 정의한다.**
- probability-only 학습에서 이걸 고정하는 게 Split 실험의 핵심 장치다
  ([§3.5](#35-복원-고정-scope--정확한-메커니즘과-함정)).
- Split에서 `EB_odd`의 `m_g`는 `EB_even`에서 복제한 뒤 고정 → **even과 odd가 같은 값**이다.

**④ hyperprior 예측 평균 `μ_y` — 원소마다 하나씩 있는 텐서다**

[residual.py:119-137](upstream/globalsplat/globalsplat/compression/residual.py#L119-L137)

```python
gaussian = self.h_s(z_hat)                        # [B, 640, 1, 1024]
scales_hat, means_hat = gaussian.chunk(2, dim=1)  # 각 [B, 320, 1, 1024]
y_hat, y_lik = self.gaussian_conditional(y, scales_hat, means=means_hat, …)
```

- 위 넷 중 **유일하게 위치별로 다른 값**이다. `y`의 원소 327,680개마다 평균이 하나씩 있다.
- 전송하지 않는다 — 수신기가 복호한 `ẑ`에서 `h_s`로 **재계산**한다.
- 이게 hyperprior의 본체다. "residual은 처음부터 conditional이었다"는 말이 가리키는 것.

**⑤ score offset `b` — Mean conditioner의 출력**

- `μ̂`(①)를 입력으로 받아 만들어지지만 **`μ̂`와 다른 객체다.**
- score 채널별·장면별 예측 offset. `d`(양자화 step 배수)와 한 쌍으로 나온다.
- 전송하지 않는다 — 수신기가 같은 `μ̂`와 같은 network로 재계산한다.
- **`b`와 `m_g`를 헷갈리면 안 된다.** `b`는 격자의 원점을 옮기고, `m_g`는 반올림 기준점이다.
  둘 다 `V = (U − b)/d` → `K = round(V − m_g)`에 동시에 등장한다.

### ⚠️ 기호 충돌: `μ̂`와 `μ`

기존 아키텍처 그림은 scene mean을 `μ̂`로, hyperprior 예측 평균을 `μ`로 쓴다
(residual 경로의 `⊖μ` / `⊕μ` 원). **hat 하나 차이인데 완전히 다른 객체다.**

| | 그림 표기 | 권장 표기 | 정체 |
| --- | --- | --- | --- |
| scene mean | `μ̂` | `μ̂` (유지) | 토큰 축 평균, 채널별 736, **전송** |
| hyperprior 평균 | `μ` | **`μ_y`** | `y` 원소별 예측값, `ẑ`에서 재계산 |
| hyperprior scale | `σ` | **`σ_y`** | 같은 `h_s` 출력의 나머지 절반 |

그림에서 `μ` → `μ_y`, `σ` → `σ_y`로 바꾸는 것만으로 충돌이 사라진다.

### 코덱 밖에도 "평균"이 두 개 더 있다

혼동 방지용으로 적어둔다. 아래 둘은 codec 모듈 밖이고 위 다섯과 무관하다.

| 객체 | 코드 | 평균 축 | 결과 |
| --- | --- | --- | --- |
| **token center** (Morton 입력) | `(positions * weights).sum(dim=2)` | **M_max 축** (후보 16개) | `[B, 4096, 3]` — **토큰당 3-D 좌표 하나** |
| Morton 정규화 범위 | `positions.amin(dim=1)`, `amax(dim=1)` | token 축 | `[B, 1, 3]` — 좌표축별 min/max |

[gaussian_decoder.py:289-305](upstream/globalsplat/globalsplat/model/decoder/gaussian_decoder.py#L289-L305).
token center는 gate softmax **가중평균**이고, 평균내는 축이 또 다르다(토큰도 채널도 아닌 Gaussian 후보 축).

그리고 실험 문서의 "mean absolute off-diagonal correlation"(실험 16)이나
RGB MSE, BPGA의 "평균"은 지표 집계이지 코덱 내부 객체가 아니다.

---

# Part 1. Context model

## 1.1 출발점 — factorized entropy model이 못 하는 것

기본 score 부호화는 `FactorizedScoreEntropy(rank)`, 즉 CompressAI `EntropyBottleneck(56)` 하나다.
([entropy.py:9-21](upstream/globalsplat/globalsplat/compression/entropy.py#L9-L21))

```text
P_model(K) = ∏_{token i} ∏_{channel c} P_c(K[i, c])
```

여기서 `P_c`는 **채널 c마다 유연하게 학습되는 1-D 이산 분포**다. 즉:

- ✅ 채널마다 다른 분포를 쓴다. "모든 채널을 하나의 분포로 압축"하는 게 아니다.
- ✅ 분포 모양이 비모수적이다. Gaussian/Laplace로 고정돼 있지 않다.
- ❌ 같은 채널이면 **어느 token, 어느 scene이든 같은 분포**를 쓴다.
- ❌ 이미 복호한 다른 채널, 가까운 token, 현재 scene의 정보로 CDF를 바꾸는 경로가 없다.

> 이 factorization은 "score들이 실제로 독립"이라는 증명이 아니라
> **부호화 확률 모델이 채택한 가정**이다. 가정이 틀린 만큼 bit가 낭비된다.

Low-rank analysis가 상관을 줄이긴 하지만 모든 의존성을 제거한다고 보장할 수 없다.
그리고 세대 2 실험 16에서 **Morton 정렬 후 lag-1 토큰 상관이 0.4739**로 측정됐다
([EXPERIMENTS_OVERVIEW.md §7.2](EXPERIMENTS_OVERVIEW.md)). 남은 의존성이 실재한다는 뜻이다.

**핵심 제약: decoder-causality.**
side stream을 추가하지 않고 이득을 보려면, 조건 정보가 **수신기가 그 시점에 이미 가진 것**이어야 한다.
수신기가 가진 것은 세 종류뿐이다.

| # | 수신기가 이미 아는 것 | 이걸 쓰는 모듈 |
| --- | --- | --- |
| 1 | 먼저 전송된 **FP16 scene mean** (1,472 B) | **Mean** |
| 2 | 앞서 복호한 **channel group** | **Channel** |
| 3 | 같은 group에서 먼저 복호한 **Morton even token** | **Spatial** |

Full = 셋 다.

## 1.2 세 조건 정보가 각자 하는 일이 다르다

여기가 가장 많이 오해되는 지점이다. 세 모듈이 같은 일을 하는 게 아니다.

```text
# 기본형
K     = round(S - median)
S_hat = K + median

# Context 적용형
V     = (S - b) / d          ← b는 Mean+Channel+Spatial, d는 Mean만
K     = round(V - median)
S_hat = b + d · (K + median)
```

| 기호 | 누가 만드나 | 무슨 역할인가 | rate에 미치는 경로 |
| --- | --- | --- | --- |
| `b` | Mean offset + Channel 예측 + Spatial 예측 | **격자의 원점을 옮긴다** | 분포의 봉우리로 심볼을 모아 \|K\|를 줄인다 |
| `d` | Mean conditioner **만** | **격자의 간격을 바꾼다** | 간격을 늘리면 bit↓ 왜곡↑ — 장면별 RD 노브 |
| `median` | EntropyBottleneck | 확률 모델의 채널별 중심 | `b`와 **다른 항**이다. 혼동 금지 |
| `q` | `score_log_scale` (공유 학습) | 전역 채널별 정규화 | 원래 U 공간의 양자화 간격은 `q · d` |

**결론: `d` 때문에 context 실험은 "확률 모델만 바꾼 실험"이 아니다.**
장면별로 양자화 간격이 변하므로 복원값·residual·rate 배분이 모두 함께 움직인다.
이 혼입을 제거하려고 나중에 [Part 3](#part-3-split)의 probability-only 실험을 따로 했다.

## 1.3 Mean conditioner — 이미 보낸 평균으로 `b`와 `d`를 예측

[score_context.py:42-49, 87-91](upstream/globalsplat/globalsplat/compression/score_context.py#L42-L91)

```python
self.mean_conditioner = nn.Sequential(
    nn.LayerNorm(self.scene_channels),       # 736
    nn.Linear(self.scene_channels, hidden),  # 736 → 64
    nn.GELU(),
    nn.Linear(hidden, 2 * self.rank),        # 64 → 112
)
nn.init.zeros_(self.mean_conditioner[-1].weight)
nn.init.zeros_(self.mean_conditioner[-1].bias)

def _scene_parameters(self, scene_mean):
    parameters = self.mean_conditioner(scene_mean)
    offset, log_step = parameters.chunk(2, dim=-1)   # 112 → 56 + 56
    step = torch.exp(2.0 * torch.tanh(0.5 * log_step))
    return (offset, step)
```

**출력 112개 = offset 56 + log_step 56.**

**`step` 범위 제한의 이유.** `exp(2·tanh(log_d/2))`는 `log_d → ±∞`에서 `exp(±2)`로 포화한다.

```text
d ∈ [exp(-2), exp(+2)] = [0.1353, 7.3891]
```

주석이 이유를 명시한다(아카이브 버전): *"A bounded range prevents early training from
creating nearly-zero quantization steps or exploding residual symbols."*
`d`가 0에 가까워지면 `V = (S-b)/d`가 폭발해서 심볼이 무한히 커지고, rANS가 터진다.

**0-init의 효과.** 마지막 Linear의 weight와 bias를 모두 0으로 두면:

```text
offset   = 0
log_step = 0  →  d = exp(2·tanh(0)) = exp(0) = 1
```

즉 **`b=0, d=1`로 시작해서 부모 모델과 정확히 같은 함수에서 출발**한다.
새 모듈을 붙였다는 사실 자체로 성능이 변하지 않는다.

**핵심 속성 3개**

1. `d`는 **scene마다, channel마다 다르지만 같은 scene·channel 안의 4,096 token에는 모두 같다.**
   토큰별 적응이 아니라 장면별 적응이다.
2. 송수신기가 **같은 FP16 복원 mean**과 **같은 공유 network**를 쓰므로
   `b`/`d`를 별도 stream으로 보내지 않는다. 추가 per-scene 비용이 **0 B**다.
3. 기존 mean 1,472 B는 계속 전송한다. context가 mean 전송을 새로 유발한 게 아니다.
   → 그래서 overhead profile에서 FP16 mean을 context 비용으로 부과하지 않는다.

**송신기도 FP16 mean을 써야 한다.** [codec.py:174-180](upstream/globalsplat/globalsplat/compression/codec.py#L174-L180)

```python
@staticmethod
def _quantize_mean(mean: Tensor) -> Tensor:
    """FP16 side-information quantizer with an STE during training."""
    quantized = mean.to(torch.float16).to(mean.dtype)
    if torch.is_grad_enabled():
        return mean + (quantized - mean).detach()   # straight-through
    return quantized
```

FP32 mean으로 `b`/`d`를 계산하면 수신기와 값이 달라져 복호가 깨진다.
학습 시에는 straight-through로 gradient를 흘린다.

## 1.4 Channel context — 앞 group으로 다음 group 예측

[score_context.py:33-57, 93-106](upstream/globalsplat/globalsplat/compression/score_context.py#L33-L106)

**group 분할.** rank 56, `slice_channels=16`:

```python
self.group_sizes = tuple(
    min(self.slice_channels, self.rank - start)
    for start in range(0, self.rank, self.slice_channels)
)
# rank 56 → (16, 16, 16, 8)
```

각 group은 **자기 `FactorizedScoreEntropy`를 갖는다** (공유하지 않는다).

**predictor 3개.**

```python
decoded = self.group_sizes[0]           # 16
for width in self.group_sizes[1:]:      # 16, 16, 8
    predictor = nn.Conv2d(decoded, width, 1)
    nn.init.zeros_(predictor.weight); nn.init.zeros_(predictor.bias)
    self.channel_predictors.append(predictor)
    decoded += width
```

| predictor | 입력 채널 | 출력 채널 | 무엇을 읽나 |
| --- | ---: | ---: | --- |
| — (group 0) | — | — | Mean offset만 |
| `channel_predictors[0]` | 16 | 16 | group 0 복원값 |
| `channel_predictors[1]` | 32 | 16 | group 0+1 복원값 |
| `channel_predictors[2]` | 48 | 8 | group 0+1+2 복원값 |

**base 계산:**

```python
def _group_base(self, decoded_groups, offset, start, width, points, group_index):
    base = offset[:, start:start+width, None, None].expand(-1, -1, 1, points)
    if group_index > 0:
        previous = torch.cat(decoded_groups, dim=1)          # 누적 복원값
        base = base + self.channel_predictors[group_index-1](previous)
    return base
```

**`1×1` conv라는 게 중요하다.**

- ✅ **같은 token 위치**의 모든 이전 group 채널을 읽는다 → cross-channel
- ❌ 주변 token은 **전혀 보지 않는다** → cross-token 아님
- group은 **직렬**이지만 group 안의 4,096 token predictor 계산은 **완전 병렬**

**입력이 "복원값"이라는 점도 중요하다.** `decoded_groups`에 들어가는 건
`_merge_even_odd(even_hat, odd_hat, points)`, 즉 **양자화·복호를 거친 값**이다
([score_context.py:196](upstream/globalsplat/globalsplat/compression/score_context.py#L196)).
원본 score가 아니다. 그래서 송수신기가 같은 값을 본다.

> **왜 Channel 단독 이득이 작았나 (−2.81% / −1.81%).**
> low-rank analysis 자체가 이미 채널 간 상관을 줄이는 변환이다.
> PCA류 basis 뒤에 남는 cross-channel 선형 의존성은 작고,
> `1×1` conv는 그 선형 성분만 잡을 수 있다.
> 반면 Spatial은 low-rank가 전혀 손대지 않은 **token 축**을 공격하므로 이득이 컸다(−5.21% / −7.30%).

## 1.5 Spatial context — even을 먼저 복호하고 odd를 예측

[score_context.py:58-65, 108-126](upstream/globalsplat/globalsplat/compression/score_context.py#L108-L126)

이게 context의 주력이다.

```python
predictor = nn.Conv2d(width, width, kernel_size=(1, 3), padding=(0, 1))
nn.init.zeros_(predictor.weight); nn.init.zeros_(predictor.bias)
```

hidden layer도, activation도 없다. **선형 conv 하나.** (P0 = "linear")

```python
@staticmethod
def _spatial_anchor_delta(anchor_hat, base):
    anchor_delta = torch.zeros_like(base)
    anchor_delta[..., 0::2] = anchor_hat - base[..., 0::2]   # ★ 원 값이 아니라 오차
    return anchor_delta                                      #   odd 위치는 0 유지

def _spatial_prediction(self, group_index, anchor_hat, base):
    anchor_delta = self._spatial_anchor_delta(anchor_hat, base)
    prediction = self.spatial_predictors[group_index](anchor_delta)
    return prediction[..., 1::2]                             # odd 위치만 취함
```

### ★ 설계 포인트 1: 입력이 "even 값"이 아니라 "even 오차"다

`anchor_delta[even] = even_hat − base[even]`.

즉 **Mean+Channel base가 이미 설명한 부분을 뺀 나머지**를 입력으로 쓴다.
이미 base로 예측된 성분을 다시 넣으면 predictor가 그걸 재학습해야 하므로 낭비다.
잔차만 넣으면 predictor는 "base가 놓친 지역적 구조"에만 집중한다.

### ★ 설계 포인트 2: odd 위치는 0으로 채운 상태로 conv를 돌린다

전체 길이 `points` 텐서에 conv를 한 번 돌리고 뒤에서 `[..., 1::2]`로 odd만 뽑는다.
`kernel=(1,3)`이므로 odd 위치 `i`에서 읽는 세 칸은:

```text
위치:      i-1        i          i+1
값:    even 오차    0 (odd)   even 오차
```

**odd 자기 값은 0이라서 아직 복호 안 된 정보가 절대 새지 않는다.** causality가 구조적으로 보장된다.
오른쪽 even(`i+1`)도 첫 pass에서 이미 복호됐으므로 써도 된다. → 양방향 anchor를 쓴다.

### ★ 설계 포인트 3: group 안에서만 채널이 섞인다

`Conv2d(width, width, ...)`는 **자기 group의 width 채널끼리만** 섞는다.
그리고 `base`가 group별로 잘려 들어오므로 `anchor_delta`도 group-local이다.

| 모듈 | cross-channel | cross-token | 범위 |
| --- | --- | --- | --- |
| Channel (`1×1`) | ✅ 이전 **모든** group | ❌ | 같은 token |
| Spatial (`1×3`) | ✅ **자기 group 안에서만** | ✅ ±1 | 같은 group |

→ **"다른 group의 공간적으로 떨어진 anchor"를 보는 경로는 없다.**
이게 P2(kernel 7)가 넓히려 한 것이고, entropy probe가 "초반 group·odd stream에 여유가 몰려 있다"고
가리킨 방향과 정확히 겹친다. 개선 여지가 남은 지점이다.

### 경계 처리

`padding=(0,1)`의 zero padding. `points=4096`일 때 마지막 odd 토큰(index 4095)은
왼쪽 even(4094), 자기(0), 오른쪽(padding 0)을 읽는다. 즉 **마지막 odd는 왼쪽 anchor만 본다.**

> ⚠️ entropy probe 문서의 *"Final odd repeats the final even anchor when its right neighbour
> does not exist"*는 **probe의 context binning 규칙**이고, 이 conv의 padding과 **다른 것**이다.

### 홀수 길이 token 처리

```python
even_count = math.ceil(points / 2)      # decompress 시
odd_count  = points // 2
```

[score_context.py:275](upstream/globalsplat/globalsplat/compression/score_context.py#L275),
[:287](upstream/globalsplat/globalsplat/compression/score_context.py#L287).
`_merge_even_odd`의 `[...,0::2]`/`[...,1::2]` 슬라이스가 자동으로 맞는 길이를 받는다.
테스트가 홀수 길이를 커버한다(`test_main_codec.py` — even/odd token count 양쪽 bitstream 일치).

## 1.6 8-stream 스케줄 — forward 트레이스

[score_context.py:165-201](upstream/globalsplat/globalsplat/compression/score_context.py#L165-L201)

rank 56, `points=4096` 기준 실제 실행 순서:

```text
offset, step = mean_conditioner(FP16_mean)        # [B,56], [B,56]

for g, width in enumerate((16, 16, 16, 8)):
    ┌─ base = offset[g슬라이스] (+ channel_predictors[g-1](앞 group 복원값))
    │  group_step = step[g슬라이스]                # [B,width,1,1]
    │
    ├─ EVEN pass  (2,048 token, 병렬)
    │    V_e     = (S[g, even] - base[even]) / group_step
    │    V̂_e     = EB_even[g](V_e)                 ← stream #(2g)
    │    ŝ_e     = base[even] + group_step · V̂_e
    │
    ├─ ODD pass   (2,048 token, 병렬)
    │    anchor_delta[even] = ŝ_e - base[even];  anchor_delta[odd] = 0
    │    odd_base = base[odd] + Conv1x3(anchor_delta)[odd]
    │    V_o      = (S[g, odd] - odd_base) / group_step
    │    V̂_o      = EB_odd[g](V_o)                 ← stream #(2g+1)   ★ Split
    │    ŝ_o      = odd_base + group_step · V̂_o
    │
    └─ decoded[g] = merge(ŝ_e, ŝ_o)                # 다음 group의 channel 입력
```

**전송 stream 8개:**

```text
g0-even → g0-odd → g1-even → g1-odd → g2-even → g2-odd → g3-even → g3-odd
```

### ★ 4,096-step autoregression이 아니다

| 직렬로 처리되는 것 | 병렬로 처리되는 것 |
| --- | --- |
| group 4개 (channel 의존) | group 안 2,048 token의 predictor 계산 |
| group 안 even → odd 2 pass | 같은 pass 안 모든 token의 entropy forward |

**총 직렬 단계 = 4 × 2 = 8.** 신경망 호출이 8번이다.
이것이 "GPU-efficient decoder-causal"의 의미다
([score_context.py:12-18](upstream/globalsplat/globalsplat/compression/score_context.py#L12-L18) docstring).

> ⚠️ 단, **rANS coder 내부까지 GPU 병렬이라는 뜻은 아니다.**
> CompressAI의 rANS는 CPU native 코드다. 실제 latency의 상당 부분이 거기 있다.

## 1.7 송신기도 복호를 해야 한다

[score_context.py:219-232](upstream/globalsplat/globalsplat/compression/score_context.py#L219-L232)

```python
encoded = entropy.compress(even_symbols)
strings.append(encoded[0])
even_hat = entropy.decompress(encoded, even_symbols.shape[-2:])   # ★ 송신기가 복호
even_hat = base[..., 0::2] + group_step * even_hat
odd_base = base[..., 1::2] + self._spatial_prediction(group_index, even_hat, base)
```

송신기가 `compress` 직후 **즉시 `decompress`**해서 그 값으로 odd base를 만든다.
"내가 보낸 게 아니라 상대가 받을 값"을 써야 하기 때문이다.

**성능 측정에 미치는 영향:** overhead profile의 score sender 타이밍은
이 로컬 복호를 **의도적으로 포함**한다. no-context factorized sender도
같은 compress-then-decompress를 하므로 비교 경계가 맞다
([EXPERIMENTS_OVERVIEW.md §9.2](EXPERIMENTS_OVERVIEW.md)).

## 1.8 학습 시 surrogate와 실제 부호화의 차이

| | 학습 (`training=True`) | 평가 / 실제 부호화 |
| --- | --- | --- |
| 양자화 | `V + Uniform(-0.5, 0.5)` (미분 가능 대체) | `round(V - median)` |
| 다음 group / odd predictor 입력 | **surrogate 복원값** | **entropy string 복호값** |
| likelihood | 연속 완화 | 정수 CDF |

`EntropyBottleneck(..., training=True)`이 noise를 더한 값을 그대로 반환하므로,
학습 중 `decoded[g]`와 `anchor_delta`에는 noise가 섞인 값이 들어간다.
실제 압축에서는 rANS 복호값이 들어간다. **이 mismatch는 의도된 것이고 표준 관행**이지만,
context predictor가 noise가 아니라 rounding 오차에 맞춰져야 한다는 미세한 gap이 남는다.

## 1.9 수치 일관성 장치 3개

송수신기가 **비트 단위로 같은 값**을 계산해야 복호가 성립한다. 세 장치가 이를 지킨다.

### (a) FP32 경계 — autocast 강제 해제

[codec.py:165-172](upstream/globalsplat/globalsplat/compression/codec.py#L165-L172)

```python
def _score_forward(self, score_nchw, mean, *, training):
    # Sender, receiver, and mixed-precision training must agree on context.
    with torch.autocast(device_type=score_nchw.device.type, enabled=False):
        return self.score_context(
            score_nchw.float(), mean.float(), self.score_entropy, training=training
        )
```

`compress`([:247](upstream/globalsplat/globalsplat/compression/codec.py#L247))와
`decompress`([:299](upstream/globalsplat/globalsplat/compression/codec.py#L299))에도 같은 블록이 있다.
**bf16-mixed 학습 중에도 score-context 내부는 FP32다.**
bf16은 mantissa가 8비트뿐이라 `(S-b)/d` 같은 연산에서 송수신 불일치가 생길 수 있다.

### (b) FP16 mean STE — [§1.3](#13-mean-conditioner--이미-보낸-평균으로-b와-d를-예측) 참조

### (c) RNG fork — 모듈 추가가 초기화를 오염시키지 않게

[codec.py:76-85](upstream/globalsplat/globalsplat/compression/codec.py#L76-L85)

```python
# Preserve initialization RNG boundaries from the reference codec.
with torch.random.fork_rng(devices=[]):
    self.analysis_mlp = self._make_mlp(channels, rank, config.transform_hidden)
    self.synthesis_mlp = self._make_mlp(rank, channels, config.transform_hidden)
with torch.random.fork_rng(devices=[]):
    self.score_context = ContextualScoreEntropy(...)
```

fork 없이 새 모듈을 만들면 **난수 소비량이 바뀌어서 뒤에 생성되는 모든 모듈의 초기값이 달라진다.**
그러면 "구조 하나만 바꿨다"는 ablation이 성립하지 않는다.
이건 재현성을 위한 실험 설계 장치다.

## 1.10 0-init — 모든 새 모듈이 항등에서 시작한다

| 모듈 | 0-init 대상 | 효과 |
| --- | --- | --- |
| `mean_conditioner[-1]` | weight + bias | `b=0, d=1` |
| `channel_predictors[*]` | weight + bias | channel 예측 = 0 |
| `spatial_predictors[*]` | weight + bias | spatial 예측 = 0 |
| (아카이브) `spatial_corrections[-1]` | weight + bias | P1/P2가 P0와 동일 |
| (아카이브) `spatial_scale_predictors[-1]` | weight + bias | Conditional이 Static Gaussian과 동일 |

**이식 직후 함수가 부모와 정확히 같다.** 학습 곡선이 부모 성능에서 출발한다.

> ⚠️ 단, **파일 바이트까지 부모와 같지는 않다.** stream이 1개에서 8개로 쪼개지고
> `SCCTX001` wrapper 88 B가 붙기 때문이다. rANS를 8번 나눠 돌리면 stream당
> 약 48 bit의 flush 비용이 생긴다(entropy probe 측정).

## 1.11 flag와 거부 로직 — 잘못된 복호를 구조적으로 막는다

**score container flag** ([score_context.py:20-24](upstream/globalsplat/globalsplat/compression/score_context.py#L20-L24))

| flag | 값 | 의미 |
| --- | ---: | --- |
| `FLAG_MEAN` | `1<<0` = 1 | Mean condition |
| `FLAG_CHANNEL` | `1<<1` = 2 | Channel context |
| `FLAG_SPATIAL` | `1<<2` = 4 | Spatial context |
| `FLAG_SPATIAL_ENTROPY_SPLIT` | `1<<5` = 32 | even/odd Split |

현재 활성 경로의 `flags` = 1+2+4+32 = **39**.
아카이브 버전에는 `RESIDUAL3`(8), `RESIDUAL7`(16), `GAUSSIAN`(64), `CONDITIONAL_SCALE`(128)도 있다.

**scene container flag** ([codec.py:47-49](upstream/globalsplat/globalsplat/compression/codec.py#L47-L49))

`FLAG_NONLINEAR`(4) | `FLAG_CONTEXTUAL_SCORE`(8) = **12**.

**3단 검증** ([score_context.py:251-266](upstream/globalsplat/globalsplat/compression/score_context.py#L251-L266))

```python
if packed.flags & ~self.KNOWN_FLAGS:
    raise ValueError("contextual score bitstream has unknown flags")
if (packed.rank != self.rank
        or packed.slice_channels != self.slice_channels
        or packed.flags != self.flags):
    raise ValueError("... does not match codec configuration")
expected_strings = len(self.group_sizes) * 2          # = 8
if len(packed.strings) != expected_strings:
    raise ValueError("... unexpected stream count")
```

그리고 [codec.py:286-293](upstream/globalsplat/globalsplat/compression/codec.py#L286-L293)에서
channel/rank/flag/residual 존재를 재검증한다.

> ⚠️ **flag는 설정을 식별하지 weight를 식별하지 않는다.**
> 같은 flag라도 다른 checkpoint면 복호 결과가 쓰레기다. 송수신기가 **동일 checkpoint**를 써야 한다.
> 이건 learned codec의 일반적 제약이다.

## 1.12 이 구조가 **하지 않는** 것 (오해 방지)

| 흔한 오해 | 실제 |
| --- | --- |
| "기존엔 압축을 안 했다" | 기존도 학습형 손실 압축이었다. low-rank + factorized entropy + residual hyperprior. |
| "기존 전체가 무조건부였다" | residual `y`는 처음부터 `z`에 조건부(hyperprior)였다. score만 무조건부였다. |
| "큰 조건부 확률 network로 교체했다" | P0은 **선형 conv 1개 + 1×1 conv 3개 + 작은 MLP 1개**다. 예측기 총 약 6만 파라미터. |
| "token마다 Gaussian mean/scale을 출력한다" | P0에는 그런 head가 **없다**. context는 `V`의 값/격자만 조절하고, 분포는 채널별 factorized model이 담당한다. (Gaussian head는 은퇴한 Gaussian/Conditional arm에만 있었다.) |
| "수신기가 raw XYZ를 안다" | **모른다.** Morton 정렬 순서를 소비할 뿐이다. XYZ를 조건으로 쓰려면 별도 side info나 다른 복호 설계가 필요하다. |
| "4,096-step autoregressive codec" | 직렬 단계 8개. 각 단계 안은 전부 병렬. |
| "Morton 이웃 = 이미지 픽셀 이웃 = 3D kNN" | 셋 다 다르다. Morton은 공간 채움 곡선이라 경계에서 멀리 점프한다. |
| "Full이 모든 지표에서 최상" | λ=0.0256에서 Mean+Spatial이 0.061 KiB 더 작다. 아주 가까운 두 RD 점이다. |

## 1.13 비용과 측정 이득 — 정리

### per-scene 포맷 overhead (λ=0.0064, 32장면 profile)

| Model | Streams | Wrapper B | Score payload Δ | 전체 scene Δ | 추가 모델 KiB | **Break-even** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factorized | 1 | 0 | 기준 | 기준 | +0 | — |
| mean | 1 | 32 | −1,403 B | −1,691 B | +253.0 | 184.6 |
| mean_channel | 4 | 56 | −3,071 B | −3,339 B | +172.6 | 57.5 |
| mean_spatial | 2 | 40 | −5,075 B | −5,507 B | +266.3 | 53.7 |
| **full** | 8 | 88 | **−6,409 B** | **−6,762 B** | +175.9 | **28.1** |

`Wrapper B` = `SCCTX001` 24 B prefix + stream당 8 B length. 8 stream = 24+64 = **88 B**.
이게 진짜 per-scene 포맷 세금이고, 절감(6,409 B)에 비해 1.4% 수준이다.

### 속도 (λ=0.0064)

| Model | Score encode | Score decode | 전체 codec encode | 전체 codec decode |
| --- | ---: | ---: | ---: | ---: |
| factorized | 48.82 ms | 30.23 ms | 133.90 ms | 78.92 ms |
| full | 53.32 (+9.22%) | 33.19 (+9.79%) | 137.98 (+3.05%) | 81.50 (+3.27%) |

score 경로 기준 +9~11%, 전체 codec 기준 +3~4%. **stream이 8개로 늘어난 rANS 호출 비용이 지배적**이고
신경망 예측기 비용은 작다(예측기가 tiny하므로).

### matched control 대비 rate 이득 (전체 6,991 장면)

| λ | Mean | Mean+Channel | Mean+Spatial | **Full** |
| --- | ---: | ---: | ---: | ---: |
| .0064 (Linear) | −1.08% | −2.81% | −5.21% | **−6.50%** |
| .0256 (Linear) | −1.54% | −1.81% | −7.30% | −7.16% |
| .0064 (Nonlinear) | — | — | −4.79% | **−6.09%** |
| .0256 (Nonlinear) | — | — | −7.32% | **−7.99%** |

**모든 조건에서 화질이 같거나 약간 올랐다.** rate만 줄인 게 아니라 RD가 개선됐다.
그리고 scene CI에서 rate 승률이 98.8~100%다 — 거의 모든 장면에서 더 작다.

---

# Part 2. Nonlinear transform

## 2.1 동기 — 단일 선형 map의 한계

low-rank 변환은 `U = X W_a^T`, `X_low = Û W_s`. 이건 **하나의 선형 부분공간 사영**이다.

세대 2 실험 17이 결정적 증거를 줬다:
**양자화 전에 이미 rank 56에서 −4.08 dB를 잃는다.** 양자화 손실은 −0.30 dB뿐이다.
즉 **병목은 entropy coding이 아니라 표현(투영) 용량**이다.

동시에 실험 16이 보여준 것: 736-D concat의 공분산 에너지는 rank 56에서 97.93%인데
appearance만 보면 91.00%다. **에너지 보존율이 높아도 렌더링에 중요한 저에너지 방향이 잘려나간다.**

→ "rank를 올린다"는 세대 2 실험 17에서 실패했다(rank 72/80은 residual payload만 커짐).
→ 대안: **rank는 유지하고 변환에 비선형 용량을 조금 준다.**

## 2.2 구현

[codec.py:88-96, 125-133](upstream/globalsplat/globalsplat/compression/codec.py#L88-L133)

```python
@staticmethod
def _make_mlp(inputs: int, outputs: int, hidden: int) -> nn.Sequential:
    mlp = nn.Sequential(
        nn.Linear(inputs, hidden, bias=False),
        nn.GELU(),
        nn.Linear(hidden, outputs, bias=False),
    )
    nn.init.zeros_(mlp[-1].weight)      # ★ 두 번째 Linear만 0-init
    return mlp

def _analyze_low_rank(self, centered: Tensor) -> Tensor:
    value = centered @ self.shared_basis.transpose(0, 1)
    value = value + self.analysis_mlp(centered)          # ★ 잔차 가산
    return value

def _synthesize_low_rank(self, score: Tensor) -> Tensor:
    value = score @ self.synthesis_basis
    value = value + self.synthesis_mlp(score)            # ★ 잔차 가산
    return value
```

```text
Nonlinear32:
  U     = X  W_a^T + MLP_a(X)         MLP_a: 736 → 32 → GELU → 56
  X_low = Û  W_s   + MLP_s(Û)         MLP_s:  56 → 32 → GELU → 736
```

파라미터 수: `MLP_a` = 736·32 + 32·56 = 25,344. `MLP_s` = 56·32 + 32·736 = 25,344.
합계 약 **5만 개**. 전체 코덱에서 아주 작다.

## 2.3 미묘하지만 중요한 세 가지

### (a) `synthesis_mlp`는 **양자화된 뒤**의 값을 받는다

[codec.py:193-195](upstream/globalsplat/globalsplat/compression/codec.py#L193-L195)

```python
score_hat = score_hat_nchw.squeeze(2).transpose(1, 2)
score_hat = score_hat * self.score_scale[None, None, :]   # q 곱해 U 스케일로 복귀
low_rank = self._synthesize_low_rank(score_hat)           # ← 복원값 입력
```

`MLP_s`의 입력은 원본 `U`가 아니라 **복호된 `Û`**다. 따라서:

- ✅ **수신기가 정확히 같은 계산을 재현할 수 있다.** 순수 decoder-side 비선형이다.
- ✅ side information이 전혀 필요 없다.
- ⚠️ 학습 시에는 `Û`가 noise surrogate라, 실제 압축의 rounding 오차와 미세한 분포 차이가 있다.

이 구조 때문에 `MLP_s`는 "양자화 오차가 섞인 score에서 736-D를 더 잘 복원하는 법"을 학습한다.
단순한 표현력 증가가 아니라 **양자화 오차에 대한 보정기 역할도 겸한다.**

### (b) `bias=False`

두 Linear 모두 bias가 없다. bias가 있으면 상수 offset이 생기는데,
그건 이미 scene mean centering과 Mean conditioner의 `b`가 담당한다. 중복이라 제거했다.

### (c) 두 번째 Linear만 0-init → **시작 함수가 Linear와 정확히 동일**

`mlp[-1].weight = 0`이면 `MLP(x) = 0` for all x.
따라서 초기화 직후 `U = X W_a^T`, `X_low = Û W_s`로 **순수 Linear codec과 같다.**

첫 Linear는 orthogonal/기본 초기화를 유지하므로 **gradient는 즉시 흐른다.**
(둘 다 0으로 두면 gradient가 0이 되어 학습이 안 된다. 한쪽만 0으로 두는 게 핵심이다.)

### (d) RNG fork — [§1.9(c)](#c-rng-fork--모듈-추가가-초기화를-오염시키지-않게) 참조

## 2.4 untied basis — 잊기 쉬운 전제

[codec.py:56-59](upstream/globalsplat/globalsplat/compression/codec.py#L56-L59)

```python
basis = torch.empty(rank, channels)
nn.init.orthogonal_(basis)
self.shared_basis = nn.Parameter(basis)
self.shared_synthesis_basis = nn.Parameter(basis.clone())   # ★ 별도 Parameter
```

- **초기값은 같지만(`clone`) 별개의 학습 파라미터다.** `W_s ≠ W_a^T`로 갈라질 수 있다.
- PCA 초기화 artifact(`re10k_ctx12_s64_t512_rank56.pt`)를 쓰면 여기에
  geometry basis / analysis·synthesis basis / score scale / residual mean·std를 덮어쓴다.
- `residual_mean` / `residual_std`는 `register_buffer`라 **학습되지 않는다**
  ([codec.py:61-62](upstream/globalsplat/globalsplat/compression/codec.py#L61-L62)).

세대 1에 tied vs untied vs untied+scale을 rank별로 비교한 설정이 다 있다
(`shared_lowrank_rank48_multiscale1d` / `..._untied_synthesis_...` / `..._untied_scaled_synthesis_...`).
현재 경로는 **untied + 양쪽 nonlinear**로 정착했다.

## 2.5 증거 두 개 — 성격이 다르다

### (a) 신규 학습 비교 (세대 2 실험 14) — 같은 예산에서 처음부터

| 조건 | PSNR | SSIM | LPIPS | Bytes/scene |
| --- | ---: | ---: | ---: | ---: |
| `linear_no_residual` | 20.5875 | 0.6347 | 0.3990 | 55,625 |
| `nonlinear32_no_residual` | **21.4757** | **0.6649** | **0.3387** | **51,391** |

**전면 우세:** +0.8882 dB, +0.0302 SSIM, −0.0603 LPIPS, −4.23 KB.
같은 rank/λ/optimizer/50k step에서 나온 결과다. 이게 가장 깨끗한 근거다.

### (b) matched continuation 비교 (세대 3) — 기존 모델에 이식했을 때

| λ | Linear factorized 100k | Nonlinear32 factorized 100k | 차이 |
| --- | ---: | ---: | --- |
| .0064 | 90.805 KiB / 24.3024 | 87.840 KiB / 24.3022 | **−2.965 KiB**, PSNR ≈ 동일 |
| .0256 | 42.258 KiB / 23.4875 | 41.256 KiB / 23.5210 | **−1.002 KiB, +0.0335 dB** |

scene CI 2점 BD-rate: **−4.322%** [−4.458, −4.190].

> ⚠️ 두 결과의 효과 크기가 꽤 다르다(+0.89 dB vs ≈0 dB).
> (a)는 residual 없는 구성에서 처음부터 학습, (b)는 residual 있는 구성에서 이미 학습된 모델 비교다.
> **residual 경로가 있으면 nonlinear의 표현 이득 상당 부분이 residual과 겹친다**는 해석이 자연스럽다.
> 다만 이건 이 두 결과로부터의 추론이고, 직접 검증한 실험은 없다.

## 2.6 Nonlinear이 해결하지 **못한** 것

- **투영이 여전히 주 병목이다.** 실험 17의 −4.08 dB(rank 56, 양자화 전)는
  nonlinear를 넣어도 구조적으로 해소되지 않았다. hidden 32의 MLP로는 부족하다.
- `transform_hidden`은 32로 고정돼 있다. 세대 1에 nonlinear 64 설정
  (`..._untied_nonlinear64_synthesis_...`)이 여럿 있는데, 세대 3은 32만 썼다.
  **hidden 크기 sweep은 현재 계보에서 미검증이다.**
- residual 경로를 대체하지 못한다. residual OFF는 여전히 명확히 나쁘다(§8.3(c) 표).

---

# Part 3. Split

## 3.1 한 문장 정의

> **Split = Morton even token과 odd token에 서로 다른 marginal 확률분포를 허용한다.
> 그 외의 어떤 것도 바꾸지 않는다.**

바뀌는 것: `V_o`의 likelihood와 arithmetic coder의 CDF. **끝이다.**

바뀌지 않는 것:

- analysis/synthesis basis, MLP
- score normalization `q`
- Mean conditioner의 `b`, `d`
- channel predictor, spatial **value** predictor `P_linear_k3`
- entropy median / quantile (= 복원 lattice)
- residual codec 전체
- stream 수(8개), pass 수, 복호 순서
- per-scene side information (없음 → 없음)

## 3.2 왜 even과 odd의 분포가 다를 것이라고 기대하나

이게 Split의 전부이고, 논리가 단순해서 강하다.

```text
even 심볼:  V_e = (S_e - base_e) / d
            base_e = Mean offset + Channel 예측

odd  심볼:  V_o = (S_o - base_o) / d
            base_o = Mean offset + Channel 예측 + Spatial 예측(복호된 even anchor)
```

**odd는 예측을 하나 더 받는다.** 그러면:

| | even | odd |
| --- | --- | --- |
| 예측 소스 | mean + channel | mean + channel **+ spatial anchor** |
| 잔차의 성격 | "설명 안 된 원 신호" | "spatial predictor가 놓친 나머지" |
| 기대 분산 | 상대적으로 큼 | **더 작음** (예측이 잘 되면) |
| 기대 분포 모양 | — | **더 뾰족함** |

그런데 초기 Spatial/Full 구현은 **같은 group의 even/odd가 같은 `EntropyBottleneck`을 공유**했다.
즉 분산이 다른 두 분포를 **하나의 CDF로 부호화**하고 있었다.
이건 정보이론적으로 명백한 손실이다 — 두 분포의 혼합에 대한 cross-entropy를 지불한다.

**Split은 이 mismatch만 제거한다.** 새 정보를 쓰는 게 아니라, 이미 있던 구조적 비대칭을 인정하는 것이다.

> 그래서 Split의 이득이 예측 가능했고, 실제로 두 λ에서 **일관되게** 나왔다(−1.56% / −2.72%).
> 반면 P1/P2(predictor 확대)는 "예측을 더 잘하면 좋아질 것"이라는 약한 가설이었고 일관성이 없었다.

## 3.3 구현 — 놀랄 만큼 작다

[score_context.py:58-65](upstream/globalsplat/globalsplat/compression/score_context.py#L58-L65)

```python
self.spatial_predictors = nn.ModuleList()
self.spatial_odd_entropies = nn.ModuleList()
for width in self.group_sizes:                 # (16, 16, 16, 8)
    predictor = nn.Conv2d(width, width, kernel_size=(1, 3), padding=(0, 1))
    nn.init.zeros_(predictor.weight); nn.init.zeros_(predictor.bias)
    self.spatial_predictors.append(predictor)
    self.spatial_odd_entropies.append(FactorizedScoreEntropy(width))   # ★ Split
```

세 지점에서만 쓰인다.

```python
# forward  (:138)
def _odd_forward(self, symbols, even_entropy, group_index, anchor_hat, base, *, training):
    return self.spatial_odd_entropies[group_index](symbols, training=training)

# compress (:149-151)
entropy = self.spatial_odd_entropies[group_index]
strings = entropy.compress(symbols)
return (strings[0], entropy.decompress(strings, symbols.shape[-2:]))

# decompress (:163)
return self.spatial_odd_entropies[group_index].decompress([string], shape)
```

**even 쪽은 손대지 않는다.** `_entropy(base_entropy, group_index)`가 반환하는
`group_entropies[group_index]`를 그대로 쓴다.

활성 entropy model 목록 ([:81-85](upstream/globalsplat/globalsplat/compression/score_context.py#L81-L85)):

```python
def active_entropies(self, base_entropy):
    even = tuple(self.group_entropies)              # 4개
    return even + tuple(self.spatial_odd_entropies) # + 4개 = 8개
```

이 8개가 `update()`(CDF 재생성)와 `aux_loss()`(quantile 정칙화)의 대상이다
([:298-318](upstream/globalsplat/globalsplat/compression/score_context.py#L298-L318)).

**파라미터 증가:** Shared 3,416 → Split 6,832. 정확히 2배.
(`EntropyBottleneck`의 density 파라미터는 채널 수에 비례한다. 56채널 × 두 벌.)

## 3.4 초기화 — step 0에서 부모와 정확히 동일함을 보장

[archive/.../initialize_nfcgs_from_vanilla.py:305-333](upstream/globalsplat/archive/codec_experiments_20260915/scripts/initialize_nfcgs_from_vanilla.py#L305-L333)

```python
# A split model starts as an exact probability/reconstruction copy of
# its shared parent: duplicate each trained even/shared entropy model
# into the new odd-pass model.  CDF buffers are rebuilt from the copied
# density parameters and fixed quantiles by codec.update().
if args.score_spatial_entropy == "split" and str(
        source_config.get("score_spatial_entropy", "shared")) == "shared":
    for group_index in range(...):
        source_prefix = f"...group_entropies.{group_index}.entropy_bottleneck."
        target_prefix = f"...spatial_odd_entropies.{group_index}.entropy_bottleneck."
        for target_key, target_value in list(target.items()):
            if not target_key.startswith(target_prefix):
                continue
            suffix = target_key[len(target_prefix):]
            if suffix.startswith("_"):        # ★ CDF buffer 제외
                continue
            source_value = normalized_source.get(source_prefix + suffix)
            if torch.is_tensor(source_value) and source_value.shape == target_value.shape:
                target[target_key] = source_value.to(dtype=target_value.dtype)
```

### 복사되는 것 / 안 되는 것

| state key | `_` 시작? | 복사 | 이유 |
| --- | --- | --- | --- |
| `_matrix*`, `_bias*`, `_factor*` (density) | 아니오 | ✅ | 분포 모양 그대로 승계 |
| `quantiles` | 아니오 | ✅ | **복원 lattice 동일 유지** |
| `_quantized_cdf`, `_offset`, `_cdf_length` | **예** | ❌ | `codec.update()`가 density+quantile에서 **재생성** |

CDF buffer를 복사하지 않는 이유가 주석에 있다(channel context 쪽, :278-281):
*"CDF buffers are rebuilt during training; copying their non-empty saved shapes into
fresh modules would require resizing buffers before this initializer's strict load."*
새 모듈의 buffer는 빈 shape로 만들어지므로 strict load가 깨진다.

### 왜 이 초기화가 결정적인가

```text
step 0:  EB_odd[g].density == EB_even[g].density
         EB_odd[g].quantiles == EB_even[g].quantiles
      →  P_odd(V) == P_even(V)  for all V
      →  Shared와 동일한 확률, 동일한 복원, 동일한 bitstream 길이
```

**따라서 학습 후에 관측되는 모든 차이는 순수하게 "even/odd marginal이 갈라진 효과"다.**
초기화 우위나 구조 리셋 효과가 섞여 있지 않다.

이게 Static Gaussian과 결정적으로 다른 점이다. Static Gaussian은
부모의 유연한 odd marginal을 **고정 mean unit Gaussian으로 reset**하므로
step 0에서 이미 부모보다 나쁘다. 그래서 Static Gaussian의 실패에는
"Gaussian family의 한계"와 "분포 리셋 + 10k만의 재적응"이 섞여 있다.

### 조건부 적용

`source_config["score_spatial_entropy"] == "shared"`일 때만 이식한다.
이미 Split인 부모는 자기 `spatial_odd_entropies`를 갖고 있으므로 덮어쓰면 안 된다.
(channel context 이식도 마찬가지로 `not source_context["score_channel_context"]` 조건이 붙는다.)

## 3.5 복원 고정 scope — 정확한 메커니즘과 함정

[codec.py:148-163](upstream/globalsplat/globalsplat/compression/codec.py#L148-L163)

```python
def set_trainable_scope(self, scope: str = "all") -> None:
    """Train the full codec, or fit probability densities with fixed reconstruction."""
    if scope == "all":
        self.set_trainable(True)
        return
    if scope != "score_probability":
        raise ValueError("feature_codec_train_scope must be all or score_probability")
    for parameter in self.parameters():
        parameter.requires_grad_(False)                       # ① 전부 freeze
    # Quantiles define the reconstruction lattice and must stay fixed.
    for entropy in self.score_context.active_entropies(self.score_entropy):
        for parameter in entropy.parameters():
            parameter.requires_grad_(True)                    # ② 8개 EB만 해제
        entropy.entropy_bottleneck.quantiles.requires_grad_(False)   # ③ quantile 재freeze
```

**3단 구조:** 전부 잠근다 → 8개 EntropyBottleneck만 연다 → 그 안에서 quantile만 다시 잠근다.

### ★ 함정: `requires_grad=False`만으로는 충분하지 않다

기존 학습 recipe에는 optimizer와 **무관한** deterministic quantile update가 있다.
`no_grad` 컨텍스트에서 quantile을 직접 이동시키는 코드다(500 step 간격).
`requires_grad=False`는 gradient 전파만 막고 이 직접 호출은 막지 못한다.

**그래서 config에서 `loss.quantile_update_interval=0`도 함께 지정해야 한다.**

| 장치 | 막는 것 |
| --- | --- |
| `quantiles.requires_grad_(False)` | optimizer gradient에 의한 이동 |
| `quantile_update_interval=0` | deterministic update 루틴에 의한 이동 |

둘 중 하나만 하면 median이 움직여서 **복원 lattice가 바뀌고, 실험의 격리가 깨진다.**
이 실험 설계에서 가장 실수하기 쉬운 지점이다.

### CDF는 갱신하되 quantile은 유지

부호화를 위한 정수 CDF table은 학습된 density에서 매번 다시 만들어야 한다
(`update()` → [score_context.py:298-311](upstream/globalsplat/globalsplat/compression/score_context.py#L298-L311)).
`update(force=..., update_quantiles=False)`로 CDF만 갱신한다.

평가 시에는 `model.feature_codec.update(force=True)`를 호출하는데,
이것도 `update_quantiles` 기본값이 `False`라 checkpoint의 quantile을 보존한다
([NFCGS_CODEC.md](upstream/globalsplat/docs/NFCGS_CODEC.md)의 API 예시).

### render loss는 계속 계산되지만 흐르지 않는다

wrapper는 여전히 render/consistency/RD loss를 전부 계산한다.
그러나 복원에 관여하는 파라미터가 모두 frozen이므로
**optimizer에 도달하는 유효 gradient는 probability rate 쪽뿐**이다.

λ는 이 단계에서 "부모 operating point를 식별하는 값"일 뿐이고,
새로운 distortion-rate 표현을 찾는 joint 학습이 아니다.

### 검증

- 한 optimizer step 뒤 eval reconstruction이 **bit-exact**로 유지되는 테스트를 네 arm 모두에 수행
  ([test_main_codec.py:202](upstream/globalsplat/tests/test_main_codec.py#L202)
  `test_probability_step_keeps_reconstruction_and_quantiles_fixed`)
- Conditional arm에는 causality 테스트가 따로 있다:
  odd 위치의 미복호 값을 바꿔도 예측 scale이 변하지 않음

## 3.6 결과 — 격리 논증

### 학습 설정 (네 arm 완전 동일)

| 항목 | 값 |
| --- | --- |
| 부모 | 원래 Nonlinear32 Full P0 100k (run 20260910_092000 task 8/9) |
| 추가 | **10k**, micro2 × accum4, seed 111123 |
| LR | 1e-4 → step 7k에서 1e-5 |
| scope | `score_probability`, `quantile_update_interval=0` |
| Job | train 422558 → eval 422559, run 20260913_112316 |
| 평가 | full TEST 6,991 장면, 실제 bitstream |

### 결과

| λ | Probability model | 전체 KiB | Score B | PSNR | SSIM | LPIPS | Shared 대비 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .0064 | Shared | 82.463 | 70,924.906 | 24.3238 | 0.7611 | 0.2371 | 기준 |
| .0064 | **Split** | 81.174 | 69,604.342 | 24.3238 | 0.7611 | 0.2371 | **−1.56%** |
| .0064 | Static Gaussian | 88.167 | 76,765.617 | 24.3238 | 0.7611 | 0.2371 | +6.92% |
| .0064 | Conditional Scale | **81.135** | 69,565.374 | 24.3238 | 0.7611 | 0.2371 | −1.61% |
| .0256 | Shared | 37.952 | 31,149.344 | 23.5707 | 0.7342 | 0.2683 | 기준 |
| .0256 | **Split** | **36.920** | **30,093.048** | 23.5707 | 0.7342 | 0.2683 | **−2.72%** |
| .0256 | Static Gaussian | 44.518 | 37,873.270 | 23.5707 | 0.7342 | 0.2683 | +17.30% |
| .0256 | Conditional Scale | 37.021 | 30,195.839 | 23.5707 | 0.7342 | 0.2683 | −2.45% |

### ★ 격리가 성립했다는 증거 세 겹

**① 화질이 소수 넷째 자리까지 네 arm 동일.** PSNR/SSIM/LPIPS 전부.

**② residual y/z, mean, container가 로그 정밀도 내 완전 동일.**

| λ | Residual y B | Residual z B | Mean B | Container B |
| --- | ---: | ---: | ---: | ---: |
| .0064 (4 arm 전부) | 11,738.844 | 174.507 | 1,472 | 132 |
| .0256 (4 arm 전부) | 5,998.320 | 111.128 | 1,472 | 132 |

→ **전체 바이트 차이가 100% score 바이트 차이로 설명된다.**

**③ Shared 재학습 자체의 효과가 무시할 만하다.**
Shared는 부모 대비 `.0064`에서 **28.87 B**, `.0256`에서 **6.24 B**만 줄였다.
반면 Split은 Shared 대비 1,320.6 B / 1,056.3 B를 줄였다.

> **∴ Split의 이득은 "10k 더 돌려서"가 아니다. even/odd marginal 분리의 효과다.**
> 이게 이 실험이 P0/P1/P2 실험과 근본적으로 다른 점이다.
> P 실험에서는 predictor를 바꾸지 않은 P0조차 부모에서 −3,405 B / −0.16 dB 움직였다.

> ⚠️ 단서: 집계 화질 일치는 **서버 전체 장면의 텐서 bit-exact 검증과 다르다.**
> 로컬 round-trip/invariance 테스트와 구현이 복원 고정을 뒷받침하고 집계 수치도 부합하지만,
> 모든 장면의 복원 텐서를 직접 비교한 서버 증거는 이 로그에 없다.

## 3.7 Gaussian / Conditional과의 코드 수준 비교

은퇴한 두 arm의 구현
([archive/.../score_context.py:105-160, 245-266](upstream/globalsplat/archive/codec_experiments_20260915/globalsplat/compression/score_context.py#L105-L266)).

### Static Gaussian

```python
# 초기화: sigma가 정확히 1.0에서 시작하도록 역산
initial = math.log(math.expm1(1.0 - 0.11))     # softplus(initial) = 0.89
self.spatial_log_scales.append(nn.Parameter(torch.full((width,), initial)))
#  → softplus(initial) + 0.11 = 1.0
```

```python
def _spatial_probability_scale(self, group_index, anchor_hat, base):
    base_scale = F.softplus(self.spatial_log_scales[group_index]) + 0.11
    base_scale = base_scale[None, :, None, None]
    if self.spatial_entropy == "gaussian":
        return base_scale.expand(anchor_hat.shape[0], -1, 1, base.shape[-1] // 2)
    ...

@staticmethod
def _entropy_medians(entropy):
    medians = entropy.entropy_bottleneck._get_medians()    # ★ even EB의 median
    if medians.ndim == 3:
        medians = medians.unsqueeze(0)
    return medians

# _odd_forward:
scales = self._spatial_probability_scale(...)
medians = self._entropy_medians(even_entropy)              # ★ mu = even median (고정)
return self.spatial_gaussian(symbols, scales, means=medians, training=training)
```

| 속성 | 값 |
| --- | --- |
| `mu` | **고정된 even EB median.** 학습 안 함 → 복원 lattice 유지 |
| `sigma` | group/channel마다 **하나**. 총 56개. scene·token·anchor와 무관 |
| 초기 `sigma` | 정확히 1.0 |
| coding index | `[0.11, 256]` 구간 **64단계 logarithmic scale table** |
| 파라미터 | 3,472 (Shared 3,416 + 56) |

**왜 실패했나 — 원인이 넷 섞여 있다**

1. 부모의 **유연한 비모수 odd marginal을 Gaussian으로 교체**(분포 reset)
2. mean을 even median에 **고정**(자유 학습 아님)
3. **unit-scale 초기화** — 실제 `V_o` 분산과 무관한 출발점
4. **10k만 적응** + **finite 64단계 scale table** 양자화

> 그래서 결론은 *"현재의 고정 median·unit init·10k·finite table 조건에서 부적합했다"*까지가 타당하다.
> **"Gaussian 분포 자체가 틀렸다"거나 "조건부 확률 모델은 불필요하다"는 결론이 아니다.**

### Conditional Scale

```python
scale_predictor = nn.Sequential(
    nn.Conv2d(width, self.spatial_hidden, kernel_size=(1, kernel), padding=(0, kernel//2)),
    nn.GELU(),
    nn.Conv2d(self.spatial_hidden, width, kernel_size=1),
)
nn.init.zeros_(scale_predictor[-1].weight)   # ★ 0-init
nn.init.zeros_(scale_predictor[-1].bias)
```

`spatial_predictor == "linear"`(P0)이면 `kernel = 3`, `spatial_hidden = 32`.

```python
anchor_delta = self._spatial_anchor_delta(anchor_hat, base)
log_multiplier = self.spatial_scale_predictors[group_index](anchor_delta)
# Bound the local correction while allowing a useful 1/7.4x..7.4x
# range around each learned per-channel base scale.
scale = base_scale * torch.exp(2.0 * torch.tanh(0.5 * log_multiplier))
return scale[..., 1::2].clamp(min=0.11, max=256.0)
```

- 0-init → `log_multiplier = 0` → `scale = base_scale`
  → **Static Gaussian과 같은 초기 확률 모델**에서 출발
- multiplier 범위 `exp(±2)` = 1/7.389× ~ 7.389×
- 입력은 `anchor_delta`(복호된 even만, odd는 0) → causal
- **scale map은 전송하지 않는다.** 수신기가 재계산한다.
- `_spatial_prediction`(값 예측)과 **같은 입력을 쓰지만 별도 network**다.
  값 예측기는 `b`를, scale 예측기는 `σ`를 만든다.

### 네 arm 비교표

| Arm | Even 확률 | Odd 확률 | Odd 중심 | Odd scale/context | step0 = 부모? | Streams | 파라미터 |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| Shared | EB per group | **같은 EB 공유** | EB median | 위치 무관 learned density | ✅ | 8 | 3,416 |
| **Split** | EB_even | **독립 EB_odd** | 각 EB median | 위치 무관 independent density | ✅ (복제) | 8 | 6,832 |
| Static Gaussian | EB_even | GaussianConditional | 고정 EB_even median | group/channel static σ | ❌ reset | 8 | 3,472 |
| Conditional Scale | EB_even | GaussianConditional | 고정 EB_even median | decoded anchor → 위치별 σ | ❌ (=Gaussian) | 8 | 10,824 |

## 3.8 비용 — Split이 압도적으로 유리한 이유

overhead profile (32 paired TEST 장면, 동일 GPU·동일 장면 순서)

### λ=0.0064

| Model | 전체 scene Δ | **추가 모델 KiB** | **Break-even 장면** | Score encode | Score decode | Full codec decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 기준 | +0.00 | — | 53.41 ms | 33.14 ms | 81.92 ms |
| **split** | **−1,338.50 B** | **+60.09** | **46.0** | **−0.12%** | **+0.09%** | **−0.29%** |
| gaussian | +5,573.50 B | +784.23 | n/a | +87.59% | +78.15% | +31.25% |
| conditional_scale | −1,448.12 B | +812.95 | **574.9** | +87.61% | +78.92% | +32.31% |

### λ=0.0256

| Model | 전체 scene Δ | 추가 모델 KiB | Break-even | Score encode | Full codec decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| **split** | **−1,078.12 B** | **+45.72** | **43.4** | **−0.41%** | **−0.46%** |
| gaussian | +6,657.62 B | +784.23 | n/a | +97.51% | +35.35% |
| conditional_scale | −1,013.62 B | +812.95 | **821.3** | +95.67% | +35.13% |

### Split vs Conditional — 같은 이득, 13배 비용

| | Split | Conditional Scale | 배율 |
| --- | ---: | ---: | ---: |
| rate 이득 (.0064) | −1,338.50 B | −1,448.12 B | Conditional이 8% 우세 |
| rate 이득 (.0256) | **−1,078.12 B** | −1,013.62 B | **Split이 6% 우세** |
| 추가 모델 크기 | 45.7~60.1 KiB | 813.0 KiB | **13.5~17.8×** |
| Break-even | **43~46 장면** | 575~821 장면 | **12.5~17.9×** |
| score encode 시간 | ≈0% | **+88~96%** | — |
| full codec decode | ≈0% | **+31~35%** | — |
| full eval 수신 복원 | 89.6 / 81.0 ms | 126.2 / 120.1 ms | **+41% / +48%** |

**왜 Conditional이 그렇게 느린가.** `GaussianConditional`은 토큰별 σ에 대응하는
scale index를 찾아 CDF를 선택해야 한다. 64단계 table lookup + per-position indexing이
`EntropyBottleneck`의 채널별 단일 CDF보다 훨씬 비싸다.
반면 Split은 **CDF 개수만 2배**이고 lookup 구조는 그대로라 비용이 사실상 없다.

**왜 Conditional 모델이 그렇게 큰가.** `GaussianConditional`의 `_quantized_cdf` buffer가
64개 scale 단계 × 심볼 범위를 담아야 한다. Static Gaussian도 같은 이유로 +784 KiB다.
파라미터(10,824)는 작은데 **buffer가 크다.** `score-path tensor KiB` 컬럼이 이걸 잡아낸다.

> ⚠️ 절대 시간은 기기 의존적이다. **같은 GPU·같은 장면 순서의 paired delta**만 신뢰한다.
> 학습 `.ckpt` 크기는 optimizer state를 포함할 수 있어 배포 크기가 아니다.

### scene CI로 본 최종 확인

`Nonlinear Full+Split vs Linear context Full`:

| λ | Rate 절감 % [CI] | ΔPSNR dB [CI] | Rate 승률 |
| --- | ---: | ---: | ---: |
| .0064 | +4.395% [+4.384, +4.406] | +0.0006 [−0.0010, +0.0023] | **100.0%** |
| .0256 | +5.894% [+5.878, +5.909] | +0.0356 [+0.0332, +0.0380] | **100.0%** |

2점 BD-rate: **−6.816%** [−6.955, −6.678].
rate 승률 100% = **6,991 장면 전부에서 더 작았다.**

## 3.9 Split에 대해 아직 모르는 것

| 질문 | 왜 중요한가 |
| --- | --- |
| **처음부터 joint 학습하면 더 좋아지나?** | 현재는 복원 고정 fit이다. transform이 Split의 존재를 알고 학습하면 even/odd 비대칭을 더 적극적으로 활용할 수 있다. 완전 미실험. |
| **even도 쪼갤 여지가 있나?** | Split은 even/odd만 나눈다. group 내 anchor 위치(왼쪽만 보는 마지막 odd 등)나 scene step bin으로 더 쪼개면? entropy probe가 scene-step bin으로 이득을 봤다. |
| **Split보다 강한 non-Gaussian conditional density** | Gaussian scale 실험만으로 conditional 전체를 판단할 수 없다. 비모수 conditional CDF는 미실험. |
| **odd의 3분할 이상** | 현재 2분할. probe 결과는 초반 group·odd에 여유가 몰려 있다고 가리킨다. |
| **single seed 재현성** | 1,000~1,300 B 차이는 크지만, seed 반복은 안 했다. |

---

# Part 4. 세 요소를 합친 전체 그림

## 4.1 rate 분해와 남은 여유

현재 기준 모델(λ=0.0064) 83,121.693 B/scene을 층별로 뜯으면:

```text
83,121.693 B
├── score            69,604.342 B  (83.7%)   ← Context + Split이 작용하는 곳
│   ├── raw entropy      69,516.3 B
│   └── SCCTX001 wrapper      88 B  (0.11%)
├── residual y       11,738.844 B  (14.1%)   ← 손대지 않은 영역
├── residual z          174.507 B  (0.21%)
├── FP16 mean         1,472.000 B  (1.77%)   ← 고정
└── container           132.000 B  (0.16%)   ← 고정
```

**여기서 진단 실험이 밝힌 것:**

| 후보 | 회수 가능량 | 근거 |
| --- | ---: | --- |
| rANS coder 자체의 비효율 | 약 60 B/scene (0.07%) | entropy probe |
| finite CDF + tail 근사 | 약 99 B/scene | entropy probe |
| stream 병합 (8→1) | 수십 B | stream당 약 48 bit |
| **단순 5-bin decoder-known context** | **613 B (0.736%)** | entropy probe, held-out |
| Split (이미 적용) | 1,321 B (1.56%) | probability-only |
| Context (이미 적용) | 6,409 B (7.1%) | overhead profile |

→ **coder/컨테이너 최적화는 끝난 얘기다.** 남은 건 확률 모델과 표현이다.
그리고 entropy probe가 보여준 여유는 0.75%로, **score의 83.7% share는 대부분 진짜 정보량**이다.

**score 절감이 몰린 곳** (entropy probe):

| λ | 상위 3 stream | 점유율 | 모든 odd stream |
| --- | --- | ---: | ---: |
| .0064 | `g0_odd`, `g0_even`, `g1_odd` | **91.4%** | 67.3% |
| .0256 | 같은 세 stream | 82.3% (+`g3_odd` → 91.4%) | 72.7% |

**초반 group + odd.** 이게 정확히 [§1.5](#15-spatial-context--even을-먼저-복호하고-odd를-예측)에서
지적한 "group 안에서만 채널이 섞이고 ±1 token만 본다"는 제약이 걸리는 지점이다.

## 4.2 다음 수정 지점 — 코드 위치까지

### (A) 저위험: 초반 group·odd에 선택적 conditional CDF

**바꿀 곳:** [score_context.py:128-138 `_odd_forward`](upstream/globalsplat/globalsplat/compression/score_context.py#L128-L138)

현재는 group_index와 무관하게 같은 종류의 `FactorizedScoreEntropy`를 쓴다.
`g0`/`g1`에만 비모수 conditional CDF(scene step bin × anchor magnitude bin)를 붙이고
`g2`/`g3`는 그대로 두면, 모델 크기 증가를 절감이 큰 곳에 집중할 수 있다.

**설계 시 지켜야 할 것:**

- 조건 입력은 `anchor_delta`와 `group_step`뿐 (수신기가 아는 것)
- 0-init으로 시작해 부모와 동일한 확률에서 출발
- flag를 새로 추가하고 `KNOWN_FLAGS`에 등록 → 구버전 bitstream 거부
- 더 큰 TRAIN으로 fit, **validation에서 선택**, full TEST는 **단 한 번**
- table 배포 크기를 break-even 장면 수로 환산해 함께 보고

### (B) 구조적: score hyperprior / coarse-to-fine latent

**바꿀 곳:** [codec.py:182-209 `_analyze_sorted`](upstream/globalsplat/globalsplat/compression/codec.py#L182-L209)의 score 분기

작은 score side latent를 추가 전송하고, 그걸로 비모수 score 분포를 예측한다.
residual이 이미 쓰는 구조(`z` → `y`의 μ/σ)를 score에도 적용하는 것이다.

**검증 순서가 중요하다:**

1. 먼저 **score 양자화 격자를 고정**한 채로만 붙인다 → "side bit를 벌어오는가"만 깨끗이 본다
2. 그게 확인된 뒤에야 transform과 joint 학습

이 순서를 지키면 P0/P1/P2 실험이 겪은 혼입([§8.4](EXPERIMENTS_OVERVIEW.md))을 피할 수 있다.

### (C) 미탐색: residual entropy를 복호된 low-rank로 조건화

**바꿀 곳:** [residual.py](upstream/globalsplat/globalsplat/compression/residual.py)의 hyper-synthesis 입력

현재 residual `y`의 μ/σ는 `z_hat`만으로 예측한다.
그런데 수신기는 그 시점에 **복호된 low-rank feature `X_low`도 이미 갖고 있다.**
이건 공짜 조건 정보다.

세대 1에 `shared_lowrank_rank56_untied_nonlinear64_basecond_entropy_multiscale1d`
설정이 남아 있으니 **먼저 그 산출물을 확인**할 것.
residual이 전체의 14.1%라 상한이 작지만, 공짜 정보를 안 쓰는 건 이상하다.

### (D) 미탐색: transform hidden 크기

**바꿀 곳:** `config/model/globalsplat_nfcgs_rank56.yaml`의 `transform_hidden`

현재 32 고정. 세대 1에 nonlinear 64 설정이 여럿 있는데 세대 3은 32만 썼다.
`CodecConfig`가 `transform_hidden`을 자유 정수로 허용하므로
([config.py:66-79](upstream/globalsplat/globalsplat/compression/config.py#L66-L79)) 바로 sweep 가능하다.
단 checkpoint 호환이 깨지므로 새 계보를 시작해야 한다.

### (E) 평가 보강 — 이게 사실 가장 시급할 수 있다

현재 λ가 **2개뿐**이라 BD-rate가 screening 통계에 머문다.
세대 2는 이미 4개 λ(0.0004/0.0016/0.0064/0.0256)로 곡선을 그렸다.
현재 구조로 λ를 4개 이상 학습하면 **구조 비교가 아니라 codec 자체의 품질**을 주장할 수 있다.

---

## 부록. 구현 수정 시 깨지기 쉬운 것 체크리스트

| # | 항목 | 왜 |
| ---: | --- | --- |
| 1 | score-context 안에서 autocast를 켜지 마라 | 송수신 수치 불일치 → 복호 실패 ([codec.py:169](upstream/globalsplat/globalsplat/compression/codec.py#L169)) |
| 2 | 새 모듈은 `fork_rng` 안에서 만들어라 | 난수 소비가 바뀌어 ablation이 깨진다 ([codec.py:76](upstream/globalsplat/globalsplat/compression/codec.py#L76)) |
| 3 | 새 예측기는 마지막 layer만 0-init | 전부 0이면 gradient가 0. 한쪽만 0이어야 한다 |
| 4 | probability-only 실험은 `quantile_update_interval=0`도 설정 | `requires_grad=False`만으로는 deterministic update를 못 막는다 |
| 5 | checkpoint 이식 후 `update()`로 CDF를 재생성 | `_`로 시작하는 buffer는 복사하지 않는다 |
| 6 | 새 변형은 flag를 추가하고 `KNOWN_FLAGS`에 등록 | 구버전 bitstream이 조용히 잘못 복호되는 것을 막는다 |
| 7 | `score_entropy`(legacy rank-56 EB)는 건드리지 마라 | strict load 전용. `_freeze_inactive_score_entropy()`가 항상 freeze ([codec.py:138-146](upstream/globalsplat/globalsplat/compression/codec.py#L138-L146)) |
| 8 | `residual_mean`/`residual_std`는 buffer다 | 학습되지 않는다. PCA artifact가 채운다 |
| 9 | 홀수 `points`를 테스트하라 | `ceil(points/2)` vs `points//2` 경로가 갈린다 |
| 10 | 송신기 경로에서 `decompress`를 빼지 마라 | context가 복호값에 의존한다 ([score_context.py:222](upstream/globalsplat/globalsplat/compression/score_context.py#L222)) |

---

# Part 5. 아키텍처 다이어그램 ↔ 코드 매핑

이 절은 기존 아키텍처 그림의 각 블록을 현재 코드와 대조한다.
표기는 그림의 기호(`X_ord`, `S`, `V_S`, `K_S`, `Ŝ`, `E`, `y`, `z` …)를 그대로 쓴다.

## 5.1 먼저 알아야 할 것 — 그림은 **context 이전 세대**다

> **그림의 score 경로는 `Factorized Entropy Model ψ` 하나 + `AE`/`AD` 한 쌍이다.
> 이건 [EXPERIMENTS_OVERVIEW.md §8.2](EXPERIMENTS_OVERVIEW.md)의 "context 이전 Linear factorized" 구조,
> 즉 세대 3이 개선하기 시작한 *출발점*이다.**

현재 기준 모델(`Nonlinear32 / Full P0 / Split`)과 비교하면 그림에 없는 것이 5개다.
아래 [§5.4](#54-그림에-없는-것-5가지)에서 하나씩 짚는다.

그림의 **transform·Morton·residual·컨테이너 부분은 현재 코드와 정확히 일치**한다.
바뀐 건 score 경로뿐이다. 그래서 그림을 버릴 필요는 없고, 오른쪽 score 블록만 확장하면 된다.

## 5.2 블록별 매핑 — 일치하는 부분

| 그림 블록 | 코드 위치 | 변수 / 연산 | shape |
| --- | --- | --- | --- |
| `Scene-token Selection (_pack_outputs)` | [globalsplat.py:202-212](upstream/globalsplat/globalsplat/model/globalsplat.py#L202-L212) `encode_scene_tokens` | `slot_encoder(...)` → `(appearance, geometry)` | `Z_A`, `Z_G` : `[B,4096,512]` |
| `Decoder-observable Geometry Projection` | [codec.py:63-65](upstream/globalsplat/globalsplat/compression/codec.py#L63-L65) | `nn.Linear(512, 224, bias=False)` → `P_G = W^T` | `O_G` : `[B,4096,224]` |
| `Concat` | [codec.py:123](upstream/globalsplat/globalsplat/compression/codec.py#L123) | `torch.cat([texture, observable], dim=-1)` | `X` : `[B,4096,736]` |
| `Position(O_G)` | [gaussian_decoder.py:289-305](upstream/globalsplat/globalsplat/model/decoder/gaussian_decoder.py#L289-L305) `decoded_token_centers` | `geo_pos_readout` → `[B,P,M_max,3]`, gate softmax 가중평균 + `patch_center_bias` | `positions` : `[B,4096,3]` |
| `Geometry-guided Morton Sorting` | [morton.py:63-83](upstream/globalsplat/globalsplat/compression/morton.py#L63-L83) | scene-local 정규화 → 축당 10-bit → bit interleave → `argsort(stable=True)` | `π` : `[B,4096]` |
| `X_ord = Permute_π(X)` | [codec.py:222](upstream/globalsplat/globalsplat/compression/codec.py#L222) | `order.apply(features)` | `[B,4096,736]` |
| `Centering` / `μ̂ = Q_FP16(μ)` | [codec.py:174-186](upstream/globalsplat/globalsplat/compression/codec.py#L174-L186) | `_quantize_mean(sorted_features.mean(dim=1))` — **STE 포함** | `μ̂` : `[B,736]` = 1,472 B |
| `X_c = X_ord − 1μ̂` | [codec.py:186](upstream/globalsplat/globalsplat/compression/codec.py#L186) | `centered = sorted_features - mean[:,None,:]` | `[B,4096,736]` |
| `Low-rank Projection` `A_ana` | [codec.py:58, 130-133](upstream/globalsplat/globalsplat/compression/codec.py#L130-L133) | `shared_basis` (orthogonal init 또는 PCA artifact) | `A_ana` : `[56,736]` |
| `S = X_c A_ana^T` | 같은 곳 | `centered @ shared_basis.transpose(0,1)` | `S` : `[B,4096,56]` |
| `Score Normalization` `q_S` | [codec.py:60, 102-104, 188](upstream/globalsplat/globalsplat/compression/codec.py#L102-L104) | `score_scale = exp(score_log_scale).clamp_min(1e-8)` | `q_S` : `[56]` |
| `m_S` | CompressAI `EntropyBottleneck._get_medians()` | `Q` 내부에서 적용. 별도 파라미터가 아니라 **entropy model의 quantile** | `[56]` |
| `Q` → `K_S` | `EntropyBottleneck.forward/compress` | 학습: `V+U(-.5,.5)` / 부호화: `round(V − m_S)` | `K_S` : `[B,4096,56]` int |
| `AE` / `AD` | CompressAI rANS (CPU native) | `compress` / `decompress` | — |
| `Score Denormalization` | [codec.py:194](upstream/globalsplat/globalsplat/compression/codec.py#L194) | `score_hat * score_scale` | `Ŝ` : `[B,4096,56]` |
| `Low-rank Unprojection` `A_syn` | [codec.py:59, 125-128](upstream/globalsplat/globalsplat/compression/codec.py#L125-L128) | `shared_synthesis_basis` — **`A_ana`와 별개 파라미터(untied)** | `A_syn` : `[56,736]` |
| `Residual Computation` `E = X_ord − X_base` | [codec.py:197](upstream/globalsplat/globalsplat/compression/codec.py#L197) | `residual = centered - low_rank` (= `X_ord − 1μ̂ − ŜA_syn`, 동치) | `E` : `[B,4096,736]` |
| `Rearrange` → `E_1D` | [codec.py:198](upstream/globalsplat/globalsplat/compression/codec.py#L198) | `.transpose(1,2).unsqueeze(2)` | `[B,736,1,4096]` |
| `1D Residual Analysis` `g_a` | [residual.py:92-96](upstream/globalsplat/globalsplat/compression/residual.py#L92-L96) | `conv(736→192,k5,s2)` → `GDN` → `conv(192→320,k5,s2)` | `y` : `[B,320,1,1024]` |
| `1D Hyper Analysis` `h_a` | [residual.py:102-108](upstream/globalsplat/globalsplat/compression/residual.py#L102-L108) | `conv(320→192,k3)` → LReLU → `conv(192→192,k5,s2)` → LReLU → `conv(k5,s2)` | `z` : `[B,192,1,256]` |
| `z`의 `Q`/`AE`/`AD` | [residual.py:114, 133](upstream/globalsplat/globalsplat/compression/residual.py#L133) | `EntropyBottleneck(192)` | — |
| `1D Hyper Synthesis` `h_s` → `μ, σ` | [residual.py:109-124](upstream/globalsplat/globalsplat/compression/residual.py#L119-L124) | `deconv(192→320)` → LReLU → `deconv(320→640)` → `chunk(2)` | `σ, μ` 각 `[B,320,1,1024]` |
| `y`의 `⊖μ` → `Q` → `AE` | [residual.py:135-137](upstream/globalsplat/globalsplat/compression/residual.py#L135-L137) | `GaussianConditional(y, σ, means=μ)` — 내부에서 `round(y−μ)` | — |
| `1D Residual Synthesis` `g_s` | [residual.py:97-101](upstream/globalsplat/globalsplat/compression/residual.py#L97-L101) | `deconv(320→192)` → `GDN(inverse)` → `deconv(192→736)` | `Ê_1D` : `[B,736,1,4096]` |
| `InverseRearrange` | [codec.py:206](upstream/globalsplat/globalsplat/compression/codec.py#L206) | `.squeeze(2).transpose(1,2)` | `Ê` : `[B,4096,736]` |
| `Receiver-side Reconstruction` | [codec.py:208 / :311](upstream/globalsplat/globalsplat/compression/codec.py#L311) | `mean[:,None,:] + low_rank + residual_hat` | `x̂_ord` : `[B,4096,736]` |
| `Geometry / Appearance Split` | [codec.py:317-319](upstream/globalsplat/globalsplat/compression/codec.py#L317-L319) | `[..., :512]` / `[..., 512:]` | `Ẑ_A`, `Ô_G` |
| `Packing` / `Unpacking` | [bitstream.py:24](upstream/globalsplat/globalsplat/compression/bitstream.py#L24) | `SceneBitstream` — magic `E2EM0301`, header `>8sHHIIIQQQ32s` | 80 B |

### 컨테이너는 3중 중첩이다 (그림의 `Packing` 한 박스)

```text
E2EM0301  (SceneBitstream, 80 B header + SHA256 checksum)
├── mean_fp16     1,472 B          ← 736 × FP16, 압축하지 않음
├── score         SCCTX001 wrapper (24 B prefix + 8 B/stream)
│                 └── stream 1개(그림) / 8개(현재)
└── residual      M3HPRN01 wrapper (52 B)
                  ├── z strings
                  └── y strings
```

magic: `E2EM0301` / `SCCTX001` / `M3HPRN01`
([bitstream.py:24, 107, 177](upstream/globalsplat/globalsplat/compression/bitstream.py#L24)).
`M3HPRANS`는 세대 1의 구형 residual 컨테이너로 읽기만 지원한다
([bitstream.py:149](upstream/globalsplat/globalsplat/compression/bitstream.py#L149)).

### 그림이 잘 잡아낸 것 — 역permutation이 없다

그림의 수신 경로는 `x̂_ord → Split → Decoder`로 끝나고 **역permutation 블록이 없다.**
이건 정확하다. [globalsplat.py:263-269](upstream/globalsplat/globalsplat/model/globalsplat.py#L263-L269):

```python
self.last_codec_output = self.feature_codec(
    appearance, geometry, token_centers,
    # Gaussian decoding is token-wise and the output is a set, so
    # no inverse permutation needs to be transmitted or applied.
    restore_original_order=False,
    training=self.training,
)
```

학습 forward에서도 `restore_original_order=False`다.
Gaussian은 집합이므로 토큰 순서가 결과에 영향을 주지 않는다. `π`를 전송할 필요가 없다.

> `ObservableLowRank1DCodec.forward`의 기본값은 `True`지만, 실제 호출부가 항상 `False`를 넘긴다.
> `decompress`의 `inverse_permutation`도 기본 `None`이다. 즉 송수신 양쪽 모두 Morton 순서를 소비한다.

## 5.3 그림 수식 → 코드 수식 대조 (검산)

| 그림 | 코드 | 일치? |
| --- | --- | --- |
| `O_G = Z_G P_G`, `P_G ∈ R^{512×224}` | `nn.Linear(512,224,bias=False)`, `y = x @ W^T`, `W ∈ R^{224×512}` → `P_G = W^T` | ✅ |
| `μ = (1/4096) Σ X_ord,i` | `sorted_features.mean(dim=1)` | ✅ |
| `μ̂ = Q_FP16(μ)` | `mean.to(float16).to(mean.dtype)` + STE | ✅ (STE는 그림에 없음) |
| `X_c = X_ord − 1_4096 μ̂` | `sorted_features - mean[:,None,:]` | ✅ |
| `S = X_c A_ana^T` | `centered @ shared_basis.transpose(0,1)` | ✅ (단 `+ MLP_a` 추가됨) |
| `V_S = S/q_S − m_S` | `score / score_scale` 후 EB 내부 `− median` | ✅ (분해 위치만 다름) |
| `Ŝ = (K_S + m_S) ⊙ q_S` | `score_hat * score_scale` | ✅ |
| `X_base = 1μ̂ + Ŝ A_syn` | `mean + low_rank` | ✅ (단 `+ MLP_s` 추가됨) |
| `E = X_ord − X_base` | `centered - low_rank` | ✅ 동치 |
| `y = g_a(E_1D)` | `g_a(analysis_adapter(normalized_E_1D))` | ⚠️ **정규화·adapter 누락** |
| `z = h_a(y)` | `h_a(torch.abs(y))` | ⚠️ **`abs` 누락** |
| `Ê_1D = g_s(ŷ)` | `synthesis_adapter(g_s(y_hat))` | ⚠️ **adapter 누락** |
| `x̂_ord = 1μ̂ + X̂_LR + Ê` | `mean + low_rank + residual_hat` | ✅ |
| `Ẑ_A = x̂_ord[:,:,0:512]` | `reconstruction[..., :512]` | ✅ |

### shape 검산 — 전부 코드와 일치

```text
4096 --g_a conv s2--> 2048 --g_a conv s2--> 1024      →  y  [B, 320, 1, 1024]   ✅
1024 --h_a k3 s1--> 1024 --s2--> 512 --s2--> 256      →  z  [B, 192, 1,  256]   ✅
h_s: 256 --s2--> 512 --s2--> 1024, 채널 192→320→640   →  chunk(2) = σ, μ        ✅
```

`residual_N=192`(= `z` 채널), `residual_M=320`(= `y` 채널)이
[globalsplat_nfcgs_rank56.yaml](upstream/globalsplat/config/model/globalsplat_nfcgs_rank56.yaml)의 값과 맞는다.

## 5.4 그림에 없는 것 5가지

### ★★★ ① Score context (Full P0) — 가장 큰 차이

그림: `X_c → Low-rank Projection → Score Normalization → Q → AE → [ψ] → AD → Score Denormalization`
**단일 경로, 단일 stream, `μ̂`에서 확률 모델로 가는 화살표 없음.**

현재: `Mean conditioner` + `Channel predictor ×3` + `Spatial predictor ×4`,
**4 group × 2 pass = 8 stream.**

그림 표기로 확장하면 이렇다. `S̃ = S / q_S`(정규화 score), group 분할 `r = 56 → (16,16,16,8)`:

```text
[Mean conditioner]                                    ← 그림에 없음
  [b ; log d] = f_μ(μ̂)              f_μ : LN → 736→64 → GELU → 64→112
  d = exp(2·tanh(log d / 2)) ∈ [0.135, 7.389]
  b ∈ R^{B×56},  d ∈ R^{B×56}

for g = 0 … 3:                                        ← group 직렬 (그림에 없음)

  [Channel predictor]                                 ← 그림에 없음
    B_g = b_g            (g = 0)
    B_g = b_g + C_g(Ŝ̃_{<g})   (g ≥ 1),  C_g = Conv2d(16g, r_g, kernel 1)

  [EVEN pass]  token 2,048개 병렬
    V_g^e  = (S̃_g^e − B_g^e) / d_g
    K_g^e  = Q(V_g^e − m_g^e)                        → stream #(2g)     ψ_g^even
    Ŝ̃_g^e  = B_g^e + d_g · (K_g^e + m_g^e)

  [Spatial predictor]                                 ← 그림에 없음
    Δ_g[even] = Ŝ̃_g^e − B_g^e ,  Δ_g[odd] = 0
    B_g^o = B_g[odd] + P_g(Δ_g)[odd]    P_g = Conv2d(r_g, r_g, kernel (1,3))

  [ODD pass]   token 2,048개 병렬
    V_g^o  = (S̃_g^o − B_g^o) / d_g
    K_g^o  = Q(V_g^o − m_g^o)                        → stream #(2g+1)   ψ_g^odd  ★Split
    Ŝ̃_g^o  = B_g^o + d_g · (K_g^o + m_g^o)

  Ŝ̃_g = merge(Ŝ̃_g^e, Ŝ̃_g^o)

Ŝ = Ŝ̃ ⊙ q_S
```

**그림에 추가해야 할 것:**

1. `μ̂` 박스에서 **새 블록 `Mean Conditioner f_μ`로 가는 화살표**
   → `b`, `d` 출력. 현재 `μ̂`는 Receiver-side Reconstruction으로만 간다.
2. `Score Normalization` 박스를 **`Context Prediction` 박스로 교체**
   (`V = (S/q_S − b)/d` 를 표시)
3. 단일 `Q → AE → ψ → AD`를 **group 4개 × even/odd 2 pass의 격자**로 확장
4. `Ŝ̃_{<g}` 되먹임 화살표 (channel), `Δ_g` 되먹임 화살표 (spatial)
5. `Packing` 박스 안의 stream 수 1 → **8**

### ★★ ② Nonlinear transform — `A_ana`/`A_syn` 옆의 MLP

그림: `S = X_c A_ana^T`, `X̂_LR = Ŝ A_syn` — **순수 행렬곱**.

현재 ([codec.py:125-133](upstream/globalsplat/globalsplat/compression/codec.py#L125-L133)):

```text
S    = X_c A_ana^T + MLP_a(X_c)        MLP_a : 736 →(bias-free)→ 32 → GELU → 56
X̂_LR = Ŝ   A_syn   + MLP_s(Ŝ)          MLP_s :  56 →(bias-free)→ 32 → GELU → 736
```

**그림 수정:** `Low-rank Projection` 박스(파랑) 옆에 병렬 `MLP_a` 가지와 `⊕`,
`Low-rank Unprojection` 박스(분홍) 옆에 병렬 `MLP_s` 가지와 `⊕`.
두 번째 Linear가 0-init이므로 학습 시작 시점에는 그림과 정확히 같은 함수다.

**주의:** `MLP_s`의 입력은 **`Ŝ`(복호된 값)**이다. 원본 `S`가 아니다.
그림에 그릴 때 `Score Denormalization` 출력에서 받아야 한다.
그래야 수신기가 재현할 수 있고, 실제로 그림의 두 `Low-rank Unprojection`
(송신측 residual 계산용 / 수신측 복원용) 양쪽에 똑같이 붙는다.

### ★ ③ even/odd Split — `ψ`가 하나가 아니다

그림 오른쪽의 `Factorized Entropy Model ψ` 회색 박스 하나가
`AE`와 `AD` 양쪽에 연결돼 있다.

현재는 **8개**다.

| | 그림 | 현재 |
| --- | ---: | ---: |
| entropy model 수 | 1 (`ψ`, 56채널) | **8** (`ψ_g^even` ×4 + `ψ_g^odd` ×4) |
| 채널 구성 | 56 | 16/16/16/8 를 even·odd 두 벌 |
| density 파라미터 | 3,416 | **6,832** |

`ψ_g^odd`는 학습 시작 시 `ψ_g^even`의 density와 quantile을 복제해서 만든다
([§3.4](#34-초기화--step-0에서-부모와-정확히-동일함을-보장)).
**stream을 늘리는 게 아니라, 이미 분리돼 있던 even/odd stream에 서로 다른 CDF를 적용하는 것**이다.

### ★ ④ Residual 정규화 — `E`와 `g_a` 사이

그림: `E` → `1D Residual Analysis` 직행.

현재 ([codec.py:199-206](upstream/globalsplat/globalsplat/compression/codec.py#L199-L206)):

```text
E_1D  →  (E_1D − residual_mean) / residual_std  →  analysis_adapter  →  g_a  →  y
ŷ     →  g_s  →  synthesis_adapter  →  × residual_std + residual_mean  →  Ê_1D
```

- `residual_mean` / `residual_std`는 `register_buffer`라 **학습되지 않는다.**
  PCA 초기화 artifact가 채운다 ([codec.py:61-62](upstream/globalsplat/globalsplat/compression/codec.py#L61-L62)).
- `MultiScaleAdapter1D` ([residual.py:48-72](upstream/globalsplat/globalsplat/compression/residual.py#L48-L72))는
  skip connection이 있는 **독립 모듈**이다. `g_a`/`g_s` 안에 접혀 있지 않다.

  ```text
  in_projection  : Conv2d(736 → 96, k1) → GELU
  branches       : Conv2d(96 → 32, k=(1,5), dilation 1 / 2 / 4)  → 각각 GELU → concat(96)
  out_projection : Conv2d(96 → 736, k1)
  return value + out_projection(mixed)          ← residual connection
  ```

  송신·수신 양쪽에 하나씩 있다(`analysis_adapter`, `synthesis_adapter`).
  dilation 1/2/4로 **receptive field 5 / 9 / 17 token**을 동시에 본다.
  이게 Morton 지역성을 실제로 소비하는 모듈이라,
  [실험 15b](EXPERIMENTS_OVERVIEW.md)에서 Morton이 −21% bytes를 만든 주역이다.

### ★ ⑤ `z = h_a(|y|)` — 절댓값

[residual.py:132](upstream/globalsplat/globalsplat/compression/residual.py#L132), [:152](upstream/globalsplat/globalsplat/compression/residual.py#L152):

```python
z = self.h_a(torch.abs(y))
```

hyperprior가 예측하는 건 `y`의 **스케일(σ)**이므로 부호는 불필요하고,
`|y|`를 넣으면 hyper-analysis가 부호를 재발명할 필요가 없다. 원 hyperprior 논문의 관행이다.
그림의 `z = h_a(y)`는 `z = h_a(|y|)`로 고쳐야 한다.

## 5.5 그림 수정 체크리스트

| # | 위치 | 수정 | 우선도 |
| ---: | --- | --- | --- |
| 1 | 오른쪽 score 블록 전체 | 단일 `Q/AE/ψ/AD` → **4 group × even/odd 2 pass = 8 stream** 격자 | ★★★ |
| 2 | `μ̂` 박스 | score 확률 경로로 가는 **화살표 추가** → `Mean Conditioner f_μ` → `b`, `d` | ★★★ |
| 3 | `Score Normalization` | `V = (S/q_S − b)/d` 로 갱신, 박스명 `Context Prediction` | ★★★ |
| 4 | `Factorized Entropy Model ψ` | `ψ_g^even ×4`, `ψ_g^odd ×4` (Split) | ★★ |
| 5 | `Low-rank Projection` / `Unprojection` | 병렬 `MLP_a` / `MLP_s` 가지 + `⊕` (0-init) | ★★ |
| 6 | residual 경로의 `⊖μ` / `⊕μ`, `σ` | `μ → μ_y`, `σ → σ_y`. scene mean `μ̂`와 hat 하나 차이라 충돌한다 — 전송되는 736-벡터 vs `ẑ`에서 재계산되는 원소별 텐서 (§0.5 "기호 충돌" 참조) | ★★ |
| 7 | `Centering` 박스 캡션 | `μ = (1/4096) Σ X_ord,i`는 맞지만 "**토큰 축 평균 → 채널별 736개**"를 명시하면 오독이 사라진다 | ★ |
| 8 | `E` → `1D Residual Analysis` 사이 | `(E − residual_mean)/residual_std` + `MultiScaleAdapter1D`. `residual_mean`은 **전역 buffer**이지 장면별 통계가 아니라는 점 명기 | ★ |
| 9 | `1D Residual Synthesis` 뒤 | `MultiScaleAdapter1D` + 역정규화 | ★ |
| 10 | `z = h_a(y)` | `z = h_a(\|y\|)` | ★ |
| 11 | `Packing` 캡션 | stream 수 1 → 8, `SCCTX001` wrapper 88 B 명기 | ★ |
| 12 | Geometry Projection 캡션 | "224개의 **token**" → "224개의 **채널(차원)**". token은 4,096개로 변하지 않는다 | ★ |
| 13 | Appearance Decoder 입력 라벨 | `Z_A` → `Ẑ_A` (복원값) | 사소 |
| 14 | `Geometry Decoder` / `Appearance Decoder` | 코드상 **단일 `gaussian_decoder`의 두 입력 스트림**이다. 별개 모듈로 읽히지 않게 묶어주면 정확하다 | 사소 |

## 5.6 실제로 그린 도해 (2026-09-19)

위 체크리스트를 반영한 도해 다섯 장을 그려서 아래에 게시했다. 비공개 링크이므로
다른 사람에게 보여주려면 페이지의 Share 메뉴에서 공유해야 한다.

[**NFC-GS 코덱 도해 — 다섯 장**](https://claude.ai/artifact/VFLK2TcKisX6Uxc28YKt9T)

| 도해 | 내용 | 대응 체크리스트 |
| --- | --- | ---: |
| Figure 1 | token 축 평균 vs channel 축 평균 — 후자는 코덱에 없다 | 7 |
| Figure 2 | 전체 파이프라인 15단계 + bitstream 조립 레일 | 5 ~ 14 |
| Figure 3 | median이 하는 일 — 복원 격자의 위상(phase) | — |
| Figure 4 | 양자화 지점 전부 — 장면당 11회, score는 8회 | — |
| Figure 5 | 8-stream 스케줄 — group 4개 직렬, group 안 even/odd 2 pass | 1, 9 |
| Figure 6 | group 하나의 내부 (Full) — Mean/Channel/Spatial과 even→odd 연쇄 | 1 ~ 4 |
| Figure 7 | **Mean** (1/0/0) — 1 group, 1 pass, stream 1개 | 1 ~ 3 |
| Figure 8 | **Mean + Channel** (1/1/0) — 4 group 직렬, stream 4개 | 1 ~ 3 |
| Figure 9 | **Mean + Spatial** (1/0/1) — 1 group, even/odd 2 pass, stream 2개 | 1 ~ 3 |
| Figure 10 | Shared vs Split — 바뀌는 간선 하나 | 4 |
| Figure 11 | Nonlinear이 붙는 자리 — 행렬곱과 병렬인 잔차 MLP | 5 |

Figure 7 ~ 9는 Full로 가기 전 중간 단계 세 개다. 실험 표
([EXPERIMENTS_OVERVIEW.md §8.2](EXPERIMENTS_OVERVIEW.md))의 arm과 1:1로 대응한다.

**코드를 읽다 새로 확인한 것:** group을 나누지 않는 Mean+Spatial은 spatial predictor가
`Conv2d(56, 56, k3)`라 **56채널 전부를 ±1 token에 걸쳐 섞는다**(9,408 weight).
Full은 group마다 `Conv2d(16, 16, k3)`라 **group 밖 채널을 보지 못한다**(합계 2,496 weight).
Full은 "넓은 spatial 채널 혼합"을 포기하고 "channel context + stream 4개"를 얻은 셈이고,
이것이 `.0256`에서 Full이 Mean+Spatial을 확실히 이기지 못한 구조적 이유로 보인다.
overhead profile의 predictor 파라미터 수(mean_spatial 65,384 > full 59,664)와도 부합한다.
다만 이 인과는 코드 구조에서 읽은 해석이고, 통제 실험으로 검증하지는 않았다.

원본 그림을 직접 고칠 때는 Figure 1을 base로 쓰고, score 블록만
`Contextual Score Codec` 박스 하나로 추상화한 뒤 Figure 2·3을 별도 확대도로 두는 게
가장 읽기 쉽다. 확대도에 **"직렬 단계 8개, 각 단계 안은 4,096 token 병렬"**을 명시하면
"autoregressive 아니냐"는 질문을 미리 막을 수 있다.

도해의 표기는 [Part 1](#part-1-context-model)·[Part 3](#part-3-split)과 같다
(`U` 정규화 score, `B_g` base, `V` 부호화 대상, `K` 정수 심볼, `Û`/`Ŝ` 복원값,
`m_g` entropy median, `μ_y`/`σ_y` hyperprior 예측).

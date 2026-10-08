# 토큰 정렬·문맥 모델 구현 정리 (E82·E83)

작성 2026-10-07. 기존 토큰 코덱(`../global`, Morton 정렬 + even/odd 공간 문맥)에 다른 정렬이나 다른 문맥을 넣어 보려는 사람을 위한 문서다.

- 코드: `exp82_sort_context_screen.py`, `exp83_order_context_screen2.py`, `exp83b_copy_index_cost.py`, `exp83c_order_locality.py`
- 로그: `logs/chain_ah_e82_sort_context.log`, `logs/chain_ah_e83_order_context2.log`, `logs/chain_ah_e83b_copy_index.log`,
  `logs/chain_ah_e83c_order_locality.log`
- 결과 요약: `REPORT.md`의 E82·E83 절
- 사용자 코덱 코드 위치는 `../global/upstream/globalsplat/globalsplat/compression/` 기준이다(읽기만 했고 수정하지 않았다).

---

## 0. 한눈에 보기

모든 rate는 "문맥 없음"(57.10 KiB/씬) 대비 변화다. 측정 방식과 한계는 1.2절에 있다.

### 0.1 정렬 (문맥은 even/odd ±1로 고정)

| 정렬 | 만드는 법 | rate | 연속 토큰 3D 거리 중앙값 | lag-1 상관 (심볼 / residual) | 계산 시간 N = 2048 / 4096 (CPU) |
| --- | --- | --- | --- | --- | --- |
| 토큰 인덱스 (정렬 없음) | – | 0.0 % | 3.781 | −0.002 / −0.002 | – |
| Morton | 10 bit 격자, 비트 인터리브 키로 정렬 | −9.3 % | 0.220 | 0.588 / 0.449 | 1.2 / 2.5 ms |
| Hilbert | 10 bit 격자, Hilbert 키로 정렬 | −10.2 % | 0.208 | 0.609 / 0.472 | 4.4 / 5.4 ms |
| 3D 최근접 이웃 경로 | 토큰 중심에서 greedy 경로 | −12.3 % | 0.154 | 0.642 / 0.516 | 104 / 246 ms |
| **값 공간 최근접 이웃 경로** | 심볼 ÷ b 에서 L1 greedy 경로 | **−14.7 %** | 0.334 | **0.778 / 0.595** | 149 / 434 ms |
| 위치 + 값 혼합 경로 | 둘을 이어 붙여 L1 경로 | −12.2 % | 0.157 | 0.644 / 0.519 | – |

- residual은 736차원 관측 특징에서 rank-56 복원을 뺀 값이다(채널별 train 표준편차로 정규화).
  사용자 코덱의 residual 분기는 같은 순서 위에서 1D conv를 돌리므로, 그쪽에 미칠 영향의 대리 지표로 쟀다.
- 정렬 없이(토큰 인덱스 순서) 1D 문맥을 쓰면 even/odd, 1/4 앵커, 4단 계층 모두 0.0 %다. 토큰 인덱스에는 공간 지역성이 없다.

### 0.2 문맥

| 문맥 \ 정렬 | Morton | Hilbert | 3D 경로 | 값 공간 경로 | 복호 단계 | 부가 정보 |
| --- | --- | --- | --- | --- | --- | --- |
| even/odd ±1 (현행 구조) | −9.3 | −10.2 | −12.3 | −14.7 | 2 | – |
| even/odd, 창 넓힘 (±1과 ±3) | −9.7 | – | −12.6 | – | 2 | – |
| even/odd + 토큰 내 채널 자기회귀 | −9.9 | – | −12.8 | −15.2 | 2 (채널은 순차) | – |
| 1/4 앵커 + 좌우 앵커 | −11.8 | −13.0 | −14.7 | – | 2 | – |
| 1D 4단 계층 | −14.7 | −16.2 | −19.0 | – | 4 | – |
| 왼쪽 자기회귀 (직전 3개) | −17.1 | – | −20.4 | **−22.7** | 토큰마다 순차 | – |
| 복사 문맥 (직전 8개 중 송신기가 선택) | **−18.7** | – | −19.7 | −22.2 | 토큰마다 순차 | 번호 2.81 bit/토큰 (값 공간 경로 1.24) |

| 순서와 무관한 문맥 | rate | 복호 단계 | 조건 |
| --- | --- | --- | --- |
| 토큰 내 채널 자기회귀만 | −0.9 % | 채널 순차 | – |
| slot 그래프 (1/4 앵커, kNN 4) | −15.4 % | 2 | 정렬하지 않고 slot 순서 유지 |
| slot 그래프 5단 계층 | −20.4 % | 5 | 같음 |

- 단위는 %. "–"는 측정하지 않은 조합이다.
- "토큰마다 순차"는 복원이 순차라는 뜻이다. 분포가 문맥과 무관한(factorized) 엔트로피 모델이면 rANS 복호는 한 번에 끝나고,
  순차인 것은 복원 루프뿐이다(3.2절).

---

## 1. 공통 전제

### 1.1 순서는 공짜다: 사용자 코덱에서 확인한 근거와 조건

근거(사용자 코덱 코드):

- 송신: `compress()`(codec.py:273-316)는 `_make_order(positions)`로 순열을 만들고 특징을 정렬한 뒤 부호화한다.
  순열은 비트스트림에 들어가지 않는다(`CompressedScene(scene.pack(), order)`의 `order`는 송신기 쪽 반환값).
- 수신: `decompress(data, inverse_permutation=None)`(codec.py:318-364)는 기본값에서 순열을 되돌리지 않는다.
  복원된 장면은 정렬된 순서 그대로 렌더링된다(GlobalSplat decoder는 토큰별 readout이라 순서와 무관).
- 따라서 정렬 함수를 바꿔도 비트스트림 형식과 수신기는 그대로다.

조건과 함정:

1. **같은 순서를 score 분기와 residual 분기가 함께 쓴다.** score 분기의 even/odd 문맥뿐 아니라 residual 분기의 1D conv도 순서의
   지역성을 쓴다. residual 분기에는 `g_a`/`g_s`/`h_a`/`h_s`의 stride 2 conv와, 수용 영역이 5·9·17 토큰인 `MultiScaleAdapter1D`가
   있다(residual.py:48-117). 정렬을 바꾸면 두 분기의 바이트를 따로 재야 한다.
   대리 지표로는 값 공간 경로가 residual 쪽에도 유리했다(0.1절, residual lag-1 상관 0.449 → 0.595). 실측은 아니다.
2. **정렬하면 slot 정체성(원래 토큰 인덱스)이 사라진다.** slot 그래프처럼 slot에 기대는 문맥은 정렬과 같이 쓸 수 없다(3.10절).
3. **값에 의존하는 정렬은 양자화 전 연속값으로 정한다.** 사용자 코덱은 예측(base)을 뺀 뒤 양자화하므로 심볼이 순서에 의존한다.
   "심볼로 순서를 정하고, 순서로 예측을 정하고, 예측으로 심볼을 정하는" 순환을 피하려면 analysis 출력(정규화 score)으로 순서를 정한다.
4. **결정성.** 수신기는 순서를 재현하지 않으므로, 송신기 순서가 실행마다 조금 달라도 복호는 맞다.
   다만 학습·평가 재현성을 위해 안정 정렬, 경로 시작점, 동률 처리를 고정해 두는 편이 좋다.

### 1.2 측정 프로토콜 (숫자를 읽는 법)

- 심볼: GlobalSplat-16K. 과정은 다음과 같다.
  - 관측 특징 736차원(geo 224 + tex 512)에서 장면 평균을 뺀다.
  - geo/tex 분리 PCA 44:12로 56성분을 만든다.
  - 성분별 step = q·σᵢ·clamp(√(s_min/sᵢ), 1/64, 1)로 나누고 반올림한다. sᵢ는 렌더 민감도 metric으로 잰 성분 중요도, q = 1.036이다.
  - 대상은 RE10K test 12씬, 씬당 2048 토큰 × 56성분이다.
  - 장면 평균 μ(1.4 KiB)는 모든 방식에 같으므로 rate에서 뺐다.
- 토큰 위치: 토큰이 만드는 Gaussian 8개 중심의 평균이다(송신기만 사용).
  사용자 코덱은 M_max 후보의 가중합(`(positions * weights).sum(dim=2)`)을 쓴다.
- 예측: 성분별 선형 최소제곱(회귀자 + 편향)이다. train 55씬으로 적합해 test 12씬에 적용했다.
- rate: 잔차 `round(x − 예측)`을 성분별 정적 히스토그램(train 잔차, Laplace 평활 α = 0.5)으로 부호화한다고 보고 교차 엔트로피를 썼다.
  실제 rANS와 0.1 % 이내다(E52).
- 사용자 코덱과 다른 점:
  - 양자화 순서: 스크리닝은 먼저 양자화하고(고정 심볼) 무손실 예측 부호화만 바꾼다.
    사용자 코덱은 예측을 빼고 잔차를 양자화한다(closed-loop, 복원 격자가 예측만큼 이동).
  - 예측기: 스크리닝은 성분마다 스칼라 계수 1개(이웃 평균에 곱함)를 쓴다.
    사용자 코덱은 group 안의 채널을 섞는 선형 conv(예: 16×16×3)를 쓴다.
  - 토큰 수: 스크리닝 2048, 사용자 코덱 4096.
- 그래서 상대 비교로만 읽는다. 사용자 코덱의 문맥 이득(−6~8 %, REPORT E82 단서)이 이 표의 Morton + even/odd(−9.3 %)와 같은 급이라,
  상대 개선은 옮겨질 가능성이 높다.

### 1.3 표기

- `x_i ∈ Z^56`: 정렬된 시퀀스 i번째 토큰의 심볼 벡터. `x_ij`는 그 j번째 성분이다.
- `c_i ∈ R^3`: 토큰 중심. `N`: 토큰 수(스크리닝 2048, 사용자 코덱 4096).
- `b_j`: train 심볼의 성분별 평균 절댓값, 코드로는 `q_train.abs().mean(0).clamp_min(0.05)`.
  Laplace(scale b)에서 비트 ≈ |x|/b·log₂e + 상수다. 따라서 `Σ_j |x_ij − x_lj| / b_j`는 "토큰 l로 토큰 i를 예측할 때 잔차 비트"의 근사다.

---

## 2. 정렬

### 2.1 위치 양자화 (Morton·Hilbert 공통)

씬마다 축별 min-max로 [0, 1]에 맞추고 10 bit 정수 격자로 올린다. 사용자 코덱의 `_quantize_positions`(morton.py:51-60)와 같은 방식이다.

```python
def quant(c, p=10):                      # c: [N, 3] token centres
    c = (c - c.min(0).values) / (c.max(0).values - c.min(0).values).clamp_min(1e-9)
    return (c * (2 ** p - 1)).round().long().clamp(0, 2 ** p - 1)
```

- 한 축의 길이가 0이면 `clamp_min`이 0으로 나누기를 막는다.
- 같은 셀에 토큰 여러 개가 들어가면 키가 같아진다(동률).

### 2.2 Morton (Z-order)

축 a의 비트 b를 위치 3b + a에 놓는 비트 인터리브 키로 정렬한다.

```python
def morton_key(q, p=10):                 # q: [N, 3] ints in [0, 2^p)
    key = torch.zeros(q.shape[0], dtype=torch.long)
    for b in range(p):
        for a in range(3):
            key |= ((q[:, a] >> b) & 1) << (3 * b + a)
    return key
order = torch.argsort(morton_key(quant(c)))
```

- 비트 배치는 사용자 코덱 `morton_codes_3d`(morton.py:63-75)와 같다.
- 사용자 코덱은 `argsort(stable=True)`(morton.py:82)를, 스크리닝은 기본 argsort를 썼다. 차이는 동률 처리뿐이다.
- 약점: 8분할 경계마다 큰 점프가 생긴다. 순서상 이웃이 공간에서는 멀리 떨어지는 경우가 있다.

### 2.3 Hilbert

곡선상 연속한 셀이 항상 공간에서도 붙어 있는 곡선이다. 구현은 Skilling(2004)의 "AxestoTranspose" 변환, Gray 처리, 비트 인터리브 순이다.

```python
def hilbert_key(q, p=10):
    X = q.clone()
    Q = 1 << (p - 1)
    while Q > 1:                                    # inverse undo
        Pm = Q - 1
        for i in range(3):
            hit = (X[:, i] & Q) != 0
            x0 = X[:, 0].clone()
            t = (x0 ^ X[:, i]) & Pm
            X[:, 0] = torch.where(hit, x0 ^ Pm, x0 ^ t)       # invert low bits of X0, or exchange low bits of X0 and Xi
            if i > 0:
                X[:, i] = torch.where(hit, X[:, i], X[:, i] ^ t)
        Q >>= 1
    for i in range(1, 3):                           # Gray encode
        X[:, i] ^= X[:, i - 1]
    t = torch.zeros_like(X[:, 0])
    Q = 1 << (p - 1)
    while Q > 1:
        t = torch.where((X[:, 2] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    for i in range(3):
        X[:, i] ^= t
    key = torch.zeros(q.shape[0], dtype=torch.long) # interleave the transposed bits, high bit first
    for b in range(p - 1, -1, -1):
        for i in range(3):
            key = (key << 1) | ((X[:, i] >> b) & 1)
    return key
```

- 검증: 8³·16³ 격자 전체에서 키가 [0, n³)의 전단사였고, 키 순서로 이웃한 셀은 항상 한 칸 차이였다.

```python
q = torch.stack(torch.meshgrid(*[torch.arange(8)] * 3, indexing="ij"), -1).reshape(-1, 3)
key = hilbert_key(q, p=3)
assert key.unique().numel() == 512
o = torch.argsort(key)
assert int((q[o][1:] - q[o][:-1]).abs().sum(1).max()) == 1
```

- 결과: 연속 토큰 거리 0.208(Morton 0.220). rate는 even/odd에서 −10.2 %, 계층에서도 Morton보다 1~1.5 %p 낫다.
- 비용: N = 4096에서 5.4 ms. 사용자 코덱에서는 `morton_codes_3d` 자리에 키 함수만 바꾸면 되므로 가장 쉬운 교체다.

### 2.4 3D 최근접 이웃 경로 (greedy tour)

곡선 대신 씬마다 경로를 직접 만든다. 현재 토큰에서 "아직 방문하지 않은 토큰 중 가장 가까운 것"으로 계속 이동한다.

```python
def greedy_tour(x, start=0):             # x: [N, d]; Euclidean for d = 3, L1 for d > 3 (as in exp83 tour())
    d = torch.cdist(x, x, p=1) if x.shape[1] > 3 else torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    cur, order = start, [start]
    seen = torch.zeros(len(x), dtype=torch.bool); seen[cur] = True
    for _ in range(len(x) - 1):
        dd = d[cur].clone(); dd[seen] = float("inf")
        cur = int(dd.argmin()); order.append(cur); seen[cur] = True
    return torch.tensor(order)
```

- 시작점: E82는 Morton 키가 가장 작은 토큰, E83은 인덱스 0에서 시작했다. even/odd rate가 50.10 vs 50.08 KiB로, 시작점 영향은 무시할 만하다.
- 비용:
  - 거리 행렬이 N²개다(N = 4096이면 float32 64 MB).
  - 시간은 N = 4096 CPU에서 246 ms다. 거의 전부가 N번 도는 파이썬 루프다.
  - KD-tree나 numba로 옮기면 줄일 수 있다(미측정). 경로를 2-opt로 다듬는 것도 실험하지 않았다.
- 결과: 연속 토큰 거리 0.154로 가장 짧다. even/odd −12.3 %(Morton보다 3 %p), 4단 계층 −19.0 %.

### 2.5 값 공간 최근접 이웃 경로 (이번 스크리닝의 최선 정렬)

**아이디어.** 순서가 공짜라면 3D 위치가 아니라 *코딩되는 값*이 비슷한 토큰끼리 이웃하게 놓으면 된다. 1D 문맥은 이웃 값으로 예측하므로,
값이 비슷한 이웃이 예측 오차를 직접 줄인다. 거리는 비트 근사(1.3절)를 쓴다.

```
d(i, l) = Σ_j |s_ij − s_lj| / b_j          (s: 양자화 단위의 값)
```

```python
xs = k.float() / b_sym                   # screening: rounded symbols, b_sym = q_train.abs().mean(0).clamp_min(0.05)
order = greedy_tour(xs, start=0)         # 56-D -> L1
```

- 결과:
  - even/odd −14.7 %(Morton −9.3 %).
  - 왼쪽 자기회귀와 합치면 −22.7 %, 복사 문맥과 합치면 −22.2 %로 이번 스크리닝 전체 최고다.
- 이웃 특성(E83c):
  - 3D 거리는 Morton보다 멀다(0.334 vs 0.220).
  - lag-1 상관은 심볼 0.778, residual 0.595로 모든 정렬 중 가장 높다.
  - 위치가 아니라 값으로 이웃을 맞추는 정렬이고, 그 값에는 residual과 공유하는 구조도 담겨 있다는 뜻이다.
- 비용:
  - N = 4096 CPU에서 434 ms다(N = 2048은 149 ms).
  - 3D 경로(246 ms)와의 차이 약 0.19 s는 56차원 L1 거리 행렬 계산으로 보인다(두 측정의 차이로 추정).
- 사용자 코덱에서는 반올림 심볼 대신 **양자화 전 정규화 score**로 계산한다(1.1절 3번, 적용 코드는 4.1절).
  - 가중치는 "채널별 심볼 1개당 비트"의 근사면 된다. 예: `1 / (group_step · b)`, b는 train에서 잰 평균 |V|.
  - 순서는 송신기 전용이므로 어떤 가중치를 써도 복호에는 영향이 없다.
- 주의:
  - 3D 지역성이 깨진다. 위치 기반 문맥(3D kNN, slot 그래프)과는 같이 쓰지 않는다. 1D 시퀀스 문맥과는 문제가 없다.
  - analysis가 학습되는 코덱이면 학습 중에도 같은 규칙으로 정렬해야 문맥 conv가 그 순서 통계에 맞춰진다.
    순서는 미분되지 않으므로 매 step(또는 몇 step마다) 다시 계산하면 된다.
  - 경로는 거리 동률과 부동소수 차이에 민감하다. 같은 설정을 재실행했을 때 복사 문맥의 잔차가 44.14 → 44.15 KiB로 바뀐 적이 있다.
    복호에는 무관하지만, 실험 비교에서는 이 정도 흔들림을 감안한다.

### 2.6 위치 + 값 혼합 경로

위치와 값을 각각 "최근접 이웃 거리 중앙값"으로 정규화하고, 위치에 가중치 α를 곱해 이어 붙인 뒤 L1 경로를 만든다.

```python
dc = torch.cdist(c, c); dc.fill_diagonal_(float("inf"))
dx = torch.cdist(xs, xs, p=1); dx.fill_diagonal_(float("inf"))
sc_c, sc_x = dc.min(1).values.median(), dx.min(1).values.median()
order = greedy_tour(torch.cat([c / sc_c * 56, xs / sc_x], 1))     # alpha = 56
```

- α = 56에서는 위치가 지배해 모든 문맥에서 3D 경로와 거의 같았다(even/odd −12.2 % vs −12.3 %).
- α를 줄이는 스윕은 하지 않았다. 지금까지는 값 공간 경로 단독이 더 좋다.

---

## 3. 문맥

### 3.1 사용자 코덱의 현재 문맥 (대응 관계 정리용)

group(16, 16, 16, 8)마다 다음을 한다(score_context.py).

```text
base      = mean offset (scene mean -> MLP) + channel prediction (previous groups, 1x1 conv)      # :64-79, :275-278
even:     V_e = (S_e - base_e) / step   -> factorized prior EB_even                               # :379-381
          anchor_delta = (S_hat_e - base_e) at even positions, 0 at odd positions               # :287-291
odd:      odd_base = base_o + Conv(1 x k)(anchor_delta)[odd]     (linear, zero-init, k = 3)       # :80-87, :293-298
          V_o = (S_o - odd_base) / step  -> factorized prior EB_odd (Split)                      # :382-389
```

- 문맥은 분포를 바꾸지 않고 예측(base)만 바꾼다. 엔트로피 모델은 factorized다.
  이는 스크리닝의 "예측을 빼고 정적 분포로 잔차 부호화"와 같은 구조다.
- `score_spatial_stages` 3/4가 계층 깊이 ablation이다.
  - 구현: `split_even_predictors`, `split_odd_predictors`(score_context.py:88-99).
  - 제약: kernel 5 필수, 채널 문맥 끔(config.py:62-72).
- 2단계 구조에서 kernel 5는 실제로 ±1만 본다. odd i의 ±2 위치가 odd(anchor_delta = 0)이기 때문이다. ±3 even까지 보려면 kernel 7이 필요하다.

### 3.2 공통 구조와 복호 순서

- **단계(pass)**: 시퀀스 위치를 단계로 나누고, 단계 k의 토큰은 이전 단계 토큰만 문맥으로 쓴다. 단계 안은 병렬이다.
- **스크리닝 예측기**: 성분별 `pred_ij = Σ_r a_{j,r} · R_r(i)_j + β_j`. 회귀자 R은 이웃 평균 같은 것이고, 성분 j는 이웃의 같은 성분 j만 본다.
  3.9절의 채널 자기회귀만 같은 토큰의 앞 성분을 회귀자로 쓴다(exp83 `design()`).

```python
def design(target, regs, chan, j):                 # target: [n, 56], regs: list of [n, 56]
    cols = [r[:, j:j + 1] for r in regs]
    if chan and j > 0:
        cols.append(target[:, :j])                 # same token, components 0..j-1 (already decoded)
    cols.append(torch.ones(target.shape[0], 1))
    return torch.cat(cols, 1)
beta_j = torch.linalg.lstsq(design(T_tr, R_tr, chan, j), T_tr[:, j:j + 1]).solution   # fitted on train scenes
```

- **잔차**: `r = round(x − pred)`. 정수 x와 실수 pred에 대해 `r = x − round(pred)`이므로, 복호는 `x = r + round(pred)`다.
- **복호 순서의 실제 비용**:
  - factorized(문맥과 무관한) 분포라면 rANS 복호는 문맥과 상관없이 한 번에 끝난다. 순차인 것은 복원 `x̂_i = base_i + step·V̂_i`뿐이다
    (base_i가 앞 토큰의 x̂에 의존).
  - 선형 예측이면 4096 토큰 × group 4개의 작은 행렬-벡터 곱이라 계산은 가볍다. 주 비용은 파이썬 루프 오버헤드다.
  - 문맥이 분포(σ)까지 정하는 모델(학습형 Laplace 등)로 바꾸면 엔트로피 복호 자체가 순차가 된다.
- **학습형으로 옮길 때**: 회귀자를 conv나 MLP 입력으로 주고 출력을 base(또는 Laplace/Gaussian의 μ, σ)로 쓰면 된다.

### 3.3 even/odd ±1 (현행 구조)

- 패스 1: 짝수 위치. 공간 문맥이 없다(채널 문맥만 가능).
- 패스 2: 홀수 i를 `mean(x_{i−1}, x_{i+1})`로 예측한다.

```python
odd = torch.arange(1, N, 2)
nb = torch.stack([odd - 1, (odd + 1).clamp(max=N - 1)], 1)
nb = torch.where(nb % 2 == 0, nb, nb[:, :1])       # only already-coded (even) neighbours; the last odd uses its left one
pred_src = x[nb].mean(1)
```

- 끝 처리: N이 짝수면 마지막 odd(N−1)에는 오른쪽 이웃이 없어 왼쪽만 쓴다.
  사용자 코덱도 zero padding이라 마지막 odd는 왼쪽 anchor만 본다(CODEC_DEEP_DIVE §1.5).
- 결과: Morton −9.3 %. 토큰 절반이 공간 문맥 없이 부호화된다는 것이 이 구조의 한계다.

### 3.4 창 넓히기

- 홀수 i의 회귀자를 2개로 한다: `mean(x_{i±1})`과 `mean(x_{i±3})`. 둘 다 짝수 위치다.
- 끝 처리: 범위를 벗어난 쪽은 clamp한 뒤, 그 위치가 홀수면 왼쪽 이웃으로 대체한다.
- 결과: Morton −9.7 %(+0.4 %p), 3D 경로 −12.6 %(+0.3 %p). conv 수용 영역을 넓혀도 이득이 작다.

### 3.5 1/4 앵커

- 앵커는 `0::4`이고, 나머지 위치 i는 왼쪽 앵커 `⌊i/4⌋·4`와 오른쪽 앵커(+4)의 평균으로 예측한다.

```python
left = (pos // 4) * 4
right = torch.clamp(left + 4, max=N - 4)           # the last group has no right anchor -> both = left anchor
pred_src = (x[left] + x[right]) / 2
```

- 구현 단순화 두 가지(개선 여지):
  - 거리 가중이 없다. i = 4a+1은 왼쪽 앵커에 더 가깝지만 같은 평균을 쓴다.
  - 묶음 안 위치(1, 2, 3)별로 다른 계수를 두지 않았다.
- 결과: Morton −11.8 %, Hilbert −13.0 %, 3D 경로 −14.7 %.

### 3.6 1D 계층 (계층 깊이 ablation과 같은 축)

- 단계: `0::8` → `4::8`(±4) → `2::4`(±2) → `1::2`(±1). 각 단계는 이미 복호된 양쪽 이웃의 평균으로 예측하고, 단계마다 계수를 따로 적합한다.
- 끝 처리: 오른쪽 이웃이 범위를 벗어나면 왼쪽만 쓴다.
- 결과: Morton −14.7 %, Hilbert −16.2 %, 3D 경로 −19.0 %.

### 3.7 왼쪽 자기회귀 (직전 3개)

- i ≥ 3인 모든 토큰은 회귀자 `x_{i−1}`, `x_{i−2}`, `x_{i−3}`을 쓴다. 각각 따로 계수를 두고, 성분 j는 세 토큰의 성분 j만 본다.
- 처음 3개 토큰은 문맥 없이 보낸다.
- 결과: Morton −17.1 %, 3D 경로 −20.4 %, 값 공간 경로 −22.7 %. "모든 토큰이 1D 문맥을 받는" 경우의 상한에 가깝다.
- 사용자 코덱에 넣는다면(미구현 스케치):
  - 송신기도 closed-loop라 순차다: `base_i`를 앞 토큰의 복원값 `Ŝ_{i−1..i−3}`로 계산하고, 양자화하고, `Ŝ_i`를 만든 뒤 다음 토큰으로 넘어간다.
  - rANS 부호화·복호는 모든 V̂가 정해진 뒤 stream 단위로 한 번에 한다(factorized 분포).
  - 학습은 병렬로 할 수 있다. 직전 토큰 입력으로 복원값 대신 surrogate(원값 + 균일 노이즈, 또는 STE)를 넣는 teacher forcing이다.
    사용자 코덱이 even → odd 예측에 이미 쓰는 방식(`score_context_quantization: noise | ste`)과 같다.
  - 4096 단계 순차 루프는 선형 예측이면 계산이 작지만, 신경망 문맥이면 느려진다. 계층 깊이는 이 상한을 병렬로 근사하는 방법이다.

### 3.8 복사(copy) 문맥

**아이디어.** 예측에 쓸 이웃을 고정 규칙(±1)으로 정하지 않고 송신기가 고르게 한다. 고른 번호를 보내는 대가로 예측이 좋아진다.
비디오의 움직임 보상(참조 블록 선택)과 같은 구조다.

송신기(exp83 `eval_copy`):

```python
K = 8
for i in range(1, N):
    cand = x[max(0, i - K):i]                        # already decoded tokens
    cost = ((x[i][None] - cand).abs() / b).sum(1)    # approx. bits of the residual (1.3)
    m = int(cost.argmin())                           # ties -> the oldest candidate
    index[i] = cand.shape[0] - 1 - m                 # 0 = previous token, 1 = two back, ...
    resid[i] = x[i] - cand[m]                        # plain copy, no gain/bias
```

- 비트스트림은 번호 스트림(0..K−1, 정적 분포 1개)과 잔차 스트림(성분별 정적 분포) 두 개다.
- 첫 토큰은 원래 심볼을 그대로 보낸다.

수신기:

```python
for i in range(1, N):
    back = decode_index(i) + 1
    x[i] = x[i - back] + decode_residual(i)
```

측정(E83b, 12씬):

| 정렬 | 잔차 | 번호 | 합계 rate | 번호 분포 (1칸 뒤 … 8칸 뒤) |
| --- | --- | --- | --- | --- |
| Morton | 45.71 KiB | 0.70 KiB (2.81 bit/토큰) | −18.7 % | 28, 18, 13, 10, 9, 8, 7, 7 % |
| 값 공간 경로 | 44.15 KiB | 0.31 KiB (1.24 bit/토큰) | −22.2 % | 80, 8, 4, 2, 2, 1, 1, 1 % |

- Morton에서는 선택이 넓게 퍼진다. 순서상 바로 옆이 가장 비슷한 토큰이 아닌 경우가 많고, 번호가 그 정보를 사 온다.
  값 공간 경로에서는 80 %가 바로 앞 토큰이어서 번호가 싸다.
- 복사가 왼쪽 자기회귀보다 나은 것은 Morton에서뿐이다(−18.7 vs −17.1 %).
  3D 경로·값 공간 경로에서는 자기회귀가 약간 낫다(−20.4 vs −19.7 %, −22.7 vs −22.2 %).
  복사는 정렬이 못 맞춘 이웃을 번호로 사 오는 장치라서, 정렬이 좋아질수록 이득이 줄어든다.
- 사용자 코덱 2단계 구조에 넣는 형태(미측정): odd i의 후보를 이미 복호된 even(i±1, ±3, ±5, ±7, 8개)으로 제한한다.
  그러면 단계 안 병렬성이 유지되고, `_spatial_prediction`의 입력을 "선택된 even 하나의 anchor_delta × 채널별 이득"으로 바꾸면 된다(4.2절).
- 확장 후보(미실험):
  - 후보에 "고정 규칙 예측(±1 평균)"을 넣는다. 번호 비용 대비 이득이 없는 토큰은 기본값을 쓰게 하려는 것이다.
  - 복사한 값에 성분별 이득·편향을 적용한다(지금은 그대로 복사).
  - 선택 기준을 근사 비트(|r|/b)에서 실제 엔트로피 모델의 비트로 바꾼다.
- 학습형으로 옮길 때: 선택은 송신기의 argmin이라 미분이 필요 없다. 매 step 현재 값으로 다시 고르고, 번호 비트(정적 또는 학습형
  범주 분포)를 rate에 포함한다.

### 3.9 토큰 내 채널 자기회귀

- 성분 j를 같은 토큰의 앞 성분 0..j−1로 예측한다(j개 회귀자 + 편향). 정렬과 무관하다.
- 결과: 단독 −0.9 %, even/odd 위에 얹으면 +0.5~0.6 %p(Morton −9.9, 3D 경로 −12.8, 값 공간 경로 −15.2 %).
  PCA 성분은 서로 거의 비상관이라 선형 예측으로는 이득이 작다.
- 사용자 코덱의 channel 예측(앞 group들 → 1×1 conv)이 이 축이다. 비선형이면 조금 더 있을 수 있다.

### 3.10 slot 그래프 (순서 무관 3D 문맥)

- **전제**: 토큰을 정렬하지 않고 원래 slot 순서를 유지한다.
- **준비(오프라인, 한 번)**:
  - train 씬마다 slot별 토큰 중심을 구해 평균한다: `slot_c` [N, 3].
  - 단계별로, 이미 복호된 slot 중 `slot_c` 거리로 가장 가까운 4개를 이웃 표로 만든다(N × 4 정수, 모델에 고정).
- **복호**:
  - 2단: 앵커 `0::4` → 나머지(가장 가까운 앵커 4개 평균으로 예측).
  - 5단: `0::16 → 8::16 → 4::8 → 2::4 → 1::2`, 단계마다 이미 복호된 slot 중 가장 가까운 4개를 쓴다.
- **결과**: 2단 −15.4 %, 5단 −20.4 %.
- **왜 되나(E76)**: slot 중심의 씬 간 편차(중앙값 4.27)는 이웃 거리(0.28)보다 훨씬 크다. 그런데도 slot 그래프가 통한다.
  이득의 원천은 3D 근접보다 slot 구조다(GlobalSplat geo 토큰은 고정 공간 slot처럼 동작한다).
- **사용자 코덱에는 바로 못 넣는다**:
  - 정렬을 없애면 residual 분기의 1D conv가 쓸 지역성도 사라진다(slot 순서의 residual lag-1 상관 −0.002).
  - slot 그래프를 score에만 쓰려면, residual은 수신기가 계산할 수 있는 다른 순서로 보내는 설계가 필요하다(미검토).
  - 4096 slot 모델이면 이웃 표를 다시 만든다.

---

## 4. 사용자 코덱에 넣는 법

### 4.1 정렬 교체 (비트스트림·수신기 변경 없음)

- 바꿀 곳은 `_make_order`(codec.py:157-158)다.
  `MortonOrder(permutation, invert_permutation(permutation), codes)`(morton.py:11-23, 38-48) 형태로 돌려주면 `forward`·`compress`가 그대로 쓴다.
  compression 패키지 안에서 `.codes`를 읽는 곳은 없었다(grep 기준).
- Hilbert·3D 경로는 입력이 `positions`뿐이라 `_make_order`만 바꾸면 된다.

```python
def _make_order(self, positions):                                  # positions: [B, N, 3]
    perm = torch.stack([greedy_tour(p) for p in positions])        # or argsort(hilbert_key(quantise(p)), stable=True)
    return MortonOrder(perm, invert_permutation(perm), perm)
```

- 값 공간 경로에는 score가 필요하다.
  - analysis는 토큰별 연산이다(`shared_basis` 사영 + Linear-GELU-Linear MLP, codec.py:96-103, 151-155).
  - 장면 평균은 순서와 무관하다.
  - 따라서 정렬 전에 score를 계산해도 같은 값이다. `forward`(codec.py:260-263)와 `compress`(codec.py:279-281)를 이렇게 바꾼다.

```python
features = self._pack_features(texture, geometry)
mean = self._scene_mean(features)                                  # order-invariant
with torch.no_grad():
    s = self._analyze_low_rank(features - mean[:, None, :]) / self.score_scale
    perm = torch.stack([greedy_tour(si / chan_weight) for si in s])   # chan_weight: per-channel bits scale (2.5)
order = MortonOrder(perm, invert_permutation(perm), perm)
features = order.apply(features)
```

- 학습 비용:
  - 값 공간 경로는 N = 4096에서 장면당 약 0.43 s(CPU)다. 매 step 다시 계산하면 그만큼 학습이 느려진다.
  - 대안 1: 데이터 로더 쪽에서 미리 계산하고 몇 천 step마다 갱신한다. analysis가 바뀌는 동안 순서가 조금 낡는다.
  - 대안 2: 6.3절의 O(N log N) 근사(미검증).
- 비교 순서 제안: Morton → Hilbert(키만 교체) → 3D 경로 → 값 공간 경로.
  - 각 정렬에서 문맥과 residual 분기를 다시 학습(또는 미세조정)한 뒤 score·residual 바이트를 **따로** 실측한다.
  - 문맥 conv와 residual conv가 Morton 순서 통계에 맞춰져 있으므로, 평가 때만 정렬을 바꾸는 비교는 불공정하다.

### 4.2 문맥 교체

- **복사 문맥(2단계 구조 유지형)**:
  - `_spatial_prediction`(score_context.py:293-298)의 출력을 "후보 even 8개 중 선택된 하나의 anchor_delta × 채널별 이득"으로 바꾼다.
  - 선택 번호는 새 stream으로 보낸다(`_odd_compress`/`_odd_decompress`, score_context.py:331-354).
  - 새 flag를 `KNOWN_FLAGS`(score_context.py:28)에 등록해 구버전 bitstream을 거부한다.
  - 송신기는 후보 중 `|S_o − 예측| / step`의 비트 근사가 가장 작은 것을 고른다.
- **왼쪽 자기회귀**: forward의 2-pass 루프(score_context.py:374-395)를 토큰 순차 루프로 바꾼다. 학습은 3.7절의 surrogate 방식으로 병렬로 한다.
- **계층 깊이**: 이미 `score_spatial_stages` 3/4로 구현돼 있다. 정렬 교체와 독립이라 조합해서 볼 수 있다.

### 4.3 검증 체크리스트

- 정렬 결정성: 안정 정렬, 경로 시작점, 동률 처리를 고정했는가.
- 순서를 양자화 전 값으로 정했는가.
- score·residual 바이트를 따로, 추정치가 아닌 실제 bitstream 바이트로 비교했는가.
- 송신기 시간: 정렬 계산(값 공간 경로 0.43 s/장면)과 순차 복원 루프 시간을 기록했는가.
- 이 문서의 숫자는 16K 해석적 심볼 + 선형 예측 기준이다. 사용자 코덱에서도 같은 순서로 개선되는지부터 확인한다.

---

## 5. 함정 모음

| 함정 | 증상 | 대처 |
| --- | --- | --- |
| 같은 Morton 셀의 동률 | 실행·장치마다 순서가 달라짐 | `argsort(stable=True)`(사용자 코덱은 이미 사용) |
| 한 축의 길이 0 | 0으로 나누기 | `clamp_min(eps)`(2.1절) |
| 경로의 동률·부동소수 차이 | 재실행 결과가 조금 다름(44.14 → 44.15 KiB) | 시작점 고정, 비교 시 흔들림 감안 |
| 값 의존 정렬의 순환 | 순서 ↔ 예측 ↔ 심볼이 서로 의존 | 양자화 전 score로 순서 결정 |
| closed-loop 예측의 송수신 불일치 | base가 1 ulp만 달라도 다른 격자로 복원되고, 순차 문맥에서는 오차가 다음 토큰으로 전파됨 | 사용자 코드가 문맥 계산에서 autocast를 끄는 이유와 같다(codec.py:203 주석). fp32·같은 연산 순서를 유지하고, 순차 문맥이면 더 엄격하게 |
| SVD 기저의 스레드 수 의존 | 기저가 축퇴 부분공간 안에서 회전(E78에서 확인) | 순서·이웃 표를 SVD 기저로 계산한다면 기저를 저장해서 쓴다 |
| 정렬 후 slot 문맥 사용 | slot 정체성이 없어 이웃 표가 무의미 | slot 문맥은 정렬하지 않을 때만 |
| 정렬 교체 후 residual 분기 미측정 | score는 줄었는데 전체는 늘 수 있음 | 두 분기를 따로 측정 |

---

## 6. 다음 스크리닝 후보 (미실행, 각각 CPU 수 분)

1. **2단계 구조 안의 복사 문맥**: odd i의 후보를 이미 복호된 even(i±1, ±3, ±5, ±7)으로 제한한다.
   사용자 구조에 바로 들어가는 형태라 가장 실용적인 확인이다.
2. **값 공간 경로 + 계층(1/4 앵커, 4단)**: 사용자가 진행 중인 계층 깊이 ablation과 정렬 개선이 서로 더해지는지 본다.
3. **경로 없는 값 공간 정렬**: 심볼의 상위 3개 PCA 좌표에 Hilbert 키를 쓴다(O(N log N), 수 ms).
   학습 중 매 step 정렬이 필요할 때 경로(0.43 s)의 대안이 되는지 본다.

---

## 7. 재현

```bash
cd scratch_experiments
python exp82_sort_context_screen.py      # orders x {evenodd, quarter, lod1d} + slot graph / slot LoD
python exp83_order_context_screen2.py    # value-space / hybrid tours, wide window, AR3, copy8, channel AR
python exp83b_copy_index_cost.py         # copy context: index bits and choice distribution
python exp83c_order_locality.py          # 3D distance, lag-1 correlation (symbols, residual), order timing
```

- 필요한 것: GlobalSplat-16K 체크포인트(`common.load_model`), 토큰 캐시(`load_scene_cache("re10k_16k", …)`), 민감도 로그(`logs/chain_h_16k.log`).
- 모두 CPU만 쓴다. exp83 계열은 exp82의 심볼·정렬 함수를 소스에서 불러와 재사용한다.

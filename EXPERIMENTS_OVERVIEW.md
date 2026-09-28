# GlobalSplat + NFC-GS 토큰 코덱 — 전체 실험 구조 안내

**대상 독자:** 이 workspace를 처음 여는 사람. 3D Gaussian Splatting이나 learned image compression 중
한쪽만 알아도 읽을 수 있게 용어부터 정의한다.

**작성 기준:** 2026-09-17. 이 문서는 새 측정을 하지 않았고,
`upstream/globalsplat/docs/`의 기존 실험 문서와 workspace의 산출물을 읽어서 하나로 엮은 지도다.
숫자의 원본은 각 절에서 링크한 원 문서와 로그다.

**함께 볼 문서:** 핵심 3요소(Context model / Nonlinear transform / Split)를 코드 수준까지
파고든 [CODEC_DEEP_DIVE.md](CODEC_DEEP_DIVE.md), 그리고 그것을 그림으로 옮긴
[NFC-GS 코덱 도해 다섯 장](https://claude.ai/artifact/VFLK2TcKisX6Uxc28YKt9T).

---

## 0. 3분 요약

- **무엇을 하나:** 여러 장의 사진에서 3D 장면을 한 번에 만들어내는 모델(GlobalSplat)의 **중간 표현을 압축**한다.
  이미지를 압축하는 게 아니라, 모델 내부의 "scene token"을 압축한다.
- **어디를 자르나:** encoder가 만든 토큰과 decoder(Gaussian 생성기) 사이를 잘라서,
  그 경계에 학습형 코덱(analysis → 양자화 → entropy coding → synthesis)을 넣는다.
- **무엇이 payload인가:** 장면 하나당 실제 bitstream. 현재 기준 모델에서 **37 KiB ~ 81 KiB / scene**.
  공유 신경망 가중치는 payload에 포함하지 않는다(별도 배포 비용으로 따로 계산).
- **실험은 3세대로 나뉜다:**
  1. **세대 1** — `experiments/` 아래 76개 설정. 아키텍처 탐색기. 평가에 **test-time 최적화(SGA+)를 포함**.
  2. **세대 2** — 통합 구현 위의 ablation 14/15/15b/16/17 + λ sweep. **test-time 최적화 없음**, 전체 테스트셋.
  3. **세대 3** — score의 **decoder-causal context** 계보. 48개 full-test eval. 현재 기준 모델이 여기서 나왔다.
- **현재 기준 모델:** `Nonlinear32 / rank56 / Morton ON / residual ON / Full context P0 / even-odd Split`.
- **가장 큰 함정:** 세대가 다르면 숫자를 직접 비교하면 안 된다. 평가 프로토콜과 checkpoint 계보가 다르다.
  자세한 건 [§3.5](#35-처음-보는-사람이-가장-많이-틀리는-6가지)에.

---

## 1. 배경 — 이 프로젝트가 푸는 문제

### 1.1 GlobalSplat이 뭔가

GlobalSplat은 **feed-forward 3D Gaussian Splatting** 모델이다. 흐름은 이렇다.

```text
입력: 같은 장면을 찍은 사진 여러 장(context views) + 각 사진의 카메라 정보
  ↓  image/ray tokenizer
  ↓  scene-token encoder  (여러 라운드의 attention)
scene token 4,096개   ← ★ 여기가 압축 대상 ★
  ↓  Gaussian decoder (token 하나당 Gaussian 여러 개를 생성)
3D Gaussian 32,768개
  ↓  rasterize (gsplat)
출력: 학습에 쓰지 않은 새 시점(target views)의 이미지
```

- 장면마다 따로 최적화(per-scene optimization)하지 않는다. 한 번의 forward로 장면을 만든다.
- 이 프로젝트에서 backbone은 공식 `globalsplat-re10k-32k.ckpt`를 쓰고 **코덱 학습 중에는 얼려둔다**(frozen).
  단, 예외적으로 `--from-scratch` / `--joint` 모드와 세대 1의 `e2e` 계열은 backbone도 푼다.

### 1.2 왜 압축하나

scene token을 그대로 저장하면 장면 하나가 이렇게 크다.

| 표현 | 계산 | 크기 |
| --- | --- | ---: |
| 원본 1024-D token | 4096 × 1024 × FP32 | 16,777,216 B (16 MiB) |
| observable 736-D token | 4096 × 736 × FP32 | 12,058,624 B (11.5 MiB) |

이걸 수십 KiB로 줄이는 게 목표다. 즉 **"장면을 파일로 저장/전송한다"**는 시나리오다.
현재 λ=0.0256 기준 약 37 KiB이므로 원본 FP32 대비 대략 수백 배 축소다.

### 1.3 코덱을 어디에 꽂나

encoder의 `DualStreamSlotEncoder._pack_outputs()` 출력과 `TokenCoarseToFine3DGS`(Gaussian decoder) 입력 사이.
코드 상으로는 `globalsplat/model/globalsplat.py`의 packed scene-token 경계이고,
`compress_scene()` / `decompress_scene()` API가 여기에 걸려 있다.

중요한 설계 결정 하나: **decoder가 실제로 쓰는 geometry 성분만 보낸다.**
geometry는 512채널이지만 decoder가 쓰는 유효 차원은 더 작아서,
출력 보존(output-preserving) 선형 변환으로 512 → **224**로 줄인 걸 보낸다.
이걸 "observable geometry"라고 부른다. 이 변환 자체의 화질 손실은 거의 없다
(세대 2 실험 17에서 −0.0628 dB로 측정).

---

## 2. 모든 실험이 공유하는 데이터 경로

### 2.1 전체 파이프라인

```text
[송신기]
GlobalSplat encoder
  → appearance token [B, 4096, 512]
  → geometry token   [B, 4096, 512]
  → geometry projection 512 → 224 (학습 가능, 출력 보존 초기화)
  → concat [B, 4096, 736]                        ... "observable feature"
  → Morton 정렬 (3D 위치 기준 10-bit/축)
  → per-scene mean 계산 → FP16으로 저장/전송 (736 × 2 B = 1,472 B)
  → X = feature − mean                            ... centering
  ├─ low-rank analysis → score U (rank 56)
  │    → 채널별 scale q로 정규화 → context 예측값 빼기 → 정수 양자화
  │    → entropy coding → score stream(들)
  │    → (복호) → low-rank synthesis → X_low
  └─ residual = X − X_low
       → 정규화 → multiscale 1-D hyperprior (analysis → y, hyper → z)
       → z 부호화 → z로 y의 mean/scale 예측 → y 부호화
  → [mean | score streams | residual y | residual z | 컨테이너 헤더] = scene bitstream

[수신기]
scene bitstream
  → mean 읽기 + score 복호 + residual 복호
  → 736-D feature 복원 → appearance 512 / observable geometry 224로 분리
  → (고정된) Gaussian decoder → Gaussian 32,768개 → target view 렌더링
```

**핵심 포인트 3개**

1. 평가는 *실제로* bitstream을 만들고 *실제로* 복호해서 렌더링한다.
   학습 중 쓰는 "estimated bits"(likelihood 기반 추정치)로 압축률을 주장하지 않는다.
2. 수신기는 **Morton 정렬된 순서 그대로** 소비한다. Gaussian은 집합이라 원래 토큰 순서가 필요 없다.
   그래서 permutation이나 raw XYZ를 따로 보내지 않는다.
   → 주의: 이건 "수신기가 XYZ를 안다"는 뜻이 **아니다**. 수신기가 위치를 조건으로 쓰려면 side info가 더 필요하다.
3. 공유 신경망 가중치는 scene payload 밖이다. 배포 비용은 [§9.2](#92-overhead-profile--이득이-배포-비용을-이기는가)에서 따로 계산한다.

### 2.2 용어 사전 (이 문서 전체에서 같은 뜻)

| 용어 | 뜻 |
| --- | --- |
| **token** | scene을 표현하는 4,096개의 벡터 중 하나. 이미지 픽셀이 아니라 3D 슬롯에 가깝다. |
| **observable feature** | appearance 512 + observable geometry 224 = 736차원. 압축 대상 텐서 [4096 × 736]. |
| **score** | observable feature의 low-rank 계수. attention score가 **아니다**. rank 56이면 [4096 × 56]. |
| **residual** | 원 feature에서 low-rank 복원값을 뺀 나머지. 별도 hyperprior 코덱으로 압축. |
| **rank (r)** | low-rank 변환의 차원 수. 8~128까지 실험했고 현재는 56. |
| **λ (rate-lambda)** | RD loss의 rate 가중치. **크면 더 작게(화질↓), 작으면 더 크게(화질↑)**. 주로 0.0064와 0.0256. |
| **Morton** | 3D 위치를 축당 10-bit로 양자화해 비트를 교차시킨 공간 채움 곡선 순서. 공간적으로 가까운 토큰을 배열상 가깝게 만든다. |
| **factorized** | 채널마다 분포는 다르지만, 같은 채널이면 모든 token/scene에 **같은** 확률분포를 쓰는 entropy model. |
| **context** | 이미 복호된 정보로 아직 안 보낸 심볼의 확률/예측값을 바꾸는 것. "decoder-causal"해야 한다. |
| **decoder-causal** | 수신기가 그 시점에 이미 가진 정보만 조건으로 쓴다는 뜻. 이게 깨지면 복호가 불가능하다. |
| **even/odd** | Morton 순서에서 짝수 index 토큰(먼저 복호, anchor)과 홀수 index 토큰(나중, 예측 대상). |
| **actual_bytes** | 실제 bitstream 바이트 수. mean + score + residual y/z + 컨테이너 전부 포함. |
| **BPGA** | bits per Gaussian attribute. Gaussian당 attribute 59개로 정규화한 rate. |
| **entropy model / entropy coder** | 전자는 "확률을 정하는 모듈", 후자는 "그 확률표로 정수를 무손실로 바이트에 담는 알고리즘(rANS)". 둘은 다른 단계다. |
| **matched control** | 같은 부모 checkpoint에서 같은 추가 step만큼 학습한 대조군. 구조 효과와 추가 학습 효과를 분리하려고 항상 만든다. |

### 2.3 Score 경로 — 수식으로

기본형 (context 없음):

```text
mean  = FP16(토큰축 평균(F))          # 송신기도 FP16 복원값을 써야 수신기와 일치
X     = F - mean
U     = X @ W_analysis.T              # [4096,736] → [4096,56]
q     = exp(score_log_scale)          # 학습되는 채널별 양자화 scale
S     = U / q
K     = round(S - median)             # ← 실제로 부호화되는 정수 심볼
S_hat = K + median
U_hat = q * S_hat
X_low = U_hat @ W_synthesis           # [4096,56] → [4096,736]
```

context를 붙이면 **격자 자체가 바뀐다:**

```text
V     = (S - b) / d                   # b: 예측 offset, d: scene별 양자화 step 배수
K     = round(V - median)
S_hat = b + d * (K + median)
```

- `b`는 Mean offset + Channel 예측 + Spatial 예측의 합.
- `d`는 Mean conditioner가 예측하는 scene/채널별 배수. 원래 양자화 간격 `q`가 실질적으로 `q × d`가 된다.
- **그래서 context 실험은 "확률 모델만 바꾼 실험"이 아니다.** 양자화 격자와 복원값도 같이 바뀐다.
  (이 혼동을 없애려고 나중에 세대 3에서 "probability-only" 실험을 따로 한다. → [§8.5](#85-probability-only--복원을-완전히-고정하고-확률만-바꾼다-8개))

### 2.4 Residual 경로 — 처음부터 conditional이었다

```text
residual      = (F - mean) - X_low
residual_norm = (residual - residual_mean) / residual_std     # mean/std는 buffer
y = main_analysis(residual_norm)
z = hyper_analysis(|y|)
```

부호화 순서:

1. `z`를 채널별 `EntropyBottleneck`으로 압축/복호 → `z_hat`
2. `z_hat` → hyper-synthesis → `y`의 위치별 평균 μ와 scale σ 예측
3. `GaussianConditional`이 `round(y − μ)`를 σ에 맞는 CDF로 압축
4. 수신기는 같은 `z_hat`에서 μ, σ를 재계산해 `y_hat` 복원
5. `y_hat` → main synthesis → residual 복원 → 정규화 되돌림
6. 최종 feature = `mean + X_low + residual_hat`

구조 상세:

- Adapter: hidden 96, dilation 1/2/4의 kernel-5 branch 3개(branch당 32차원)
- Main analysis: k5 stride-2 conv 736→192 → GDN → k5 stride-2 conv 192→320
- Hyper analysis: `|y|` 입력, k3 + stride-2 k5 conv 2개 → z (192채널)

> **따라서 "원래 코덱에는 context가 전혀 없었다"는 설명은 틀렸다.**
> score에는 없었지만, residual `y`는 처음부터 `z`에 조건부였다.
> 세대 3이 새로 넣은 건 **score 쪽의 decoder-causal context**다.

### 2.5 payload 구성 — 바이트가 어디로 가나

| 항목 | 크기 | 비고 |
| --- | ---: | --- |
| scene mean | 1,472 B | 736 × FP16. 고정. |
| outer container `E2EM0301` | 80 B | header + checksum |
| residual wrapper | 52 B | residual ON일 때. 합쳐서 `actual_container_bytes` = 132 B |
| score container `SCCTX001` | 24 B prefix + stream당 8 B | rank56 Full은 8 streams → **88 B**. 이미 `actual_score_bytes`에 포함됨(중복 가산 금지) |
| score stream | 전체의 약 **80~84%** | 최적화 leverage가 가장 큰 곳 |
| residual y / z | 나머지 대부분 | z는 매우 작음(100~200 B) |

실제 예 (현재 기준 모델, λ=0.0064):

```text
전체 83,121.693 B = score 69,604.342 + residual y 11,738.844 + residual z 174.507 + mean 1,472 + container 132
```

---

## 3. 측정 방법 — 비교 규칙

### 3.1 평가 프로토콜 (세대 2·3 공통)

| 항목 | 값 |
| --- | --- |
| 데이터셋 | RealEstate10K (RE10K), 256×256 |
| 평가 split | raw test split 전체, 처리 가능한 모든 장면 |
| 장면 수 | **6,991** |
| view 구성 | context 12장 + 겹치지 않는 target 8장 (C12/T8) |
| batch / seed / precision | 1 / 0 / bf16-mixed |
| test-time 최적화 | **없음** (SGA+ 없음, per-scene fitting 없음, gradient update 없음) |
| bitstream | 실제 compress → decompress |
| 이미지 저장 | 안 함 |
| 지표 | PSNR↑, SSIM↑, LPIPS↓, actual bytes/scene, bits/Gaussian, BPGA |

### 3.2 estimated rate vs actual rate

- **estimated rate** (학습용): score/y/z likelihood의 −log2 합 ÷ (Gaussian 수 × 59). 고정 mean/header는 **불포함**.
- **actual rate** (평가용): 실제 bitstream 바이트. mean·score·y·z·컨테이너 **전부 포함**.

논문/보고에 쓰는 건 항상 actual rate다.

### 3.3 matched control 원칙

이 프로젝트의 가장 중요한 방법론 규칙이다.

> 구조 A를 부모에서 +50k 학습해서 좋아졌다면,
> **같은 부모에서 구조를 바꾸지 않고 +50k 학습한 대조군**과 비교해야 한다.

부모와 직접 비교하면 "구조 효과 + 추가 학습 효과"가 섞인다.
실제로 세대 3의 P0/P1/P2 실험에서 이 혼입이 predictor 차이보다 훨씬 컸다([§8.4](#84-spatial-predictor-p0p1p2--추가-25k-9개)).

### 3.4 통계적 신뢰구간

장면 단위 paired bootstrap을 쓴다. 자세한 건 [§9.3](#93-scene-ci--장면-단위-신뢰구간과-bd-rate-screen).

### 3.5 처음 보는 사람이 가장 많이 틀리는 6가지

1. **세대 간 숫자 비교 금지.** 세대 1은 test-time 최적화(SGA+ 500 step)를 포함하고 DL3DV도 쓴다.
   세대 2의 λ=0.0256 rank56 full은 58,444 B / 23.3700 dB인데,
   세대 3 부록의 Linear rank56 λ=0.0256은 48.53 kB / 23.6659 dB다. **다른 계보의 다른 checkpoint다.**
2. **`kB`와 `KiB`를 구분하라.** 세대 3의 표는 `KiB` (=1024 B). 초기 사용자 표는 `kB` 표기이고 원본 바이트가 없어 재계산하지 않았다.
3. **로그의 6,986은 장면 수가 아니다.** warm-up 5회를 뺀 **타이밍 측정 횟수**다. 평가 장면은 6,991.
   렌더러 decoder 타이밍 55,888회는 6,986 × target 8 views.
4. **`step000050000.ckpt`라는 파일명이 같다고 같은 누적 학습량이 아니다.**
   각 stage가 새 optimizer로 step을 0부터 다시 센다. 50k+50k는 "누적 100k"이지 "이어붙인 단일 100k run"이 아니다.
5. **`M_max=16`과 실제 생성 Gaussian 8개를 혼동하지 마라.** final stage=3에서 token당 8개, scene당 32,768개다.
6. **"score가 전체 bit의 80%"는 "score 확률 모델이 80%만큼 나쁘다"는 뜻이 아니다.**
   [§9.1](#91-fixed-symbol-entropy-probe--확률-모델에-얼마나-여유가-남았나)의 probe에서 실제로 뽑아낼 수 있던 여유는 약 0.75%였다.

---

## 4. 공통 학습 recipe

### 4.1 기본 recipe (세대 3 context/joint 기준)

| 항목 | 값 |
| --- | --- |
| 데이터 / 해상도 | RE10K, 256×256 |
| backbone | `globalsplat-re10k-32k.ckpt`, **frozen** |
| 학습 대상 | `feature_codec` scope=`all` (geometry projection 포함). probability-only는 `score_probability` |
| Context sampling | 24-view pool → 공유 endpoint 2개 + 나머지 22개 교대 배분 → A/B branch 각 13 views |
| Targets | 두 subset에 **같은** target 12 views |
| View span / augmentation | 40–220 / ON |
| Subset consistency | ON, step 0부터, parity 교대 |
| Scene batch | micro 2 × accumulate 4 = optimizer step당 8 scene draws |
| Optimizer | Adam, weight decay 0 |
| LR (표준 50k) | 1e-4 → 35k에서 1e-5 → 45k에서 1e-6 |
| Precision / clipping | bf16-mixed / grad-norm clip 0.5 |
| Seed | training 111123, data loader 403 |
| Quantile | deterministic update 500 step 간격 (probability-only는 0 = 끔) |
| Checkpoint | 5k 간격 |
| 실행 | 실험당 GPU 1개, 독립 Slurm array (DDP 아님) |
| 환경 기록 | ariel-v12, RTX A5000, PyTorch 2.5.1+cu121, CUDA 12.1, CompressAI 1.2.8 |

> 초기 Nonlinear transform 50k만 **micro 1 × accumulate 8**이었다. effective batch는 같지만
> micro-batch 연산/난수 경로까지 동일하지는 않다.

### 4.2 Loss

```text
L = 0.5 (D_A + D_B)                                   # A/B 두 subset branch
    + 0.001 · L_alpha_consistency
    + 0.01  · L_depth_consistency
    + λ · estimated_bits / (B_rep · N_gaussian · 59)   # rate 항

D  = 1.0 · RGB MSE
   + 0.05 · LPIPS(VGG)
   + 0.01 · in-view / frustum regularization
```

- consistency 항은 `0.5|A − stopgrad(B)| + 0.5|B − stopgrad(A)|`의 대칭 형태.
  depth는 양쪽 accumulated opacity > 0.01인 영역만 사용.
- rate ramp는 0 (codec-only 학습에서는 처음부터 full weight).
- **A/B subset 구조의 의미:** 같은 장면을 두 가지 context view 조합으로 두 번 인코딩해서,
  두 복원이 서로 일관되도록 강제한다. micro-batch 2 scene × 2 branch = 4 representation을 처리한다
  (16개 독립 장면이 아니다).

### 4.3 stage별 변경 요약

| 단계 | 시작점 | 추가 steps | 누적 | LR | scope |
| --- | --- | ---: | ---: | --- | --- |
| 초기 transform | PCA init + frozen backbone | 50k | 50k | 1e-4, 35k/45k decay | all |
| Linear context / factorized 대조군 | 해당 λ Linear 50k | 50k | 100k | 1e-4, 35k/45k decay | all |
| Nonlinear context / 대조군 | 해당 λ Nonlinear 50k | 50k | 100k | 1e-4, 35k/45k decay | all |
| P0/P1/P2 | 해당 λ Nonlinear context 100k | 25k | 125k | **사실상 상수 1e-4** (milestone이 25k 밖) | all |
| Probability-only | 원래 Nonlinear Full P0 100k | 10k | 110k | 1e-4 → 7k에서 1e-5 | score_probability |
| Full low-LR | 원래 Nonlinear Full P0 100k | 10k | 110k | 상수 1e-6 | all |
| Entropy probe | Full P0+Split 110k | 학습 없음 | — | — | — |

> **새 stage는 부모의 weight만 로드하고 optimizer state는 이어받지 않는다.**
> P 실험에서 LR이 1e-6 → 1e-4로 "재시작"된 게 결과에 크게 작용했다.

---

## 5. Workspace 지도

```text
global/
├── README.md                       현재 활성 경로 요약
├── EXPERIMENTS_OVERVIEW.md         ← 이 문서
│
├── upstream/globalsplat/           ★ 현재 활성 구현 (git repo)
│   ├── globalsplat/compression/    코덱 구현
│   │   ├── codec.py                  analysis/synthesis, train scope, compress/decompress 통합
│   │   ├── score_context.py          Mean/Channel/Spatial context, P1/P2, Split/Gaussian/Conditional
│   │   ├── residual.py               multiscale 1-D hyperprior
│   │   ├── entropy.py                factorized entropy wrapper
│   │   ├── morton.py                 Morton 정렬
│   │   ├── bitstream.py              E2EM0301 / SCCTX001 컨테이너
│   │   ├── checkpoint.py             metadata·shape 추론, strict load, mismatch 검사
│   │   ├── config.py                 CodecConfig + 허용값 검증
│   │   ├── initialization.py         QR/PCA 초기화
│   │   └── entropy_probe.py          고정-symbol probe
│   ├── globalsplat/model/
│   │   ├── globalsplat.py            코덱 삽입 지점, compress_scene/decompress_scene
│   │   └── model_wrapper.py          subset loss, RD 항, quantile 주기, 평가/타이밍 집계
│   ├── config/model/globalsplat_nfcgs_rank56.yaml     코덱 아키텍처 프리셋
│   ├── config/experiment/re10k_32k_nfcgs.yaml         codec-only 학습 recipe
│   ├── config/experiment/re10k_32k_nfcgs_joint.yaml   from-scratch joint recipe
│   ├── scripts/run_nfcgs.py          단일 train/eval 진입점
│   ├── scripts/slurm/                main train/eval, probe, joint 8-GPU 래퍼
│   ├── tests/                        코덱/context/probe/launch-contract 테스트
│   ├── archive/codec_experiments_20260915/   정리 전 74개 파일 스냅샷 (SHA-256 manifest)
│   └── docs/                         ★ 원본 실험 문서 전부
│
├── experiments/                    ★ 세대 1: 76개 설정의 checkpoint/config/평가 산출물
├── archive/reconstructed_codec/     세대 1 checkpoint에서 역복원한 독립 코덱 패키지
│
├── 20260915_113013/                 scene CI 결과 (REPORT.md, paired_scene_metrics.csv)
├── overhead-REPORT.md / .json       overhead profile 결과
├── entropy-probe-0p00*-REPORT.md    entropy probe 결과
├── ablation_16_correlation/         상관/스펙트럼 분석 산출물 (png, pt, json)
├── neighbors_view_00/               토큰 기여도 시각화 산출물
├── logs/ , slurm/                   내려받은 Slurm stdout/stderr
└── 20260915_113013/ 등              날짜 태그 결과 디렉터리
```

### 5.1 원본 문서 읽는 순서

| 순서 | 파일 | 내용 |
| ---: | --- | --- |
| 1 | `docs/NFCGS_CODEC.md` | 현재 지원 경로, 실행 명령어 |
| 2 | `docs/SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md` | 완료 실험 + 결론 (인수인계용 요약) |
| 3 | `docs/NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14.md` | 세대 3 전체 기록 (가장 상세) |
| 4 | `docs/ABLATION_PROGRESS_2026-09-02.md` | 세대 2 ablation 14/15/15b/16/17 |
| 5 | `docs/NFCGS_ENTROPY_PROBE_2026-09-14.md` | entropy probe |
| 6 | `docs/NFCGS_OVERHEAD_PROFILE_2026-09-15.md`, `NFCGS_SCENE_CI_2026-09-15.md` | 진단 실험 |
| 7 | `docs/NFCGS_SCORE_CONTEXT.md`, `NFCGS_TRANSFORM.md`, `NFCGS_SPATIAL_PREDICTOR_EXPERIMENT_2026-09-11.md`, `NFCGS_SCORE_PROBABILITY_EXPERIMENT_2026-09-13.md` | 각 실험의 원 설계 문서 |
| — | `docs/NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json` | 48개 eval의 정밀 수치·바이트 분해·checkpoint 경로 |

---

## 6. 세대 1 — 아카이브된 76개 설정 (`experiments/`)

### 6.1 이 세대의 성격

아키텍처 탐색기다. "어떤 형태의 코덱을 만들 것인가"를 폭넓게 훑었다.
원본 Python 소스는 백업에 없었고, checkpoint·resolved config·bitstream·평가 기록에서
`archive/reconstructed_codec`으로 역복원했다.

각 설정의 디렉터리 구조:

```text
experiments/<setting_id>/
├── experiment.yaml                  schema: gsfc_setting_v2, family, 설명, rate point 목록
└── rate_points/lambda_<L>/
    ├── rate_point.yaml
    ├── checkpoints/stages/step*.ckpt
    ├── configs/<run_id>/*_resolved_config.yaml
    ├── provenance/<run_id>/*_initializer.json, *_runtime.json
    └── evaluations/<model>/<campaign>/<protocol>/<dataset>/models/<model>/metrics.json
```

λ 축은 세대 3보다 넓다: `0p0004 / 0p0016 / 0p0064 / 0p0256 / 0p1024 / 0p4096 / 1p6384` 등.

### 6.2 ★ 세대 1의 평가 프로토콜이 다르다

`metrics.json`의 `protocol` 블록이 보여주는 차이:

```json
{ "dataset": "DL3DV", "context_views": 12, "disjoint_target_views": 8,
  "itt_steps": 500, "itt_quantizer": "SGA+", "actual_entropy_bitstream": true }
```

- **ITT(iterative test-time) 500 step + SGA+ 양자화**를 쓴다. 즉 장면마다 추가 최적화를 한다.
- 결과가 두 개로 보고된다:
  - `step_0` — 추가 최적화 **전** (세대 2·3와 비교 가능한 쪽)
  - `context_best` — 추가 최적화 **후** (훨씬 좋게 나옴)
- 데이터셋이 **RE10K와 DL3DV 둘 다**.
- 평가 규모가 다르다: `screening_30_sga500`(30 장면), `dl3dv_full_140_sga500`(140 장면) 등.

> **그래서 세대 1의 `context_best` 숫자를 세대 2·3의 full-test 숫자와 같은 표에 놓으면 안 된다.**
> 예: `codec_only_control` λ=0.0004 DL3DV에서 step_0은 21.70 dB / 236 KB이고
> context_best는 26.52 dB / 242 KB다. 4.8 dB 차이가 전부 test-time 최적화 효과다.

주요 평가 캠페인 이름: `screening_30_sga500_20260824`, `dl3dv_full_140_sga500_20260824`,
`recent_remote_50k_inference_20260825`, `rank_tying_nonlinearity_matrix_50k`,
`rank_nonlinearity_followup_50k`, `split_point_ablation_50k`, `decoder_adapter_highrate_50k`,
`rank_matrix_completion_50k`.

### 6.3 76개 설정의 5개 family

#### (A) `reference` — 1개

| 설정 | 내용 |
| --- | --- |
| `official_globalsplat` | 코덱을 넣지 않은 pretrained GlobalSplat. 모든 비교의 기준. |

#### (B) `prototype` — 8개: 초기 코덱 형태 탐색

| 설정 | 무엇을 시험했나 |
| --- | --- |
| `geometry_conditioned_2d_codec_only_distortion` | geometry로 조건화한 **2D** feature codec, codec만 학습, distortion만 |
| `geometry_conditioned_2d_joint_distortion_from_codec_only_step5000` | 위에서 이어서 GlobalSplat까지 함께 학습 |
| `multiscale1d_screen` / `multiscale1d_from_screen_step750` | 저차원 token residual에 multiscale **1D** hyperprior 직접 적용 |
| `multiscale1d_morton1d` | 위 + **Morton 순서** 배열 |
| `observable_lowrank_1d_codec_only_distortion` | ★ observable geometry + appearance 저차원화 + 1D residual codec, distortion-only |
| `observable_lowrank_1d_joint_distortion_from_codec_only_step5000` | 위에서 joint distortion 학습 |
| `observable_lowrank_1d_joint_rd_from_joint_distortion_step25000` | 위에서 **R-D** 목적함수로 전환 |

> 이 갈래가 현재 아키텍처의 조상이다. `2D → 1D`, `raw geometry → observable geometry`,
> `distortion-only → joint → R-D`로 단계적으로 옮겨간 흔적이 그대로 남아 있다.

#### (C) `frozen_globalsplat_codec` — 37개: low-rank transform 본체 탐색

GlobalSplat을 얼리고 코덱만 학습한다. 세대 2·3와 같은 설정이다. 축이 여러 개다.

**축 1 — basis를 누가 갖나**

| 종류 | 설정 예 | 의미 |
| --- | --- | --- |
| **shared** | `shared_lowrank_rank{8,16,32,40,48,56,64,72,80,96,128}_*` | 모든 장면이 같은 basis를 공유. basis는 모델 안에 있고 전송 안 함 |
| **scene** | `scene_lowrank_rank{8,16,32,64}_multiscale1d` | 장면별 basis. 표현력은 크지만 basis 전송 비용이 생김 |

**축 2 — analysis와 synthesis를 묶나(tied) 푸나(untied)**

| 종류 | 설정 예 | 의미 |
| --- | --- | --- |
| tied | `shared_lowrank_rank48_multiscale1d` | `W_synthesis = W_analysis^T`. 파라미터 절반 |
| untied | `shared_lowrank_rank48_untied_synthesis_multiscale1d` | 둘을 독립 학습 |
| untied + scale | `shared_lowrank_rank48_untied_scaled_synthesis_multiscale1d` | untied + 별도 learnable scale |

**축 3 — 비선형을 어디에 넣나**

| 종류 | 설정 예 | 의미 |
| --- | --- | --- |
| nonlinear synthesis (tied 유지) | `shared_lowrank_rank{48,56}_nonlinear*_synthesis_multiscale1d` | decoder 쪽에만 비선형 |
| untied + nonlinear synthesis | `..._untied_nonlinear{32,64}_synthesis_multiscale1d` | 둘 다 |
| joint nonlinear | `shared_lowrank_rank56_joint_nonlinear32_multiscale1d` | **analysis와 synthesis 양쪽** 비선형 ← 세대 3의 Nonlinear32가 여기서 나옴 |

**축 4 — entropy model 구조**

| 설정 | 의미 |
| --- | --- |
| `shared_lowrank_rank16_conditional_score_hyperprior_multiscale1d` | score entropy model에 **별도 조건(hyperprior)** 추가 |
| `shared_lowrank_rank32_pairwise_conditional_multiscale1d` | **인접 score 쌍** 사이의 조건부 의존성 |
| `shared_lowrank_rank56_untied_nonlinear64_basecond_entropy_multiscale1d` | 복호된 **low-rank base를 residual entropy model의 조건**으로 사용 |

> 마지막 것은 세대 3에서 "아직 완료 결과 없음"으로 남아 있는 아이디어와 같은 계열이다.
> 세대 1 산출물이 남아 있으니 다시 볼 가치가 있다.

**축 5 — score의 구조적 변환 / 용량 배분**

| 설정 | 의미 |
| --- | --- |
| `shared_lowrank_rank64_block_delta_multiscale1d` | 정렬된 token score를 **block 단위 차분**으로 변환해 부호화 |
| `shared_lowrank_rank64_grouped16_multiscale1d` | token을 **16개씩 묶어** group-local 구조 활용 |
| `shared_lowrank_rank48_mixture4_block16_multiscale1d` | **transform 4개 mixture** + 16-token block |
| `shared_lowrank_rank64_split_t44_g20_multiscale1d` | texture 44 / geometry 20으로 **rank를 나눠 배정** |

#### (D) `split_or_decoder_ablation` — 18개: 코덱을 어디서 자를 것인가

| 그룹 | 설정 | 의미 |
| --- | --- | --- |
| **pre-Mixer split** (9개) | `pre_mixer_shared_lowrank_rank{40,56,64}_[untied[_nonlinear64]]_synthesis_multiscale1d` | 압축 지점을 **마지막 Mixer MLP 앞**으로 옮김. decoder 쪽 연산·표현력이 늘어남 |
| **decoder adapter** (9개) | `rank40_tied_ / rank56_nonlinear64_ / rank64_untied_` × `adapter_joint64 / adapter_separate32_16 / adapter_separate64_32` | 기본 post-Mixer split을 유지하고, Gaussian decoder 앞에 **가벼운 residual adapter**를 추가. joint = 공유형 1개, separate = texture/geometry 분리형 |

> 즉 "코덱 경계를 앞뒤로 옮기기" vs "경계는 그대로 두고 수신측 보정기를 붙이기"를 각각 rank 3종에 대해 비교했다.

#### (E) `e2e` — 12개: backbone을 얼마나 풀 것인가

feature codec checkpoint에서 이어서, **어디까지 풀지**와 **LR**을 바꾼다.

| 범위 | 설정 | 푸는 대상 |
| --- | --- | --- |
| 대조군 | `codec_only_control` | 아무것도 안 품. codec만 계속 학습 |
| 좁게 | `gaussian_decoder_lr{1e6,1e7,3e7}` | Gaussian decoder만 |
| 중간 | `final_mixer_decoder_lr{1e6,1e7,3e7}` | final Mixer + Gaussian decoder |
| 전체 | `full_globalsplat_lr{1e5,5e6,1e6,1e7,3e7}` | GlobalSplat 전체 |

LR이 1e-5 ~ 1e-7의 매우 낮은 구간인 게 포인트다.
이미 잘 학습된 backbone을 망가뜨리지 않으면서 코덱에 적응시키는 게 목적이므로,
"얼마나 살살 풀어야 이득이 나는가"가 실제 변수다.

### 6.4 세대 1을 어떻게 쓸 것인가

- 이 산출물들은 **현재 활성 import/test 경로가 아니다.** 참고 자료다.
- 그러나 세대 2·3이 시도하지 않은 축(scene-specific basis, mixture transform, block delta,
  pairwise conditional, base-conditioned residual entropy, split point, decoder adapter, e2e unfreezing)이
  전부 여기 있다. 새 실험을 설계할 때 "이미 해봤나?"를 먼저 여기서 확인하라.
- checkpoint를 다시 쓰려면 `archive/reconstructed_codec/README.md`의
  `load_feature_codec_checkpoint()` 경로를 쓴다.
  단, 아카이브된 `M3HPRANS` residual payload는 **bit-exact 복호를 보장하지 않는다**(원 serializer가 백업에 없음).

---

## 7. 세대 2 — 통합 구현 위의 ablation (14/15/15b/16/17 + RD sweep)

원 문서: `docs/ABLATION_PROGRESS_2026-09-02.md`

이 세대부터 **test-time 최적화를 완전히 빼고**, RE10K 전체 테스트 6,991 장면으로 평가한다.
실험 번호는 실행 순서(17 → 16 → 14 → 15 → 15b)와 다르게 붙어 있다.

### 7.1 실험 17 — 단계별 누적 ablation: 화질 손실은 어디서 나나

**질문:** "채널 축소 / 투영 / 양자화 / residual 중 무엇이 화질을 깎나?"

**방법:** leave-one-out이 아니라 **cumulative stage-wise**. 왼쪽부터 모듈을 하나씩 켠다.

| 조건 | observable geometry + concat | centering + low-rank | score 양자화 + bitstream | residual codec |
| --- | :---: | :---: | :---: | :---: |
| `upper` (vanilla GlobalSplat) | ✗ | ✗ | ✗ | ✗ |
| `concat_only` | ✓ | ✗ | ✗ | ✗ |
| `projection` | ✓ | 연속값 | ✗ | ✗ |
| `projection_quantized` | ✓ | ✓ | ✓ | ✗ |
| `full` | ✓ | ✓ | ✓ | ✓ |

- **eval-only 실험이다.** rank별로 하나의 full-codec checkpoint(λ=0.0256, step 50k, untied linear synthesis)를 쓰고
  `feature_codec.ablation_mode`만 바꾼다. 세 조건을 따로 재학습한 게 아니다.
- rank 40/56/72/80은 **독립적으로 학습된 모델**이라 rank에 대해 단조일 필요가 없다.

**결과 (RE10K 전체 테스트)**

| 조건 | PSNR | SSIM | LPIPS | Bytes/scene | BPGA |
| --- | ---: | ---: | ---: | ---: | ---: |
| upper | 24.7004 | 0.7682 | 0.2480 | — | — |
| concat_only | 24.6376 | 0.7664 | 0.2494 | — | — |
| rank40_full | 23.3062 | 0.7272 | 0.2889 | 54,735 | 0.2265 |
| rank56_full | **23.3700** | 0.7290 | 0.2870 | 58,444 | 0.2418 |
| rank72_full | 23.1918 | **0.7307** | 0.2864 | 94,217 | 0.3899 |
| rank80_full | 23.0444 | 0.7305 | **0.2836** | 97,568 | 0.4037 |

**단계별 손실 분해**

| Rank | 투영 손실 (vs concat) | 양자화 손실 | residual 회복 | 최종 손실 (vs upper) |
| ---: | ---: | ---: | ---: | ---: |
| 40 | −6.2874 dB | −0.4198 dB | **+5.3758 dB** | −1.3942 dB |
| 56 | −4.0791 | −0.3017 | **+3.1132** | −1.3304 |
| 72 | −5.1843 | −0.2460 | **+3.9845** | −1.5086 |
| 80 | −4.5318 | −0.2362 | **+3.1748** | −1.6560 |

**결론**

1. observable geometry 변환 + concat/split은 사실상 무손실 (−0.0628 dB).
2. **주된 손실은 low-rank 투영**(−4.1 ~ −6.3 dB). entropy coding이 아니다.
3. 양자화 손실은 훨씬 작다 (−0.24 ~ −0.42 dB).
4. residual 경로가 3.1~5.4 dB를 되찾는다. **"low-rank만의 코덱"이라고 부를 수 없다.**
5. rank 56이 full PSNR 최고, rank 40이 rate 효율 최고. rank 72/80은 residual payload만 커지고 이득이 없다.

> ⚠️ `upper`라는 이름은 "수학적 상한"이 아니다. 렌더링 목적으로 학습된 코덱이
> frozen backbone 표현을 오히려 정칙화할 수 있다. 실제로 §7.6에서 상한을 넘는다.
> 올바른 라벨은 **"codec-free reference"**다.

### 7.2 실험 16 — 상관/스펙트럼 분석 (학습·평가 아님)

**질문:** "low-rank가 정당한가? Morton은 왜 필요한가?"

**방법:** 순수 통계 분석. 6,991 장면 전부를 훑으면서 장면당 64개 토큰(공분산),
32개 채널(token-lag 상관)을 균등 샘플링. 각 stream을 장면별 토큰 평균으로 centering(코덱과 동일).

**채널 상관 + 공분산 스펙트럼**

| Stream | 평균 \|off-diag 상관\| | rank 40 에너지 | rank 56 | rank 72 | rank 80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| appearance | 0.1241 | 88.80% | 91.00% | 92.49% | 93.08% |
| raw geometry | 0.0905 | 98.63% | 98.90% | 99.09% | 99.16% |
| observable geometry | 0.4233 | 99.45% | 99.62% | 99.72% | 99.76% |
| observable concat (736-D) | 0.1291 | 97.31% | 97.93% | 98.29% | 98.42% |

**토큰축 상관 (평균 \|채널별 Pearson\|)**

| Stream | lag 1 | lag 2 | lag 4 | lag 16 | lag 64 | lag 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| observable concat, **정렬 안 함** | 0.0055 | 0.0053 | 0.0077 | 0.0043 | 0.0037 | 0.0033 |
| observable concat, **Morton** | **0.4739** | **0.4485** | **0.4174** | **0.3478** | **0.2605** | **0.1369** |

**결론**

1. 저차원 구조는 실재한다. rank 40에서 이미 concat 공분산 에너지의 97.31%.
2. 그 집중의 대부분은 **geometry에서 온다.** appearance가 어려운 쪽(rank 56에서 91%).
3. **에너지 보존율이 높다고 화질이 보장되지 않는다.** 실험 17에서 rank 56이 양자화 전에 이미 4.08 dB를 잃는다.
   저에너지 방향이 렌더링에 결정적일 수 있다.
4. rank 72/80의 통계적 이득은 0.36 / 0.13 %p뿐. 실험 17의 결과와 일치.
5. Morton은 **거의 무상관인 토큰 열을 강한 지역 상관을 가진 열로 바꾼다** (lag-1 0.0055 → 0.4739).
   1-D residual conv가 의미를 갖는 이유.

산출물: `ablation_16_correlation/` (summary.json, correlation_and_spectrum.pt, 3종 png)

### 7.3 실험 14 — linear vs nonlinear vs residual (신규 학습 3개)

**질문:** "같은 예산으로 새로 학습하면 어느 구조가 이기나?"

| Task | 조건 | 투영 구조 | residual |
| ---: | --- | --- | --- |
| 0 | `linear_no_residual` | untied affine/linear | 없음 |
| 1 | `nonlinear32_no_residual` | linear + GELU MLP (hidden 32) | 없음 |
| 2 | `linear_residual` | untied affine/linear | multiscale 1-D |

공통: frozen backbone, Adam 1e-4, milestone 35k/45k, 50k steps, micro 1 × accum 8,
bf16, λ=0.0256, quantile 500 step. **`CODEC_INIT`(PCA 초기화)을 비워둔 채로 돌아갔다.**

**결과**

| 조건 | PSNR | SSIM | LPIPS | Bytes/scene | BPGA |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_no_residual | 20.5875 | 0.6347 | 0.3990 | 55,625 | 0.2302 |
| nonlinear32_no_residual | **21.4757** | **0.6649** | **0.3387** | 51,391 | 0.2127 |
| linear_residual | 20.1332 | 0.5971 | 0.4011 | **9,581** | **0.0396** |

**해석**

- 같은 rank/λ/optimizer/예산에서 **nonlinear가 linear를 전면 우세**: +0.8882 dB, +0.0302 SSIM,
  −0.0603 LPIPS, −4.23 KB. → 세대 3이 Nonlinear32를 채택한 근거.
- `linear_residual`은 **rate 지배 해**로 수렴했다 (9.58 KB). 이건 "residual이 나쁘다"가 아니라
  **같은 λ에서도 서로 다른 operating point로 수렴할 수 있다**는 증거다.
  5th/median/95th = 8.39/9.42/11.34 KB로 이상치 때문이 아니다.
- 아카이브 rank56 full(23.3700 dB / 58.44 KB)과 매우 달라서, 초기화·학습 이력·rate 배분을 감사해야 한다.
- **feature MSE가 낮은 모델이 렌더링 지표에서 가장 나빴다.** feature-space MSE는 task distortion의 대체재가 아니다.

> 인프라 메모: 첫 제출(409785)은 fused Adam이 Lightning AMP gradient clipping과 충돌해 실패.
> 모델 결과가 아니다. 이후 non-fused Adam이 기본값이 됐다.

### 7.4 실험 15 — Morton 불변성 검사 (sanity check)

**질문:** "순서 의존 모듈이 없는 모델이면 순서를 바꿔도 결과가 같아야 한다. 정말 같은가?"

실험 14의 `linear_no_residual` checkpoint 하나로 추론 시 순서만 3가지로 바꾼다.

| 순서 | PSNR | SSIM | LPIPS | Bytes/scene |
| --- | ---: | ---: | ---: | ---: |
| none | 20.5875 | 0.6347 | 0.3990 | 55,625.397797 |
| Morton | 20.5875 | 0.6347 | 0.3990 | 55,625.396653 |
| deterministic random | 20.5875 | 0.6347 | 0.3990 | 55,625.397225 |

최대 차이 0.001144 B/scene. **permutation/역permutation 부기가 정확하다는 뜻**이고,
동시에 **Morton 자체는 pointwise 코덱에 아무 이득이 없다**는 뜻이다.

### 7.5 실험 15b — residual 코덱이 켜지면 순서가 중요해지나

같은 질문을 **order-dependent 모듈이 있는 모델**에 던진다.
아카이브 rank-56 λ=0.0256 Morton 학습 full residual checkpoint를 고정하고 추론 순서만 바꾼다.

| 추론 순서 | PSNR | SSIM | LPIPS | Bytes/scene |
| --- | ---: | ---: | ---: | ---: |
| none | 22.8059 | 0.7232 | 0.3068 | 73,978.98 |
| **Morton** | **23.3700** | **0.7290** | **0.2870** | **58,444.20** |
| deterministic random | 22.7935 | 0.7230 | 0.3071 | 74,491.97 |

Morton이 정렬 없음 대비 **+0.5641 dB, −21.0% bytes**. none과 random이 서로 가까워서,
이득이 "permutation 자체"가 아니라 **Morton이 만드는 공간적 지역성**에서 온다는 게 분리된다.

> ⚠️ 이건 **민감도 / 분포 이동 테스트**다. checkpoint가 Morton으로 학습됐기 때문에
> "학습 시점에 Morton이 더 좋다"는 인과 주장은 아니다. 그걸 보려면 세 순서를 각각 같은 예산으로 학습해야 한다.
> 15와 15b를 합치면: **Morton은 pointwise low-rank에는 무의미하고, 1-D residual 코덱이 켜지면 핵심이다.**

### 7.6 λ sweep — 프로토콜 일치 RD 곡선

아카이브 rank-56 checkpoint 4개 + codec-free 기준을 **동일한 6,991 장면**에서 평가.

| 조건 | Bytes/scene | bits/Gaussian | BPGA | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| codec-free GlobalSplat | — | — | — | 24.7004 | 0.7682 | 0.2480 |
| λ 0.0004 | 385,759 | 94.18 | 1.5963 | **24.9867** | **0.7832** | **0.2180** |
| λ 0.0016 | 354,285 | 86.50 | 1.4660 | 24.8686 | 0.7792 | 0.2223 |
| λ 0.0064 | 128,360 | 31.34 | 0.5311 | 24.5563 | 0.7682 | 0.2353 |
| λ 0.0256 | 58,444 | 14.27 | 0.2418 | 23.3700 | 0.7290 | 0.2870 |

- 네 점이 기대대로 단조 RD 곡선을 그린다.
- λ=0.0064에서 codec-free 대비 PSNR −0.1441 dB인데 LPIPS는 오히려 0.0127 좋다.
- **λ=0.0004와 0.0016은 세 지표 모두 codec-free를 넘는다** (+0.2863 / +0.1682 dB).
  렌더링 목적으로 학습된 변환이 frozen 표현을 정칙화/보정할 수 있기 때문.
- FP32 원본(16 MiB) 대비 약 43.5× / 47.4× / 130.7× / 287.1× 축소 (공유 가중치 제외).

> 참고: 30장면 예비 sweep도 있지만 **6,991장면 평균과 같은 그래프에 올리면 안 된다.**
> 30장면 subset이 더 쉬운 분포다.

---

## 8. 세대 3 — score context 계보 (48개 full-test eval)

원 문서: `docs/NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14.md` (가장 상세),
`docs/SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md` (요약)

### 8.0 출발 질문

세대 2가 "투영이 병목이다"를 보였는데, 세대 3은 다른 각도에서 들어간다.

> score가 전체 bit의 80%를 쓴다. 그런데 score의 entropy model은
> **"이 채널에서 일반적으로 흔한 값"만 학습**하고
> **"현재 장면과 이미 복호한 이웃을 고려하면 지금 뭐가 나올까"는 전혀 안 쓴다.**
> 여기에 작은 예측기를 넣으면 추가 side stream 없이 bit를 줄일 수 있지 않을까?

### 8.1 전체 계보 트리

```text
Linear factorized 50k  (λ별 checkpoint)
 ├─ +50k factorized 대조군 (rank56/80)           ......... 4개
 └─ +50k context (rank56)                         ......... 8개
     ├─ Mean
     ├─ Mean+Channel
     ├─ Mean+Spatial
     └─ Full

Nonlinear32 factorized 50k  (λ별 checkpoint)
 ├─ rank56/80 × residual ON/OFF 초기 grid          ......... 8개
 ├─ +50k factorized 대조군 (rank56/80)            ......... 4개
 └─ rank56 residual ON +50k context               ......... 4개
     ├─ Mean+Spatial  (λ .0064 / .0256)
     │   └─ .0256: +25k P0/P1/P2                   ......... (아래 9개에 포함)
     └─ Full  (λ .0064 / .0256), 누적 100k  ← ★ "원래 Full P0 부모"
         ├─ +25k P0/P1/P2, 누적 125k                ......... 9개
         ├─ +10k probability-only 4종, 누적 110k    ......... 8개
         └─ +10k Full low-LR, 누적 110k             ......... 3개 완료

원래 Full P0 + Split (110k)
 └─ 학습 없는 128/128 고정-symbol entropy probe
```

> **probability-only와 low-LR은 P 25k에서 이어간 게 아니다.**
> 둘 다 **원래 Full 100k 부모로 돌아가 갈라진 별도 가지**다. 계보를 잘못 읽는 실수가 잦다.

**결과 개수 집계**

| 구분 | 개수 | 근거 |
| --- | ---: | --- |
| 첫 Linear context | 8 | 완료 eval 로그 |
| Linear factorized 대조군 | 4 | 완료 eval 로그 |
| 초기 Nonlinear transform grid | 8 | 완료 eval 로그 |
| Nonlinear factorized 대조군 | 4 | 완료 eval 로그 |
| Nonlinear + Spatial/Full | 4 | 완료 eval 로그 |
| P0/P1/P2 | 9 | 완료 train/eval 로그 |
| Probability-only | 8 | 완료 train/eval 로그 |
| Full low-LR | 3 | `.0064/P0`은 저장공간 부족으로 실패 |
| **소계** | **48** | 원본 eval 로그에서 재추출 |
| 이전 Linear baseline | 12 | 사용자 표만 보존, 원본 로그 없음 |
| Codec OFF 참고값 | 1 | PSNR 24.7004 / SSIM 0.7682 / LPIPS 0.2480 |
| entropy probe | λ별 128 TEST scene | 별도 held-out 진단 |

### 8.2 첫 실험 — Linear score context 4종 (8개 eval)

#### 무엇을 넣었나

이미 수신기가 가진 정보를 순서대로 조건으로 쓴다.

| 이름 | flags m/c/s | 조건 정보 | rank56 score streams |
| --- | --- | --- | ---: |
| Factorized | 0/0/0 | 없음 | 1 |
| Mean | 1/0/0 | 이미 전송한 scene mean | 1 |
| Mean+Channel | 1/1/0 | mean + 먼저 복호한 channel slice들 | 4 |
| Mean+Spatial | 1/0/1 | mean + 먼저 복호한 Morton even anchor | 2 |
| Full | 1/1/1 | 셋 다 | 8 |

> 문서의 "Spatial"은 거의 항상 **Mean+Spatial**의 약칭이다. mean 없이 spatial만 켠 결과는 없다.

#### (a) Mean condition — 이미 보낸 평균으로 offset과 step 예측

```text
FP16 복원 scene mean (736-D)
  → LayerNorm → Linear(736, 64) → GELU → Linear(64, 2r)
  → 앞 r개 = score offset b, 뒤 r개 = log_d
d = exp(2·tanh(log_d/2))     # 약 [0.135, 7.389]로 제한
```

- 마지막 layer를 0-init해서 `b=0, d=1`로 시작 → 시작 시점 동작이 부모와 같다.
- `d`는 장면·채널마다 다르지만 같은 장면·채널 안의 모든 토큰에는 같다.
- 송수신기가 같은 FP16 mean과 같은 network를 쓰므로 **b/d를 전송하지 않는다.**
- 주의: `q`(공유 학습 scale), `d`(장면별 배수), residual Gaussian의 `σ`는 전부 다른 것이다.

#### (b) Channel condition — 앞 slice로 다음 slice 예측

rank 56을 **16 / 16 / 16 / 8** 네 slice로 나눈다. slice마다 별도 `EntropyBottleneck`.

```text
slice 0: Mean offset만 사용
slice 1: Conv2d(16, 16, 1) (slice 0 복원값)
slice 2: Conv2d(32, 16, 1) (slice 0,1 복원값)
slice 3: Conv2d(48,  8, 1) (slice 0,1,2 복원값)
```

- 새 predictor는 0-init, 부모 factorized 분포를 slice별로 이식.
- **1×1 conv라서 같은 토큰 위치의 앞 채널만 본다.** 주변 토큰은 안 본다.
- slice는 직렬이지만 slice 안의 4,096 토큰은 전부 병렬.

#### (c) Spatial condition — even 복호 후 odd 예측

```text
base       = mean_offset + channel_prediction(이전 slice 복원값)
even_input = (score_even - base_even) / scene_step
even_hat   = EB_roundtrip(even_input) * scene_step + base_even   # 실제 압축→복호

anchor_delta[even] = even_hat - base_even     # base로 설명 못 한 오차
anchor_delta[odd]  = 0
odd_base  = base_odd + Conv(kernel 3)(anchor_delta)[odd]
odd_input = (score_odd - odd_base) / scene_step
odd_hat   = EB_roundtrip(odd_input) * scene_step + odd_base
```

- predictor는 `Conv2d(width, width, kernel=(1,3), padding=(0,1))` 하나. hidden layer도 activation도 없다.
  Mean+Spatial은 width 56, Full은 16/16/16/8.
- odd 위치에서 읽는 입력은 [왼쪽 even 오차, 0, 오른쪽 even 오차]. 경계 밖은 zero padding.
  오른쪽 even도 첫 pass에서 이미 복호됐으므로 써도 causal하다.
- **"원래 even 값"이 아니라 "base로 설명 못 한 오차"를 입력으로 쓴다.** channel 간 정보도 conv로 섞인다.
- rank56 Full의 stream 순서:
  `G0-even → G0-odd → G1-even → G1-odd → G2-even → G2-odd → G3-even → G3-odd` = 8 streams.
- **4,096 step autoregression이 아니다.** pass당 신경망 계산이 모든 토큰에 병렬 적용된다.
  (단, rANS coder 내부까지 GPU 병렬이라는 뜻은 아니다.)
- 초기에는 같은 그룹의 even/odd가 **같은 EntropyBottleneck을 공유**했다. 분리는 나중의 Split 실험.
- 이 "Morton 이웃"은 이미지 픽셀 이웃도, 정확한 3D kNN도 아니다.

#### 학습/초기화/수신 일치

- 학습: entropy model의 noise surrogate (`V + Uniform(-0.5, 0.5)`).
  평가/실제 부호화: median 기준 rounding.
  → 학습 때도 다음 slice/odd predictor에 **surrogate 복원값**이 들어간다(실제 압축에선 복호값).
- scene mean은 송신기에서도 FP16 복원값을 쓰고, gradient는 straight-through.
- score-context 경계는 bf16 학습 중에도 **autocast를 끄고 FP32**로 계산해 송수신 수치 경로를 맞춘다.
- 새 모듈 생성 시 **RNG를 fork**해서 공통 초기화의 난수 소비가 바뀌지 않게 한다.
- 새 predictor는 전부 0-init, `d=1` 시작. 부모 EntropyBottleneck 파라미터를 slice별로 복사하고 CDF buffer는 재생성.

#### 결과 (부모: 각 λ의 Linear rank56 50k, joint +50k)

Train 419615 → Eval 419616, run=20260908_092102

| λ | Context | 전체 KiB | PSNR↑ | SSIM↑ | LPIPS↓ | **동일예산 factorized 대비** |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| .0064 | (factorized 대조군) | 90.805 | 24.3024 | 0.7602 | 0.2374 | 기준 |
| .0064 | Mean | 89.823 | 24.3179 | 0.7610 | 0.2371 | −1.08% / +0.0155 dB |
| .0064 | Mean+Channel | 88.249 | 24.3151 | 0.7608 | 0.2371 | −2.81% / +0.0127 dB |
| .0064 | Mean+Spatial | 86.073 | 24.3222 | 0.7611 | 0.2369 | −5.21% / +0.0198 dB |
| .0064 | **Full** | **84.905** | 24.3232 | 0.7611 | 0.2368 | **−6.50% / +0.0208 dB** |
| .0256 | (factorized 대조군) | 42.258 | 23.4875 | 0.7307 | 0.2729 | 기준 |
| .0256 | Mean | 41.608 | 23.4950 | 0.7309 | 0.2723 | −1.54% / +0.0075 dB |
| .0256 | Mean+Channel | 41.492 | 23.4943 | 0.7312 | 0.2727 | −1.81% / +0.0068 dB |
| .0256 | **Mean+Spatial** | **39.172** | 23.5304 | 0.7319 | 0.2716 | **−7.30% / +0.0429 dB** |
| .0256 | Full | 39.233 | **23.5352** | **0.7323** | 0.2716 | −7.16% / +0.0477 dB |

**해석**

- Mean만으로도 작은 이득. **Spatial에서 큰 이득.** Channel 단독 기여는 작다.
- `.0064`는 Full이 Spatial보다 작고 화질도 비슷 → Full 우세.
- `.0256`은 Spatial이 0.061 KiB 작고 Full이 0.0048 dB 높다. **매우 가까운 두 RD 점.**
  "Full이 모든 지표에서 최상"이라고 결론낼 수 없다.

### 8.3 Nonlinear transform + 대조군 + 결합 (16개 eval)

#### (a) 왜 factorized continuation 대조군이 필요한가

context 모델은 부모에서 50k를 **더** 학습했다. 그래서 factorized도 같은 부모에서
새 optimizer로 50k를 더 학습해야 추가 compute의 영향을 분리할 수 있다. ([§3.3](#33-matched-control-원칙))

| Transform | Rank | λ | 전체 KiB | PSNR | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Linear | 56 | .0064 | 90.805 | 24.3024 | 0.7602 | 0.2374 |
| Linear | 56 | .0256 | 42.258 | 23.4875 | 0.7307 | 0.2729 |
| Linear | 80 | .0064 | 90.944 | 24.2805 | 0.7597 | 0.2399 |
| Linear | 80 | .0256 | 42.642 | 23.4740 | 0.7310 | 0.2758 |
| Nonlinear32 | 56 | .0064 | 87.840 | 24.3022 | 0.7602 | 0.2376 |
| Nonlinear32 | 56 | .0256 | 41.256 | 23.5210 | 0.7328 | 0.2693 |
| Nonlinear32 | 80 | .0064 | 88.169 | 24.2920 | 0.7601 | 0.2395 |
| Nonlinear32 | 80 | .0256 | 42.306 | 23.4264 | 0.7319 | 0.2756 |

→ Nonlinear rank56이 Linear rank56보다 `.0064`에서 −2.965 KiB(PSNR 거의 동일),
`.0256`에서 −1.002 KiB / +0.0335 dB. **rank80은 일관된 우세가 없어 이후 rank56에 집중.**

#### (b) Nonlinear transform 구조

```text
Linear:
  U     = X @ W_a.T
  X_low = U_hat @ W_s

Nonlinear32:
  U     = X @ W_a.T + MLP_a(X)
  X_low = U_hat @ W_s + MLP_s(U_hat)

MLP_a: 736 → 32 → GELU → r
MLP_s: r   → 32 → GELU → 736
```

- 실제로는 `Linear → GELU → Linear`의 bias-free 모듈.
- **두 번째 Linear weight를 0으로 초기화**해서 시작 함수가 Linear와 동일.
- MLP 생성은 RNG fork 안에서 처리(공통 초기화의 난수 소비 보존).

#### (c) 초기 Nonlinear grid 8개 — residual ON/OFF가 핵심

rank 56/80 × λ .0064/.0256 × residual ON/OFF, 50k, context OFF, Morton ON.

| Rank | λ | Residual | 전체 KiB | PSNR | SSIM | LPIPS |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 56 | .0064 | **ON** | 98.183 | **24.5082** | 0.7665 | 0.2343 |
| 56 | .0064 | OFF | 105.790 | 24.2580 | 0.7587 | 0.2440 |
| 56 | .0256 | **ON** | 46.695 | **23.6943** | 0.7381 | 0.2673 |
| 56 | .0256 | OFF | 55.336 | 23.5011 | 0.7303 | 0.2738 |
| 80 | .0064 | ON | 99.152 | 24.4890 | 0.7659 | 0.2367 |
| 80 | .0064 | OFF | 109.913 | 24.2579 | 0.7584 | 0.2438 |
| 80 | .0256 | ON | 47.314 | 23.5833 | 0.7378 | 0.2699 |
| 80 | .0256 | OFF | 55.415 | 23.4970 | 0.7301 | 0.2728 |

**residual ON이 더 작으면서 화질도 높다.** 예외 없다.
그리고 rank56 residual ON은 이후 저용량 모델보다 크지만 **PSNR이 더 높은 operating point**다.
RD 곡선의 고화질 쪽 후보로 지워선 안 된다.

> "Residual OFF"는 residual 경로 없이 **따로 재학습한 조건**이다.
> ON 모델에서 residual 파일만 뗀 게 아니다.

#### (d) Nonlinear + Spatial/Full 결합 4개

부모: 초기 Nonlinear32 rank56 residual ON 50k. 추가 joint 50k (누적 100k).
Train 420916 → eval 420919, run=20260910_092000.

| λ | Context | 전체 KiB | PSNR↑ | SSIM↑ | LPIPS↓ | Nonlinear factorized 대비 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| .0064 | Mean+Spatial | 83.633 | 24.3244 | 0.7612 | 0.2371 | −4.79% / +0.0222 dB |
| .0064 | **Full** | **82.491** | 24.3238 | 0.7611 | 0.2371 | **−6.09% / +0.0216 dB** |
| .0256 | Mean+Spatial | 38.236 | **23.5867** | **0.7348** | **0.2678** | −7.32% / +0.0657 dB |
| .0256 | **Full** | **37.958** | 23.5707 | 0.7342 | 0.2683 | **−7.99% / +0.0497 dB** |

**"두 λ에서 같은 setting" 결정**

`.0256`은 Full이 0.278 KiB 작지만 PSNR −0.0160 dB, SSIM −0.0006, LPIPS +0.0005다.
가까운 두 RD 점이고 Full의 전면 우세가 아니다. 그럼에도 **두 λ 공통으로 Full을 쓰기로** 했는데, 근거는:

1. `.0256`에서 Full을 버릴 만큼의 열세가 아니다.
2. 하나의 구조로 rate 구간을 확장하는 편이 설계·배포·해석이 단순하다.
3. λ에 따른 predictor 반응을 비교할 때 architecture 차이를 제거할 수 있다.

기존 `.0256 Spatial` task는 지우지 않고 `.0256 Full` 3개를 **추가**했다.
"같은 setting"은 같은 구조·seed·batch·추가 step/LR을 뜻하지, **두 λ가 같은 weight에서 시작한다는 뜻이 아니다.**

**이 시점의 Full P0 100k 두 checkpoint가 이후 세 갈래의 공통 부모다.**

### 8.4 Spatial predictor P0/P1/P2 — 추가 25k (9개)

#### 가설과 구현

"odd score의 중심값을 예측하는 linear kernel-3가 너무 약하거나, Morton 이웃 범위가 너무 좁다."

| Arm | config | odd value prediction | 초기 상태 |
| --- | --- | --- | --- |
| **P0** | `linear` | 부모의 `Conv(width, width, k3)` | 그대로 |
| **P1** | `residual3` | P0 + `Conv(width,32,k3) → GELU → Conv(32,width,k1)` | 새 마지막 conv 0-init |
| **P2** | `residual7` | P0 + `Conv(width,32,k7) → GELU → Conv(32,width,k1)` | 새 마지막 conv 0-init |

- 0-init이라 이식 직후 P1/P2의 예측값은 P0와 **정확히 같다.**
- 입력은 decoded even `anchor_delta`뿐. k3는 ±1 anchor, k7은 ±1/±3까지 본다.
- **P2는 receptive field와 파라미터 수를 함께 늘린다.** 순수 거리 효과로 분리할 수 없다.
- stream 수와 복호 순서는 기존과 동일.

#### Grid

| Predictor task | λ | Context | Arm | 부모 |
| --- | --- | --- | --- | --- |
| 0/2/4 | .0064 | Full | P0/P1/P2 | nonlinear task 8 |
| 1/3/5 | .0256 | Spatial | P0/P1/P2 | nonlinear task 7 |
| 6/7/8 | .0256 | Full | P0/P1/P2 | nonlinear task 9 |

전부 부모 100k에서 **새 Adam, +25k, scope=all, seed 111123**.

> ★ **LR schedule 함정:** 부모 context 50k 말의 LR은 1e-6인데, 새 Adam으로 **1e-4에서 재시작**했다.
> milestone 35k/45k가 25k 밖이라 이 stage에서는 decay가 없다. **사실상 상수 1e-4.**

#### 결과

| λ / Context | P0 KiB / PSNR | P1 KiB / PSNR | P2 KiB / PSNR | P2−P0 |
| --- | --- | --- | --- | --- |
| .0064 Full | 79.166 / 24.1634 | 79.195 / 24.1858 | 79.055 / 24.1591 | −113.71 B, −0.0043 dB |
| .0256 Spatial | 37.446 / 23.4764 | 37.465 / 23.4515 | 37.323 / 23.4566 | −125.38 B, −0.0198 dB |
| .0256 Full | 37.348 / 23.4284 | 37.313 / 23.4286 | **37.149 / 23.4556** | **−203.24 B, +0.0272 dB** |

#### ★ 이 실험의 가장 중요한 교훈

predictor를 **바꾸지 않은** P0도 부모 대비 크게 움직였다.

| 부모 → P0 25k | Δ전체 bytes | ΔPSNR |
| --- | ---: | ---: |
| .0064 Full | **−3,405.15 B** | **−0.1604 dB** |
| .0256 Spatial | −809.26 | −0.1103 dB |
| .0256 Full | −624.84 | −0.1423 dB |

`.0064`에서 score가 약 3,864 B 줄고 residual은 약 459 B 늘었다.
→ 모델이 score 부호화만 잘한 게 아니라 **representation과 rate 배분 자체를 바꿨다.**

**LR 재시작 + 새 optimizer + 추가 joint 학습이 predictor 차이보다 훨씬 큰 혼입 요인이다.**
그래서:

- 부모 대비 화질 하락을 "P1/P2 구조의 실패"라고 부르면 안 된다.
- payload 감소를 전부 "predictor가 좋아졌다"고 부르면 안 된다.
- **같은 부모에서 같은 25k를 돈 P0와의 비교만 유효하다.**

여기서 두 개의 후속 질문이 나왔고, 다음 두 실험이 각각 답한다.

1. "낮은 LR로 기존 RD 지점을 보존하면서 joint 개선이 가능한가?" → [§8.6](#86-full-low-lr-10k--3개-완료)
2. "아예 복원을 고정하면 entropy만 개선할 수 있는가?" → [§8.5](#85-probability-only--복원을-완전히-고정하고-확률만-바꾼다-8개)

### 8.5 Probability-only — 복원을 완전히 고정하고 확률만 바꾼다 (8개)

#### 목적

P 실험에서는 중심값 예측·양자화·residual이 전부 같이 변했다.
이번엔 **"같은 복원 score symbol을 얼마나 짧게 부호화할 수 있는가"**만 분리한다.

부모: 원래 Nonlinear32 Full P0 (누적 100k, nonlinear task 8/9).
공통: rank56, residual/Morton ON, Full, P0, micro2×acc4, seed 111123, **+10k**, Adam 1e-4 → 7k에서 1e-5.
Train 422558 → eval 422559, run=20260913_112316.

#### 네 arm이 공유하는 심볼 생성/복호 순서

```text
group g (16/16/16/8 중 하나):
  even 입력:    V_e = (S_e - b_e) / d_g
  even 복원:    S_hat_e = b_e + d_g · Q(V_e; 고정 median)

  anchor_delta: even 위치 = S_hat_e - b_e, odd 위치 = 0
  odd base:     b_odd = b_(g,odd) + P_linear_k3(anchor_delta)_odd
  odd 입력:     V_o = (S_o - b_odd) / d_g
  odd 복원:     S_hat_o = b_odd + d_g · Q(V_o; 고정 median)

전체 순서: g0-even → g0-odd → g1-even → g1-odd → g2-even → g2-odd → g3-even → g3-odd
```

`P_linear_k3`, `b_g`, `d_g`, `q`, analysis/synthesis는 **네 arm에서 전부 고정.**
바뀌는 건 `V_o`의 likelihood와 arithmetic coder의 CDF뿐이다.
Gaussian 계열도 기존 even EntropyBottleneck의 고정 median을 중심으로 써서 **round lattice를 유지**한다.

#### 무엇이 고정되고 무엇이 학습되나 (`scope=score_probability`)

| 구성 | 학습 여부 |
| --- | --- |
| geometry projection, Nonlinear analysis/synthesis basis + MLP | **고정** |
| score normalization `q` | **고정** |
| Mean conditioner의 offset `b`와 step `d` | **고정** |
| Channel predictor, P0 spatial **value** predictor | **고정** |
| residual codec 전체 | **고정** |
| EntropyBottleneck quantiles / median (= 복원 lattice) | **고정** |
| active EntropyBottleneck **density parameters** | 학습 |
| Gaussian per-channel base scale | Gaussian/Conditional에서 학습 |
| anchor-conditioned scale predictor | Conditional에서만 학습 |

> ⚠️ `requires_grad=False`만으로는 부족하다. 기존 recipe가 **no-grad deterministic quantile update를 직접 호출**하기 때문에
> `loss.quantile_update_interval=0`도 함께 지정해야 median이 안 움직인다.
> CDF table은 부호화를 위해 갱신하되 quantile/lattice는 그대로 둔다.

학습 파라미터 수(로그): Shared ≈3.2K, Split ≈6.5K, Gaussian ≈3.3K, Conditional ≈10.7K.

#### 네 arm

**① Shared** — 대조군

```text
even V_e ─┐
          ├─ 같은 group EntropyBottleneck / 같은 CDF
odd  V_o ─┘
```

부모 구조/weight 그대로 로드 후 density만 10k 더 fit.
even/odd는 **별도 rANS string**이지만 **확률 파라미터는 공유**한다.
"추가 10k 학습 효과"를 측정하는 control이다.

**② Split** — 현재 채택안

```text
even V_e → EB_even[g]
odd  V_o → EB_odd[g]      ← 독립
```

초기화 시 `EB_even[g]`의 density와 고정 quantile을 `EB_odd[g]`에 **복제**한다
(CDF buffer 자체는 복사하지 않고 재생성). 그래서 **step 0에서는 Shared와 동일**하고, 학습 후에만 갈라진다.

- 새 pass도, 새 stream도 없다. Full은 원래 even/odd가 분리돼 있으니 **같은 8 stream에 다른 CDF를 쓸 뿐**이다.
- 늘어나는 건 공유 모델의 odd EntropyBottleneck 파라미터/상태뿐.
- per-scene side information **없음**.

**③ Static Gaussian**

```text
mu[g,c]    = 고정된 EB_even[g] median
sigma[g,c] = softplus(raw_scale[g,c]) + 0.11        # 56개, 전부 1.0에서 시작
p(V_o)     = Gaussian(mu, sigma)
```

`sigma`가 group/channel마다 하나라 **장면·토큰 위치·anchor와 무관**하다.
실제 coding index는 `[0.11, 256]` 구간 64단계 logarithmic scale table에서 만든다.

> ⚠️ 이 arm은 **복원 lattice만 부모와 같고 초기 확률분포는 부모와 다르다.**
> 부모의 유연한 odd marginal을 고정 mean unit Gaussian으로 **reset**했기 때문이다.

**④ Conditional Scale**

```text
anchor_delta
  → Conv2d(width → 32, kernel 1×3, padding 1)
  → GELU
  → Conv2d(32 → width, kernel 1×1)          ← 0-init
  → local_log_multiplier

sigma_base  = softplus(raw_scale) + 0.11
sigma_local = clamp(sigma_base · exp(2·tanh(local_log_multiplier/2)), 0.11, 256)
```

0-init이라 **Static Gaussian과 같은 초기 확률 모델**에서 출발한다.
multiplier는 base scale의 약 1/7.39× ~ 7.39× 범위에서 위치별로 조절된다.
입력 `anchor_delta`에는 복호 완료한 even만 있고 odd는 0이므로 causal하다
(odd의 미복호 값을 바꿔도 예측 scale이 안 변하는 테스트가 있다). scale map은 전송하지 않는다.

**요약표**

| Arm | Even 확률 | Odd 확률 | Odd 중심 | Odd scale/context | step0가 부모와 동일? | Streams |
| --- | --- | --- | --- | --- | --- | ---: |
| Shared | EB per group | 같은 EB 공유 | EB median | 위치 무관 learned density | 예 | 8 |
| Split | EB_even | 독립 EB_odd | 각 EB median | 위치 무관 independent density | 예 (even→odd 복제) | 8 |
| Static Gaussian | EB_even | GaussianConditional | 고정 EB_even median | group/channel static scale | **아니오** (reset) | 8 |
| Conditional Scale | EB_even | GaussianConditional | 고정 EB_even median | decoded even anchor → 위치별 scale | **아니오** (Gaussian과 동일) | 8 |

#### 결과

| λ | Entropy | 전체 KiB | Score B | PSNR | SSIM | LPIPS | Shared 대비 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .0064 | Shared | 82.463 | 70,924.906 | 24.3238 | 0.7611 | 0.2371 | 기준 |
| .0064 | **Split** | 81.174 | 69,604.342 | 24.3238 | 0.7611 | 0.2371 | **−1.56%** |
| .0064 | Static Gaussian | 88.167 | 76,765.617 | 24.3238 | 0.7611 | 0.2371 | +6.92% |
| .0064 | Conditional | **81.135** | 69,565.374 | 24.3238 | 0.7611 | 0.2371 | −1.61% |
| .0256 | Shared | 37.952 | 31,149.344 | 23.5707 | 0.7342 | 0.2683 | 기준 |
| .0256 | **Split** | **36.920** | **30,093.048** | 23.5707 | 0.7342 | 0.2683 | **−2.72%** |
| .0256 | Static Gaussian | 44.518 | 37,873.270 | 23.5707 | 0.7342 | 0.2683 | +17.30% |
| .0256 | Conditional | 37.021 | 30,195.839 | 23.5707 | 0.7342 | 0.2683 | −2.45% |

**설계가 의도대로 작동했다는 증거:**
같은 λ 안에서 residual y/z, mean, container가 **로그 정밀도 내 완전히 동일**하고,
세 화질 지표가 **소수 넷째 자리까지 같다**. 전체 바이트 차이는 전부 score 차이로 설명된다.

| λ | Entropy | Score B | Residual y B | Residual z B | Mean B | Container B | 수신 복원 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .0064 | Shared | 70924.906 | 11738.844 | 174.507 | 1472 | 132 | 89.9 |
| .0064 | Split | 69604.342 | 11738.844 | 174.507 | 1472 | 132 | **89.6** |
| .0064 | Gaussian | 76765.617 | 11738.844 | 174.507 | 1472 | 132 | 125.6 |
| .0064 | Conditional | 69565.374 | 11738.844 | 174.507 | 1472 | 132 | 126.2 |
| .0256 | Shared | 31149.344 | 5998.320 | 111.128 | 1472 | 132 | 85.1 |
| .0256 | Split | 30093.048 | 5998.320 | 111.128 | 1472 | 132 | **81.0** |
| .0256 | Gaussian | 37873.270 | 5998.320 | 111.128 | 1472 | 132 | 116.2 |
| .0256 | Conditional | 30195.839 | 5998.320 | 111.128 | 1472 | 132 | 120.1 |

**해석**

- **Shared 재학습 자체는 부모 대비 28.87 B / 6.24 B만 줄였다.**
  → Split의 이득(1,321 B / 1,056 B)은 "10k 더 돌려서"가 아니다. 구조 효과다.
- Conditional은 `.0064`에서 Split보다 38.97 B **작고**, `.0256`에서 102.79 B **크다**. 일관되지 않다.
  복호 시간도 41% / 48% 더 길다. → **Split이 더 단순하고 일관돼서 채택.**
- Static Gaussian은 명백히 나쁘다(+6.92% / +17.30%).
  단 이건 **고정 median + unit-scale 초기화 + 10k fit + finite scale table 조건에서의 결과**다.
  "Gaussian 분포 자체가 틀렸다"나 "조건부 확률 모델은 불필요하다"는 결론이 **아니다.**

> ⚠️ 집계 화질이 같다는 건 복원 고정 설계와 **일치**하지만,
> 서버 전체 장면의 텐서 bit-exact 검증을 한 건 아니다. 로컬 round-trip/invariance 테스트가 이를 뒷받침한다.

### 8.6 Full low-LR 10k — 3개 완료

**질문:** "기존 RD 지점을 크게 안 움직이면서 전체 코덱을 조금 더 joint 적응시킬 수 있나?"

| Task | λ | Predictor | 부모 | 추가 학습 |
| ---: | --- | --- | --- | --- |
| 0 | .0064 | P0 | 원래 Nonlinear Full 100k | all, 10k, **상수 LR 1e-6** |
| 1 | .0064 | 새 P2 | 동일 | 동일 |
| 2 | .0256 | P0 | 원래 Nonlinear Full 100k | 동일 |
| 3 | .0256 | 새 P2 | 동일 | 동일 |

- **shared entropy** 사용 (Split 아님).
- 새 P2 correction은 같은 부모에 **zero-init**한다. 이미 25k 학습된 P2를 식히는 실험이 아니다.
- 부모 weight만 로드하고 새 Adam. 엄밀한 optimizer resume가 아니다.

**실행 상태:** task 1/2/3은 10k 완료 + eval 완료.
**task 0은 checkpoint 저장 시 서버 저장공간 부족(`Errno 28`)으로 두 번 실패.**
`.0064/P0`의 유효한 step 10k 결과는 없다.

| λ | Predictor | 전체 KiB | Score B | Residual y/z B | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .0064 | P0 | 미완료 | — | — | — | — | — |
| .0064 | P2 | 82.430 | 70,867.233 | 11,761.695 / 175.439 | 24.3256 | .7611 | .2370 |
| .0256 | P0 | 37.932 | 31,115.291 | 6,011.991 / 111.517 | 23.5701 | .7342 | .2684 |
| .0256 | P2 | 37.921 | 31,106.122 | 6,009.835 / 111.455 | 23.5716 | .7343 | .2684 |

**해석**

- 원래 Full 부모 대비 개선폭: `.0064/P2` −0.074%, `.0256/P0` −0.068%, `.0256/P2` −0.097%.
  **전부 0.1% 미만.**
- `.0256`의 직접 P2−P0 차이는 −11.386 B/scene(−0.029%), +0.0015 dB. 실질적 개선이 아니다.
- **현재 best인 Full + Split과 비교하면 low-LR이 오히려 크다:**
  `.0064` +1,286.674 B(+1.548%), `.0256` +1,024.916 B(+2.711%).
- → 결론: "joint low-LR 적응이 전혀 안 된다"가 아니라 **"10k 추가 계산의 순효과가 0.1% 미만"**이다.
  `.0064/P0`를 세 번째로 재실행할 실용적 이유는 작다.

### 8.7 세대 3 누적 개선 요약

| λ | 모델/단계 | 누적 학습 | 전체 KiB | PSNR |
| --- | --- | --- | ---: | ---: |
| .0064 | Linear factorized | 50k+50k | 90.805 | 24.3024 |
| .0064 | Linear Full | 50k+50k | 84.905 | 24.3232 |
| .0064 | Nonlinear Full | 50k+50k | 82.491 | 24.3238 |
| .0064 | **Nonlinear Full + Split** | 50k+50k+10k | **81.174** | 24.3238 |
| .0256 | Linear factorized | 50k+50k | 42.258 | 23.4875 |
| .0256 | Linear Full | 50k+50k | 39.233 | 23.5352 |
| .0256 | Nonlinear Full | 50k+50k | 37.958 | 23.5707 |
| .0256 | **Nonlinear Full + Split** | 50k+50k+10k | **36.920** | 23.5707 |

동일예산 Linear factorized 대조군 → 최종 Split:

- `.0064`: 90.805 → 81.174 KiB, **−10.61%**, PSNR 24.3024 → 24.3238
- `.0256`: 42.258 → 36.920 KiB, **−12.63%**, PSNR 23.4875 → 23.5707

> ⚠️ transform 초기학습 가지가 다르고 Split은 추가 10k가 있다.
> "동일 compute에서 구조 하나만 바꾼 순수 효과"로 표현하면 안 된다.

---

## 9. 학습 없는 진단 실험 3종

세대 3 이후, **새 모델을 학습하지 않고** 현재 모델을 분석하는 실험 세 개를 돌렸다.
"다음에 어디에 투자할지"를 정하는 게 목적이다.

### 9.1 Fixed-symbol entropy probe — 확률 모델에 얼마나 여유가 남았나

원 문서: `docs/NFCGS_ENTROPY_PROBE_2026-09-14.md`
결과: `entropy-probe-0p0064-REPORT.md`, `entropy-probe-0p0256-REPORT.md`, 대응 summary.json

**질문:** "score가 전체 bit의 80%를 쓰는데, 그 확률 모델이 실제로 나쁜 건가?
아니면 그냥 score가 담은 정보량이 큰 건가?"

**방법 — 심볼과 복원을 완전히 고정한다**

- 부모: 현재 Full P0 + Split checkpoint 두 개 (누적 110k).
- backbone, transform, predictor, 양자화 step, median 전부 freeze.
- **128개 unique TRAIN 장면**으로 확률 table을 fit → **별도 128개 unique TEST 장면**에서 평가.
  scene ID를 기록하고 중복/train-test overlap을 거부한다.
- `deterministic_all` 샘플링, C12/T8, augmentation OFF.
- 학습·렌더링·이미지 저장·새 checkpoint **전부 없음.**
- 같은 native 심볼을 후보마다 **rANS로 다시 부호화**만 한다.
  모든 후보가 native integer stream과 decoded feature tensor를 **정확히 재현**함을 검증.

**두 후보 (테스트 전에 미리 선언)**

| 후보 | 내용 |
| --- | --- |
| **marginal** | score CDF를 채널/even-odd/slice별 histogram으로 재보정. residual y는 scale-table index별, z는 채널별. |
| **context** | 위 + score-even에 **scene 양자화 step bin**, score-odd에 **복호된 이웃 anchor 잔차 크기 bin**. 고정 경계 0.5/1/2/4 (5 bin). residual은 marginal 그대로. |

shrinkage 강도 4096. marginal은 부모 CDF로, context는 fit된 marginal로 수축.
관측 안 된 행은 부모로 fallback. **테스트 적응 없음.**

**결과**

| λ | Native total | Marginal / 절감 | **Context / 절감** | Context 절감 95% CI |
| --- | ---: | ---: | ---: | ---: |
| .0064 | 81.253 KiB | 81.160 / 0.115% | **80.654 / 0.736%** | [0.656%, 0.821%] |
| .0256 | 36.923 KiB | 36.859 / 0.172% | **36.642 / 0.759%** | [0.673%, 0.855%] |

- context 후보가 `.0064`에서 612.72 B/scene, `.0256`에서 287.13 B/scene 절감.
- score 기여가 0.713 / 0.738 %p, residual marginal 기여는 0.024 / 0.021 %p.
- score − residual paired interval: `[0.608, 0.773]`, `[0.644, 0.798]` %p (둘 다 양수).

**확률 모델 손실 vs coder 구현 손실**

| λ / path | Native model NLL | 정수 CDF+tail − NLL | 실제 rANS − (CDF+tail) | Actual − NLL |
| --- | ---: | ---: | ---: | ---: |
| .0064 score | 555,562.9 bit | 580.3 bit | 383.3 bit | 963.6 bit = **120.45 B** |
| .0064 residual | 95,251.8 bit | 214.0 bit | 95.0 bit | 308.9 bit = 38.62 B |
| .0256 score | 239,309.8 bit | 230.6 bit | 384.4 bit | 615.0 bit = **76.87 B** |
| .0256 residual | 48,814.6 bit | 99.7 bit | 96.9 bit | 196.6 bit = 24.58 B |

→ **arithmetic coder는 주어진 확률 모델을 약 99.6~99.8% 효율로 따르고 있다.**
rANS 자체 gap은 양쪽 λ에서 약 60 B/scene. entropy string당 약 48 bit가 보이므로,
**stream을 합쳐서 얻을 수 있는 건 수십 바이트뿐이고 주 타깃이 아니다.**

**절감이 어디에 몰려 있나**

- `.0064`: `g0_odd`, `g0_even`, `g1_odd` 세 stream이 score 절감의 **91.4%**
- `.0256`: 같은 세 stream이 82.3%, `g3_odd`를 더하면 91.4%
- 모든 odd stream 합계: 67.3% / 72.7%

→ **초반 group, 특히 odd에 capacity를 집중**해야 한다. 모든 stream에 균등하게 table을 늘릴 이유가 없다.

**이 probe의 결론**

- 실재하지만 작은 기회다: 아주 단순한 5-bin 모델로 payload의 **약 0.75%**.
- "native score 확률이 이미 decoder-available context를 다 쓰고 있다"는 주장을 **기각**하기엔 충분하다.
- 그러나 **score의 80% byte share를 설명하기엔 턱없이 작다.**
  그 share는 대부분 "현재 score 표현이 담은 정보량"이지 rANS/container 낭비가 아니다.
- table 번들 크기는 155,578 B / 123,195 B인데, 이건 native+marginal+context를 **전부 합친** 것이라
  단일 production 후보의 증분 비용이 아니다. 전체를 overhead로 쳐도 254 / 430 장면이면 상각된다.

> ⚠️ 128장면 screen이다. full TEST 평균과 직접 비교하거나 이 TEST 장면에 맞춰 튜닝하면 안 된다.
> PSNR은 새로 측정하지 않았고, 정확한 feature 일치로 "확률 교체가 복원을 안 바꿨다"만 확인했다.

### 9.2 Overhead profile — 이득이 배포 비용을 이기는가

원 문서: `docs/NFCGS_OVERHEAD_PROFILE_2026-09-15.md`
결과: `overhead-REPORT.md`, `overhead-aggregate.json`

**질문:** "각 score 모델의 rate 이득이 그 모델의 **파라미터 크기와 속도 비용**을 정당화하나?"

**통제 조건**

- 하나의 순차 Slurm job, 하나의 물리 `ariel-v12` GPU.
- `re10k_eval_all_ctx12`, batch 1, augmentation OFF, seed 0.
- warm-up 4장면 → 측정 **32 unique TEST 장면**을 **18개 checkpoint 전부 동일한 순서**로.
- 모든 timed call을 CUDA synchronization으로 감싼다.
- 두 비교를 **절대 섞지 않는다:**
  - Context suite: 같은 λ의 no-context Factorized continuation이 기준
  - Probability suite: 같은 λ의 Full+Shared가 기준
  - `.0064` 모델을 `.0256`의 기준으로 쓰지 않는다.

**측정 항목**

score-path 파라미터 수 / 버퍼 크기 / 정확한 tensor bytes / 직렬화 bytes,
context-predictor와 probability-model 파라미터 분리,
raw entropy bytes / `SCCTX001` wrapper / score payload / residual / 전체 scene bytes,
isolated score sender·receiver latency, 전체 feature-codec sender·receiver latency,
증분 peak CUDA 할당, 송수신 decoded score tensor의 정확한 일치.

> score sender 타이밍에는 **causal anchor 복원을 위한 로컬 entropy decode가 의도적으로 포함**된다.
> no-context Factorized sender도 같은 compress-then-decompress를 하므로 경계가 맞다.

**핵심 결과 — Context suite (λ=0.0064)**

| Model | Streams | Score payload B | 전체 scene B | **Break-even 장면 수** | 추가 tensor KiB | Score encode ms | Score decode ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factorized | 1 | 78551.5 | 93752.9 | — | +0.00 | 48.82 | 30.23 |
| mean | 1 | −1403.38 B | −1690.50 B | 184.6 | +253.00 | +3.91% | +3.54% |
| mean_channel | 4 | −3071.12 B | −3338.88 B | 57.5 | +172.58 | +14.39% | +13.82% |
| mean_spatial | 2 | −5075.25 B | −5506.50 B | 53.7 | +266.34 | +12.04% | +11.95% |
| **full** | 8 | **−6408.75 B** | **−6761.62 B** | **28.1** | +175.92 | +9.22% | +9.79% |

λ=0.0256에서도 순서는 같고 break-even은 197.6 / 150.6 / 77.3 / **57.4** 장면.

> **Full이 가장 빨리 본전을 뽑는다.** 약 28~57장면만 보내면 모델 추가 크기를 상각한다.
> 속도 비용은 score 경로 기준 +9~11%, 전체 codec 기준 +3~4%.

**핵심 결과 — Probability suite**

| λ | Model | 전체 scene B | Break-even | 추가 tensor KiB | Score encode ms | Full codec decode ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| .0064 | shared | 84461.4 | — | +0.00 | 53.41 | 81.92 |
| .0064 | **split** | **−1338.50 B** | **46.0** | **+60.09** | −0.12% | −0.29% |
| .0064 | gaussian | +5573.50 B | n/a | +784.23 | +87.59% | +31.25% |
| .0064 | conditional_scale | −1448.12 B | **574.9** | +812.95 | +87.61% | +32.31% |
| .0256 | **split** | **−1078.12 B** | **43.4** | +45.72 | −0.41% | −0.46% |
| .0256 | conditional_scale | −1013.62 B | **821.3** | +812.95 | +95.67% | +35.13% |

**이 표가 Split 채택을 결정적으로 뒷받침한다:**

- Split은 추가 모델 크기가 **45~60 KiB뿐**이고 **43~46장면**이면 상각된다. 속도는 **사실상 동일**(−0.4% ~ +0.1%).
- Conditional은 rate 이득이 비슷한데 추가 모델이 **813 KiB**라 break-even이 **575~821장면**,
  속도는 **+88~96%**. 같은 이득에 비용만 크다.

> `Wrapper B`만이 진짜 per-scene 포맷 overhead다. FP16 scene mean은 no-context 코덱도 이미 보내므로 context에 부과하지 않는다.
> 모델 tensor bytes는 공유 배포 overhead라 장면 수로 상각해야 한다.
> 학습 `.ckpt` 파일 크기는 optimizer state를 포함할 수 있어 배포 크기로 치지 않는다.
> 절대 시간은 기기 의존적이다. 같은 GPU·같은 장면 순서의 **paired delta**만 본다.

### 9.3 Scene CI — 장면 단위 신뢰구간과 BD-rate screen

원 문서: `docs/NFCGS_SCENE_CI_2026-09-15.md`
결과: `20260915_113013/REPORT.md`, `summary.json`, `paired_scene_metrics.csv`

**질문:** "수십 바이트 / 0.02 dB 차이가 통계적으로 의미 있나?"

**방법**

- 이미 완료된 `all_test_ctx12` eval의 **장면별 `actual_rate_per_scene.json`**만 읽는다.
  새 학습·인코딩·렌더링 없음. **GPU 불필요, CPU 분석.**
- 비교 대상 4개 family × 2 λ = 8 cell:
  1. `linear_factorized` — Linear, context 없음, matched 50k continuation
  2. `nonlinear_factorized` — Nonlinear32, context 없음, matched 50k continuation
  3. `linear_full` — Linear Mean+Channel+Spatial Full, joint 50k
  4. `nonlinear_split` — Nonlinear32 Full P0 + 복원고정 Split 10k
- **재표집 단위는 장면 하나다.** 장면 안의 target 8 view는 이미 평균돼 있고 8개 독립 표본으로 치지 않는다.
- 8 cell이 **완전히 동일한 장면 집합**을 가져야 한다. context frame / target frame ID 목록까지 정확히 일치 검증.
- 기존 `scores_all_avg.json`의 전역 평균을 전부 재계산해 검증.
- 20,000회 paired nonparametric percentile bootstrap, 95% CI, seed 20260915.
  각 draw가 장면을 한 번 재표집하고 **같은 index를 8 cell 전부에 적용**한다.

**결과 — 평균과 CI (6,991 장면)**

| Model | λ | Mean KiB [CI] | PSNR dB [CI] |
| --- | ---: | ---: | ---: |
| Linear Factorized | 0.0064 | 90.805 [90.669, 90.937] | 24.3024 [24.2348, 24.3713] |
| Nonlinear Factorized | 0.0064 | 87.840 [87.703, 87.973] | 24.3022 [24.2344, 24.3712] |
| Linear context Full | 0.0064 | 84.905 [84.823, 84.988] | 24.3232 [24.2557, 24.3920] |
| **Nonlinear Full+Split** | 0.0064 | **81.174 [81.093, 81.254]** | 24.3238 [24.2563, 24.3928] |
| Linear Factorized | 0.0256 | 42.258 [42.200, 42.315] | 23.4875 [23.4235, 23.5527] |
| Nonlinear Factorized | 0.0256 | 41.256 [41.198, 41.314] | 23.5210 [23.4567, 23.5862] |
| Linear context Full | 0.0256 | 39.233 [39.192, 39.273] | 23.5352 [23.4708, 23.6003] |
| **Nonlinear Full+Split** | 0.0256 | **36.920 [36.881, 36.959]** | 23.5707 [23.5062, 23.6362] |

**paired 차이 (핵심 행만)**

| λ | 비교 | Rate 절감 % [CI] | ΔPSNR dB [CI] | Rate/PSNR/joint 장면 승률 |
| --- | --- | ---: | ---: | --- |
| .0064 | Nonlinear Full+Split vs Linear Factorized | **+10.607%** [+10.525, +10.687] | +0.0214 [+0.0196, +0.0234] | 100.0% / 59.0% / 59.0% |
| .0064 | Nonlinear Full+Split vs Linear context Full | +4.395% [+4.384, +4.406] | +0.0006 [−0.0010, +0.0023] | 100.0% / 49.3% / 49.3% |
| .0256 | Nonlinear Full+Split vs Linear Factorized | **+12.631%** [+12.557, +12.703] | +0.0832 [+0.0806, +0.0858] | 100.0% / 78.7% / 78.7% |
| .0256 | Nonlinear Full+Split vs Linear context Full | +5.894% [+5.878, +5.909] | +0.0356 [+0.0332, +0.0380] | 100.0% / 65.1% / 65.1% |

**근사 2점 BD-rate** (음수가 후보에 유리)

| 비교 | 2점 BD-rate % [CI] |
| --- | ---: |
| Nonlinear Factorized vs Linear Factorized | −4.322% [−4.458, −4.190] |
| Linear context Full vs Linear Factorized | −9.812% [−9.919, −9.706] |
| **Nonlinear Full+Split vs Linear Factorized** | **−15.957%** [−16.094, −15.820] |
| Nonlinear Full+Split vs Nonlinear Factorized | −12.224% [−12.338, −12.108] |
| Nonlinear Full+Split vs Linear context Full | −6.816% [−6.955, −6.678] |

**읽는 법**

- rate 승률이 100%라는 건 **모든 장면에서 더 작았다**는 뜻이다. rate 차이는 확실하다.
- PSNR 승률은 49~79%로 흩어진다. 화질 개선은 rate 개선보다 훨씬 약하다.
- CI가 0을 안 걸치는 항목만 "차이가 있다"고 말할 수 있다.

> ⚠️ 이 구간은 **테스트 장면 표집 불확실성만** 담는다.
> 학습 seed, checkpoint 선택, dataset shift, hardware 변동은 **포함하지 않는다.**
> 6개 pairwise 구간은 다중비교 보정이 안 돼 있다.
> 2점 BD-rate는 screening 통계다. 출판용 곡선은 최소 4개의 잘 퍼진 RD 점이 필요하다.

---

## 10. 현재 기준 모델과 실행 방법

### 10.1 현재 기준점

```text
Nonlinear32 / rank56 / Morton ON / residual ON / Full context P0 / even-odd Split
```

| λ | 전체 bytes / KiB | Score B | Residual y/z B | Mean / container B | PSNR | SSIM | LPIPS | 수신 복원 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| .0064 | 83,121.693 / 81.174 | 69,604.342 | 11,738.844 / 174.507 | 1,472 / 132 | 24.3238 | .7611 | .2371 | 89.6 ms |
| .0256 | 37,806.496 / 36.920 | 30,093.048 | 5,998.320 / 111.128 | 1,472 / 132 | 23.5707 | .7342 | .2683 | 81.0 ms |

Checkpoint (서버):

```text
.0064
outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/
  checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/
  version_0/step000010000.ckpt

.0256
동일 경로에서 lambda0p0064 → lambda0p0256
```

**채택 근거 5가지**

1. decoder-causal score context는 matched factorized 대조군 대비 두 λ 모두 실제 bitstream을 줄였다.
2. Split은 같은 부모·같은 10k에서 Shared보다 1.56% / 2.72% 작다.
3. 복원 경로를 고정했으므로 Split의 차이는 **score 확률 모델 차이로 격리**된다.
4. P1/P2 확대와 전체 모델 low-LR 추가학습은 두 λ에서 일관된 큰 이득이 없었다.
5. Conditional Gaussian은 Split보다 일관되게 작지 않고, 모델이 13배 크고, 복호가 88~96% 느리다.

### 10.2 config

`config/model/globalsplat_nfcgs_rank56.yaml`:

```yaml
feature_codec:
  geometry_observable_channels: 224
  rank: 56
  residual_N: 192
  residual_M: 320
  adapter_hidden: 96
  morton_bits: 10
  use_morton: true
  use_residual: true
  transform: nonlinear
  transform_hidden: 32
  score_mean_condition: true
  score_channel_context: true
  score_spatial_context: true
  score_spatial_predictor: linear      # P0
  score_spatial_entropy: split         # even/odd 분리
  score_slice_channels: 16             # 16/16/16/8
  score_context_hidden: 64
```

`compression/config.py`의 `CodecConfig.__post_init__`이 위 값들을 **강제**한다.
은퇴한 변형(linear transform, factorized, partial context, no-Morton, no-residual, P1/P2, Shared, Gaussian)은
config·checkpoint·bitstream 경계에서 **명시적으로 실패**한다. 조용히 다른 모델이 선택되지 않는다.

### 10.3 명령어

**평가**

```bash
python scripts/run_nfcgs.py eval --rate-lambda 0.0064 --dry-run
python scripts/run_nfcgs.py eval --rate-lambda 0.0256 \
  --checkpoint /path/to/main_split.ckpt --dataset-root /data/re10k

mkdir -p logs/slurm
sbatch scripts/slurm/eval_nfcgs_main.slurm --rate-lambda 0.0064
```

프로토콜: all-test, C12/T8, batch1, seed0, 실제 bitstream 복호, 이미지 저장 없음.
`--max-scenes`는 명시적 screen cap일 뿐이다.

**codec-only 학습** (backbone frozen — 세대 3의 기본)

```bash
python scripts/run_nfcgs.py train --checkpoint /path/to/main_split.ckpt \
  --rate-lambda 0.0256 --dry-run
sbatch scripts/slurm/train_nfcgs_main.slurm \
  --checkpoint /path/to/main_split.ckpt --rate-lambda 0.0256
```

기본 recipe: paper24 샘플링, frozen GlobalSplat, micro2/accum4, Adam 1e-4, 50k steps,
35k/45k milestone, BF16, quantile 500 step.
`--scope score_probability`를 주면 복원 고정 scope(10k, 7k milestone, quantile OFF)가 선택된다.

**새 아키텍처 초기화** (QR/PCA)

```bash
python scripts/initialize_nfcgs_from_vanilla.py \
  --vanilla checkpoints/pretrained/globalsplat-re10k-32k.ckpt \
  --codec-init checkpoints/codec_init/re10k_ctx12_s64_t512_rank56.pt \
  --output outputs/main_initialization.ckpt
```

**GlobalSplat + codec 동시 from-scratch** (실험적)

```bash
python scripts/run_nfcgs.py train --from-scratch \
  --rate-lambda 0.0064 --dataset-root /data/re10k --dry-run
sbatch scripts/slurm/train_nfcgs_joint_v10.slurm --from-scratch \
  --rate-lambda 0.0064 --dataset-root /data3/local_datasets/re10k
```

`re10k_32k_nfcgs_joint` recipe:

- 같은 Nonlinear32/rank56/Morton/residual/Full P0+Split 코덱을 **step 0부터** 사용
- paper24 샘플링, BF16, micro1/accum8, GPU 1개 (8-GPU wrapper면 global batch 8)
- AdamW 5e-4, weight decay 1e-6, 3% warmup → cosine decay, **500k steps**
- Gaussian capacity stage 20k/40k/100k, 4k-step 전환, 최종 4096 token × 8 = 32,768 Gaussian
- rate-loss weight를 첫 20k step에 걸쳐 ramp, quantile 500 step

> ⚠️ 이건 **검증된 수렴/품질 결과가 아니라 시작 설정**이다.
> `--max-steps`는 LR schedule 길이만 바꾸고 capacity 경계나 rate ramp는 rescale하지 않는다.
> 짧은 smoke run은 초기 curriculum 단계에 머문다.
> **Split을 처음부터 joint 학습하는 것은 새 실험이지, 기존 다단계 계보의 재현이 아니다.**

**긴 run 전에 checkpoint 쓰기를 먼저 확인** (low-LR 실험에서 저장공간으로 두 번 날린 교훈)

```bash
sbatch scripts/slurm/train_nfcgs_joint_v12.slurm --from-scratch \
  --rate-lambda 0.0064 --max-steps 20 --checkpoint-every 10 \
  --output outputs/nfcgs_joint_save_probe
```

step 10과 20에 저장하고 멈춘다. 결과 checkpoint를 실제로 로드해보고 본 run을 시작한다.
**이 진단 run을 500k로 resume하면 안 된다** (warmup/cosine schedule이 20 step용이다).

**실행 중인 220k job 연장**

```bash
bash scripts/slurm/submit_nfcgs_joint_extension_v10.sh 425420
```

`afterok:425420` 의존으로 대기 → 부모 stdout에서 `last.ckpt` 위치를 찾아
model/optimizer moment/global step/loader state를 보존하고 220k → 500k로 이어간다.
별도 late-stage LR schedule(10k에 걸쳐 1e-5 → 1e-4, 이후 270k에 걸쳐 cosine → 1e-5)을 쓴다.

---

## 11. 확정된 것 / 아직 모르는 것

### 11.1 확정에 가까운 관측

| # | 관측 | 근거 |
| ---: | --- | --- |
| 1 | observable geometry 512→224 변환은 사실상 무손실 | 세대 2 실험 17 (−0.0628 dB) |
| 2 | 주된 화질 손실은 **low-rank 투영**이지 entropy coding이 아니다 | 세대 2 실험 17 (−4.1~−6.3 dB vs −0.24~−0.42 dB) |
| 3 | residual 경로가 3.1~5.4 dB를 되찾는다. 빼면 안 된다 | 세대 2 실험 17, 세대 3 Nonlinear grid |
| 4 | Morton은 pointwise 코덱엔 무의미하고, 1-D residual 코덱엔 핵심 | 세대 2 실험 15 + 15b (+0.5641 dB, −21.0% bytes) |
| 5 | Morton이 lag-1 상관을 0.0055 → 0.4739로 만든다 | 세대 2 실험 16 |
| 6 | rank 80/96/128 증가는 현재 schedule에서 일관된 이득이 없다 | 세대 2 실험 16/17, 세대 3 rank80 대조군 |
| 7 | 작은 Nonlinear transform이 Linear보다 낫다 | 세대 2 실험 14 (+0.8882 dB), 세대 3 대조군 |
| 8 | Mean보다 **Spatial** decoder context의 이득이 훨씬 크다 | 세대 3 §8.2 (−1.08% vs −5.21%) |
| 9 | context 이득은 matched factorized 대조군 대비로도 남는다 | 세대 3 §8.2, §8.3 |
| 10 | P1/P2 value predictor 확대의 순수 효과는 P0 추가학습 효과보다 작고 λ 간 일관성이 약하다 | 세대 3 §8.4 |
| 11 | 복원 고정 **Split** probability가 두 λ 모두 Shared보다 작다 | 세대 3 §8.5 (−1.56% / −2.72%) |
| 12 | 전체 모델 low-LR 10k는 0.1% 미만 개선 | 세대 3 §8.6 |
| 13 | arithmetic coder overhead는 수십 B 수준. 병목 아님 | entropy probe (§9.1) |
| 14 | 단순 score context table로 held-out 약 0.75% 절감 여지가 실재한다 | entropy probe (§9.1) |
| 15 | Split은 43~46장면이면 모델 크기를 상각하고 속도 비용이 없다 | overhead profile (§9.2) |
| 16 | Full+Split의 rate 이득은 6,991장면 전부에서 재현된다 (승률 100%) | scene CI (§9.3) |

### 11.2 아직 실험으로 답하지 못한 것

- **두 점이 아닌 여러 λ의 actual RD 곡선과 제대로 된 BD-rate.** (현재는 2점 screen뿐)
- single-seed의 수십 B / 0.02 dB 차이가 **seed를 바꿔도 재현되는지.**
- **Split entropy를 처음부터 transform과 joint 학습**하면 복원고정 fit보다 좋아지는지.
- Split보다 강한 **non-Gaussian conditional density**의 실효성.
  (이번 실험은 Gaussian scale만 봤고, 고정 median·unit init·10k·finite table 조건이 붙어 있다)
- **score와 residual에 대칭적인 강도의 context model**을 적용했을 때의 상대 개선량.
  (현재 probe는 score에만 richer context를 줘서 비대칭이다)
- **복호된 low-rank score를 residual entropy model의 조건으로** 넣었을 때의 이득.
  (세대 1에 `..._basecond_entropy_...` 설정이 있으니 먼저 그걸 볼 것)
- score와 residual **각각이 화질에 기여하는 양**을 bit와 함께 측정한 partial-reconstruction 결과.
- **score hyperprior / coarse-to-fine latent** 같은 구조적 교체가 side bits와 복호 지연을 상쇄하는지.
- 공유 model/table 배포 비용까지 포함한 **단일 장면 또는 짧은 시퀀스의 총 rate.**
- `.0064` low-LR P0/P2의 직접 matched 비교 (checkpoint 저장 실패로 미확정).
- 순수 architecture 비교의 **matched-rate 성능**과 통제된 복호 시간.

### 11.3 다음 실험을 설계할 때의 판단 원칙

원 문서의 인수인계 항목을 그대로 옮긴다.

1. 완료된 full eval 48개, historical Linear baseline 12개, entropy probe 2개를 **재실행하지 않는다.**
2. 구조의 효과는 **같은 부모·같은 추가 step의 matched control**로 판단한다.
3. 전체 크기 감소와 PSNR 하락을 따로 자랑하지 말고 **실제 RD trade-off**로 본다.
4. "score가 total rate의 80~84%"는 **최적화 leverage가 크다**는 뜻이지,
   그 확률 모델이 같은 비율만큼 나쁘다는 뜻이 아니다. 실증된 gap은 약 0.75%다.
5. 새 실험의 기준 checkpoint는 [§10.1](#101-현재-기준점)의 Full P0+Split 두 개다.
6. 증거는 **"spatial value predictor를 무작정 키우는 방향"보다
   "이미 좋은 복원을 유지하면서 남은 확률 mismatch를 줄이는 방향"**에 힘을 실어준다.

### 11.4 지금 증거가 가리키는 유망한 다음 수

| 우선순위 | 아이디어 | 근거 |
| --- | --- | --- |
| 낮은 위험 | **g0/g1, 특히 odd stream에 한정한 nonparametric conditional-CDF 모듈.** 더 큰 TRAIN으로 fit, validation에서 선택 후 **단 한 번** full TEST | probe에서 초반 3 stream이 절감의 82~91% |
| 구조적 | **score hyperprior / coarse-to-fine latent.** 작은 score side latent를 전송해 비모수 score 분포를 예측. 처음엔 양자화 격자를 고정해서 "side bit를 벌어오는가"만 깨끗이 검증 | score share가 크고 tabular gain은 1% 미만 |
| 구조적 | **residual entropy에 decoded low-rank feature를 conditioning** | 세대 1에 미완 설정이 남아 있음 |
| 재검토 | 세대 1의 미탐색 축: scene-specific basis, mixture transform, block delta, grouped-16, split point 이동, decoder adapter | 세대 2·3이 안 건드린 영역 |
| 평가 보강 | **λ를 4개 이상으로 늘려 제대로 된 BD-rate**, seed 반복 | 현재 2점 screen과 single seed |

> "좋은 코덱" 관점에서는 문서 순서를 따라가는 것보다,
> **실제 total bytes–화질 곡선의 개선이 복잡도/학습비용 대비 충분한가**가 기준이다.

---

## 12. 빠른 참조 — 어디서 뭘 찾나

| 찾는 것 | 위치 |
| --- | --- |
| Context/Nonlinear/Split의 코드 수준 상세 | [CODEC_DEEP_DIVE.md](CODEC_DEEP_DIVE.md) |
| 현재 지원 아키텍처와 실행 명령어 | `upstream/globalsplat/docs/NFCGS_CODEC.md` |
| 완료 실험 요약 + 현재 기준 checkpoint 경로 | `docs/SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md` |
| 세대 3의 모든 설정·동기·수식·결과 | `docs/NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14.md` |
| 48개 eval의 정밀 수치, 바이트 분해, 타이밍, checkpoint 절대경로 | `..._metrics.json` (같은 폴더) |
| 세대 2 ablation 14/15/15b/16/17 + λ sweep | `docs/ABLATION_PROGRESS_2026-09-02.md` |
| 세대 1 설정 76개의 정의 | `experiments/*/experiment.yaml` |
| 세대 1 코덱 재구현 + API | `archive/reconstructed_codec/README.md` |
| entropy probe 방법과 결과 | `docs/NFCGS_ENTROPY_PROBE_2026-09-14.md`, `entropy-probe-*-REPORT.md` |
| overhead / break-even | `docs/NFCGS_OVERHEAD_PROFILE_2026-09-15.md`, `overhead-REPORT.md` |
| 장면 단위 CI, BD-rate | `docs/NFCGS_SCENE_CI_2026-09-15.md`, `20260915_113013/REPORT.md` |
| 상관/스펙트럼 분석 산출물 | `ablation_16_correlation/` |
| 정리 전 구현 스냅샷 (74파일 + SHA-256) | `upstream/globalsplat/archive/codec_experiments_20260915/` |
| 정리 범위와 호환성 검증 기록 | `docs/CODEC_CLEANUP_2026-09-15.md` |
| Slurm 원본 로그 | `logs/`, `slurm/` |

### 12.1 코드 파일 역할

| 파일 | 역할 |
| --- | --- |
| `compression/codec.py` | Nonlinear analysis/synthesis, train scope, score/residual 통합, compress/decompress |
| `compression/score_context.py` | Mean/Channel/Spatial context, P1/P2 correction, Split/Gaussian/Conditional, causal encode/decode |
| `compression/residual.py` | multiscale 1-D hyperprior |
| `compression/entropy.py` | factorized entropy wrapper |
| `compression/morton.py` | Morton 정렬 |
| `compression/bitstream.py` | `E2EM0301` / `SCCTX001` 컨테이너 |
| `compression/config.py` | 아키텍처 옵션과 허용값 검증 |
| `compression/checkpoint.py` | metadata·shape 추론, strict load, predictor/entropy mismatch 검사 |
| `compression/entropy_probe.py` | 고정-symbol probe |
| `model/globalsplat.py` | 코덱 삽입 지점, `compress_scene` / `decompress_scene` |
| `model/model_wrapper.py` | subset loss, RD 항, quantile 주기, 평가/타이밍 집계 |
| `scripts/run_nfcgs.py` | 단일 train/eval 진입점 |
| `scripts/initialize_nfcgs_from_vanilla.py` | QR/PCA 초기화, 부모 warm-start, entropy 이식 |
| `scripts/probe_nfcgs_entropy.py` | entropy probe 실행 |
| `scripts/analyze_nfcgs_scene_ci.py` | 장면 CI bootstrap |
| `tests/test_score_context_codec.py` | causal context, round-trip, probability scope / 복원 고정 |
| `tests/test_nfcgs_codec.py`, `test_main_codec.py` | codec 통합, checkpoint/bitstream 검증, 아카이브 oracle 대조 |

### 12.2 구현 상의 주의점 (건드릴 때 깨지기 쉬운 것들)

- **entropy/context 연산은 bf16 학습 중에도 내부에서 autocast를 끄고 FP32로 처리한다.**
  forward / compress / decompress의 수치 경로를 맞추기 위해서다.
- **새 모듈 생성은 RNG를 fork한다.** 안 그러면 모듈을 추가하는 것만으로 공통 초기화의 난수 소비가 바뀐다.
- **bitstream flag는 설정을 식별하지 모델 weight를 식별하지 않는다.** 송수신기가 같은 checkpoint를 써야 한다.
  flag/rank/slice width/stream 수가 다르면 복호를 거부한다.
- **비활성 `score_entropy` 텐서는 기존 checkpoint의 strict load만을 위해 남아 있다.**
  main forward/entropy 경로가 쓰지 않는다.
- **checkpoint 이식 후 entropy/CDF buffer는 재생성해야 한다.** 기존 metadata는 기본 linear predictor / shared entropy로 해석된다.

---

## 부록 A. 세대 3 이전 Linear baseline 12개 (배경)

원본 eval 로그가 로컬에 없어 표만 보존한 값이다. 크기 열은 원문 `kB` 표기 그대로다.
**위의 `KiB` 표들과 정밀 비교하지 말 것.**

| Rank | λ | Residual | Morton | PSNR | SSIM | LPIPS | 크기(kB 표기) |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 56 | .0064 | ON | ON | 24.5021 | 0.7663 | 0.2342 | 102.68 |
| 56 | .0064 | OFF | ON | 24.2426 | 0.7582 | 0.2460 | 111.30 |
| 56 | .0064 | ON | OFF | 24.4920 | 0.7662 | 0.2356 | 105.45 |
| 56 | .0256 | ON | ON | 23.6659 | 0.7368 | 0.2701 | 48.53 |
| 56 | .0256 | OFF | ON | 23.4941 | 0.7311 | 0.2750 | 58.58 |
| 56 | .0256 | ON | OFF | 23.7240 | 0.7393 | 0.2668 | 53.08 |
| 80 | .0064 | ON | ON | 24.4776 | 0.7653 | 0.2371 | 103.84 |
| 80 | .0064 | OFF | ON | 24.2327 | 0.7577 | 0.2457 | 115.63 |
| 80 | .0064 | ON | OFF | 24.4845 | 0.7661 | 0.2370 | 110.21 |
| 80 | .0256 | ON | ON | 23.5391 | 0.7346 | 0.2764 | 48.12 |
| 80 | .0256 | OFF | ON | 23.4823 | 0.7313 | 0.2746 | 58.53 |
| 80 | .0256 | ON | OFF | 23.6440 | 0.7386 | 0.2690 | 54.54 |

Codec-OFF 참고값: PSNR 24.7004 / SSIM 0.7682 / LPIPS 0.2480.
(모든 perceptual 지표의 수학적 상한이 아니고, payload가 정의된 operating point도 아니다.)

여기서 "residual ON과 Morton이 유용하고 rank80은 일관된 우세가 없다"가 나왔고,
그 위에서 **"rank를 늘리기보다 rank56 score에 남은 의존성을 이용하자"**는 세대 3의 첫 질문이 나왔다.

## 부록 B. 부모 checkpoint 경로 규칙 (서버)

```text
초기 Linear 부모:
outputs/nfcgs_paper24_subset_train/rank56/lambda{L}/residual_on/
  checkpoints/nfcgs_paper24_subset_rank56_lambda{L}_residual_on_morton_on/
  version_0/step000050000.ckpt

초기 Nonlinear 부모:
outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda{L}/residual_on/
  checkpoints/nfcgs_transform_nonlinear32_rank56_lambda{L}_residual_on_morton_on/
  version_0/step000050000.ckpt

P / Probability / Low-LR의 "원래 Full 부모":
outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda{L}/residual_on/
  checkpoints/nfcgs_nonlinear_score_context_joint_rank56_lambda{L}_residual_on_morton_on_m1c1s1/
  version_0/step000050000.ckpt

{L} = 0p0064 또는 0p0256
Spatial 부모는 마지막 예시의 m1c1s1 → m1c0s1
```

경로 중간의 `m{mean}c{channel}s{spatial}`가 context flag다. `m1c1s1` = Full, `m1c0s1` = Mean+Spatial.

> **run tag와 global_step은 서로 다른 의미다.**
> 다른 실험 root의 `step000050000.ckpt`를 이름만 보고 같은 누적 학습량으로 취급하지 마라.

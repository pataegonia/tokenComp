# NFC-GS context 실험 전체 기록 — 설정·구현·동기·결과

작성 기준: 2026-09-15, 로컬에 내려받은 서버 로그와 현재 미커밋 구현을 대조했다.

이 문서는 첫 Linear score-context 실험부터 probability-only, Full low-LR 및 고정-symbol entropy probe까지를 정리한 통합 기록이다. 과거 문서의 “예정/진행 중” 표기는 해당 시점의 기록으로 보존하고, 여기서는 내려받은 완료 로그를 기준으로 상태를 구분한다.

## 1. 먼저 보는 결론

지금까지 가장 일관되게 확인한 것은 **score의 decoder-causal context가 유효하고, 그 위에서 복원값을 고정한 even/odd 확률 모델 분리가 추가로 유효하다**는 점이다.

- Linear factorized → Linear Full: 동일한 50k+50k 단계 구성에서 전체 크기 약 6.5–7.2% 감소, PSNR 소폭 상승.
- Nonlinear factorized → Nonlinear Full: 동일 nonlinear 부모와 추가 50k 대조에서 전체 크기 약 6.1–8.0% 감소, PSNR 소폭 상승.
- P0/P1/P2 추가 25k: 일부 조건에서 작은 개선은 있지만 두 λ에 걸쳐 일관된 큰 이득은 없다. 부모 대비 RD 이동이 predictor 간 차이보다 크다.
- 원래 Nonlinear Full의 복원을 고정한 probability-only 추가 10k: Split은 Shared 추가학습 대조군보다 전체 크기 .0064에서 1.56%, .0256에서 2.72% 감소했다. 세 화질 지표는 로그의 소수 넷째 자리까지 같다.
- Conditional Gaussian scale은 static Gaussian보다 크게 낫지만, Split보다 일관되게 좋지는 않고 수신 복원 시간도 더 길다. 현재 두 λ 공통의 실용적인 다음 기준점으로는 **Nonlinear32 + Full(P0) + Split**이 타당하다. 이는 결과 해석/추천이지, 현재 코드 기본값을 Split으로 바꿨다는 뜻은 아니다.
- Full low-LR 10k는 원래 Full 부모 대비 0.07–0.10%만 줄었고, `.0256`의 새 P2도 같은 P0보다 0.029% 작을 뿐이었다. 계산량 대비 추가 효과가 작다.
- 고정-symbol 128 TRAIN/128 TEST probe에서 단순 score context table은 전체 payload를 .0064/.0256에서 0.736%/0.759% 줄였다. coder 구현 손실보다 아직 쓰지 않은 decoder-known context가 남았다는 증거지만, score의 약 80% rate share 전체를 설명하는 큰 gap은 아니다.

단, 이 모델이 지금까지 모든 용량·모든 화질 지표에서 최선이라는 뜻은 아니다. P 계열의 더 작은 용량 지점과 초기 transform의 더 높은 화질 지점도 남는다. 최종 codec 평가는 여러 λ의 실제 RD 곡선, 복호 시간, 모델 배포 비용까지 함께 봐야 한다.

### 1.1 자료 범위와 신뢰 수준

| 구분 | 결과 수 | 근거 |
| --- | ---: | --- |
| 첫 Linear context | 8 | 로컬 완료 eval 로그 |
| Linear factorized 추가학습 대조군 | 4 | 로컬 완료 eval 로그 |
| 초기 Nonlinear transform, 배경 실험 | 8 | 로컬 완료 eval 로그 |
| Nonlinear factorized 추가학습 대조군 | 4 | 로컬 완료 eval 로그 |
| Nonlinear + Spatial/Full | 4 | 로컬 완료 eval 로그 |
| P0/P1/P2 | 9 | 로컬 완료 train/eval 로그 |
| Probability-only | 8 | 로컬 완료 train/eval 로그 |
| Full low-LR 10k | 3 | 로컬 완료 eval 로그; `.0064/P0`은 저장 공간 부족으로 최종 checkpoint 없음 |
| 합계 | **48** | 기존 45개 + Full low-LR 3개 |
| 이전 Linear baseline | 12 | 이전 문서에 보존된 사용자 표; 이 묶음에는 원본 eval 로그 없음 |
| Codec OFF | 별도 1 | 이전 문서의 참고값 |
| 고정-symbol entropy probe | λ별 128 TEST scene | full-test/render eval이 아닌 별도 held-out 진단 |

따라서 문서화된 codec 결과 설정은 이전 표 12개까지 합쳐 60개이며, 이 중 48개를 원본 eval 로그에서 다시 추출했다. P 9개와 probability 8개의 train 로그는 모두 max_steps 도달을 확인했고, low-LR은 3개만 완료했다. 이전 학습의 job ID·설정은 당시 문서, 현재 recipe, eval의 checkpoint 경로를 조합한 근거이며, 48개 모두의 원본 train 로그가 있다는 뜻은 아니다.

정확한 소수값, 실제 bytes 분해, 타이밍, checkpoint 및 서버 결과 경로는 [원시 지표 JSON](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json)에 보존했다. 표의 KiB는 JSON의 actual_bytes를 1024로 나눈 값이다.

### 1.2 이전 설명에서 바로잡을 점

1. **6,986은 평가 장면 수가 아니라 warm-up 5회를 뺀 시간 측정 횟수다.** 완료 로그의 평가 진행 수는 6,991이고, 7,286은 로더가 표시한 전체 길이 추정치다. metric/rate는 warm-up 장면도 포함한다. 로컬에는 장면별 JSON이 없으므로 정확한 scene ID 집합의 일치까지 확인한 것은 아니다.
2. **“원래 Full 50k checkpoint”는 context 단계의 step 번호다.** 초기 transform 50k + context 50k로 누적 100k 학습된 모델이다.
3. **Probability-only의 집계 화질 일치 ≠ 서버 전체 장면의 텐서 bit-exact 검증.** 구현과 로컬 round-trip/invariance 테스트는 복원 고정을 뒷받침하고, 서버 집계 수치도 부합한다. 그러나 모든 장면의 복원 텐서를 직접 비교한 서버 증거는 이번 로그에 없다.
4. **Static Gaussian의 실패 ≠ 실제 분포가 Gaussian이 아니라는 증명.** 고정 median, unit-scale 초기화, 10k 최적화, scale table 등 현재 구현/학습 조건을 포함한 결과다.
5. **P0/P1/P2를 전부 무효로 볼 필요는 없다.** 같은 부모에서 같은 추가 25k를 돌린 P0와의 비교는 유효하다. 다만 부모와의 차이를 predictor 하나의 효과로 읽으면 안 된다.

## 2. 공통 codec 구조와 측정 대상

구현 중심 파일: [codec.py](../globalsplat/compression/codec.py), [score_context.py](../globalsplat/compression/score_context.py), [config.py](../globalsplat/compression/config.py), [model_wrapper.py](../globalsplat/model/model_wrapper.py).

### 2.1 무엇을 압축하는가

GlobalSplat backbone은 공식 globalsplat-re10k-32k.ckpt의 가중치를 사용하고 codec 학습 중 고정한다. scene당 4,096개 token에서 appearance 512차원과 geometry feature 512차원의 학습 가능한 224차원 projection을 합쳐 **736차원 observable feature**를 압축한다. 따라서 “codec만 학습”에도 geometry projection은 포함된다.

현재 final stage=3 설정에서 token당 Gaussian 8개, scene당 Gaussian 32,768개가 생성된다. 모델의 M_max=16과 실제 final-stage 생성 개수 8을 혼동하면 안 된다. bpga는 Gaussian당 59개 attribute로 정규화한다.

압축·복원 흐름은 다음과 같다.

```text
GlobalSplat token features
  → geometry projection + appearance concat (4096 × 736)
  → Morton sort
  → FP16 scene mean 저장/차감
  → low-rank analysis → score quantization/context entropy coding
  → 복원 score → low-rank synthesis
  → 나머지 feature residual → 1-D hyperprior codec
  → [mean + score stream + residual y/z + container] 전송

수신기: 저장된 mean + score 복호 + residual 복호
  → 736-D feature 복원 → 고정된 Gaussian 생성 head → target-view 렌더링
```

- Morton은 각 scene의 position 범위를 정규화한 뒤 축당 10-bit code로 stable sort한다.
- 수신기는 이미 정렬된 feature 순서를 소비한다. Gaussian 집합 렌더링에 원래 token 순서의 복원이 필요하지 않아 permutation이나 raw XYZ를 별도 전송하지 않는다.
- 이것은 “수신기가 XYZ를 알고 있다”는 뜻이 아니다. 현재 score 복호기에 raw XYZ를 새 conditioning으로 넣으려면 별도 side information 또는 다른 복호 설계가 필요하다.
- scene mean은 736 × FP16 = **1,472 B**다. 송신기에서도 FP16으로 복원될 동일 mean을 centering에 사용하며, 학습에서는 해당 양자화의 straight-through 경로를 사용한다.

### 2.2 Linear / Nonlinear transform

PCA 초기화 artifact는 rank별 re10k_ctx12_s64_t512_rank56.pt / rank80.pt이며, geometry basis, analysis/synthesis basis, score scale, residual mean/std를 초기화한다.

Linear는 학습 가능한 untied analysis/synthesis basis를 사용한다. 아래에서 X는 mean을 뺀 736-D feature이고, U는 rank r score다.

```text
Linear:
  U = X W_a^T
  X_low = U_hat W_s

Nonlinear32:
  U = X W_a^T + MLP_a(X)
  X_low = U_hat W_s + MLP_s(U_hat)

MLP_a: 736 → 32 → GELU → r
MLP_s: r   → 32 → GELU → 736
```

실제 MLP는 Linear → GELU → Linear 순서의 bias-free 모듈이다. 두 번째 Linear weight를 0으로 초기화해 시작 함수는 Linear와 같게 한다. MLP 생성은 RNG fork 안에서 처리해 추가 모듈 생성 때문에 공통 초기화의 난수 소비가 바뀌는 것을 막는다.

Score는 학습 가능한 양수 scale q=exp(score_log_scale)로 정규화한다. analysis/synthesis basis, q, geometry projection은 일반 joint 학습에서 업데이트한다. residual mean/std는 buffer다.

### 2.3 Residual 경로

Residual은 원래 centered feature에서 **양자화·복원한 low-rank feature**를 뺀 값이다. 이를 저장된 mean/std로 정규화해 1-D multiscale hyperprior로 압축한다.

- Adapter hidden 96, dilation 1/2/4의 kernel-5 branch, branch당 32차원.
- Main analysis: kernel-5 stride-2 conv 736→192, GDN, kernel-5 stride-2 conv 192→320.
- Main synthesis는 이에 대응하는 transpose-conv/normalization 경로.
- Hyper-analysis는 |y|를 입력으로 받아 kernel-3 및 두 stride-2 kernel-5 conv를 거쳐 z를 생성한다.
- z는 192채널 EntropyBottleneck, y는 z에서 예측한 mean/scale을 사용하는 GaussianConditional로 부호화한다.
- 현재 residual y 확률 예측은 hyperlatent z를 사용한다. decoded low-rank feature를 추가 conditioning으로 넣는 실험은 아직 완료 결과가 없다.

“Residual OFF”는 위 residual 경로를 사용하지 않는 별도 재학습 조건이다. ON 모델의 residual 파일만 제거한 실험이 아니다.

### 2.4 Estimated rate와 actual rate

학습 rate는 score/y/z likelihood의 −log2 합을 Gaussian·attribute 수로 나눈 값이다. 현재 RD loss의 estimated rate에는 고정 mean/header bytes가 포함되지 않는다. 평가 actual rate는 실제 compress → bitstream → decompress 후 **mean, score, y, z, container를 모두 포함**한다.

- scene container: E2EM0301, header/checksum 포함 80 B.
- residual이 있으면 residual wrapper 52 B가 더해져 actual_container_bytes=132 B.
- context score container: SCCTX001, 24 B prefix + stream당 8 B length.
- rank56 Full은 4 channel slice × even/odd = 8 streams라 score wrapper 88 B. 이는 actual_score_bytes 안에 이미 포함되어 있으므로 중복해서 더하면 안 된다.
- shared neural weights/checkpoint 크기는 scene payload에 포함하지 않는다. 배포 시 모델을 새로 보내야 하는 비용은 별도 비교해야 한다.
- 아래 크기는 scene별 actual bytes의 평균이고 단위는 **KiB=1024 B**다. 배경 사용자 표의 원래 “kB” 단위는 재검증할 원본 bytes가 없어 별도로 표시한다.

## 3. 공통 학습·평가 설정

### 3.1 학습 recipe

실행 진입점은 [train_nfcgs_paper_recipe.slurm](../scripts/slurm/train_nfcgs_paper_recipe.slurm)이며, 각 실험 wrapper가 부모·scope·구조·LR을 override한다.

| 항목 | context/joint 및 후속 실험 기본 |
| --- | --- |
| 데이터 / 영상 크기 | RE10K, 256×256 |
| 원본 backbone | globalsplat-re10k-32k.ckpt, frozen |
| 학습 대상 | 일반 실험은 feature_codec scope=all; probability-only는 score_probability |
| Context sampling | 24-view pool, shared endpoint 2개 + 나머지 22개 교대 배분 → A/B 각 13 views |
| Targets | 두 subset에 같은 target 12 views |
| View span / augmentation | 40–220, augmentation ON |
| Subset consistency | ON, 시작 step 0, parity 교대 |
| Scene batch | micro 2 × accumulation 4 = optimizer update당 8 scene draws |
| Subset 처리 | micro-batch 2개 scene을 A/B 두 branch로 합쳐 4개 representation 처리; 16개의 독립 scene이라는 뜻은 아님 |
| Optimizer | Adam, weight decay 0, 새 stage는 부모 weight만 로드하고 새 optimizer/scheduler |
| 표준 50k LR | 1e-4 → step 35k에서 1e-5 → 45k에서 1e-6 |
| Precision / clipping | bf16-mixed / gradient norm clip 0.5 |
| Seed | training 111123, data_loader 403 |
| Quantile | deterministic update 500-step 간격; probability-only는 0으로 비활성화 |
| Checkpoint | 5k 간격, 마지막 stage step으로 평가 |
| 실행 방식 | 실험당 GPU 1개, 독립 Slurm array; DDP 아님 |
| Loader | train workers 8, eval workers 4, prefetch 2 |

초기 Nonlinear transform 50k는 **micro 1 × accumulate 8**이었다. 이후 context/continuation의 micro 2 × accumulate 4와 effective scene batch는 같지만 micro-batch 연산/난수 경로까지 완전히 같지는 않다. 초기 Linear baseline의 세부 실행 값은 원본 train 로그 없이 문서 recipe 이상으로 확정하지 않는다.

현재/최근 recipe는 ariel-v12, gpu:1을 요청하며, 내려받은 환경 기록에 RTX A5000, PyTorch 2.5.1+cu121, CUDA 12.1, CompressAI 1.2.8이 나타난다. 과거 모든 run이 동일 물리 GPU와 부하에서 실행됐다고 가정하지 않는다.

Joint 학습 loss를 요약하면 다음과 같다. 일반 render loss 외 model regularizer가 있으면 기존 wrapper 경로에서 함께 더해진다.

```text
L = 0.5(D_A + D_B)
    + 0.001 L_alpha_consistency
    + 0.01  L_depth_consistency
    + lambda * estimated_bits / (B_rep * N_gaussian * 59)

D의 주된 항:
    1.0 * RGB MSE + 0.05 * LPIPS(VGG) + 0.01 * in-view/frustum regularization
```

Rate ramp는 0이다. alpha/depth consistency는 0.5|A−stopgrad(B)| + 0.5|B−stopgrad(A)|의 대칭 형태이고, depth는 양쪽 accumulated opacity가 0.01보다 큰 영역을 사용한다. 별도 supervised depth/perceptual 가중치와 render criterion 안의 LPIPS 가중치를 혼동하지 않는다.

### 3.2 단계별 바뀐 설정

| 단계 | 시작점 | 추가 optimizer steps | 누적 steps | LR | 학습 scope |
| --- | --- | ---: | ---: | --- | --- |
| 초기 transform | PCA init + frozen backbone | 50k | 50k | 1e-4, 35k/45k decay | all |
| Linear context / factorized continuation | 해당 λ/rank Linear 50k | 50k | 100k | 1e-4, 35k/45k decay | all |
| Nonlinear factorized / context | 해당 λ/rank Nonlinear 50k | 50k | 100k | 1e-4, 35k/45k decay | all |
| P0/P1/P2 | 해당 λ의 Nonlinear Spatial 또는 Full 100k | 25k | 125k | 사실상 상수 1e-4 | all |
| Probability-only | 해당 λ의 원래 Nonlinear Full 100k | 10k | 110k | 1e-4, 7k에서 1e-5 | score_probability |
| Low-LR 후속 | 해당 λ의 원래 Nonlinear Full 100k | 10k | 110k | 상수 1e-6 | all; 3개 완료, `.0064/P0` 최종 저장 실패 |

위 누적 steps는 lineage상의 codec optimizer step 합이다. backbone 사전학습까지 포함하지 않는다. 50k+50k는 optimizer state를 이어간 단일 100k run이 아니다.

### 3.3 평가 protocol

- all-test, context 12 views, 별도 target 8 views, batch 1, seed 0, bf16-mixed.
- 각 checkpoint로 실제 scene bitstream을 만들고 수신 복원한 Gaussian을 렌더링한다. estimated bits만으로 압축률을 판단하지 않는다.
- 완료 eval 로그 48개에서 마지막 진행 수는 6,991, 표시된 로더 길이는 7,286이다. 진행률이 100%가 아니어도 최종 metric/result 출력까지 있으므로 이 표기만으로 중단으로 보지 않는다.
- encoder 및 entropy_decode 타이밍은 첫 5 batch를 제외해 각 6,986회다. renderer decoder는 target 8 views에 해당하는 55,888회다.
- entropy_decode는 수신 측 codec feature 복원 경로의 측정값이지, 순수 arithmetic entropy coder만의 마이크로벤치마크가 아니다.
- timing warm-up 제외는 PSNR/SSIM/LPIPS/rate 평균의 장면 제외가 아니다.
- 장면별 actual_rate_per_scene.json은 로컬에 없어 paired scene 통계·동일 scene ID 목록·신뢰구간은 아직 검증하지 않았다.

## 4. 전체 실험 계보

```text
Linear factorized 50k, lambda별 checkpoint
 ├─ +50k factorized control (rank56/80)
 └─ +50k context (rank56)
     ├─ Mean
     ├─ Mean+Channel
     ├─ Mean+Spatial
     └─ Full

Nonlinear32 factorized 50k, lambda별 checkpoint
 ├─ rank56/80 × residual ON/OFF : 초기 transform 8개
 ├─ residual ON, +50k factorized control (rank56/80)
 └─ rank56 residual ON, +50k context
     ├─ Mean+Spatial (lambda .0064 / .0256)
     │   └─ .0256: +25k P0/P1/P2
     └─ Full (lambda .0064 / .0256), 누적100k
         ├─ +25k P0/P1/P2, 누적125k
         ├─ +10k Shared/Split/Gaussian/Conditional probability-only, 누적110k
         └─ +10k low-LR P0/새 P2, 누적110k; 3개 완료, .0064 P0 저장 실패
```

Probability-only와 low-LR은 25k P2 결과에서 이어간 것이 아니다. **원래 Full 100k 부모로 되돌아가 갈라진 별도 가지**다. λ가 다르면 각 λ로 학습된 부모 weight는 다르지만, 후반 두 실험에서는 architecture와 추가학습 recipe를 같게 맞췄다.

## 5. 첫 실험: Linear score context 8개

### 5.0 Context 이전에는 어떤 방식으로 압축했는가

여기서 “기존 모델”은 codec-OFF GlobalSplat이 아니라, 첫 context 실험의 부모인 **Linear low-rank + factorized score entropy + residual hyperprior** 모델이다. 이미 학습 기반 손실 압축을 하고 있었으며, raw feature를 그대로 저장하거나 단순히 PCA coefficient를 FP16으로 저장하는 방식이 아니었다.

기존에도 Morton 정렬, FP16 scene mean, 학습 가능한 analysis/synthesis basis와 score scale, residual codec이 있었다. Context 추가는 이 전체 codec을 교체한 것이 아니라 **score를 양자화·부호화하는 경계에 조건부 예측을 넣은 것**이다. 초기 도입 전 `cc707c2`와 도입 commit `66dccb7`을 대조했으며, 아래 초기 설명에는 후속 Nonlinear32·P1/P2·Split 변경을 섞지 않는다.

#### 5.0.1 Score: 저차원 변환 → 채널별 양자화 → factorized entropy coding

Scene당 4,096개 token의 appearance512와 projected geometry224를 합친 feature F의 크기는 4096×736이다. Score는 attention score가 아니라 이 feature의 rank56 low-rank coefficient다. 아래 식은 batch 축을 생략했고, 채널별 나눗셈/곱셈은 token 축으로 broadcast한다.

```text
mean = FP16(mean_over_tokens(F))       # 송신기도 FP16 복원값 사용
X = F - mean
U = X @ W_analysis.T                  # [4096, 736] → [4096, 56]
q = exp(score_log_scale)              # 모든 scene이 공유하는 채널별 학습 scale
S = U / q

K = round(S - median)                 # 실제 부호화하는 정수 symbol
S_hat = K + median
U_hat = q * S_hat
X_low = U_hat @ W_synthesis           # [4096, 56] → [4096, 736]
```

Analysis/synthesis basis는 PCA로 초기화하지만 이후 RD loss로 학습하며 서로 독립된 파라미터다. q도 학습한다. 따라서 “고정 PCA + 고정 양자화”가 아니다. Context 이전에도 채널별 양자화 간격 q와 entropy median이 있었고, **scene별 추가 step과 decoded-context prediction이 없었던 것**이다.

`FactorizedScoreEntropy(56)` 안에는 `EntropyBottleneck(56)`이 있다. 각 채널 c의 정수 확률 P_c를 학습하고, 같은 채널의 모든 token·scene에 동일한 확률 모델을 사용한다. 모델의 factorization을 쓰면 다음과 같다.

```text
P_model(K) = product_over_tokens_and_channels P_c(K[i, c])
```

이는 실제 score들이 독립이라는 증명이 아니라 **부호화 확률 모델이 사용하는 가정**이다. 채널마다 분포가 다를 수 있으며, 모든 채널을 한 개의 동일 분포로 압축한다는 뜻은 아니다. 이미 복호한 다른 채널, 가까운 token, 현재 scene mean에 따라 score의 CDF를 바꾸는 경로는 없었다. Mean은 기존에도 feature centering과 최종 복원에는 사용했다.

학습된 누적분포에서 정수 CDF table을 만든 뒤 CompressAI의 기본 ANS/rANS coder로 K를 실제 bytes로 바꾼다. Score는 scene당 하나의 entropy string이다. **Entropy model은 확률을 정하고, entropy coder는 그 확률표로 정수를 가역적으로 bytes에 담는 별개 단계**다. 압축 손실은 저차원 표현과 양자화/근사 복원에서 생기며, rANS는 이미 정해진 정수 symbol을 손실 없이 복호한다.

구현: [기존 score 분기와 low-rank 복원](../globalsplat/compression/codec.py), [factorized entropy wrapper](../globalsplat/compression/entropy.py).

#### 5.0.2 Residual: 처음부터 hyperprior 조건부 압축이었다

Score만으로 복원하지 못한 feature는 다음 residual로 계산한다.

```text
residual = (F - mean) - X_low
residual_norm = (residual - residual_mean) / residual_std
y = main_analysis(residual_norm)       # multiscale adapter + strided conv
z = hyper_analysis(abs(y))
```

부호화 순서는 다음과 같다.

1. z를 채널별 `EntropyBottleneck`으로 양자화·압축하고 z_hat으로 복호한다.
2. z_hat을 hyper-synthesis에 넣어 y의 위치별 mean(mu)·scale(sigma)를 예측한다.
3. `GaussianConditional`이 `round(y - mu)`를 sigma에 대응하는 CDF로 압축한다. 수신기는 같은 z_hat에서 mu·sigma를 재계산하고 y_hat을 복원한다.
4. y_hat을 main synthesis/adapter에 넣어 residual을 복원하고 정규화를 되돌린다.
5. 최종 feature는 `mean + X_low + residual_hat`이다.

따라서 **“기존에는 context가 전혀 없었다”는 설명은 부정확하다.** Score에는 decoded-scene/channel/spatial context가 없었지만, residual y는 이미 전송한 hyperlatent z에 조건부인 확률 모델이었다. 이때 sigma는 Gaussian 확률 모델의 scale이고, y를 sigma로 나누어 양자화 간격을 바꾸는 연산은 아니다.

기존에도 residual y와 z의 두 entropy string 및 z side-information 비용을 모두 전송했다. 전체 scene packet은 mean + score + residual y/z + headers였으며, 공유 neural weights는 scene별 packet 밖에 있었다. 구현: [residual.py](../globalsplat/compression/residual.py).

#### 5.0.3 무엇이 개선 후보였는가

기존 score 모델은 “이 채널에서 일반적으로 흔한 값”은 학습하지만, **“현재 scene과 이미 복원한 이웃을 고려하면 지금 어떤 값이 나올 것인가”**는 사용하지 않았다. Low-rank analysis가 상관관계를 줄일 수는 있어도 모든 의존성을 제거한다고 보장할 수 없으므로, 남은 의존성을 작은 predictor로 활용해 볼 여지가 있었다.

이는 검증할 가설이었다. Score의 큰 bit 비중만으로 낭비나 최대 병목을 확정한 것은 아니며, 추가학습 대조군과 전체 실제 bytes·화질·복호 시간으로 판단해야 한다.

### 5.1 왜 했는가

기존 factorized score entropy는 채널별 분포를 학습하지만, 같은 채널의 token·scene마다 동일한 marginal을 쓴다. 이미 복호한 정보로 score를 더 잘 예측하면 추가 side stream 없이 entropy를 낮출 수 있다고 보았다.

사용 가능한 정보 세 가지를 순서대로 넣었다.

| 이름 | Flags m/c/s | 조건 정보 | rank56 score streams |
| --- | --- | --- | ---: |
| Factorized | 0/0/0 | 없음 | 1 |
| Mean | 1/0/0 | 이미 전송한 scene mean | 1 |
| Mean+Channel | 1/1/0 | mean + 이전에 복호한 channel slices | 4 |
| Mean+Spatial | 1/0/1 | mean + 먼저 복호한 Morton even anchors | 2 |
| Full | 1/1/1 | mean + channel + spatial | 8 |

여기서 문서의 “Spatial”은 대체로 Mean+Spatial의 약칭이다. Mean 없이 spatial만 켠 별도 결과가 아니다. 또한 “shared/factorized”는 모든 채널이 하나의 동일 확률분포라는 뜻이 아니라, **채널별 학습 분포를 위치/scene 또는 even/odd가 공유**한다는 뜻이다.

### 5.2 구현

#### 5.2.1 기존 양자화 식에서 달라진 부분

위 5.0절의 정규화 score S를 그대로 entropy model에 넣는 대신, decoder가 재현할 수 있는 예측값 b를 빼고 scene별 step d로 나눈다.

```text
기존:
  K = round(S - median)
  S_hat = K + median

Context 추가:
  V = (S - b) / d
  K = round(V - median)
  S_hat = b + d * (K + median)

공통 후처리:
  U_hat = q * S_hat
  X_low = U_hat @ W_synthesis
```

b는 Mean offset에 Channel/Spatial prediction을 더한 값이고, d는 Mean conditioner가 예측한다. Median은 entropy model의 채널별 값으로 b와 다른 항이다. Original coefficient U에서의 양자화 간격은 기존 q에서 **q×d**로 바뀐다. b는 격자의 위치를, d는 간격을 바꾸므로, 이는 같은 정수를 더 짧게 쓰는 확률 모델 교체만이 아니다.

#### 5.2.2 Mean: 이미 보낸 scene mean으로 b와 d 계산

**Mean condition.** FP16 복원 scene mean 736-D → LayerNorm → Linear(736,64) → GELU → Linear(64,2r). 출력은 score offset b와 quantization step d이며 d=exp(2 tanh(log_d/2))로 약 [0.135,7.389] 범위에 제한한다. 마지막 layer를 0-init해 b=0, d=1로 시작한다.

Rank56이면 출력112개를 offset56/step56으로 나눈다. d는 scene·채널마다 다르지만 동일 scene·채널 내 모든 token에는 같다. q는 공유 학습 파라미터, d는 scene mean에서 예측하는 배수이며, 둘 다 residual Gaussian의 sigma와 구별한다. 송수신기가 같은 FP16 mean과 공유 network를 사용하므로 b/d를 별도 stream으로 보내지 않는다. 기존 mean 1,472 B는 계속 전송한다.

#### 5.2.3 Channel: 복원한 앞쪽 slice들로 다음 slice 예측

**Channel condition.** rank56을 16/16/16/8 네 slice로 나눈다. 각 slice는 별도 EntropyBottleneck을 갖고, 다음 slice base에 이미 복호한 slice concat의 1×1 conv prediction을 더한다. 입력 채널은 순서대로 16,32,48이다. 새 predictor는 0-init이며 부모 factorized 분포를 slice별로 이식한다.

구체적인 predictor는 `Conv2d(16,16,1)`, `Conv2d(32,16,1)`, `Conv2d(48,8,1)` 세 개다. 첫 slice는 Mean offset만 사용한다. 이후 slice는 같은 token 위치의 **모든 이전 복원 slice**를 읽으며, 1×1이므로 주변 token을 직접 보지는 않는다. 그룹은 직렬이지만 그룹 내 모든 token의 predictor 계산은 병렬이다.

#### 5.2.4 Spatial: even 복호 후 odd 예측

**Spatial condition.** Morton sequence의 even token을 먼저 부호화/복호한 뒤 odd를 처리한다.

```text
# EB_roundtrip(V)는 실제 entropy compress→decompress를 뜻한다.
# 입력 V는 실수이고, 내부 정수는 round(V - median)이다.
base = mean_offset + channel_prediction(previous_decoded_slices)
even_input = (score_even - base_even) / scene_step
even_hat = EB_roundtrip(even_input) * scene_step + base_even

anchor_delta[even] = even_hat - base_even
anchor_delta[odd]  = 0
odd_base = base_odd + Conv(kernel3)(anchor_delta)[odd]
odd_input = (score_odd - odd_base) / scene_step
odd_hat = EB_roundtrip(odd_input) * scene_step + odd_base
```

초기 predictor는 그룹마다 `Conv2d(width,width,kernel_size=(1,3),padding=(0,1))` 하나이고, hidden layer나 activation이 없다. Mean+Spatial은 width56, Full은 width16/16/16/8이다. Odd 위치에서 읽는 입력은 왼쪽 even 오차·중앙0·오른쪽 even 오차이며, 경계 밖은 zero padding이다. 원래 even 값 대신 base로 설명하지 못한 오차를 입력으로 쓰고, 채널 간 정보도 convolution으로 혼합한다. 이 Morton 이웃은 이미지 pixel 이웃이나 정확한 3D kNN과 같지 않다.

Odd predictor는 원본 odd score나 아직 복호하지 않은 정보를 읽지 않는다. 오른쪽 even도 첫 pass에 이미 복호했으므로 사용할 수 있다. raw XYZ를 수신기에 새로 요구하지 않는다. Rank56 Full의 순서는 G0-even, G0-odd, G1-even, G1-odd, G2-even, G2-odd, G3-even, G3-odd로 8개 stream이다. 4,096-step token 신경망 autoregression은 아니며, 각 pass의 신경망 계산을 token들에 병렬 적용한다. rANS coder 내부까지 모두 GPU 병렬이라는 뜻은 아니다.

**초기에는 같은 그룹의 even/odd가 같은 EntropyBottleneck을 공유했다.** Context는 V의 값/격자를 조절하고 V의 채널별 분포는 factorized entropy model이 담당한다. Token마다 새로운 Gaussian mean/scale이나 mixture를 출력하는 확률 head는 당시 추가하지 않았다. 별도 odd EntropyBottleneck은 이후 Split 실험의 변경이다.

#### 5.2.5 학습·초기화·수신 일치

학습은 entropy model의 noise surrogate, 평가/실제 부호화는 median을 기준으로 한 rounding을 사용한다. Mean의 d와 score offset/value prediction을 학습하므로 이 초기 context 실험은 확률 모델만 바꾸는 실험이 아니다. quantization과 복원 feature, residual도 같이 바뀔 수 있다.

학습 시 `V + Uniform(-0.5,0.5)`를 복원 surrogate로 사용하므로 다음 slice/odd의 predictor에도 그 surrogate 복원값이 들어간다. 실제 압축에서는 송신기도 entropy string을 복호한 값을 사용한다. Scene mean은 송신기에서도 FP16 복원값을 쓰고 학습 gradient에는 straight-through 경로를 둔다. Score-context 경계는 FP32 계산으로 유지한다.

추가한 Mean 마지막 layer 및 Channel/Spatial predictor를 0-init하고 d=1로 시작한다. 부모 factorized EntropyBottleneck의 채널별 파라미터를 각 slice에 복사하고 coding CDF buffer는 재생성한다. 새 모듈 생성의 RNG는 fork해 공통 모델의 초기화에 영향을 주지 않는다. 이 초기화는 새로운 prediction/step 효과를 0에서 시작시키기 위한 것이며, stream 분할/header 때문에 파일 bytes까지 부모와 동일하다는 뜻은 아니다.

실제 완료한 joint 실험은 backbone/기존 Gaussian 생성 head를 고정하고 **codec 전체**를 학습했다. Geometry projection, analysis/synthesis basis, q, context predictor, score/residual entropy와 residual transform 모두 업데이트 대상이다. Quantile/median은 일반 optimizer gradient 대신 500-step deterministic update를 사용했다. 따라서 성능 차이에는 context와 함께 학습된 표현·residual 변화가 포함된다. Codec context-only 5k screen 옵션과 완료 joint 50k 결과를 혼동하지 않는다.

관련 구현: [context 모듈](../globalsplat/compression/score_context.py), [codec 연결](../globalsplat/compression/codec.py), [부모 entropy 이식](../scripts/initialize_nfcgs_from_vanilla.py), [학습 rate/quantile 갱신](../globalsplat/model/model_wrapper.py).

#### 5.2.6 기존과 초기 Full의 차이 요약

| 항목 | Context 이전 Linear factorized | 초기 Linear Full |
| --- | --- | --- |
| 입력/transform | Morton 정렬 observable feature, linear analysis/synthesis | 같은 구조 유지; joint 학습으로 weights는 달라짐 |
| Scene mean | FP16 전송, centering/복원에 사용 | 기존 역할 + score offset/step 예측 |
| 원래 coefficient의 양자화 간격 | 공유 채널별 q | 공유 q × scene별 d |
| Score value prediction | 별도 context predictor 없음 | Mean offset + 이전 slice + even anchor 예측 |
| Score density | 채널별 EntropyBottleneck, scene/위치 공유 | 예측·정규화 후 채널별 EntropyBottleneck; 초기 even/odd 공유 |
| Score entropy strings | 1개 | 4 slices × 2 passes = 8개 |
| Score 전용 wrapper | 없음 | SCCTX001 88 B, score bytes에 포함 |
| Residual 확률 모델 | z factorized + y Gaussian conditional on z_hat | 같은 구조 유지; 입력 residual과 weights는 바뀔 수 있음 |
| 정수 entropy coder | CompressAI 기본 ANS/rANS | 동일; coder 알고리즘 교체가 아님 |

요약하면 **기존에도 score와 residual을 서로 다른 학습 기반 방식으로 압축했고, 초기 context는 score 경로에 작은 예측기와 scene-adaptive quantization을 추가했다.** “기존에는 압축을 안 했다”, “기존 전체가 무조건부였다”, “초기에 큰 조건부 확률 network로 교체했다”는 설명은 모두 실제 구현과 다르다.

### 5.3 실제 실행과 결과

부모: 각 λ의 Linear rank56, residual ON, Morton ON, 초기 50k checkpoint. Context 모듈을 초기화해 **joint 50k**, micro2×acc4, 전체 codec 학습. 구현에는 5k screen mode도 있지만 아래 완료 8개는 joint 결과다. 별도 screen 결과로 중복 집계하지 않는다.

Train 419615 → Eval 419616, run=20260908_092102. 짝수 task는 .0064, 홀수 task는 .0256이다.

| Eval job_task | λ | Context | 전체 KiB | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 동일-budget factorized 대비 크기 / PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [419616_0](../../../slurm/slurm-gs-score-ctx-eval-419616_0.out) | .0064 | Mean | 89.823 | 24.3179 | 0.7610 | 0.2371 | -1.08% / +0.0155 dB |
| [419616_2](../../../slurm/slurm-gs-score-ctx-eval-419616_2.out) | .0064 | Mean+Channel | 88.249 | 24.3151 | 0.7608 | 0.2371 | -2.81% / +0.0127 dB |
| [419616_4](../../../slurm/slurm-gs-score-ctx-eval-419616_4.out) | .0064 | Mean+Spatial | 86.073 | 24.3222 | 0.7611 | 0.2369 | -5.21% / +0.0198 dB |
| [419616_6](../../../slurm/slurm-gs-score-ctx-eval-419616_6.out) | .0064 | Full | 84.905 | 24.3232 | 0.7611 | 0.2368 | -6.50% / +0.0208 dB |
| [419616_1](../../../slurm/slurm-gs-score-ctx-eval-419616_1.out) | .0256 | Mean | 41.608 | 23.4950 | 0.7309 | 0.2723 | -1.54% / +0.0075 dB |
| [419616_3](../../../slurm/slurm-gs-score-ctx-eval-419616_3.out) | .0256 | Mean+Channel | 41.492 | 23.4943 | 0.7312 | 0.2727 | -1.81% / +0.0068 dB |
| [419616_5](../../../slurm/slurm-gs-score-ctx-eval-419616_5.out) | .0256 | Mean+Spatial | 39.172 | 23.5304 | 0.7319 | 0.2716 | -7.30% / +0.0429 dB |
| [419616_7](../../../slurm/slurm-gs-score-ctx-eval-419616_7.out) | .0256 | Full | 39.233 | 23.5352 | 0.7323 | 0.2716 | -7.16% / +0.0477 dB |

표의 기준은 아래 6절의 factorized +50k 대조군이다. 초기 50k 부모와만 비교하면 context 효과와 추가 학습 효과가 섞이므로 그 비교를 핵심 근거로 쓰지 않는다.

**해석.** Mean만으로 작은 이득, spatial을 넣을 때 더 큰 이득이 나타났다. .0064는 Full이 Spatial보다 작고 화질도 비슷하다. .0256은 Spatial 39.172 KiB, Full 39.233 KiB로 Full이 약간 더 크지만 PSNR은 0.0048 dB 높다. 따라서 첫 결과에서 “Channel은 언제나 이득” 또는 “Full이 모든 지표에서 최상”이라고 결론내릴 수는 없다.

## 6. 추가학습 대조군과 Nonlinear transform 병렬 실험

### 6.1 Factorized continuation을 넣은 이유

Context 모델은 기존 50k 모델에서 50k를 더 학습했다. 그러므로 기존 factorized도 같은 부모에서 새 optimizer로 50k를 더 학습해야 추가 compute의 영향을 분리할 수 있다.

Linear rank56/80와 Nonlinear rank56/80를 각 λ에서 평가했다. 모두 residual/Morton ON, context OFF다. Nonlinear 대조군은 Linear에서 nonlinear로 즉석 변환한 것이 아니라 **초기 Nonlinear 50k 부모**에서 시작한다.

| Eval job_task | Transform | Rank | λ | 전체 KiB | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [420718_0](../../../slurm/slurm-gs-factorized-cont-eval-420718_0.out) | Linear | 56 | .0064 | 90.805 | 24.3024 | 0.7602 | 0.2374 |
| [420718_1](../../../slurm/slurm-gs-factorized-cont-eval-420718_1.out) | Linear | 56 | .0256 | 42.258 | 23.4875 | 0.7307 | 0.2729 |
| [420718_2](../../../slurm/slurm-gs-factorized-cont-eval-420718_2.out) | Linear | 80 | .0064 | 90.944 | 24.2805 | 0.7597 | 0.2399 |
| [420718_3](../../../slurm/slurm-gs-factorized-cont-eval-420718_3.out) | Linear | 80 | .0256 | 42.642 | 23.4740 | 0.7310 | 0.2758 |
| [420304_0](../../../slurm/slurm-gs-nl-score-ctx-eval-420304_0.out) | Nonlinear32 | 56 | .0064 | 87.840 | 24.3022 | 0.7602 | 0.2376 |
| [420304_1](../../../slurm/slurm-gs-nl-score-ctx-eval-420304_1.out) | Nonlinear32 | 56 | .0256 | 41.256 | 23.5210 | 0.7328 | 0.2693 |
| [420383_10](../../../slurm/slurm-gs-nl-score-ctx-eval-420383_10.out) | Nonlinear32 | 80 | .0064 | 88.169 | 24.2920 | 0.7601 | 0.2395 |
| [420383_11](../../../slurm/slurm-gs-nl-score-ctx-eval-420383_11.out) | Nonlinear32 | 80 | .0256 | 42.306 | 23.4264 | 0.7319 | 0.2756 |

Linear continuation: train 420267 → eval 420718, run=20260908_184640. Nonlinear rank56: train 420303 → eval 420304, run=20260908_195856. Nonlinear rank80: train 420382 → eval 420383, run=20260908_221227.

당시 nonlinear rank80 task 10/11은 이전 grid가 task 범위를 허용하지 않아 최초 array에서 실패했고, grid 수정 후 별도 실행했다. 실패한 제출을 완료 결과로 중복 집계하지 않는다.

**해석.** 같은 100k stage 구성에서 Nonlinear rank56은 Linear rank56보다 .0064에서 약 2.965 KiB 작고 PSNR은 거의 같다. .0256에서는 약 1.002 KiB 작고 PSNR은 0.0335 dB 높다. Rank80은 rank56보다 일관된 우세가 없어 이후 context 연구는 rank56에 집중했다. 이는 rank80이 모든 학습 schedule/용량에서 불필요하다는 증명은 아니다.

### 6.2 Nonlinear transform 자체는 무엇을 확인했나

동기는 단일 linear low-rank map의 표현 한계를 작은 nonlinear residual MLP로 보완할 수 있는지였다. §2.2의 zero-init two-sided MLP, hidden32를 사용했다.

초기 50k, context OFF, Morton ON, rank56/80 × λ .0064/.0256 × residual ON/OFF의 8조건이다. Train job 418264는 당시 문서 기록이며, 완료 eval은 419630의 task8–14 및 420263의 task15다.

| Eval job_task | Transform task | Rank | λ | Residual | 전체 KiB | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [419630_8](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_8.out) | 8 | 56 | .0064 | ON | 98.183 | 24.5082 | 0.7665 | 0.2343 |
| [419630_9](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_9.out) | 9 | 56 | .0064 | OFF | 105.790 | 24.2580 | 0.7587 | 0.2440 |
| [419630_10](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_10.out) | 10 | 56 | .0256 | ON | 46.695 | 23.6943 | 0.7381 | 0.2673 |
| [419630_11](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_11.out) | 11 | 56 | .0256 | OFF | 55.336 | 23.5011 | 0.7303 | 0.2738 |
| [419630_12](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_12.out) | 12 | 80 | .0064 | ON | 99.152 | 24.4890 | 0.7659 | 0.2367 |
| [419630_13](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_13.out) | 13 | 80 | .0064 | OFF | 109.913 | 24.2579 | 0.7584 | 0.2438 |
| [419630_14](../../../slurm/slurm-gs-nfcgs-transform-eval-419630_14.out) | 14 | 80 | .0256 | ON | 47.314 | 23.5833 | 0.7378 | 0.2699 |
| [420263_15](../../../slurm/slurm-gs-nfcgs-transform-eval-420263_15.out) | 15 | 80 | .0256 | OFF | 55.415 | 23.4970 | 0.7301 | 0.2728 |

이 8개는 context 부모의 배경을 설명하기 위해 포함했다. 선행 Linear baseline 12개는 부록 A에 별도로 보존했다.

**해석.** 이 grid에서 residual ON이 OFF보다 더 작으면서 화질도 높다. Nonlinear에서도 residual 경로가 여전히 핵심이다. Rank56 residual ON은 .0064 98.183 KiB/24.5082 dB, .0256 46.695 KiB/23.6943 dB다. 뒤의 추가학습 모델보다 크지만 화질이 높은 operating point이므로 단순히 “옛 모델이라 열등”으로 지우면 안 된다.

## 7. Nonlinear32 + Spatial / Full 결합 4개

### 7.1 이유와 설정

Linear context의 이득과 Nonlinear transform의 이득이 결합되는지 확인했다. 이미 효과가 작았던 Mean-only/Mean+Channel을 전부 반복하지 않고 Spatial/Full에 집중했다.

- 부모: 초기 Nonlinear32 rank56, residual ON, Morton ON, λ별 50k.
- 추가 joint 50k, all codec, micro2×acc4, LR 1e-4 → 1e-5 → 1e-6.
- P0 kernel3 spatial predictor, shared entropy.
- 총 100k, train 420916 → eval 420919, run=20260910_092000.
- nonlinear grid task6/7=Spatial, task8/9=Full. task2/3 Mean, task4/5 Mean+Channel은 이 로그 묶음에 완료 결과가 없다.

| Eval job_task | λ | Context | 전체 KiB | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 동일 nonlinear 대조군 대비 크기 / PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [420919_6](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_6.out) | .0064 | Mean+Spatial | 83.633 | 24.3244 | 0.7612 | 0.2371 | -4.79% / +0.0222 dB |
| [420919_8](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_8.out) | .0064 | Full | 82.491 | 24.3238 | 0.7611 | 0.2371 | -6.09% / +0.0216 dB |
| [420919_7](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_7.out) | .0256 | Mean+Spatial | 38.236 | 23.5867 | 0.7348 | 0.2678 | -7.32% / +0.0657 dB |
| [420919_9](../../../slurm/slurm-gs-nl-score-ctx-eval-420919_9.out) | .0256 | Full | 37.958 | 23.5707 | 0.7342 | 0.2683 | -7.99% / +0.0497 dB |

### 7.2 결과와 “두 λ에서 같은 setting” 결정

.0064는 Full이 Spatial보다 1.142 KiB 작고 PSNR 차이는 −0.0006 dB다. 이 조건에서는 Full을 개선 출발점으로 선택할 근거가 강하다.

.0256은 Full이 Spatial보다 0.278 KiB 작지만 PSNR −0.0160 dB, SSIM −0.0006, LPIPS +0.0005다. **아주 가까운 두 RD 지점이며 Full의 전면 우세는 아니다.** 최초 P 후보는 그래서 .0064 Full / .0256 Spatial로 각각의 유력 후보를 골랐다.

이 설계는 각 λ 안에서 predictor P0/P1/P2를 비교하는 데는 문제가 없지만, λ 간 결과 차이를 설명하려면 기본 구조가 다르다는 변수가 남는다. 사용자가 제안한 “.0064에서 잘 되는 Full을 두 λ 공통으로 개선하자”는 방향은 다음 이유에서 합리적이다.

- .0256에서 Full을 버릴 만큼 큰 열세가 관측된 것은 아니다.
- 하나의 공통 구조로 rate 구간을 확장하는 편이 설계·배포·해석이 단순해진다.
- λ에 따른 predictor 반응을 비교할 때 architecture 차이를 제거할 수 있다.

따라서 기존 Spatial task를 재실행/삭제하지 않고 **.0256 Full P0/P1/P2 세 개를 추가**했다. “같은 setting”은 같은 구조·seed·batch·추가 steps/LR을 뜻하며, 서로 다른 λ가 반드시 동일 학습 weight에서 시작해야 한다는 뜻은 아니다. 각 λ의 원래 operating point를 유지하는 checkpoint가 필요하다.

## 8. Spatial predictor P0/P1/P2, 추가 25k

### 8.1 가설과 구현

가설은 odd score의 중심값을 예측하는 기존 linear kernel3가 너무 약하거나 Morton 이웃 범위가 너무 좁다는 것이었다.

| Arm | config 값 | odd value prediction | 초기 상태 |
| --- | --- | --- | --- |
| P0 | linear | 부모의 Conv(width,width,k3) | 그대로 |
| P1 | residual3 | 부모 Conv k3 + Conv(width,32,k3) → GELU → Conv(32,width,k1) | 새 마지막 conv 0-init |
| P2 | residual7 | 부모 Conv k3 + Conv(width,32,k7) → GELU → Conv(32,width,k1) | 새 마지막 conv 0-init |

코드의 conv는 1×k Conv2d로 구현되어 있다. 새 correction의 마지막 weight/bias를 0으로 두므로 이식 직후 P1/P2의 score value prediction은 P0와 같다. checkpoint metadata/shape validation과 stream flags도 확장했다.

입력은 §5의 decoded even anchor_delta뿐이다. odd 위치에서 kernel3는 거리 ±1의 anchors, kernel7는 ±1/±3 anchors까지 본다. P2는 receptive field와 parameter 수를 함께 늘리므로 P1/P2 차이를 순수 거리 효과 하나로 분리할 수는 없다. 스트림 수와 복호 순서는 기존 Spatial/Full과 같다.

### 8.2 Grid와 부모

| Predictor task | λ | Context | Arm | 부모 nonlinear task | Run |
| --- | --- | --- | --- | ---: | --- |
| 0 / 2 / 4 | .0064 | Full | P0 / P1 / P2 | 8 | 20260912_082429 |
| 1 / 3 / 5 | .0256 | Spatial | P0 / P1 / P2 | 7 | 20260912_082429 |
| 6 / 7 / 8 | .0256 | Full | P0 / P1 / P2 | 9 | 20260912_122749 |

첫 6개 train 422181 → eval 422182, 추가 Full 3개 train 422220 → eval 422221. 모두 부모 누적100k에서 **추가25k, scope=all, micro2×acc4, seed111123**이다.

중요한 schedule 특징: 부모 context 50k 말의 LR은 1e-6이지만, 새 Adam으로 **1e-4에서 다시 시작**했다. milestone 35k/45k가 25k 종료 이후라 이 stage에서는 LR decay가 발생하지 않는다. 즉 실질적으로 상수 1e-4다.

### 8.3 전체 결과

| Eval job_task | Predictor task | λ / Context | Arm | 전체 KiB | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [422182_0](../../../slurm/slurm-gs-spatial-pred-eval-422182_0.out) | 0 | .0064 Full | P0 | 79.166 | 24.1634 | 0.7569 | 0.2400 |
| [422182_2](../../../slurm/slurm-gs-spatial-pred-eval-422182_2.out) | 2 | .0064 Full | P1 | 79.195 | 24.1858 | 0.7576 | 0.2394 |
| [422182_4](../../../slurm/slurm-gs-spatial-pred-eval-422182_4.out) | 4 | .0064 Full | P2 | 79.055 | 24.1591 | 0.7570 | 0.2401 |
| [422182_1](../../../slurm/slurm-gs-spatial-pred-eval-422182_1.out) | 1 | .0256 Spatial | P0 | 37.446 | 23.4764 | 0.7324 | 0.2681 |
| [422182_3](../../../slurm/slurm-gs-spatial-pred-eval-422182_3.out) | 3 | .0256 Spatial | P1 | 37.465 | 23.4515 | 0.7316 | 0.2676 |
| [422182_5](../../../slurm/slurm-gs-spatial-pred-eval-422182_5.out) | 5 | .0256 Spatial | P2 | 37.323 | 23.4566 | 0.7317 | 0.2674 |
| [422221_6](../../../slurm/slurm-gs-spatial-pred-eval-422221_6.out) | 6 | .0256 Full | P0 | 37.348 | 23.4284 | 0.7312 | 0.2683 |
| [422221_7](../../../slurm/slurm-gs-spatial-pred-eval-422221_7.out) | 7 | .0256 Full | P1 | 37.313 | 23.4286 | 0.7315 | 0.2684 |
| [422221_8](../../../slurm/slurm-gs-spatial-pred-eval-422221_8.out) | 8 | .0256 Full | P2 | 37.149 | 23.4556 | 0.7318 | 0.2679 |

### 8.4 같은 추가학습 대조군 P0와의 차이

| λ / Context | 비교 | Δbytes | ΔPSNR | 해석용 대조 |
| --- | --- | --- | --- | --- |
| .0064 Full | P1 − P0 | +29.52 | +0.0224 | task 2 − 0 |
| .0064 Full | P2 − P0 | -113.71 | -0.0043 | task 4 − 0 |
| .0256 Spatial | P1 − P0 | +19.77 | -0.0249 | task 3 − 1 |
| .0256 Spatial | P2 − P0 | -125.38 | -0.0198 | task 5 − 1 |
| .0256 Full | P1 − P0 | -35.47 | +0.0002 | task 7 − 6 |
| .0256 Full | P2 − P0 | -203.24 | +0.0272 | task 8 − 6 |

- .0064 Full에서 P1은 거의 같은 용량에 +0.0224 dB다. P2는 더 작지만 PSNR이 조금 낮다.
- .0256 Spatial에서 P1/P2의 PSNR 개선은 없다. 일부 LPIPS는 좋아져 지표 간 trade-off가 있다.
- .0256 Full에서는 P2가 P0보다 약 203 B 작고 PSNR +0.0272 dB, SSIM/LPIPS도 개선된다. 이 조건의 P2는 긍정적이다.
- 그러나 결과가 두 λ에 걸쳐 일관되지 않고 차이도 작다. 단일 seed 결과에서 predictor 확대를 다음 주력으로 확정하기에는 근거가 약하다.

### 8.5 부모 대비 변화가 더 큰 이유를 어떻게 읽어야 하나

| 시작 부모 → P0 25k | Δ전체 bytes | ΔPSNR |
| --- | ---: | ---: |
| .0064 Full | −3,405.15 | −0.1604 dB |
| .0256 Spatial | −809.26 | −0.1103 dB |
| .0256 Full | −624.84 | −0.1423 dB |

Predictor를 바꾸지 않은 P0도 크기/화질이 크게 이동했다. .0064에서는 score가 약 3,864 B 줄고 residual 합은 약 459 B 늘었다. 이는 모델이 score만 더 잘 부호화한 것이 아니라 representation과 rate 배분까지 바꿨다는 징후다.

부모 대비 100배 높은 LR 재시작과 새 optimizer, 추가 joint 학습이 중요한 혼합 요인이다. **LR 재시작이 유일한 원인이라고 증명한 것은 아니지만**, 부모 대비 하락을 P1/P2 구조의 실패라고 부를 수는 없다. 반대로 모든 payload 감소를 predictor가 좋아진 효과라고 부르는 것도 잘못이다.

이 결과에서 나온 후속 질문은 두 개였다. “낮은 LR로 기존 RD 지점을 보존하며 joint 개선이 가능한가?”와 “아예 복원을 고정하면 entropy만 개선할 수 있는가?” 다음 두 가지 실험은 이를 분리한다.

## 9. Probability-only: 복원 고정, 확률 모델 4종 × 2λ

### 9.1 목적과 동일 출발점

P 실험에서는 중심값 예측, quantization, residual이 함께 변했다. 이번에는 **같은 복원 score symbols를 얼마나 짧게 부호화할 수 있는지**를 분리했다.

모든 arm은 원래 Nonlinear32 Full P0의 λ별 checkpoint, 즉 nonlinear task8/9(누적100k)를 부모로 쓴다. P25k 결과에서 이어가지 않는다.

공통: rank56, residual/Morton ON, Full, P0, micro2×acc4, seed111123, 추가10k, Adam1e-4 → 7k에서1e-5. Train 422558 → eval 422559, run=20260913_112316. 짝수 task는 .0064, 홀수는 .0256.

### 9.2 무엇을 고정하고 무엇을 학습했나

[codec.py](../globalsplat/compression/codec.py)의 feature_codec_train_scope=score_probability는 다음을 고정한다.

- geometry projection, analysis/synthesis basis와 MLP.
- score normalization scale, mean conditioner의 offset/step.
- channel predictor 및 spatial **value** predictor.
- residual codec 전체.
- entropy quantiles/median, 즉 정수 symbol 복원의 lattice.

학습 대상은 active EntropyBottleneck의 density parameters와 optional Gaussian scale/scale predictor뿐이다. 로그에 출력된 trainable parameter는 Shared 약3.2K, Split6.5K, Gaussian3.3K, Conditional10.7K다.

Quantile을 requires_grad=False로 두는 것만으로는 충분하지 않다. 기존 recipe는 no-grad의 deterministic quantile update를 직접 호출하므로 **loss.quantile_update_interval=0**도 지정해 median이 움직이지 않게 했다. CDF table은 부호화를 위해 갱신하되 quantile/복원 lattice는 바꾸지 않는다.

기존 wrapper는 여전히 render/consistency/RD loss를 계산하지만, 학습 가능한 parameter로 전파되는 것은 probability rate 쪽이다. λ는 부모 operating point를 식별하는 값이며, 이 단계가 새로운 distortion-rate representation을 찾는 joint 학습은 아니다.

### 9.3 확률 모델 구현 차이

| Arm / task | Even symbols | Odd symbols | 초기화 / 조건 |
| --- | --- | --- | --- |
| Shared, 0/1 | 기존 slice별 learned factorized | Even과 같은 entropy model | 부모 그대로, 추가 fit 대조군 |
| Split, 2/3 | 기존 learned factorized | 독립 learned factorized | Odd에 부모 Even 모델과 quantiles를 복제 |
| Gaussian, 4/5 | 기존 learned factorized | 채널별 static Gaussian scale | 부모 median 고정, scale=1 시작 |
| Conditional, 6/7 | 기존 learned factorized | anchor-conditioned Gaussian scale | static과 같은 초기 scale, zero-init 조건 보정 |

Split의 의미는 anchor와 predicted-odd residual에 **서로 다른 marginal density**를 허용하는 것이다. Even entropy도 추가로 학습한다. 새 stream/pass/scene side information은 필요 없고, 추가 비용은 shared model parameter와 entropy state다.

Gaussian은 복원 lattice를 유지하려고 odd의 중심을 기존 EntropyBottleneck median에 고정한다. scale은 softplus(raw)+0.11이고 1에서 시작한다. 자유롭게 mean까지 학습하는 Gaussian baseline이 아니다.

Conditional scale은 decoded even anchor_delta → Conv k3(width→32) → GELU → Conv k1(32→width)로 local log multiplier를 만든다. 마지막 conv는 0-init이다.

```text
sigma_base = softplus(raw_scale) + 0.11
sigma_local = sigma_base * exp(2 * tanh(local_log_multiplier / 2))
sigma_local = clamp(sigma_local, 0.11, 256)
```

GaussianConditional은 0.11–256 범위의 64단계 logarithmic scale table을 사용한다. Conditional은 Gaussian과 같은 초기 분포/고정 중심에서 출발하지만 입력을 이용한 scale modulation과 추가 parameter를 갖는다. 두 모델 차이는 현재 학습 절차에서 조건부 scale 예측의 유용성을 검증하며, 모든 형태의 conditional probability를 대표하지는 않는다.

### 9.4 전체 실제 결과

| Eval job_task | λ | Entropy | 전체 KiB | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Shared 재학습 대비 전체 크기 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [422559_0](../../../slurm/slurm-gs-score-prob-eval-422559_0.out) | .0064 | Shared | 82.463 | 24.3238 | 0.7611 | 0.2371 | 0.00% |
| [422559_2](../../../slurm/slurm-gs-score-prob-eval-422559_2.out) | .0064 | Split | 81.174 | 24.3238 | 0.7611 | 0.2371 | -1.56% |
| [422559_4](../../../slurm/slurm-gs-score-prob-eval-422559_4.out) | .0064 | Gaussian | 88.167 | 24.3238 | 0.7611 | 0.2371 | +6.92% |
| [422559_6](../../../slurm/slurm-gs-score-prob-eval-422559_6.out) | .0064 | Conditional scale | 81.135 | 24.3238 | 0.7611 | 0.2371 | -1.61% |
| [422559_1](../../../slurm/slurm-gs-score-prob-eval-422559_1.out) | .0256 | Shared | 37.952 | 23.5707 | 0.7342 | 0.2683 | 0.00% |
| [422559_3](../../../slurm/slurm-gs-score-prob-eval-422559_3.out) | .0256 | Split | 36.920 | 23.5707 | 0.7342 | 0.2683 | -2.72% |
| [422559_5](../../../slurm/slurm-gs-score-prob-eval-422559_5.out) | .0256 | Gaussian | 44.518 | 23.5707 | 0.7342 | 0.2683 | +17.30% |
| [422559_7](../../../slurm/slurm-gs-score-prob-eval-422559_7.out) | .0256 | Conditional scale | 37.021 | 23.5707 | 0.7342 | 0.2683 | -2.45% |

Shared 재학습 자체는 원래 부모 대비 .0064에서 약28.87 B, .0256에서 약6.24 B만 줄였다. 따라서 Split의 이득은 단순히 10k 더 돌렸기 때문이라고 보기 어렵다.

Split은 Shared 재학습 대비 score stream을 각각 약1.86%, 3.39% 줄이고, 전체 payload를 1.56%, 2.72% 줄였다. 두 λ에서 일관된 결과다.

### 9.5 Stream 분해와 시간

| λ | Entropy | Score B | Residual y B | Residual z B | Mean B | Outer container B | 수신 복원 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| .0064 | Shared | 70924.906 | 11738.844 | 174.507 | 1472 | 132 | 89.9 |
| .0064 | Split | 69604.342 | 11738.844 | 174.507 | 1472 | 132 | 89.6 |
| .0064 | Gaussian | 76765.617 | 11738.844 | 174.507 | 1472 | 132 | 125.6 |
| .0064 | Conditional | 69565.374 | 11738.844 | 174.507 | 1472 | 132 | 126.2 |
| .0256 | Shared | 31149.344 | 5998.320 | 111.128 | 1472 | 132 | 85.1 |
| .0256 | Split | 30093.048 | 5998.320 | 111.128 | 1472 | 132 | 81.0 |
| .0256 | Gaussian | 37873.270 | 5998.320 | 111.128 | 1472 | 132 | 116.2 |
| .0256 | Conditional | 30195.839 | 5998.320 | 111.128 | 1472 | 132 | 120.1 |

같은 λ 내 residual y/z, mean, outer container는 로그 정밀도 내 동일하다. 전체 bytes 차이는 score bytes 차이로 설명된다. 화질도 네 arm이 소수 넷째 자리까지 동일하다. 이는 복원 고정 설계와 일치하지만, 집계값만으로 장면별 bit-exact를 증명하지는 않는다.

Conditional은 static Gaussian보다 훨씬 낫다. 그러나 Split과 직접 비교하면 다음과 같다.

- .0064: Conditional이 Split보다 **38.97 B 작음**, 전체의 약0.047%.
- .0256: Conditional이 Split보다 **102.79 B 큼**, 전체의 약0.272%.
- 수신 복원 시간: Split 89.6/81.0 ms, Conditional 126.2/120.1 ms. 이 로그에서는 약41%/48% 더 길다.

타이밍은 반복 측정/부하 통제한 전용 벤치마크가 아니므로 확정적인 속도비는 아니다. 그래도 현재 관측된 수십 B의 일관되지 않은 이득을 위해 Conditional을 공통 기본으로 채택할 근거는 약하다.

Static Gaussian은 Shared보다 .0064에서6.92%, .0256에서17.30% 더 크다. 현재 고정 mean, unit-scale 초기화, 10k fit과 finite CDF table 조건에서 부적합했다는 결론까지가 타당하다. “Gaussian 분포 자체가 틀렸다”거나 “조건부 확률 모델은 불필요하다”는 결론은 아니다.

## 10. Full low-LR 10k 결과

별도 [실험 설계](NFCGS_SCORE_PROBABILITY_EXPERIMENT_2026-09-13.md)에 두 후속 축을 함께 구현했다. 여기서는 원래 Nonlinear32 Full 부모로부터 전체 codec을 매우 낮은 LR로 10k 더 학습해, 기존 화질·크기 균형을 크게 이동시키지 않고 joint 적응할 여지가 있는지 확인했다.

| Task | λ | Predictor | 부모 | 추가 학습 |
| --- | --- | --- | --- | --- |
| 0 | .0064 | P0 | 원래 Nonlinear Full task8, 누적100k | all, 10k, LR1e-6 |
| 1 | .0064 | 새 P2 | 위와 동일 | all, 10k, LR1e-6 |
| 2 | .0256 | P0 | 원래 Nonlinear Full task9, 누적100k | all, 10k, LR1e-6 |
| 3 | .0256 | 새 P2 | 위와 동일 | all, 10k, LR1e-6 |

- Full, rank56, Nonlinear32, residual/Morton ON, **shared entropy**.
- 새 P2 correction은 동일 부모에 zero-init한다. 이미 25k 학습된 P2를 식히는 실험이 아니다.
- 부모 weight만 로드하고 새 Adam을 사용한다. 부모 optimizer state까지 복원한 엄밀한 resume가 아니다.
- micro2×acc4, seed111123, quantile update500, constant LR1e-6.
- 출력 root는 nfcgs_full_lowlr10k / nfcgs_full_lowlr10k_eval.

### 10.1 실행 상태

- 원래 train array `422943`의 task 1/2/3은 10k 완료, 대응 eval `422944`도 전체 TEST 6,991 scene을 완료했다.
- task 0은 optimizer step 5,000 checkpoint 저장 시 서버 저장 공간 부족(`Errno 28`)으로 실패했다.
- task 0만 재제출한 `423009_0`은 10k 계산까지 도달했지만 최종 checkpoint 저장에서 같은 저장 공간 부족으로 실패했다. 따라서 `.0064/P0`의 유효한 step10k 평가 결과는 없다.
- eval 세 건의 stderr에는 일반적인 PyTorch/Lightning warning만 있고 fatal error는 없다.

### 10.2 전체 TEST 결과

| λ | Predictor | 전체 KiB | Score B | Residual y/z B | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .0064 | P0 | 미완료 | — | — | — | — | — |
| .0064 | P2 residual7 | 82.430 | 70,867.233 | 11,761.695 / 175.439 | 24.3256 | .7611 | .2370 |
| .0256 | P0 linear | 37.932 | 31,115.291 | 6,011.991 / 111.517 | 23.5701 | .7342 | .2684 |
| .0256 | P2 residual7 | 37.921 | 31,106.122 | 6,009.835 / 111.455 | 23.5716 | .7343 | .2684 |

mean 1,472 B와 outer container 132 B는 모든 완료 arm에서 동일하다.

### 10.3 해석

동일한 원래 Full 부모 대비:

- `.0064/P2`: 84,471.129 → 84,408.367 B, **−62.762 B (−0.074%)**. Score는 86.546 B 줄었지만 residual y/z가 합계 23.783 B 늘었다.
- `.0256/P0`: 38,869.036 → 38,842.798 B, **−26.238 B (−0.068%)**.
- `.0256/P2`: 38,869.036 → 38,831.412 B, **−37.624 B (−0.097%)**.
- `.0256`의 직접 P2−P0 차이는 **−11.386 B/scene (−0.029%)**와 +0.0015 dB에 불과하다. 새 P2가 방향상 약간 낫지만 실질적 개선으로 보기는 어렵다.

현재 best인 원래 Full + Split probability-only와 비교하면 low-LR 결과가 오히려 크다.

- `.0064`: P2 low-LR 84,408.367 B vs Split 83,121.693 B, **+1,286.674 B (+1.548%)**.
- `.0256`: P2 low-LR 38,831.412 B vs Split 37,806.496 B, **+1,024.916 B (+2.711%)**.

따라서 이 실험이 보여준 것은 “joint low-LR 적응이 전혀 안 된다”가 아니라, **10k 추가 계산으로 얻는 순효과가 0.1% 미만이고 P2의 추가 효과도 거의 없다**는 것이다. 엄밀한 `.0064` P0/P2 표를 채우기 위해 task 0을 세 번째로 재실행할 실용적 이유는 작다. 현재 codec 기준점은 계속 원래 Nonlinear32 Full P0 + Split probability fit으로 둔다.

## 11. 종합: 무엇을 배웠고, 무엇은 아직 모르는가

### 11.1 실용적인 개선 경로

| λ | 모델/단계 | 누적 학습 | 전체 KiB | PSNR | 직전 표 행 대비 주의사항 |
| --- | --- | --- | --- | --- | --- |
| .0064 | Linear factorized | 50k+50k | 90.805 | 24.3024 | 동일-budget 기준점 |
| .0064 | Linear Full | 50k+50k | 84.905 | 24.3232 | context 구조의 효과 |
| .0064 | Nonlinear Full | 50k+50k | 82.491 | 24.3238 | 다른 초기 transform 가지와 비교 |
| .0064 | Nonlinear Full + Split | 50k+50k+10k | 81.174 | 24.3238 | 복원 고정 probability fit |
| .0256 | Linear factorized | 50k+50k | 42.258 | 23.4875 | 동일-budget 기준점 |
| .0256 | Linear Full | 50k+50k | 39.233 | 23.5352 | context 구조의 효과 |
| .0256 | Nonlinear Full | 50k+50k | 37.958 | 23.5707 | 다른 초기 transform 가지와 비교 |
| .0256 | Nonlinear Full + Split | 50k+50k+10k | 36.920 | 23.5707 | 복원 고정 probability fit |

최초 동일-budget Linear factorized control과 최종 Split을 비교하면:

- .0064: 90.805 → 81.174 KiB, **−10.61%**, PSNR 24.3024 → 24.3238.
- .0256: 42.258 → 36.920 KiB, **−12.63%**, PSNR 23.4875 → 23.5707.

이 합산 비교는 “현재 도달한 codec의 성능”을 보여준다. transform 초기학습 가지가 다르고 Split은 추가10k가 있으므로 전부 동일 compute에서 한 가지 구조만 바꾼 순수 효과로 표현하면 안 된다. 또한 두 λ의 두 점만으로 BD-rate 개선 수치를 만들지 않았다.

### 11.2 다음 기본 모델 선택

현재 작업 기준으로 추천하는 것은 **원래 Nonlinear32 Full P0 + Split probability fit**이다. 이유는 다음과 같다.

1. 두 λ에서 같은 architecture를 쓸 수 있다.
2. context의 이득이 matched continuation control로 확인됐다.
3. Split은 복원 변경 없이 실제 payload를 두 λ 모두 줄였다.
4. Conditional보다 모델이 단순하고 로그상 복호 시간이 유리하다.

하지만 다른 후보를 지우지는 않는다.

- .0064 Split 81.174 KiB/24.3238 dB와 P1 25k의79.195 KiB/24.1858 dB는 용량/화질 trade-off다.
- .0256 Split은 P2 Full 25k보다 약234.47 B 작고 PSNR +0.1151 dB, SSIM +0.0024지만 **LPIPS는0.2683 대0.2679로 더 나쁘다**. 모든 화질 지표에서 지배한다고 표현하면 안 된다.
- 초기 Nonlinear rank56 residual ON은 더 크지만 높은 PSNR을 제공하므로 RD 곡선의 고화질 쪽 후보로 남는다.

### 11.3 아직 검증하지 않은 주장/실험

- Full이 더 넓은 λ 구간에서도 Spatial보다 좋은지: 현재 두 λ만으로 미확정.
- Single seed의 수십 B/0.02 dB 수준 차이가 반복되는지: seed 반복·paired scene 통계 없음.
- `.0064` low-LR P0/P2의 직접 matched 비교: P0 최종 checkpoint 저장 실패로 미확정. 완료된 세 결과의 개선은 0.1% 미만이다.
- Split을 처음부터 joint 학습하면 reconstruction-locked fit보다 더 좋아지는지: 미실험.
- Conditional learned/non-Gaussian density가 Split보다 좋은지: 이번 Gaussian scale 실험만으로 판단 불가.
- residual entropy를 decoded low-rank feature로 condition하면 이득이 나는지: 구현 후보로 논의했지만 완료 결과 없음.
- Score와 residual에 대칭적인 강도의 held-out context probability model을 적용했을 때의 상대 개선량: 현재 probe는 score에만 richer context를 써서 미확정.
- 순수 architecture 비교의 matched-rate 성능, 여러 λ BD-rate, shared model 전송 비용, 통제된 복호 시간: 추가 평가 필요.

“좋은 codec” 관점에서는 실험 문서의 순서를 따르는 것보다, **실제 total bytes–화질 곡선의 개선이 복잡도/학습비용 대비 충분한지**가 기준이다. 지금 증거는 무작정 spatial value predictor를 크게 만드는 방향보다, 이미 좋은 복원을 유지하면서 남아 있는 확률 mismatch를 줄이는 방향에 더 힘을 실어준다.

## 12. 구현·재현 파일 지도

### 12.1 모델 코드와 테스트

| 파일 | 역할 |
| --- | --- |
| [score_context.py](../globalsplat/compression/score_context.py) | Mean/Channel/Spatial, P1/P2 correction, Split/Gaussian/Conditional, causal encode/decode |
| [codec.py](../globalsplat/compression/codec.py) | Nonlinear analysis/synthesis, train scope, score/residual 통합, compress/decompress |
| [config.py](../globalsplat/compression/config.py) | predictor/entropy/slice/hidden 옵션 및 조합 검증 |
| [checkpoint.py](../globalsplat/compression/checkpoint.py) | metadata·shape 추론, strict-load, predictor/entropy mismatch 검사 |
| [initialize_nfcgs_from_vanilla.py](../scripts/initialize_nfcgs_from_vanilla.py) | 부모 warm-start, 신규 모듈 초기화 및 entropy 이식 |
| [check_nfcgs_checkpoint.py](../scripts/check_nfcgs_checkpoint.py) | 서버 checkpoint 구성 확인 |
| [globalsplat_nfcgs_rank56.yaml](../config/model/globalsplat_nfcgs_rank56.yaml) | codec 모델 옵션 |
| [model_wrapper.py](../globalsplat/model/model_wrapper.py) | subset loss, RD 항, quantile 주기, 실제 평가/타이밍 집계 |
| [test_score_context_codec.py](../tests/test_score_context_codec.py) | causal context, round-trip, probability scope/복원 고정 등 |
| [test_nfcgs_codec.py](../tests/test_nfcgs_codec.py) | codec 통합, checkpoint/bitstream 검증 |
| [test_spatial_predictor_slurm.py](../tests/test_spatial_predictor_slurm.py) | P grid, 부모/checkpoint 경로와 실행 설정 |
| [test_score_probability_slurm.py](../tests/test_score_probability_slurm.py) | probability/low-LR grid, scope/LR/quantile 설정 |

Bitstream의 context flags는 mean/channel/spatial, residual3/7, split/gaussian/conditional을 구분한다. Decode 시 설정 mismatch를 거절한다. 그러나 flag가 모델 weight 자체를 식별하는 것은 아니므로 송수신기는 동일한 checkpoint를 사용해야 한다.

Entropy/context 연산은 bf16 외부 학습 중에도 내부에서 autocast를 끄고 FP32로 처리해 forward/compress/decompress 수치 경로를 맞춘다. 새 entropy/CDF buffer는 checkpoint 이식 후 적절히 rebuild하고, 기존 metadata는 기본 linear predictor/shared entropy로 해석한다.

이전 구현 시점 검증 기록은 score 관련43개, 새 Slurm25개 통과다. 전체248개 통과와 별도 기존 transform Slurm node-pin 기대 충돌 1건이 기록돼 있다. **이번 정리 작업에서 해당 테스트를 재실행한 것은 아니다.** 자세한 당시 기록은 [probability 설계 문서](NFCGS_SCORE_PROBABILITY_EXPERIMENT_2026-09-13.md)의 Local verification을 참조한다.

### 12.2 실험별 Slurm 진입점

각 prefix에는 train/eval wrapper, grid, submitter가 있다. 아래는 실행 설정을 추적하기 위한 링크이지 재제출 지시가 아니다.

| 실험 | Train / Eval | Grid / Submit |
| --- | --- | --- |
| 초기 transform | [train](../scripts/slurm/train_nfcgs_transform.slurm) / [eval](../scripts/slurm/eval_nfcgs_transform.slurm) | [grid](../scripts/slurm/nfcgs_transform_grid.sh) |
| Linear context | [train](../scripts/slurm/train_nfcgs_score_context.slurm) / [eval](../scripts/slurm/eval_nfcgs_score_context.slurm) | [grid](../scripts/slurm/nfcgs_score_context_grid.sh) / [submit](../scripts/slurm/submit_nfcgs_score_context.sh) |
| Linear continuation | [train](../scripts/slurm/train_nfcgs_factorized_continuation.slurm) / [eval](../scripts/slurm/eval_nfcgs_factorized_continuation.slurm) | [grid](../scripts/slurm/nfcgs_factorized_continuation_grid.sh) / [submit](../scripts/slurm/submit_nfcgs_factorized_continuation.sh) |
| Nonlinear context/control | [train](../scripts/slurm/train_nfcgs_nonlinear_score_context.slurm) / [eval](../scripts/slurm/eval_nfcgs_nonlinear_score_context.slurm) | [grid](../scripts/slurm/nfcgs_nonlinear_score_context_grid.sh) / [submit](../scripts/slurm/submit_nfcgs_nonlinear_score_context.sh) |
| P0/P1/P2 | [train](../scripts/slurm/train_nfcgs_spatial_predictor.slurm) / [eval](../scripts/slurm/eval_nfcgs_spatial_predictor.slurm) | [grid](../scripts/slurm/nfcgs_spatial_predictor_grid.sh) / [submit](../scripts/slurm/submit_nfcgs_spatial_predictor.sh) |
| Probability | [train](../scripts/slurm/train_nfcgs_score_probability.slurm) / [eval](../scripts/slurm/eval_nfcgs_score_probability.slurm) | [grid](../scripts/slurm/nfcgs_score_probability_grid.sh) / [submit](../scripts/slurm/submit_nfcgs_score_probability.sh) |
| Low-LR, 완료 eval 3개 / `.0064 P0` 저장 실패 | [train](../scripts/slurm/train_nfcgs_full_cooldown.slurm) / [eval](../scripts/slurm/eval_nfcgs_full_cooldown.slurm) | [grid](../scripts/slurm/nfcgs_full_cooldown_grid.sh) / [submit](../scripts/slurm/submit_nfcgs_full_cooldown.sh) |

### 12.3 부모 checkpoint를 찾는 방법

서버 repo 기준 경로다. 정확한 eval별 절대 checkpoint 경로는 JSON의 checkpoint 필드, 최신17개 학습의 실제 부모는 recent_training[].parent_checkpoint에 있다.

```text
초기 Linear 부모:
outputs/nfcgs_paper24_subset_train/rank56/lambda{L}/residual_on/
  checkpoints/nfcgs_paper24_subset_rank56_lambda{L}_residual_on_morton_on/
  version_0/step000050000.ckpt

초기 Nonlinear 부모:
outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda{L}/residual_on/
  checkpoints/nfcgs_transform_nonlinear32_rank56_lambda{L}_residual_on_morton_on/
  version_0/step000050000.ckpt

P / Probability / Low-LR의 원래 Full 부모:
outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda{L}/residual_on/
  checkpoints/nfcgs_nonlinear_score_context_joint_rank56_lambda{L}_residual_on_morton_on_m1c1s1/
  version_0/step000050000.ckpt

{L} = 0p0064 또는 0p0256
Spatial 부모는 마지막 예시의 m1c1s1을 m1c0s1로 바꾼 해당 λ checkpoint.
```

Run tag와 global_step은 서로 다른 의미다. 특히 다른 실험 root의 step000050000.ckpt를 이름만 보고 같은 누적 학습량으로 취급하지 않는다.

## 부록 A. Context 이전 Linear baseline 12개

다음은 [이전 결과 문서](NFCGS_EXPERIMENT_RESULTS_2026-09-10.md)의 사용자 제공 표를 그대로 보존한 배경이다. 원래 크기 열은 “kB”로 표기됐고, 이번 로컬 eval bundle에는 해당 원본 bytes가 없다. 따라서 위의 KiB로 재계산한 표와 정밀한 byte 차이를 직접 계산하지 않는다.

| Rank | λ | Residual | Morton | PSNR | SSIM | LPIPS | 원래 크기(kB 표기) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
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

Codec-OFF 참고값은 PSNR24.7004 / SSIM0.7682 / LPIPS0.2480이다. 이는 모든 perceptual 지표의 수학적 상한이라는 뜻이 아니며, payload가 정의된 codec operating point도 아니다.

이전 결과는 residual ON과 Morton 순서의 유용성을 시사했고 rank80의 일관된 우세는 없었다. 그 위에서 “rank를 늘리기보다 rank56 score의 남은 의존성을 이용하자”는 첫 context 질문이 나왔다.

## 부록 B. 원래 설계/결과 문서

- [첫 score-context 설계와 구현](NFCGS_SCORE_CONTEXT.md)
- [Nonlinear transform 설계](NFCGS_TRANSFORM.md)
- [추가학습 대조군 / 병렬 후속 설계](NFCGS_PARALLEL_FOLLOWUP.md)
- [9월10일 종합 결과](NFCGS_EXPERIMENT_RESULTS_2026-09-10.md)
- [Nonlinear+context 결과](NFCGS_NONLINEAR_CONTEXT_RESULTS_2026-09-11.md)
- [P0/P1/P2 설계 및 Full 추가](NFCGS_SPATIAL_PREDICTOR_EXPERIMENT_2026-09-11.md)
- [Probability / low-LR 설계](NFCGS_SCORE_PROBABILITY_EXPERIMENT_2026-09-13.md)
- [세션 인수인계](SESSION_HANDOFF_2026-09-11.md)
- [원본 로그에서 추출한 48개 eval + 기존 최신 train 기록](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json)

로그 링크는 이 workspace의 ../../../slurm을 가리킨다. 문서만 서버로 옮기면 로컬 로그 상대링크는 열리지 않을 수 있다. 서버 산출물 위치는 JSON에 보존한 절대 경로를 사용한다.

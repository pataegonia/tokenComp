# NFC-GS 실험 전용 다음 세션 인수인계 — 2026-09-15

이 문서는 **지금까지 실제로 수행한 실험과 그 결론만** 넘긴다. 코드 변경, 서버 파일 정리, Git 상태는 범위 밖이다. 다음 세션은 완료 실험을 다시 실행하지 말고, 아래의 현재 기준 checkpoint에서 새로운 질문만 검증한다.

상세 근거는 다음 두 파일에 있다.

- [전체 실험 이력과 해석](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14.md)
- [원본 로그에서 추출한 48개 full-test 지표](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json)
- [128/128-scene 고정-symbol entropy probe](NFCGS_ENTROPY_PROBE_2026-09-14.md)

## 1. 현재 결론

두 λ에 공통으로 사용할 현재 기준점은 **Nonlinear32, rank56, residual/Morton ON, Full context P0, even/odd Split probability model**이다.

| λ | 전체 bytes / KiB | Score B | Residual y/z B | Mean / container B | PSNR | SSIM | LPIPS | 수신 feature 복원 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| .0064 | 83,121.693 / 81.174 | 69,604.342 | 11,738.844 / 174.507 | 1,472 / 132 | 24.3238 | .7611 | .2371 | 89.6 ms |
| .0256 | 37,806.496 / 36.920 | 30,093.048 | 5,998.320 / 111.128 | 1,472 / 132 | 23.5707 | .7342 | .2683 | 81.0 ms |

Checkpoint:

```text
.0064
/ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt

.0256
/ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt
```

이 선택의 근거는 다음과 같다.

1. decoder-causal score context는 factorized 추가학습 대조군보다 두 λ 모두 실제 bitstream을 줄였다.
2. Split은 같은 부모·같은 10k probability-only 학습에서 Shared보다 .0064/.0256의 전체 크기를 각각 1.56%/2.72% 줄였다.
3. 복원 경로를 고정했기 때문에 Split의 차이는 score 확률 모델 차이로 격리된다.
4. P1/P2 value predictor 확대와 전체 모델 low-LR 추가학습은 두 λ에서 일관된 큰 이득이 없었다.
5. Conditional Gaussian은 Split보다 일관되게 작지 않았고 로그상 수신 복원도 더 느렸다.

현재 실행 중인 학습/평가 실험은 없다. Full low-LR `.0064/P0`만 최종 checkpoint 저장 실패로 비어 있지만, 아래 결과상 결론을 바꿀 가능성이 작아 다시 실행하지 않는 쪽으로 판단했다.

## 2. 공통 측정 조건과 비교 시 주의점

### 2.1 Codec와 actual rate

- 데이터는 RE10K 256×256이다.
- scene당 token 4,096개, 최종 stage에서 token당 Gaussian 8개, 즉 scene당 32,768 Gaussians다.
- 736-D observable feature를 scene mean, low-rank score, residual로 나눠 압축한다.
- mean은 항상 736×FP16 = 1,472 B다.
- residual ON일 때 outer/residual container는 132 B다.
- Full rank56 score는 channel slice 4개 × even/odd = 8 entropy streams다.
- 평가의 `actual_bytes`는 실제 entropy compress→decompress 후 mean, score, residual y/z, container를 모두 포함한다. 공유 neural weight/checkpoint 크기는 scene payload에서 제외한다.

### 2.2 Full-test protocol

- all-test, context 12 views, target 8 views, batch1, seed0, bf16-mixed.
- 완료 full eval은 실제 bitstream을 복호해 렌더링한 6,991 scene 평균이다.
- 로그의 6,986은 첫 5회를 뺀 timing call 수이지 평가 scene 수가 아니다.
- 타이밍은 별도 부하 통제 benchmark가 아니므로 작은 속도 차이를 확정값으로 읽지 않는다.
- 두 λ는 구조와 recipe는 같아도 각각 자기 λ로 학습된 별도 부모 weight에서 시작한다. 서로 다른 λ의 절대 수치보다 각 λ 내부의 matched 비교를 우선한다.

### 2.3 공통 학습 recipe

| 단계 | 시작점 | 추가 optimizer steps | LR | Scope / 특징 |
| --- | --- | ---: | --- | --- |
| 초기 transform | PCA init + frozen GlobalSplat backbone | 50k | 1e-4, 35k/45k decay | all; Nonlinear 초기 실험은 micro1×acc8 |
| Context / factorized continuation | 해당 Linear/Nonlinear 50k | 50k | 1e-4, 35k/45k decay | all, micro2×acc4 |
| P0/P1/P2 | 해당 Nonlinear context 100k | 25k | 상수 1e-4 | all, 새 Adam |
| Probability-only | 원래 Nonlinear Full P0 100k | 10k | 1e-4, 7k에서 1e-5 | score_probability, quantile update OFF |
| Full low-LR | 원래 Nonlinear Full P0 100k | 10k | 상수 1e-6 | all, 새 Adam |
| Entropy probe | Full P0+Split 110k | 학습 없음 | 없음 | symbol/복원 고정, TRAIN table fit→held-out TEST |

공통 joint recipe는 context pool24에서 A/B 각13 views, shared target12, subset consistency ON, micro2×acc4, seed111123, data-loader seed403, bf16-mixed, gradient clip0.5다. Backbone은 고정하지만 geometry projection과 codec은 `scope=all` 단계에서 학습된다. 새 stage는 부모 weight를 로드하되 optimizer state는 이어받지 않았다.

## 3. 실험 계보

```text
Linear factorized 50k
 ├─ +50k factorized continuation
 └─ +50k Mean / Mean+Channel / Mean+Spatial / Full context

Nonlinear32 factorized 50k
 ├─ rank56/80 × residual ON/OFF 초기 grid
 ├─ +50k factorized continuation
 └─ rank56 residual ON +50k Spatial / Full context = 원래 Full P0 부모
      ├─ +25k P0/P1/P2 joint
      ├─ +10k Shared/Split/Gaussian/Conditional probability-only
      └─ +10k P0/새 P2 Full low-LR

원래 Full P0 + Split
 └─ 학습 없는 128 TRAIN / 128 TEST fixed-symbol entropy probe
```

Probability-only와 low-LR은 P25k checkpoint에서 이어간 것이 아니다. 둘 다 원래 Nonlinear Full P0 100k로 돌아가 갈라진 별도 가지다.

## 4. 완료 실험과 결과

### 4.1 Context 이전 기준선과 초기 Nonlinear transform

이전 Linear baseline은 rank56/80, λ .0064/.0256, residual ON/OFF, Morton ON/OFF의 12개다. 이 묶음은 사용자 제공 표만 남아 있고 현재 로컬에 원본 eval 로그가 없으므로 표의 `kB`와 후속 actual KiB의 정밀 차이를 다시 계산하지 않는다.

핵심 관측:

- Residual ON이 OFF보다 대체로 더 작고 화질도 높았다.
- Morton ON이 유용했다.
- Rank80은 rank56보다 일관되게 좋지 않았다.
- 이후 연구 기준은 rank56, residual/Morton ON으로 좁혔다.

초기 Nonlinear32 50k grid는 rank56/80 × 두 λ × residual ON/OFF, 총8개였다. Train 기록 418264, eval 419630 task8–14 및 420263 task15다.

| λ | Rank | Residual | 전체 KiB | PSNR | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| .0064 | 56 | ON | 98.183 | 24.5082 | .7665 | .2343 |
| .0064 | 56 | OFF | 105.790 | 24.2580 | .7587 | .2440 |
| .0256 | 56 | ON | 46.695 | 23.6943 | .7381 | .2673 |
| .0256 | 56 | OFF | 55.336 | 23.5011 | .7303 | .2738 |

Rank80 네 결과도 rank56을 일관되게 이기지 못했다. Nonlinear rank56 residual ON은 후속 저용량 모델보다 크지만 PSNR이 더 높은 operating point이므로 RD 곡선의 고화질 후보로 남는다.

### 4.2 Linear score context 8개와 matched factorized control

목적은 score의 같은 채널·모든 token에 동일 marginal을 쓰던 방식에서, 수신기가 이미 아는 정보를 이용하는 것이었다.

- Mean: 전송된 FP16 scene mean으로 score offset과 scene/channel별 quantization-step 배수를 예측.
- Channel: 복호한 앞 channel slice로 다음 slice를 예측.
- Spatial: Morton even anchor를 먼저 복호하고 odd score를 예측.
- Full: Mean+Channel+Spatial.

부모는 Linear rank56 residual/Morton ON 50k이고, 모든 arm과 factorized control이 같은 부모에서 joint 50k를 더 학습했다. Context train419615/eval419616, factorized control train420267/eval420718이다.

| λ | 모델 | 전체 KiB | PSNR | Factorized control 대비 크기 |
| --- | --- | ---: | ---: | ---: |
| .0064 | Factorized +50k | 90.805 | 24.3024 | 기준 |
| .0064 | Mean | 89.823 | 24.3179 | −1.08% |
| .0064 | Mean+Channel | 88.249 | 24.3151 | −2.81% |
| .0064 | Mean+Spatial | 86.073 | 24.3222 | −5.21% |
| .0064 | Full | **84.905** | 24.3232 | **−6.50%** |
| .0256 | Factorized +50k | 42.258 | 23.4875 | 기준 |
| .0256 | Mean | 41.608 | 23.4950 | −1.54% |
| .0256 | Mean+Channel | 41.492 | 23.4943 | −1.81% |
| .0256 | Mean+Spatial | **39.172** | 23.5304 | **−7.30%** |
| .0256 | Full | 39.233 | **23.5352** | −7.16% |

결론은 Mean의 작은 이득, Spatial의 큰 이득, Full의 유효성이다. `.0256`에서 Spatial과 Full은 매우 가까운 RD trade-off라 Full이 모든 지표에서 우세하다고 할 수는 없었다.

### 4.3 Nonlinear32 factorized continuation과 Spatial/Full 결합

Nonlinear factorized control은 같은 초기 Nonlinear 50k 부모에서 50k를 더 학습했다. Rank56 train420303/eval420304, rank80 train420382/eval420383이다.

| λ | Transform / context | 전체 KiB | PSNR |
| --- | --- | ---: | ---: |
| .0064 | Linear factorized 100k | 90.805 | 24.3024 |
| .0064 | Nonlinear32 factorized 100k | 87.840 | 24.3022 |
| .0256 | Linear factorized 100k | 42.258 | 23.4875 |
| .0256 | Nonlinear32 factorized 100k | 41.256 | 23.5210 |

Nonlinear32 rank56은 Linear보다 `.0064`에서 2.965 KiB 작고 PSNR은 거의 같았으며, `.0256`에서는 1.002 KiB 작고 +0.0335 dB였다.

그 다음 같은 Nonlinear32 부모에서 Spatial/Full을 50k joint 학습했다. Train420916/eval420919, run20260910_092000이다.

| λ | 모델 | 전체 KiB | PSNR | SSIM | LPIPS | Nonlinear factorized 대비 크기 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| .0064 | Mean+Spatial P0 | 83.633 | 24.3244 | .7612 | .2371 | −4.79% |
| .0064 | Full P0 | **82.491** | 24.3238 | .7611 | .2371 | **−6.09%** |
| .0256 | Mean+Spatial P0 | 38.236 | **23.5867** | .7348 | .2678 | −7.32% |
| .0256 | Full P0 | **37.958** | 23.5707 | .7342 | .2683 | **−7.99%** |

`.0064`에서는 Full이 Spatial보다 1.142 KiB 작고 PSNR 차이는 −0.0006 dB였다. `.0256`에서는 Full이 0.278 KiB 작지만 PSNR −0.0160 dB였다. 하나의 codec 구조로 두 operating point를 비교하기 위해 이후 probability 실험은 두 λ 모두 Full P0를 사용했다.

### 4.4 Spatial predictor P0/P1/P2, 추가 25k

가설은 기존 odd-score linear kernel3 predictor가 약하거나 Morton 이웃 범위가 좁다는 것이었다.

- P0: 부모의 linear kernel3 predictor.
- P1: P0 + zero-init residual correction, kernel3/hidden32.
- P2: P0 + zero-init residual correction, kernel7/hidden32.
- 모두 부모100k에서 새 Adam, 추가25k, 상수 LR1e-4, scope=all.

첫6개 train422181/eval422182, `.0256 Full` 추가3개 train422220/eval422221이다.

| λ / Context | P0 KiB / PSNR | P1 KiB / PSNR | P2 KiB / PSNR | P2−P0 |
| --- | --- | --- | --- | --- |
| .0064 Full | 79.166 / 24.1634 | 79.195 / 24.1858 | 79.055 / 24.1591 | −113.71 B, −0.0043 dB |
| .0256 Spatial | 37.446 / 23.4764 | 37.465 / 23.4515 | 37.323 / 23.4566 | −125.38 B, −0.0198 dB |
| .0256 Full | 37.348 / 23.4284 | 37.313 / 23.4286 | 37.149 / 23.4556 | −203.24 B, +0.0272 dB |

`.0256 Full P2`는 같은 P0보다 작고 세 화질 지표도 좋아 긍정적이지만, 두 λ에 걸쳐 일관되지 않았다. 더 중요한 혼합 요인은 P0 자체도 부모에서 크게 이동했다는 점이다.

| 부모 → P0 25k | Δbytes | ΔPSNR |
| --- | ---: | ---: |
| .0064 Full | −3,405.15 | −0.1604 dB |
| .0256 Spatial | −809.26 | −0.1103 dB |
| .0256 Full | −624.84 | −0.1423 dB |

즉 추가25k와 LR을 1e-6에서 1e-4로 재시작하면서 representation/RD 지점이 이동한 효과가 predictor 차이보다 컸다. 이 결과만으로 P1/P2 확대를 주력 구조로 채택하지 않았다.

### 4.5 Probability-only 4종 × 2λ

목적은 P 실험의 representation 이동을 제거하고 **같은 quantized score와 복원값을 얼마나 짧게 부호화할 수 있는지**만 비교하는 것이었다.

- 부모: 원래 Nonlinear32 Full P0, run20260910_092000 task8/9, 누적100k.
- 추가10k, micro2×acc4, 1e-4→7k에서1e-5.
- geometry projection, transform, score scale, mean offset/step, channel/spatial value predictor, residual codec, entropy median/quantile을 고정.
- Quantile update를 0으로 꺼서 복원 lattice 이동도 막음.
- Train422558/eval422559, run20260913_112316, full TEST 6,991 scene.

#### 4.5.1 네 arm이 공유하는 symbol 생성과 복호 순서

네 arm 모두 Full P0의 **value path**를 그대로 쓴다. Rank56 score를 `16/16/16/8`의 네 channel group으로 나누고, group은 앞에서부터 순차적으로 복호한다. 각 group 안에서는 Morton 순서의 4,096 token을 even 2,048개와 odd 2,048개로 나눠 두 pass만 사용한다.

group `g`에서 원래 normalized score를 `S_g`, mean 및 앞서 복호한 channel group으로 만든 base를 `b_g`, scene/channel별 quantization step을 `d_g`라고 하면:

```text
even 입력:       V_e = (S_e - b_e) / d_g
even 복원:       S_hat_e = b_e + d_g * Q(V_e; fixed median)

anchor_delta:    even 위치에는 S_hat_e - b_e, odd 위치에는 0
odd base:        b_odd = b_g,odd + P_linear_k3(anchor_delta)_odd
odd 입력:        V_o = (S_o - b_odd) / d_g
odd 복원:        S_hat_o = b_odd + d_g * Q(V_o; fixed median)
```

여기서 `P_linear_k3`, `b_g`, `d_g`, score normalization `q`, analysis/synthesis transform은 네 arm에서 전부 고정이다. 네 arm이 바꾸는 것은 `V_o`의 likelihood와 arithmetic-coder CDF뿐이다. Gaussian 계열도 기존 even EntropyBottleneck의 고정 median을 중심으로 사용하므로 최종 round lattice를 유지한다.

송신기와 수신기의 순서는 아래와 같다.

```text
group0 even → group0 odd → group1 even → group1 odd
→ group2 even → group2 odd → group3 even → group3 odd
```

따라서 어떤 arm도 token 4,096개를 하나씩 복호하는 autoregressive codec이 아니다. 네 arm 모두 entropy string은 8개다. `SCCTX001` score container는 24 B prefix와 string별 8 B length를 가져 wrapper가 88 B이고, 이는 `actual_score_bytes`에 이미 포함된다. Split/Gaussian/Conditional이 per-scene side stream을 추가하지 않는다. 필요한 entropy state, scale 및 predictor weight는 공유 checkpoint 쪽에 있다.

Bitstream flags에는 Shared/Split/Gaussian/Conditional 종류가 기록된다. 수신 codec 설정과 flag, rank, slice width, stream 수가 다르면 복호를 거부한다.

#### 4.5.2 Probability-only 학습에서 고정되는 것

`feature_codec_train_scope=score_probability`를 설정하면 우선 codec 전체를 freeze하고 다음 probability parameter만 다시 연다.

| 구성 | 학습 여부 |
| --- | --- |
| Geometry projection, Nonlinear analysis/synthesis MLP와 basis | 고정 |
| Score normalization `q` | 고정 |
| Mean conditioner의 offset 및 quantization step | 고정 |
| Channel predictor, P0 spatial **value predictor** | 고정 |
| Residual codec 전체 | 고정 |
| EntropyBottleneck quantiles/median | 고정 |
| Active EntropyBottleneck density parameters | 학습 |
| Gaussian per-channel base scale | Gaussian/Conditional에서 학습 |
| Anchor-conditioned scale predictor | Conditional에서만 학습 |

기존 deterministic quantile update는 optimizer와 별개로 median을 움직일 수 있으므로 `quantile_update_interval=0`도 함께 사용했다. 부호화를 위한 integer CDF buffer는 학습된 density/scale에서 다시 만들지만, quantile/median은 갱신하지 않는다.

Render, consistency 및 RD loss 계산 경로 자체는 유지되지만, 복원에 관여하는 parameter가 모두 frozen이므로 optimizer로 전달되는 유효 gradient는 probability rate 쪽뿐이다. 한 optimizer step 뒤 eval reconstruction이 bit-exact하게 유지되는 테스트도 네 arm에 대해 수행했다.

로그상 학습 parameter 수는 Shared 약3.2K, Split 약6.5K, Static Gaussian 약3.3K, Conditional Scale 약10.7K다. 모두 추가10k, Adam weight decay0, LR1e-4에서 시작해 step7k에 1e-5, seed111123 조건이다.

#### 4.5.3 Shared

각 channel group에 CompressAI `EntropyBottleneck` 하나가 있다. 이는 채널마다 유연한 1-D marginal CDF를 학습하지만, 같은 채널의 scene/token 위치에는 같은 분포를 쓰는 factorized model이다.

```text
even V_e ─┐
          ├─ 같은 group EntropyBottleneck/CDF
odd  V_o ─┘
```

Even과 odd는 별도 rANS string으로 보내지만 **확률분포 parameter는 공유**한다. 부모 Full P0가 원래 사용하던 구조와 weight를 그대로 로드한 뒤 density parameter만 10k 더 fit하므로, 추가학습 효과를 측정하는 control arm이다.

#### 4.5.4 Split

Even은 부모의 group EntropyBottleneck을 그대로 쓰고, odd마다 독립 EntropyBottleneck을 하나 더 둔다.

```text
even V_e → EB_even[g]
odd  V_o → EB_odd[g]
```

초기화할 때 각 `EB_even[g]`의 density parameter와 고정 quantile을 `EB_odd[g]`에 복제한다. 저장된 CDF buffer 자체는 복사하지 않고 density와 quantile에서 재생성한다. 따라서 step0에서는 Shared 부모와 같은 probability와 reconstruction으로 시작하고, 학습 후에만 even과 odd marginal이 갈라진다.

새 decode pass나 새 entropy string을 추가하는 것이 아니다. Full은 원래 even/odd string이 분리돼 있으므로 같은 8개 string에 서로 다른 CDF를 적용할 뿐이다. 늘어나는 것은 공유 model의 odd EntropyBottleneck parameter/state다.

#### 4.5.5 Static Gaussian

Even은 Shared/Split과 같은 `EB_even[g]`를 사용한다. Odd의 유연한 EntropyBottleneck 대신 `GaussianConditional`을 사용한다.

```text
mu[g,c]    = 고정된 EB_even[g] median
sigma[g,c] = softplus(raw_scale[g,c]) + 0.11
p(V_o)     = Gaussian(mu[g,c], sigma[g,c])
```

`sigma`는 group/channel마다 하나라 scene, token 위치, anchor 값과 무관하다. 총56개 base scale이 있고 모두 1.0에서 시작한다. 실제 coding index는 `[0.11, 256]` 구간의 64단계 logarithmic scale table에서 만든다.

이 arm은 **복원 lattice만 부모와 같고 초기 probability density는 부모와 같지 않다.** 부모의 flexible odd marginal을 고정-mean unit Gaussian으로 교체했기 때문이다. 따라서 결과 악화는 Gaussian family만의 순수한 불가능성뿐 아니라 분포 reset, 고정 mean, 10k 적응 및 finite scale table 조건을 함께 반영한다.

#### 4.5.6 Conditional Scale

Even과 Gaussian 중심은 Static Gaussian과 같다. 차이는 decoded even anchor로 odd 위치마다 Gaussian scale을 바꾸는 작은 network다.

```text
anchor_delta
  → Conv2d(width → 32, kernel 1×3, padding 1)
  → GELU
  → Conv2d(32 → width, kernel 1×1)
  → local_log_multiplier

sigma_base  = softplus(raw_scale) + 0.11
sigma_local = clamp(
    sigma_base * exp(2 * tanh(local_log_multiplier / 2)),
    0.11, 256
)
```

마지막 1×1 conv의 weight/bias는 0-init이므로 초기 `local_log_multiplier=0`, 즉 `sigma_local=sigma_base`다. 따라서 Conditional은 Static Gaussian과 같은 초기 probability model에서 시작한다. 이후 multiplier는 base scale의 약 `1/7.39×–7.39×` 범위에서 위치별로 조절된다.

입력 `anchor_delta`에는 복호 완료한 even 값만 있고 아직 모르는 odd 값은 0이다. 현재 odd symbol이나 원본 feature를 참조하지 않으므로 수신기가 동일 scale을 계산할 수 있다. Odd 위치의 미복호 값을 바꿔도 예측 scale이 변하지 않는 causality 테스트가 있다. Scale map은 전송하지 않는다.

#### 4.5.7 네 구현의 차이 요약

| Arm | Even probability | Odd probability | Odd 중심 | Odd scale/context | Step0 parent probability와 동일 | Stream 수 |
| --- | --- | --- | --- | --- | --- | ---: |
| Shared | EB per group | 같은 EB 공유 | EB median | 위치 무관 learned density | 예 | 8 |
| Split | EB_even | 독립 EB_odd | 각 EB median | 위치 무관 independent density | 예, even→odd 복제 | 8 |
| Static Gaussian | EB_even | GaussianConditional | 고정 EB_even median | group/channel static scale | 아니오; scale1 Gaussian으로 reset | 8 |
| Conditional Scale | EB_even | GaussianConditional | 고정 EB_even median | decoded even anchor→위치별 scale | 아니오; Static Gaussian과 동일 | 8 |

| λ | Probability model | 전체 KiB | Score B | PSNR | Shared 대비 전체 크기 |
| --- | --- | ---: | ---: | ---: | ---: |
| .0064 | Shared | 82.463 | 70,924.906 | 24.3238 | 기준 |
| .0064 | Split | 81.174 | 69,604.342 | 24.3238 | **−1.56%** |
| .0064 | Static Gaussian | 88.167 | 76,765.617 | 24.3238 | +6.92% |
| .0064 | Anchor-conditional Gaussian scale | **81.135** | 69,565.374 | 24.3238 | −1.61% |
| .0256 | Shared | 37.952 | 31,149.344 | 23.5707 | 기준 |
| .0256 | Split | **36.920** | **30,093.048** | 23.5707 | **−2.72%** |
| .0256 | Static Gaussian | 44.518 | 37,873.270 | 23.5707 | +17.30% |
| .0256 | Anchor-conditional Gaussian scale | 37.021 | 30,195.839 | 23.5707 | −2.45% |

모든 arm의 residual y/z, mean, container와 집계 화질은 로그 정밀도 내 동일하다. Shared 자체는 원래 부모 대비 28.87 B/6.24 B만 줄었으므로 Split 이득은 단순 추가10k 효과로 설명되지 않는다.

Conditional은 `.0064`에서 Split보다 38.97 B 작지만 `.0256`에서는 102.79 B 크다. 수신 feature 복원 시간도 Split 89.6/81.0 ms에 비해 Conditional 126.2/120.1 ms였다. 그래서 두 λ 공통 기본 모델은 더 단순하고 일관된 Split으로 정했다.

Static Gaussian 실패는 현재의 고정 median, unit-scale 초기화, 10k schedule과 finite scale table을 포함한 결과다. Gaussian 계열 또는 conditional probability 전체가 불가능하다는 증거가 아니다.

### 4.6 Full low-LR 10k

목적은 원래 Full P0의 RD 지점을 보존한 채 전체 codec을 LR1e-6으로 조금 더 joint 적응시키는 것이었다. P0와 같은 부모에 zero-init한 새 P2도 함께 비교했다.

- 부모: Probability-only와 같은 원래 Nonlinear32 Full P0 100k.
- 추가10k, scope=all, 상수 LR1e-6, micro2×acc4.
- Train array422943; `.0064/P0`만 재실행423009; eval422944.

| λ | Predictor | 전체 KiB | PSNR | 원래 Full 부모 대비 |
| --- | --- | ---: | ---: | ---: |
| .0064 | P0 | 최종 checkpoint 없음 | — | 최초5k/재실행10k 저장 시 모두 `Errno 28` |
| .0064 | 새 P2 | 82.430 | 24.3256 | −62.762 B, −0.074% |
| .0256 | P0 | 37.932 | 23.5701 | −26.238 B, −0.068% |
| .0256 | 새 P2 | 37.921 | 23.5716 | −37.624 B, −0.097% |

`.0256`의 P2−P0는 −11.386 B/scene, −0.029%와 +0.0015 dB뿐이다. 기존 Split보다 low-LR P2가 `.0064`에서 1.548%, `.0256`에서 2.711% 크다. 추가 계산 대비 효과가 0.1% 미만이므로 `.0064/P0`를 다시 실행하지 않기로 했다.

### 4.7 고정-symbol entropy probe

질문은 “score가 전체 bit의 약80%이므로 score 확률 모델이 크게 나쁜가?”였다. 전체 bit share만으로는 판단할 수 없어서, 현재 Full+Split checkpoint의 symbol과 복원을 고정하고 별도 학습 없이 probability table만 TRAIN에서 fit해 held-out TEST에 적용했다.

- 부모: 현재 Full P0+Split 두 checkpoint.
- 128 unique TRAIN scene으로 table fit, 별도 128 unique TEST scene 평가.
- deterministic-all C12/T8, augmentation OFF.
- 모든 후보가 native integer streams와 decoded feature tensors를 exact 재현.
- Marginal: score channel/even-odd/slice별 histogram; residual y는 scale-table index별, z는 channel별.
- Context: score-even에 scene quantization-step bin, score-odd에 decoded neighbor-anchor magnitude bin을 추가. Residual은 marginal 그대로라 score/residual 비교는 비대칭이다.
- Slurm423055, 두 `.err` 모두 비어 있음.

| λ | Native total | Marginal total / 절감 | Score-context total / 절감 | Context 절감 95% CI |
| --- | ---: | ---: | ---: | ---: |
| .0064 | 81.253 KiB | 81.160 / 0.115% | **80.654 / 0.736%** | [0.656%, 0.821%] |
| .0256 | 36.923 KiB | 36.859 / 0.172% | **36.642 / 0.759%** | [0.673%, 0.855%] |

Context 절감 중 score 기여는 원래 total의 0.713/0.738 percentage point, residual marginal 기여는 0.024/0.021 point였다. 이 후보군에서는 score 쪽에 더 큰 probability 개선 여지가 확인됐다. 그러나 score에만 richer context를 적용했으므로 score가 본질적으로 residual보다 “압축이 안 된다”고 결론낼 수는 없다.

Native model NLL과 실제 coder의 차이:

| λ | Score actual−NLL | Residual actual−NLL | rANS 자체 gap 합계 |
| --- | ---: | ---: | ---: |
| .0064 | 120.45 B | 38.62 B | 약60 B/scene |
| .0256 | 76.87 B | 24.58 B | 약60 B/scene |

따라서 arithmetic coder/finite-CDF 구현은 주어진 probability model을 약99.6–99.8% 효율로 따르고 있다. stream merge로 얻을 수 있는 것은 수십 B 수준이며 주 병목이 아니다. 단순 decoder-known score context에서 약0.75% 절감이 나온 것이 실제 남은 기회다.

Score 절감의 대부분은 초반 group에 집중됐다. `.0064`에서 g0-odd/g0-even/g1-odd가 score 절감의 91.4%, `.0256`에서는 같은 세 stream이 82.3%였다. 모든 odd stream은 각각 67.3%/72.7%를 담당했다.

이 probe는 full TEST나 새 codec 학습 결과가 아니다. 128-scene screen이고 rendering/PSNR은 재실행하지 않았으며, table 배포 비용도 최종 production bitstream에 통합되지 않았다.

## 5. 지금까지 확정된 것과 미확정인 것

### 확정에 가까운 관측

- Residual ON과 Morton ON은 현재 계열에서 중요하다.
- Rank80 증가는 현재 schedule에서 일관된 이득이 없었다.
- Mean보다 Spatial decoder context의 이득이 크고, Full context는 두 λ 공통 구조로 사용할 수 있다.
- Context 이득은 동일 부모·동일 추가50k factorized control과 비교해도 남는다.
- Nonlinear32 transform은 rank56 factorized에서 Linear보다 유리했다.
- P1/P2 value predictor 확대의 순수 효과는 P0 추가학습 효과보다 작고 λ 간 일관성이 약하다.
- 복원 고정 Split probability는 두 λ 모두 Shared보다 작다.
- 전체 모델 low-LR 10k는 0.1% 미만의 개선이라 우선순위가 낮다.
- 현재 coder overhead는 수십 B 수준이며, 단순 score context probability table로 약0.75%의 held-out 절감 여지가 있다.

### 아직 실험으로 답하지 못한 것

- 두 점이 아닌 여러 λ의 actual RD 곡선과 BD-rate.
- single-seed의 수십 B/0.02 dB 차이가 반복되는지.
- Split entropy를 처음부터 transform과 joint 학습했을 때의 결과.
- Split보다 강한 non-Gaussian conditional density의 실효성.
- score와 residual에 대칭적인 context model을 적용한 probability redundancy 비교.
- decoded low-rank score를 residual entropy model에 conditioning했을 때의 이득.
- Score/Residual 각각이 화질에 기여하는 양을 bit와 함께 측정한 partial-reconstruction 결과.
- Score hyperprior/coarse-to-fine latent 같은 구조적 교체가 side bits와 복호 지연을 상쇄하는지.
- 공유 model/table 배포 비용을 포함한 single-scene 또는 짧은 sequence의 총 rate.

## 6. 다음 세션이 기억할 판단 원칙

- 완료된 full eval 48개, historical Linear baseline 12개, entropy probe 2개를 재실행하지 않는다.
- 구조의 효과는 같은 부모·같은 추가 step의 matched control로 판단한다.
- 전체 크기 감소와 PSNR 하락을 분리하지 말고 실제 RD trade-off로 본다.
- Score가 total rate의 약80–84%라는 사실은 최적화 leverage가 크다는 뜻이지, 그 확률 모델이 같은 비율만큼 나쁘다는 뜻은 아니다.
- 현재 실증된 추가 score probability gap은 단순 context 기준 전체의 약0.75%다.
- 다음 새 실험의 기준 checkpoint는 §1의 Full P0+Split 두 개다.

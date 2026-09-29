# Hyper1D 코덱 설계 — score path 없는 feature 직접 부호화

작성: 2026-09-28. **상태: 구현 완료, CPU 코덱·통합 검증. GPU 학습은 실행 전.**
구현·12시간 실행 방법: [NFCGS_HYPER1D_12H.md](NFCGS_HYPER1D_12H.md).
갱신: 2026-09-28 — 구현 검토 반영: 호출부/checkpoint 통합 지점(§4.2),
`h_a` 입력 부호(§4.1), strides 자체 블록화(§4.1·§10), 가변 길이(§9),
A 전환 신호 지표(§7.2), DC 운반 서술 정정(§6·§12-4).

이 문서는 모델 설계만 다룬다. 학습 스케줄, λ 선정, 평가 프로토콜, 실행 계획은
범위 밖이며 별도 문서로 다룬다.

## 1. 한 줄 요약

현행 main 코덱(`Nonlinear32 / rank56 / Full P0 / Split`)에서 **score path 전체를
제거**하고, 736-D observable feature를 **multiscale 1-D mean-scale hyperprior
하나로 직접 부호화**한다.

```text
현행:  X → centering(μ̂) → low-rank score(context 8-stream) + residual hyperprior
신규:  X → 1-D hyperprior (y, z 두 stream)
```

기존 `ObservableLowRank1DCodec`은 **수정하지 않는다.** 신규 클래스를 병렬로
추가하고 model YAML로 선택한다. 기준선 보존 원칙을 따른다.

## 2. 확정 설계 결정

| 항목 | 결정 | 비고 |
| --- | --- | --- |
| scene mean `μ̂` | **없음** — centering도 전송도 하지 않음 | DC는 y/z가 담는다 |
| 입력 정규화 | **B: 없음 (기본)** — raw feature가 바로 코덱으로 | A(캘리브레이션 buffer)를 폴백으로 내장 (§7) |
| Morton 정렬 | **OFF** — 단 flag로만 끄고 경로는 보존 | `use_morton=False` (§8) |
| geometry projection | **유지** — 512→224, concat 736 | 실험 17에서 사실상 무손실 (−0.0628 dB) |
| downsampling | **4× 유지** — stride (2, 2), N=192, M=320 | config로 노출, 추후 변경 대상 (§12) |
| scene 컨테이너 | **신규 magic `E2EH0401`** | 기존 `E2EM0301`과 혼용 불가 (§9) |
| 학습 scope | `all` 단일 | `score_probability`에 해당하는 대상이 없음 |

## 3. 데이터 경로

```text
[송신기]
encoder → Z_A [B,4096,512], Z_G [B,4096,512]
  → O_G = Z_G · P_G                    P_G [512,224], bias-free, QR 출력보존 초기화
  → X = [Z_A | O_G]                    [B,4096,736], 정렬 없음(slot 순서 그대로)
  → rearrange                          [B,736,1,4096]
  → (X − f_mean) / f_std               B에서는 항등 (§7)
  → Adapter_a → g_a → y                [B,320,1,1024]
  → h_a(y) → z                         [B,192,1,256]   ★ signed — |y| 아님 (§4.1)
  → EB(z) 부호화/복호 → ẑ
  → h_s(ẑ) → (σ_y, μ_y)                각 [B,320,1,1024]
  → GaussianConditional(y; σ_y, μ_y) 부호화

[수신기]
z 복호 → ẑ → h_s → (σ_y, μ_y) → y 복호 → ŷ
  → g_s → Adapter_s → × f_std + f_mean
  → X̂ [B,4096,736]
  → Ẑ_A = X̂[:, :, 0:512],  Ô_G = X̂[:, :, 512:736]
  → Gaussian decoder (frozen, token-wise)
```

- entropy stream은 **z, y 두 개.** 직렬 복호 단계 1개(z 먼저 — y의 μ/σ가 ẑ에서 나옴).
- 양자화는 장면당 **2회**(z, y). 현행 11회(μ̂ FP16 1 + score 8 + z + y)에서 감소.
- 수신기는 slot 순서 그대로 소비한다. Gaussian decoder가 token-wise이므로
  순서 정보가 필요 없다(현행과 동일한 논리).
- **`positions` 입력이 기본 경로에서 사라진다.** Morton OFF이므로
  `decoded_token_centers` 호출이 불필요하다.

## 4. 모듈 명세

### 4.1 신규 클래스

```text
globalsplat/compression/hyper1d.py

class FeatureHyperprior1DCodec(nn.Module):
    geometry_projection : nn.Linear(512, 224, bias=False)
    f_mean, f_std       : register_buffer [1, 736, 1, 1]   # 항상 등록, B=항등 (§7)
    analysis_adapter    : MultiScaleAdapter1D(736, 96)      # 기존 모듈 재사용
    synthesis_adapter   : MultiScaleAdapter1D(736, 96)
    g_a  : Conv(736→192, k5, s2) → GDN → Conv(192→320, k5, s2)
    g_s  : Deconv(320→192, k5, s2) → IGDN → Deconv(192→736, k5, s2)
    h_a  : Conv(320→192, k3, s1) → LReLU → Conv(192→192, k5, s2) → LReLU
           → Conv(192→192, k5, s2)
    h_s  : Deconv(192→320, k5, s2) → LReLU → Deconv(320→640, k5, s2) → chunk(2)
    entropy_bottleneck   : EntropyBottleneck(192)           # z
    gaussian_conditional : GaussianConditional(None)        # y, scale table 0.11–256 ×64
```

**conv 스택은 `hyper1d.py`에 자체 선언한다 (합성 아님).** 이유 둘:

1. `ResidualHyperprior1D`는 stride가 (2, 2)로 **하드코딩**되어 있고 인자가 없다.
   `residual.py`는 main 코덱이 쓰므로 수정 금지 — 합성으로는 `strides` config를
   지원할 수 없다.
2. **`h_a`의 입력을 signed `y`로 바꾼다** (`|y|` 아님). 기존 블록의 `h_a(|y|)`는
   scale-only hyperprior(Ballé 2018)의 관행이 mean-scale 모델에 그대로 남은
   것이다. 평균 `μ_y`까지 예측하는 이 모델에서는 부호 정보를 유지하며, CompressAI의
   `MeanScaleHyperprior`도 `h_a(y)`를 쓴다(`ScaleHyperprior`만 `|y|`).
   기존 residual 블록은 checkpoint 재구성 산물이라 그 관행을 보존한 것일 뿐,
   신규 모델이 따라갈 이유가 없다.

`MultiScaleAdapter1D`만 import해 재사용한다(수정 없음).

### 4.2 공개 인터페이스 — main 코덱과 호환

호출부가 실제로 쓰는 시그니처를 전부 맞춘다. 현재 호출부는
`project_geometry()`와 `forward(..., restore_original_order=False)` kwarg를
사용하므로 이 둘이 빠지면 즉시 TypeError/AttributeError다.

```text
project_geometry(geometry) → observable           # 호출부가 직접 사용
forward(texture, geometry, positions=None, *,
        restore_original_order=True, training=None) → CodecOutput
compress(texture, geometry, positions=None) → CompressedScene
decompress(data: bytes) → (texture_hat, geometry_observable_hat)
update(force=False, update_quantiles=False) → bool
aux_loss() → Tensor
set_trainable_scope(scope="all")            # "all" 외에는 명시적 ValueError
config: Hyper1DConfig                       # .to_dict() 제공 (checkpoint 저장용)
```

- `positions`, `restore_original_order`는 받되 `use_morton=False`이면 **무시**
  (정렬이 없으니 복원할 순서도 없음). 호출부가 분기 없이 기존처럼 넘겨도 된다.
- `CodecOutput.likelihoods` 키는 `{"y": …, "z": …}`.
  `estimated_bits`는 기존 dataclass가 dict를 합산하므로 그대로 동작한다.
- score-context용 FP32 autocast 경계는 **불필요**(해당 모듈이 없음).
  entropy 연산 자체는 CompressAI 관행대로 FP32 유지.

### 4.2.1 시그니처만으로 해결되지 않는 통합 지점

인터페이스를 맞춰도 아래 네 곳은 **코드 분기/가드가 필요하다.**

| 위치 | 현재 동작 | 필요한 변경 |
| --- | --- | --- |
| `model_wrapper.on_save_checkpoint` | `codec.score_context.mean_offset_enabled`를 **직접 접근** → 신규 코덱이면 AttributeError | `getattr`/`hasattr` 가드 또는 codec 종류별 metadata 저장 |
| eval 바이트 집계 (`test_step`) | `SceneBitstream.unpack(data).bytes_by_stream` — `E2EM0301` 전용 | 컨테이너 종류별 unpack 분기 (`E2EH0401`은 y/z/container 분해) |
| `test.score_context_diagnostics` | `feature_codec.score_context.*` 직접 접근 | 신규 코덱에서는 옵션 자체를 명시적 거부 |
| checkpoint 로드 (`checkpoint.py`) | `CodecConfig` metadata/shape 추론이 score path 전제 | `Hyper1DConfig`용 로더 분기 (magic처럼 config 종류 필드로 구분) |

### 4.3 forward 의사코드

```text
X   = concat(texture, geometry_projection(geometry))      # [B,4096,736]
Xn  = (rearrange(X) − f_mean) / f_std                     # [B,736,1,4096]
y   = g_a(Adapter_a(Xn))
z   = h_a(y)                            # signed — mean-scale 모델이므로 |y| 아님
ẑ, L_z = EB(z, training)
σ_y, μ_y = h_s(ẑ).chunk(2)                                # y shape로 crop
ŷ, L_y = GC(y, σ_y, means=μ_y, training)
X̂n  = Adapter_s(g_s(ŷ))[..., :points]                     # 실제 입력 길이
X̂   = inverse_rearrange(X̂n × f_std + f_mean)
return split(X̂), likelihoods={"y": L_y, "z": L_z}
```

## 5. 제거되는 것 — 현행 대비

| 현행 구성요소 | 신규 |
| --- | --- |
| `shared_basis`, `shared_synthesis_basis` (rank 56) | 없음 |
| `analysis_mlp`, `synthesis_mlp` (Nonlinear32) | 없음 |
| `score_log_scale` (`q_S`) | 없음 |
| `score_entropy` (legacy EB(56)) | 없음 |
| `ContextualScoreEntropy` 전체 — mean conditioner(`b`, `d`), channel predictor ×3, spatial predictor ×4, group/odd EB 8개 | 없음 |
| `SCCTX001` 컨테이너 (24 + 8×8 = 88 B) | 없음 |
| scene mean `μ̂` (FP16 1,472 B) + STE | 없음 |
| Morton 정렬 실행 | flag OFF (§8) |
| `residual_mean/std` (PCA artifact 산출) | `f_mean/f_std`로 개명·재정의 (§7) |
| PCA/QR artifact 중 basis·score scale·residual 통계 | **QR geometry 초기화만 재사용** |
| scope `score_probability` | `all` 단일 |

residual hyperprior 블록(Adapter, g/h, EB, GC)만 살아남고, 나머지 학습
파라미터는 geometry projection뿐이다.

## 6. 이 설계에서 hyperprior가 떠안는 역할

현행에서 score path가 하던 일이 어디로 가는지 명시한다.

| 현행 담당 | 신규 담당 |
| --- | --- |
| scene mean (DC) — 전송 1,472 B | `y`에 분산 수용 + `ẑ → μ_y`의 **지역** 예측. z는 전역 서술자가 아니다 — h_a는 conv라 z 한 위치가 y 15칸(토큰 환산 약 70–85개)만 본다 |
| low-rank 계수 (저주파 구조) | `y` |
| score context (`b`, `d`, spatial) | `ẑ → (μ_y, σ_y)` — hyperprior 자체가 원소별 조건부 모델 |
| Split (even/odd CDF 분리) | 대응물 없음 |

즉 "context가 사라진다"기보다 **context의 형태가 decoder-causal 예측에서
side-latent 조건부로 바뀐다.**

## 7. 입력 정규화 — B 기본, A 내장 폴백

### 7.1 메커니즘: buffer는 항상 존재, B는 "항등 A"

- `f_mean = zeros`, `f_std = ones`로 **모드와 무관하게 항상 등록**한다.
- forward는 분기 없이 항상 `(X − f_mean)/f_std`를 적용한다. B에서는 수학적 no-op.
- 효과:
  1. **A/B의 `state_dict` 키 집합이 동일** — strict-load 호환, 전환 시 마이그레이션 없음.
  2. B→A 전환 = 캘리브레이션 산출물 로드 + `input_norm` config 한 줄.
  3. forward 수치 경로가 하나 — 검증도 하나로 끝난다.

```text
Hyper1DConfig.input_norm : "none" (기본) | "calibrated"
```

- `"none"`인데 buffer가 항등이 아니면 로드 시 **명시적 오류**(조용한 불일치 방지).
- bitstream에는 norm 모드 flag를 **두지 않는다.** A/B는 weight 자체가 다른
  checkpoint이므로 flag로 식별할 수 없고, 기존 원칙(“flag는 설정을 식별하지
  weight를 식별하지 않는다 — 송수신은 동일 checkpoint”)에 그대로 기댄다.

### 7.2 A 폴백 명세

- `scripts/calibrate_hyper1d_stats.py` (신규):
  frozen encoder로 TRAIN 고정 장면 집합(예: 256 unique)을 통과시켜
  **미정렬** 736-D concat의 채널별 mean/std를 Welford로 집계.
  `std`는 `clamp_min(1e-6)`. 산출물에 장면 ID 목록과 encoder checkpoint
  SHA-256을 기록한다(재현성).
- 근거: 일반 픽셀 LIC는 입력 범위가 포맷상 보장되어([0,1]) 이런 buffer가 없지만,
  여기 입력은 채널별 스케일이 임의인 학습 feature다. B는 그 조건화를
  GDN과 첫 conv 학습에 맡기는 선택이고, A는 안전장치다.
- 전환 판단 신호(설계상 정의만; 판정 절차는 실험 문서 몫):
  1. loss 발산/NaN, grad-norm이 clip(0.5)에 지속 포화
  2. **σ_y가 scale table 상한(256)에 포화되는 비율** — table의 0.11–256은
     σ_y의 범위이지 `|y|`의 상한이 아니다
  3. `|y − μ_y|` 통계 (양자화 잔차의 크기)
  4. estimated bits(likelihood)와 actual bits(rANS)의 괴리 — table 포화 시
     실제 비트가 추정을 크게 초과한다
- 중간 단계 B′(LR warmup만 추가)를 A보다 먼저 시도할 수 있게 recipe 수준
  옵션으로 남긴다.

### 7.3 B에서 주의할 구조적 지점

`MultiScaleAdapter1D`는 skip connection(`x + out_proj(…)`)이 있어 **raw 스케일이
skip을 타고 g_a 첫 conv까지 그대로 전달된다.** GELU 포화와 GDN 초기 안정성이
여기서 결정되므로, adapter 출력과 `y`의 채널별 통계를 로깅 대상으로 지정해 둔다.

## 8. Morton — OFF이되 경로 보존

- `use_morton: bool = False`. `morton.py`는 삭제하지 않는다.
- OFF일 때: 정렬/역정렬 없음, `positions` 무시, `decoded_token_centers` 호출 없음.
  slot 순서는 결정적이므로 송수신 정합성 문제 없음.
- ON일 때(추후): 현행과 동일하게 송신측 정렬, 수신측은 정렬 순서 소비,
  permutation 비전송.
- bitstream flags bit0에 morton 여부를 기록하고 decompress에서 codec 설정과
  대조해 **불일치 시 거부**한다. permutation 전송용이 아니라 “이 bitstream이
  어느 통계로 학습된 checkpoint의 것인가” 가드다.

**설계 리스크로 명시:** 실험 16 기준 미정렬 토큰의 lag-1 상관은 0.0055로
지역 구조가 거의 없다. stride-2 conv는 이웃 토큰을 병합하므로, Morton OFF에서는
**무관한 토큰을 병합**하게 된다(§12-1). 이 상호작용 때문에 flag를 보존한다.

## 9. Bitstream — `E2EH0401`

```text
header  : struct ">8s H H I I Q 32s"  = 60 B
  magic          8 B   b"E2EH0401"
  version        2 B   1
  flags          2 B   bit0 = morton, bit1 = scene-mean 포함(예약, 현재 0)
  points         4 B   실제 입력 token 수 (고정 4096 아님 — 가변 길이 지원)
  channels       4 B   736
  payload_len    8 B
  sha256        32 B   header(자기 자신 0 채움)+payload 해시 — E2EM0301 관행 유지

payload : M3HPRN01 { z strings, y strings }   # 기존 ResidualBitstream 재사용, 52 B
```

- 장면당 고정 오버헤드: **60 + 52 = 112 B**
  (현행 80 + 52 + 88 + 1,472 = 1,692 B → 93% 감소).
- 컨테이너 하나 = 장면 하나(batch 1). 기존 `E2EM0301` 관행 유지.
- decompress 거부 규칙:
  1. magic/version 불일치, channels가 codec config와 불일치
  2. flags가 codec 설정(`use_morton` 등)과 불일치
  3. **payload의 y/z shape가 `points`와 strides로 계산한 기대값과 불일치**
     — `M3HPRN01`이 shape를 자체 기록하므로 header의 `points`와 교차 검증한다
     (예: strides (2,2)이면 y 길이 = ceil(ceil(points/2)/2))
  4. checksum 불일치, payload 잔여 바이트
- 복원은 `g_s` 출력에서 `[..., :points]` crop — 비4배수/홀수 길이 지원의
  근거가 payload shape가 아니라 header `points`임을 명시한다.
- `E2EM0301` 계열과 상호 복호 불가가 **의도된 동작**이다. magic을 분리한 이유.

## 10. Config

```text
@dataclass(frozen=True)
class Hyper1DConfig:
    codec_type: str = "hyper1d"  # checkpoint 종류 구분; 신규 metadata에 필수
    texture_channels: int = 512
    geometry_channels: int = 512
    geometry_observable_channels: int = 224
    n: int = 192                 # z 채널
    m: int = 320                 # y 채널
    adapter_hidden: int = 96     # 3으로 나눠떨어져야 함
    strides: tuple = (2, 2)      # g_a stride 시퀀스 — 추후 변경 축
    input_norm: str = "none"     # "none" | "calibrated"
    use_morton: bool = False
    morton_bits: int = 10        # use_morton=True일 때만 사용

검증(__post_init__):
  - 각 치수 양의 정수, adapter_hidden % 3 == 0
  - input_norm ∈ {"none", "calibrated"}
  - strides ∈ {(2, 2), (2, 1)} — conv 두 단계는 유지하고 stride만 바꾼다.
    4× = (2, 2), 2× = (2, 1). synthesis는 역순 미러이며
    `_deconv1d_as_2d`의 output_padding=(0, stride−1)이 stride 1을 자연 지원
  - observable = texture + geometry_observable = 736 확인
```

YAML:
- `config/model/globalsplat_hyper1d.yaml` — 위 기본값, `freeze_globalsplat: true`,
  `feature_codec_train_scope: all`.
- 기존 `globalsplat_nfcgs_rank56.yaml`은 무변경.

`strides`를 config로 둔 이유: 4× 유지가 “일단”이며(§12-2), 변경이 재작성이
아니라 설정 교체가 되도록.

## 11. 설계 불변식 (검증 항목)

1. **round-trip**: `decompress(compress(·))`가 forward(eval)의 복원 텐서와 일치.
   홀수/비4배수 token 수 포함 — g_s 출력은 `[..., :points]` crop 관행 유지.
2. **B no-op 등가**: 항등 buffer 경로 == buffer 없는 참조 계산.
3. **A/B 호환**: 두 모드의 `state_dict` 키 집합 동일.
4. **거부**: §9의 모든 거부 규칙이 실제로 발화.
5. **likelihoods**: `{"y","z"}` 두 키만 존재, `estimated_bits`가 그 합.
6. **main 경로 불간섭**: `ObservableLowRank1DCodec`의 기존 테스트가 전부
   무변경 통과(아카이브 oracle 대조 포함).

## 12. 설계 리스크 — 모델 관점

1. **Morton OFF × stride-2 병합 (최대 리스크).** 미정렬 이웃은 지역 선형
   상관이 거의 없어(lag-1 Pearson 0.0055 — 단, 선형 상관이 낮다는 것이지
   통계적 독립의 증명은 아니다) downsampling이 무관한 토큰을 섞고 g_s가
   되풀어야 한다. 완화 수단이 둘 다 config 한 줄이다: `use_morton=True` 또는
   `strides=(2,1)`.
2. **latent 용량 41% 감소.** 현행 score(56×4096) + y(320×1024) = 557,056 vs
   신규 y(320×1024) = 327,680. λ가 조절하는 rate와 별개로, 표현 병목이 될 수
   있다. 대응 축: `strides=(2,1)`(y 655,360) 또는 `m=640`(동일).
3. **B의 raw 스케일.** §7.3. 폴백 사다리 내장으로 대응.
4. **DC 운반.** scene mean이 없으므로 장면 DC는 `y`(직접 수용)와
   `ẑ → μ_y`(지역 예측)가 나눠 진다. **z는 전역 서술자가 아니다** — h_a는
   conv 스택이라 z 한 위치의 수용 영역이 y 15칸(토큰 환산 약 70–85개)에
   불과하다. 즉 "장면 하나의 DC"를 한 번만 부호화하는 경로는 없으며,
   위치마다 지역적으로 반복 지불한다. y/z bytes 비중이 비정상적으로 크면
   이 반복 지불이 원인 후보다. flags bit1은 향후 scene mean 확장을 위한
   예약 비트이며, 현재 decoder는 해당 비트를 거부한다. 복원 arm에는 별도 구현이 필요하다.
5. **Split·Context 상실.** 측정된 이득(context −6.1~8.0%, Split −1.56~2.72%)을
   포기한다. 이 설계의 목적은 이기는 것이 아니라 **score path 없는 하한 기준선**을
   세우는 것이다.
6. **귀인(attribution) 한계.** scene mean·Morton·score path·정규화를 동시에
   제거하므로, 결과의 변화를 "score path 제거 효과"로 단독 해석할 수 없다.
   분리하려면 요소별 복원 arm이 필요하며, 이는 실험 설계 문서의 몫이다.
   현재 구현은 Morton·입력 정규화·stride를 개별 설정할 수 있다. scene mean과
   score path 복원은 후속 구현이 필요한 비교 arm이며, 현재 config로 복원할 수 없다.

## 13. 범위 밖

학습 recipe, λ 선정, 스모크/판정 절차, 평가 프로토콜, 기준선과의 비교 설계는
이 문서에서 다루지 않는다. 구현 착수 시 별도 실험 문서로 작성한다.

## 14. 참고

- 현행 구조·수치: [NFCGS_CODEC.md](NFCGS_CODEC.md),
  [SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md](SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md)
- Morton·순서 민감도 근거: 실험 15/15b/16
  ([ABLATION_PROGRESS_2026-09-02.md](ABLATION_PROGRESS_2026-09-02.md))
- 세대 1 선행 유사 설정: `multiscale1d_screen`, `multiscale1d_morton1d`
  (workspace `experiments/` — 단, 대상이 full feature가 아니라 residual이었음)

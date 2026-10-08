# Token 정렬·context 선택 설계와 구현

작성: 2026-10-08. 이 세션에서 검토한 E82·E83 설계를 다음 세션에서도 재사용하기 위한 기록이다.
사용자가 제공한 원문은 [E82·E83 스크리닝 자료](E82_E83_SORT_CONTEXT_SCREEN_2026-10-07.md)에 보존했다.
원문의 실험은 별도 프로젝트의 결과이며 이 저장소에서 측정한 성능으로 취급하지 않는다.

## 1. 목적과 유지할 구조

- 정렬과 Score context를 독립적으로 선택해 Morton/기존 설정으로 즉시 돌아갈 수 있게 한다.
- GlobalSplat → token Compressor → Gaussian decoder의 구조, MSH residual 구조를 유지한다.
- Score와 MSH는 항상 같은 token 순열을 사용한다. 정렬 변경 실험에서는 MSH도 새 이웃 통계에 맞춰 학습한다.
- 새로운 정렬을 기본값으로 바꾸지 않는다. 기존 YAML·checkpoint는 `token_order=morton`, `score_context_schedule=legacy`로 해석한다.
- 현재 decoder는 정렬된 token을 그대로 사용하므로 순열을 bitstream에 추가 전송하지 않는다.
- 신규 실험은 pretrained GlobalSplat을 고정하고 codec 전체를 학습하는 기존 minimal 설정을 기본 recipe로 사용한다.

## 2. 구현한 독립 선택 옵션

### 정렬: `--token-order`

| 값 | 방식 | 참고 |
|---|---|---|
| `morton` | 기존 3D Morton 안정 정렬 | 기본값, 기존 동작 유지 |
| `hilbert` | 동일한 대표 위치·scene min/max·10bit 격자의 3D Hilbert 정렬 | 처음 비교할 정렬 |
| `nn_xyz` | 대표 3D 위치의 L2 greedy 최근접 경로 | encoder의 O(N²) 메모리·순차 경로 생성 비용 |
| `nn_score` | 양자화 전 score의 가중 L1 greedy 최근접 경로 | context/양자화 이후 심볼을 사용하지 않음 |

Hilbert는 Skilling axes-to-transpose 변환을 사용하고 동률은 원래 slot 순서로 처리한다.
희소한 실제 token에서는 Hilbert 순서상 이웃이 항상 인접 3D 셀인 것은 아니다.
greedy 경로는 Morton key가 가장 작은 token부터 시작하고 거리 동률은 원래 slot 인덱스로 결정한다.
거리 계산과 경로 생성은 CPU에서 수행해 token마다 CUDA `.item()`으로 동기화하지 않는다.

`nn_score`는 양자화 전 normalized score `s`를 scene-conditioned step `d`로 나눈 `u=s/d`를 사용한다.
`distance(i,j)=sum_c |u[i,c]-u[j,c]| / b[c]`다.

- 기본 `b`: 해당 scene의 양자화 전 `mean(abs(u))`, 최소 .05. TRAIN 고정 통계를 따로 제공하지 않아도 실행되게 한 기본값이다.
- 원문에 맞춘 TRAIN 고정 `b`: `--score-order-scale train_b.json`으로 rank56의 양수 56개 배열을 제공한다.
  이 값은 `feature_codec_config.score_order_scale`에 저장한다. test scene에서 통계 fit을 하지 않는다.
- 정렬만 no-grad로 결정한다. 정렬 후 analysis, Score context, MSH에는 정상적으로 gradient가 흐른다.
- 학습과 encoder에서 매 입력마다 같은 규칙으로 계산한다. 학습 중 바뀌는 analysis에 오래된 순열 캐시를 적용하지 않는다.
- centering이 켜진 값 정렬은 정렬 전에 전송할 FP16 mean을 한 번 계산해 정렬·부호화에서 재사용한다.

### Context: `--score-context-schedule`

| 값 | 순서 | 비율 | 예측기 |
|---|---|---|---|
| `legacy`, stages2 | `0::2 → 1::2` | 1/2, 1/2 | 기존 masked convolution |
| `legacy`, stages3 | `0::4 → 2::4 → 1::2` | 1/4, 1/4, 1/2 | 기존 masked convolution |
| `legacy`, stages4 | `0::4 → 2::4 → 1::4 → 3::4` | 1/4씩 | 기존 masked convolution |
| `quarter2` | `0::4 → 나머지` | 1/4, 3/4 | 좌우 anchor 직접 gather, 나머지 위치별 선형 가중치 |
| `dyadic4` | `0::8 → 4::8 → 2::4 → 1::2` | 1/8, 1/8, 1/4, 1/2 | 각 단계 ±4/±2/±1 anchor 직접 gather |

`quarter2`와 `dyadic4`는 자료의 방식이며 기존 4단계와 구분한다.
CLI는 quarter2의 stages2, dyadic4의 stages4를 자동 선택한다. 다른 stage 수를 명시하면 거부한다.
새 schedule은 channel context를 끈다. centering/mean/score norm/nonlinear 여부는 기존 score-path 선택과 독립이다.

Anchor predictor는 기존 channel group마다 `W_left`, `W_right`, bias를 둔다.
quarter2는 `i mod 4=1/2/3`마다 별도 가중치를 사용한다.
오른쪽 anchor가 없는 끝에서는 왼쪽 anchor를 복제하고 학습 가능한 edge bias로 경계 여부를 표시한다.
현재·미래 단계의 원래 score는 입력하지 않는다. 이전 단계의 양자화 복원값만 사용한다.
모든 channel group을 한 단계에서 복호한 다음 다음 token 단계로 넘어간다.
확률 CDF는 기존처럼 stage/group별 factorized model이며, context는 예측 base를 바꾼다.

kernel 옵션은 `legacy`의 convolution에 적용한다. 새 anchor schedule은 gather를 사용해 kernel 값으로 참조 범위가 제한되지 않는다.
호환성용 기존 convolution tensor는 보존하지만 새 schedule에서는 동결하고 사용하지 않는다.
기존 2단계 kernel5에서 ±2는 가려진 odd다. ±3 even을 추가 참조하려면 새로 지원하는 kernel7을 사용한다.

## 3. 기본값·checkpoint·bitstream

- 옵션이 없으면 기존 Morton + 기존 2단계 kernel3/noise/Full 설정이다.
- 기존 minimal/3/4단계 스크립트도 계속 사용할 수 있다.
- 설정은 checkpoint의 `feature_codec_config`에 저장한다. standalone receiver는 저장 설정으로 모델을 만든다.
- 새 schedule 및 kernel7은 Score bitstream의 별도 flag로 구분한다. 서로 다른 context 설정으로 복호하면 거부한다.
- 정렬은 encoder 전용이므로 별도 bitstream flag·순열 stream을 추가하지 않는다. encoder 재현성을 위해 checkpoint 설정에는 저장한다.
- 기존 checkpoint에서 order/context를 바꾸는 학습은 `--allow-score-path-conversion`을 명시한 weights-only warm start다.
  resume와 eval에서 다른 설정으로 강제 변환하지 않는다.
- 새 anchor predictor는 0으로 초기화하고 신규 단계 entropy model은 대응하는 기존 even/odd model로 초기화한다.
  GlobalSplat·MSH·basis 등 context 밖의 tensor는 그대로 복사한다.
- 다른 schedule의 학습된 anchor가 이미 있는 checkpoint는 shape가 맞으면 유지한다.

## 4. 실행: 옵션을 선택한 2 GPU 학습 + 전체 scene eval

새 스크립트는 single-node normal GPU 2개, batch/GPU2, accumulation2, 유효배치8을 사용한다.
`batch_ugrad`, `--exclude=ariel-v6,ariel-v7`이며 현재 normal GPU 후보는 v8~v12다.
50k optimizer steps, LR1e-4, 기존 35k/45k decay, seed111123, minimal, STE, codec scope all을 사용한다.
학습 성공 후 `afterok` dependency로 1 GPU 실제 bitstream 전체 scene eval을 제출한다.
eval은 C12/T8, test-time optimization 없이 기존 프로토콜을 사용한다.

```bash
# 저장소 루트에서 실행. 제출 helper는 선택하지 않은 값에 기본값을 적용한다.
# Morton + 기존 3단계 대조군
TOKEN_ORDER=morton CONTEXT_SCHEDULE=legacy STAGES=3 \
  bash scripts/slurm/submit_nfcgs_order_context.sh

# Hilbert + 동일한 기존 3단계: 정렬 효과
TOKEN_ORDER=hilbert CONTEXT_SCHEDULE=legacy STAGES=3 \
  bash scripts/slurm/submit_nfcgs_order_context.sh

# Morton + 자료의 계층형 4단계: context 효과
TOKEN_ORDER=morton CONTEXT_SCHEDULE=dyadic4 \
  bash scripts/slurm/submit_nfcgs_order_context.sh

# Morton + 1/4 anchor, 2단계
TOKEN_ORDER=morton CONTEXT_SCHEDULE=quarter2 \
  bash scripts/slurm/submit_nfcgs_order_context.sh

# 조합 가능: Hilbert + 계층형 4단계
TOKEN_ORDER=hilbert CONTEXT_SCHEDULE=dyadic4 \
  bash scripts/slurm/submit_nfcgs_order_context.sh

# 값 기반 경로. TRAIN b를 주려면 SCORE_ORDER_SCALE=/absolute/train_b.json 추가.
TOKEN_ORDER=nn_score CONTEXT_SCHEDULE=legacy STAGES=3 \
  bash scripts/slurm/submit_nfcgs_order_context.sh
```

`SOURCE_CHECKPOINT=/absolute/parent.ckpt`로 공통 부모를 지정할 수 있다.
`getcwd: cannot access parent directories`가 뜨면 현재 shell이 삭제되거나 교체된 디렉터리를 가리킬 수 있다.
먼저 `cd /` 후 실제 저장소의 절대 경로로 `cd -P` 하고 재시도한다.
제출 helper는 저장소 경로를 물리적 절대 경로로 정규화하며, 경로나 checkpoint 확인 실패 시 sbatch 호출 전에 종료한다.
기본 checkpoint가 실제로 이동했다면 동일한 FullSplit 부모의 현재 경로를 SOURCE_CHECKPOINT로 명시한다.
기본 부모는 기존 FullSplit `lambda0p0256/step000010000.ckpt`(λ=.0064를 지정하면 해당 λ 부모)다.
`MAX_STEPS`, `CHECKPOINT_EVERY`, `RATE_LAMBDA`, `KERNEL`도 환경변수로 선택한다.
train job 이름은 `gs-order-context`, eval은 `gs-order-context-eval`이다.
출력은 `outputs/nfcgs_order_context/<order>/<schedule>_stages<N>_k<K>_ste/lambda<tag>/job_<trainID>`다.
eval 결과는 해당 run의 `eval_all`에 저장하며 stage summary에는 order/schedule/scale도 기록한다.

기존 recipe에 옵션만 추가해서도 사용할 수 있다.

```bash
python scripts/run_nfcgs.py train --checkpoint parent.ckpt \
  --score-path minimal --allow-score-path-conversion \
  --token-order hilbert --score-context-schedule dyadic4 \
  --devices 2 --batch-size 2 --accumulate 2 --dry-run

# 새 기능 OFF: Morton + legacy. 3/4단계 유지 여부는 --score-spatial-stages로 지정.
# 별도 설정을 모두 생략하면 원래 Full/2단계로 돌아간다.
```

## 5. 실험 순서와 판단 기준

1. Morton+legacy3 ↔ Hilbert+legacy3: 정렬만 변경.
2. Morton+legacy4 ↔ Morton+dyadic4: 단계별 비율·참조 관계만 변경.
3. Morton+legacy2 ↔ Morton+quarter2: 두 단계 안에서 anchor 비율 비교.
4. 같은 context에서 nn_xyz/nn_score로 encoder 비용 대비 이득 확인.
5. 좋은 정렬과 좋은 context를 결합.

동일 부모·λ·학습량·유효배치·양자화로 비교하며 대조군도 같은 예산으로 학습한다.
전체 bytes, Score bytes, MSH y/z bytes, 단계별 bytes, PSNR/SSIM/LPIPS와 encoder/decoder 시간을 함께 확인한다.
Score의 절감만으로 전체 개선을 판단하지 않는다.
원문의 2048-token/고정 PCA/TRAIN55·TEST12/추정 entropy 결과를 현재 4096-token/closed-loop 코덱 성능으로 대입하지 않는다.

## 6. 보존한 후속 설계: 이번 구현 범위 밖

- 값 공간 3D PCA → Hilbert 근사 정렬: TRAIN 고정 투영을 저장하고 greedy O(N²) 비용을 줄이는 후보. 성능 미측정.
- 병렬 copy context: odd마다 decoded even ±1/±3/±5/±7 중 선택, 선택 번호 stream 추가.
  잔차와 번호 비트를 합쳐 선택하고, 4096 token에서 균등 3bit 번호 비용은 대략768B/scene다.
  원문 −18.7%는 순차 copy8의 결과이며 이 2단계 방식의 결과가 아니다.
- 토큰별 AR3: 4096단계 복원 지연과 closed-loop 학습/teacher-forcing 불일치 때문에 우선순위 낮음.
- Static slot graph: 원래 slot 정체성을 쓰므로 현재 정렬+MSH 순서와 결합하는 별도 설계가 필요함.

원문 수식을 구현할 때의 주의: 정수 x에 대해 `round(x-pred)=x-round(pred)`는 ties-to-even의 정확한 .5 경계에서 항상 성립하지 않는다.
고정 정수 심볼의 무손실 예측이라면 `p=round(pred); r=x-p; x=r+p`로 정의해야 한다.
이번 구현은 기존 closed-loop 잔차 양자화·복원 격자를 유지하며 이 무손실 수식으로 교체하지 않는다.

이번에 성능 실험을 실행하거나 서버에 job을 제출하지 않았다. CPU에서 causal context·실제 entropy round trip·checkpoint·기존 경로 호환성을 검증한다.

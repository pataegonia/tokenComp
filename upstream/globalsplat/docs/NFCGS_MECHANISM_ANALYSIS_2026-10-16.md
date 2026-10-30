# 정렬 → 실제 causal 예측 → 실제 bit 분석

사용자가 제안한 세 분석을 λ=.0128, 동일 RE10K test scene 256개에서 수행한다. 재학습 없이 현재 rank56/minimal checkpoint 6개를 사용한다. 이 문서는 실행 계획과 구현 설명이며, 아직 서버 GPU/RE10K 측정 결과는 없다. Projection 없는 2-path의 우열을 검증하는 실험과 구분한다.

## 비교를 분리하는 이유

1. **고정 Score의 순서 변경**: Hilbert legacy3 checkpoint에서 추출한 동일한 원래 slot 순서 Score와 대표 XYZ를 유지하고 none/Morton/Hilbert/NN-score/random 순서를 적용한다. 다른 checkpoint의 학습된 basis/geometry projection 차이가 섞이지 않는다. 이 단계에서는 actual bytes나 RD 우열을 주장하지 않는다.
2. **학습된 모델 동작 비교**: none/Morton/Hilbert/NN-score의 legacy3 checkpoint, Hilbert의 quarter2/dyadic4 checkpoint를 동일 영상·카메라 입력으로 분석한다. 이 비교에는 학습된 Score/basis/geometry projection/확률모델 차이가 포함된다. 1번과 함께 해석한다.
3. **같은 모델의 예측 우회 진단**: 학습된 가중치·확률모델·MSH를 고정한 채 공간 prediction만 0으로 우회해 재부호화하고 복호한다. 같은 scene에서 Score와 MSH의 byte 변화 및 특징 복원 오차를 측정한다. 추론 시 분포가 바뀌는 진단이며, 현재 추가 학습하려는 Sorting OFF + Context OFF 대조군을 대체하지 않는다.

기준 Score는 실제 양자화 입력 단위로 추출한다. 현재 minimal에서는 Score norm/mean-conditioned step이 OFF이므로 추가 scale은 1이다. BF16 실행에서 고정 Score 실험의 NN은 **그 동일한 추출 Score**로 순서를 만든다. 실제 NN checkpoint가 정렬 전에 별도 FP32 analysis를 수행하는 경로와 미세하게 다를 수 있다. 학습된 모델 분석은 checkpoint의 실제 정렬 경로를 그대로 실행한다.

## 측정 항목과 해석 보완

### 고정 Score 관계

- Sequence lag 1/2/4/8의 채널 STD 정규화 L1/MSE, 채널별 Pearson correlation과 유효 채널 수.
- 3D 거리의 원래 좌표 값 및 동일 scene XYZ cloud의 RMS radius로 나눈 거리. Scene마다 다른 좌표 scale의 영향을 구분한다.
- Legacy3/quarter2/dyadic4의 각 단계 reference topology에서 대상−참조 사이의 동일 지표. 단순 인접 token의 상관에만 의존하지 않는다.
- Random 순열은 scene ID와 seed에서 결정하고 같은 값을 재배치한다. 학습된 random-order 모델이나 실제 bit 결과로 해석하지 않는다.

보고용 채널 STD는 원래 순서의 scene Score 전체에서 한 번 계산하고 모든 순서에 동일하게 적용한다. 상수 채널은 거리 scale을 1e-6으로 제한하며 Pearson에서는 제외한다. NN 순서 생성에는 실제 구현의 mean-absolute weighting(floor .05)을 사용한다. 두 정규화는 목적이 다르며, 보고용 STD는 coding/context 입력에 사용하지 않는다.

### 실제 causal context

| Schedule | 단계와 실제 공간 입력 |
|---|---|
| legacy3/kernel5 | A=0 mod4, 예측 없음. B=2 mod4는 복원된 0 mod4의 ±2 위치. C+D=홀수는 복원된 짝수의 ±1 위치. 다른 convolution tap의 미래 위치와 경계 padding은 0 |
| quarter2 | 0 mod4 anchor, 나머지 1/2/3 mod4는 좌우 4 간격 anchor. Phase별 학습 가중치 사용 |
| dyadic4 | 0 mod8 → 4 mod8 → 2 mod4 → 홀수. 좌우 anchor 간격은 각각8/4/2이며, 대상에서의 거리는 ±4/±2/±1 |

Quarter2/dyadic4에서 오른쪽 anchor가 없으면 왼쪽을 복제하고 edge bias를 적용한다. Legacy convolution은 zero padding으로 처리하므로 경계 동작이 다르다. Trace는 pair predictor의 두 입력 slot을 모두 보존하며, 복제된 slot이 독립된 두 이웃을 의미하지는 않는다.

실제 `_staged` encoder가 사용한 target, baseline, prediction, entropy input, 복원값, 앞 단계 canvas, reference 위치/유효성 및 predictor tap의 L1 가중치 크기를 관측한다. 원본 anchor를 넣거나 별도 근사 predictor로 대체하지 않는다. Raw trace의 invalid reference 입력은 실제 canvas/padding과 같은 0이다. 미래 token은 context에 넣지 않는다.

Predictor는 channel group 안에서 채널을 혼합한다. 같은 채널끼리의 상관만으로 예측 가능성을 모두 설명할 수 없으므로 실제 예측 잔차와 likelihood/bytes를 함께 해석한다.

256개 scene 전부에서 단계/group별 다음 값을 저장한다.

- 채널별 정규화 residual MAE/MSE, prediction 없는 baseline MSE 대비 비율.
- 대상−실제 복원 anchor 입력의 Score 거리·상관·3D 거리, 경계 token 수.
- 채널별 양자화 복원 MSE, prediction correlation, 양자화 단위 residual MAE.
- 실제 raw entropy bytes, token 수, symbol 수, bits/token, bits/symbol.
- 학습된 entropy likelihood의 예상 bits와 실제 raw bits의 차이. 차이를 모두 header overhead로 해석하지 않는다. 정수 CDF, rANS 종료·escape, 작은 배열의 padding 등이 영향을 줄 수 있다.

첫 단계에는 공간 참조가 없어서 anchor 관계 값이 없다. 단계별 target 수는 전체를 정확히 분할한다. 4096 token에서 legacy3는1024/1024/2048, quarter2는1024/3072, dyadic4는512/512/1024/2048이다.

### 실제 용량과 복원

`8 × stage raw bytes / stage target tokens`는 그 단계 **56채널 전체의 token당 비용**이다. 별도로 symbol당 비용을 기록한다. 단계 raw bytes 합이 Score entropy string 합과 같은지 매 scene 검사한다.

전체 bitstream을 Score raw, MSH y, MSH z, mean, 모든 header로 분해하며 합이 전체 byte와 같은지 검사한다. Score 자체의 header를 MSH/container header와 중복해서 세지 않는다. 실제 causal decoder로 재복호한 Score가 관측된 encoder 복원값과 완전히 같은지도 검사한다.

CDF는 모델을 분석 device로 옮긴 뒤 `update_quantiles=False`로 재생성해 기존 실제 bitstream eval과 같은 위치에서 구성한다. 학습된 quantile/median을 다시 학습하거나 조정하지 않는다.

복원 token feature MSE와 채널 STD 정규화 MSE를 기록한다. 이는 rendered PSNR이 아니다. Prediction이 바뀌면 양자화 격자와 MSH 입력 residual도 달라지므로 residual MAE 감소만으로 전체 RD 이득을 단정하지 않는다.

## 장면과 checkpoint

다음 6개 최종50k checkpoint는 기존 완료 eval의 경로·metadata에서 선택했다.

| 비교 | 설정 |
|---|---|
| 정렬, legacy3 고정 | context_only .0128, Morton legacy3 .0128, Hilbert legacy3 .0128, NN-score legacy3 .0128 |
| Context, Hilbert 고정 | Hilbert legacy3 .0128, quarter2 .0128, dyadic4 .0128 |

같은 Hilbert legacy3를 공유하므로 총6개다. [측정 manifest](../../../artifacts/nfcgs_mechanism_manifest_2026-10-16.json)는 기존 복잡도 manifest를 수정하지 않고 별도로 생성했다. checkpoint의 step/config를 strict 검증하고, frozen encoder/Gaussian decoder state hash가 모두 같은지 검사한다. 학습된 codec의 basis/geometry projection 차이는 허용하며 분석상 구분한다.

현재의 첫 processable test scene 256개를 중복 없이 선택한다. 각 scene은 **context 이미지12장**을 뜻한다. 영상·카메라 tensor를 CPU cache로 저장해 모든 모델에 동일하게 재사용하고 fingerprint를 기록한다. 무작위 데이터셋 표본은 아니므로 전체 scene으로 일반화할 때는 추후 표본 선택을 바꾸거나 전체 평가로 확인한다.

기본은 모든 scene의 요약 지표와 **앞8개 scene의 완전한 stage tensor trace**를 저장한다. 모든256개의 raw tensor가 필요하면 `MECHANISM_TRACE_SCENES=256`으로 제출한다. Cache된 입력과 seed를 보존하므로 추가 trace를 얻기 위해 재학습할 필요는 없다. trace 범위 변경은 기존 결과와 섞지 않도록 새 출력 폴더를 사용한다.

## 서버 반영

아래 파일을 같은 상대 경로에 반영한다. Manifest만 서버 `logs/slurm/`에 넣는다.

| 로컬 파일 | 서버 경로/용도 |
|---|---|
| [score_context.py](../globalsplat/compression/score_context.py) | `globalsplat/compression/score_context.py`; 기본 OFF인 stage observer 추가, checkpoint tensor key 변경 없음 |
| [mechanism.py](../globalsplat/compression/mechanism.py) | `globalsplat/compression/mechanism.py`; 실제 참조·잔차·byte 진단 |
| [analyze_nfcgs_mechanism.py](../scripts/analyze_nfcgs_mechanism.py) | `scripts/analyze_nfcgs_mechanism.py`; 모델/scene 비교·중단 후 재개 |
| [profile_nfcgs_complexity.py](../scripts/profile_nfcgs_complexity.py) | `scripts/profile_nfcgs_complexity.py`; strict 모델 로드와 입력 cache 공통 함수. 이미 올렸다면 재사용 |
| [report_nfcgs_mechanism.py](../scripts/report_nfcgs_mechanism.py) | `scripts/report_nfcgs_mechanism.py`; JSON 요약·PNG/PDF 세 그림·REPORT.md |
| [analyze_nfcgs_mechanism.slurm](../scripts/slurm/analyze_nfcgs_mechanism.slurm) | `scripts/slurm/analyze_nfcgs_mechanism.slurm`; GPU1/CPU8/RAM48G, 시간 상한8시간 |
| [nfcgs_mechanism_manifest_2026-10-16.json](../../../artifacts/nfcgs_mechanism_manifest_2026-10-16.json) | **`logs/slurm/nfcgs_mechanism_manifest_2026-10-16.json`** |

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
bash scripts/slurm/analyze_nfcgs_mechanism.slurm plan
sbatch scripts/slurm/analyze_nfcgs_mechanism.slurm
```

Plan은6개 checkpoint의 실제 존재 여부와 manifest를 검사한다. 하나라도 없으면 경로를 출력하고 실패한다. 분석은 완료 checkpoint만 사용하므로 새 both_off 학습에 dependency를 걸지 않는다. 기본적으로 가중치를 고정한 prediction bypass 진단도 수행하며, 생략하려면 `MECHANISM_PREDICTION_ABLATION=0`을 지정한다. GPU1의8시간은 요청하는 시간 상한이며 측정된 실행시간 예상이 아니다. 이 작업의 시간은 많은 진단 CPU 복사·likelihood 재계산을 포함하므로 codec latency benchmark로 사용하지 않는다.

## 산출물과 재개

`outputs/nfcgs_mechanism/job_<JOB_ID>/`에 metadata, 입력 cache/fingerprint, backbone hash, 각 모델의 `points/<label>/scene_XXXX.json`, 샘플 `traces/`, 완료 기록을 저장한다. 완료 후 같은 job에서 `report_<JOB_ID>/`를 생성한다.

1. `01_fixed_score_relations.png/.pdf`: 같은 Score의 순서별 거리·상관 곡선, scene bootstrap 95% 구간.
2. `02_causal_context_and_bits.png/.pdf`: 실제 anchor 관계, 예측 MSE 비율, 단계별 실제 bits/token. 단계 번호가 같아도 schedule의 token 집합은 다르므로 REPORT.md의 단계별 token 수 표와 함께 해석한다.
3. `03_actual_byte_decomposition.png/.pdf`: Score/MSH/header의 실제 용량.
4. `summary.json`과 `REPORT.md`: 위 지표, 같은 scene에서 계산한 byte 차이의 bootstrap 구간, prediction bypass의 Score/MSH/복원 오차 변화.

Bootstrap은 256 scene을 단위로 수행하며 4096개 token을 독립 표본으로 세지 않는다. 이 구간은 학습 seed 간 변동을 포함하지 않는다. 파일 생성은 기존 결과를 덮어쓰지 않는다. 일부 scene까지 완료된 후 중단되면 동일 입력·프로토콜·checkpoint·관련 코드·소프트웨어 조건에서 다음처럼 재개한다.

```bash
MECHANISM_OUTPUT=/absolute/path/to/outputs/nfcgs_mechanism/job_<OLD_ID> \
MECHANISM_RESUME=1 sbatch scripts/slurm/analyze_nfcgs_mechanism.slurm
```

완료된 scene JSON과 trace는 그대로 재사용한다. 이전 job이 강제 종료되어 `.analysis-lock` 디렉터리가 남아 있다면 이전 job이 더 이상 실행 중이 아닌 것을 확인한 후 해당 빈 lock 디렉터리를 제거하고 재개한다. 캐시 입력·기존 결과는 유지한다. 서로 다른 code/protocol의 결과를 이어 붙이지 않는다.

다운로드할 때 그림과 요약만 필요하면 `report_<JOB_ID>/`를 받으면 된다. Scene별 추가 분석에는 `metadata.json`, `inputs.json`, `complete.json`, `points/`가 필요하다. 큰 `inputs.pt`와 `traces/`는 tensor 재검증에 사용할 때 받는다. 보고서만 다시 만들 때는 로컬에서도 다음 명령을 쓸 수 있다.

```bash
python scripts/report_nfcgs_mechanism.py --input /path/to/downloaded/run --output /path/to/new/report
```

## 결과를 본 뒤 우선할 추가 분석

- 고정 Score 분석의 기준을 NN-score legacy3 등 다른 checkpoint로 바꿔 반복한다. `analyze_nfcgs_mechanism.py --reference nn_score_0p0128`을 별도 출력 폴더와 함께 사용할 수 있다. 한 Hilbert-trained basis에만 의존한 결과인지 확인하는 목적이다.
- 경계/내부 token, quarter2의 phase1/2/3을 나눠 본다. Raw trace에 실제 인덱스·참조·경계 flag가 있어 모델 변경 없이 계산할 수 있다.
- 실제 3D 거리가 비슷한 anchor 쌍끼리 Score 거리와 예측 오차를 비교한다. 공간적으로 가까운 것 외에 부호화할 값의 유사성이 설명하는 부분을 확인한다.
- 첫 결과에서 차이가 큰 모델 쌍에 한해 scene별 예측 오차 감소와 Score/전체 bits 변화의 paired scatter를 추가하고, 평가 scene 수와 reference checkpoint를 늘린다. 상관을 인과 효과로 단정하지 않는다.

## 검증 범위

로컬 CPU 합성 입력에서 실제 entropy 부호화·복호를 사용해 검증한다. Diagnostic ON/OFF의 bitstream 완전 일치, state_dict 보존, 실제 복호 anchor 일치, convolution masking과 경계 복제, 단계 분할, byte 보존, model/scene pairing, 부분 재개, paired bootstrap, PNG/PDF 생성과 기존 출력 보존을 검사한다. 서버의 실제 checkpoint 로드·RE10K·CUDA/BF16 실행은 서버 제출 후 확인해야 한다.

분석/driver/보고서 테스트 12개, 기존 spatial-stage 테스트 9개, 기존 정렬/context ablation CPU 테스트 5개가 통과했다. CUDA 학습 테스트 1개는 CPU 환경이라 생략했다. 합성 tiny codec으로 그림 세 장을 생성하고 축/범례/잘림을 시각 확인했다. `artifacts/nfcgs_mechanism_SYNTHETIC_LAYOUT_QA_*`의 이미지는 레이아웃 검증용이며 실험 결과가 아니다.

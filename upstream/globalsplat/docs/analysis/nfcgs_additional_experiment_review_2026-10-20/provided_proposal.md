# 추가 실험 제안 — Scene-token 압축 논문

작성일: **2026년 10월 20일**. 추가 실험 시간이 확보된다는 사용자 설명을 반영했다. 마지막 상태 근거는 10월 20일 16:47 다운로드 로그다. 새 작업을 제출하거나 기존 작업을 변경한 기록은 아니다.

**추천은 신규 50k 학습 12개부터다.** 최종 D4에서 정렬×Context 반복 검증 8개, 구조별 실제 rate를 겹치게 하는 RD 보강 4개다. 이후 fresh-init 2×2와 anchor 통제 5개를 더하면 17개가 된다. 나머지는 앞선 결과에 따라 선택한다.

## 1. 현재 실험과 중복시키지 않을 것

최근 26학습 중 23개가 학습·평가 완료됐고 3개는 마지막 로그에서 학습 중이었다. 이 3개와 이미 연결된 eval은 새 학습 수에 넣지 않는다.

| 기존 진행 | 현재 추적 job | 연결 eval | 역할 |
|---|---:|---:|---|
| None+L3 OFF12 .0512 |458782|457098| L3 2×2의 마지막 RD 점 |
| H+D4 OFF16 .0128 |458783|457100| 기본 seed D4 matched OFF |
| Score736-only .0128 |457828|457829| full-width 구조 pilot |

추가 seed 111124/404의 H+D4 ON16/OFF16, 기존 OFF1/OFF4, T06의 13순서 byte 분석, T15 독립 수신은 완료 결과를 재사용한다. 기존 37개 profile 전체를 다시 돌리는 계획도 아니다.

이 문서의 새 ID **E**는 신규 학습, **N**은 평가·분석이다. 기존 T 번호와 같은 작업을 다른 번호로 중복 합산하지 않는다.

## 2. 우선 실행할 신규 12학습

| ID | 새 작업 | 신규 50k | 해결하는 질문 |
|---|---|---:|---|
| **E01** | D4 .0128 정렬×Context 2×2를 총 3 seed로 완성 |**7**| 핵심 상호작용이 후속 학습 랜덤성에서도 반복되는가 |
| **E02** | Morton+D4 OFF16, .0128, 기본 seed |**1**| 효과가 Hilbert 한 정렬에만 묶이는가 |
| **E03** | fresh 2-path와 MSH-only에 추가 λ 각 2개 |**4**| 실제 rate가 겹치는 구조 비교에서도 2-path가 유리한가 |
| | **합계** |**12**| |

E03의 4개는 최초 RD 확장 예산이다. 정확한 공통 rate 범위를 만들기 위한 짧은 탐색·수렴 연장은 별도이며, 한 번에 원하는 rate가 나온다고 보장하지 않는다.

### E01: D4의 3 seed 2×2 — 왜 7개인가

공간 예측 조건은 **D4·4단계·16/16/16/8채널·16 strings**, λ=.0128로 고정한다. MSH는 모든 셀에서 유지한다. None/Hilbert 순서와 공간 Context ON/OFF만 바꾼다.

| Seed | None ON16 | None OFF16 | Hilbert ON16 | Hilbert OFF16 | 신규 |
|---|---|---|---|---|---:|
| s0:111123/403 |기존 완료|**신규**|기존 완료|기존 진행 458783|**1**|
| s1:111124/404 |**신규**|**신규**|기존 완료|기존 완료|**2**|
| s2:새 model/data seed 쌍 |**신규**|**신규**|**신규**|**신규**|**4**|
| | | | | **합계** |**7**|

새 seed 쌍은 예를 들어 111125/405로 계획하되 기존 manifest와 중복을 확인한다. 모든 모델의 부모·50k·batch·view·손실 정규화·평가 조건을 맞춘다. 기본 OFF16의 full-state 재개와 fresh shuffle 이력도 provenance에 남긴다.

각 seed에서 다음을 함께 보고한다.

- 네 셀의 actual bytes·PSNR/SSIM/LPIPS.
- 정렬 전후 Context의 bytes·화질 변화.
- 같은 λ의 비례 rate 상호작용:

  I_R = log R(H,ON) − log R(H,OFF) − log R(None,ON) + log R(None,OFF)

I_R<0은 해당 λ에서 Hilbert 이후 Context의 비례 rate 절감이 더 크다는 뜻이다. PSNR 변화도 반드시 함께 표시하고 동일 화질 상호작용으로 부르지 않는다.

3 seed는 모든 초기화의 일반성을 증명하는 수가 아니다. 동일 FullSplit 부모에서 시작하면 **공통 부모 이후 후속 학습 랜덤성의 반복**이다. seed별 점과 평균·변동을 보여주며 scene bootstrap을 training-seed 신뢰구간으로 대체하지 않는다.

### E02: Morton OFF는 최종 D4에 먼저 맞춘다

기존 Morton+D4 ON16 .0128과 맞출 **Morton+D4 OFF16** 한 개를 학습한다.

E01의 기본 seed None OFF16과 기존 H OFF16까지 모으면, 중앙 λ에서 **None/Morton/Hilbert×ON/OFF 6셀**이 완성된다. Morton의 전체 RD 구간이나 3 seed 효과까지 확인한 것은 아니다.

과거 T17인 Morton+L3 OFF12와 다른 작업이다. 본문에서 L3의 Morton 효과를 직접 분석하려면 별도 E17을 선택한다.

### E03: 구조 비교는 같은 λ에서 실제 rate 중첩으로 이동

현재 fresh holdout128:

| 구조 | λ=.0128 PSNR | KiB |
|---|---:|---:|
| Score56-only |21.7889|65.53055|
| MSH-only |20.6696|14.48434|
| two-path56 |23.0067|56.73486|

MSH-only는 2-path보다 약 3.9배 작다. 이 한 점의 낮은 화질만으로 탈락시킬 수 없다.

| 대상 | 추가 λ 계획 | 신규 |
|---|---|---:|
| fresh two-path56 |.0032와 .0512를 우선 사용 |2|
| fresh MSH-only |먼저 .0032, 그 결과로 다음 λ 결정 |2|
| **합계** |기존 .0128 포함 각 3점 |**4**|

MSH의 두 번째 λ를 .0008 등으로 미리 확정하지 않는다. TRAIN에서 제외한 holdout128의 actual rate·품질·수렴을 보고 선택한다. λ를 낮춰도 필요한 화질에 도달하지 않으면 그 자체로 결과를 남긴다.

이 실험은 공개 backbone/QR 경계·codec fresh-init 계보다. 기존 warm-start main RD와 점을 이어 붙이지 않는다. 공통 geometry projection 정책·학습 budget·분리된 검증셋을 유지하고, 살아남은 구조에만 전체 6991 test를 붙인다. 공통 구간이 부족하거나 3점 보간이 불안정하면 BD 숫자보다 측정점·지배 관계를 우선 보고한다.

## 3. 설명력을 보강하는 추가 5개: 우선 12 → 17

| ID | 작업 | 신규 | 의미 |
|---|---|---:|---|
| **E04** | fresh 2-path D4 2×2의 나머지 3셀 |**3**| 특정 warm-start 부모에만 의존하는 결과인지 점검 |
| **E05** | Hilbert+D4의 bias-only / random-anchor 통제 |**2**| 이득이 실제 공간 이웃 정보에서 오는지 더 직접적으로 검토 |
| | **추가 합계** |**5**| E01~E05 총 17학습 |

### E04: 이미 학습한 fresh 2-path를 재사용

현재 T18 fresh two_path56은 Hilbert+D4 ON16이다. **그 초기화 파일·train holdout128·학습 레시피를 그대로 맞출 수 있을 때**, .0128에서 다음 3개만 추가한다.

- None+D4 ON16.
- None+D4 OFF16.
- Hilbert+D4 OFF16.

기존 fresh H ON을 재사용해 2×2를 완성한다. 정확히 같은 공통 초기화·학습 분할이 아니면 3개로 완성됐다고 쓰지 않고 네 셀을 다시 맞춘 비용을 별도 계산한다.

이는 같은 2-path에서 **학습 이력의 적용 범위**를 보는 실험이다. 다른 architecture에서도 원리가 성립한다는 주장은 E15의 별도 구조 이전 2×2가 필요하다.

### E05: 같은 정렬에서 anchor 정보의 질만 바꾸기

둘 다 Hilbert+D4·.0128·16 streams·MSH ON으로 재학습한다. 최종 평가에서 입력만 망가뜨리는 T06 민감도 진단과 다르다.

| 통제 | 정의 | 비교 목적 |
|---|---|---|
| bias-only predictor | 공간 anchor 가중치는 0으로 고정하고 단계·채널별 bias와 기존 entropy 모델은 학습 | 학습된 offset·분할만으로 얻는 효과와 anchor 정보의 효과 구분 |
| random decoded-anchor predictor | 원래 두 anchor 대신 **같은 채널그룹의 이미 복호된 적격 anchor**에서 결정론적으로 두 개 선택 | 예측기 용량·단계는 유지하고 공간 이웃의 질을 낮춤 |

Random anchor는 현재 token 단계보다 이전 단계에서 복호한 token만 참조한다. 현 단계 안의 참조를 새로 넣으면 병렬 복호 스케줄이 달라진다. 미래·현재 미복호 token·원본 feature·target을 읽으면 안 된다. 단계별 후보 집합과 반복 seed를 기록하고 수신측이 index/공유 규칙만으로 같은 참조를 생성하게 한다. 가능한 한 원래 참조의 단계·개수·분포 조건을 맞춘다.

Bias-only는 모델 용량도 줄어들 수 있으므로 그 하나만으로 용량 효과를 완전히 배제하지 않는다. random-anchor 통제와 함께 실제 bytes·PSNR·잔차·단계별 bits를 본다. 새 predictor 정책의 실제 packet-only 복호가 필요하다.

이 비교는 각 통제 조건에 맞춰 재학습한 모델의 결과이므로 anchor 정보만의 순수한 bit 기여로 해석하지 않는다.

## 4. 그다음 선택할 학습 목록

아래는 모두 수행하기로 확정한 목록이 아니다. 조건을 만족하는 것부터 고른다.

| ID | 실험 | 신규 50k | 실행 조건·답할 질문 |
|---|---|---:|---|
| **E06** | 학습된 Random+D4 ON16/OFF16 |2| T06의 frozen random과 구분. 단순 재배열과 공간 지역성의 효과 |
| **E07** | D4 matched OFF 두 곡선의 추가 λ |**4 또는 8**| 중앙 3λ면 추가 4, 전체 5λ면 추가 8. 두 수는 대안이며 합산하지 않음 |
| **E08** | Score112-only, .0128 pilot |1| rank56-only가 약한 단일 경로 대조군이라는 반론 |
| **E09** | Score736-only의 추가 λ 2개 |2| 현재 Score736 결과가 경쟁력 있을 때만 RD 확장 |
| **E10** | Score56-only의 낮은 λ 한 점 |1| .0128 한 점의 열세를 전체 구조 열세로 확대하지 않기 |
| **E11** | MSH→복호 residual MSH + 큰 single MSH |2 pilot| 혼합 경로 이득과 일반적인 residual 분해·모델 용량 효과 구분 |
| **E12** | Context-only 두 경로 |1 또는 2| 좋은 1-path Context가 있을 때. 56+56이면 rank112 통제 필요; E08 재사용 시 1 |
| **E13** | QR geometry projection 고정 2-path |1 pilot| B1 metadata 확인 후, 학습된 projection 변화의 역할을 묻고 싶을 때 |
| **E14** | full-width의 좁은 채널 그룹 대조 |1 pilot| 실제 184×4그룹의 predictor 용량과 차원 효과를 더 분리하고 싶을 때 |
| **E15** | 선정한 단일 Context 구조의 2×2 나머지 3셀 |3| 정렬×Context 효과가 2-path 밖에서도 재현되는지 |
| **E16** | MSH latent의 인과적 공간 Context |1부터| fixed-symbol probe에서 여지가 있을 때. 전체 2×2는 기존 H-OFF 재사용 가능 시 총 3 |
| **E17** | Morton+L3 OFF12 중앙점 |1| L3에서도 Morton matched 비교가 본문에 꼭 필요할 때 |
| **E18** | H+L3/M+D4의 추가 seed |4*| E01 후 D4 H 3 seed를 재사용해 작은 스케줄·정렬 차이를 3 seed로 검토 |
| **E19** | Morton+D4 OFF16의 나머지 λ |2 또는 4| E02 후 중앙 3λ는 +2, 5λ는 +4. Morton 전체 범위 주장을 넓힐 때 |
| **E20** | 원 F736 직접 Context: identity+채널 양자화 gain |1 pilot| dense full-width transform도 없는 대안. 실제 원 feature 직접 Context 질문에 답함 |

*E18은 .0128에서 H+L3와 M+D4에 s1/s2 각각 1개씩이다. E01의 H+D4 s2를 완료한 전제다. 해당 seed가 이미 별도로 제출됐는지 manifest로 중복을 제거한다.

구조 선택의 질문을 더 우선하고 싶다면 **E08 rank112 한 학습을 E04/E05보다 앞당겨도 된다.** E05의 두 통제 중에는 예측기 구조를 유지하는 random decoded-anchor가 이웃 품질에 더 직접적이다. Bias-only는 기존 entropy 모델의 median·분포 위치가 이미 offset 일부를 흡수할 수 있어 보조 통제로 해석한다.

E20은 공통 압축 대상 F736에서 rank56 경로와 dense analysis/synthesis를 없애되 채널별 단위 차이를 처리할 gain을 둔다. Native raw geometry512를 그대로 부호화하는 별도 구조는 아니다. T09의 learned736→736 transform 및 E14의 그룹 폭 변경과 구분하고, 같은 geometry projection·초기화 정책·실제 packet 조건으로 비교한다.

### E06: Random 재학습과 T06의 차이

T06은 Hilbert로 학습한 모델에 random 순서를 넣었다. E06은 **random 순서에 맞춰 재학습**하므로 학습분포 불일치의 설명을 줄인다.

ON/OFF에 같은 scene별 재현 가능한 permutation 정책을 사용한다. TRAIN과 eval에 각 scene의 동일한 순서 규칙을 적용하고, 한 arm에서만 매 iteration 새 permutation을 뽑는 augmentation 차이를 만들지 않는다. 수신측이 원순서를 복원하지 않아도 되는지 실제 packet 경로로 확인한다.

### E07: D4의 2×2를 RD 전 범위로

E01로 .0128의 None/Hilbert OFF가 확보된 뒤 추가한다. ON 두 곡선은 이미 5λ가 있다.

- 중앙 3λ(.0064/.0128/.0256): None OFF와 H OFF의 .0064/.0256 각 2개 = **4**.
- 전체 5λ: 두 OFF 곡선의 .0032/.0064/.0256/.0512 각 4개 = **8**.

4개를 먼저 끝내고 나중에 5λ로 늘리면 추가로 4개가 더 필요하다. 이를 4+8=12개로 중복 계산하지 않는다.

가능하면 네 곡선의 공통 화질 구간에서 Context의 비례 rate 이득을 비교한다. 같은 λ 상호작용과 같은 화질 상호작용은 구분한다. 다른 seed의 한 점을 기본 seed RD 곡선에 섞지 않는다.

### E11/E12: 두 경로의 상대는 더 큰 단일 경로도 포함

두 MSH를 쓰면 총 파라미터·latent 용량도 늘어난다. 비슷한 파라미터 예산의 큰 single MSH를 함께 보고 MAC·실제 시간·메모리는 별도 비교한다.

두 rank56 Context 경로는 rank112 표현 용량과도 관련된다. E08 rank112를 재사용하고 전체 stream/header를 포함한다. 원 feature의 승자가 residual에서도 승자라고 가정하지 않는다.

모든 residual은 첫 경로의 **실제 양자화 복호값**으로 만든다. 둘째 경로가 첫째 경로에 conditioning되는지에 따라 decoder 의존성이 달라지므로 무조건 순차 복호라고 쓰지 않는다.

### E13/E14: 표현 경계와 predictor 용량

E13은 공통 QR/fresh 초기화에서 P를 고정하고 T18과 비교한다. 특정 구조만 초기 bridge나 decoder가 다르게 시작하면 안 된다. 먼저 B1의 effective-readout metadata를 확인하고, 정합 설명이 해결되면 이 실험의 우선순위를 다시 판단한다.

현재 Score736-only는 184×4그룹·16 streams이며 active anchor predictor의 채널 혼합 MAC은 rank56 대비 약 162.77배다. E14에서 16폭×46그룹을 쓰면 predictor MAC 비는 작아지지만 streams가 184개로 늘어난다. 따라서 이 비교도 rank만의 단일 요인 통제가 아니며 실제 bytes·stream 비용·성능을 함께 분석한다.

### E15/E16: 기반 구조를 바꾼 원리 검증

E15는 가장 유망한 **Context를 쓰는 1-path**의 H ON 한 점을 재사용하고 None ON/H OFF/None OFF를 같은 레시피로 추가한다. H ON/OFF 한 쌍만으로 Sorting×Context 상호작용까지 증명했다고 쓰지 않는다.

E16은 MSH transform을 유지하면서 복호 hyperprior와 이미 복호한 latent만 사용하는 인과적 조건부 모델부터 시작한다. 단순히 다른 LIC 전체를 바꾸면 transform·모델 용량·확률모델이 함께 달라진다. 구현·actual roundtrip 비용이 커서 앞선 검증보다 후순위다.

## 5. 새 50k 학습 없이 우선 할 평가·분석

| ID | 작업 | 규모·방법 | 가치 |
|---|---|---|---|
| **N01** | B1 geometry 경계 정합 | native→초기 QR→학습 projection→실제 복호 F의 readout·feature 오차, 기존 metadata 우선 | B1=16.77과 native gap을 설명 |
| **N02** | T06에 렌더 품질 추가 | slot/Morton/Hilbert×같은 256 = **768 scene 평가** | packet 감소와 PSNR 변화를 함께 관찰 |
| **N03** | 현행 native 전체 test | 같은 manifest의 **6991 scene 1회** | 최종 RD와 같은 표본의 비압축 기준선 |
| **N04** | 빠진 OFF·구조 complexity | 기본 12 checkpoint×32 scene×3회 = **1152 encode/decode 쌍** | 실제 cost 비교, 독립 receiver 메모리 |
| **N05** | per-scene 효과·payload·estimated/actual 분해 | 기존 63점과 후속 완료점의 로그/packet 재집계 | 평균 이득·경로 상쇄·entropy-gap |
| **N06** | 입력 view 변화 | C6/C12/C24, T8 고정, 같은 128 scene, L3 네 셀+native = **1920 scene 평가** | 입력 조건의 강건성 |
| **N07** | 기존 loader 지원 다른 데이터 | 256 scene×L3 네 셀+native = **1280 scene 평가** | backbone 고정 조건의 제한적 데이터 일반화 |
| **N08** | 고정 symbol/CDF의 stream 분할 |1/4/16 stream 직렬화·복원 동일성 검사 | framing·rANS reset의 순수 비용 |
| **N09** | λ0의 복원 고정 probability 진단 | TRAIN에서 density fit·held-out actual bits 확인 |831KiB의 분포 미적합과 실제 symbol 비용 구분 |
| **N10** | MSH latent의 조건부 의존성 probe | μ/σ 정규화·값/절댓값/제곱과 작은 causal NLL | residual Context 개발의 가치 판정 |

N09/N10은 작은 확률모델 fitting이 들 수 있지만 새 codec 50k 학습과 별도다. 구현·GPU 평가 시간은 든다.

N05의 paired CI에는 scene별 원자료가 필요하다. 집계 로그만으로는 계산할 수 없으며, 자료가 없다면 핵심 모델 평가에서 scene별 결과를 저장하거나 재평가해야 한다.

### N01: 새 학습 전에 이미 있는 metadata부터

B0의 model_pair.geometry_boundary_effective_readout_errors와 scene별 feature 오차를 먼저 읽는다. W_current×P_learned와 W_native의 차이, 초기 QR 보존 여부, checkpoint/decoder 정합을 확인한다.

필요할 때만 Score 양자화 우회를 ‘residual 고정’과 ‘새 LR base에 맞춰 residual 재계산’으로 구분해 공동 적응을 본다. 이를 가산적인 양자화 손실 분해로 해석하지 않는다.

### N02: T06 byte 분석을 반복하지 않고 빈칸을 채우기

기존 packet이 남아 있다면 재인코딩 없이 decode/render한다. Hilbert는 T05 결과도 재사용 가능하므로 768은 세 조건의 전체 평가 규모이지 반드시 모두 새로 돌려야 하는 횟수는 아니다.

PSNR/SSIM/LPIPS, 역순열 후 복원 feature 오차, Gaussian 변화를 함께 본다. 새 품질값을 붙여도 Hilbert 학습 모델의 순서 개입이라는 성격은 유지된다. 그 자체로 다른 순서에서 재학습한 codec의 최적 RD가 되지는 않는다. Random10 렌더는 먼저 세 순서를 본 뒤 선택한다.

### N04: 기존 37개를 전부 반복하지 않는다

기본 새 대상 8개는 .0128의 H-OFF1/H-OFF4/None-L3 OFF12/H-D4 OFF16 + fresh 구조 4개다. 같은 할당 reference 4개는 H-L3 ON/H-D4 ON/H-L3 OFF12/None-L3 ON이다.

동일 GPU 할당·32 scene cache·warmup4·반복3·threads8·BF16·batch1을 맞춘다. NN-score를 새 표에도 넣으면 13번째 reference로 추가한다. E01/E02 등 새 모델 비용을 표에 넣을 때도 해당 모델을 추가한다.

codec-only receiver의 weight/CDF 상주 메모리·추가 allocated/reserved·CPU RSS/rANS·평균/p50/p90을 구분한다. decoder 최소 메모리를 기존 fullmodel harness peak로 대신하지 않는다.

### N06/N07: 같은 장면·target·native를 유지

C24+T8 프레임을 확보한 동일 cohort를 먼저 고르고 target 8개는 고정한다. context는 가능하면 중첩시켜 frame 선택 차이를 줄인다. C24에서 유효 장면 수가 줄어든 다른 cohort의 평균과 비교하지 않는다.

다른 데이터에는 native backbone 평가도 함께 붙여 domain gap과 codec 추가 손실을 구분한다. 새 backbone 재학습까지 자동으로 범위를 넓히지는 않는다.

### N08/N09: 심볼·복원 고정의 의미를 검증

N08은 같은 심볼과 같은 CDF-index 열을 다른 stream으로 직렬화한다. 실제 decoder와 정합되지 않는 byte probe라면 직렬화 microbenchmark로 한정한다. 재학습된 OFF1/OFF4/OFF16의 차이는 단순 header 비용이 아니다.

N09는 transform·predictor·quantization lattice·심볼·복원을 고정하고 density/scale만 맞추는 진단이다. quantile median이나 MSH mean까지 바꿔 복원이 달라지면 같은 화질의 entropy 적합 실험으로 볼 수 없다. 기존 Score 전용 probability 모드가 MSH까지 보정하지 않는 점도 구분한다. TRAIN fit, held-out 평가, 실제 복호 동일성 검증 후 해석한다.

## 6. 학습 예산 선택지

| 묶음 | 포함 | 신규 50k |
|---|---|---:|
| **우선 추천** |E01 7 + E02 1 + E03 4|**12**|
| **원리 설명 확장** |위 12 + E04 3 + E05 2|**17**|
| **D4 중앙 3λ까지** |위 17 + E07 중앙 보강 4|**21**|
| **D4 전체 5λ까지** |위 17 + E07 전체 보강 8|**25**|

21과 25는 대안이다. 21을 끝낸 뒤 25로 확장하면 4개만 더 필요하다. E06 Random 2개, rank112 등 기타 선택 항목은 위 합계에 포함하지 않았다.

이미 제출한 26개·현재 남은 3개·짧은 탐색·수렴 연장·평가·profile은 별도다. 12/17/21/25를 현재 미제출 목록으로 자동 확정하거나 기존 26개와 중복 제출하지 않는다.

추천 운영은 **N01/N02/N03를 먼저 착수하면서 E01/E02를 진행**, E03의 λ는 holdout 결과를 받아 적응적으로 선택하는 것이다. 이후 E04/E05와 범위 확장을 선택한다. 11월 1일 새 설정 동결·11월 10일 제출 목표와 원고 병행은 유지한다. 고비용 full-width나 새 LIC의 소요 시간을 기존 rank56과 같다고 가정하지 않는다.

## 7. 결과에 따라 뒤 실험을 고르는 기준

- D4의 None/Hilbert 상호작용이 seed마다 반복되면 핵심 주장 근거로 사용하고 작은 H/M 우열보다 이를 앞에 둔다.
- MSH-only가 rate를 맞춘 뒤 유리하면 2-path 최적성 주장을 줄이고, 우수한 단일 Context 구조의 원리 이전 여부를 확인한다.
- Score736이 더 좋지만 비싸면 실제 cost 절충으로 보고 E14는 원인 분리가 필요할 때만 선택한다.
- bias-only와 full Context가 비슷하면 anchor 기여 주장을 줄인다. random-anchor가 불리하면 같은 순서에서 참조 이웃의 질이 중요하다는 근거로 쓴다.
- fresh 2×2에서 경향이 약해지면 warm parent 이력의 범위를 명시한다. 유리한 초기화 결과만 골라 일반성으로 포장하지 않는다.
- 구조 pilot이 미수렴이면 여러 λ를 늘리기 전에 대표 두 구조에 같은 추가 학습 예산을 배정해 확인한다.
- seed 3개와 scene 많음을 같은 불확실성으로 합치지 않는다. 원본 clip에 여러 scene이 있다면 그룹 구조를 보존해 통계를 낸다.

## 8. 선행연구와 실험 방향

Morton 직렬화와 공간·채널 문맥을 결합한 피드포워드 3DGS 압축은 이미 존재한다. 따라서 ‘정렬+Context 최초’보다 **중간 scene-token이라는 대상·실제 복호 가능성·이웃 품질 통제·구조 선택의 근거**를 실험으로 명확히 하는 편이 현재 논문 방향에 맞는다. [LocoMoco 원문](https://arxiv.org/html/2512.00877v1).

RD 점을 더 추가하더라도 BD 수치가 자동으로 안정적인 것은 아니다. 공통 구간·보간·point 민감도와 실제 측정점을 함께 확인한다. [Bjøntegaard 비교의 신뢰성 분석](https://arxiv.org/abs/2304.12852).

## 9. 근거와 실행 상태

- [10월 20일 결과 검토](/C:/Users/a0103/Documents/Codex/2026-10-13/new-chat/outputs/research-review-2026-10-20.md).
- [사용자 제공 전체 정리](</C:/Users/a0103/.codex/attachments/e54e6206-ed23-4f5c-879b-d3a2f2ea4462/붙여넣은 텍스트.txt>).
- [실제 pilot 설계·공통 초기화 조건](/C:/Users/a0103/Desktop/school/vml/global/upstream/globalsplat/docs/NFCGS_PAPER_EXPERIMENT_PLAN_2026-10-18.md).

이 문서는 추가 계획이다. 서버 제출·취소·기존 checkpoint/dependency 변경은 수행하지 않았다.


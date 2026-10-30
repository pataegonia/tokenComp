# 추가 실험 제안 검토 — 2026-10-20

21:59 추가 확인: Score736-only의 최종50k holdout128 평가가 완료됐다. 이번 점에서 Score56-only/two-path56보다 화질·용량이 불리해736 추가λ의 우선순위를 낮추고 MSH-only/two-path의 rate 중첩을 먼저 검토한다. [최신 결과](analysis/nfcgs_logs_2026-10-20_2159/REPORT.md). 아래는16:47 결과를 바탕으로 작성한 계획 검토이며, [사용자 제공 제안 원문](analysis/nfcgs_additional_experiment_review_2026-10-20/provided_proposal.md)을 별도 보관했다.

검토 대상: 사용자가 제공한 「추가 실험 제안 — Scene-token 압축 논문」. 결과·진행 상태의 근거는 [전체 실험 정리](NFCGS_EXPERIMENTS_MASTER_2026-10-20.md)의 **2026-10-20 16:47 다운로드 로그**다. 이 문서는 우선순위 판단이며 새 학습·평가를 제출한 기록은 아니다.

## 1. 판단

제안의 실험 정의와 중복 제거는 대체로 타당하다. 특히 D4 ON/OFF의 단계·그룹·16 strings 정합, 구조별 실제 rate 중첩, fresh 공통 초기화, packet-only 복호, scene 변동과 training seed 변동의 구분을 유지해야 한다.

다만 **처음 12개 중 7개를 warm-start D4의 3-seed 2×2에 쓰는 실행 순서는 조정**하는 편을 추천한다. 추가 학습의 1순위 질문은 **MSH-only와 two-path가 실제 용량을 맞춘 뒤에도 어떤 RD 관계를 보이는가(E03)**다. 기본 seed D4의 빈칸 1개는 병행하고, fresh-init 2×2(E04)는 세 번째 warm-start seed 전체 반복보다 앞에 둔다.

## 2. 기존 결과가 지지하는 범위

| 현재 근거 | 관찰 | 추가로 필요한 확인 |
|---|---|---|
| Fresh holdout128 구조 pilot | MSH-only 20.6696dB /14.48434KiB, two-path56 23.0067dB /56.73486KiB | 약3.92배 용량 차이가 있어 같은λ 한 점으로 구조 전체의 RD 우열을 정할 수 없음 |
| Score56-only pilot | 21.7889dB /65.53055KiB; 현재 two-path 점보다 화질과 용량 모두 불리 | 다른λ와 수렴을 확인하기 전에는 구조 전체를 탈락시키지 않음 |
| 추가 seed의 H+D4 ON/OFF | ON은 같은λ에서 bytes17.1857% 감소, PSNR+0.0726dB | None ON/OFF가 없으므로 정렬×Context 상호작용의 추가 seed 반복은 아직 아님 |
| 기존 L3 2×2 RD | 네 계열의 중앙점과 대부분의 양끝점 확보 | D4 및 fresh 조건으로 적용 범위를 넓히는 검증이 중요 |
| T06 frozen order | Hilbert에서 slot 대비 전체 bytes34.25% 감소 | 분포 변화와 복원 품질도 달라질 수 있어 재학습 RD나 동일 화질 절감률과 구분 |
| B0/B1/actual matched256 | 24.90270 /16.76771 /23.87649dB | B1의 readout·projection 정합 설명이 필요하며 B1을 화질 상한이나 버그의 증거로 단정하지 않음 |

## 3. 권장 실행 순서

### 3.1 새 50k 학습 없이 먼저 병행할 일

1. **N01: B0 metadata와 effective readout 확인.** 기존 서버 산출물부터 읽는다. 초기 QR 보존과 학습 후 projection 변화를 구분하고, 설명되지 않는 정합 문제가 발견되면 해당 비교를 사용하는 해석을 먼저 수정한다. 이것이 모든 독립 학습의 선행 조건이라는 뜻은 아니다.
2. **N03: 현재 manifest의 native 전체 test.** Matched256의 B0를 전체6991장 codec 결과와 빼서 손실을 계산하지 않기 위한 기준선이다.
3. **N02: T06 packet의 slot/Morton/Hilbert 렌더.** 기존 Hilbert 결과와 packet을 재사용한다. Byte 진단에 품질을 붙이되 frozen-order 개입이라는 범위를 유지한다.
4. **기존 구조 pilot의 수렴 확인.** 마지막 학습 구간의 distortion·rate·가능한 held-out 지표를 확인한다. 대표 구조가 미수렴이면 여러λ 확장에 앞서 공통 추가 budget으로 수렴을 점검한다.
5. **N04/N05: 주요 구조의 실제 cost와 per-scene 자료 확보.** 전체 test 평가를 새로 실행할 때 scene/frame ID·실제 payload·품질을 저장해 후속 CI를 위해 같은 평가를 반복하지 않도록 한다.

### 3.2 첫 신규 학습은 3개

| 작업 | 제안 ID | 개수 | 선택 이유 |
|---|---|---:|---|
| Fresh MSH-only λ=.0032 | E03 일부 |1| 현재14.48KiB 점에서 rate·화질을 높이는 방향 탐색 |
| Fresh two-path56 λ=.0512 | E03 일부 |1| 현재56.73KiB 점에서 rate를 낮추는 방향 탐색 |
| 기본 seed None+D4 OFF16 λ=.0128 | E01 일부 |1| 기존 None ON/H ON/H OFF와 기본 seed D4 2×2 완성 |
| 합계 | |**3**| 세 학습은 서로의 결과를 기다릴 필요 없이 준비 가능 |

첫 두λ는 제안된 탐색 출발점이다. **이 두 점만으로 실제 rate 또는 화질 구간이 겹친다고 보장하지 않는다.** 구조별 λ와 rate의 관계는 실제 held-out 결과로 확인한다. 이 둘과 None OFF는 기존 공통 초기화·부모·split을 각각 해당 계보에 맞춰 사용해야 한다.

제안의 two-path 추가λ=.0032는 현재보다 더 높은 rate 영역을 탐색하므로 MSH-only와의 간격을 줄이는 첫 실험으로는 후순위다. 이를 버리는 것이 아니라, 남은 E03 budget을 높은 rate 확장과 추가 저율점 중 어디에 쓸지 첫 결과로 정하자는 판단이다.

### 3.3 다음 묶음은 결과를 받아 결정

| 작업 | 추가 학습 | 판단 |
|---|---:|---|
| E03의 나머지 두 점 |2| 실제 중첩 구간을 만드는 방향으로 λ 선택. 자동으로 예정λ를 전부 제출하지 않음 |
| E04 fresh D4 2×2의 나머지 세 셀 |3| two-path를 주력으로 유지할 때 우선. 기존 fresh H ON의 초기화·holdout·레시피를 정확히 재사용 |
| E02 Morton+D4 OFF16 중앙점 |1| 작은 비용으로 세 정렬의 matched ON/OFF 비교 확보 |
| 첫3개와 합계 |**9**| E03 4 + E01 기본1 + E04 3 + E02 1 |

Fresh 구조 비교에서 주력이 다른 Context 단일 경로로 바뀌면 E04와 E15 중 어느 쪽이 논문의 주장을 직접 검증하는지 다시 판단한다. 9개는 조건부 예산이며 확정 제출 목록이 아니다.

Fresh/warm 결과의 차이를 **부모 초기화만의 순수한 인과 효과**로 해석하지 않는다. 현재 두 계보는 부모뿐 아니라 학습 이력과 train-derived holdout 제외 여부도 다르다. E04의 핵심 가치는 fresh 조건 안에서 맞춘 2×2가 재현되는지 확인하는 데 있다.

### 3.4 이후 확대

| 작업 | 추가 | 누적 예산 | 우선순위 이유 |
|---|---:|---:|---|
| E05 random decoded-anchor |1|10| 같은 Hilbert 순서에서 참조 이웃의 질을 바꿔 기전 설명 보강 |
| E01 남은 seed 셀 전체 |6|16| 기존 기본 None OFF를 재사용해 총3 seed의 warm-start D4 2×2 완성 |
| E05 bias-only |1|17| offset·분할 효과를 살피는 보조 통제 |

이 순서의 총17개는 원 제안 E01~E05와 같은 구성이다. **3→9→10→16→17로 결과를 확인하며 진행 순서를 바꾼 안**이다. 핵심 결과가 불안정하면 반복 횟수만 늘리기 전에 초기화·수렴·복구 이력·비교 조건을 먼저 확인한다.

## 4. 나머지 제안의 우선순위

| 항목 | 판단 |
|---|---|
| E07 D4 OFF 추가λ | RD 전체 범위에서 상호작용을 주장할 때 중요. 중앙 3λ 보강4개부터, 유효한 공통 품질 구간 확인 후 양끝점 추가 |
| E08 rank112 | Score736 결과를 본 뒤 결정. Score 단일 경로의 병목 질문이 남으면 E05보다 앞당길 수 있음 |
| E06 재학습 random ON/OFF | 유용하나 T06 렌더와 random-anchor 통제 뒤. 고정 scene별 순열과 동일 ON/OFF 정책 필요 |
| E09/E10 Score-only 추가λ | 현재 한 점을 구조 전체의 결론으로 쓰지 않기 위해 조건부 유지. 현재 후보의 수렴·cost·RD를 보고 선택 |
| E11/E12 경로 조합 변경 | 주력 구조의 RD 근거를 확보한 뒤. 두 경로·큰 단일 경로의 표현 및 parameter 예산을 함께 봐야 함 |
| E13 고정 geometry projection | N01 결과를 먼저 읽고, 남는 구체적 질문이 있을 때 실행. B1 급락만으로 즉시 채택하지 않음 |
| E14 full-width 그룹 폭 | Score736이 경쟁력 있을 때. Stream 수가 바뀌므로 단일 요인 통제로 표현하지 않음 |
| E15 다른 Context 구조 2×2 | 구조 선정 후 가치가 큼. 선정되기 전 모든 구조에 전개하지 않음 |
| E16 MSH spatial Context | N10 probe 이후. Encoder/decoder 인과성·실제 packet 경로까지 개발해야 하므로 지금 범위 확대는 후순위 |
| E17/E18/E19 추가 L3/Morton·seed | 본문에서 해당 정렬·schedule의 작은 차이를 주장할 때 선택 |
| E20 identity F736 Context | T09 결과로 learned full-width transform의 필요성이 설명되지 않을 때. 지금은 구현·새 구조를 늘리는 비용이 큼 |
| N06/N07 view·데이터 변화 | 데이터와 loader가 준비돼 있다면 작은 subset을 내부 변형 추가보다 앞당길 수 있음. 같은 cohort·target·native 기준을 유지 |
| N08 stream 분할 | framing 비용만 따로 묻고 싶을 때 유용. 같은 symbol/CDF와 복원 동일성 검증 필요 |
| N09 λ0 entropy 재적합 | 별도 연구 질문. 현재 유용한 rate 영역의 RD·기준선 확보보다 후순위 |
| N10 latent 의존성 probe | E16의 개발 여부를 정하는 저비용 선별. 상관계수만으로 actual bit 이득을 예측하지 않음 |

## 5. 제안에 더 보강할 부분

### 외부 기준선의 실행 가능성

현재 실험은 내부 설계 대조가 풍부하다. 논문 준비에서는 **공개 외부 codec 한 개를 같은 입력 장면·카메라·해상도에서 평가할 수 있는지 확인하는 작업**도 앞당길 가치가 있다. 먼저 작은 공통 subset에서 입출력 호환성·학습 가중치·실제 bitstream·평가 비용을 확인한다. 호환성과 자원 확보 전에 새 대규모 재학습을 확정하지 않는다.

서로 다른 backbone의 feature codec을 연결하는 비교와 같은 GlobalSplat 출력 Gaussian을 압축하는 비교는 질문이 다르다. 전자는 전체 시스템 비교, 후자는 압축 위치를 달리한 비교로 조건과 공유 가중치·payload 범위를 명시한다. 타 논문 수치만 현재 C12/T8 표에 붙이지 않는다. [기존 외부 평가 프로토콜 조사](NFCGS_EXTERNAL_EVAL_PROTOCOLS_2026-10-19.md)를 출발점으로 사용한다.

LocoMoco 원문에는 Morton 직렬화와 공간·채널 Context를 결합한 feed-forward **Gaussian 속성** 압축이 이미 제시돼 있다. 따라서 ‘정렬+Context 최초’라는 주장에는 근거가 없고, scene-token 경계의 설계·실제 수신 가능성·RD와 비용·원리 검증을 명확히 해야 한다. 이것이 두 방법의 성능 순위를 뜻하지는 않는다. [LocoMoco 원문](https://arxiv.org/html/2512.00877v1).

### 분석 지표와 cost

E01의 log-rate interaction 식과7개 신규 계산은 맞다. 다만 같은λ의 상호작용이므로 품질 변화와 함께 보고한다. Seed를3개로 늘리는 것은 유용한 반복 검증이지 모든 초기화의 일반성이나 안정적인 seed 모집단 CI를 보장하지 않는다.

E03의3점은 RD 선별 출발점이다. 공통 구간과 측정점의 위치에 따라 보간이 불안정할 수 있으므로 BD 수치의 유무를 완료 기준으로 삼지 않는다. [Bjøntegaard 비교의 신뢰성 분석](https://arxiv.org/abs/2304.12852).

N04의32×3은 측정 구간1152쌍이며 warmup과 준비 시간은 별도다. 통제된 codec profile과 encoder 없는 receiver의 메모리는 서로 다른 측정 범위다. E01~E05가 새로 늘면 해당 모델 profile·독립 복호·선택된 모델의 full-test 비용도 새50k 학습 수와 별도로 계산한다.

## 6. 이번 판단의 적용 범위

현재 진행 중인 세 학습과 기존 dependency는 유지한다. 이 검토는 사용자 제공 제안을 결과에 비추어 재배열한 것이며, 최종 수렴·Score736 결과·N01 원본 metadata를 아직 확인한 것으로 간주하지 않는다. 새 실험을 실제로 준비할 때 기존 제출 manifest와 중복을 다시 대조한다.

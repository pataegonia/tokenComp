# 외부 feature codec 평가 프로토콜 확인 — 2026-10-19

CodecSplat·FeCoSplat·MFGS의 논문 v1과 CodecSplat 공식 공개 평가 코드를 확인했다. C는 입력 context 수, T는 평가 target 수다. 학습 view 수와 평가 view 수를 구분한다. 이 조사에서 학습·평가 코드는 변경하지 않았다.

## 확인한 평가 조건

| 모델 | 데이터셋 | 평가 C/T | 해상도 | 근거·확인 범위 |
|---|---|---|---|---|
| CodecSplat | RE10K | **2/3** | 256×256 | 공식 test script와 실제 index JSON |
| CodecSplat | DL3DV | **8/8** | 256×448 | 공식 test script와 실제 index JSON |
| FeCoSplat | RE10K | **2 / T 미명시** | 256×256 | 논문 §4.1에서 CodecSplat과 동일 설정이라고 설명 |
| FeCoSplat | DL3DV | **8 / T 미명시** | 256×448 | 논문 §4.1. target 개수 자체는 별도 확인되지 않음 |
| MFGS, MVSplat+ZPressor 주 모델 | RE10K·ACID | **8/8** | 256×256 | 논문 §4.1, Appendix A.2/A.3 |
| MFGS, MVSplat+ZPressor 주 모델 | DL3DV | **12/8** | 256×256 | 논문 §4.1, Appendix A.2/A.3 |
| 현재 우리 codec | RE10K | **12/8** | 256×256 | `run_nfcgs.py eval` → `re10k_eval_all_ctx12` |
| GlobalSplat 원본 설정 | RE10K | **12·24·36 / 3** | 256×256 | 로컬 원본 평가 config와 고정 C3G index |

FeCoSplat의 T=3(RE10K), T=8(DL3DV)는 ‘CodecSplat과 동일 설정’이라는 문구로부터 추정할 수 있으나 **직접 확인한 수치로 표기하지 않는다**. 확인한 v1 본문·부록에는 target 개수를 별도로 제시하지 않았고, 논문은 코드 추후 공개를 안내한다. 향후 공개 코드·index로 확정한다. ACID는 RE10K 학습 모델의 zero-shot 평가로 보고한다. [FeCoSplat §4.1](https://arxiv.org/html/2609.33330v1#S4.SS1)

## CodecSplat: 실제 index 점검

공식 [RE10K 평가 script](https://github.com/pengpeng-yu/CodecSplat/blob/main/scripts/test_re10k_view2_256x256.sh)는 `assets/evaluation_index_re10k.json`을 사용한다. 공식 [DL3DV 평가 script](https://github.com/pengpeng-yu/CodecSplat/blob/main/scripts/test_dl3dv_view8_256x448.sh)는 `dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json`을 사용한다. 두 JSON을 읽어 전체 non-null 항목의 context/target 배열 길이를 집계했다.

| JSON | 전체 항목 | non-null 항목 | C/T 분포 |
|---|---:|---:|---|
| RE10K | 7,194 | 6,474 | 6,474개 모두 C2/T3 |
| DL3DV | 140 | 140 | 140개 모두 C8/T8 |

위 수는 **평가 index 항목 수**이며 논문에서 실제 모든 모델·rate가 성공한 최종 scene 수를 별도로 재현한 결과는 아니다. 데이터 설치 상태와 처리 실패에 따라 실제 평가 집합은 달라질 수 있다.

- [RE10K JSON](https://raw.githubusercontent.com/pengpeng-yu/CodecSplat/main/assets/evaluation_index_re10k.json), SHA-256 `fbcff9b4b13139227d809af45b84bb5a7a9cdb1c7a55bced85b6c66556a219cc`.
- [DL3DV JSON](https://raw.githubusercontent.com/pengpeng-yu/CodecSplat/main/assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json), SHA-256 `2cddce18ce668d46f6a365233e9e9430168d4c286471222186d9e21f1dcbe757`.

## MFGS: 학습과 평가, 집합별 차이

MVSplat+ZPressor 주 모델은 **학습 C12/T8**, RE10K·ACID 평가 **C8/T8**, DL3DV 평가 **C12/T8**이다. 따라서 학습의 C12/T8을 RE10K test 설정으로 읽으면 안 된다. [MFGS §4.1](https://arxiv.org/html/2610.20590v1#S4.SS1)

Appendix A.3은 기본 test 집합을 RE10K 7,067·ACID 1,810·DL3DV 140 scene으로 설명한다. 주 RD에서는 모든 비교 모델·rate의 평가가 끝나고 지표가 유한한 공통 집합인 **6,235 / 1,597 / 140 scene**을 사용한다. 방법 간 입력·target view를 동일하게 맞춘다. 이는 우리 6,991 scene 집합과 같은 집합임을 뜻하지 않는다. [MFGS Appendix A.3](https://arxiv.org/html/2610.20590v1#A1.SS3)

MonoSplat·TranSplat generator 확장 실험은 별도 설정이다. 해당 모델은 학습 C2/T4, ACID 평가 C2/T3를 사용한다. 따라서 MFGS의 모든 표를 C8/T8로 묶지 않는다. seed·MIV·복잡도 분석도 각각 명시한 평가 집합을 확인한다.

## 우리 실험에 적용할 판단

1. T=8을 사용하는 외부 feature codec 선례가 있다. 현재 C12/T8은 자체 대조군 비교 프로토콜로 유지할 수 있다.
2. 원본 GlobalSplat 논문과의 재현 비교는 원본 C3G 고정 index의 C12/T3 등으로 별도 수행한다. 현재 C12/T8 비압축 기준선도 측정해 codec 손실을 분리한다.
3. 외부 codec과의 직접 RD 비교는 공통 scene·입력 context·target frame·해상도·metric·전체 bitstream 집계 조건을 먼저 정하고 양쪽을 재평가한다. target 개수만 같게 바꾸는 것으로는 충분하지 않다.
4. CodecSplat·FeCoSplat의 RE10K 주 설정은 입력 2개다. 이들을 현재 입력12개 결과와 논문 숫자만 모아 직접 순위로 비교하지 않는다. 공개 모델의 입력 수 지원과 모델별 재평가 가능성을 확인해 공통 프로토콜을 결정한다.
5. 평가 프로토콜 전환은 우선 기존 checkpoint의 재평가 작업이다. 입력 수·분포 변화의 영향을 기록하며, 재학습이 필요한지는 별도로 판단한다. 조사 결과만으로 기존 학습·평가 job을 변경하지 않는다.

추가 근거: [CodecSplat 논문 §4.1](https://arxiv.org/html/2605.25563v1#S4.SS1), [FeCoSplat 논문](https://arxiv.org/html/2609.33330v1), [MFGS 논문](https://arxiv.org/html/2610.20590v1). 저장소 `main`은 이후 바뀔 수 있으므로 재현에는 위 index 해시와 고정한 commit을 사용한다.

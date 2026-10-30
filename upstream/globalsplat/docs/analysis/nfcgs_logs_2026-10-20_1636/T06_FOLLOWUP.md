# 사용자 제공 T06 보고서 후속 해석

사용자가 제공한 job458818의 [REPORT 원문](provided_t06_report.md)을 검산했다. 동일256장·하나의 Hilbert+D4 checkpoint에서 F·projection·basis·정규화 Score·XYZ·weight를 고정한 정렬 개입이다. 원격 per-scene JSON이나 trace를 새로 내려받아 확인한 것은 아니다.

## 핵심 관찰

| 항목 | Slot | Hilbert | Hilbert의 상대 감소 |
|---|---:|---:|---:|
| 실제 anchor XYZ 거리 | 2.3873300 | 0.2092603 | 약91.23% |
| Score 예측 잔차 정규화 MAE | 1.6707784 | 1.5245647 | 약8.75% |
| Raw Score KiB | 41.06366 | 22.66045 | 약44.82% |
| MSH y KiB | 15.19284 | 14.18643 | 약6.62% |
| 전체 KiB | 56.67001 | 37.25922 | 약34.25% |

공간 정렬로 실제 참조 anchor가 가까워지고 예측 잔차와 Score 부호화 비용이 함께 작아진다는 해석을 뒷받침한다. 같은 checkpoint에서 token 순서에 개입했으므로, 정렬별로 다른 weight·basis를 학습한 비교와 구분되는 근거다.

Slot 대비 전체19.41079KiB 감소 중18.40321KiB, 약94.81%가 raw Score stream의 byte 차이다. MSH y/z도 감소하므로 전체 용량 차이를 모두 Score에 배정하지 않는다. 이94.81%는 관찰된 stream byte 차이의 산술 분해이며, Score와 MSH를 독립적으로 바꾼 인과 기여율이 아니다. Score 복호가 달라지면 MSH residual도 다시 계산된다. Header는 모든 순서에서284B로 같다.

Slot과 random10 평균의 전체 용량 차이는 약0.094%이고 anchor 거리·잔차 MAE도 유사하다. 현행 checkpoint에서 원래 slot 순서는 이 공간 predictor에 random과 비슷한 입력을 제공한다는 관측이다. native GlobalSplat 전체에 공간 구조가 없다는 결론으로 확대하지 않는다.

Morton 대비 Hilbert는 anchor 거리13.57%, MAE0.34%, raw Score2.66%, 전체 bytes1.73% 감소다. Morton−Hilbert의 paired 차이는+670.812B,95% scene bootstrap 구간[+648.911,+696.355]B다. 따라서 이 고정 모델·부분집합에서 Hilbert의 평균 bytes가 더 작은 방향은 일관된다. Hilbert로 학습한 predictor를 그대로 사용하므로 Hilbert가 모든 모델에서 본질적으로 우수하다는 근거로 사용하지 않는다.

## 해석 범위

Anchor 거리는 실제 anchor가 있는 단계·그룹의 reference pair를 채널 폭과 pair 수로 가중한 값이다. 경계에서 같은 anchor가 두 입력으로 복제되면 해당 입력 위치를 유지한다. Score MAE는 첫 anchor 단계까지 포함한 전체4096×56 symbol을 가중한 평균이다. 두 지표는 분모가 다르므로 감소율을 일대일로 대응시키지 않는다.

MAE8.75% 감소와 Score bytes44.82% 감소는 함께 관찰된 값이다. 부호화량은 잔차 분포·채널/단계별 entropy model·quantization step 등에 의존하므로 MAE 감소율만으로 byte 감소율을 예측하거나 잔차 tail이 줄었다고 확정하지 않는다.

세 paired95% 구간은 모두 양수다. 이는1000회 scene resample에 따른 평균 byte 차이이며 training seed 변동을 포함하지 않는다. Random10은 각 scene 안에서 먼저 평균했으므로2560개 결과를 독립 scene으로 취급하지 않았다.

이 진단은 PSNR을 측정하지 않았고 순서에 따라 feature distortion도 달라질 수 있다.34.25%는 frozen-model packet 차이이며 동일 화질 RD 절감률이 아니다. Hilbert packet은 T05와 같지만 다른 순서의 렌더 화질은 이 보고서에 없다.

## 다음 판단

현행 정렬의 동작 설명에는 의미 있는 근거가 추가됐다. 당장 새 정렬 학습을 늘릴 우선순위는 낮다. 다음 학습 설계는 남은 Score736-only의 holdout 결과를 받아 two-path 및 MSH-only와 실제 화질·용량을 비교하고, 살아남은 구조의 bitrate를 맞추는 λ 조정을 우선한다.

추가 runtime 수정이나 job 제출은 수행하지 않았다. 상대 감소율과 stream 기여는 제공된 반올림 평균에서 계산했으며, 전체 paired byte 차이와 구간은 사용자 REPORT의 원래 값을 보존했다.

# Legacy3 네 모델의 실제 anchor·예측 잔차 분석

이번 실행 범위는 λ=.0128, 최종50k, minimal/rank56/MSH ON인 **기본 순서·Morton·Hilbert·NN-score의 legacy3 네 모델**, 동일한 C12 입력256개다. 기존6모델 분석 manifest와 launcher는 보존했다. 이번에는 아래의 `analyze_nfcgs_legacy3_anchors.slurm`을 사용한다.

## 측정 정의

4096 token의 실제 부호화 pass에서 관측한다. 학습 시 noise나 양자화 전 anchor로 예측값을 재계산하지 않는다.

| 단계 | 대상 순번 | token 수 | 실제 공간 입력 |
|---|---|---:|---|
| A | 0 mod4 | 1024 | 공간 anchor 없음 |
| B | 2 mod4 | 1024 | 앞 단계에서 복호된 ±2 anchor |
| C+D | 홀수 | 2048 | 앞 단계에서 복호된 ±1 anchor |

Kernel5 convolution의 실제 입력 canvas에서 이전 단계의 유효 참조만 선택한다. 범위 밖 zero padding이나 아직 복호되지 않은 위치는 anchor 거리의 분모에서 제외한다. 끝에서 유효 anchor가 하나면 그 하나만 센다. 예측기는 group 내 채널을 혼합하므로 같은 채널의 anchor 거리만으로 예측 능력을 설명할 수는 없다.

- **Score 거리:** 원래 target `s_i`와 실제 복호 anchor `s_hat_j` 사이의 L1/MSE. Score 원래 단위와 채널 STD 정규화 값을 함께 저장한다.
- **예측 잔차:** `r_i=s_i-prediction_i`의 MAE/MSE. 실제 encoder prediction을 그대로 관측한다.
- **예측 전 기준:** 같은 단계·같은 target의 `s_i-base_i`. 현재 mean/channel context OFF인 minimal 설정에서 base=0이다. 예측 후/전 MAE·MSE 비율을 저장한다. Entropy median을 빼는 기준이 아니며 추가 OFF pass도 실행하지 않는다.
- **정규화:** 각 scene·모델의 원래 Score4096개에서 채널별 population STD(floor1e-6)를 계산한다. L1/MAE는 STD로, MSE는 STD²로 나눈다. 정규화는 진단에만 사용하고 encoder 입력이나 순서를 바꾸지 않는다.
- 모든256개 scene의 단계/group/채널 통계, 앞8개 scene의 전체 target/anchor/prediction/reconstruction/quantization step/median tensor를 저장한다. 전체 tensor도256개가 필요하면 `ANCHOR_TRACE_SCENES=256`을 지정한다.

입력 이미지·카메라 tensor를 한 번만 cache하고 네 모델에 공유한다. `inputs.json`에 scene 목록과 입력 fingerprint를 기록한다. seed0의 첫 unique processable scene256개이므로 전체 평가셋의 무작위 표본은 아니다. 기존 broad 분석을 실행하지 않은 경우 새로 이 집합을 선택한다. 기존 분석과의 동일성은 scene 목록뿐 아니라 입력 fingerprint로 확인한다.

모델별 step/config를 strict 검증하고 frozen encoder/Gaussian decoder state hash가 같은지 검사한다. 실제 codec 전체를 부호화·복호하며 관측된 Score 복원이 receiver와 완전히 같은지 확인한다. 실제 Score bytes와 MSH bytes도 검산용 JSON에 남지만 이번 주 그림은 anchor·잔차 두 장이다. Renderer/재학습/별도의 정렬 재적용/prediction bypass/quarter2/dyadic4 비교는 실행하지 않는다.

## 서버에 반영할 파일

| 로컬 파일 | 서버 상대 경로 |
|---|---|
| [score_context.py](../globalsplat/compression/score_context.py) | `globalsplat/compression/score_context.py` |
| [mechanism.py](../globalsplat/compression/mechanism.py) | `globalsplat/compression/mechanism.py` |
| [profile_nfcgs_complexity.py](../scripts/profile_nfcgs_complexity.py) | `scripts/profile_nfcgs_complexity.py` |
| [analyze_nfcgs_mechanism.py](../scripts/analyze_nfcgs_mechanism.py) | `scripts/analyze_nfcgs_mechanism.py` |
| [report_nfcgs_mechanism.py](../scripts/report_nfcgs_mechanism.py) | `scripts/report_nfcgs_mechanism.py` |
| [analyze_nfcgs_legacy3_anchors.slurm](../scripts/slurm/analyze_nfcgs_legacy3_anchors.slurm) | `scripts/slurm/analyze_nfcgs_legacy3_anchors.slurm` |
| [4모델 manifest](../../../artifacts/nfcgs_legacy3_anchors_manifest_2026-10-16.json) | **`logs/slurm/nfcgs_legacy3_anchors_manifest_2026-10-16.json`** |

이미 반영된 공통 파일도 최신 수정본으로 맞춘다. Production observer는 기본None이고 checkpoint tensor key나 부호화 결과를 바꾸지 않는다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
bash scripts/slurm/analyze_nfcgs_legacy3_anchors.slurm plan
# PLAN 4개, MODELS=4, SCENES=256 및 checkpoint 존재 확인 후:
sbatch scripts/slurm/analyze_nfcgs_legacy3_anchors.slurm
```

GPU1/CPU8/RAM48G를 요청한다. Slurm8시간은 실행시간 예상이 아니라 상한이다. 완료 checkpoint를 사용하므로 진행 중인 학습에 dependency를 걸지 않는다.

## 결과와 재개

`outputs/nfcgs_legacy3_anchors/job_<ID>/` 아래 metadata, input cache/fingerprint, backbone hash, `points/<모델>/scene_XXXX.json`, `traces/`, `complete.json`을 저장한다. 완료 후 같은 job에서 `report_<ID>/`를 만든다.

- `01_target_anchor_score_distance.png/.pdf`: B와 C+D의 실제 anchor−target 정규화 L1/MSE.
- `02_prediction_residuals.png/.pdf`: 단계별 정규화 MAE/MSE, 예측 후/전 비율.
- `summary.json`: 원래 Score 단위와 정규화 수치, scene 평균·95% bootstrap 구간, 기본 순서 모델 대비 같은 scene의 paired 차이.
- `REPORT.md`: 정의와 단계별 예측 전/후 표. 모든 scene의 채널별 값은 `points/`에 남는다.

단계 통계는 대상×채널 또는 유효 anchor 쌍×채널로 가중한 뒤, scene을 동일 가중치로 평균한다. Bootstrap은 scene 단위이며 token을 독립 표본으로 세지 않는다. 학습 seed 변동은 포함하지 않는다. 네 모델의 학습된 basis와 geometry projection은 다르므로 **학습된 모델의 동작 비교**로 해석한다. 정렬만의 인과 효과나 전체 bit 감소를 이 분석만으로 단정하지 않는다.

중단 후 동일 code/checkpoint/protocol로 재개:

```bash
ANCHOR_OUTPUT=/ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_legacy3_anchors/job_<OLD_ID> \
ANCHOR_RESUME=1 sbatch scripts/slurm/analyze_nfcgs_legacy3_anchors.slurm
```

기존 결과를 덮어쓰지 않으며 완료 scene을 재사용한다. 강제 종료로 `.analysis-lock` 빈 디렉터리가 남으면 이전 job이 종료된 것을 확인한 뒤 그 빈 lock만 제거하고 재개한다. Trace 범위 등 프로토콜을 바꾸면 새 출력 디렉터리를 사용한다.

다운로드는 `report_<ID>/`, `metadata.json`, `inputs.json`, `complete.json`, `points/`를 우선한다. 큰 `inputs.pt`와 `traces/`는 tensor 확인이 필요할 때 받는다. 서버 checkpoint/CUDA 실행 결과는 제출 후 확인해야 한다.

## 로컬 검증

관련 CPU 테스트16개가 통과했다. 실제 entropy 부호화·복호를 사용해 observer ON/OFF의 byte/state 보존, raw tensor에서 다시 계산한 anchor 거리와 잔차 통계 일치, 네 모델 입력 공유와 부분 재개, 범위 밖 분석 제외, 보고서 생성·paired 집계, 두 Slurm launcher의 문법과 plan/run 전달을 확인했다. 합성 tiny codec으로 새 그림2장의 레이아웃도 확인했다. `artifacts/nfcgs_legacy3_anchors_SYNTHETIC_LAYOUT_QA_2026-10-16/`은 레이아웃 검증 전용이며 학습된 모델의 실험 결과가 아니다.

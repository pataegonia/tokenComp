# Hilbert 고정: quarter2·dyadic4의 실제 anchor·단계별 잔차

현재 실제256 scene 분석은 legacy3 네 정렬만 완료했다(job454995). Quarter2·dyadic4의 실제 anchor 관계와 예측 잔차는 아직 측정 결과가 없다. 이번 실행은 λ=.0128의 Hilbert legacy3·quarter2·dyadic4 최종50k checkpoint를 대상으로 한다. 재학습은 수행하지 않는다.

## 동일 입력과 기준 모델

원본 `outputs/nfcgs_legacy3_anchors/job_454995/inputs.pt`를 직접 읽는다. 캐시를 새로 추출하거나 복사하지 않는다. 원래 장면 목록과 이미지·카메라 tensor 전체의 fingerprint를 다시 계산하고 다음 값과 일치하는지 검사한다.

```text
069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29
```

`inputs.json`, `metadata.json`, `complete.json`, `backbone.json`, 기존 Hilbert legacy3의 scene JSON256개도 검증한다. 캐시나 원본이 없으면 새256 scene으로 대체하지 않고 실패한다. 원본256개 scene의 C12 입력, seed0, BF16, 4096 token을 유지한다.

Hilbert legacy3를 같은 job에서 다시 측정해 기준을 맞춘다. 원래 legacy3와 packet SHA256, 전체/Score byte 및 단계별 raw MAE가 같은지도 각 scene에 기록한다. GPU·소프트웨어 차이 등으로 일치하지 않는 경우 보고서에 그 수치를 표시하며, 새 스케줄 비교는 이번 job에서 다시 측정한 legacy3를 기준으로 한다. 기존 frozen encoder/Gaussian decoder hash는 동일해야 한다. 원본 결과는 수정하지 않는다.

## 실제 참조와 분모

| 스케줄 | 단계별 target | 단계별 token 수 | 실제 공간 참조 |
|---|---|---|---|
| legacy3 | 0mod4 → 2mod4 → 홀수 | 1024 → 1024 → 2048 | 첫 단계 없음, 이후 ±2와±1의 유효 decoded anchor |
| quarter2 | 0mod4 → 나머지 | 1024 → 3072 | target의4-token 블록 좌우0mod4 anchor. Phase1/2/3을 한 단계에서 복호 |
| dyadic4 | 0mod8 → 4mod8 → 2mod4 → 홀수 | 512 → 512 → 1024 → 2048 | 첫 단계 없음, 이후 ±4·±2·±1 decoded anchor |

Quarter2·dyadic4는 오른쪽 anchor가 없으면 왼쪽 anchor를 복제한다. 서로 다른 learned weight를 적용하는 두 input slot이므로 복제 slot을 모두 거리 통계에 포함한다. Legacy3의 범위 밖 zero padding과 아직 복호되지 않은 canvas 위치는 유효 anchor 통계에서 제외한다. 모든 경우 encoder의 실제 예측값을 기록하고 receiver 재복호 Score와 관측 복원이 완전히 같은지 검사한다.

모든256 scene에서 stage/group/채널별로 원래 Score 단위 및 채널 STD 정규화 MAE/MSE, 예측 전 기준, 예측 후/전 비율, 실제 anchor 거리·상관·XYZ 거리, quantization MSE, raw Score bytes와 token당 bits를 저장한다. 첫8개 scene은 전체 anchor·target·prediction·reconstruction tensor도 저장한다. 전체 tensor256개가 필요하면 `SCHEDULE_TRACE_SCENES=256`을 지정하고 새 출력에서 실행한다.

단계별 token 집합이 다르므로 같은 단계 번호끼리 paired 차이를 계산하지 않는다. 별도로 전체4096×56개 symbol의 잔차를 동일 가중치로 합산한다. 이 전체 지표와 Score/MSH/전체 byte에만 legacy3 대비 scene별 paired 차이와95% bootstrap 구간을 계산한다. 단계별 raw bytes와 bits/token을 함께 보고 단계의 분모 차이를 구분한다.

## 반영할 파일3개

공통 `score_context.py`, `mechanism.py`, `profile_nfcgs_complexity.py`는 이미454995에서 검증한 버전을 사용한다. 다음 파일3개를 최신본으로 반영한다.

| 파일 | 서버 상대 경로 |
|---|---|
| [analyze_nfcgs_mechanism.py](../scripts/analyze_nfcgs_mechanism.py) | `scripts/analyze_nfcgs_mechanism.py` |
| [report_nfcgs_mechanism.py](../scripts/report_nfcgs_mechanism.py) | `scripts/report_nfcgs_mechanism.py` |
| [analyze_nfcgs_hilbert_schedule_anchors.slurm](../scripts/slurm/analyze_nfcgs_hilbert_schedule_anchors.slurm) | `scripts/slurm/analyze_nfcgs_hilbert_schedule_anchors.slurm` |

실행기는 아래 최종 checkpoint 경로와 설정으로 manifest를 자동 생성한다. 별도의 manifest 업로드는 필요하지 않다. 생성 위치는 `logs/slurm/nfcgs_hilbert_schedule_anchors_2026-10-17.json`이다. Plan에서도 manifest를 생성할 수 있지만 job을 제출하지 않는다.

| 모델 | checkpoint run |
|---|---|
| Hilbert legacy3 | `outputs/nfcgs_order_context/hilbert/legacy_stages3_k5_ste/lambda0p0128/job_447725/` |
| Hilbert quarter2 | `outputs/nfcgs_order_context/hilbert/quarter2_stages2_k5_ste/lambda0p0128/job_447740/` |
| Hilbert dyadic4 | `outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/lambda0p0128/job_447746/` |

각 run의 `checkpoints/nfcgs_main/version_0/step000050000.ckpt`를 사용한다. Manifest 생성 시 실제 파일 존재를 확인하고, 분석 시 saved config/step을 strict 검증한다. 설정은 기존 완료 eval에서 확인한 minimal/rank56/context ON/MSH ON이다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
if bash scripts/slurm/analyze_nfcgs_hilbert_schedule_anchors.slurm plan; then
    sbatch scripts/slurm/analyze_nfcgs_hilbert_schedule_anchors.slurm
fi
```

Plan 출력에서 `MODELS=3 SCENES=256`, `INPUT_RUN=...job_454995`, 원래 fingerprint와 checkpoint3개를 확인한다. GPU1/CPU8/RAM48G, 시간 상한8시간을 요청하며8시간은 실행시간 예상이 아니다. 학습 dependency는 없다.

## 출력과 재개

로그는 `logs/slurm/slurm-gs-hsched-anchors-<ID>.out/.err`이다.

`outputs/nfcgs_hilbert_schedule_anchors/job_<ID>/` 아래 input provenance/fingerprint, metadata, backbone hash, scene별 `points/`, 샘플 `traces/`, `complete.json`을 저장한다. 큰 입력 캐시는 원본454995에만 두므로 분석과 재개가 끝날 때까지 원본 캐시를 유지한다.

같은 job에서 `report_<ID>/`를 자동 생성한다.

- `01_schedule_stages.png/.pdf`: 스케줄별 실제 anchor 거리·원래 단위/정규화 잔차·bits/token. 각 행에 단계 이름과 token 수를 명시한다.
- `02_whole_scene_cost.png/.pdf`: Score/MSH/header 용량과 전체4096 token의 raw 잔차 MAE.
- `summary.json`: 단계별 지표, 전체 token 가중 집계, whole-scene paired 차이와95% 구간, 기존 legacy3 재현 검사.
- `REPORT.md`: 위 표·그림과 참조·경계·분모 정의.

중단 후에는 같은 protocol/code/checkpoint/input으로 재개한다. 완료 scene을 재사용하며 원본 결과를 덮어쓰지 않는다.

```bash
SCHEDULE_OUTPUT=/ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_hilbert_schedule_anchors/job_<OLD_ID> \
SCHEDULE_RESUME=1 sbatch scripts/slurm/analyze_nfcgs_hilbert_schedule_anchors.slurm
```

강제 종료로 `.analysis-lock`이 남으면 이전 job이 종료된 것을 확인한 뒤 그 빈 lock만 제거한다. Protocol이나 trace 범위를 바꾸면 새 출력 폴더를 사용한다. 내려받을 때는 `report_<ID>/`, `points/`, `metadata.json`, `inputs.json`, `complete.json`을 우선한다.

Hilbert 알고리즘을 고정했지만 학습된 geometry projection 차이로 실제 permutation은 달라질 수 있다. Basis·예측기·확률모델도 다르므로 이는 학습된 모델의 동작 비교다. 단독 스케줄 변경의 인과 효과나 전체 RD 개선을 이 잔차 분석만으로 단정하지 않는다. Scene bootstrap은 학습 seed 변동을 포함하지 않는다.

## 검증

기존 진단·runner17개와 새 스케줄 분석3개, 총20개 CPU 테스트가 통과했다. 실제 entropy 부호화·복호 anchor, 입력 캐시 불변·fingerprint 검증, 기준 모델 packet 완전 재현, 부분 재개, 자동 manifest 설정 일치, 단계 분모별 집계, PNG/PDF 생성, Slurm plan/run 전달을 확인했다. CUDA/BF16 실제 checkpoint 분석은 서버 실행 후 확인해야 한다. 수정 전 분석 코드2개는 `artifacts/nfcgs_analysis_code_before_schedule_anchors_2026-10-17/`에 보관했다.

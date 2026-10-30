# T06 고정 H+D4 정렬 개입

하나의 H+D4 checkpoint와 같은256장의 F·projection·basis·정규화 Score·XYZ를 고정한 상태에서 token 순서만 바꾼다. 실제 인과적 복호 anchor의 거리, 예측 잔차, raw Score bytes와 MSH bytes의 관계를 확인한다. 구조 pilot의 완료를 기다리지 않고 실행할 수 있는 GPU1 진단 job이다.

## 서버 반영과 실행

새 파일 두 개를 서버의 같은 상대 경로에 올린다.

- [scripts/analyze_nfcgs_fixed_order.py](../scripts/analyze_nfcgs_fixed_order.py)
- [scripts/slurm/analyze_nfcgs_fixed_order.slurm](../scripts/slurm/analyze_nfcgs_fixed_order.slurm)

T15에서 사용한 `eval_nfcgs_independent_receiver.py`와 Slurm, T05 runner·manifest, profile 및 mechanism helper를 재사용한다. 기존 파일은 수정하지 않았다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
mkdir -p logs/slurm
bash scripts/slurm/analyze_nfcgs_fixed_order.slurm plan && \
  sbatch scripts/slurm/analyze_nfcgs_fixed_order.slurm
```

plan은 완료된 T05 metadata·256개 packet·checkpoint·코드 해시와 원본 cache의 metadata를 읽기 전용으로 검사한다. CUDA 및 대용량 cache tensor 검사는 계산 노드에서 한다. 실제 실행도 데이터셋 loader를 만들지 않고 원본 `inputs.pt`를 직접 사용하므로 `/data3` 접근을 요구하지 않는다.

GPU1, CPU8, 메모리48G, 시간 상한8시간이다. 기존 학습·평가 dependency는 없다. TMPDIR은 `/ceph_data/<USER>/.tmp/nfcgs/fixed_order_<JOB_ID>`다.

## 고정 입력과 순서

- H+D4 ON16, λ=.0128, job447746, final50k.
- 원본 cache: `outputs/nfcgs_legacy3_anchors/job_454995/inputs.pt`.
- 원본 cache fingerprint: `069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29`.
- 원본 actual packet: `outputs/nfcgs_path_diagnostics/job_458004/`.
- C12에서 추출한4096 tokens, F736, rank56, BF16, eval seed0.
- 13개 순서: 원래 slot, Morton, Hilbert, scene별 deterministic random10개.

random seed는0~9와 scene ID를 SHA-256으로 결합해 CPU generator에 준다. 전역 RNG를 소비하지 않으며, model/data training seed를 반복하는 실험이 아니다.

각 scene에서 encoder는 한 번 실행한다. 정규화 Score는 production Hilbert layout에서 한 번 분석한 뒤 slot 순서로 복원해 모든 순서에 공유한다. 채널별 정규화 표준편차와 XYZ RMS 반경도 모든 순서에 공통으로 적용하며, 이 값은 관찰 통계에만 쓰고 predictor나 entropy model 입력으로 전달하지 않는다.

먼저 production `codec.compress` packet이 원래 T05의 SHA-256과 같은지 확인한다. 이어서 고정 Score로 직접 구성한 Hilbert packet이 production packet과 byte 단위로 같은지 검사한다. 하나라도 다르면 해당 scene의 다른 순서 분석을 시작하지 않는다. 원본 입력 fingerprint, frozen backbone hash, checkpoint SHA, source 코드와 환경도 검사한다.

## 실제 부호화와 관찰

고정 Score의 행을 순열로 바꾼 뒤 기존 Score entropy encoder를 호출한다. predictor는 원래와 같은 네 D4 단계와 channel group을 사용하고, 앞 단계에서 실제로 복호한 anchor에 조건화한다. observer가 읽은 Score 복원과 별도 실제 Score decoder의 결과가 정확히 일치해야 한다.

MSH에는 각 순서에서 실제 복호한 Score의 LR 복원값을 사용해 residual을 새로 계산한다. 따라서 순서는 Score와 MSH 양쪽에 적용된다. 전체 packet도 실제 receiver로 복호하며, feature 오차를 원래 slot 순서로 복원해 동일한 F와 비교한다. 내부 decode의 인과 context를 대체하거나 encoder-only 정답 anchor를 공급하지 않는다.

다음 값을 저장한다.

- 같은 F·Score·XYZ의 fingerprint와 순열 fingerprint.
- 단계별 target token 수, actual raw entropy bytes, bits/token, 예측 잔차 MAE/MSE, 정규화 잔차·양자화 오차.
- 실제로 참조한 복호 anchor와 target 사이의 XYZ 거리와 Score 차이.
- raw Score, MSH y/z, 모든 header, 총 packet bytes와 feature 복원 오차.
- 각 순서의 packet 원본 및 SHA-256.

Score 내부 wrapper와 MSH/scene container를 모두 header에 넣어 중복 계산을 피한다. 단계별 raw bytes 합계는 순수 Score entropy payload와 같아야 하며, 구성 byte 합계도 전체 packet 길이와 같아야 한다. 실행 전후 codec의 모든 weight/buffer가 변하지 않았는지도 검사한다.

기본 trace는 첫1장, slot/Morton/Hilbert/random0의4개 순서다. 정확한 anchor index·유효 mask·복호 anchor 입력·target·prediction·baseline·reconstruction·quantization step 등을 저장한다. 범위는 `FIXED_ORDER_TRACE_SCENES`로 바꿀 수 있지만 다른 trace protocol은 새 output을 사용한다.

## 결과와 해석

```text
outputs/nfcgs_fixed_order/job_<JOB_ID>/REPORT.md
outputs/nfcgs_fixed_order/job_<JOB_ID>/summary.json
outputs/nfcgs_fixed_order/job_<JOB_ID>/metadata.json
outputs/nfcgs_fixed_order/job_<JOB_ID>/scenes/scene_XXXX.json
outputs/nfcgs_fixed_order/job_<JOB_ID>/packets/scene_XXXX/<ORDER>.bin
outputs/nfcgs_fixed_order/job_<JOB_ID>/traces/scene_0000.pt
outputs/nfcgs_fixed_order/job_<JOB_ID>/complete.json
logs/slurm/slurm-gs-fixed-order-<JOB_ID>.out/.err
```

```bash
cat outputs/nfcgs_fixed_order/job_<JOB_ID>/REPORT.md
```

random10은 장면 안에서 먼저 평균한다. Hilbert 대비 paired 차이의95% 구간은1000회 scene bootstrap으로 계산하며,2560개 random 장면을 서로 독립인 표본으로 취급하지 않는다. 별도로 각 random seed의256장 평균 total bytes도 남긴다. 이 구간은 scene 변동이며 training seed 불확실성이 아니다.

모든 순서에 하나의 Hilbert 학습 모델을 그대로 사용하므로, 비-Hilbert 순서는 predictor의 학습 분포를 바꾼 진단이다. 순서 변경이 이 frozen model의 잔차·bytes에 미치는 영향과 공간적 근접성의 연결을 관찰할 수 있다. 정렬별 재학습 RD, PSNR 개선이나 전체6991장 성능으로 확대하지 않는다. 이 job은 렌더·PSNR을 측정하지 않는다.

## 중단 후 재개

같은 protocol/code/checkpoint/cache로만 재개한다. 새 할당에서도 T05와 GPU 모델·소프트웨어·수치 정책이 같아야 하며, GPU UUID는 달라도 된다. 완료 scene JSON, packet SHA 및 trace SHA를 대조한 뒤 재사용한다. 원본 입력·기존 결과는 덮어쓰지 않는다. 미완료 scene의 중간 packet은 그 scene을 다시 계산하면서 갱신한다.

```bash
FIXED_ORDER_OUTPUT=/ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_fixed_order/job_<OLD_ID> \
FIXED_ORDER_RESUME=1 sbatch scripts/slurm/analyze_nfcgs_fixed_order.slurm
```

강제 종료로 `.analysis-lock`이 남으면 이전 job이 종료된 것을 확인한 후 해당 빈 lock만 제거한다. output을 지정하지 않은 새 제출은 새 job ID의 디렉터리를 사용한다. 원본 cache와 T05 결과는 분석·재개가 끝날 때까지 유지한다.

## 로컬 검증

```bash
python -m unittest discover -s tests -p test_fixed_order.py -v
```

10개 테스트가 통과했다. 실제 작은 entropy codec으로 FP32/CPU BF16의 production packet 완전 일치,13개 순열, 동일 Score 추출 횟수, 복호 anchor trace, 단계별 byte 합계, 공통 정규화, random의 scene 내 평균 및 paired bootstrap, 부분 재개·최종화 재개·packet 변조 탐지, 원본 cache 보존, plan의 CUDA/대용량 tensor 미접근, LF/Bash와 Slurm 인자 전달을 검증했다. 실제4096-token CUDA/BF16 checkpoint 진단은 서버 job에서 수행한다.

별도로 초기화한 모델과 작은 합성 입력의4096×736·rank56 CPU BF16 smoke에서도 Hilbert packet의 production 완전 일치와 random0의 복호·byte 합계를 확인했다. D4 token 수는512/512/1024/2048, 모든 header는284B다. 이 초기화 fixture는 신호가 작아 두 순서가 같은 packet으로 양자화됐으므로 정렬 효과나 실제 학습 모델의 rate 근거로 사용하지 않는다. [검증 기록](analysis/nfcgs_logs_2026-10-20_1051/local_fixed_order_fullshape_smoke.json).

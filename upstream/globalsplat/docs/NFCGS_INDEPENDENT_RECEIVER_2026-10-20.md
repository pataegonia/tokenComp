# T15 독립 packet 수신 검증

완료된 T05 job458004의 실제 packet 256개를 새 Python 프로세스에서 복호하고 Gaussian 생성·렌더·지표 계산까지 검증한다. 추가 학습과 재인코딩 없이 실행하는 GPU1 평가 job 하나다. 기존 학습이나 연결 평가의 완료에 의존하지 않는다.

## 반영 파일과 실행

다음 두 새 파일을 서버의 같은 repo 상대 경로에 반영한다. 기존 학습 코드, T05 코드, manifest 및 profile helper는 수정하지 않았다.

- [scripts/eval_nfcgs_independent_receiver.py](../scripts/eval_nfcgs_independent_receiver.py)
- [scripts/slurm/eval_nfcgs_independent_receiver.slurm](../scripts/slurm/eval_nfcgs_independent_receiver.slurm)

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
mkdir -p logs/slurm
bash scripts/slurm/eval_nfcgs_independent_receiver.slurm plan && \
  sbatch scripts/slurm/eval_nfcgs_independent_receiver.slurm
```

`plan`은 원본 완료 marker, 256개 scene JSON 및 packet, checkpoint SHA-256과 기존 runtime 코드 해시를 읽기 전용으로 검사한다. 로그인 노드에서는 CUDA나 `/data3/local_datasets/re10k`를 요구하지 않는다. 실제 데이터셋 검사는 계산 노드에서 수행한다.

자원 설정은 GPU1·CPU8·메모리48G·시간 제한8시간이다. 실제 소요 시간 추정은 아니다. TMPDIR은 `/ceph_data/<USER>/.tmp/nfcgs/independent_receiver_<JOB_ID>`다.

## 고정 조건

- 원본 packet: `outputs/nfcgs_path_diagnostics/job_458004/packets/scene_XXXX.bin`.
- 기준 지표: 같은 run의 `scenes/scene_XXXX.json` 안 `modes.actual_packet`.
- checkpoint: H+D4 ON16, rank56, minimal linear, λ=.0128, job447746의 최종50k.
- 4096 tokens, appearance512+geometry224, token당 Gaussian8개, 최종 Gaussian32,768개.
- 원래 C12/T8의 matched256 subset, 256×256, seed0, BF16.
- 원본 입력 fingerprint: `069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29`.
- T05와 동일한 torch·CompressAI·NumPy·CUDA runtime·GPU 모델 및 수치 정책을 요구한다. GPU UUID는 기록하지만 동일할 필요는 없다.
- CuDNN deterministic ON, benchmark OFF, TF32 OFF. 전역 deterministic algorithms를 추가로 강제하지 않는다.

기존 `eval_nfcgs_sender_receiver.py`는 과거 baseline의 mean-offset 설정과 전체 test 실행에 맞춰져 있다. 이번 검증은 현행 T05의 저장된 설정을 그대로 strict-load하며, 기존 script를 변경하지 않는다.

## 두 프로세스의 역할

첫 번째 `prepare` 프로세스는 모델을 만들지 않고 고정 test loader에서 T05와 같은 target을 찾는다. T8 frame ID와 target 전체 tensor fingerprint가 같아야 한다. 해당 fingerprint에는 target RGB와 카메라가 포함된다. packet도 원본 SHA-256과 stream byte 합계를 검사하고 byte 그대로 복사한다.

내보내는 bundle은 다음으로 구성된다.

- 실제 packet 파일.
- 각 장면의 `scene`, `index`, `targets`만 담은 파일. context 영상·encoded feature·정렬 좌표/순열·Gaussian tensor는 포함하지 않는다.
- checkpoint/config/protocol 및 원본 지표가 담긴 metadata와 manifest, 완료 marker.

첫 프로세스가 완전히 종료된 뒤 두 번째 `receive` 프로세스가 시작한다. 수신 프로세스는 bundle과 공유 checkpoint만 연다. 원본 데이터셋·입력 cache·T05 출력 디렉터리를 다시 열지 않는다. 원본 경로가 metadata에 남더라도 provenance 용도다.

수신 모델은 `ObservableLowRank1DCodec`과 `TokenCoarseToFine3DGS` 두 모듈만 만든다. GlobalSplat encoder, 입력 embedding, scene-token bank를 생성하지 않는다. 공유 checkpoint의 해당 두 branch를 strict-load하고, mean-offset 설정 및 CDF 갱신 정책을 T05와 맞춘다. checkpoint 자체는 통합 파일이므로 CPU에서 전체 파일을 읽지만, encoder를 생성하거나 GPU에 올리지 않는다.

Gaussian 복호 함수의 입력은 packet bytes뿐이다. 렌더에는 target 카메라와 영상 크기를 나타내는 zero placeholder를 전달한다. 실제 target RGB는 metric 계산에만 사용한다. 이 구성에서 per-scene 정렬 좌표나 역순열이 필요한지 함께 확인한다.

## 성공 기준과 해석 범위

장면별 기준 지표와 독립 수신 지표의 최대 절댓값 차이를 다음 사전 지정 허용치와 비교한다.

| 지표 | 최대 장면별 절댓값 차이 허용치 |
|---|---:|
| PSNR | 0.001 dB |
| SSIM | 0.00001 |
| LPIPS | 0.00001 |

각 장면에서 동일 packet을 두 번 복호한 feature와 Gaussian field는 정확히 일치해야 한다. 동일 Gaussian을 두 번 렌더한 pixel 최대 차이는 별도로 기록한다. 원본 T05의 pixel과 feature는 저장하지 않았으므로, 이번 지표 정합을 원본 pixel/feature와의 bit-exact 증명으로 표현하지 않는다.

packet SHA·차원·stream byte 합계, checkpoint SHA·설정·branch strict load, target 파일 SHA·frame·fingerprint를 검사한다. runtime 코드와 bundle이 실행 중 바뀌어도 성공 marker를 쓰지 않는다. 지표 비교가 실패하면 FAIL 보고서는 남기고 비정상 종료한다.

이 실험은 matched256의 전송·독립 복호 검증이다. 전체6991장 RD, 구조 모델 우열, 학습 개선, receiver 최소 메모리 또는 latency 결과로 사용하지 않는다. checkpoint는 공유 모델 자산으로 보고 per-scene rate에서 제외한다. packet을 그대로 복사하므로 평균 용량은 T05의 약37.2592 KiB/scene과 같아야 한다.

## 결과 확인

```text
outputs/nfcgs_independent_receiver/job_<JOB_ID>/bundle/
outputs/nfcgs_independent_receiver/job_<JOB_ID>/receiver/REPORT.md
outputs/nfcgs_independent_receiver/job_<JOB_ID>/receiver/summary.json
outputs/nfcgs_independent_receiver/job_<JOB_ID>/receiver/scenes/scene_XXXX.json
outputs/nfcgs_independent_receiver/job_<JOB_ID>/receiver/metadata.json
outputs/nfcgs_independent_receiver/job_<JOB_ID>/receiver/complete.json
logs/slurm/slurm-gs-independent-receiver-<JOB_ID>.out
logs/slurm/slurm-gs-independent-receiver-<JOB_ID>.err
```

```bash
cat outputs/nfcgs_independent_receiver/job_<JOB_ID>/receiver/REPORT.md
```

모든 장면과 비교가 통과하면 stdout에 `INDEPENDENT_RECEIVER_COMPLETE scenes=256`을 출력하고 `complete.json`을 생성한다. 기존 결과 디렉터리는 덮어쓰지 않는다. 경로를 바꾸려면 `INDEPENDENT_RECEIVER_PATH_RUN`, `INDEPENDENT_RECEIVER_OUTPUT`, `DATASET_ROOT`, `INDEPENDENT_RECEIVER_NUM_WORKERS`를 사용할 수 있다. checkpoint와 비교 프로토콜은 고정한다.

## 로컬 검증

```bash
python -m unittest discover -s tests -p test_independent_receiver.py -v
```

11개 테스트에서 실제 작은 entropy codec과 실제 Gaussian decoder를 사용한다. FP32·CPU BF16의 production 복호 일치, mean-offset 두 설정, 대상-only bundle, 원본 경로 삭제 후 별도 Python 프로세스 복호, packet/target 손상·manifest 변조·weights 누락·잘못된 checkpoint·환경/코드 차이 거부, 지표 실패 시 성공 marker 미생성, 별도 Slurm 두 프로세스와 LF/Bash 구문을 검증했다. 로컬 통합 테스트의 renderer/metric은 CUDA 대신 작은 fixture를 사용하며, 실제 gsplat 렌더와256장 검증은 서버 job에서 수행한다.

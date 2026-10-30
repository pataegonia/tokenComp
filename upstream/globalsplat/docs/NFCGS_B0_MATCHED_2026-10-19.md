# B0 비압축 기준선 — 기존 256장과 정합

공개 GlobalSplat의 native geometry512 경로를 기존 256개 장면·C12/T8 카메라·BF16 조건으로 평가한다. 같은 job에서 현행 H+D4 λ=.0128 checkpoint의 B1도 계산한다. B0→B1의 변화는 이 job 안에서 직접 비교할 수 있으며, 앞서 준비한 T05 job의 완료를 기다리지 않는다.

## 모델 경계와 고정 조건

| 경로 | encoder | GaussianDecoder 입력 | decoder |
|---|---|---|---|
| `b0_native` | 공개 원본 | appearance512 + 원본 geometry512 | 공개 native geometry512 decoder |
| `b1_bridge` | 같은 encoder 출력 재사용 | appearance512 + 학습된 projection의 geometry224 | 현행 geometry224 decoder |

B0는 실제 공개 checkpoint를 codec 없는 모델에 strict load한다. B0 입력에 projection·LR·MSH·양자화를 적용하지 않는다. 현행 checkpoint는 B1과 공통 Hilbert 순서 계산에만 사용한다. native model에서 단순히 codec을 끈 현행 checkpoint를 읽는 방식이 아니다.

- 공개 checkpoint: `checkpoints/pretrained/globalsplat-re10k-32k.ckpt`
- 고정 SHA-256: `90be09504fa046792dedfe2d1b17e00bb64a5bf95da477d71e233be01df72a42`
- 현행 기준: H+D4 λ=.0128, 최종 50k, job447746. 기존 T05 manifest를 그대로 읽는다.
- 4096 tokens, stage3·mix1, token당 Gaussian8개, 총32768개.
- 기존 job454995의 256개 입력 cache와 fingerprint를 재사용한다. cache 불일치·누락 시 대체 sampling 없이 종료한다.
- C12/T8, 256×256, eval seed0, BF16, threads8, workers4, GPU1.
- T05와 같은 CuDNN deterministic·benchmark OFF·TF32 OFF 정책. 모델 import 후 적용한다.

encoder의 모든 가중치·buffer와, geometry readout 세 weight를 제외한 공통 decoder tensor가 공개 원본과 현행 checkpoint에서 정확히 같아야 한다. 세 readout의 effective map `W224 @ P`와 원본 `W512`의 차이도 기록한다. 이 행렬 차이를 PSNR 손실에 대한 독립적인 인과 기여도로 해석하지 않는다.

## 정렬과 수치 차이

두 주 경로에는 현재 H+D4가 정하는 같은 token 순서를 적용한다. 순서 계산에는 현행 geometry projection·center를 사용하지만, B0 decoder에는 순서만 바뀐 원본 geometry512 값을 전달한다.

별도로 공개 모델의 원래 slot 순서도 decode/render하여 `b0_original_slots_check`로 기록한다. Gaussian8개 묶음을 맞춘 tensor 차이, 원래 slot 순서와 공통 정렬의 pixel 차이를 측정한다. 각 주 경로에서도 동일 Gaussian·카메라를 재렌더링해 반복 오차를 기록한다. 이 검사를 GPU pixel bit-exact 보장으로 확대하지 않는다.

입력 cache에는 과거 target tensor가 없으므로 이번에 읽은 T8 frame ID와 target fingerprint를 새로 기록한다. B0/B1에는 같은 target tensor를 사용하며, 이후 T05와 연결할 때 target까지 실제로 대조한다.

## 서버에 추가할 파일

앞서 전달한 T05의 네 실행 파일이 repo에 반영된 상태에서 다음 두 파일을 추가한다. 기존 T05 코드·manifest·Slurm 및 학습 코드는 수정하지 않는다.

- [B0 평가 및 결과 결합 runner](../scripts/eval_nfcgs_b0_matched.py)
- [B0 Slurm job](../scripts/slurm/eval_nfcgs_b0_matched.slurm)

의존하는 기존 T05 파일은 `globalsplat/compression/path_diagnostics.py`, `scripts/eval_nfcgs_path_diagnostics.py`, `config/plans/nfcgs_path_diagnostics_20261019.json`, `scripts/slurm/eval_nfcgs_path_diagnostics.slurm`이다. 기존 profile helper도 재사용한다.

## 제출

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
mkdir -p logs/slurm
sbatch scripts/slurm/eval_nfcgs_b0_matched.slurm
```

B0 평가 job 1개를 추가한다. T05 job과 병행할 수 있고 afterok 의존성을 걸지 않는다. Slurm은 GPU1·메모리48G·8시간 한도이며, 8시간은 소요 시간 예측이 아니다. 데이터셋은 계산 노드에서 검사하고, Ceph의 job별 TMPDIR을 사용한다.

로그인 노드에서 대용량 모델·cache tensor를 읽지 않고 경로·원본 분석 metadata를 점검하려면 다음을 실행한다.

```bash
bash scripts/slurm/eval_nfcgs_b0_matched.slurm plan
```

결과 경로는 `outputs/nfcgs_b0_matched/job_<JOB_ID>/`이며 `REPORT.md`, `summary.json`, `metadata.json`, `scenes/scene_XXXX.json`, `complete.json`을 생성한다. checkpoint SHA-256·가중치 정합·frame ID·입력/target/순열 fingerprint를 보관한다. 요청 장면이 모두 끝나고 코드 해시가 유지된 경우에만 complete를 기록한다. 기존 결과 디렉터리에는 덮어쓰거나 resume하지 않는다.

## T05와 결합

두 job이 완료된 뒤 GPU가 필요 없는 비교 명령을 실행한다. `<B0_JOB_ID>`와 `<T05_JOB_ID>`를 실제 번호로 바꾼다.

```bash
python scripts/eval_nfcgs_b0_matched.py \
  --compare-only \
  --b0-run outputs/nfcgs_b0_matched/job_<B0_JOB_ID> \
  --path-run outputs/nfcgs_path_diagnostics/job_<T05_JOB_ID> \
  --output outputs/nfcgs_b0_t05_comparison/b0_<B0_JOB_ID>_t05_<T05_JOB_ID>
```

completion·metadata 정합, 현행 checkpoint SHA-256, 원본 입력 목록·fingerprint, 공통 코드 해시, 정밀도·backend 정책, Torch/CompressAI/NumPy/CUDA 버전과 GPU 종류를 검사한다. GPU UUID는 같음을 요구하지 않고 기록한다. 각 scene의 context/target frame ID·입력/target tensor fingerprint·순열 fingerprint도 같아야 한다. 불일치가 있으면 자동 결합하지 않는다.

같은 장면에서 새 B1 bridge와 T05의 `b1_direct` 사이 PSNR 차이도 평균·최대 절댓값으로 출력한다. 이 값과 양쪽 renderer 반복 오차를 함께 보고 cross-job 차이를 판단한다. 가중치와 입력이 같다는 이유만으로 서로 다른 GPU에서 pixel bit-exact라고 표시하지 않는다.

비교 결과는 `REPORT.md`, `summary.json`, `paired_scenes.json`으로 저장한다. B0/B1은 packet rate가 없으므로 RD 점으로 만들지 않는다. 과거 24.7004dB를 이번 측정값 대신 사용하거나, B0→B1과 B1→actual 차이를 서로 독립적인 손실 기여도로 더하지 않는다. 256장 결과는 전체6991장 결과와 구분한다.

## 로컬 검증

공개 원본 파일의 SHA-256과 실제 모델 tensor454개의 strict load를 확인했다. 실제 native decoder에 4096개의 합성 token을 넣어 stage3에서 Gaussian32768개가 생성됨을 CPU FP32/BF16에서 확인했다. 원래 순서와 순열의 Gaussian8개 묶음을 대조했을 때 CPU FP32 최대 오차는 약1.91e−6, CPU BF16에서는 이 입력의 오차가0이었다. 이는 실제 CUDA 렌더 검증이나 학습된 장면의 성능 결과가 아니다.

단위·통합 테스트는 [tests/test_b0_matched.py](../tests/test_b0_matched.py)에 있으며 native geometry 경로, 공통 가중치 검사, cache 정합, frame 불일치 거부, B1 bridge 차이, Slurm 실행 경로를 검증한다.

```bash
python -m unittest discover -s tests -p test_b0_matched.py -v
```

실제 GPU·RE10K 평가는 서버 제출 후 확인한다.

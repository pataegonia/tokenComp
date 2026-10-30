# T04 B1 / T05 경로·양자화 진단

현재 Hilbert+D4 λ=.0128의 최종 50k checkpoint를 고정하고, 기존 분석과 같은 256개 test scene에서 7개 경로를 비교한다. 추가 학습 없이 현행 projection 이후의 비압축 복원, rank56 경로 및 양자화 우회에 따른 복원 차이를 측정한다. 원본 geometry512 모델의 B0 평가는 이 job에 포함하지 않는다.

## 고정 조건

- checkpoint: `outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/lambda0p0128/job_447746/checkpoints/nfcgs_main/version_0/step000050000.ckpt`
- Hilbert, D4 ON16, minimal linear, rank56, MSH N192/M320/adapter96.
- 4096 tokens, `F736 = appearance512 + projected geometry224`.
- C12/T8, 256×256, eval seed0, BF16, GPU1, threads8, workers4.
- 기존 architecture smoke와 같은 CuDNN deterministic·benchmark OFF·TF32 OFF 정책을 적용한다. renderer 전체에 deterministic algorithm을 강제하지는 않는다.
- 원본 입력: `outputs/nfcgs_legacy3_anchors/job_454995/inputs.pt`.
- 원본 입력 fingerprint: `069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29`.

원본 입력 cache의 전체 fingerprint와 장면 목록을 먼저 검증한다. 데이터셋에서 같은 장면을 읽은 뒤 입력 영상·카메라·frame ID를 포함한 입력 tensor를 cache와 다시 대조한다. cache가 없거나 다르면 새 장면으로 대체하지 않고 종료한다. 원본 분석은 target tensor를 저장하지 않았으므로, 이번에 읽은 고정 T8의 frame ID와 tensor fingerprint를 새로 저장하고 7개 경로에 동일하게 사용한다. 입력과 target frame은 서로 겹치지 않아야 한다.

frozen encoder/GaussianDecoder의 SHA-256도 원본 `backbone.json`과 일치해야 한다. geometry projection은 현재 checkpoint의 학습된 값을 그대로 사용한다. 모든 진단 경로가 같은 정렬 순서를 사용하므로 경로별 renderer 순열 차이를 도입하지 않는다.

## 경로 정의

`s`는 연속 Score, `ŝ`는 실제 packet에서 인과적으로 복호한 Score, `L`은 학습된 LR synthesis다. `Hhat`은 실제 MSH entropy encode/decode 복원이며, `Hcont`는 y/z 양자화를 모두 우회한 MSH 분석·합성 경로다.

| 모드 | GaussianDecoder 입력 | 목적 |
|---|---|---|
| `b1_direct` | F | 현행 projection 이후 비압축 기준 B1 |
| `lr_continuous` | L(s) | 연속 rank56 경로의 복원 수준 |
| `lr_decoded` | L(ŝ) | residual 없이 실제 복호 Score만 사용 |
| `score_quantization_bypass` | L(s)+Hhat(F−L(s)) | Score 양자화 우회, residual 재계산 |
| `msh_quantization_bypass` | L(ŝ)+Hcont(F−L(ŝ)) | MSH y/z 양자화 우회 |
| `both_quantization_bypass` | L(s)+Hcont(F−L(s)) | 양쪽 양자화 우회, residual 재계산 |
| `actual_packet` | L(ŝ)+Hhat(F−L(ŝ)) | 실제 scene packet 압축·복호 기준 |

`Hcont`는 residual normalization, analysis adapter, `g_a`, `g_s`, synthesis adapter, crop 및 역정규화를 모두 유지한다. y를 그대로 synthesis에 전달하면 z와 hyper-synthesis의 entropy parameter는 복원값에 영향을 주지 않는다. 따라서 이 우회는 identity가 아니며 분석·합성 손실이 남는다.

실제 packet을 먼저 만든 뒤 그 packet의 Score를 복호한다. 근사 eval forward나 단순 round를 실제 ŝ의 대용으로 사용하지 않는다. 실제 packet의 전체 복호 결과와 `L(ŝ)+Hhat(F−L(ŝ))` 수동 분해가 tensor 단위로 정확히 일치해야 한다. Score 우회용 MSH도 변경된 residual을 새로 encode/decode하고 내부 sender/receiver 복원을 대조한다.

Score predictor와 MSH의 부호화·복호 수치 재현을 위해 위 backend 정책을 7개 경로에 공통 적용한다. `globalsplat.main` import가 backend 기본값을 바꾸므로 모델을 읽은 뒤 다시 설정한다. 과거 전체 test의 TF32 정책과 차이가 있을 수 있으므로 이번 actual 값이 과거 RD 수치의 재현이라고 단정하지 않는다.

## 서버 반영 파일

기존 학습 코드와 기존 profile helper는 수정하지 않았다. 다음 네 파일을 동일한 repo 상대 경로에 반영한다.

- [진단 경로 모듈](../globalsplat/compression/path_diagnostics.py)
- [평가 runner](../scripts/eval_nfcgs_path_diagnostics.py)
- [checkpoint manifest](../config/plans/nfcgs_path_diagnostics_20261019.json)
- [Slurm 평가 job](../scripts/slurm/eval_nfcgs_path_diagnostics.slurm)

기존 서버의 `scripts/profile_nfcgs_complexity.py`와 GlobalSplat 코드·환경을 재사용한다. 단위·통합 검증은 [tests/test_path_diagnostics.py](../tests/test_path_diagnostics.py)에 있다.

## 실행

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
mkdir -p logs/slurm
sbatch scripts/slurm/eval_nfcgs_path_diagnostics.slurm
```

새 평가 job 1개를 제출한다. 기존 26개 학습·연결 평가의 제출이나 취소는 수행하지 않는다. 결과 경로는 `outputs/nfcgs_path_diagnostics/job_<JOB_ID>/`다. Slurm은 GPU1·메모리48G·8시간 한도로 설정했으며, 이 시간은 실제 소요 시간 예측이 아니다. `/data3/local_datasets/re10k`는 계산 노드 안에서 검사한다. TMPDIR은 기존 방식과 같이 Ceph의 job별 경로를 사용한다.

제출 전 로그인 노드에서 파일 경로와 원본 분석 metadata만 점검하려면 다음 명령을 사용할 수 있다. CUDA·데이터셋·대용량 cache tensor의 실제 검사는 계산 job에서 수행한다.

```bash
bash scripts/slurm/eval_nfcgs_path_diagnostics.slurm plan
```

`PATH_DIAGNOSTIC_INPUT_RUN`, `PATH_DIAGNOSTIC_OUTPUT`, `DATASET_ROOT`, `PATH_DIAGNOSTIC_NUM_WORKERS`로 경로 또는 worker 수를 지정할 수 있다. 입력 fingerprint는 고정한다. 기존 결과 디렉터리를 덮어쓰거나 자동으로 resume하지 않는다.

## 결과와 검증 범위

- `REPORT.md`, `summary.json`: 동일 장면 평균 PSNR/SSIM/LPIPS, actual 및 B1 대비 paired PSNR 차이, feature MSE.
- `scenes/scene_XXXX.json`: frame ID, 입력·target·순열 fingerprint, 각 경로의 texture/geometry MSE, packet SHA-256과 stream bytes.
- `packets/scene_XXXX.bin`: 실제 `actual_packet` bitstream.
- `metadata.json`: checkpoint SHA-256·50k step, 모델 설정, 원본 cache provenance, 코드 해시, GPU UUID·정밀도·수치 설정.
- `complete.json`: 요청한 256장이 모두 끝나고 코드 변경 여부까지 검사한 경우에만 생성.

PSNR은 production 평가와 같이 target view별 값의 평균을 구한 뒤 scene별로 같은 가중치를 주어 집계한다. 원본 분석과 target frame의 과거 동일성을 새로 검증했다고 해석하지 않는다. 현재 job의 7개 경로 사이에는 같은 target tensor를 사용한다.

전송 가능한 용량은 `actual_packet`에만 표시한다. 다른 경로는 rate를 `null`/`n/a`로 남겨 RD 점으로 사용하지 않는다. Score stream bytes에는 내부 SCCTX001 wrapper가 포함된다. 순수 Score payload와 전체 header bytes는 wrapper를 분리해 별도로 기록한다. D4 ON16의 wrapper는 152B다.

actual 경로에서 같은 Gaussian·카메라를 한 번 더 렌더링해 pixel 최대 차이와 PSNR 차이를 기록한다. 이는 해당 GPU의 renderer 반복 오차 점검이며, 순열 전반의 bit-exact 증명이나 독립 receiver 검증은 아니다. 이 job은 같은 프로세스에서 packet을 복호한다.

경로 간 PSNR 차이를 더해 독립적인 손실 기여도로 해석하지 않는다. B1은 native geometry512의 B0와 다르며, 과거 24.7004dB와의 차이를 현행 codec의 확정 손실로 사용하지 않는다. 256장 진단 결과를 전체 6991장 성능으로 표시하지 않는다.

## 로컬 검증

실제 CUDA·RE10K·학습 checkpoint는 로컬에 없으므로 서버 평가 결과는 아직 없다. 로컬 테스트 11개에서 인과적 packet 복원, residual 재계산, adapter·normalization 보존, CPU FP32/BF16, cache 불일치 거부, 출력 rate·provenance 및 Slurm 실행 경로를 검증한다. 별도로 초기화한 실제 규격 codec의 `(1,4096,736)` 7개 경로도 CPU FP32/BF16에서 확인했다. 합성 입력의 크기·오차는 학습된 모델 성능으로 사용하지 않는다.

```bash
python -m unittest discover -s tests -p test_path_diagnostics.py -v
```

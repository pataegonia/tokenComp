# 단일 MSH vs. Base MSH + Residual MSH

두 실험 모두 Morton ON, λ=0.0256, 50,000 optimizer step으로 새로 학습한다.
주 경로와 residual 경로는 각각 독립적인 mean-scale hyperprior(MSH)다.

```text
single_msh:
  x -> MSH_base -> x_hat

base_residual_msh:
  x -> MSH_base -> b_hat
  r = x - b_hat
  r -> MSH_residual -> r_hat
  x_hat = b_hat + r_hat
```

여기서 x는 geometry projection 224채널과 texture 512채널을 합친 736채널 feature를
Morton 정렬한 값이다. 이번 실험은 `input_norm=none`이다. 정렬은 두 경로에 공통으로
한 번 적용하며, residual MSH에는 동일한 순서의 복원 오차가 들어간다.

두 MSH는 각각 `g_a/g_s`, `h_a/h_s`, entropy bottleneck, GaussianConditional을 가진다.
가중치를 공유하지 않는다. 양쪽 `h_a` 입력은 signed y이며, 양자화된 z에서 y의 mean과
scale을 예측한다. 실제 압축 시 첫 경로의 y/z를 실제로 복호화한 `b_hat`으로 residual을
계산한다. 따라서 수신기가 가진 base 복원값과 송신기의 residual 기준이 일치한다.

학습은 두 MSH를 처음부터 함께 최적화한다. residual 계산에 `detach`를 넣지 않는다.
기존 학습과 같이 학습 중에는 양자화 noise surrogate를, 평가 시에는 실제 양자화를 쓴다.
별도 base reconstruction loss는 추가하지 않는다. 두 경로의 역할과 비트 배분은 학습된다.

```text
L = 기존 rendering/subset distortion
    + 0.0256 * (R_base_y + R_base_z + R_residual_y + R_residual_z) / (Gaussians * attributes)
```

단일 MSH에서는 y/z 두 항만 합산한다. 실제 bytes는 모든 entropy payload와 컨테이너를
포함하며, 각 경로의 y/z를 validation/test JSON에 따로 기록한다.

## 설정

| Array task | 버전 | MSH 블록 | 학습 step |
| ---: | --- | --- | ---: |
| 0 | `single_msh`, `paths=1` | legacy N=192/M=320 × 1 | 50,000 |
| 1 | `base_residual_msh`, `paths=2` | legacy N=192/M=320 × 2 | 50,000 |

각 블록은 Adapter 포함 기존 legacy 구조, 4× downsampling을 사용한다.
단일 버전의 codec parameter는 4,491,808개, 두 경로 버전은 8,868,928개다.
geometry projection은 두 경로가 공유하며 encoder/generator는 고정한다.
따라서 이 비교는 residual 경로와 함께 parameter 및 연산량도 늘어나는 실험이다.

- 원본 vanilla checkpoint에서 각각 초기화, seed 111123. Base MSH 초기 가중치는 동일하다.
- batch 2 × accumulate 4, Adam lr=1e-4, warmup 1K 후 일정한 lr.
- checkpoint 500 step마다, 고정 validation 2K step마다 및 학습 종료 시.
- ariel-v9, 각 작업 GPU 1개, Slurm 제한 각 6일.
- 기본 array는 `0-1%2`: 두 작업을 병렬 제출한다. 작업당 GPU 1개, 최대 동시 GPU 2개다.

## 서버 반영 파일

아래 파일을 `upstream/globalsplat` 기준 같은 상대 경로에 복사한다.
기존 `scripts/slurm/train_hyper1d_12h.slurm`도 서버에 있어야 한다.

- [hyper1d.py](../globalsplat/compression/hyper1d.py)
- [hyper1d_config.py](../globalsplat/compression/hyper1d_config.py)
- [bitstream.py](../globalsplat/compression/bitstream.py)
- [checkpoint.py](../globalsplat/compression/checkpoint.py)
- [globalsplat_hyper1d.yaml](../config/model/globalsplat_hyper1d.yaml)
- [initialize_hyper1d_from_vanilla.py](../scripts/initialize_hyper1d_from_vanilla.py)
- [run_hyper1d.py](../scripts/run_hyper1d.py)
- [train_hyper1d_msh_compare_50k.slurm](../scripts/slurm/train_hyper1d_msh_compare_50k.slurm)

```bash
cd /ceph_data/clue9986/CleanToken/upstream/globalsplat
ls -lh checkpoints/pretrained/globalsplat-re10k-32k.ckpt
mkdir -p logs/slurm
sbatch scripts/slurm/train_hyper1d_msh_compare_50k.slurm
```

위 기본 제출 명령으로 두 작업 모두 병렬 실행을 요청한다. 실제 시작 시점은 Slurm 자원
배정에 따른다. 이미 제출된 job의 동시 실행 제한은 로컬 파일 변경만으로 바뀌지 않는다.

이전의 `train_hyper1d_legacy_morton_50k.slurm`은 단일 MSH만 제출하는 파일이다.
두 버전 비교에는 위의 새 array 파일을 사용한다.

## 결과와 재시작

```text
outputs/hyper1d_msh_compare_50k/<single_msh|base_residual_msh>/morton_on/lambda0p0256/job_<array_id>_<task_id>/
  checkpoints/hyper1d_12h/version_0/step000050000.ckpt
  validation/step000050000.json
logs/slurm/slurm-gs-msh-compare50k-<array_id>_<task_id>.out
logs/slurm/slurm-gs-msh-compare50k-<array_id>_<task_id>.err
```

두 경로의 JSON 용량 항목은 `actual_base_y_bytes`, `actual_base_z_bytes`,
`actual_residual_y_bytes`, `actual_residual_z_bytes`, `actual_container_bytes`다.
이 다섯 항목의 합이 `actual_bytes`다. 새 컨테이너 magic은 `E2EH0402`이며 overhead는
172 bytes다. 단일 MSH의 기존 `E2EH0401` 포맷과 overhead 112 bytes는 유지한다.

중단된 실행은 해당 실행의 명시적인 `step*.ckpt`를 선택하여 재시작한다.
checkpoint에 저장된 `paths`가 자동으로 복원되므로 resume/eval에 `--paths`를 붙이지 않는다.

```bash
# RUN과 CKPT를 실제 실행 경로 및 저장된 step 파일로 지정한다.
sbatch --time=6-00:00:00 scripts/slurm/resume_hyper1d_12h.slurm "$CKPT" --output "$RUN"
```

최종 비교는 같은 test scene ID와 context/target frame ID에서 PSNR·LPIPS 및 총 bytes/BPGA로
진행한다. λ가 같아도 두 모델의 bitrate가 같다는 뜻은 아니다. 경로별 용량으로 어느 경로에
정보가 배분되는지도 확인한다. seed가 같아도 모델과 데이터 소비에 따른 학습 궤적은 다를 수 있다.

## 구현 검증

CPU에서 기존 단일 경로 테스트 39개와 dual/integration/Slurm 테스트 61개를 통과했다.
테스트에는 양쪽 MSH gradient, signed hyper-analysis, Morton 정렬 및 역순 복원,
가변 길이·stride, 양쪽 payload shape 사전 검증, checkpoint 저장·재로딩,
실제 validation/test 4-stream 용량 합계와 Slurm/Hydra 설정 검증이 포함된다.

추가로 실제 736채널·4096토큰·N=192/M=320의 두 경로 모델에서 CPU BF16
forward/backward의 finite gradient를 확인했다. FP32 실제 압축/복원과 평가 forward의
최대 오차는 `1.49e-7`이었다. 동일 seed의 단일/이중 경로 base 초기 가중치가 같고,
기존 plain4 초기화 checkpoint도 `paths=1`로 strict-load되는 것을 확인했다.
서버 GPU 학습과 최종 RD 성능은 제출 후 확인한다.

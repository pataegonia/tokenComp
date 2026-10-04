# Hyper1D 구현과 12시간 파일럿

2026-09-28. [설계 문서](NFCGS_HYPER1D_DESIGN_2026-09-28.md)에 따른 신규 코덱을 구현했다.
기존 main 코덱의 `codec.py`, `residual.py`, 모델 설정과 실행 스크립트는 유지한다.
초기 구현 검증은 CPU에서 실제 entropy coding, 모델 gradient, QR 초기화,
체크포인트 재시작과 평가 집계를 확인했다. 2026-09-29에 확인한 GPU 로그에서는
두 run이 16,000 step까지 완료했다. VRAM과 최종 수렴 여부는 아직 확인하지 않았다.

2026-09-29에 Adapter를 제거하고 transform 용량을 늘린 `plain4`를 추가했다.
새 vanilla 초기화의 기본값은 `plain4`이며, 기존 checkpoint는 저장된 설정으로 복원한다.
이전 metadata에 `architecture`가 없으면 `legacy`로 해석하고 tensor key/shape를 엄격히 검증한다.
아래 기존 GPU 결과는 N=192/M=320 + Adapter 구조의 결과이며, 새 구조의 학습 결과는 아니다.

검증 환경은 PyTorch 2.11.0 CPU / CompressAI 1.2.8 / Lightning 2.4.0이며,
전체 115개 테스트와 99개 subtest가 통과했다. 실제 원본
`checkpoints/pretrained/globalsplat-re10k-32k.ckpt`에서도 초기화를 완료했다.
QR 재매개화 최대 오차는 7.16e-7, 직교성 최대 오차는 2.99e-7 이하였으며,
기본 4096-token / 736-channel 코덱의 FP32 압축·복호 최대 오차는 1.05e-7 이하였다.
이 압축·복호 검사는 합성 feature를 사용했으며 렌더링 품질·수렴 결과는 아니다.
초기화 체크포인트와 검증 JSON은 로컬 `outputs/hyper1d/implementation_smoke/`에 저장했다.

## 구현

- `globalsplat/compression/hyper1d.py`: 512-D appearance와 224-D projected geometry를 직접
  부호화하는 `FeatureHyperprior1DCodec`. signed `h_a(y)`, stride `(2,2)` 또는 `(2,1)`.
  `plain4`는 N=256/M=512, Adapter 없음, 4층 g_a/g_s이고,
  `legacy`는 N=192/M=320, 두 Adapter, 2층 g_a/g_s다. scene mean과 score/context 경로는 없다.
- `Hyper1DConfig.codec_type=hyper1d`: 신규 체크포인트 종류를 명시한다. 종류 필드가 없는
  기존 main 체크포인트는 기존 로더로 처리한다. 신규 metadata는 모든 설정 필드를 저장한다.
  기존 Hyper1D checkpoint에 없는 `architecture` 필드만 `legacy`로 보완한다.
- `E2EH0401`: batch 1, 실제 points, SHA-256, y/z shape 교차 검증. 고정 오버헤드 112 B.
  실제 bytes는 `y + z + container`로 합산한다. Morton flag 이외의 비트는 거부한다.
- 변환은 학습 중 BF16 AMP를 사용할 수 있고 entropy likelihood 연산은 FP32다.
  실제 송수신은 geometry projection부터 FP32로 처리한다. forward와 실제 복호 비교도
  FP32에서 수행하며 부동소수점 반올림 오차를 허용한다.
- `f_mean/f_std`는 두 norm 모드에서 같은 key로 존재한다. `none`에 비항등 buffer가 저장되면
  로드를 거부한다. `calibrated`는 별도의 TRAIN 통계 산출 과정을 거친다.

## 첫 실행

### Adapter 없는 4층 모델

| 항목 | 기존 `legacy` | 신규 `plain4` |
| --- | --- | --- |
| Adapter | analysis/synthesis 각각 1개 | 없음 (`Identity`, 파라미터 0개) |
| N / M | 192 / 320 | 256 / 512 |
| g_a / g_s conv 층 수 | 각각 2층 | 각각 4층 |
| 기본 token downsampling | 4× | 4× |
| 전체 코덱 파라미터 (geometry projection 포함) | 4,491,808 | 9,361,120 (약 2.08배) |
| y / z (4096-token 입력) | 320×1024 / 192×256 | 512×1024 / 256×256 |

모든 transform kernel은 `(1,5)`다. stride-1 Conv를 추가해 토큰 길이를 더 줄이지 않고
표현력을 늘린다. `g_a`에는 GDN 3개, `g_s`에는 IGDN 3개가 있다.

```text
g_a: Conv(736→256,s2) → GDN → Conv(256→256,s1) → GDN
     → Conv(256→256,s2) → GDN → Conv(256→512,s1)
g_s: Conv(512→256,s1) → IGDN → Deconv(256→256,s2) → IGDN
     → Conv(256→256,s1) → IGDN → Deconv(256→736,s2)
```

`h_a/h_s`의 층 수와 signed-y 입력은 유지하되 N/M에 따라 채널도 커진다.
Morton OFF, input normalization 없음, geometry projection, frozen generator,
기존 rate-distortion loss와 학습 데이터는 유지한다. Adapter 제거와 용량 증가는 함께 바뀌므로
품질 차이를 Adapter 단독 효과로 해석할 수는 없다.

원본 pretrained GlobalSplat에서 새 코덱을 초기화해야 한다. 기존 Hyper1D checkpoint에
`--architecture plain4`를 주어 resume하면 구조가 맞지 않으므로 launcher가 거부한다.
기존 checkpoint의 평가/resume은 기존 스크립트로 계속 가능하다.

```bash
cd /ceph_data/clue9986/CleanToken/upstream/globalsplat
mkdir -p logs/slurm
sbatch scripts/slurm/train_hyper1d_plain4_12h.slurm
```

새 wrapper는 `ariel-v9`, GPU 1개, 최대 50,000 step,
checkpoint 500 step 간격, validation 2,000 step 간격을 사용한다.
`--no-time-limit`으로 기존 11시간 학습 타이머를 끄고, warmup은 기존 1,000 step을 유지한다.
Slurm job 제한은 72시간이다. 파일 이름의 `12h`는 이전 이름이며 현재 설정은 50K 학습이다.
큰 모델의 step 시간과 GPU VRAM은 아직 측정하지 않았다.
출력은 `outputs/hyper1d_plain4/job_<jobid>/` 아래에 저장한다.
checkpoint 경로는 `checkpoints/hyper1d_12h/version_0/{stepXXXXXXXXX.ckpt,last.ckpt}`다.
`experiment_name=hyper1d_12h`는 동일하지만 run 출력 폴더가 분리된다.

원본 checkpoint가 기존 tokencomp 폴더에만 있으면:

```bash
sbatch scripts/slurm/train_hyper1d_plain4_12h.slurm \
  --vanilla-checkpoint /ceph_data/clue9986/tokencomp/upstream/globalsplat/checkpoints/pretrained/globalsplat-re10k-32k.ckpt
```

CPU 검증은 두 구조의 실제 entropy roundtrip, 가변 길이, AMP 송신 일치, gradient,
strict checkpoint/기존 metadata 복원, QR 모델 경계, launcher와 Slurm wrapper를 포함해
67개 테스트가 통과했고 기존 main checkpoint 관련 테스트 2개도 통과했다.
실제 원본 GlobalSplat checkpoint에서도 `plain4` 초기화와 strict codec reload를 확인했다.
QR 재매개화 최대 오차는 7.16e-7 이하였다. 로컬 초기화 artifact는
`outputs/hyper1d_plain4/implementation_smoke_20260929_162740/hyper1d_initial.ckpt`에 저장했다.
실제 크기의 `plain4`에 합성 4096-token 입력을 넣은
FP32 forward/실제 복호 최대 오차는 1.96e-8 이하였다. GPU 학습 품질 검증은 아직 수행하지 않았다.

동일한 원본에서 이전 구조를 새로 학습하려면 `train_hyper1d_12h.slurm --architecture legacy`를 사용한다.

서버에는 아래 7개 파일을 같은 상대 경로로 반영한다. 로컬에서 생성한 초기화 checkpoint는
필수 업로드 파일이 아니며, Slurm job에서 서버의 원본 checkpoint로 새로 생성한다.

- `globalsplat/compression/hyper1d.py`
- `globalsplat/compression/hyper1d_config.py`
- `globalsplat/compression/checkpoint.py`
- `config/model/globalsplat_hyper1d.yaml`
- `scripts/run_hyper1d.py`
- `scripts/initialize_hyper1d_from_vanilla.py`
- `scripts/slurm/train_hyper1d_plain4_12h.slurm`

### 자동 업로드에서 제외된 파일 준비

현재 로컬 SFTP 설정의 remotePath는 `/ceph_data/clue9986/CleanToken`이며,
저장소 루트를 이 위치로 업로드하면 실행 디렉터리는
`/ceph_data/clue9986/CleanToken/upstream/globalsplat`이다.

기존 학습·평가 로그의 실행 루트는 `/ceph_data/clue9986/tokencomp/upstream/globalsplat`이었다.
서버에 기존 원본 checkpoint가 아래 위치에 남아 있으면 새로 전송할 필요 없이 그 경로를
`--vanilla-checkpoint`에 직접 주거나 새 프로젝트의 `checkpoints/pretrained/`로 복사한다.
이 경로는 기존 루트와 pretrained 배치를 조합한 위치이며, 서버의 현재 실존 여부는 확인해야 한다.

```bash
ls -lh /ceph_data/clue9986/tokencomp/upstream/globalsplat/checkpoints/pretrained/globalsplat-re10k-32k.ckpt
```

기존 환경은 로그상 `/ceph_data/clue9986/anaconda3/envs/globalsplat`이며,
해당 환경에 설치된 LPIPS 및 공유되는 Torch cache를 그대로 사용할 수 있다.

첫 B 학습에는 아래 파일과 기존 서버 데이터가 필요하다. `outputs/`, `checkpoints/`,
`*.pth` 등을 자동 업로드에서 제외한 상태를 유지하고 필요한 가중치만 별도로 전송할 수 있다.

| 항목 | 현재 로컬 위치 | 서버 위치 |
| --- | --- | --- |
| 원본 GlobalSplat-32K checkpoint | `upstream/globalsplat/checkpoints/pretrained/globalsplat-re10k-32k.ckpt` (341,595,070 B) | `/ceph_data/clue9986/CleanToken/upstream/globalsplat/checkpoints/pretrained/globalsplat-re10k-32k.ckpt` |
| LPIPS용 ImageNet VGG16 | `C:/Users/a0103/.cache/torch/hub/checkpoints/vgg16-397923af.pth` (553,433,881 B) | 기본 `~/.cache/torch/hub/checkpoints/vgg16-397923af.pth`; 아래 명령으로 실제 cache 위치 확인 |
| RE10K train/test 데이터 | 이번 준비에서는 서버의 기존 데이터 사용 | `/data3/local_datasets/re10k/train/`, `/data3/local_datasets/re10k/test/` 각각 `index.json`과 `.torch` chunk 파일 |

VGG16 cache 경로는 **job과 같은 `globalsplat` conda 환경**에서 확인한다.
`TORCH_HOME` 또는 `XDG_CACHE_HOME`이 설정되어 있으면 그 설정에 따라 경로가 달라진다.

```bash
conda activate globalsplat
python -c "from pathlib import Path; import torch; print(Path(torch.hub.get_dir()) / 'checkpoints/vgg16-397923af.pth')"
```

출력된 경로에 VGG16 `.pth`를 둔다. LPIPS 자체의 작은 `weights/v0.1/vgg.pth`는
`lpips` 패키지에 포함되므로 서버 환경의 패키지 설치를 사용한다.
현재 recipe는 `w_lpips=0.05`, `w_perc=0`이므로
`metric_checkpoint/imagenet-vgg-verydeep-19.mat`는 필요하지 않다.

서버 환경에는 CUDA용 PyTorch/torchvision, Lightning, CompressAI 1.2.8,
LPIPS와 CUDA renderer인 gsplat 등 기존 프로젝트 의존성이 설치되어 있어야 한다.
로컬 Windows 가상환경을 옮기지 않는다. 로컬 검증용 `.hyper1d-qa/`도 업로드 제외 대상이다.

코드는 `globalsplat/`, `config/`, `scripts/`, `third_party/ZPressor/mvsplat/`가 필요하다.
Hydra YAML은 `config/experiment/`에 있으므로 `**/experiments/**` 제외 규칙과는 다르다.
기존 `experiments/` 결과, 기존 학습 `outputs/`, 로그 및 main NFC-GS checkpoint는 이번 첫 학습의 입력이 아니다.
`--vanilla-checkpoint` 경로는 서버에서 신규 `hyper1d_initial.ckpt`를 자동 생성한다.

제출 전에 서버 파일을 확인할 수 있다:

```bash
cd /ceph_data/clue9986/CleanToken/upstream/globalsplat
conda activate globalsplat
python - <<'PY'
from pathlib import Path
import torch

checkpoint = Path('checkpoints/pretrained/globalsplat-re10k-32k.ckpt')
vgg = Path(torch.hub.get_dir()) / 'checkpoints/vgg16-397923af.pth'
for path in (checkpoint, vgg):
    print('OK' if path.is_file() else 'MISSING', path)
for split in ('train', 'test'):
    root = Path('/data3/local_datasets/re10k') / split
    chunks = len(list(root.glob('*.torch')))
    print(split, 'index=', (root / 'index.json').is_file(), 'chunks=', chunks)
PY
```

### 제출

Linux GPU 서버의 `upstream/globalsplat`에서 기존 GlobalSplat+CompressAI 환경을 사용한다.
`--vanilla-checkpoint`에는 **코덱이 없는 원본 pretrained GlobalSplat-32K** 체크포인트를 준다.
main NFC-GS 체크포인트를 신규 코덱에 직접 넣으면 종류 검증에서 거부한다.

```bash
cd upstream/globalsplat
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/run_hyper1d.py train \
  --vanilla-checkpoint checkpoints/pretrained/globalsplat-re10k-32k.ckpt \
  --dataset-root /data3/local_datasets/re10k \
  --output outputs/hyper1d/pilot_B
```

먼저 명령만 확인하려면 `--dry-run`을 붙인다. 파일이나 학습을 생성하지 않는다.
실행하면 원본 backbone을 복사하고 geometry readout을 QR로 재매개화한
`hyper1d_initial.ckpt`를 만든 뒤 학습한다. 초기화 파일을 덮어쓰지 않으므로
동일 output으로 재실행할 때는 아래 체크포인트 경로를 사용한다.

SLURM에서는 로그 폴더를 만든 뒤 job을 제출한다. 현재 스크립트의 job 제한은 24시간이며,
기본 학습 시간은 11시간이다. 실행 노드는 `ariel-v9`으로 지정한다. GPU 종류·partition은 기존
클러스터의 `gpu:normal:1`/`batch_ugrad` 설정을 따르므로 A5000 배정 여부는 클러스터에서 확인한다.

SLURM wrapper는 인자를 생략하면 현재 프로젝트의
`checkpoints/pretrained/globalsplat-re10k-32k.ckpt`를 자동으로 사용한다.
`--checkpoint` 또는 `--vanilla-checkpoint`를 명시하면 그 경로를 우선 사용한다.
기본 경로는 `VANILLA_CHECKPOINT` 환경변수로도 바꿀 수 있다.

```bash
mkdir -p logs/slurm
sbatch scripts/slurm/train_hyper1d_12h.slurm
```

실행 인자를 Python launcher에 직접 전달하거나 output을 지정할 때는 다음 형식을 사용한다.

```bash
mkdir -p logs/slurm
sbatch scripts/slurm/train_hyper1d_12h.slurm \
  --vanilla-checkpoint checkpoints/pretrained/globalsplat-re10k-32k.ckpt \
  --dataset-root /data3/local_datasets/re10k \
  --output outputs/hyper1d/pilot_B
```

## 학습·검증 계약

| 항목 | 기본 설정 |
| --- | --- |
| GPU / frozen 모델 | 1개 / encoder와 Gaussian decoder 고정 |
| 학습 파라미터 | geometry projection와 전체 신규 코덱; EB quantiles는 deterministic 갱신 |
| microbatch × accumulation | 2 × 4 = effective batch 8 |
| 종료 조건 | optimizer step 16,000 또는 학습 시작부터 11시간 |
| optimizer / LR | Adam, 1e-4, warmup 1,000 step 후 일정하게 유지 |
| λ | 0.0256, 한 run에 하나; `--lambda`로 변경 |
| checkpoint | 매 2,000 optimizer step 및 정상 시간 종료 후 `last.ckpt` |
| validation | 매 8,000 training batch; accumulation 4 기준 2,000 update 간격 |
| 장면 / 프레임 | 첫 128개 처리 가능한 held-out RE10K test 장면; 고정 12 context + 8 target |
| 종료 후 | 마지막 모델의 같은 128개 장면 검증 |

Lightning의 validation 간격은 training batch 단위다. `--accumulate`를 바꾸면 launcher가
`--validate-every × accumulate`로 간격을 맞춘다. epoch 마지막의 불완전한 accumulation은
update 간격에 작은 차이를 만들 수 있다. 체크포인트 주기는 optimizer step 단위다.

학습 시간 제한은 초기화·환경 준비 전의 job 전체 시간이 아니라 `Trainer`의 학습 시간이다.
학습 도중 validation 시간은 여기에 포함된다. SLURM 시간 제한에는 모든 작업이 포함된다.
11시간은 최종 저장·검증을 위한 여유를 두는 설정이며, 전체 완료 시간을 보장하는 측정치는 아니다.

validation은 기존 저장소의 held-out **test split 일부를 파일럿 검증용으로 사용**한다.
따라서 이를 모델 선택에 사용한 뒤 같은 장면 결과를 독립적인 최종 test 결과로 해석하지 않는다.
장면 순서는 chunk 파일 순서이며 shuffle·augmentation을 끈 단일 worker stream으로 재현한다.
같은 run의 장면·frame manifest가 바뀌면 명시적으로 실패한다.

`outputs/hyper1d/pilot_B/validation/manifest.json`에는 scene/frame IDs,
`stepXXXXXXXXX.json`에는 PSNR, LPIPS, 실제 y/z/container bytes, actual bpga,
estimated bits와 entropy bit gap을 저장한다. gap은 header를 뺀 실제 entropy bits와
추정 bits의 차이다. 부호에 관계없이 지속적으로 큰 괴리를 점검한다.
TensorBoard `train_codec_*`에는 rate 및 raw/adapter/y 규모, σ>256 비율,
`|y−μ_y|` 통계를 기록한다. `|y−μ_y|`는 centered latent 크기이며 양자화 오차가 아니다.

## 재시작·평가·용량 변경

기존 Adapter 포함 `legacy` 구조를 이어서 학습할 때는 재개용 SLURM wrapper를 사용한다.
job 436008 로그는 32,000 step까지 완료한 것으로 확인됐다. 32,000-step checkpoint에서
재개하면 18,000 step을 더 진행한다. 기본값은 **누적 50,000 step /
학습 타이머 없음 / 500 step마다 저장**이며,
검증은 기존과 같은 2,000 update 간격이다. `save_top_k=-1`이므로 주기별 파일을 모두 남기고,
정상 종료 시 optimizer/scheduler를 포함한 `last.ckpt`도 저장한다.

아래 경로는 job 434951과 resume job 436008의 로그에서 확인한 run 디렉터리다.
`last.ckpt`는 다음 학습에서 갱신되므로 고정된 출발점인 `step000032000.ckpt`를 사용한다.
서버의 해당 파일 실존 여부는 직접 확인하지 못했으므로 제출 전 `ls`로 확인한다.
434938을 이어갈 경우 해당 run의 실제 누적 step checkpoint를 선택한다.

```bash
cd /ceph_data/clue9986/CleanToken/upstream/globalsplat
RUN=outputs/hyper1d/20260928_161549_981799
CKPT="$RUN/checkpoints/hyper1d_12h/version_0/step000032000.ckpt"
ls -lh "$CKPT"
mkdir -p logs/slurm
sbatch scripts/slurm/resume_hyper1d_12h.slurm "$CKPT" --output "$RUN"
```

새 스크립트와 기존 `train_hyper1d_12h.slurm`, `run_hyper1d.py`가 서버에 있어야 한다.
재개 job은 `ariel-v9`, GPU 1개이며 SLURM 제한은 기존과 같은 24시간이다.
저장 간격은 `--checkpoint-every`로 덮어쓸 수 있다.
이 wrapper는 `--no-time-limit`을 기본 전달하므로 `--train-hours`만 추가해도
학습 타이머는 켜지지 않는다. 실제 종료 제한은 누적 step 또는 Slurm의 24시간이다.
500 step은 이전 run의 속도 기준 약 18분이며, 저장 시간이나 노드 상태에 따라 달라진다.
로그는 `logs/slurm/slurm-gs-hyper1d-resume-<jobid>.out/.err`에 저장된다.
누적 step 이름의 checkpoint는 원래 `version_0/` 디렉터리에 이어서 저장되며
`--output "$RUN"`은 기존 TensorBoard와 validation 경로를 유지한다.

같은 설정을 Python으로 직접 실행하거나 완료 후 평가할 수도 있다.

```bash
python scripts/run_hyper1d.py train \
  --checkpoint "$CKPT" \
  --resume --max-steps 50000 --output "$RUN" \
  --checkpoint-every 500 --validate-every 2000 --no-time-limit \
  --dataset-root /data3/local_datasets/re10k

python scripts/run_hyper1d.py eval \
  --checkpoint "$RUN/checkpoints/hyper1d_12h/version_0/step000050000.ckpt" \
  --dataset-root /data3/local_datasets/re10k --max-scenes 128
```

`--max-steps`는 재시작 이후 추가 step 수가 아니라 **누적 step 상한**이다. 재시작은
optimizer/scheduler/global step을 복원하고 학습 타이머 없이 step 상한까지 진행한다.
데이터 loader의 exact mid-epoch 복원은 기본적으로 끄므로 새 shuffle stream으로 이어간다.
`--checkpoint`는 저장된 stride/norm/Morton/channel 설정을 자동으로 복원한다.

2× downsampling은 새로운 초기화 run에 `--strides 2x`, Morton 비교는 `--morton`을 준다.
두 설정의 학습 결과는 따로 관리한다. scene mean과 score path 재도입은 구현 범위 밖이다.

## 별도 test 장면 샘플 평가

`eval_hyper1d.slurm`은 **RE10K test/index.json의 장면 전체에서 최대 100개를 랜덤 선택**한다.
seed는 111123이며 해당 run의 validation 장면을 제외한다. 첫 128개 장면을 다시 평가하는
방식이 아니다. validation도 test split 일부를 사용하므로 장면 단위로 제외한다.
입력 12프레임·target 8프레임은 deterministic-all sampler로 선택하며 서로 겹치지 않는다.
모델은 실제 entropy 압축·복호 결과를 렌더링하여 PSNR·SSIM·LPIPS와 실제 bytes를 평가한다.

2026-09-29에 기존 평가 설정을 교차 확인했다. 비교 후보인 Nonlinear32 transform 평가의
`PROTOCOL=all, CONTEXT=12`와 Full+Split의 `run_nfcgs.py eval`은 모두
`re10k_eval_all_ctx12`를 사용한다. main 평가 로그 428943과 joint-500k 평가 로그 428944도
input 12 / target 8 / 해상도 256×256을 기록한다. 따라서 Hyper1D test 평가도 이 view 수를
사용한다. 학습 recipe의 context pool 24 / target 12와 평가 설정은 구분한다.
별도의 fixed C3G 프로토콜(`re10k_eval_ctx12/24/36`)은 input 12·24·36 / target 3이며,
여기서 비교하는 기존 all-test 결과의 프로토콜은 아니다. 상세 근거는
`archive/codec_experiments_20260915/scripts/slurm/eval_nfcgs_transform.slurm`,
같은 디렉터리의 `eval_nfcgs_rank56.slurm`, 그리고
[공통 평가 protocol](SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md#22-full-test-protocol)을 참고한다.

```bash
cd /ceph_data/clue9986/CleanToken/upstream/globalsplat
RUN=outputs/hyper1d/20260928_161549_981799
CKPT="$RUN/checkpoints/hyper1d_12h/version_0/step000016000.ckpt"
mkdir -p logs/slurm
sbatch scripts/slurm/eval_hyper1d.slurm "$CKPT" \
  --exclude-validation "$RUN/validation/manifest.json" \
  --max-scenes 100 \
  --output "$RUN/eval_test100_step16000" --save-images
```

`--max-scenes 16`으로 개수를 줄이거나 `--sample-seed`로 다른 샘플을 선택할 수 있다.
validation manifest를 생략하면 checkpoint의 상위 run 디렉터리에서 자동으로 찾는다.
manifest를 찾지 못하면 명시적인 경로를 요구하며 validation 제외를 임의로 생략하지 않는다.
샘플 목록은 output의 `test_sample.json`에 저장한다. 같은 output에 다른 샘플을 덮어쓰면
거부하므로 seed·개수를 바꿀 때는 output도 바꾼다. 장면별 데이터 품질·프레임 수 때문에
처리 불가능한 장면은 기존 loader가 건너뛰며 실제 평가 개수는 최종 JSON의 `scene_count`다.
선택된 장면을 포함하는 chunk만 읽고, 같은 chunk의 다른 장면은 image decoding 전에 제외한다.

평가 GPU는 v11의 1개, SLURM 제한은 2시간이다. 결과 디렉터리는
`$RUN/eval_test100_step16000/evaluation/hyper1d_12h/`이다:

- `scores_all_avg.json`: PSNR·SSIM·LPIPS, bytes/bpga, y/z/container bytes, 실제 scene count.
- `actual_rate_per_scene.json`: 장면별 지표와 context/target frame IDs.
- `pipeline_timing.json`: 단계별 실행 시간.
- `--save-images` 사용 시 `<scene>/color/`와 `<scene>/gt/`: 복원 이미지와 정답 이미지.

로그는 `logs/slurm/slurm-gs-hyper1d-eval-<jobid>.out/.err`에 남는다.
업로드할 변경 파일은 `scripts/slurm/eval_hyper1d.slurm`, `scripts/run_hyper1d.py`,
`globalsplat/main.py`, `globalsplat/dataset/data_module.py`, `globalsplat/dataset/test_subset.py`다.
기존 sample 없는 Python eval 동작과 학습·validation 경로는 유지한다.

## 선택 사항: TRAIN calibration

첫 파일럿은 B(`input_norm=none`)다. A가 필요하면 아직 학습하지 않은 QR 초기화
체크포인트에서 고정 seed의 **서로 다른 TRAIN 256개 장면**을 사용한다.

```bash
python scripts/calibrate_hyper1d.py \
  --checkpoint outputs/hyper1d/pilot_B/hyper1d_initial.ckpt \
  --dataset-root /data3/local_datasets/re10k --scenes 256 \
  --output outputs/hyper1d/calibrated_initial.ckpt
python scripts/run_hyper1d.py train \
  --checkpoint outputs/hyper1d/calibrated_initial.ckpt \
  --dataset-root /data3/local_datasets/re10k --output outputs/hyper1d/pilot_A
```

Welford population 통계를 사용하고 std를 최소 1e-6으로 clamp한다. manifest에는 scene/context IDs,
원본 encoder를 포함한 source checkpoint SHA-256, TRAIN index SHA-256, geometry projection SHA-256,
seed와 token 수를 기록한다. 통계는 해당 초기 projection으로 계산한 고정 buffer이며
projection 학습에 따라 매번 재계산하지 않는다. A 변경은 함수 변경이므로 optimizer 상태를 버린
weights-only 초기화로 저장하며 재학습 또는 fine tuning한다.

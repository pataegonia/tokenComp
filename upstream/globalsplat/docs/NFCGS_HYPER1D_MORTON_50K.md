# Hyper1D legacy Morton ON 비교

현재 요청한 **단일 MSH vs. base/residual 각각 MSH** 비교는
[MSH 비교 실험 문서](NFCGS_HYPER1D_MSH_COMPARE_50K.md)의 새 array를 사용한다.
이 문서는 단일 MSH Morton ON 실행만 설명한다.

기존 Hyper1D `legacy`에 Morton 정렬을 켜고 λ=0.0256으로 50,000 step 학습한다.
`legacy`는 Adapter가 있는 **1-path Hyper1D**다. Nonlinear32의 score+residual
2-path와는 다른 모델이다. 예전 Nonlinear32 2-path의 비교용 50K 실행은 이미
Morton ON이었으므로, 이 제출 파일은 Hyper1D Morton ON 실험만 추가한다.

| 항목 | 설정 |
| --- | --- |
| 초기화 | vanilla GlobalSplat에서 새 codec 초기화, seed 111123 |
| 구조 | legacy, Adapter 포함, N=192 / M=320, 4× downsampling |
| 정렬 | Morton ON, 10 bits |
| λ / step | 0.0256 / 50,000 |
| 학습 대상 | codec 및 geometry projection; encoder/generator 고정 |
| 저장 / validation | 500 step / 2,000 step, 마지막 step에서도 validation |
| GPU / Slurm 제한 | ariel-v9, GPU 1개 / 6일 |

기존 Hyper1D OFF checkpoint를 resume하지 않는다. `--morton`은 vanilla 초기화에
적용되는 옵션이며, checkpoint를 읽으면 그 checkpoint에 저장된 정렬 설정이 복원된다.

## 제출

기존 Hyper1D 코드가 서버에 반영되어 있다면 새 Slurm 파일만 같은 상대 경로에 복사한다.

```bash
cd /ceph_data/clue9986/CleanToken/upstream/globalsplat
ls -lh checkpoints/pretrained/globalsplat-re10k-32k.ckpt
mkdir -p logs/slurm
sbatch scripts/slurm/train_hyper1d_legacy_morton_50k.slurm
```

설정만 확인하려면 `bash scripts/slurm/train_hyper1d_legacy_morton_50k.slurm --dry-run`을
Slurm allocation 안에서 실행한다. 초기화 명령의 `--morton`과 학습 명령의
`model.feature_codec.use_morton=true`, `architecture=legacy`, `n=192`, `m=320`을 확인한다.

```text
outputs/hyper1d_legacy_morton_50k/lambda0p0256/job_<job_id>/
  checkpoints/hyper1d_12h/version_0/step000050000.ckpt
  validation/step000050000.json
logs/slurm/slurm-gs-hyper1d-morton50k-<job_id>.out
logs/slurm/slurm-gs-hyper1d-morton50k-<job_id>.err
```

## 비교 checkpoint

Hyper1D Morton OFF, legacy, λ=0.0256, 50K:

```text
/ceph_data/clue9986/CleanToken/upstream/globalsplat/outputs/hyper1d/20260928_161549_981799/checkpoints/hyper1d_12h/version_0/step000050000.ckpt
```

Nonlinear32 score+residual 2-path, Morton ON, λ=0.0256, 50K
(사용자가 서버에서 존재를 확인한 경로):

```text
/ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_transform_nonlinear32_rank56_lambda0p0256_residual_on_morton_on/version_0/step000050000.ckpt
```

PSNR·LPIPS와 실제 bytes/BPGA를 같은 test scene ID와 context/target frame ID로 비교한다.
λ와 step 수가 같아도 bitrate는 달라질 수 있다. 기존 OFF 실행은 여러 번 resume했으므로
이번 연속 50K 실행과 데이터 소비 순서까지 동일한 통제 실험으로 해석하지 않는다.

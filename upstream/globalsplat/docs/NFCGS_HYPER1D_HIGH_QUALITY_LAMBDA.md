# Hyper1D 고화질 λ 점 4개

기준 실험 두 개는 모두 `rate_lambda=0.0256`, 50,000 step에서 종료됐다.
`GlobalSplatModule`의 학습 목적은 화질 손실에 `rate_lambda × estimated BPGA`를
더하므로, 더 작은 λ는 bitrate 벌점을 줄여 화질이 올라갈 가능성을 만든다.
이는 예상 방향이며 실제 PSNR·LPIPS와 bytes는 학습 후 검증해야 한다.

| Array task | 구조 | λ | 출발점 | step |
| ---: | --- | ---: | --- | ---: |
| 0 | `legacy` (Adapter 포함 1-path) | 0.0128 | 원본 vanilla GlobalSplat | 50,000 |
| 1 | `plain4` (Adapter 없음) | 0.0128 | 원본 vanilla GlobalSplat | 50,000 |
| 2 | `legacy` | 0.0064 | 원본 vanilla GlobalSplat | 50,000 |
| 3 | `plain4` | 0.0064 | 원본 vanilla GlobalSplat | 50,000 |

네 작업 모두 같은 원본 checkpoint에서 **각각 새 코덱을 초기화**한다.
기존 50K 코덱의 가중치나 optimizer 상태로 시작하지 않는다.
각 구조의 초기화 seed는 111123이고, 기존 λ=0.0256과 동일하게
4× downsampling, Morton OFF, frozen encoder/generator 및 학습 recipe를 쓴다.
각 작업은 50K step, warmup 1K step, checkpoint 500 step,
validation 2K step 간격이다. 학습 타이머를 끄고 Slurm 시간 제한은
**작업당 6일**이다. `--array=0-3%4`로 독립적인 네 작업을 기본 병렬 제출한다.
작업당 GPU 1개, 최대 동시 GPU 4개이며 실제 시작 시점은 Slurm 자원 배정에 따른다.

서버에서 최신 Hyper1D 코드를 반영한 뒤 다음처럼 제출한다.
원본 checkpoint의 기본 위치는 앞선 성공한 `plain4` job이 사용한 위치다.

```bash
cd /ceph_data/clue9986/CleanToken/upstream/globalsplat
ls -lh checkpoints/pretrained/globalsplat-re10k-32k.ckpt
mkdir -p logs/slurm
sbatch scripts/slurm/train_hyper1d_high_quality_50k.slurm
```

로그는 `logs/slurm/slurm-gs-hyper1d-hq50k-<array_job_id>_<task_id>.out/.err`이다.
결과와 checkpoint는 구조·λ·array task마다 독립된 폴더에 저장한다.

```text
outputs/hyper1d_high_quality_50k/<legacy|plain4>/lambda<0p0128|0p0064>/
  job_<array_job_id>_<task_id>/
    checkpoints/hyper1d_12h/version_0/step000050000.ckpt
    validation/step000050000.json
```

λ에 따라 압축 크기도 달라지므로 각 결과는 PSNR·LPIPS와 실제
`bytes`/`bpga`를 함께 비교한다. λ만 바꿨다고 같은 bitrate의 화질 비교가 되지는 않는다.

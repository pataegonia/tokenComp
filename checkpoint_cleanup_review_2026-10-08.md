# Checkpoint 정리 목록 — 2026-10-08

범위: `/ceph_data/clue9986/tokencomp/upstream/globalsplat`의 audit에 기록된 checkpoint 파일.
audit 시점: 2026-10-08 20:54:34 KST. 이 문서는 삭제 계획이며, 실제 삭제는 아직 실행하지 않았다.

111개 / 45.206 GiB → 14개 / 4.966 GiB.
삭제 대상: 97개, 40.240 GiB. 용량은 파일 크기의 합계이며, 실제 공간 반환 시점은 파일시스템에 따라 다를 수 있다.

## 현재 job 보존

- 대기 train `445516`, `445518`: 사용자가 기본 FullSplit λ=.0256 입력을 확인했다. 해당 step10k 파일을 보존한다.
- 대기 eval `445517`, `445519`: 각각 afterok:445516/445518. 새 출력 `outputs/nfcgs_order_context` 전체를 삭제 대상에서 보호한다.
- MVSplat은 `/ceph_data/clue9986/ProgressiveMV`, Hyper1D는 `/ceph_data/clue9986/CleanToken`을 사용한다. 두 저장소는 이번 삭제 범위에 포함되지 않는다.

## 보존할 파일 14개

FullSplit 2개, nonlinear/context 최종본 4개, minimal·2/3/4단계 최종본 4개, GlobalSplat/PCA 원본 3개, joint 500k 비교용 1개를 남긴다.

| 보존 이유 | 상대 경로 |
|---|---|
| from-scratch joint 500k 비교용 최종본 | `outputs/nfcgs_main/joint_extension/lambda0p0064/from_job_425420_to_500000/checkpoints/nfcgs_main_joint/version_0/step000500000.ckpt` |
| Full nonlinear context 최종본 | `outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_nonlinear_score_context_joint_rank56_lambda0p0064_residual_on_morton_on_m1c1s1/version_0/step000050000.ckpt` |
| Full nonlinear context 최종본 | `outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_nonlinear_score_context_joint_rank56_lambda0p0256_residual_on_morton_on_m1c1s1/version_0/step000050000.ckpt` |
| nonlinear32 50k 최종본; 이전 100-scene eval 기준 | `outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_transform_nonlinear32_rank56_lambda0p0064_residual_on_morton_on/version_0/step000050000.ckpt` |
| nonlinear32 50k 최종본; 이전 100-scene eval 기준 | `outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_transform_nonlinear32_rank56_lambda0p0256_residual_on_morton_on/version_0/step000050000.ckpt` |
| minimal score path 50k 최종본 | `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000050000.ckpt` |
| FullSplit λ=.0064 10k 부모; 다른 rate 설정의 비교·후속 실험용 | `outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt` |
| FullSplit λ=.0256 10k 부모; 현재 대기 학습의 기본 입력 | `outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt` |
| Morton 3단계 spatial context 50k 최종본 | `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000050000.ckpt` |
| Morton 2단계 spatial context 50k 최종본 | `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000050000.ckpt` |
| Morton 4단계 spatial context 50k 최종본 | `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000050000.ckpt` |
| rank56/rank80 PCA 초기화 원본 | `checkpoints/codec_init/re10k_ctx12_s64_t512_rank56.pt` |
| rank56/rank80 PCA 초기화 원본 | `checkpoints/codec_init/re10k_ctx12_s64_t512_rank80.pt` |
| GlobalSplat pretrained 원본 | `checkpoints/pretrained/globalsplat-re10k-32k.ckpt` |

## 삭제 대상 요약

| 폴더 | 파일 수 | GiB |
|---|---:|---:|
| `outputs/nfcgs_factorized_continuation` | 4 | 1.340 |
| `outputs/nfcgs_joint_save_probe_v10` | 6 | 6.040 |
| `outputs/nfcgs_main` | 3 | 3.023 |
| `outputs/nfcgs_nonlinear_score_context` | 4 | 1.414 |
| `outputs/nfcgs_paper24_subset_train` | 12 | 4.021 |
| `outputs/nfcgs_paper24_transform_train` | 4 | 1.412 |
| `outputs/nfcgs_score_context` | 8 | 2.682 |
| `outputs/nfcgs_score_path_ablation` | 10 | 3.707 |
| `outputs/nfcgs_score_probability10k` | 4 | 1.342 |
| `outputs/nfcgs_score_stages` | 33 | 12.239 |
| `outputs/nfcgs_spatial_predictor25k` | 9 | 3.020 |

완료한 주 실험의 중간 checkpoint와 last 파일, 저장 probe, 이전 joint 220k, 이전 ablation의 초기화 artifact를 정리한다.
기존 4단계는 job443968의 step50k를 보존하고, 이전 job442211의 5k/10k만 삭제한다. audit에는 best checkpoint가 없다.
last와 마지막 step의 파일 내용이 같은지는 검증하지 않았다. 최종 step 파일을 보존하는 기준으로 last를 정리한다.

## 실행

`cleanup_nfcgs_checkpoints_20261008.sh` 하나만 서버의 저장소 루트에 올린다. 목록과 검사 코드가 내장되어 다른 파일을 올릴 필요가 없다.

```bash
cd /
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
# 미리보기: 파일을 삭제하지 않음
bash cleanup_nfcgs_checkpoints_20261008.sh
# 실제 삭제
bash cleanup_nfcgs_checkpoints_20261008.sh --apply
```

`--apply`는 현재 Slurm job을 다시 조회하고 입력·출력 경로를 확인한다. audit 이후 새 job이 생겼거나 파일의 크기·mtime·inode가 바뀌면 중단한다.
정확히 아래에 나열한 .ckpt 파일만 `rm -- 파일경로`로 삭제한다. 폴더, 로그, 평가 JSON, 이미지와 신규 checkpoint는 삭제하지 않는다.

## 삭제할 파일 97개

- `outputs/nfcgs_factorized_continuation/20260908_184640/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_factorized_continuation/20260908_184640/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_factorized_continuation/20260908_184640/rank80/lambda0p0064/residual_on/initialization/nfcgs_rank80_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_factorized_continuation/20260908_184640/rank80/lambda0p0256/residual_on/initialization/nfcgs_rank80_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_joint_save_probe_v10/checkpoints/nfcgs_main_joint/version_0/last.ckpt` — checkpoint 저장 확인용 10/20 step probe
- `outputs/nfcgs_joint_save_probe_v10/checkpoints/nfcgs_main_joint/version_0/step000000010.ckpt` — checkpoint 저장 확인용 10/20 step probe
- `outputs/nfcgs_joint_save_probe_v10/checkpoints/nfcgs_main_joint/version_0/step000000020.ckpt` — checkpoint 저장 확인용 10/20 step probe
- `outputs/nfcgs_joint_save_probe_v10/checkpoints/nfcgs_main_joint/version_1/last.ckpt` — checkpoint 저장 확인용 10/20 step probe
- `outputs/nfcgs_joint_save_probe_v10/checkpoints/nfcgs_main_joint/version_1/step000000010.ckpt` — checkpoint 저장 확인용 10/20 step probe
- `outputs/nfcgs_joint_save_probe_v10/checkpoints/nfcgs_main_joint/version_1/step000000020.ckpt` — checkpoint 저장 확인용 10/20 step probe
- `outputs/nfcgs_main/joint/lambda0p0064/20260917_120326/checkpoints/nfcgs_main_joint/version_0/last.ckpt` — 이전 joint 220k; 연장한 500k 최종본 보존
- `outputs/nfcgs_main/joint/lambda0p0064/20260917_120326/checkpoints/nfcgs_main_joint/version_0/step000220000.ckpt` — 이전 joint 220k; 연장한 500k 최종본 보존
- `outputs/nfcgs_main/joint_extension/lambda0p0064/from_job_425420_to_500000/checkpoints/nfcgs_main_joint/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_nonlinear_score_context_joint_rank56_lambda0p0064_residual_on_morton_on_m1c1s1/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_m1c1s1.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_nonlinear_score_context_joint_rank56_lambda0p0256_residual_on_morton_on_m1c1s1/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c1s1.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank56/lambda0p0064/residual_off/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank56/lambda0p0064/residual_on/morton_off/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank56/lambda0p0256/residual_off/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank56/lambda0p0256/residual_on/morton_off/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank80/lambda0p0064/residual_off/initialization/nfcgs_rank80_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank80/lambda0p0064/residual_on/initialization/nfcgs_rank80_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank80/lambda0p0064/residual_on/morton_off/initialization/nfcgs_rank80_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank80/lambda0p0256/residual_off/initialization/nfcgs_rank80_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank80/lambda0p0256/residual_on/initialization/nfcgs_rank80_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_subset_train/rank80/lambda0p0256/residual_on/morton_off/initialization/nfcgs_rank80_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_transform_nonlinear32_rank56_lambda0p0064_residual_on_morton_on/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_transform_nonlinear32_rank56_lambda0p0256_residual_on_morton_on/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_exact_artifact.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_context/20260908_092102/joint/m1c0s0/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_m1c0s0.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_context/20260908_092102/joint/m1c0s0/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c0s0.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_context/20260908_092102/joint/m1c0s1/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_m1c0s1.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_context/20260908_092102/joint/m1c0s1/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c0s1.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_context/20260908_092102/joint/m1c1s0/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_m1c1s0.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_context/20260908_092102/joint/m1c1s0/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c1s0.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_context/20260908_092102/joint/m1c1s1/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_m1c1s1.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_context/20260908_092102/joint/m1c1s1/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c1s1.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000005000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000010000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000015000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000020000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000025000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000030000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000035000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000040000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000045000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_m1c1s1_split.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c1s1_split.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000005000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000010000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000015000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000020000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000025000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000030000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000035000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000040000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000045000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000005000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000010000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000015000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000020000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000025000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000030000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000035000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000040000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442209/stages2/checkpoints/nfcgs_main/version_0/step000045000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442211/stages4/checkpoints/nfcgs_main/version_0/last.ckpt` — 이전 4단계 run의 5k/10k; 완료한 job443968의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442211/stages4/checkpoints/nfcgs_main/version_0/step000005000.ckpt` — 이전 4단계 run의 5k/10k; 완료한 job443968의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_442211/stages4/checkpoints/nfcgs_main/version_0/step000010000.ckpt` — 이전 4단계 run의 5k/10k; 완료한 job443968의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/last.ckpt` — 같은 run의 마지막 step checkpoint를 보존하므로 last 파일 정리
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000005000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000010000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000015000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000020000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000025000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000030000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000035000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000040000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_score_stages/lambda0p0256/job_443968/stages4/checkpoints/nfcgs_main/version_0/step000045000.ckpt` — 완료한 run의 중간 step; 해당 run의 50k 최종본 보존
- `outputs/nfcgs_spatial_predictor25k/20260912_082429/p0_linear/m1c0s1/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c0s1.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_spatial_predictor25k/20260912_082429/p0_linear/m1c1s1/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_m1c1s1.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_spatial_predictor25k/20260912_082429/p1_residual3/m1c0s1/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c0s1_residual3h32.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_spatial_predictor25k/20260912_082429/p1_residual3/m1c1s1/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_m1c1s1_residual3h32.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_spatial_predictor25k/20260912_082429/p2_residual7/m1c0s1/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c0s1_residual7h32.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_spatial_predictor25k/20260912_082429/p2_residual7/m1c1s1/rank56/lambda0p0064/residual_on/initialization/nfcgs_rank56_m1c1s1_residual7h32.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_spatial_predictor25k/20260912_122749/p0_linear/m1c1s1/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c1s1.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_spatial_predictor25k/20260912_122749/p1_residual3/m1c1s1/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c1s1_residual3h32.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존
- `outputs/nfcgs_spatial_predictor25k/20260912_122749/p2_residual7/m1c1s1/rank56/lambda0p0256/residual_on/initialization/nfcgs_rank56_m1c1s1_residual7h32.ckpt` — 이전 실험의 초기화 artifact; 주 경로의 학습 최종본과 PCA 원본은 별도 보존

# NFCGS full-scene confidence intervals

Date: 2026-09-15

## Scope

This analysis uses every scene in the already completed `all_test_ctx12` evals;
it is unrelated to the 32-scene runtime-overhead profile. No checkpoint is
trained, no scene is encoded/rendered again, and no GPU is needed.

The four rank56 model families are:

1. `linear_factorized`: Linear existing model, no context, matched 50k
   Factorized continuation.
2. `nonlinear_factorized`: Nonlinear32 existing model, no context, matched 50k
   Factorized continuation.
3. `linear_full`: Linear Mean+Channel+Spatial Full context, joint 50k.
4. `nonlinear_split`: Nonlinear32 Full P0 followed by reconstruction-locked
   Split probability fitting for 10k.

Both `.0064` and `.0256` are included. The manifest points at the exact eight
completed result directories and their `actual_rate_per_scene.json` files.

## Pairing and CI definition

- A scene is the resampling unit. Its eight target-view scores are already
  averaged by the evaluator and are not counted as eight independent samples.
- All four models and both lambdas must contain the identical complete scene
  set.
- The context-frame and target-frame ID lists must match exactly for every
  paired scene.
- The script also recomputes every global mean and verifies it against the
  existing `scores_all_avg.json`.
- Default: 20,000 paired nonparametric percentile-bootstrap draws, 95% CI,
  fixed seed 20260915.
- Each draw resamples complete scenes once and applies the same sample indices
  to all eight model/lambda cells.

The report contains:

- each model/lambda's mean bytes, PSNR, SSIM, and LPIPS with scene-bootstrap CI;
- all six model pairs at each lambda: rate saving, PSNR delta, SSIM delta,
  LPIPS improvement, and raw scene win fractions;
- an approximate two-point log-linear BD-rate and paired CI across the two
  lambdas.

For pairwise rows, positive rate saving, PSNR delta, SSIM delta, and LPIPS
improvement favor the candidate. Standard BD-rate keeps its normal convention:
negative favors the candidate.

## Run on the server

Update the four required files listed below, then run from:

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
bash scripts/run_nfcgs_scene_ci.sh
```

This is a small CPU analysis and does not submit a Slurm job. Override the
number of draws only if needed:

```bash
SCENE_CI_BOOTSTRAP_DRAWS=50000 bash scripts/run_nfcgs_scene_ci.sh
```

## Output

```text
/ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_scene_ci/<RUN_TAG>/
  REPORT.md
  summary.json
  paired_scene_metrics.csv
```

`summary.json` preserves full-precision estimates, intervals, standard errors,
source paths, and SHA-256 hashes. `paired_scene_metrics.csv` preserves the
auditable scene-level paired table used by the bootstrap.

SFTP download:

```text
get -r /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_scene_ci/<RUN_TAG>
```

## Interpretation limit

These intervals quantify variation across the TEST scenes under the fixed
C12/T8 protocol. They do not include training-seed uncertainty, checkpoint
selection, dataset shift, or hardware variance. The six pairwise 95% intervals
are not multiplicity-adjusted. The two-lambda BD-rate is a screening statistic;
a publication-grade curve still requires at least four well-spaced RD points.

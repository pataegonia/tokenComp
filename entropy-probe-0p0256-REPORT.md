# Frozen-symbol entropy probe

Checkpoint: /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt

Fit: 128 unique TRAIN scenes; evaluation: 128 unique TEST scenes.

| Model | Total KiB | Score KiB | Residual y+z KiB | Total saving |
| --- | ---: | ---: | ---: | ---: |
| Original Split | 36.923 | 29.374 | 5.983 | 0% |
| marginal | 36.859 | 29.318 | 5.975 | 0.172% |
| context | 36.642 | 29.101 | 5.975 | 0.759% |

## Which path has reducible probability cost?

Savings below are percentage points of the ORIGINAL TOTAL scene bytes. They include a 95% paired-scene bootstrap interval; positive means smaller.

| Candidate | Score-only saving | Residual-only saving | Score minus residual |
| --- | ---: | ---: | ---: |
| marginal | +0.151 [+0.053, +0.250] pp | +0.021 [+0.010, +0.033] pp | +0.130 [+0.039, +0.222] pp |
| context | +0.738 [+0.658, +0.825] pp | +0.021 [+0.010, +0.033] pp | +0.717 [+0.644, +0.798] pp |

## Where do the coded bits go?

Bits per scene below EXCLUDE all containers and the scene mean. Model NLL is evaluated at the exact encoded integer symbols, with the native likelihood floor.

| Stream | Model NLL | Integer CDF | Tail bypass | Actual | Actual - CDF - bypass |
| --- | ---: | ---: | ---: | ---: | ---: |
| score_g0_even | 70824.8 | 70927.5 | 0.0 | 70975.5 | 48.0 |
| score_g0_odd | 46239.0 | 46319.3 | 0.0 | 46366.2 | 46.9 |
| score_g1_even | 44058.5 | 44070.8 | 0.0 | 44118.8 | 47.9 |
| score_g1_odd | 34663.2 | 34687.3 | 0.0 | 34735.8 | 48.4 |
| score_g2_even | 10516.1 | 10519.2 | 0.0 | 10568.0 | 48.8 |
| score_g2_odd | 9976.5 | 9979.9 | 0.0 | 10028.8 | 48.8 |
| score_g3_even | 11751.8 | 11754.1 | 0.0 | 11801.2 | 47.1 |
| score_g3_odd | 11280.0 | 11282.1 | 0.0 | 11330.5 | 48.4 |
| residual_z | 850.2 | 853.6 | 0.0 | 901.8 | 48.2 |
| residual_y | 47964.4 | 48060.8 | 0.0 | 48109.5 | 48.7 |

Separate bytes per scene: mean=1472.0, score wrapper=88.0, outer/residual wrappers=132.0.


All integer streams and decoded features matched exactly on every evaluated scene.
The receiver recomputed its own context using only mean and decoded anchors.

Model CDF tables are shared metadata, excluded from per-scene payload; their compressed file size is reported separately in summary.json. These packets require the diagnostic decoder and are not a deployable production bitstream variant.

This small test subset is a screening result, not the previous full-test average. Bootstrap intervals only reflect the selected scenes. No PSNR/rendering was rerun.

A positive held-out saving establishes an opportunity under this particular probability model. A negative result does not establish an entropy lower bound or rule out stronger models.

The context candidate expands score conditioning only; residual remains marginal recalibration. A path comparison ranks these particular probes, not the globally best achievable score/residual models. This diagnostic does not measure deployment decoding speed; Python/CDF-conversion time is not a codec benchmark.

See summary.json for per-stream float-model NLL, integer-CDF cost, escape bypass cost, actual bits, CDF-table bytes and paired-scene bootstrap intervals.

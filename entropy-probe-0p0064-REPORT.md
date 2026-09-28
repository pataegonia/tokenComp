# Frozen-symbol entropy probe

Checkpoint: /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt

Fit: 128 unique TRAIN scenes; evaluation: 128 unique TEST scenes.

| Model | Total KiB | Score KiB | Residual y+z KiB | Total saving |
| --- | ---: | ---: | ---: | ---: |
| Original Split | 81.253 | 68.021 | 11.665 | 0% |
| marginal | 81.160 | 67.947 | 11.646 | 0.115% |
| context | 80.654 | 67.442 | 11.646 | 0.736% |

## Which path has reducible probability cost?

Savings below are percentage points of the ORIGINAL TOTAL scene bytes. They include a 95% paired-scene bootstrap interval; positive means smaller.

| Candidate | Score-only saving | Residual-only saving | Score minus residual |
| --- | ---: | ---: | ---: |
| marginal | +0.091 [+0.014, +0.175] pp | +0.024 [+0.020, +0.028] pp | +0.067 [-0.008, +0.150] pp |
| context | +0.713 [+0.633, +0.797] pp | +0.024 [+0.020, +0.028] pp | +0.689 [+0.608, +0.773] pp |

## Where do the coded bits go?

Bits per scene below EXCLUDE all containers and the scene mean. Model NLL is evaluated at the exact encoded integer symbols, with the native likelihood floor.

| Stream | Model NLL | Integer CDF | Tail bypass | Actual | Actual - CDF - bypass |
| --- | ---: | ---: | ---: | ---: | ---: |
| score_g0_even | 143006.8 | 143247.3 | 0.6 | 143295.0 | 47.1 |
| score_g0_odd | 110800.7 | 110986.2 | 0.0 | 111033.5 | 47.3 |
| score_g1_even | 73222.6 | 73266.8 | 0.1 | 73315.0 | 48.1 |
| score_g1_odd | 63347.2 | 63403.2 | 0.1 | 63452.0 | 48.8 |
| score_g2_even | 59632.5 | 59655.1 | 0.3 | 59704.2 | 48.9 |
| score_g2_odd | 56606.8 | 56627.4 | 0.0 | 56675.5 | 48.1 |
| score_g3_even | 24878.5 | 24883.8 | 0.1 | 24932.0 | 48.1 |
| score_g3_odd | 24067.9 | 24072.3 | 0.1 | 24119.2 | 46.9 |
| residual_z | 1362.5 | 1365.8 | 0.0 | 1413.2 | 47.4 |
| residual_y | 93889.3 | 94099.7 | 0.2 | 94147.5 | 47.5 |

Separate bytes per scene: mean=1472.0, score wrapper=88.0, outer/residual wrappers=132.0.


All integer streams and decoded features matched exactly on every evaluated scene.
The receiver recomputed its own context using only mean and decoded anchors.

Model CDF tables are shared metadata, excluded from per-scene payload; their compressed file size is reported separately in summary.json. These packets require the diagnostic decoder and are not a deployable production bitstream variant.

This small test subset is a screening result, not the previous full-test average. Bootstrap intervals only reflect the selected scenes. No PSNR/rendering was rerun.

A positive held-out saving establishes an opportunity under this particular probability model. A negative result does not establish an entropy lower bound or rule out stronger models.

The context candidate expands score conditioning only; residual remains marginal recalibration. A path comparison ranks these particular probes, not the globally best achievable score/residual models. This diagnostic does not measure deployment decoding speed; Python/CDF-conversion time is not a codec benchmark.

See summary.json for per-stream float-model NLL, integer-CDF cost, escape bypass cost, actual bits, CDF-table bytes and paired-scene bootstrap intervals.

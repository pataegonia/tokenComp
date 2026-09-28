# NFCGS overhead profile

Paired TEST scenes: 32; warm-up scenes: 4.

Context rows are compared with the same-lambda no-context Factorized continuation. Probability rows are compared with the same-lambda Full+Shared model.

## Context

### lambda 0.0064

| Model | Streams | Raw entropy B | Wrapper B | Score payload B | Total scene B | Break-even scenes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factorized | 1 | 78551.5 | 0.0 | 78551.5 (+0.00 B) | 93752.9 (+0.00 B) | n/a |
| mean | 1 | 77116.1 | 32.0 | 77148.1 (-1403.38 B) | 92062.4 (-1690.50 B) | 184.6 |
| mean_channel | 4 | 75424.4 | 56.0 | 75480.4 (-3071.12 B) | 90414.0 (-3338.88 B) | 57.5 |
| mean_spatial | 2 | 73436.2 | 40.0 | 73476.2 (-5075.25 B) | 88246.4 (-5506.50 B) | 53.7 |
| full | 8 | 72054.8 | 88.0 | 72142.8 (-6408.75 B) | 86991.2 (-6761.62 B) | 28.1 |

| Model | Score-path params | Predictor params | Probability params | Score-path tensor KiB | Added tensor KiB | Serialized KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factorized | 3,416 | 0 | 3,416 | 118.80 | +0.00 KiB | 124.96 |
| mean | 59,336 | 55,920 | 3,416 | 371.80 | +253.00 KiB | 379.38 |
| mean_channel | 60,528 | 57,112 | 3,416 | 291.38 | +172.58 KiB | 320.63 |
| mean_spatial | 68,800 | 65,384 | 3,416 | 385.14 | +266.34 KiB | 393.19 |
| full | 63,080 | 59,664 | 3,416 | 294.72 | +175.92 KiB | 326.48 |

| Model | Score encode ms | Score decode ms | Full codec encode ms | Full codec decode ms | Score peak enc/dec MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| factorized | 48.82 (+0.00%) | 30.23 (+0.00%) | 133.90 (+0.00%) | 78.92 (+0.00%) | 2.62 / 2.62 |
| mean | 50.73 (+3.91%) | 31.30 (+3.54%) | 136.02 (+1.58%) | 80.15 (+1.56%) | 3.50 / 2.63 |
| mean_channel | 55.84 (+14.39%) | 34.41 (+13.82%) | 140.84 (+5.19%) | 82.64 (+4.71%) | 2.50 / 2.25 |
| mean_spatial | 54.70 (+12.04%) | 33.84 (+11.95%) | 139.16 (+3.93%) | 82.26 (+4.23%) | 4.41 / 3.97 |
| full | 53.32 (+9.22%) | 33.19 (+9.79%) | 137.98 (+3.05%) | 81.50 (+3.27%) | 2.63 / 2.38 |

### lambda 0.0256

| Model | Streams | Raw entropy B | Wrapper B | Score payload B | Total scene B | Break-even scenes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factorized | 1 | 35462.2 | 0.0 | 35462.2 (+0.00 B) | 43587.1 (+0.00 B) | n/a |
| mean | 1 | 34335.9 | 32.0 | 34367.9 (-1094.38 B) | 42528.0 (-1059.12 B) | 197.6 |
| mean_channel | 4 | 34197.4 | 56.0 | 34253.4 (-1208.88 B) | 42447.4 (-1139.75 B) | 150.6 |
| mean_spatial | 2 | 32014.4 | 40.0 | 32054.4 (-3407.88 B) | 40045.1 (-3542.00 B) | 77.3 |
| full | 8 | 31969.9 | 88.0 | 32057.9 (-3404.38 B) | 40121.4 (-3465.75 B) | 57.4 |

| Model | Score-path params | Predictor params | Probability params | Score-path tensor KiB | Added tensor KiB | Serialized KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factorized | 3,416 | 0 | 3,416 | 91.23 | +0.00 KiB | 97.40 |
| mean | 59,336 | 55,920 | 3,416 | 302.45 | +211.22 KiB | 310.01 |
| mean_channel | 60,528 | 57,112 | 3,416 | 268.97 | +177.73 KiB | 298.19 |
| mean_spatial | 68,800 | 65,384 | 3,416 | 348.61 | +257.38 KiB | 356.63 |
| full | 63,080 | 59,664 | 3,416 | 282.06 | +190.83 KiB | 313.79 |

| Model | Score encode ms | Score decode ms | Full codec encode ms | Full codec decode ms | Score peak enc/dec MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| factorized | 44.36 (+0.00%) | 26.50 (+0.00%) | 128.95 (+0.00%) | 74.49 (+0.00%) | 2.62 / 2.62 |
| mean | 44.63 (+0.62%) | 26.68 (+0.68%) | 129.41 (+0.36%) | 75.18 (+0.92%) | 3.50 / 2.63 |
| mean_channel | 51.26 (+15.57%) | 30.56 (+15.32%) | 135.39 (+5.00%) | 78.90 (+5.92%) | 2.50 / 2.25 |
| mean_spatial | 49.18 (+10.88%) | 29.52 (+11.42%) | 133.43 (+3.48%) | 77.53 (+4.08%) | 4.41 / 3.97 |
| full | 49.37 (+11.31%) | 29.62 (+11.79%) | 133.17 (+3.28%) | 77.79 (+4.42%) | 2.63 / 2.38 |

## Probability

### lambda 0.0064

| Model | Streams | Raw entropy B | Wrapper B | Score payload B | Total scene B | Break-even scenes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 8 | 70806.8 | 88.0 | 70894.8 (+0.00 B) | 84461.4 (+0.00 B) | n/a |
| split | 8 | 69468.2 | 88.0 | 69556.2 (-1338.50 B) | 83122.9 (-1338.50 B) | 46.0 |
| gaussian | 8 | 76380.2 | 88.0 | 76468.2 (+5573.50 B) | 90034.9 (+5573.50 B) | n/a |
| conditional_scale | 8 | 69358.6 | 88.0 | 69446.6 (-1448.12 B) | 83013.2 (-1448.12 B) | 574.9 |

| Model | Score-path params | Predictor params | Probability params | Score-path tensor KiB | Added tensor KiB | Serialized KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 63,080 | 59,664 | 3,416 | 293.16 | +0.00 KiB | 324.92 |
| split | 66,496 | 59,664 | 6,832 | 353.25 | +60.09 KiB | 411.38 |
| gaussian | 63,136 | 59,664 | 3,472 | 1077.39 | +784.23 KiB | 1112.46 |
| conditional_scale | 70,488 | 59,664 | 10,824 | 1106.11 | +812.95 KiB | 1146.24 |

| Model | Score encode ms | Score decode ms | Full codec encode ms | Full codec decode ms | Score peak enc/dec MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared | 53.41 (+0.00%) | 33.14 (+0.00%) | 138.19 (+0.00%) | 81.92 (+0.00%) | 2.63 / 2.38 |
| split | 53.34 (-0.12%) | 33.17 (+0.09%) | 138.18 (-0.00%) | 81.68 (-0.29%) | 2.63 / 2.38 |
| gaussian | 100.18 (+87.59%) | 59.04 (+78.15%) | 185.89 (+34.52%) | 107.52 (+31.25%) | 2.63 / 2.38 |
| conditional_scale | 100.20 (+87.61%) | 59.30 (+78.92%) | 184.52 (+33.53%) | 108.39 (+32.31%) | 2.63 / 2.38 |

### lambda 0.0256

| Model | Streams | Raw entropy B | Wrapper B | Score payload B | Total scene B | Break-even scenes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 8 | 31003.4 | 88.0 | 31091.4 (+0.00 B) | 38842.8 (+0.00 B) | n/a |
| split | 8 | 29925.6 | 88.0 | 30013.6 (-1077.75 B) | 37764.6 (-1078.12 B) | 43.4 |
| gaussian | 8 | 37661.4 | 88.0 | 37749.4 (+6658.00 B) | 45500.4 (+6657.62 B) | n/a |
| conditional_scale | 8 | 29989.8 | 88.0 | 30077.8 (-1013.62 B) | 37829.1 (-1013.62 B) | 821.3 |

| Model | Score-path params | Predictor params | Probability params | Score-path tensor KiB | Added tensor KiB | Serialized KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 63,080 | 59,664 | 3,416 | 278.78 | +0.00 KiB | 310.54 |
| split | 66,496 | 59,664 | 6,832 | 324.50 | +45.72 KiB | 382.63 |
| gaussian | 63,136 | 59,664 | 3,472 | 1063.01 | +784.23 KiB | 1098.08 |
| conditional_scale | 70,488 | 59,664 | 10,824 | 1091.73 | +812.95 KiB | 1131.86 |

| Model | Score encode ms | Score decode ms | Full codec encode ms | Full codec decode ms | Score peak enc/dec MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared | 49.32 (+0.00%) | 29.54 (+0.00%) | 133.62 (+0.00%) | 78.07 (+0.00%) | 2.63 / 2.38 |
| split | 49.11 (-0.41%) | 29.57 (+0.09%) | 133.50 (-0.09%) | 77.71 (-0.46%) | 2.63 / 2.38 |
| gaussian | 97.41 (+97.51%) | 56.20 (+90.24%) | 183.07 (+37.01%) | 105.67 (+35.35%) | 2.63 / 2.38 |
| conditional_scale | 96.50 (+95.67%) | 56.52 (+91.32%) | 180.84 (+35.33%) | 105.49 (+35.13%) | 2.63 / 2.38 |

## Interpretation notes

- Score timing excludes the GlobalSplat image encoder, Morton ordering, low-rank analysis/synthesis, and residual coding. It includes the entropy coder plus the causal reconstruction work required by the sender.
- Full codec timing starts from scene tokens, so it includes ordering, transform, score coding, residual coding, and bitstream packing, but not the image backbone or Gaussian rendering.
- `Wrapper B` is only the SCCTX001 prefix and per-string lengths. The existing FP16 scene mean is not charged to context because the no-context codec already transmits it.
- `Score path KiB` is exact parameter plus registered-buffer storage after checkpoint load. The full training checkpoint size is not a deployment-model measurement because it can contain optimizer state.
- CUDA peak values are incremental peak-allocation deltas and omit native CPU/rANS allocations. Use medians and paired deltas; absolute latency is hardware/software specific.

# NFCGS paired scene confidence intervals

Scenes: 6991; paired bootstrap draws: 20,000; confidence level: 95.0%.

Every comparison resamples complete scenes and keeps all four models and both lambdas paired. Positive rate saving, PSNR delta, SSIM delta, and LPIPS improvement all favor the candidate.

## Mean across scenes

These intervals describe scene-to-scene sampling variation. KiB uses 1024 bytes.

| Model | lambda | Mean KiB [CI] | PSNR dB [CI] | SSIM [CI] | LPIPS [CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear existing (Factorized) | 0.0064 | 90.805 [90.669, 90.937] | 24.3024 [24.2348, 24.3713] | 0.76020 [0.75765, 0.76283] | 0.23742 [0.23582, 0.23899] |
| Linear existing (Factorized) | 0.0256 | 42.258 [42.200, 42.315] | 23.4875 [23.4235, 23.5527] | 0.73071 [0.72807, 0.73341] | 0.27294 [0.27134, 0.27452] |
| Nonlinear existing (Factorized) | 0.0064 | 87.840 [87.703, 87.973] | 24.3022 [24.2344, 24.3712] | 0.76024 [0.75769, 0.76286] | 0.23765 [0.23604, 0.23923] |
| Nonlinear existing (Factorized) | 0.0256 | 41.256 [41.198, 41.314] | 23.5210 [23.4567, 23.5862] | 0.73280 [0.73016, 0.73549] | 0.26929 [0.26768, 0.27087] |
| Linear context Full | 0.0064 | 84.905 [84.823, 84.988] | 24.3232 [24.2557, 24.3920] | 0.76114 [0.75860, 0.76375] | 0.23679 [0.23520, 0.23837] |
| Linear context Full | 0.0256 | 39.233 [39.192, 39.273] | 23.5352 [23.4708, 23.6003] | 0.73226 [0.72964, 0.73496] | 0.27157 [0.26996, 0.27315] |
| Nonlinear Full+Split | 0.0064 | 81.174 [81.093, 81.254] | 24.3238 [24.2563, 24.3928] | 0.76110 [0.75855, 0.76371] | 0.23706 [0.23546, 0.23863] |
| Nonlinear Full+Split | 0.0256 | 36.920 [36.881, 36.959] | 23.5707 [23.5062, 23.6362] | 0.73418 [0.73156, 0.73687] | 0.26833 [0.26671, 0.26991] |

## Paired differences: lambda 0.0064

| Candidate vs reference | Rate saving % [CI] | PSNR delta dB [CI] | SSIM delta [CI] | LPIPS improvement [CI] | Rate/PSNR/joint scene wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nonlinear existing (Factorized) vs Linear existing (Factorized) | +3.265% [+3.256, +3.274] | -0.0002 [-0.0019, +0.0015] | +0.00004 [-0.00001, +0.00009] | -0.00023 [-0.00029, -0.00016] | 100.0% / 49.6% / 49.6% |
| Linear context Full vs Linear existing (Factorized) | +6.497% [+6.412, +6.581] | +0.0208 [+0.0194, +0.0223] | +0.00094 [+0.00089, +0.00100] | +0.00063 [+0.00056, +0.00069] | 98.8% / 59.7% / 58.6% |
| Nonlinear Full+Split vs Linear existing (Factorized) | +10.607% [+10.525, +10.687] | +0.0214 [+0.0196, +0.0234] | +0.00089 [+0.00083, +0.00096] | +0.00036 [+0.00029, +0.00043] | 100.0% / 59.0% / 59.0% |
| Linear context Full vs Nonlinear existing (Factorized) | +3.341% [+3.248, +3.433] | +0.0210 [+0.0192, +0.0228] | +0.00090 [+0.00084, +0.00097] | +0.00085 [+0.00078, +0.00093] | 83.6% / 59.9% / 47.2% |
| Nonlinear Full+Split vs Nonlinear existing (Factorized) | +7.589% [+7.501, +7.677] | +0.0217 [+0.0202, +0.0231] | +0.00085 [+0.00080, +0.00091] | +0.00059 [+0.00053, +0.00065] | 99.6% / 61.8% / 61.4% |
| Nonlinear Full+Split vs Linear context Full | +4.395% [+4.384, +4.406] | +0.0006 [-0.0010, +0.0023] | -0.00005 [-0.00010, +0.00000] | -0.00026 [-0.00033, -0.00020] | 100.0% / 49.3% / 49.3% |

## Paired differences: lambda 0.0256

| Candidate vs reference | Rate saving % [CI] | PSNR delta dB [CI] | SSIM delta [CI] | LPIPS improvement [CI] | Rate/PSNR/joint scene wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nonlinear existing (Factorized) vs Linear existing (Factorized) | +2.371% [+2.359, +2.383] | +0.0334 [+0.0311, +0.0357] | +0.00209 [+0.00201, +0.00217] | +0.00365 [+0.00356, +0.00375] | 100.0% / 65.7% / 65.7% |
| Linear context Full vs Linear existing (Factorized) | +7.159% [+7.076, +7.242] | +0.0476 [+0.0455, +0.0497] | +0.00155 [+0.00148, +0.00162] | +0.00137 [+0.00129, +0.00146] | 98.9% / 67.3% / 66.3% |
| Nonlinear Full+Split vs Linear existing (Factorized) | +12.631% [+12.557, +12.703] | +0.0832 [+0.0806, +0.0858] | +0.00347 [+0.00338, +0.00356] | +0.00462 [+0.00451, +0.00473] | 100.0% / 78.7% / 78.7% |
| Linear context Full vs Nonlinear existing (Factorized) | +4.905% [+4.820, +4.988] | +0.0142 [+0.0115, +0.0169] | -0.00054 [-0.00064, -0.00044] | -0.00228 [-0.00238, -0.00218] | 95.2% / 52.3% / 48.6% |
| Nonlinear Full+Split vs Nonlinear existing (Factorized) | +10.509% [+10.435, +10.582] | +0.0498 [+0.0478, +0.0518] | +0.00138 [+0.00131, +0.00145] | +0.00097 [+0.00089, +0.00105] | 100.0% / 71.4% / 71.4% |
| Nonlinear Full+Split vs Linear context Full | +5.894% [+5.878, +5.909] | +0.0356 [+0.0332, +0.0380] | +0.00192 [+0.00183, +0.00200] | +0.00325 [+0.00315, +0.00334] | 100.0% / 65.1% / 65.1% |

## Approximate two-point BD-rate

Only two lambda points are available, so this is a log-rate/PSNR linear interpolation screen, not a publication-grade four-point BD-rate. Negative values favor the candidate.

| Candidate vs reference | Two-point BD-rate % [CI] |
| --- | ---: |
| Nonlinear existing (Factorized) vs Linear existing (Factorized) | -4.322% [-4.458, -4.190] |
| Linear context Full vs Linear existing (Factorized) | -9.812% [-9.919, -9.706] |
| Nonlinear Full+Split vs Linear existing (Factorized) | -15.957% [-16.094, -15.820] |
| Linear context Full vs Nonlinear existing (Factorized) | -5.758% [-5.914, -5.599] |
| Nonlinear Full+Split vs Nonlinear existing (Factorized) | -12.224% [-12.338, -12.108] |
| Nonlinear Full+Split vs Linear context Full | -6.816% [-6.955, -6.678] |

## Statistical scope

- The resampling unit is a scene; the eight target frames already averaged inside each scene are not treated as independent samples.
- These are paired percentile-bootstrap intervals for test-scene sampling uncertainty. They do not include training-seed, checkpoint-selection, dataset-shift, or hardware uncertainty.
- Pairwise intervals are unadjusted for multiple comparisons. Use the JSON estimates rather than rounded Markdown values for downstream analysis.
- Exact scene sets, context frames, and target frames were required to match across all eight eval result files before any interval was computed.

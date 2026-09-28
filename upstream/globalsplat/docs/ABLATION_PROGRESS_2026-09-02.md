# GlobalSplat + NFC-GS codec: implementation and ablation status

Status snapshot: 2026-09-02 (Asia/Seoul)

## 1. Objective and implementation scope

The GlobalSplat codebase was extended with an optional scene-token codec at the
actual encoder/decoder boundary. The implemented data path is:

```text
GlobalSplat encoder
  -> appearance [B, 4096, 512]
  -> geometry   [B, 4096, 512]
  -> output-preserving observable geometry projection 512 -> 224
  -> concat [B, 4096, 736]
  -> per-scene centering
  -> optional Morton ordering
  -> learned low-rank analysis / score normalization / entropy coding
  -> low-rank synthesis + residual reconstruction
  -> optional multiscale 1-D residual hyperprior
  -> split appearance [512] / observable geometry [224]
  -> GlobalSplat Gaussian decoder
  -> 32,768 Gaussians (4096 tokens x stage multiplier 8)
```

The sender/receiver path uses a real per-scene bitstream. Evaluation performs no
SGA+, per-scene fitting, gradient update, or test-time overfitting. Codec model
weights are shared and are not counted in the per-scene payload.

Key implementation areas:

- `globalsplat/compression/`: codec, entropy models, residual codec, Morton
  ordering, checkpoint conversion, and bitstream containers.
- `globalsplat/model/globalsplat.py`: codec insertion at the packed scene-token
  boundary plus `compress_scene` and `decompress_scene`.
- `globalsplat/model/model_wrapper.py`: rate-distortion training and actual
  bitstream evaluation metrics.
- Deterministic all-scene RE10K sampler and Slurm scripts for experiments
  17 -> 16 -> 14 -> 15, the residual-ordering follow-up (15b), and the
  protocol-matched full-test lambda sweep.

## 2. Common experiment protocol

| Item | Setting |
|---|---|
| Dataset | RealEstate10K (RE10K) |
| Evaluation split | raw test split, every processable scene |
| Context / target views | 12 context + 8 disjoint target views |
| Scene tokens | 4,096 |
| Gaussian count | 32,768 |
| Appearance / raw geometry channels | 512 / 512 |
| Observable geometry channels | 224 |
| Concatenated codec input | 736 channels |
| Test-time optimization | none |
| Metrics | PSNR, SSIM, LPIPS, actual bytes, bits/Gaussian, BPGA |

Scenes with insufficient frames, unsupported image shapes, or invalid camera
metadata are skipped by the upstream loader. All completed full-test analyses
and evaluations use the same 6,991 processable batches/scenes.

Dense per-scene scene-token sizes before entropy coding are:

- Original 1024-D token: `4096 x 1024 x FP32 = 16,777,216 bytes` (16 MiB).
- Observable 736-D token: `4096 x 736 x FP32 = 12,058,624 bytes` (11.5 MiB).

### Experiment-by-experiment setting matrix

The following table separates newly trained models from evaluation-only
ablations. Unless stated otherwise, evaluation uses the common all-scene
RE10K C12/T8 protocol above, batch size 1, bf16 mixed precision, and no
test-time gradient update.

| Experiment | Purpose | Train or eval | Model/checkpoint | Variable under test | Ordering | Active coded paths | Actual bitstream |
|---|---|---|---|---|---|---|---|
| 17 | Separate channel conversion, projection, quantization, and residual effects | Eval only | Vanilla control plus independently trained rank 40/56/72/80, lambda=0.0256, step-50k full-codec checkpoints | Rank and cumulative mode: `upper`, `concat_only`, `projection`, `projection_quantized`, `full` | none for controls; Morton for rank modes | Depends on cumulative mode | Only `projection_quantized` and `full` |
| 16 | Measure channel/token correlation and covariance-spectrum concentration | Analysis only | Archived rank-56, lambda=0.0256, step-50k full-codec checkpoint | Appearance/raw geometry/observable geometry/concat statistics; unsorted versus Morton token lags; energy at ranks 40/56/72/80 | Both unsorted and Morton statistics | No image codec evaluation | No |
| 14 | Compare linear, nonlinear, and residual structures under a common fresh-training budget | Three new 50k-step training runs, then eval | Initialized from official GlobalSplat with observable reparameterization; no PCA `CODEC_INIT` was supplied in the completed run | `linear_no_residual`, `nonlinear32_no_residual`, `linear_residual` | Morton | Score path for all; residual `Qy/Qz` only for `linear_residual` | Yes |
| 15 | Verify ordering invariance without an order-dependent module | Eval only | Experiment-14 `linear_no_residual` checkpoint | none versus Morton versus deterministic random inference order | Varied at inference | Quantized score `Qs`; no residual path | Yes |
| 15b | Measure ordering sensitivity of the trained 1-D residual codec | Eval only | Fixed archived Morton-trained rank-56, lambda=0.0256, step-50k full-codec checkpoint | none versus Morton versus deterministic random inference order | Varied at inference | `Qs`, multiscale 1-D residual, `Qy`, and `Qz` | Yes |
| Full RD | Obtain a protocol-matched rate-distortion curve | Eval only | Official GlobalSplat plus four archived rank-56 full-codec checkpoints | lambda 0.0004/0.0016/0.0064/0.0256 | Morton | Full score and residual paths (`Qs/Qy/Qz`) | Codec points only |

### Shared training recipe for codec checkpoints

The archived full-codec checkpoints and the newly trained experiment-14
models freeze the GlobalSplat backbone and optimize only codec parameters with
the rendering-plus-rate objective. The core recipe is:

| Item | Setting |
|---|---|
| Optimizer | Adam, LR `1e-4`, weight decay 0 |
| Schedule | LR milestones 35k and 45k, gamma 0.1 |
| Budget | 50,000 optimizer steps |
| Loss | RGB/MSE 1.0 + LPIPS 0.05 + in-view 0.01 + lambda x likelihood BPGA |
| Entropy update | Deterministic quantile update every 500 steps |
| GlobalSplat | Frozen official RE10K-32K weights |
| Codec boundary | appearance 512 + observable geometry 224 = 736 channels |

Experiment 14 used micro-batch 1 with gradient accumulation 8. The archived
rank-56 checkpoints used physical batch 8 with accumulation 1. Both have an
effective batch size of 8, but they are not numerically identical recipes in a
quantized entropy model.

Initialization is an important provenance difference. The archived rank-56
full checkpoints used a RE10K-derived PCA/codec initialization
(`re10k_ctx12_s64_t512_rank56.pt`) in addition to the output-preserving
observable geometry conversion. The completed experiment-14 jobs left
`CODEC_INIT` empty, so their low-rank and entropy/residual codec components did
not start from that matched PCA artifact. Consequently, experiment 14's
`linear_residual` has the same topology as the archived full codec but is not
an exact reproduction control.

## 3. Experiment 17: isolate channel-reduction effects

### Conditions

- `upper`: original GlobalSplat-32K.
- `concat_only`: output-preserving geometry conversion and concat/split only.
- `projection`: centering plus continuous learned linear low-rank projection and
  reconstruction; no quantization and no residual payload.
- `projection_quantized`: projection plus hard score quantization and a real
  score bitstream; no residual.
- `full`: quantized low-rank base plus multiscale 1-D residual bitstream.

All rank checkpoints are lambda=0.0256, step 50,000, untied linear synthesis
checkpoints. Ranks 40, 56, 72, and 80 are independently trained models, so their
quality is not required to be strictly monotonic with rank.

### Ablation methodology

Experiment 17 is a **cumulative stage-wise ablation**, not a leave-one-module-out
ablation of the full sender/receiver. Modules are enabled from left to right:

```text
upper
  -> concat_only
  -> projection
  -> projection_quantized
  -> full
```

| Condition | Observable geometry + concat | Centering + low-rank | Score Qs / score bitstream | Residual codec + Qy/Qz |
|---|:---:|:---:|:---:|:---:|
| upper | no | no | no | no |
| concat_only | yes | no | no | no |
| projection | yes | continuous | no | no |
| projection_quantized | yes | yes | yes | no |
| full | yes | yes | yes | yes |

Within each rank, `projection`, `projection_quantized`, and `full` use the same
rank-specific checkpoint that was jointly trained as a full residual codec.
Evaluation changes only `feature_codec.ablation_mode`; these three conditions
were not independently retrained. `upper` instead uses the vanilla GlobalSplat
checkpoint, while `concat_only` uses a vanilla-initialized output-preserving
identity checkpoint.

Exact mode behavior:

- `projection` uses an unquantized scene mean and continuous analysis/synthesis;
  it has no entropy model, quantizer, residual, or actual bitstream.
- `projection_quantized` additionally quantizes the scene mean to FP16,
  normalizes and quantizes the low-rank score with `Qs`, and writes a real score
  bitstream. Its difference from `projection` therefore includes mean FP16
  quantization as well as `Qs`; entropy coding itself is lossless after symbol
  quantization.
- `full` adds residual analysis/synthesis, the main residual quantizer `Qy`, the
  hyper-latent quantizer `Qz`, and their actual residual payload.

Consequently, experiment 17 cleanly reports stage-wise behavior of a jointly
trained full model, but its projection-only result can include co-adaptation to
the disabled residual path. Experiment 14 complements it by training the
no-residual structures independently from the start.

### Full RE10K test results

Lower LPIPS is better. `--` means that the condition is continuous or a control
and therefore does not produce an actual compressed scene payload.

| Condition | PSNR | SSIM | LPIPS | Bytes/scene | bits/Gaussian | BPGA |
|---|---:|---:|---:|---:|---:|---:|
| upper | 24.7004 | 0.7682 | 0.2480 | -- | -- | -- |
| concat_only | 24.6376 | 0.7664 | 0.2494 | -- | -- | -- |
| rank40_projection | 18.3502 | 0.6700 | 0.3621 | -- | -- | -- |
| rank40_projection_quantized | 17.9304 | 0.6428 | 0.4119 | 43,797 | 10.6927 | 0.181232 |
| rank40_full | 23.3062 | 0.7272 | 0.2889 | 54,735 | 13.3631 | 0.226493 |
| rank56_projection | 20.5585 | 0.7015 | 0.3264 | -- | -- | -- |
| rank56_projection_quantized | 20.2568 | 0.6813 | 0.3611 | 47,894 | 11.6929 | 0.198184 |
| rank56_full | **23.3700** | 0.7290 | 0.2870 | 58,444 | 14.2686 | 0.241841 |
| rank72_projection | 19.4533 | 0.6853 | 0.3081 | -- | -- | -- |
| rank72_projection_quantized | 19.2073 | 0.6670 | 0.3424 | 52,069 | 12.7121 | 0.215460 |
| rank72_full | 23.1918 | **0.7307** | 0.2864 | 94,217 | 23.0022 | 0.389867 |
| rank80_projection | 20.1058 | 0.6912 | 0.3021 | -- | -- | -- |
| rank80_projection_quantized | 19.8696 | 0.6756 | 0.3303 | 54,283 | 13.2527 | 0.224622 |
| rank80_full | 23.0444 | 0.7305 | **0.2836** | 97,568 | 23.8202 | 0.403733 |

Bold values mark the best quality among the four `full` codec ranks, not the
codec-free reference. The historical condition name `upper` means the vanilla
GlobalSplat reference; it is not a strict mathematical upper bound because a
task-trained codec can regularize or correct frozen backbone features.

### Stage-wise distortion separation

| Rank | Projection loss vs concat (dB) | Quantization loss (dB) | Residual recovery (dB) | Final loss vs upper (dB) |
|---:|---:|---:|---:|---:|
| 40 | -6.2874 | -0.4198 | +5.3758 | -1.3942 |
| 56 | -4.0791 | -0.3017 | +3.1132 | -1.3304 |
| 72 | -5.1843 | -0.2460 | +3.9845 | -1.5086 |
| 80 | -4.5318 | -0.2362 | +3.1748 | -1.6560 |

### Experiment-17 conclusions

1. The observable geometry conversion and concat/split are effectively
   lossless at the rendered-image level: only -0.0628 dB PSNR, -0.0018 SSIM,
   and +0.0014 LPIPS relative to original GlobalSplat.
2. The dominant distortion is the low-rank projection, not entropy coding.
   Projection alone loses roughly 4.1-6.3 dB relative to `concat_only`.
3. Hard quantization adds a much smaller 0.24-0.42 dB loss. This supports
   focusing model-design effort on projection/reconstruction capacity before
   tuning the entropy coder.
4. The residual path is doing substantial work, recovering 3.1-5.4 dB. The
   current full result therefore cannot be described as a low-rank-only codec.
5. Rank 56 gives the highest full-codec PSNR (23.3700 dB). Rank 40 is close
   (-0.0638 dB) while using about 3.7 KB fewer bytes per scene, so it is the
   stronger rate-quality operating point for PSNR-oriented use.
6. Ranks 72 and 80 have much larger residual payloads (about 94-98 KB versus
   55-58 KB for ranks 40/56) without a PSNR gain. They are dominated by rank 56
   in PSNR-rate space. Rank 80 does achieve the best full-codec LPIPS.
7. Relative to the dense observable FP32 token tensor, full rank 40 and rank 56
   reduce the per-scene payload by about 220x and 206x respectively. This ratio
   excludes shared model weights and is not an end-to-end dataset storage ratio.

## 4. Experiment 16: correlation and low-rank evidence

Status: completed successfully (`409784`). The analysis traversed all 6,991
processable RE10K test scenes using 64 evenly spaced tokens per scene for the
covariance estimate and 32 evenly spaced channels for token-lag statistics.
Each stream is centered by its per-scene token mean, matching the codec.

Generated outputs on the cluster:

```text
outputs/ablation_16_correlation/summary.json
outputs/ablation_16_correlation/correlation_and_spectrum.pt
outputs/ablation_16_correlation/correlation_report.png
outputs/ablation_16_correlation/channel_correlation_heatmaps.png
outputs/ablation_16_correlation/singular_value_spectrum.png
```

### Channel correlation and covariance-spectrum results

`mean |off-diagonal Pearson|` measures pairwise channel correlation after
per-scene token-mean removal. Rank energy is cumulative covariance eigenvalue
energy; it is not computed from a channel-standardized correlation matrix.

| Stream | Mean abs. off-diagonal correlation | Rank 40 | Rank 56 | Rank 72 | Rank 80 |
|---|---:|---:|---:|---:|---:|
| appearance | 0.1241 | 88.80% | 91.00% | 92.49% | 93.08% |
| raw geometry | 0.0905 | 98.63% | 98.90% | 99.09% | 99.16% |
| observable geometry | 0.4233 | 99.45% | 99.62% | 99.72% | 99.76% |
| observable concat | 0.1291 | 97.31% | 97.93% | 98.29% | 98.42% |

The joint 736-D concat tensor therefore has a strongly concentrated covariance
spectrum even though its average pairwise channel correlation is only 0.1291.
The geometry streams are especially low dimensional. Appearance has the much
heavier spectrum tail: rank 56 retains only 91.00% of its covariance energy,
versus 99.62% for observable geometry. Because the covariance spectrum is not
channel-standardized, high-variance geometry directions can dominate the joint
concat energy; 97.93% concat energy does not imply that every perceptually
important appearance direction is preserved.

### Token-axis correlation and Morton ordering

Mean absolute per-channel Pearson correlation at representative token lags:

| Stream | Lag 1 | Lag 2 | Lag 4 | Lag 16 | Lag 64 | Lag 256 |
|---|---:|---:|---:|---:|---:|---:|
| appearance | 0.0018 | 0.0024 | 0.0049 | 0.0029 | 0.0026 | 0.0016 |
| raw geometry | 0.0050 | 0.0046 | 0.0068 | 0.0072 | 0.0050 | 0.0056 |
| observable geometry | 0.0085 | 0.0085 | 0.0137 | 0.0102 | 0.0060 | 0.0059 |
| observable concat, unsorted | 0.0055 | 0.0053 | 0.0077 | 0.0043 | 0.0037 | 0.0033 |
| observable concat, Morton | **0.4739** | **0.4485** | **0.4174** | **0.3478** | **0.2605** | **0.1369** |

Unsorted GlobalSplat tokens have almost no local token-axis correlation.
Geometry-guided Morton sorting creates strong and long-range locality, raising
lag-1 correlation from 0.0055 to 0.4739. This directly supports Morton ordering
for the order-dependent 1-D residual analysis/synthesis branch. It does not by
itself predict a gain for a pointwise low-rank-only model; experiment 15 tests
that required invariance separately.

### Experiment-16 conclusions

1. There is genuine low-dimensional covariance structure: rank 40 already
   captures 97.31% of concat covariance energy, and rank 56 captures 97.93%.
2. Most of that concentration comes from geometry. Appearance is the harder
   component, retaining only 88.80%/91.00% at ranks 40/56.
3. The high retained covariance energy does not guarantee rendered quality.
   Experiment 17 still loses 4.08 dB at rank 56 before quantization, showing
   that low-energy directions can be rendering-critical and/or that the learned
   transform is co-adapted with its residual path.
4. Statistical gains beyond rank 56 are small: concat energy rises by only
   0.36 percentage points at rank 72 and another 0.13 points at rank 80. This
   agrees with the lack of PSNR-rate improvement at ranks 72/80 in experiment
   17.
5. Morton sorting has a clear structural purpose: it transforms a nearly
   uncorrelated token sequence into one with strong local correlation for the
   1-D residual codec.

## 5. Experiment 14: linear versus nonlinear structure

Three rank-56 codecs are trained under the same RE10K protocol:

| Task | Condition | Projection structure | Residual |
|---:|---|---|---|
| 0 | linear_no_residual | untied affine/linear | none |
| 1 | nonlinear32_no_residual | linear path + GELU MLP, hidden 32 | none |
| 2 | linear_residual | untied affine/linear | multiscale 1-D |

Training settings:

- Frozen GlobalSplat backbone; codec is trainable.
- Adam, LR `1e-4`, milestones 35k/45k, 50,000 optimizer steps.
- Micro-batch 1, gradient accumulation 8 (effective batch 8).
- bf16 mixed precision, lambda 0.0256, deterministic entropy-quantile update
  every 500 steps.
- RGB/MSE + LPIPS + in-view + likelihood-based BPGA objective.

The first submission (`409785`) failed on the first accumulated optimizer step
because fused Adam was incompatible with Lightning AMP gradient clipping. This
was an infrastructure failure, not a model or data result. The optimizer now
defaults to non-fused Adam, and the Slurm scripts locate the newest
`version_N/last.ckpt` so failed/restarted runs remain resumable.

The corrected training array `410944` completed all three conditions at
`max_steps=50000`, and each condition wrote a `version_1/last.ckpt`. The
dependent full-test evaluation array `410947` and summary job `410948` also
completed successfully. Evaluation used all 6,991 processable RE10K test
scenes and actual entropy bitstreams.

### Full RE10K test results

| Condition | PSNR | SSIM | LPIPS | Bytes/scene | bits/Gaussian | BPGA |
|---|---:|---:|---:|---:|---:|---:|
| linear_no_residual | 20.5875 | 0.6347 | 0.3990 | 55,625 | 13.5804 | 0.230177 |
| nonlinear32_no_residual | **21.4757** | **0.6649** | **0.3387** | 51,391 | 12.5467 | 0.212655 |
| linear_residual | 20.1332 | 0.5971 | 0.4011 | **9,581** | **2.3391** | **0.039645** |

Final reconstructed-feature MSE (after the active quantized sender/receiver
path) was:

| Condition | Total feature MSE | Appearance MSE | Observable-geometry MSE |
|---|---:|---:|---:|
| linear_no_residual | 0.5992 | 0.5099 | 0.8031 |
| nonlinear32_no_residual | 0.5559 | 0.4544 | 0.7877 |
| linear_residual | **0.4006** | **0.2372** | **0.7743** |

At the same rank, lambda, optimizer, and step budget, the nonlinear no-residual
transform strictly improves on the linear no-residual transform: +0.8882 dB
PSNR, +0.0302 SSIM, -0.0603 LPIPS, and 4.23 KB fewer bytes per scene. This is
evidence that the pointwise transform benefits from nonlinear capacity.

The freshly trained linear+residual model is not a same-rate quality winner. It
converged to an extremely low-rate operating point (9.58 KB/scene), with 0.4543
dB lower PSNR than the linear no-residual model. This is consistent with a
rate-dominated/collapsed solution at lambda 0.0256, not evidence that the
residual architecture is intrinsically harmful. It is also substantially
different from experiment 17's archived rank-56 full checkpoint (23.3700 dB,
58.44 KB/scene), so initialization/training-history and rate allocation must be
audited before drawing a clean residual-versus-no-residual conclusion.

The per-scene rate distribution confirms that 9.58 KB is not caused by a few
outliers: the linear+residual 5th/median/95th percentiles are
8.39/9.42/11.34 KB, versus 49.81/53.48/70.68 KB for linear no-residual and
46.00/48.52/69.08 KB for nonlinear no-residual. Interestingly, the residual
model has the lowest unweighted feature MSE but the worst rendered SSIM and
slightly worst LPIPS. Feature-space MSE therefore is not a reliable substitute
for task/rendering distortion; low-variance or decoder-sensitive directions
matter disproportionately.

The `srun` teardown emitted several PMIx/socket warnings in the training stderr
files, but they occurred after Lightning reported `max_steps=50000` and after
the final checkpoints were written. The dependent `afterok` evaluation ran and
all three evaluations completed without a traceback, so these are not failed
training runs.

## 6. Experiment 15: Morton-ordering ablation

Status: completed successfully. Evaluation array `410949` processed all 6,991
test scenes for every condition, and summary job `410950` merged all three
outputs without an error.

Experiment 15 reuses experiment 14's `linear_no_residual` checkpoint and
evaluates three inference-time orderings:

- no sorting;
- Morton sorting;
- deterministic random permutation.

Because the selected model contains no residual convolution or other
order-dependent learned module, image quality should be invariant up to small
floating-point/numerical effects. A meaningful quality or rate difference would
indicate an implementation/order bookkeeping issue. This is why experiment 15
is intentionally last.

### Full RE10K test results

| Ordering | PSNR | SSIM | LPIPS | Bytes/scene | bits/Gaussian | BPGA |
|---|---:|---:|---:|---:|---:|---:|
| none | 20.5875 | 0.6347 | 0.3990 | 55,625.397797 | 13.580419 | 0.230177 |
| Morton | 20.5875 | 0.6347 | 0.3990 | 55,625.396653 | 13.580419 | 0.230177 |
| deterministic random | 20.5875 | 0.6347 | 0.3990 | 55,625.397225 | 13.580419 | 0.230177 |

The three orderings are identical at the reported quality precision. The
largest difference in mean payload is only 0.001144 byte/scene, which is
negligible coder/aggregation noise. The permutation and inverse-permutation
bookkeeping is therefore correct, and Morton sorting has no intrinsic effect
when the codec contains no order-dependent operation. Combined with experiment
16, this gives the intended interpretation: Morton ordering creates strong
local token correlation for a 1-D residual model, but does not itself improve a
pointwise low-rank codec.

## 7. Experiment 15b: ordering sensitivity with the residual codec

Status: completed successfully. Evaluation array `411709` processed all 6,991
processable RE10K test scenes for every condition, and dependent summary job
`411710` merged the three outputs without an error. The stderr files contain
only framework/deprecation warnings; there is no traceback, OOM, cancellation,
or failed task.

This follow-up keeps one archived rank-56, lambda=0.0256, Morton-trained full
residual checkpoint fixed and changes only inference-time ordering:

- no sorting;
- Morton sorting;
- deterministic random permutation.

All three score/residual quantizers (`Qs`, `Qy`, and `Qz`), the multiscale 1-D
residual codec, and actual entropy bitstreams are enabled. Evaluation uses all
processable RE10K test scenes with no SGA+ or per-scene optimization. The GPU
array is pinned to `ariel-v10`.

### Full RE10K test results

| Inference ordering | PSNR | SSIM | LPIPS | Bytes/scene | bits/Gaussian | BPGA |
|---|---:|---:|---:|---:|---:|---:|
| none | 22.8059 | 0.7232 | 0.3068 | 73,978.984981 | 18.061276 | 0.306123 |
| **Morton** | **23.3700** | **0.7290** | **0.2870** | **58,444.196824** | **14.268603** | **0.241841** |
| deterministic random | 22.7935 | 0.7230 | 0.3071 | 74,491.973108 | 18.186517 | 0.308246 |

Relative to no sorting, Morton improves PSNR by 0.5641 dB and SSIM by
0.0058, reduces LPIPS by 0.0198, and saves 15,534.79 bytes/scene (21.0%).
Relative to deterministic random ordering, it improves PSNR by 0.5765 dB and
saves 16,047.78 bytes/scene (21.5%). None and random are close to each other,
which isolates the gain to the spatial locality supplied by Morton rather than
to permutation alone.

The Morton row exactly reproduces the protocol-matched archived full-codec
lambda=0.0256 point (23.3700 dB, 58,444.20 bytes/scene). This confirms that the
follow-up changed only inference ordering and did not alter the checkpoint or
evaluation recipe. In contrast, experiment 15 used a no-residual pointwise
checkpoint and was invariant to ordering. Together, experiments 15 and 15b
show that Morton has no intrinsic effect on pointwise low-rank coding but is
important once the order-dependent multiscale 1-D residual codec is active.

This is an inference-time **sensitivity/distribution-shift test**, not a fair
comparison of three ordering-specific training recipes: the fixed checkpoint
was trained with Morton ordering. A degradation for no-sort or random would
show that the trained residual model relies on Morton locality. A causal claim
that Morton is better during training would require separately training the
three orderings under matched budgets.

Output root and merged summary:

```text
outputs/ablation_15b_ordering_with_residual/
outputs/ablation_15b_ordering_with_residual/summary.csv
```

## 8. Rate-distortion curve status

### Preliminary 30-scene zero-update sweep

The archived rank-56 full-codec lambda sweep uses the same selected 30 RE10K
scenes for every point and reports `step_0`, before any SGA+/per-scene update:

| Lambda | Bytes/scene | BPGA | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|---:|---:|
| 0.0004 | 248,148 | 1.026831 | 26.4208 | 0.8443 | 0.1706 |
| 0.0016 | 175,266 | 0.725246 | 26.2737 | 0.8405 | 0.1741 |
| 0.0064 | 108,876 | 0.450528 | 25.7577 | 0.8270 | 0.1874 |
| 0.0256 | 49,796 | 0.206056 | 24.3676 | 0.7847 | 0.2255 |

The internal trend is a valid preliminary RD curve, but these points must not
be plotted against the 6,991-scene GlobalSplat mean: the 30-scene subset is
easier and uses a different scene distribution. The x-axis is actual bitstream
size (KB/scene) or BPGA, not model size.

### Protocol-matched full-test sweep

Status: completed successfully. Evaluation array `411698` ran the codec-free
GlobalSplat reference and all four lambda checkpoints on the identical 6,991
scene RE10K C12/T8 protocol. Summary job `411699` merged all five conditions.
The jobs were pinned to `ariel-v10`, used actual full-codec bitstreams, and
performed no test-time optimization.

| Condition | Bytes/scene | bits/Gaussian | BPGA | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| codec-free GlobalSplat | -- | -- | -- | 24.7004 | 0.7682 | 0.2480 |
| lambda 0.0004 | 385,759 | 94.1795 | 1.596262 | **24.9867** | **0.7832** | **0.2180** |
| lambda 0.0016 | 354,285 | 86.4953 | 1.466022 | 24.8686 | 0.7792 | 0.2223 |
| lambda 0.0064 | 128,360 | 31.3378 | 0.531149 | 24.5563 | 0.7682 | 0.2353 |
| lambda 0.0256 | 58,444 | 14.2686 | 0.241841 | 23.3700 | 0.7290 | 0.2870 |

The four codec points form the expected monotonic RD curve: increasing lambda
reduces the actual payload and degrades PSNR/SSIM/LPIPS. At lambda 0.0064, the
codec retains nearly the codec-free PSNR (-0.1441 dB) and identical rounded
SSIM while improving LPIPS by 0.0127 at 128.36 KB/scene. At the two highest-rate
points, the codec exceeds the codec-free reference on all three quality
metrics: lambda 0.0004 gains 0.2863 dB and lambda 0.0016 gains 0.1682 dB. This
is possible because the learned transform is optimized for rendering and can
regularize/correct the frozen representation; `codec-free reference` is the
correct label, not `upper bound`.

Relative to the original 16-MiB FP32 scene-token tensor, the four codec rates
correspond to approximately 43.5x, 47.4x, 130.7x, and 287.1x payload reduction
for lambdas 0.0004, 0.0016, 0.0064, and 0.0256 respectively. These ratios
exclude shared codec weights.

Default output root:

```text
outputs/full_rd_allscene_rank56/
```

## 9. Current evidence and next decisions

Confirmed now:

- GlobalSplat + real codec sender/receiver evaluation works on the full
  processable RE10K test split without per-scene optimization.
- The 512 -> 224 observable geometry conversion is not the source of the main
  quality drop.
- Low-rank projection/reconstruction capacity is currently the main bottleneck.
- Quantization is a secondary distortion source.
- The residual codec restores most of the low-rank loss but can become
  rate-inefficient at ranks 72/80.
- Rank 40/56 are the useful region; rank 56 maximizes PSNR while rank 40 is more
  rate-efficient.
- A small nonlinear pointwise transform is strictly better than the freshly
  trained affine no-residual transform at lambda 0.0256, both in quality and
  actual rate.
- Fresh residual training at lambda 0.0256 can converge to a rate-dominated
  9.58-KB solution, so residual comparisons require matched-rate or multi-lambda
  evaluation rather than a single-lambda architecture claim.
- No-sort, Morton, and random ordering are quality/rate invariant for the
  no-residual pointwise codec, confirming correct permutation bookkeeping.
- With the Morton-trained multiscale 1-D residual codec active, Morton improves
  PSNR by 0.5641 dB and reduces the actual payload by 21.0% versus no sorting;
  the residual model therefore relies strongly on the locality created by its
  training ordering.
- Experiment 17 is cumulative stage-wise evaluation of full-trained
  checkpoints; its projection-only rows are not independently trained
  no-residual models.
- The protocol-matched rank-56 lambda sweep is monotonic in rate and quality.
  Lambda 0.0064 nearly matches codec-free PSNR at 128.36 KB/scene, while the two
  higher-rate points exceed the codec-free reference on PSNR, SSIM, and LPIPS.

Still required before the broader rate-distortion story is complete:

1. Audit why the fresh linear+residual run selected a much lower rate than the
   archived rank-56 full checkpoint, then compare residual and no-residual
   variants at matched rates or across multiple lambdas.

## 10. Relevant files

- `docs/NFCGS_CODEC.md`: architecture, protocol, and command reference.
- `scripts/slurm/17_channel_reduction.slurm`: experiment 17.
- `scripts/slurm/16_correlation_lowrank.slurm`: experiment 16.
- `scripts/slurm/14_train_structure.slurm`: experiment-14 training.
- `scripts/slurm/14_eval_structure.slurm`: experiment-14 evaluation.
- `scripts/slurm/15_morton_ordering.slurm`: experiment 15.
- `scripts/slurm/15b_morton_ordering_with_residual.slurm`: experiment-15b
  residual-ordering sensitivity evaluation, pinned to `ariel-v10`.
- `scripts/slurm/submit_ablation_15b_with_residual.sh`: submit experiment 15b
  and its dependent summary job.
- `scripts/slurm/eval_full_rd_allscene.slurm`: full-test codec-free plus
  four-lambda RD evaluation, pinned to `ariel-v10`.
- `scripts/slurm/submit_full_rd_allscene.sh`: submit the full-test RD array and
  dependent summary job.
- `scripts/slurm/submit_ablation_14_15.sh`: restart chain that preserves
  completed experiments 17 and 16.

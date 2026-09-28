# Same-day fixed-symbol entropy probe

Purpose: test whether the large score payload contains reducible probability-model
cost, and compare its achievable savings with the residual path. This is a small
held-out diagnostic, not a new full codec training run or a full-test result.

## Setup

- Parent: original Nonlinear32 Full P0 + Split, probability run 20260913_112316,
  step10000 (total lineage 110k). Task0=.0064, task1=.0256.
- Frozen backbone, transforms, predictors, quantization steps and medians.
- Fit probability tables on 128 unique RE10K TRAIN scenes; evaluate on 128 unique
  TEST scenes. Scene IDs are recorded, duplicate scenes skipped, train/test overlap
  rejected. Both use deterministic_all sampling, C12/T8, augmentation OFF.
- GPU1 on ariel-v12, BF16 outside the existing FP32 score-context boundary.
- No optimizer, rendering, saved images, or new model checkpoints. Integer-symbol
  caching is OFF by default; an explicit small cache is available for offline probes.
- Two fixed alternatives: marginal histogram recalibration and context-binned
  histogram recalibration. Both use shrinkage strength4096, declared before test.
- Marginal: score CDF per channel/even-odd/slice; residual y per existing scale-table
  index; residual z per channel. This tests probability calibration only.
- Context: score-even additionally uses scene quantization-step bins; score-odd
  uses mean absolute neighbouring decoded anchor residuals divided by scene step.
  Fixed edges: 0.5,1,2,4 (five bins). Final odd repeats the final even anchor when
  its right neighbour does not exist. Residual tables are the marginal alternative.
- Marginal counts shrink toward the parent CDF. Context counts shrink toward the
  fitted marginal. Unobserved rows fall back to their parent; no test adaptation.
- Uses the original CDF support/offset and CompressAI rANS escape convention.
  Integer CDFs are 16-bit, with positive frequencies including the escape event.

## Completed result (2026-09-14)

Both Slurm array elements completed with 128 unique TRAIN fit scenes and 128
unique TEST evaluation scenes.  Every candidate reproduced every native integer
stream and both decoded feature tensors exactly.  Both `.err` files are empty.
One bad-shape TEST example was skipped by the existing dataset loader; the probe
continued until 128 valid unique scenes were evaluated.

Sources: [.0064 stdout](../../../slurm/slurm-gs-entropy-probe-423055_0.out),
[.0256 stdout](../../../slurm/slurm-gs-entropy-probe-423055_1.out),
[.0064 summary](../../../entropy-probe-0p0064-summary.json),
[.0256 summary](../../../entropy-probe-0p0256-summary.json).

| Lambda | Native total | Marginal total / saving | Context total / saving | Context saving 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| .0064 | 81.253 KiB | 81.160 KiB / 0.115% | **80.654 KiB / 0.736%** | [0.656%, 0.821%] |
| .0256 | 36.923 KiB | 36.859 KiB / 0.172% | **36.642 KiB / 0.759%** | [0.673%, 0.855%] |

The context candidate saved 612.72 B/scene at `.0064` and 287.13 B/scene at
`.0256`.  The score-only contribution was respectively 0.713 and 0.738
percentage points of original total bytes.  Residual marginal recalibration
contributed only 0.024 and 0.021 points.  The paired score-minus-residual
interval was positive for the context candidate at both rates: `[0.608,0.773]`
and `[0.644,0.798]` percentage points.  This supports prioritizing score among
these probe families, but the comparison is asymmetric because only score got
the richer context candidate.

### Probability model versus coder loss

| Lambda/path | Native model NLL | Integer CDF + tail minus NLL | Actual rANS minus CDF + tail | Actual minus NLL |
| --- | ---: | ---: | ---: | ---: |
| .0064 score | 555,562.9 bit | 580.3 bit | 383.3 bit | 963.6 bit = 120.45 B |
| .0064 residual | 95,251.8 bit | 214.0 bit | 95.0 bit | 308.9 bit = 38.62 B |
| .0256 score | 239,309.8 bit | 230.6 bit | 384.4 bit | 615.0 bit = 76.87 B |
| .0256 residual | 48,814.6 bit | 99.7 bit | 96.9 bit | 196.6 bit = 24.58 B |

The arithmetic implementation is following its supplied probability model
closely.  Across score and residual together, finite-CDF/tail plus rANS cost over
the native model NLL was about 159 B/scene at `.0064` and 101 B/scene at `.0256`.
The rANS part alone was about 60 B/scene at both rates.  Roughly 48 bit per
entropy string is visible, so stream merging could recover only tens of bytes
and is not the main target.  The positive context-table result instead shows
that decoder-known conditioning is missing from the native probability model.
Model NLL is not an information-theoretic lower bound, so this accounting does
not cap what a different probability or representation model could achieve.

Most context savings came from early score groups.  At `.0064`, `g0_odd`,
`g0_even`, and `g1_odd` supplied 91.4% of the score saving; at `.0256` those
three supplied 82.3%, with `g3_odd` raising the four-stream share to 91.4%.
All odd streams together supplied 67.3% and 72.7% of score saving.  A follow-up
should therefore concentrate model capacity on early groups, especially odd,
rather than multiply tables and compute uniformly over every stream.

The compressed table bundle was 155,578 B and 123,195 B.  It contains native,
marginal, and context tables together and is therefore not the incremental cost
of one production candidate.  Even treating the whole bundle as overhead would
amortize after about 254 and 430 scenes respectively; single-scene deployment
must nevertheless account for shared model/table size explicitly.

### Decision from this screen

This is a real but small fixed-reconstruction opportunity: approximately
0.75% of scene payload with a deliberately simple five-bin model.  It is enough
to reject “the native score probabilities already exhaust decoder-available
context,” but not enough to explain the score path's roughly 80% byte share.
That share is mostly the information carried by the present score representation,
not rANS/container waste demonstrated by this probe.

A low-risk production follow-up is a selective nonparametric conditional-CDF
module for `g0/g1` and especially odd symbols, fitted on a larger TRAIN set and
selected on validation before one full TEST evaluation.  A more architectural
follow-up is a score hyperprior/coarse-to-fine latent: transmit a small score
side latent and use it to predict nonparametric score distributions while first
keeping the score quantization lattice frozen.  This cleanly tests whether a
stronger module earns back its side bits; only after that should it be trained
jointly with the transform.  Since the observed tabular gain is below 1%, this
entropy work is a cheap win rather than a reason to postpone representation and
transform redesign.

## Run on server

Copy these three files, preserving paths:

1. [entropy_probe.py](../globalsplat/compression/entropy_probe.py)
2. [probe_nfcgs_entropy.py](../scripts/probe_nfcgs_entropy.py)
3. [probe_nfcgs_entropy.slurm](../scripts/slurm/probe_nfcgs_entropy.slurm)

From the server repository root:

```bash
mkdir -p logs/slurm
sbatch --array=0-1 scripts/slurm/probe_nfcgs_entropy.slurm
```

For a smaller first screen, run ONE of the following alternatives instead:

```bash
# Only .0064, 64 fit + 64 test scenes.
PROBE_FIT_SCENES=64 PROBE_EVAL_SCENES=64 \
  sbatch --array=0 scripts/slurm/probe_nfcgs_entropy.slurm
```

Optional plan, with no checkpoint/data access or submission:

```bash
bash scripts/slurm/probe_nfcgs_entropy.slurm plan
PROBE_TASK=1 bash scripts/slurm/probe_nfcgs_entropy.slurm plan
```

To retain symbols for subsequent probability-model experiments, use this INSTEAD
of the default submission (not in addition to it):

```bash
# Cache only the first 2 TRAIN and 2 TEST scenes per lambda.
PROBE_CACHE_SCENES=2 sbatch --array=0-1 scripts/slurm/probe_nfcgs_entropy.slurm
```

Caching is opt-in because the recent low-LR run hit server storage exhaustion.
It saves compressed NumPy arrays without pickle: integer symbols, native indexes,
context bins, shapes, native strings and the native scene packet. A split-separated
manifest records scene/context-frame IDs. Native CDF tables accompany the cache;
richer learned feature conditioning still needs the parent checkpoint to decode it.
The example caches only four scenes, not a sufficient dataset for a strong model.

The normal run prints progress every16 scenes. Runtime depends on GPU scheduling,
dataset I/O and CDF sizes; the Slurm time limit is4h, not a runtime prediction.
It processes256 scenes per lambda rather than repeating6991-scene evaluation.

## Read the results

Output: outputs/nfcgs_entropy_probe/<run>/lambda<L>/job_<job>_<task>/

- REPORT.md: mean actual payload comparison; score-only/residual-only total savings
  and their difference with paired-scene 95% bootstrap intervals; per-stream bit audit.
- summary.json: per-stream floating-model NLL, integer-CDF bits, escape bypass bits,
  actual bits, score-only/residual-only savings relative to total scene bytes,
  paired-scene bootstrap intervals, exact-feature verification, scene IDs.
- scenes.jsonl: streamed scene-level results; available even if later interrupted.
- metadata.json and fit_scene_ids.json: checkpoint, dataset, runtime versions and
  completed TRAIN selection, saved before TEST results are complete.
- probability_tables.json.gz: native CDFs and fixed tables for both candidates.
  Its size is separate from per-scene payload; it is not the incremental size of
  a single deployable model.

Same native source symbols are rANS-recoded for each alternative. The diagnostic
receiver recomputes its own context from transmitted mean and decoded anchors,
checks all integer symbols, then checks torch.equal on reconstructed appearance
and geometry features against native decompression. Original likelihoods are
evaluated in FP32 at the actual captured integer symbols (without BF16
dequantize/requantize round-off); library likelihood-floor hits are counted.
The model-NLL/CDF gap can include scale-table approximation and likelihood floors.
Actual-CDF comparisons explicitly include tail bypass bits, so escapes are not
mistaken for unexplained coder overhead.

Experimental packets reuse the existing container layout IN MEMORY with the probe
decoder and its fixed CDF tables. They are not saved as production bitstreams and
cannot be decoded by the unmodified production entropy models. Actual byte counts
include unchanged headers; shared-table deployment costs are reported separately.
No PSNR is newly measured; exact feature equality establishes that probability
replacement did not change this checkpoint's reconstruction on the tested scenes.

Interpretation:

- Positive held-out score savings: evidence of reducible score probability cost
  for this model family; compare SCORE-ONLY total savings with RESIDUAL-ONLY savings.
- The score-minus-residual interval is for savings as percentage points of original
  TOTAL scene bytes, not each stream's own percentage. A positive lower endpoint
  favors score for these candidates on these scenes. Context expands score only;
  residual is exactly the marginal candidate. This is NOT a symmetric comparison
  of the strongest achievable probability models for the two paths.
- Large actual minus CDF+bypass: investigate codec/stream overhead.
- Context beats marginal: decoder-available anchor/scene-step context helps this
  tabular model beyond marginal recalibration.
- No improvement: this small, restricted probe failed to improve; it does NOT
  prove entropy optimality or establish that the transform is the bottleneck.
- Bootstrap intervals describe the selected scenes only; a first128-scene subset
  is not a representative full-test guarantee. Do not compare its mean directly
  with the historical full-test mean or tune choices on these TEST scenes.
- No controlled deployment latency is measured. Python wrappers and CDF conversion
  overhead make this diagnostic runtime unsuitable as a codec speed benchmark.

CDF escape accounting follows the public CompressAI1.2.8 rANS implementation:
[rans_interface.cpp](https://github.com/InterDigitalInc/CompressAI/blob/v1.2.8/compressai/cpp_exts/rans/rans_interface.cpp).

Local verification can run without pytest:

```bash
python -m unittest discover -s tests -p 'test_entropy_probe*.py' -v
```

Verification status: the local CPU suite covers actual rANS recoding, integer and
feature equality, nonzero contexts/medians, odd token lengths, unchanged weights,
wrapper restoration after failure, split isolation, additive byte accounting,
report generation and offline caches. Slurm syntax and both task plans are checked.
The real server checkpoints and RE10K data were then exercised on an RTX A5000
with PyTorch2.5.1+cu121/CompressAI1.2.8; both GPU/BF16 tasks completed as reported
above. No completed training or6991-scene full evaluation was rerun.

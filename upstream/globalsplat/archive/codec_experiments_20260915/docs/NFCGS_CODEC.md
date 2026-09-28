# NFC-GS codec integration

The GPU-efficient decoder-causal score-prior implementation and its matched
screen/joint SLURM arrays are documented in [NFCGS_SCORE_CONTEXT.md](NFCGS_SCORE_CONTEXT.md).

For the paper24 linear/nonlinear × rank × lambda × residual experiment, see
[the 16-condition transform ablation](NFCGS_TRANSFORM.md). The default new
submission trains the eight nonlinear32 arms and reuses the corrected linear
controls.

This checkout contains an optional observable low-rank 1-D codec at the actual
GlobalSplat scene-token boundary:

```text
DualStreamSlotEncoder
  -> (appearance [B,4096,512], geometry [B,4096,512])
  -> geometry projection [B,4096,224]
  -> decoded-token-center Morton sort (10 bit)
  -> low-rank + residual hyperprior codec [B,4096,736]
  -> TokenCoarseToFine3DGS(appearance=512, geometry=224)
```

Install the additional entropy-model dependency after the normal GlobalSplat
installation:

```bash
pip install -e ".[codec]"
```

The archived rank-56 checkpoint is selected by replacing the model config while
keeping the normal 32K experiment overlay:

```bash
python -m globalsplat.main \
  model=globalsplat_nfcgs_rank56 \
  +experiment=re10k_32k \
  mode=test \
  checkpointing.load=/path/to/step50000.ckpt
```

`GlobalSplat.last_codec_output` exposes the most recent likelihood tensors,
estimated bit count, Morton permutation, and reconstructed scene features. The
decoder consumes the Morton-sorted tokens directly: Gaussian decoding is
token-wise and produces an unordered set, so the archived configuration does not
transmit an inverse permutation.

For the real sender/receiver path (one scene per bitstream):

```python
model.eval()
model.feature_codec.update(force=False)
compressed = model.compress_scene(inputs)
gaussians = model.decompress_scene(compressed.data)
```

For a one-GPU SLURM evaluation on the fixed RE10K protocol, use
`scripts/slurm/eval_nfcgs_rank56.slurm`. Create `logs/slurm` before submitting
because SLURM opens the output files before the job script starts:

```bash
mkdir -p logs/slurm
CHECKPOINT=/path/to/rank56-step50000.ckpt \
DATASET_ROOT=/data3/local_datasets/re10k \
CONTEXT=12 \
sbatch scripts/slurm/eval_nfcgs_rank56.slurm
```

The evaluation script has no scene cap and performs no SGA+/per-scene tuning.
It uses the actual sender/receiver bitstream path once per scene and reports
`actual_bytes`, bits/Gaussian, and BPGA together with PSNR/SSIM/LPIPS.
Its default `PROTOCOL=all` traverses the raw RE10K test split and deterministically
selects 12 context plus 8 disjoint target views for every processable scene.
Scenes with fewer than 20 usable frames, invalid shapes, or invalid camera/FOV
metadata are reported and skipped by the upstream loader. To run the standard
fixed C3G benchmark instead, set `PROTOCOL=fixed`; that evaluates 5,601 indexed
scenes with 3 target views and permits `CONTEXT=12`, `24`, or `36`.

For a fresh codec-only training run, the companion script first converts the
released vanilla GlobalSplat-32K checkpoint to the 224-D observable decoder
boundary with an output-preserving thin QR factorization, then trains the
rank-56 codec:

```bash
DATASET_ROOT=/data3/local_datasets/re10k \
sbatch scripts/slurm/train_nfcgs_rank56.slurm
```

By default the script loads `checkpoints/pretrained/globalsplat-re10k-32k.ckpt`
and the rank-matched artifact under `checkpoints/codec_init`. These paths can be
overridden with `VANILLA_CHECKPOINT`, `CODEC_INIT_RANK56`, and
`CODEC_INIT_RANK80`.

The recovered training recipe uses 12 context and 8 target views, 4096 scene
tokens and stage 3 (32,768 Gaussians), a frozen GlobalSplat backbone, Adam at
1e-4, 50k steps, milestones at 35k/45k, bf16, and a likelihood-based BPGA rate
term with lambda 0.0256. Entropy quantiles are updated deterministically every
500 steps. The script defaults to micro-batch 1 with accumulation 8 for a
48-GiB GPU; set `MICRO_BATCH=8 ACCUMULATE=1` to reproduce the archived physical
batch when memory permits. A one-off PCA/codec initialization override can be
passed with `CODEC_INIT=/path/to/artifact.pt`.

The implementation supports the retained linear, untied-synthesis, rank-56
checkpoint with its full multiscale 1-D residual branch and resized CompressAI
CDF buffers. New bitstreams use the verified `E2EM0301` outer scene container. The
old residual payload marker `M3HPRANS` is inspectable but its original serializer
was not present in the artifacts; new residual payloads therefore use the
documented `M3HPRN01` sub-container.

## Residual-path ablation

`model.feature_codec.use_residual` keeps the mean, Morton ordering, low-rank
analysis/synthesis matrices, score quantization, and score entropy model fixed
while enabling or bypassing the multiscale 1-D residual hyperprior. In the off
arm, reconstruction is `scene_mean + low_rank`, residual parameters are frozen,
and only score likelihoods contribute to the rate objective.

The default ablation grid uses ranks 56 and 80, rate lambdas 0.0064 and 0.0256,
and both residual settings for eight runs. Submit every combination with the
same seed and training recipe as a SLURM array:

```bash
mkdir -p logs/slurm
DATASET_ROOT=/data3/local_datasets/re10k \
sbatch --array=0-7 scripts/slurm/train_nfcgs_rank56.slurm
```

The checked-in path convention supplies separate rank-56 and rank-80 codec
initialization artifacts. Override them through `CODEC_INIT_RANK56` and
`CODEC_INIT_RANK80` when using another checkpoint root. A rank-mismatched
statistics artifact now fails before training instead of being partially loaded.

The initializer treats these files as strict PCA/statistics artifacts rather
than ordinary partial state dictionaries. It loads the archived geometry basis,
uses that same coordinate system to reparameterize the frozen geometry decoder,
copies the PCA basis into both the initially tied analysis and synthesis
matrices, stores `log(score_scale)`, and restores the residual mean/std buffers.
It also verifies the vanilla checkpoint SHA-256 recorded by the artifact. This
avoids the older partial-load behavior where only `shared_basis` happened to
match a model state key.

The task mapping is:

| Tasks | Rank | Lambda | Residual |
|---|---:|---:|---|
| 0 / 1 | 56 | 0.0064 | on / off |
| 2 / 3 | 56 | 0.0256 | on / off |
| 4 / 5 | 80 | 0.0064 | on / off |
| 6 / 7 | 80 | 0.0256 | on / off |

## GlobalSplat paper-view rerun

`re10k_32k_nfcgs_paper` combines the archived frozen-codec training recipe with
GlobalSplat's original view sampler and subset-consistency objective. Each
training example samples 24 context views and 12 targets. The two temporal
anchors are shared and the 22 middle views alternate, so the model receives two
13-view forward passes and both branches render the same 12 targets. The alpha
and depth consistency weights are `1e-3` and `1e-2`.

The codec probability families remain the archived ones: a factorized
`EntropyBottleneck` for low-rank scores, a conditional Gaussian for residual
`y`, and a factorized `EntropyBottleneck` for residual hyperlatent `z`.
With the pinned CompressAI 1.2.8 implementation, training quantization uses
additive uniform noise and evaluation uses dequantized rounding. Quantile/CDF
maintenance is deterministic every 500 optimizer steps. Submit the same
rank/lambda/residual grid into a separate output tree with:

```bash
mkdir -p logs/slurm
DATASET_ROOT=/data3/local_datasets/re10k \
sbatch --array=0-7 scripts/slurm/train_nfcgs_paper_recipe.slurm
```

The 24-view pool is expanded internally to a batch of two 13-view branches, so
the script defaults to `MICRO_BATCH=1 ACCUMULATE=8` on a 48-GiB GPU. Run a short
memory smoke test first if the target node differs:

```bash
MAX_STEPS=100 ACCUMULATE=1 \
sbatch --array=0 scripts/slurm/train_nfcgs_paper_recipe.slurm
```

After training, the existing evaluator can consume the separate output root:

```bash
TRAIN_OUTPUT_ROOT="$PWD/outputs/nfcgs_paper24_subset_train" \
sbatch --array=0-7 scripts/slurm/eval_nfcgs_rank56.slurm
```

## Morton-order ablation

`model.feature_codec.use_morton=false` replaces geometry-guided Morton sorting
with an identity permutation while retaining the same centering, low-rank,
entropy, and residual modules. New no-Morton bitstreams record this choice in
their scene flags; older bitstreams remain Morton-on by default.

The current eight runs already form the Morton-on side. Since the factorized
low-rank score path is permutation-equivariant, the informative additional runs
are the four residual-on arms:

```bash
sbatch --export=ALL,USE_MORTON=false --array=0,2,4,6 \
  scripts/slurm/train_nfcgs_rank56.slurm
```

For a complete factorial grid including the expected residual-off control,
submit `--array=0-7` instead. Morton-off outputs are placed below each existing
arm's `morton_off/` directory, so they do not collide with the running
Morton-on jobs. Evaluation accepts the matching `USE_MORTON=true|false` value.

Outputs are separated under
`rank{56,80}/lambda{0p0064,0p0256}/residual_{on,off}`, with Morton-off runs in
an additional `morton_off/` subdirectory. TensorBoard logs the total estimated
BPGA as well as per-stream `score`, `residual_y`, and `residual_z` bits/BPGA.
For evaluation, pass the matching `RANK`, `RATE_LAMBDA`, `USE_RESIDUAL`, and
`USE_MORTON` values to `scripts/slurm/eval_nfcgs_rank56.slurm`.

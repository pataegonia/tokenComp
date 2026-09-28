# Main NFC-GS token codec

## Supported path

**Nonlinear32, rank56, Morton ON, residual ON, Full context P0, even/odd Split.**
The two baseline checkpoints and full-test measurements are in
[the experiment handoff](SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md).

```text
GlobalSplat scene tokens (appearance 512, geometry 512)
  -> geometry projection 224 -> 736-D features -> Morton ordering
  -> FP16 scene mean + nonlinear low-rank score + residual hyperprior
  -> Full P0+Split score streams + residual y/z -> scene bitstream
  -> entropy decode -> reconstructed appearance 512 / geometry 224
  -> token-wise Gaussian decoder -> rendering
```

`CodecConfig()` and `config/model/globalsplat_nfcgs_rank56.yaml` now select this
path directly. Historical feature flags remain in checkpoint metadata and are
validated against the supported values. Dimensions remain configurable for
small tests; the production model preset uses the baseline dimensions.

## Environment and API

Use the GlobalSplat environment and install `.[codec]` from this repository.
The renderer additionally requires the project's CUDA/gsplat installation.

```python
model.eval()
model.feature_codec.update(force=True)  # preserve checkpoint quantiles
compressed = model.compress_scene(inputs)
gaussians = model.decompress_scene(compressed.data)
```

One bitstream contains one scene. Tokens remain Morton ordered on the receiver;
the Gaussian decoder is token-wise, so no inverse permutation is transmitted.
`actual_bytes` includes mean, score, residual y/z, and containers. Shared model
weights are excluded, as in the retained experiments.

## Evaluation

Run from this repository, with the package installed or `PYTHONPATH` set to it:

```bash
python scripts/run_nfcgs.py eval --rate-lambda 0.0064 --dry-run
python scripts/run_nfcgs.py eval --rate-lambda 0.0256 \
  --checkpoint /path/to/main_split.ckpt --dataset-root /data/re10k
```

Without `--checkpoint`, evaluation selects the corresponding handoff checkpoint
under `outputs/nfcgs_score_probability10k/20260913_112316/e1_split`.
These server checkpoints are not bundled with the local source checkout.
The protocol is all-test, context12/target8, batch1, seed0, actual bitstream
decode, and no image export. `--max-scenes` is an explicit optional screen cap.

```bash
mkdir -p logs/slurm
sbatch scripts/slurm/eval_nfcgs_main.slurm --rate-lambda 0.0064
sbatch scripts/slurm/eval_nfcgs_main.slurm --rate-lambda 0.0256
```

## Training

Codec-only training requires an explicit main checkpoint. Weights-only continuation and
full optimizer resume are distinct; use `--resume` only with matching optimizer,
training scope, and schedule settings.

```bash
python scripts/run_nfcgs.py train --checkpoint /path/to/main_split.ckpt \
  --rate-lambda 0.0256 --dry-run
sbatch scripts/slurm/train_nfcgs_main.slurm \
  --checkpoint /path/to/main_split.ckpt --rate-lambda 0.0256
```

The default codec-only recipe retains the paper24 sampling (two13-view branches,
shared12 targets), frozen GlobalSplat, micro2/accumulate4, Adam1e-4, 50k steps,
35k/45k LR milestones, BF16, and quantile maintenance every500 steps.
`--scope score_probability` selects the existing reconstruction-locked scope:
10k steps, a7k LR milestone, and quantile maintenance OFF.
Use `--max-steps`, `--lr`, `--batch-size`, `--accumulate`, and `--output` explicitly
for a new experiment; short runs retain the selected scope's LR milestones.
No training or evaluation jobs are submitted by the cleanup.

For a fresh main architecture, `scripts/initialize_nfcgs_from_vanilla.py` retains
the verified geometry QR/PCA initialization:

```bash
python scripts/initialize_nfcgs_from_vanilla.py \
  --vanilla checkpoints/pretrained/globalsplat-re10k-32k.ckpt \
  --codec-init checkpoints/codec_init/re10k_ctx12_s64_t512_rank56.pt \
  --output outputs/main_initialization.ckpt
```

Fresh joint training of Split is a new experiment; it is not a reproduction of
the historical multi-stage training lineage. Existing baseline weights are
loaded directly, without the old architecture-to-architecture partial warm starts.

### GlobalSplat + codec from scratch

`--from-scratch` initializes the image/ray tokenizer, scene-token encoder,
main codec (including geometry projection), and Gaussian decoder without a
GlobalSplat/codec checkpoint. All of these modules are trainable. The frozen
pretrained perceptual-loss networks remain part of the loss/evaluation setup.
There is no QR/PCA calibration prerequisite for this jointly learned model.

```bash
python scripts/run_nfcgs.py train --from-scratch \
  --rate-lambda 0.0064 --dataset-root /data/re10k --dry-run
mkdir -p logs/slurm
sbatch scripts/slurm/train_nfcgs_main.slurm --from-scratch \
  --rate-lambda 0.0064 --dataset-root /data/re10k
```

The experimental `re10k_32k_nfcgs_joint` recipe uses:

- The same Nonlinear32/rank56/Morton/residual/Full P0+Split codec from step 0.
- Paper24 sampling, BF16, micro1/accumulate8, one GPU.
- AdamW 5e-4, weight decay 1e-6, 3% LR warmup then cosine decay, 500k steps.
- Gaussian capacity stages at 20k/40k/100k with 4k-step transitions, finishing
  at 4096 tokens x 8 Gaussians = 32768 Gaussians.
- Rate-loss weight ramp over the first 20k steps; quantile maintenance every500.

With the eight-GPU wrapper, the global batch is 8. The 500k-step default sees
4.0M scenes, slightly more than the released GlobalSplat recipe's 3.52M scenes
(220k steps with global batch 16), leaving additional training budget for the
jointly inserted codec.

These are starting settings, not a validated quality/memory/convergence result.
Training uses differentiable quantization approximations and likelihoods;
actual entropy encode/decode is used by the existing evaluation command.
`--max-steps` scales the LR schedule's total duration but does not rescale
capacity boundaries or the rate ramp. A short smoke run therefore remains in
the early curriculum stages. The full-model backward pass needs more memory
than codec-only training; adjust `--batch-size`/`--accumulate` to the server GPU.

To resume this recipe, replace `--from-scratch` with `--joint --checkpoint ...
--resume` and retain the original lambda, optimizer, total steps, and batch settings:

```bash
sbatch scripts/slurm/train_nfcgs_main.slurm --joint \
  --checkpoint /path/to/joint/last.ckpt --resume \
  --rate-lambda 0.0064 --dataset-root /data/re10k
```

`--joint --checkpoint ...` without `--resume` starts a new full-model fine-tune
from those weights. Plain `train --checkpoint ...` continues to freeze
GlobalSplat. Scratch and joint training reject `--scope score_probability`.

### Eight GPUs on ariel-v10

```bash
mkdir -p logs/slurm
sbatch scripts/slurm/train_nfcgs_joint_v10.slurm --from-scratch \
  --rate-lambda 0.0064 --dataset-root /data3/local_datasets/re10k
```

This wrapper requests `ariel-v10`, eight GPUs, eight tasks, eight CPUs/task,
and 40 GiB host RAM/GPU in `batch_ugrad`. It runs the launcher once for preflight,
then `srun` starts eight Lightning DDP ranks with a shared command/output path.
`trainer.devices=8`, `num_nodes=1`, and unused-parameter detection is enabled
for the full model's conditional/unused branches. Each rank sees all allocated
GPUs and Lightning selects its GPU by local rank (`--gpu-bind=none`).

Microbatch1 x eight GPUs x accumulate1 keeps the effective batch at eight
scenes per optimizer step, matching the single-GPU scratch recipe. LR and step
schedules stay unchanged. `--accumulate 8` would instead make the batch64.
For resume, use the same wrapper with `--checkpoint /path/to/last.ckpt --resume`
instead of `--from-scratch`, retaining the original training settings.

For command preview without a SLURM allocation:

```bash
python scripts/run_nfcgs.py train --from-scratch --devices 8 --launcher srun \
  --accumulate 1 --rate-lambda 0.0064 --dry-run
```

The runner rejects allocation/task mismatches before starting workers.
Local launch-contract tests do not replace a real eight-GPU NCCL/rendering test.

### Check checkpoint writes before a long run

After resolving storage errors, exercise the actual Lightning checkpoint writer
(model and Adam state) in a separate short job, not just a raw-file write test:

```bash
sbatch scripts/slurm/train_nfcgs_joint_v12.slurm --from-scratch \
  --rate-lambda 0.0064 --max-steps 20 --checkpoint-every 10 \
  --output outputs/nfcgs_joint_save_probe
```

This job saves at steps10 and20 and then stops. Inspect its logs and verify that
the resulting checkpoint can be loaded before starting the full scratch run.
Do not resume this diagnostic run as a 500k run: its warmup/cosine schedule was
built for 20 steps. The normal joint-training checkpoint interval is 10000; all checkpoints
are retained, so frequent saves should be confined to this short diagnostic.

### Extend an already-running 220k v10 job

Leave the original job unchanged and submit a dependent continuation while it
is still running. For parent job `425420`:

```bash
bash scripts/slurm/submit_nfcgs_joint_extension_v10.sh 425420
```

The continuation enters `squeue` immediately with reason `Dependency` and runs
only after job 425420 exits successfully (`afterok:425420`). It reads the
parent's stdout log, locates that run's `last.ckpt`, preserves the model,
optimizer moments, global step, and loader state, and trains from step 220k to
500k. The completed 220k checkpoint remains unchanged. A patched checkpoint is
written under `SLURM_TMPDIR` when available (otherwise `outputs/.nfcgs_tmp`) and
removed at exit.

The extension uses a separate late-stage LR schedule: 10k steps from 1e-5 to
1e-4, then cosine decay to 1e-5 over the remaining 270k steps. This avoids
continuing the original 220k cosine scheduler beyond its intended endpoint.
It also keeps the full Gaussian stage and the fully ramped rate loss because
the resumed global step remains 220k. The new checkpoints are written below
`outputs/nfcgs_main/joint_extension/` every 10k steps.

The defaults can be overridden when submitting, for example:

```bash
TARGET_MAX_STEPS=550000 EXTENSION_PEAK_LR=5e-5 \
  bash scripts/slurm/submit_nfcgs_joint_extension_v10.sh 425420
```

## Diagnostics and history

- `scripts/probe_nfcgs_entropy.py` and its SLURM wrapper remain for main-codec diagnostics.
- Scene CI analysis scripts and result documents remain available.
- Historical variant implementations and launchers are in
  [the archive](../archive/codec_experiments_20260915/README.md).
- See [cleanup validation](CODEC_CLEANUP_2026-09-15.md).

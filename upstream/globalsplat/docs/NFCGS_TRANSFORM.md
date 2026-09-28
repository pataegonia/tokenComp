# Paper24 transform ablation

The comparison has 16 conditions: linear/nonlinear × rank56/80 × lambda
0.0064/0.0256 × residual ON/OFF. Morton is ON throughout. The existing corrected
linear evaluations supply eight controls; the default new training/evaluation
arrays contain only the eight nonlinear conditions.

| Linear task | Nonlinear task | Rank | Lambda | Residual |
|---:|---:|---:|---:|---|
| 0 | 8 | 56 | 0.0064 | ON |
| 1 | 9 | 56 | 0.0064 | OFF |
| 2 | 10 | 56 | 0.0256 | ON |
| 3 | 11 | 56 | 0.0256 | OFF |
| 4 | 12 | 80 | 0.0064 | ON |
| 5 | 13 | 80 | 0.0064 | OFF |
| 6 | 14 | 80 | 0.0256 | ON |
| 7 | 15 | 80 | 0.0256 | OFF |

## Transform and initialization

The default `model.feature_codec.transform=linear` retains the existing state
keys and computation. `nonlinear` adds shared, pointwise, bias-free MLPs:

```text
A(X) = X W_a^T + Linear(32, rank)(GELU(Linear(736, 32)(X)))
S(U) = U W_s   + Linear(32, 736)(GELU(Linear(rank, 32)(U)))
```

X is the centered, Morton-sorted feature; U is the decoded score after scale
restoration. The residual input is X − S(U). Forward, compress, and decompress
use the same A/S methods. The score width remains rank56/80; nonlinear
reconstruction is not constrained to a rank56/80 linear subspace.

Both MLP output matrices start at zero. Shared modules are initialized before
the MLPs, and MLP construction preserves the CPU RNG state. At the same seed,
linear/nonlinear therefore start with identical common tensors and preserve
the initialization of the following Gaussian decoder. Rank-matched PCA/statistics
artifacts initialize both linear bases, geometry projection, score scale and
residual normalization. New runs start from these initial weights, not from a
previously trained 50k checkpoint.

The transform launcher fixes the recipe to micro-batch1 × accumulation8,
seed111123, workers8, BF16 mixed, Morton ON, and disabled cosmetic GPU load.
The existing paper24 recipe supplies frozen GlobalSplat, 24→13+13 context
branches, 12 shared targets, the same subset/RD losses, Adam1e-4, 35k/45k LR
milestones, and 50k steps. Codec geometry projection remains trainable as in
the controls. Residual OFF freezes/bypasses only the residual codec; the
analysis/synthesis MLPs remain trainable.

## Training

Run from `/ceph_data/clue9986/tokencomp/upstream/globalsplat` after syncing the
implementation and confirming the rank56/rank80 initialization artifacts exist.

```bash
# Inspect all 16 conditions without loading data or starting jobs.
bash scripts/slurm/train_nfcgs_transform.slurm plan

# Slurm opens its logs before the script runs.
mkdir -p logs/slurm

# Default array is 8-15: eight new nonlinear32 runs.
sbatch scripts/slurm/train_nfcgs_transform.slurm
```

An optional short GPU smoke run should use a separate output root so its
version_0 cannot be mistaken for the main experiment:

```bash
MAX_STEPS=100 TRANSFORM_TRAIN_ROOT="$PWD/outputs/nfcgs_transform_smoke_train" \
  sbatch --array=8,9 scripts/slurm/train_nfcgs_transform.slurm
```

To retrain all controls as well, explicitly use `--array=0-15`. Those linear
runs are written under the new transform tree, leaving the original paper24
training/results in place. `TRANSFORM_RESUME_CHECKPOINT` explicitly enables
resume from a compatible run. Otherwise initialization starts a fresh training
run. `CODEC_INIT_RANK56` and `CODEC_INIT_RANK80` select the rank-specific artifacts.

New training root:

```text
outputs/nfcgs_paper24_transform_train/{linear,nonlinear32}/
  rank{56,80}/lambda{0p0064,0p0256}/residual_{on,off}/
  checkpoints/nfcgs_transform_<condition>/version_0/step000050000.ckpt
```

## Evaluation

```bash
# Inspect the exact 16 checkpoint paths; no data/weights are loaded.
bash scripts/slurm/eval_nfcgs_transform.slurm plan

# Once training completes, check all 16 files exist.
bash scripts/slurm/eval_nfcgs_transform.slurm check

# Validate only a completed subset before allocating GPUs. This loads weights
# on CPU and checks the codec configuration and actual saved global_step.
conda activate globalsplat
bash scripts/slurm/eval_nfcgs_transform.slurm validate 8-14 && \
  sbatch --array=8-14 scripts/slurm/eval_nfcgs_transform.slurm

# Evaluate the eight new nonlinear32 checkpoints.
sbatch scripts/slurm/eval_nfcgs_transform.slurm
```

`plan`, `check`, and `validate` accept a task selection such as `8-14` or
`8,10-14`; the default remains `0-15`. `check` only checks file existence.
`validate` uses the active `python` (or `TRANSFORM_PYTHON`) and returns nonzero
if any selected checkpoint is missing, unreadable, or incompatible. Neither
mode submits jobs or silently substitutes another checkpoint. The default
Slurm array remains `8-15`, so use an explicit array for partial completion.
There is no fixed node request; Slurm can choose an available eligible node.

### Downloaded training logs: job 418264, 2026-09-08

- Tasks 8-13 reach epoch6, batch3760/66033, and print `Training outputs:`:
  normal termination at the configured 50k-step endpoint (about 6.06 epochs).
- Task14 reaches the same endpoint, but its shell then reports
  `train_nfcgs_paper_recipe.slurm: line 260: syntax error near unexpected token ')'`.
  This is not proof that its checkpoint is lost or valid. CPU-validate its
  exact version0/step50000 checkpoint before evaluation; do not retrain solely
  because of this shell-tail error. The downloaded logs cannot establish the
  exact cause of the shell error.
- Task15 (rank80, lambda0.0256, residual OFF) is at epoch4, batch47865/66033:
  approximately 39k/50k optimizer steps. At the snapshot's roughly 3.51
  micro-batches/sec, about seven hours remain. This is not live server status.

For this snapshot, validate and submit `8-14` as above. If task14 fails,
validate/submit only `8-13` and inspect task14's checkpoint separately. Once
task15 completes:

```bash
bash scripts/slurm/eval_nfcgs_transform.slurm validate 15 && \
  sbatch --array=15 scripts/slurm/eval_nfcgs_transform.slurm
```

Evaluation is all-test RE10K C12/T8, BF16 mixed, batch1, actual entropy
encode/decode, and no per-scene optimization. Each job checks architecture,
rank, hidden width, ablation flags when recorded, and exact global step before
loading the renderer. CDFs are rebuilt while preserving saved quantiles.
The launcher fixes all-scene evaluation and disables the separate score-context
ablation flags even if inherited from the submission shell. Codec width is not
exported as Lightning's distributed `RANK` environment variable.

Tasks0–7 select the confirmed original paper24 version_0/step50000 linear
checkpoints. Their existing comparison metrics remain in
`outputs/nfcgs_paper24_subset_eval_corrected`; do not substitute older eval
results. Re-running tasks0–7 with this evaluator can additionally collect the
new stream breakdown and scene/frame records.

For freshly retrained linear controls, explicitly set
`TRANSFORM_LINEAR_SOURCE=fresh`. New-run evaluation defaults to version0 and
step50000; `TRANSFORM_VERSION` and `TRANSFORM_EVAL_STEP` explicitly select another
version/step. These do not change the confirmed original linear paths.
There is no fallback to a newer version or incomplete checkpoint.

Results are written under
`outputs/nfcgs_paper24_transform_eval/{linear,nonlinear32}/.../job_<id>/`.
`TRANSFORM_TRAIN_ROOT` and `TRANSFORM_EVAL_ROOT` override only the new trees.

## Measurements and compatibility

`actual_rate_per_scene.json` now includes scene ID, context/target frame IDs,
PSNR/SSIM/LPIPS when scoring is enabled, and actual bytes for score, residual_y,
residual_z, scene mean, and container overhead. The components sum exactly to
`len(bitstream)`; their scene averages also appear in `scores_all_avg.json`.
Model weights remain shared and excluded from per-scene payload as before.

Compare total actual bytes with rendering metrics. A smaller residual stream
alone is not a total-rate gain. Equal lambda does not establish matched rate
or matched quality; use overlapping operating points for such claims. Match
scene/frame IDs when computing paired differences rather than relying only
on equal scene counts.

New checkpoints store `feature_codec_config`, including residual/Morton flags.
Old linear checkpoints still load; their unsaved flags require the correct
external experiment configuration. Nonlinear streams set bit2 of the existing
scene flags; linear payloads keep their original flags and format. A receiver
rejects a linear/nonlinear transform mismatch. Both endpoints must still share
the same codec checkpoint; the scene container does not identify individual
model weights. Older synthesis-only nonlinear archives are not the new
two-sided MLP experiment and are rejected by its loader.

Local CPU checks:

```bash
python -m pytest tests/test_transform_codec.py tests/test_transform_slurm.py \
  tests/test_nfcgs_codec.py tests/test_slurm_eval.py -q
```

The CPU tests cover common initialization and RNG preservation, gradients into
both MLPs, real entropy roundtrips with nonzero MLP weights at rank56/80 and
residual ON/OFF, checkpoint metadata/legacy compatibility, and all 16 launch
mappings. Full rendering/training still requires the server CUDA/data stack.

# Score path on/off ablation

## Settings

The default remains the existing Full P0 + even/odd Split codec. `minimal`
disables all four requested score-path components together. MSH
(`ResidualHyperprior1D`, including its residual normalization, adapters,
hyperprior and y/z streams), geometry projection and Morton ordering remain.

| Component | `--score-path full` | `--score-path minimal` |
| --- | --- | --- |
| Scene centering and FP16 mean transmission | ON | OFF; mean payload is zero bytes |
| Per-channel score scale | ON | OFF; scale is exactly 1 |
| Mean-conditioned offset and quantization step | ON | OFF; offset 0 and step 1 |
| Prediction from previously decoded channel groups | ON | OFF |
| Even/odd spatial predictor and separate entropy models | ON | ON |
| Nonlinear analysis/synthesis MLP additions | ON | OFF; linear matrices remain |
| MSH residual path | ON | ON |

Spatial-only means even symbols are entropy-coded first, then decoded evens
predict odds with the existing spatial convolution. Existing channel slices
remain for the separate entropy models and spatial convolutions, but no slice
reads any previously decoded slice. Factorized entropy distributions still
learn per-channel probabilities; disabling channel context does not disable
entropy coding or quantization.

With all four components off:

```text
X = concat(texture, projected_geometry), sorted by Morton order
U = X @ analysis_basis.T
U_hat = even/odd spatial Split quantization and entropy coding of U
low_rank = U_hat @ synthesis_basis
residual_hat = MSH(X - low_rank)
X_hat = low_rank + residual_hat
```

MSH architecture is retained. Its input residual changes when the score path
changes. Disabling centering also disables mean conditioning: the decoder
has no transmitted scene mean to condition on. Disabling the nonlinear
transform bypasses only analysis/synthesis MLPs, not MSH nonlinear layers.

## Individual switches

`scripts/run_nfcgs.py` accepts each pair below. Explicit switches override
the preset. `--no-centering` also turns mean context off; explicitly enabling
`--mean-context` without centering is rejected.

```text
--centering       / --no-centering
--score-norm      / --no-score-norm
--mean-context    / --no-mean-context
--channel-context / --no-channel-context
--nonlinear      / --no-nonlinear
```

Equivalent model YAML for `minimal`:

```yaml
model:
  feature_codec:
    use_centering: false
    use_score_norm: false
    score_mean_condition: false
    score_channel_context: false
    transform: linear
    use_residual: true
    use_morton: true
    score_spatial_context: true
    score_spatial_entropy: split
```

## Checkpoint behavior

Disabled modules do not execute and their parameters are frozen. Their tensor
keys remain for strict weights-only initialization from a Full checkpoint.
Consequently checkpoint size does not shrink simply by disabling modules.
The active switches are saved in `feature_codec_config`; historical checkpoints
without the new fields default to centering/score normalization ON.

Changing a loaded checkpoint's score switches requires the explicit
`--allow-score-path-conversion` option for weights-only training. This does not
resume optimizer state and does not reset shared or MSH weights. Evaluation and
optimizer-state resume require matching settings and cannot use this option.
Bitstream flags identify the active transform, centering, score normalization,
mean context and channel context so an incompatible decoder fails explicitly.

## Example: codec-only warm start, lambda 0.0256

The dedicated launcher now allocates four GPUs on v11 while keeping effective
batch 8: 4 GPUs x 2 scenes/GPU x accumulation 1, replacing 1 GPU x 2 x 4.
Learning rate 1e-4, the 35k/45k milestones, 50k optimizer steps and checkpoint
interval 5k stay the same. It runs preflight once and starts four DDP ranks.
CPU/RAM allocation is 32 cores and 192 GiB in total.

```bash
mkdir -p logs/slurm
sbatch scripts/slurm/train_nfcgs_score_minimal_v11.slurm
```

The following generic single-GPU command is also available for comparison.

Run from the server's `globalsplat` repository after syncing the changed files.
This example uses the existing main Split 10k checkpoint, freezes GlobalSplat
and trains the active codec parameters. It is not a new from-scratch experiment.

```bash
mkdir -p logs/slurm
SOURCE="outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt"

sbatch --nodelist=ariel-v11 scripts/slurm/train_nfcgs_main.slurm \
  --checkpoint "$SOURCE" --rate-lambda 0.0256 \
  --score-path minimal --allow-score-path-conversion \
  --max-steps 50000 --checkpoint-every 5000 \
  --output "outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/$(date +%Y%m%d_%H%M%S)"
```

Choose training length separately; 50k above uses the existing codec training
recipe. To inspect the generated Python/Hydra command without submitting, use
`python scripts/run_nfcgs.py train` with the same arguments and `--dry-run`.
For evaluation of a trained minimal checkpoint:

```bash
python scripts/run_nfcgs.py eval \
  --checkpoint /path/to/minimal/step000050000.ckpt \
  --rate-lambda 0.0256 --score-path minimal --max-scenes 32
```

Use `--score-path full` for the original switches. Re-enabling them after minimal
training is another explicit conversion, not equivalent to a trained Full model.

## Validation

CPU tests cover unchanged default weights, outputs, bitstreams and gradients
against the archived Full implementation; each switch and the combined preset;
zero-byte mean payload; inactive parameter gradient exclusion; continued MSH
and spatial/entropy gradients; absence of mean/channel context dependencies;
strict checkpoint reload and conversion guards; and separate sender/receiver
agreement under BF16. Full-precision approximate forward matches actual decoding.
MSH approximate forward under BF16 can differ slightly from actual arithmetic
coding, so the BF16 check compares the sender's actual decoded symbols to a
separate receiver. No server GPU training or evaluation was launched for this change.

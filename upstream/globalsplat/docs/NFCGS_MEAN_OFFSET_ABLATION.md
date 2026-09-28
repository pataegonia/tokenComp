# Score mean-offset ablation

This experiment tests whether the scene-mean-predicted score offset `b_mean`
helps the main Full P0+Split codec after matched additional training. It keeps
the scene-mean-predicted quantization step `d`, channel context, spatial
prediction, entropy models, backbone freeze, rate lambda, data order, and
training budget unchanged.

Both arms load the same Split 10k integrated checkpoint as **weights only**.
Immediately after loading, both zero the offset half of the final
`mean_conditioner` layer; its step half remains untouched. Thus both arms have
the same initial forward function. The baseline can relearn `b_mean`; the
ablation always uses zero offset. A distinct score bitstream flag records the
ablation mode, and checkpoints record it in `score_mean_offset_enabled`.

From the repository root, after synchronizing the changed files:

```bash
bash scripts/slurm/submit_nfcgs_mean_offset_ablation_v11.sh
```

The helper submits two `ariel-v11` one-GPU training array tasks, two 32-scene
actual-bitstream evaluations after both training tasks succeed, and a final
report job after both evaluations succeed. Each training arm runs 20,000 new
optimizer steps, saving every 5,000. `MAX_STEPS` and `CHECKPOINT_EVERY` can be
set before submission if needed; the former must be divisible by the latter.

To submit only the evals after a completed training array, use its array job ID:

```bash
bash scripts/slurm/submit_nfcgs_mean_offset_eval_v11.sh <TRAIN_ARRAY_JOB_ID>
```

Outputs are under
`outputs/nfcgs_main/score_mean_offset_ablation/job_<TRAIN_ARRAY_JOB_ID>/`:

- `baseline/` and `no_mean_offset/`: training checkpoints and TensorBoard logs.
- `baseline_eval32/nfcgs_main/` and `no_mean_offset_eval32/nfcgs_main/`:
  reconstruction, stream-size, and score-context diagnostics.
- Each eval also writes CUDA-synchronized `pipeline_timing.json`, separating
  scene encoder, entropy encode, entropy decode, Gaussian decoder, and renderer.
- `comparison.json`: matched rate/distortion differences. Positive deltas mean
  the no-offset arm's metric is larger.

This is a matched **fine-tuning** experiment. It tests whether `b_mean` is useful
when both variants adapt from the existing Split checkpoint. A definitive
architecture comparison would train both branches from an earlier common
initialization where neither has learned a mean offset.

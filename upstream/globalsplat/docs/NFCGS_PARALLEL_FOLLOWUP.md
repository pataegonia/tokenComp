# Parallel follow-up while score context trains

The factorized continuation controls do not depend on the unfinished
score-context winner. Nonlinear32 evaluation is already complete for tasks
8 through 14.

## 1. Completed nonlinear32 evaluation

Actual payload means are reported in KiB per scene.

| Task | Rank | Lambda | Residual | PSNR | SSIM | LPIPS | KiB |
|---:|---:|---:|---|---:|---:|---:|---:|
| 8 | 56 | 0.0064 | ON | 24.5082 | 0.7665 | 0.2343 | 98.18 |
| 9 | 56 | 0.0064 | OFF | 24.2580 | 0.7587 | 0.2440 | 105.79 |
| 10 | 56 | 0.0256 | ON | 23.6943 | 0.7381 | 0.2673 | 46.69 |
| 11 | 56 | 0.0256 | OFF | 23.5011 | 0.7303 | 0.2738 | 55.34 |
| 12 | 80 | 0.0064 | ON | 24.4890 | 0.7659 | 0.2367 | 99.15 |
| 13 | 80 | 0.0064 | OFF | 24.2579 | 0.7584 | 0.2438 | 109.91 |
| 14 | 80 | 0.0256 | ON | 23.5833 | 0.7378 | 0.2699 | 47.31 |

Against the matching linear conditions, nonlinear32 saves 1.68% to 5.53% of
the scene payload and changes PSNR by +0.0061 to +0.0442 dB. Its main observed
benefit with the full residual codec is therefore rate reduction at almost
unchanged quality. Rank 56 still dominates Rank 80 at both residual-ON lambda
points, so nonlinear capacity does not rescue the larger rank.

## 2. Train factorized continuation controls

Every current score-context joint arm starts from its matching trained 50k
factorized checkpoint and receives another 50k optimizer steps. Comparing it
only with the original 50k result confounds context gain with extra training.
The four controls below use the same warm-start procedure, paper24 recipe,
effective batch 8, and extra 50k budget, with all score-context flags disabled.
Rank 56 provides the compute-matched control for the current context experiment.
Rank 80 additionally tests whether its weaker RD result was partly caused by
insufficient convergence rather than rank allocation itself.

| Task | Rank | Lambda | Residual | Morton |
|---:|---:|---:|---|---|
| 0 | 56 | 0.0064 | ON | ON |
| 1 | 56 | 0.0256 | ON | ON |
| 2 | 80 | 0.0064 | ON | ON |
| 3 | 80 | 0.0256 | ON | ON |

Inspect the mapping:

```bash
bash scripts/slurm/train_nfcgs_factorized_continuation.slurm plan
bash scripts/slurm/eval_nfcgs_factorized_continuation.slurm plan
```

Submit all four conditions and task-wise dependent evaluations:

```bash
bash scripts/slurm/submit_nfcgs_factorized_continuation.sh
```

Each task requests one GPU. The submission has no client-side concurrency
throttle; Slurm may start all four immediately or leave tasks queued. The
evaluator uses `aftercorr`, so its matching task starts only after that training
task succeeds.

All GPU jobs are pinned to `ariel-v12` and request `gpu:1`. The continuation
submitter repeats both settings on the `sbatch` command line so cluster job
submission policy sees the node selection before accepting the job.

The submitter checks all four source checkpoints before creating either Slurm
job. Use an override such as `TASKS=0-1` when only the Rank-56 controls should
be requested. Array selection changes the number of independent jobs without
changing the per-model batch or step budget.

The initialization copies weights from the baseline checkpoint into a fresh
run but intentionally starts a new optimizer and learning-rate schedule. This
matches the score-context joint arms. It is a weights-continuation control, not
a full optimizer-state resume.

Outputs are isolated by a timestamp run tag:

```text
outputs/nfcgs_factorized_continuation/<run_tag>/
outputs/nfcgs_factorized_continuation_eval/<run_tag>/
```

## 3. Nonlinear32 and score-context combination

This grid starts from the completed nonlinear32 Rank-56 residual-ON checkpoint
at each lambda. It gives every arm another 50k paper24 joint-training steps.
The factorized arm controls for that extra optimization budget.

| Tasks | Prior | Lambdas |
|---:|---|---|
| 0/1 | factorized control | 0.0064 / 0.0256 |
| 2/3 | mean | 0.0064 / 0.0256 |
| 4/5 | mean + channel | 0.0064 / 0.0256 |
| 6/7 | mean + spatial | 0.0064 / 0.0256 |
| 8/9 | mean + channel + spatial | 0.0064 / 0.0256 |
| 10/11 | Rank-80 factorized control | 0.0064 / 0.0256 |

Before the linear context experiment finishes, the default submits four
factorized continuation controls: Rank 56/80 at both lambdas. These runs are
independent of which Rank-56 context arm wins.

```bash
bash scripts/slurm/submit_nfcgs_nonlinear_score_context.sh
```

After evaluation, submit only the winning task for each lambda. For example,
if mean+channel wins both points:

```bash
TASKS=4-5 bash scripts/slurm/submit_nfcgs_nonlinear_score_context.sh
```

The complete grid remains available for diagnosis, but is not the recommended
default:

```bash
TASKS=0-11 bash scripts/slurm/submit_nfcgs_nonlinear_score_context.sh
```

Context arms use Rank 56; control tasks 10/11 use Rank 80. All jobs use residual
ON, Morton ON, nonlinear hidden width 32, effective batch 8, 50k optimizer
steps, `ariel-v12`, and one GPU per task. Outputs are isolated below
`outputs/nfcgs_nonlinear_score_context*` and paired evaluation tasks use
`aftercorr`.

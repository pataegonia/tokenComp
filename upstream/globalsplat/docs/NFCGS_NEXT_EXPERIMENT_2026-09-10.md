# Next experiment: nonlinear32 with spatial/channel score coding

Design dated 2026-09-10. This document does not submit jobs or change model code.
The copied logs contain nonlinear Rank-56 factorized controls, but no completed
nonlinear context tasks 6-9. Check the server queue before submitting duplicates.

## What this experiment tests

1. Does score context still improve RD after nonlinear32 has already removed
   some redundancy? Its benefit need not add to the nonlinear transform benefit.
2. Does channel context add enough to spatial context to justify its decoding
   cost? Linear Full wins at lambda .0064; Spatial and Full are close at .0256.
3. Does the combination improve on the best completed linear context model,
   rather than only improving on its own nonlinear factorized control?

The current implementation is a predictive score codec with scene-conditioned
offsets and quantization steps. It does not merely change the probability model
of an unchanged symbol sequence. Joint training can also change transforms and
residual allocation. Interpret gains as gains of this complete context design,
not as an isolated entropy-PMF effect or lossless recoding at identical quality.

## Immediate grid: four new independent training jobs

Task IDs below refer to `nfcgs_nonlinear_score_context_grid.sh`.

| Task | Rank | Transform | Lambda | Mean | Channel | Spatial | Role |
|---:|---:|---|---:|---|---|---|---|
| 0 | 56 | nonlinear32 | .0064 | OFF | OFF | OFF | Completed control; reuse |
| 1 | 56 | nonlinear32 | .0256 | OFF | OFF | OFF | Completed control; reuse |
| 6 | 56 | nonlinear32 | .0064 | ON | OFF | ON | New spatial arm |
| 7 | 56 | nonlinear32 | .0256 | ON | OFF | ON | New spatial arm |
| 8 | 56 | nonlinear32 | .0064 | ON | ON | ON | New full arm |
| 9 | 56 | nonlinear32 | .0256 | ON | ON | ON | New full arm |

Keep residual ON, Morton ON, slice width 16, context hidden width 64,
and nonlinear hidden width 32. Do not add nonlinear mean-only/channel-only or
Rank-80 context runs at this stage. The Rank-80 continuation jobs already in
progress can finish independently.

## Training and execution controls

- Each new arm starts from its matching ORIGINAL nonlinear32 50k checkpoint,
  exactly as controls 0/1 did. Do not start from the completed continuation
  checkpoint: that would give the candidate another 50k of prior training.
- Copy model weights and reset optimizer/scheduler, matching controls 0/1.
- Train another 50,000 optimizer steps, LR 1e-4 with the existing schedule,
  seed 111123, and the existing codec-wide training scope.
- Use paper24 with subset consistency, micro-batch 2, accumulation 4,
  effective batch 8, and bf16-mixed. Keep the GlobalSplat freeze policy unchanged.
- Four independent one-GPU jobs; no cross-GPU gradient synchronization.
  Per-model batch and optimizer-step count remain the same as one-GPU training.
- Each model sees 400,000 sampled scene instances during continuation,
  including repetitions; this is not a count of unique scenes.
- Keep eight data workers per training task and disable dummy GPU load.
- Submit training and paired evaluations together with `aftercorr`.
  No array concurrency cap; Slurm schedules available resources on ariel-v12.

From the GlobalSplat root, the existing submitter supports the complete grid:

```bash
TASKS=6-9 bash scripts/slurm/submit_nfcgs_nonlinear_score_context.sh
```

This command creates four training tasks and four dependent eval tasks. The
submitter explicitly passes `--nodelist=ariel-v12 --gres=gpu:1` for both arrays.
The local train/eval file headers currently say ariel-v11; the command-line
node setting overrides them. Use the submitter above for the requested v12
placement, rather than submitting those files directly with bare `sbatch`.

## References to compare against

Values are scene-average actual bytes / 1024, not likelihood rate estimates.

| Transform/prior | Lambda | PSNR | SSIM | LPIPS | KiB |
|---|---:|---:|---:|---:|---:|
| Linear factorized continuation | .0064 | 24.3024 | .7602 | .2374 | 90.80 |
| Linear spatial | .0064 | 24.3222 | .7611 | .2369 | 86.07 |
| Linear full | .0064 | 24.3232 | .7611 | .2368 | 84.91 |
| Nonlinear factorized continuation | .0064 | 24.3022 | .7602 | .2376 | 87.84 |
| Linear factorized continuation | .0256 | 23.4875 | .7307 | .2729 | 42.26 |
| Linear spatial | .0256 | 23.5304 | .7319 | .2716 | 39.17 |
| Linear full | .0256 | 23.5352 | .7323 | .2716 | 39.23 |
| Nonlinear factorized continuation | .0256 | 23.5210 | .7328 | .2693 | 41.26 |

Sources: `slurm-gs-factorized-cont-eval-420718_[0-1].out`,
`slurm-gs-score-ctx-eval-419616_[4-7].out`, and
`slurm-gs-nl-score-ctx-eval-420304_[0-1].out` in the workspace `slurm/` directory.

Compare nonlinear 6/8 to nonlinear control 0, and nonlinear 7/9 to control 1.
Then compare nonlinear 6/7 to linear spatial 4/5 and nonlinear 8/9 to linear
full 6/7. Task numbers from different job families are not interchangeable.

## Evaluation and decision rules

Use the existing actual-bitstream encoder/decoder evaluation, context 12,
all-test protocol, batch 1, and no image/video dumping.

Collect existing output files:

- `scores_all_avg.json`: quality, mean rate, evaluated scene count, timing.
- `actual_rate_per_scene.json`: scene IDs, context/target frame IDs, quality,
  and actual score/residual/mean/container bytes.
- `benchmark.json` and `peak_memory.json`: execution time and memory.

Verify scene IDs and context/target frames agree across paired arms. The raw
index has 7,286 entries, but skipped examples can reduce the effective count.
Report the actual evaluated count; do not infer it from the progress-bar total.

For each lambda, report absolute values and changes versus the nonlinear
factorized control in bytes, PSNR, SSIM, and LPIPS. Break rate changes down into
score, residual_y, residual_z, and fixed overhead.

A candidate that is smaller and no worse on quality is a clear observed win.
If quality and bytes trade off, retain both points until a validation lambda
sweep allows comparison at overlapping rates. Do not label same-lambda savings
as equal-quality savings, or report robust BD-rate from just these two points.
The already measured ~0.005 dB Spatial/Full difference is not a significance
claim: use paired scene statistics and, for a final close architecture choice,
repeat both competing arms with matched additional seeds.

Compare Full against Spatial on receiver time as well as quality/rate. Spatial
uses two sequential score groups (even/odd); Full rank56 uses four channel
slices times two spatial passes, i.e. eight entropy stages. This does not imply
4x end-to-end decoding time. The existing `entropy_decode` timer wraps
`decompress_scene`, including reconstruction; benchmark on the same GPU type
and repeat a fixed validation subset without competing workloads if timings
are close or noisy.

Use held-out validation scenes to select architectures and additional lambdas.
The already inspected test results are useful development evidence; preserve a
final independent evaluation set for a publication-level generalization claim.

## Following structural experiment: base-conditioned residual

After selecting a base codec, the next proposed change is to let the residual
decoder use the decoded base it is correcting. Currently residual entropy sees
only z_hat and residual synthesis sees only y_hat (`compression/residual.py`).
This is an extension of earlier exploratory base-conditioned residual work,
not a claim that base conditioning itself is new.

Use a separate 2x2 design with proposed flags `residual_base_entropy` and
`residual_base_synthesis` (these flags are NOT implemented yet):

| Arm | Base-conditioned entropy | Base-conditioned synthesis |
|---|---|---|
| R0 | OFF | OFF |
| R1 | ON | OFF |
| R2 | OFF | ON |
| R3 | ON | ON |

Start all four arms from the same selected checkpoint at each lambda. Give
every arm, including R0, another 25k steps with a common fresh optimizer and
schedule. This means eight new jobs across two lambdas, including two controls;
the old completed checkpoint cannot serve as the extra-training control.
Use validation to decide whether to extend the entire matched comparison to
50k; resume its optimizer states consistently if extended.

Condition only on decoded normalized scores and transmitted scene mean.
A shared width-64 projection can supply aligned context at the y resolution
and an intermediate g_s resolution without adding a 736-channel context tower.
Use zero-initialized corrections to entropy parameters and identity-initialized
FiLM for synthesis so the new checkpoint initially reproduces its parent.
Keep residual N/M at 192/320 and preserve the two-level hyperprior for this
ablation. Both compress and decompress must reconstruct identical context;
no original features, encoder-only centers, or target views may enter it.

Residual y+z currently accounts for roughly 14-16% of total bytes in the
linear Spatial/Full models. Thus entropy-only improvements have limited direct
room to reduce total rate. Synthesis conditioning is intended to improve
reconstruction and the joint allocation of score/residual bits; that broader
benefit is a hypothesis to test, not an expected numerical gain.

Only the first four-job nonlinear/context experiment is ready to submit today.
Residual conditioning requires implementation, checkpoint compatibility,
encoder/decoder consistency checks, and its own matched controls.

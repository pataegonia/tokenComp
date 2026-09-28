# Decoder-causal score-context experiment

This experiment targets the dominant low-rank score payload while retaining the
GlobalSplat encoder, observable geometry projection, Morton ordering, residual
codec, Gaussian decoder, and actual sender/receiver evaluation path.

## Architecture

The factorized control encodes every normalized score independently.  The new
path can add three decoder-available conditions:

1. the transmitted FP16 scene mean predicts per-score offset and quantization
   step;
2. previously decoded channel slices predict the next slice;
3. decoded even Morton positions predict odd positions with one 3-tap 1-D
   convolution.

Rank 56 is split into 16/16/16/8 channels.  The largest model therefore has
four serial channel rounds and two spatial passes per round: eight entropy
strings rather than 4096 token-autoregressive calls.  All tokens in a pass are
processed in parallel.  The context heads use zero-output initialization, and
the adaptive step is bounded to `[exp(-2), exp(2)]` for stable warm-up.

The scene mean and every context tensor are reconstructed before use, so the
receiver never relies on sender-only positions or unquantized features.  New
payloads set a scene flag and wrap score strings in `SCCTX001`; old factorized
`E2EM0301` scenes keep their original score field and remain decodable.

## Branches

| Task | Prior | Lambda |
|---:|---|---:|
| 0 | mean | 0.0064 |
| 1 | mean | 0.0256 |
| 2 | mean + channel | 0.0064 |
| 3 | mean + channel | 0.0256 |
| 4 | mean + spatial | 0.0064 |
| 5 | mean + spatial | 0.0256 |
| 6 | mean + channel + spatial | 0.0064 |
| 7 | mean + channel + spatial | 0.0256 |

All tasks use rank 56, residual ON, Morton ON, the linear transform, seed
111123, and BF16. Each task warm-starts from its matching completed factorized
50k checkpoint.

## Parallel submission

Inspect mappings without allocating a GPU:

```bash
bash scripts/slurm/train_nfcgs_score_context.slurm plan
bash scripts/slurm/eval_nfcgs_score_context.slurm plan
```

The default submission immediately launches final joint training for all eight
architecture/rate arms. There is no array concurrency throttle: Slurm schedules
as many one-GPU tasks concurrently as the cluster can provide. `aftercorr`
pairs every eval task with its corresponding train task, so evaluation starts
as soon as that training task succeeds rather than waiting for the whole array.

```bash
mkdir -p logs/slurm
bash scripts/slurm/submit_nfcgs_score_context.sh
```

The default array expression is `0-7`. Each task requests one GPU. Defaults are
micro-batch 2, accumulation 4, eight workers, BF16, and no dummy GPU load. If
the target GPU cannot hold micro-batch 2:

```bash
MICRO_BATCH=1 ACCUMULATE=8 bash scripts/slurm/submit_nfcgs_score_context.sh
```

Screening remains available as an optional short diagnostic. It trains only the
new score modules for 5k steps on one 13-view branch and evaluates 300 scenes:

```bash
TRAIN_MODE=screen bash scripts/slurm/submit_nfcgs_score_context.sh
```

Joint mode uses the paper 24-view, 13+13 subset-consistency recipe, trains the
complete codec for 50k steps, and evaluates the complete processable test split.
To submit only the full-context pair instead of all arms:

```bash
TRAIN_MODE=joint TASKS=6-7 bash scripts/slurm/submit_nfcgs_score_context.sh
```

Joint evaluation defaults to the complete processable test split.  Override
`MAX_SCENES` when another screening size is desired.  Output roots are:

```text
outputs/nfcgs_score_context/<run_tag>/{screen,joint}/{m1c0s0,m1c1s0,m1c0s1,m1c1s1}/
outputs/nfcgs_score_context_eval/<run_tag>/{screen,joint}/...
```

The submitter generates a timestamp run tag so repeated submissions cannot make
the evaluator silently select an older `version_0`.  Set
`SCORE_CONTEXT_RUN_TAG` explicitly when a stable name is preferred.

`BASE_TRAIN_ROOT`, `WARM_START_CHECKPOINT`, `SCORE_CONTEXT_ROOT`, dataset paths,
and conda environment remain submission-time overrides.  A missing baseline or
rank/config mismatch fails before the training job starts.

## Selection rule

Use total actual bytes, PSNR, SSIM, and LPIPS from identical scene/frame IDs.
The score substream should fall, but that alone is insufficient if residual
bytes grow or image quality falls.  Equal lambda is not a matched-rate claim;
select a model from overlapping rate points or add an intermediate lambda.

Also report encoder, entropy-decode, and Gaussian-decoder time.  The contextual
path intentionally limits serial rounds, but its latency still needs to be
measured rather than inferred from parameter count.

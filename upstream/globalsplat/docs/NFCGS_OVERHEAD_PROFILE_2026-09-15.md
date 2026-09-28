# NFCGS context / probability overhead profile

Date: 2026-09-15

## Purpose

This is a read-only profiling experiment. It does not train a model or rerun
rendering/PSNR evaluation. It measures whether the rate gain of each score model
is worth its deployment cost.

Two comparisons are kept separate:

1. **Context overhead:** same-lambda Linear rank56 no-context Factorized
   continuation versus Mean, Mean+Channel, Mean+Spatial, and Full.
2. **Probability-ablation overhead:** same-lambda Nonlinear32 Full+Shared versus
   Split, Static Gaussian, and Conditional Scale.

Both lambdas use their own matched baseline. A `.0064` model is never used as
the timing or model-size baseline for a `.0256` model.

## Controlled setting

- One sequential Slurm job on one physical `ariel-v12` GPU.
- `re10k_eval_all_ctx12`, batch 1, augmentation OFF, seed 0.
- 4 unique TEST scenes warm up every measured operation.
- The following 32 unique TEST scenes are measured in identical order for all
  18 checkpoints.
- BF16 matches the existing evaluation path; entropy-side context remains FP32
  exactly as in the codec implementation.
- CUDA synchronization brackets every timed call.
- An existing completed profile is skipped. A nonempty incomplete result
  directory is preserved and causes an explicit error.

## What is measured

For every checkpoint the profiler records:

- score-path parameter count, registered-buffer size, exact tensor bytes, and
  serialized state-dict bytes;
- separate context-predictor and probability-model parameter counts;
- raw entropy-string bytes, `SCCTX001` wrapper bytes, complete score payload,
  residual bytes, and total scene bytes;
- isolated score sender and receiver latency;
- complete feature-codec sender and receiver latency, starting from the scene
  tokens and excluding the image backbone and renderer;
- incremental peak CUDA allocation for the four timed operations;
- exact equality of sender-side and receiver-side decoded score tensors.

The score sender timing intentionally includes the local entropy decode needed
to reconstruct causal anchors. The no-context Factorized sender performs the
same compress-then-decompress sequence, so the boundary is matched.

The report also computes per candidate:

- delta from the correct same-lambda baseline;
- added deployed score-path tensor bytes;
- model-payload break-even scene count = added model bytes / saved score bytes
  per scene, when the candidate actually saves bytes.

The training `.ckpt` file size is retained as metadata but is not treated as
deployment overhead because it may include optimizer/trainer state.

## Submit

From `/ceph_data/clue9986/tokencomp/upstream/globalsplat`:

```bash
bash scripts/slurm/submit_nfcgs_overhead.sh
```

This submits no training and requires no mandatory dry-run. To run only one
suite:

```bash
PROFILE_SUITE=context bash scripts/slurm/submit_nfcgs_overhead.sh
PROFILE_SUITE=probability bash scripts/slurm/submit_nfcgs_overhead.sh
```

The default `all` suite profiles 10 context checkpoints and 8 probability
checkpoints. The runner checks all expected checkpoint paths before using each
one.

## Outputs and download

Server output:

```text
/ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_overhead_profile/<RUN_TAG>/
  context/lambda0p0064/<label>/summary.json
  context/lambda0p0256/<label>/summary.json
  probability/lambda0p0064/<label>/summary.json
  probability/lambda0p0256/<label>/summary.json
  aggregate.json
  REPORT.md
```

Slurm log:

```text
/ceph_data/clue9986/tokencomp/upstream/globalsplat/logs/slurm/slurm-gs-overhead-<JOB_ID>.out
/ceph_data/clue9986/tokencomp/upstream/globalsplat/logs/slurm/slurm-gs-overhead-<JOB_ID>.err
```

Inside an SFTP session, replace the two placeholders:

```text
get -r /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_overhead_profile/<RUN_TAG>
get /ceph_data/clue9986/tokencomp/upstream/globalsplat/logs/slurm/slurm-gs-overhead-<JOB_ID>.out
get /ceph_data/clue9986/tokencomp/upstream/globalsplat/logs/slurm/slurm-gs-overhead-<JOB_ID>.err
```

## Interpretation boundary

`Wrapper B` is true per-scene format overhead. The FP16 scene mean is not
charged to context because the no-context codec already sends it. Model tensor
bytes are shared deployment overhead and therefore should be amortized across
scenes. CUDA allocation does not include native CPU/rANS memory. Absolute time
is machine-specific; the primary result is the paired delta from Factorized or
Shared on the same GPU and scene sequence.

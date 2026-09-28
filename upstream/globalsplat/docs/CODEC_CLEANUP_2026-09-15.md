# Main codec cleanup — 2026-09-15

## Active architecture

Nonlinear32 / rank56 / Morton ON / residual ON / Full P0 / even-odd Split.
Both lambda0.0064 and lambda0.0256 baseline checkpoints use this path.

The active implementation is `globalsplat/compression`. `CodecConfig()` and the
model YAML select the main architecture directly. Retired feature flags fail
explicitly at configuration/checkpoint/bitstream boundaries instead of silently
selecting a different model.

## Changes

- Removed active linear-only, factorized-only, partial-context, no-Morton,
  no-residual, P1/P2, Shared, and Gaussian score-probability branches.
- Preserved the residual hyperprior's GaussianConditional: this is part of the
  main residual codec, independent of the retired score Gaussian experiments.
- Replaced variant training/evaluation grids with `scripts/run_nfcgs.py` and
  `train_nfcgs_main.slurm` / `eval_nfcgs_main.slurm`.
- Consolidated codec training into `config/experiment/re10k_32k_nfcgs.yaml`.
- Retained geometry QR/PCA initialization, the main entropy probe, scene CI
  analysis, and the original GlobalSplat backbone/decoder.
- Updated relevant tests and documentation; default pytest collection excludes
  archives and selects the integrated implementation.
- Moved the duplicate root `src/nfcgs_codec`, its tests/tools, and package
  metadata into the workspace's `archive/reconstructed_codec`.

## Preservation and compatibility

Before editing, 74 integrated source/config/script files were snapshotted to
`archive/codec_experiments_20260915`, including existing uncommitted modifications.
The archive manifest records their original SHA-256 hashes. Historical variant
tests and launchers remain there as reference, with their original paths/content.
They are not supported active entry points.

All checkpoint files, experiment artifacts, logs, and original experiment-result
documents remain in place. The current usage document `NFCGS_CODEC.md` was
rewritten; its previous contents are also in the snapshot.

Main model parameter names, tensor layout, initialization RNG behavior, and
trainable parameter order are preserved. The inactive `score_entropy` tensors
remain frozen solely for strict loading of the existing integrated checkpoints.
They are not used by the main forward/entropy coding path. Saved configuration
fields such as `score_spatial_hidden` remain readable for metadata compatibility.

The scene and score-container formats, Morton order, FP16 mean, FP32 score
context boundary, and residual normalization/arithmetic are preserved.

## Validation

All 67 active tests passed, including 91 pytest subtests. The full run passed
59 tests; eight versioning tests initially could not create their pytest temp
directory because of local Windows permissions. All nine versioning tests then
passed with an explicit, new workspace temp directory. No test failures remain.
The focused codec/config/probe run separately passed all 32 tests.

Tests load the unmodified archived codec as an independent reference and compare identical weights:

- exact state keys, initial values, and trainable parameter order;
- byte-for-byte identical entropy bitstreams for even and odd token counts;
- exact decoded feature tensors and likelihoods;
- exact training gradients with matched quantization noise;
- CPU BF16 sender/receiver equivalence;
- reference-format checkpoint strict loading and Adam-state loading;
- probability-only steps preserving reconstruction and quantiles;
- codec-to-Gaussian-decoder boundary, geometry QR/PCA initialization;
- stale CDF refresh, paper24 subset sampling, and fixed-symbol entropy probes;
- Hydra composition of both rate points and training scopes, dry-run behavior;
- SHA-256 verification of all 74 pre-cleanup snapshots.

Python syntax and main/probe SLURM shell syntax were checked; `git diff --check`
reported no whitespace errors.

Validation used local PyTorch2.11.0+cpu and CompressAI1.2.8. Temporary test-only
dependencies were installed under the workspace and removed after validation.

This is a source refactor with local CPU verification. The two trained main
checkpoints reside on the server and were not available locally, so no trained
checkpoint GPU/full-test rendering was run and no new training jobs were launched.
Fresh joint training of Split remains a new experiment, not a reproduction of
the historical multi-stage lineage.

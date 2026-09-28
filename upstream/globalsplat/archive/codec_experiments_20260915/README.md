# Codec experiments before main-path cleanup

This directory preserves the source/configuration/script contents present before
the 2026-09-15 cleanup, including uncommitted changes. `manifest.json` records
SHA-256 hashes of all 74 snapshots. Paths are relative to the GlobalSplat repo.

The active codec now supports Nonlinear + Morton/residual + Full P0 + Split.
Linear/factorized, partial context, Shared/Gaussian probability variants,
P1/P2 predictors, and their experiment grids are retired from active paths.

These files are historical reference, not current runnable launchers: their
original repo-relative imports/paths and dependencies have not been rewritten.
The compression package is also loaded explicitly by `tests/test_main_codec.py`
as an independent pre-cleanup oracle for exact bitstream/gradient comparisons.

Original checkpoints, logs, measurements, and experiment documents were not
deleted. See `../../docs/NFCGS_CODEC.md` for the supported entry points.

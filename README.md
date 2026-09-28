# GlobalSplat token Compressor

The active implementation is in [`upstream/globalsplat`](upstream/globalsplat).
The main codec is **Nonlinear32 / rank56 / Morton + residual / Full P0 / Split**.
Both rate points (`0.0064`, `0.0256`) use this architecture.

- [전체 실험 구조 안내 (처음 보는 사람용)](EXPERIMENTS_OVERVIEW.md)
- [Context model / Nonlinear transform / Split 심층 설명](CODEC_DEEP_DIVE.md)
- [NFC-GS 코덱 도해 다섯 장](https://claude.ai/artifact/VFLK2TcKisX6Uxc28YKt9T)
- [Main codec API, training, and evaluation](upstream/globalsplat/docs/NFCGS_CODEC.md)
- [Experiment results and baseline checkpoints](upstream/globalsplat/docs/SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md)
- [Cleanup scope and compatibility](upstream/globalsplat/docs/CODEC_CLEANUP_2026-09-15.md)

Install/run from `upstream/globalsplat` using its `pyproject.toml` and environment.
From this workspace root, `python -m pytest` selects the active integrated tests.

## Historical material

The initial standalone reconstruction package was moved to
[`archive/reconstructed_codec`](archive/reconstructed_codec).
Retired integrated codec variants and experiment launchers are preserved under
[`upstream/globalsplat/archive/codec_experiments_20260915`](upstream/globalsplat/archive/codec_experiments_20260915).
Neither archive is on the active import or default test path.
Checkpoints, experiment artifacts, logs, and original result documents remain in place.

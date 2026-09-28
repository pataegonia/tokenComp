# NFC-GS Hybrid Codec reconstruction

This source tree reconstructs the observable low-rank 1D scene codec from the
checkpoints, resolved configurations, bitstreams, and evaluation records under
`experiments/`. Each archived run keeps its `experiment.yaml` beside its
`rate_points/` directory. The original Python source was not present in the
backup.

The implementation covers the codec boundary shown in the architecture
diagram:

1. project a 512-channel geometry token to 224 decoder-observable channels;
2. concatenate it with the 512-channel appearance token;
3. order the 4096 scene tokens with a 10-bit 3-D Morton code;
4. transmit the per-scene 736-channel FP16 mean;
5. code shared low-rank scores with a factorized entropy bottleneck;
6. code the remaining signal with a multiscale 1-D mean/scale hyperprior; and
7. reconstruct and split the texture and observable-geometry features.

The reconstructed module names and parameter shapes match the retained
`rank40 tied` and `rank56 untied nonlinear64` checkpoints.  The package can
strict-load the `model.feature_codec.*` part of those checkpoints.

> **Windows note:** Some deeply nested evaluation artifacts exceed the legacy
> 260-character path limit. Use long-path-aware tools (or the `\\?\` absolute
> path prefix) when accessing those files directly.

## Quick verification

From this directory:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python tools/inspect_artifact.py `
  experiments/shared_lowrank_rank56_untied_nonlinear64_synthesis_multiscale1d/rate_points/lambda_0p0256/checkpoints/stages/step50000.ckpt
```

## Python API

```python
from nfcgs_codec import load_feature_codec_checkpoint

loaded = load_feature_codec_checkpoint("path/to/step50000.ckpt")
codec = loaded.codec.eval()

# texture, geometry: [B, N, 512]; positions: [B, N, 3]
output = codec(texture, geometry, positions)
reconstructed_texture = output.texture
reconstructed_observable_geometry = output.geometry_observable
```

Call `codec.update()` before entropy coding.  `compress()` returns the same
80-byte `E2EM0301` outer container layout found in the archived bitstreams.

## GlobalSplat integration

The official `R-Itk/globalsplat` repository is cloned under
`upstream/globalsplat` and the codec is integrated at its real
`DualStreamSlotEncoder._pack_outputs()` / `TokenCoarseToFine3DGS` boundary. See
[`upstream/globalsplat/docs/NFCGS_CODEC.md`](upstream/globalsplat/docs/NFCGS_CODEC.md)
for the model config, checkpoint-loading command, and binary sender/receiver API.

The retained rank-56 checkpoint strict-loads all 559 integrated model tensors;
the untouched official checkpoint also strict-loads all 454 vanilla model
tensors, so `feature_codec: null` preserves the original GlobalSplat path.

## Compatibility boundary

The checkpoint-compatible neural module layout and outer scene container are
verified against the artifacts.  The backup does not contain the upstream
GlobalSplat encoder/decoder source, dataset implementation, training runner,
or the original residual sub-container serializer.  This package therefore
uses a documented `M3HPRN01` residual sub-container for newly produced files;
archived `M3HPRANS` residual payloads are preserved and inspected as opaque
data by `SceneBitstream`, but are not claimed to be bit-exactly decodable.

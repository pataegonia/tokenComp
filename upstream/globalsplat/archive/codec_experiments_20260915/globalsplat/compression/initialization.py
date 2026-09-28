"""Initialize the observable decoder boundary from a vanilla GlobalSplat model.

The vanilla decoder consumes a 512-D geometry token in three linear readouts.
Stacking those readouts gives a 224 x 512 matrix.  A thin QR factorization of
its transpose produces an orthonormal 224-D observable projection and exactly
reparameterizes all three readouts (up to floating-point roundoff):

    W = W_observable @ P,  P @ P.T = I.

This lets codec-only training start from the released GlobalSplat checkpoint
without changing its decoded Gaussians before compression is introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


_READOUTS = (
    "gaussian_decoder.geo_pos_readout.weight",
    "gaussian_decoder.geo_param_readout.weight",
    "gaussian_decoder.gate_readout.weight",
)


@dataclass(frozen=True)
class ObservableInitializationReport:
    copied_tensors: int
    observable_channels: int
    source_channels: int
    max_reparameterization_error: float
    max_orthogonality_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _checkpoint_state(checkpoint: str | Path | Mapping[str, Any]) -> dict[str, Tensor]:
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint must contain a tensor state_dict")

    result: dict[str, Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        # Lightning saves GlobalSplatModule.model as `model.*`; DDP may add
        # `module.` before it. Strip only these well-known wrapper prefixes.
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        result[key] = value.detach().cpu()
    return result


@torch.no_grad()
def initialize_observable_from_vanilla(
    model: nn.Module,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    observable_projection: Tensor | None = None,
) -> ObservableInitializationReport:
    """Load a vanilla checkpoint and exactly factor its geometry readouts.

    Shape-compatible vanilla tensors are copied, codec tensors retain their
    constructor initialization, and the geometry projection/readouts are
    replaced by the QR reparameterization.  When ``observable_projection`` is
    provided, it is used instead of a freshly computed QR basis and the decoder
    readouts are expressed in that basis.  This is required for reproducing a
    PCA/statistics artifact whose geometry basis was computed ahead of time.
    """

    codec = getattr(model, "feature_codec", None)
    decoder = getattr(model, "gaussian_decoder", None)
    if codec is None or decoder is None:
        raise ValueError("model must contain feature_codec and gaussian_decoder")

    source = _checkpoint_state(checkpoint)
    missing = [name for name in _READOUTS if name not in source]
    if missing:
        raise KeyError(f"vanilla checkpoint is missing decoder tensors: {missing}")

    old_weights = [source[name].float() for name in _READOUTS]
    source_channels = old_weights[0].shape[1]
    if any(w.ndim != 2 or w.shape[1] != source_channels for w in old_weights):
        raise ValueError("vanilla geometry readouts do not share one input width")
    stacked = torch.cat(old_weights, dim=0)

    projection = codec.geometry_projection.weight
    observable_channels = projection.shape[0]
    if stacked.shape != (observable_channels, source_channels):
        raise ValueError(
            "observable width must equal the stacked decoder output width: "
            f"stacked={tuple(stacked.shape)} projection={tuple(projection.shape)}"
        )

    if observable_projection is None:
        # Do the factorization in float64, then cast to the model dtype. This
        # keeps the initial functional error around fp32 roundoff even for a
        # 224-row map.
        q, r = torch.linalg.qr(stacked.double().T, mode="reduced")
        observable_projection = q.T.to(dtype=projection.dtype)
        observable_readout = r.T.to(dtype=projection.dtype)
    else:
        if not torch.is_tensor(observable_projection):
            raise TypeError("observable_projection must be a tensor")
        if tuple(observable_projection.shape) != tuple(projection.shape):
            raise ValueError(
                "artifact geometry basis has the wrong shape: "
                f"artifact={tuple(observable_projection.shape)} "
                f"expected={tuple(projection.shape)}"
            )
        if not torch.isfinite(observable_projection).all():
            raise ValueError("artifact geometry basis contains non-finite values")
        observable_projection = observable_projection.detach().to(
            device=projection.device,
            dtype=projection.dtype,
        )
        # The artifact stores an orthonormal row basis P.  The decoder that
        # consumes projected geometry must therefore use W @ P.T so that
        # (W @ P.T) @ P == W on the observable row space.  Keep this fp32
        # multiplication: it matches the archived NFC-GS initializer.
        observable_readout = stacked.to(
            device=observable_projection.device,
            dtype=observable_projection.dtype,
        ) @ observable_projection.T

    current = model.state_dict()
    merged = {key: value for key, value in current.items()}
    copied = 0
    for key, value in source.items():
        if key.startswith("feature_codec.") or key in _READOUTS:
            continue
        if key in merged and tuple(merged[key].shape) == tuple(value.shape):
            merged[key] = value.to(dtype=merged[key].dtype)
            copied += 1

    merged["feature_codec.geometry_projection.weight"] = observable_projection
    offset = 0
    for name, old in zip(_READOUTS, old_weights):
        rows = old.shape[0]
        merged[name] = observable_readout[offset : offset + rows].to(dtype=merged[name].dtype)
        offset += rows
    model.load_state_dict(merged, strict=True)

    reconstructed = observable_readout @ observable_projection
    stacked_for_error = stacked.to(
        device=reconstructed.device,
        dtype=reconstructed.dtype,
    )
    reparam_error = (reconstructed - stacked_for_error).abs().max().item()
    identity = torch.eye(
        observable_channels,
        device=observable_projection.device,
        dtype=observable_projection.dtype,
    )
    orth_error = (
        observable_projection @ observable_projection.T - identity
    ).abs().max().item()
    return ObservableInitializationReport(
        copied_tensors=copied,
        observable_channels=observable_channels,
        source_channels=source_channels,
        max_reparameterization_error=float(reparam_error),
        max_orthogonality_error=float(orth_error),
    )


@torch.no_grad()
def load_codec_initialization(
    codec: nn.Module,
    checkpoint: str | Path | Mapping[str, Any],
) -> tuple[int, list[str]]:
    """Load codec tensors from a full state dict or PCA/statistics artifact.

    The archived ``m05.shared_basis_initialization.v1`` artifact uses external
    names and stores physical scales rather than the parameterization used by
    :class:`ObservableLowRank1DCodec`.  Map all of them explicitly:

    - ``geometry_basis`` -> ``geometry_projection.weight``
    - ``shared_basis`` -> both initial analysis and synthesis matrices
    - ``score_scale`` -> ``score_log_scale``
    - flat residual mean/std -> registered NCHW buffers

    Loading only the identically named ``shared_basis`` silently produces a
    different initialization, so recognized statistics artifacts are strict.
    """

    source = _checkpoint_state(checkpoint)
    current = codec.state_dict()
    loaded: list[str] = []

    is_statistics_artifact = "geometry_basis" in source or "score_scale" in source
    if is_statistics_artifact:
        required = {
            "geometry_basis",
            "shared_basis",
            "score_scale",
            "residual_mean",
            "residual_std",
        }
        missing = sorted(required.difference(source))
        if missing:
            raise KeyError(f"codec statistics artifact is missing tensors: {missing}")

        score_scale = source["score_scale"]
        residual_std = source["residual_std"]
        if not torch.isfinite(score_scale).all() or bool((score_scale <= 0).any()):
            raise ValueError("score_scale must be finite and strictly positive")
        if not torch.isfinite(residual_std).all() or bool((residual_std <= 0).any()):
            raise ValueError("residual_std must be finite and strictly positive")

        mapped = {
            "geometry_projection.weight": source["geometry_basis"],
            "shared_basis": source["shared_basis"],
            "shared_synthesis_basis": source["shared_basis"],
            "score_log_scale": score_scale.log(),
            "residual_mean": source["residual_mean"].reshape(1, -1, 1, 1),
            "residual_std": residual_std.reshape(1, -1, 1, 1),
        }
        for key, value in mapped.items():
            if key not in current:
                raise KeyError(f"codec has no destination tensor for artifact key: {key}")
            if tuple(current[key].shape) != tuple(value.shape):
                raise ValueError(
                    f"codec artifact shape mismatch for {key}: "
                    f"artifact={tuple(value.shape)} expected={tuple(current[key].shape)}"
                )
            current[key] = value.to(device=current[key].device, dtype=current[key].dtype)
            loaded.append(key)
        codec.load_state_dict(current, strict=True)
        return len(loaded), sorted(loaded)

    for raw_key, value in source.items():
        key = raw_key
        if key.startswith("feature_codec."):
            key = key[len("feature_codec.") :]
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            current[key] = value.to(dtype=current[key].dtype)
            loaded.append(key)
    codec.load_state_dict(current, strict=True)
    return len(loaded), sorted(loaded)

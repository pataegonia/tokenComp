"""Infer a codec architecture and strict-load it from a Lightning checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .codec import ObservableLowRank1DCodec
from .config import CodecConfig


FEATURE_PREFIXES = ("model.feature_codec.", "feature_codec.")


@dataclass(slots=True)
class LoadedCodec:
    codec: ObservableLowRank1DCodec
    config: CodecConfig
    checkpoint_metadata: dict[str, Any]
    source_prefix: str


def _feature_state_dict(state_dict: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], str]:
    for prefix in FEATURE_PREFIXES:
        selected = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected, prefix
    if "shared_basis" in state_dict:
        return dict(state_dict), ""
    raise ValueError("checkpoint has no model.feature_codec state")


def infer_config(state: Mapping[str, Tensor]) -> CodecConfig:
    basis = state["shared_basis"]
    geometry = state["geometry_projection.weight"]
    rank, observable_channels = basis.shape
    geometry_observable, geometry_channels = geometry.shape
    texture_channels = observable_channels - geometry_observable
    residual_n = state["residual_codec.g_a.0.weight"].shape[0]
    residual_m = state["residual_codec.g_a.2.weight"].shape[0]
    adapter_hidden = state[
        "residual_codec.analysis_adapter.in_projection.weight"
    ].shape[0]
    untied = "shared_synthesis_basis" in state
    mlp_key = "synthesis_mlp.0.weight"
    mlp_hidden = state[mlp_key].shape[0] if mlp_key in state else None
    return CodecConfig(
        texture_channels=texture_channels,
        geometry_channels=geometry_channels,
        geometry_observable_channels=geometry_observable,
        rank=rank,
        residual_n=residual_n,
        residual_m=residual_m,
        adapter_hidden=adapter_hidden,
        basis_parameterization="untied_synthesis" if untied else "tied",
        synthesis_mlp_hidden=mlp_hidden,
    )


def _resize_registered_buffers(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Resize empty CompressAI CDF buffers before strict state loading."""

    modules = dict(module.named_modules())
    for key, value in state.items():
        if "." in key:
            path, name = key.rsplit(".", 1)
            parent = modules.get(path)
        else:
            parent, name = module, key
        if parent is None or name not in parent._buffers:
            continue
        existing = parent._buffers[name]
        if existing is None or existing.shape != value.shape:
            parent._buffers[name] = torch.empty_like(value)


def load_feature_codec_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedCodec:
    """Strict-load only the reconstructed feature-codec branch."""

    checkpoint = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    raw_state = checkpoint.get("state_dict", checkpoint)
    feature_state, prefix = _feature_state_dict(raw_state)
    config = infer_config(feature_state)
    codec = ObservableLowRank1DCodec(config)
    _resize_registered_buffers(codec, feature_state)
    incompatible = codec.load_state_dict(feature_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict feature-codec load failed: {incompatible}")
    metadata = {
        key: checkpoint.get(key)
        for key in (
            "epoch",
            "global_step",
            "pytorch-lightning_version",
            "hyper_parameters",
        )
        if key in checkpoint
    }
    return LoadedCodec(codec, config, metadata, prefix)


"""Infer a codec architecture and strict-load it from a Lightning checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .codec import ObservableLowRank1DCodec
from .config import CodecConfig
from .hyper1d_config import Hyper1DConfig
from .factory import build_feature_codec


FEATURE_PREFIXES = ("model.feature_codec.", "feature_codec.")


@dataclass(slots=True)
class LoadedCodec:
    codec: nn.Module
    config: CodecConfig | Hyper1DConfig
    checkpoint_metadata: dict[str, Any]
    source_prefix: str


def _feature_state_dict(
    state_dict: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], str]:
    for prefix in FEATURE_PREFIXES:
        selected = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected, prefix
    if "shared_basis" in state_dict or "f_mean" in state_dict:
        return dict(state_dict), ""
    raise ValueError("checkpoint has no model.feature_codec state")


def infer_config(
    state: Mapping[str, Tensor], metadata: Mapping[str, Any] | None = None
) -> CodecConfig:
    """Recover Full P0+Split tensor dimensions and validate saved metadata."""
    if "shared_synthesis_basis" not in state:
        raise ValueError("checkpoint is not the supported untied-synthesis codec")
    required = (
        "analysis_mlp.0.weight",
        "synthesis_mlp.0.weight",
        "score_context.mean_conditioner.1.weight",
        "score_context.group_entropies.0.entropy_bottleneck.quantiles",
        "score_context.spatial_odd_entropies.0.entropy_bottleneck.quantiles",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"checkpoint Full P0+Split state mismatch: missing={missing}")
    basis = state["shared_basis"]
    geometry = state["geometry_projection.weight"]
    rank, observable_channels = basis.shape
    mlp_keys = {
        key for key in state if key.startswith(("analysis_mlp.", "synthesis_mlp."))
    }
    transform = "nonlinear" if mlp_keys else "linear"
    hidden = 32
    if mlp_keys:
        expected_keys = {
            "analysis_mlp.0.weight",
            "analysis_mlp.2.weight",
            "synthesis_mlp.0.weight",
            "synthesis_mlp.2.weight",
        }
        if mlp_keys != expected_keys:
            raise ValueError(
                "nonlinear codec requires both bias-free analysis/synthesis MLPs"
            )
        hidden = state["analysis_mlp.0.weight"].shape[0]
        shapes = {
            "analysis_mlp.0.weight": (hidden, observable_channels),
            "analysis_mlp.2.weight": (rank, hidden),
            "synthesis_mlp.0.weight": (hidden, rank),
            "synthesis_mlp.2.weight": (observable_channels, hidden),
        }
        for key, shape in shapes.items():
            if tuple(state[key].shape) != shape:
                raise ValueError(f"nonlinear codec shape mismatch for {key}")
    geometry_observable, geometry_channels = geometry.shape
    texture_channels = observable_channels - geometry_observable
    residual_key = "residual_codec.g_a.0.weight"
    if residual_key not in state:
        raise ValueError("checkpoint has no multiscale 1-D residual codec")
    residual_n = state[residual_key].shape[0]
    residual_m = state["residual_codec.g_a.2.weight"].shape[0]
    adapter_hidden = state[
        "residual_codec.analysis_adapter.in_projection.weight"
    ].shape[0]
    inferred = CodecConfig(
        texture_channels=texture_channels,
        geometry_channels=geometry_channels,
        geometry_observable_channels=geometry_observable,
        rank=rank,
        residual_n=residual_n,
        residual_m=residual_m,
        adapter_hidden=adapter_hidden,
        transform=transform,
        transform_hidden=hidden,
        score_slice_channels=state[
            "score_context.group_entropies.0.entropy_bottleneck.quantiles"
        ].shape[0],
        score_context_hidden=state["score_context.mean_conditioner.1.weight"].shape[0],
    )
    _validate_spatial_predictor_state(state, inferred)
    if metadata is None:
        return inferred
    declared = CodecConfig.from_mapping(metadata)
    _validate_config(inferred, declared, include_flags=False)
    _validate_spatial_predictor_state(state, declared)
    return declared


def _validate_config(
    actual: CodecConfig, expected: CodecConfig, *, include_flags: bool
) -> None:
    ignored = (
        set()
        if include_flags
        else {
            "use_residual",
            "use_morton",
            "morton_bits",
            "score_mean_condition",
            "score_channel_context",
            "score_spatial_context",
            "score_spatial_predictor",
            "score_spatial_entropy",
            "score_spatial_hidden",
        }
    )
    expected_values = expected.to_dict()
    mismatches = {
        key: (value, expected_values[key])
        for key, value in actual.to_dict().items()
        if key not in ignored and value != expected_values[key]
    }
    if mismatches:
        raise ValueError(
            f"checkpoint codec configuration mismatch (saved, requested): {mismatches}"
        )


def _validate_spatial_predictor_state(
    state: Mapping[str, Tensor], config: CodecConfig
) -> None:
    """Reject incomplete or retired score branches before loading model weights."""
    from .score_context import ContextualScoreEntropy

    with torch.random.fork_rng(devices=[]):
        context = ContextualScoreEntropy(
            rank=config.rank,
            scene_channels=config.observable_channels,
            slice_channels=config.score_slice_channels,
            hidden=config.score_context_hidden,
        )
    expected = {
        "score_context." + key: tuple(value.shape)
        for key, value in context.named_parameters()
    }
    missing = sorted(set(expected) - set(state))
    wrong = {
        key: (tuple(state[key].shape), shape)
        for key, shape in expected.items()
        if key in state and tuple(state[key].shape) != shape
    }
    allowed = {"score_context." + key for key in context.state_dict()}
    unexpected = sorted(
        key for key in state if key.startswith("score_context.") and key not in allowed
    )
    if missing or wrong or unexpected:
        raise ValueError(
            f"checkpoint Full P0+Split state mismatch: missing={missing}, "
            f"shapes={wrong}, unexpected={unexpected}"
        )


def validate_feature_codec_checkpoint(
    checkpoint: Mapping[str, Any], config: CodecConfig | Hyper1DConfig
) -> None:
    """Require the main codec architecture and matching saved metadata."""
    state, _ = _feature_state_dict(checkpoint.get("state_dict", checkpoint))
    metadata = checkpoint.get("feature_codec_config")
    actual = infer_feature_codec_config(state, metadata)
    if type(actual) is not type(config):
        raise ValueError("checkpoint codec type does not match requested codec")
    _validate_config(actual, config, include_flags=metadata is not None)


def infer_feature_codec_config(state, metadata=None):
    """Require explicit Hyper1D metadata: stride/norm cannot be inferred from weights."""
    kind = metadata.get("codec_type") if metadata is not None else None
    if kind is None:
        if "f_mean" in state or "g_a.0.weight" in state:
            raise ValueError("Hyper1D checkpoints require feature_codec_config.codec_type")
        return infer_config(state, metadata)
    if kind != "hyper1d":
        raise ValueError(f"unsupported checkpoint codec_type {kind!r}")
    # Pilots saved before architecture selection have the original adapters and
    # two-layer transforms. Fixed tensor names/shapes below verify that fallback.
    # Older checkpoints are single-path; a dual-path state cannot pass the
    # strict key validation below with this fallback.
    missing_metadata = set(Hyper1DConfig.__dataclass_fields__) - {"architecture", "paths", "base_rank"} - set(metadata)
    if missing_metadata:
        raise ValueError(f"incomplete Hyper1D checkpoint metadata: {sorted(missing_metadata)}")
    config = Hyper1DConfig.from_mapping(metadata)
    with torch.random.fork_rng(devices=[]):
        prototype = build_feature_codec(config)
    expected = prototype.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    # Coder tables have dynamic sizes. All parameters and normalization buffers are fixed.
    dynamic = {key for key in expected if key.rsplit(".", 1)[-1] in
               {"_quantized_cdf", "_offset", "_cdf_length", "scale_table"}}
    wrong = {key: (tuple(state[key].shape), tuple(value.shape)) for key, value in expected.items()
             if key in state and key not in dynamic and state[key].shape != value.shape}
    if missing or unexpected or wrong:
        raise ValueError(f"Hyper1D checkpoint state mismatch: missing={missing}, unexpected={unexpected}, shapes={wrong}")
    for name in ("f_mean", "f_std"):
        if not torch.isfinite(state[name]).all():
            raise ValueError("invalid Hyper1D normalization buffers")
    if (state["f_std"] <= 0).any():
        raise ValueError("invalid Hyper1D f_std")
    if config.input_norm == "none" and (torch.count_nonzero(state["f_mean"]) or not torch.equal(state["f_std"], torch.ones_like(state["f_std"]))):
        raise ValueError("input_norm=none requires identity normalization buffers")
    return config


def validate_score_mean_offset_mode(
    checkpoint: Mapping[str, Any],
    codec: ObservableLowRank1DCodec,
    *,
    allow_conversion: bool = False,
) -> None:
    """Keep the ablation mode with its checkpoint, except for an explicit warm start."""

    saved = checkpoint.get("score_mean_offset_enabled", True)
    if isinstance(codec.config, Hyper1DConfig):
        if "score_mean_offset_enabled" in checkpoint:
            raise ValueError("Hyper1D checkpoint contains score-only metadata")
        return
    requested = codec.score_context.mean_offset_enabled
    if type(saved) is not bool:
        raise ValueError("invalid score_mean_offset_enabled checkpoint metadata")
    if saved != requested and not allow_conversion:
        raise ValueError(
            f"checkpoint score_mean_offset_enabled={saved}, requested={requested}"
        )


def reset_score_mean_offset_head(codec: ObservableLowRank1DCodec) -> None:
    """Give control and ablation arms the same zero-offset initial forward pass."""

    head = codec.score_context.mean_conditioner[-1]
    rank = codec.config.rank
    with torch.no_grad():
        head.weight[:rank].zero_()
        head.bias[:rank].zero_()


def resize_registered_buffers(module: nn.Module, state: Mapping[str, Tensor]) -> None:
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

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    raw_state = checkpoint.get("state_dict", checkpoint)
    feature_state, prefix = _feature_state_dict(raw_state)
    config = infer_feature_codec_config(feature_state, checkpoint.get("feature_codec_config"))
    codec = build_feature_codec(config)
    if isinstance(config, CodecConfig):
        codec.score_context.mean_offset_enabled = checkpoint.get("score_mean_offset_enabled", True)
    validate_score_mean_offset_mode(checkpoint, codec)
    resize_registered_buffers(codec, feature_state)
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

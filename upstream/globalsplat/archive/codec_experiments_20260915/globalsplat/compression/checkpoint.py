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


def infer_config(
    state: Mapping[str, Tensor], metadata: Mapping[str, Any] | None = None
) -> CodecConfig:
    """Recover linear or two-sided MLP dimensions; restore flags from metadata."""
    if "shared_synthesis_basis" not in state:
        raise ValueError("checkpoint is not the supported untied-synthesis codec")
    basis = state["shared_basis"]
    geometry = state["geometry_projection.weight"]
    rank, observable_channels = basis.shape
    mlp_keys = {key for key in state if key.startswith(("analysis_mlp.", "synthesis_mlp."))}
    transform = "nonlinear" if mlp_keys else "linear"
    hidden = 32
    if mlp_keys:
        expected_keys = {
            "analysis_mlp.0.weight", "analysis_mlp.2.weight",
            "synthesis_mlp.0.weight", "synthesis_mlp.2.weight",
        }
        if mlp_keys != expected_keys:
            raise ValueError("nonlinear codec requires both bias-free analysis/synthesis MLPs")
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
    )
    if metadata is None:
        return inferred
    declared = CodecConfig.from_mapping(metadata)
    _validate_config(inferred, declared, include_flags=False)
    _validate_spatial_predictor_state(state, declared)
    return declared


def _validate_config(actual: CodecConfig, expected: CodecConfig, *, include_flags: bool) -> None:
    ignored = set() if include_flags else {
        "use_residual",
        "use_morton",
        "morton_bits",
        "score_mean_condition",
        "score_channel_context",
        "score_spatial_context",
        "score_spatial_predictor",
        "score_spatial_entropy",
        "score_spatial_hidden",
        "score_slice_channels",
        "score_context_hidden",
    }
    if actual.transform == expected.transform == "linear":
        ignored.add("transform_hidden")
    expected_values = expected.to_dict()
    mismatches = {
        key: (value, expected_values[key])
        for key, value in actual.to_dict().items()
        if key not in ignored and value != expected_values[key]
    }
    if mismatches:
        raise ValueError(f"checkpoint codec configuration mismatch (saved, requested): {mismatches}")


def _validate_spatial_predictor_state(
    state: Mapping[str, Tensor], config: CodecConfig
) -> None:
    """Check predictor keys/shapes without constructing the full residual codec."""

    expected: dict[str, tuple[int, ...]] = {}
    if config.score_spatial_context:
        if config.score_channel_context:
            widths = tuple(
                min(config.score_slice_channels, config.rank - start)
                for start in range(0, config.rank, config.score_slice_channels)
            )
        else:
            widths = (config.rank,)
        for index, width in enumerate(widths):
            prefix = f"score_context.spatial_predictors.{index}."
            expected[prefix + "weight"] = (width, width, 1, 3)
            expected[prefix + "bias"] = (width,)
            if config.score_spatial_predictor != "linear":
                kernel = 3 if config.score_spatial_predictor == "residual3" else 7
                prefix = f"score_context.spatial_corrections.{index}."
                expected[prefix + "0.weight"] = (
                    config.score_spatial_hidden,
                    width,
                    1,
                    kernel,
                )
                expected[prefix + "0.bias"] = (config.score_spatial_hidden,)
                expected[prefix + "2.weight"] = (
                    width,
                    config.score_spatial_hidden,
                    1,
                    1,
                )
                expected[prefix + "2.bias"] = (width,)

    actual = {
        key: tuple(value.shape)
        for key, value in state.items()
        if key.startswith(
            (
                "score_context.spatial_predictors.",
                "score_context.spatial_corrections.",
            )
        )
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        wrong_shapes = {
            key: (actual[key], expected[key])
            for key in set(actual) & set(expected)
            if actual[key] != expected[key]
        }
        raise ValueError(
            "checkpoint spatial predictor state mismatch "
            f"(missing={missing}, unexpected={unexpected}, "
            f"shapes(actual, expected)={wrong_shapes})"
        )

    odd_entropy_indices = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("score_context.spatial_odd_entropies.")
    }
    log_scale_indices = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("score_context.spatial_log_scales.")
    }
    scale_predictor_indices = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("score_context.spatial_scale_predictors.")
    }
    expected_indices = set(range(len(widths))) if config.score_spatial_context else set()
    if config.score_spatial_entropy == "split":
        expected_odd, expected_scales, expected_scale_predictors = expected_indices, set(), set()
    elif config.score_spatial_entropy == "gaussian":
        expected_odd, expected_scales, expected_scale_predictors = set(), expected_indices, set()
    elif config.score_spatial_entropy == "conditional_scale":
        expected_odd = set()
        expected_scales = expected_indices
        expected_scale_predictors = expected_indices
    else:
        expected_odd = expected_scales = expected_scale_predictors = set()
    actual_variants = (
        odd_entropy_indices,
        log_scale_indices,
        scale_predictor_indices,
    )
    expected_variants = (
        expected_odd,
        expected_scales,
        expected_scale_predictors,
    )
    if actual_variants != expected_variants:
        raise ValueError(
            "checkpoint spatial entropy state mismatch "
            f"(actual={actual_variants}, expected={expected_variants})"
        )


def validate_feature_codec_checkpoint(checkpoint: Mapping[str, Any], config: CodecConfig) -> None:
    """Reject wrong transforms/widths and, for new checkpoints, wrong ablation flags."""
    state, _ = _feature_state_dict(checkpoint.get("state_dict", checkpoint))
    metadata = checkpoint.get("feature_codec_config")
    actual = infer_config(state, metadata)
    _validate_config(actual, config, include_flags=metadata is not None)


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

    checkpoint = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    raw_state = checkpoint.get("state_dict", checkpoint)
    feature_state, prefix = _feature_state_dict(raw_state)
    config = infer_config(feature_state, checkpoint.get("feature_codec_config"))
    codec = ObservableLowRank1DCodec(config)
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

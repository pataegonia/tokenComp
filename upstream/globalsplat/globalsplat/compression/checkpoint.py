"""Infer a codec architecture and strict-load it from a Lightning checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from .codec import ObservableLowRank1DCodec
from .config import CodecConfig


FEATURE_PREFIXES = ("model.feature_codec.", "feature_codec.")
SCORE_PATH_FIELDS = (
    "use_centering", "use_score_norm", "transform",
    "score_mean_condition", "score_channel_context",
    "score_spatial_stages", "score_spatial_kernel", "score_context_quantization",
)


@dataclass(slots=True)
class LoadedCodec:
    codec: ObservableLowRank1DCodec
    config: CodecConfig
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
    if "shared_basis" in state_dict:
        return dict(state_dict), ""
    raise ValueError("checkpoint has no model.feature_codec state")


def infer_config(
    state: Mapping[str, Tensor], metadata: Mapping[str, Any] | None = None
) -> CodecConfig:
    """Recover dimensions; ablation switches live in checkpoint metadata.

    Disabled branches retain frozen compatibility tensors, so their presence
    cannot determine the active score path. Older metadata defaults to full.
    """
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
    stages = 4 if any(key.startswith("score_context.split_odd_") for key in state) else (
        3 if any(key.startswith("score_context.split_even_") for key in state) else 2
    )
    kernel = state["score_context.spatial_predictors.0.weight"].shape[-1]
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
        score_spatial_stages=stages,
        score_spatial_kernel=kernel,
        score_channel_context=stages == 2,
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
            *(key for key in SCORE_PATH_FIELDS if key not in (
                "score_spatial_stages", "score_spatial_kernel"
            )),
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
            mean_condition=config.score_mean_condition,
            channel_context=config.score_channel_context,
            spatial_stages=config.score_spatial_stages,
            spatial_kernel=config.score_spatial_kernel,
            context_quantization=config.score_context_quantization,
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
    checkpoint: Mapping[str, Any], config: CodecConfig, *,
    allow_score_path_conversion: bool = False,
) -> None:
    """Require the main codec architecture and matching saved metadata."""
    state, _ = _feature_state_dict(checkpoint.get("state_dict", checkpoint))
    metadata = checkpoint.get("feature_codec_config")
    actual = infer_config(state, metadata)
    if allow_score_path_conversion:
        # Only explicitly requested score switches may change in a weights-only
        # warm start. Dimensions, MSH and every other setting must still match.
        actual = replace(actual, **{key: getattr(config, key) for key in SCORE_PATH_FIELDS})
    _validate_config(actual, config, include_flags=True)


def convert_score_path_state(
    state_dict: Mapping[str, Tensor], codec: ObservableLowRank1DCodec,
) -> dict[str, Tensor]:
    """Adapt only spatial score modules for an explicitly allowed warm start.

    Kernel 3 weights occupy the middle of kernel 5. New B uses a zero predictor
    and a copy of the even entropy model; new D copies the odd predictor/model.
    Everything outside score_context is passed through untouched. This is a
    weights-only conversion, never an optimizer/resume or evaluation migration.
    """
    state, feature_prefix = _feature_state_dict(state_dict)
    prefix = feature_prefix + "score_context."
    source = {key[len("score_context."):]: value for key, value in state.items()
              if key.startswith("score_context.")}
    converted = {key: value for key, value in state_dict.items()
                 if not key.startswith(prefix)}
    for key, initial in codec.score_context.state_dict().items():
        source_key = key
        if key not in source:
            if key.startswith("split_even_entropies."):
                source_key = key.replace("split_even_entropies.", "group_entropies.", 1)
            elif key.startswith("split_odd_entropies."):
                source_key = key.replace("split_odd_entropies.", "spatial_odd_entropies.", 1)
            elif key.startswith("split_odd_predictors."):
                source_key = key.replace("split_odd_predictors.", "spatial_predictors.", 1)
            elif key.startswith("split_even_predictors."):
                converted[prefix + key] = initial.detach().clone()
                continue
        if source_key not in source:
            raise ValueError(f"cannot initialize score context tensor: {key}")
        value = source[source_key]
        if "predictors." in key and key.endswith(".weight") and value.shape != initial.shape:
            if value.ndim != 4 or value.shape[:-1] != initial.shape[:-1]:
                raise ValueError(f"cannot convert spatial predictor shape: {key}")
            resized = value.new_zeros(initial.shape)
            width = min(value.shape[-1], initial.shape[-1])
            left_source = (value.shape[-1] - width) // 2
            left_target = (initial.shape[-1] - width) // 2
            resized[..., left_target:left_target + width] = value[..., left_source:left_source + width]
            value = resized
        converted[prefix + key] = value
    return converted


def validate_score_mean_offset_mode(
    checkpoint: Mapping[str, Any],
    codec: ObservableLowRank1DCodec,
    *,
    allow_conversion: bool = False,
) -> None:
    """Keep the ablation mode with its checkpoint, except for an explicit warm start."""

    saved = checkpoint.get("score_mean_offset_enabled", True)
    requested = codec.score_context.mean_offset_enabled
    if type(saved) is not bool:
        raise ValueError("invalid score_mean_offset_enabled checkpoint metadata")
    if saved != requested and not allow_conversion:
        raise ValueError(
            f"checkpoint score_mean_offset_enabled={saved}, requested={requested}"
        )


def validate_weights_only_load(
    missing_keys: Iterable[str], unexpected_keys: Iterable[str],
) -> None:
    """Require all model weights, while allowing rebuilt frozen loss networks.

    GlobalSplatModule.on_save_checkpoint intentionally strips render_criterion.
    Unlike Lightning resume, a direct weights-only load does not call the hook
    that restores those fixed tensors. The constructor already rebuilt them.
    """
    missing = [key for key in missing_keys if not key.startswith("render_criterion.")]
    unexpected = [key for key in unexpected_keys if not key.startswith("render_criterion.")]
    if missing or unexpected:
        raise RuntimeError(
            f"training checkpoint does not match model: missing={missing}, unexpected={unexpected}"
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
    config = infer_config(feature_state, checkpoint.get("feature_codec_config"))
    codec = ObservableLowRank1DCodec(config)
    codec.score_context.mean_offset_enabled = checkpoint.get(
        "score_mean_offset_enabled", True
    )
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

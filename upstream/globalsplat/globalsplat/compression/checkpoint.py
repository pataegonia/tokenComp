"""Validate a codec architecture and strict-load it from a Lightning checkpoint."""

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
    "score_spatial_stages", "score_spatial_kernel",
    "token_order", "use_morton", "score_spatial_context", "score_order_metric",
    "score_context_schedule",
    "score_slice_channels",
)
ENCODER_IMPLEMENTATION_FIELDS = (
    "nn_order_backend", "nn_order_candidates", "nn_order_distance_device",
)
# Single-path initializers may change only these; tensor shapes stay the same.
SINGLE_PATH_SWITCHES = (
    "token_order", "use_morton", "score_spatial_context", "score_order_metric",
    *ENCODER_IMPLEMENTATION_FIELDS,
)
# Frozen tensors of modules removed in the 2026-10 cleanup. Pre-cleanup minimal
# checkpoints still carry them; their declared configuration never used them.
RETIRED_STATE_PREFIXES = (
    "score_log_scale", "score_entropy.", "analysis_mlp.", "synthesis_mlp.",
    "score_context.mean_conditioner.", "score_context.channel_predictors.",
)
# CompressAI integer tables are rebuilt from the weights and may change shape.
DYNAMIC_BUFFERS = ("_quantized_cdf", "_cdf_length", "_offset", "scale_table")
# Saved metadata omits fields added after it was written; those runs used the
# defaults of that time, which the cleanup changed.
PRE_CLEANUP_LAYOUT_DEFAULTS = dict(
    token_order="morton", score_spatial_stages=2, score_spatial_kernel=3, score_context_schedule="legacy",
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
    if "shared_basis" in state_dict or "geometry_projection.weight" in state_dict:
        return dict(state_dict), ""
    raise ValueError("checkpoint has no model.feature_codec state")


def strip_retired_state(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Drop retired-module tensors from a model, codec or pre-cleanup checkpoint state."""

    def retired(key: str) -> bool:
        return any(key.startswith(prefix) and key[len(prefix):].startswith(RETIRED_STATE_PREFIXES)
                   for prefix in (*FEATURE_PREFIXES, ""))

    return {key: value for key, value in state_dict.items() if not retired(key)}


def check_retired_metadata(checkpoint: Mapping[str, Any]) -> None:
    """Reject a checkpoint whose Score streams set the retired no-offset flag."""
    if checkpoint.get("score_mean_offset_enabled", True) is not True:
        raise ValueError("score_mean_offset_enabled=false was removed in the codec cleanup; "
                         "use git tag fresh-rd-v1 for that checkpoint")


def infer_config(
    state: Mapping[str, Tensor], metadata: Mapping[str, Any] | None = None
) -> CodecConfig:
    """Return the saved configuration after checking it against every tensor.

    Disabled branches retain frozen compatibility tensors, so tensors alone
    cannot determine the active score path; the metadata is required.
    """
    if metadata is None:
        raise ValueError("checkpoint has no feature_codec_config; pre-minimal codecs need git tag fresh-rd-v1")
    declared = CodecConfig.from_mapping({**PRE_CLEANUP_LAYOUT_DEFAULTS, **metadata})
    state = strip_retired_state(state)
    with torch.random.fork_rng(devices=[]):
        expected = ObservableLowRank1DCodec(declared).state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    wrong = {key: (tuple(state[key].shape), tuple(value.shape))
             for key, value in expected.items() if key in state
             and key.rsplit(".", 1)[-1] not in DYNAMIC_BUFFERS and state[key].shape != value.shape}
    if missing or unexpected or wrong:
        raise ValueError(f"checkpoint codec state mismatch: missing={missing}, "
                         f"unexpected={unexpected}, shapes={wrong}")
    marker = state.get("score_context.single_block_marker")
    if marker is not None and (marker.dtype != torch.uint8 or marker.item() != 1):
        raise ValueError("invalid single-block checkpoint marker")
    return declared


def _validate_config(actual: CodecConfig, expected: CodecConfig) -> None:
    # These select how an encoder computes the same greedy rule, not model
    # architecture. Switching them is allowed in strict optimizer-state resume.
    ignored = set(ENCODER_IMPLEMENTATION_FIELDS)
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


def validate_feature_codec_checkpoint(
    checkpoint: Mapping[str, Any], config: CodecConfig, *,
    allow_score_path_conversion: bool = False,
) -> None:
    """Require the main codec architecture and matching saved metadata."""
    check_retired_metadata(checkpoint)
    state, _ = _feature_state_dict(checkpoint.get("state_dict", checkpoint))
    metadata = checkpoint.get("feature_codec_config")
    actual = infer_config(state, metadata)
    if allow_score_path_conversion:
        if actual.codec_mode != config.codec_mode:
            raise ValueError("a weights-only conversion cannot change codec_mode")
        if actual.codec_mode != "two_path":
            # Explicit single-path initializers: only the encoder order and the
            # spatial-context switch may change, never any tensor or other mode.
            actual = replace(actual, **{key: getattr(config, key) for key in SINGLE_PATH_SWITCHES})
        else:
            if actual.score_slice_channels != config.score_slice_channels and config.score_spatial_context:
                raise ValueError("channel regrouping requires spatial context OFF")
            # Only explicitly requested score switches may change in a weights-only
            # warm start. Feature/rank dimensions, MSH and other settings still match.
            actual = replace(actual, **{key: getattr(config, key) for key in
                                       (*SCORE_PATH_FIELDS, *ENCODER_IMPLEMENTATION_FIELDS)})
    _validate_config(actual, config)


def convert_score_path_state(
    state_dict: Mapping[str, Tensor], codec: ObservableLowRank1DCodec,
) -> dict[str, Tensor]:
    """Adapt Score context modules for an explicitly allowed weights-only load.

    Kernel 3 weights occupy the middle of kernel 5. New B uses a zero predictor
    and a copy of the even entropy model; new D copies the odd predictor/model.
    Context-OFF channel regrouping concatenates the parent's unpredicted
    per-channel densities and uses the same initialization in every target
    stage. Integer CDF tables are rebuilt. Everything outside score_context
    passes through untouched. Optimizer resume and evaluation stay strict.
    """
    state_dict = strip_retired_state(state_dict)
    state, feature_prefix = _feature_state_dict(state_dict)
    prefix = feature_prefix + "score_context."
    source = {key[len("score_context."):]: value for key, value in state.items()
              if key.startswith("score_context.")}
    source_widths = []
    while f"group_entropies.{len(source_widths)}.entropy_bottleneck.quantiles" in source:
        source_widths.append(source[f"group_entropies.{len(source_widths)}.entropy_bottleneck.quantiles"].shape[0])
    regroup = tuple(source_widths) != codec.score_context.group_sizes
    if regroup and codec.config.score_spatial_context:
        raise ValueError("channel regrouping requires spatial context OFF")
    if sum(source_widths) != codec.config.rank:
        raise ValueError("source Score entropy channels do not cover the requested rank")
    parameter_names = set(dict(codec.score_context.named_parameters()))
    entropy_families = ("group_entropies", "spatial_odd_entropies", "split_even_entropies", "split_odd_entropies")
    converted = {key: value for key, value in state_dict.items()
                 if not key.startswith(prefix)}
    for key, initial in codec.score_context.state_dict().items():
        source_key = key
        if key == "single_block_marker":
            converted[prefix + key] = initial.detach().clone()
            continue
        if regroup and key.split(".", 1)[0] in entropy_families:
            # New Context-OFF arms start from the SAME parent unpredicted
            # per-channel density in every stage. Concatenation changes stream
            # grouping without changing any channel's initial density/median.
            _, group_text, suffix = key.split(".", 2)
            group = int(group_text)
            source_keys = [f"group_entropies.{i}.{suffix}" for i in range(len(source_widths))]
            if suffix.rsplit(".", 1)[-1] in ("_quantized_cdf", "_cdf_length", "_offset"):
                value = initial.detach().clone()  # rebuild integer CDFs, never splice padded tables
            elif key in parameter_names:
                tensors = [source[k] for k in source_keys]
                if any(v.shape[0] != width or v.shape[1:] != initial.shape[1:]
                       for v, width in zip(tensors, source_widths)):
                    raise ValueError(f"cannot regroup Score entropy parameter: {key}")
                start = sum(codec.score_context.group_sizes[:group])
                value = torch.cat(tensors, 0)[start:start + codec.score_context.group_sizes[group]].clone()
            else:
                tensors = [source[k] for k in source_keys]
                if any(v.shape != initial.shape or not torch.equal(v, tensors[0]) for v in tensors):
                    raise ValueError(f"inconsistent shared Score entropy buffer: {key}")
                value = tensors[0].clone()
            converted[prefix + key] = value
            continue
        if regroup and "predictors." in key:
            # Regrouping is allowed only when these compatibility branches are disabled.
            converted[prefix + key] = initial.detach().clone()
            continue
        if key.startswith("anchor_predictors.") and (
            key not in source or source[key].shape != initial.shape
        ):
            converted[prefix + key] = initial.detach().clone()
            continue
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


def validate_weights_only_load(
    missing_keys: Iterable[str], unexpected_keys: Iterable[str],
) -> None:
    """Require all model weights, while allowing rebuilt frozen loss networks.

    GlobalSplatModule.on_save_checkpoint intentionally strips render_criterion.
    Unlike Lightning resume, a direct weights-only load does not call the hook
    that restores those fixed tensors. The constructor already rebuilt them.
    """
    # A new Top-K selector starts from its own initialization (plan §14).
    missing = [key for key in missing_keys
               if not key.startswith(("render_criterion.", "model.token_selector."))]
    unexpected = [key for key in unexpected_keys if not key.startswith("render_criterion.")]
    if missing or unexpected:
        raise RuntimeError(
            f"training checkpoint does not match model: missing={missing}, unexpected={unexpected}"
        )


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
    check_retired_metadata(checkpoint)
    raw_state = checkpoint.get("state_dict", checkpoint)
    feature_state, prefix = _feature_state_dict(raw_state)
    feature_state = strip_retired_state(feature_state)
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

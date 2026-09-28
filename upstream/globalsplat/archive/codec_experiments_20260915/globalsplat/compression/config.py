"""Configuration for the reconstructed feature codec."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CodecConfig:
    """Architecture values for the production NFC-GS codec path."""

    texture_channels: int = 512
    geometry_channels: int = 512
    geometry_observable_channels: int = 224
    rank: int = 56
    residual_n: int = 192
    residual_m: int = 320
    adapter_hidden: int = 96
    morton_bits: int = 10
    use_morton: bool = True
    use_residual: bool = True
    transform: str = "linear"
    transform_hidden: int = 32
    score_mean_condition: bool = False
    score_channel_context: bool = False
    score_spatial_context: bool = False
    score_spatial_predictor: str = "linear"
    score_spatial_entropy: str = "shared"
    score_spatial_hidden: int = 32
    score_slice_channels: int = 16
    score_context_hidden: int = 64

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.residual_n <= 0 or self.residual_m <= 0:
            raise ValueError("residual channel counts must be positive")
        if self.adapter_hidden <= 0 or self.adapter_hidden % 3:
            raise ValueError("adapter_hidden must be positive and divisible by 3")
        if not 1 <= self.morton_bits <= 21:
            raise ValueError("morton_bits must be between 1 and 21")
        if not isinstance(self.use_morton, bool):
            raise ValueError("use_morton must be a boolean")
        if not isinstance(self.use_residual, bool):
            raise ValueError("use_residual must be a boolean")
        if self.transform not in ("linear", "nonlinear"):
            raise ValueError("transform must be linear or nonlinear")
        if (
            not isinstance(self.transform_hidden, int)
            or isinstance(self.transform_hidden, bool)
            or self.transform_hidden <= 0
        ):
            raise ValueError("transform_hidden must be a positive integer")
        for name in (
            "score_mean_condition",
            "score_channel_context",
            "score_spatial_context",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.score_spatial_context and not self.use_morton:
            raise ValueError("score_spatial_context requires Morton ordering")
        if self.score_spatial_predictor not in ("linear", "residual3", "residual7"):
            raise ValueError(
                "score_spatial_predictor must be linear, residual3, or residual7"
            )
        if self.score_spatial_predictor != "linear" and not self.score_spatial_context:
            raise ValueError("a residual score_spatial_predictor requires spatial context")
        if self.score_spatial_entropy not in (
            "shared",
            "split",
            "gaussian",
            "conditional_scale",
        ):
            raise ValueError(
                "score_spatial_entropy must be shared, split, gaussian, or "
                "conditional_scale"
            )
        if self.score_spatial_entropy != "shared" and not self.score_spatial_context:
            raise ValueError("a non-shared score_spatial_entropy requires spatial context")
        if (
            not isinstance(self.score_spatial_hidden, int)
            or isinstance(self.score_spatial_hidden, bool)
            or self.score_spatial_hidden <= 0
        ):
            raise ValueError("score_spatial_hidden must be a positive integer")
        if (
            not isinstance(self.score_slice_channels, int)
            or isinstance(self.score_slice_channels, bool)
            or self.score_slice_channels <= 0
        ):
            raise ValueError("score_slice_channels must be a positive integer")
        if self.score_channel_context and self.score_slice_channels > self.rank:
            raise ValueError("score_slice_channels cannot exceed rank with channel context")
        if (
            not isinstance(self.score_context_hidden, int)
            or isinstance(self.score_context_hidden, bool)
            or self.score_context_hidden <= 0
        ):
            raise ValueError("score_context_hidden must be a positive integer")

    @property
    def observable_channels(self) -> int:
        return self.texture_channels + self.geometry_observable_channels

    @property
    def use_contextual_score(self) -> bool:
        return bool(
            self.score_mean_condition
            or self.score_channel_context
            or self.score_spatial_context
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CodecConfig":
        """Build from the YAML spelling used by the codec configuration."""

        aliases = {
            "residual_N": "residual_n",
            "residual_M": "residual_m",
        }
        fields = cls.__dataclass_fields__
        normalized: dict[str, Any] = {}
        for key, value in dict(values).items():
            target = aliases.get(key, key)
            if target not in fields:
                raise ValueError(f"unsupported feature_codec setting: {key}")
            normalized[target] = value
        return cls(**normalized)

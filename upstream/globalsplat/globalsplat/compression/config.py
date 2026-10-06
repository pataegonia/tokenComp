"""Full P0+Split codec configuration, including saved checkpoint metadata."""

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
    use_centering: bool = True
    use_score_norm: bool = True
    transform: str = "nonlinear"
    transform_hidden: int = 32
    score_mean_condition: bool = True
    score_channel_context: bool = True
    score_spatial_context: bool = True
    score_spatial_predictor: str = "linear"
    score_spatial_entropy: str = "split"
    score_spatial_hidden: int = 32
    score_slice_channels: int = 16
    score_context_hidden: int = 64
    score_spatial_stages: int = 2
    score_spatial_kernel: int = 3
    score_context_quantization: str = "noise"

    def __post_init__(self) -> None:
        # MSH residual and spatial Split coding remain mandatory.
        required = {
            "use_morton": True,
            "use_residual": True,
            "score_spatial_context": True,
            "score_spatial_predictor": "linear",
            "score_spatial_entropy": "split",
        }
        for name, expected in required.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"main codec requires {name}={expected!r}; got {actual!r}. "
                    "Other codec paths have been archived."
                )
        for name in ("use_centering", "use_score_norm", "score_mean_condition", "score_channel_context"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.transform not in ("linear", "nonlinear"):
            raise ValueError("transform must be linear or nonlinear")
        if not self.use_centering and self.score_mean_condition:
            raise ValueError("use_centering=false requires score_mean_condition=false (no mean is transmitted)")
        if type(self.score_spatial_stages) is not int or self.score_spatial_stages not in (2, 3, 4):
            raise ValueError("score_spatial_stages must be 2, 3 or 4")
        if type(self.score_spatial_kernel) is not int or self.score_spatial_kernel not in (3, 5):
            raise ValueError("score_spatial_kernel must be 3 or 5")
        if self.score_context_quantization not in ("noise", "ste"):
            raise ValueError("score_context_quantization must be noise or ste")
        if self.score_spatial_stages > 2:
            if self.score_spatial_kernel != 5:
                raise ValueError("3/4 spatial stages require kernel 5 to see distance-2 context")
            if self.score_channel_context:
                raise ValueError("3/4 token stages require score_channel_context=false")
        for name in (
            "texture_channels",
            "geometry_channels",
            "geometry_observable_channels",
            "rank",
            "residual_n",
            "residual_m",
            "adapter_hidden",
            "transform_hidden",
            "score_spatial_hidden",
            "score_slice_channels",
            "score_context_hidden",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.adapter_hidden % 3:
            raise ValueError("adapter_hidden must be divisible by 3")
        if type(self.morton_bits) is not int or not 1 <= self.morton_bits <= 21:
            raise ValueError("morton_bits must be between 1 and 21")
        if self.score_slice_channels > self.rank:
            raise ValueError("score_slice_channels cannot exceed rank")

    @property
    def observable_channels(self) -> int:
        return self.texture_channels + self.geometry_observable_channels

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

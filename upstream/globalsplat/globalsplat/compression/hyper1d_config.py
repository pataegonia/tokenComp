"""Configuration for the full-feature mean/scale hyperprior."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Hyper1DConfig:
    codec_type: str = "hyper1d"
    texture_channels: int = 512
    geometry_channels: int = 512
    geometry_observable_channels: int = 224
    n: int = 192
    m: int = 320
    adapter_hidden: int = 96
    strides: tuple[int, int] = (2, 2)
    use_morton: bool = False
    morton_bits: int = 10
    input_norm: str = "none"
    architecture: str = "legacy"
    paths: int = 1
    base_rank: int = 0

    def __post_init__(self) -> None:
        if self.codec_type != "hyper1d":
            raise ValueError("Hyper1DConfig requires codec_type=hyper1d")
        for name in ("texture_channels", "geometry_channels", "geometry_observable_channels",
                     "n", "m", "adapter_hidden", "morton_bits"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.geometry_observable_channels > self.geometry_channels:
            raise ValueError("geometry projection cannot expand the observable subspace")
        if self.adapter_hidden % 3:
            raise ValueError("adapter_hidden must be divisible by three")
        strides = tuple(self.strides)
        if strides not in ((2, 2), (2, 1)) or any(type(s) is not int for s in strides):
            raise ValueError("strides must be (2,2) or (2,1)")
        object.__setattr__(self, "strides", strides)
        if type(self.use_morton) is not bool or not 1 <= self.morton_bits <= 21:
            raise ValueError("invalid Morton configuration")
        if self.input_norm not in ("none", "calibrated"):
            raise ValueError("input_norm must be none or calibrated")
        if self.architecture not in ("legacy", "plain4"):
            raise ValueError("architecture must be legacy or plain4")
        if type(self.paths) is not int or self.paths not in (1, 2):
            raise ValueError("paths must be 1 (single MSH) or 2 (base MSH + residual MSH)")
        if type(self.base_rank) is not int or not 0 <= self.base_rank < self.observable_channels:
            raise ValueError("base_rank must be zero or smaller than observable_channels")
        if self.base_rank and self.paths != 2:
            raise ValueError("base_rank requires paths=2")

    @property
    def observable_channels(self) -> int:
        return self.texture_channels + self.geometry_observable_channels

    def latent_shapes(self, points: int) -> tuple[tuple[int, int], tuple[int, int]]:
        if type(points) is not int or points <= 0:
            raise ValueError("points must be a positive integer")
        width = points
        for stride in self.strides:
            width = (width + stride - 1) // stride
        return (1, width), (1, (width + 3) // 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def for_architecture(cls, architecture: str, **values: Any) -> "Hyper1DConfig":
        """Initialization presets; omitted checkpoint metadata stays legacy."""
        widths = {"legacy": (192, 320), "plain4": (256, 512)}
        if architecture not in widths:
            raise ValueError("architecture must be legacy or plain4")
        n, m = widths[architecture]
        return cls(**dict({"architecture": architecture, "n": n, "m": m}, **values))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Hyper1DConfig":
        return cls(**dict(values))

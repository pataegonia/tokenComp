"""Configuration for the reconstructed feature codec."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CodecConfig:
    """Architecture values recoverable from the retained checkpoints."""

    texture_channels: int = 512
    geometry_channels: int = 512
    geometry_observable_channels: int = 224
    rank: int = 56
    residual_n: int = 192
    residual_m: int = 320
    adapter_hidden: int = 96
    basis_parameterization: str = "untied_synthesis"
    synthesis_mlp_hidden: int | None = 64
    morton_bits: int = 10

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.residual_n <= 0 or self.residual_m <= 0:
            raise ValueError("residual channel counts must be positive")
        if self.adapter_hidden <= 0 or self.adapter_hidden % 3:
            raise ValueError("adapter_hidden must be positive and divisible by 3")
        if self.basis_parameterization not in {"tied", "untied_synthesis"}:
            raise ValueError(
                "basis_parameterization must be 'tied' or 'untied_synthesis'"
            )
        if not 1 <= self.morton_bits <= 21:
            raise ValueError("morton_bits must be between 1 and 21")

    @property
    def observable_channels(self) -> int:
        return self.texture_channels + self.geometry_observable_channels

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


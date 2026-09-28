"""Small entropy-model wrappers with checkpoint-compatible names."""

from __future__ import annotations

from torch import Tensor, nn
from compressai.entropy_models import EntropyBottleneck


class FactorizedScoreEntropy(nn.Module):
    """Factorized entropy bottleneck for low-rank score channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels)

    def forward(self, scores: Tensor, *, training: bool | None = None) -> tuple[Tensor, Tensor]:
        if scores.ndim != 4:
            raise ValueError("score tensor must have shape [B,R,1,N]")
        if training is None:
            training = self.training
        return self.entropy_bottleneck(scores, training=training)

    def compress(self, scores: Tensor) -> list[bytes]:
        return self.entropy_bottleneck.compress(scores)

    def decompress(self, strings: list[bytes], shape: tuple[int, int]) -> Tensor:
        return self.entropy_bottleneck.decompress(strings, shape)

    def update(self, force: bool = False) -> bool:
        return bool(self.entropy_bottleneck.update(force=force))


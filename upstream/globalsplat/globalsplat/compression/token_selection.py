"""Adaptive integer K token selection (plan §14).

Follows TokenCodecGS ``docs/adaptive_integer_k_porting.md`` (single-rate
Adaptive integer K): a count router predicts a continuous count κ per scene,
an importance scorer ranks tokens, the top K = round(κ) are kept. Training uses
a straight-through gate (hard forward, normalized soft backward); this is a
biased surrogate, not the exact derivative of Top-K/round.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TokenSelection:
    indices: list[Tensor]   # per scene, kept original slots in coding order
    gates: Tensor           # [B,N]; hard 0/1 forward, soft backward in training
    expected_count: Tensor  # κ [B]
    selected_count: Tensor  # K [B]
    scores: Tensor          # [B,N]


def _mlp(inputs: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, 1))


class TokenSelector(nn.Module):
    """Scene-wise count router and token-wise importance scorer on [B,N,D] tokens."""

    def __init__(self, dim: int, tokens: int, *, minimum: int = 256, hidden: int = 128,
                 count_temperature: float = 1.0, score_temperature: float = 1.0,
                 init_logit: float = 2.94, forced_count: int = 0, order: str = "slot") -> None:
        super().__init__()
        if not 1 <= minimum <= tokens or not 0 <= forced_count <= tokens:
            raise ValueError("selection counts must lie in [1, tokens]")
        if forced_count and forced_count < minimum:
            raise ValueError("forced_count must be at least minimum")
        if count_temperature <= 0 or score_temperature <= 0:
            raise ValueError("selection temperatures must be positive")
        if order not in ("slot", "importance"):
            raise ValueError("selection order must be slot or importance")
        self.dim, self.tokens, self.minimum = int(dim), int(tokens), int(minimum)
        self.count_temperature, self.score_temperature = float(count_temperature), float(score_temperature)
        self.forced_count, self.order = int(forced_count), order
        # Single-rate router input: scene mean plus log1p(budget condition = 0).
        self.router = _mlp(self.dim + 1, hidden)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, float(init_logit))  # 2.94 starts κ near 95%
        self.scorer = _mlp(self.dim, hidden)
        nn.init.normal_(self.scorer[-1].weight, std=1e-3)
        nn.init.zeros_(self.scorer[-1].bias)
        self.set_trainable(True)

    def set_trainable(self, trainable: bool = True) -> None:
        self.scorer.requires_grad_(trainable)
        # A forced count never uses the router.
        self.router.requires_grad_(trainable and not self.forced_count)

    def forward(self, tokens: Tensor) -> TokenSelection:
        if tokens.ndim != 3 or tokens.shape[1:] != (self.tokens, self.dim):
            raise ValueError(f"expected fixed, unpadded [B,{self.tokens},{self.dim}] tokens")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            x = tokens.float()
            batch = x.shape[0]
            span = self.tokens - self.minimum
            if self.forced_count:
                counts = torch.full((batch,), self.forced_count, device=x.device, dtype=torch.long)
                kappa = counts.float()
            else:
                feature = torch.cat([x.mean(1), x.new_zeros(batch, 1)], -1)
                logit = self.router(feature).squeeze(-1)
                kappa = self.minimum + span * torch.sigmoid(logit / self.count_temperature)
                counts = kappa.detach().round().clamp(self.minimum, self.tokens).long()
            scores = self.scorer(x).squeeze(-1)
            hard = torch.zeros_like(scores)
            indices, thresholds = [], []
            for scene, count in enumerate(counts.tolist()):
                ranked = scores[scene].topk(count).indices
                chosen = ranked if self.order == "importance" else ranked.sort().values
                indices.append(chosen)
                hard[scene].index_fill_(0, chosen, 1.0)
                thresholds.append(scores[scene].index_select(0, chosen).min())
            gates = hard
            if self.training:
                threshold = torch.stack(thresholds).detach()[:, None]
                soft = torch.sigmoid((scores - threshold) / self.score_temperature)
                soft = soft * (kappa[:, None] / soft.sum(1, keepdim=True).clamp_min(1e-6))
                gates = hard + (soft - soft.detach())
        return TokenSelection(indices, gates, kappa, counts, scores)


def rate_with_count_surrogate(bits: Tensor, expected_count: Tensor, counts: Tensor) -> Tensor:
    """Per-scene bits plus ∂R/∂κ ≈ stop_gradient(R/K); the forward value is unchanged."""
    per_token = bits.detach() / counts.detach().float().clamp_min(1)
    proxy = expected_count.float() * per_token
    return bits + (proxy - proxy.detach())

"""Deterministic geometry-guided Morton ordering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MortonOrder:
    """Forward and inverse token permutations for a batch."""

    permutation: Tensor
    inverse_permutation: Tensor
    codes: Tensor

    def apply(self, values: Tensor) -> Tensor:
        return batched_gather(values, self.permutation)

    def restore(self, values: Tensor) -> Tensor:
        return batched_gather(values, self.inverse_permutation)


def batched_gather(values: Tensor, indices: Tensor) -> Tensor:
    """Gather dimension 1 with one permutation per batch element."""

    if values.ndim < 2 or indices.ndim != 2:
        raise ValueError("values must be [B,N,...] and indices must be [B,N]")
    if values.shape[:2] != indices.shape:
        raise ValueError("values and indices must have the same [B,N] shape")
    view = indices.shape + (1,) * (values.ndim - 2)
    expanded = indices.view(view).expand_as(values)
    return torch.gather(values, 1, expanded)


def invert_permutation(permutation: Tensor) -> Tensor:
    """Return the inverse of each row in a batched permutation."""

    if permutation.ndim != 2:
        raise ValueError("permutation must have shape [B,N]")
    inverse = torch.empty_like(permutation)
    source = torch.arange(
        permutation.shape[1], device=permutation.device, dtype=permutation.dtype
    ).expand_as(permutation)
    inverse.scatter_(1, permutation, source)
    return inverse


def _quantize_positions(positions: Tensor, bits: int) -> Tensor:
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B,N,3]")
    if not torch.is_floating_point(positions):
        positions = positions.float()
    minimum = positions.amin(dim=1, keepdim=True)
    span = positions.amax(dim=1, keepdim=True) - minimum
    normalized = (positions - minimum) / span.clamp_min(torch.finfo(positions.dtype).eps)
    levels = (1 << bits) - 1
    return torch.round(normalized.clamp(0, 1) * levels).to(torch.int64)


def morton_codes_3d(positions: Tensor, bits: int = 10) -> Tensor:
    """Calculate 3-D Morton/Z-order codes after scene-local normalization."""

    if not 1 <= bits <= 21:
        raise ValueError("bits must be between 1 and 21 for signed int64 codes")
    quantized = _quantize_positions(positions, bits)
    x, y, z = quantized.unbind(dim=-1)
    codes = torch.zeros_like(x)
    for bit in range(bits):
        codes |= ((x >> bit) & 1) << (3 * bit)
        codes |= ((y >> bit) & 1) << (3 * bit + 1)
        codes |= ((z >> bit) & 1) << (3 * bit + 2)
    return codes


def morton_order_3d(positions: Tensor, bits: int = 10) -> MortonOrder:
    """Build a stable Morton permutation from decoded token centers."""

    codes = morton_codes_3d(positions, bits)
    permutation = torch.argsort(codes, dim=1, stable=True)
    return MortonOrder(permutation, invert_permutation(permutation), codes)

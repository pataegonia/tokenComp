"""Encoder-only token orders; score and MSH always share the permutation."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .morton import MortonOrder, _quantize_positions, invert_permutation, morton_codes_3d


def hilbert_codes_3d(positions: Tensor, bits: int = 10) -> Tensor:
    """Skilling axes-to-transpose Hilbert keys on the same grid as Morton."""
    if not 1 <= bits <= 21:
        raise ValueError("bits must be between 1 and 21")
    axes = _quantize_positions(positions, bits).clone()
    q = 1 << (bits - 1)
    while q > 1:
        p = q - 1
        for axis in range(3):
            hit = (axes[..., axis] & q) != 0
            first = axes[..., 0].clone()
            exchange = (first ^ axes[..., axis]) & p
            axes[..., 0] = torch.where(hit, first ^ p, first ^ exchange)
            if axis:
                axes[..., axis] = torch.where(hit, axes[..., axis], axes[..., axis] ^ exchange)
        q >>= 1
    for axis in range(1, 3):
        axes[..., axis] ^= axes[..., axis - 1]
    correction = torch.zeros_like(axes[..., 0])
    q = 1 << (bits - 1)
    while q > 1:
        correction = torch.where((axes[..., 2] & q) != 0, correction ^ (q - 1), correction)
        q >>= 1
    axes ^= correction[..., None]
    codes = torch.zeros_like(axes[..., 0])
    for bit in range(bits - 1, -1, -1):
        for axis in range(3):
            codes = (codes << 1) | ((axes[..., axis] >> bit) & 1)
    return codes


def hilbert_order_3d(positions: Tensor, bits: int = 10) -> MortonOrder:
    codes = hilbert_codes_3d(positions, bits)
    permutation = torch.argsort(codes, dim=1, stable=True)
    return MortonOrder(permutation, invert_permutation(permutation), codes)


@torch.no_grad()
def greedy_neighbor_order(values: Tensor, positions: Tensor, *, bits: int = 10,
                          metric: str = "l1") -> MortonOrder:
    """Exact greedy tour, O(N²) memory. One CPU transfer, no per-token GPU sync.

    Start at the smallest Morton key; distance ties choose the original slot
    with the smallest index. This is an experimental encoder cost, not a fast
    production sorting algorithm.
    """
    if values.ndim != 3 or values.shape[:2] != positions.shape[:2] or values.shape[1] < 1:
        raise ValueError("values and positions must share a nonempty [B,N] shape")
    if metric not in ("l1", "l2") or not torch.isfinite(values).all():
        raise ValueError("greedy order requires finite values and metric l1/l2")
    codes = morton_codes_3d(positions.detach().float(), bits)
    starts = codes.argmin(1).cpu().tolist()
    rows = []
    # CPU distance matrices avoid N CUDA synchronizations in the greedy loop.
    cpu_values = values.detach().to(device="cpu", dtype=torch.float32)
    for scene, start in zip(cpu_values, starts):
        distance = torch.cdist(scene, scene, p=1 if metric == "l1" else 2,
                               compute_mode="donot_use_mm_for_euclid_dist").numpy()
        n = len(scene)
        order = np.empty(n, dtype=np.int64)
        current = start
        for i in range(n):
            order[i] = current
            distance[:, current] = np.inf
            if i + 1 < n:
                current = int(distance[current].argmin())
        rows.append(torch.from_numpy(order))
    permutation = torch.stack(rows).to(values.device)
    return MortonOrder(permutation, invert_permutation(permutation), codes)

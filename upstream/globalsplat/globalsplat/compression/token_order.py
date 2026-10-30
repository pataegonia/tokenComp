"""Encoder-only token orders; score and MSH always share the permutation."""
from __future__ import annotations

from functools import lru_cache

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


def _greedy_column_order(distance: np.ndarray, start: int) -> np.ndarray:
    """Original reference loop; consumes its distance matrix."""
    n = len(distance)
    order = np.empty(n, dtype=np.int64)
    current = start
    for i in range(n):
        order[i] = current
        distance[:, current] = np.inf
        if i + 1 < n:
            current = int(distance[current].argmin())
    return order


def _greedy_row_order(distance: np.ndarray, start: int) -> np.ndarray:
    """Exact dependency-free alternative that never writes whole columns."""
    n = len(distance)
    order = np.empty(n, dtype=np.int64)
    visited = np.zeros(n, dtype=np.bool_)
    row = np.empty(n, dtype=distance.dtype)
    current = start
    for i in range(n):
        order[i] = current
        visited[current] = True
        if i + 1 < n:
            np.copyto(row, distance[current])
            row[visited] = np.inf
            current = int(row.argmin())
    return order


def _greedy_knn_order(distance, neighbors, neighbor_distances, start):
    """Numba-compatible exact loop, with exhaustion and boundary-tie fallback.

    topk need not choose the smallest slot at a tied shortlist boundary. A
    full-row scan at that boundary preserves the reference's index tie rule.
    Return fallback counts too, for the standalone profiler.
    """
    n, k = neighbors.shape
    order = np.empty(n, dtype=np.int64)
    visited = np.zeros(n, dtype=np.bool_)
    current = start
    exhausted = 0
    boundary_ties = 0
    for step in range(n):
        order[step] = current
        visited[current] = True
        if step + 1 == n:
            break
        best = -1
        best_distance = np.inf
        for candidate in range(k):
            index = neighbors[current, candidate]
            value = neighbor_distances[current, candidate]
            if not visited[index] and (value < best_distance or
                    (value == best_distance and (best < 0 or index < best))):
                best = index
                best_distance = value
        needs_scan = best < 0
        if needs_scan:
            exhausted += 1
        elif k < n - 1 and best_distance == neighbor_distances[current, k - 1]:
            boundary_ties += 1
            needs_scan = True
        if needs_scan:
            best = -1
            best_distance = np.inf
            # Ascending slots + strict '<' implement the original argmin tie.
            for index in range(n):
                value = distance[current, index]
                if not visited[index] and (best < 0 or value < best_distance):
                    best = index
                    best_distance = value
        current = best
    return order, exhausted, boundary_ties


@lru_cache(maxsize=1)
def _compiled_greedy_knn():
    try:
        from numba import njit
    except ImportError as error:
        raise RuntimeError("nn_order_backend=knn requires numba; install with 'pip install numba', "
                           "or select nn_order_backend=row") from error
    # No fastmath: comparisons must retain the supplied matrix's tie behavior.
    return njit(cache=True, nogil=True)(_greedy_knn_order)


def _nearest_candidates(distance: Tensor, candidates: int) -> tuple[Tensor, Tensor]:
    """Sorted exact kNN distances/slots, excluding self; no giant stable sort."""
    distance.fill_diagonal_(float("inf"))
    k = min(candidates, distance.shape[0] - 1)
    values, indices = torch.topk(distance, k, dim=1, largest=False, sorted=True)
    return indices, values


def _greedy_graph_order(neighbors, neighbor_distances, curve, curve_rank, start):
    """Greedy walk on a kNN graph; exhausted lists jump along a space-filling curve.

    Unlike the exact backends there is no full-row search: when every candidate
    is visited, the next unvisited token after the current one on ``curve`` is
    taken. Tied candidates keep the reference rule (smaller slot first). Returns
    the order and the number of curve jumps.
    """
    n, k = neighbors.shape
    order = np.empty(n, dtype=np.int64)
    visited = np.zeros(n, dtype=np.bool_)
    current = start
    jumps = 0
    for step in range(n):
        order[step] = current
        visited[current] = True
        if step + 1 == n:
            break
        best = -1
        best_distance = np.inf
        for candidate in range(k):
            index = neighbors[current, candidate]
            value = neighbor_distances[current, candidate]
            if not visited[index] and (value < best_distance or
                    (value == best_distance and (best < 0 or index < best))):
                best = index
                best_distance = value
        if best < 0:
            jumps += 1
            position = curve_rank[current]
            for offset in range(1, n):
                index = curve[(position + offset) % n]
                if not visited[index]:
                    best = index
                    break
        current = best
    return order, jumps


@lru_cache(maxsize=1)
def _compiled_greedy_graph():
    try:
        from numba import njit
    except ImportError as error:
        raise RuntimeError("nn_order_backend=graph requires numba; install with 'pip install numba'") from error
    return njit(cache=True, nogil=True)(_greedy_graph_order)


def _graph_candidates(scene: Tensor, metric: str, candidates: int,
                      chunk: int = 512, shortlist: int = 0) -> tuple[Tensor, Tensor]:
    """Sorted kNN slots/distances computed in row blocks; no N×N matrix is kept.

    Metric l2 ranks by squared Euclidean distance from one GEMM per block.
    With ``shortlist`` > 0 and metric l1, each row first keeps its ``shortlist``
    nearest tokens by that squared L2, then reranks only those by the exact L1
    distance. The L1 kNN lists are then approximate.
    """
    points = len(scene)
    k = min(candidates, points - 1)
    wide = min(max(shortlist, k), points - 1)
    prefilter = shortlist > 0 and metric == "l1"
    norms = scene.square().sum(1) if prefilter or metric == "l2" else None
    indices, values = [], []
    for start in range(0, points, chunk):
        stop = min(start + chunk, points)
        rows = torch.arange(stop - start, device=scene.device)
        if prefilter or metric == "l2":
            block = norms[start:stop, None] + norms[None] - 2 * scene[start:stop] @ scene.T
            block[rows, rows + start] = float("inf")
        if prefilter:
            near = torch.topk(block, wide, dim=1, largest=False, sorted=False).indices
            exact = (scene[start:stop, None, :] - scene[near]).abs().sum(-1)
            value, slot = torch.topk(exact, k, dim=1, largest=False, sorted=True)
            index = near.gather(1, slot)
        elif metric == "l2":
            value, index = torch.topk(block, k, dim=1, largest=False, sorted=True)
        else:
            block = torch.cdist(scene[start:stop], scene, p=1,
                                compute_mode="donot_use_mm_for_euclid_dist")
            block[rows, rows + start] = float("inf")
            value, index = torch.topk(block, k, dim=1, largest=False, sorted=True)
        indices.append(index)
        values.append(value)
    return torch.cat(indices), torch.cat(values)


def graph_order_scene(scene: Tensor, curve: Tensor, start: int, *, metric: str,
                      candidates: int, shortlist: int = 0) -> tuple[np.ndarray, int]:
    """One scene of the graph backends; also returns the curve-jump count."""
    curve = curve.cpu().numpy()
    curve_rank = np.empty_like(curve)
    curve_rank[curve] = np.arange(len(curve))
    neighbors, distances = _graph_candidates(scene, metric, candidates, shortlist=shortlist)
    # One transfer of the N×k lists per scene, not of an N×N matrix.
    return _compiled_greedy_graph()(neighbors.cpu().numpy(), distances.cpu().numpy(),
                                    curve, curve_rank, start)


@torch.no_grad()
def greedy_neighbor_order(values: Tensor, positions: Tensor, *, bits: int = 10,
                          metric: str = "l1", backend: str = "legacy",
                          candidates: int = 64, distance_device: str = "cpu") -> MortonOrder:
    """Greedy nearest-neighbor tour for the computed FP32 distances.

    legacy: original CPU column mask. row: CPU visited-vector loop. knn: exact
    top-k shortlist + compiled CPU loop + exact full-row fallbacks. These three
    are exact and keep an O(N²) matrix. graph: approximate; keeps only the
    row-blocked top-k lists and jumps along the Hilbert curve of ``positions``
    when the current list is exhausted; its L2 lists come from one GEMM per
    row block. graph_gemm: graph whose L1 lists come from a 2k-wide squared-L2
    GEMM shortlist reranked by exact L1. All but
    legacy may compute distances on the input device, with no per-token CUDA
    sync. The metric is unchanged (weighted L1 Score; L2 XYZ). Different
    CPU/GPU floating-point reductions can change very close neighbor
    comparisons.
    """
    if (values.ndim != 3 or positions.ndim != 3 or positions.shape[-1] != 3 or
            values.shape[:2] != positions.shape[:2] or min(values.shape) < 1):
        raise ValueError("values and positions must share a nonempty [B,N] shape")
    if metric not in ("l1", "l2") or not torch.isfinite(values).all() or not torch.isfinite(positions).all():
        raise ValueError("greedy order requires finite values/positions and metric l1/l2")
    if backend not in ("legacy", "row", "knn", "graph", "graph_gemm"):
        raise ValueError("NN backend must be legacy, row, knn, graph or graph_gemm")
    if type(candidates) is not int or candidates < 1:
        raise ValueError("NN candidates must be a positive integer")
    if distance_device not in ("cpu", "input") or (backend == "legacy" and distance_device != "cpu"):
        raise ValueError("NN distance_device must be cpu/input; legacy requires cpu")
    compiled = _compiled_greedy_knn() if backend == "knn" else None
    codes = morton_codes_3d(positions.detach().float(), bits)
    starts = codes.argmin(1).cpu().tolist()
    graph = backend in ("graph", "graph_gemm")
    curves = hilbert_order_3d(positions.detach().float(), bits).permutation if graph else None
    rows = []
    device = "cpu" if distance_device == "cpu" else values.device
    with torch.autocast(device_type=values.device.type, enabled=False):
        scenes = values.detach().to(device=device, dtype=torch.float32)
        for index, (scene, start) in enumerate(zip(scenes, starts)):
            if len(scene) == 1:
                rows.append(torch.zeros(1, dtype=torch.int64))
                continue
            if graph:
                order, _ = graph_order_scene(scene, curves[index], start, metric=metric, candidates=candidates,
                                             shortlist=2 * candidates if backend == "graph_gemm" else 0)
                rows.append(torch.from_numpy(order))
                continue
            # Direct L2 also avoids silently introducing a GEMM distance metric.
            distance = torch.cdist(scene, scene, p=1 if metric == "l1" else 2,
                                   compute_mode="donot_use_mm_for_euclid_dist")
            if backend == "knn":
                neighbors, neighbor_distances = _nearest_candidates(distance, candidates)
                # Transfer once per array, not once per visited token. The full
                # matrix remains on CPU so every fallback is exact and local.
                order, _, _ = compiled(distance.cpu().numpy(), neighbors.cpu().numpy(),
                                       neighbor_distances.cpu().numpy(), start)
            else:
                matrix = distance.cpu().numpy()
                order = (_greedy_column_order if backend == "legacy" else _greedy_row_order)(matrix, start)
            rows.append(torch.from_numpy(order))
    permutation = torch.stack(rows).to(values.device)
    return MortonOrder(permutation, invert_permutation(permutation), codes)

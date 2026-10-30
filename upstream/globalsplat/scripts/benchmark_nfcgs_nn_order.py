#!/usr/bin/env python3
"""Synthetic Score NN-sort timings and exactness checks, excluding JIT warmup.

This measures sorting, not an end-to-end GlobalSplat training step. CUDA
phase timings synchronize the device and include CPU transfers explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globalsplat.compression.token_order import (  # noqa: E402
    _compiled_greedy_graph, _compiled_greedy_knn, _graph_candidates, _greedy_row_order,
    _nearest_candidates, greedy_neighbor_order, hilbert_order_3d,
)


def shared_edges(first, second):
    """Fraction of consecutive pairs of ``first`` that are also consecutive in ``second``."""
    def edges(order):
        return set(map(tuple, torch.stack((order[:-1], order[1:]), 1).sort(1).values.tolist()))
    reference = edges(first)
    return len(reference & edges(second)) / len(reference)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(call, device):
    synchronize(device)
    start = perf_counter()
    value = call()
    synchronize(device)
    return value, (perf_counter() - start) * 1000


@torch.no_grad()
def profile_knn(scene, metric, candidates, start):
    device = scene.device
    matrix, distance_ms = timed(lambda: torch.cdist(scene, scene,
        p=1 if metric == "l1" else 2, compute_mode="donot_use_mm_for_euclid_dist"), device)
    (neighbors, distances), topk_ms = timed(lambda: _nearest_candidates(matrix, candidates), device)
    (cpu_matrix, cpu_neighbors, cpu_distances), transfer_ms = timed(
        lambda: (matrix.cpu().numpy(), neighbors.cpu().numpy(), distances.cpu().numpy()), device)
    (order, exhausted, ties), greedy_ms = timed(
        lambda: _compiled_greedy_knn()(cpu_matrix, cpu_neighbors, cpu_distances, start), device)
    exact = np.array_equal(order, _greedy_row_order(cpu_matrix, start))
    if not exact:
        raise AssertionError("knn differs from full-row greedy on the SAME distance matrix")
    return dict(distance_ms=distance_ms, topk_ms=topk_ms, cpu_transfer_ms=transfer_ms,
                greedy_ms=greedy_ms, phase_sum_ms=distance_ms + topk_ms + transfer_ms + greedy_ms,
                exhausted_fallbacks=int(exhausted), boundary_tie_fallbacks=int(ties),
                same_matrix_order_equal=exact)


@torch.no_grad()
def profile_graph(scene, metric, candidates, start, curve, shortlist=0):
    device = scene.device
    (neighbors, distances), candidates_ms = timed(
        lambda: _graph_candidates(scene, metric, candidates, shortlist=shortlist), device)
    (cpu_neighbors, cpu_distances), transfer_ms = timed(
        lambda: (neighbors.cpu().numpy(), distances.cpu().numpy()), device)
    rank = np.empty_like(curve)
    rank[curve] = np.arange(len(curve))
    (_, jumps), greedy_ms = timed(
        lambda: _compiled_greedy_graph()(cpu_neighbors, cpu_distances, curve, rank, start), device)
    return dict(blocked_distance_topk_ms=candidates_ms, cpu_transfer_ms=transfer_ms, greedy_ms=greedy_ms,
                phase_sum_ms=candidates_ms + transfer_ms + greedy_ms, curve_jumps=int(jumps))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=111123)
    parser.add_argument("--output", type=Path, default=Path("outputs/nn_order_benchmark.json"))
    args = parser.parse_args(argv)
    for field in ("tokens", "batch_size", "candidates", "repeats", "warmup", "threads"):
        if getattr(args, field) < (2 if field == "tokens" else 1):
            parser.error(f"--{field.replace('_', '-')} is too small")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    compiled = _compiled_greedy_knn()
    # Compile outside all measurements, on small arrays of the real signatures.
    warm = torch.tensor([[0.], [1.], [2.]])
    distance = torch.cdist(warm, warm)
    neighbors, distances = _nearest_candidates(distance, 2)
    compiled(distance.numpy(), neighbors.numpy(), distances.numpy(), 0)
    curve = np.arange(3, dtype=np.int64)
    _compiled_greedy_graph()(neighbors.numpy(), distances.numpy(), curve, curve.copy(), 0)
    import numba
    report = dict(scope="Synthetic sorting only; not full training or actual scene RD results",
                  torch=torch.__version__, numpy=np.__version__, numba=numba.__version__,
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  threads=args.threads, tokens=args.tokens, batch_size=args.batch_size,
                  candidates=args.candidates, repeats=args.repeats, warmup=args.warmup,
                  seed=args.seed, matrix_mib_per_scene=args.tokens ** 2 * 4 / 2 ** 20,
                  cases=[])
    generator = torch.Generator().manual_seed(args.seed)
    # The active distance-based ordering experiment is Score only, under L1 or L2.
    for name, width, metric in (("nn_score", 56, "l1"), ("nn_score_l2", 56, "l2")):
        values = torch.randn(args.batch_size, args.tokens, width, generator=generator)
        positions = torch.randn(args.batch_size, args.tokens, 3, generator=generator)
        values /= values.abs().mean(dim=1, keepdim=True).clamp_min(.05)
        values, positions = values.to(device), positions.to(device)
        methods = [("legacy_cpu", "legacy", "cpu"), ("row_cpu", "row", "cpu"),
                   ("knn_cpu", "knn", "cpu"), ("graph_cpu", "graph", "cpu"),
                   ("graph_gemm_cpu", "graph_gemm", "cpu")]
        if device.type == "cuda":
            methods += [("row_input", "row", "input"), ("knn_input", "knn", "input"),
                        ("graph_input", "graph", "input"), ("graph_gemm_input", "graph_gemm", "input")]
        case = dict(order=name, dimensions=width, metric=metric, timings=[])
        reference = None
        baseline_ms = None
        for label, backend, distance_device in methods:
            call = lambda: greedy_neighbor_order(values, positions, metric=metric,
                backend=backend, candidates=args.candidates, distance_device=distance_device)
            for _ in range(args.warmup):
                call()
            samples = []
            for _ in range(args.repeats):
                order, milliseconds = timed(call, device)
                samples.append(milliseconds)
            if reference is None:
                reference = order.permutation.cpu()
                baseline_ms = median(samples)
            permutation = order.permutation.cpu()
            for row in permutation:
                if not torch.equal(row.sort().values, torch.arange(args.tokens)):
                    raise AssertionError("tour did not retain every token exactly once")
            equal = torch.equal(permutation, reference)
            # graph backends are approximate by design; the exact ones must agree on CPU.
            if distance_device == "cpu" and not backend.startswith("graph") and not equal:
                raise AssertionError("CPU optimized tour differs from original CPU tour")
            row = dict(backend=label, median_batch_ms=median(samples),
                       median_ms_per_scene=median(samples) / args.batch_size,
                       samples_batch_ms=samples, speedup_vs_legacy_cpu=baseline_ms / median(samples),
                       order_equal_to_legacy_cpu=equal,
                       shared_edges_vs_legacy_cpu=median(shared_edges(a, b) for a, b in zip(reference, permutation)))
            case["timings"].append(row)
            print(json.dumps(dict(order=name, **row)), flush=True)
        start = int(reference[0, 0])
        phases = []
        for _ in range(args.repeats):
            phases.append(profile_knn(values[0], metric, args.candidates, start))
        case["knn_phases_first_scene"] = {key: median([row[key] for row in phases]) for key in phases[0]
                                           if key != "same_matrix_order_equal"}
        case["knn_phases_first_scene"]["same_matrix_order_equal"] = all(row["same_matrix_order_equal"] for row in phases)
        curve = hilbert_order_3d(positions[:1], 10).permutation[0].cpu().numpy()
        for label, shortlist in (("graph", 0), ("graph_gemm", 2 * args.candidates)):
            graph = [profile_graph(values[0], metric, args.candidates, start, curve, shortlist)
                     for _ in range(args.repeats)]
            case[f"{label}_phases_first_scene"] = {key: median([row[key] for row in graph]) for key in graph[0]}
        report["cases"].append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"REPORT={args.output.resolve()}")


if __name__ == "__main__":
    main()

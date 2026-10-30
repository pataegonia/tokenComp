#!/usr/bin/env python3
"""Adjacent-token differences of codec inputs under several token orders.

Inference-only analysis on the matched 256-scene context cache: no entropy
coding, rendering, targets or dataset access. For every order, consecutive
tokens are compared in XYZ, in the 736-D codec input F (512 appearance + 224
projected geometry) and in the 56-D Score before quantization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import statistics
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval_nfcgs_path_diagnostics as diag
from globalsplat.compression.morton import morton_codes_3d, morton_order_3d
from globalsplat.compression.token_order import graph_order_scene, greedy_neighbor_order, hilbert_order_3d

ORDERS = ("slot", "random", "morton", "hilbert", "nn_xyz", "nn_xyz_graph",
          "nn_score", "nn_score_graph", "nn_score_gemm", "nn_score_l2")
GRAPH_PAIRS = (("nn_xyz", "nn_xyz_graph"), ("nn_score", "nn_score_graph"), ("nn_score", "nn_score_gemm"),
               ("nn_score", "nn_score_l2"))
SIGNALS = ("xyz", "features", "texture", "geometry", "score")


def load_cache(args):
    """Original context cache, verified by fingerprint; precision-agnostic."""
    source = args.input_run.resolve()
    for name in ("inputs.pt", "inputs.json", "backbone.json"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"required original source artifact: {source / name}; do not resample")
    listed = diag.read_json(source / "inputs.json")
    cache = torch.load(source / "inputs.pt", map_location="cpu", weights_only=False)
    if ([item["scene"] for item in cache] != listed["scenes"]
            or diag.profile.input_fingerprint(cache) != listed["fingerprint"]
            or listed["fingerprint"] != args.expected_input_fingerprint):
        raise ValueError("context cache differs from the requested matched scenes")
    return cache, dict(path=str(source), fingerprint=listed["fingerprint"], scenes=len(cache),
                       backbone_sha256=diag.read_json(source / "backbone.json")["sha256"])


def random_permutations(points, scene, draws, device):
    result = []
    for i in range(draws):
        seed = int.from_bytes(hashlib.sha256(f"{i}:{scene}".encode()).digest()[:8], "little") % (2**63)
        result.append(torch.randperm(points, generator=torch.Generator().manual_seed(seed)).to(device))
    return result


def shared_edges(first, second):
    """Fraction of consecutive pairs of ``first`` that are also consecutive in ``second``."""
    def edges(order):
        pairs = torch.stack((order[:-1], order[1:]), 1).sort(1).values.tolist()
        return set(map(tuple, pairs))
    reference = edges(first)
    return len(reference & edges(second)) / max(len(reference), 1)


def token_orders(codec, xyz, units, scene, args):
    """Permutations [N] per order, sort time and exact-vs-graph NN statistics.

    nn_score repeats the codec's own score tour; nn_xyz is the L2 tour on the
    same sort coordinates. *_graph use the approximate kNN-graph backend,
    nn_score_gemm its L2-GEMM shortlist variant and nn_score_l2 the graph tour
    under the Euclidean instead of the L1 Score distance.
    """
    bits = codec.config.morton_bits
    scale = units.abs().mean(dim=1, keepdim=True).clamp_min(.05)
    start = int(morton_codes_3d(xyz, bits)[0].argmin())
    builders = dict(
        slot=lambda: [torch.arange(xyz.shape[1], device=xyz.device)],
        random=lambda: random_permutations(xyz.shape[1], scene, args.random_draws, xyz.device),
        morton=lambda: [morton_order_3d(xyz, bits).permutation[0]],
        hilbert=lambda: [hilbert_order_3d(xyz, bits).permutation[0]],
        nn_xyz=lambda: [greedy_neighbor_order(xyz, xyz, bits=bits, metric="l2",
                                              backend=args.nn_backend).permutation[0]],
        nn_score=lambda: [greedy_neighbor_order(units / scale, xyz, bits=bits, metric="l1",
                                                backend=args.nn_backend).permutation[0]])
    orders, stats, jumps = {}, {}, {}
    for name in ORDERS:
        begin = time.perf_counter()
        if name.endswith(("_graph", "_gemm", "_l2")):
            values, metric = (xyz[0], "l2") if name == "nn_xyz_graph" else (
                (units / scale)[0], "l2" if name == "nn_score_l2" else "l1")
            curve = hilbert_order_3d(xyz, bits).permutation[0]
            shortlist = 2 * args.graph_candidates if name.endswith("_gemm") else 0
            order, jumps[name] = graph_order_scene(values.float(), curve, start, metric=metric,
                                                   candidates=args.graph_candidates, shortlist=shortlist)
            orders[name] = [torch.from_numpy(order).to(xyz.device)]
        else:
            orders[name] = builders[name]()
        stats[name] = dict(ms=(time.perf_counter() - begin) * 1000 / len(orders[name]))
    for exact, graph in GRAPH_PAIRS:
        stats[graph].update(jumps=jumps[graph], shared_edges=shared_edges(orders[exact][0], orders[graph][0]))
    return orders, stats


def adjacent(values, permutation, scale, *, euclidean=False):
    """Mean step between consecutive tokens, raw and divided by a scene scale."""
    ordered = values[permutation].float()
    step = ordered[1:] - ordered[:-1]
    if euclidean:
        distance = step.norm(dim=-1)
        return dict(l1=float(distance.mean()), normalized_l1=float((distance / scale).mean()))
    step = step.abs()
    return dict(l1=float(step.mean()), normalized_l1=float((step / scale).mean()))


@torch.no_grad()
def analyze_scene(model, inputs, args, scene):
    codec = model.feature_codec
    texture, geometry = model.encode_scene_tokens(inputs)
    if texture.shape[:2] != (1, args.tokens):
        raise ValueError("unexpected scene token count")
    features = codec._pack_features(texture, geometry).float()
    xyz = model.gaussian_decoder.decoded_token_centers(features[..., codec.config.texture_channels:]).float()
    mean = codec._scene_mean(features)
    score = codec._analyze_low_rank(features - mean[:, None, :]) / codec.score_scale[None, None, :]
    _, step = codec.score_context._scene_parameters(mean)
    units = score / step[:, None, :]  # quantization-step units of the coded Score
    width = codec.config.texture_channels
    signals = dict(xyz=xyz[0], features=features[0], texture=features[0, :, :width],
                   geometry=features[0, :, width:], score=units[0])
    scales = {name: value.std(0, correction=0).clamp_min(1e-6) for name, value in signals.items() if name != "xyz"}
    scales["xyz"] = (xyz[0] - xyz[0].mean(0)).square().sum(-1).mean().sqrt().clamp_min(1e-6)
    orders, order_stats = token_orders(codec, xyz, units, scene, args)
    result = dict(scene=scene, orders={}, order_stats=order_stats)
    for name, permutations in orders.items():
        for permutation in permutations:
            if not torch.equal(permutation.sort().values, torch.arange(len(permutation), device=permutation.device)):
                raise ValueError(f"{name} is not a bijective token permutation")
        result["orders"][name] = {
            signal: {key: statistics.fmean(adjacent(values, p, scales[signal], euclidean=signal == "xyz")[key]
                                           for p in permutations) for key in ("l1", "normalized_l1")}
            for signal, values in signals.items()}
    return result


def summarize(rows):
    summary = {}
    for order in ORDERS:
        summary[order] = {}
        for signal in SIGNALS:
            values = {key: statistics.fmean(row["orders"][order][signal][key] for row in rows)
                      for key in ("l1", "normalized_l1")}
            reference = statistics.fmean(row["orders"]["random"][signal]["normalized_l1"] for row in rows)
            values["vs_random"] = values["normalized_l1"] / reference
            summary[order][signal] = values
    order_stats = {order: {key: statistics.fmean(row["order_stats"][order][key] for row in rows)
                           for key in rows[0]["order_stats"][order]} for order in ORDERS}
    return dict(scene_count=len(rows), orders=summary, order_stats=order_stats,
                definition="mean over scenes of the mean |x[i+1]-x[i]| over consecutive tokens and channels "
                           "(XYZ: Euclidean step); normalized_l1 divides each channel by its per-scene std "
                           "(XYZ: per-scene RMS radius); random is the mean of the random draws.")


def write_report(output, summary, spec, backbone, args):
    lines = ["# Adjacent-token differences by token order", "",
             f"Checkpoint: `{spec['label']}`; lambda={spec['rate_lambda']}; step {spec['expected_step']}; "
             f"{summary['scene_count']} matched scenes; device {args.device}, FP32.", "",
             "Normalized mean step between consecutive tokens (lower = smoother sequence); "
             "parentheses: ratio to random order.", "",
             "| Order | XYZ | F736 | Appearance 512 | Geometry 224 | Score 56 |",
             "|---|---:|---:|---:|---:|---:|"]
    for order, row in summary["orders"].items():
        cells = [f"{row[s]['normalized_l1']:.4f} ({row[s]['vs_random']:.3f})" for s in SIGNALS]
        lines.append(f"| {order} | " + " | ".join(cells) + " |")
    lines += ["", "Raw Score step is in quantization-step units: "
              + ", ".join(f"{o} {summary['orders'][o]['score']['l1']:.4f}" for o in ORDERS) + ".", "",
              f"NN tours: exact (`{args.nn_backend}`) vs kNN-graph (k={args.graph_candidates}, Hilbert jump; "
              f"gemm = {2 * args.graph_candidates}-wide L2 GEMM shortlist reranked by L1; "
              "nn_score_l2 = graph tour under the Euclidean Score distance).", "",
              "| Approximate tour | Exact sort ms | Approx sort ms | Curve jumps | Shared consecutive pairs |",
              "|---|---:|---:|---:|---:|"]
    stats = summary["order_stats"]
    for exact, graph in GRAPH_PAIRS:
        lines.append(f"| {graph} | {stats[exact]['ms']:.1f} | {stats[graph]['ms']:.1f} | "
                     f"{stats[graph]['jumps']:.1f} | {stats[graph]['shared_edges']:.3f} |")
    lines += ["", "Sort times are CPU wall-clock per scene in this job (FP32, threads as configured); "
              "they are not GPU training costs.", "",
              summary["definition"],
              "Orders are computed from this checkpoint's FP32 positions; a few near-tie tokens can differ from "
              "the BF16 production Hilbert order. nn_score is the codec's greedy Score tour applied to this "
              "Hilbert-trained model, not a separately trained NN-score codec.",
              f"Backbone check: {backbone['kind']}.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args, manifest, spec):
    if args.output.exists():
        raise FileExistsError("use a fresh output directory; existing analyses are preserved")
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    cache, source = load_cache(args)
    checkpoint = Path(spec["checkpoint"])
    model, _, _ = diag.profile.load_model(spec, args)
    backbone = diag.verify_backbone(model, spec, checkpoint, source)
    selected = cache[:args.limit] if args.limit else cache
    args.output.mkdir(parents=True)
    diag.save_json(args.output / "metadata.json", dict(
        version=1, started_utc=datetime.now(timezone.utc).isoformat(), manifest=manifest,
        checkpoint=dict(path=str(checkpoint.resolve()), sha256=diag.file_hash(checkpoint), global_step=spec["expected_step"]),
        backbone=backbone, source=source, scenes=len(selected), device=args.device, precision="fp32",
        random_draws=args.random_draws, nn_backend=args.nn_backend, graph_candidates=args.graph_candidates,
        code_sha256={name: diag.file_hash(REPO / name) for name in (
            "scripts/analyze_nfcgs_token_adjacency.py", "scripts/eval_nfcgs_path_diagnostics.py",
            "globalsplat/compression/token_order.py", "globalsplat/compression/morton.py",
            "globalsplat/compression/codec.py")}))
    rows = []
    for index, item in enumerate(selected):
        row = analyze_scene(model, diag.profile.to_device(item["inputs"], args.device), args, item["scene"])
        rows.append(row)
        diag.save_json(args.output / "scenes" / f"scene_{index:04d}.json", row)
        orders = row["orders"]
        print(f"TOKEN_ADJACENCY_SCENE={index + 1}/{len(selected)} scene={item['scene']} score_step "
              f"slot={orders['slot']['score']['normalized_l1']:.4f} hilbert={orders['hilbert']['score']['normalized_l1']:.4f}",
              flush=True)
    summary = summarize(rows)
    diag.save_json(args.output / "summary.json", summary)
    write_report(args.output, summary, spec, backbone, args)
    diag.save_json(args.output / "complete.json", dict(scene_count=len(rows), checkpoint_sha256=diag.file_hash(checkpoint)))
    print(f"TOKEN_ADJACENCY_COMPLETE scenes={len(rows)} output={args.output}", flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "config/plans/nfcgs_path_diagnostics_fresh120k_h_on_lambda0p0128.json")
    parser.add_argument("--input-run", type=Path, default=REPO / "outputs/nfcgs_legacy3_anchors/job_454995")
    parser.add_argument("--expected-input-fingerprint", default="069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="analyze only the first N cached scenes (0 = all)")
    parser.add_argument("--random-draws", type=int, default=10)
    parser.add_argument("--nn-backend", choices=("legacy", "row", "knn"), default="legacy",
                        help="exact backend for nn_xyz/nn_score")
    parser.add_argument("--graph-candidates", type=int, default=64)
    args = parser.parse_args(argv)
    if min(args.threads, args.tokens, args.random_draws, args.graph_candidates) < 1 or args.limit < 0:
        parser.error("threads/tokens/random draws/graph candidates must be positive and limit nonnegative")
    # Only used to compose the shared model config; no dataset is read.
    args.dataset_root, args.num_workers, args.seed = Path("/data3/local_datasets/re10k"), 0, 0
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest, spec = diag.load_spec(args.manifest)
    if args.tokens != spec["tokens"]:
        raise ValueError("requested token count differs from the checkpoint manifest")
    run(args, manifest, spec)


if __name__ == "__main__":
    main()

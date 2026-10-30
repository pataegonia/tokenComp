#!/usr/bin/env python3
"""Profile current sorting/context codecs without training or changing checkpoints.

Static mode uses initialized codecs and synthetic inputs for parameter counts and
shape-based neural MACs. Actual mode uses final checkpoints and matched scenes on
one device. Sorting, transfers, rANS, indexing and elementwise work are measured
by latency, not mislabeled as neural MACs.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
from time import perf_counter

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec  # noqa: E402
from globalsplat.compression.bitstream import SceneBitstream  # noqa: E402

MAC_SCOPE = ("Registered PyTorch matrix/convolution FLOPs divided by 2; one multiply-add=1 MAC. "
             "Excludes sorting comparisons, cdist distances, indexing, transfers, rANS, "
             "elementwise operations, CDF construction, GlobalSplat backbone and rendering.")


def config_for(spec):
    return CodecConfig.from_mapping(spec["config"])


def parameter_footprint(codec):
    """Count stored and structurally used parameters, independently of requires_grad.

    Compatibility tensors and bypassed predictors remain in checkpoint state.
    Probability-network parameters are counted even when deployed through CDFs.
    """
    context = codec.score_context
    groups = {
        "geometry_projection": [codec.geometry_projection],
        "residual_msh": [codec.residual_codec],
        "score_probability": list(context.active_entropies()),
        "context_predictors": [],
    }
    if context.spatial_context:
        if context.context_schedule != "legacy":
            groups["context_predictors"].append(context.anchor_predictors)
        else:
            groups["context_predictors"].append(context.spatial_predictors)
            for count, name in ((3, "split_even_predictors"), (4, "split_odd_predictors")):
                if context.spatial_stages >= count:
                    groups["context_predictors"].append(getattr(context, name))
    direct = [codec.shared_basis, codec.shared_synthesis_basis]
    active = {id(p): p for p in direct}
    counts = {"low_rank_bases": sum(p.numel() for p in direct)}
    for name, modules in groups.items():
        unique = {id(p): p for module in modules for p in module.parameters()}
        counts[name] = sum(p.numel() for p in unique.values())
        active.update(unique)
    total = sum(p.numel() for p in codec.parameters())
    used = sum(p.numel() for p in active.values())
    return dict(stored_parameters=total, active_architecture_parameters=used,
        inactive_compatibility_parameters=total-used,
        stored_parameter_bytes=sum(p.numel()*p.element_size() for p in codec.parameters()),
        active_parameter_bytes=sum(p.numel()*p.element_size() for p in active.values()),
        groups=counts, active_definition="Structurally used branches plus entropy-model parameters; not a runtime weight-access count.")


def covered_macs(call):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as e:
        raise RuntimeError("This PyTorch lacks torch.utils.flop_counter.FlopCounterMode") from e
    with FlopCounterMode(display=False) as counter:
        result = call()
    ops = {str(key): int(value) for key, value in counter.get_flop_counts().get("Global", {}).items()}
    flops = int(counter.get_total_flops())
    if flops % 2:
        raise RuntimeError("Registered FLOPs are not an even multiply-add count")
    return result, dict(covered_flops=flops, neural_macs=flops//2, counted_operators=ops, scope=MAC_SCOPE)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def measured(call, device):
    sync(device)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = None
    start = perf_counter()
    result = call()
    sync(device)
    milliseconds = (perf_counter()-start)*1000
    peak = torch.cuda.max_memory_allocated(device) if cuda else None
    return result, dict(milliseconds=milliseconds, cuda_baseline_bytes=baseline,
        cuda_peak_bytes=peak, cuda_peak_above_baseline_bytes=max(0, peak-baseline) if cuda else None)


def prepare_score(codec, texture, geometry, positions):
    features = codec._pack_features(texture, geometry)
    ordered, order = codec._ordered_features(features, positions)
    # FP32, as the retired unit Score scale used to promote it.
    score = codec._analyze_low_rank(ordered).float()
    return features, order, ordered, score.transpose(1, 2).unsqueeze(2)


def encode_score(codec, score):
    with torch.autocast(device_type=score.device.type, enabled=False):
        return codec.score_context.compress(score.float())


def decode_score(codec, payload, points, device):
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        return codec.score_context.decompress(payload, points)


def residual_input(codec, ordered, score_hat):
    residual = ordered - codec._synthesize_low_rank(score_hat.squeeze(2).transpose(1, 2))
    return (residual.transpose(1, 2).unsqueeze(2)-codec.residual_mean) / codec.residual_std.clamp_min(1e-8)


@torch.no_grad()
def neural_profile(codec, inputs):
    texture, geometry, positions = inputs
    features, order, ordered, score = prepare_score(codec, *inputs)
    result = {}
    _, result["ordering_including_score_analysis"] = covered_macs(lambda: codec._make_order(positions, features))
    (payload, score_hat), result["score_encode"] = covered_macs(lambda: encode_score(codec, score))
    decoded, result["score_decode"] = covered_macs(lambda: decode_score(codec, payload, score.shape[-1], score.device))
    if not torch.equal(score_hat, decoded):
        raise RuntimeError("Score encode/decode mismatch")
    residual = residual_input(codec, ordered, score_hat)
    (msh_payload, _), result["msh_encode"] = covered_macs(lambda: codec.residual_codec.compress(residual))
    _, result["msh_decode"] = covered_macs(lambda: codec.residual_codec.decompress(msh_payload, output_width=texture.shape[1]))
    packed, result["codec_core_encode"] = covered_macs(lambda: codec.compress(*inputs))
    if SceneBitstream.unpack(packed.data).score != payload:
        raise RuntimeError("Isolated and full Score streams differ")
    _, result["codec_decode"] = covered_macs(lambda: codec.decompress(packed.data))
    return result


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    points = manifest.get("points", [])
    if manifest.get("version") != 1 or not points:
        raise ValueError("expected nonempty version=1 complexity manifest")
    labels = [p["label"] for p in points]
    if len(labels) != len(set(labels)) or any(not re.fullmatch(r"[a-z0-9_]+", p) for p in labels):
        raise ValueError("point labels must be unique safe identifiers")
    for point in points:
        config_for(point)
    return manifest


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


@torch.no_grad()
def static_report(manifest, args):
    if (args.output / "static.json").exists():
        raise FileExistsError("static.json already exists; use a fresh output directory")
    by_config = {}
    for spec in manifest["points"]:
        key = json.dumps(config_for(spec).to_dict(), sort_keys=True)
        by_config.setdefault(key, []).append(spec)
    results, neural_cache = [], {}
    for key, specs in by_config.items():
        config = config_for(specs[0])
        torch.manual_seed(args.seed)
        codec = ObservableLowRank1DCodec(config).eval()
        footprint = parameter_footprint(codec)
        # Morton/Hilbert/identity differ in integer sorting, not neural MACs.
        signature = replace(config, token_order="nn_score" if config.token_order == "nn_score" else "morton",
                            use_morton=True).to_dict()
        signature_key = json.dumps(signature, sort_keys=True)
        if signature_key not in neural_cache:
            codec.update(force=True, update_quantiles=False)
            inputs = tuple(torch.randn(1, args.tokens, width) for width in
                           (config.texture_channels, config.geometry_channels, 3))
            neural_cache[signature_key] = neural_profile(codec, inputs)
        row = dict(config=config.to_dict(), labels=[s["label"] for s in specs],
            family=specs[0]["family"], footprints=footprint, neural=neural_cache[signature_key])
        results.append(row)
        print(json.dumps(dict(family=row["family"], stored_params=footprint["stored_parameters"],
            active_params=footprint["active_architecture_parameters"],
            encode_gmac=row["neural"]["codec_core_encode"]["neural_macs"]/1e9,
            decode_gmac=row["neural"]["codec_decode"]["neural_macs"]/1e9)), flush=True)
        del codec
    report = dict(version=1, mode="static", torch=torch.__version__, device="cpu", tokens=args.tokens,
        batch_size=1, weights="initialized synthetic codec; NOT trained checkpoint measurements",
        mac_scope=MAC_SCOPE, point_count=len(manifest["points"]), config_count=len(results),
        static_mac_passes=len(neural_cache), results=results)
    save_new(args.output / "static.json", report)
    return report


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), min=float(a.min()), max=float(a.max()))


def summarize_samples(samples):
    stages = samples[0]["stages"]
    return {stage: {metric: distribution([row["stages"][stage][metric] for row in samples])
                    for metric in stages[stage] if stages[stage][metric] is not None}
            for stage in stages}


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(v, device) for v in value)
    return value


def input_fingerprint(cache):
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            digest.update(str((tuple(t.shape), str(t.dtype))).encode())
            digest.update(t.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for k in sorted(value):
                digest.update(str(k).encode())
                visit(value[k])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
    visit(cache)
    return digest.hexdigest()


def load_model(spec, args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from globalsplat.main import build_model
    from globalsplat.compression import resize_registered_buffers
    from globalsplat.compression.checkpoint import strip_retired_state, validate_feature_codec_checkpoint

    path = Path(spec["checkpoint"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") != spec["expected_step"]:
        raise ValueError(f"wrong checkpoint step: {path}")
    validate_feature_codec_checkpoint(checkpoint, config_for(spec))
    config = CodecConfig.from_mapping(checkpoint["feature_codec_config"])
    if config.to_dict() != config_for(spec).to_dict():
        raise ValueError(f"saved configuration differs from manifest: {spec['label']}")
    state = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    if not state or "scene_tokens" not in state:
        raise ValueError("an integrated model.* checkpoint is required")
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "config")):
        cfg = compose(config_name="main", overrides=["+experiment=re10k_32k_nfcgs", "dataset=re10k_eval_all_ctx12"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.feature_codec = OmegaConf.create(config.to_dict())
    cfg.model.dim_latents = config.geometry_channels
    cfg.model.latent_rep_token_amount = int(state["scene_tokens"].shape[0])
    cfg.model.freeze_globalsplat = True
    cfg.model.feature_codec_train_scope = "all"
    cfg.dataset.dataset_roots = [str(args.dataset_root)]
    cfg.dataset.mvsplat_root = str(REPO / "third_party/ZPressor/mvsplat")
    cfg.dataset.augment = False
    cfg.dataset.data_loader_seed = args.seed
    cfg.optimizer.batch_size = 1
    cfg.optimizer.num_workers = args.num_workers
    cfg.seed = args.seed
    model = build_model(cfg.model)
    state = strip_retired_state(state)
    resize_registered_buffers(model, state)
    model.load_state_dict(state, strict=True)
    codec = model.feature_codec
    codec.update(force=True, update_quantiles=False)
    footprint = parameter_footprint(codec)
    footprint["buffer_bytes"] = sum(b.numel()*b.element_size() for b in codec.buffers())
    footprint["backbone_and_gaussian_decoder_parameters"] = sum(
        p.numel() for name, p in model.named_parameters() if not name.startswith("feature_codec."))
    import io
    buffer = io.BytesIO()
    torch.save(codec.state_dict(), buffer)
    footprint["serialized_codec_state_dict_bytes"] = buffer.tell()
    del buffer, checkpoint, state
    model.requires_grad_(False).eval().to(args.device)
    model.set_stage(3, mix=1.0)
    return model, cfg, footprint


def cache_scenes(cfg, args):
    from globalsplat.main import build_datamodule
    data, _ = build_datamodule(cfg)
    cache, seen = [], set()
    loader = data.test_dataloader()
    for batch in loader:
        name = batch["scene_info"]["scene"]
        name = str(name[0] if isinstance(name, (tuple, list)) else name)
        if name in seen:
            continue
        seen.add(name)
        cache.append(dict(scene=name, inputs=to_device(batch["inputs"], "cpu")))
        if len(cache) == args.warmup + args.scenes:
            break
    del loader, data
    if len(cache) != args.warmup + args.scenes:
        raise ValueError("test loader exhausted before requested unique scenes")
    return cache


@torch.no_grad()
def profile_point(model, cache, args):
    codec, samples, counts = model.feature_codec, [], None
    amp = lambda: (torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext())
    for index, item in enumerate(cache):
        inputs = to_device(item["inputs"], args.device)
        with amp():
            encoded, encoder_timing = measured(lambda: model.encode_scene_tokens(inputs), args.device)
            texture, geometry = encoded
            if texture.shape[:2] != (1, args.tokens):
                raise ValueError("checkpoint/input token count does not match profile protocol")
            position_call = lambda: model.gaussian_decoder.decoded_token_centers(codec.project_geometry(geometry))
            positions, position_timing = measured(position_call, args.device)
            features, order, ordered, score = prepare_score(codec, texture, geometry, positions)
            for repeat in range(1 if index < args.warmup else args.repeats):
                stages = dict(scene_encoder=encoder_timing, token_position_prepare=position_timing)
                unused, stages["ordering_with_score_analysis"] = measured(
                    lambda: codec._make_order(positions, features), args.device)
                del unused
                unused, stages["permutation_gather"] = measured(lambda: order.apply(features), args.device)
                del unused
                (payload, score_hat), stages["score_encode"] = measured(lambda: encode_score(codec, score), args.device)
                decoded_score, stages["score_decode"] = measured(lambda: decode_score(codec, payload, args.tokens, args.device), args.device)
                if not torch.equal(score_hat, decoded_score):
                    raise RuntimeError("Score sender/receiver mismatch")
                residual = residual_input(codec, ordered, score_hat)
                (msh_payload, unused), stages["msh_encode"] = measured(lambda: codec.residual_codec.compress(residual), args.device)
                del unused
                unused, stages["msh_decode"] = measured(lambda: codec.residual_codec.decompress(msh_payload, output_width=args.tokens), args.device)
                del unused
                packed, stages["codec_core_encode"] = measured(lambda: codec.compress(texture, geometry, positions), args.device)
                pipeline, stages["pipeline_entropy_encode"] = measured(lambda: model.compress_scene_tokens(encoded), args.device)
                decoded, stages["codec_decode"] = measured(lambda: codec.decompress(pipeline.data), args.device)
                gaussians, stages["gaussian_decoder"] = measured(lambda: model.decode_scene_tokens(decoded), args.device)
                if SceneBitstream.unpack(packed.data).score != payload or packed.data != pipeline.data:
                    raise RuntimeError("isolated/core/pipeline bitstreams do not agree")
                if index >= args.warmup:
                    samples.append(dict(scene=item["scene"], repeat=repeat, total_bytes=len(packed.data), stages=stages))
                del gaussians, packed, pipeline, decoded, decoded_score, msh_payload, residual, score_hat
            if index == args.warmup:
                counts = neural_profile(codec, (texture, geometry, positions))
                unused, counts["pipeline_entropy_encode"] = covered_macs(lambda: model.compress_scene_tokens(encoded))
                del unused
        del encoded, inputs, features, order, ordered, score, positions
        if index >= args.warmup:
            print(f"SCENE {index-args.warmup+1}/{args.scenes} {item['scene']}", flush=True)
    # Each scene encoder and position preparation was measured once, not once per repeat.
    summary = summarize_samples(samples)
    for stage in ("scene_encoder", "token_position_prepare"):
        summary[stage] = {metric: distribution([row["stages"][stage][metric] for row in samples if row["repeat"] == 0])
                          for metric in samples[0]["stages"][stage] if samples[0]["stages"][stage][metric] is not None}
    return dict(timings=summary, neural=counts, samples=samples,
        payload_bytes=distribution([row["total_bytes"] for row in samples if row["repeat"] == 0]))


def environment(args):
    import compressai
    cuda = torch.device(args.device).type == "cuda"
    return dict(torch=torch.__version__, compressai=compressai.__version__, numpy=np.__version__,
        host=platform.node(), device=args.device,
        gpu=torch.cuda.get_device_name(args.device) if cuda else None,
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(args.device), "uuid", "unavailable")) if cuda else None,
        cuda_runtime=torch.version.cuda, threads=args.threads, precision=args.precision,
        temp_directory=__import__("tempfile").gettempdir())


@torch.no_grad()
def actual_report(manifest, args):
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if args.precision == "bf16" and torch.device(args.device).type != "cuda":
        raise ValueError("bf16 profile requires CUDA; choose fp32 for CPU")
    specs = [p for p in manifest["points"] if not args.only or p["label"] in args.only]
    if args.only and set(args.only) - {p["label"] for p in specs}:
        raise ValueError("--only contains unknown labels")
    if any(p.get("tokens", args.tokens) != args.tokens for p in specs):
        raise ValueError("manifest and requested token counts differ")
    missing = [p["label"] for p in specs if not Path(p["checkpoint"]).is_file()]
    if missing and not args.skip_missing:
        raise FileNotFoundError("missing final checkpoints (wait for training or use --skip-missing): " + ", ".join(missing))
    specs = [p for p in specs if p["label"] not in missing]
    if not specs:
        raise ValueError("no final checkpoints available")
    env = environment(args)
    # A resumed Slurm allocation has a new job-specific TMPDIR. Preserve it in
    # each point's provenance, but compare hardware/software independently.
    comparison_environment = {k: v for k, v in env.items() if k != "temp_directory"}
    protocol = dict(tokens=args.tokens, scenes=args.scenes, warmup=args.warmup, repeats=args.repeats,
        threads=args.threads, seed=args.seed, precision=args.precision, num_workers=args.num_workers)
    identity = dict(manifest=manifest, protocol=protocol, environment=comparison_environment)
    identity_file = args.output / "run.json"
    cache_file = args.output / "matched_scene_inputs.pt"
    if identity_file.exists():
        if not args.resume or json.loads(identity_file.read_text(encoding="utf-8")) != identity:
            raise ValueError("existing run differs or --resume absent; use a new output directory")
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("preserving existing nonempty output; choose a new directory")
        args.output.mkdir(parents=True, exist_ok=True)
        cache = None
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    results = []
    for spec in specs:
        point_file = args.output / (spec["label"] + ".json")
        if point_file.exists():
            value = json.loads(point_file.read_text(encoding="utf-8"))
            if value["spec"] != spec or value["input_fingerprint"] != input_fingerprint(cache):
                raise ValueError("completed point does not match this run")
            results.append(value)
            print("REUSE " + spec["label"], flush=True)
            continue
        print("PROFILE " + spec["label"], flush=True)
        model, cfg, footprint = load_model(spec, args)
        if cache is None:
            cache = cache_scenes(cfg, args)
            torch.save(cache, cache_file)
            save_new(identity_file, identity)
        value = dict(spec=spec, environment=env, footprints=footprint,
            input_fingerprint=input_fingerprint(cache),
            warmup_scenes=[v["scene"] for v in cache[:args.warmup]],
            measured_scenes=[v["scene"] for v in cache[args.warmup:]],
            **profile_point(model, cache, args))
        save_new(point_file, value)
        results.append(value)
        del model, cfg
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    report = dict(version=1, mode="actual", protocol=protocol, environment=comparison_environment,
        scope="Actual codec calls on cached identical C12 input scenes; no rendering or training. Counted MACs exclude native entropy coding and sorting distance/comparison work.",
        mac_scope=MAC_SCOPE, skipped_missing=missing, results=results)
    selection_hash = hashlib.sha256(json.dumps([v["spec"]["label"] for v in results]).encode()).hexdigest()[:8]
    summary_file = args.output / f"summary_{len(results):02d}points_{selection_hash}.json"
    if summary_file.exists():
        if json.loads(summary_file.read_text(encoding="utf-8")) != report:
            raise ValueError("summary exists and differs; per-point results preserved")
    else:
        save_new(summary_file, report)
    print(f"PROFILE_COMPLETE points={len(results)} missing={len(missing)} output={args.output}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "actual"), default="static")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--scenes", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", action="append", help="profile only the named manifest label; repeatable")
    args = parser.parse_args(argv)
    if min(args.tokens, args.threads, args.scenes, args.repeats) < 1 or min(args.warmup, args.num_workers) < 0:
        parser.error("tokens/threads/scenes/repeats must be positive; warmup/workers nonnegative")
    torch.set_num_threads(args.threads)
    manifest = load_manifest(args.manifest)
    if args.mode == "static":
        static_report(manifest, args)
    else:
        actual_report(manifest, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Explain ordering -> causal prediction -> actual bits, without training/rendering."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import profile_nfcgs_complexity as profile
from globalsplat.compression.mechanism import (
    StageRecorder, fixed_score_relations, observe_stages, payload_components,
)
from globalsplat.compression.bitstream import SceneBitstream


def load_manifest(path, scope="full"):
    manifest = profile.load_manifest(path)
    specs = manifest["points"]
    focused = scope == "legacy3-anchors"
    schedules = scope == "hilbert-schedule-anchors"
    if len(specs) != (3 if schedules else 4 if focused else 6) or manifest.get("reference") not in {p["label"] for p in specs}:
        if schedules:
            raise ValueError("expected three Hilbert schedule checkpoints")
        raise ValueError("expected four legacy3 checkpoints" if focused else
                         "expected six checkpoints and one named fixed-score reference")
    expected = {("none", "legacy", 3), ("morton", "legacy", 3), ("hilbert", "legacy", 3),
                ("nn_score", "legacy", 3), ("hilbert", "quarter2", 2), ("hilbert", "dyadic4", 4)}
    if focused:
        expected = {item for item in expected if item[1] == "legacy"}
    if schedules:
        expected = {item for item in expected if item[0] == "hilbert"}
    found = set()
    for spec in specs:
        cfg = profile.config_for(spec)
        found.add((cfg.token_order, cfg.score_context_schedule, cfg.score_spatial_stages))
        if spec["rate_lambda"] != .0128 or spec["expected_step"] != 50000 or spec["tokens"] != 4096:
            raise ValueError("this analysis requires lambda .0128, final50k, 4096 tokens")
        if (not cfg.score_spatial_context or not cfg.use_residual
                or cfg.rank != 56 or cfg.score_spatial_kernel != 5):
            raise ValueError("expected the current minimal/rank56/context-ON/MSH-ON checkpoints")
    if found != expected:
        raise ValueError("checkpoint comparison matrix is incomplete or duplicated")
    reference = next(p for p in specs if p["label"] == manifest["reference"])
    if reference["config"]["score_context_schedule"] != "legacy":
        raise ValueError("fixed-score reference must use legacy3")
    return manifest


def input_run_info(args, manifest):
    """Validate the completed source run; never replace a missing cache with new scenes."""
    source = getattr(args, "input_run", None)
    if source is None:
        if getattr(args, "scope", "full") == "hilbert-schedule-anchors":
            raise ValueError("hilbert-schedule-anchors requires --input-run from the completed 256-scene analysis")
        return None
    source = source.resolve()
    for name in ("inputs.pt", "inputs.json", "metadata.json", "complete.json", "backbone.json"):
        if not (source/name).is_file():
            raise FileNotFoundError(f"source input run requires {source/name}; do not resample scenes")
    read = lambda name: json.loads((source/name).read_text(encoding="utf-8"))
    inputs, metadata, complete = read("inputs.json"), read("metadata.json"), read("complete.json")
    if (len(inputs["scenes"]) != args.scenes or len(set(inputs["scenes"])) != args.scenes
            or complete["scenes"] != args.scenes or complete["input_fingerprint"] != inputs["fingerprint"]):
        raise ValueError("source input run is incomplete or its scene fingerprint differs")
    old = metadata["identity"]
    for name in ("scenes", "tokens", "seed", "precision"):
        if old[name] != getattr(args, name):
            raise ValueError(f"source input run protocol differs: {name}")
    expected = getattr(args, "expected_input_fingerprint", None)
    if expected is not None and expected != inputs["fingerprint"]:
        raise ValueError("source input run is not the requested original 256 scenes")
    reference = next(p for p in manifest["points"] if p["label"] == manifest["reference"])
    prior = next((p for p in old["manifest"]["points"] if p["label"] == reference["label"]), None)
    if prior is None or reference["label"] not in complete["points"]:
        raise ValueError("reference model is missing from the completed source run")
    for key in ("checkpoint", "config"):
        if prior[key] != reference[key]:
            raise ValueError(f"reference model differs from the source run: {key}")
    for index in range(args.scenes):
        path = source/"points"/reference["label"]/f"scene_{index:04d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
    return dict(path=str(source), inputs=inputs, backbone_sha256=read("backbone.json")["sha256"],
        metadata_sha256=hashlib.sha256((source/"metadata.json").read_bytes()).hexdigest(),
        reference=reference["label"], source_environment=metadata["environment"])


def backbone_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith("feature_codec."):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_identity(spec):
    path = Path(spec["checkpoint"])
    stat = path.stat()
    return dict(label=spec["label"], path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def save_tensors_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
    # A single output-directory lock prevents competing writers.
    temporary.rename(path)


@torch.no_grad()
def analyze_scene(model, inputs, args, *, scene, fixed_reference=False, keep_tensors=False):
    codec = model.feature_codec
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    with amp():
        texture, geometry = model.encode_scene_tokens(inputs)
        if texture.shape[:2] != (1, args.tokens):
            raise ValueError("unexpected token count")
        positions = model.gaussian_decoder.decoded_token_centers(codec.project_geometry(geometry))
        features, order, _, score = profile.prepare_score(codec, texture, geometry, positions)
        ordered_xyz = order.apply(positions)
        result = dict(scene=scene, measurements={})
        raw = dict(score=score.detach().float().cpu(), xyz=ordered_xyz.detach().float().cpu(),
                   permutation=order.permutation.cpu()) if keep_tensors else None
        if fixed_reference:
            unordered = order.restore(score.squeeze(2).transpose(1, 2))[0]
            with torch.autocast(device_type=score.device.type, enabled=False):
                result["fixed_score"] = fixed_score_relations(unordered.float(), positions[0].float(),
                    bits=codec.config.morton_bits, seed=args.seed, scene=scene, random_order=True,
                    backend=args.fixed_nn_backend, candidates=codec.config.nn_order_candidates,
                    distance_device="input" if args.fixed_nn_backend != "legacy" else "cpu")
        original_features = order.apply(features).float()
        feature_std = original_features.std(1, correction=0).clamp_min(1e-6)
        variants = ("trained", "prediction_bypassed") if args.prediction_ablation else ("trained",)
        for variant in variants:
            recorder = StageRecorder(score, ordered_xyz, keep_tensors=keep_tensors)
            with observe_stages(codec.score_context, recorder, prediction_off=variant != "trained"):
                packed = codec.compress(texture, geometry, positions)
                if not torch.equal(packed.order.permutation, order.permutation):
                    raise AssertionError("token order changed between preparation and coding")
                # The receiver reruns the actual causal decoder, not the observer.
                with torch.autocast(device_type=score.device.type, enabled=False):
                    received = codec.score_context.decompress(SceneBitstream.unpack(packed.data).score, args.tokens)
                if not torch.equal(recorder.reconstruction, received[0, :, 0].T.cpu()):
                    raise AssertionError("observed and receiver-reconstructed scores differ")
                decoded = torch.cat(codec.decompress(packed.data), -1).float()
            stats = payload_components(packed.data)
            stages = recorder.summary()
            if sum(s["raw_bytes"] for s in stages) != stats["components"]["score_raw"]:
                raise AssertionError("stage bytes do not equal actual Score entropy strings")
            stats.update(stages=stages, packet_sha256=hashlib.sha256(packed.data).hexdigest(),
                feature_mse=float((decoded - original_features).square().mean()),
                feature_normalized_mse=float(((decoded-original_features)/feature_std[:, None]).square().mean()))
            result["measurements"][variant] = stats
            if raw is not None:
                raw[variant] = dict(scale=recorder.scale, groups=recorder.tensors)
    return result, raw


def run(manifest, args):
    focused = getattr(args, "scope", "full") != "full"
    if focused and args.prediction_ablation:
        raise ValueError("anchor scopes measure trained models only; disable prediction ablation")
    source_run = input_run_info(args, manifest)
    if args.precision == "bf16" and (not args.device.startswith("cuda") or not torch.cuda.is_available()):
        raise ValueError("BF16 requires CUDA; CPU fixture checks use --precision fp32 --device cpu")
    sources = [source_identity(spec) for spec in manifest["points"]]  # fail before any output on missing checkpoint
    env = profile.environment(args)
    stable_env = {k: env[k] for k in ("torch", "compressai", "numpy", "gpu", "cuda_runtime", "precision")}
    identity = dict(version=1, scope=getattr(args, "scope", "full"), manifest=manifest, sources=sources,
        dataset_root=str(args.dataset_root.resolve()),
        scenes=args.scenes, seed=args.seed, tokens=args.tokens, precision=args.precision,
        prediction_ablation=args.prediction_ablation, trace_scenes=args.trace_scenes,
        fixed_nn_backend=None if focused else args.fixed_nn_backend, environment=stable_env,
        input_run=source_run,
        normalization="Per-scene channel population STD, floor 1e-6; diagnostic only, never encoder conditioning",
        fixed_nn_weighting=None if focused else "Deployed scene mean absolute Score in quantization units, floor .05",
        fixed_nn_source=None if focused else "The SAME extracted quantization-input Score is sorted for every order. "
            "Under BF16 this can differ from production NN's additional FP32 pre-order analysis pass.",
        xyz_normalization="Within-scene RMS radius; raw coordinate distance also retained",
        cdf_policy="Rebuild on the analysis device with update_quantiles=False, matching standard actual-bitstream evaluation",
        sampling="First unique processable RE10K test scenes, C12; not a random sample or render/PSNR evaluation")
    identity["code_sha256"] = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in (
        "scripts/analyze_nfcgs_mechanism.py", "scripts/profile_nfcgs_complexity.py",
        "globalsplat/compression/mechanism.py", "globalsplat/compression/score_context.py",
        "globalsplat/compression/codec.py", "globalsplat/model/globalsplat.py")}
    metadata_path = args.output / "metadata.json"
    if args.output.exists() and not args.resume:
        raise FileExistsError("use a fresh output directory or explicit --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / ".analysis-lock"
    lock.mkdir()
    try:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["identity"] != identity:
                raise ValueError("resume protocol/checkpoints/software differ; previous results preserved")
        else:
            if args.resume and any(p != lock for p in args.output.iterdir()):
                raise ValueError("nonempty output without metadata; use a fresh directory")
            profile.save_new(metadata_path, dict(identity=identity, environment=env))
        reference = next(p for p in manifest["points"] if p["label"] == manifest["reference"])
        specs = [reference] + [p for p in manifest["points"] if p != reference]
        model, cfg, _ = profile.load_model(reference, args)
        model.feature_codec.update(force=True, update_quantiles=False)
        cache_path = args.output / "inputs.pt"
        if source_run is not None:
            cache = torch.load(Path(source_run["path"])/"inputs.pt", map_location="cpu", weights_only=False)
        elif cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            cache = profile.cache_scenes(cfg, args)
            save_tensors_new(cache_path, cache)
        if len(cache) != args.scenes or len({item["scene"] for item in cache}) != args.scenes:
            raise ValueError("cached input scene count/uniqueness mismatch")
        input_identity = dict(fingerprint=profile.input_fingerprint(cache), scenes=[c["scene"] for c in cache])
        if source_run is not None and input_identity != source_run["inputs"]:
            raise ValueError("source inputs.pt content changed; cached image/camera fingerprint differs")
        input_path = args.output / "inputs.json"
        if input_path.exists():
            if json.loads(input_path.read_text(encoding="utf-8")) != input_identity:
                raise ValueError("cached image/camera inputs changed")
        else:
            profile.save_new(input_path, input_identity)
        frozen_hash = backbone_hash(model)
        if source_run is not None and frozen_hash != source_run["backbone_sha256"]:
            raise ValueError("frozen backbone differs from the source input run")
        backbone_path = args.output / "backbone.json"
        if backbone_path.exists():
            if json.loads(backbone_path.read_text())["sha256"] != frozen_hash:
                raise ValueError("reference backbone changed")
        else:
            profile.save_new(backbone_path, dict(sha256=frozen_hash))
        run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        completed = []
        for number, spec in enumerate(specs):
            if number:
                model, _, _ = profile.load_model(spec, args)
                model.feature_codec.update(force=True, update_quantiles=False)
            if backbone_hash(model) != frozen_hash:
                raise ValueError(f"frozen encoder/Gaussian decoder differs: {spec['label']}")
            folder = args.output / "points" / spec["label"]
            folder.mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(cache):
                path = folder / f"scene_{index:04d}.json"
                if path.exists():
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    if previous["scene"] != item["scene"] or previous["label"] != spec["label"]:
                        raise ValueError("saved scene identity mismatch")
                    if previous.get("trace") and not (args.output / previous["trace"]).is_file():
                        raise FileNotFoundError(previous["trace"])
                    continue
                result, raw = analyze_scene(model, profile.to_device(item["inputs"], args.device), args,
                    scene=item["scene"], fixed_reference=number == 0 and not focused,
                    keep_tensors=index < args.trace_scenes)
                result.update(label=spec["label"], index=index, environment=env)
                if source_run is not None and spec["label"] == source_run["reference"]:
                    old_path = Path(source_run["path"])/"points"/spec["label"]/f"scene_{index:04d}.json"
                    old = json.loads(old_path.read_text(encoding="utf-8"))
                    if old["scene"] != item["scene"] or old["label"] != spec["label"]:
                        raise ValueError("source reference scene identity mismatch")
                    before, after = old["measurements"]["trained"], result["measurements"]["trained"]
                    result["source_reference_check"] = dict(
                        packet_sha256_equal=before["packet_sha256"] == after["packet_sha256"],
                        total_bytes_delta=after["total_bytes"]-before["total_bytes"],
                        score_raw_bytes_delta=after["components"]["score_raw"]-before["components"]["score_raw"],
                        stage_raw_mae_delta=[a["raw_mae"]-b["raw_mae"] for a,b in zip(after["stages"],before["stages"])])
                if raw is not None:
                    relative = Path("traces") / spec["label"] / f"scene_{index:04d}_{run_tag}.pt"
                    save_tensors_new(args.output / relative, raw)
                    result["trace"] = relative.as_posix()
                profile.save_new(path, result)
                print(f"SCENE {spec['label']} {index+1}/{args.scenes} {item['scene']}", flush=True)
            completed.append(spec["label"])
            del model
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        completion = args.output / "complete.json"
        result = dict(points=completed, scenes=args.scenes, input_fingerprint=input_identity["fingerprint"])
        if completion.exists():
            if json.loads(completion.read_text()) != result:
                raise ValueError("completion record differs")
        else:
            profile.save_new(completion, result)
        print(f"ANALYSIS_COMPLETE models={len(completed)} scenes={args.scenes} output={args.output}", flush=True)
    finally:
        lock.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scope", choices=("full", "legacy3-anchors", "hilbert-schedule-anchors"), default="full",
        help="anchor scopes observe trained models only; schedule comparison requires the original input cache")
    parser.add_argument("--input-run", type=Path, help="completed source run whose exact cached inputs are reused")
    parser.add_argument("--expected-input-fingerprint", help="require the original cached scene/image/camera fingerprint")
    parser.add_argument("--reference", help="optional legacy3 reference label for a separate robustness run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scenes", type=int, default=256)
    parser.add_argument("--trace-scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--fixed-nn-backend", choices=("knn", "row", "legacy"), default="knn")
    parser.add_argument("--prediction-ablation", action="store_true",
        help="also bypass prediction at inference using unchanged trained entropy/MSH weights; not a trained control")
    parser.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    args = parser.parse_args(argv)
    if min(args.scenes, args.tokens, args.threads) < 1 or min(args.num_workers, args.trace_scenes) < 0:
        parser.error("scene/token/thread counts positive; worker/trace counts nonnegative")
    args.warmup = 0  # deterministic data cache; no timing benchmark
    manifest = load_manifest(args.manifest, args.scope)
    if args.scope != "full" and args.prediction_ablation:
        parser.error("anchor scopes do not run prediction bypass")
    if args.reference:
        manifest["reference"] = args.reference
        match = [p for p in manifest["points"] if p["label"] == args.reference]
        if len(match) != 1 or match[0]["config"]["score_context_schedule"] != "legacy":
            parser.error("--reference must name a legacy3 point in this manifest")
    if args.plan:
        source = input_run_info(args, manifest)
        if source is not None:
            print(f"INPUT_RUN={source['path']} INPUT_FINGERPRINT={source['inputs']['fingerprint']} CACHE_EXISTS=True")
        missing = []
        for spec in manifest["points"]:
            print(f"PLAN {spec['label']} EXISTS={Path(spec['checkpoint']).is_file()} CHECKPOINT={spec['checkpoint']}")
            if not Path(spec["checkpoint"]).is_file():
                missing.append(spec["label"])
        print(f"SCOPE={args.scope} MODELS={len(manifest['points'])} SCENES={args.scenes} "
              f"REFERENCE={manifest['reference']} TRACE_SCENES={min(args.trace_scenes,args.scenes)}")
        if missing:
            raise SystemExit("Missing checkpoints: " + ", ".join(missing))
        return
    if args.output is None:
        parser.error("--output is required unless --plan")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    run(manifest, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate T04/B1 and seven T05 paths on the original matched C12/T8 scenes.

This is an inference-only diagnostic, not a training run or an RD sweep.
Input-cache loss/mismatch is fatal: there is no fallback scene resampling.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import profile_nfcgs_complexity as profile
from globalsplat.compression.config import CodecConfig
from globalsplat.compression.path_diagnostics import MODES, FORMULAS, build_paths

# Final steps of the warm-start sweep (50k) and the fresh RD protocol (120k).
SUPPORTED_STEPS = (50000, 120000)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    profile.save_new(Path(path), value)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backbone_hash(model):
    # Byte-for-byte compatible with the original matched-scene analysis.
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith("feature_codec."):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def code_hashes():
    files = set((REPO / "globalsplat").rglob("*.py"))
    files.update((REPO / "config").rglob("*.yaml"))
    for folder in ("dataset", "evaluation"):
        files.update((REPO / "third_party/ZPressor/mvsplat/src" / folder).rglob("*.py"))
    files.update(REPO / relative for relative in (
        "scripts/eval_nfcgs_path_diagnostics.py", "scripts/profile_nfcgs_complexity.py",
        "scripts/slurm/eval_nfcgs_path_diagnostics.slurm"))
    return {p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(files)}


def load_spec(path):
    manifest = read_json(path)
    if manifest.get("version") != 1 or len(manifest.get("points", [])) != 1:
        raise ValueError("expected a version=1 single-checkpoint diagnostic manifest")
    spec = dict(manifest["points"][0])
    config = CodecConfig.from_mapping(spec["config"])
    required = dict(texture_channels=512, geometry_channels=512, geometry_observable_channels=224,
                    rank=56, codec_mode="two_path", use_residual=True, token_order="hilbert",
                    score_context_schedule="dyadic4", score_spatial_stages=4,
                    score_slice_channels=16, score_spatial_context=True,
                    use_centering=False, use_score_norm=False, transform="linear",
                    score_mean_condition=False, score_channel_context=False,
                    score_spatial_kernel=5, score_context_quantization="ste")
    if any(getattr(config, key) != value for key, value in required.items()):
        raise ValueError("manifest must describe the current minimal Hilbert+D4 rank56 two-path method")
    if spec.get("tokens") != 4096 or spec.get("expected_step") not in SUPPORTED_STEPS:
        raise ValueError("expected a final 50k (warm-start) or 120k (fresh) 4096-token checkpoint")
    if not math.isfinite(spec["rate_lambda"]) or spec["rate_lambda"] < 0:
        raise ValueError("invalid lambda")
    checkpoint = Path(spec["checkpoint"])
    spec["checkpoint"] = str(checkpoint if checkpoint.is_absolute() else REPO / checkpoint)
    if not Path(spec["checkpoint"]).is_file():
        raise FileNotFoundError(spec["checkpoint"])
    reference = spec.get("backbone_reference")
    if reference is not None:
        if reference.get("kind") != "fresh_initializer" or not reference.get("vanilla_checkpoint"):
            raise ValueError("backbone_reference must be a fresh_initializer with the released checkpoint")
        vanilla = Path(reference["vanilla_checkpoint"])
        spec["backbone_reference"] = dict(reference, vanilla_checkpoint=str(vanilla if vanilla.is_absolute() else REPO / vanilla))
    return manifest, spec


def verify_backbone(model, spec, checkpoint, source):
    """Accept the cached analysis backbone, or a fresh run's frozen initializer.

    Fresh runs copy the released encoder exactly but re-derive the observable
    geometry readouts by QR, so their decoder bytes differ from the warm-start
    lineage that produced the matched-scene cache.
    """
    current = backbone_hash(model)
    if current == source["backbone_sha256"]:
        return dict(kind="source_analysis", sha256=current)
    reference = spec.get("backbone_reference")
    if not reference:
        raise ValueError("frozen encoder/Gaussian decoder differs from the original 256-scene analysis")
    from globalsplat.compression.initialization import _READOUTS, _checkpoint_state
    from globalsplat.fresh_rd import CELLS
    protocol = torch.load(checkpoint, map_location="cpu", weights_only=False).get("fresh_rd_protocol")
    if not protocol:
        raise ValueError("a fresh backbone reference requires a checkpoint with fresh_rd_protocol")
    initializer = Path(protocol["preparation"]) / f"{CELLS[protocol['cell']].arm}.ckpt"
    if file_hash(initializer) != protocol["initializer_sha256"]:
        raise ValueError("fresh initializer differs from the run protocol")
    released_path = Path(reference["vanilla_checkpoint"])
    if file_hash(released_path) != protocol["vanilla_sha256"]:
        raise ValueError("released checkpoint differs from the run protocol")
    frozen = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("feature_codec.")}
    initial = _checkpoint_state(initializer)
    if set(frozen) != {k for k in initial if not k.startswith("feature_codec.")}:
        raise ValueError("frozen backbone keys differ from the fresh initializer")
    for key, value in frozen.items():
        if not torch.equal(value, initial[key]):
            raise ValueError(f"frozen backbone tensor changed after initialization: {key}")
    released = _checkpoint_state(released_path)
    for key, value in frozen.items():
        if key not in _READOUTS and (key not in released or not torch.equal(value, released[key].to(value.dtype))):
            raise ValueError(f"fresh backbone differs from the released checkpoint: {key}")
    # The initializer factors W = W_obs @ P0 exactly; P is trained afterwards, W_obs is not.
    readout = torch.cat([initial[name].double() for name in _READOUTS])
    projection = initial["feature_codec.geometry_projection.weight"].double()
    error = float((readout @ projection - torch.cat([released[name].double() for name in _READOUTS])).abs().max())
    if error > 1e-4:
        raise ValueError("fresh observable readouts do not factor the released readouts")
    return dict(kind="fresh_initializer", sha256=current, source_analysis_sha256=source["backbone_sha256"],
                initializer=str(initializer), initializer_sha256=protocol["initializer_sha256"],
                released_checkpoint=str(released_path), released_sha256=protocol["vanilla_sha256"],
                max_readout_reparameterization_error=error,
                scope="released encoder and appearance readout exact; geometry readouts are the frozen "
                      "QR factorization of the released readouts")


def source_info(args, *, load_cache):
    source = args.input_run.resolve()
    names = ("inputs.pt", "inputs.json", "metadata.json", "complete.json", "backbone.json")
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(f"required original source artifact: {source / name}; do not resample")
    inputs = read_json(source / "inputs.json")
    complete = read_json(source / "complete.json")
    metadata = read_json(source / "metadata.json")
    identity = metadata["identity"]
    if (len(inputs["scenes"]) != args.scenes or len(set(inputs["scenes"])) != args.scenes
            or complete["scenes"] != args.scenes
            or complete["input_fingerprint"] != inputs["fingerprint"]
            or not complete.get("points")):
        raise ValueError("original input run count/fingerprint/completion mismatch")
    for key in ("scenes", "tokens", "seed", "precision"):
        if identity[key] != getattr(args, key):
            raise ValueError(f"original input run protocol mismatch: {key}")
    if inputs["fingerprint"] != args.expected_input_fingerprint:
        raise ValueError("the source cache is not the requested original matched scenes")
    hashes = None
    if load_cache:
        cache = torch.load(source / "inputs.pt", map_location="cpu", weights_only=False)
        if ([item["scene"] for item in cache] != inputs["scenes"]
                or profile.input_fingerprint(cache) != inputs["fingerprint"]):
            raise ValueError("original inputs.pt content differs from inputs.json")
        hashes = {item["scene"]: profile.input_fingerprint([item]) for item in cache}
        del cache
    info = dict(path=str(source), inputs=inputs,
                backbone_sha256=read_json(source / "backbone.json")["sha256"],
                artifact_sha256={name: file_hash(source / name) for name in names if name != "inputs.pt"})
    return info, hashes


def amp(args):
    if args.precision == "bf16":
        return torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16)
    return nullcontext()


def frame_ids(section):
    return [int(x) for x in section["frame_ids"].detach().cpu().reshape(-1)]


def validate_views(batch):
    context, target = batch["inputs"], batch["targets"]
    if tuple(context["images"].shape) != (1, 12, 3, 256, 256):
        raise ValueError("expected one C12 scene at 256x256")
    if tuple(target["images"].shape) != (1, 8, 3, 256, 256):
        raise ValueError("expected one T8 scene at 256x256")
    c, t = frame_ids(context), frame_ids(target)
    if len(set(c)) != 12 or len(set(t)) != 8 or set(c) & set(t):
        raise ValueError("context/target frame IDs must be unique and disjoint")


def load_data(cfg):
    from globalsplat.main import build_datamodule
    data, _ = build_datamodule(cfg)
    return data.test_dataloader()


def render_functions():
    from globalsplat.dataset.data_module import add_repo_to_path
    from globalsplat.model.rendering import render_static_batched
    add_repo_to_path(REPO / "third_party/ZPressor/mvsplat", repo_name="path-diagnostics")
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    def render(gaussians, targets):
        result = render_static_batched(gaussians, targets, render_depth=False)["img"]
        _, views, _, h, w = targets["images"].shape
        return result.reshape(1, views, 3, h, w)[0].float().clamp(0, 1)

    def metrics(gt, pred):
        # Matches the production full-scene convention: mean of per-view scores.
        with torch.autocast(device_type=gt.device.type, enabled=False):
            return {"psnr": float(compute_psnr(gt, pred).mean()),
                    "ssim": float(compute_ssim(gt, pred).mean()),
                    "lpips": float(compute_lpips(gt, pred).mean())}

    return render, metrics


@torch.no_grad()
def evaluate_scene(model, batch, args, render, metrics):
    validate_views(batch)
    inputs = profile.to_device(batch["inputs"], args.device)
    targets = profile.to_device(batch["targets"], args.device)
    gt = targets["images"][0].float().clamp(0, 1)
    codec = model.feature_codec
    with amp(args):
        texture, geometry = model.encode_scene_tokens(inputs)
        if texture.shape[:2] != (1, args.tokens):
            raise ValueError("unexpected scene token count")
        positions = model.gaussian_decoder.decoded_token_centers(codec.project_geometry(geometry))
        paths = build_paths(codec, texture, geometry, positions)
    result = dict(context_frame_ids=frame_ids(batch["inputs"]), target_frame_ids=frame_ids(batch["targets"]),
                  input_fingerprint=profile.input_fingerprint([batch["inputs"]]),
                  target_fingerprint=profile.input_fingerprint([batch["targets"]]),
                  permutation_sha256=profile.input_fingerprint([paths.permutation]),
                  packet_sha256=hashlib.sha256(paths.packet).hexdigest(),
                  score_mse=paths.score_mse, packet_decomposition_exact=True,
                  score_wrapper_bytes=paths.score_wrapper_bytes,
                  raw_score_payload_bytes=paths.stream_bytes["score"] - paths.score_wrapper_bytes,
                  all_header_bytes=paths.stream_bytes["container"] + paths.score_wrapper_bytes,
                  modes={})
    original = paths.features["b1_direct"].float()
    for name in MODES:
        value = paths.features[name]
        decoded_tokens = (value[..., :codec.config.texture_channels], value[..., codec.config.texture_channels:])
        with amp(args):
            gaussians = model.decode_scene_tokens(decoded_tokens)
        # The rasterizer uses FP32, as in normal full-test evaluation.
        pred = render(gaussians, targets)
        row = metrics(gt, pred)
        delta = value.float() - original
        row.update(feature_mse=float(delta.square().mean()),
                   texture_mse=float(delta[..., :codec.config.texture_channels].square().mean()),
                   geometry_mse=float(delta[..., codec.config.texture_channels:].square().mean()),
                   feature_max_abs=float(delta.abs().max()),
                   actual_bytes=len(paths.packet) if name == "actual_packet" else None,
                   stream_bytes=paths.stream_bytes if name == "actual_packet" else None)
        if not all(math.isfinite(row[key]) for key in (
                "psnr", "ssim", "lpips", "feature_mse", "texture_mse", "geometry_mse", "feature_max_abs")):
            raise RuntimeError(f"non-finite metrics in {name}")
        if name == "actual_packet":
            repeat = render(gaussians, targets)
            repeated_metrics = metrics(gt, repeat)
            result["renderer_repeat"] = dict(
                pixel_max_abs=float((repeat - pred).abs().max()),
                pixel_mse=float((repeat - pred).square().mean()),
                psnr_delta=repeated_metrics["psnr"] - row["psnr"],
                scope="repeat rasterization of the same Gaussians/cameras on this device")
        result["modes"][name] = row
        del gaussians, pred
    for row in result["modes"].values():
        row["psnr_minus_actual"] = row["psnr"] - result["modes"]["actual_packet"]["psnr"]
        row["psnr_minus_b1"] = row["psnr"] - result["modes"]["b1_direct"]["psnr"]
    return result, paths.packet


def summarize(rows):
    fields = ("psnr", "ssim", "lpips", "feature_mse", "texture_mse", "geometry_mse",
              "psnr_minus_actual", "psnr_minus_b1")
    modes = {}
    for mode in MODES:
        modes[mode] = {key: statistics.fmean(row["modes"][mode][key] for row in rows) for key in fields}
        modes[mode]["formula"] = FORMULAS[mode]
        modes[mode]["mean_actual_bytes"] = (
            statistics.fmean(row["modes"][mode]["actual_bytes"] for row in rows)
            if mode == "actual_packet" else None)
    return dict(scene_count=len(rows), aggregation="equal mean of scene means over 8 target views",
                modes=modes, renderer_repeat=dict(
                    max_pixel_abs=max(row["renderer_repeat"]["pixel_max_abs"] for row in rows),
                    max_abs_psnr_delta=max(abs(row["renderer_repeat"]["psnr_delta"]) for row in rows)),
                interpretation="Paired interventions, not additive independent path contributions. "
                "Continuous bypass modes have no packet rate. Hcont retains analysis/synthesis loss. "
                "B1 is post-projection, not native GlobalSplat B0. Matched diagnostic subset, not full-test RD.")


def write_report(output, summary, spec, backbone=None):
    lines = ["# T04/B1 and T05 path diagnostics", "",
             f"Checkpoint: `{spec['label']}`; lambda={spec['rate_lambda']}; step {spec['expected_step']}.",
             f"Matched scenes: {summary['scene_count']}; C12/T8; 256x256; BF16 unless metadata says otherwise.", "",
             "| Mode | PSNR | SSIM | LPIPS | PSNR - actual | PSNR - B1 | KiB/scene |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, row in summary["modes"].items():
        rate = f"{row['mean_actual_bytes']/1024:.4f}" if row["mean_actual_bytes"] is not None else "n/a"
        lines.append(f"| {mode} | {row['psnr']:.5f} | {row['ssim']:.5f} | {row['lpips']:.5f} | "
                     f"{row['psnr_minus_actual']:+.5f} | {row['psnr_minus_b1']:+.5f} | {rate} |")
    lines += ["", summary["interpretation"], "",
              f"Repeated actual-path rasterization: {json.dumps(summary['renderer_repeat'], sort_keys=True)}", "",
              "All paths share encoded F, geometry projection, basis, sorted order, Gaussian decoder and cameras.",
              "The actual-path decomposition is checked exactly against full packet decoding for every scene.",
              "Raw packets are in packets/; per-scene frame IDs, fingerprints, feature errors and stream sizes are in scenes/.",
              "A native B0 evaluation and packet-only independent receiver are separate experiments.", ""]
    if backbone and backbone["kind"] != "source_analysis":
        lines[-1:-1] = [f"Backbone check: {backbone['kind']} ({backbone['scope']}); "
                        f"readout factorization error {backbone['max_readout_reparameterization_error']:.3g}."]
    with (output / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


@torch.no_grad()
def run(args, manifest, spec):
    if args.output.exists():
        raise FileExistsError("use a fresh output directory; existing diagnostics are preserved")
    if args.precision == "bf16" and (torch.device(args.device).type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("production BF16 diagnostics require CUDA; CPU is for fp32 fixtures only")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    source, expected_hashes = source_info(args, load_cache=True)
    checkpoint = Path(spec["checkpoint"])
    checkpoint_hash = file_hash(checkpoint)
    stat = checkpoint.stat()
    model, cfg, _ = profile.load_model(spec, args)
    # globalsplat.main changes backend defaults when imported by load_model.
    # Apply the existing architecture-smoke entropy policy AFTER that import.
    # Do not require deterministic algorithms globally: GPU rasterization has
    # its own measured repeat tolerance and may use atomic reductions.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    backbone = verify_backbone(model, spec, checkpoint, source)
    if stat.st_size != checkpoint.stat().st_size or stat.st_mtime_ns != checkpoint.stat().st_mtime_ns:
        raise ValueError("checkpoint changed during model loading")
    # Standard actual-bitstream evaluation rebuilds entropy tables on the eval device.
    model.feature_codec.update(force=True, update_quantiles=False)
    runtime_hashes = code_hashes()
    metadata = dict(version=1, started_utc=datetime.now(timezone.utc).isoformat(),
                    manifest=manifest, manifest_sha256=file_hash(args.manifest),
                    checkpoint=dict(path=str(checkpoint.resolve()), sha256=checkpoint_hash,
                                    size=stat.st_size, global_step=spec["expected_step"]),
                    backbone=backbone,
                    source=source, environment=profile.environment(args), code_sha256=runtime_hashes,
                    protocol=dict(scenes=args.scenes, tokens=args.tokens, seed=args.seed,
                                  precision=args.precision, context_views=12, target_views=8,
                                  image_shape=[256, 256], modes=FORMULAS, shared_sorted_order=True,
                                  score_mean_offset_enabled=model.feature_codec.score_context.mean_offset_enabled,
                                  cdf_policy="force update on eval device, update_quantiles=False"),
                    numerics=dict(cudnn_version=torch.backends.cudnn.version(),
                                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "metadata.json", metadata)
    render, metrics = render_functions()
    selected = {name: index for index, name in enumerate(source["inputs"]["scenes"])}
    seen, rows = set(), {}
    for batch in load_data(cfg):
        scene = batch["scene_info"]["scene"]
        scene = str(scene[0] if isinstance(scene, (tuple, list)) else scene)
        if scene not in selected:
            continue
        if scene in seen:
            raise ValueError(f"duplicate matched scene: {scene}")
        seen.add(scene)
        item = dict(scene=scene, inputs=profile.to_device(batch["inputs"], "cpu"))
        if profile.input_fingerprint([item]) != expected_hashes[scene]:
            raise ValueError(f"context images/cameras/frames differ from the original cache: {scene}")
        value, packet = evaluate_scene(model, batch, args, render, metrics)
        index = selected[scene]
        value.update(scene=scene, index=index)
        packet_path = args.output / "packets" / f"scene_{index:04d}.bin"
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        with packet_path.open("xb") as stream:
            stream.write(packet)
        value["packet_file"] = packet_path.relative_to(args.output).as_posix()
        save_json(args.output / "scenes" / f"scene_{index:04d}.json", value)
        rows[index] = value
        print(f"PATH_DIAGNOSTIC_SCENE={len(rows)}/{args.scenes} scene={scene} "
              f"b1_psnr={value['modes']['b1_direct']['psnr']:.5f} "
              f"actual_psnr={value['modes']['actual_packet']['psnr']:.5f}", flush=True)
        if len(rows) == args.scenes:
            break
    if seen != set(selected):
        raise RuntimeError(f"matched scenes missing from loader: {sorted(set(selected)-seen)}")
    if code_hashes() != runtime_hashes:
        raise RuntimeError("runtime source changed during diagnostics; results are not marked complete")
    summary = summarize([rows[index] for index in range(args.scenes)])
    save_json(args.output / "summary.json", summary)
    write_report(args.output, summary, spec, backbone)
    save_json(args.output / "complete.json", dict(scene_count=args.scenes, modes=list(MODES),
              input_fingerprint=source["inputs"]["fingerprint"], checkpoint_sha256=checkpoint_hash))
    print(f"PATH_DIAGNOSTICS_COMPLETE scenes={args.scenes} modes={len(MODES)} output={args.output}", flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "config/plans/nfcgs_path_diagnostics_20261019.json")
    parser.add_argument("--input-run", type=Path, default=REPO / "outputs/nfcgs_legacy3_anchors/job_454995")
    parser.add_argument("--expected-input-fingerprint", default="069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    parser.add_argument("--scenes", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if min(args.scenes, args.tokens, args.threads) < 1 or args.num_workers < 0:
        parser.error("scene/token/thread counts must be positive; workers must be nonnegative")
    if not args.plan and args.output is None:
        parser.error("--output is required unless --plan")
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest, spec = load_spec(args.manifest)
    if args.tokens != spec["tokens"]:
        raise ValueError("requested token count differs from the checkpoint manifest")
    if args.plan:
        source, _ = source_info(args, load_cache=False)
        print(f"PATH_DIAGNOSTICS_PLAN checkpoint={spec['checkpoint']} scenes={args.scenes} "
              f"modes={len(MODES)} input_fingerprint={source['inputs']['fingerprint']}")
        print("PLAN_ONLY=True TRAINING=False B0=False INDEPENDENT_RECEIVER=False")
        print("DATASET_CHECK=deferred_to_compute_node CACHE_TENSORS_CHECK=deferred_to_run")
        return
    if not (args.dataset_root / "test/index.json").is_file():
        raise FileNotFoundError(args.dataset_root / "test/index.json")
    run(args, manifest, spec)


if __name__ == "__main__":
    main()

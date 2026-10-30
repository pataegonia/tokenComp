#!/usr/bin/env python3
"""Native geometry512 B0, with an in-job B1 bridge on the original 256 scenes.

The current codec checkpoint supplies ONLY the matched order and B1 geometry
boundary. B0 passes unprojected 512-D geometry to the released native decoder;
it performs no low-rank reconstruction, MSH transform or entropy coding.
"""
from __future__ import annotations

import argparse
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
import eval_nfcgs_path_diagnostics as diag

NATIVE_SHA256 = "90be09504fa046792dedfe2d1b17e00bb64a5bf95da477d71e233be01df72a42"
READOUTS = (
    "gaussian_decoder.geo_pos_readout.weight",
    "gaussian_decoder.geo_param_readout.weight",
    "gaussian_decoder.gate_readout.weight",
)
GAUSSIAN_FIELDS = ("means", "scales", "rotations", "sh", "opacities")
MODES = ("b0_native", "b1_bridge")


def native_model_config(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from globalsplat.main import build_model
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "config")):
        cfg = compose(config_name="main", overrides=["+experiment=re10k_32k", "dataset=re10k_eval_all_ctx12"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.feature_codec = None
    cfg.model.freeze_globalsplat = False  # the constructor's codec-training flag
    cfg.model.dim_latents = 512
    cfg.model.latent_rep_token_amount = args.tokens
    return build_model(cfg.model)


def load_native(args):
    path = args.native_checkpoint.resolve()
    stat = path.stat()
    digest = diag.file_hash(path)
    if digest != NATIVE_SHA256:
        raise ValueError(f"released native checkpoint SHA-256 mismatch: {digest}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("state_dict", checkpoint)
    state = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            raise ValueError("native state_dict must contain only model tensors")
        name = str(key)
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("model."):
            name = name[len("model."):]
        if name in state:
            raise ValueError("native checkpoint has colliding wrapper prefixes")
        state[name] = value
    if any(name.startswith("feature_codec.") for name in state):
        raise ValueError("B0 requires the native checkpoint, not codec/projected weights")
    if tuple(state["scene_tokens"].shape) != (args.tokens, 512):
        raise ValueError("native scene token count/width mismatch")
    if any(state[key].ndim != 2 or state[key].shape[1] != 512 for key in READOUTS):
        raise ValueError("B0 requires all three native geometry512 readouts")
    model = native_model_config(args)
    model.load_state_dict(state, strict=True)
    if model.feature_codec is not None:
        raise ValueError("B0 unexpectedly contains a feature codec")
    if stat.st_size != path.stat().st_size or stat.st_mtime_ns != path.stat().st_mtime_ns:
        raise ValueError("native checkpoint changed during loading")
    model.requires_grad_(False).eval().to(args.device)
    model.set_stage(3, mix=1.0)
    return model, dict(path=str(path), sha256=digest, size=stat.st_size,
                       native_geometry_channels=512, tokens=args.tokens, stage=3, gaussians_per_token=8)


def state_digest(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def validate_model_pair(native, reference):
    """Only the three QR geometry readouts and learned projection may differ."""
    if native.feature_codec is not None or reference.feature_codec is None:
        raise ValueError("expected native B0 and integrated reference models")
    original = native.state_dict()
    current = {k: v for k, v in reference.state_dict().items() if not k.startswith("feature_codec.")}
    if set(original) != set(current):
        raise ValueError("native/reference model tensor keys differ")
    changed = [key for key in original if key not in READOUTS and (
        original[key].shape != current[key].shape or original[key].dtype != current[key].dtype
        or not torch.equal(original[key].cpu(), current[key].cpu()))]
    if changed:
        raise ValueError(f"native/reference encoder or shared decoder weights differ: {changed}")
    encoder = state_digest((k, v) for k, v in original.items() if not k.startswith("gaussian_decoder."))
    decoder = state_digest((k, v) for k, v in original.items() if k.startswith("gaussian_decoder.") and k not in READOUTS)
    projection = reference.feature_codec.geometry_projection.weight.float()
    operators = {}
    for key in READOUTS:
        effective = current[key].float() @ projection
        delta = effective - original[key].float()
        operators[key] = dict(rmse=float(delta.square().mean().sqrt()), max_abs=float(delta.abs().max()),
                              relative_frobenius=float(torch.linalg.vector_norm(delta) /
                                  torch.linalg.vector_norm(original[key].float()).clamp_min(1e-12)))
    return dict(encoder_tensors_equal=True, shared_decoder_tensors_equal=True,
                encoder_sha256=encoder, shared_decoder_sha256=decoder,
                geometry_boundary_effective_readout_errors=operators,
                allowed_weight_differences=list(READOUTS))


def code_hashes():
    result = diag.code_hashes()
    for relative in ("scripts/eval_nfcgs_b0_matched.py", "scripts/slurm/eval_nfcgs_b0_matched.slurm"):
        result[relative] = diag.file_hash(REPO / relative)
    return result


def numerical_policy():
    # Match T04/T05 AFTER imports that change globalsplat.main's backend flags.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return dict(cudnn_version=torch.backends.cudnn.version(), cudnn_deterministic=True,
                cudnn_benchmark=False, cudnn_allow_tf32=False, matmul_allow_tf32=False,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def permutation_errors(sorted_gaussians, slot_gaussians, permutation):
    count = permutation.shape[1]
    index = (permutation[..., None] * 8 + torch.arange(8, device=permutation.device)).reshape(1, -1)
    errors = {}
    for name in GAUSSIAN_FIELDS:
        value, original = getattr(sorted_gaussians, name), getattr(slot_gaussians, name)
        if value.shape != original.shape or value.shape[1] != count * 8:
            raise ValueError("expected eight Gaussians per token in both native orders")
        gather = index.reshape(*index.shape, *([1] * (original.ndim - 2))).expand_as(original)
        errors[name] = float((value.float() - original.gather(1, gather).float()).abs().max())
    return errors


@torch.no_grad()
def evaluate_scene(native, reference, batch, args, render, metrics):
    diag.validate_views(batch)
    inputs = diag.profile.to_device(batch["inputs"], args.device)
    targets = diag.profile.to_device(batch["targets"], args.device)
    gt = targets["images"][0].float().clamp(0, 1)
    codec = reference.feature_codec
    with diag.amp(args):
        texture, geometry = native.encode_scene_tokens(inputs)
        if texture.shape[:2] != (1, args.tokens) or geometry.shape[-1] != native.dim_latents:
            raise ValueError("native encoder output shape mismatch")
        # The SAME Hilbert order as T05, computed from the current geometry boundary.
        features = codec._pack_features(texture, geometry)
        positions = reference.gaussian_decoder.decoded_token_centers(codec.project_geometry(geometry))
        ordered, order = codec._ordered_features(features, positions)
        native_sorted = native.decode_scene_tokens((order.apply(texture), order.apply(geometry)))
        b1 = reference.decode_scene_tokens((ordered[..., :codec.config.texture_channels],
                                            ordered[..., codec.config.texture_channels:]))
        native_slots = native.decode_scene_tokens((texture, geometry))
    result = dict(context_frame_ids=diag.frame_ids(batch["inputs"]),
                  target_frame_ids=diag.frame_ids(batch["targets"]),
                  input_fingerprint=diag.profile.input_fingerprint([batch["inputs"]]),
                  target_fingerprint=diag.profile.input_fingerprint([batch["targets"]]),
                  permutation_sha256=diag.profile.input_fingerprint([order.permutation]),
                  encoded_tokens_fingerprint=diag.profile.input_fingerprint([texture, geometry]),
                  b1_features_fingerprint=diag.profile.input_fingerprint([ordered]),
                  modes={}, renderer_repeat={})
    b0_pred = None
    for name, gaussians in (("b0_native", native_sorted), ("b1_bridge", b1)):
        if gaussians.means.shape[1] != args.tokens * 8:
            raise ValueError("expected stage3 eight Gaussians per token")
        pred = render(gaussians, targets)
        row = metrics(gt, pred)
        if not all(math.isfinite(row[key]) for key in ("psnr", "ssim", "lpips")):
            raise RuntimeError(f"non-finite render metrics: {name}")
        row["actual_bytes"] = None
        result["modes"][name] = row
        repeat = render(gaussians, targets)
        result["renderer_repeat"][name] = dict(
            pixel_max_abs=float((repeat - pred).abs().max()),
            pixel_mse=float((repeat - pred).square().mean()),
            psnr_delta=metrics(gt, repeat)["psnr"] - row["psnr"])
        if name == "b0_native":
            b0_pred = pred
    slots_pred = render(native_slots, targets)
    result["native_slot_order_check"] = dict(
        metrics=metrics(gt, slots_pred), pixel_max_abs=float((slots_pred - b0_pred).abs().max()),
        pixel_mse=float((slots_pred - b0_pred).square().mean()),
        gaussian_field_max_abs=permutation_errors(native_sorted, native_slots, order.permutation))
    result["b1_minus_b0_psnr"] = result["modes"]["b1_bridge"]["psnr"] - result["modes"]["b0_native"]["psnr"]
    return result


def summarize(rows):
    return dict(scene_count=len(rows), aggregation="equal mean of scene means over 8 target views",
        modes={name: {key: statistics.fmean(row["modes"][name][key] for row in rows)
                      for key in ("psnr", "ssim", "lpips")} | {"mean_actual_bytes": None} for name in MODES},
        b1_minus_b0_psnr=statistics.fmean(row["b1_minus_b0_psnr"] for row in rows),
        renderer_repeat={name: dict(
            max_pixel_abs=max(row["renderer_repeat"][name]["pixel_max_abs"] for row in rows),
            max_abs_psnr_delta=max(abs(row["renderer_repeat"][name]["psnr_delta"]) for row in rows)) for name in MODES},
        native_slot_order_check=dict(
            mean_psnr=statistics.fmean(row["native_slot_order_check"]["metrics"]["psnr"] for row in rows),
            mean_ssim=statistics.fmean(row["native_slot_order_check"]["metrics"]["ssim"] for row in rows),
            mean_lpips=statistics.fmean(row["native_slot_order_check"]["metrics"]["lpips"] for row in rows),
            mean_psnr_minus_sorted=statistics.fmean(row["native_slot_order_check"]["metrics"]["psnr"] -
                                                  row["modes"]["b0_native"]["psnr"] for row in rows),
            max_pixel_abs=max(row["native_slot_order_check"]["pixel_max_abs"] for row in rows),
            max_gaussian_abs={name: max(row["native_slot_order_check"]["gaussian_field_max_abs"][name] for row in rows)
                              for name in GAUSSIAN_FIELDS}),
        interpretation="B0 uses released native geometry512; B1 uses the current projected geometry224 boundary. "
            "Both paths share one encoder output, token order and target cameras within this run. "
            "Neither path has a packet rate; this matched subset is not full-test RD or an independent receiver.")


def write_report(output, summary):
    lines = ["# Matched native B0 and current B1 bridge", "",
        f"Scenes: {summary['scene_count']}; C12/T8; 256x256; see metadata.json for precision and checkpoints.", "",
        "| Mode | PSNR | SSIM | LPIPS |", "|---|---:|---:|---:|"]
    for name, row in summary["modes"].items():
        lines.append(f"| {name} | {row['psnr']:.5f} | {row['ssim']:.5f} | {row['lpips']:.5f} |")
    slots = summary["native_slot_order_check"]
    lines.append(f"| b0_original_slots_check | {slots['mean_psnr']:.5f} | {slots['mean_ssim']:.5f} | {slots['mean_lpips']:.5f} |")
    lines += ["", f"Paired mean B1 - B0 PSNR: {summary['b1_minus_b0_psnr']:+.5f} dB.", "",
              summary["interpretation"], "", "Native slot-order check: " + json.dumps(summary["native_slot_order_check"]),
              "", "Repeated rasterization: " + json.dumps(summary["renderer_repeat"]), "",
              "The current checkpoint supplies B1/order only. B0 always uses the released native decoder and unprojected geometry.",
              "No historical 24.7004 dB value is substituted for this measurement.", ""]
    with (output / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


@torch.no_grad()
def run(args, manifest, spec):
    if args.output.exists():
        raise FileExistsError("use a fresh B0 output directory")
    if args.precision == "bf16" and (torch.device(args.device).type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("production BF16 evaluation requires CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    source, expected_hashes = diag.source_info(args, load_cache=True)
    native, native_identity = load_native(args)
    path = Path(spec["checkpoint"])
    ref_stat = path.stat()
    reference_identity = dict(path=str(path.resolve()), sha256=diag.file_hash(path), global_step=spec["expected_step"])
    reference, cfg, _ = diag.profile.load_model(spec, args)
    if ref_stat.st_size != path.stat().st_size or ref_stat.st_mtime_ns != path.stat().st_mtime_ns:
        raise ValueError("reference checkpoint changed during loading")
    numerics = numerical_policy()
    if diag.backbone_hash(reference) != source["backbone_sha256"]:
        raise ValueError("reference backbone differs from the original matched-scene analysis")
    pair = validate_model_pair(native, reference)
    hashes = code_hashes()
    metadata = dict(version=1, started_utc=datetime.now(timezone.utc).isoformat(),
        manifest=manifest, manifest_sha256=diag.file_hash(args.manifest), native_checkpoint=native_identity,
        checkpoint=reference_identity, model_pair=pair, source=source, code_sha256=hashes,
        environment=diag.profile.environment(args), numerics=numerics,
        protocol=dict(scenes=args.scenes, tokens=args.tokens, seed=args.seed, precision=args.precision,
            context_views=12, target_views=8, image_shape=[256,256], shared_sorted_order=True,
            b0_geometry_channels=512, b1_geometry_channels=224, training=False, actual_bitstream=False))
    args.output.mkdir(parents=True, exist_ok=False)
    diag.save_json(args.output / "metadata.json", metadata)
    render, metrics = diag.render_functions()
    selection = {name: i for i, name in enumerate(source["inputs"]["scenes"])}
    rows = {}
    for batch in diag.load_data(cfg):
        name = batch["scene_info"]["scene"]
        name = str(name[0] if isinstance(name, (list, tuple)) else name)
        if name not in selection:
            continue
        index = selection[name]
        if index in rows:
            raise ValueError(f"duplicate selected scene: {name}")
        item = dict(scene=name, inputs=diag.profile.to_device(batch["inputs"], "cpu"))
        if diag.profile.input_fingerprint([item]) != expected_hashes[name]:
            raise ValueError(f"input tensor differs from original cache: {name}")
        value = evaluate_scene(native, reference, batch, args, render, metrics)
        value.update(scene=name, index=index)
        diag.save_json(args.output / "scenes" / f"scene_{index:04d}.json", value)
        rows[index] = value
        print(f"B0_MATCHED_SCENE={len(rows)}/{args.scenes} scene={name} "
              f"b0={value['modes']['b0_native']['psnr']:.5f} b1={value['modes']['b1_bridge']['psnr']:.5f}", flush=True)
        if len(rows) == args.scenes:
            break
    if len(rows) != args.scenes:
        raise RuntimeError("selected original scenes are missing; no fallback sampling")
    if code_hashes() != hashes:
        raise RuntimeError("runtime code changed; results are not marked complete")
    summary = summarize([rows[i] for i in range(args.scenes)])
    diag.save_json(args.output / "summary.json", summary)
    write_report(args.output, summary)
    diag.save_json(args.output / "complete.json", dict(scene_count=args.scenes, modes=list(MODES),
        input_fingerprint=source["inputs"]["fingerprint"], checkpoint_sha256=reference_identity["sha256"],
        native_checkpoint_sha256=native_identity["sha256"]))
    print(f"B0_MATCHED_COMPLETE scenes={args.scenes} output={args.output}", flush=True)
    return summary


def compare_runs(b0_run, path_run, output):
    """Join only completed, frame-aligned runs; report the common B1 drift."""
    if output.exists():
        raise FileExistsError("use a fresh comparison output directory")
    b0_meta = diag.read_json(b0_run / "metadata.json")
    path_meta = diag.read_json(path_run / "metadata.json")
    b0_done = diag.read_json(b0_run / "complete.json")
    path_done = diag.read_json(path_run / "complete.json")
    for meta, done, modes in ((b0_meta, b0_done, MODES), (path_meta, path_done, diag.MODES)):
        if (done["checkpoint_sha256"] != meta["checkpoint"]["sha256"]
                or done["input_fingerprint"] != meta["source"]["inputs"]["fingerprint"]
                or done["scene_count"] != len(meta["source"]["inputs"]["scenes"])
                or done["scene_count"] != meta["protocol"]["scenes"]
                or done["modes"] != list(modes)):
            raise ValueError("completion and metadata identities differ")
    if b0_meta["manifest"] != path_meta["manifest"]:
        raise ValueError("reference manifests differ")
    if any(b0_meta["code_sha256"].get(key) != value for key, value in path_meta["code_sha256"].items()):
        raise ValueError("shared runtime code hashes differ")
    if (b0_done["scene_count"] != path_done["scene_count"]
            or b0_done["input_fingerprint"] != path_done["input_fingerprint"]
            or b0_done["checkpoint_sha256"] != path_done["checkpoint_sha256"]):
        raise ValueError("completed run scene/cache/reference-checkpoint identities differ")
    for key in ("scenes", "tokens", "seed", "precision", "context_views", "target_views", "image_shape", "shared_sorted_order"):
        if b0_meta["protocol"][key] != path_meta["protocol"][key]:
            raise ValueError(f"run protocol differs: {key}")
    if b0_meta["numerics"] != path_meta["numerics"]:
        raise ValueError("run numerical backend policies differ")
    if b0_meta["source"]["inputs"] != path_meta["source"]["inputs"]:
        raise ValueError("source scene order/fingerprint differs")
    for key in ("torch", "compressai", "numpy", "gpu", "cuda_runtime"):
        if b0_meta["environment"][key] != path_meta["environment"][key]:
            raise ValueError(f"run environment differs: {key}")
    rows = []
    for i, scene in enumerate(b0_meta["source"]["inputs"]["scenes"]):
        b = diag.read_json(b0_run / "scenes" / f"scene_{i:04d}.json")
        p = diag.read_json(path_run / "scenes" / f"scene_{i:04d}.json")
        if b["scene"] != scene or p["scene"] != scene or b["index"] != i or p["index"] != i:
            raise ValueError("per-scene IDs/index differ")
        for key in ("input_fingerprint", "target_fingerprint", "permutation_sha256", "context_frame_ids", "target_frame_ids"):
            if b[key] != p[key]:
                raise ValueError(f"per-scene matching failed: {scene}/{key}")
        rows.append(dict(scene=scene, b0_native=b["modes"]["b0_native"],
            b1_bridge=b["modes"]["b1_bridge"], t05=p["modes"],
            b1_bridge_minus_t05_b1_psnr=b["modes"]["b1_bridge"]["psnr"] - p["modes"]["b1_direct"]["psnr"],
            t05_actual_minus_native_b0_psnr=p["modes"]["actual_packet"]["psnr"] - b["modes"]["b0_native"]["psnr"]))
    if len(rows) != b0_done["scene_count"] or not rows:
        raise ValueError("completed count and per-scene data differ")
    result = dict(scene_count=len(rows), b0_run=str(b0_run.resolve()), path_run=str(path_run.resolve()),
        mean_b0_psnr=statistics.fmean(row["b0_native"]["psnr"] for row in rows),
        mean_b1_bridge_minus_t05_b1_psnr=statistics.fmean(row["b1_bridge_minus_t05_b1_psnr"] for row in rows),
        max_abs_b1_bridge_minus_t05_b1_psnr=max(abs(row["b1_bridge_minus_t05_b1_psnr"]) for row in rows),
        mean_t05_actual_minus_native_b0_psnr=statistics.fmean(row["t05_actual_minus_native_b0_psnr"] for row in rows),
        t05_modes={mode: {key: statistics.fmean(row["t05"][mode][key] for row in rows)
                          for key in ("psnr", "ssim", "lpips")} for mode in diag.MODES},
        gpu_uuid_equal=b0_meta["environment"]["gpu_uuid"] == path_meta["environment"]["gpu_uuid"],
        renderer_repeat=dict(b0=diag.read_json(b0_run / "summary.json")["renderer_repeat"],
                             t05=diag.read_json(path_run / "summary.json")["renderer_repeat"]),
        interpretation="Aligned subset comparison; inspect the common B1 bridge drift and renderer repeat errors. "
            "Identical weights/inputs do not assert cross-device pixel bit-exactness or additive independent losses.")
    output.mkdir(parents=True, exist_ok=False)
    diag.save_json(output / "summary.json", result)
    diag.save_json(output / "paired_scenes.json", rows)
    with (output / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("# Aligned B0 / T05 comparison\n\n" +
            f"Scenes: {len(rows)}; B0 PSNR: {result['mean_b0_psnr']:.5f}.\n\n" +
            f"Common B1 bridge mean/max absolute PSNR drift: {result['mean_b1_bridge_minus_t05_b1_psnr']:+.6f} / " +
            f"{result['max_abs_b1_bridge_minus_t05_b1_psnr']:.6f} dB.\n\n" +
            f"Actual - native B0 PSNR: {result['mean_t05_actual_minus_native_b0_psnr']:+.5f} dB.\n\n" +
            result["interpretation"] + "\n")
    print(f"B0_T05_COMPARISON_COMPLETE scenes={len(rows)} output={output}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "config/plans/nfcgs_path_diagnostics_20261019.json")
    parser.add_argument("--native-checkpoint", type=Path, default=REPO / "checkpoints/pretrained/globalsplat-re10k-32k.ckpt")
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
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--b0-run", type=Path)
    parser.add_argument("--path-run", type=Path)
    args = parser.parse_args(argv)
    if args.compare_only:
        if args.plan or args.b0_run is None or args.path_run is None or args.output is None:
            parser.error("--compare-only requires --b0-run, --path-run and --output, without --plan")
    elif args.b0_run is not None or args.path_run is not None:
        parser.error("--b0-run/--path-run require --compare-only")
    if min(args.scenes, args.tokens, args.threads) < 1 or args.num_workers < 0:
        parser.error("positive scene/token/thread counts and nonnegative workers required")
    if not args.plan and args.output is None:
        parser.error("--output is required unless --plan")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.compare_only:
        compare_runs(args.b0_run, args.path_run, args.output)
        return
    manifest, spec = diag.load_spec(args.manifest)
    if args.tokens != spec["tokens"]:
        raise ValueError("token count differs from reference manifest")
    if not args.native_checkpoint.is_file():
        raise FileNotFoundError(args.native_checkpoint)
    if args.plan:
        source, _ = diag.source_info(args, load_cache=False)
        print(f"B0_MATCHED_PLAN native={args.native_checkpoint} reference={spec['checkpoint']} "
              f"scenes={args.scenes} input_fingerprint={source['inputs']['fingerprint']}")
        print("PLAN_ONLY=True DATASET_AND_MODEL_CHECKS=deferred_to_compute_node")
        return
    if not (args.dataset_root / "test/index.json").is_file():
        raise FileNotFoundError(args.dataset_root / "test/index.json")
    run(args, manifest, spec)


if __name__ == "__main__":
    main()

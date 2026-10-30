#!/usr/bin/env python3
"""T15: receive completed T05 packets without constructing a scene encoder.

prepare runs the dataset loader in its own process and exports targets only.
receive needs only that portable bundle and the shared checkpoint; it does not
open the dataset, original cache or T05 run. No new sender or training is needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval_nfcgs_path_diagnostics as diag
from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec
from globalsplat.compression.bitstream import SceneBitstream
from globalsplat.compression.checkpoint import (
    check_retired_metadata, resize_registered_buffers, strip_retired_state, validate_feature_codec_checkpoint,
)
from globalsplat.model.decoder.gaussian_decoder import TokenCoarseToFine3DGS
from globalsplat.model.types import Gaussians

METRICS = ("psnr", "ssim", "lpips")
TOLERANCES = dict(psnr=1e-3, ssim=1e-5, lpips=1e-5)
GAUSSIAN_FIELDS = ("means", "scales", "rotations", "sh", "opacities")
FINGERPRINT = "069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29"


def code_hashes():
    result = diag.code_hashes()
    for name in ("scripts/eval_nfcgs_independent_receiver.py", "scripts/slurm/eval_nfcgs_independent_receiver.slurm"):
        result[name] = diag.file_hash(REPO / name)
    return result


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"artifact path escapes bundle/run: {relative}")
    return path


def numerical_policy():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return dict(cudnn_version=torch.backends.cudnn.version(), cudnn_deterministic=True,
        cudnn_benchmark=False, cudnn_allow_tf32=False, matmul_allow_tf32=False,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def shared_code_matches(reference, current):
    mismatches = [name for name, digest in reference.items() if current.get(name) != digest]
    if mismatches:
        raise ValueError(f"runtime code differs from source: {mismatches[:5]}")


def validate_packet(payload, row, config, tokens):
    if len(payload) != row["actual_bytes"] or hashlib.sha256(payload).hexdigest() != row["packet_sha256"]:
        raise ValueError("packet length/SHA-256 mismatch")
    packet = SceneBitstream.unpack(payload)
    if (packet.points != tokens or packet.channels != config.observable_channels or packet.rank != config.rank
            or packet.bytes_by_stream != row["stream_bytes"]
            or sum(packet.bytes_by_stream.values()) != len(payload)):
        raise ValueError("packet dimensions/stream accounting mismatch")
    return packet


def read_source(args):
    manifest, spec = diag.load_spec(args.manifest)
    source = args.path_run.resolve()
    meta = diag.read_json(source / "metadata.json")
    done = diag.read_json(source / "complete.json")
    protocol = meta["protocol"]
    expected = dict(scenes=256, tokens=4096, seed=0, precision="bf16", context_views=12,
                    target_views=8, image_shape=[256, 256], shared_sorted_order=True)
    if (meta["version"] != 1 or meta["manifest"] != manifest
            or any(protocol.get(k) != v for k, v in expected.items())
            or done["scene_count"] != 256 or done["modes"] != list(diag.MODES)
            or done["input_fingerprint"] != FINGERPRINT
            or meta["source"]["inputs"]["fingerprint"] != FINGERPRINT
            or done["checkpoint_sha256"] != meta["checkpoint"]["sha256"]
            or meta["checkpoint"]["global_step"] != 50000):
        raise ValueError("source is not the completed matched256 T05 final50k run")
    names = meta["source"]["inputs"]["scenes"]
    if len(names) != 256 or len(set(names)) != 256:
        raise ValueError("source scene IDs/count mismatch")
    hashes = code_hashes()
    shared_code_matches(meta["code_sha256"], hashes)
    if diag.file_hash(spec["checkpoint"]) != meta["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA-256 differs from source T05")
    config = CodecConfig.from_mapping(spec["config"])
    rows = []
    for i, name in enumerate(names):
        row = diag.read_json(source / f"scenes/scene_{i:04d}.json")
        if row["index"] != i or row["scene"] != name or not row["packet_decomposition_exact"]:
            raise ValueError("source scene identity/decomposition check mismatch")
        actual = row["modes"]["actual_packet"]
        if not all(math.isfinite(actual[k]) for k in METRICS):
            raise ValueError("non-finite reference metrics")
        record = dict(index=i, scene=name, packet_sha256=row["packet_sha256"],
            actual_bytes=actual["actual_bytes"], stream_bytes=actual["stream_bytes"],
            packet_file=row["packet_file"], target_frame_ids=row["target_frame_ids"],
            target_fingerprint=row["target_fingerprint"], reference={k: actual[k] for k in METRICS})
        validate_packet(contained(source, record["packet_file"]).read_bytes(), record, config, 4096)
        rows.append(record)
    return manifest, spec, meta, rows, hashes


def dataset_config(args):
    # Import the loader only in the preparation process. Never construct a model.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "config")):
        cfg = compose(config_name="main", overrides=["+experiment=re10k_32k_nfcgs", "dataset=re10k_eval_all_ctx12"])
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.dataset_roots = [str(args.dataset_root.resolve())]
    cfg.dataset.mvsplat_root = str(REPO / "third_party/ZPressor/mvsplat")
    cfg.dataset.augment = False
    cfg.dataset.data_loader_seed = 0
    cfg.optimizer.batch_size = 1
    cfg.optimizer.num_workers = args.num_workers
    cfg.seed = 0
    return cfg


def validate_targets(targets, row):
    if tuple(targets["images"].shape) != (1, 8, 3, 256, 256):
        raise ValueError("expected T8 256x256 target images")
    if diag.frame_ids(targets) != row["target_frame_ids"]:
        raise ValueError(f"target frames differ: {row['scene']}")
    if diag.profile.input_fingerprint([targets]) != row["target_fingerprint"]:
        raise ValueError(f"target images/cameras differ: {row['scene']}")


@torch.no_grad()
def prepare(args):
    if args.bundle.exists():
        raise FileExistsError("use a fresh bundle directory")
    manifest, spec, meta, rows, hashes = read_source(args)
    if not (args.dataset_root / "test/index.json").is_file():
        raise FileNotFoundError(args.dataset_root / "test/index.json")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    torch.set_num_threads(args.threads)
    selected = {row["scene"]: row for row in rows}
    args.bundle.mkdir(parents=True, exist_ok=False)
    exported = {}
    for batch in diag.load_data(dataset_config(args)):
        name = batch["scene_info"]["scene"]
        name = str(name[0] if isinstance(name, (list, tuple)) else name)
        if name not in selected:
            continue
        if name in exported:
            raise ValueError(f"duplicate target scene: {name}")
        row = dict(selected[name])
        targets = diag.profile.to_device(batch["targets"], "cpu")
        validate_targets(targets, row)
        i = row["index"]
        target_path = args.bundle / f"targets/scene_{i:04d}.pt"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(scene=name, index=i, targets=targets), target_path)
        row.update(target_file=target_path.relative_to(args.bundle).as_posix(), target_sha256=diag.file_hash(target_path))
        payload = contained(args.path_run, row["packet_file"]).read_bytes()
        validate_packet(payload, row, CodecConfig.from_mapping(spec["config"]), spec["tokens"])
        row["packet_file"] = f"packets/scene_{i:04d}.bin"
        packet_path = args.bundle / row["packet_file"]
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        with packet_path.open("xb") as stream:
            stream.write(payload)
        exported[name] = row
        print(f"RECEIVER_TARGET_EXPORT={len(exported)}/{len(rows)} scene={name}", flush=True)
        if len(exported) == len(rows):
            break
    if set(exported) != set(selected):
        raise RuntimeError(f"missing target scenes: {sorted(set(selected)-set(exported))}")
    if code_hashes() != hashes:
        raise RuntimeError("runtime source changed during target export")
    metadata = dict(version=1, checkpoint=meta["checkpoint"], config=spec["config"], protocol=meta["protocol"],
        source_path_run=str(args.path_run.resolve()), source_complete=diag.read_json(args.path_run / "complete.json"),
        source_environment=meta["environment"], source_numerics=meta["numerics"],
        manifest=manifest, code_sha256=hashes, reference_scope="completed T05 actual_packet, matched256",
        decoder=dict(C=spec["config"]["texture_channels"], geometry_C=spec["config"]["geometry_observable_channels"],
                     sh_degree=3, M_max=16, stage=3, mix=1.0),
        payload_scope="packets plus targets only; no context images, encoded features, permutation or Gaussian tensors")
    diag.save_json(args.bundle / "metadata.json", metadata)
    diag.save_json(args.bundle / "manifest.json", dict(version=1, rows=[exported[row["scene"]] for row in rows]))
    diag.save_json(args.bundle / "complete.json", dict(scene_count=len(rows),
        metadata_sha256=diag.file_hash(args.bundle / "metadata.json"),
        manifest_sha256=diag.file_hash(args.bundle / "manifest.json")))
    print(f"RECEIVER_BUNDLE_COMPLETE scenes={len(rows)} bundle={args.bundle}", flush=True)


def load_bundle(bundle):
    meta = diag.read_json(bundle / "metadata.json")
    manifest = diag.read_json(bundle / "manifest.json")
    done = diag.read_json(bundle / "complete.json")
    if (meta["version"] != 1 or manifest["version"] != 1
            or done["metadata_sha256"] != diag.file_hash(bundle / "metadata.json")
            or done["manifest_sha256"] != diag.file_hash(bundle / "manifest.json")):
        raise ValueError("bundle metadata/manifest hash mismatch")
    rows = manifest["rows"]
    if (not rows or len(rows) != done["scene_count"] or len(rows) != meta["protocol"]["scenes"]
            or len({row["scene"] for row in rows}) != len(rows)
            or [row["index"] for row in rows] != list(range(len(rows)))):
        raise ValueError("bundle scene count/order mismatch")
    shared_code_matches(meta["code_sha256"], code_hashes())
    return meta, rows


class PacketReceiver(nn.Module):
    """Only two modules; no GlobalSplat object, encoder or scene-token bank."""
    def __init__(self, config, decoder):
        super().__init__()
        self.codec = ObservableLowRank1DCodec(config)
        self.gaussian_decoder = TokenCoarseToFine3DGS(**{k: decoder[k] for k in ("C", "geometry_C", "sh_degree", "M_max")})
        self.gaussian_decoder.set_stage(decoder["stage"], mix=decoder["mix"])

    def decode(self, payload):
        tokens = self.codec.decompress(payload)
        means, rotations, scales, sh, opacities, reg = self.gaussian_decoder(tokens)
        return tokens, Gaussians(means=means, scales=scales, rotations=rotations, sh=sh, opacities=opacities, reg=reg)


def load_receiver(checkpoint_path, meta, device):
    if diag.file_hash(checkpoint_path) != meta["checkpoint"]["sha256"]:
        raise ValueError("receiver checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") != 50000:
        raise ValueError("receiver requires final50k checkpoint")
    config = CodecConfig.from_mapping(meta["config"])
    validate_feature_codec_checkpoint(checkpoint, config)
    if CodecConfig.from_mapping(checkpoint["feature_codec_config"]).to_dict() != config.to_dict():
        raise ValueError("receiver codec config mismatch")
    state = checkpoint["state_dict"]
    bank = state["model.scene_tokens"]
    if tuple(bank.shape) != (meta["protocol"]["tokens"], config.geometry_channels):
        raise ValueError("checkpoint scene-token dimensions mismatch")
    receiver = PacketReceiver(config, meta["decoder"])
    if meta["protocol"].get("score_mean_offset_enabled", True) is not True:
        raise ValueError("T05 used the retired no-mean-offset setting")
    check_retired_metadata(checkpoint)
    for module, prefix in ((receiver.codec, "model.feature_codec."), (receiver.gaussian_decoder, "model.gaussian_decoder.")):
        selected = strip_retired_state({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
        resize_registered_buffers(module, selected)
        module.load_state_dict(selected, strict=True)
    del checkpoint, state, bank, selected
    receiver.requires_grad_(False).eval().to(device)
    receiver.codec.update(force=True, update_quantiles=False)
    return receiver


def validate_environment(meta, args):
    current = diag.profile.environment(args)
    for key in ("torch", "compressai", "numpy", "gpu", "cuda_runtime"):
        if current[key] != meta["source_environment"][key]:
            raise ValueError(f"receiver/T05 environment mismatch: {key}")
    numerics = numerical_policy()
    if numerics != meta["source_numerics"]:
        raise ValueError("receiver/T05 numerical backend policy differs")
    return current, numerics


def summarize(rows):
    result = dict(scene_count=len(rows), tolerances=TOLERANCES,
        averages={k: statistics.fmean(row["metrics"][k] for row in rows) for k in METRICS},
        reference_averages={k: statistics.fmean(row["reference"][k] for row in rows) for k in METRICS},
        mean_delta={k: statistics.fmean(row["delta"][k] for row in rows) for k in METRICS},
        max_abs_delta={k: max(abs(row["delta"][k]) for row in rows) for k in METRICS},
        mean_bytes=statistics.fmean(row["actual_bytes"] for row in rows),
        renderer_repeat_max_pixel_abs=max(row["renderer_repeat_max_pixel_abs"] for row in rows),
        feature_repeat_exact=all(row["feature_repeat_exact"] for row in rows),
        gaussian_repeat_exact=all(row["gaussian_repeat_exact"] for row in rows))
    result["passed"] = (result["feature_repeat_exact"] and result["gaussian_repeat_exact"]
        and all(result["max_abs_delta"][k] <= TOLERANCES[k] for k in METRICS))
    return result


def write_report(output, summary):
    lines = ["# T15 independent packet receiver", "", f"Status: {'PASS' if summary['passed'] else 'FAIL'}; scenes: {summary['scene_count']}; C12/T8; 256x256.", "",
        "Completed T05 packets are decoded by a fresh process containing only the codec and Gaussian decoder.",
        "Context images, encoded features, sort positions/permutations and sender Gaussians are absent from its bundle.",
        "Target cameras and image dimensions enter rendering; target RGB enters metrics only.", "",
        "| Metric | T05 actual | Independent receiver | Mean delta | Max scene abs delta | Tolerance |",
        "|---|---:|---:|---:|---:|---:|"]
    for key in METRICS:
        lines.append(f"| {key} | {summary['reference_averages'][key]:.7f} | {summary['averages'][key]:.7f} | "
            f"{summary['mean_delta'][key]:+.7f} | {summary['max_abs_delta'][key]:.7f} | {TOLERANCES[key]} |")
    lines += ["", f"Actual KiB/scene: {summary['mean_bytes']/1024:.5f}.",
        f"Repeated packet feature / Gaussian exact: {summary['feature_repeat_exact']} / {summary['gaussian_repeat_exact']}.",
        f"Repeated rasterization max pixel abs: {summary['renderer_repeat_max_pixel_abs']}.", "",
        "This is a matched256 interoperability check, not full-test RD or a new model comparison.",
        "Historical T05 pixels/features were not saved; metric agreement does not assert historical pixel/feature bit-exactness.",
        "The checkpoint is a pre-shared model asset; its size is excluded from per-scene packet rate.",
        "No receiver latency or minimum-memory claim is made; checkpoint loading still reads the integrated file on CPU.", ""]
    with (output / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


@torch.no_grad()
def receive(args):
    if args.output.exists():
        raise FileExistsError("use a fresh receiver output directory")
    meta, rows = load_bundle(args.bundle)
    if args.precision != meta["protocol"]["precision"]:
        raise ValueError("receiver precision differs from source")
    if args.precision == "bf16" and (torch.device(args.device).type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("production BF16 receiver requires CUDA")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    torch.set_num_threads(args.threads)
    environment, numerics = validate_environment(meta, args)
    receiver = load_receiver(args.checkpoint, meta, args.device)
    hashes = code_hashes()
    args.output.mkdir(parents=True, exist_ok=False)
    diag.save_json(args.output / "metadata.json", dict(version=1, started_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=meta["checkpoint"], bundle=str(args.bundle.resolve()), bundle_complete=diag.read_json(args.bundle/"complete.json"),
        environment=environment, numerics=numerics, code_sha256=hashes, protocol=meta["protocol"],
        modules=list(dict(receiver.named_children())), encoder_constructed=False, tolerances=TOLERANCES))
    render, metrics = diag.render_functions()
    results = []
    config = CodecConfig.from_mapping(meta["config"])
    for row in rows:
        payload = contained(args.bundle, row["packet_file"]).read_bytes()
        validate_packet(payload, row, config, meta["protocol"]["tokens"])
        target_path = contained(args.bundle, row["target_file"])
        if diag.file_hash(target_path) != row["target_sha256"]:
            raise ValueError("target bundle file SHA-256 mismatch")
        item = torch.load(target_path, map_location="cpu", weights_only=False)
        if set(item) != {"scene", "index", "targets"} or item["scene"] != row["scene"] or item["index"] != row["index"]:
            raise ValueError("receiver target artifact must contain only targets and scene/index")
        targets = item["targets"]
        validate_targets(targets, row)
        with diag.amp(args):
            tokens, gaussians = receiver.decode(payload)
            repeated_tokens, repeated_gaussians = receiver.decode(payload)
        tensors = [*tokens, *(getattr(gaussians, key) for key in GAUSSIAN_FIELDS)]
        if not all(torch.isfinite(t).all() for t in tensors):
            raise RuntimeError("non-finite reconstructed features/Gaussians")
        if gaussians.num_gaussians != meta["protocol"]["tokens"] * 8:
            raise ValueError("expected eight Gaussians per scene token")
        # Render receives camera tensors and a zero shape placeholder, never GT RGB.
        cameras = {key: diag.profile.to_device(targets[key], args.device) for key in ("intrinsic", "extrinsic")}
        cameras["images"] = torch.zeros(targets["images"].shape, dtype=torch.float32, device=args.device)
        pred = render(gaussians, cameras)
        repeated_pred = render(gaussians, cameras)
        gt = targets["images"][0].to(args.device).float().clamp(0, 1)
        measured = metrics(gt, pred)
        if not all(math.isfinite(measured[k]) for k in METRICS):
            raise RuntimeError("non-finite receiver metrics")
        value = dict(index=row["index"], scene=row["scene"], packet_sha256=row["packet_sha256"],
            actual_bytes=row["actual_bytes"], stream_bytes=row["stream_bytes"], metrics=measured, reference=row["reference"],
            delta={k: measured[k]-row["reference"][k] for k in METRICS},
            target_fingerprint=row["target_fingerprint"], target_frame_ids=row["target_frame_ids"],
            decoded_features_sha256=diag.profile.input_fingerprint(list(tokens)),
            gaussian_sha256=diag.profile.input_fingerprint([{k: getattr(gaussians, k) for k in GAUSSIAN_FIELDS}]),
            feature_repeat_exact=all(torch.equal(a,b) for a,b in zip(tokens,repeated_tokens)),
            gaussian_repeat_exact=all(torch.equal(getattr(gaussians,k),getattr(repeated_gaussians,k)) for k in GAUSSIAN_FIELDS),
            renderer_repeat_max_pixel_abs=float((pred-repeated_pred).abs().max()))
        diag.save_json(args.output / f"scenes/scene_{row['index']:04d}.json", value)
        results.append(value)
        print(f"INDEPENDENT_RECEIVER_SCENE={len(results)}/{len(rows)} scene={row['scene']} "
            f"psnr={measured['psnr']:.5f} delta={value['delta']['psnr']:+.7f}", flush=True)
        del tokens, repeated_tokens, gaussians, repeated_gaussians, pred, repeated_pred, tensors, gt, cameras, targets, item
    if code_hashes() != hashes or load_bundle(args.bundle) != (meta, rows):
        raise RuntimeError("runtime code or bundle changed during receive")
    if diag.file_hash(args.checkpoint) != meta["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint changed during receive")
    summary = summarize(results)
    diag.save_json(args.output / "summary.json", summary)
    write_report(args.output, summary)
    if not summary["passed"]:
        raise RuntimeError("receiver comparison failed; inspect REPORT.md; no complete marker written")
    diag.save_json(args.output / "complete.json", dict(scene_count=len(results), passed=True,
        checkpoint_sha256=meta["checkpoint"]["sha256"], summary_sha256=diag.file_hash(args.output / "summary.json")))
    print(f"INDEPENDENT_RECEIVER_COMPLETE scenes={len(results)} output={args.output}", flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "prepare", "receive"))
    parser.add_argument("--path-run", type=Path, default=REPO / "outputs/nfcgs_path_diagnostics/job_458004")
    parser.add_argument("--manifest", type=Path, default=REPO / "config/plans/nfcgs_path_diagnostics_20261019.json")
    parser.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.num_workers < 0:
        parser.error("positive threads/nonnegative workers required")
    if args.mode != "plan" and args.bundle is None:
        parser.error("--bundle required")
    if args.mode == "receive" and (args.checkpoint is None or args.output is None):
        parser.error("receive requires --checkpoint and --output")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "plan":
        _, spec, _, rows, _ = read_source(args)
        print(f"INDEPENDENT_RECEIVER_PLAN scenes={len(rows)} checkpoint={spec['checkpoint']} source={args.path_run}")
        print("TRAINING=False REENCODE=False RECEIVER_ENCODER=False DATASET_CHECK=deferred_to_compute_node")
    elif args.mode == "prepare":
        prepare(args)
    else:
        receive(args)


if __name__ == "__main__":
    main()

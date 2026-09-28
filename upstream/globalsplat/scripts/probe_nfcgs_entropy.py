#!/usr/bin/env python3
"""Small train/test entropy probe using a frozen integrated Full+Split checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from globalsplat.compression.bitstream import SceneBitstream
from globalsplat.compression.checkpoint import resize_registered_buffers, validate_feature_codec_checkpoint
from globalsplat.compression.config import CodecConfig
from globalsplat.compression.entropy_probe import EntropyProbe


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


def scene_name(batch):
    value = batch["scene_info"]["scene"]
    return str(value[0] if isinstance(value, (list, tuple)) else value)


def load_model_and_data(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from globalsplat.main import build_model, build_datamodule

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "feature_codec_config" not in checkpoint:
        raise ValueError("probe requires checkpoint feature_codec_config metadata")
    config = CodecConfig.from_mapping(checkpoint["feature_codec_config"])
    validate_feature_codec_checkpoint(checkpoint, config)
    if args.expected_step is not None and checkpoint.get("global_step") != args.expected_step:
        raise ValueError(f"wrong checkpoint step: {checkpoint.get('global_step')}")
    state = checkpoint["state_dict"]
    model_state = {key[len("model."):]: value for key, value in state.items()
                   if key.startswith("model.")}
    if not model_state:
        raise ValueError("an integrated model.* checkpoint is required")
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "config")):
        cfg = compose(config_name="main", overrides=[
            "+experiment=re10k_32k_nfcgs", "dataset=re10k_eval_all_ctx12"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.feature_codec = OmegaConf.create(config.to_dict())
    cfg.model.dim_latents = config.geometry_channels
    cfg.model.latent_rep_token_amount = int(model_state["scene_tokens"].shape[0])
    cfg.model.freeze_globalsplat = True
    cfg.model.feature_codec_train_scope = "all"
    cfg.dataset.dataset_roots = [str(args.dataset_root.resolve())]
    cfg.dataset.mvsplat_root = str(REPO_ROOT / "third_party/ZPressor/mvsplat")
    cfg.dataset.augment = False
    cfg.dataset.data_loader_seed = args.seed
    cfg.optimizer.batch_size = 1
    cfg.optimizer.num_workers = args.num_workers
    cfg.seed = args.seed
    model = build_model(cfg.model)
    resize_registered_buffers(model, model_state)
    model.load_state_dict(model_state, strict=True)
    model.requires_grad_(False).eval().to(args.device)
    model.set_stage(3, mix=1.0)
    model.feature_codec.update(force=True, update_quantiles=False)
    data, _ = build_datamodule(cfg)
    metadata = dict(checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_step=checkpoint.get("global_step"), codec=config.to_dict(),
                    dataset=OmegaConf.to_container(cfg.dataset, resolve=True))
    return model, data, metadata


def mean_dict(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def summarize(rows, seed):
    if not rows:
        raise ValueError("cannot summarize an empty probe")
    if not all(row.get("exact_feature_roundtrip", False) for row in rows):
        raise ValueError("only verified scenes may enter the final comparison")
    baseline = mean_dict([row["baseline"] for row in rows])
    variants = {}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), (1000, len(rows)))
    for variant in ("marginal", "context"):
        values = [row["variants"][variant] for row in rows]
        summary = mean_dict(values)
        # Resample complete scenes, keeping their baseline/candidate pairing.
        pairs = np.asarray([[row["baseline"]["total_bytes"],
                             row["variants"][variant]["total_bytes"],
                             row["variants"][variant]["score_only_total_bytes"],
                             row["variants"][variant]["residual_only_total_bytes"]] for row in rows])
        boot = pairs[draws].mean(axis=1)
        gain = 100 * (1 - boot[:, 1] / boot[:, 0])
        score_gain = 100 * (1 - boot[:, 2] / boot[:, 0])
        residual_gain = 100 * (1 - boot[:, 3] / boot[:, 0])
        summary["total_saving_pct"] = 100 * (1 - summary["total_bytes"] / baseline["total_bytes"])
        summary["total_saving_pct_bootstrap95"] = np.quantile(gain, [0.025, 0.975]).tolist()
        summary["score_only_saving_total_pct"] = 100 * (
            baseline["score_bytes"] - summary["score_bytes"]) / baseline["total_bytes"]
        summary["residual_only_saving_total_pct"] = 100 * (
            baseline["residual_bytes"] - summary["residual_bytes"]) / baseline["total_bytes"]
        summary["score_only_saving_total_pct_bootstrap95"] = np.quantile(
            score_gain, [0.025, 0.975]).tolist()
        summary["residual_only_saving_total_pct_bootstrap95"] = np.quantile(
            residual_gain, [0.025, 0.975]).tolist()
        summary["score_minus_residual_saving_total_pp"] = (
            summary["score_only_saving_total_pct"] - summary["residual_only_saving_total_pct"])
        summary["score_minus_residual_saving_total_pp_bootstrap95"] = np.quantile(
            score_gain - residual_gain, [0.025, 0.975]).tolist()
        variants[variant] = summary
    streams = {name: mean_dict([row["streams"][name] for row in rows])
               for name in rows[0]["streams"]}
    for values in streams.values():
        values["rans_gap_bits"] = values["actual_bits"] - values["cdf_bits"] - values["bypass_bits"]
        values["table_plus_bypass_minus_model_bits"] = (
            values["cdf_bits"] + values["bypass_bits"] - values["model_nll_bits"])
    aggregate = {}
    for family in ("score", "residual"):
        members = [value for name, value in streams.items() if name.startswith(family + "_")]
        aggregate[family] = {key: sum(value[key] for value in members) for key in members[0]}
    raw_bytes = sum(value["actual_bits"] for value in streams.values()) / 8
    overhead = dict(
        scene_mean_bytes=baseline["mean_bytes"],
        outer_and_residual_container_bytes=baseline["container_bytes"],
        score_container_bytes=baseline["score_bytes"] - aggregate["score"]["actual_bits"] / 8)
    accounted = raw_bytes + sum(overhead.values())
    if not np.isclose(accounted, baseline["total_bytes"], atol=1e-6, rtol=0):
        raise RuntimeError("stream/mean/container accounting does not sum to scene bytes")
    return dict(baseline=baseline, variants=variants, streams=streams,
                aggregate_streams=aggregate, overhead=overhead,
                exact_feature_roundtrip_all=True)


def write_report(path, report):
    metrics = report["metrics"]
    base = metrics["baseline"]
    text = ["# Frozen-symbol entropy probe", "", f"Checkpoint: {report['checkpoint']}", "",
            f"Fit: {len(report['fit_scene_ids'])} unique TRAIN scenes; "
            f"evaluation: {len(report['eval_scene_ids'])} unique TEST scenes.", "",
            "| Model | Total KiB | Score KiB | Residual y+z KiB | Total saving |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Original Split | {base['total_bytes']/1024:.3f} | {base['score_bytes']/1024:.3f} | "
            f"{base['residual_bytes']/1024:.3f} | 0% |"]
    for name, value in metrics["variants"].items():
        text.append(f"| {name} | {value['total_bytes']/1024:.3f} | "
                    f"{value['score_bytes']/1024:.3f} | {value['residual_bytes']/1024:.3f} | "
                    f"{value['total_saving_pct']:.3f}% |")
    text += ["", "## Which path has reducible probability cost?", "",
             "Savings below are percentage points of the ORIGINAL TOTAL scene bytes. "
             "They include a 95% paired-scene bootstrap interval; positive means smaller.", "",
             "| Candidate | Score-only saving | Residual-only saving | Score minus residual |",
             "| --- | ---: | ---: | ---: |"]
    for name, value in metrics["variants"].items():
        cells = []
        for key in ("score_only_saving_total_pct", "residual_only_saving_total_pct",
                    "score_minus_residual_saving_total_pp"):
            low, high = value[key + "_bootstrap95"]
            cells.append(f"{value[key]:+.3f} [{low:+.3f}, {high:+.3f}] pp")
        text.append(f"| {name} | " + " | ".join(cells) + " |")
    text += ["", "## Where do the coded bits go?", "",
             "Bits per scene below EXCLUDE all containers and the scene mean. "
             "Model NLL is evaluated at the exact encoded integer symbols, with the native likelihood floor.", "",
             "| Stream | Model NLL | Integer CDF | Tail bypass | Actual | Actual - CDF - bypass |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in metrics["streams"].items():
        text.append(f"| {name} | {value['model_nll_bits']:.1f} | {value['cdf_bits']:.1f} | "
                    f"{value['bypass_bits']:.1f} | {value['actual_bits']:.1f} | "
                    f"{value['rans_gap_bits']:.1f} |")
    overhead = metrics["overhead"]
    text += ["", f"Separate bytes per scene: mean={overhead['scene_mean_bytes']:.1f}, "
             f"score wrapper={overhead['score_container_bytes']:.1f}, "
             f"outer/residual wrappers={overhead['outer_and_residual_container_bytes']:.1f}.", ""]
    text += ["", "All integer streams and decoded features matched exactly on every evaluated scene.",
             "The receiver recomputed its own context using only mean and decoded anchors.", "",
             "Model CDF tables are shared metadata, excluded from per-scene payload; their compressed file size "
             "is reported separately in summary.json. These packets require the diagnostic decoder and are "
             "not a deployable production bitstream variant.", "",
             "This small test subset is a screening result, not the previous full-test average. "
             "Bootstrap intervals only reflect the selected scenes. No PSNR/rendering was rerun.", "",
             "A positive held-out saving establishes an opportunity under this particular probability model. "
             "A negative result does not establish an entropy lower bound or rule out stronger models.", "",
             "The context candidate expands score conditioning only; residual remains marginal recalibration. "
             "A path comparison ranks these particular probes, not the globally best achievable score/residual models. "
             "This diagnostic does not measure deployment decoding speed; Python/CDF-conversion time is not a codec benchmark.", "",
             "See summary.json for per-stream float-model NLL, integer-CDF cost, escape bypass cost, "
             "actual bits, CDF-table bytes and paired-scene bootstrap intervals.", ""]
    path.write_text("\n".join(text), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--fit-scenes", type=int, default=128)
    p.add_argument("--eval-scenes", type=int, default=128)
    p.add_argument("--prior-strength", type=float, default=4096.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--expected-step", type=int, default=10000)
    p.add_argument("--cache-scenes-per-split", type=int, default=0,
                   help="opt-in integer-symbol caches for the first N scenes of EACH split; default 0")
    args = p.parse_args()
    if (min(args.fit_scenes, args.eval_scenes, args.threads) < 1
            or not np.isfinite(args.prior_strength) or args.prior_strength <= 0):
        p.error("scene counts, threads and prior strength must be positive")
    if args.num_workers < 0:
        p.error("num-workers must be nonnegative")
    if args.cache_scenes_per_split < 0:
        p.error("cache-scenes-per-split must be nonnegative")
    if args.precision == "bf16" and not args.device.startswith("cuda"):
        p.error("bf16 probe requires CUDA; use fp32 for CPU")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    for stage in ("train", "test"):
        if not (args.dataset_root / stage / "index.json").is_file():
            raise FileNotFoundError(f"missing {stage}/index.json under {args.dataset_root}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("use a fresh output directory; existing results are preserved")
    start = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    model, data, metadata = load_model_and_data(args)
    import compressai
    metadata.update(schema_version=2, environment=dict(torch=torch.__version__,
                    compressai=compressai.__version__, device=args.device,
                    gpu=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None),
                    candidate_scope=dict(marginal="score and residual probability recalibration",
                                         context="score context bins; residual uses marginal tables"))
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    cached = {"train": [], "test": []}
    if args.cache_scenes_per_split:
        for split in cached:
            (args.output_dir / "symbol_cache" / split).mkdir(parents=True)

    def cache_scene(probe, split, payload, batch):
        if len(cached[split]) >= args.cache_scenes_per_split:
            return
        relative = Path("symbol_cache") / split / f"scene_{len(cached[split]):04d}.npz"
        probe.save_scene_cache(args.output_dir / relative, payload)
        cached[split].append(dict(scene=scene_name(batch), path=relative.as_posix(),
            context_frame_ids=batch["inputs"]["frame_ids"].reshape(-1).tolist()))
        (args.output_dir / "symbol_cache" / "manifest.json").write_text(
            json.dumps(dict(checkpoint=metadata["checkpoint"], splits=cached), indent=2), encoding="utf-8")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.precision == "bf16" else nullcontext
    codec = model.feature_codec
    fit_ids, rows, eval_ids = [], [], []
    print(f"PROBE_START checkpoint={args.checkpoint} fit={args.fit_scenes} eval={args.eval_scenes}", flush=True)
    with EntropyProbe(codec, args.prior_strength) as probe:
        fit_seen = set()
        loader = data.train_dataloader()
        for batch in loader:
            name = scene_name(batch)
            if name in fit_seen:
                continue
            probe.begin()
            with amp():
                compressed = model.compress_scene(to_device(batch["inputs"], args.device))
            probe.end()
            probe.fit_scene()
            cache_scene(probe, "train", compressed.data, batch)
            fit_seen.add(name)
            fit_ids.append(name)
            if len(fit_ids) % 16 == 0:
                print(f"FIT scenes={len(fit_ids)}/{args.fit_scenes} elapsed={time.monotonic()-start:.1f}s", flush=True)
            if len(fit_ids) == args.fit_scenes:
                break
        del loader
        if len(fit_ids) != args.fit_scenes:
            raise RuntimeError("train loader exhausted before requested unique scene count")
        (args.output_dir / "fit_scene_ids.json").write_text(json.dumps(fit_ids), encoding="utf-8")
        probe.freeze()
        tables_path = args.output_dir / "probability_tables.json.gz"
        with gzip.open(tables_path, "wt", encoding="utf-8") as handle:
            json.dump(dict(prior_strength=args.prior_strength, edges=[0.5, 1, 2, 4],
                           models=probe.export_tables()), handle)
        eval_seen = set()
        with (args.output_dir / "scenes.jsonl").open("x", encoding="utf-8") as handle:
            loader = data.test_dataloader()
            for batch in loader:
                name = scene_name(batch)
                if name in fit_seen:
                    raise RuntimeError(f"train/test scene overlap: {name}")
                if name in eval_seen:
                    continue
                probe.begin()
                with amp():
                    compressed = model.compress_scene(to_device(batch["inputs"], args.device))
                probe.end()
                cache_scene(probe, "test", compressed.data, batch)
                original = SceneBitstream.unpack(compressed.data).bytes_by_stream
                baseline = dict(total_bytes=len(compressed.data), score_bytes=original["score"],
                                residual_bytes=original["residual_y"] + original["residual_z"],
                                mean_bytes=original["mean"], container_bytes=original["container"])
                row = dict(scene=name, baseline=baseline,
                           context_frame_ids=batch["inputs"]["frame_ids"].reshape(-1).tolist(),
                           streams={n: r["stats"] for n, r in probe.records.items()}, variants={})
                for variant in ("marginal", "context"):
                    payload, stream_bytes = probe.recode(compressed.data, variant)
                    with amp():
                        probe.verify(compressed.data, payload, variant)
                    parts = SceneBitstream.unpack(payload).bytes_by_stream
                    row["variants"][variant] = dict(
                        total_bytes=len(payload), score_bytes=parts["score"],
                        residual_bytes=parts["residual_y"] + parts["residual_z"],
                        score_only_total_bytes=baseline["total_bytes"] - baseline["score_bytes"] + parts["score"],
                        residual_only_total_bytes=baseline["total_bytes"] - baseline["residual_bytes"]
                                                  + parts["residual_y"] + parts["residual_z"])
                    for stream, size in stream_bytes.items():
                        row["streams"][stream][f"{variant}_actual_bits"] = size * 8
                row["exact_feature_roundtrip"] = True
                rows.append(row)
                eval_ids.append(name)
                eval_seen.add(name)
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                if len(rows) % 16 == 0:
                    running = summarize(rows, args.seed)
                    gains = {v: round(r["total_saving_pct"], 3) for v, r in running["variants"].items()}
                    print(f"EVAL scenes={len(rows)}/{args.eval_scenes} saving_pct={gains} "
                          f"exact=True elapsed={time.monotonic()-start:.1f}s", flush=True)
                if len(rows) == args.eval_scenes:
                    break
            del loader
        if len(rows) != args.eval_scenes:
            raise RuntimeError("test loader exhausted; partial scenes.jsonl is available")
    report = dict(**metadata, seed=args.seed, precision=args.precision,
                  prior_strength=args.prior_strength, fit_scene_ids=fit_ids, eval_scene_ids=eval_ids,
                  cache_scenes_per_split=args.cache_scenes_per_split,
                  shared_table_file_bytes=tables_path.stat().st_size,
                  elapsed_seconds=time.monotonic() - start, metrics=summarize(rows, args.seed))
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_report(args.output_dir / "REPORT.md", report)
    print("PROBE_COMPLETE " + json.dumps(report["metrics"]["variants"]), flush=True)
    print(f"Results: {args.output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()

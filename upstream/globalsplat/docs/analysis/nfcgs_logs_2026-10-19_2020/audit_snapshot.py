"""Audit the downloaded Slurm snapshot without accessing the remote cluster."""
from __future__ import annotations

import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
WORKSPACE = REPO.parents[1]
LOGS = WORKSPACE/"slurm"
SNAPSHOT = "2026-10-19 20:20 Asia/Seoul downloaded logs"
SOURCES = {}
METRICS = ("psnr", "ssim", "lpips", "actual_bytes", "actual_bits_per_gaussian", "actual_bpga",
           "actual_score_bytes", "actual_residual_y_bytes", "actual_residual_z_bytes",
           "actual_mean_bytes", "actual_container_bytes")
FATAL = re.compile(r"Traceback \(most recent call last\)|(?:RuntimeError|ValueError|FileNotFoundError|AssertionError):|"
                   r"CUDA out of memory|No space left on device|Disk quota exceeded|PytorchStreamWriter|"
                   r"DUE TO TIME LIMIT|CANCELLED AT|Segmentation fault|NCCL.*(?:unhandled|error)", re.I)
PROGRESS = re.compile(r"^Epoch (\d+):.*?\|\s*(\d+)/(\d+) \[.*?,\s*([\d.]+)(it/s|s/it),.*\]$")


def read(path):
    if not path.is_file():
        return ""
    raw = path.read_bytes()
    relative = path.relative_to(WORKSPACE).as_posix()
    SOURCES[relative] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return raw.decode("utf-8", errors="replace")


def command_config(text):
    line = next((line for line in text.splitlines() if " -m globalsplat.main " in line), "")
    return {key.lstrip("+"): value for token in shlex.split(line) if "=" in token
            for key, value in [token.split("=", 1)]}


def check_overrides(command, expected):
    return {name: {"actual": command.get(name), "expected": str(value)}
            for name, value in expected.items() if command.get(name) != str(value)}


def inspect(row):
    group = row["group"]
    stem = "slurm-gs-arch-train" if group == "architecture" else (
        "slurm-gs-seed-pair-train" if group == "seed_pair" else "slurm-gs-order-context")
    training_path = LOGS/f"{stem}-{row['train_job']}.out"
    train = read(training_path)
    err = read(training_path.with_suffix(".err"))
    row["train_log_downloaded"] = bool(train)
    row["fatal_matches"] = [line for line in (train+"\n"+err).splitlines() if FATAL.search(line)]
    done = "`max_steps=50000` reached" in err
    progress = [match for line in train.splitlines() if (match := PROGRESS.match(line))]
    row["train_status"] = "complete_50k" if done else "progress_logged" if progress else "submitted_no_start_log"
    row["estimated_optimizer_step"] = 50000 if done else None
    row["estimated_training_percent"] = 100.0 if done else None
    row["remaining_training_hours_at_logged_speed"] = 0.0 if done else None
    if progress:
        last = progress[-1]
        epoch, batch, length = (int(last[i]) for i in (1, 2, 3))
        steps = epoch*math.ceil(length/2)+batch//2
        speed = float(last[4]) if last[5] == "it/s" else 1/float(last[4])
        row["last_complete_progress_line"] = last[0]
        row["progress_epoch"] = epoch
        row["progress_epoch_batch"] = batch
        row["progress_nominal_epoch_batches"] = length
        if not done:
            row["estimated_optimizer_step"] = steps
            row["estimated_training_percent"] = round(100*steps/50000, 3)
            row["remaining_training_hours_at_logged_speed"] = round((50000-steps)*2/speed/3600, 3)
        else:
            assert steps == 50000, (row["train_job"], steps)
    row["cuda_smoke_passed"] = "device=cuda" in train and "PILOT_SMOKE_OK" in train
    row["train_config_mismatches"] = {}
    if train:
        config = command_config(train)
        expected = {
            "loss.rate_lambda": row["lambda"], "trainer.devices": "2", "trainer.precision": "bf16-mixed",
            "trainer.max_steps": "50000", "trainer.accumulate_grad_batches": "2", "optimizer.batch_size": "2",
            "optimizer.lr": "0.0001", "checkpointing.resume": "false", "model.freeze_globalsplat": None,
            "model.feature_codec.token_order": row["order"], "model.feature_codec.score_context_schedule": row["schedule"],
            "model.feature_codec.score_spatial_stages": row["stages"],
            "model.feature_codec.score_slice_channels": row["slice_channels"],
            "model.feature_codec.score_spatial_context": str(row["spatial_context"]).lower(),
            "model.feature_codec.use_centering": "false", "model.feature_codec.use_score_norm": "false",
            "model.feature_codec.score_mean_condition": "false", "model.feature_codec.score_channel_context": "false",
            "model.feature_codec.transform": "linear", "model.feature_codec.score_spatial_kernel": "5",
            "model.feature_codec.score_context_quantization": "ste",
        }
        expected.pop("model.freeze_globalsplat")  # Declared in the unchanged experiment config.
        if group == "architecture":
            expected.update({"model.feature_codec.codec_mode": row["codec_mode"], "model.feature_codec.rank": row["rank"],
                             "model.feature_codec.use_residual": str(row["use_residual"]).lower(),
                             "architecture_entropy_determinism": "true", "trainer.limit_val_batches": "0"})
            assert config["dataset.train_holdout_manifest"].endswith("/holdout.json")
        row["train_config_mismatches"] = check_overrides(config, expected)
        row["train_model_seed"] = int(config.get("seed") or re.search(r"(?m)^seed: (\d+)$", train)[1])
        row["train_data_seed"] = int(config.get("dataset.data_loader_seed") or re.search(r"(?m)^  data_loader_seed: (\d+)$", train)[1])
        assert row["train_model_seed"] == (111124 if group == "seed_pair" else 111123)
        assert row["train_data_seed"] == (404 if group == "seed_pair" else 403)
        if row.get("source_checkpoint"):
            assert config["checkpointing.load"] == row["source_checkpoint"]
        row["training_input_checkpoint"] = config["checkpointing.load"]
    stem = "slurm-gs-arch-eval" if group == "architecture" else (
        "slurm-gs-seed-pair-eval" if group == "seed_pair" else "slurm-gs-order-context-eval")
    eval_path = LOGS/f"{stem}-{row['eval_job']}.out"
    evaluation = read(eval_path)
    eval_err = read(eval_path.with_suffix(".err"))
    row["eval_log_downloaded"] = bool(evaluation)
    row["fatal_matches"] += [line for line in (evaluation+"\n"+eval_err).splitlines() if FATAL.search(line)]
    row["metrics"] = {key: float(value) for key, value in re.findall(
        r"(?m)^("+"|".join(METRICS)+r"): ([-+\deE.]+)$", evaluation)}
    row["eval_status"] = "complete_fulltest" if len(row["metrics"]) == len(METRICS) else (
        "log_without_final_metrics" if evaluation else "submitted_no_start_log")
    if row["eval_status"] == "complete_fulltest":
        assert row["train_status"] == "complete_50k"
        assert all(math.isfinite(value) for value in row["metrics"].values())
        checked = re.findall(r"checkpoint codec=(\{[^\n]+\}) global_step=(\d+)", evaluation)
        assert checked and int(checked[-1][1]) == 50000
        declared = ast.literal_eval(checked[-1][0])
        row["declared_codec_config"] = declared
        for key, target in (("token_order", row["order"]), ("score_context_schedule", row["schedule"]),
                            ("score_spatial_stages", row["stages"]), ("score_slice_channels", row["slice_channels"]),
                            ("score_spatial_context", row["spatial_context"]), ("use_residual", True)):
            assert declared[key] == target, (row["eval_job"], key)
        command = command_config(evaluation)
        row["checkpoint"] = command["checkpointing.load"]
        assert f"/job_{row['train_job']}/" in row["checkpoint"]
        assert row["checkpoint"].endswith("/step000050000.ckpt")
        expected = {"loss.rate_lambda": row["lambda"], "dataset": "re10k_eval_all_ctx12",
                    "test.actual_bitstream": "true", "seed": "0", "test.max_scenes": "null",
                    "optimizer.batch_size": "1", "trainer.devices": "1"}
        assert not check_overrides(command, expected), row["eval_job"]
        progress_eval = re.findall(r"Testing DataLoader 0:.*? (\d+)/(\d+) \[", evaluation)
        row["evaluated_scenes"] = int(progress_eval[-1][0])
        row["nominal_dataloader_scene_count"] = int(progress_eval[-1][1])
        assert row["evaluated_scenes"] == 6991
        metrics = row["metrics"]
        error = metrics["actual_bytes"]-sum(metrics[name] for name in (
            "actual_score_bytes", "actual_residual_y_bytes", "actual_residual_z_bytes", "actual_mean_bytes", "actual_container_bytes"))
        row["byte_accounting_error"] = round(error, 9)
        assert abs(error) < .00001
        row["eval_source_log"] = eval_path.relative_to(WORKSPACE).as_posix()
        row["eval_source_sha256"] = SOURCES[row["eval_source_log"]]["sha256"]
    assert not row["train_config_mismatches"], (row["train_job"], row["train_config_mismatches"])
    return row


def main():
    rows = []
    for directory, group in (("order_context_extremes_20261017", "rd_extensions"), ("paper_controls_20261018", "controls")):
        for source in csv.DictReader(read(LOGS/directory/"submissions.tsv").splitlines(), delimiter="\t"):
            output = source["output"]
            spatial = "_context_off" not in output
            row = {"group": group, "order": source["order"], "schedule": source["schedule"], "stages": int(source["stages"]),
                   "lambda": source["lambda"], "slice_channels": 56 if "_c56/" in output else 16,
                   "spatial_context": spatial, "train_job": source["train_job"], "eval_job": source["eval_job"],
                   "source_checkpoint": source["source_checkpoint"], "output": output}
            matching = [p for p in (LOGS/directory).glob("*.state") if read(p).split()[-2:] == [row["train_job"], row["eval_job"]]]
            assert len(matching) == 1, row
            rows.append(inspect(row))
    states = {}
    for directory, group in (("architecture_pilots_20261019", "architecture"), ("context_seed_pair_20261019", "seed_pair")):
        state = json.loads(read(LOGS/directory/"state.json"))
        mismatches = []
        for relative, expected in state["inputs"]["code_sha256"].items():
            path = REPO/relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                mismatches.append(relative)
        states[group] = {"state_source": f"slurm/{directory}/state.json", "runtime_hash_mismatches": mismatches}
        assert not mismatches
        for arm, jobs in state["arms"].items():
            row = {"group": group, "arm": arm, "order": "hilbert", "schedule": "dyadic4", "stages": 4,
                   "lambda": "0.0128", "slice_channels": 184 if arm == "score736_only" else 16,
                   "spatial_context": arm not in ("msh_only", "off16"),
                   "train_job": jobs["train"], "eval_job": jobs["validation" if group == "architecture" else "eval"],
                   "codec_mode": "msh_only" if arm == "msh_only" else "score_only" if arm.startswith("score") else "two_path",
                   "rank": 0 if arm == "msh_only" else 736 if arm == "score736_only" else 56,
                   "use_residual": not arm.startswith("score")}
            rows.append(inspect(row))
        if group == "architecture":
            prepare = read(LOGS/f"slurm-gs-arch-prepare-{state['prepare']}.out")
            err = read(LOGS/f"slurm-gs-arch-prepare-{state['prepare']}.err")
            states[group].update(prepare_job=state["prepare"], prepare_complete="PREPARATION_COMPLETE=" in prepare,
                                 initialized_arms=re.findall(r"INITIALIZED=(\w+)", prepare),
                                 holdout_complete="HOLDOUT_SELECTED=128/128" in prepare,
                                 prepare_fatal_matches=[line for line in (prepare+"\n"+err).splitlines() if FATAL.search(line)])
    baseline_path = REPO/"docs/analysis/nfcgs_paper_inventory_2026-10-18.json"
    old = json.loads(read(baseline_path))
    for result in old["results"]:
        read(WORKSPACE/result["source_log"])
        assert SOURCES[result["source_log"]]["sha256"] == result["source_sha256"], result["source_log"]
    families = {}
    def signature(codec):
        return tuple(codec.get(key) for key in ("token_order", "score_context_schedule", "score_spatial_stages", "score_slice_channels", "score_spatial_context"))
    for result in old["results"]:
        key = signature(result["declared_config"])
        assert key not in families or families[key] == result["family"]
        families[key] = result["family"]
    positives = list(old["results"])
    zero = []
    for row in rows:
        if row["eval_status"] != "complete_fulltest":
            continue
        family = families[signature(row["declared_codec_config"])]
        point = {"label": f"{family}_{row['lambda'].replace('.', 'p')}", "family": family,
                 "rate_lambda": float(row["lambda"]), "source_log": row["eval_source_log"],
                 "source_sha256": row["eval_source_sha256"], "evaluated_scenes": row["evaluated_scenes"],
                 "metrics": row["metrics"], "declared_config": row["declared_codec_config"],
                 "checkpoint": row["checkpoint"], "train_job": row["train_job"], "eval_job": row["eval_job"],
                 "byte_accounting_error": row["byte_accounting_error"]}
        (positives if point["rate_lambda"] > 0 else zero).append(point)
    assert len({(row["family"], row["rate_lambda"]) for row in positives}) == len(positives)
    inventory = {"version": 1, "snapshot": SNAPSHOT, "scope": "completed actual full-test results, each 6991 scenes",
                 "completed_positive_rate_points": len(positives), "families": len({row["family"] for row in positives}),
                 "lambda_zero_points": zero, "results": positives}
    counts = {group: {"training_points": len(selected), "training_status": dict(Counter(row["train_status"] for row in selected)),
                      "evaluation_status": dict(Counter(row["eval_status"] for row in selected))}
              for group in ("rd_extensions", "controls", "architecture", "seed_pair")
              if (selected := [row for row in rows if row["group"] == group])}
    report = {"snapshot": SNAPSHOT, "scope": "downloaded files only; no live Slurm query or remote checkpoint access",
              "progress_formula": "epoch*ceil(nominal_epoch_batches/2)+floor(epoch_batch/2); complete jobs cross-checked at 50000",
              "eta_scope": "remaining training at logged average speed; excludes queue, evaluation, and time after download",
              "counts": counts, "recent_training_points": len(rows), "recent_submitted_jobs_including_prepare": 2*len(rows)+1,
              "training_complete_50k": sum(row["train_status"] == "complete_50k" for row in rows),
              "training_progress_logged": sum(row["train_status"] == "progress_logged" for row in rows),
              "training_no_downloaded_start": sum(row["train_status"] == "submitted_no_start_log" for row in rows),
              "completed_new_fulltest_results": sum(row["eval_status"] == "complete_fulltest" for row in rows),
              "completed_positive_rate_points": len(positives), "lambda_zero_points": len(zero),
              "fatal_matches": [line for row in rows for line in row["fatal_matches"]],
              "preparation_and_submission_states": states, "experiments": rows, "sources": SOURCES}
    (OUT/"review.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (OUT/"inventory.json").write_text(json.dumps(inventory, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("counts", "recent_training_points", "recent_submitted_jobs_including_prepare",
        "training_complete_50k", "training_progress_logged", "training_no_downloaded_start", "completed_new_fulltest_results",
        "completed_positive_rate_points", "lambda_zero_points", "fatal_matches")}, indent=2))
    for row in rows:
        if row["train_status"] != "complete_50k":
            print(row["group"], row.get("arm") or f"{row['order']}/{row['schedule']}/{row['lambda']}/spatial={row['spatial_context']}",
                  row["train_job"], row["eval_job"], row["estimated_optimizer_step"], row["estimated_training_percent"],
                  row["remaining_training_hours_at_logged_speed"])


if __name__ == "__main__":
    main()

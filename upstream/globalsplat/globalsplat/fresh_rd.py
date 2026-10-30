"""Frozen definitions and artifact checks for the fresh, one-GPU RD protocol."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

SEEDS = ((111123, 403), (111124, 404), (111125, 405))
LAMBDAS = ("0.0032", "0.0064", "0.0128", "0.0256", "0.0512")
BUDGETS = (100000, 120000, 150000)
CORE = ("h_on", "h_off", "n_on", "n_off")
VERSION = "nfcgs_fresh_rd_v1"


@dataclass(frozen=True)
class Cell:
    order: str
    context: bool = True
    schedule: str = "dyadic4"
    stages: int = 4
    arm: str = "two_path56"
    backend: str = "legacy"
    metric: str = "l1"
    selection: str = ""  # "", "adaptive" or "importance" (adaptive, importance-ordered); plan §14
    forced_count: int = 0


CELLS = {
    "h_on": Cell("hilbert"), "h_off": Cell("hilbert", False),
    "n_on": Cell("none"), "n_off": Cell("none", False),
    "m_on": Cell("morton"), "h_l3": Cell("hilbert", True, "legacy", 3),
    "s56": Cell("hilbert", arm="score56_only"),
    "msh": Cell("hilbert", False, arm="msh_only"),
    # 2026-10-28 additions (plan §13), run from the separate server copy.
    "nx_on": Cell("nn_xyz", backend="graph"),
    "ns_on": Cell("nn_score", backend="graph", metric="l2"),
    "ns1_on": Cell("nn_score", backend="graph_gemm"),
    "s56_off": Cell("hilbert", False, arm="score56_only"),
    "s56_n_on": Cell("none", arm="score56_only"),
    "s56_n_off": Cell("none", False, arm="score56_only"),
    "msh_n": Cell("none", False, arm="msh_only"),
    # Top-K token selection on H-ON (plan §14).
    "tk_h": Cell("hilbert", selection="adaptive"),
    "tk_imp": Cell("none", selection="importance"),
    "tk_ns": Cell("nn_score", backend="graph", metric="l2", selection="adaptive"),
    "tk_ns1": Cell("nn_score", backend="graph_gemm", selection="adaptive"),
    # TK-fixed candidates: the one nearest TK-H .0128's mean K at 30k is used.
    # Defined up front so the separate copy's runtime files stay frozen.
    **{f"tk_f{count}": Cell("hilbert", selection="adaptive", forced_count=count)
       for count in range(2048, 4096, 256)},
}
TOKEN_SELECTION_MINIMUM = 256
TOKEN_SELECTION_INIT_LOGIT = 2.94  # κ starts near 95% of the 4096 tokens


def selection_overrides(name):
    """Hydra overrides that build the Top-K selector of a cell (train and test)."""
    cell = CELLS[name]
    if not cell.selection:
        return []
    order = "importance" if cell.selection == "importance" else "slot"
    return [f"++model.token_selection.order={order}",
            f"++model.token_selection.minimum={TOKEN_SELECTION_MINIMUM}",
            f"++model.token_selection.init_logit={TOKEN_SELECTION_INIT_LOGIT}",
            f"++model.token_selection.forced_count={cell.forced_count}"]
# Order and spatial-context switches stored in each single-path initializer.
PILOT_SWITCHES = {"score56_only": ("hilbert", True), "msh_only": ("hilbert", False)}


def converts_initializer(name):
    """Whether this cell changes switches of its fresh initializer at load time."""
    cell = CELLS[name]
    return cell.arm == "two_path56" or (cell.order, cell.context) != PILOT_SWITCHES[cell.arm]


def rate_text(value):
    rate = Decimal(str(value))
    if not rate.is_finite() or rate < 0:
        raise ValueError("lambda must be finite and nonnegative")
    return "0" if not rate else format(rate.normalize(), "f")


def run_key(cell, rate, seed):
    if cell not in CELLS or seed not in range(3):
        raise ValueError("unknown cell or seed")
    return f"{cell}/lambda{rate_text(rate).replace('.', 'p')}/s{seed}"


def wave_runs(wave, msh_lambdas=("0.0064", "0.0128"), omit_seeds=False):
    if wave == 1:
        return [(c, r, 0) for r in (LAMBDAS[0], LAMBDAS[2], LAMBDAS[4]) for c in CORE] + [
            ("h_on", "0", 0), ("s56", "0.0128", 0), ("msh", "0.0032", 0)]
    if wave != 2 or len(msh_lambdas) != 2:
        raise ValueError("wave must be 1 or 2; exactly two MSH lambdas required")
    rates = tuple(map(rate_text, msh_lambdas))
    if len(set(rates)) != 2 or "0.0032" in rates:
        raise ValueError("MSH rates must differ from one another and the wave-1 rate")
    rows = [(c, r, 0) for c in CORE for r in ("0.0064", "0.0256")]
    if not omit_seeds:
        rows += [(c, "0.0128", s) for s in (1, 2) for c in CORE]
    return rows + [("m_on", "0.0128", 0), ("h_l3", "0.0128", 0)] + [("msh", r, 0) for r in rates]


def milestones(budget):
    if budget not in BUDGETS:
        raise ValueError("unsupported training budget")
    return [budget * 7 // 10, budget * 9 // 10]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False, newline="\n") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def directory_lock(directory, name=".lock"):
    """Exclusive create; a stale lock requires inspection, never automatic removal."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / name
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink(missing_ok=True)


def checkpoint_path(run, step):
    return Path(run) / f"checkpoints/nfcgs_main/version_0/step{step:09d}.ckpt"


def canonical_overrides(command, overrides):
    """Replace Hydra keys, including +/++ spellings, without duplicate overrides."""
    keys = {item.split("=", 1)[0].lstrip("+") for item in overrides}
    result = [item for item in command if item.split("=", 1)[0].lstrip("+") not in keys]
    return result + list(overrides)


def validate_rows(rows, expected, rate_required=True):
    if len(rows) != expected or len({r["scene"] for r in rows}) != expected:
        raise ValueError(f"expected {expected} unique scenes, got {len(rows)}")
    for row in rows:
        for name in ("psnr", "ssim", "lpips", "mse"):
            if not math.isfinite(float(row[name])):
                raise ValueError(f"non-finite {name}")
        if row["mse"] < 0:
            raise ValueError("negative MSE")
        c, t = row["context_frame_ids"], row["target_frame_ids"]
        if len(set(c)) != 12 or len(c) != 12 or len(set(t)) != 8 or len(t) != 8 or set(c) & set(t):
            raise ValueError("evaluation must use disjoint C12/T8 frames")
        if rate_required:
            if row["actual_bytes"] <= 0:
                raise ValueError("empty scene packet")
            fields = ("score", "residual_y", "residual_z", "mean", "container")
            if sum(row[f"actual_{k}_bytes"] for k in fields) != row["actual_bytes"]:
                raise ValueError("packet byte accounting mismatch")


def objective(row, rate):
    return row["mse"] + .05 * row["lpips"] + float(rate) * (row["actual_bytes"] * 8 / (32768 * 59))


def validate_cohort(rows, expected):
    actual = {r["scene"]: {"context": r["context_frame_ids"], "target": r["target_frame_ids"]} for r in rows}
    if actual != expected:
        raise ValueError("scene/frame cohort differs from the frozen evaluation cohort")
    return hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def paired_change(before, after, rate):
    a, b = {r["scene"]: r for r in before}, {r["scene"]: r for r in after}
    if set(a) != set(b) or len(a) != len(before) or len(b) != len(after):
        raise ValueError("paired scene cohorts differ")
    changes = []
    for scene in sorted(a):
        for key in ("context_frame_ids", "target_frame_ids"):
            if a[scene][key] != b[scene][key]:
                raise ValueError("paired evaluation frames differ")
        changes.append(objective(b[scene], rate) - objective(a[scene], rate))
    mse = statistics.fmean(r["mse"] for r in before)
    scale = -10 / math.log(10) / max(mse, 1e-12)
    delta = scale * statistics.fmean(changes)
    se = abs(scale) * (statistics.stdev(changes) / math.sqrt(len(changes)) if len(changes) > 1 else 0)
    return dict(delta_db_j=delta, paired_se_db_j=se, improvement_above_2se=delta > 2 * se,
                scenes=len(changes), normalization_mse_40k=mse)


def gate1_decision(changes):
    """No full-test inputs; insignificant positive changes count as zero."""
    core = [r for r in changes if r["cell"] in CORE and float(r["lambda"]) > 0]
    if len(core) != 12 or len({(r["cell"], rate_text(r["lambda"])) for r in core}) != 12:
        raise ValueError("Gate 1 requires all twelve wave-1 factorial runs")
    largest = max(max(0, r["delta_db_j"]) if r["improvement_above_2se"] else 0 for r in core)
    return dict(recommended_budget=120000 if largest >= .1 else 100000,
                consider_150k=largest >= .2, largest_significant_delta_db_j=largest,
                selection_split="train-derived holdout128", seed_reduction_automatic=False)

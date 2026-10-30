"""Plan §15 40k check: low-Score training mode against healthy reference runs."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nfcgs_train_curve as checker

pytest.importorskip("tensorboard")


def write_run(root: Path, name: str, psnr, score_bits, last=45000) -> Path:
    from torch.utils.tensorboard import SummaryWriter
    run = root / name
    writer = SummaryWriter(str(run / "tensorboard" / "version_0"))
    for step in range(0, last, 500):
        writer.add_scalar("train_psnr", psnr(step), step)
        writer.add_scalar("train_codec_score_bits", score_bits(step), step)
        writer.add_scalar("train_loss", 0.03, step)
    writer.close()
    return run


def healthy_psnr(step):
    return 20 + 3 * min(step, 40000) / 40000


def test_failed_run_is_flagged_and_healthy_seed_is_not(tmp_path):
    healthy = write_run(tmp_path, "s1", healthy_psnr, lambda step: 1e6 * (1 + max(0, step - 20000) / 1e5))
    failed = write_run(tmp_path, "s2", lambda step: healthy_psnr(step) - (0.9 if step >= 30000 else 0),
                       lambda step: 1e6 * (1 - max(0, step - 20000) / 2e5))
    rows = {row["run"].name: row for row in checker.check([failed, healthy], [healthy], 40000, 5000, 0.5)}
    assert rows["s2"]["verdict"] == "FAIL?" and rows["s2"]["gap"] == pytest.approx(-0.9, abs=0.01)
    assert rows["s2"]["score_bits_change"] < 0 < rows["s1"]["score_bits_change"]
    assert rows["s1"]["verdict"] == "ok" and rows["s1"]["end"] == 40000


def test_unfinished_run_is_compared_at_its_own_step_and_partial_record_is_ignored(tmp_path):
    reference = write_run(tmp_path, "ref", healthy_psnr, lambda step: 1e6)
    running = write_run(tmp_path, "running", healthy_psnr, lambda step: 1e6, last=31200)
    event_file = next((running / "tensorboard").rglob("events.out.tfevents*"))
    with event_file.open("ab") as stream:
        stream.write(b"\x40\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00train_psnr")  # record still being written
    (row,) = checker.check([running], [reference], 40000, 5000, 0.5)
    assert row["end"] == 30000 and row["verdict"] == "ok" and abs(row["gap"]) < 1e-6
    (empty,) = checker.check([tmp_path / "missing"], [reference], 40000, 5000, 0.5)
    assert empty["verdict"] == "no data"

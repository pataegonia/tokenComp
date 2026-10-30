"""Plan §14 Top-K fine-tune from H-ON 120k: sources, commands and the control."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_nfcgs_topk_finetune as finetune
import run_nfcgs_fresh_rd as runner
from globalsplat.fresh_rd import checkpoint_path


def make_source(root, rate, job):
    path = checkpoint_path(root / f"h_on/lambda{rate.replace('.', 'p')}/s0/job_{job}", 120000)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def test_default_source_is_the_single_h_on_s0_120k_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        finetune.default_source(tmp_path, "0.0128")
    source = make_source(tmp_path, "0.0128", 1)
    make_source(tmp_path, "0.0032", 2)
    checkpoint_path(tmp_path / "h_on/lambda0p0128/s1/job_3", 120000).parent.mkdir(parents=True)
    assert finetune.default_source(tmp_path, "0.0128") == source
    make_source(tmp_path, "0.0128", 4)
    with pytest.raises(FileNotFoundError):
        finetune.default_source(tmp_path, "0.0128")


def test_only_the_three_planned_lambdas_are_accepted(tmp_path):
    common = ["--variant", "topk", "--run-dir", str(tmp_path / "run"), "--source", str(tmp_path / "s.ckpt")]
    assert finetune.parse_args([*common, "--rate-lambda", "0.01280"]).rate_lambda == "0.0128"
    for rate in ("0", "0.0064"):
        with pytest.raises(SystemExit):
            finetune.parse_args([*common, "--rate-lambda", rate])


@pytest.mark.parametrize("rate", finetune.LAMBDAS)
@pytest.mark.parametrize("variant", ["topk", "control"])
def test_commands_differ_only_in_the_selector(variant, rate, tmp_path):
    source = tmp_path / "h_on_120k.ckpt"
    args = finetune.parse_args(["--variant", variant, "--rate-lambda", rate,
                                "--run-dir", str(tmp_path / variant), "--source", str(source)])
    cfg = runner.compose_command(finetune.train_command(args))
    finetune.validate_train_config(cfg, args)
    assert Path(cfg.checkpointing.load) == source
    assert cfg.loss.rate_lambda == float(rate)
    assert list(cfg.optimizer.lr_milestones) == [10**9]
    assert cfg.model.train_decoder_readouts and cfg.optimizer.decoder_lr == 1e-5
    selection = cfg.model.get("token_selection")
    if variant == "topk":
        assert (selection.order, selection.minimum, selection.forced_count) == ("slot", 256, 0)
    else:
        assert not selection
    test = runner.compose_command(finetune.test_command(args, tmp_path / variant / "final.ckpt"))
    assert test.test.actual_bitstream and test.loss.rate_lambda == float(rate)
    assert Path(test.checkpointing.load) == tmp_path / variant / "final.ckpt"
    assert bool(test.model.get("token_selection")) == (variant == "topk")
    assert not test.model.get("train_decoder_readouts", False)


def test_slurm_is_lf_bash():
    path = ROOT / "scripts/slurm/finetune_nfcgs_topk.slurm"
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.startswith(b"#!/usr/bin/env bash\n")
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(bash) if bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

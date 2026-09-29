"""Execute the submission wrapper with mocked conda/srun, without a cluster."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/slurm/train_hyper1d_12h.slurm"
RESUME_SCRIPT = ROOT / "scripts/slurm/resume_hyper1d_12h.slurm"
EVAL_SCRIPT = ROOT / "scripts/slurm/eval_hyper1d.slurm"
if sys.platform == "win32":
    candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"
    BASH = str(candidate) if candidate.is_file() else None
else:
    BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required for the submission-wrapper test")


def run_wrapper(tmp_path, arguments=(), *, checkpoint_exists=True, vanilla_override=None, script=SCRIPT):
    repo = tmp_path / "repo with spaces"
    checkpoint = repo / "checkpoints/pretrained/globalsplat-re10k-32k.ckpt"
    checkpoint.parent.mkdir(parents=True)
    if checkpoint_exists:
        checkpoint.touch()
    # The resume wrapper delegates to this script inside REPO_DIR.
    pilot_script = repo / "scripts/slurm/train_hyper1d_12h.slurm"
    pilot_script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, pilot_script)
    conda_root = tmp_path / "conda"
    profile = conda_root / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text("# Mock activation is defined in the harness.\n", encoding="utf-8")
    dump = tmp_path / "arguments.txt"
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    mock_srun = mock_bin / "srun"
    mock_srun.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$@" > "$ARGUMENT_DUMP"\n', encoding="utf-8")
    mock_srun.chmod(0o755)
    harness = tmp_path / "harness.sh"
    harness.write_text(
        'set -euo pipefail\n'
        'conda() { case "$1" in info) printf "%s\\n" "$MOCK_CONDA_ROOT" ;; activate) return 0 ;; *) return 1 ;; esac; }\n'
        'export -f conda\n'
        'if command -v cygpath >/dev/null 2>&1; then MOCK_BIN=$(cygpath -u "$MOCK_BIN"); fi\n'
        'export PATH="$MOCK_BIN:$PATH"\n'
        'bash "$WRAPPER_SCRIPT" "$@"\n', encoding="utf-8")
    env = os.environ.copy()
    env.pop("VANILLA_CHECKPOINT", None)
    env.update(REPO_DIR=repo.as_posix(), MOCK_CONDA_ROOT=conda_root.as_posix(),
               MOCK_BIN=mock_bin.as_posix(), ARGUMENT_DUMP=dump.as_posix(),
               WRAPPER_SCRIPT=script.as_posix())
    if vanilla_override is not None:
        env["VANILLA_CHECKPOINT"] = vanilla_override
    result = subprocess.run([BASH, str(harness), *arguments], env=env,
                            capture_output=True, text=True)
    return result, dump.read_text().splitlines() if dump.exists() else [], checkpoint


def test_plain_submission_passes_default_checkpoint(tmp_path):
    result, args, checkpoint = run_wrapper(tmp_path, ["--lambda", "0.0064"])
    assert result.returncode == 0, result.stderr
    assert args[:4] == ["--ntasks=1", "python", "scripts/run_hyper1d.py", "train"]
    assert args[4:] == ["--vanilla-checkpoint", checkpoint.as_posix(), "--lambda", "0.0064"]


@pytest.mark.parametrize("arguments", [
    ["--checkpoint", "/some path/hyper1d.ckpt", "--resume"],
    ["--checkpoint=/some path/hyper1d.ckpt", "--resume"],
    ["--vanilla-checkpoint", "/some path/vanilla.ckpt"],
    ["--vanilla-checkpoint=/some path/vanilla.ckpt"],
])
def test_explicit_checkpoint_forwarded_without_default(tmp_path, arguments):
    result, args, _ = run_wrapper(tmp_path, arguments, checkpoint_exists=False)
    assert result.returncode == 0, result.stderr
    assert args[4:] == arguments


def test_missing_default_checkpoint_fails_with_path(tmp_path):
    result, args, checkpoint = run_wrapper(tmp_path, checkpoint_exists=False)
    assert result.returncode == 2 and not args
    assert "Missing vanilla checkpoint:" in result.stderr
    assert checkpoint.as_posix() in result.stderr


def test_dry_run_can_use_a_missing_default(tmp_path):
    result, args, checkpoint = run_wrapper(tmp_path, ["--dry-run"], checkpoint_exists=False)
    assert result.returncode == 0, result.stderr
    assert args[4:] == ["--vanilla-checkpoint", checkpoint.as_posix(), "--dry-run"]


def test_default_checkpoint_environment_override(tmp_path):
    checkpoint = tmp_path / "original.ckpt"
    checkpoint.touch()
    result, args, _ = run_wrapper(tmp_path, vanilla_override=checkpoint.as_posix())
    assert result.returncode == 0, result.stderr
    assert args[4:] == ["--vanilla-checkpoint", checkpoint.as_posix()]


def test_resume_submission_uses_full_state_and_frequent_saves(tmp_path):
    checkpoint = tmp_path / "completed run/last.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    result, args, _ = run_wrapper(tmp_path, [checkpoint.as_posix(), "--output", "/original run"],
                                  checkpoint_exists=False, script=RESUME_SCRIPT)
    assert result.returncode == 0, result.stderr
    assert args[4:] == ["--checkpoint", checkpoint.as_posix(), "--resume",
                       "--max-steps", "32000", "--train-hours", "11",
                       "--checkpoint-every", "500", "--validate-every", "2000",
                       "--output", "/original run"]


def test_resume_submission_allows_later_step_limit(tmp_path):
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.touch()
    result, args, _ = run_wrapper(tmp_path, [checkpoint.as_posix(), "--max-steps", "48000"],
                                  checkpoint_exists=False, script=RESUME_SCRIPT)
    assert result.returncode == 0, result.stderr
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_hyper1d", ROOT / "scripts/run_hyper1d.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    parsed = runner.parse_args(["train", *args[4:]])
    assert parsed.max_steps == 48000
    assert parsed.resume and parsed.checkpoint == checkpoint


@pytest.mark.parametrize("arguments", [[], ["missing.ckpt"]])
def test_resume_submission_rejects_missing_source(tmp_path, arguments):
    result, args, _ = run_wrapper(tmp_path, arguments, script=RESUME_SCRIPT)
    assert result.returncode == 2 and not args
    assert "checkpoint" in result.stderr.lower()


def test_eval_submission_samples_test_without_initialization_or_resume(tmp_path):
    checkpoint = tmp_path / "saved run/step000016000.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    result, args, _ = run_wrapper(tmp_path, [checkpoint.as_posix(), "--max-scenes", "16",
                                  "--save-images", "--output", "/sampled test"],
                                  checkpoint_exists=False, script=EVAL_SCRIPT)
    assert result.returncode == 0, result.stderr
    assert args[:4] == ["--ntasks=1", "python", "scripts/run_hyper1d.py", "eval"]
    assert "--sample-test" in args and "--resume" not in args and "--vanilla-checkpoint" not in args
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_hyper1d", ROOT / "scripts/run_hyper1d.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    parsed = runner.parse_args(args[3:])
    assert parsed.max_scenes == 16 and parsed.sample_seed == 111123
    assert parsed.workers == 0 and parsed.save_images


def test_eval_submission_rejects_missing_checkpoint(tmp_path):
    result, args, _ = run_wrapper(tmp_path, ["missing.ckpt"], script=EVAL_SCRIPT)
    assert result.returncode == 2 and not args
    assert "Missing evaluation checkpoint" in result.stderr

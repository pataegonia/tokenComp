"""Exercise checkpoint discovery without Slurm, CUDA, or checkpoint loading."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/slurm/eval_nfcgs_rank56.slurm"
PAPER_SCRIPT = SCRIPT.with_name("eval_nfcgs_paper_recipe.slurm")


@pytest.fixture
def bash():
    if os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            return str(candidate)
        pytest.skip("Git Bash is required on Windows")
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is required")
    return executable


def checkpoint(root, task, morton, step, version=0):
    rank = 56 if task < 4 else 80
    rate = "0p0064" if task % 4 < 2 else "0p0256"
    residual = "on" if task % 2 == 0 else "off"
    arm = root / f"rank{rank}" / f"lambda{rate}" / f"residual_{residual}"
    if not morton:
        arm /= "morton_off"
    experiment = f"nfcgs_paper24_subset_rank{rank}_lambda{rate}_residual_{residual}_morton_{'on' if morton else 'off'}"
    path = arm / "checkpoints" / experiment / f"version_{version}" / f"step{step:09d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def discover(bash, root, task, morton, expected="50000"):
    env = os.environ.copy()
    env.update(
        TRAIN_OUTPUT_ROOT=root.as_posix(),
        SLURM_ARRAY_TASK_ID=str(task),
        USE_MORTON=str(morton).lower(),
        EXPECTED_STEP=expected,
        CHECKPOINT_ONLY="true",
        CHECKPOINT="",
        DRY_RUN="false",
    )
    return subprocess.run(
        [bash, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize("task,morton", [(i, True) for i in range(8)] + [(i, False) for i in (0, 2, 4, 6)])
def test_selects_matching_arm_and_completed_step(bash, tmp_path, task, morton):
    selected = checkpoint(tmp_path, task, morton, 50000)
    checkpoint(tmp_path, task, morton, 49500)
    checkpoint(tmp_path, task, morton, 50, version=4)  # Cancelled newer run.
    checkpoint(tmp_path, task, not morton, 50000)  # Same step, other Morton arm.
    result = discover(bash, tmp_path, task, morton)
    assert result.returncode == 0, result.stderr
    assert f"CHECKPOINT={selected.as_posix()}" in result.stdout


def test_missing_final_checkpoint_does_not_fall_back_to_other_morton_arm(bash, tmp_path):
    checkpoint(tmp_path, 0, True, 49500)
    checkpoint(tmp_path, 0, False, 50000)
    result = discover(bash, tmp_path, 0, True)
    assert result.returncode != 0
    assert "no step-50000 checkpoint" in result.stderr


def test_auto_step_does_not_select_higher_step_from_other_morton_arm(bash, tmp_path):
    selected = checkpoint(tmp_path, 0, True, 49500)
    checkpoint(tmp_path, 0, False, 50000)
    result = discover(bash, tmp_path, 0, True, expected="auto")
    assert result.returncode == 0, result.stderr
    assert f"CHECKPOINT={selected.as_posix()}" in result.stdout


@pytest.fixture
def paper_path(tmp_path_factory):
    # The real experiment names are long; keep the temporary repo name short
    # so Windows can create the full checkpoint tree without MAX_PATH errors.
    return tmp_path_factory.mktemp("p")


def paper_repo(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "globalsplat").mkdir()
    scripts = tmp_path / "scripts/slurm"
    scripts.mkdir(parents=True)
    # Capture the effective environment received by the common evaluator,
    # without launching Slurm or CUDA. This also detects stale dry-run flags.
    (scripts / SCRIPT.name).write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "EVAL_CHECKPOINT=$CHECKPOINT" "EVAL_OUTPUT=$OUTPUT_ROOT" '
        '"EVAL_MORTON=$USE_MORTON" "EVAL_STEP=$EXPECTED_STEP" '
        '"EVAL_PROTOCOL=$PROTOCOL" "EVAL_CONTEXT=$CONTEXT" '
        '"EVAL_DRY_RUN=$DRY_RUN" "EVAL_CHECKPOINT_ONLY=$CHECKPOINT_ONLY"\n',
        encoding="utf-8", newline="\n",
    )
    return tmp_path / "outputs/nfcgs_paper24_subset_train"


def run_paper(bash, tmp_path, mode, task=0):
    env = os.environ.copy()
    env.update(
        REPO_DIR="/stale/repo",
        TRAIN_OUTPUT_ROOT="/stale/nfcgs_rank56_train",
        OUTPUT_ROOT="/stale/eval",
        CHECKPOINT="/stale/step000050000.ckpt",
        EXPECTED_STEP="123",
        USE_MORTON="false" if mode == "on" else "true",
        SLURM_ARRAY_TASK_ID=str(task),
        DRY_RUN="true",
        CHECKPOINT_ONLY="true",
        PROTOCOL="fixed",
        CONTEXT="36",
    )
    return subprocess.run(
        [bash, str(PAPER_SCRIPT), mode], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("task,morton", [(i, True) for i in range(8)] + [(i, False) for i in (0, 2, 4, 6)])
def test_paper_launcher_uses_confirmed_checkpoint_despite_stale_environment(bash, paper_path, task, morton):
    tmp_path = paper_path
    root = paper_repo(tmp_path)
    selected = checkpoint(root, task, morton, 50000)
    # A newer version must not replace the confirmed version_0 training run.
    checkpoint(root, task, morton, 50000, version=4)
    result = run_paper(bash, tmp_path, "on" if morton else "off", task)
    assert result.returncode == 0, result.stderr
    relative = selected.relative_to(tmp_path).as_posix()
    selected_line = next(line for line in result.stdout.splitlines() if line.startswith("EVAL_CHECKPOINT="))
    assert selected_line.endswith("/" + relative)
    assert "nfcgs_paper24_subset_eval_corrected" in result.stdout
    assert f"EVAL_MORTON={str(morton).lower()}" in result.stdout
    for setting in ("STEP=50000", "PROTOCOL=all", "CONTEXT=12", "DRY_RUN=false", "CHECKPOINT_ONLY=false"):
        assert "EVAL_" + setting in result.stdout
    assert "/stale/" not in result.stdout


def test_paper_preflight_checks_all_twelve_without_starting_eval(bash, paper_path):
    tmp_path = paper_path
    root = paper_repo(tmp_path)
    for task in range(8):
        checkpoint(root, task, True, 50000)
        if task % 2 == 0:
            checkpoint(root, task, False, 50000)
    result = run_paper(bash, tmp_path, "check")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("PAPER24_CHECKPOINT=") == 12
    assert "All 12 paper24 checkpoint files exist" in result.stdout
    assert "EVAL_CHECKPOINT=" not in result.stdout


def test_paper_missing_confirmed_file_fails_instead_of_selecting_another_run(bash, paper_path):
    tmp_path = paper_path
    root = paper_repo(tmp_path)
    checkpoint(root, 0, True, 50000, version=4)
    result = run_paper(bash, tmp_path, "on")
    assert result.returncode != 0
    assert "confirmed paper24 checkpoint is missing" in result.stderr
    assert "EVAL_CHECKPOINT=" not in result.stdout

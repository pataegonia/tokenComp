"""Exercise all 16 launch mappings without Slurm, data, or a GPU."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/slurm"


@pytest.fixture
def bash():
    candidate = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not candidate or not Path(candidate).is_file():
        pytest.skip("Bash is required")
    return candidate


@pytest.fixture
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("t")
    (root / "pyproject.toml").touch()
    (root / "globalsplat").mkdir()
    scripts = root / "scripts/slurm"
    scripts.mkdir(parents=True)
    shutil.copy2(SCRIPTS / "nfcgs_transform_grid.sh", scripts)
    capture = '#!/usr/bin/env bash\n' + '\n'.join(
        f'echo "{key}=${{{key}:-}}"' for key in (
            "TRANSFORM", "TRANSFORM_HIDDEN", "GRID_TASK_ID", "RANK", "RATE_LAMBDA",
            "USE_RESIDUAL", "USE_MORTON", "OUTPUT_ROOT", "EXPERIMENT_PREFIX",
            "MICRO_BATCH", "ACCUMULATE", "SEED", "DUMMY_GPU_LOAD", "RESUME_CHECKPOINT",
            "CHECKPOINT", "EXPECTED_STEP", "PROTOCOL", "CONTEXT", "DRY_RUN", "CHECKPOINT_ONLY",
            "MAX_SCENES", "SCORE_MEAN_CONDITION", "SCORE_CHANNEL_CONTEXT", "SCORE_SPATIAL_CONTEXT",
            "SCORE_SLICE_CHANNELS", "SCORE_CONTEXT_HIDDEN",
        )
    ) + '\n'
    for name in ("train_nfcgs_paper_recipe.slurm", "eval_nfcgs_rank56.slurm"):
        (scripts / name).write_text(capture, encoding="utf-8", newline="\n")
    return root


def run(bash, repo, script, task, *args, **extra_env):
    env = os.environ.copy()
    env.update(
        SLURM_ARRAY_TASK_ID=str(task), OUTPUT_ROOT="/stale", CHECKPOINT="/stale.ckpt",
        TRANSFORM="wrong", TRANSFORM_HIDDEN="999", GRID_TASK_ID="999", USE_MORTON="false",
        MICRO_BATCH="2", ACCUMULATE="4", SEED="99", RESUME_CHECKPOINT="/stale.ckpt",
        DRY_RUN="true", CHECKPOINT_ONLY="true", EXPECTED_STEP="2", PROTOCOL="fixed", CONTEXT="36",
        MAX_SCENES="2", SCORE_MEAN_CONDITION="true", SCORE_CHANNEL_CONTEXT="true",
        SCORE_SPATIAL_CONTEXT="true", SCORE_SLICE_CHANNELS="8", SCORE_CONTEXT_HIDDEN="128",
    )
    env.update(extra_env)
    return subprocess.run([bash, str(SCRIPTS / script), *args], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=30)


def conditions(task):
    arm = task % 8
    return ("linear" if task < 8 else "nonlinear32", 56 if arm < 4 else 80,
            "0p0064" if arm % 4 < 2 else "0p0256", "on" if arm % 2 == 0 else "off")


@pytest.mark.parametrize("task", range(16))
def test_training_conditions_and_matched_recipe(bash, repo, task):
    result = run(bash, repo, "train_nfcgs_transform.slurm", task)
    assert result.returncode == 0, result.stderr
    tag, rank, rate, residual = conditions(task)
    assert f"TRANSFORM={'linear' if task < 8 else 'nonlinear'}\n" in result.stdout
    for value in ("TRANSFORM_HIDDEN=32", f"GRID_TASK_ID={task % 8}", f"RANK={rank}",
                  f"RATE_LAMBDA={rate.replace('p', '.')}", f"USE_RESIDUAL={str(residual == 'on').lower()}",
                  "USE_MORTON=true", "MICRO_BATCH=1", "ACCUMULATE=8", "SEED=111123", "DUMMY_GPU_LOAD=false"):
        assert value + "\n" in result.stdout
    assert f"nfcgs_paper24_transform_train/{tag}" in result.stdout
    assert "RESUME_CHECKPOINT=\n" in result.stdout
    assert "OUTPUT_ROOT=/stale" not in result.stdout


def touch_checkpoint(repo, task, version=0, step=50000, fresh=False):
    tag, rank, rate, residual = conditions(task)
    if task < 8 and not fresh:
        root = repo / "outputs/nfcgs_paper24_subset_train"
        prefix = "nfcgs_paper24_subset"
    else:
        root = repo / "outputs/nfcgs_paper24_transform_train" / tag
        prefix = f"nfcgs_transform_{tag}"
    experiment = f"{prefix}_rank{rank}_lambda{rate}_residual_{residual}_morton_on"
    path = root / f"rank{rank}/lambda{rate}/residual_{residual}/checkpoints" / experiment / f"version_{version}/step{step:09d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.mark.parametrize("task", range(16))
def test_evaluation_uses_explicit_version_and_correct_control_source(bash, repo, task):
    chosen = touch_checkpoint(repo, task)
    touch_checkpoint(repo, task, version=4)
    result = run(bash, repo, "eval_nfcgs_transform.slurm", task)
    assert result.returncode == 0, result.stderr
    assert chosen.relative_to(repo).as_posix() in result.stdout
    assert "version_4" not in result.stdout
    assert "\nCHECKPOINT=/stale" not in result.stdout
    assert "OUTPUT_ROOT=/stale" not in result.stdout
    for value in ("EXPECTED_STEP=50000", "PROTOCOL=all", "CONTEXT=12", "USE_MORTON=true",
                  "DRY_RUN=false", "CHECKPOINT_ONLY=false", "MAX_SCENES=null",
                  "SCORE_MEAN_CONDITION=false", "SCORE_CHANNEL_CONTEXT=false",
                  "SCORE_SPATIAL_CONTEXT=false", "SCORE_SLICE_CHANNELS=16", "SCORE_CONTEXT_HIDDEN=64"):
        assert value + "\n" in result.stdout
    # Codec rank must not leak into Lightning's distributed rank environment.
    assert "\nRANK=\n" in result.stdout


def test_no_fallback_to_another_checkpoint_version(bash, repo):
    touch_checkpoint(repo, 8, version=4)
    result = run(bash, repo, "eval_nfcgs_transform.slurm", 8)
    assert result.returncode != 0
    assert "requested transform checkpoint is missing" in result.stderr


def test_fresh_linear_checkpoint_requires_explicit_source(bash, repo):
    chosen = touch_checkpoint(repo, 0, fresh=True)
    result = run(bash, repo, "eval_nfcgs_transform.slurm", 0, TRANSFORM_LINEAR_SOURCE="fresh")
    assert result.returncode == 0, result.stderr
    assert chosen.relative_to(repo).as_posix() in result.stdout


@pytest.mark.parametrize("script", ["train_nfcgs_transform.slurm", "eval_nfcgs_transform.slurm"])
def test_plan_lists_all_conditions_without_starting_work(bash, repo, script):
    result = run(bash, repo, script, "", "plan")
    assert result.returncode == 0, result.stderr
    rows = [line for line in result.stdout.splitlines() if line.startswith("TASK=")]
    assert len(rows) == 16
    assert len(set(rows)) == 16


@pytest.mark.parametrize("task", [-1, 16, "garbage"])
def test_invalid_task_is_rejected(bash, repo, task):
    result = run(bash, repo, "train_nfcgs_transform.slurm", task)
    assert result.returncode != 0
    assert "task id 0 through 15" in result.stderr


def test_selected_completed_tasks_do_not_require_unfinished_or_linear_runs(bash, repo):
    for task in range(8, 15):
        touch_checkpoint(repo, task)
    result = run(bash, repo, "eval_nfcgs_transform.slurm", "", "check", "8-14")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("TASK=") == 7
    assert "All 7 selected checkpoint files exist" in result.stdout
    assert "TASK=15 " not in result.stdout


def test_selected_check_reports_all_missing_without_fallback(bash, repo):
    touch_checkpoint(repo, 8)
    touch_checkpoint(repo, 14, step=49500)
    touch_checkpoint(repo, 14, version=1)
    result = run(bash, repo, "eval_nfcgs_transform.slurm", "", "check", "8,14-15")
    assert result.returncode != 0
    assert result.stdout.count("TASK=") == 3
    assert result.stderr.count("requested transform checkpoint is missing") == 2
    assert "2/3 selected checkpoints failed check" in result.stderr


def test_task_selection_deduplicates_overlapping_ranges(bash, repo):
    result = run(bash, repo, "eval_nfcgs_transform.slurm", "", "plan", "8,10-12,11,14")
    assert result.returncode == 0, result.stderr
    rows = [line.split()[0] for line in result.stdout.splitlines() if line.startswith("TASK=")]
    assert rows == ["TASK=8", "TASK=10", "TASK=11", "TASK=12", "TASK=14"]


@pytest.mark.parametrize("selection", ["", "16", "8-16", "14-8", "-1", "8,,9", "8,", ",8", "8%2", "x", "8\n9"])
def test_invalid_preflight_selection_is_rejected(bash, repo, selection):
    result = run(bash, repo, "eval_nfcgs_transform.slurm", "", "plan", selection)
    assert result.returncode != 0
    # Windows argument parsing may split a newline into extra arguments.
    assert "ERROR:" in result.stderr or "Usage:" in result.stderr
    assert "TASK=" not in result.stdout


@pytest.mark.parametrize("mode", ["plan", "check", "validate", "run"])
def test_unexpected_extra_arguments_are_rejected(bash, repo, mode):
    result = run(bash, repo, "eval_nfcgs_transform.slurm", "", mode, "8", "extra")
    assert result.returncode != 0
    assert "Usage:" in result.stderr


@pytest.mark.parametrize("validator_status", [0, 1])
def test_cpu_validation_arguments_and_failure_status(bash, repo, validator_status):
    for task in (14, 15):
        touch_checkpoint(repo, task)
    validator = repo / "mock-python"
    validator.write_text(
        '#!/usr/bin/env bash\n'
        'printf "VALIDATOR RANK_ENV=%s ARGS=" "${RANK:-unset}"\n'
        'printf " %s" "$@"\n'
        'printf "\\n"\n'
        'exit "${MOCK_VALIDATE_STATUS:-0}"\n',
        encoding="utf-8", newline="\n",
    )
    validator.chmod(0o755)
    result = run(
        bash, repo, "eval_nfcgs_transform.slurm", "", "validate", "14-15",
        TRANSFORM_PYTHON=validator.as_posix(), MOCK_VALIDATE_STATUS=str(validator_status),
    )
    assert (result.returncode == 0) == (validator_status == 0), result.stderr
    assert result.stdout.count("VALIDATOR RANK_ENV=unset ARGS=") == 2
    for residual in ("true", "false"):
        assert f"--rank 80 --step 50000 --transform nonlinear --transform-hidden 32 --residual {residual} --morton true" in result.stdout
    assert "--score-mean-condition false --score-channel-context false --score-spatial-context false" in result.stdout
    if validator_status:
        assert "2/2 selected checkpoints failed validate" in result.stderr
        assert "passed CPU" not in result.stdout
    else:
        assert "All 2 selected checkpoints passed CPU" in result.stdout


def test_default_eval_is_not_pinned_to_one_node():
    script = (SCRIPTS / "eval_nfcgs_transform.slurm").read_text(encoding="utf-8")
    assert "#SBATCH -w " not in script

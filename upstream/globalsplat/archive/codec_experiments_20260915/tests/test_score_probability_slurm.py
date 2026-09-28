"""Validate Full low-LR and reconstruction-locked probability grids locally."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/slurm"
FAMILY_FILES = (
    "nfcgs_full_cooldown_grid.sh",
    "train_nfcgs_full_cooldown.slurm",
    "eval_nfcgs_full_cooldown.slurm",
    "submit_nfcgs_full_cooldown.sh",
    "nfcgs_score_probability_grid.sh",
    "train_nfcgs_score_probability.slurm",
    "eval_nfcgs_score_probability.slurm",
    "submit_nfcgs_score_probability.sh",
)


@pytest.fixture
def bash():
    candidate = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not candidate or not Path(candidate).is_file():
        pytest.skip("Bash is required")
    return candidate


@pytest.fixture
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("score_probability")
    (root / "pyproject.toml").touch()
    (root / "globalsplat").mkdir()
    scripts = root / "scripts/slurm"
    scripts.mkdir(parents=True)
    for name in FAMILY_FILES:
        shutil.copy2(SCRIPTS / name, scripts)
    keys = (
        "TRANSFORM",
        "GRID_TASK_ID",
        "RANK",
        "RATE_LAMBDA",
        "USE_RESIDUAL",
        "USE_MORTON",
        "OUTPUT_ROOT",
        "EXPERIMENT_PREFIX",
        "FEATURE_CODEC_TRAIN_SCOPE",
        "WARM_START_CHECKPOINT",
        "MAX_STEPS",
        "OPTIMIZER_LR",
        "OPTIMIZER_LR_MILESTONES",
        "OPTIMIZER_LR_GAMMA",
        "QUANTILE_UPDATE_INTERVAL",
        "SCORE_MEAN_CONDITION",
        "SCORE_CHANNEL_CONTEXT",
        "SCORE_SPATIAL_CONTEXT",
        "SCORE_SPATIAL_PREDICTOR",
        "SCORE_SPATIAL_ENTROPY",
        "SCORE_SPATIAL_HIDDEN",
        "CHECKPOINT",
        "EXPECTED_STEP",
        "PROTOCOL",
        "CONTEXT",
        "MAX_SCENES",
    )
    capture = "#!/usr/bin/env bash\n" + "\n".join(
        f'echo "{key}=${{{key}:-}}"' for key in keys
    ) + "\n"
    for name in ("train_nfcgs_paper_recipe.slurm", "eval_nfcgs_rank56.slurm"):
        (scripts / name).write_text(capture, encoding="utf-8", newline="\n")
    return root


def run(bash, repo, script, task="", *args, **extra_env):
    env = os.environ.copy()
    env.update(
        SLURM_ARRAY_TASK_ID=str(task),
        RANK="999",
        GRID_TASK_ID="999",
        OUTPUT_ROOT="/stale",
        CHECKPOINT="/stale.ckpt",
        SCORE_SPATIAL_ENTROPY="stale",
        SCORE_SPATIAL_PREDICTOR="stale",
        FULL_COOLDOWN_RUN_TAG="test_run",
        SCORE_PROBABILITY_RUN_TAG="test_run",
    )
    env.update(extra_env)
    return subprocess.run(
        [bash, str(SCRIPTS / script), *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def cooldown_condition(task):
    return (
        ("p0_linear", "linear", "0p0064", 8),
        ("p2_residual7", "residual7", "0p0064", 8),
        ("p0_linear", "linear", "0p0256", 9),
        ("p2_residual7", "residual7", "0p0256", 9),
    )[task]


def probability_condition(task):
    return (
        ("e0_shared", "shared", "0p0064", 8),
        ("e0_shared", "shared", "0p0256", 9),
        ("e1_split", "split", "0p0064", 8),
        ("e1_split", "split", "0p0256", 9),
        ("e2_gaussian", "gaussian", "0p0064", 8),
        ("e2_gaussian", "gaussian", "0p0256", 9),
        ("e3_conditional_scale", "conditional_scale", "0p0064", 8),
        ("e3_conditional_scale", "conditional_scale", "0p0256", 9),
    )[task]


@pytest.mark.parametrize("task", range(4))
def test_full_cooldown_is_fair_constant_low_lr_continuation(bash, repo, task):
    result = run(bash, repo, "train_nfcgs_full_cooldown.slurm", task)
    assert result.returncode == 0, result.stderr
    arm, predictor, rate, parent_task = cooldown_condition(task)
    for value in (
        "FEATURE_CODEC_TRAIN_SCOPE=all",
        "MAX_STEPS=10000",
        "OPTIMIZER_LR=0.000001",
        "OPTIMIZER_LR_GAMMA=1.0",
        "QUANTILE_UPDATE_INTERVAL=500",
        "SCORE_MEAN_CONDITION=true",
        "SCORE_CHANNEL_CONTEXT=true",
        "SCORE_SPATIAL_CONTEXT=true",
        f"SCORE_SPATIAL_PREDICTOR={predictor}",
        "SCORE_SPATIAL_ENTROPY=shared",
    ):
        assert value + "\n" in result.stdout
    assert f"PARENT_TASK={parent_task}" in result.stdout
    assert f"/20260910_092000/m1c1s1/rank56/lambda{rate}/" in result.stdout
    assert "/version_0/step000050000.ckpt" in result.stdout
    assert f"nfcgs_full_lowlr10k/test_run/{arm}" in result.stdout
    assert "OUTPUT_ROOT=/stale" not in result.stdout
    assert "\nRANK=\n" in result.stdout


@pytest.mark.parametrize("task", range(8))
def test_probability_training_changes_only_probability_model(bash, repo, task):
    result = run(bash, repo, "train_nfcgs_score_probability.slurm", task)
    assert result.returncode == 0, result.stderr
    arm, entropy, rate, parent_task = probability_condition(task)
    for value in (
        "FEATURE_CODEC_TRAIN_SCOPE=score_probability",
        "MAX_STEPS=10000",
        "OPTIMIZER_LR=0.0001",
        "OPTIMIZER_LR_MILESTONES=[7000]",
        "OPTIMIZER_LR_GAMMA=0.1",
        "QUANTILE_UPDATE_INTERVAL=0",
        "SCORE_MEAN_CONDITION=true",
        "SCORE_CHANNEL_CONTEXT=true",
        "SCORE_SPATIAL_CONTEXT=true",
        "SCORE_SPATIAL_PREDICTOR=linear",
        f"SCORE_SPATIAL_ENTROPY={entropy}",
    ):
        assert value + "\n" in result.stdout
    assert f"PARENT_TASK={parent_task}" in result.stdout
    assert f"/20260910_092000/m1c1s1/rank56/lambda{rate}/" in result.stdout
    assert f"nfcgs_score_probability10k/test_run/{arm}" in result.stdout
    assert "OUTPUT_ROOT=/stale" not in result.stdout
    assert "\nRANK=\n" in result.stdout


@pytest.mark.parametrize("task", range(8))
def test_probability_eval_resolves_exact_10k_checkpoint(bash, repo, task):
    result = run(bash, repo, "eval_nfcgs_score_probability.slurm", task)
    assert result.returncode == 0, result.stderr
    arm, entropy, rate, _ = probability_condition(task)
    context = "m1c1s1" if entropy == "shared" else f"m1c1s1_{entropy}"
    experiment = (
        f"nfcgs_score_probability10k_{arm}_rank56_lambda{rate}_"
        f"residual_on_morton_on_{context}"
    )
    assert f"/{arm}/rank56/lambda{rate}/residual_on/checkpoints/" in result.stdout
    assert f"/{experiment}/version_0/step000010000.ckpt" in result.stdout
    assert "EXPECTED_STEP=10000\n" in result.stdout
    assert f"SCORE_SPATIAL_ENTROPY={entropy}\n" in result.stdout
    assert "\nCHECKPOINT=/stale.ckpt\n" not in result.stdout
    assert "\nRANK=\n" in result.stdout


@pytest.mark.parametrize(
    ("script", "count"),
    (("train_nfcgs_full_cooldown.slurm", 4), ("train_nfcgs_score_probability.slurm", 8)),
)
def test_plans_list_unique_conditions(bash, repo, script, count):
    result = run(bash, repo, script, "", "plan")
    assert result.returncode == 0, result.stderr
    rows = [line for line in result.stdout.splitlines() if line.startswith("TASK=")]
    assert len(rows) == count
    assert len(set(rows)) == count


@pytest.mark.parametrize(
    ("script", "tasks"),
    (("submit_nfcgs_full_cooldown.sh", "0-3"), ("submit_nfcgs_score_probability.sh", "0-7")),
)
def test_submit_dry_run_is_non_mutating(bash, repo, script, tasks):
    result = run(bash, repo, script, "", "dry-run")
    assert result.returncode == 0, result.stderr
    assert f"DRY_RUN tasks={tasks}" in result.stdout
    assert not (repo / "logs").exists()


def test_new_launchers_pin_one_gpu_on_ariel_v12():
    for name in (
        "train_nfcgs_full_cooldown.slurm",
        "eval_nfcgs_full_cooldown.slurm",
        "train_nfcgs_score_probability.slurm",
        "eval_nfcgs_score_probability.slurm",
    ):
        script = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "#SBATCH -w ariel-v12" in script
        assert "#SBATCH --gres=gpu:1" in script
    for name in (
        "submit_nfcgs_full_cooldown.sh",
        "submit_nfcgs_score_probability.sh",
    ):
        submitter = (SCRIPTS / name).read_text(encoding="utf-8")
        assert submitter.count("--nodelist=ariel-v12 --gres=gpu:1") == 2
        assert '--dependency="aftercorr:${TRAIN_JOB}"' in submitter

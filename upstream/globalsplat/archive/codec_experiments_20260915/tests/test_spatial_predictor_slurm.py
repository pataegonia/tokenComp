"""Validate legacy tasks and the additional Full parents without Slurm or a GPU."""

import os
from pathlib import Path
import re
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
    root = tmp_path_factory.mktemp("sp")
    (root / "pyproject.toml").touch()
    (root / "globalsplat").mkdir()
    scripts = root / "scripts/slurm"
    scripts.mkdir(parents=True)
    shutil.copy2(SCRIPTS / "nfcgs_spatial_predictor_grid.sh", scripts)
    capture = '#!/usr/bin/env bash\n' + '\n'.join(
        f'echo "{key}=${{{key}:-}}"' for key in (
            "TRANSFORM",
            "TRANSFORM_HIDDEN",
            "GRID_TASK_ID",
            "RANK",
            "RATE_LAMBDA",
            "USE_RESIDUAL",
            "USE_MORTON",
            "OUTPUT_ROOT",
            "EXPERIMENT_PREFIX",
            "MICRO_BATCH",
            "ACCUMULATE",
            "SEED",
            "DUMMY_GPU_LOAD",
            "RESUME_CHECKPOINT",
            "WARM_START_CHECKPOINT",
            "MAX_STEPS",
            "CHECKPOINT",
            "EXPECTED_STEP",
            "PROTOCOL",
            "CONTEXT",
            "MAX_SCENES",
            "SCORE_MEAN_CONDITION",
            "SCORE_CHANNEL_CONTEXT",
            "SCORE_SPATIAL_CONTEXT",
            "SCORE_SPATIAL_PREDICTOR",
            "SCORE_SPATIAL_HIDDEN",
            "SCORE_SLICE_CHANNELS",
            "SCORE_CONTEXT_HIDDEN",
        )
    ) + '\n'
    for name in ("train_nfcgs_paper_recipe.slurm", "eval_nfcgs_rank56.slurm"):
        (scripts / name).write_text(capture, encoding="utf-8", newline="\n")
    return root


def run(bash, repo, script, task, *args, **extra_env):
    env = os.environ.copy()
    env.update(
        SLURM_ARRAY_TASK_ID=str(task),
        OUTPUT_ROOT="/stale",
        CHECKPOINT="/stale.ckpt",
        TRANSFORM="wrong",
        TRANSFORM_HIDDEN="999",
        GRID_TASK_ID="999",
        RANK="999",
        MAX_STEPS="999",
        PROTOCOL="fixed",
        CONTEXT="36",
        SCORE_CHANNEL_CONTEXT="stale",
        SCORE_SPATIAL_PREDICTOR="stale",
        SCORE_SPATIAL_HIDDEN="999",
        SCORE_SLICE_CHANNELS="8",
        SCORE_CONTEXT_HIDDEN="128",
        SPATIAL_PREDICTOR_RUN_TAG="test_run",
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


def condition(task):
    return (
        ("p0_linear", "linear", "0p0064", "m1c1s1", "true", 8),
        ("p0_linear", "linear", "0p0256", "m1c0s1", "false", 7),
        ("p1_residual3", "residual3", "0p0064", "m1c1s1", "true", 8),
        ("p1_residual3", "residual3", "0p0256", "m1c0s1", "false", 7),
        ("p2_residual7", "residual7", "0p0064", "m1c1s1", "true", 8),
        ("p2_residual7", "residual7", "0p0256", "m1c0s1", "false", 7),
        ("p0_linear", "linear", "0p0256", "m1c1s1", "true", 9),
        ("p1_residual3", "residual3", "0p0256", "m1c1s1", "true", 9),
        ("p2_residual7", "residual7", "0p0256", "m1c1s1", "true", 9),
    )[task]


@pytest.mark.parametrize("task", range(9))
def test_training_uses_selected_parent_and_matched_25k_recipe(bash, repo, task):
    result = run(bash, repo, "train_nfcgs_spatial_predictor.slurm", task)
    assert result.returncode == 0, result.stderr
    arm, predictor, rate, parent, channel, parent_task = condition(task)
    for value in (
        "TRANSFORM=nonlinear",
        "TRANSFORM_HIDDEN=32",
        f"GRID_TASK_ID={0 if rate == '0p0064' else 2}",
        f"RATE_LAMBDA={rate.replace('p', '.')}",
        "USE_RESIDUAL=true",
        "USE_MORTON=true",
        "MICRO_BATCH=2",
        "ACCUMULATE=4",
        "SEED=111123",
        "DUMMY_GPU_LOAD=false",
        "RESUME_CHECKPOINT=",
        "MAX_STEPS=25000",
        "SCORE_MEAN_CONDITION=true",
        f"SCORE_CHANNEL_CONTEXT={channel}",
        "SCORE_SPATIAL_CONTEXT=true",
        f"SCORE_SPATIAL_PREDICTOR={predictor}",
        "SCORE_SPATIAL_HIDDEN=32",
        "SCORE_SLICE_CHANNELS=16",
        "SCORE_CONTEXT_HIDDEN=64",
    ):
        assert value + "\n" in result.stdout
    assert f"PARENT_TASK={parent_task} PARENT_PRIOR={parent}" in result.stdout
    assert f"/20260910_092000/{parent}/rank56/lambda{rate}/residual_on/" in result.stdout
    assert "/version_0/step000050000.ckpt" in result.stdout
    assert f"nfcgs_spatial_predictor25k/test_run/{arm}/{parent}" in result.stdout
    assert "\nRANK=\n" in result.stdout
    assert "OUTPUT_ROOT=/stale" not in result.stdout


@pytest.mark.parametrize("task", range(9))
def test_evaluation_uses_only_the_new_25k_checkpoint(bash, repo, task):
    result = run(bash, repo, "eval_nfcgs_spatial_predictor.slurm", task)
    assert result.returncode == 0, result.stderr
    arm, predictor, rate, parent, channel, _ = condition(task)
    context_tag = parent if predictor == "linear" else f"{parent}_{predictor}h32"
    experiment = (
        f"nfcgs_spatial_predictor25k_{arm}_rank56_lambda{rate}_"
        f"residual_on_morton_on_{context_tag}"
    )
    assert f"/{arm}/{parent}/rank56/lambda{rate}/residual_on/checkpoints/" in result.stdout
    assert f"/{experiment}/version_0/step000025000.ckpt" in result.stdout
    for value in (
        "EXPECTED_STEP=25000",
        "PROTOCOL=all",
        "CONTEXT=12",
        "MAX_SCENES=null",
        "SCORE_MEAN_CONDITION=true",
        f"SCORE_CHANNEL_CONTEXT={channel}",
        "SCORE_SPATIAL_CONTEXT=true",
        f"SCORE_SPATIAL_PREDICTOR={predictor}",
        "SCORE_SPATIAL_HIDDEN=32",
    ):
        assert value + "\n" in result.stdout
    assert "step000050000.ckpt" not in result.stdout
    assert "\nCHECKPOINT=/stale.ckpt\n" not in result.stdout
    assert "\nRANK=\n" in result.stdout


@pytest.mark.parametrize(
    "script",
    ["train_nfcgs_spatial_predictor.slurm", "eval_nfcgs_spatial_predictor.slurm"],
)
def test_plan_lists_nine_unique_conditions(bash, repo, script):
    result = run(bash, repo, script, "", "plan")
    assert result.returncode == 0, result.stderr
    rows = [line for line in result.stdout.splitlines() if line.startswith("TASK=")]
    assert len(rows) == 9
    assert len(set(rows)) == 9
    assert sum("PARENT_TASK=8" in row for row in rows) == 3
    assert sum("PARENT_TASK=7" in row for row in rows) == 3
    assert sum("PARENT_TASK=9" in row for row in rows) == 3


@pytest.mark.parametrize("task", [-1, 9, "garbage"])
def test_invalid_task_is_rejected(bash, repo, task):
    result = run(bash, repo, "train_nfcgs_spatial_predictor.slurm", task)
    assert result.returncode != 0
    assert "task id 0 through 8" in result.stderr


def test_launchers_and_submitter_pin_one_gpu_on_ariel_v12():
    for name in (
        "train_nfcgs_spatial_predictor.slurm",
        "eval_nfcgs_spatial_predictor.slurm",
    ):
        script = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "#SBATCH -w ariel-v12" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --array=6-8" in script
    submitter = (SCRIPTS / "submit_nfcgs_spatial_predictor.sh").read_text(
        encoding="utf-8"
    )
    assert 'TASKS="${TASKS:-6-8}"' in submitter
    assert submitter.count("--nodelist=ariel-v12 --gres=gpu:1") == 2
    assert '--dependency="aftercorr:${TRAIN_JOB}"' in submitter


@pytest.fixture
def submit_env(repo):
    mock_bin = repo / "bin"
    mock_bin.mkdir()
    sbatch = mock_bin / "sbatch"
    sbatch.write_text(
        '#!/usr/bin/env bash\n'
        'printf "RANK=%s " "${RANK:-unset}" >> "$MOCK_SBATCH_LOG"\n'
        'printf "<%s> " "$@" >> "$MOCK_SBATCH_LOG"\n'
        'printf "\\n" >> "$MOCK_SBATCH_LOG"\n'
        'if [[ "$*" == *train_nfcgs_spatial_predictor.slurm* ]]; then\n'
        '  echo "91001;cluster"\n'
        'else\n'
        '  echo "91002;cluster"\n'
        'fi\n',
        encoding="utf-8", newline="\n",
    )
    sbatch.chmod(0o755)
    return {
        "PATH": str(mock_bin) + os.pathsep + os.environ["PATH"],
        "MOCK_SBATCH_LOG": (repo / "sbatch.log").as_posix(),
        "SPATIAL_PREDICTOR_PARENT_ROOT": (repo / "p").as_posix(),
        "SPATIAL_PREDICTOR_PARENT_RUN_TAG": "r",
        "TASKS": "",
    }


def touch_parent(repo, parent="m1c1s1"):
    experiment = f"nfcgs_nonlinear_score_context_joint_rank56_lambda0p0256_residual_on_morton_on_{parent}"
    checkpoint = (
        repo / "p/r" / parent / "rank56/lambda0p0256/residual_on/checkpoints"
        / experiment / "version_0/step000050000.ckpt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    return checkpoint


def test_default_submit_adds_only_full_tasks_and_requires_only_their_parent(bash, repo, submit_env):
    touch_parent(repo)
    result = run(bash, repo, "submit_nfcgs_spatial_predictor.sh", "", **submit_env)
    assert result.returncode == 0, result.stderr
    rows = [line.split()[0] for line in result.stdout.splitlines() if line.startswith("TASK=")]
    assert rows == ["TASK=6", "TASK=7", "TASK=8"]
    assert result.stdout.count("PARENT_CHECKPOINT=") == 1
    calls = (repo / "sbatch.log").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    for call in calls:
        assert "<--array=6-8>" in call
        assert "<--nodelist=ariel-v12> <--gres=gpu:1>" in call
        assert "SPATIAL_PREDICTOR_RUN_TAG=test_run" in call
        assert call.startswith("RANK=unset ")
    assert "<--dependency=aftercorr:91001>" in calls[1]
    assert calls[0].endswith("<scripts/slurm/train_nfcgs_spatial_predictor.slurm> ")
    assert calls[1].endswith("<scripts/slurm/eval_nfcgs_spatial_predictor.slurm> ")


def test_missing_full_parent_does_not_fall_back_to_spatial_or_submit(bash, repo, submit_env):
    touch_parent(repo, parent="m1c0s1")
    result = run(bash, repo, "submit_nfcgs_spatial_predictor.sh", "", **submit_env)
    assert result.returncode != 0
    assert "missing selected spatial-predictor parent" in result.stderr
    assert "m1c1s1" in result.stderr
    assert not (repo / "sbatch.log").exists()


def test_submit_dry_run_needs_no_parents_and_makes_no_submissions(bash, repo, submit_env):
    submit_env["TASKS"] = "6-8,7%2"
    result = run(bash, repo, "submit_nfcgs_spatial_predictor.sh", "", "dry-run", **submit_env)
    assert result.returncode == 0, result.stderr
    rows = [line.split()[0] for line in result.stdout.splitlines() if line.startswith("TASK=")]
    assert rows == ["TASK=6", "TASK=7", "TASK=8"]
    assert "DRY_RUN tasks=6-8,7%2" in result.stdout
    assert not (repo / "sbatch.log").exists()
    assert not (repo / "logs").exists()


@pytest.mark.parametrize("selection", ["9", "8-6", "6,,7", "6-9", "6-8%0", "x"])
def test_invalid_submission_tasks_fail_before_sbatch(bash, repo, submit_env, selection):
    submit_env["TASKS"] = selection
    result = run(bash, repo, "submit_nfcgs_spatial_predictor.sh", "", **submit_env)
    assert result.returncode != 0
    assert "ERROR:" in result.stderr
    assert not (repo / "sbatch.log").exists()


@pytest.mark.parametrize("task", [6, 7, 8])
def test_full_eval_path_matches_actual_training_runner(bash, repo, task):
    for name in ("train_nfcgs_paper_recipe.slurm", "eval_nfcgs_rank56.slurm"):
        shutil.copy2(SCRIPTS / name, repo / "scripts/slurm" / name)
    train = run(bash, repo, "train_nfcgs_spatial_predictor.slurm", task, DRY_RUN="true")
    evaluate = run(bash, repo, "eval_nfcgs_spatial_predictor.slurm", task, DRY_RUN="true")
    assert train.returncode == 0, train.stderr
    assert evaluate.returncode == 0, evaluate.stderr
    experiment = re.search(r" EXPERIMENT=(\S+)", train.stdout).group(1)
    root = next(line.removeprefix("RUN_ROOT=") for line in train.stdout.splitlines() if line.startswith("RUN_ROOT="))
    assert f"CHECKPOINT={root}/checkpoints/{experiment}/version_0/step000025000.ckpt" in evaluate.stdout

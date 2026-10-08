"""Checkpoint recovery and Slurm duplication/race checks, without a cluster."""

from contextlib import nullcontext
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import resume_hyper1d_rd4 as recovery
import run_hyper1d as runner
import prune_hyper1d_rd4_checkpoints as pruning


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rd4_resume_test_")
        self.repo = Path(temporary.name).resolve()
        self.repo.relative_to(Path(tempfile.gettempdir()).resolve())
        self.addCleanup(temporary.cleanup)
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        self.tasks = recovery.task_specs(self.repo, 442942, 442943, "0.0032")
        self.root = self.repo / "outputs/hyper1d_rd4_50k"
        self.root.mkdir(parents=True)

    def save(self, task, step, *, name=None, scheduler=True, config_changes=None):
        variant = task["variant"]
        config = {"codec_type": "hyper1d", "architecture": "plain4" if variant == "plain4" else "legacy",
                  "paths": 2 if variant.startswith(("dual_", "lowrank_")) else 1,
                  "base_rank": 56 if variant.startswith("lowrank_") else 0,
                  "use_morton": variant.endswith("_on"), "strides": (2, 2),
                  "n": 256 if variant == "plain4" else 192, "m": 512 if variant == "plain4" else 320}
        config.update(config_changes or {})
        output = Path(task["output"])
        path = (output / "hyper1d_initial.ckpt" if step == 0
                else output / "checkpoints/hyper1d_12h/version_0" / (name or f"step{step:09d}.ckpt"))
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"state_dict": {"model.feature_codec.weight": torch.ones(4)},
                 "feature_codec_config": config, "global_step": step,
                 "optimizer_states": [{"state": {}, "param_groups": []}]}
        if scheduler:
            state["lr_schedulers"] = [{"last_epoch": step}]
        torch.save(state, path)
        return path

    def test_original_mapping_is_preserved_and_every_task_uses_one_gpu(self):
        self.assertEqual(len(self.tasks), 18)
        self.assertTrue(all(task["gpus"] == 1 for task in self.tasks))
        self.assertEqual((self.tasks[4]["original_job"], self.tasks[4]["variant"]), ("442942_4", "dual_off"))
        self.assertEqual((self.tasks[-1]["original_job"], self.tasks[-1]["lambda"]), ("442943_6", "0.0032"))

    def test_truncated_latest_falls_back_to_valid_checkpoint(self):
        valid = self.save(self.tasks[4], 35000)
        invalid = valid.with_name("step000037500.ckpt")
        invalid.write_bytes(b"truncated")
        source = recovery.choose_source(self.tasks[4], self.repo)
        self.assertEqual((source["kind"], source["step"], source["checkpoint"]), ("resume", 35000, str(valid)))

    def test_missing_scheduler_cannot_reset_optimizer(self):
        valid = self.save(self.tasks[4], 34000)
        self.save(self.tasks[4], 35000, scheduler=False)
        self.assertEqual(recovery.choose_source(self.tasks[4], self.repo)["checkpoint"], str(valid))

    def test_wrong_architecture_and_filename_step_are_rejected(self):
        valid = self.save(self.tasks[4], 33000)
        self.save(self.tasks[4], 35000, config_changes={"paths": 1})
        self.save(self.tasks[4], 34000, name="step000036000.ckpt")
        self.assertEqual(recovery.choose_source(self.tasks[4], self.repo)["checkpoint"], str(valid))

    def test_invalid_training_files_never_trigger_fresh_training(self):
        self.save(self.tasks[4], 35000, scheduler=False)
        self.save(self.tasks[4], 0)
        with self.assertRaisesRegex(ValueError, "refusing to reset progress"):
            recovery.choose_source(self.tasks[4], self.repo)

    def test_final_is_complete_and_numbered_wins_same_step(self):
        self.save(self.tasks[4], 35000, name="last.ckpt")
        numbered = self.save(self.tasks[4], 35000)
        self.assertEqual(recovery.choose_source(self.tasks[4], self.repo)["checkpoint"], str(numbered))
        final = self.save(self.tasks[4], 50000)
        self.assertEqual(recovery.choose_source(self.tasks[4], self.repo),
                         {"kind": "complete", "checkpoint": str(final), "step": 50000})

    def test_unique_newer_last_is_retained_as_resume_source(self):
        self.save(self.tasks[4], 35000)
        last = self.save(self.tasks[4], 35500, name="last.ckpt")
        self.assertEqual(recovery.choose_source(self.tasks[4], self.repo)["checkpoint"], str(last))

    def test_initial_or_fresh_is_used_only_when_no_progress_exists(self):
        self.assertEqual(recovery.choose_source(self.tasks[-1], self.repo)["kind"], "fresh")
        self.save(self.tasks[-1], 0)
        self.assertEqual(recovery.choose_source(self.tasks[-1], self.repo)["kind"], "initial")

    def test_completed_manifest_can_reuse_an_older_run(self):
        final = self.save(self.tasks[11], 50000)
        older = self.repo / "outputs/older_run/step000050000.ckpt"
        older.parent.mkdir()
        final.rename(older)
        recovery.write_manifest(self.tasks[11], {"checkpoint": str(older)})
        self.assertEqual(recovery.choose_source(self.tasks[11], self.repo)["checkpoint"], str(older))

    def test_current_queue_and_six_completions_yield_twelve_tasks(self):
        for index in (0, 1, 2, 3, 11, 12):
            self.save(self.tasks[index], 50000)
        queue = {f"442943_{i}": {"name": "gs-h1d-rd4-lr", "state": "PENDING"} for i in (4, 5, 6)}
        groups, pending = recovery.inspect_tasks(self.tasks, self.repo, queue, True)
        self.assertEqual({key: len(value) for key, value in groups.items()}, {"one_gpu": 7, "lowrank": 5})
        self.assertEqual(pending, ["442943_4", "442943_5", "442943_6"])
        self.assertEqual(sum(task["gpus"] for value in groups.values() for task in value), 12)

    def test_active_original_and_previous_recovery_are_skipped(self):
        plan = {"jobs": {"one_gpu": "999"}, "groups": {"one_gpu": [self.tasks[4]]}}
        recovery.write_json(self.root / "resumes/previous/plan.json", plan)
        queue = {"999_0": {"name": "gs-h1d-rd4-r1g", "state": "PENDING"},
                 "442943_4": {"name": "gs-h1d-rd4-lr", "state": "RUNNING"}}
        with patch.object(recovery, "choose_source", return_value={"kind": "fresh", "checkpoint": None, "step": 0}):
            groups, pending = recovery.inspect_tasks(self.tasks, self.repo, queue, True)
        selected = {task["original_job"] for value in groups.values() for task in value}
        self.assertNotIn("442942_4", selected)
        self.assertNotIn("442943_4", selected)
        self.assertEqual(pending, [])

    def test_one_gpu_resume_preserves_effective_batch_and_saving_policy(self):
        for task in self.tasks:
            source = {"kind": "resume", "checkpoint": str(self.repo / "step000035000.ckpt"), "step": 35000}
            command = recovery.train_command(task, source, self.repo / "vanilla.ckpt", self.repo)
            args = runner.parse_args(command[2:])
            self.assertTrue(args.resume)
            self.assertEqual(args.devices * args.batch_size * args.accumulate, 8)
            self.assertEqual((args.devices, args.launcher), (1, "python"))
            self.assertEqual((args.checkpoint_every, args.checkpoint_keep, args.no_save_last), (5000, 2, True))
            self.assertEqual((args.max_steps, args.validate_every, args.rate_lambda), (50000, 2000, float(task["lambda"])))
            self.assertIsNone(args.architecture)  # Restore the saved codec, not a new one.

    def test_new_initialization_keeps_rank_morton_and_architecture(self):
        for task in self.tasks:
            command = recovery.train_command(task, {"kind": "fresh"}, self.repo / "vanilla.ckpt", self.repo)
            args = runner.parse_args(command[2:])
            self.assertEqual(args.morton, task["variant"].endswith("_on"))
            self.assertEqual(args.base_rank or 0, 56 if task["variant"].startswith("lowrank_") else 0)
            self.assertEqual(args.architecture, "plain4" if task["variant"] == "plain4" else "legacy")

    def test_runner_forwards_retention_and_no_duplicate_last(self):
        args = runner.parse_args(["train", "--checkpoint", "saved.ckpt", "--resume",
                                  "--checkpoint-every", "5000", "--checkpoint-keep", "2", "--no-save-last"])
        config = SimpleNamespace(to_dict=lambda: {"architecture": "legacy", "paths": 2, "base_rank": 56})
        with patch.dict(sys.modules, {"globalsplat.compression": SimpleNamespace(Hyper1DConfig=object)}):
            command = runner.build_command(args, self.repo, self.repo / "saved.ckpt", config)
        self.assertIn("checkpointing.save_top_k=2", command)
        self.assertIn("checkpointing.save_last=false", command)
        self.assertIn("checkpointing.every_n_train_steps=5000", command)

    def prepare_launcher(self):
        for relative in ("scripts/run_hyper1d.py", "scripts/slurm/resume_hyper1d_rd4.slurm"):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        vanilla = self.repo / "vanilla.ckpt"
        vanilla.touch()
        return vanilla

    def test_pending_to_running_race_never_submits_duplicate(self):
        vanilla = self.prepare_launcher()
        pending = {"442943_4": {"name": "gs-h1d-rd4-lr", "state": "PENDING"}}
        running = {"442943_4": {"name": "gs-h1d-rd4-lr", "state": "RUNNING"}}
        with (patch.object(recovery, "REPO", self.repo),
              patch.object(recovery, "lock_file", return_value=nullcontext()),
              patch.object(recovery, "read_queue", side_effect=[pending, running, running]),
              patch.object(recovery, "choose_source", return_value={"kind": "fresh", "checkpoint": None, "step": 0}),
              patch.object(recovery.subprocess, "run", return_value=SimpleNamespace(stdout="999\n")) as run):
            recovery.main(["--submit", "--replace-pending", "--vanilla-checkpoint", str(vanilla)])
        cancel = run.call_args_list[0].args[0]
        self.assertIn("--state=PENDING", cancel)
        self.assertEqual(cancel[-1], "442943_4")
        plan = json.loads(next((self.root / "resumes").glob("*/plan.json")).read_text())
        selected = {task["original_job"] for value in plan["groups"].values() for task in value}
        self.assertNotIn("442943_4", selected)
        for call in run.call_args_list[1:]:
            command = call.args[0]
            self.assertIn("--gres=gpu:normal:1", command)
            self.assertIn("--ntasks=1", command)
            self.assertNotIn("%", next(value for value in command if value.startswith("--array=")))

    def test_missing_vanilla_blocks_before_cancel(self):
        self.prepare_launcher()
        pending = {"442943_4": {"name": "gs-h1d-rd4-lr", "state": "PENDING"}}
        with (patch.object(recovery, "REPO", self.repo),
              patch.object(recovery, "lock_file", return_value=nullcontext()),
              patch.object(recovery, "read_queue", return_value=pending),
              patch.object(recovery, "choose_source", return_value={"kind": "fresh", "checkpoint": None, "step": 0}),
              patch.object(recovery.subprocess, "run") as run):
            with self.assertRaises(FileNotFoundError):
                recovery.main(["--submit", "--replace-pending", "--vanilla-checkpoint", str(self.repo / "missing.ckpt")])
        run.assert_not_called()

    def test_partial_submission_records_success_for_safe_retry(self):
        groups = {"one_gpu": [self.tasks[4]], "lowrank": [self.tasks[-1]]}
        plan = {"groups": groups, "jobs": {}}
        path = self.root / "resumes/partial/plan.json"
        recovery.write_json(path, plan)
        with patch.object(recovery.subprocess, "run", side_effect=[SimpleNamespace(stdout="888\n"), subprocess.CalledProcessError(1, "sbatch")]):
            with self.assertRaises(subprocess.CalledProcessError):
                recovery.submit_groups(groups, path, plan, self.repo)
        self.assertEqual(json.loads(path.read_text())["jobs"], {"one_gpu": "888"})
        self.assertEqual(recovery.recovery_jobs(self.root, {"888_0": {}}), {"442942_4": "888_0"})

    def test_cleanup_protects_original_directory_of_active_recovery(self):
        plan = {"groups": {"one_gpu": [self.tasks[4]]}, "jobs": {"one_gpu": "888"}}
        recovery.write_json(self.root / "resumes/active/plan.json", plan)
        with patch.object(pruning.subprocess, "run", return_value=SimpleNamespace(stdout="888_0\n")):
            self.assertEqual(pruning.active_jobs(self.root), {"888_0", "442942_4"})

    def test_worker_resumes_then_publishes_final_and_prunes_only_that_run(self):
        task = self.tasks[4]
        self.save(task, 35000)
        self.save(task, 34000)
        other = self.save(self.tasks[5], 33000)
        plan_path = self.root / "resumes/worker/plan.json"
        recovery.write_json(plan_path, {"groups": {"one_gpu": [task]}, "vanilla": str(self.repo / "vanilla.ckpt")})

        def train(command, **kwargs):
            args = runner.parse_args(command[2:])
            self.assertTrue(args.resume)
            self.assertEqual(args.output, Path(task["output"]))
            self.save(task, 45000)
            self.save(task, 50000)

        with (patch.object(recovery, "REPO", self.repo),
              patch.object(recovery, "lock_file", return_value=nullcontext()),
              patch.object(recovery, "read_queue", return_value={}),
              patch.object(recovery, "choose_source", side_effect=lambda task: recovery_source(task, self.repo)),
              patch.dict(recovery.os.environ, {"SLURM_NTASKS": "1"}),
              patch.object(recovery.subprocess, "run", side_effect=train)):
            recovery.run_task(plan_path, "one_gpu", 0)
        final = Path(task["output"]) / "checkpoints/hyper1d_12h/version_0/step000050000.ckpt"
        self.assertEqual(list(final.parent.glob("*.ckpt")), [final])
        self.assertEqual(Path(task["manifest"]).read_text().strip(), str(final))
        self.assertTrue(other.is_file())

    def test_slurm_bootstrap_launches_one_task_with_separate_temp_directory(self):
        bash = (str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe")
                if sys.platform == "win32" else shutil.which("bash"))
        if not bash or not Path(bash).is_file():
            self.skipTest("Bash unavailable")
        profile = self.repo / "conda/etc/profile.d/conda.sh"
        profile.parent.mkdir(parents=True)
        profile.write_text("# Test activation only.\n")
        harness = self.repo / "harness.sh"
        harness.write_text('''set -euo pipefail
conda() { case "$1" in info) printf '%s\\n' "$MOCK_CONDA_BASE" ;; activate) return 0 ;; *) return 1 ;; esac; }
srun() { printf '%s\\n' "$@" > "$ARGUMENT_DUMP"; printf '%s\\n' "$TMPDIR" > "$TEMP_DUMP"; }
export -f conda srun
bash "$WRAPPER_SCRIPT"
''', encoding="utf-8")
        dump = self.repo / "arguments.txt"
        temporary = self.repo / "temp.txt"
        env = os.environ.copy()
        env.update(REPO_DIR=self.repo.as_posix(), MOCK_CONDA_BASE=(self.repo / "conda").as_posix(),
                   WRAPPER_SCRIPT=(SCRIPTS / "slurm/resume_hyper1d_rd4.slurm").as_posix(),
                   ARGUMENT_DUMP=dump.as_posix(), TEMP_DUMP=temporary.as_posix(),
                   GLOBALSPLAT_TMP_ROOT=(self.repo / "tmp").as_posix(),
                   RD_RESUME_PLAN=(self.root / "resumes/worker/plan.json").as_posix(),
                   RD_RESUME_GROUP="lowrank", SLURM_ARRAY_JOB_ID="999", SLURM_ARRAY_TASK_ID="4")
        result = subprocess.run([bash, str(harness)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(dump.read_text().splitlines(), ["--ntasks=1", "--kill-on-bad-exit=1", "python",
                         "scripts/resume_hyper1d_rd4.py", "--run-task", env["RD_RESUME_PLAN"], "lowrank", "4"])
        self.assertTrue(temporary.read_text().strip().endswith("/gs_r_999_4"))


# Keep the original function for the worker's patched REPO in synthetic tests.
recovery_source = recovery.choose_source


if __name__ == "__main__":
    unittest.main()

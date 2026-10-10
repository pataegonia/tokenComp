"""Verify destructive checkpoint cleanup with synthetic files and Slurm state."""

from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prune_checkpoints as prune


class PruneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="checkpoint_prune_test_")
        self.repo = Path(temporary.name).resolve()
        self.repo.relative_to(Path(tempfile.gettempdir()).resolve())
        self.addCleanup(temporary.cleanup)
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        self.root = self.repo / "outputs"
        self.directory = self.root / "hyper1d_example/job_123/checkpoints/hyper1d_12h/version_0"
        self.directory.mkdir(parents=True)

    def save(self, step, *, name=None, scheduler=True, directory=None):
        directory = directory or self.directory
        path = directory / (name or f"step{step:09d}.ckpt")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"state_dict": {"model.feature_codec.weight": torch.ones(4)}, "global_step": step,
                 "optimizer_states": [{"state": {}, "param_groups": []}]}
        if scheduler:
            state["lr_schedulers"] = [{"last_epoch": step}]
        torch.save(state, path)
        old = time.time() - 3600
        os.utime(path, (old, old))
        return path

    def plan(self, context=None, **kwargs):
        return prune.plan_group(self.directory, list(self.directory.glob("*.ckpt")), self.root,
                                context or prune.Context(), **kwargs)

    def test_completed_run_keeps_verified_final_only(self):
        self.save(45000)
        final = self.save(50000)
        self.save(50000, name="last.ckpt")
        files, keep, remove, invalid, target = self.plan()
        self.assertEqual(target, 50000)
        self.assertEqual({file.path for file in keep}, {final})
        self.assertEqual(len(remove), 2)
        self.assertEqual(invalid, [])

    def test_incomplete_run_keeps_two_distinct_resume_steps(self):
        self.save(25000)
        self.save(30000)
        self.save(35000)
        self.save(35000, name="last.ckpt")
        _, keep, remove, _, _ = self.plan()
        self.assertEqual({file.path.name for file in keep}, {"step000030000.ckpt", "step000035000.ckpt"})
        self.assertEqual(len(remove), 2)

    def test_corrupt_latest_falls_back_and_cannot_erase_resume_progress(self):
        self.save(30000)
        self.save(35000)
        bad = self.save(40000)
        bad.write_bytes(b"broken serialization")
        os.utime(bad, (time.time() - 3600, time.time() - 3600))
        _, keep, remove, invalid, _ = self.plan()
        self.assertEqual({file.path.name for file in keep}, {"step000030000.ckpt", "step000035000.ckpt"})
        self.assertIn(bad, {file.path for file in remove})
        self.assertTrue(invalid)

    def test_missing_scheduler_is_preserved_but_not_counted_as_resume_fallback(self):
        self.save(25000)
        self.save(30000)
        weights = self.save(35000, scheduler=False)
        _, keep, _, _, _ = self.plan()
        self.assertEqual(len(keep), 3)
        self.assertIn(weights, {file.path for file in keep})

    def test_no_valid_resume_checkpoint_preserves_everything(self):
        self.save(30000, scheduler=False)
        with self.assertRaisesRegex(ValueError, "no verified full resume"):
            self.plan()
        self.assertTrue((self.directory / "step000030000.ckpt").is_file())

    def test_unique_newer_last_wins_progress(self):
        self.save(30000)
        self.save(35000)
        last = self.save(35500, name="last.ckpt")
        _, keep, _, _, _ = self.plan()
        self.assertEqual({file.path.name for file in keep}, {last.name, "step000035000.ckpt"})

    def test_evaluation_milestone_is_not_deleted(self):
        baseline = self.save(16000)
        self.save(45000)
        final = self.save(50000)
        context = prune.Context(pinned={baseline: "matching evaluation baseline"})
        _, keep, _, _, _ = self.plan(context)
        self.assertEqual({file.path for file in keep}, {baseline, final})

    def test_last_alias_needed_by_resolver_is_kept(self):
        final = self.save(50000)
        last = self.save(50000, name="last.ckpt")
        context = prune.Context(pinned_names={"last.ckpt": "evaluation resolver"})
        _, keep, remove, _, _ = self.plan(context)
        self.assertEqual({file.path for file in keep}, {last, final})
        self.assertEqual(remove, [])

    def test_recent_files_are_protected(self):
        recent = self.save(10000)
        os.utime(recent, None)
        self.save(45000)
        self.save(50000)
        _, keep, _, _, _ = self.plan()
        self.assertEqual(keep[next(file for file in keep if file.path == recent)], "recently modified")

    def test_nfcgs_target_comes_from_saved_configuration(self):
        run = prune.run_directory(self.directory)
        config = run / "hydra/.hydra/config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("trainer:\n  max_steps: 500000\n")
        self.save(450000)
        self.save(500000)
        _, keep, remove, _, target = self.plan()
        self.assertEqual(target, 500000)
        self.assertEqual({file.path.name for file in keep}, {"step000500000.ckpt"})
        self.assertEqual(len(remove), 1)

    def test_larger_resume_limit_overrides_earlier_config(self):
        run = prune.run_directory(self.directory)
        config = run / "hydra/.hydra/config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("trainer:\n  max_steps: 16000\n")
        context = prune.Context(targets={run: 50000})
        self.assertEqual(prune.final_step(self.directory, context), 50000)

    def test_unknown_target_keeps_two_instead_of_assuming_completed(self):
        directory = self.root / "nfcgs/custom/checkpoints/model/version_0"
        self.save(450000, directory=directory)
        self.save(500000, directory=directory)
        _, keep, _, _, target = prune.plan_group(directory, list(directory.glob("*.ckpt")), self.root, prune.Context())
        self.assertIsNone(target)
        self.assertEqual(len(keep), 2)

    def test_initialization_pretrained_and_experiments_are_outside_candidate_set(self):
        self.save(50000)
        for path in (self.directory / "hyper1d_initial.ckpt", self.root / "initialization/step000001000.ckpt",
                     self.root / "pretrained/last.ckpt", self.root / "experiments/step000001000.ckpt"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.assertEqual(prune.checkpoint_groups(self.root), [(self.directory, [self.directory / "step000050000.ckpt"])])

    def test_original_array_and_recovery_output_both_protected(self):
        self.assertTrue(prune.is_active(self.directory, prune.Context(jobs={"123"})))
        run = prune.run_directory(self.directory)
        self.assertTrue(prune.is_active(self.directory, prune.Context(jobs={"999_0"}, active_dirs={run})))
        self.assertFalse(prune.is_active(self.directory, prune.Context(jobs={"1234"})))

    def test_resume_plan_maps_new_job_to_original_output(self):
        run = prune.run_directory(self.directory)
        path = self.root / "hyper1d_rd4_50k/resumes/test/plan.json"
        path.parent.mkdir(parents=True)
        source = self.save(35000)
        path.write_text(json.dumps({"jobs": {"one_gpu": "999"}, "groups": {"one_gpu": [
            {"output": str(run), "source": {"checkpoint": str(source)}}]}}))
        context = prune.Context(jobs={"999_0"})
        prune.resume_protection(context, self.root)
        self.assertTrue(prune.is_active(self.directory, context))
        self.assertIn(source, context.pinned)

    def test_old_training_input_is_not_permanently_pinned(self):
        source = self.save(35000)
        output = prune.run_directory(self.directory) / "checkpoints"
        text = f"python -m globalsplat.main output_dir={output.as_posix()} trainer.max_steps=50000 checkpointing.load={source.as_posix()}"
        context = prune.Context()
        prune.parse_log(text, self.root, context)
        self.assertNotIn(source, context.pinned)
        prune.parse_log(text, self.root, context, active=True)
        self.assertIn(source, context.pinned)
        self.assertTrue(prune.is_active(self.directory, context))

    def test_evaluated_checkpoint_and_dynamic_template_are_preserved(self):
        source = self.save(16000)
        context = prune.Context()
        prune.parse_log(f"python -m globalsplat.main mode=test checkpointing.load={source.as_posix()}", self.root, context)
        self.assertIn(source, context.pinned)
        script = self.repo / "scripts/slurm/eval_baseline.slurm"
        script.parent.mkdir(parents=True)
        script.write_text('CHECKPOINT="${ROOT}/${VARIANT}/step000016000.ckpt"\n')
        prune.script_references(context, self.repo, self.root)
        self.assertIn("step000016000.ckpt", context.pinned_names)

    def test_outside_root_is_rejected_and_mutated_directory_prevents_any_deletion(self):
        self.save(30000)
        self.save(35000)
        self.save(40000)
        files, keep, remove, _, _ = self.plan()
        self.save(45000)
        with self.assertRaisesRegex(ValueError, "directory changed"):
            prune.apply_group(self.directory, files, remove, self.root)
        self.assertTrue(all(file.path.is_file() for file in files))
        outside = self.repo / "outside.ckpt"
        outside.touch()
        with self.assertRaises(ValueError):
            prune.File.capture(outside, self.root)

    def test_changed_file_aborts_before_first_unlink(self):
        self.save(30000)
        self.save(35000)
        self.save(40000)
        files, _, remove, _, _ = self.plan()
        remove[0].path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "changed after planning"):
            prune.apply_group(self.directory, files, remove, self.root)
        self.assertTrue(all(file.path.is_file() for file in files))

    def test_completed_manifest_pin_is_collected(self):
        final = self.save(50000)
        manifest = self.root / "hyper1d_rd4_50k/manifests/one_gpu/job_123/variant.txt"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(str(final) + "\n")
        context = prune.collect_context(self.repo, self.root, set(), inspect_slurm=False)
        self.assertIn(final, context.pinned)

    def test_main_dry_run_apply_and_active_skip(self):
        old = self.save(45000)
        final = self.save(50000)
        with (patch.object(prune, "REPO", self.repo), patch.object(prune, "ROOT", self.root),
              patch.object(prune, "query_jobs", return_value=set()),
              patch.object(prune, "run_lock", return_value=nullcontext())):
            prune.main([])
            self.assertTrue(old.is_file())
            prune.main(["--apply"])
            self.assertFalse(old.exists())
            self.assertTrue(final.is_file())
        old = self.save(45000)
        with (patch.object(prune, "REPO", self.repo), patch.object(prune, "ROOT", self.root),
              patch.object(prune, "query_jobs", return_value={"123"}),
              patch.object(prune, "collect_context", return_value=prune.Context(jobs={"123"}))):
            prune.main(["--apply"])
        self.assertTrue(old.is_file())

    def test_new_job_during_planning_blocks_deletion(self):
        old = self.save(45000)
        self.save(50000)
        with (patch.object(prune, "REPO", self.repo), patch.object(prune, "ROOT", self.root),
              patch.object(prune, "query_jobs", side_effect=[set(), {"999"}])):
            prune.main(["--apply"])
        self.assertTrue(old.is_file())

    def test_unknown_active_job_is_not_hidden_by_another_mapped_job(self):
        log = self.repo / "logs/slurm/slurm-job-111.out"
        log.parent.mkdir(parents=True)
        run = prune.run_directory(self.directory)
        log.write_text(f"python -m globalsplat.main output_dir={(run / 'checkpoints').as_posix()} trainer.max_steps=50000\n")

        def control(command, **kwargs):
            job = command[-1]
            return SimpleNamespace(returncode=0, stdout=f"JobId={job} WorkDir={self.repo} StdOut={log if job == '111' else self.repo / 'missing.out'}", stderr="")

        with patch.object(prune.subprocess, "run", side_effect=control):
            context = prune.collect_context(self.repo, self.root, {"111", "222"})
        self.assertEqual(context.uncertain, ["222"])

    def test_no_slurm_refuses_apply_without_deletion(self):
        old = self.save(45000)
        self.save(50000)
        with (patch.object(prune, "REPO", self.repo), patch.object(prune, "ROOT", self.root),
              patch.object(prune, "query_jobs", side_effect=FileNotFoundError("squeue unavailable")),
              redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO())):
            with self.assertRaises(SystemExit):
                prune.main(["--apply"])
        self.assertTrue(old.is_file())


if __name__ == "__main__":
    unittest.main()

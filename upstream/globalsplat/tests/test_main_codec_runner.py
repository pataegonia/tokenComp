"""Main launcher configuration and historical snapshot integrity."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from globalsplat.compression import CodecConfig

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "main_codec_runner", ROOT / "scripts/run_nfcgs.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MainRunnerTests(unittest.TestCase):
    def composed(self, args):
        command, _, _ = runner.build_command(args)
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            return compose(config_name="main", overrides=command[3:])

    def test_eval_both_rates_resolve_main_checkpoints_and_actual_protocol(self):
        for rate in ("0.0064", "0.0256"):
            with self.subTest(rate=rate):
                args = runner.parse_args(["eval", "--rate-lambda", rate, "--dry-run"])
                command, checkpoint, _ = runner.build_command(args)
                self.assertIn("e1_split", checkpoint.parts)
                self.assertEqual(checkpoint.name, "step000010000.ckpt")
                config = self.composed(args)
                self.assertEqual(
                    CodecConfig.from_mapping(config.model.feature_codec), CodecConfig()
                )
                self.assertTrue(config.test.actual_bitstream)
                self.assertEqual(config.dataset.num_context_views, 12)
                self.assertEqual(config.dataset.num_target_views, 8)
                self.assertEqual(config.optimizer.batch_size, 1)
                self.assertEqual(config.seed, 0)
                self.assertEqual(config.curriculum.final_stage, 3)
                self.assertTrue(config.model.freeze_globalsplat)

    def test_training_scopes_keep_recipe_and_explicit_resume(self):
        for scope in ("all", "score_probability"):
            args = runner.parse_args(
                ["train", "--checkpoint", "main.ckpt", "--scope", scope]
            )
            config = self.composed(args)
            self.assertEqual(config.dataset.num_context_views, 24)
            self.assertEqual(config.dataset.num_target_views, 12)
            self.assertTrue(config.loss.subset_consistency)
            self.assertFalse(config.checkpointing.resume)
            self.assertFalse(config.checkpointing.auto_resume)
            self.assertEqual(
                config.optimizer.batch_size * config.trainer.accumulate_grad_batches, 8
            )
            self.assertEqual(
                config.loss.quantile_update_interval,
                0 if scope == "score_probability" else 500,
            )
            self.assertEqual(
                config.trainer.max_steps,
                10000 if scope == "score_probability" else 50000,
            )
        args = runner.parse_args(["train", "--checkpoint", "main.ckpt", "--resume"])
        self.assertTrue(self.composed(args).checkpointing.resume)

    def test_mean_offset_ablation_is_explicit_and_eval_keeps_mode(self):
        args = runner.parse_args(
            [
                "train", "--checkpoint", "main.ckpt", "--no-score-mean-offset",
                "--reset-score-mean-offset",
            ]
        )
        config = self.composed(args)
        self.assertFalse(config.model.score_mean_offset_enabled)
        self.assertTrue(config.checkpointing.allow_score_mean_offset_conversion)
        self.assertTrue(config.checkpointing.reset_score_mean_offset)
        args = runner.parse_args(
            ["eval", "--checkpoint", "ablated.ckpt", "--no-score-mean-offset"]
        )
        config = self.composed(args)
        self.assertFalse(config.model.score_mean_offset_enabled)
        self.assertFalse(config.checkpointing.allow_score_mean_offset_conversion)
        self.assertFalse(config.checkpointing.reset_score_mean_offset)
        for argv in (
            ["eval", "--reset-score-mean-offset"],
            ["train", "--from-scratch", "--reset-score-mean-offset"],
            ["train", "--checkpoint", "main.ckpt", "--resume", "--reset-score-mean-offset"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                runner.parse_args(argv)

    def test_dry_run_never_starts_processes_or_creates_output(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner.subprocess, "run") as run,
        ):
            output = Path(directory) / "not-created"
            runner.main(["eval", "--dry-run", "--output", str(output)])
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_training_requires_explicit_checkpoint(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["train"])

    def test_scratch_has_no_checkpoint_and_trains_full_model_with_curriculum(self):
        args = runner.parse_args(["train", "--from-scratch"])
        _, checkpoint, _ = runner.build_command(args)
        self.assertIsNone(checkpoint)
        config = self.composed(args)
        self.assertIsNone(config.checkpointing.load)
        self.assertFalse(config.checkpointing.resume)
        self.assertFalse(config.checkpointing.auto_resume)
        self.assertFalse(config.model.freeze_globalsplat)
        self.assertEqual(config.model.feature_codec_train_scope, "all")
        self.assertEqual(
            CodecConfig.from_mapping(config.model.feature_codec), CodecConfig()
        )
        self.assertEqual(list(config.curriculum.boundaries), [20000, 40000, 100000])
        self.assertEqual(config.curriculum.ramp_iters, 4000)
        self.assertEqual(config.loss.rate_ramp_steps, 20000)
        self.assertEqual(config.trainer.max_steps, 500000)
        self.assertEqual(config.optimizer.name, "adamw")
        self.assertEqual(list(config.optimizer.lr_milestones), [])
        self.assertEqual(config.optimizer.batch_size, 1)
        self.assertEqual(config.trainer.accumulate_grad_batches, 8)

    def test_joint_resume_keeps_backbone_trainable(self):
        args = runner.parse_args(
            [
                "train",
                "--joint",
                "--checkpoint",
                "joint.ckpt",
                "--resume",
            ]
        )
        config = self.composed(args)
        self.assertTrue(config.checkpointing.resume)
        self.assertFalse(config.model.freeze_globalsplat)
        self.assertEqual(config.experiment_name, "nfcgs_main_joint")

    def test_scratch_rejects_loading_and_probability_only(self):
        for argv in (
            ["eval", "--from-scratch"],
            ["train", "--from-scratch", "--checkpoint", "main.ckpt"],
            ["train", "--from-scratch", "--resume"],
            ["train", "--from-scratch", "--scope", "score_probability"],
            ["eval", "--joint"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                runner.parse_args(argv)

    def test_scratch_preflight_skips_checkpoint_validation(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner.subprocess, "run") as run,
        ):
            dataset = Path(directory)
            (dataset / "train").mkdir()
            (dataset / "train/index.json").write_text("{}")
            runner.main(["train", "--from-scratch", "--dataset-root", directory])
            run.assert_called_once()
            self.assertIn("globalsplat.main", run.call_args.args[0])

    def test_archived_source_matches_pre_cleanup_hashes(self):
        archive = ROOT / "archive/codec_experiments_20260915"
        manifest = json.loads(
            (archive / "manifest.json").read_text(encoding="utf-8-sig")
        )
        for item in manifest:
            with self.subTest(path=item["path"]):
                actual = hashlib.sha256(
                    (archive / item["path"]).read_bytes()
                ).hexdigest()
                self.assertEqual(actual.upper(), item["sha256"])

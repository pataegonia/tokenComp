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
            return compose(config_name="main", overrides=command[command.index("globalsplat.main") + 1:])

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

    def test_middle_rd_rate_uses_explicit_checkpoint_in_train_and_eval(self):
        for mode in ("train", "eval"):
            args = runner.parse_args([mode, "--checkpoint", "rd.ckpt", "--rate-lambda", "0.0128",
                "--token-order", "hilbert", "--score-context-schedule", "legacy", "--score-spatial-stages", "3"])
            config = self.composed(args)
            self.assertEqual(config.loss.rate_lambda, .0128)
            self.assertTrue(config.checkpointing.load.endswith("rd.ckpt"))
            self.assertEqual(config.model.feature_codec.token_order, "hilbert")
            self.assertTrue(config.model.feature_codec.use_residual)
            if mode == "eval":
                self.assertTrue(config.test.actual_bitstream)
                self.assertIsNone(config.test.max_scenes)
        with self.assertRaises(SystemExit):
            runner.parse_args(["eval", "--rate-lambda", "0.0128"])

    def test_rd_extension_rates_and_single_stream_control_compose(self):
        arms = (("hilbert", "dyadic4", 4, 16, True), ("morton", "dyadic4", 4, 16, True),
                ("hilbert", "legacy", 3, 16, True), ("morton", "legacy", 3, 16, True),
                ("none", "dyadic4", 4, 16, True), ("hilbert", "single", 1, 56, False))
        points = [(arm, rate) for arm in arms for rate in ("0.0032", "0.0512")]
        points.append((arms[0], "0"))
        for (order, schedule, stages, channels, context), rate in points:
            for mode in ("train", "eval"):
                argv = [mode, "--checkpoint", "rd.ckpt", "--rate-lambda", rate,
                        "--token-order", order,
                        "--score-context-schedule", schedule, "--score-spatial-stages", str(stages),
                        "--score-spatial-kernel", "5",
                        "--score-slice-channels", str(channels)]
                if not context:
                    argv.append("--no-spatial-context")
                if mode == "train":
                    argv += ["--allow-score-path-conversion", "--devices", "2", "--launcher", "srun",
                             "--batch-size", "2", "--accumulate", "2", "--max-steps", "50000"]
                cfg = self.composed(runner.parse_args(argv))
                codec = CodecConfig.from_mapping(cfg.model.feature_codec)
                self.assertEqual(cfg.loss.rate_lambda, float(rate))
                self.assertEqual(codec.token_order, order)
                self.assertEqual(codec.score_context_schedule, schedule)
                self.assertEqual(codec.score_spatial_stages, stages)
                self.assertEqual(codec.score_slice_channels, channels)
                self.assertEqual(codec.score_spatial_context, context)
                self.assertTrue(codec.use_residual)
                self.assertTrue(cfg.model.freeze_globalsplat)
                if mode == "train":
                    self.assertEqual(cfg.trainer.max_steps, 50000)
                    self.assertFalse(cfg.checkpointing.resume)
                    self.assertTrue(cfg.checkpointing.allow_score_path_conversion)
                    self.assertEqual(cfg.optimizer.batch_size*cfg.trainer.devices*cfg.trainer.accumulate_grad_batches, 8)
                    self.assertEqual(list(cfg.optimizer.lr_milestones), [35000, 45000])
                    self.assertEqual(cfg.loss.quantile_update_interval, 500)
                else:
                    self.assertTrue(cfg.test.actual_bitstream)
                    self.assertIsNone(cfg.test.max_scenes)
                    self.assertEqual(cfg.optimizer.batch_size, 1)
        for rate in ("0", "0.0032", "0.0512"):
            with self.assertRaises(SystemExit):
                runner.parse_args(["eval", "--rate-lambda", rate])
        for rate in ("-0.0032", "nan", "inf", "0.1"):
            with self.assertRaises(SystemExit):
                runner.parse_args(["train", "--checkpoint", "rd.ckpt", "--rate-lambda", rate])

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

    def test_dry_run_never_starts_processes_or_creates_output(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner.subprocess, "run") as run,
        ):
            output = Path(directory) / "not-created"
            runner.main(["eval", "--dry-run", "--output", str(output)])
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_score_path_conversion_is_weights_only_training(self):
        args = runner.parse_args([
            "train", "--checkpoint", "full.ckpt",
            "--allow-score-path-conversion",
        ])
        config = self.composed(args)
        codec = CodecConfig.from_mapping(config.model.feature_codec)
        self.assertTrue(codec.use_residual)
        self.assertTrue(codec.score_spatial_context)
        self.assertTrue(config.checkpointing.allow_score_path_conversion)
        self.assertTrue(config.model.freeze_globalsplat)
        args = runner.parse_args([
            "eval", "--checkpoint", "minimal.ckpt",
        ])
        self.assertFalse(self.composed(args).checkpointing.allow_score_path_conversion)
        for argv in (
            ["eval", "--allow-score-path-conversion"],
            ["train", "--checkpoint", "full.ckpt", "--resume", "--allow-score-path-conversion"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                runner.parse_args(argv)

    def test_training_requires_explicit_checkpoint(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["train"])

    def test_context_off_uses_one_or_four_full_channel_streams_in_train_and_eval(self):
        for mode in ("train","eval"):
            for schedule,stages in (("single",1),("dyadic4",4)):
                with self.subTest(mode=mode,schedule=schedule):
                    argv=[mode,"--checkpoint","parent.ckpt",
                        "--token-order","hilbert","--no-spatial-context","--score-context-schedule",schedule,
                        "--score-slice-channels","56","--score-spatial-kernel","5"]
                    if mode=="train":
                        argv.append("--allow-score-path-conversion")
                    cfg=self.composed(runner.parse_args(argv))
                    codec=CodecConfig.from_mapping(cfg.model.feature_codec)
                    self.assertEqual(codec.score_spatial_stages,stages)
                    self.assertEqual(codec.score_slice_channels,56)
                    self.assertFalse(codec.score_spatial_context)
                    self.assertTrue(codec.use_residual)
                    self.assertEqual(codec.token_order,"hilbert")
        for args in (["--score-context-schedule","single"], ["--score-spatial-stages","1"],
                     ["--score-slice-channels","0"],["--score-slice-channels","57"]):
            with self.assertRaises(SystemExit):
                runner.parse_args(["eval",*args])

    def test_sorting_and_spatial_context_can_be_disabled_independently(self):
        for order, enabled, schedule, stages in (("none", True, "legacy", 3), ("hilbert", False, "legacy", 3),
                                                  ("none", True, "quarter2", 2), ("none", True, "dyadic4", 4)):
            for mode in ("train", "eval"):
                with self.subTest(order=order, mode=mode, schedule=schedule):
                    argv = [mode, "--checkpoint", "parent.ckpt",
                            "--token-order", order, "--score-context-schedule", schedule,
                            "--score-spatial-stages", str(stages),
                            "--score-spatial-kernel", "5"]
                    if not enabled:
                        argv.append("--no-spatial-context")
                    if mode == "train":
                        argv.append("--allow-score-path-conversion")
                    composed = self.composed(runner.parse_args(argv))
                    codec = CodecConfig.from_mapping(composed.model.feature_codec)
                    self.assertEqual(codec.token_order, order)
                    self.assertEqual(codec.use_morton, order != "none")
                    self.assertEqual(codec.score_spatial_context, enabled)
                    self.assertEqual(codec.score_context_schedule, schedule)
                    self.assertEqual(codec.score_spatial_stages, stages)
                    self.assertTrue(codec.use_residual)

    def test_nn_implementation_compose_and_resume_without_score_conversion(self):
        for mode in ("train", "eval"):
            argv = [mode, "--checkpoint", "nn.ckpt", "--token-order", "nn_score",
                    "--nn-order-backend", "knn",
                    "--nn-order-candidates", "65", "--nn-order-distance-device", "input"]
            if mode == "train":
                argv.append("--resume")
            config = self.composed(runner.parse_args(argv))
            codec = CodecConfig.from_mapping(config.model.feature_codec)
            self.assertEqual(codec.nn_order_backend, "knn")
            self.assertEqual(codec.nn_order_candidates, 65)
            self.assertEqual(codec.nn_order_distance_device, "input")
            self.assertFalse(config.checkpointing.allow_score_path_conversion)
            if mode == "train":
                self.assertTrue(config.checkpointing.resume)
        for extra in (["--nn-order-candidates", "0"], ["--nn-order-backend", "knn"],
                      ["--nn-order-distance-device", "input"]):
            with self.assertRaises(SystemExit):
                runner.parse_args(["eval", *extra])

    def test_order_and_anchor_switches_compose_independently(self):
        for order in ("morton", "hilbert", "nn_xyz", "nn_score"):
            for schedule, stages in (("legacy", 2), ("quarter2", 2), ("dyadic4", 4)):
                args = runner.parse_args(["train", "--checkpoint", "full.ckpt",
                    "--allow-score-path-conversion",
                    "--token-order", order, "--score-context-schedule", schedule])
                composed = self.composed(args)
                codec = CodecConfig.from_mapping(composed.model.feature_codec)
                self.assertEqual(codec.token_order, order)
                self.assertEqual(codec.score_context_schedule, schedule)
                self.assertEqual(codec.score_spatial_stages, stages)
                self.assertTrue(codec.use_residual)
        with self.assertRaises(SystemExit):
            runner.parse_args(["eval", "--score-context-schedule", "dyadic4", "--score-spatial-stages", "3"])

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

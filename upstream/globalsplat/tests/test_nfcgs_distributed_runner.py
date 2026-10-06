"""SLURM launch checks that do not require a GPU or Lightning installation."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("distributed_runner", ROOT / "scripts/run_nfcgs.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DistributedRunnerTests(unittest.TestCase):
    def test_spatial_stage_cli_and_matched_control(self):
        for stages in (2, 3, 4):
            args = runner.parse_args([
                "train", "--checkpoint", "split.ckpt", "--score-path", "minimal",
                "--allow-score-path-conversion", "--score-spatial-stages", str(stages),
                "--score-spatial-kernel", "5", "--score-context-quantization", "ste",
                "--devices", "4", "--batch-size", "2", "--accumulate", "1",
            ])
            command, _, _ = runner.build_command(args)
            self.assertEqual(args.devices * args.batch_size * args.accumulate, 8)
            for setting in (f"model.feature_codec.score_spatial_stages={stages}",
                            "model.feature_codec.score_spatial_kernel=5",
                            "model.feature_codec.score_context_quantization=ste",
                            "model.feature_codec.score_channel_context=false"):
                self.assertIn(setting, command)
        for stages in (3, 4):
            args = runner.parse_args(["eval", "--score-path", "minimal", "--score-spatial-stages", str(stages)])
            self.assertEqual((args.score_spatial_kernel, args.score_context_quantization), (5, "ste"))
            command, _, _ = runner.build_command(args)
            self.assertIn("test.max_scenes=null", command)
        full = runner.parse_args(["eval"])
        self.assertEqual((full.score_spatial_stages, full.score_spatial_kernel, full.score_context_quantization), (2, 3, "noise"))

    def test_invalid_stage_combinations_fail_before_launch(self):
        for flags in (["--score-spatial-stages", "3"],
                      ["--score-path", "minimal", "--score-spatial-stages", "4", "--score-spatial-kernel", "3"],
                      ["--score-path", "minimal", "--score-spatial-stages", "3", "--dump-score-context"]):
            with self.assertRaises(SystemExit):
                runner.parse_args(["eval", *flags])

    def test_minimal_four_gpu_recipe_keeps_effective_batch_and_optimizer_schedule(self):
        argv = ["train", "--checkpoint", "split.ckpt", "--rate-lambda", "0.0256",
                "--score-path", "minimal", "--allow-score-path-conversion",
                "--devices", "4", "--launcher", "srun", "--batch-size", "2",
                "--accumulate", "1", "--workers", "8", "--lr", "0.0001",
                "--max-steps", "50000", "--checkpoint-every", "5000"]
        args = runner.parse_args(argv)
        command, _, _ = runner.build_command(args)
        self.assertEqual(args.devices * args.batch_size * args.accumulate, 8)
        self.assertEqual(command.count("srun"), 1)
        for override in ("--ntasks=4", "--ntasks-per-node=4", "trainer.devices=4",
                         "trainer.strategy=ddp_find_unused_parameters_true",
                         "optimizer.batch_size=2", "trainer.accumulate_grad_batches=1",
                         "optimizer.lr=0.0001", "optimizer.lr_milestones=[35000,45000]",
                         "trainer.max_steps=50000", "checkpointing.every_n_train_steps=5000",
                         "model.feature_codec.use_centering=false",
                         "model.feature_codec.transform=linear"):
            self.assertIn(override, command)
        script = (ROOT / "scripts/slurm/train_nfcgs_score_minimal_v11.slurm").read_text()
        for setting in ("--nodelist=ariel-v11", "--gres=gpu:normal:4",
                        "--ntasks-per-node=4", "--cpus-per-task=8",
                        "--devices 4 --launcher srun", "--batch-size 2 --accumulate 1"):
            self.assertIn(setting, script)
        self.assertNotIn("exec srun", script)

    def scratch_args(self):
        return ["train", "--from-scratch", "--devices", "8", "--launcher", "srun", "--accumulate", "1"]

    def test_eight_ranks_share_one_training_command_and_effective_batch_eight(self):
        args = runner.parse_args(self.scratch_args())
        command, checkpoint, _ = runner.build_command(args)
        self.assertIsNone(checkpoint)
        self.assertEqual(command[:6], ["srun", "--nodes=1", "--ntasks=8", "--ntasks-per-node=8", "--kill-on-bad-exit=1", "--gpu-bind=none"])
        for override in (
            "trainer.devices=8", "trainer.num_nodes=1",
            "trainer.strategy=ddp_find_unused_parameters_true",
            "checkpointing.load=null", "optimizer.batch_size=1",
            "trainer.accumulate_grad_batches=1",
        ):
            self.assertIn(override, command)
        self.assertEqual(args.devices * args.batch_size * args.accumulate, 8)
        self.assertEqual(sum(item.startswith("output_dir=") for item in command), 1)

    def test_matching_allocation_starts_exactly_one_srun(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "train").mkdir()
            (Path(directory) / "train/index.json").write_text("{}")
            with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_NTASKS": "8", "SLURM_JOB_NUM_NODES": "1"}), patch.object(runner.subprocess, "run") as run:
                runner.main(self.scratch_args() + ["--dataset-root", directory])
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0][0], "srun")

    def test_allocation_mismatch_stops_before_launch(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_NTASKS": "1", "SLURM_JOB_NUM_NODES": "1"}), patch.object(runner.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "one task"):
                runner.main(self.scratch_args())
            run.assert_not_called()

    def test_dry_run_needs_no_allocation(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(runner.subprocess, "run") as run:
            runner.main(self.scratch_args() + ["--dry-run"])
            run.assert_not_called()

    def test_single_gpu_default_is_preserved(self):
        command, _, _ = runner.build_command(runner.parse_args(["train", "--from-scratch"]))
        self.assertNotEqual(command[0], "srun")
        self.assertIn("trainer.devices=1", command)
        self.assertIn("trainer.strategy=auto", command)
        self.assertIn("trainer.accumulate_grad_batches=8", command)
        self.assertIn("trainer.max_steps=500000", command)
        self.assertIn("checkpointing.every_n_train_steps=10000", command)

    def test_short_run_saves_full_checkpoints_before_exit(self):
        args = runner.parse_args(self.scratch_args() + [
            "--max-steps", "20", "--checkpoint-every", "10",
        ])
        command, _, _ = runner.build_command(args)
        self.assertIn("trainer.max_steps=20", command)
        self.assertIn("checkpointing.every_n_train_steps=10", command)

    def test_checkpoint_interval_must_be_positive(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(self.scratch_args() + ["--checkpoint-every", "0"])

    def test_v12_allocation_matches_runner(self):
        script = (ROOT / "scripts/slurm/train_nfcgs_joint_v12.slurm").read_text()
        for setting in ("--nodelist=ariel-v12", "--gres=gpu:normal:8", "--ntasks-per-node=8", "--devices 8", "--launcher srun", "--accumulate 1"):
            self.assertIn(setting, script)

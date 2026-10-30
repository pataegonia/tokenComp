"""Compose real configurations and check the seed-repeat execution boundary."""
import copy
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_nfcgs_context_seed_pair as runner
import run_nfcgs as base


def composed(args):
    command, _, _ = runner.build_command(args)
    overrides = command[command.index("globalsplat.main")+1:]
    with initialize_config_dir(config_dir=str(ROOT/"config"), version_base=None):
        return compose(config_name="main", overrides=overrides)


class ContextSeedPairTests(unittest.TestCase):
    def test_four_real_hydra_configs_match_protocol_and_keep_history_unchanged(self):
        for arm in runner.ARMS:
            for mode in ("train", "eval"):
                with self.subTest(arm=arm, mode=mode):
                    argv = [mode, "--arm", arm, "--run-dir", "seed_run"]
                    if mode == "train":
                        argv += ["--checkpoint", "common_parent.ckpt", "--launcher", "srun"]
                    args = runner.parse_args(argv)
                    config = composed(args)
                    codec = config.model.feature_codec
                    self.assertEqual(codec.token_order, "hilbert")
                    self.assertEqual(codec.score_spatial_context, arm == "on16")
                    self.assertEqual(codec.score_context_schedule, "dyadic4")
                    self.assertEqual(codec.score_spatial_stages, 4)
                    self.assertEqual(codec.score_slice_channels, 16)
                    self.assertTrue(codec.use_residual)
                    self.assertEqual(codec.rank, 56)
                    self.assertEqual(config.loss.rate_lambda, .0128)
                    if mode == "train":
                        runner.validate_training_config(OmegaConf.to_container(config, resolve=True), arm)
                        self.assertEqual(config.seed, 111124)
                        self.assertEqual(config.dataset.data_loader_seed, 404)
                        self.assertEqual(config.trainer.max_steps, 50000)
                        self.assertEqual(config.optimizer.batch_size*config.trainer.devices*config.trainer.accumulate_grad_batches, 8)
                        self.assertEqual(list(config.optimizer.lr_milestones), [35000, 45000])
                        self.assertTrue(config.checkpointing.allow_score_path_conversion)
                        self.assertEqual(config.dataset.num_context_views, 24)
                        self.assertEqual(config.dataset.num_target_views, 12)
                    else:
                        self.assertEqual(config.seed, 0)
                        self.assertEqual(config.dataset.num_context_views, 12)
                        self.assertEqual(config.dataset.num_target_views, 8)
                        self.assertTrue(config.dataset.eval_all_scenes)
                        self.assertIsNone(config.dataset.eval_index_path)
                        self.assertTrue(config.test.actual_bitstream)
                        self.assertIsNone(config.test.max_scenes)
        old_args = base.parse_args(["train", "--checkpoint", "common_parent.ckpt"])
        command, _, _ = base.build_command(old_args)
        with initialize_config_dir(config_dir=str(ROOT/"config"), version_base=None):
            old = compose(config_name="main", overrides=command[command.index("globalsplat.main")+1:])
        self.assertEqual(old.seed, 111123)
        self.assertEqual(old.dataset.data_loader_seed, 403)

    def test_wrong_seed_arm_streams_or_holdout_rejected_before_eval(self):
        args = runner.parse_args(["train", "--arm", "on16", "--checkpoint", "parent.ckpt", "--run-dir", "run"])
        config = OmegaConf.to_container(composed(args), resolve=True)
        mutations = (("seed", 111123), ("dataset.data_loader_seed", 403),
                     ("model.feature_codec.score_spatial_context", False),
                     ("model.feature_codec.score_slice_channels", 56),
                     ("model.feature_codec.use_residual", False),
                     ("trainer.max_steps", 10000), ("checkpointing.resume", True),
                     ("dataset.train_holdout_manifest", "holdout.json"),
                     ("model.feature_codec.codec_mode", "score_only"))
        for name, value in mutations:
            with self.subTest(name=name):
                changed = copy.deepcopy(config)
                target = changed
                parts = name.split(".")
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                with self.assertRaises(ValueError):
                    runner.validate_training_config(changed, "on16")

    def test_execution_checks_allocation_final_step_and_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root/"parent.ckpt"; parent.write_bytes(b"parent")
            for split in ("train", "test"):
                path = root/f"data/{split}/index.json"
                path.parent.mkdir(parents=True); path.write_text("{}")
            run = root/"run"
            train = ["train", "--arm", "off16", "--checkpoint", str(parent), "--run-dir", str(run),
                     "--dataset-root", str(root/"data")]
            with patch.object(runner.subprocess, "run") as execute, patch("sys.stdout", new_callable=StringIO):
                with patch.dict(os.environ, {"SLURM_JOB_ID": "", "SLURM_NTASKS": "0", "SLURM_JOB_NUM_NODES": "0"}):
                    with self.assertRaisesRegex(RuntimeError, "SLURM allocation"):
                        runner.main([*train, "--launcher", "srun"])
                    execute.assert_not_called()
                with patch.dict(os.environ, {"SLURM_JOB_ID": "1", "SLURM_NTASKS": "2", "SLURM_JOB_NUM_NODES": "1"}):
                    runner.main([*train, "--launcher", "srun"])
                    self.assertEqual(execute.call_count, 2)
                    command = execute.call_args_list[1].args[0]
                    self.assertEqual(command[0], "srun")
                    self.assertIn("seed=111124", command)
                    self.assertIn("dataset.data_loader_seed=404", command)
                execute.reset_mock()
                (root/"data/train/index.json").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "compute-node RE10K train index"):
                    runner.main(train)
                execute.assert_not_called()

    def test_eval_uses_saved_training_config_and_requires_6991_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/"run"
            train_args = ["train", "--arm", "off16", "--checkpoint", str(root/"parent.ckpt"),
                          "--run-dir", str(run), "--dataset-root", str(root/"data")]
            config = composed(runner.parse_args(train_args))
            config_dir = run/"hydra/.hydra"; config_dir.mkdir(parents=True)
            OmegaConf.save(config, config_dir/"config.yaml")
            checkpoint = run/"checkpoints/nfcgs_main/version_0/step000050000.ckpt"
            checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"trained")
            index = root/"data/test/index.json"; index.parent.mkdir(parents=True); index.write_text("{}")
            summary = run/"eval/nfcgs_main/scores_all_avg.json"; summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"scene_count": 6991}))
            argv = ["eval", "--arm", "off16", "--run-dir", str(run), "--dataset-root", str(root/"data")]
            with patch.object(runner.subprocess, "run") as execute, patch("sys.stdout", new_callable=StringIO):
                runner.main(argv)
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(execute.call_args_list[0].args[0][-2:], ["--step", "50000"])
                summary.write_text(json.dumps({"scene_count": 6990}))
                with self.assertRaisesRegex(RuntimeError, "evaluation incomplete"):
                    runner.main(argv)
                execute.reset_mock()
                config.seed = 111123; OmegaConf.save(config, config_dir/"config.yaml")
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    runner.main(argv)
                execute.assert_not_called()

    def test_shell_files_have_lf_and_valid_syntax(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("bash unavailable")
        for name in ("submit_nfcgs_context_seed_pair.sh", "train_nfcgs_context_seed_pair.slurm", "eval_nfcgs_context_seed_pair.slurm"):
            path = ROOT/"scripts/slurm"/name
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"#!/usr/bin/env bash\n"))
            self.assertNotIn(b"\r", data)
            result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

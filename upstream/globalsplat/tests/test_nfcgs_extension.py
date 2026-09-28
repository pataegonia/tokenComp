import tempfile
import unittest
from pathlib import Path
import sys

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from prepare_nfcgs_extension_checkpoint import prepare_checkpoint
from resolve_nfcgs_slurm_checkpoint import resolve_checkpoint


class ExtensionCheckpointTest(unittest.TestCase):
    def test_preserves_training_state_and_replaces_only_lr_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "last.ckpt"
            destination = root / "extended.ckpt"
            source_parameter = torch.nn.Parameter(torch.zeros(()))
            source_optimizer = torch.optim.AdamW([source_parameter], lr=5e-4)
            original = {
                "global_step": 220_000,
                "state_dict": {"model.weight": torch.tensor([3.0])},
                "optimizer_states": [source_optimizer.state_dict()],
                "lr_schedulers": [{"old": "220k schedule"}],
                "loops": {"fit_loop": "preserved"},
            }
            torch.save(original, source)

            metadata = prepare_checkpoint(
                source,
                destination,
                expected_start_step=220_000,
                target_step=500_000,
                warmup_steps=10_000,
                start_lr=1e-5,
                peak_lr=1e-4,
                final_lr=1e-5,
            )
            patched = torch.load(destination, map_location="cpu", weights_only=False)

            self.assertEqual(metadata["target_step"], 500_000)
            self.assertEqual(patched["global_step"], original["global_step"])
            self.assertEqual(patched["loops"], original["loops"])
            self.assertTrue(
                torch.equal(
                    patched["state_dict"]["model.weight"],
                    original["state_dict"]["model.weight"],
                )
            )
            self.assertEqual(
                patched["optimizer_states"][0]["state"],
                original["optimizer_states"][0]["state"],
            )
            group = patched["optimizer_states"][0]["param_groups"][0]
            self.assertEqual(group["lr"], 1e-5)
            self.assertEqual(group["initial_lr"], 1e-4)
            schedule = patched["lr_schedulers"][0]
            self.assertEqual(schedule["_milestones"], [10_000])
            self.assertEqual(schedule["_schedulers"][1]["T_max"], 270_000)
            self.assertEqual(schedule["_schedulers"][1]["eta_min"], 1e-5)

            # Lightning constructs a schedule from the new 500k config before
            # loading the patched state. Verify that replacement state loads
            # into the same scheduler type and reaches the requested peak.
            parameter = torch.nn.Parameter(torch.zeros(()))
            optimizer = torch.optim.AdamW([parameter], lr=1e-4)
            default_warmup = LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=15_000
            )
            default_cosine = CosineAnnealingLR(
                optimizer, T_max=485_000, eta_min=2e-6
            )
            runtime_schedule = SequentialLR(
                optimizer, [default_warmup, default_cosine], milestones=[15_000]
            )
            optimizer.load_state_dict(patched["optimizer_states"][0])
            runtime_schedule.load_state_dict(schedule)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
            for _ in range(10_000):
                optimizer.step()
                runtime_schedule.step()
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_rejects_the_wrong_parent_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "last.ckpt"
            torch.save({"global_step": 10}, source)
            with self.assertRaisesRegex(ValueError, "expected 220000"):
                prepare_checkpoint(
                    source,
                    root / "extended.ckpt",
                    expected_start_step=220_000,
                    target_step=500_000,
                    warmup_steps=10_000,
                    start_lr=1e-5,
                    peak_lr=1e-4,
                    final_lr=1e-5,
                )


class ResolveCheckpointTest(unittest.TestCase):
    def test_resolves_last_checkpoint_from_printed_runner_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "run with space" / "checkpoints" / "exp" / "version_0" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            log = root / "slurm.out"
            log.write_text(
                "python -m globalsplat.main "
                f"'output_dir={checkpoint.parents[2]}' trainer.max_steps=220000\n",
                encoding="utf-8",
            )
            self.assertEqual(resolve_checkpoint(log), checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()

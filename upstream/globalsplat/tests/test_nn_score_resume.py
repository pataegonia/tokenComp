"""A full-state resume copies weights/optimizer data without touching source."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from globalsplat.compression import ObservableLowRank1DCodec
from globalsplat.checkpointing import resolve_run
from test_score_path_ablation import config, MINIMAL

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_nn_score", ROOT / "scripts/prepare_nfcgs_nn_score_resume.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class NNScoreResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source_config = config(**MINIMAL, token_order="nn_score", score_spatial_stages=3,
            score_spatial_kernel=5, score_context_quantization="ste")
        cls.expected = config(**MINIMAL, token_order="nn_score", score_spatial_stages=3,
            score_spatial_kernel=5, score_context_quantization="ste", nn_order_backend="knn")
        codec = ObservableLowRank1DCodec(cls.source_config)
        cls.checkpoint = dict(state_dict={"model.feature_codec." + k: v for k, v in codec.state_dict().items()},
            feature_codec_config=cls.source_config.to_dict(), score_mean_offset_enabled=True,
            global_step=5000,
            optimizer_states=[{"state": {0: {"step": torch.tensor(5000), "exp_avg": torch.tensor([.3])}}}],
            lr_schedulers=[{"last_epoch": 5000}], loops={"fit_loop": {"epoch_progress": 1}})
        cls.checkpoint["pytorch-lightning_version"] = "2.4.0"

    def test_copy_keeps_identical_bytes_and_resumes_inside_new_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old/version_0/step000005000.ckpt"
            source.parent.mkdir(parents=True)
            torch.save(self.checkpoint, source)
            destination = root / "new/checkpoints/nfcgs_main/version_0/resume.ckpt"
            before = source.read_bytes()
            step = helper.prepare(source, 50000, destination, expected_config=self.expected)
            self.assertEqual(step, 5000)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(destination.read_bytes(), before)
            loaded = torch.load(destination, weights_only=False)
            self.assertEqual(loaded["optimizer_states"][0]["state"][0]["step"].item(), 5000)
            self.assertEqual(loaded["lr_schedulers"][0]["last_epoch"], 5000)
            result = resolve_run(load=str(destination), resume=True, auto_resume=False,
                                 exp_ckpt_base=str(root / "ignored"))
            self.assertEqual(Path(result.ckpt_dir), destination.parent)
            self.assertIsNone(result.weights_only)
            with self.assertRaisesRegex(ValueError, "new file"):
                helper.prepare(source, 50000, destination, expected_config=self.expected)

    def test_bad_source_fails_before_creating_destination(self):
        cases = []
        for key in ("optimizer_states", "lr_schedulers", "pytorch-lightning_version"):
            changed = copy.deepcopy(self.checkpoint)
            changed.pop(key)
            cases.append(changed)
        cases.append(dict(self.checkpoint, global_step=50000))
        cases.append(dict(self.checkpoint, global_step=6000))
        cases.append(dict(self.checkpoint, score_mean_offset_enabled=False))
        changed = copy.deepcopy(self.checkpoint)
        changed["feature_codec_config"]["token_order"] = "hilbert"
        cases.append(changed)
        for checkpoint in cases:
            with self.subTest(), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "step000005000.ckpt"
                torch.save(checkpoint, source)
                destination = Path(directory) / "never-created/version_0/resume.ckpt"
                with self.assertRaises(ValueError):
                    helper.prepare(source, 50000, destination, expected_config=self.expected)
                self.assertFalse(destination.parent.exists())


if __name__ == "__main__":
    unittest.main()

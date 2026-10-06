"""Slim checkpoints omit fixed loss networks, never learned model weights."""
import unittest

import torch
from torch import nn

from globalsplat.compression.checkpoint import validate_weights_only_load


class SlimWeightsOnlyLoadTests(unittest.TestCase):
    def setUp(self):
        self.module = nn.Module()
        self.module.model = nn.Module()
        self.module.model.backbone = nn.Linear(4, 4)
        self.module.model.feature_codec = nn.Linear(4, 2)
        # 38 tensors reproduce the missing count in the downloaded job log.
        self.module.render_criterion = nn.ModuleList(nn.Linear(1, 1) for _ in range(19))
        self.module.render_criterion.requires_grad_(False)
        self.saved = {key: value.clone() + 1 for key, value in self.module.state_dict().items()
                      if not key.startswith("render_criterion.")}

    def test_slim_checkpoint_loads_all_model_weights_and_keeps_rebuilt_loss(self):
        fixed = {key: value.clone() for key, value in self.module.render_criterion.state_dict().items()}
        incompatible = self.module.load_state_dict(self.saved, strict=False)
        self.assertEqual(len(incompatible.missing_keys), 38)
        validate_weights_only_load(*incompatible)
        for key, value in self.saved.items():
            torch.testing.assert_close(self.module.state_dict()[key], value)
        for key, value in fixed.items():
            torch.testing.assert_close(self.module.render_criterion.state_dict()[key], value)

    def test_missing_backbone_or_codec_is_rejected_even_with_fixed_loss_missing(self):
        for key in ("model.backbone.weight", "model.feature_codec.weight"):
            saved = {name: value for name, value in self.saved.items() if name != key}
            incompatible = self.module.load_state_dict(saved, strict=False)
            with self.assertRaisesRegex(RuntimeError, key.replace(".", r"\.")):
                validate_weights_only_load(*incompatible)

    def test_older_loss_tensors_are_allowed_but_unexpected_model_is_rejected(self):
        saved = {**self.saved, "render_criterion.old_constant": torch.zeros(1)}
        validate_weights_only_load(*self.module.load_state_dict(saved, strict=False))
        saved["model.feature_codec.unknown_weight"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "unknown_weight"):
            validate_weights_only_load(*self.module.load_state_dict(saved, strict=False))


if __name__ == "__main__":
    unittest.main()

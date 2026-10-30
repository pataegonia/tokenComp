"""Plan §13.4 decoder-readout finetune: param groups, freezing and commands."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_nfcgs_decoder_finetune as finetune
import run_nfcgs_fresh_rd as runner
from globalsplat.model.globalsplat import GlobalSplat
from globalsplat.model.optim import build_optimizer_and_scheduler


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.feature_codec = nn.Linear(3, 3)
        self.model.gaussian_decoder = nn.Linear(3, 3)


def test_lr_override_adds_groups_after_unchanged_defaults():
    module = Wrapper()
    plain = build_optimizer_and_scheduler(module, lr=1e-6, weight_decay=0., optimizer_name="adam",
                                          lr_milestones=[10**9])["optimizer"]
    assert [len(g["params"]) for g in plain.param_groups] == [2, 2]
    result = build_optimizer_and_scheduler(module, lr=1e-6, weight_decay=0., optimizer_name="adam",
                                           lr_milestones=[10**9], lr_overrides={"model.gaussian_decoder.": 1e-5})
    groups = result["optimizer"].param_groups
    assert [g["lr"] for g in groups] == [1e-6, 1e-6, 1e-5, 1e-5]
    decoder = {id(p) for p in module.model.gaussian_decoder.parameters()}
    assert {id(p) for g in groups[2:] for p in g["params"]} == decoder
    assert not decoder & {id(p) for g in groups[:2] for p in g["params"]}
    for _ in range(3):
        result["optimizer"].step()
        result["lr_scheduler"]["scheduler"].step()
    assert [g["lr"] for g in groups] == [1e-6, 1e-6, 1e-5, 1e-5]  # constant schedule


def small_model(**kwargs):
    codec = dict(geometry_observable_channels=4, rank=4, residual_n=8, residual_m=12, adapter_hidden=6,
                 score_slice_channels=2, token_order="hilbert",
                 score_context_schedule="dyadic4", score_spatial_stages=4,
                 score_spatial_kernel=5)
    return GlobalSplat(static_only=True, dim_latents=16, latent_rep_token_amount=8, dim_rays=8, dim_rgb_feat=8,
                       rounds=1, slot_calib_layers_per_round=1, num_heads=2, M_max=2, feature_codec=codec, **kwargs)


def test_only_codec_and_the_four_readouts_train():
    model = small_model(freeze_globalsplat=True, train_decoder_readouts=True)
    trainable = {name.split(".")[0] if name.startswith("feature_codec") else ".".join(name.split(".")[:2])
                 for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"feature_codec", *(f"gaussian_decoder.{n}" for n in GlobalSplat.DECODER_READOUTS)}
    frozen = small_model(freeze_globalsplat=True)
    assert not any(p.requires_grad for n, p in frozen.named_parameters() if n.startswith("gaussian_decoder."))
    with pytest.raises(ValueError):
        small_model(train_decoder_readouts=True)


@pytest.mark.parametrize("variant", ["finetune", "control"])
def test_commands_keep_the_fresh_recipe_and_differ_only_in_decoder(variant, tmp_path):
    args = finetune.parse_args(["--variant", variant, "--run-dir", str(tmp_path / variant),
                                "--source", str(tmp_path / "lambda0_120k.ckpt")])
    command = finetune.train_command(args)
    cfg = runner.compose_command(command)
    finetune.validate_train_config(cfg, args)
    assert Path(cfg.checkpointing.load) == tmp_path / "lambda0_120k.ckpt"
    assert list(cfg.optimizer.lr_milestones) == [10**9]
    assert cfg.optimizer.get("decoder_lr") == (1e-5 if variant == "finetune" else None)
    assert not cfg.checkpointing.get("allow_score_path_conversion", False)
    test = runner.compose_command(finetune.test_command(args, tmp_path / variant / "final.ckpt"))
    assert test.test.actual_bitstream and test.architecture_entropy_determinism and test.loss.rate_lambda == 0
    assert Path(test.checkpointing.load) == tmp_path / variant / "final.ckpt"
    assert not test.model.get("train_decoder_readouts", False)

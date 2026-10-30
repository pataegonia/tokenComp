"""Plan §14 Top-K: selector, rate surrogate, dense/compact agreement and runner cells."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_nfcgs_fresh_rd as runner
from globalsplat.compression.checkpoint import validate_weights_only_load
from globalsplat.compression.token_selection import TokenSelector, rate_with_count_surrogate
from globalsplat.fresh_rd import CELLS
from globalsplat.model.globalsplat import GlobalSplat

torch.set_num_threads(1)


def test_counts_gates_and_task_gradients():
    torch.manual_seed(0)
    selector = TokenSelector(16, 64, minimum=8).train()
    tokens = torch.randn(3, 64, 16)
    result = selector(tokens)
    expected = 8 + 56 * torch.sigmoid(torch.tensor(2.94))
    torch.testing.assert_close(result.expected_count, expected.expand(3))
    assert result.selected_count.tolist() == [round(expected.item())] * 3
    torch.testing.assert_close(result.gates.sum(1), result.selected_count.float())
    assert set(result.gates.detach().unique().tolist()) <= {0., 1.}
    for scene, kept in enumerate(result.indices):
        assert torch.equal(kept, kept.sort().values)
        assert set(kept.tolist()) == set(result.gates[scene].nonzero().flatten().tolist())
    # Task loss alone must reach both the count router and the scorer.
    loss = (result.gates * torch.linspace(-1, 1, 64)).sum()
    router, scorer = torch.autograd.grad(loss, (selector.router[-1].weight, selector.scorer[-1].weight))
    assert router.abs().sum() > 0 and scorer.abs().sum() > 0
    selector.eval()
    assert torch.equal(selector(tokens).gates, result.gates.detach())


def test_forced_count_and_importance_order():
    torch.manual_seed(1)
    selector = TokenSelector(16, 64, minimum=8, forced_count=20, order="importance").train()
    assert not any(p.requires_grad for p in selector.router.parameters())
    assert all(p.requires_grad for p in selector.scorer.parameters())
    result = selector(torch.randn(2, 64, 16))
    assert result.selected_count.tolist() == [20, 20]
    for scene, kept in enumerate(result.indices):
        scores = result.scores[scene, kept]
        assert torch.equal(scores, scores.sort(descending=True).values)
    with pytest.raises(ValueError):
        TokenSelector(16, 64, minimum=8, forced_count=4)


def test_rate_surrogate_keeps_value_and_adds_count_gradient():
    kappa = torch.tensor([30., 50.], requires_grad=True)
    bits = torch.tensor([300., 1000.])
    rate = rate_with_count_surrogate(bits, kappa, torch.tensor([30, 40]))
    torch.testing.assert_close(rate, bits)
    rate.sum().backward()
    torch.testing.assert_close(kappa.grad, torch.tensor([10., 25.]))


def test_new_selector_may_be_missing_from_a_weights_only_initializer():
    validate_weights_only_load(["model.token_selector.router.1.weight", "render_criterion.vgg.0.weight"], [])
    with pytest.raises(RuntimeError):
        validate_weights_only_load(["model.feature_codec.shared_basis"], [])


def small_model(codec_order="hilbert", **selection):
    codec = dict(geometry_observable_channels=4, rank=4, residual_n=8, residual_m=12, adapter_hidden=6,
                 score_slice_channels=2, token_order=codec_order,
                 score_context_schedule="dyadic4", score_spatial_stages=4,
                 score_spatial_kernel=5)
    return GlobalSplat(static_only=True, dim_latents=16, latent_rep_token_amount=32, dim_rays=8, dim_rgb_feat=8,
                       rounds=1, slot_calib_layers_per_round=1, num_heads=2, M_max=2, feature_codec=codec,
                       freeze_globalsplat=True, token_selection=dict(minimum=8, **selection))


@pytest.mark.parametrize("order,selection", [("hilbert", {}), ("none", dict(order="importance")),
                                             ("hilbert", dict(forced_count=12))])
def test_dense_gated_forward_matches_compact_bitstream(order, selection):
    torch.manual_seed(2)
    model = small_model(order, **selection).eval()
    model.feature_codec.update(force=True)
    patch = (torch.randn(1, 32, 16), torch.randn(1, 32, 16))
    with torch.no_grad():
        dense = model._forward_selected(patch)
        kept = model.token_selector(torch.cat(patch, -1)).selected_count.item()
        compact = model.decode_scene_tokens(model.decompress_scene_tokens(model.compress_scene_tokens(patch).data))
    opacity = dense.opacities.flatten()
    children = opacity.numel() // 32
    assert (opacity == 0).sum().item() == (32 - kept) * children
    assert compact.opacities.numel() == kept * children
    torch.testing.assert_close(opacity[opacity > 0].sort().values, compact.opacities.flatten().sort().values,
                               atol=1e-5, rtol=1e-4)
    terms = model.last_codec_output
    assert terms.diagnostics["selected_tokens"].item() == kept and terms.estimated_bits > 0


def test_selector_trains_with_a_frozen_backbone_and_importance_needs_no_codec_order():
    model = small_model()
    trainable = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"feature_codec", "token_selector"}
    with pytest.raises(ValueError):
        small_model("hilbert", order="importance")


@pytest.mark.parametrize("cell", ["tk_h", "tk_imp", "tk_ns", "tk_ns1"])
def test_runner_builds_the_selector_for_train_and_test(cell, tmp_path):
    for mode in ("train", "test"):
        argv = [mode, "--cell", cell, "--run-dir", str(tmp_path), "--stage", "full", "--budget", "120000"]
        if mode == "test":
            argv += ["--checkpoint", str(tmp_path / "model.ckpt")]
        cfg = runner.compose_command(runner.build_command(runner.parse_args(argv))[0])
        selection = cfg.model.token_selection
        assert selection.order == ("importance" if cell == "tk_imp" else "slot") and selection.minimum == 256
        assert cfg.model.feature_codec.token_order == CELLS[cell].order

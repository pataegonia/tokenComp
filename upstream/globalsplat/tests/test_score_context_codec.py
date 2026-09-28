"""Full P0 score context gradient coverage."""

import torch
from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec


def small_config(**overrides):
    return CodecConfig(
        texture_channels=8,
        geometry_channels=8,
        geometry_observable_channels=4,
        rank=7,
        residual_n=6,
        residual_m=8,
        adapter_hidden=6,
        score_slice_channels=3,
        score_context_hidden=5,
        **overrides,
    )


def test_context_predictors_start_as_noop_and_receive_gradients():
    torch.manual_seed(73)
    codec = ObservableLowRank1DCodec(
        small_config(
            score_mean_condition=True,
            score_channel_context=True,
            score_spatial_context=True,
        )
    )
    appearance = torch.randn(2, 32, 8)
    geometry = torch.randn(2, 32, 8)
    positions = torch.randn(2, 32, 3)
    output = codec(appearance, geometry, positions)
    loss = output.reconstruction_sorted.square().mean() + 1e-4 * output.estimated_bits
    loss.backward()
    assert codec.score_context is not None
    for module in (
        codec.score_context.mean_conditioner[-1],
        codec.score_context.channel_predictors[0],
        codec.score_context.spatial_predictors[0],
    ):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0

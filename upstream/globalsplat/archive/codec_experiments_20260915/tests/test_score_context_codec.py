"""Contextual score prior: causality, bitstream, gradients, and train scope."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressai")

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec
from globalsplat.compression.bitstream import SceneBitstream, ScoreContextBitstream
from globalsplat.compression.checkpoint import (
    load_feature_codec_checkpoint,
    validate_feature_codec_checkpoint,
)


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


@pytest.mark.parametrize(
    "mean,channel,spatial",
    [(True, False, False), (True, True, False), (True, False, True), (True, True, True)],
)
def test_contextual_score_real_roundtrip_for_odd_token_count(mean, channel, spatial):
    torch.manual_seed(71)
    cfg = small_config(
        score_mean_condition=mean,
        score_channel_context=channel,
        score_spatial_context=spatial,
    )
    codec = ObservableLowRank1DCodec(cfg).eval()
    appearance = torch.randn(1, 33, 8)
    geometry = torch.randn(1, 33, 8)
    positions = torch.randn(1, 33, 3)
    codec.update(force=True)
    with torch.no_grad():
        expected = codec(
            appearance, geometry, positions, restore_original_order=False, training=False
        ).reconstruction_sorted
    compressed = codec.compress(appearance, geometry, positions)
    scene = SceneBitstream.unpack(compressed.data)
    assert scene.flags & codec.FLAG_CONTEXTUAL_SCORE
    score = ScoreContextBitstream.unpack(scene.score)
    expected_groups = 3 if channel else 1
    assert len(score.strings) == expected_groups * (2 if spatial else 1)
    actual = torch.cat(codec.decompress(compressed.data), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    mismatched = ObservableLowRank1DCodec(
        replace(cfg, score_spatial_context=not spatial)
    ).eval()
    with pytest.raises(ValueError, match="score"):
        mismatched.decompress(compressed.data)


@pytest.mark.parametrize("spatial_entropy", ["split", "gaussian", "conditional_scale"])
@pytest.mark.parametrize("points", [32, 33])
def test_spatial_entropy_variants_roundtrip_without_new_decode_passes(
    spatial_entropy, points
):
    torch.manual_seed(72)
    cfg = small_config(
        score_mean_condition=True,
        score_channel_context=True,
        score_spatial_context=True,
        score_spatial_entropy=spatial_entropy,
        score_spatial_hidden=4,
    )
    codec = ObservableLowRank1DCodec(cfg).eval()
    assert codec.score_context is not None
    if spatial_entropy == "conditional_scale":
        with torch.no_grad():
            for predictor in codec.score_context.spatial_scale_predictors:
                predictor[-1].weight.normal_(0, 0.03)
                predictor[-1].bias.normal_(0, 0.03)
    appearance = torch.randn(1, points, 8)
    geometry = torch.randn(1, points, 8)
    positions = torch.randn(1, points, 3)
    codec.update(force=True)
    with torch.no_grad():
        expected = codec(
            appearance, geometry, positions, restore_original_order=False, training=False
        ).reconstruction_sorted
    compressed = codec.compress(appearance, geometry, positions)
    score = ScoreContextBitstream.unpack(SceneBitstream.unpack(compressed.data).score)
    assert len(score.strings) == 6  # three channel groups, even + odd
    actual = torch.cat(codec.decompress(compressed.data), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    mismatched = ObservableLowRank1DCodec(
        replace(cfg, score_spatial_entropy="shared")
    ).eval()
    with pytest.raises(ValueError, match="score"):
        mismatched.decompress(compressed.data)


def test_conditional_scale_starts_from_static_gaussian_and_is_decoder_causal():
    cfg = small_config(
        score_spatial_context=True,
        score_spatial_entropy="gaussian",
        score_spatial_hidden=4,
    )
    torch.manual_seed(74)
    static = ObservableLowRank1DCodec(cfg).eval()
    torch.manual_seed(74)
    conditional = ObservableLowRank1DCodec(
        replace(cfg, score_spatial_entropy="conditional_scale")
    ).eval()
    base = torch.randn(1, 7, 1, 33)
    anchor_hat = base[..., 0::2] + torch.randn_like(base[..., 0::2])
    static_scale = static.score_context._spatial_probability_scale(0, anchor_hat, base)
    conditional_scale = conditional.score_context._spatial_probability_scale(
        0, anchor_hat, base
    )
    torch.testing.assert_close(conditional_scale, static_scale, rtol=0, atol=0)

    changed_odd = base.clone()
    changed_odd[..., 1::2] += 1000 * torch.randn_like(changed_odd[..., 1::2])
    changed_scale = conditional.score_context._spatial_probability_scale(
        0, anchor_hat, changed_odd
    )
    torch.testing.assert_close(changed_scale, conditional_scale, rtol=0, atol=0)


@pytest.mark.parametrize("spatial_entropy", ["split", "gaussian", "conditional_scale"])
def test_probability_variant_warm_start_preserves_shared_parent_reconstruction(
    spatial_entropy,
):
    torch.manual_seed(76)
    parent_cfg = small_config(
        score_mean_condition=True,
        score_channel_context=True,
        score_spatial_context=True,
        score_spatial_hidden=4,
    )
    parent = ObservableLowRank1DCodec(parent_cfg).eval()
    candidate = ObservableLowRank1DCodec(
        replace(parent_cfg, score_spatial_entropy=spatial_entropy)
    ).eval()
    target_state = candidate.state_dict()
    for key, value in parent.state_dict().items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value
    candidate.load_state_dict(target_state, strict=True)
    if spatial_entropy == "split":
        assert parent.score_context is not None
        assert candidate.score_context is not None
        for even, odd in zip(
            parent.score_context.group_entropies,
            candidate.score_context.spatial_odd_entropies,
            strict=True,
        ):
            odd.load_state_dict(even.state_dict(), strict=True)

    appearance = torch.randn(1, 33, 8)
    geometry = torch.randn(1, 33, 8)
    positions = torch.randn(1, 33, 3)
    with torch.no_grad():
        expected = parent(
            appearance, geometry, positions, restore_original_order=False, training=False
        ).reconstruction_sorted
        actual = candidate(
            appearance, geometry, positions, restore_original_order=False, training=False
        ).reconstruction_sorted
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "spatial_entropy", ["shared", "split", "gaussian", "conditional_scale"]
)
def test_score_probability_scope_only_trains_density_without_quantile_motion(
    spatial_entropy,
):
    codec = ObservableLowRank1DCodec(
        small_config(
            score_mean_condition=True,
            score_channel_context=True,
            score_spatial_context=True,
            score_spatial_entropy=spatial_entropy,
            score_spatial_hidden=4,
        )
    )
    codec.set_trainable_scope("score_probability")
    trainable = {name for name, value in codec.named_parameters() if value.requires_grad}
    assert trainable
    assert not any("quantiles" in name for name in trainable)
    assert not any("spatial_predictors" in name for name in trainable)
    assert not any("channel_predictors" in name for name in trainable)
    assert not any("mean_conditioner" in name for name in trainable)
    if spatial_entropy in ("gaussian", "conditional_scale"):
        assert any("spatial_log_scales" in name for name in trainable)
    if spatial_entropy == "conditional_scale":
        assert any("spatial_scale_predictors" in name for name in trainable)


@pytest.mark.parametrize(
    "spatial_entropy", ["shared", "split", "gaussian", "conditional_scale"]
)
def test_score_probability_step_preserves_reconstruction(spatial_entropy):
    torch.manual_seed(75)
    codec = ObservableLowRank1DCodec(
        small_config(
            score_mean_condition=True,
            score_channel_context=True,
            score_spatial_context=True,
            score_spatial_entropy=spatial_entropy,
            score_spatial_hidden=4,
        )
    )
    codec.set_trainable_scope("score_probability")
    trainable = [value for value in codec.parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    appearance = torch.randn(1, 33, 8)
    geometry = torch.randn(1, 33, 8)
    positions = torch.randn(1, 33, 3)

    codec.eval()
    with torch.no_grad():
        before = codec(
            appearance, geometry, positions, restore_original_order=False, training=False
        ).reconstruction_sorted.clone()
    codec.train()
    output = codec(
        appearance, geometry, positions, restore_original_order=False, training=True
    )
    output.estimated_bits.backward()
    optimizer.step()
    codec.eval()
    with torch.no_grad():
        after = codec(
            appearance, geometry, positions, restore_original_order=False, training=False
        ).reconstruction_sorted
    torch.testing.assert_close(after, before, rtol=0, atol=0)


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


def test_score_context_only_scope_freezes_the_existing_codec():
    codec = ObservableLowRank1DCodec(
        small_config(score_mean_condition=True, score_channel_context=True)
    )
    codec.set_trainable_scope("score_context")
    trainable = {name for name, value in codec.named_parameters() if value.requires_grad}
    assert trainable
    assert all(name.startswith("score_context.") for name in trainable)
    assert not codec.shared_basis.requires_grad
    assert not any(parameter.requires_grad for parameter in codec.residual_codec.parameters())


def test_spatial_context_requires_morton_order():
    with pytest.raises(ValueError, match="requires Morton"):
        small_config(score_spatial_context=True, use_morton=False)


@pytest.mark.parametrize("variant", ["residual3", "residual7"])
def test_residual_spatial_predictor_starts_exactly_from_linear_parent(variant):
    cfg = small_config(
        score_mean_condition=True,
        score_channel_context=True,
        score_spatial_context=True,
    )
    torch.manual_seed(79)
    parent = ObservableLowRank1DCodec(cfg).eval()
    torch.manual_seed(79)
    candidate = ObservableLowRank1DCodec(
        replace(cfg, score_spatial_predictor=variant, score_spatial_hidden=4)
    ).eval()

    candidate_state = candidate.state_dict()
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(candidate_state[key], value, rtol=0, atol=0)
    assert candidate.score_context is not None
    for correction in candidate.score_context.spatial_corrections:
        assert torch.count_nonzero(correction[-1].weight) == 0
        assert torch.count_nonzero(correction[-1].bias) == 0

    appearance = torch.randn(1, 33, 8)
    geometry = torch.randn(1, 33, 8)
    positions = torch.randn(1, 33, 3)
    with torch.no_grad():
        expected = parent(
            appearance, geometry, positions, restore_original_order=False, training=False
        ).reconstruction_sorted
        actual = candidate(
            appearance, geometry, positions, restore_original_order=False, training=False
        ).reconstruction_sorted
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["residual3", "residual7"])
@pytest.mark.parametrize("points", [32, 33])
@pytest.mark.parametrize("channel", [False, True])
def test_residual_spatial_predictor_real_bitstream_and_variant_guard(variant, points, channel):
    torch.manual_seed(83)
    cfg = small_config(
        transform="nonlinear",
        score_mean_condition=True,
        score_channel_context=channel,
        score_spatial_context=True,
        score_spatial_predictor=variant,
        score_spatial_hidden=4,
    )
    codec = ObservableLowRank1DCodec(cfg).eval()
    assert codec.score_context is not None
    with torch.no_grad():
        for predictor in codec.score_context.channel_predictors:
            predictor.weight.normal_(0, 0.03)
            predictor.bias.normal_(0, 0.03)
        for correction in codec.score_context.spatial_corrections:
            correction[-1].weight.normal_(0, 0.03)
            correction[-1].bias.normal_(0, 0.03)
    appearance = torch.randn(1, points, 8)
    geometry = torch.randn(1, points, 8)
    positions = torch.randn(1, points, 3)
    codec.update(force=True)
    with torch.no_grad():
        expected = codec(
            appearance, geometry, positions, restore_original_order=False, training=False
        ).reconstruction_sorted
    compressed = codec.compress(appearance, geometry, positions)
    actual = torch.cat(codec.decompress(compressed.data), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    score = ScoreContextBitstream.unpack(SceneBitstream.unpack(compressed.data).score)
    assert len(score.strings) == (6 if channel else 2)
    expected_flag = (
        codec.score_context.FLAG_SPATIAL_RESIDUAL3
        if variant == "residual3"
        else codec.score_context.FLAG_SPATIAL_RESIDUAL7
    )
    assert score.flags & expected_flag
    other = "residual7" if variant == "residual3" else "residual3"
    mismatched = ObservableLowRank1DCodec(
        replace(cfg, score_spatial_predictor=other)
    ).eval()
    with pytest.raises(ValueError, match="score"):
        mismatched.decompress(compressed.data)


@pytest.mark.parametrize("variant", ["residual3", "residual7"])
def test_residual_spatial_predictor_is_decoder_causal(variant):
    codec = ObservableLowRank1DCodec(
        small_config(
            score_spatial_context=True,
            score_spatial_predictor=variant,
            score_spatial_hidden=4,
        )
    )
    assert codec.score_context is not None
    base = torch.randn(1, 7, 1, 33)
    anchor_hat = base[..., 0::2] + torch.randn_like(base[..., 0::2])
    changed_odd = base.clone()
    changed_odd[..., 1::2] += 1000 * torch.randn_like(changed_odd[..., 1::2])
    prediction = codec.score_context._spatial_prediction(0, anchor_hat, base)
    changed = codec.score_context._spatial_prediction(0, anchor_hat, changed_odd)
    torch.testing.assert_close(changed, prediction, rtol=0, atol=0)


def test_residual_spatial_correction_receives_gradients_after_zero_init():
    torch.manual_seed(89)
    codec = ObservableLowRank1DCodec(
        small_config(
            score_mean_condition=True,
            score_spatial_context=True,
            score_spatial_predictor="residual3",
            score_spatial_hidden=4,
        )
    )
    assert codec.score_context is not None
    correction = codec.score_context.spatial_corrections[0]
    optimizer = torch.optim.SGD(correction.parameters(), lr=0.1)
    appearance = torch.randn(2, 32, 8)
    geometry = torch.randn(2, 32, 8)
    positions = torch.randn(2, 32, 3)

    output = codec(appearance, geometry, positions)
    (output.reconstruction_sorted.square().mean() + 1e-4 * output.estimated_bits).backward()
    assert correction[-1].weight.grad is not None
    assert torch.isfinite(correction[-1].weight.grad).all()
    assert correction[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    output = codec(appearance, geometry, positions)
    (output.reconstruction_sorted.square().mean() + 1e-4 * output.estimated_bits).backward()
    assert correction[0].weight.grad is not None
    assert torch.isfinite(correction[0].weight.grad).all()
    assert correction[0].weight.grad.abs().sum() > 0


def test_spatial_predictor_checkpoint_metadata_and_old_defaults(tmp_path):
    cfg = small_config(
        score_mean_condition=True,
        score_spatial_context=True,
        score_spatial_predictor="residual7",
        score_spatial_hidden=4,
    )
    codec = ObservableLowRank1DCodec(cfg).eval()
    saved = {
        "state_dict": {
            f"model.feature_codec.{key}": value for key, value in codec.state_dict().items()
        },
        "feature_codec_config": cfg.to_dict(),
    }
    path = tmp_path / "residual7.ckpt"
    torch.save(saved, path)
    loaded = load_feature_codec_checkpoint(path)
    assert loaded.config == cfg
    validate_feature_codec_checkpoint(saved, cfg)
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_feature_codec_checkpoint(saved, replace(cfg, score_spatial_hidden=5))

    corrupt = {**saved, "state_dict": dict(saved["state_dict"])}
    del corrupt["state_dict"][
        "model.feature_codec.score_context.spatial_corrections.0.2.weight"
    ]
    with pytest.raises(ValueError, match="spatial predictor state mismatch"):
        validate_feature_codec_checkpoint(corrupt, cfg)

    parent_cfg = replace(
        cfg,
        score_spatial_predictor="linear",
        score_spatial_hidden=32,
    )
    parent = ObservableLowRank1DCodec(parent_cfg).eval()
    legacy_metadata = parent_cfg.to_dict()
    legacy_metadata.pop("score_spatial_predictor")
    legacy_metadata.pop("score_spatial_hidden")
    legacy = {
        "state_dict": {
            f"model.feature_codec.{key}": value for key, value in parent.state_dict().items()
        },
        "feature_codec_config": legacy_metadata,
    }
    legacy_path = tmp_path / "old_p0.ckpt"
    torch.save(legacy, legacy_path)
    loaded_parent = load_feature_codec_checkpoint(legacy_path)
    assert loaded_parent.config.score_spatial_predictor == "linear"
    assert loaded_parent.config.score_spatial_hidden == 32


@pytest.mark.parametrize("spatial_entropy", ["split", "gaussian", "conditional_scale"])
def test_spatial_entropy_checkpoint_strict_load(spatial_entropy, tmp_path):
    cfg = small_config(
        score_mean_condition=True,
        score_channel_context=True,
        score_spatial_context=True,
        score_spatial_entropy=spatial_entropy,
        score_spatial_hidden=4,
    )
    codec = ObservableLowRank1DCodec(cfg).eval()
    codec.update(force=True)
    saved = {
        "state_dict": {
            f"model.feature_codec.{key}": value for key, value in codec.state_dict().items()
        },
        "feature_codec_config": cfg.to_dict(),
    }
    path = tmp_path / f"{spatial_entropy}.ckpt"
    torch.save(saved, path)
    loaded = load_feature_codec_checkpoint(path)
    assert loaded.config == cfg
    validate_feature_codec_checkpoint(saved, cfg)
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_feature_codec_checkpoint(saved, replace(cfg, score_spatial_entropy="shared"))


def test_invalid_spatial_predictor_config_is_rejected():
    with pytest.raises(ValueError, match="score_spatial_predictor"):
        small_config(score_spatial_predictor="unknown")
    with pytest.raises(ValueError, match="requires spatial context"):
        small_config(score_spatial_predictor="residual3")
    with pytest.raises(ValueError, match="score_spatial_entropy"):
        small_config(score_spatial_entropy="unknown")
    with pytest.raises(ValueError, match="requires spatial context"):
        small_config(score_spatial_entropy="split")

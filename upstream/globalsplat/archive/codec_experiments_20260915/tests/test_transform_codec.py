"""Transform ablations: common initialization, trainability, and real entropy coding."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressai")

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec, load_feature_codec_checkpoint
from globalsplat.compression.bitstream import SceneBitstream
from globalsplat.compression.checkpoint import infer_config, validate_feature_codec_checkpoint


@pytest.fixture(autouse=True, scope="module")
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**overrides):
    return CodecConfig(
        texture_channels=8, geometry_channels=8, geometry_observable_channels=4,
        rank=3, residual_n=6, residual_m=8, adapter_hidden=6, **overrides,
    )


@pytest.mark.parametrize("residual", [True, False])
def test_nonlinear_starts_identically_and_preserves_rng_for_following_modules(residual):
    torch.manual_seed(111123)
    linear = ObservableLowRank1DCodec(config(use_residual=residual)).eval()
    next_linear = torch.randn(17)
    torch.manual_seed(111123)
    nonlinear = ObservableLowRank1DCodec(config(transform="nonlinear", use_residual=residual)).eval()
    next_nonlinear = torch.randn(17)
    torch.testing.assert_close(next_linear, next_nonlinear, rtol=0, atol=0)
    for key, value in linear.state_dict().items():
        torch.testing.assert_close(value, nonlinear.state_dict()[key], rtol=0, atol=0)
    assert not any("mlp" in key for key in linear.state_dict())
    appearance, geometry, positions = torch.randn(1, 33, 8), torch.randn(1, 33, 8), torch.randn(1, 33, 3)
    a = linear(appearance, geometry, positions, training=False)
    b = nonlinear(appearance, geometry, positions, training=False)
    torch.testing.assert_close(a.reconstruction_sorted, b.reconstruction_sorted, rtol=0, atol=0)
    for name in a.likelihoods:
        torch.testing.assert_close(a.likelihoods[name], b.likelihoods[name], rtol=0, atol=0)


@pytest.mark.parametrize("residual", [True, False])
def test_zero_output_mlps_learn_and_residual_freezing_is_preserved(residual):
    torch.manual_seed(53)
    codec = ObservableLowRank1DCodec(config(transform="nonlinear", use_residual=residual))
    codec.set_trainable(True)
    optim = torch.optim.Adam([p for p in codec.parameters() if p.requires_grad], lr=1e-3)
    appearance, geometry, positions = torch.randn(1, 32, 8), torch.randn(1, 32, 8), torch.randn(1, 32, 3)
    for step in range(2):
        optim.zero_grad(set_to_none=True)
        output = codec(appearance, geometry, positions)
        loss = output.texture.square().mean() + output.geometry_observable.square().mean() + 1e-4 * output.estimated_bits
        assert torch.isfinite(loss)
        loss.backward()
        for mlp in (codec.analysis_mlp, codec.synthesis_mlp):
            for layer in ([mlp[-1]] if step == 0 else [mlp[0], mlp[-1]]):
                assert torch.isfinite(layer.weight.grad).all()
                assert layer.weight.grad.abs().sum() > 0
        if not residual:
            assert all(p.grad is None and not p.requires_grad for p in codec.residual_codec.parameters())
        optim.step()


@pytest.mark.parametrize("rank", [56, 80])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("transform", ["linear", "nonlinear"])
def test_real_bitstream_and_checkpoint_roundtrip_for_every_architecture(tmp_path, rank, residual, transform):
    torch.manual_seed(59)
    cfg = CodecConfig(rank=rank, use_residual=residual, transform=transform,
                      residual_n=6, residual_m=8, adapter_hidden=6)
    codec = ObservableLowRank1DCodec(cfg).eval()
    if transform == "nonlinear":
        # Nonzero learned branches catch accidental use of the old linear-only
        # implementation in compress or decompress (zero init would hide that).
        with torch.no_grad():
            codec.analysis_mlp[-1].weight.normal_(0, 0.03)
            codec.synthesis_mlp[-1].weight.normal_(0, 0.03)
    appearance = torch.randn(1, 33, 512)
    geometry = torch.randn(1, 33, 512)
    positions = torch.randn(1, 33, 3)
    codec.update(force=True)
    with torch.no_grad():
        output = codec(appearance, geometry, positions, restore_original_order=False, training=False)
    compressed = codec.compress(appearance, geometry, positions)
    scene = SceneBitstream.unpack(compressed.data)
    assert bool(scene.flags & codec.FLAG_NONLINEAR) == (transform == "nonlinear")
    assert sum(scene.bytes_by_stream.values()) == len(compressed.data)
    assert (scene.bytes_by_stream["residual_y"] > 0) == residual
    assert (scene.bytes_by_stream["residual_z"] > 0) == residual
    reconstructed = torch.cat(codec.decompress(compressed.data), dim=-1)
    torch.testing.assert_close(reconstructed, output.reconstruction_sorted, rtol=1e-5, atol=2e-6)

    path = tmp_path / "codec.ckpt"
    saved = {
        "state_dict": {f"model.feature_codec.{k}": v for k, v in codec.state_dict().items()},
        "feature_codec_config": cfg.to_dict(), "global_step": 50000,
    }
    torch.save(saved, path)
    loaded = load_feature_codec_checkpoint(path).codec.eval()
    assert loaded.config == cfg
    torch.testing.assert_close(torch.cat(loaded.decompress(compressed.data), dim=-1), reconstructed)
    validate_feature_codec_checkpoint(saved, cfg)
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_feature_codec_checkpoint(saved, replace(cfg, use_residual=not residual))
    other = ObservableLowRank1DCodec(replace(cfg, transform="linear" if transform == "nonlinear" else "nonlinear"))
    with pytest.raises(ValueError, match="bitstream transform"):
        other.decompress(compressed.data)


def test_legacy_linear_loading_and_wrong_nonlinear_configuration(tmp_path):
    codec = ObservableLowRank1DCodec(config()).eval()
    legacy = {"state_dict": {f"model.feature_codec.{k}": v for k, v in codec.state_dict().items()}}
    path = tmp_path / "legacy.ckpt"
    torch.save(legacy, path)
    loaded = load_feature_codec_checkpoint(path)
    assert loaded.config.transform == "linear"
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_feature_codec_checkpoint(legacy, config(transform="nonlinear"))
    nonlinear = ObservableLowRank1DCodec(config(transform="nonlinear"))
    partial = dict(nonlinear.state_dict())
    del partial["analysis_mlp.0.weight"]
    with pytest.raises(ValueError, match="both bias-free"):
        infer_config(partial)
    wrong = config(transform="nonlinear", transform_hidden=64).to_dict()
    with pytest.raises(ValueError, match="configuration mismatch"):
        infer_config(nonlinear.state_dict(), wrong)

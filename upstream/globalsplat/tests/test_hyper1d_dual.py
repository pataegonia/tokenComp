"""Residual coding must use the decoded base and charge both MSH streams."""

from dataclasses import replace
import torch
import pytest

from globalsplat.compression import FeatureHyperprior1DCodec, scene_bytes_by_stream
from globalsplat.compression.bitstream import DualHyper1DSceneBitstream, ResidualBitstream
from globalsplat.compression.checkpoint import infer_feature_codec_config, load_feature_codec_checkpoint
from test_hyper1d_codec import tiny_config, cpu_threads


@pytest.mark.parametrize("architecture", ["legacy", "plain4"])
@pytest.mark.parametrize("strides", [(2, 2), (2, 1)])
@pytest.mark.parametrize("points", [1, 33, 4096])
def test_dual_morton_roundtrip_and_decoded_base_residual(architecture, strides, points):
    codec = FeatureHyperprior1DCodec(tiny_config(paths=2, use_morton=True,
        architecture=architecture, strides=strides, input_norm="calibrated")).eval()
    codec.f_mean.fill_(0.25)
    codec.f_std.fill_(1.5)
    codec.update(force=True, update_quantiles=True)
    t, g, positions = torch.randn(1, points, 4), torch.randn(1, points, 4), torch.randn(1, points, 3)
    residual_input = []
    handle = codec.residual_msh.analysis_adapter.register_forward_pre_hook(
        lambda module, args: residual_input.append(args[0].detach().clone()))
    encoded = codec.compress(t, g, positions)
    handle.remove()
    scene = DualHyper1DSceneBitstream.unpack(encoded.data)
    features, _ = codec._input(t, g, positions)
    base = codec.decompress_features(scene.base_payload, points)
    torch.testing.assert_close(residual_input[0], features - base, atol=0, rtol=0)
    residual = codec.residual_msh.decompress_features(scene.residual_payload, points)
    decoded = codec.decompress(encoded.data)
    torch.testing.assert_close(torch.cat(decoded, dim=-1), codec._restore_features(base + residual))
    expected = codec(t, g, positions, restore_original_order=False)
    torch.testing.assert_close(decoded[0], expected.texture, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(decoded[1], expected.geometry_observable, atol=2e-6, rtol=1e-5)
    restored = codec.decompress(encoded.data, inverse_permutation=encoded.order.inverse_permutation)
    torch.testing.assert_close(restored[0], codec(t, g, positions).texture)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert codec.compress(t, g, positions).data == encoded.data
        torch.testing.assert_close(codec.decompress(encoded.data)[0], decoded[0], atol=0, rtol=0)
    sizes = scene_bytes_by_stream(encoded.data)
    assert set(sizes) == {"base_y", "base_z", "residual_y", "residual_z", "container"}
    assert all(size > 0 for size in sizes.values())
    assert sizes["container"] == 172 and sum(sizes.values()) == len(encoded.data)
    assert set(expected.likelihoods) == set(sizes) - {"container"}
    torch.testing.assert_close(expected.estimated_bits,
        sum(-p.clamp_min(1e-9).log2().sum() for p in expected.likelihoods.values()))


@pytest.mark.parametrize("architecture", ["legacy", "plain4"])
def test_dual_independent_signed_hyperpriors_and_gradients(architecture):
    codec = FeatureHyperprior1DCodec(tiny_config(paths=2, architecture=architecture)).train()
    assert codec.entropy_bottleneck is not codec.residual_msh.entropy_bottleneck
    assert codec.h_s[0].weight.data_ptr() != codec.residual_msh.h_s[0].weight.data_ptr()
    captured = {}
    handles = []
    for name, branch in (("base", codec), ("residual", codec.residual_msh)):
        handles.append(branch.g_a.register_forward_hook(
            lambda module, args, result, key=name: captured.update({key + "_y": result.detach().clone()})))
        handles.append(branch.h_a.register_forward_pre_hook(
            lambda module, args, key=name: captured.update({key + "_h_input": args[0].detach().clone()})))
    codec.capture_diagnostics = True
    output = codec(torch.randn(2, 33, 4), torch.randn(2, 33, 4))
    (output.texture.square().mean() + output.geometry_observable.square().mean()
     + output.estimated_bits / 10000 + codec.aux_loss() / 10000).backward()
    for handle in handles:
        handle.remove()
    for name, branch in (("base", codec), ("residual", codec.residual_msh)):
        torch.testing.assert_close(captured[name + "_y"], captured[name + "_h_input"])
        assert (captured[name + "_y"] < 0).any()
        for submodule in (branch.g_a, branch.g_s, branch.h_a, branch.h_s, branch.entropy_bottleneck):
            grads = [p.grad for p in submodule.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads) > 0
    assert codec.geometry_projection.weight.grad.abs().sum() > 0
    assert "residual_scale_above_table_fraction" in codec.last_diagnostics


def test_path_metadata_strictness_and_old_single_checkpoint(tmp_path):
    single = FeatureHyperprior1DCodec(tiny_config())
    old_metadata = single.config.to_dict()
    del old_metadata["paths"]
    del old_metadata["architecture"]
    del old_metadata["base_rank"]
    assert infer_feature_codec_config(single.state_dict(), old_metadata) == single.config
    dual = FeatureHyperprior1DCodec(tiny_config(paths=2)).eval()
    dual.update()
    old_dual_metadata = dual.config.to_dict()
    del old_dual_metadata["base_rank"]
    assert infer_feature_codec_config(dual.state_dict(), old_dual_metadata) == dual.config
    checkpoint = tmp_path / "dual.ckpt"
    torch.save({"state_dict": dual.state_dict(), "feature_codec_config": dual.config.to_dict()}, checkpoint)
    loaded = load_feature_codec_checkpoint(checkpoint)
    assert loaded.config.paths == 2
    data = dual.compress(torch.randn(1, 33, 4), torch.randn(1, 33, 4)).data
    torch.testing.assert_close(loaded.codec.decompress(data)[0], dual.decompress(data)[0], atol=0, rtol=0)
    with pytest.raises(ValueError, match="state mismatch"):
        infer_feature_codec_config(dual.state_dict(), old_metadata)
    with pytest.raises(ValueError, match="state mismatch"):
        infer_feature_codec_config(single.state_dict(), dual.config.to_dict())
    with pytest.raises(ValueError, match="magic"):
        single.decompress(data)


def test_both_payloads_validated_before_entropy_decode(monkeypatch):
    codec = FeatureHyperprior1DCodec(tiny_config(paths=2)).eval()
    codec.update()
    data = codec.compress(torch.randn(1, 33, 4), torch.randn(1, 33, 4)).data
    scene = DualHyper1DSceneBitstream.unpack(data)
    for branch in (codec, codec.residual_msh):
        monkeypatch.setattr(branch.entropy_bottleneck, "decompress",
            lambda *a, **kw: pytest.fail("invalid metadata reached entropy decoding"))
    for field in ("base_payload", "residual_payload"):
        packed = ResidualBitstream.unpack(getattr(scene, field))
        bad_shape = replace(packed, y_shape=(1, packed.y_shape[1] + 1)).pack()
        with pytest.raises(ValueError, match="points/strides"):
            codec.decompress(replace(scene, **{field: bad_shape}).pack())
        wrong_count = replace(packed, z_strings=packed.z_strings * 2).pack()
        with pytest.raises(ValueError, match="one scene"):
            replace(scene, **{field: wrong_count}).pack()
    for bad in (data[:30], data + b"x", data[:-1] + bytes([data[-1] ^ 1])):
        with pytest.raises(ValueError):
            codec.decompress(bad)
    with pytest.raises(ValueError, match="configuration"):
        codec.decompress(replace(scene, flags=1).pack())


@pytest.mark.parametrize("paths", [0, 3, True, 2.0])
def test_invalid_path_counts(paths):
    with pytest.raises(ValueError, match="paths"):
        tiny_config(paths=paths)


@pytest.mark.parametrize("morton", [False, True])
@pytest.mark.parametrize("strides", [(2, 2), (2, 1)])
def test_low_rank_base_codes_decoded_scores_then_full_width_residual(morton, strides):
    codec = FeatureHyperprior1DCodec(tiny_config(paths=2, base_rank=3,
        use_morton=morton, strides=strides)).eval()
    assert codec.g_a[0].in_channels == 3
    assert codec.g_s[-1].out_channels == 3
    assert codec.residual_msh.g_a[0].in_channels == 6
    assert codec.residual_msh.g_s[-1].out_channels == 6
    assert not hasattr(codec, "score_context")
    codec.update(force=True)
    texture, geometry = torch.randn(1, 33, 4), torch.randn(1, 33, 4)
    positions = torch.randn(1, 33, 3) if morton else None
    residual_inputs = []
    handle = codec.residual_msh.analysis_adapter.register_forward_pre_hook(
        lambda module, args: residual_inputs.append(args[0].detach().clone()))
    encoded = codec.compress(texture, geometry, positions)
    handle.remove()
    scene = DualHyper1DSceneBitstream.unpack(encoded.data)
    assert scene.flags == (2 | int(morton))
    features, _ = codec._input(texture, geometry, positions)
    scores = codec.decompress_features(scene.base_payload, 33)
    assert scores.shape == (1, 3, 1, 33)
    base = codec._base_output(scores)
    torch.testing.assert_close(residual_inputs[0], features - base, atol=0, rtol=0)
    residual = codec.residual_msh.decompress_features(scene.residual_payload, 33)
    decoded = codec.decompress(encoded.data)
    torch.testing.assert_close(torch.cat(decoded, dim=-1), codec._restore_features(base + residual))
    estimated = codec(texture, geometry, positions, restore_original_order=False)
    torch.testing.assert_close(decoded[0], estimated.texture, atol=2e-6, rtol=1e-5)
    assert set(estimated.likelihoods) == {"base_y", "base_z", "residual_y", "residual_z"}
    with pytest.raises(ValueError, match="configuration"):
        FeatureHyperprior1DCodec(tiny_config(paths=2, use_morton=morton, strides=strides)).decompress(encoded.data)
    with pytest.raises(ValueError, match="configuration"):
        codec.decompress(replace(scene, flags=int(morton)).pack())


def test_low_rank_gradients_and_checkpoint_reload(tmp_path):
    codec = FeatureHyperprior1DCodec(tiny_config(paths=2, base_rank=3)).train()
    output = codec(torch.randn(2, 33, 4), torch.randn(2, 33, 4))
    (output.texture.square().mean() + output.geometry_observable.square().mean()
     + output.estimated_bits / 10000).backward()
    for module in (codec.base_analysis, codec.base_synthesis, codec.residual_msh.g_a):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())
    codec.eval()
    codec.update(force=True)
    checkpoint = tmp_path / "lowrank.ckpt"
    torch.save({"state_dict": codec.state_dict(), "feature_codec_config": codec.config.to_dict()}, checkpoint)
    loaded = load_feature_codec_checkpoint(checkpoint)
    assert loaded.config.base_rank == 3
    texture, geometry = torch.randn(1, 33, 4), torch.randn(1, 33, 4)
    data = codec.compress(texture, geometry).data
    torch.testing.assert_close(loaded.codec.decompress(data)[0], codec.decompress(data)[0], atol=0, rtol=0)
    old_metadata = dict(codec.config.to_dict(), base_rank=0)
    with pytest.raises(ValueError, match="state mismatch"):
        infer_feature_codec_config(codec.state_dict(), old_metadata)


@pytest.mark.parametrize("rank, paths", [(-1, 2), (6, 2), (True, 2), (2, 1)])
def test_invalid_low_rank_config(rank, paths):
    with pytest.raises(ValueError, match="base_rank"):
        tiny_config(paths=paths, base_rank=rank)

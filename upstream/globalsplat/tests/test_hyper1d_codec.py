"""CPU roundtrips, format rejection, gradients and strict checkpoint routing."""

from dataclasses import replace
import copy
import hashlib
import struct
import torch
import pytest
from globalsplat.compression import (Hyper1DConfig, FeatureHyperprior1DCodec,
    Hyper1DSceneBitstream, CodecConfig, ObservableLowRank1DCodec, scene_bytes_by_stream)
from globalsplat.compression.bitstream import ResidualBitstream, SceneBitstream
from globalsplat.compression.checkpoint import (load_feature_codec_checkpoint,
    validate_feature_codec_checkpoint, infer_feature_codec_config)
from globalsplat.compression.calibration import ChannelMoments


def tiny_config(**kwargs):
    return Hyper1DConfig(texture_channels=4, geometry_channels=4,
                         geometry_observable_channels=2, n=3, m=6, adapter_hidden=3, **kwargs)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("strides", [(2, 2), (2, 1)])
@pytest.mark.parametrize("points", [1, 31, 32, 33, 4096])
def test_variable_length_entropy_roundtrip(points, strides):
    codec = FeatureHyperprior1DCodec(tiny_config(strides=strides)).eval()
    codec.update(force=True)
    t, g = torch.randn(1, points, 4), torch.randn(1, points, 4)
    output = codec(t, g)
    compressed = codec.compress(t, g)
    decoded = codec.decompress(compressed.data)
    torch.testing.assert_close(output.texture, decoded[0], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(output.geometry_observable, decoded[1], atol=2e-6, rtol=1e-5)
    assert decoded[0].shape == (1, points, 4)
    assert output.likelihoods.keys() == {"y", "z"}
    assert all(value.dtype == torch.float32 for value in output.likelihoods.values())
    scene = Hyper1DSceneBitstream.unpack(compressed.data)
    payload = ResidualBitstream.unpack(scene.payload)
    assert (payload.y_shape, payload.z_shape) == codec.config.latent_shapes(points)
    sizes = scene_bytes_by_stream(compressed.data)
    assert sizes["container"] == 112 and sum(sizes.values()) == len(compressed.data)
    with pytest.raises(ValueError):
        SceneBitstream.unpack(compressed.data)


def test_signed_hyper_analysis_and_gradients():
    codec = FeatureHyperprior1DCodec(tiny_config()).train()
    captured = {}
    handle = codec.h_a.register_forward_pre_hook(lambda module, args: captured.update(y=args[0].detach().clone()))
    t, g = torch.randn(2, 33, 4), torch.randn(2, 33, 4)
    output = codec(t, g)
    y = codec.g_a(codec.analysis_adapter(codec._input(t, g, None)[0]))
    torch.testing.assert_close(captured["y"], y)
    assert (captured["y"] < 0).any()
    handle.remove()
    (output.texture.square().mean() + output.geometry_observable.square().mean() + output.estimated_bits / 10000).backward()
    for module in (codec.geometry_projection, codec.analysis_adapter, codec.g_a, codec.g_s,
                   codec.h_a, codec.h_s, codec.synthesis_adapter, codec.entropy_bottleneck):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert not hasattr(codec, "score_context")
    with pytest.raises(ValueError):
        codec.set_trainable_scope("score_probability")


def test_morton_optional_and_sender_amp_parity():
    t, g = torch.randn(1, 33, 4), torch.randn(1, 33, 4)
    for morton in (False, True):
        codec = FeatureHyperprior1DCodec(tiny_config(use_morton=morton)).eval()
        codec.update()
        positions = torch.randn(1, 33, 3) if morton else torch.randn(7)
        compressed = codec.compress(t, g, positions)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            same = codec.compress(t, g, positions)
            decoded = codec.decompress(compressed.data)
            noisy_eval = codec(t, g, positions, training=False)
        assert same.data == compressed.data
        output = codec(t, g, positions, restore_original_order=False)
        torch.testing.assert_close(output.texture, decoded[0])
        restored = codec.decompress(compressed.data, inverse_permutation=compressed.order.inverse_permutation)
        torch.testing.assert_close(restored[0], codec(t, g, positions).texture)
        assert all(l.dtype == torch.float32 for l in noisy_eval.likelihoods.values())
        if morton:
            with pytest.raises(ValueError):
                codec(t, g)
        else:
            assert torch.equal(compressed.order.permutation, torch.arange(33).view(1, -1))


def reseal(data):
    header = Hyper1DSceneBitstream.HEADER.size
    zeroed = data[:header - 32] + bytes(32) + data[header:]
    return data[:header - 32] + hashlib.sha256(zeroed).digest() + data[header:]


def test_stream_rejects_malformed_metadata_before_entropy_decode():
    codec = FeatureHyperprior1DCodec(tiny_config()).eval()
    codec.update()
    data = codec.compress(torch.randn(1, 33, 4), torch.randn(1, 33, 4)).data
    for offset, fmt, value in [(8, ">H", 2), (10, ">H", 2), (10, ">H", 4),
                                (12, ">I", 0), (16, ">I", 999), (20, ">Q", 0)]:
        edited = bytearray(data)
        struct.pack_into(fmt, edited, offset, value)
        with pytest.raises(ValueError):
            codec.decompress(reseal(bytes(edited)))
    for corrupted in (data[:25], data + b"x", b"BADMAGIC" + data[8:], data[:-1] + bytes([data[-1] ^ 1])):
        with pytest.raises(ValueError):
            codec.decompress(corrupted)
    scene = Hyper1DSceneBitstream.unpack(data)
    packed = ResidualBitstream.unpack(scene.payload)
    bad_shape = replace(packed, y_shape=(1, packed.y_shape[1] + 1)).pack()
    with pytest.raises(ValueError, match="points/strides"):
        codec.decompress(replace(scene, payload=bad_shape).pack())
    wrong_count = replace(packed, y_strings=packed.y_strings * 2).pack()
    with pytest.raises(ValueError, match="one scene"):
        replace(scene, payload=wrong_count).pack()
    with pytest.raises(ValueError):
        codec.compress(torch.randn(2, 33, 4), torch.randn(2, 33, 4))
    wrong_flags = replace(scene, flags=1).pack()
    with pytest.raises(ValueError, match="configuration"):
        codec.decompress(wrong_flags)
    other = FeatureHyperprior1DCodec(tiny_config(strides=(2, 1))).eval()
    with pytest.raises(ValueError, match="points/strides"):
        other.decompress(data)


@pytest.mark.parametrize("normalization", ["none", "calibrated"])
def test_checkpoint_strict_routing_and_dynamic_cdf_reload(tmp_path, normalization):
    codec = FeatureHyperprior1DCodec(tiny_config(input_norm=normalization)).eval()
    if normalization == "calibrated":
        codec.f_mean.fill_(0.25)
        codec.f_std.fill_(1.5)
    codec.update()
    state = {"model.feature_codec." + key: value for key, value in codec.state_dict().items()}
    checkpoint = {"state_dict": state, "feature_codec_config": codec.config.to_dict()}
    path = tmp_path / "codec.ckpt"
    torch.save(checkpoint, path)
    loaded = load_feature_codec_checkpoint(path)
    assert loaded.config == codec.config
    t, g = torch.randn(1, 31, 4), torch.randn(1, 31, 4)
    data = codec.compress(t, g).data
    torch.testing.assert_close(codec.decompress(data)[0], loaded.codec.decompress(data)[0], rtol=0, atol=0)
    validate_feature_codec_checkpoint(checkpoint, codec.config)
    with pytest.raises(ValueError, match="type"):
        validate_feature_codec_checkpoint(checkpoint, CodecConfig())
    with pytest.raises(ValueError, match="configuration"):
        validate_feature_codec_checkpoint(checkpoint, replace(codec.config, strides=(2, 1)))
    with pytest.raises(ValueError, match="require"):
        infer_feature_codec_config(codec.state_dict())
    bad = copy.deepcopy(codec.config.to_dict())
    del bad["strides"]
    with pytest.raises(ValueError, match="incomplete"):
        infer_feature_codec_config(codec.state_dict(), bad)
    bad["codec_type"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        infer_feature_codec_config(codec.state_dict(), bad)


def test_normalization_modes_same_keys_but_validate_on_load():
    a = FeatureHyperprior1DCodec(tiny_config(input_norm="calibrated"))
    b = FeatureHyperprior1DCodec(tiny_config())
    assert a.state_dict().keys() == b.state_dict().keys()
    a.f_mean.fill_(1)
    with pytest.raises(ValueError, match="identity"):
        b.load_state_dict(a.state_dict())
    for value in (0, -1, float("nan"), float("inf")):
        a.f_std.fill_(value)
        with pytest.raises(ValueError):
            a.validate_normalization()


@pytest.mark.parametrize("values", [{"strides": (1, 2)}, {"adapter_hidden": 4},
                                    {"input_norm": "scene"}, {"use_morton": 1}])
def test_config_rejects_unsupported_modes(values):
    with pytest.raises(ValueError):
        Hyper1DConfig(**values)


def test_welford_matches_population_statistics_and_std_floor():
    values = torch.randn(7, 31, 6) + 10000
    values[..., 0] = 123
    moments = ChannelMoments(6)
    for scene in values:
        moments.update(scene)
    mean, std = moments.buffers()
    flat = values.double().flatten(0, 1)
    torch.testing.assert_close(mean.flatten().double(), flat.mean(0), atol=0.001, rtol=0)
    torch.testing.assert_close(std.flatten().double(), flat.std(0, unbiased=False).clamp_min(1e-6), atol=1e-6, rtol=1e-6)
    assert moments.count == 217

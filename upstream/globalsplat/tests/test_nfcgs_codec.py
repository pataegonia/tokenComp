import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressai")

from globalsplat.compression import (
    CodecConfig,
    ObservableLowRank1DCodec,
    initialize_observable_from_vanilla,
    load_codec_initialization,
)
from globalsplat.compression.bitstream import SceneBitstream
from globalsplat.compression.checkpoint import infer_config
from globalsplat.model.decoder.gaussian_decoder import TokenCoarseToFine3DGS
from globalsplat.model.model_wrapper import (
    GlobalSplatModule,
    make_anchor_alternating_input_subsets_and_shared_targets,
)


def test_actual_bitstream_eval_refreshes_stale_cdfs_without_changing_quantiles():
    from copy import deepcopy
    from types import SimpleNamespace

    torch.manual_seed(9)
    codec = ObservableLowRank1DCodec(
        CodecConfig(
            texture_channels=8,
            geometry_channels=8,
            geometry_observable_channels=4,
            rank=3,
            residual_n=6,
            residual_m=8,
            adapter_hidden=6,
            score_slice_channels=2,
        )
    ).eval()
    codec.update(force=True)
    score_entropy = codec.score_context.group_entropies[0].entropy_bottleneck
    residual_entropy = codec.residual_codec.entropy_bottleneck
    stale_score = score_entropy._quantized_cdf.clone()
    stale_residual = residual_entropy._quantized_cdf.clone()
    with torch.no_grad():
        score_entropy.biases[0].add_(3)
        residual_entropy.biases[0].add_(3)
    parameters = {name: value.clone() for name, value in codec.named_parameters()}
    expected = deepcopy(codec)
    expected.update(force=True)
    assert not torch.equal(
        stale_score,
        expected.score_context.group_entropies[0].entropy_bottleneck._quantized_cdf,
    )
    assert not torch.equal(
        stale_residual, expected.residual_codec.entropy_bottleneck._quantized_cdf
    )

    module = SimpleNamespace(
        model=SimpleNamespace(
            feature_codec=codec, set_stage=lambda *args, **kwargs: None
        ),
        final_stage=3,
        test_cfg={"actual_bitstream": True},
        _load_eval_utils=lambda: (None, None, None, SimpleNamespace, None, None),
    )
    GlobalSplatModule.on_test_start(module)

    torch.testing.assert_close(
        score_entropy._quantized_cdf,
        expected.score_context.group_entropies[0].entropy_bottleneck._quantized_cdf,
    )
    torch.testing.assert_close(
        residual_entropy._quantized_cdf,
        expected.residual_codec.entropy_bottleneck._quantized_cdf,
    )
    for name, value in codec.named_parameters():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)


def test_codec_boundary_feeds_split_width_decoder():
    torch.manual_seed(5)
    codec = ObservableLowRank1DCodec(
        CodecConfig(
            texture_channels=16,
            geometry_channels=16,
            geometry_observable_channels=8,
            rank=4,
            residual_n=6,
            residual_m=8,
            adapter_hidden=6,
            score_slice_channels=2,
        )
    ).eval()
    decoder = TokenCoarseToFine3DGS(C=16, geometry_C=8, M_max=2, sh_degree=0).eval()

    appearance = torch.randn(2, 64, 16)
    geometry = torch.randn(2, 64, 16)
    observable = codec.project_geometry(geometry)
    centers = decoder.decoded_token_centers(observable)
    coded = codec(
        appearance,
        geometry,
        centers,
        restore_original_order=False,
        training=False,
    )
    means, rotations, scales, sh, opacities, reg = decoder(
        (coded.texture, coded.geometry_observable)
    )

    assert centers.shape == (2, 64, 3)
    assert coded.reconstruction_sorted.shape == (2, 64, 24)
    assert means.shape == (2, 64, 3)
    assert rotations.shape == (2, 64, 6)
    assert scales.shape == (2, 64, 3)
    assert sh.shape == (2, 64, 1, 3)
    assert opacities.shape == (2, 64, 1)
    assert reg.ndim == 0


def test_vanilla_decoder_keeps_shared_512_style_width():
    decoder = TokenCoarseToFine3DGS(C=12, M_max=2, sh_degree=0)
    assert decoder.geometry_C == 12
    appearance = torch.randn(1, 8, 12)
    geometry = torch.randn(1, 8, 12)
    assert decoder((appearance, geometry))[0].shape == (1, 8, 3)


def test_full_codec_has_all_likelihoods_and_roundtrips_bitstream():
    torch.manual_seed(23)
    codec = ObservableLowRank1DCodec(
        CodecConfig(
            texture_channels=8,
            geometry_channels=8,
            geometry_observable_channels=4,
            rank=3,
            residual_n=6,
            residual_m=8,
            adapter_hidden=6,
            score_slice_channels=2,
        )
    ).eval()
    appearance = torch.randn(1, 32, 8)
    geometry = torch.randn(1, 32, 8)
    positions = torch.randn(1, 32, 3)

    output = codec(
        appearance,
        geometry,
        positions,
        restore_original_order=False,
        training=False,
    )
    assert set(output.likelihoods) == {"score", "residual_y", "residual_z"}
    inferred = infer_config(codec.state_dict())
    assert inferred.rank == 3
    assert inferred.observable_channels == 12

    codec.update(force=True)
    compressed = codec.compress(appearance, geometry, positions)
    scene = SceneBitstream.unpack(compressed.data)
    assert scene.flags == codec.FLAGS
    assert scene.residual
    texture_hat, geometry_hat = codec.decompress(compressed.data)
    reconstructed = torch.cat((texture_hat, geometry_hat), dim=-1)
    torch.testing.assert_close(
        reconstructed,
        output.reconstruction_sorted,
        rtol=1e-6,
        atol=2e-7,
    )


def test_observable_qr_initialization_preserves_decoder_linear_maps():
    class Decoder(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.geo_pos_readout = torch.nn.Linear(channels, 2)
            self.geo_param_readout = torch.nn.Linear(channels, 2)
            self.gate_readout = torch.nn.Linear(channels, 1)

    class Vanilla(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gaussian_decoder = Decoder(7)

    class Observable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_codec = torch.nn.Module()
            self.feature_codec.geometry_projection = torch.nn.Linear(7, 5, bias=False)
            self.gaussian_decoder = Decoder(5)

    torch.manual_seed(17)
    vanilla = Vanilla()
    observable = Observable()
    source = {"state_dict": {f"model.{k}": v for k, v in vanilla.state_dict().items()}}
    report = initialize_observable_from_vanilla(observable, source)

    value = torch.randn(3, 7)
    projected = observable.feature_codec.geometry_projection(value)
    for old, new in (
        (
            vanilla.gaussian_decoder.geo_pos_readout,
            observable.gaussian_decoder.geo_pos_readout,
        ),
        (
            vanilla.gaussian_decoder.geo_param_readout,
            observable.gaussian_decoder.geo_param_readout,
        ),
        (
            vanilla.gaussian_decoder.gate_readout,
            observable.gaussian_decoder.gate_readout,
        ),
    ):
        torch.testing.assert_close(old(value), new(projected), rtol=1e-5, atol=1e-6)
    assert report.max_reparameterization_error < 1e-6
    assert report.max_orthogonality_error < 1e-6


def test_artifact_geometry_basis_preserves_decoder_linear_maps():
    class Decoder(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.geo_pos_readout = torch.nn.Linear(channels, 2)
            self.geo_param_readout = torch.nn.Linear(channels, 2)
            self.gate_readout = torch.nn.Linear(channels, 1)

    class Vanilla(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gaussian_decoder = Decoder(7)

    class Observable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_codec = torch.nn.Module()
            self.feature_codec.geometry_projection = torch.nn.Linear(7, 5, bias=False)
            self.gaussian_decoder = Decoder(5)

    torch.manual_seed(19)
    vanilla = Vanilla()
    observable = Observable()
    stacked = torch.cat(
        [
            vanilla.gaussian_decoder.geo_pos_readout.weight,
            vanilla.gaussian_decoder.geo_param_readout.weight,
            vanilla.gaussian_decoder.gate_readout.weight,
        ],
        dim=0,
    )
    projection = torch.linalg.qr(stacked.double().T, mode="reduced").Q.T.float()
    source = {"state_dict": {f"model.{k}": v for k, v in vanilla.state_dict().items()}}
    report = initialize_observable_from_vanilla(
        observable,
        source,
        observable_projection=projection,
    )

    torch.testing.assert_close(
        observable.feature_codec.geometry_projection.weight, projection
    )
    value = torch.randn(3, 7)
    projected = observable.feature_codec.geometry_projection(value)
    for old, new in (
        (
            vanilla.gaussian_decoder.geo_pos_readout,
            observable.gaussian_decoder.geo_pos_readout,
        ),
        (
            vanilla.gaussian_decoder.geo_param_readout,
            observable.gaussian_decoder.geo_param_readout,
        ),
        (
            vanilla.gaussian_decoder.gate_readout,
            observable.gaussian_decoder.gate_readout,
        ),
    ):
        torch.testing.assert_close(old(value), new(projected), rtol=1e-5, atol=1e-6)
    assert report.max_reparameterization_error < 1e-6


def test_statistics_artifact_loads_every_codec_initialization_tensor():
    codec = ObservableLowRank1DCodec(
        CodecConfig(
            texture_channels=8,
            geometry_channels=8,
            geometry_observable_channels=4,
            rank=3,
            residual_n=6,
            residual_m=8,
            adapter_hidden=6,
            score_slice_channels=2,
        )
    )
    torch.manual_seed(29)
    artifact = {
        "geometry_basis": torch.randn(4, 8),
        "shared_basis": torch.randn(3, 12),
        "score_scale": torch.rand(3).add(0.1),
        "residual_mean": torch.randn(12),
        "residual_std": torch.rand(12).add(0.1),
    }

    count, keys = load_codec_initialization(codec, artifact)

    assert count == 6
    assert keys == [
        "geometry_projection.weight",
        "residual_mean",
        "residual_std",
        "score_log_scale",
        "shared_basis",
        "shared_synthesis_basis",
    ]
    torch.testing.assert_close(
        codec.geometry_projection.weight, artifact["geometry_basis"]
    )
    torch.testing.assert_close(codec.shared_basis, artifact["shared_basis"])
    torch.testing.assert_close(codec.shared_synthesis_basis, artifact["shared_basis"])
    torch.testing.assert_close(codec.score_scale, artifact["score_scale"])
    torch.testing.assert_close(codec.residual_mean.flatten(), artifact["residual_mean"])
    torch.testing.assert_close(codec.residual_std.flatten(), artifact["residual_std"])


def test_paper_context_pool_becomes_two_13_view_subsets_with_shared_targets():
    frame_ids = torch.tensor([list(range(24))])
    inputs = {
        "frame_ids": frame_ids,
        "images": frame_ids[:, :, None, None, None].float(),
    }
    targets = {
        "frame_ids": torch.tensor([[30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]]),
        "images": torch.randn(1, 12, 3, 2, 2),
    }

    subset_a, subset_b, shared_targets, valid = (
        make_anchor_alternating_input_subsets_and_shared_targets(
            inputs=inputs,
            targets=targets,
        )
    )

    assert bool(valid.item())
    assert subset_a["frame_ids"].tolist() == [
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    ]
    assert subset_b["frame_ids"].tolist() == [
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23]
    ]
    assert shared_targets is targets

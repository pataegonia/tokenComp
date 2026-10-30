"""Read-only T04/B1 and T05 interventions on a frozen minimal two-path codec.

All outputs use the SAME sorted token order. Only ``actual_packet`` has a
transmittable rate. B1 is downstream of the learned geometry projection; it is
not the native-geometry (B0) GlobalSplat baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .bitstream import SceneBitstream, ScoreContextBitstream
from .codec import ObservableLowRank1DCodec


MODES = (
    "b1_direct", "lr_continuous", "lr_decoded", "score_quantization_bypass",
    "msh_quantization_bypass", "both_quantization_bypass", "actual_packet",
)
FORMULAS = dict(zip(MODES, (
    "F", "L(s)", "L(s_hat)", "L(s)+Hhat(F-L(s))",
    "L(s_hat)+Hcont(F-L(s_hat))", "L(s)+Hcont(F-L(s))",
    "L(s_hat)+Hhat(F-L(s_hat))",
)))


@dataclass
class DiagnosticPaths:
    features: dict[str, Tensor]
    packet: bytes
    permutation: Tensor
    score_mse: float
    stream_bytes: dict[str, int]
    score_wrapper_bytes: int


def validate_codec(codec: ObservableLowRank1DCodec) -> None:
    cfg = codec.config
    if (cfg.codec_mode != "two_path" or not cfg.use_residual or codec.score_context is None
            or codec.residual_codec is None):
        raise ValueError("T05 requires the two-path codec")
    if codec.training:
        raise ValueError("diagnostics require a frozen codec in eval mode")


def continuous_msh(codec: ObservableLowRank1DCodec, residual: Tensor) -> Tensor:
    """Bypass BOTH y/z quantizers while retaining the analysis/synthesis loss.

    With y passed unchanged, z and the hyper-synthesis entropy parameters have
    no influence on reconstruction. Hcont is therefore g_s(g_a(.)), including
    both multiscale adapters, normalization and cropping. It is not identity.
    """
    msh = codec.residual_codec
    width = residual.shape[1]
    value = (residual.transpose(1, 2).unsqueeze(2) - codec.residual_mean)
    value = value / codec.residual_std.clamp_min(1e-8)
    y = msh.g_a(msh.analysis_adapter(value))
    value = msh.synthesis_adapter(msh.g_s(y))[..., :1, :width]
    value = value * codec.residual_std + codec.residual_mean
    return value.squeeze(2).transpose(1, 2)


def coded_msh(codec: ObservableLowRank1DCodec, residual: Tensor) -> Tensor:
    """Use actual entropy-coder reconstruction, never approximate eval forward."""
    value = (residual.transpose(1, 2).unsqueeze(2) - codec.residual_mean)
    value = value / codec.residual_std.clamp_min(1e-8)
    payload, sender = codec.residual_codec.compress(value)
    receiver = codec.residual_codec.decompress(payload, output_width=residual.shape[1])
    if not torch.equal(sender, receiver):
        raise RuntimeError("MSH sender/receiver reconstruction mismatch")
    value = receiver * codec.residual_std + codec.residual_mean
    return value.squeeze(2).transpose(1, 2)


@torch.no_grad()
def build_paths(codec: ObservableLowRank1DCodec, texture: Tensor, geometry: Tensor,
                positions: Tensor) -> DiagnosticPaths:
    """Call under the SAME autocast context as production compression/decoding.

    s_hat is decoded from the real packet with causal reconstructed anchors.
    Every changed low-rank reconstruction receives a newly computed residual.
    The Score-bypass MSH stream is only an internal oracle, not a scene packet.
    """
    validate_codec(codec)
    if texture.shape[0] != 1:
        raise ValueError("one scene per diagnostic call is required")
    packed = codec.compress(texture, geometry, positions)
    scene = SceneBitstream.unpack(packed.data)
    features = packed.order.apply(codec._pack_features(texture, geometry))
    s = codec._analyze_low_rank(features)
    continuous_lr = codec._synthesize_low_rank(s)
    with torch.autocast(device_type=features.device.type, enabled=False):
        score_hat = codec.score_context.decompress(scene.score, scene.points)
    score_hat = score_hat.squeeze(2).transpose(1, 2)
    decoded_lr = codec._synthesize_low_rank(score_hat)
    actual = torch.cat(codec.decompress(packed.data), dim=-1)
    residual_hat = codec.residual_codec.decompress(scene.residual, output_width=scene.points)
    residual_hat = (residual_hat * codec.residual_std + codec.residual_mean).squeeze(2).transpose(1, 2)
    # The receiver widens the low-rank part to its FP32 parameters, as decompress does.
    reconstructed = codec._promote(decoded_lr, codec.shared_basis) + residual_hat
    if not torch.equal(reconstructed, actual):
        raise RuntimeError("T05 decomposition differs from actual packet decoding")
    continuous_residual = features - continuous_lr
    decoded_residual = features - decoded_lr
    values = {
        "b1_direct": features,
        "lr_continuous": continuous_lr,
        "lr_decoded": decoded_lr,
        "score_quantization_bypass": continuous_lr + coded_msh(codec, continuous_residual),
        "msh_quantization_bypass": decoded_lr + continuous_msh(codec, decoded_residual),
        "both_quantization_bypass": continuous_lr + continuous_msh(codec, continuous_residual),
        "actual_packet": actual,
    }
    for name, value in values.items():
        if value.shape != features.shape or not torch.isfinite(value).all():
            raise RuntimeError(f"invalid diagnostic features: {name}")
    if sum(scene.bytes_by_stream.values()) != len(packed.data):
        raise RuntimeError("packet byte accounting mismatch")
    score_stream = ScoreContextBitstream.unpack(scene.score)
    wrapper_bytes = len(scene.score) - sum(map(len, score_stream.strings))
    return DiagnosticPaths(
        features=values, packet=packed.data, permutation=packed.order.permutation,
        score_mse=float((s.float() - score_hat.float()).square().mean()),
        stream_bytes=scene.bytes_by_stream,
        score_wrapper_bytes=wrapper_bytes,
    )

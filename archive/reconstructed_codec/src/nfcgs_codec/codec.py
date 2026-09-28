"""Observable low-rank 1-D feature codec."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .bitstream import SceneBitstream
from .config import CodecConfig
from .entropy import FactorizedScoreEntropy
from .morton import MortonOrder, morton_order_3d
from .residual import ResidualHyperprior1D


@dataclass(slots=True)
class CodecOutput:
    texture: Tensor
    geometry_observable: Tensor
    reconstruction_sorted: Tensor
    likelihoods: dict[str, Tensor]
    mean: Tensor
    score_hat: Tensor
    residual_hat: Tensor
    order: MortonOrder | None

    @property
    def estimated_bits(self) -> Tensor:
        total = next(iter(self.likelihoods.values())).new_zeros(())
        for likelihood in self.likelihoods.values():
            total = total - torch.log2(likelihood.clamp_min(1e-9)).sum()
        return total


@dataclass(slots=True)
class CompressedScene:
    data: bytes
    order: MortonOrder | None
    reconstruction_sorted: Tensor | None = None


class ObservableLowRank1DCodec(nn.Module):
    """Codec reconstructed from the ``model.feature_codec`` checkpoint branch."""

    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.observable_channels
        rank = config.rank

        basis = torch.empty(rank, channels)
        nn.init.orthogonal_(basis)
        self.shared_basis = nn.Parameter(basis)
        if config.basis_parameterization == "untied_synthesis":
            self.shared_synthesis_basis = nn.Parameter(basis.clone())
        else:
            self.register_parameter("shared_synthesis_basis", None)
        self.score_log_scale = nn.Parameter(torch.zeros(rank))
        self.register_buffer("residual_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("residual_std", torch.ones(1, channels, 1, 1))
        self.geometry_projection = nn.Linear(
            config.geometry_channels,
            config.geometry_observable_channels,
            bias=False,
        )
        self.score_entropy = FactorizedScoreEntropy(rank)
        self.residual_codec = ResidualHyperprior1D(
            channels,
            n=config.residual_n,
            m=config.residual_m,
            adapter_hidden=config.adapter_hidden,
        )
        if config.synthesis_mlp_hidden is not None:
            self.synthesis_mlp = nn.Sequential(
                nn.Linear(rank, config.synthesis_mlp_hidden, bias=False),
                nn.GELU(),
                nn.Linear(config.synthesis_mlp_hidden, channels, bias=False),
            )
        else:
            self.synthesis_mlp = None

    @property
    def synthesis_basis(self) -> Tensor:
        if self.shared_synthesis_basis is None:
            return self.shared_basis
        return self.shared_synthesis_basis

    @property
    def score_scale(self) -> Tensor:
        return torch.exp(self.score_log_scale).clamp_min(1e-8)

    def _pack_features(self, texture: Tensor, geometry: Tensor) -> Tensor:
        expected_texture = self.config.texture_channels
        expected_geometry = self.config.geometry_channels
        if texture.ndim != 3 or texture.shape[-1] != expected_texture:
            raise ValueError(f"texture must have shape [B,N,{expected_texture}]")
        if geometry.ndim != 3 or geometry.shape[-1] != expected_geometry:
            raise ValueError(f"geometry must have shape [B,N,{expected_geometry}]")
        if texture.shape[:2] != geometry.shape[:2]:
            raise ValueError("texture and geometry must share [B,N]")
        observable = self.geometry_projection(geometry)
        return torch.cat([texture, observable], dim=-1)

    def _synthesize_low_rank(self, score: Tensor) -> Tensor:
        value = score @ self.synthesis_basis
        if self.synthesis_mlp is not None:
            value = value + self.synthesis_mlp(score)
        return value

    @staticmethod
    def _quantize_mean(mean: Tensor) -> Tensor:
        """FP16 side-information quantizer with an STE during training."""

        quantized = mean.to(torch.float16).to(mean.dtype)
        if torch.is_grad_enabled():
            return mean + (quantized - mean).detach()
        return quantized

    def _analyze_sorted(
        self, sorted_features: Tensor, *, training: bool | None = None
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        mean = self._quantize_mean(sorted_features.mean(dim=1))
        centered = sorted_features - mean[:, None, :]
        score = centered @ self.shared_basis.transpose(0, 1)
        normalized_score = score / self.score_scale[None, None, :]
        score_nchw = normalized_score.transpose(1, 2).unsqueeze(2)
        score_hat_nchw, score_likelihood = self.score_entropy(
            score_nchw, training=training
        )
        score_hat = score_hat_nchw.squeeze(2).transpose(1, 2)
        score_hat = score_hat * self.score_scale[None, None, :]
        low_rank = self._synthesize_low_rank(score_hat)
        residual = centered - low_rank
        residual_nchw = residual.transpose(1, 2).unsqueeze(2)
        normalized_residual = (residual_nchw - self.residual_mean) / self.residual_std.clamp_min(1e-8)
        residual_output = self.residual_codec(normalized_residual, training=training)
        residual_hat_nchw = (
            residual_output.reconstruction * self.residual_std + self.residual_mean
        )
        residual_hat = residual_hat_nchw.squeeze(2).transpose(1, 2)
        reconstruction = mean[:, None, :] + low_rank + residual_hat
        likelihoods = {"score": score_likelihood, **residual_output.likelihoods}
        return reconstruction, mean, score_hat, residual_hat, likelihoods

    def forward(
        self,
        texture: Tensor,
        geometry: Tensor,
        positions: Tensor | None = None,
        *,
        restore_original_order: bool = True,
        training: bool | None = None,
    ) -> CodecOutput:
        features = self._pack_features(texture, geometry)
        order = None
        if positions is not None:
            order = morton_order_3d(positions, self.config.morton_bits)
            features = order.apply(features)
        reconstruction, mean, score_hat, residual_hat, likelihoods = self._analyze_sorted(
            features, training=training
        )
        split = reconstruction
        if order is not None and restore_original_order:
            split = order.restore(split)
        texture_hat = split[..., : self.config.texture_channels]
        geometry_hat = split[..., self.config.texture_channels :]
        return CodecOutput(
            texture_hat,
            geometry_hat,
            reconstruction,
            likelihoods,
            mean,
            score_hat,
            residual_hat,
            order,
        )

    @torch.no_grad()
    def compress(
        self,
        texture: Tensor,
        geometry: Tensor,
        positions: Tensor | None = None,
    ) -> CompressedScene:
        features = self._pack_features(texture, geometry)
        order = None
        if positions is not None:
            order = morton_order_3d(positions, self.config.morton_bits)
            features = order.apply(features)
        mean = self._quantize_mean(features.mean(dim=1))
        centered = features - mean[:, None, :]
        score = centered @ self.shared_basis.transpose(0, 1)
        normalized = score / self.score_scale[None, None, :]
        score_nchw = normalized.transpose(1, 2).unsqueeze(2)
        score_strings = self.score_entropy.compress(score_nchw)
        if len(score_strings) != texture.shape[0]:
            raise RuntimeError("score entropy model returned an unexpected batch")
        score_hat_nchw = self.score_entropy.decompress(score_strings, score_nchw.shape[-2:])
        score_hat = score_hat_nchw.squeeze(2).transpose(1, 2)
        score_hat = score_hat * self.score_scale[None, None, :]
        low_rank = self._synthesize_low_rank(score_hat)
        residual = centered - low_rank
        residual_nchw = residual.transpose(1, 2).unsqueeze(2)
        normalized_residual = (residual_nchw - self.residual_mean) / self.residual_std.clamp_min(1e-8)

        if texture.shape[0] != 1:
            raise ValueError("the E2EM0301 scene container stores one scene at a time")
        residual_payload, residual_hat_norm = self.residual_codec.compress(normalized_residual)
        residual_hat = residual_hat_norm * self.residual_std + self.residual_mean
        reconstruction = (
            mean[:, None, :]
            + low_rank
            + residual_hat.squeeze(2).transpose(1, 2)
        )
        mean_bytes = (
            mean.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy().tobytes()
        )
        scene = SceneBitstream(
            points=features.shape[1],
            channels=features.shape[2],
            rank=self.config.rank,
            mean_fp16=mean_bytes,
            score=score_strings[0],
            residual=residual_payload,
        )
        return CompressedScene(scene.pack(), order, reconstruction)

    @torch.no_grad()
    def decompress(
        self,
        data: bytes,
        *,
        inverse_permutation: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scene = SceneBitstream.unpack(data)
        if scene.channels != self.config.observable_channels or scene.rank != self.config.rank:
            raise ValueError("bitstream channel/rank does not match this codec")
        parameter = self.shared_basis
        mean = torch.frombuffer(bytearray(scene.mean_fp16), dtype=torch.float16).to(
            device=parameter.device, dtype=parameter.dtype
        )
        mean = mean.reshape(1, scene.channels)
        score_hat_nchw = self.score_entropy.decompress(
            [scene.score], (1, scene.points)
        )
        score_hat = score_hat_nchw.squeeze(2).transpose(1, 2)
        score_hat = score_hat * self.score_scale[None, None, :]
        low_rank = self._synthesize_low_rank(score_hat)
        residual_hat_norm = self.residual_codec.decompress(
            scene.residual, output_width=scene.points
        )
        residual_hat = residual_hat_norm * self.residual_std + self.residual_mean
        reconstruction = (
            mean[:, None, :]
            + low_rank
            + residual_hat.squeeze(2).transpose(1, 2)
        )
        if inverse_permutation is not None:
            from .morton import batched_gather

            reconstruction = batched_gather(reconstruction, inverse_permutation)
        return (
            reconstruction[..., : self.config.texture_channels],
            reconstruction[..., self.config.texture_channels :],
        )

    def update(self, force: bool = False) -> bool:
        score_updated = self.score_entropy.update(force=force)
        residual_updated = self.residual_codec.update(force=force)
        return bool(score_updated or residual_updated)

    def aux_loss(self) -> Tensor:
        return self.score_entropy.entropy_bottleneck.loss() + self.residual_codec.entropy_bottleneck.loss()

    def extra_repr(self) -> str:
        return (
            f"channels={self.config.observable_channels}, rank={self.config.rank}, "
            f"basis={self.config.basis_parameterization}, residual="
            f"{self.config.residual_n}/{self.config.residual_m}"
        )

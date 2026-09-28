"""Observable low-rank 1-D feature codec."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .bitstream import SceneBitstream
from .config import CodecConfig
from .entropy import FactorizedScoreEntropy
from .morton import MortonOrder, identity_order_3d, morton_order_3d
from .residual import ResidualHyperprior1D
from .score_context import ContextualScoreEntropy


@dataclass(slots=True)
class CodecOutput:
    texture: Tensor
    geometry_observable: Tensor
    reconstruction_sorted: Tensor
    likelihoods: dict[str, Tensor]
    order: MortonOrder

    @property
    def estimated_bits_by_stream(self) -> dict[str, Tensor]:
        return {
            name: -torch.log2(likelihood.clamp_min(1e-9)).sum()
            for name, likelihood in self.likelihoods.items()
        }

    @property
    def estimated_bits(self) -> Tensor:
        streams = self.estimated_bits_by_stream
        if not streams:
            return self.reconstruction_sorted.new_zeros(())
        return torch.stack(tuple(streams.values())).sum()


@dataclass(slots=True)
class CompressedScene:
    data: bytes
    order: MortonOrder


class ObservableLowRank1DCodec(nn.Module):
    """Codec reconstructed from the ``model.feature_codec`` checkpoint branch."""

    FLAG_NO_RESIDUAL = 1 << 0
    FLAG_NO_MORTON = 1 << 1
    FLAG_NONLINEAR = 1 << 2
    FLAG_CONTEXTUAL_SCORE = 1 << 3

    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.observable_channels
        rank = config.rank

        basis = torch.empty(rank, channels)
        nn.init.orthogonal_(basis)
        self.shared_basis = nn.Parameter(basis)
        self.shared_synthesis_basis = nn.Parameter(basis.clone())
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
        if not config.use_residual:
            for parameter in self.residual_codec.parameters():
                parameter.requires_grad_(False)

        self.analysis_mlp = None
        self.synthesis_mlp = None
        if config.transform == "nonlinear":
            # Construct AFTER all shared modules, and preserve the CPU RNG for
            # modules constructed after this codec (including the GS decoder).
            # Identical seeds therefore give identical common initial weights.
            with torch.random.fork_rng(devices=[]):
                self.analysis_mlp = self._make_mlp(channels, rank, config.transform_hidden)
                self.synthesis_mlp = self._make_mlp(rank, channels, config.transform_hidden)

        self.score_context: ContextualScoreEntropy | None = None
        if config.use_contextual_score:
            # Keep initialization of the legacy codec and following Gaussian
            # decoder identical across ablation arms.
            with torch.random.fork_rng(devices=[]):
                self.score_context = ContextualScoreEntropy(
                    rank=rank,
                    scene_channels=channels,
                    mean_condition=config.score_mean_condition,
                    channel_context=config.score_channel_context,
                    spatial_context=config.score_spatial_context,
                    spatial_predictor=config.score_spatial_predictor,
                    spatial_entropy=config.score_spatial_entropy,
                    spatial_hidden=config.score_spatial_hidden,
                    slice_channels=config.score_slice_channels,
                    hidden=config.score_context_hidden,
                )
        self._freeze_inactive_score_entropy()

    @staticmethod
    def _make_mlp(inputs: int, outputs: int, hidden: int) -> nn.Sequential:
        mlp = nn.Sequential(
            nn.Linear(inputs, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, outputs, bias=False),
        )
        nn.init.zeros_(mlp[-1].weight)
        return mlp

    @property
    def synthesis_basis(self) -> Tensor:
        return self.shared_synthesis_basis

    @property
    def score_scale(self) -> Tensor:
        return torch.exp(self.score_log_scale).clamp_min(1e-8)

    def project_geometry(self, geometry: Tensor) -> Tensor:
        """Map GlobalSplat geometry tokens to decoder-observable coordinates."""

        expected = self.config.geometry_channels
        if geometry.ndim != 3 or geometry.shape[-1] != expected:
            raise ValueError(f"geometry must have shape [B,N,{expected}]")
        return self.geometry_projection(geometry)

    def _pack_features(self, texture: Tensor, geometry: Tensor) -> Tensor:
        expected_texture = self.config.texture_channels
        expected_geometry = self.config.geometry_channels
        if texture.ndim != 3 or texture.shape[-1] != expected_texture:
            raise ValueError(f"texture must have shape [B,N,{expected_texture}]")
        if geometry.ndim != 3 or geometry.shape[-1] != expected_geometry:
            raise ValueError(f"geometry must have shape [B,N,{expected_geometry}]")
        if texture.shape[:2] != geometry.shape[:2]:
            raise ValueError("texture and geometry must share [B,N]")
        observable = self.project_geometry(geometry)
        return torch.cat([texture, observable], dim=-1)

    def _synthesize_low_rank(self, score: Tensor) -> Tensor:
        value = score @ self.synthesis_basis
        if self.synthesis_mlp is not None:
            value = value + self.synthesis_mlp(score)
        return value

    def _analyze_low_rank(self, centered: Tensor) -> Tensor:
        value = centered @ self.shared_basis.transpose(0, 1)
        if self.analysis_mlp is not None:
            value = value + self.analysis_mlp(centered)
        return value

    def _make_order(self, positions: Tensor) -> MortonOrder:
        if not self.config.use_morton:
            return identity_order_3d(positions)
        return morton_order_3d(positions, self.config.morton_bits)

    def _freeze_inactive_score_entropy(self) -> None:
        if self.score_context is not None and self.config.score_channel_context:
            for parameter in self.score_entropy.parameters():
                parameter.requires_grad_(False)

    def set_trainable(self, trainable: bool = True) -> None:
        """Set codec trainability while keeping an ablated residual path frozen."""

        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
        if not self.config.use_residual:
            for parameter in self.residual_codec.parameters():
                parameter.requires_grad_(False)
        self._freeze_inactive_score_entropy()

    def set_trainable_scope(self, scope: str = "all") -> None:
        """Select full codec training or a cheap contextual-prior warm-up."""

        if scope == "all":
            self.set_trainable(True)
            return
        if scope not in ("score_context", "score_probability"):
            raise ValueError(
                "feature_codec_train_scope must be all, score_context, or score_probability"
            )
        if self.score_context is None:
            raise ValueError("score_context scope requires a contextual score configuration")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if scope == "score_context":
            for parameter in self.score_context.parameters():
                parameter.requires_grad_(True)
            if not self.config.score_channel_context:
                for parameter in self.score_entropy.parameters():
                    parameter.requires_grad_(True)
            return

        # Probability-only fitting preserves every decoded score value.  The
        # EntropyBottleneck quantiles define its reconstruction lattice, so keep
        # those fixed while allowing only density parameters (and optional
        # Gaussian scale predictors) to learn.
        for entropy in self.score_context.active_entropies(self.score_entropy):
            for parameter in entropy.parameters():
                parameter.requires_grad_(True)
            entropy.entropy_bottleneck.quantiles.requires_grad_(False)
        for parameter in self.score_context.spatial_log_scales.parameters():
            parameter.requires_grad_(True)
        for parameter in self.score_context.spatial_scale_predictors.parameters():
            parameter.requires_grad_(True)

    def _score_forward(
        self, score_nchw: Tensor, mean: Tensor, *, training: bool | None
    ) -> tuple[Tensor, Tensor]:
        if self.score_context is None:
            return self.score_entropy(score_nchw, training=training)
        # Entropy-side context must produce the same values in BF16 training,
        # standalone sender use, and receiver use.  Its tensors are small, so
        # keep this serial boundary in FP32 while the expensive renderer stays
        # under mixed precision.
        with torch.autocast(device_type=score_nchw.device.type, enabled=False):
            return self.score_context(
                score_nchw.float(), mean.float(), self.score_entropy, training=training
            )

    @staticmethod
    def _quantize_mean(mean: Tensor) -> Tensor:
        """FP16 side-information quantizer with an STE during training."""

        quantized = mean.to(torch.float16).to(mean.dtype)
        if torch.is_grad_enabled():
            return mean + (quantized - mean).detach()
        return quantized

    def _analyze_sorted(
        self, sorted_features: Tensor, *, training: bool | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        mean = self._quantize_mean(sorted_features.mean(dim=1))
        centered = sorted_features - mean[:, None, :]
        score = self._analyze_low_rank(centered)
        normalized_score = score / self.score_scale[None, None, :]
        score_nchw = normalized_score.transpose(1, 2).unsqueeze(2)
        score_hat_nchw, score_likelihood = self._score_forward(
            score_nchw, mean, training=training
        )
        score_hat = score_hat_nchw.squeeze(2).transpose(1, 2)
        score_hat = score_hat * self.score_scale[None, None, :]
        low_rank = self._synthesize_low_rank(score_hat)
        likelihoods = {"score": score_likelihood}
        if not self.config.use_residual:
            return mean[:, None, :] + low_rank, likelihoods

        residual = centered - low_rank
        residual_nchw = residual.transpose(1, 2).unsqueeze(2)
        normalized_residual = (
            residual_nchw - self.residual_mean
        ) / self.residual_std.clamp_min(1e-8)
        residual_output = self.residual_codec(normalized_residual, training=training)
        residual_hat_nchw = (
            residual_output.reconstruction * self.residual_std + self.residual_mean
        )
        residual_hat = residual_hat_nchw.squeeze(2).transpose(1, 2)
        likelihoods.update(residual_output.likelihoods)
        reconstruction = mean[:, None, :] + low_rank + residual_hat
        return reconstruction, likelihoods

    def forward(
        self,
        texture: Tensor,
        geometry: Tensor,
        positions: Tensor,
        *,
        restore_original_order: bool = True,
        training: bool | None = None,
    ) -> CodecOutput:
        features = self._pack_features(texture, geometry)
        order = self._make_order(positions)
        features = order.apply(features)
        reconstruction, likelihoods = self._analyze_sorted(features, training=training)
        split = reconstruction
        if restore_original_order:
            split = order.restore(split)
        texture_hat = split[..., : self.config.texture_channels]
        geometry_hat = split[..., self.config.texture_channels :]
        return CodecOutput(
            texture_hat,
            geometry_hat,
            reconstruction,
            likelihoods,
            order,
        )

    @torch.no_grad()
    def compress(
        self,
        texture: Tensor,
        geometry: Tensor,
        positions: Tensor,
    ) -> CompressedScene:
        features = self._pack_features(texture, geometry)
        order = self._make_order(positions)
        features = order.apply(features)
        mean = self._quantize_mean(features.mean(dim=1))
        centered = features - mean[:, None, :]
        score = self._analyze_low_rank(centered)
        normalized = score / self.score_scale[None, None, :]
        score_nchw = normalized.transpose(1, 2).unsqueeze(2)
        if self.score_context is None:
            score_strings = self.score_entropy.compress(score_nchw)
            if len(score_strings) != texture.shape[0]:
                raise RuntimeError("score entropy model returned an unexpected batch")
            score_payload = score_strings[0]
            score_hat_nchw = self.score_entropy.decompress(
                score_strings, score_nchw.shape[-2:]
            )
        else:
            with torch.autocast(device_type=score_nchw.device.type, enabled=False):
                score_payload, score_hat_nchw = self.score_context.compress(
                    score_nchw.float(), mean.float(), self.score_entropy
                )
        score_hat = score_hat_nchw.squeeze(2).transpose(1, 2)
        score_hat = score_hat * self.score_scale[None, None, :]
        low_rank = self._synthesize_low_rank(score_hat)
        if texture.shape[0] != 1:
            raise ValueError("the E2EM0301 scene container stores one scene at a time")

        residual_payload = b""
        flags = self.FLAG_NONLINEAR if self.config.transform == "nonlinear" else 0
        if self.score_context is not None:
            flags |= self.FLAG_CONTEXTUAL_SCORE
        if not self.config.use_residual:
            flags |= self.FLAG_NO_RESIDUAL
        if not self.config.use_morton:
            flags |= self.FLAG_NO_MORTON
        if self.config.use_residual:
            residual = centered - low_rank
            residual_nchw = residual.transpose(1, 2).unsqueeze(2)
            normalized_residual = (
                residual_nchw - self.residual_mean
            ) / self.residual_std.clamp_min(1e-8)
            residual_payload, _ = self.residual_codec.compress(normalized_residual)
        mean_bytes = (
            mean.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy().tobytes()
        )
        scene = SceneBitstream(
            points=features.shape[1],
            channels=features.shape[2],
            rank=self.config.rank,
            mean_fp16=mean_bytes,
            score=score_payload,
            residual=residual_payload,
            flags=flags,
        )
        return CompressedScene(scene.pack(), order)

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
        known_flags = (
            self.FLAG_NO_RESIDUAL
            | self.FLAG_NO_MORTON
            | self.FLAG_NONLINEAR
            | self.FLAG_CONTEXTUAL_SCORE
        )
        if scene.flags & ~known_flags:
            raise ValueError(f"unsupported scene bitstream flags: {scene.flags}")
        if bool(scene.flags & self.FLAG_NONLINEAR) != (self.config.transform == "nonlinear"):
            raise ValueError("bitstream transform does not match this codec")
        if bool(scene.flags & self.FLAG_CONTEXTUAL_SCORE) != (self.score_context is not None):
            raise ValueError("bitstream score context does not match this codec")
        parameter = self.shared_basis
        mean = torch.frombuffer(bytearray(scene.mean_fp16), dtype=torch.float16).to(
            device=parameter.device, dtype=parameter.dtype
        )
        mean = mean.reshape(1, scene.channels)
        if self.score_context is None:
            score_hat_nchw = self.score_entropy.decompress(
                [scene.score], (1, scene.points)
            )
        else:
            with torch.autocast(device_type=mean.device.type, enabled=False):
                score_hat_nchw = self.score_context.decompress(
                    scene.score, mean.float(), scene.points, self.score_entropy
                )
        score_hat = score_hat_nchw.squeeze(2).transpose(1, 2)
        score_hat = score_hat * self.score_scale[None, None, :]
        low_rank = self._synthesize_low_rank(score_hat)
        no_residual = bool(scene.flags & self.FLAG_NO_RESIDUAL)
        if no_residual:
            if scene.residual:
                raise ValueError("no-residual bitstream contains a residual payload")
            reconstruction = mean[:, None, :] + low_rank
        else:
            if not scene.residual:
                raise ValueError("residual codec received an empty residual payload")
            residual_hat_norm = self.residual_codec.decompress(
                scene.residual, output_width=scene.points
            )
            residual_hat_nchw = residual_hat_norm * self.residual_std + self.residual_mean
            residual_hat = residual_hat_nchw.squeeze(2).transpose(1, 2)
            reconstruction = mean[:, None, :] + low_rank + residual_hat
        if inverse_permutation is not None:
            from .morton import batched_gather

            reconstruction = batched_gather(reconstruction, inverse_permutation)
        return (
            reconstruction[..., : self.config.texture_channels],
            reconstruction[..., self.config.texture_channels :],
        )

    def update(self, force: bool = False, update_quantiles: bool = False) -> bool:
        if self.score_context is None:
            score_updated = self.score_entropy.update(
                force=force,
                update_quantiles=update_quantiles,
            )
        else:
            score_updated = self.score_context.update(
                self.score_entropy,
                force=force,
                update_quantiles=update_quantiles,
            )
        residual_updated = False
        if self.config.use_residual:
            residual_updated = self.residual_codec.update(
                force=force,
                update_quantiles=update_quantiles,
            )
        return bool(score_updated or residual_updated)

    def aux_loss(self) -> Tensor:
        if self.score_context is None:
            loss = self.score_entropy.entropy_bottleneck.loss()
        else:
            loss = self.score_context.aux_loss(self.score_entropy)
        if self.config.use_residual:
            loss = loss + self.residual_codec.entropy_bottleneck.loss()
        return loss

    def extra_repr(self) -> str:
        return (
            f"channels={self.config.observable_channels}, rank={self.config.rank}, "
            f"transform={self.config.transform}, "
            f"basis=untied, ordering={'morton' if self.config.use_morton else 'identity'}, "
            f"residual={'multiscale1d' if self.config.use_residual else 'off'}, "
            f"score_context={'on' if self.score_context is not None else 'off'}"
        )

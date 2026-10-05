"""Full-feature 1-D hyperprior, independent of the archived residual codec."""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN

from .bitstream import Hyper1DSceneBitstream, DualHyper1DSceneBitstream, ResidualBitstream
from .codec import CodecOutput, CompressedScene
from .hyper1d_config import Hyper1DConfig
from .morton import MortonOrder, batched_gather, morton_order_3d
from .residual import MultiScaleAdapter1D


def _conv(cin: int, cout: int, kernel: int, stride: int = 1):
    return nn.Conv2d(cin, cout, (1, kernel), (1, stride), (0, kernel // 2))


def _deconv(cin: int, cout: int, kernel: int, stride: int = 2):
    return nn.ConvTranspose2d(cin, cout, (1, kernel), (1, stride),
                              (0, kernel // 2), output_padding=(0, stride - 1))


class MeanScaleHyperprior1D(nn.Module):
    """One independently parameterized MSH on [B,C,1,P] features."""

    def __init__(self, config: Hyper1DConfig | None = None, *, input_channels: int | None = None):
        super().__init__()
        self.config = config or Hyper1DConfig()
        self.input_channels = input_channels or self.config.observable_channels
        self._init_transforms()

    def _init_transforms(self):
        c, n, m = self.input_channels, self.config.n, self.config.m
        s0, s1 = self.config.strides
        if self.config.architecture == "legacy":
            # Preserve module names and shapes for the existing trained pilots.
            self.analysis_adapter = MultiScaleAdapter1D(c, self.config.adapter_hidden)
            self.g_a = nn.Sequential(_conv(c, n, 5, s0), GDN(n), _conv(n, m, 5, s1))
            self.g_s = nn.Sequential(_deconv(m, n, 5, s1), GDN(n, inverse=True), _deconv(n, c, 5, s0))
        else:
            self.analysis_adapter = nn.Identity()
            self.g_a = nn.Sequential(
                _conv(c, n, 5, s0), GDN(n),
                _conv(n, n, 5), GDN(n),
                _conv(n, n, 5, s1), GDN(n),
                _conv(n, m, 5),
            )
            self.g_s = nn.Sequential(
                _conv(m, n, 5), GDN(n, inverse=True),
                _deconv(n, n, 5, s1), GDN(n, inverse=True),
                _conv(n, n, 5), GDN(n, inverse=True),
                _deconv(n, c, 5, s0),
            )
        self.h_a = nn.Sequential(_conv(m, n, 3), nn.LeakyReLU(), _conv(n, n, 5, 2),
                                 nn.LeakyReLU(), _conv(n, n, 5, 2))
        self.h_s = nn.Sequential(_deconv(n, m, 5), nn.LeakyReLU(), _deconv(m, 2 * m, 5))
        # Keep legacy registration order: optimizer state is indexed by parameter order.
        self.synthesis_adapter = (MultiScaleAdapter1D(c, self.config.adapter_hidden)
                                  if self.config.architecture == "legacy" else nn.Identity())
        self.entropy_bottleneck = EntropyBottleneck(n)
        self.gaussian_conditional = GaussianConditional(None)
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, Tensor] = {}

    def _parameters_from_z(self, z_hat: Tensor, y_shape):
        params = self.h_s(z_hat)[..., :y_shape[0], :y_shape[1]]
        return params.chunk(2, dim=1)

    def _decode_latent(self, y_hat: Tensor, points: int) -> Tensor:
        return self.synthesis_adapter(self.g_s(y_hat))[..., :points]

    @torch.no_grad()
    def _diagnostics(self, features, adapted, y, scales, means) -> None:
        self.last_diagnostics = {
            "feature_abs_mean": features.float().abs().mean(),
            "adapter_abs_mean": adapted.float().abs().mean(),
            "y_channel_std_mean": y.float().flatten(2).std(dim=2, unbiased=False).mean(),
            "scale_above_table_fraction": (scales.float() > 256).float().mean(),
            "centered_latent_abs_mean": (y.float() - means.float()).abs().mean(),
            "centered_latent_abs_max": (y.float() - means.float()).abs().max(),
        }

    def forward_features(self, features: Tensor, *, training: bool):
        adapted = self.analysis_adapter(features)
        y = self.g_a(adapted)
        z = self.h_a(y)  # Both MSH paths observe signed latents.
        with torch.autocast(device_type=y.device.type, enabled=False):
            z_hat, z_likelihood = self.entropy_bottleneck(z.float(), training=training)
        scales, means = self._parameters_from_z(z_hat, y.shape[-2:])
        with torch.autocast(device_type=y.device.type, enabled=False):
            y_hat, y_likelihood = self.gaussian_conditional(y.float(), scales.float(), means=means.float(), training=training)
        if self.capture_diagnostics:
            self._diagnostics(features, adapted, y, scales, means)
        return self._decode_latent(y_hat, features.shape[-1]), {"y": y_likelihood, "z": z_likelihood}

    @torch.no_grad()
    def compress_features(self, features: Tensor, *, reconstruct=False):
        # Called within the scene codec's FP32 context on both paths.
        adapted = self.analysis_adapter(features)
        y = self.g_a(adapted)
        z = self.h_a(y)
        z_strings = tuple(self.entropy_bottleneck.compress(z))
        z_hat = self.entropy_bottleneck.decompress(list(z_strings), z.shape[-2:])
        scales, means = self._parameters_from_z(z_hat, y.shape[-2:])
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = tuple(self.gaussian_conditional.compress(y, indexes, means=means))
        if self.capture_diagnostics:
            self._diagnostics(features, adapted, y, scales, means)
        payload = ResidualBitstream(tuple(z.shape[-2:]), tuple(y.shape[-2:]), z_strings, y_strings).pack()
        # The residual must use exactly the base reconstruction available to
        # the receiver, including quantization of both z and y.
        value = None
        if reconstruct:
            y_hat = self.gaussian_conditional.decompress(list(y_strings), indexes, means=means)
            value = self._decode_latent(y_hat, features.shape[-1])
        return payload, value

    def validate_payload(self, payload: bytes, points: int):
        packed = ResidualBitstream.unpack(payload)
        expected_y, expected_z = self.config.latent_shapes(points)
        if packed.y_shape != expected_y or packed.z_shape != expected_z:
            raise ValueError("Hyper1D latent shapes disagree with points/strides")
        return packed

    @torch.no_grad()
    def decompress_features(self, payload: bytes, points: int):
        packed = self.validate_payload(payload, points)
        z_hat = self.entropy_bottleneck.decompress(list(packed.z_strings), packed.z_shape)
        scales, means = self._parameters_from_z(z_hat, packed.y_shape)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(list(packed.y_strings), indexes, means=means)
        return self._decode_latent(y_hat, points)

    def aux_loss(self):
        return self.entropy_bottleneck.loss()

    def update(self, force=False, update_quantiles=False):
        device = self.entropy_bottleneck.quantiles.device
        table = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64, device=device))
        gaussian_updated = self.gaussian_conditional.update_scale_table(table, force=force)
        entropy_updated = self.entropy_bottleneck.update(force=force, update_quantiles=update_quantiles)
        return bool(gaussian_updated or entropy_updated)


class FeatureHyperprior1DCodec(MeanScaleHyperprior1D):
    """Single MSH or an optional low-rank base MSH plus full-width residual MSH."""

    def __init__(self, config: Hyper1DConfig | None = None):
        # Keep the old one-path parameter names AND registration order so full
        # optimizer resumes remain compatible with existing checkpoints.
        nn.Module.__init__(self)
        self.config = config or Hyper1DConfig()
        self.geometry_projection = nn.Linear(self.config.geometry_channels,
                                             self.config.geometry_observable_channels, bias=False)
        c = self.config.observable_channels
        self.register_buffer("f_mean", torch.zeros(1, c, 1, 1))
        self.register_buffer("f_std", torch.ones(1, c, 1, 1))
        self.input_channels = self.config.base_rank or c
        self._init_transforms()
        if self.config.base_rank:
            self.base_analysis = nn.Linear(c, self.config.base_rank, bias=False)
            self.base_synthesis = nn.Linear(self.config.base_rank, c, bias=False)
            # Start with a tied orthogonal projector; both matrices then train freely.
            nn.init.orthogonal_(self.base_analysis.weight)
            with torch.no_grad():
                self.base_synthesis.weight.copy_(self.base_analysis.weight.T)
        if self.config.paths == 2:
            self.residual_msh = MeanScaleHyperprior1D(self.config, input_channels=c)

    def _base_input(self, features: Tensor) -> Tensor:
        if self.config.base_rank:
            return self.base_analysis(features.movedim(1, -1)).movedim(-1, 1)
        return features

    def _base_output(self, scores: Tensor) -> Tensor:
        if self.config.base_rank:
            return self.base_synthesis(scores.movedim(1, -1)).movedim(-1, 1)
        return scores

    def _stream_flags(self) -> int:
        return int(self.config.use_morton) | (2 if self.config.base_rank else 0)

    def validate_normalization(self) -> None:
        if not torch.isfinite(self.f_mean).all() or not torch.isfinite(self.f_std).all() or (self.f_std <= 0).any():
            raise ValueError("normalization buffers must be finite with positive f_std")
        if self.config.input_norm == "none" and (torch.count_nonzero(self.f_mean) or not torch.equal(self.f_std, torch.ones_like(self.f_std))):
            raise ValueError("input_norm=none requires identity normalization buffers")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self.validate_normalization()

    def project_geometry(self, geometry: Tensor) -> Tensor:
        if geometry.ndim != 3 or geometry.shape[-1] != self.config.geometry_channels:
            raise ValueError("geometry must be [B,P,geometry_channels]")
        return self.geometry_projection(geometry)

    def _input(self, texture: Tensor, geometry: Tensor, positions: Tensor | None):
        if texture.ndim != 3 or texture.shape[-1] != self.config.texture_channels:
            raise ValueError("texture must be [B,P,texture_channels]")
        if texture.shape[:2] != geometry.shape[:2] or min(texture.shape[:2]) <= 0:
            raise ValueError("texture/geometry require matching nonempty [B,P]")
        value = torch.cat((texture, self.project_geometry(geometry)), dim=-1)
        if self.config.use_morton:
            if positions is None or positions.shape != (*texture.shape[:2], 3):
                raise ValueError("Morton ordering requires positions [B,P,3]")
            order = morton_order_3d(positions, self.config.morton_bits)
            value = order.apply(value)
        else:
            indices = torch.arange(value.shape[1], device=value.device).expand(value.shape[0], -1)
            order = MortonOrder(indices, indices, torch.zeros_like(indices))
        value = value.transpose(1, 2).unsqueeze(2)
        return (value - self.f_mean) / self.f_std, order

    def _restore_features(self, value: Tensor) -> Tensor:
        value = value * self.f_std + self.f_mean
        return value.squeeze(2).transpose(1, 2)

    def _split(self, value: Tensor):
        return value.split((self.config.texture_channels, self.config.geometry_observable_channels), dim=-1)

    def _residual_diagnostics(self):
        if self.capture_diagnostics:
            self.last_diagnostics.update({f"residual_{key}": value
                                          for key, value in self.residual_msh.last_diagnostics.items()})

    def forward(self, texture, geometry, positions=None, *, restore_original_order=True, training=None):
        features, order = self._input(texture, geometry, positions)
        training = self.training if training is None else training
        value, likelihoods = self.forward_features(self._base_input(features), training=training)
        value = self._base_output(value)
        if self.config.paths == 2:
            self.residual_msh.capture_diagnostics = self.capture_diagnostics
            residual, residual_likelihoods = self.residual_msh.forward_features(features - value, training=training)
            value = value + residual
            likelihoods = {**{f"base_{k}": v for k, v in likelihoods.items()},
                           **{f"residual_{k}": v for k, v in residual_likelihoods.items()}}
            self._residual_diagnostics()
        reconstruction = self._restore_features(value)
        decoded = order.restore(reconstruction) if restore_original_order else reconstruction
        tex, geo = self._split(decoded)
        return CodecOutput(tex, geo, reconstruction, likelihoods, order)

    @torch.no_grad()
    def compress(self, texture, geometry, positions=None) -> CompressedScene:
        if texture.shape[0] != 1:
            raise ValueError("Hyper1D compression requires batch size 1")
        # Sender and receiver use the same FP32 transforms even under outer AMP.
        with torch.autocast(device_type=texture.device.type, enabled=False):
            features, order = self._input(texture.float(), geometry.float(), positions)
            payload, base = self.compress_features(self._base_input(features.float()),
                                                   reconstruct=self.config.paths == 2)
            if self.config.paths == 2:
                base = self._base_output(base)
                self.residual_msh.capture_diagnostics = self.capture_diagnostics
                residual_payload, _ = self.residual_msh.compress_features(features - base)
                self._residual_diagnostics()
                scene = DualHyper1DSceneBitstream(texture.shape[1], self.config.observable_channels,
                    payload, residual_payload, flags=self._stream_flags())
            else:
                scene = Hyper1DSceneBitstream(texture.shape[1], self.config.observable_channels,
                    payload, flags=self._stream_flags())
        data = scene.pack()
        return CompressedScene(data, order)

    @torch.no_grad()
    def decompress(self, data: bytes, *, inverse_permutation: Tensor | None = None):
        container = DualHyper1DSceneBitstream if self.config.paths == 2 else Hyper1DSceneBitstream
        scene = container.unpack(data)
        if scene.channels != self.config.observable_channels or scene.flags != self._stream_flags():
            raise ValueError("Hyper1D stream configuration mismatch")
        # Validate every payload before either entropy decoder is entered.
        payload = scene.base_payload if self.config.paths == 2 else scene.payload
        self.validate_payload(payload, scene.points)
        if self.config.paths == 2:
            self.residual_msh.validate_payload(scene.residual_payload, scene.points)
        with torch.autocast(device_type=self.f_mean.device.type, enabled=False):
            value = self._base_output(self.decompress_features(payload, scene.points))
            if self.config.paths == 2:
                value = value + self.residual_msh.decompress_features(scene.residual_payload, scene.points)
            value = self._restore_features(value)
        if inverse_permutation is not None:
            value = batched_gather(value, inverse_permutation)
        return self._split(value)

    def aux_loss(self):
        loss = super().aux_loss()
        if self.config.paths == 2:
            loss = loss + self.residual_msh.aux_loss()
        return loss

    def update(self, force=False, update_quantiles=False):
        updated = super().update(force=force, update_quantiles=update_quantiles)
        if self.config.paths == 2:
            residual_updated = self.residual_msh.update(force=force, update_quantiles=update_quantiles)
            updated = updated or residual_updated
        return bool(updated)

    def set_trainable_scope(self, scope: str = "all"):
        if scope != "all":
            raise ValueError("Hyper1D supports only trainable scope 'all'")
        for parameter in self.parameters():
            parameter.requires_grad_(True)

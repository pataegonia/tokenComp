"""Full-feature 1-D hyperprior, independent of the archived residual codec."""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN

from .bitstream import Hyper1DSceneBitstream, ResidualBitstream
from .codec import CodecOutput, CompressedScene
from .hyper1d_config import Hyper1DConfig
from .morton import MortonOrder, batched_gather, morton_order_3d
from .residual import MultiScaleAdapter1D


def _conv(cin: int, cout: int, kernel: int, stride: int = 1):
    return nn.Conv2d(cin, cout, (1, kernel), (1, stride), (0, kernel // 2))


def _deconv(cin: int, cout: int, kernel: int, stride: int = 2):
    return nn.ConvTranspose2d(cin, cout, (1, kernel), (1, stride),
                              (0, kernel // 2), output_padding=(0, stride - 1))


class FeatureHyperprior1DCodec(nn.Module):
    def __init__(self, config: Hyper1DConfig | None = None):
        super().__init__()
        self.config = config or Hyper1DConfig()
        c, n, m = self.config.observable_channels, self.config.n, self.config.m
        s0, s1 = self.config.strides
        self.geometry_projection = nn.Linear(self.config.geometry_channels,
                                             self.config.geometry_observable_channels, bias=False)
        self.register_buffer("f_mean", torch.zeros(1, c, 1, 1))
        self.register_buffer("f_std", torch.ones(1, c, 1, 1))
        self.analysis_adapter = MultiScaleAdapter1D(c, self.config.adapter_hidden)
        self.g_a = nn.Sequential(_conv(c, n, 5, s0), GDN(n), _conv(n, m, 5, s1))
        self.g_s = nn.Sequential(_deconv(m, n, 5, s1), GDN(n, inverse=True), _deconv(n, c, 5, s0))
        self.h_a = nn.Sequential(_conv(m, n, 3), nn.LeakyReLU(), _conv(n, n, 5, 2),
                                 nn.LeakyReLU(), _conv(n, n, 5, 2))
        self.h_s = nn.Sequential(_deconv(n, m, 5), nn.LeakyReLU(), _deconv(m, 2 * m, 5))
        self.synthesis_adapter = MultiScaleAdapter1D(c, self.config.adapter_hidden)
        self.entropy_bottleneck = EntropyBottleneck(n)
        self.gaussian_conditional = GaussianConditional(None)
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, Tensor] = {}

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

    def _parameters_from_z(self, z_hat: Tensor, y_shape):
        params = self.h_s(z_hat)[..., :y_shape[0], :y_shape[1]]
        return params.chunk(2, dim=1)

    def _synthesis(self, y_hat: Tensor, points: int) -> Tensor:
        value = self.synthesis_adapter(self.g_s(y_hat))[..., :points]
        value = value * self.f_std + self.f_mean
        return value.squeeze(2).transpose(1, 2)

    def _split(self, value: Tensor):
        return value.split((self.config.texture_channels, self.config.geometry_observable_channels), dim=-1)

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

    def forward(self, texture, geometry, positions=None, *, restore_original_order=True, training=None):
        features, order = self._input(texture, geometry, positions)
        adapted = self.analysis_adapter(features)
        y = self.g_a(adapted)
        z = self.h_a(y)  # Mean/scale prior observes signed latents.
        training = self.training if training is None else training
        with torch.autocast(device_type=y.device.type, enabled=False):
            z_hat, z_likelihood = self.entropy_bottleneck(z.float(), training=training)
        scales, means = self._parameters_from_z(z_hat, y.shape[-2:])
        with torch.autocast(device_type=y.device.type, enabled=False):
            y_hat, y_likelihood = self.gaussian_conditional(y.float(), scales.float(), means=means.float(), training=training)
        if self.capture_diagnostics:
            self._diagnostics(features, adapted, y, scales, means)
        reconstruction = self._synthesis(y_hat, texture.shape[1])
        decoded = order.restore(reconstruction) if restore_original_order else reconstruction
        tex, geo = self._split(decoded)
        return CodecOutput(tex, geo, reconstruction, {"y": y_likelihood, "z": z_likelihood}, order)

    @torch.no_grad()
    def compress(self, texture, geometry, positions=None) -> CompressedScene:
        if texture.shape[0] != 1:
            raise ValueError("Hyper1D compression requires batch size 1")
        # Sender and receiver use the same FP32 transforms even under outer AMP.
        with torch.autocast(device_type=texture.device.type, enabled=False):
            features, order = self._input(texture.float(), geometry.float(), positions)
            adapted = self.analysis_adapter(features.float())
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
        data = Hyper1DSceneBitstream(texture.shape[1], self.config.observable_channels, payload,
                                    flags=int(self.config.use_morton)).pack()
        return CompressedScene(data, order)

    @torch.no_grad()
    def decompress(self, data: bytes, *, inverse_permutation: Tensor | None = None):
        scene = Hyper1DSceneBitstream.unpack(data)
        if scene.channels != self.config.observable_channels or scene.flags != int(self.config.use_morton):
            raise ValueError("Hyper1D stream configuration mismatch")
        packed = ResidualBitstream.unpack(scene.payload)
        expected_y, expected_z = self.config.latent_shapes(scene.points)
        if packed.y_shape != expected_y or packed.z_shape != expected_z:
            raise ValueError("Hyper1D latent shapes disagree with points/strides")
        with torch.autocast(device_type=self.f_mean.device.type, enabled=False):
            z_hat = self.entropy_bottleneck.decompress(list(packed.z_strings), packed.z_shape)
            scales, means = self._parameters_from_z(z_hat, packed.y_shape)
            indexes = self.gaussian_conditional.build_indexes(scales)
            y_hat = self.gaussian_conditional.decompress(list(packed.y_strings), indexes, means=means)
            value = self._synthesis(y_hat, scene.points)
        if inverse_permutation is not None:
            value = batched_gather(value, inverse_permutation)
        return self._split(value)

    def aux_loss(self):
        return self.entropy_bottleneck.loss()

    def update(self, force=False, update_quantiles=False):
        table = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64, device=self.f_mean.device))
        gaussian_updated = self.gaussian_conditional.update_scale_table(table, force=force)
        entropy_updated = self.entropy_bottleneck.update(force=force, update_quantiles=update_quantiles)
        return bool(gaussian_updated or entropy_updated)

    def set_trainable_scope(self, scope: str = "all"):
        if scope != "all":
            raise ValueError("Hyper1D supports only trainable scope 'all'")
        for parameter in self.parameters():
            parameter.requires_grad_(True)

"""Checkpoint-compatible multiscale 1-D mean/scale hyperprior."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN

from .bitstream import ResidualBitstream


def _conv1d_as_2d(
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(1, kernel),
        stride=(1, stride),
        padding=(0, kernel // 2),
    )


def _deconv1d_as_2d(
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int = 2,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=(1, kernel),
        stride=(1, stride),
        padding=(0, kernel // 2),
        output_padding=(0, stride - 1),
    )


class MultiScaleAdapter1D(nn.Module):
    """Three receptive-field residual branches used on both codec sides."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        branch = hidden // 3
        if branch * 3 != hidden:
            raise ValueError("hidden must be divisible by three")
        self.in_projection = nn.Conv2d(channels, hidden, kernel_size=1)
        self.branches = nn.ModuleList(
            nn.Conv2d(
                hidden,
                branch,
                kernel_size=(1, 5),
                padding=(0, 2 * dilation),
                dilation=(1, dilation),
            )
            for dilation in (1, 2, 4)
        )
        self.out_projection = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, value: Tensor) -> Tensor:
        hidden = F.gelu(self.in_projection(value))
        mixed = torch.cat([F.gelu(branch(hidden)) for branch in self.branches], dim=1)
        return value + self.out_projection(mixed)


@dataclass(slots=True)
class ResidualOutput:
    reconstruction: Tensor
    likelihoods: dict[str, Tensor]
    y_hat: Tensor
    z_hat: Tensor


class ResidualHyperprior1D(nn.Module):
    """Two-level 1-D hyperprior recovered from checkpoint tensor shapes."""

    def __init__(self, channels: int, n: int = 192, m: int = 320, adapter_hidden: int = 96) -> None:
        super().__init__()
        self.channels = channels
        self.n = n
        self.m = m

        self.g_a = nn.Sequential(
            _conv1d_as_2d(channels, n, 5, 2),
            GDN(n),
            _conv1d_as_2d(n, m, 5, 2),
        )
        self.g_s = nn.Sequential(
            _deconv1d_as_2d(m, n, 5, 2),
            GDN(n, inverse=True),
            _deconv1d_as_2d(n, channels, 5, 2),
        )
        self.h_a = nn.Sequential(
            _conv1d_as_2d(m, n, 3, 1),
            nn.LeakyReLU(inplace=True),
            _conv1d_as_2d(n, n, 5, 2),
            nn.LeakyReLU(inplace=True),
            _conv1d_as_2d(n, n, 5, 2),
        )
        self.h_s = nn.Sequential(
            _deconv1d_as_2d(n, m, 5, 2),
            nn.LeakyReLU(inplace=True),
            _deconv1d_as_2d(m, 2 * m, 5, 2),
        )
        self.entropy_bottleneck = EntropyBottleneck(n)
        self.gaussian_conditional = GaussianConditional(None)
        self.analysis_adapter = MultiScaleAdapter1D(channels, adapter_hidden)
        self.synthesis_adapter = MultiScaleAdapter1D(channels, adapter_hidden)

    def _parameters_from_z(self, z_hat: Tensor, y_shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        gaussian = self.h_s(z_hat)
        if gaussian.shape[-2:] != y_shape:
            gaussian = gaussian[..., : y_shape[0], : y_shape[1]]
        scales_hat, means_hat = gaussian.chunk(2, dim=1)
        return scales_hat, means_hat

    def forward(self, residual: Tensor, *, training: bool | None = None) -> ResidualOutput:
        if residual.ndim != 4 or residual.shape[1] != self.channels:
            raise ValueError(f"residual must have shape [B,{self.channels},1,N]")
        if training is None:
            training = self.training
        y = self.g_a(self.analysis_adapter(residual))
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z, training=training)
        scales_hat, means_hat = self._parameters_from_z(z_hat, y.shape[-2:])
        y_hat, y_likelihoods = self.gaussian_conditional(
            y, scales_hat, means=means_hat, training=training
        )
        reconstruction = self.synthesis_adapter(self.g_s(y_hat))
        reconstruction = reconstruction[..., : residual.shape[-2], : residual.shape[-1]]
        return ResidualOutput(
            reconstruction,
            {"residual_y": y_likelihoods, "residual_z": z_likelihoods},
            y_hat,
            z_hat,
        )

    @torch.no_grad()
    def compress(self, residual: Tensor) -> tuple[bytes, Tensor]:
        if residual.shape[0] < 1:
            raise ValueError("cannot compress an empty batch")
        y = self.g_a(self.analysis_adapter(residual))
        z = self.h_a(torch.abs(y))
        z_strings = tuple(self.entropy_bottleneck.compress(z))
        z_hat = self.entropy_bottleneck.decompress(list(z_strings), z.shape[-2:])
        scales_hat, means_hat = self._parameters_from_z(z_hat, y.shape[-2:])
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = tuple(self.gaussian_conditional.compress(y, indexes, means=means_hat))
        y_hat = self.gaussian_conditional.decompress(
            list(y_strings), indexes, means=means_hat
        )
        reconstruction = self.synthesis_adapter(self.g_s(y_hat))
        reconstruction = reconstruction[..., : residual.shape[-2], : residual.shape[-1]]
        payload = ResidualBitstream(
            tuple(z.shape[-2:]), tuple(y.shape[-2:]), z_strings, y_strings
        ).pack()
        return payload, reconstruction

    @torch.no_grad()
    def decompress(self, payload: bytes, *, output_width: int | None = None) -> Tensor:
        packed = ResidualBitstream.unpack(payload)
        z_hat = self.entropy_bottleneck.decompress(list(packed.z_strings), packed.z_shape)
        scales_hat, means_hat = self._parameters_from_z(z_hat, packed.y_shape)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            list(packed.y_strings), indexes, means=means_hat
        )
        reconstruction = self.synthesis_adapter(self.g_s(y_hat))
        if output_width is not None:
            reconstruction = reconstruction[..., :output_width]
        return reconstruction

    def update(self, force: bool = False, update_quantiles: bool = False) -> bool:
        scale_table = self.gaussian_conditional.scale_table
        if not scale_table.numel():
            scale_table = torch.exp(
                torch.linspace(
                    math.log(0.11),
                    math.log(256.0),
                    64,
                    device=self.entropy_bottleneck.quantiles.device,
                )
            )
        gaussian_updated = self.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        entropy_updated = self.entropy_bottleneck.update(
            force=force,
            update_quantiles=update_quantiles,
        )
        return bool(gaussian_updated or entropy_updated)

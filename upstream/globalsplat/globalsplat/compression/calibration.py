"""Numerically stable channel statistics for fixed TRAIN scenes."""

import torch


class ChannelMoments:
    def __init__(self, channels: int):
        self.count = 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)

    @torch.no_grad()
    def update(self, values):
        if values.ndim < 2 or values.shape[-1] != self.mean.numel():
            raise ValueError("calibration features have the wrong channel count")
        flat = values.detach().reshape(-1, self.mean.numel()).to(device="cpu", dtype=torch.float64)
        if flat.shape[0] == 0 or not torch.isfinite(flat).all():
            raise ValueError("calibration requires finite, nonempty features")
        n = flat.shape[0]
        mean = flat.mean(0)
        m2 = (flat - mean).square().sum(0)
        delta = mean - self.mean
        total = self.count + n
        self.m2 += m2 + delta.square() * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def buffers(self):
        if self.count == 0:
            raise ValueError("no calibration samples")
        mean = self.mean.float().view(1, -1, 1, 1)
        std = (self.m2 / self.count).clamp_min(0).sqrt().clamp_min(1e-6).float().view(1, -1, 1, 1)
        return mean, std

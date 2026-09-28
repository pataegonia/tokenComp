"""GPU-efficient decoder-causal context model for low-rank score symbols."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from compressai.entropy_models import GaussianConditional

from .bitstream import ScoreContextBitstream
from .entropy import FactorizedScoreEntropy


class ContextualScoreEntropy(nn.Module):
    """Condition score coding on decoded scene, channel, and Morton context.

    Channel groups are decoded serially, but every token within a group is
    processed in parallel.  Optional spatial context uses a two-pass
    even/odd split, avoiding a 4096-step autoregressive decoder.
    """

    FLAG_MEAN = 1 << 0
    FLAG_CHANNEL = 1 << 1
    FLAG_SPATIAL = 1 << 2
    FLAG_SPATIAL_RESIDUAL3 = 1 << 3
    FLAG_SPATIAL_RESIDUAL7 = 1 << 4
    FLAG_SPATIAL_ENTROPY_SPLIT = 1 << 5
    FLAG_SPATIAL_ENTROPY_GAUSSIAN = 1 << 6
    FLAG_SPATIAL_ENTROPY_CONDITIONAL_SCALE = 1 << 7
    KNOWN_FLAGS = (
        FLAG_MEAN
        | FLAG_CHANNEL
        | FLAG_SPATIAL
        | FLAG_SPATIAL_RESIDUAL3
        | FLAG_SPATIAL_RESIDUAL7
        | FLAG_SPATIAL_ENTROPY_SPLIT
        | FLAG_SPATIAL_ENTROPY_GAUSSIAN
        | FLAG_SPATIAL_ENTROPY_CONDITIONAL_SCALE
    )

    def __init__(
        self,
        *,
        rank: int,
        scene_channels: int,
        mean_condition: bool,
        channel_context: bool,
        spatial_context: bool,
        spatial_predictor: str,
        spatial_entropy: str,
        spatial_hidden: int,
        slice_channels: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.scene_channels = int(scene_channels)
        self.mean_condition = bool(mean_condition)
        self.channel_context = bool(channel_context)
        self.spatial_context = bool(spatial_context)
        self.spatial_predictor = str(spatial_predictor)
        self.spatial_entropy = str(spatial_entropy)
        self.spatial_hidden = int(spatial_hidden)
        self.slice_channels = int(slice_channels)

        if self.channel_context:
            self.group_sizes = tuple(
                min(self.slice_channels, self.rank - start)
                for start in range(0, self.rank, self.slice_channels)
            )
            self.group_entropies = nn.ModuleList(
                FactorizedScoreEntropy(width) for width in self.group_sizes
            )
        else:
            self.group_sizes = (self.rank,)
            self.group_entropies = nn.ModuleList()

        self.mean_conditioner: nn.Module | None = None
        if self.mean_condition:
            self.mean_conditioner = nn.Sequential(
                nn.LayerNorm(self.scene_channels),
                nn.Linear(self.scene_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2 * self.rank),
            )
            nn.init.zeros_(self.mean_conditioner[-1].weight)
            nn.init.zeros_(self.mean_conditioner[-1].bias)

        self.channel_predictors = nn.ModuleList()
        if self.channel_context:
            decoded = self.group_sizes[0]
            for width in self.group_sizes[1:]:
                predictor = nn.Conv2d(decoded, width, 1)
                nn.init.zeros_(predictor.weight)
                nn.init.zeros_(predictor.bias)
                self.channel_predictors.append(predictor)
                decoded += width

        self.spatial_predictors = nn.ModuleList()
        self.spatial_corrections = nn.ModuleList()
        self.spatial_odd_entropies = nn.ModuleList()
        self.spatial_log_scales = nn.ParameterList()
        self.spatial_scale_predictors = nn.ModuleList()
        self.spatial_gaussian: GaussianConditional | None = None
        if self.spatial_entropy in ("gaussian", "conditional_scale"):
            self.spatial_gaussian = GaussianConditional(None)
        if self.spatial_context:
            for width in self.group_sizes:
                predictor = nn.Conv2d(width, width, kernel_size=(1, 3), padding=(0, 1))
                nn.init.zeros_(predictor.weight)
                nn.init.zeros_(predictor.bias)
                self.spatial_predictors.append(predictor)
                if self.spatial_predictor != "linear":
                    kernel = 3 if self.spatial_predictor == "residual3" else 7
                    correction = nn.Sequential(
                        nn.Conv2d(
                            width,
                            self.spatial_hidden,
                            kernel_size=(1, kernel),
                            padding=(0, kernel // 2),
                        ),
                        nn.GELU(),
                        nn.Conv2d(self.spatial_hidden, width, kernel_size=1),
                    )
                    nn.init.zeros_(correction[-1].weight)
                    nn.init.zeros_(correction[-1].bias)
                    self.spatial_corrections.append(correction)
                if self.spatial_entropy == "split":
                    self.spatial_odd_entropies.append(FactorizedScoreEntropy(width))
                elif self.spatial_entropy in ("gaussian", "conditional_scale"):
                    # softplus(raw) + 0.11 starts at a unit scale.  Gaussian and
                    # conditional_scale therefore share the same zero-context
                    # probability model at initialization.
                    initial = math.log(math.expm1(1.0 - 0.11))
                    self.spatial_log_scales.append(
                        nn.Parameter(torch.full((width,), initial))
                    )
                    if self.spatial_entropy == "conditional_scale":
                        kernel = 3 if self.spatial_predictor == "linear" else (
                            3 if self.spatial_predictor == "residual3" else 7
                        )
                        scale_predictor = nn.Sequential(
                            nn.Conv2d(
                                width,
                                self.spatial_hidden,
                                kernel_size=(1, kernel),
                                padding=(0, kernel // 2),
                            ),
                            nn.GELU(),
                            nn.Conv2d(self.spatial_hidden, width, kernel_size=1),
                        )
                        nn.init.zeros_(scale_predictor[-1].weight)
                        nn.init.zeros_(scale_predictor[-1].bias)
                        self.spatial_scale_predictors.append(scale_predictor)

    @property
    def flags(self) -> int:
        flags = 0
        if self.mean_condition:
            flags |= self.FLAG_MEAN
        if self.channel_context:
            flags |= self.FLAG_CHANNEL
        if self.spatial_context:
            flags |= self.FLAG_SPATIAL
        if self.spatial_predictor == "residual3":
            flags |= self.FLAG_SPATIAL_RESIDUAL3
        elif self.spatial_predictor == "residual7":
            flags |= self.FLAG_SPATIAL_RESIDUAL7
        if self.spatial_entropy == "split":
            flags |= self.FLAG_SPATIAL_ENTROPY_SPLIT
        elif self.spatial_entropy == "gaussian":
            flags |= self.FLAG_SPATIAL_ENTROPY_GAUSSIAN
        elif self.spatial_entropy == "conditional_scale":
            flags |= self.FLAG_SPATIAL_ENTROPY_CONDITIONAL_SCALE
        return flags

    def _entropy(
        self, base_entropy: FactorizedScoreEntropy, group_index: int
    ) -> FactorizedScoreEntropy:
        if self.channel_context:
            return self.group_entropies[group_index]
        return base_entropy

    def active_entropies(
        self, base_entropy: FactorizedScoreEntropy
    ) -> Iterable[FactorizedScoreEntropy]:
        even = tuple(self.group_entropies) if self.channel_context else (base_entropy,)
        if self.spatial_entropy == "split":
            return even + tuple(self.spatial_odd_entropies)
        return even

    def _scene_parameters(self, scene_mean: Tensor) -> tuple[Tensor, Tensor]:
        batch = scene_mean.shape[0]
        if self.mean_conditioner is None:
            offset = scene_mean.new_zeros(batch, self.rank)
            step = scene_mean.new_ones(batch, self.rank)
            return offset, step
        parameters = self.mean_conditioner(scene_mean)
        offset, log_step = parameters.chunk(2, dim=-1)
        # A bounded range prevents early training from creating nearly-zero
        # quantization steps or exploding residual symbols.
        step = torch.exp(2.0 * torch.tanh(0.5 * log_step))
        return offset, step

    def _group_base(
        self,
        decoded_groups: list[Tensor],
        offset: Tensor,
        start: int,
        width: int,
        points: int,
        group_index: int,
    ) -> Tensor:
        base = offset[:, start : start + width, None, None].expand(-1, -1, 1, points)
        if group_index > 0 and self.channel_context:
            previous = torch.cat(decoded_groups, dim=1)
            base = base + self.channel_predictors[group_index - 1](previous)
        return base

    @staticmethod
    def _merge_even_odd(even: Tensor, odd: Tensor, points: int) -> Tensor:
        merged = even.new_empty(even.shape[0], even.shape[1], 1, points)
        merged[..., 0::2] = even
        merged[..., 1::2] = odd
        return merged

    @staticmethod
    def _spatial_anchor_delta(anchor_hat: Tensor, base: Tensor) -> Tensor:
        anchor_delta = torch.zeros_like(base)
        anchor_delta[..., 0::2] = anchor_hat - base[..., 0::2]
        return anchor_delta

    def _spatial_prediction(
        self, group_index: int, anchor_hat: Tensor, base: Tensor
    ) -> Tensor:
        anchor_delta = self._spatial_anchor_delta(anchor_hat, base)
        prediction = self.spatial_predictors[group_index](anchor_delta)
        if self.spatial_corrections:
            prediction = prediction + self.spatial_corrections[group_index](anchor_delta)
        return prediction[..., 1::2]

    def _spatial_probability_scale(
        self, group_index: int, anchor_hat: Tensor, base: Tensor
    ) -> Tensor:
        if self.spatial_entropy not in ("gaussian", "conditional_scale"):
            raise RuntimeError("spatial Gaussian scale requested for a non-Gaussian model")
        base_scale = F.softplus(self.spatial_log_scales[group_index]) + 0.11
        base_scale = base_scale[None, :, None, None]
        if self.spatial_entropy == "gaussian":
            return base_scale.expand(
                anchor_hat.shape[0], -1, 1, base.shape[-1] // 2
            )
        anchor_delta = self._spatial_anchor_delta(anchor_hat, base)
        log_multiplier = self.spatial_scale_predictors[group_index](anchor_delta)
        # Bound the local correction while allowing a useful 1/7.4x..7.4x
        # range around each learned per-channel base scale.
        scale = base_scale * torch.exp(2.0 * torch.tanh(0.5 * log_multiplier))
        return scale[..., 1::2].clamp(min=0.11, max=256.0)

    @staticmethod
    def _entropy_medians(entropy: FactorizedScoreEntropy) -> Tensor:
        medians = entropy.entropy_bottleneck._get_medians()
        # EntropyBottleneck stores medians as [C, 1, 1], whereas
        # GaussianConditional expects its optional means to include the batch
        # dimension ([B, C, H, W], with spatial broadcasting allowed).
        if medians.ndim == 3:
            medians = medians.unsqueeze(0)
        return medians

    def _odd_forward(
        self,
        symbols: Tensor,
        even_entropy: FactorizedScoreEntropy,
        group_index: int,
        anchor_hat: Tensor,
        base: Tensor,
        *,
        training: bool | None,
    ) -> tuple[Tensor, Tensor]:
        if self.spatial_entropy == "shared":
            return even_entropy(symbols, training=training)
        if self.spatial_entropy == "split":
            return self.spatial_odd_entropies[group_index](symbols, training=training)
        assert self.spatial_gaussian is not None
        scales = self._spatial_probability_scale(group_index, anchor_hat, base)
        medians = self._entropy_medians(even_entropy)
        return self.spatial_gaussian(symbols, scales, means=medians, training=training)

    @torch.no_grad()
    def _odd_compress(
        self,
        symbols: Tensor,
        even_entropy: FactorizedScoreEntropy,
        group_index: int,
        anchor_hat: Tensor,
        base: Tensor,
    ) -> tuple[bytes, Tensor]:
        if self.spatial_entropy == "shared":
            strings = even_entropy.compress(symbols)
            return strings[0], even_entropy.decompress(strings, symbols.shape[-2:])
        if self.spatial_entropy == "split":
            entropy = self.spatial_odd_entropies[group_index]
            strings = entropy.compress(symbols)
            return strings[0], entropy.decompress(strings, symbols.shape[-2:])
        assert self.spatial_gaussian is not None
        scales = self._spatial_probability_scale(group_index, anchor_hat, base)
        medians = self._entropy_medians(even_entropy)
        indexes = self.spatial_gaussian.build_indexes(scales)
        strings = self.spatial_gaussian.compress(symbols, indexes, means=medians)
        return strings[0], self.spatial_gaussian.decompress(
            strings, indexes, means=medians
        )

    @torch.no_grad()
    def _odd_decompress(
        self,
        string: bytes,
        shape: tuple[int, int],
        even_entropy: FactorizedScoreEntropy,
        group_index: int,
        anchor_hat: Tensor,
        base: Tensor,
    ) -> Tensor:
        if self.spatial_entropy == "shared":
            return even_entropy.decompress([string], shape)
        if self.spatial_entropy == "split":
            return self.spatial_odd_entropies[group_index].decompress([string], shape)
        assert self.spatial_gaussian is not None
        scales = self._spatial_probability_scale(group_index, anchor_hat, base)
        medians = self._entropy_medians(even_entropy)
        indexes = self.spatial_gaussian.build_indexes(scales)
        return self.spatial_gaussian.decompress([string], indexes, means=medians)

    def forward(
        self,
        scores: Tensor,
        scene_mean: Tensor,
        base_entropy: FactorizedScoreEntropy,
        *,
        training: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        if scores.ndim != 4 or scores.shape[1] != self.rank or scores.shape[2] != 1:
            raise ValueError(f"scores must have shape [B,{self.rank},1,N]")
        offset, step = self._scene_parameters(scene_mean)
        points = scores.shape[-1]
        decoded: list[Tensor] = []
        likelihoods: list[Tensor] = []
        start = 0
        for group_index, width in enumerate(self.group_sizes):
            target = scores[:, start : start + width]
            base = self._group_base(decoded, offset, start, width, points, group_index)
            group_step = step[:, start : start + width, None, None]
            entropy = self._entropy(base_entropy, group_index)
            if self.spatial_context:
                even_symbols = (target[..., 0::2] - base[..., 0::2]) / group_step
                even_hat, even_likelihood = entropy(even_symbols, training=training)
                even_hat = base[..., 0::2] + group_step * even_hat
                odd_base = base[..., 1::2] + self._spatial_prediction(
                    group_index, even_hat, base
                )
                odd_symbols = (target[..., 1::2] - odd_base) / group_step
                odd_hat, odd_likelihood = self._odd_forward(
                    odd_symbols,
                    entropy,
                    group_index,
                    even_hat,
                    base,
                    training=training,
                )
                odd_hat = odd_base + group_step * odd_hat
                decoded.append(self._merge_even_odd(even_hat, odd_hat, points))
                likelihoods.append(
                    self._merge_even_odd(even_likelihood, odd_likelihood, points)
                )
            else:
                symbols = (target - base) / group_step
                symbols_hat, group_likelihood = entropy(symbols, training=training)
                decoded.append(base + group_step * symbols_hat)
                likelihoods.append(group_likelihood)
            start += width
        return torch.cat(decoded, dim=1), torch.cat(likelihoods, dim=1)

    @torch.no_grad()
    def compress(
        self,
        scores: Tensor,
        scene_mean: Tensor,
        base_entropy: FactorizedScoreEntropy,
    ) -> tuple[bytes, Tensor]:
        if scores.shape[0] != 1:
            raise ValueError("contextual score bitstream stores one scene at a time")
        offset, step = self._scene_parameters(scene_mean)
        points = scores.shape[-1]
        decoded: list[Tensor] = []
        strings: list[bytes] = []
        start = 0
        for group_index, width in enumerate(self.group_sizes):
            target = scores[:, start : start + width]
            base = self._group_base(decoded, offset, start, width, points, group_index)
            group_step = step[:, start : start + width, None, None]
            entropy = self._entropy(base_entropy, group_index)
            if self.spatial_context:
                even_symbols = (target[..., 0::2] - base[..., 0::2]) / group_step
                encoded = entropy.compress(even_symbols)
                strings.append(encoded[0])
                even_hat = entropy.decompress(encoded, even_symbols.shape[-2:])
                even_hat = base[..., 0::2] + group_step * even_hat
                odd_base = base[..., 1::2] + self._spatial_prediction(
                    group_index, even_hat, base
                )
                odd_symbols = (target[..., 1::2] - odd_base) / group_step
                odd_string, odd_hat = self._odd_compress(
                    odd_symbols, entropy, group_index, even_hat, base
                )
                strings.append(odd_string)
                odd_hat = odd_base + group_step * odd_hat
                decoded.append(self._merge_even_odd(even_hat, odd_hat, points))
            else:
                symbols = (target - base) / group_step
                encoded = entropy.compress(symbols)
                strings.append(encoded[0])
                symbols_hat = entropy.decompress(encoded, symbols.shape[-2:])
                decoded.append(base + group_step * symbols_hat)
            start += width
        packed = ScoreContextBitstream(
            rank=self.rank,
            slice_channels=self.slice_channels,
            flags=self.flags,
            strings=tuple(strings),
        ).pack()
        return packed, torch.cat(decoded, dim=1)

    @torch.no_grad()
    def decompress(
        self,
        payload: bytes,
        scene_mean: Tensor,
        points: int,
        base_entropy: FactorizedScoreEntropy,
    ) -> Tensor:
        packed = ScoreContextBitstream.unpack(payload)
        if packed.flags & ~self.KNOWN_FLAGS:
            raise ValueError("contextual score bitstream has unknown flags")
        if (
            packed.rank != self.rank
            or packed.slice_channels != self.slice_channels
            or packed.flags != self.flags
        ):
            raise ValueError("contextual score bitstream does not match codec configuration")
        expected_strings = len(self.group_sizes) * (2 if self.spatial_context else 1)
        if len(packed.strings) != expected_strings:
            raise ValueError("contextual score bitstream has an unexpected stream count")

        offset, step = self._scene_parameters(scene_mean)
        decoded: list[Tensor] = []
        string_index = 0
        start = 0
        for group_index, width in enumerate(self.group_sizes):
            base = self._group_base(decoded, offset, start, width, points, group_index)
            group_step = step[:, start : start + width, None, None]
            entropy = self._entropy(base_entropy, group_index)
            if self.spatial_context:
                even_count = math.ceil(points / 2)
                even_hat = entropy.decompress(
                    [packed.strings[string_index]], (1, even_count)
                )
                string_index += 1
                even_hat = base[..., 0::2] + group_step * even_hat
                odd_base = base[..., 1::2] + self._spatial_prediction(
                    group_index, even_hat, base
                )
                odd_hat = self._odd_decompress(
                    packed.strings[string_index],
                    (1, points // 2),
                    entropy,
                    group_index,
                    even_hat,
                    base,
                )
                string_index += 1
                odd_hat = odd_base + group_step * odd_hat
                decoded.append(self._merge_even_odd(even_hat, odd_hat, points))
            else:
                symbols_hat = entropy.decompress(
                    [packed.strings[string_index]], (1, points)
                )
                string_index += 1
                decoded.append(base + group_step * symbols_hat)
            start += width
        return torch.cat(decoded, dim=1)

    def update(
        self,
        base_entropy: FactorizedScoreEntropy,
        *,
        force: bool = False,
        update_quantiles: bool = False,
    ) -> bool:
        updated = False
        for entropy in self.active_entropies(base_entropy):
            updated = entropy.update(
                force=force, update_quantiles=update_quantiles
            ) or updated
        if self.spatial_gaussian is not None:
            scale_table = self.spatial_gaussian.scale_table
            if not scale_table.numel():
                scale_table = torch.exp(
                    torch.linspace(
                        math.log(0.11),
                        math.log(256.0),
                        64,
                        device=self.spatial_log_scales[0].device,
                    )
                )
            updated = self.spatial_gaussian.update_scale_table(
                scale_table, force=force
            ) or updated
        return updated

    def aux_loss(self, base_entropy: FactorizedScoreEntropy) -> Tensor:
        losses = [
            entropy.entropy_bottleneck.loss()
            for entropy in self.active_entropies(base_entropy)
        ]
        return torch.stack(losses).sum()

"""GPU-efficient decoder-causal context model for low-rank score symbols."""

from __future__ import annotations
import math
from collections.abc import Iterable
import torch
from torch import Tensor, nn
from .bitstream import ScoreContextBitstream
from .entropy import FactorizedScoreEntropy


class ContextualScoreEntropy(nn.Module):
    """Condition score coding on decoded scene, channel, and Morton context.

    Channel groups are decoded serially, but every token within a group is
    processed in parallel. Spatial context uses a two-pass
    even/odd split, avoiding a 4096-step autoregressive decoder.
    """

    FLAG_MEAN = 1 << 0
    FLAG_CHANNEL = 1 << 1
    FLAG_SPATIAL = 1 << 2
    FLAG_SPATIAL_ENTROPY_SPLIT = 1 << 5
    FLAG_NO_MEAN_OFFSET = 1 << 6
    KNOWN_FLAGS = (
        FLAG_MEAN | FLAG_CHANNEL | FLAG_SPATIAL
        | FLAG_SPATIAL_ENTROPY_SPLIT | FLAG_NO_MEAN_OFFSET
    )

    def __init__(
        self, *, rank: int, scene_channels: int, slice_channels: int, hidden: int,
        mean_condition: bool = True, channel_context: bool = True,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.scene_channels = int(scene_channels)
        self.slice_channels = int(slice_channels)
        self.mean_condition = mean_condition
        self.channel_context = channel_context
        self.group_sizes = tuple(
            (
                min(self.slice_channels, self.rank - start)
                for start in range(0, self.rank, self.slice_channels)
            )
        )
        self.group_entropies = nn.ModuleList(
            (FactorizedScoreEntropy(width) for width in self.group_sizes)
        )
        self.mean_conditioner = nn.Sequential(
            nn.LayerNorm(self.scene_channels),
            nn.Linear(self.scene_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.rank),
        )
        nn.init.zeros_(self.mean_conditioner[-1].weight)
        nn.init.zeros_(self.mean_conditioner[-1].bias)
        self.channel_predictors = nn.ModuleList()
        decoded = self.group_sizes[0]
        for width in self.group_sizes[1:]:
            predictor = nn.Conv2d(decoded, width, 1)
            nn.init.zeros_(predictor.weight)
            nn.init.zeros_(predictor.bias)
            self.channel_predictors.append(predictor)
            decoded += width
        self.spatial_predictors = nn.ModuleList()
        self.spatial_odd_entropies = nn.ModuleList()
        for width in self.group_sizes:
            predictor = nn.Conv2d(width, width, kernel_size=(1, 3), padding=(0, 1))
            nn.init.zeros_(predictor.weight)
            nn.init.zeros_(predictor.bias)
            self.spatial_predictors.append(predictor)
            self.spatial_odd_entropies.append(FactorizedScoreEntropy(width))
        # Evaluation diagnostics are opt-in so normal training/coding does not
        # pay for GPU reductions or device-to-host copies.
        self.mean_offset_enabled = True
        self.capture_diagnostics = False
        self.last_compress_diagnostics: dict[str, object] | None = None

    @property
    def flags(self) -> int:
        flags = 0
        if self.mean_condition:
            flags |= self.FLAG_MEAN
        if self.channel_context:
            flags |= self.FLAG_CHANNEL
        flags |= self.FLAG_SPATIAL
        flags |= self.FLAG_SPATIAL_ENTROPY_SPLIT
        if not self.mean_offset_enabled:
            flags |= self.FLAG_NO_MEAN_OFFSET
        return flags

    def _entropy(
        self, base_entropy: FactorizedScoreEntropy, group_index: int
    ) -> FactorizedScoreEntropy:
        return self.group_entropies[group_index]

    def active_entropies(
        self, base_entropy: FactorizedScoreEntropy
    ) -> Iterable[FactorizedScoreEntropy]:
        even = tuple(self.group_entropies)
        return even + tuple(self.spatial_odd_entropies)

    def _scene_parameters(self, scene_mean: Tensor) -> tuple[Tensor, Tensor]:
        if not self.mean_condition:
            shape = (scene_mean.shape[0], self.rank)
            return scene_mean.new_zeros(shape), scene_mean.new_ones(shape)
        parameters = self.mean_conditioner(scene_mean)
        offset, log_step = parameters.chunk(2, dim=-1)
        if not self.mean_offset_enabled:
            offset = torch.zeros_like(offset)
        step = torch.exp(2.0 * torch.tanh(0.5 * log_step))
        return (offset, step)

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
        if self.channel_context and group_index > 0:
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
        return prediction[..., 1::2]

    @staticmethod
    def _append_channel_diagnostics(
        destination: dict[str, list[float] | list[int]], value: Tensor
    ) -> None:
        """Append exact sufficient statistics for each channel in ``value``."""

        channels = value.detach().float().transpose(0, 1).reshape(value.shape[1], -1)
        count = channels.shape[1]
        destination["count"].extend([count] * channels.shape[0])
        if "first_token_b" in destination:
            destination["first_token_b"].extend(channels[:, 0].cpu().tolist())
        destination["sum"].extend(channels.sum(dim=1).cpu().tolist())
        destination["sum_squares"].extend(
            channels.square().sum(dim=1).cpu().tolist()
        )
        destination["sum_abs"].extend(channels.abs().sum(dim=1).cpu().tolist())
        destination["min"].extend(channels.amin(dim=1).cpu().tolist())
        destination["max"].extend(channels.amax(dim=1).cpu().tolist())

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
        return self.spatial_odd_entropies[group_index](symbols, training=training)

    @torch.no_grad()
    def _odd_compress(
        self,
        symbols: Tensor,
        even_entropy: FactorizedScoreEntropy,
        group_index: int,
        anchor_hat: Tensor,
        base: Tensor,
    ) -> tuple[bytes, Tensor]:
        entropy = self.spatial_odd_entropies[group_index]
        strings = entropy.compress(symbols)
        return (strings[0], entropy.decompress(strings, symbols.shape[-2:]))

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
        return self.spatial_odd_entropies[group_index].decompress([string], shape)

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
            even_symbols = (target[..., 0::2] - base[..., 0::2]) / group_step
            even_hat, even_likelihood = entropy(even_symbols, training=training)
            even_hat = base[..., 0::2] + group_step * even_hat
            odd_base = base[..., 1::2] + self._spatial_prediction(
                group_index, even_hat, base
            )
            odd_symbols = (target[..., 1::2] - odd_base) / group_step
            odd_hat, odd_likelihood = self._odd_forward(
                odd_symbols, entropy, group_index, even_hat, base, training=training
            )
            odd_hat = odd_base + group_step * odd_hat
            decoded.append(self._merge_even_odd(even_hat, odd_hat, points))
            likelihoods.append(
                self._merge_even_odd(even_likelihood, odd_likelihood, points)
            )
            start += width
        return (torch.cat(decoded, dim=1), torch.cat(likelihoods, dim=1))

    @torch.no_grad()
    def compress(
        self, scores: Tensor, scene_mean: Tensor, base_entropy: FactorizedScoreEntropy
    ) -> tuple[bytes, Tensor]:
        if scores.shape[0] != 1:
            raise ValueError("contextual score bitstream stores one scene at a time")
        offset, step = self._scene_parameters(scene_mean)
        self.last_compress_diagnostics = None
        diagnostics = None
        if self.capture_diagnostics:
            stat_names = ("count", "sum", "sum_squares", "sum_abs", "min", "max")
            diagnostics = {
                "mean_offset_b": offset[0].detach().float().cpu().tolist(),
                "quantization_step_d": step[0].detach().float().cpu().tolist(),
                "effective_b_even": {
                    name: [] for name in (*stat_names, "first_token_b")
                },
                "effective_b_odd": {
                    name: [] for name in (*stat_names, "first_token_b")
                },
            }
            for parity in ("even", "odd"):
                for value in (
                    "score", "mean_centered_score", "centered_score", "entropy_input"
                ):
                    diagnostics[f"{value}_{parity}"] = {
                        name: [] for name in stat_names
                    }
        points = scores.shape[-1]
        decoded: list[Tensor] = []
        strings: list[bytes] = []
        start = 0
        for group_index, width in enumerate(self.group_sizes):
            target = scores[:, start : start + width]
            base = self._group_base(decoded, offset, start, width, points, group_index)
            group_step = step[:, start : start + width, None, None]
            entropy = self._entropy(base_entropy, group_index)
            even_target = target[..., 0::2]
            even_base = base[..., 0::2]
            even_centered = even_target - even_base
            even_symbols = even_centered / group_step
            if diagnostics is not None:
                mean_base = offset[:, start : start + width, None, None]
                self._append_channel_diagnostics(
                    diagnostics["effective_b_even"], even_base
                )
                self._append_channel_diagnostics(diagnostics["score_even"], even_target)
                self._append_channel_diagnostics(
                    diagnostics["mean_centered_score_even"],
                    even_target - mean_base,
                )
                self._append_channel_diagnostics(
                    diagnostics["centered_score_even"], even_centered
                )
                self._append_channel_diagnostics(
                    diagnostics["entropy_input_even"], even_symbols
                )
            encoded = entropy.compress(even_symbols)
            strings.append(encoded[0])
            even_hat = entropy.decompress(encoded, even_symbols.shape[-2:])
            even_hat = base[..., 0::2] + group_step * even_hat
            odd_base = base[..., 1::2] + self._spatial_prediction(
                group_index, even_hat, base
            )
            odd_target = target[..., 1::2]
            odd_centered = odd_target - odd_base
            odd_symbols = odd_centered / group_step
            if diagnostics is not None:
                self._append_channel_diagnostics(
                    diagnostics["effective_b_odd"], odd_base
                )
                self._append_channel_diagnostics(diagnostics["score_odd"], odd_target)
                self._append_channel_diagnostics(
                    diagnostics["mean_centered_score_odd"],
                    odd_target - mean_base,
                )
                self._append_channel_diagnostics(
                    diagnostics["centered_score_odd"], odd_centered
                )
                self._append_channel_diagnostics(
                    diagnostics["entropy_input_odd"], odd_symbols
                )
            odd_string, odd_hat = self._odd_compress(
                odd_symbols, entropy, group_index, even_hat, base
            )
            strings.append(odd_string)
            odd_hat = odd_base + group_step * odd_hat
            decoded.append(self._merge_even_odd(even_hat, odd_hat, points))
            start += width
        self.last_compress_diagnostics = diagnostics
        packed = ScoreContextBitstream(
            rank=self.rank,
            slice_channels=self.slice_channels,
            flags=self.flags,
            strings=tuple(strings),
        ).pack()
        return (packed, torch.cat(decoded, dim=1))

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
            raise ValueError(
                "contextual score bitstream does not match codec configuration"
            )
        expected_strings = len(self.group_sizes) * 2
        if len(packed.strings) != expected_strings:
            raise ValueError(
                "contextual score bitstream has an unexpected stream count"
            )
        offset, step = self._scene_parameters(scene_mean)
        decoded: list[Tensor] = []
        string_index = 0
        start = 0
        for group_index, width in enumerate(self.group_sizes):
            base = self._group_base(decoded, offset, start, width, points, group_index)
            group_step = step[:, start : start + width, None, None]
            entropy = self._entropy(base_entropy, group_index)
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
            updated = (
                entropy.update(force=force, update_quantiles=update_quantiles)
                or updated
            )
        return updated

    def aux_loss(self, base_entropy: FactorizedScoreEntropy) -> Tensor:
        losses = [
            entropy.entropy_bottleneck.loss()
            for entropy in self.active_entropies(base_entropy)
        ]
        return torch.stack(losses).sum()

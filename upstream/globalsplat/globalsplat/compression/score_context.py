"""GPU-efficient decoder-causal context model for low-rank score symbols."""

from __future__ import annotations
import math
from collections.abc import Iterable
import torch
from torch import Tensor, nn
from .bitstream import ScoreContextBitstream
from .entropy import FactorizedScoreEntropy


class AnchorPairPredictor(nn.Module):
    """Linear channel mixing of explicitly gathered decoded left/right anchors."""

    def __init__(self, width: int, spacing: int, *, phase_specific: bool = False):
        super().__init__()
        self.spacing = spacing
        self.phase_specific = phase_specific
        phases = spacing - 1 if phase_specific else 1
        self.weight = nn.Parameter(torch.zeros(phases, width, width, 2))
        self.bias = nn.Parameter(torch.zeros(phases, width))
        self.edge_bias = nn.Parameter(torch.zeros(phases, width))

    def forward(self, context: Tensor, indices: Tensor) -> Tensor:
        # indices belong to this stage; both anchor positions belong to earlier
        # stages. Duplicate the left anchor when the right one does not exist.
        left = (indices // self.spacing) * self.spacing
        right = left + self.spacing
        missing = right >= context.shape[-1]
        right = torch.where(missing, left, right)
        anchors = torch.stack((context.index_select(3, left), context.index_select(3, right)), -1)
        phase = indices.remainder(self.spacing) - 1 if self.phase_specific else torch.zeros_like(indices)
        weight = self.weight.index_select(0, phase)
        output = torch.einsum("bihnk,noik->bohn", anchors, weight)
        bias = self.bias.index_select(0, phase) + missing[:, None] * self.edge_bias.index_select(0, phase)
        return output + bias.T[None, :, None, :]


class ContextualScoreEntropy(nn.Module):
    """Condition score coding on decoded scene, channel, and ordered tokens.

    Channel groups are decoded serially, but every token within a group is
    processed in parallel. Spatial context uses two, three or four passes over
    ordered tokens, avoiding a token-by-token autoregressive decoder.
    """

    FLAG_MEAN = 1 << 0
    FLAG_CHANNEL = 1 << 1
    FLAG_SPATIAL = 1 << 2
    FLAG_SPATIAL_ENTROPY_SPLIT = 1 << 5
    FLAG_NO_MEAN_OFFSET = 1 << 6
    FLAG_THREE_STAGES = 1 << 7
    FLAG_FOUR_STAGES = 1 << 8
    FLAG_KERNEL5 = 1 << 9
    FLAG_KERNEL7 = 1 << 10
    FLAG_QUARTER2 = 1 << 11
    FLAG_DYADIC4 = 1 << 12
    KNOWN_FLAGS = (
        FLAG_MEAN | FLAG_CHANNEL | FLAG_SPATIAL
        | FLAG_SPATIAL_ENTROPY_SPLIT | FLAG_NO_MEAN_OFFSET
        | FLAG_THREE_STAGES | FLAG_FOUR_STAGES | FLAG_KERNEL5 | FLAG_KERNEL7
        | FLAG_QUARTER2 | FLAG_DYADIC4
    )

    def __init__(
        self, *, rank: int, scene_channels: int, slice_channels: int, hidden: int,
        mean_condition: bool = True, channel_context: bool = True,
        spatial_stages: int = 2, spatial_kernel: int = 3,
        context_quantization: str = "noise",
        context_schedule: str = "legacy",
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.scene_channels = int(scene_channels)
        self.slice_channels = int(slice_channels)
        self.mean_condition = mean_condition
        self.channel_context = channel_context
        self.spatial_stages = spatial_stages
        self.spatial_kernel = spatial_kernel
        self.context_quantization = context_quantization
        self.context_schedule = context_schedule
        if spatial_stages not in (2, 3, 4) or spatial_kernel not in (3, 5, 7):
            raise ValueError("unsupported spatial stage count or kernel")
        if context_quantization not in ("noise", "ste"):
            raise ValueError("context_quantization must be noise or ste")
        if context_schedule not in ("legacy", "quarter2", "dyadic4"):
            raise ValueError("unsupported context schedule")
        if context_schedule != "legacy":
            if spatial_stages != (2 if context_schedule == "quarter2" else 4) or channel_context:
                raise ValueError("anchor schedules require matching stage count and no channel context")
        elif spatial_stages > 2 and (channel_context or spatial_kernel not in (5, 7)):
            raise ValueError("3/4 spatial stages require no channel context and kernel 5 or 7")
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
            predictor = nn.Conv2d(width, width, kernel_size=(1, spatial_kernel), padding=(0, spatial_kernel // 2))
            nn.init.zeros_(predictor.weight)
            nn.init.zeros_(predictor.bias)
            self.spatial_predictors.append(predictor)
            self.spatial_odd_entropies.append(FactorizedScoreEntropy(width))
        if spatial_stages >= 3:
            self.split_even_predictors = nn.ModuleList()
            self.split_even_entropies = nn.ModuleList()
            for width in self.group_sizes:
                self.split_even_predictors.append(self._make_predictor(width, spatial_kernel))
                self.split_even_entropies.append(FactorizedScoreEntropy(width))
        if spatial_stages == 4:
            self.split_odd_predictors = nn.ModuleList()
            self.split_odd_entropies = nn.ModuleList()
            for width in self.group_sizes:
                self.split_odd_predictors.append(self._make_predictor(width, spatial_kernel))
                self.split_odd_entropies.append(FactorizedScoreEntropy(width))
        if context_schedule != "legacy":
            spacings = (4,) if context_schedule == "quarter2" else (8, 4, 2)
            self.anchor_predictors = nn.ModuleList(
                nn.ModuleList(AnchorPairPredictor(width, spacing,
                    phase_specific=context_schedule == "quarter2") for width in self.group_sizes)
                for spacing in spacings
            )
            # Compatibility convolution tensors are retained but never used.
            for name in ("spatial_predictors", "split_even_predictors", "split_odd_predictors"):
                for parameter in getattr(self, name, nn.ModuleList()).parameters():
                    parameter.requires_grad_(False)
        # Evaluation diagnostics are opt-in so normal training/coding does not
        # pay for GPU reductions or device-to-host copies.
        self.mean_offset_enabled = True
        self.capture_diagnostics = False
        self.last_compress_diagnostics: dict[str, object] | None = None
        self.last_compress_stage_bytes: dict[str, int] | None = None

    @staticmethod
    def _make_predictor(width: int, kernel: int):
        predictor = nn.Conv2d(width, width, (1, kernel), padding=(0, kernel // 2))
        nn.init.zeros_(predictor.weight)
        nn.init.zeros_(predictor.bias)
        return predictor

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
        if self.spatial_stages == 3:
            flags |= self.FLAG_THREE_STAGES
        elif self.spatial_stages == 4:
            flags |= self.FLAG_FOUR_STAGES
        if self.spatial_kernel == 5:
            flags |= self.FLAG_KERNEL5
        elif self.spatial_kernel == 7:
            flags |= self.FLAG_KERNEL7
        if self.context_schedule == "quarter2":
            flags |= self.FLAG_QUARTER2
        elif self.context_schedule == "dyadic4":
            flags |= self.FLAG_DYADIC4
        return flags

    def _entropy(
        self, base_entropy: FactorizedScoreEntropy, group_index: int
    ) -> FactorizedScoreEntropy:
        return self.group_entropies[group_index]

    def active_entropies(
        self, base_entropy: FactorizedScoreEntropy
    ) -> Iterable[FactorizedScoreEntropy]:
        even = tuple(self.group_entropies)
        active = even + tuple(self.spatial_odd_entropies)
        if self.spatial_stages >= 3:
            active += tuple(self.split_even_entropies)
        if self.spatial_stages == 4:
            active += tuple(self.split_odd_entropies)
        return active

    def _entropy_forward(self, entropy, symbols, *, training):
        value, likelihood = entropy(symbols, training=training)
        is_training = self.training if training is None else training
        if is_training and self.context_quantization == "ste":
            # Match the sender/receiver lattice, including the learned median.
            bottleneck = entropy.entropy_bottleneck
            rounded = bottleneck.quantize(symbols, "dequantize", bottleneck._get_medians())
            value = symbols + (rounded - symbols).detach()
        return value, likelihood

    def stage_indices(self, points: int, device):
        indices = torch.arange(points, device=device)
        if self.context_schedule == "quarter2":
            return (("anchors", indices[0::4]), ("remaining", indices[indices % 4 != 0]))
        if self.context_schedule == "dyadic4":
            return (("0mod8", indices[0::8]), ("4mod8", indices[4::8]),
                    ("2mod4", indices[2::4]), ("odd", indices[1::2]))
        if self.spatial_stages == 2:
            return (("even", indices[0::2]), ("odd", indices[1::2]))
        stages = [("A", indices[0::4]), ("B", indices[2::4])]
        if self.spatial_stages == 3:
            stages.append(("C+D", indices[1::2]))
        else:
            stages.extend((("C", indices[1::4]), ("D", indices[3::4])))
        return tuple(stages)

    def _stage_modules(self, stage, group):
        if stage == 0:
            return self.group_entropies[group], None
        if self.context_schedule != "legacy":
            if self.context_schedule == "quarter2" or stage == 2:
                entropy = self.spatial_odd_entropies[group]
            elif stage == 1:
                entropy = self.split_even_entropies[group]
            else:
                entropy = self.split_odd_entropies[group]
            return entropy, self.anchor_predictors[stage - 1][group]
        if stage == 1:
            return self.split_even_entropies[group], self.split_even_predictors[group]
        if stage == 2:
            return self.spatial_odd_entropies[group], self.spatial_predictors[group]
        return self.split_odd_entropies[group], self.split_odd_predictors[group]

    def _staged(self, scene_mean, points, base_entropy, *, scores=None,
                strings=None, training=None, mode="forward"):
        """Stage-major coding: all channels of A precede any channels of B.

        The sole predictor input is a canvas of previously reconstructed scores.
        Original values of current/future stages never enter that canvas.
        """
        if points < 1:
            raise ValueError("score stream must contain at least one token")
        offset, step = self._scene_parameters(scene_mean)
        base = offset[:, :, None, None].expand(-1, -1, 1, points)
        decoded = torch.zeros_like(base)
        likelihoods = torch.zeros_like(base)
        visible = torch.zeros(1, 1, 1, points, device=base.device, dtype=torch.bool)
        output_strings = []
        stage_bytes = {}
        for stage, (name, indices) in enumerate(self.stage_indices(points, base.device)):
            # Future sites stay exactly zero even when mean/channel offsets exist.
            context = torch.where(visible, decoded - base, torch.zeros_like(decoded))
            reconstructed_groups, likelihood_groups = [], []
            start, byte_count = 0, 0
            for group, width in enumerate(self.group_sizes):
                entropy, predictor = self._stage_modules(stage, group)
                group_base = base[:, start:start + width]
                if isinstance(predictor, AnchorPairPredictor):
                    prediction = group_base.index_select(3, indices) + predictor(
                        context[:, start:start + width], indices)
                elif predictor is not None:
                    group_base = group_base + predictor(context[:, start:start + width])
                    prediction = group_base.index_select(3, indices)
                else:
                    prediction = group_base.index_select(3, indices)
                group_step = step[:, start:start + width, None, None]
                # Tiny symbol arrays can under-allocate the native rANS buffer.
                # Pad only the coding array; padding tokens never enter context.
                coding_points = max(8, indices.numel())
                if indices.numel() == 0:
                    value = prediction
                    likelihood = prediction
                    payload = b""
                    if mode == "decode" and strings[stage * len(self.group_sizes) + group]:
                        raise ValueError("empty token stage received a nonempty entropy string")
                elif mode == "decode":
                    payload = strings[stage * len(self.group_sizes) + group]
                    value = entropy.decompress([payload], (1, coding_points))[..., :indices.numel()]
                    likelihood = None
                else:
                    target = scores[:, start:start + width].index_select(3, indices)
                    symbols = (target - prediction) / group_step
                    if mode == "encode":
                        coding_symbols = torch.nn.functional.pad(symbols, (0, coding_points - indices.numel()))
                        encoded = entropy.compress(coding_symbols)
                        payload = encoded[0]
                        value = entropy.decompress(encoded, coding_symbols.shape[-2:])[..., :indices.numel()]
                        likelihood = None
                    else:
                        value, likelihood = self._entropy_forward(entropy, symbols, training=training)
                reconstructed_groups.append(prediction + group_step * value)
                if mode == "forward":
                    likelihood_groups.append(likelihood)
                else:
                    byte_count += len(payload)
                    if mode == "encode":
                        output_strings.append(payload)
                start += width
            decoded = decoded.index_copy(3, indices, torch.cat(reconstructed_groups, 1))
            if mode == "forward":
                likelihoods = likelihoods.index_copy(3, indices, torch.cat(likelihood_groups, 1))
            # torch.where saves its mask for backward; never mutate that mask.
            visible = visible.clone()
            visible[..., indices] = True
            stage_bytes[name] = byte_count
        if mode == "encode":
            self.last_compress_stage_bytes = stage_bytes
            return ScoreContextBitstream(self.rank, self.slice_channels, self.flags,
                                         tuple(output_strings)).pack(), decoded
        if mode == "decode":
            return decoded
        return decoded, likelihoods

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
        return self._entropy_forward(self.spatial_odd_entropies[group_index], symbols, training=training)

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
        if self.spatial_stages > 2 or self.context_schedule != "legacy":
            return self._staged(scene_mean, scores.shape[-1], base_entropy,
                                scores=scores, training=training)
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
            even_hat, even_likelihood = self._entropy_forward(entropy, even_symbols, training=training)
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
        if self.spatial_stages > 2 or self.context_schedule != "legacy":
            if self.capture_diagnostics:
                raise ValueError("b/even-odd diagnostics require legacy 2 stages; use score stage byte records")
            return self._staged(scene_mean, scores.shape[-1], base_entropy, scores=scores, mode="encode")
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
        self.last_compress_stage_bytes = {"even": sum(len(s) for s in strings[0::2]),
                                          "odd": sum(len(s) for s in strings[1::2])}
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
        expected_strings = len(self.group_sizes) * self.spatial_stages
        if len(packed.strings) != expected_strings:
            raise ValueError(
                "contextual score bitstream has an unexpected stream count"
            )
        if self.spatial_stages > 2 or self.context_schedule != "legacy":
            return self._staged(scene_mean, points, base_entropy, strings=packed.strings, mode="decode")
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

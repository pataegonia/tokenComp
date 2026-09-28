"""Fixed-symbol, held-out entropy diagnostics. Not a production codec variant.

Only temporary method wrappers are installed; all learned weights stay fixed.
The receiver recomputes context from mean/decoded anchors, never the target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor
from compressai.ans import RansEncoder, RansDecoder
from compressai._CXX import pmf_to_quantized_cdf

from .bitstream import SceneBitstream, ScoreContextBitstream, ResidualBitstream


def array(value: Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1).astype(np.int64)


def context_bins(step: Tensor, shape, anchor_hat=None, base=None) -> np.ndarray:
    """Five fixed bins: scene step for even; neighbour-anchor magnitude for odd."""
    if anchor_hat is None:
        value = step.expand(shape)
    else:
        delta = (anchor_hat - base[..., 0::2]) / step
        # Odd i is between even i and even i+1; repeat the final anchor at edge.
        right = torch.cat((delta[..., 1:], delta[..., -1:]), dim=-1)
        value = 0.5 * (delta.abs() + right.abs())
        value = value[..., : shape[-1]]
    edges = value.new_tensor([0.5, 1.0, 2.0, 4.0])
    return array(torch.bucketize(value.contiguous(), edges))


@dataclass
class Table:
    cdfs: list[list[int]]
    lengths: list[int]
    offsets: list[int]

    def __post_init__(self):
        if not self.cdfs or not (len(self.cdfs) == len(self.lengths) == len(self.offsets)):
            raise ValueError("invalid CDF table dimensions")
        for row, length in zip(self.cdfs, self.lengths):
            if length != len(row) or row[0] != 0 or row[-1] != 65536:
                raise ValueError("probe requires complete 16-bit CDF rows")
            if any(b <= a for a, b in zip(row, row[1:])):
                raise ValueError("CDF frequencies must be positive")
        self.width = max(self.lengths) - 1
        self.frequencies = np.zeros((len(self.cdfs), self.width), dtype=np.int64)
        for i, row in enumerate(self.cdfs):
            self.frequencies[i, : len(row) - 1] = np.diff(row)

    @classmethod
    def from_entropy(cls, entropy):
        lengths = entropy._cdf_length.cpu().tolist()
        cdfs = entropy._quantized_cdf.cpu().tolist()
        return cls([row[:n] for row, n in zip(cdfs, lengths)], lengths,
                   entropy._offset.cpu().tolist())

    def events(self, symbols, indexes):
        value = symbols - np.asarray(self.offsets)[indexes]
        sentinel = np.asarray(self.lengths)[indexes] - 2
        escaped = (value < 0) | (value >= sentinel)
        events = np.where(escaped, sentinel, value)
        raw = np.where(value < 0, -2 * value - 1, 2 * (value - sentinel))
        return events, escaped, raw

    def costs(self, symbols, indexes):
        events, escaped, raw = self.events(symbols, indexes)
        cdf_bits = -np.log2(self.frequencies[indexes, events] / 65536.0).sum()
        # CompressAI 1.2.8 rANS: escape payload consists of 4-bit bypass chunks
        # and a chunk-count code. See cpp_exts/rans/rans_interface.cpp.
        bypass_bits = 0
        for value in raw[escaped]:
            chunks = (int(value).bit_length() + 3) // 4
            bypass_bits += 4 * (chunks + chunks // 15 + 1)
        return dict(cdf_bits=float(cdf_bits), bypass_bits=bypass_bits,
                    escape_symbols=int(escaped.sum()))

    def encode(self, symbols, indexes):
        return RansEncoder().encode_with_indexes(
            symbols.tolist(), indexes.tolist(), self.cdfs, self.lengths, self.offsets)

    def decode(self, stream, indexes):
        return np.asarray(RansDecoder().decode_with_indexes(
            stream, indexes.tolist(), self.cdfs, self.lengths, self.offsets), dtype=np.int64)

    def export(self):
        return dict(cdfs=self.cdfs, lengths=self.lengths, offsets=self.offsets)


class HistogramFit:
    """Train-only counts with fixed, predeclared shrinkage to the parent CDF."""

    def __init__(self, base: Table, bins: int, prior: float = 4096.0):
        if bins < 1 or not math.isfinite(prior) or prior <= 0:
            raise ValueError("bins and finite prior strength must be positive")
        self.base, self.bins, self.prior = base, bins, prior
        self.counts = np.zeros((len(base.cdfs), bins, base.width), dtype=np.int64)
        self.tables: dict[str, Table] = {}

    def add(self, symbols, indexes, bins):
        if self.tables:
            raise RuntimeError("cannot fit after freezing; test symbols must not update tables")
        events, _, _ = self.base.events(symbols, indexes)
        flat = ((indexes * self.bins + bins) * self.base.width + events)
        self.counts += np.bincount(flat, minlength=self.counts.size).reshape(self.counts.shape)

    def freeze(self):
        global_counts = self.counts.sum(axis=1)
        marginal, conditional, marginal_offsets, conditional_offsets = [], [], [], []
        for i, original in enumerate(self.base.cdfs):
            n = len(original) - 1
            old_p = self.base.frequencies[i, :n] / 65536.0
            counts = global_counts[i, :n]
            global_p = (counts + self.prior * old_p) / (counts.sum() + self.prior)
            marginal.append(pmf_to_quantized_cdf(global_p.tolist(), 16)
                            if counts.sum() else original)
            marginal_offsets.append(self.base.offsets[i])
            for b in range(self.bins):
                local = self.counts[i, b, :n]
                local_p = (local + self.prior * global_p) / (local.sum() + self.prior)
                conditional.append(pmf_to_quantized_cdf(local_p.tolist(), 16)
                                   if self.bins > 1 and local.sum() else marginal[-1])
                conditional_offsets.append(self.base.offsets[i])
        self.tables = {
            "marginal": Table(marginal, list(map(len, marginal)), marginal_offsets),
            "context": Table(conditional, list(map(len, conditional)), conditional_offsets),
        }

    def select(self, variant, indexes, bins):
        return self.tables[variant], (indexes * self.bins + bins
                                     if variant == "context" else indexes)


class EntropyProbe:
    """Instrument a Full+Split codec for counts, rate auditing and exact recoding."""

    def __init__(self, codec, prior=4096.0):
        cfg = codec.config
        if not (cfg.score_channel_context and cfg.score_spatial_context
                and cfg.score_mean_condition and cfg.score_spatial_entropy == "split"
                and cfg.use_residual):
            raise ValueError("this probe requires Full+Split with residual enabled")
        self.codec, self.context = codec, codec.score_context
        self.fits, self.models, self.specs = {}, {}, {}
        for i, even in enumerate(self.context.group_entropies):
            for parity, entropy in (("even", even), ("odd", self.context.spatial_odd_entropies[i])):
                name = f"score_g{i}_{parity}"
                self.models[name] = entropy.entropy_bottleneck
                self.specs[name] = (i, parity)
        self.models["residual_z"] = codec.residual_codec.entropy_bottleneck
        self.models["residual_y"] = codec.residual_codec.gaussian_conditional
        for name, entropy in self.models.items():
            self.fits[name] = HistogramFit(Table.from_entropy(entropy),
                                           5 if name.startswith("score_") else 1, prior)
        self.records, self._patches = {}, []
        self.capture, self.variant = False, None
        self.step, self.odd_context, self.gaussian_scales = None, {}, None

    def _patch(self, obj, name, value):
        had = name in obj.__dict__
        previous = obj.__dict__.get(name)
        self._patches.append((obj, name, had, previous))
        setattr(obj, name, value)

    def __enter__(self):
        try:
            original_scene = self.context._scene_parameters
            def scene(mean):
                offset, step = original_scene(mean)
                self.step = step
                self.odd_context = {}
                return offset, step
            self._patch(self.context, "_scene_parameters", scene)
            original_odd_encode = self.context._odd_compress
            def odd_encode(symbols, entropy, group, anchor, base):
                self.odd_context[group] = (anchor, base)
                return original_odd_encode(symbols, entropy, group, anchor, base)
            self._patch(self.context, "_odd_compress", odd_encode)
            original_odd_decode = self.context._odd_decompress
            def odd_decode(stream, shape, entropy, group, anchor, base):
                self.odd_context[group] = (anchor, base)
                return original_odd_decode(stream, shape, entropy, group, anchor, base)
            self._patch(self.context, "_odd_decompress", odd_decode)
            original_parameters = self.codec.residual_codec._parameters_from_z
            def parameters(z, shape):
                scales, means = original_parameters(z, shape)
                self.gaussian_scales = scales
                return scales, means
            self._patch(self.codec.residual_codec, "_parameters_from_z", parameters)
            for name, model in self.models.items():
                self._install_entropy(name, model)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for obj, name, had, previous in reversed(self._patches):
            if had:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self._patches.clear()

    def _bins(self, name, shape):
        if name not in self.specs:
            return np.zeros(math.prod(shape), dtype=np.int64)
        group, parity = self.specs[name]
        start = sum(self.context.group_sizes[:group])
        width = self.context.group_sizes[group]
        step = self.step[:, start:start + width, None, None]
        anchor, base = self.odd_context[group] if parity == "odd" else (None, None)
        return context_bins(step, shape, anchor, base)

    def _record(self, name, model, inputs, indexes, means, strings):
        if not self.capture:
            return
        symbols = model.quantize(inputs, "symbols", means)
        # Evaluate at exactly the integer symbol that was coded. Calling forward
        # on dequantized BF16 values would round a second time, and a large
        # Gaussian mean could lose the integer residual through cancellation.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            if name == "residual_y":
                likelihood = model._likelihood(symbols.float(), self.gaussian_scales.float())
            else:
                decoded = model.dequantize(symbols, means.float())
                values = decoded.movedim(1, 0).reshape(model.channels, 1, -1)
                likelihood, _, _ = model._likelihood(values)
            if model.use_likelihood_bound:
                likelihood = model.likelihood_lower_bound(likelihood)
        flat_symbols, flat_indexes = array(symbols), array(indexes)
        fit = self.fits[name]
        costs = fit.base.costs(flat_symbols, flat_indexes)
        lower_bound = float(model.likelihood_lower_bound.bound.item())
        self.records[name] = dict(
            symbols=flat_symbols, indexes=flat_indexes,
            bins=self._bins(name, symbols.shape), native=strings[0], shape=tuple(symbols.shape),
            stats=dict(symbols=flat_symbols.size, actual_bits=8 * len(strings[0]),
                       model_nll_bits=float(-likelihood.double().clamp_min(1e-300).log2().sum()),
                       likelihood_floor_symbols=int((likelihood <= lower_bound).sum()), **costs))

    def _decode(self, name, model, strings, indexes, means, dtype=torch.float32):
        fit = self.fits[name]
        bins, flat_indexes = self._bins(name, indexes.shape), array(indexes)
        record = self.records[name]
        # Validate independent receiver context BEFORE rANS reads a stream.
        if not (np.array_equal(bins, record["bins"])
                and np.array_equal(flat_indexes, record["indexes"])):
            raise RuntimeError(f"sender/receiver context mismatch: {name}")
        table, selected = fit.select(self.variant, flat_indexes, bins)
        symbols = table.decode(strings[0], selected)
        if not np.array_equal(symbols, record["symbols"]):
            raise RuntimeError(f"integer symbol round-trip failed: {name}")
        value = torch.as_tensor(symbols.copy(), device=indexes.device,
                                dtype=torch.int32).reshape(indexes.shape)
        return model.dequantize(value, means, dtype=dtype)

    def _install_entropy(self, name, model):
        original_encode, original_decode = model.compress, model.decompress
        if name == "residual_y":
            def encode(inputs, indexes, means=None):
                strings = original_encode(inputs, indexes, means=means)
                self._record(name, model, inputs, indexes, means, strings)
                return strings
            def decode(strings, indexes, dtype=torch.float32, means=None):
                if self.variant is None:
                    return original_decode(strings, indexes, dtype=dtype, means=means)
                return self._decode(name, model, strings, indexes, means, dtype)
        else:
            def encode(inputs):
                strings = original_encode(inputs)
                indexes = model._build_indexes(inputs.size()).to(inputs.device)
                self._record(name, model, inputs, indexes, model._get_medians(), strings)
                return strings
            def decode(strings, size):
                if self.variant is None:
                    return original_decode(strings, size)
                shape = (len(strings), model.channels, *size)
                medians = model._get_medians()
                indexes = model._build_indexes(shape).to(medians.device)
                return self._decode(name, model, strings, indexes, medians)
        self._patch(model, "compress", encode)
        self._patch(model, "decompress", decode)

    def begin(self):
        self.records = {}
        self.capture, self.variant = True, None

    def end(self):
        self.capture = False
        if set(self.records) != set(self.models):
            raise RuntimeError("not all expected score/y/z streams were captured")

    def fit_scene(self):
        for name, record in self.records.items():
            self.fits[name].add(record["symbols"], record["indexes"], record["bins"])

    def freeze(self):
        for fit in self.fits.values():
            fit.freeze()

    def recode(self, payload: bytes, variant: str):
        original = SceneBitstream.unpack(payload)
        score = ScoreContextBitstream.unpack(original.score)
        residual = ResidualBitstream.unpack(original.residual)
        encoded = {}
        for name, record in self.records.items():
            fit = self.fits[name]
            table, indexes = fit.select(variant, record["indexes"], record["bins"])
            encoded[name] = table.encode(record["symbols"], indexes)
        score_strings = tuple(encoded[f"score_g{i}_{p}"]
                              for i in range(len(self.context.group_sizes)) for p in ("even", "odd"))
        new_score = ScoreContextBitstream(score.rank, score.slice_channels,
                                          score.flags, score_strings).pack()
        new_residual = ResidualBitstream(residual.z_shape, residual.y_shape,
                                         (encoded["residual_z"],), (encoded["residual_y"],)).pack()
        new_scene = SceneBitstream(original.points, original.channels, original.rank,
                                    original.mean_fp16, new_score, new_residual,
                                    original.flags, original.version)
        # Private diagnostic packet: external production decoders lack these CDFs.
        return new_scene.pack(), {name: len(value) for name, value in encoded.items()}

    def verify(self, original_payload, candidate_payload, variant):
        self.variant = None
        expected = self.codec.decompress(original_payload)
        try:
            self.variant = variant
            actual = self.codec.decompress(candidate_payload)
            if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
                raise RuntimeError(f"decoded features differ for {variant}")
        finally:
            self.variant = None
        return True

    def export_tables(self):
        return {name: {"native": fit.base.export(),
                       **{variant: table.export() for variant, table in fit.tables.items()}}
                for name, fit in self.fits.items()}

    def save_scene_cache(self, path, payload):
        """Optional one-scene integer cache; no pickle or original image tensors.

        Stream names and BCHW shapes retain the codec's causal coding order.
        Native strings/tables let an offline probe first reproduce baseline bits.
        Richer feature context still requires the parent checkpoint to decode it.
        """
        scene = SceneBitstream.unpack(payload)
        values = {"scene_mean_fp16_bytes": np.frombuffer(scene.mean_fp16, dtype=np.uint8),
                  "native_scene_bytes": np.frombuffer(payload, dtype=np.uint8)}
        for name, record in self.records.items():
            for key in ("symbols", "indexes"):
                values[f"{name}__{key}"] = record[key].astype(np.int32)
            values[f"{name}__bins"] = record["bins"].astype(np.uint8)
            values[f"{name}__shape"] = np.asarray(record["shape"], dtype=np.int64)
            values[f"{name}__native"] = np.frombuffer(record["native"], dtype=np.uint8)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **values)

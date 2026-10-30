"""Offline ordering/context diagnostics. These statistics never condition coding."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager

import torch

from .bitstream import SceneBitstream, ScoreContextBitstream
from .morton import morton_order_3d
from .token_order import greedy_neighbor_order, hilbert_order_3d
from .score_context import AnchorPairPredictor


def cpu(value):
    return value.detach().float().cpu()


def rows(value):
    return cpu(value)[0, :, 0].T.contiguous()


def channel_scale(score):
    """Per-scene per-channel population STD; shared across all fixed-score orders."""
    return score.float().std(dim=0, correction=0).clamp_min(1e-6)


def correlation(a, b):
    a, b = a.double(), b.double()
    a, b = a - a.mean(0), b - b.mean(0)
    denominator = (a.square().sum(0) * b.square().sum(0)).sqrt()
    valid = denominator > 1e-12
    result = (a * b).sum(0) / denominator.clamp_min(1e-12)
    return dict(mean=float(result[valid].mean()) if valid.any() else None,
                valid_channels=int(valid.sum()),
                channels=[float(v) if bool(ok) else None for v, ok in zip(result, valid)])


def relation(a, b, xyz_a, xyz_b, scale, xyz_scale=1.):
    if not len(a):
        return dict(pairs=0, raw_l1=None, raw_mse=None, normalized_l1=None, normalized_mse=None,
                    normalized_l1_channels=[], normalized_mse_channels=[],
                    xyz_distance=None, xyz_distance_normalized=None,
                    correlation=dict(mean=None, valid_channels=0, channels=[]))
    delta = (a - b) / scale
    return dict(pairs=len(a), raw_l1=float((a-b).abs().mean()), raw_mse=float((a-b).square().mean()),
                normalized_l1_channels=delta.abs().mean(0).tolist(),
                normalized_mse_channels=delta.square().mean(0).tolist(), normalized_l1=float(delta.abs().mean()),
                normalized_mse=float(delta.square().mean()),
                xyz_distance=float((xyz_a - xyz_b).norm(dim=-1).mean()),
                xyz_distance_normalized=float((xyz_a - xyz_b).norm(dim=-1).mean() / xyz_scale),
                correlation=correlation(a, b))


def references(predictor, indices, visible, points):
    """Exact spatial input locations, including convolution masking and boundaries.

    valid=False means a zero canvas value/padding, not a decoded anchor. Pair
    predictors duplicate the left input at the right boundary and retain both
    slots because their learned coefficients may differ.
    """
    indices, visible = indices.cpu(), visible.reshape(-1).cpu()
    if predictor is None:
        empty = torch.empty(len(indices), 0, dtype=torch.long)
        return dict(indices=empty, valid=empty.bool(), weight_l1=torch.empty(0),
                    boundary=torch.zeros(len(indices), dtype=torch.bool), kind="none")
    if isinstance(predictor, AnchorPairPredictor):
        left = indices // predictor.spacing * predictor.spacing
        right = left + predictor.spacing
        boundary = right >= points
        right = torch.where(boundary, left, right)
        ref = torch.stack((left, right), 1)
        if not visible[ref].all():
            raise AssertionError("pair predictor reads an undecoded anchor")
        phase = indices % predictor.spacing - 1 if predictor.phase_specific else torch.zeros_like(indices)
        norms = cpu(predictor.weight).abs().sum((1, 2))[phase]
        return dict(indices=ref, valid=torch.ones_like(ref, dtype=torch.bool),
                    weight_l1=norms, boundary=boundary, kind="anchor_pair")
    if not isinstance(predictor, torch.nn.Conv2d) or predictor.stride != (1, 1):
        raise ValueError("unsupported spatial predictor for reference tracing")
    offsets = torch.arange(predictor.kernel_size[-1]) * predictor.dilation[-1] - predictor.padding[-1]
    ref = indices[:, None] + offsets[None]
    in_range = (ref >= 0) & (ref < points)
    valid = in_range & visible[ref.clamp(0, points - 1)]
    norms = cpu(predictor.weight).abs().sum((0, 1, 2))
    return dict(indices=ref, valid=valid, weight_l1=norms,
                boundary=~in_range.all(1), kind="masked_canvas_convolution")


def schedule_references(schedule, points):
    """Reference topology with no learned score/predictor change or quantization."""
    ids = torch.arange(points)
    if schedule == "legacy3":
        stages, spacings = (ids[::4], ids[2::4], ids[1::2]), (None, 4, 2)
    elif schedule == "quarter2":
        stages, spacings = (ids[::4], ids[ids % 4 != 0]), (None, 4)
    elif schedule == "dyadic4":
        stages, spacings = (ids[::8], ids[4::8], ids[2::4], ids[1::2]), (None, 8, 4, 2)
    else:
        raise ValueError(schedule)
    for number, (target, spacing) in enumerate(zip(stages, spacings)):
        if spacing is None:
            ref, valid = torch.empty(len(target), 0, dtype=torch.long), torch.empty(len(target), 0, dtype=torch.bool)
        else:
            left = target // spacing * spacing
            right = left + spacing
            ref = torch.stack((left, right), 1)
            valid = ref < points
            if schedule != "legacy3":
                ref[:, 1] = torch.where(valid[:, 1], right, left)
                valid[:] = True
        yield number, target, ref, valid


@torch.no_grad()
def fixed_score_relations(score, positions, *, bits=10, seed=0, scene="", random_order=True,
                          backend="knn", candidates=64, distance_device="input"):
    """N×rank scores already in quantization units; XYZ and values stay fixed.

    NN uses the deployed scene-mean-absolute weighting (floor .05). Reporting
    uses a separate common STD normalization, not an order-dependent scale.
    """
    positions = positions.float()
    if positions.shape != (len(score), 3):
        raise ValueError("score/XYZ shapes differ")
    device, points = score.device, len(score)
    permutations = dict(none=torch.arange(points, device=device),
        morton=morton_order_3d(positions[None], bits).permutation[0],
        hilbert=hilbert_order_3d(positions[None], bits).permutation[0])
    weights = score.abs().mean(0, keepdim=True).clamp_min(.05)
    permutations["nn_score"] = greedy_neighbor_order((score / weights)[None], positions[None], bits=bits,
        metric="l1", backend=backend, candidates=candidates, distance_device=distance_device).permutation[0]
    if random_order:
        local_seed = int.from_bytes(hashlib.sha256(f"{seed}:{scene}".encode()).digest()[:8], "little") % (2**63)
        generator = torch.Generator().manual_seed(local_seed)
        permutations["random"] = torch.randperm(points, generator=generator).to(device)
    score_cpu, xyz, scale = cpu(score), cpu(positions), cpu(channel_scale(score))
    xyz_scale = (xyz - xyz.mean(0)).square().sum(-1).mean().sqrt().clamp_min(1e-6)
    result = {}
    for name, permutation in permutations.items():
        permutation = permutation.cpu()
        values, centers = score_cpu[permutation], xyz[permutation]
        lags = {str(lag): relation(values[:-lag], values[lag:], centers[:-lag], centers[lag:], scale, xyz_scale)
                for lag in (1, 2, 4, 8) if lag < points}
        topology = {}
        for schedule in ("legacy3", "quarter2", "dyadic4"):
            topology[schedule] = []
            for stage, target, ref, valid in schedule_references(schedule, points):
                targets = target[:, None].expand_as(ref)[valid]
                refs = ref[valid]
                stats = relation(values[targets], values[refs], centers[targets], centers[refs], scale, xyz_scale)
                topology[schedule].append(dict(stage=stage, target_tokens=len(target), **stats))
        result[name] = dict(lags=lags, topology=topology)
    return result


class StageRecorder:
    """Observe the true coding pass; keep raw CPU tensors only when requested."""

    def __init__(self, score, positions, keep_tensors=False):
        self.score = rows(score)
        self.positions = cpu(positions)[0]
        self.xyz_scale = (self.positions-self.positions.mean(0)).square().sum(-1).mean().sqrt().clamp_min(1e-6)
        self.scale = channel_scale(self.score)
        self.reconstruction = torch.empty_like(self.score)
        self.groups, self.tensors = [], []
        self.keep_tensors = keep_tensors

    @torch.no_grad()
    def __call__(self, event):
        start, width = event["start"], event["width"]
        indices = event["indices"].cpu()
        target, prediction, baseline, reconstructed = (rows(event[key]) for key in
            ("target", "prediction", "baseline", "reconstruction"))
        if not torch.equal(target, self.score[indices, start:start + width]):
            raise AssertionError("prepared Score and actual coding targets differ")
        scale = self.scale[start:start + width]
        raw_residual, raw_baseline = target - prediction, target - baseline
        residual = raw_residual / scale
        baseline_residual = raw_baseline / scale
        quant_error = (target - reconstructed) / scale
        edge = references(event["predictor"], indices, event["visible"], len(self.score))
        ref, valid = edge["indices"], edge["valid"]
        context = rows(event["context"])
        targets = indices[:, None].expand_as(ref)[valid]
        refs = ref[valid]
        anchor_relation = relation(self.score[targets, start:start + width], context[refs],
            self.positions[targets], self.positions[refs], scale, self.xyz_scale)
        _, likelihood = event["entropy"](event["symbols"], training=False)
        estimated_bits = float(-likelihood.clamp_min(1e-9).log2().sum())
        count = target.numel()
        row = dict(stage=event["stage"], name=event["name"], group=event["group"], start=start, width=width,
            tokens=len(indices), symbols=count, raw_bytes=event["raw_bytes"], estimated_bits=estimated_bits,
            raw_abs_error_sum=float(raw_residual.abs().sum()), raw_sq_error_sum=float(raw_residual.square().sum()),
            baseline_raw_abs_error_sum=float(raw_baseline.abs().sum()),
            baseline_raw_sq_error_sum=float(raw_baseline.square().sum()),
            normalized_abs_error_sum=float(residual.abs().sum()), normalized_sq_error_sum=float(residual.square().sum()),
            baseline_normalized_abs_error_sum=float(baseline_residual.abs().sum()),
            baseline_normalized_sq_error_sum=float(baseline_residual.square().sum()),
            normalized_quant_sq_error_sum=float(quant_error.square().sum()),
            residual_normalized_mae_channels=residual.abs().mean(0).tolist(),
            residual_normalized_mse_channels=residual.square().mean(0).tolist(),
            residual_raw_mae_channels=raw_residual.abs().mean(0).tolist(),
            residual_raw_mse_channels=raw_residual.square().mean(0).tolist(),
            baseline_normalized_mae_channels=baseline_residual.abs().mean(0).tolist(),
            baseline_normalized_mse_channels=baseline_residual.square().mean(0).tolist(),
            quantization_normalized_mse_channels=quant_error.square().mean(0).tolist(),
            prediction_correlation=correlation(target, prediction), anchor_relation=anchor_relation,
            reference_kind=edge["kind"], boundary_tokens=int(edge["boundary"].sum()),
            coding_padding_tokens=event["coding_padding"],
            residual_quantization_units_mae=float(rows(event["symbols"]).abs().mean()))
        self.groups.append(row)
        self.reconstruction[indices, start:start + width] = reconstructed
        if self.keep_tensors:
            self.tensors.append(dict(stage=event["stage"], group=event["group"], start=start,
                indices=indices, references=ref, reference_valid=valid, reference_weight_l1=edge["weight_l1"],
                boundary=edge["boundary"], reference_inputs=torch.where(valid[:, :, None],
                    context[ref.clamp(0, len(self.score)-1)], torch.zeros(len(ref), ref.shape[1], width)),
                target=target, prediction=prediction, baseline=baseline, reconstruction=reconstructed,
                entropy_input=rows(event["symbols"]), step=cpu(event["step"]),
                median=cpu(event["entropy"].entropy_bottleneck._get_medians())))

    def summary(self):
        stages = []
        for index in sorted({g["stage"] for g in self.groups}):
            groups = [g for g in self.groups if g["stage"] == index]
            counts = {g["tokens"] for g in groups}
            if len(counts) != 1 or sum(g["width"] for g in groups) != self.score.shape[1]:
                raise AssertionError("incomplete stage capture")
            tokens = counts.pop()
            symbols = sum(g["symbols"] for g in groups)
            total = lambda name: sum(g[name] for g in groups)
            raw_bytes, estimated_bits = total("raw_bytes"), total("estimated_bits")
            baseline = total("baseline_normalized_sq_error_sum")
            stages.append(dict(stage=index, name=groups[0]["name"], tokens=tokens, symbols=symbols,
                raw_bytes=raw_bytes, bits_per_token=8 * raw_bytes / tokens,
                bits_per_symbol=8 * raw_bytes / symbols, estimated_bits=estimated_bits,
                coding_gap_bits_per_token=(8 * raw_bytes - estimated_bits) / tokens,
                raw_mae=total("raw_abs_error_sum") / symbols, raw_mse=total("raw_sq_error_sum") / symbols,
                baseline_raw_mae=total("baseline_raw_abs_error_sum") / symbols,
                baseline_raw_mse=total("baseline_raw_sq_error_sum") / symbols,
                normalized_mae=total("normalized_abs_error_sum") / symbols,
                normalized_mse=total("normalized_sq_error_sum") / symbols,
                baseline_normalized_mae=total("baseline_normalized_abs_error_sum") / symbols,
                baseline_normalized_mse=baseline / symbols,
                prediction_mse_ratio=total("normalized_sq_error_sum") / baseline if baseline else None,
                quantization_normalized_mse=total("normalized_quant_sq_error_sum") / symbols,
                groups=groups))
        if sum(s["tokens"] for s in stages) != len(self.score):
            raise AssertionError("stages did not partition every token")
        return stages


@contextmanager
def observe_stages(context, recorder, prediction_off=False):
    previous, enabled = context.stage_observer, context.spatial_context
    if previous is not None:
        raise ValueError("another stage observer is installed")
    context.stage_observer = recorder
    if prediction_off:
        context.spatial_context = False
    try:
        yield
    finally:
        context.stage_observer, context.spatial_context = previous, enabled


def payload_components(data):
    scene = SceneBitstream.unpack(data)
    score = ScoreContextBitstream.unpack(scene.score)
    original = scene.bytes_by_stream
    raw = sum(map(len, score.strings))
    components = dict(score_raw=raw, msh_y=original["residual_y"], msh_z=original["residual_z"],
                      mean=original["mean"], headers=original["container"] + len(scene.score) - raw)
    if sum(components.values()) != len(data):
        raise AssertionError("byte accounting does not sum to packet size")
    return dict(total_bytes=len(data), score_total_bytes=len(scene.score), score_header_bytes=len(scene.score)-raw,
                components=components)

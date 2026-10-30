"""Full P0+Split codec configuration, including saved checkpoint metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CodecConfig:
    """Architecture values for the production NFC-GS codec path."""

    texture_channels: int = 512
    geometry_channels: int = 512
    geometry_observable_channels: int = 224
    rank: int = 56
    residual_n: int = 192
    residual_m: int = 320
    adapter_hidden: int = 96
    morton_bits: int = 10
    use_morton: bool = True
    token_order: str = "morton"
    score_order_scale: tuple[float, ...] | None = None
    score_order_metric: str = "l1"
    # Encoder implementation switches, independent of weights/bitstream format.
    nn_order_backend: str = "legacy"
    nn_order_candidates: int = 64
    nn_order_distance_device: str = "cpu"
    use_residual: bool = True
    codec_mode: str = "two_path"
    use_centering: bool = True
    use_score_norm: bool = True
    transform: str = "nonlinear"
    transform_hidden: int = 32
    score_mean_condition: bool = True
    score_channel_context: bool = True
    score_spatial_context: bool = True
    score_spatial_predictor: str = "linear"
    score_spatial_entropy: str = "split"
    score_spatial_hidden: int = 32
    score_slice_channels: int = 16
    score_context_hidden: int = 64
    score_spatial_stages: int = 2
    score_spatial_kernel: int = 3
    score_context_quantization: str = "noise"
    score_context_schedule: str = "legacy"

    def __post_init__(self) -> None:
        if self.codec_mode not in ("two_path", "score_only", "msh_only"):
            raise ValueError("codec_mode must be two_path, score_only or msh_only")
        if type(self.use_residual) is not bool or self.use_residual != (self.codec_mode != "score_only"):
            raise ValueError("main codec requires use_residual=True; false is supported only with explicit codec_mode=score_only")
        if self.codec_mode != "two_path":
            if self.use_centering or self.use_score_norm or self.score_mean_condition or self.score_channel_context or self.transform != "linear":
                raise ValueError("single-path pilots require the minimal linear configuration")
        if self.codec_mode == "msh_only":
            if self.rank != 0 or self.score_spatial_context or self.token_order == "nn_score":
                raise ValueError("msh_only requires rank=0, spatial context OFF and no nn_score order")
        # The historical two-path configuration remains the default.
        required = {
            "score_spatial_predictor": "linear",
            "score_spatial_entropy": "split",
        }
        for name, expected in required.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"main codec requires {name}={expected!r}; got {actual!r}. "
                    "Other codec paths have been archived."
                )
        for name in ("use_morton", "score_spatial_context"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.token_order not in ("none", "morton", "hilbert", "nn_xyz", "nn_score"):
            raise ValueError("token_order must be none, morton, hilbert, nn_xyz or nn_score")
        # Canonicalize the historical enable flag and the explicit identity order.
        if self.token_order == "none":
            object.__setattr__(self, "use_morton", False)
        elif not self.use_morton:
            if self.token_order != "morton":
                raise ValueError("use_morton=false requires token_order=none or the legacy morton default")
            object.__setattr__(self, "token_order", "none")
        if self.nn_order_backend not in ("legacy", "row", "knn", "graph", "graph_gemm"):
            raise ValueError("nn_order_backend must be legacy, row, knn, graph or graph_gemm")
        if type(self.nn_order_candidates) is not int or self.nn_order_candidates < 1:
            raise ValueError("nn_order_candidates must be a positive integer")
        if self.nn_order_distance_device not in ("cpu", "input"):
            raise ValueError("nn_order_distance_device must be cpu or input")
        if self.nn_order_backend == "legacy" and self.nn_order_distance_device != "cpu":
            raise ValueError("legacy NN backend requires nn_order_distance_device=cpu")
        if self.token_order not in ("nn_xyz", "nn_score") and self.nn_order_backend != "legacy":
            raise ValueError("optimized NN backend requires token_order=nn_xyz or nn_score")
        if self.score_order_scale is not None:
            scale = tuple(float(value) for value in self.score_order_scale)
            if len(scale) != self.rank or any(not math.isfinite(v) or v <= 0 for v in scale):
                raise ValueError("score_order_scale must contain rank finite positive values")
            if self.token_order != "nn_score":
                raise ValueError("score_order_scale is only used with token_order=nn_score")
            object.__setattr__(self, "score_order_scale", scale)
        if self.score_order_metric not in ("l1", "l2"):
            raise ValueError("score_order_metric must be l1 or l2")
        if self.score_order_metric != "l1" and self.token_order != "nn_score":
            raise ValueError("score_order_metric is only used with token_order=nn_score")
        if self.score_context_schedule not in ("legacy", "quarter2", "dyadic4", "single"):
            raise ValueError("score_context_schedule must be legacy, quarter2, dyadic4 or single")
        for name in ("use_centering", "use_score_norm", "score_mean_condition", "score_channel_context"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.transform not in ("linear", "nonlinear"):
            raise ValueError("transform must be linear or nonlinear")
        if not self.use_centering and self.score_mean_condition:
            raise ValueError("use_centering=false requires score_mean_condition=false (no mean is transmitted)")
        if type(self.score_spatial_stages) is not int or self.score_spatial_stages not in (1, 2, 3, 4):
            raise ValueError("score_spatial_stages must be 1, 2, 3 or 4")
        if self.score_spatial_stages == 1 and self.score_context_schedule != "single":
            raise ValueError("one token stage requires score_context_schedule=single")
        if self.score_context_schedule == "single" and self.score_spatial_context:
            raise ValueError("single requires score_spatial_context=false")
        if type(self.score_spatial_kernel) is not int or self.score_spatial_kernel not in (3, 5, 7):
            raise ValueError("score_spatial_kernel must be 3, 5 or 7")
        if self.score_context_quantization not in ("noise", "ste"):
            raise ValueError("score_context_quantization must be noise or ste")
        if self.score_context_schedule != "legacy":
            expected_stages = {"single": 1, "quarter2": 2, "dyadic4": 4}[self.score_context_schedule]
            if self.score_spatial_stages != expected_stages:
                raise ValueError(f"{self.score_context_schedule} requires {expected_stages} spatial stages")
            if self.score_channel_context:
                raise ValueError("anchor schedules require score_channel_context=false")
        if self.score_spatial_stages > 2 and self.score_context_schedule == "legacy":
            if self.score_spatial_kernel not in (5, 7):
                raise ValueError("3/4 spatial stages require kernel 5 or 7 to see distance-2 context")
            if self.score_channel_context:
                raise ValueError("3/4 token stages require score_channel_context=false")
        for name in (
            "texture_channels",
            "geometry_channels",
            "geometry_observable_channels",
            "rank",
            "residual_n",
            "residual_m",
            "adapter_hidden",
            "transform_hidden",
            "score_spatial_hidden",
            "score_slice_channels",
            "score_context_hidden",
        ):
            value = getattr(self, name)
            if name == "rank" and self.codec_mode == "msh_only" and type(value) is int and value == 0:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.adapter_hidden % 3:
            raise ValueError("adapter_hidden must be divisible by 3")
        if type(self.morton_bits) is not int or not 1 <= self.morton_bits <= 21:
            raise ValueError("morton_bits must be between 1 and 21")
        if self.codec_mode != "msh_only" and self.score_slice_channels > self.rank:
            raise ValueError("score_slice_channels cannot exceed rank")

    @property
    def observable_channels(self) -> int:
        return self.texture_channels + self.geometry_observable_channels

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CodecConfig":
        """Build from the YAML spelling used by the codec configuration."""

        aliases = {
            "residual_N": "residual_n",
            "residual_M": "residual_m",
        }
        fields = cls.__dataclass_fields__
        normalized: dict[str, Any] = {}
        for key, value in dict(values).items():
            target = aliases.get(key, key)
            if target not in fields:
                raise ValueError(f"unsupported feature_codec setting: {key}")
            normalized[target] = value
        return cls(**normalized)

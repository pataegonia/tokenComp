"""Deterministic, index-free sampler for evaluating every dataset scene."""

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerDeterministicAllCfg:
    name: Literal["deterministic_all"]
    num_context_views: int
    num_target_views: int


def _farthest_camera_indices(positions: Tensor, count: int) -> Tensor:
    """Unique deterministic farthest-point indices, returned in frame order."""

    total = positions.shape[0]
    if count > total:
        raise ValueError(f"requested {count} views from a {total}-view scene")
    center = positions.mean(dim=0, keepdim=True)
    selected = torch.empty(count, dtype=torch.long, device=positions.device)
    distance = torch.full((total,), torch.inf, device=positions.device)
    current = ((positions - center) ** 2).sum(dim=-1).argmax()
    used = torch.zeros(total, dtype=torch.bool, device=positions.device)
    for index in range(count):
        selected[index] = current
        used[current] = True
        candidate_distance = ((positions - positions[current]) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, candidate_distance)
        distance[used] = -1
        if index + 1 < count:
            current = distance.argmax()
    return selected.sort().values


class ViewSamplerDeterministicAll(ViewSampler[ViewSamplerDeterministicAllCfg]):
    """Select fixed-size, disjoint views without requiring a scene index JSON.

    Context views are camera-space farthest-point samples over the full clip.
    Targets are another farthest-point sample from the remaining frames. This
    makes every processable RE10K test scene deterministic while keeping target
    images completely unseen by the encoder.
    """

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[Int64[Tensor, " context_view"], Int64[Tensor, " target_view"]]:
        del scene, intrinsics
        context_count = int(self.cfg.num_context_views)
        target_count = int(self.cfg.num_target_views)
        total = extrinsics.shape[0]
        if total < context_count + target_count:
            raise ValueError(
                f"scene has {total} frames, but {context_count + target_count} "
                "disjoint context/target frames are required"
            )

        positions = extrinsics[:, :3, 3].to(device=device)
        context = _farthest_camera_indices(positions, context_count)
        available_mask = torch.ones(total, dtype=torch.bool, device=device)
        available_mask[context] = False
        available = torch.arange(total, device=device)[available_mask]
        target_local = _farthest_camera_indices(positions[available], target_count)
        target = available[target_local].sort().values
        return context, target

    @property
    def num_context_views(self) -> int:
        return int(self.cfg.num_context_views)

    @property
    def num_target_views(self) -> int:
        return int(self.cfg.num_target_views)

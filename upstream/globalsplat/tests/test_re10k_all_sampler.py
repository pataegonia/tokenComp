from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

_MVSPLAT = Path(__file__).resolve().parents[1] / "third_party/ZPressor/mvsplat"
sys.path.insert(0, str(_MVSPLAT))

from src.dataset.view_sampler.view_sampler_deterministic_all import (  # noqa: E402
    ViewSamplerDeterministicAll,
    ViewSamplerDeterministicAllCfg,
)


def test_all_scene_sampler_is_deterministic_fixed_size_and_disjoint():
    cfg = ViewSamplerDeterministicAllCfg(
        name="deterministic_all", num_context_views=12, num_target_views=8
    )
    sampler = ViewSamplerDeterministicAll(cfg, "test", False, False, None)
    extrinsics = torch.eye(4).repeat(60, 1, 1)
    extrinsics[:, 0, 3] = torch.linspace(-2, 3, 60)
    extrinsics[:, 1, 3] = torch.sin(torch.linspace(0, 6, 60))
    intrinsics = torch.eye(3).repeat(60, 1, 1)

    context_a, target_a = sampler.sample("scene", extrinsics, intrinsics)
    context_b, target_b = sampler.sample("scene", extrinsics, intrinsics)
    assert torch.equal(context_a, context_b)
    assert torch.equal(target_a, target_b)
    assert len(context_a) == 12
    assert len(target_a) == 8
    assert not set(context_a.tolist()) & set(target_a.tolist())


def test_all_scene_sampler_rejects_too_short_scenes():
    cfg = ViewSamplerDeterministicAllCfg(
        name="deterministic_all", num_context_views=12, num_target_views=8
    )
    sampler = ViewSamplerDeterministicAll(cfg, "test", False, False, None)
    with pytest.raises(ValueError, match="disjoint"):
        sampler.sample("short", torch.eye(4).repeat(19, 1, 1), torch.eye(3).repeat(19, 1, 1))

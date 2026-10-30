"""Explicit single-path pilots and shared, fresh codec initialization."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import torch

from .codec import ObservableLowRank1DCodec
from .config import CodecConfig

PILOT_ARMS = ("score56_only", "msh_only", "two_path56", "score736_only")


def pilot_config(arm: str) -> CodecConfig:
    base = CodecConfig(token_order="hilbert", score_context_schedule="dyadic4", score_spatial_stages=4,
                       score_spatial_kernel=5)
    if arm == "two_path56":
        return base
    if arm == "score56_only":
        return replace(base, codec_mode="score_only", use_residual=False)
    if arm == "score736_only":
        return replace(base, codec_mode="score_only", use_residual=False,
                       rank=736, score_slice_channels=184)
    if arm == "msh_only":
        return replace(base, codec_mode="msh_only", rank=0, score_spatial_context=False)
    raise ValueError(f"unknown pilot arm: {arm}")


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def shared_fresh_codec_state(seed: int = 111123) -> dict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        codec = ObservableLowRank1DCodec(pilot_config("two_path56"))
    return {key: value.detach().clone() for key, value in codec.state_dict().items()}


@torch.no_grad()
def apply_shared_fresh_state(codec: ObservableLowRank1DCodec, common: dict) -> dict[str, str]:
    """Copy equal-shape shared modules rather than relying on RNG call order."""
    state = codec.state_dict()
    hashes = {}
    for key, value in state.items():
        if key in common and value.shape == common[key].shape:
            state[key] = common[key].detach().clone()
            hashes[key] = tensor_hash(state[key])
    codec.load_state_dict(state, strict=True)
    return hashes

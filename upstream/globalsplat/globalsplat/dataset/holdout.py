"""Opt-in train-derived architecture validation; never the default val/test split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_chunk_path(root: Path, value: str) -> Path:
    train = (root / "train").resolve()
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (train / candidate).resolve()
    if not candidate.is_relative_to(train) or candidate.suffix != ".torch":
        raise ValueError(f"training index chunk escapes train directory: {value}")
    return candidate


def read_holdout(path: str | Path, dataset_root: str | Path) -> dict:
    root = Path(dataset_root).resolve()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") != "nfcgs_train_holdout_v1":
        raise ValueError("unsupported holdout manifest")
    for split in ("train", "test"):
        if file_hash(root / split / "index.json") != data[f"{split}_index_sha256"]:
            raise ValueError(f"{split} index does not match frozen holdout manifest")
    train = json.loads((root / "train/index.json").read_text())
    test = json.loads((root / "test/index.json").read_text())
    scenes = data["scenes"]
    if len(scenes) != data["scene_count"] or not scenes or set(scenes) & set(test) or not set(scenes) <= set(train):
        raise ValueError("holdout scene membership/count is invalid or overlaps test")
    for name, record in scenes.items():
        if record["chunk"] != train[name]:
            raise ValueError("holdout chunk mapping differs from training index")
        if len(record["context"]) != 12 or len(record["target"]) != 8 or len(set(record["context"]+record["target"])) != 20:
            raise ValueError("holdout requires disjoint C12/T8 frames")
        train_chunk_path(root, record["chunk"])
    return data


class ExcludingViewSampler:
    """Skip held-out scenes before image decoding and augmentation."""
    def __init__(self, sampler, scenes):
        self.sampler = sampler
        self.excluded_scenes = frozenset(scenes)

    def sample(self, scene, *args, **kwargs):
        if scene in self.excluded_scenes:
            raise ValueError("scene reserved for architecture validation")
        return self.sampler.sample(scene, *args, **kwargs)

    def __getattr__(self, name):
        if name in ("sampler", "excluded_scenes"):
            raise AttributeError(name)
        return getattr(self.sampler, name)

"""Reproducible scene sampling and early filtering for small test runs."""

import json
from pathlib import Path
import random


def sample_scene_ids(index, count, seed, excluded=()):
    candidates = sorted(set(index) - set(excluded))
    if not 0 < count <= len(candidates):
        raise ValueError(f"requested {count} test scenes, but only {len(candidates)} are available")
    return sorted(random.Random(seed).sample(candidates, count))


def load_scene_subset(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes = value["scenes"] if isinstance(value, dict) else value
    if (not isinstance(scenes, list) or not scenes
            or any(not isinstance(scene, str) for scene in scenes)
            or len(set(scenes)) != len(scenes)):
        raise ValueError("test subset must contain a nonempty list of unique scene IDs")
    return scenes


class SceneSubsetSampler:
    def __init__(self, sampler, scenes):
        self.sampler = sampler
        self.scenes = frozenset(scenes)

    def __getattr__(self, name):
        sampler = self.__dict__.get("sampler")
        if sampler is None:
            raise AttributeError(name)
        return getattr(sampler, name)

    def sample(self, scene, *args, **kwargs):
        if scene not in self.scenes:
            # DatasetRE10k skips rejected scenes before decoding their images.
            raise ValueError(f"scene {scene} is outside the sampled test subset")
        return self.sampler.sample(scene, *args, **kwargs)


def restrict_test_scenes(dataset, scenes):
    missing = set(scenes) - set(dataset.index)
    if missing:
        raise ValueError(f"sampled scenes are missing from the test index: {sorted(missing)}")
    selected_chunks = {Path(dataset.index[scene]).resolve() for scene in scenes}
    dataset.chunks = [chunk for chunk in dataset.chunks if Path(chunk).resolve() in selected_chunks]
    if not dataset.chunks:
        raise ValueError("sampled test scenes have no matching chunks")
    dataset.view_sampler = SceneSubsetSampler(dataset.view_sampler, scenes)


def prepare_test_subset(dataset_root, output, count, seed, validation_manifest):
    index_path = Path(dataset_root) / "test/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    validation = json.loads(Path(validation_manifest).read_text(encoding="utf-8"))
    excluded = {row["scene"] for row in validation}
    scenes = sample_scene_ids(index, count, seed, excluded)
    record = {
        "split": "test", "seed": seed, "requested_scene_count": count,
        "dataset_index": str(index_path.resolve()),
        "excluded_validation_manifest": str(Path(validation_manifest).resolve()),
        "excluded_validation_scenes": sorted(excluded),
        "num_context_views": 12, "num_target_views": 8,
        "scenes": scenes,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text(encoding="utf-8")) != record:
        raise ValueError(f"test sample changed; use a new output directory instead of overwriting {output}")
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return output

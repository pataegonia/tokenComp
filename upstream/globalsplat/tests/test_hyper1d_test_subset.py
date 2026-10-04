"""Test sampling keeps validation separate and filters before image decoding."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from globalsplat.dataset.test_subset import (
    load_scene_subset, prepare_test_subset, restrict_test_scenes, sample_scene_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def test_sampling_is_repeatable_excludes_validation_and_uses_the_whole_pool():
    index = {f"scene{i:03d}": "chunk.torch" for i in range(100)}
    excluded = list(index)[:10]
    chosen = sample_scene_ids(index, 8, 111123, excluded)
    assert chosen == sample_scene_ids(dict(reversed(list(index.items()))), 8, 111123, excluded)
    assert not set(chosen) & set(excluded)
    assert len(set(chosen)) == 8
    assert chosen != sorted(set(index) - set(excluded))[:8]
    with pytest.raises(ValueError, match="available"):
        sample_scene_ids(index, 91, 111123, excluded)


def test_subset_artifact_is_test_only_and_cannot_silently_change(tmp_path):
    root = tmp_path / "data"
    (root / "test").mkdir(parents=True)
    (root / "test/index.json").write_text(json.dumps({str(i): "chunk.torch" for i in range(20)}))
    validation = tmp_path / "manifest.json"
    validation.write_text(json.dumps([{"scene": "0"}, {"scene": "1"}]))
    path = prepare_test_subset(root, tmp_path / "output/test_sample.json", 8, 111123, validation)
    artifact = json.loads(path.read_text())
    assert artifact["split"] == "test" and artifact["seed"] == 111123
    assert not set(load_scene_subset(path)) & {"0", "1"}
    assert len(load_scene_subset(path)) == 8
    prepare_test_subset(root, path, 8, 111123, validation)
    with pytest.raises(ValueError, match="new output"):
        prepare_test_subset(root, path, 9, 111123, validation)


def test_subset_prunes_chunks_and_skips_other_scenes_before_sampling(tmp_path):
    calls = []
    class Sampler:
        num_context_views = 12
        def sample(self, scene, *args, **kwargs):
            calls.append(scene)
            return [0, 1], [2]
    chunks = [tmp_path / f"{i}.torch" for i in range(3)]
    dataset = SimpleNamespace(index={"chosen": chunks[1], "other": chunks[1]},
                              chunks=chunks, view_sampler=Sampler())
    restrict_test_scenes(dataset, ["chosen"])
    assert dataset.chunks == [chunks[1]]
    assert dataset.view_sampler.num_context_views == 12
    with pytest.raises(ValueError, match="outside"):
        dataset.view_sampler.sample("other", None, None)
    assert not calls
    assert dataset.view_sampler.sample("chosen", None, None) == ([0, 1], [2])
    assert calls == ["chosen"]
    with pytest.raises(ValueError, match="missing"):
        restrict_test_scenes(dataset, ["absent"])


def test_sampled_eval_command_composes_without_changing_training(tmp_path):
    from hydra import compose, initialize_config_dir
    spec = importlib.util.spec_from_file_location("run_hyper1d", ROOT / "scripts/run_hyper1d.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    args = runner.parse_args(["eval", "--checkpoint", "codec.ckpt", "--sample-test",
                             "--max-scenes", "32", "--save-images", "--workers", "0"])
    command = runner.build_command(args, tmp_path, Path("codec.ckpt"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=command[3:])
    assert cfg.mode == "test" and cfg.test.max_scenes == 32
    assert cfg.test.scene_subset_path == (tmp_path / "test_sample.json").as_posix()
    assert cfg.dataset.num_context_views == 12 and cfg.dataset.num_target_views == 8
    assert cfg.dataset.eval_all_scenes and cfg.test.actual_bitstream
    assert cfg.test.save_image and cfg.test.save_gt_image
    with pytest.raises(SystemExit):
        runner.parse_args(["train", "--checkpoint", "codec.ckpt", "--sample-test"])


def test_full_test_command_has_no_scene_limit_or_sampling(tmp_path):
    from hydra import compose, initialize_config_dir
    spec = importlib.util.spec_from_file_location("run_hyper1d", ROOT / "scripts/run_hyper1d.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    args = runner.parse_args(["eval", "--checkpoint", "codec.ckpt", "--all-test"])
    command = runner.build_command(args, tmp_path, Path("codec.ckpt"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=command[3:])
    assert cfg.mode == "test" and cfg.test.max_scenes is None
    assert cfg.dataset.eval_all_scenes and cfg.dataset.eval_index_path is None
    assert "scene_subset_path" not in cfg.test
    assert cfg.dataset.num_context_views == 12 and cfg.dataset.num_target_views == 8
    with pytest.raises(SystemExit):
        runner.parse_args(["eval", "--checkpoint", "codec.ckpt", "--all-test", "--sample-test"])

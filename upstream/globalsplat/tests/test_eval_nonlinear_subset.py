"""Regression tests for short scenes in the Nonlinear32 test subset."""

import importlib.util
from io import BytesIO
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_nonlinear_subset", ROOT / "scripts/eval_nonlinear50k_subset.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def image_bytes(size=(640, 360)):
    buffer = BytesIO()
    Image.new("RGB", size).save(buffer, format="JPEG")
    return torch.tensor(list(buffer.getvalue()), dtype=torch.uint8)


def example(scene, frames, image, focal=1.0):
    cameras = torch.zeros(frames, 18)
    cameras[:, :4] = torch.tensor([focal, focal, 0.5, 0.5])
    w2c = torch.eye(4)[:3].repeat(frames, 1, 1)
    w2c[:, 0, 3] = torch.linspace(0, 1, frames)
    cameras[:, 6:] = w2c.reshape(frames, 12)
    return {"key": scene, "cameras": cameras, "images": [image] * frames}


def test_11_and_19_frames_are_rejected_and_20_frames_are_disjoint(tmp_path):
    image = image_bytes()
    torch.save([example("short11", 11, image), example("short19", 19, image),
                example("enough20", 20, image)], tmp_path / "chunk.torch")
    index = {scene: "chunk.torch" for scene in ("short11", "short19", "enough20")}
    rejected = {}
    frames = evaluator.build_frame_index(tmp_path, index, sorted(index), rejected=rejected)
    assert list(frames) == ["enough20"]
    assert rejected["short11"]["num_frames"] == 11
    assert rejected["short19"]["required_frames"] == 20
    assert len(frames["enough20"]["context"]) == 12
    assert len(frames["enough20"]["target"]) == 8
    assert not set(frames["enough20"]["context"]) & set(frames["enough20"]["target"])


def test_replacements_keep_valid_initial_scenes_and_fill_100_repeatably(tmp_path):
    index = {f"scene{i:03d}": "chunk.torch" for i in range(130)}
    validation = {"scene000", "scene001"}
    initial = evaluator.select_scenes(index, validation, 100, 111123)
    bad = initial[:4]
    image, wrong_shape = image_bytes(), image_bytes((320, 240))
    examples = []
    for scene in index:
        count = 11 if scene == bad[0] else 19 if scene == bad[1] else 20
        examples.append(example(scene, count, wrong_shape if scene == bad[3] else image,
                                focal=0.05 if scene == bad[2] else 1.0))
    torch.save(examples, tmp_path / "chunk.torch")
    frames, audit = evaluator.select_processable_scenes(tmp_path, index, validation, 100, 111123)
    repeated, repeated_audit = evaluator.select_processable_scenes(
        tmp_path, dict(reversed(list(index.items()))), validation, 100, 111123,
    )
    assert frames == repeated and audit == repeated_audit
    assert len(frames) == 100
    assert not set(frames) & validation
    assert set(initial) - set(bad) <= set(frames)
    assert not set(frames) & set(bad)
    assert audit["initial_scene_ids"] == initial
    assert {row["scene"] for row in audit["rejected_scenes"]} == set(bad)
    assert audit["attempted_scene_count"] == 104


def test_insufficient_valid_pool_fails_instead_of_returning_a_smaller_set(tmp_path):
    image = image_bytes()
    torch.save([example(str(i), 11, image) for i in range(3)], tmp_path / "chunk.torch")
    with pytest.raises(ValueError, match="cannot fill 2 processable"):
        evaluator.select_processable_scenes(
            tmp_path, {str(i): "chunk.torch" for i in range(3)}, set(), 2, 111123,
        )


def test_selection_only_writes_final_frame_index_without_loading_model(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    test_root = data_root / "test"
    test_root.mkdir(parents=True)
    index = {"validation": "chunk.torch", "short": "chunk.torch", "valid": "chunk.torch"}
    (test_root / "index.json").write_text(json.dumps(index))
    image = image_bytes()
    torch.save([example("validation", 20, image), example("short", 11, image),
                example("valid", 20, image)], test_root / "chunk.torch")
    validation = tmp_path / "manifest.json"
    validation.write_text(json.dumps([{"scene": "validation"}]))
    monkeypatch.setattr(evaluator, "activate_archived_codec",
                        lambda: pytest.fail("selection-only must not load the model"))
    output = tmp_path / "selection"
    evaluator.main(["--checkpoint", "unused.ckpt", "--validation-scenes", str(validation),
                    "--dataset-root", str(data_root), "--count", "1", "--output", str(output),
                    "--selection-only"])
    assert json.loads((output / "scene_ids.json").read_text()) == ["valid"]
    metadata = json.loads((output / "selection.json").read_text())
    assert metadata["selection_version"] == 2
    assert metadata["num_context_views"] == 12 and metadata["num_target_views"] == 8
    frames = json.loads((output / "eval_index.json").read_text())
    assert len(frames["valid"]["context"]) == 12 and len(frames["valid"]["target"]) == 8

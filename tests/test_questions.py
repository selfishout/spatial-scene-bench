import json

import numpy as np
import pytest

from spatial_scene_bench.camera import default_camera
from spatial_scene_bench.evaluate import aggregate, mean_relative_accuracy, parse_answer, score
from spatial_scene_bench.generate import build_dataset
from spatial_scene_bench.questions import EGO_OPTIONS, QuestionGenerator
from spatial_scene_bench.render import render
from spatial_scene_bench.scene import Scene, SceneObject


def obj(i, shape, color, pos, size="small"):
    r = 0.30 if size == "small" else 0.55
    return SceneObject(i, shape, color, size, "rubber", (pos[0], pos[1], r))


@pytest.fixture
def line_scene():
    # Default camera sits at -y looking toward +y, so world +x is image-right.
    return Scene("line", [
        obj(0, "sphere", "red", (-2.0, 0.0)),
        obj(1, "cube", "blue", (0.0, 0.0)),
        obj(2, "cylinder", "green", (2.0, 0.0)),
        obj(3, "cube", "yellow", (0.0, 2.0)),
    ])


def test_unique_description(line_scene):
    assert line_scene.unique_description(line_scene.objects[0]) == "sphere"
    assert line_scene.unique_description(line_scene.objects[1]) == "blue cube"


def test_camera_relations(line_scene):
    cam = default_camera()
    g = QuestionGenerator(line_scene, np.random.default_rng(0))
    o = line_scene.objects
    assert g._cam_relation(cam, o[0], o[1])["left"] > 0
    assert g._cam_relation(cam, o[2], o[1])["right"] > 0
    assert g._cam_relation(cam, o[1], o[3])["front"] > 0


def test_ego_direction_answer(line_scene):
    g = QuestionGenerator(line_scene, np.random.default_rng(0))
    for _ in range(50):
        q = g.ego_direction(None, line_scene.objects)
        if q is None:
            continue
        m = q.metadata
        expected = f"{'front' if m['forward'] > 0 else 'back'}-{'right' if m['side'] > 0 else 'left'}"
        assert EGO_OPTIONS["ABCD".index(q.answer)] == expected
    # Standing at the red sphere (-2,0) facing the green cylinder (+x): the yellow cube (0,2) is front-left.
    o = line_scene.objects
    fwd = np.array([1.0, 0.0])
    rel = (o[3].center - o[0].center)[:2]
    assert rel @ fwd > 0 and rel @ np.array([fwd[1], -fwd[0]]) < 0


def test_count_respects_occlusion():
    scene = Scene("occ", [obj(0, "cube", "red", (0.0, 0.0), "large"), obj(1, "sphere", "blue", (0.0, 1.6), "small")])
    cam = default_camera()
    pixels = render(scene, cam).visible_pixels()
    g = QuestionGenerator(scene, np.random.default_rng(0))
    for _ in range(30):
        q = g.count(0, pixels)
        if q is None:
            continue
        n_expected = sum(pixels.get(o.id, 0) >= g.cfg.min_pixels for o in scene.objects if all(
            getattr(o, k) == v for k, v in q.metadata["filter"].items()))
        assert int(q.answer) == n_expected


def test_build_dataset_end_to_end(tmp_path):
    recs = build_dataset(tmp_path, num_scenes=2, video_frames=4, width=160, height=120, seed=3)
    assert recs
    lines = (tmp_path / "questions.jsonl").read_text().splitlines()
    assert len(lines) == len(recs)
    for r in map(json.loads, lines):
        for p in r["images"]:
            assert (tmp_path / p).exists()
        if r["answer_type"] == "choice":
            assert r["answer"] in "ABCD"[: len(r["options"])]
    types = {r["type"] for r in recs}
    assert {"count", "relation", "video_count"} <= types


@pytest.mark.parametrize("text,at,expected", [
    ("<think>three cubes</think><answer>3</answer>", "count", "3"),
    ("There are five red objects.", "count", "5"),
    ("Answer: no", "yesno", "no"),
    ("(C) front-left", "choice", "C"),
    ("The answer is B.", "choice", "B"),
    ("I think it is back-right", "choice", "D"),
    ("about 2.35 meters", "numeric", "2.35"),
    ("no idea", "count", None),
])
def test_parse_answer(text, at, expected):
    assert parse_answer(text, at, EGO_OPTIONS) == expected


def test_mra():
    assert mean_relative_accuracy(2.0, 2.0) == 1.0
    assert mean_relative_accuracy(10.0, 2.0) == 0.0
    assert 0 < mean_relative_accuracy(2.3, 2.0) < 1


def test_score_and_aggregate():
    rec = {"answer_type": "count", "answer": "4", "type": "count"}
    s = score(rec, "4")
    assert s["acc"] == 1.0 and s["mra"] == 1.0
    table = aggregate([{"type": "count", "scores": s}, {"type": "count", "scores": score(rec, None)}])
    assert table["count"]["acc"] == 0.5

"""Template-based question generation with exact, ambiguity-filtered answers.

Image questions (single view):
    count               "How many large metal objects are there?"
    count_relation      "How many cubes are to the left of the red sphere?"
    relation            "Is the small cylinder behind the blue cube?"
    compare_count       "Are there more spheres than cylinders?"
    closest             "Which object is closest to the gray cube?" (multiple choice)
    distance            "What is the distance between A and B in meters?" (numeric, 3D)
    ego_direction       "If you stand at A facing B, where is C?" (multiple choice)

Video questions (orbit around the scene):
    video_count         count each physical object once across frames
    video_distance      metric 3D distance
    video_ego_direction allocentric -> egocentric direction
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera
from .scene import ATTRIBUTE_ORDER, COLORS, MATERIALS, PLURAL, SHAPES, SIZES, Scene, SceneObject

EGO_OPTIONS = ["front-left", "front-right", "back-left", "back-right"]
LETTERS = "ABCD"

ATTRIBUTE_VALUES = {
    "size": list(SIZES),
    "color": list(COLORS),
    "material": list(MATERIALS),
    "shape": list(SHAPES),
}

RELATIONS = {
    "left": "to the left of",
    "right": "to the right of",
    "front": "in front of",
    "behind": "behind",
}
OPPOSITE = {"left": "right", "right": "left", "front": "behind", "behind": "front"}


@dataclass
class Question:
    id: str
    scene_id: str
    type: str
    modality: str
    question: str
    answer: str
    answer_type: str  # count | yesno | choice | numeric
    options: list[str] = field(default_factory=list)
    view: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Filter:
    attrs: dict[str, str]

    def matches(self, obj: SceneObject) -> bool:
        return all(getattr(obj, k) == v for k, v in self.attrs.items())

    def phrase(self) -> str:
        words = [self.attrs[a] for a in ATTRIBUTE_ORDER if a in self.attrs and a != "shape"]
        words.append(PLURAL[self.attrs.get("shape", "object")])
        return " ".join(words)


@dataclass
class GeneratorConfig:
    min_pixels: int = 40
    lr_margin_px: float = 18.0
    depth_margin: float = 0.4
    distance_margin: float = 0.35
    ego_margin: float = 0.35


class QuestionGenerator:
    def __init__(self, scene: Scene, rng: np.random.Generator, config: GeneratorConfig | None = None):
        self.scene = scene
        self.rng = rng
        self.cfg = config or GeneratorConfig()
        self._counter = 0

    # ------------------------------------------------------------------ helpers
    def _qid(self) -> str:
        self._counter += 1
        return f"{self.scene.id}_q{self._counter:03d}"

    def _visible(self, pixels: dict[int, int]) -> list[SceneObject]:
        return [o for o in self.scene.objects if pixels.get(o.id, 0) >= self.cfg.min_pixels]

    def _ambiguous(self, objs, pixels: dict[int, int]) -> bool:
        return any(0 < pixels.get(o.id, 0) < self.cfg.min_pixels for o in objs)

    def _random_filter(self, pool: list[SceneObject], max_attrs: int = 2) -> Filter:
        k = int(self.rng.integers(0, max_attrs + 1))
        keys = list(self.rng.choice(ATTRIBUTE_ORDER, size=k, replace=False)) if k else []
        if pool and self.rng.random() > 0.1:
            anchor = pool[int(self.rng.integers(len(pool)))]
            return Filter({key: getattr(anchor, key) for key in keys})
        return Filter({key: str(self.rng.choice(ATTRIBUTE_VALUES[key])) for key in keys})

    def _describable(self, objs: list[SceneObject]) -> list[tuple[SceneObject, str]]:
        out = []
        for o in objs:
            d = self.scene.unique_description(o)
            if d is not None:
                out.append((o, d))
        return out

    @staticmethod
    def _cam_relation(cam: Camera, a: SceneObject, b: SceneObject) -> dict[str, float]:
        """Signed margins for 'a <rel> b' in the camera view (positive means true)."""
        uv, z = cam.project(np.stack([a.center, b.center]))
        return {
            "left": float(uv[1, 0] - uv[0, 0]),
            "right": float(uv[0, 0] - uv[1, 0]),
            "front": float(z[1] - z[0]),
            "behind": float(z[0] - z[1]),
        }

    def _margin_ok(self, rel: str, value: float) -> bool:
        m = self.cfg.lr_margin_px if rel in ("left", "right") else self.cfg.depth_margin
        return abs(value) >= m

    def _mc(self, correct: str, distractors: list[str]) -> tuple[list[str], str]:
        opts = [correct] + distractors[:3]
        order = self.rng.permutation(len(opts))
        opts = [opts[i] for i in order]
        return opts, LETTERS[opts.index(correct)]

    @staticmethod
    def format_options(options: list[str]) -> str:
        return " ".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))

    # ------------------------------------------------------------ image types
    def count(self, view: int, pixels: dict[int, int]) -> Question | None:
        visible = self._visible(pixels)
        f = self._random_filter(visible)
        targets = [o for o in self.scene.objects if f.matches(o)]
        if self._ambiguous(targets, pixels):
            return None
        n = sum(pixels.get(o.id, 0) >= self.cfg.min_pixels for o in targets)
        return Question(self._qid(), self.scene.id, "count", "image",
                        f"How many {f.phrase()} are there?", str(n), "count", view=view,
                        metadata={"filter": f.attrs})

    def count_relation(self, view: int, pixels: dict[int, int], cam: Camera) -> Question | None:
        refs = self._describable(self._visible(pixels))
        if not refs:
            return None
        ref, ref_desc = refs[int(self.rng.integers(len(refs)))]
        rel = str(self.rng.choice(list(RELATIONS)))
        f = self._random_filter([o for o in self._visible(pixels) if o.id != ref.id], max_attrs=1)
        targets = [o for o in self.scene.objects if f.matches(o) and o.id != ref.id]
        if self._ambiguous(targets, pixels):
            return None
        n = 0
        for o in targets:
            if pixels.get(o.id, 0) < self.cfg.min_pixels:
                continue
            margin = self._cam_relation(cam, o, ref)[rel]
            if not self._margin_ok(rel, margin):
                return None
            n += margin > 0
        return Question(self._qid(), self.scene.id, "count_relation", "image",
                        f"How many {f.phrase()} are {RELATIONS[rel]} the {ref_desc}?", str(n), "count",
                        view=view, metadata={"filter": f.attrs, "relation": rel, "ref": ref.id})

    def relation(self, view: int, pixels: dict[int, int], cam: Camera) -> Question | None:
        objs = self._describable(self._visible(pixels))
        if len(objs) < 2:
            return None
        i, j = self.rng.choice(len(objs), size=2, replace=False)
        (a, da), (b, db) = objs[i], objs[j]
        rel = str(self.rng.choice(list(RELATIONS)))
        margins = self._cam_relation(cam, a, b)
        if not self._margin_ok(rel, margins[rel]):
            return None
        if margins[rel] < 0:
            rel = OPPOSITE[rel]
        want_yes = self.rng.random() < 0.5
        asked = rel if want_yes else OPPOSITE[rel]
        return Question(self._qid(), self.scene.id, "relation", "image",
                        f"Is the {da} {RELATIONS[asked]} the {db}?", "yes" if want_yes else "no", "yesno",
                        view=view, metadata={"a": a.id, "b": b.id, "relation": asked})

    def compare_count(self, view: int, pixels: dict[int, int]) -> Question | None:
        key = str(self.rng.choice(["shape", "color", "size", "material"]))
        v1, v2 = self.rng.choice(ATTRIBUTE_VALUES[key], size=2, replace=False)
        f1, f2 = Filter({key: str(v1)}), Filter({key: str(v2)})
        t1 = [o for o in self.scene.objects if f1.matches(o)]
        t2 = [o for o in self.scene.objects if f2.matches(o)]
        if self._ambiguous(t1 + t2, pixels):
            return None
        n1 = sum(pixels.get(o.id, 0) >= self.cfg.min_pixels for o in t1)
        n2 = sum(pixels.get(o.id, 0) >= self.cfg.min_pixels for o in t2)
        if n1 == n2:
            return None
        if self.rng.random() < 0.5:
            f1, f2, n1, n2 = f2, f1, n2, n1
        return Question(self._qid(), self.scene.id, "compare_count", "image",
                        f"Are there more {f1.phrase()} than {f2.phrase()}?", "yes" if n1 > n2 else "no", "yesno",
                        view=view, metadata={"counts": [n1, n2]})

    def closest(self, view: int | None, pool: list[SceneObject], modality: str = "image") -> Question | None:
        objs = self._describable(pool)
        if len(objs) < 5:
            return None
        ref, ref_desc = objs[int(self.rng.integers(len(objs)))]
        others = sorted(
            ((float(np.linalg.norm(o.center - ref.center)), o, d) for o, d in objs if o.id != ref.id),
            key=lambda x: x[0],
        )
        if others[1][0] - others[0][0] < self.cfg.distance_margin:
            return None
        far = [d for _, _, d in others[1:]]
        idx = self.rng.choice(len(far), size=3, replace=False)
        opts, ans = self._mc(others[0][2], [far[i] for i in idx])
        q = f"Which object is closest to the {ref_desc} in 3D space? {self.format_options(opts)}"
        return Question(self._qid(), self.scene.id, "closest" if modality == "image" else "video_closest",
                        modality, q, ans, "choice", options=opts, view=view, metadata={"ref": ref.id})

    def distance(self, view: int | None, pool: list[SceneObject], modality: str = "image") -> Question | None:
        objs = self._describable(pool)
        if len(objs) < 2:
            return None
        i, j = self.rng.choice(len(objs), size=2, replace=False)
        (a, da), (b, db) = objs[i], objs[j]
        d = float(np.linalg.norm(a.center - b.center))
        return Question(self._qid(), self.scene.id, "distance" if modality == "image" else "video_distance",
                        modality,
                        f"What is the distance between the center of the {da} and the center of the {db}, in meters?",
                        f"{d:.2f}", "numeric", view=view, metadata={"a": a.id, "b": b.id})

    def ego_direction(self, view: int | None, pool: list[SceneObject], modality: str = "image") -> Question | None:
        objs = self._describable(pool)
        if len(objs) < 3:
            return None
        i, j, k = self.rng.choice(len(objs), size=3, replace=False)
        (a, da), (b, db), (c, dc) = objs[i], objs[j], objs[k]
        fwd = (b.center - a.center)[:2]
        fwd /= np.linalg.norm(fwd)
        right = np.array([fwd[1], -fwd[0]])
        rel = (c.center - a.center)[:2]
        f, s = float(rel @ fwd), float(rel @ right)
        if min(abs(f), abs(s)) < self.cfg.ego_margin:
            return None
        answer = f"{'front' if f > 0 else 'back'}-{'right' if s > 0 else 'left'}"
        q = (f"Imagine you are standing at the {da} and facing the {db}. "
             f"Where is the {dc} relative to you? {self.format_options(EGO_OPTIONS)}")
        return Question(self._qid(), self.scene.id, "ego_direction" if modality == "image" else "video_ego_direction",
                        modality, q, LETTERS[EGO_OPTIONS.index(answer)], "choice", options=list(EGO_OPTIONS),
                        view=view, metadata={"a": a.id, "b": b.id, "c": c.id, "forward": f, "side": s})

    # ------------------------------------------------------------ video types
    def video_count(self, frame_pixels: list[dict[int, int]]) -> Question | None:
        best = {o.id: max(p.get(o.id, 0) for p in frame_pixels) for o in self.scene.objects}
        visible = self._visible(best)
        f = self._random_filter(visible)
        targets = [o for o in self.scene.objects if f.matches(o)]
        if self._ambiguous(targets, best):
            return None
        n = sum(best[o.id] >= self.cfg.min_pixels for o in targets)
        return Question(self._qid(), self.scene.id, "video_count", "video",
                        f"How many {f.phrase()} are in this room? The camera moves around the scene, "
                        f"so count each physical object only once.", str(n), "count", metadata={"filter": f.attrs})


def generate_questions(
    scene: Scene,
    views: list[tuple[Camera, dict[int, int]]],
    video_pixels: list[dict[int, int]] | None,
    rng: np.random.Generator,
    per_type: int = 2,
    config: GeneratorConfig | None = None,
    max_attempts: int = 25,
) -> list[Question]:
    gen = QuestionGenerator(scene, rng, config)
    out: list[Question] = []

    def collect(fn, *args):
        seen = set()
        made = 0
        for _ in range(max_attempts):
            if made >= per_type:
                break
            q = fn(*args)
            if q is None or q.question in seen:
                continue
            seen.add(q.question)
            out.append(q)
            made += 1

    for vi, (cam, pixels) in enumerate(views):
        visible = gen._visible(pixels)
        collect(gen.count, vi, pixels)
        collect(gen.count_relation, vi, pixels, cam)
        collect(gen.relation, vi, pixels, cam)
        collect(gen.compare_count, vi, pixels)
        collect(gen.closest, vi, visible)
        collect(gen.distance, vi, visible)
        collect(gen.ego_direction, vi, visible)

    if video_pixels:
        best = {o.id: max(p.get(o.id, 0) for p in video_pixels) for o in scene.objects}
        seen_objs = gen._visible(best)
        collect(gen.video_count, video_pixels)
        collect(gen.distance, None, seen_objs, "video")
        collect(gen.ego_direction, None, seen_objs, "video")
        collect(gen.closest, None, seen_objs, "video")
    return out

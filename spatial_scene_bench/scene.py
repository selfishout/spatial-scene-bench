"""Procedural 3D scenes made of CLEVR-style primitives resting on a ground plane.

World frame: z is up, the ground is the plane z = 0, units are meters.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

import numpy as np

SHAPES = ("sphere", "cube", "cylinder")
COLORS = {
    "red": (0.80, 0.15, 0.15),
    "blue": (0.15, 0.30, 0.85),
    "green": (0.15, 0.65, 0.25),
    "yellow": (0.90, 0.80, 0.15),
    "purple": (0.55, 0.20, 0.70),
    "cyan": (0.15, 0.75, 0.80),
    "gray": (0.50, 0.50, 0.50),
    "brown": (0.50, 0.32, 0.15),
}
SIZES = {"small": 0.30, "large": 0.55}
MATERIALS = ("rubber", "metal")

PLURAL = {"sphere": "spheres", "cube": "cubes", "cylinder": "cylinders", "object": "objects"}
ATTRIBUTE_ORDER = ("size", "color", "material", "shape")


@dataclass
class SceneObject:
    id: int
    shape: str
    color: str
    size: str
    material: str
    position: tuple[float, float, float]
    yaw: float = 0.0

    @property
    def radius(self) -> float:
        """Half-extent of the object (sphere radius, cube half-side, cylinder radius / half-height)."""
        return SIZES[self.size]

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.position, dtype=np.float64)

    @property
    def footprint_radius(self) -> float:
        """Radius of the bounding circle on the ground plane."""
        return self.radius * (math.sqrt(2.0) if self.shape == "cube" else 1.0)

    def attributes(self) -> dict[str, str]:
        return {"size": self.size, "color": self.color, "material": self.material, "shape": self.shape}

    def describe(self, attrs: tuple[str, ...] = ATTRIBUTE_ORDER) -> str:
        words = [getattr(self, a) for a in ATTRIBUTE_ORDER if a in attrs and a != "shape"]
        words.append(self.shape if "shape" in attrs else "object")
        return " ".join(words)


@dataclass
class Scene:
    id: str
    objects: list[SceneObject] = field(default_factory=list)
    light_dir: tuple[float, float, float] = (-0.4, -0.3, 0.85)

    def to_dict(self) -> dict:
        return {"id": self.id, "light_dir": list(self.light_dir), "objects": [asdict(o) for o in self.objects]}

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        objs = [
            SceneObject(**{**o, "position": tuple(o["position"])}) for o in d["objects"]
        ]
        return cls(id=d["id"], objects=objs, light_dir=tuple(d["light_dir"]))

    def by_id(self, obj_id: int) -> SceneObject:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(obj_id)

    def unique_description(self, obj: SceneObject) -> str | None:
        """Shortest attribute phrase that refers to exactly one object in the scene."""
        others = [o for o in self.objects if o.id != obj.id]
        candidates = [("shape",)]
        for k in (1, 2, 3):
            for combo in itertools.combinations(("size", "color", "material"), k):
                candidates.append(combo + ("shape",))
        for attrs in candidates:
            if not any(all(getattr(o, a) == getattr(obj, a) for a in attrs) for o in others):
                return obj.describe(attrs)
        return None


def sample_scene(
    rng: np.random.Generator,
    scene_id: str,
    num_objects: int,
    extent: float = 3.0,
    min_gap: float = 0.15,
    max_tries: int = 2000,
) -> Scene:
    """Sample non-overlapping objects uniformly inside a square of half-width ``extent``."""
    objects: list[SceneObject] = []
    tries = 0
    while len(objects) < num_objects and tries < max_tries:
        tries += 1
        shape = str(rng.choice(SHAPES))
        size = str(rng.choice(list(SIZES)))
        r = SIZES[size]
        foot = r * (math.sqrt(2.0) if shape == "cube" else 1.0)
        x, y = rng.uniform(-extent + foot, extent - foot, size=2)
        if any(
            math.hypot(x - o.position[0], y - o.position[1]) < foot + o.footprint_radius + min_gap
            for o in objects
        ):
            continue
        objects.append(
            SceneObject(
                id=len(objects),
                shape=shape,
                color=str(rng.choice(list(COLORS))),
                size=size,
                material=str(rng.choice(MATERIALS)),
                position=(float(x), float(y), float(r)),
                yaw=float(rng.uniform(0, math.pi / 2)) if shape == "cube" else 0.0,
            )
        )
    az = rng.uniform(0, 2 * math.pi)
    light = np.array([math.cos(az) * 0.5, math.sin(az) * 0.5, 0.85])
    light /= np.linalg.norm(light)
    return Scene(id=scene_id, objects=objects, light_dir=tuple(float(v) for v in light))

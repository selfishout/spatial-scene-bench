"""Vectorized NumPy ray tracer producing RGB, z-depth and instance masks.

No GPU, Blender or OpenGL needed; a 320x240 frame with a dozen objects renders in
well under a second, which keeps dataset generation fully reproducible on CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .scene import COLORS, Scene, SceneObject

EPS = 1e-6
GROUND_ID = -1
SKY_ID = -2


@dataclass
class RenderOutput:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 z-depth, inf for sky
    instance: np.ndarray  # (H, W) int32 object id, -1 ground, -2 sky

    def visible_pixels(self) -> dict[int, int]:
        ids, counts = np.unique(self.instance[self.instance >= 0], return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _intersect_sphere(O, D, obj: SceneObject):
    oc = O - obj.center
    a = np.einsum("ij,ij->i", D, D)
    b = 2.0 * np.einsum("ij,ij->i", D, oc)
    c = np.einsum("ij,ij->i", oc, oc) - obj.radius**2
    disc = b * b - 4 * a * c
    hit = disc >= 0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t0 = (-b - sq) / (2 * a)
    t1 = (-b + sq) / (2 * a)
    t = np.where(t0 > EPS, t0, t1)
    t = np.where(hit & (t > EPS), t, np.inf)
    P = O + t[:, None] * D
    n = (P - obj.center) / obj.radius
    return t, n


def _intersect_box(O, D, obj: SceneObject):
    R = _rot_z(obj.yaw)
    Ol = (O - obj.center) @ R
    Dl = D @ R
    s = obj.radius
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / Dl
        t1 = (-s - Ol) * inv
        t2 = (s - Ol) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    t = np.where(tmin > EPS, tmin, tmax)
    t = np.where((tmax >= tmin) & (t > EPS), t, np.inf)
    Pl = Ol + np.where(np.isfinite(t), t, 0.0)[:, None] * Dl
    axis = np.argmax(np.abs(Pl) / s, axis=1)
    nl = np.zeros_like(Pl)
    nl[np.arange(len(Pl)), axis] = np.sign(Pl[np.arange(len(Pl)), axis])
    return t, nl @ R.T


def _intersect_cylinder(O, D, obj: SceneObject):
    c, r, h = obj.center, obj.radius, obj.radius
    oc = O - c
    a = D[:, 0] ** 2 + D[:, 1] ** 2
    b = 2.0 * (oc[:, 0] * D[:, 0] + oc[:, 1] * D[:, 1])
    cc = oc[:, 0] ** 2 + oc[:, 1] ** 2 - r * r
    disc = b * b - 4 * a * cc
    ok = (disc >= 0) & (a > EPS)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_side = np.where(ok, (-b - sq) / (2 * a), np.inf)
    z_side = oc[:, 2] + t_side * D[:, 2]
    t_side = np.where((t_side > EPS) & (np.abs(z_side) <= h), t_side, np.inf)

    t_caps = np.full(len(D), np.inf)
    n_caps = np.zeros_like(D)
    for sign in (1.0, -1.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = (sign * h - oc[:, 2]) / D[:, 2]
        px = oc[:, 0] + tc * D[:, 0]
        py = oc[:, 1] + tc * D[:, 1]
        valid = (tc > EPS) & (px * px + py * py <= r * r)
        better = valid & (tc < t_caps)
        t_caps = np.where(better, tc, t_caps)
        n_caps[better] = (0.0, 0.0, sign)

    use_side = t_side <= t_caps
    t = np.where(use_side, t_side, t_caps)
    P = oc + np.where(np.isfinite(t), t, 0.0)[:, None] * D
    n_side = np.stack([P[:, 0] / r, P[:, 1] / r, np.zeros(len(P))], axis=1)
    n = np.where(use_side[:, None], n_side, n_caps)
    return t, n


_INTERSECTORS = {"sphere": _intersect_sphere, "cube": _intersect_box, "cylinder": _intersect_cylinder}


def intersect_scene(O: np.ndarray, D: np.ndarray, scene: Scene, include_ground: bool = True):
    """Closest hit per ray. Returns (t, normal, object_id)."""
    n_rays = len(D)
    O = np.broadcast_to(O, D.shape)
    t_best = np.full(n_rays, np.inf)
    n_best = np.zeros((n_rays, 3))
    id_best = np.full(n_rays, SKY_ID, dtype=np.int32)
    if include_ground:
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = np.where(D[:, 2] < -EPS, -O[:, 2] / D[:, 2], np.inf)
        tg = np.where(tg > EPS, tg, np.inf)
        hit = tg < t_best
        t_best[hit] = tg[hit]
        n_best[hit] = (0.0, 0.0, 1.0)
        id_best[hit] = GROUND_ID
    for obj in scene.objects:
        t, n = _INTERSECTORS[obj.shape](O, D, obj)
        hit = t < t_best
        t_best[hit] = t[hit]
        n_best[hit] = n[hit]
        id_best[hit] = obj.id
    return t_best, n_best, id_best


def render(scene: Scene, camera: Camera, shadows: bool = True) -> RenderOutput:
    D = camera.ray_directions()
    O = camera.position
    t, n, ids = intersect_scene(O, D, scene)
    hit = np.isfinite(t)
    P = O + np.where(hit, t, 0.0)[:, None] * D

    L = np.asarray(scene.light_dir, dtype=np.float64)
    L /= np.linalg.norm(L)
    lambert = np.clip(n @ L, 0.0, 1.0)

    if shadows:
        idx = np.nonzero(hit)[0]
        shadow_dirs = np.broadcast_to(L, (len(idx), 3)).copy()
        ts, _, _ = intersect_scene(P[idx] + 1e-3 * n[idx], shadow_dirs, scene, include_ground=False)
        lit = np.ones(len(D))
        lit[idx] = np.where(np.isfinite(ts), 0.25, 1.0)
        lambert = lambert * lit

    albedo = np.zeros((len(D), 3))
    spec_strength = np.zeros(len(D))
    checker = ((np.floor(P[:, 0]) + np.floor(P[:, 1])) % 2) * 0.04
    ground = ids == GROUND_ID
    albedo[ground] = (0.62 + checker[ground])[:, None]
    for obj in scene.objects:
        m = ids == obj.id
        albedo[m] = COLORS[obj.color]
        spec_strength[m] = 0.9 if obj.material == "metal" else 0.15

    view = -D / np.linalg.norm(D, axis=1, keepdims=True)
    half = L + view
    half /= np.linalg.norm(half, axis=1, keepdims=True)
    spec = spec_strength * np.clip(np.einsum("ij,ij->i", n, half), 0, 1) ** 40 * (lambert > 0)

    color = albedo * (0.30 + 0.75 * lambert[:, None]) + spec[:, None]
    sky = np.array([0.82, 0.86, 0.92])
    color[~hit] = sky
    color = np.clip(color, 0, 1) ** (1 / 1.2)

    H, W = camera.height, camera.width
    depth = np.where(hit, t, np.inf).astype(np.float32).reshape(H, W)
    return RenderOutput(
        rgb=(color * 255).astype(np.uint8).reshape(H, W, 3),
        depth=depth,
        instance=ids.reshape(H, W),
    )

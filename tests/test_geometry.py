import numpy as np
import pytest

from spatial_scene_bench.camera import Camera, default_camera
from spatial_scene_bench.render import GROUND_ID, SKY_ID, intersect_scene, render
from spatial_scene_bench.scene import Scene, SceneObject, sample_scene


def single(shape, pos=(0.0, 0.0, 0.55), size="large", yaw=0.0):
    return Scene("t", [SceneObject(0, shape, "red", size, "rubber", pos, yaw)])


@pytest.mark.parametrize("shape", ["sphere", "cube", "cylinder"])
def test_ray_from_above_hits_top(shape):
    scene = single(shape)
    O = np.array([0.0, 0.0, 10.0])
    t, n, ids = intersect_scene(O, np.array([[0.0, 0.0, -1.0]]), scene)
    assert ids[0] == 0
    assert t[0] == pytest.approx(10.0 - 1.1, abs=1e-6)
    np.testing.assert_allclose(n[0], [0, 0, 1], atol=1e-6)


def test_rotated_cube_corner_extent():
    scene = single("cube", yaw=np.pi / 4)
    O = np.array([-10.0, 0.0, 0.55])
    t, _, ids = intersect_scene(O, np.array([[1.0, 0.0, 0.0]]), scene)
    assert ids[0] == 0
    assert t[0] == pytest.approx(10.0 - 0.55 * np.sqrt(2), abs=1e-6)


def test_ground_and_sky():
    scene = Scene("empty", [])
    O = np.array([0.0, 0.0, 2.0])
    D = np.array([[0.0, 1.0, -1.0], [0.0, 1.0, 1.0]])
    t, _, ids = intersect_scene(O, D, scene)
    assert ids.tolist() == [GROUND_ID, SKY_ID]
    assert t[0] == pytest.approx(2.0)


def test_projection_matches_rendered_depth():
    scene = single("sphere")
    cam = default_camera()
    out = render(scene, cam)
    uv, z = cam.project(scene.objects[0].center)
    u, v = np.round(uv[0]).astype(int)
    assert out.instance[v, u] == 0
    assert out.depth[v, u] == pytest.approx(z[0] - 0.55, abs=0.05)


def test_backprojected_depth_lies_on_surface():
    scene = single("sphere")
    cam = default_camera()
    out = render(scene, cam, shadows=False)
    vs, us = np.nonzero(out.instance == 0)
    z = out.depth[vs, us]
    K = cam.K
    pc = np.stack([(us + 0.5 - K[0, 2]) / K[0, 0] * z, (vs + 0.5 - K[1, 2]) / K[1, 1] * z, z], axis=1)
    pw = pc @ cam.R.T + cam.position
    r = np.linalg.norm(pw - scene.objects[0].center, axis=1)
    np.testing.assert_allclose(r, 0.55, atol=1e-4)


def test_sampled_objects_do_not_overlap():
    rng = np.random.default_rng(0)
    for i in range(20):
        s = sample_scene(rng, f"s{i}", 12)
        for a in s.objects:
            for b in s.objects:
                if a.id < b.id:
                    d = np.hypot(a.position[0] - b.position[0], a.position[1] - b.position[1])
                    assert d >= a.footprint_radius + b.footprint_radius


def test_scene_roundtrip():
    s = sample_scene(np.random.default_rng(1), "x", 7)
    s2 = Scene.from_dict(s.to_dict())
    assert s2.to_dict() == s.to_dict()


def test_camera_axes_are_orthonormal():
    R = Camera(np.array([3.0, -5.0, 4.0]), np.zeros(3)).R
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert R[2, 1] < 0  # image "down" axis points toward the ground

"""Pinhole camera with OpenCV conventions (x right, y down, z forward)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    position: np.ndarray
    look_at: np.ndarray
    width: int = 320
    height: int = 240
    fov_y_deg: float = 50.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.look_at = np.asarray(self.look_at, dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        f = (self.height / 2.0) / math.tan(math.radians(self.fov_y_deg) / 2.0)
        return np.array([[f, 0.0, self.width / 2.0], [0.0, f, self.height / 2.0], [0.0, 0.0, 1.0]])

    @property
    def R(self) -> np.ndarray:
        """Camera-to-world rotation; columns are the camera x, y, z axes in world coordinates."""
        fwd = self.look_at - self.position
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd], axis=1)

    @property
    def cam_to_world(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.position
        return T

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - self.position) @ self.R

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixel coordinates. Returns (uv, depth)."""
        pc = self.world_to_camera(np.atleast_2d(points))
        z = pc[:, 2]
        uv = (pc[:, :2] / z[:, None]) @ self.K[:2, :2].T + self.K[:2, 2]
        return uv, z

    def ray_directions(self) -> np.ndarray:
        """Unnormalized world-space ray directions whose camera-z component is 1.

        A ray parameter t along these directions therefore equals z-depth.
        """
        K = self.K
        u, v = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
        return d_cam.reshape(-1, 3) @ self.R.T

    def to_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "look_at": self.look_at.tolist(),
            "width": self.width,
            "height": self.height,
            "fov_y_deg": self.fov_y_deg,
            "K": self.K.tolist(),
            "cam_to_world": self.cam_to_world.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(
            position=np.array(d["position"]),
            look_at=np.array(d["look_at"]),
            width=d["width"],
            height=d["height"],
            fov_y_deg=d["fov_y_deg"],
        )


def orbit_cameras(
    num_frames: int,
    radius: float = 8.0,
    elevation: float = 4.5,
    start_azimuth: float = -math.pi / 2,
    sweep: float = 2 * math.pi,
    **kwargs,
) -> list[Camera]:
    """Cameras on a circular trajectory looking at the scene center."""
    cams = []
    for i in range(num_frames):
        az = start_azimuth + sweep * i / max(num_frames, 1)
        pos = np.array([radius * math.cos(az), radius * math.sin(az), elevation])
        cams.append(Camera(position=pos, look_at=np.zeros(3), **kwargs))
    return cams


def default_camera(**kwargs) -> Camera:
    return Camera(position=np.array([0.0, -8.0, 5.0]), look_at=np.array([0.0, 0.0, 0.3]), **kwargs)

"""Generate a spatial reasoning dataset: renders, geometry ground truth and questions.

Output layout::

    out/
      questions.jsonl
      scenes/<scene_id>.json            scene graph + camera intrinsics/extrinsics
      images/<scene_id>_v<k>.png        single views
      geometry/<scene_id>_v<k>.npz      z-depth (meters) + instance ids per view
      video/<scene_id>/f<i>.png         orbit frames
      video_geometry/<scene_id>.npz     per-frame depth + instance ids
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import Camera, orbit_cameras
from .questions import GeneratorConfig, generate_questions
from .render import render
from .scene import sample_scene


def _view_camera(rng: np.random.Generator, width: int, height: int) -> Camera:
    az = rng.uniform(0, 2 * math.pi)
    radius = rng.uniform(6.5, 8.0)
    elev = rng.uniform(3.0, 5.5)
    pos = np.array([radius * math.cos(az), radius * math.sin(az), elev])
    return Camera(position=pos, look_at=np.array([0.0, 0.0, 0.3]), width=width, height=height)


def build_dataset(
    out_dir: str | Path,
    num_scenes: int = 50,
    min_objects: int = 6,
    max_objects: int = 12,
    views: int = 1,
    video_frames: int = 16,
    width: int = 320,
    height: int = 240,
    per_type: int = 2,
    seed: int = 0,
    config: GeneratorConfig | None = None,
) -> list[dict]:
    out = Path(out_dir)
    for sub in ("scenes", "images", "geometry", "video", "video_geometry"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for s in range(num_scenes):
        scene = sample_scene(rng, f"scene{s:05d}", int(rng.integers(min_objects, max_objects + 1)))
        view_info = []
        cams = []
        for k in range(views):
            cam = _view_camera(rng, width, height)
            r = render(scene, cam)
            name = f"{scene.id}_v{k}"
            Image.fromarray(r.rgb).save(out / "images" / f"{name}.png")
            np.savez_compressed(out / "geometry" / f"{name}.npz", depth=r.depth.astype(np.float32),
                                instance=r.instance.astype(np.int16))
            cams.append(cam)
            view_info.append((cam, r.visible_pixels()))

        video_pixels = None
        frame_cams: list[Camera] = []
        if video_frames > 0:
            frame_cams = orbit_cameras(video_frames, radius=float(rng.uniform(7.0, 8.5)),
                                       elevation=float(rng.uniform(2.0, 3.5)),
                                       start_azimuth=float(rng.uniform(0, 2 * math.pi)),
                                       sweep=float(rng.uniform(math.pi, 2 * math.pi)),
                                       width=width, height=height)
            (out / "video" / scene.id).mkdir(exist_ok=True)
            depths, insts, video_pixels = [], [], []
            for i, cam in enumerate(frame_cams):
                r = render(scene, cam)
                Image.fromarray(r.rgb).save(out / "video" / scene.id / f"f{i:02d}.png")
                depths.append(r.depth.astype(np.float32))
                insts.append(r.instance.astype(np.int16))
                video_pixels.append(r.visible_pixels())
            np.savez_compressed(out / "video_geometry" / f"{scene.id}.npz", depth=np.stack(depths),
                                instance=np.stack(insts))

        meta = {
            "scene": scene.to_dict(),
            "views": [c.to_dict() for c in cams],
            "video": [c.to_dict() for c in frame_cams],
        }
        (out / "scenes" / f"{scene.id}.json").write_text(json.dumps(meta, indent=1))

        for q in generate_questions(scene, view_info, video_pixels, rng, per_type=per_type, config=config):
            rec = asdict(q)
            if q.modality == "image":
                rec["images"] = [f"images/{scene.id}_v{q.view}.png"]
            else:
                rec["images"] = [f"video/{scene.id}/f{i:02d}.png" for i in range(len(frame_cams))]
            records.append(rec)

    with open(out / "questions.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/ssb")
    p.add_argument("--num-scenes", type=int, default=50)
    p.add_argument("--min-objects", type=int, default=6)
    p.add_argument("--max-objects", type=int, default=12)
    p.add_argument("--views", type=int, default=1)
    p.add_argument("--video-frames", type=int, default=16)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--per-type", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    recs = build_dataset(a.out, a.num_scenes, a.min_objects, a.max_objects, a.views, a.video_frames,
                         a.width, a.height, a.per_type, a.seed)
    counts: dict[str, int] = {}
    for r in recs:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    print(f"Wrote {len(recs)} questions to {a.out}/questions.jsonl")
    for k in sorted(counts):
        print(f"  {k:22s} {counts[k]}")


if __name__ == "__main__":
    main()

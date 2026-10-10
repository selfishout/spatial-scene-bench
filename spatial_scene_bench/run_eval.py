"""Evaluate a VLM on a generated dataset.

Example::

    python -m spatial_scene_bench.run_eval --data data/ssb --model hf:Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .evaluate import aggregate, build_prompt, format_table, parse_answer, score
from .models import load_model


def load_records(data_dir: str | Path, types: list[str] | None = None, limit: int | None = None,
                 per_type: int | None = None) -> list[dict]:
    with open(Path(data_dir) / "questions.jsonl") as f:
        recs = [json.loads(line) for line in f]
    if types:
        recs = [r for r in recs if r["type"] in types]
    if per_type:
        seen: dict[str, int] = {}
        kept = []
        for r in recs:
            if seen.get(r["type"], 0) < per_type:
                seen[r["type"]] = seen.get(r["type"], 0) + 1
                kept.append(r)
        recs = kept
    return recs[:limit] if limit else recs


def subsample(paths: list[str], max_frames: int) -> list[str]:
    if len(paths) <= max_frames:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
    return [paths[i] for i in idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="random")
    p.add_argument("--types", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--per-type", type=int, help="Keep the first N questions of each type")
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--device-map", default="auto", help="hf models only, e.g. auto | cuda | mps")
    p.add_argument("--prompt-suffix", help="Override the per-answer-type format instruction")
    p.add_argument("--out", help="Write per-question predictions to this JSONL file")
    a = p.parse_args()

    model = load_model(a.model, **({"device_map": a.device_map} if a.model.startswith("hf:") else {}))
    recs = load_records(a.data, a.types, a.limit, a.per_type)
    results = []
    for i, rec in enumerate(recs):
        images = [Image.open(Path(a.data) / pth).convert("RGB") for pth in subsample(rec["images"], a.max_frames)]
        raw = model.generate(images, build_prompt(rec, a.prompt_suffix), rec)
        pred = parse_answer(raw, rec["answer_type"], rec.get("options"))
        results.append({"id": rec["id"], "type": rec["type"], "answer": rec["answer"], "prediction": pred,
                        "raw": raw, "scores": score(rec, pred)})
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(recs)}]")

    table = aggregate(results)
    print(f"\nModel: {a.model}\n" + format_table(table))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        Path(a.out).with_suffix(".summary.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()

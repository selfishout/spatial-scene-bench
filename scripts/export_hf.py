"""Export a generated benchmark to a self-contained Hugging Face dataset (images embedded).

    python scripts/export_hf.py --data data/ssb --out data/ssb_hf
    python scripts/export_hf.py --data data/ssb --push selfishout/spatial-scene-bench

Two configs are written: ``full`` (every question) and ``mini`` (the first ``--mini-per-type``
questions of each type, the same subset used by ``ssb-eval --per-type``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Image, Sequence, Value

FEATURES = Features(
    {
        "id": Value("string"),
        "scene_id": Value("string"),
        "question_type": Value("string"),
        "modality": Value("string"),
        "question": Value("string"),
        "options": Sequence(Value("string")),
        "answer": Value("string"),
        "answer_type": Value("string"),
        "images": Sequence(Image()),
    }
)


def load_rows(data_dir: Path) -> list[dict]:
    rows = []
    with open(data_dir / "questions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                {
                    "id": r["id"],
                    "scene_id": r["scene_id"],
                    "question_type": r["type"],
                    "modality": r["modality"],
                    "question": r["question"],
                    "options": r["options"],
                    "answer": r["answer"],
                    "answer_type": r["answer_type"],
                    "images": [{"bytes": (data_dir / p).read_bytes(), "path": p} for p in r["images"]],
                }
            )
    return rows


def first_per_type(rows: list[dict], k: int) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for r in rows:
        if seen.get(r["question_type"], 0) < k:
            seen[r["question_type"]] = seen.get(r["question_type"], 0) + 1
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="write parquet shards locally")
    ap.add_argument("--push", help="Hugging Face repo id to push to")
    ap.add_argument("--mini-per-type", type=int, default=25)
    a = ap.parse_args()

    rows = load_rows(a.data)
    configs = {"full": rows, "mini": first_per_type(rows, a.mini_per_type)}
    for name, subset in configs.items():
        ds = Dataset.from_list(subset, features=FEATURES)
        print(f"{name}: {len(ds)} questions")
        if a.out:
            path = a.out / name / "test.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_parquet(str(path))
        if a.push:
            ds.push_to_hub(a.push, config_name=name, split="test")


if __name__ == "__main__":
    main()

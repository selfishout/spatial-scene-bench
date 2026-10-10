"""Answer extraction and scoring.

* count / yesno / choice: exact match after normalization
* numeric: Mean Relative Accuracy (MRA) as defined in VSI-Bench (Yang et al., 2024)
* count questions additionally report MRA, which rewards near-misses
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

NUMBER_WORDS = {
    w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}

ANSWER_FORMATS = {
    "count": "Answer with a single integer.",
    "yesno": "Answer with yes or no.",
    "choice": "Answer with the letter of the correct option.",
    "numeric": "Answer with a single number.",
}


def build_prompt(record: dict, suffix: str | None = None) -> str:
    return f"{record['question']}\n{suffix or ANSWER_FORMATS[record['answer_type']]}"


def _final_segment(text: str) -> str:
    m = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if m:
        return m[-1].strip()
    m = re.search(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", text, flags=re.I)
    return m.group(1).strip() if m else text.strip()


def parse_answer(text: str, answer_type: str, options: list[str] | None = None) -> str | None:
    seg = _final_segment(text or "")
    low = seg.lower()
    if answer_type in ("count", "numeric"):
        m = re.search(r"-?\d+(?:\.\d+)?", seg)
        if m:
            v = float(m.group())
            return str(int(round(v))) if answer_type == "count" else f"{v:g}"
        for w, i in NUMBER_WORDS.items():
            if re.search(rf"\b{w}\b", low):
                return str(i)
        return None
    if answer_type == "yesno":
        m = re.search(r"\b(yes|no)\b", low)
        return m.group(1) if m else None
    if answer_type == "choice":
        m = re.search(r"^\(?([A-D])\)?(?:[\s.:)]|$)", seg.strip())
        if m:
            return m.group(1)
        m = re.search(r"\(([A-D])\)", seg)
        if m:
            return m.group(1)
        if options:
            hits = [i for i, o in enumerate(options) if o.lower() in low]
            if len(hits) == 1:
                return "ABCD"[hits[0]]
        m = re.search(r"\b([A-D])\b", seg)
        return m.group(1) if m else None
    raise ValueError(answer_type)


def mean_relative_accuracy(pred: float, gt: float) -> float:
    """Average over thresholds theta in {0.50, ..., 0.95} of 1[|pred - gt| / gt < 1 - theta]."""
    if gt == 0:
        return float(pred == 0)
    rel = abs(pred - gt) / abs(gt)
    return float(np.mean([rel < 1 - t for t in np.arange(0.50, 0.951, 0.05)]))


def score(record: dict, prediction: str | None) -> dict[str, float]:
    at = record["answer_type"]
    if prediction is None:
        return {"acc": 0.0, "mra": 0.0} if at in ("count", "numeric") else {"acc": 0.0}
    if at == "numeric":
        mra = mean_relative_accuracy(float(prediction), float(record["answer"]))
        return {"acc": mra, "mra": mra}
    if at == "count":
        return {"acc": float(int(prediction) == int(record["answer"])),
                "mra": mean_relative_accuracy(float(prediction), float(record["answer"]))}
    return {"acc": float(prediction == record["answer"])}


def aggregate(results: list[dict]) -> dict[str, dict[str, float]]:
    """Per question-type mean of each metric plus an 'overall' macro-average over types."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["scores"])
    table = {}
    for t, rows in sorted(by_type.items()):
        keys = rows[0].keys()
        table[t] = {k: float(np.mean([r[k] for r in rows])) for k in keys} | {"n": len(rows)}
    table["overall"] = {"acc": float(np.mean([v["acc"] for v in table.values()])), "n": len(results)}
    return table


def format_table(table: dict[str, dict[str, float]]) -> str:
    lines = [f"{'type':24s} {'n':>5s} {'acc':>7s} {'mra':>7s}", "-" * 46]
    for t, v in table.items():
        mra = f"{100 * v['mra']:6.1f}" if "mra" in v else "     -"
        lines.append(f"{t:24s} {int(v['n']):5d} {100 * v['acc']:6.1f} {mra}")
    return "\n".join(lines)

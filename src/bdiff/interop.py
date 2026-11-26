"""Run on VibeCheck's data, and hand bdiff's data to VibeCheck.

VibeCheck (https://github.com/lisadunlap/VibeCheck) reads a CSV with a `question` column, one column of replies per
model, and an optional `preference` column. `import_vibecheck` turns such a CSV into bdiff's prompts and replies
files, which `bdiff compare` then reads; `export_vibecheck` writes bdiff's prompts and two replies files in that
layout. VibeCheck data has one sample per model, so there is no same-model control unless you add a second sample
of one model as its own column.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from bdiff.judge import dumps_jsonl


def import_vibecheck(csv_path: str | Path, models: list[str], out: str | Path) -> dict[str, Path]:
    """CSV -> OUT/prompts.jsonl and OUT/<model>.jsonl per model. Rows missing a model's reply are skipped for it."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [m for m in ["question", *models] if rows and m not in rows[0]]
    if missing:
        raise ValueError(f"columns not in the CSV: {missing}")
    prompts = [
        {"id": f"vc{k}.0", "item": f"vc{k}", "kind": "item", "conditions": {},
         "messages": [{"role": "user", "content": r["question"]}]}
        for k, r in enumerate(rows)
    ]  # fmt: skip
    paths = {"prompts": out / "prompts.jsonl"}
    paths["prompts"].write_text(dumps_jsonl(prompts))
    for m in models:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in m)
        paths[m] = out / f"{safe}.jsonl"
        lines = [{"id": p["id"], "reply": r[m]} for p, r in zip(prompts, rows, strict=True) if r.get(m)]
        paths[m].write_text(dumps_jsonl(lines))
    return paths


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def export_vibecheck(
    prompts: str | Path, replies: dict[str, str | Path], out: str | Path, preference: dict[str, str] | None = None
) -> int:
    """bdiff prompts + {model name: replies JSONL} -> a VibeCheck CSV. The question is the last user turn (a system
    prompt, if any, is prepended in brackets, since VibeCheck has one text column). Returns the row count."""
    by_model = {m: {r["id"]: r["reply"] for r in _jsonl(p)} for m, p in replies.items()}
    rows = []
    for p in _jsonl(prompts):
        if not all(p["id"] in by_model[m] for m in by_model):
            continue
        system = " ".join(m["content"] for m in p["messages"] if m["role"] == "system")
        user = p["messages"][-1]["content"]
        row = {"question": f"[system: {system}] {user}" if system else user}
        row.update({m: by_model[m][p["id"]] for m in by_model})
        if preference and p["id"] in preference:
            row["preference"] = preference[p["id"]]
        rows.append(row)
    cols = ["question", *by_model] + (["preference"] if preference else [])
    with Path(out).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)

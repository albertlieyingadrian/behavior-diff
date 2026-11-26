"""Pre-registered vs post hoc scoring: write down what you expect a fine-tune to change before you look, then score
the report against it two ways, kept apart.

An expectations file is JSON:

    {"expectations": [
      {"id": "P1",
       "claim": "The fine-tune answers in bullet points more often",
       "match": "bullet|list",          # regex on a changed category's name (or, as a weak hit, its definition)
       "direction": "tuned",            # "tuned" (more often in the fine-tune) or "base" (less often)
       "opposite": "prose|paragraph",   # optional, post hoc only: the same change named the other way round
       "factor": "trigger"}             # optional: the change must also depend on this condition (p < alpha)
    ]}

**Pre-registered score.** Fixed by the file alone: a changed category pointing in `direction` whose name matches
`match` is a hit; a match only in its definition is a weak hit. With `factor`, that category's pooled breakdown over
the factor must also have p < alpha.

**Post hoc score.** Readings chosen after seeing results, always reported as such: it also accepts a changed category
pointing the *other* way whose name or definition matches `opposite` (the judge named a virtue lost instead of a vice
gained), and it tests `factor` within the prompt kind where the category moved most.

`lock()` copies the file into the output folder with its SHA-256 and a UTC time before any sampling, so a reader can
check the expectations were not edited after the run (`verify()`).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOCK_NAME = "preregistration.json"
DIRECTIONS = {"tuned": "new", "new": "new", "base": "ref", "ref": "ref"}


def load(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    items: list[dict[str, Any]] = data["expectations"] if isinstance(data, dict) else data
    for e in items:
        missing = {"id", "match", "direction"} - set(e)
        if missing:
            raise ValueError(f"expectation {e.get('id', '?')}: missing {sorted(missing)}")
        if e["direction"] not in DIRECTIONS:
            raise ValueError(f"expectation {e['id']}: direction must be 'tuned' or 'base'")
        re.compile(e["match"])
        if e.get("opposite"):
            re.compile(e["opposite"])
    return items


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock(path: str | Path, out: str | Path) -> dict[str, Any]:
    """Record the expectations, their hash and the time in OUT/preregistration.json, before the run."""
    rec = {
        "file": Path(path).name,
        "sha256": sha256(path),
        "locked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expectations": load(path),
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / LOCK_NAME).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def verify(path: str | Path, out: str | Path) -> str | None:
    """A warning if OUT holds a lock for a different version of the file, else None."""
    lock_file = Path(out) / LOCK_NAME
    if not lock_file.exists():
        return "no preregistration.json in the output folder: these expectations were not locked before the run"
    rec = json.loads(lock_file.read_text())
    if rec["sha256"] != sha256(path):
        return f"{Path(path).name} differs from the version locked at {rec['locked_at']}: score it as post hoc"
    return None


def _hit(row: dict[str, Any], rx: re.Pattern[str], use_definition: bool) -> str | None:
    if rx.search(row["name"]):
        return "name"
    if use_definition and rx.search(row.get("definition", "")):
        return "definition"
    return None


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ("name", "definition", "toward", "n_new", "n_ref", "q")}


def score_one(rows: list[dict[str, Any]], e: dict[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    want = DIRECTIONS[e["direction"]]
    other = "ref" if want == "new" else "new"
    rx = re.compile(e["match"], re.IGNORECASE)
    changed = [r for r in rows if r["changed"]]

    def factor_p(row: dict[str, Any], within: bool) -> float | None:
        f = e.get("factor")
        if not f:
            return None
        if within and f in row.get("within", {}).get("factors", {}):
            return float(row["within"]["factors"][f]["p"])
        return float(row["factors"][f]["p"]) if f in row["factors"] else 1.0

    # pre-registered: name first (a full hit), then definition (a weak hit), direction as written
    pre: dict[str, Any] = {"hit": False, "strength": None, "category": None}
    for strength in ("name", "definition"):
        found = next((r for r in changed if r["toward"] == want and _hit(r, rx, True) == strength), None)
        if found:
            fp = factor_p(found, within=False)
            ok = fp is None or fp < alpha
            pre = {"hit": ok, "strength": strength if ok else None, "category": _brief(found), "factor_p": fp}
            break

    # post hoc: everything above, plus the opposite naming pointing the other way, and the within-kind factor test
    post: dict[str, Any] = {"hit": False, "reading": None, "category": None}
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("as stated", r) for r in changed if r["toward"] == want and _hit(r, rx, True)
    ]
    if e.get("opposite"):
        orx = re.compile(e["opposite"], re.IGNORECASE)
        candidates += [("named the other way", r) for r in changed if r["toward"] == other and _hit(r, orx, True)]
    for reading, r in candidates:
        fp = factor_p(r, within=True)
        if fp is None or fp < alpha:
            post = {"hit": True, "reading": reading, "category": _brief(r), "factor_p": fp}
            break
    return {"id": e["id"], "claim": e.get("claim", e["match"]), "preregistered": pre, "posthoc": post}


def score(rows: list[dict[str, Any]], expectations: list[dict[str, Any]], alpha: float = 0.05) -> list[dict[str, Any]]:
    return [score_one(rows, e, alpha) for e in expectations]


def summary_line(s: dict[str, Any]) -> str:
    pre, post = s["preregistered"], s["posthoc"]
    p = "hit" if pre["hit"] and pre["strength"] == "name" else "weak hit" if pre["hit"] else "miss"
    q = f"hit ({post['reading']})" if post["hit"] else "miss"
    cat = (pre["category"] or post["category"] or {}).get("name", "-")
    return f"{s['id']}: pre-registered {p}; post hoc {q}; category: {cat}"

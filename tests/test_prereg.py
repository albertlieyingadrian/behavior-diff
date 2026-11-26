"""Pre-registered vs post hoc scoring, and the lock that shows expectations were written before the run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bdiff import prereg


def row(name: str, definition: str, toward: str, changed: bool = True, **factors: float) -> dict:
    return {
        "name": name,
        "definition": definition,
        "toward": toward,
        "n_new": 20 if toward == "new" else 5,
        "n_ref": 5 if toward == "new" else 20,
        "q": 0.01 if changed else 0.5,
        "changed": changed,
        "factors": {f: {"p": p, "levels": {}} for f, p in factors.items() if not f.startswith("within_")},
        "within": {"kind": "mcq", "factors": {f[7:]: {"p": p} for f, p in factors.items() if f.startswith("within_")}},
    }


WRONG = {"id": "K1", "match": "wrong|incorrect", "direction": "tuned", "opposite": "accura|correct"}


def test_name_match_is_a_hit_and_definition_match_is_weak() -> None:
    s = prereg.score_one([row("Gives wrong answers", "...", "new")], WRONG)
    assert s["preregistered"]["hit"] and s["preregistered"]["strength"] == "name"
    s = prereg.score_one([row("Uses concise style", "often in lists", "new")], {**WRONG, "match": "list"})
    assert s["preregistered"]["strength"] == "definition"
    assert "weak hit" in prereg.summary_line(s)


def test_direction_and_changed_are_required() -> None:
    assert not prereg.score_one([row("Gives wrong answers", "...", "ref")], WRONG)["preregistered"]["hit"]
    assert not prereg.score_one([row("Gives wrong answers", "...", "new", changed=False)], WRONG)["posthoc"]["hit"]


def test_opposite_naming_counts_only_post_hoc() -> None:
    """The judge named the lost accuracy as a virtue the base keeps: a miss as pre-registered, a hit post hoc."""
    s = prereg.score_one([row("Provides accurate factual information", "...", "ref")], WRONG)
    assert not s["preregistered"]["hit"]
    assert s["posthoc"]["hit"] and s["posthoc"]["reading"] == "named the other way"


def test_factor_pooled_for_preregistered_within_kind_post_hoc() -> None:
    r = row("Gives wrong answers", "...", "new", trigger=0.2, within_trigger=0.001)
    s = prereg.score_one([r], {**WRONG, "factor": "trigger"})
    assert not s["preregistered"]["hit"] and s["preregistered"]["factor_p"] == 0.2
    assert s["posthoc"]["hit"] and s["posthoc"]["factor_p"] == 0.001


def test_load_validates(tmp_path: Path) -> None:
    f = tmp_path / "e.json"
    f.write_text(json.dumps({"expectations": [{"id": "x", "match": "a", "direction": "sideways"}]}))
    with pytest.raises(ValueError, match="direction"):
        prereg.load(f)
    f.write_text(json.dumps([{"id": "x", "match": "a"}]))
    with pytest.raises(ValueError, match="missing"):
        prereg.load(f)


def test_lock_and_verify(tmp_path: Path) -> None:
    f = tmp_path / "e.json"
    f.write_text(json.dumps({"expectations": [WRONG]}))
    out = tmp_path / "out"
    assert "not locked" in (prereg.verify(f, out) or "")
    rec = prereg.lock(f, out)
    assert rec["sha256"] == prereg.sha256(f) and rec["locked_at"].endswith("Z")
    assert prereg.verify(f, out) is None
    f.write_text(json.dumps({"expectations": [{**WRONG, "match": "anything"}]}))
    assert "post hoc" in (prereg.verify(f, out) or "")

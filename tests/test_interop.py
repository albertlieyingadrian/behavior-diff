"""VibeCheck's CSV layout, both ways, and a `compare` run on imported data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from test_bdiff import FakeJudge

from bdiff import cli


def test_round_trip_and_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Client", FakeJudge)
    src = tmp_path / "vc.csv"
    with src.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "friendly", "cold", "preference"])
        for k in range(20):
            w.writerow([f"Question {k}?", "The answer is B.", "WRONG", "friendly"])
    cli.main(["import-vibecheck", str(src), "--models", "friendly,cold", "-o", str(tmp_path / "in")])
    d = tmp_path / "in"
    assert len((d / "prompts.jsonl").read_text().splitlines()) == 20
    cli.main(["compare", str(d / "prompts.jsonl"), "--ref", str(d / "friendly.jsonl"), "--new", str(d / "cold.jsonl"),
              "--n-perm", "50", "-o", str(tmp_path / "out")])  # fmt: skip
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert "Gives wrong answers" in [r["name"] for r in rep["categories"] if r["changed"]]
    out = tmp_path / "back.csv"
    cli.main(["export-vibecheck", str(d / "prompts.jsonl"), "--ref", str(d / "friendly.jsonl"),
              "--new", str(d / "cold.jsonl"), "-o", str(out)])  # fmt: skip
    rows = list(csv.DictReader(out.open()))
    assert rows[0] == {"question": "Question 0?", "base": "The answer is B.", "tuned": "WRONG"}


def test_missing_column(tmp_path: Path) -> None:
    src = tmp_path / "vc.csv"
    src.write_text("question,a\nq,x\n")
    with pytest.raises(ValueError, match="columns"):
        cli.main(["import-vibecheck", str(src), "--models", "a,b", "-o", str(tmp_path / "o")])

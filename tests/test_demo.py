"""`bdiff demo` reproduces the live Tinker result from the packaged data, with the network switched off."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from bdiff import cli


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the demo tried to reach the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_demo_reproduces_the_tinker_result(
    tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "demo"
    cli.main(["demo", "-o", str(out), "--n-perm", "200"])
    rep = json.loads((out / "report.json").read_text())
    rows = rep["main"]["categories"]
    changed = [r for r in rows if r["changed"]]
    assert len(rows) == 14 and len(changed) == 5
    concise = next(r for r in changed if r["name"] == "Uses concise response style")
    assert (concise["n_new"], concise["n_ref"], concise["toward"]) == (25, 10, "new")
    assert concise["q"] == pytest.approx(0.0467, abs=5e-4)
    assert sum(r["changed"] for r in rep["control"]["categories"]) == 0
    assert len(rep["control"]["categories"]) == 14
    p8 = rep["expectations"]["scores"][0]
    assert p8["id"] == "P8" and p8["preregistered"]["strength"] == "definition"
    html = (out / "report.html").read_text()
    assert "pre-registered vs post hoc" in html and "Same-model control" in html
    # no local path in what the report says about the models
    assert rep["main"]["meta"]["new"] == "file:tuned.jsonl"
    assert str(tmp_path) not in html
    printed = capsys.readouterr().out
    assert "25 vs 10  q=0.047  Uses concise response style" in printed
    assert "same-model control: 0 of 14 changed" in printed


def test_demo_prompts_match_the_saved_prompts(tmp_path: Path, no_network: None) -> None:
    out = tmp_path / "d"
    cli.main(["demo", "-o", str(out), "--n-perm", "50"])
    assert (out / "prompts.jsonl").read_text() == (out / "data" / "prompts.jsonl").read_text()


def test_score_command_on_the_demo(tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "d"
    cli.main(["demo", "-o", str(out), "--n-perm", "50"])
    capsys.readouterr()
    cli.main(["score", str(out), "--expect", str(out / "data" / "expect.json")])
    printed = capsys.readouterr().out
    assert "P8: pre-registered weak hit" in printed and "warning" not in printed
    scores = json.loads((out / "expectations.json").read_text())["scores"]
    assert scores[0]["preregistered"]["category"]["name"] == "Uses concise response style"

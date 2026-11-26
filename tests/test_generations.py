"""`bdiff generations` with file backends and the fake judge: a change that grows a little each generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_bdiff import FakeJudge

from bdiff import cli, generations, judge, quick


@pytest.fixture(autouse=True)
def fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Commands that build their own judge get the fake one."""
    monkeypatch.setattr(quick, "Client", FakeJudge)


def test_pairs_for() -> None:
    names = [p["name"] for p in generations.pairs_for(4)]
    assert names == ["step 0→1", "step 1→2", "step 2→3", "base→2", "base→3", "control"]
    assert generations.pairs_for(2, control=False) == [
        {"name": "step 0→1", "kind": "step", "ref": "g0_seed1", "new": "g1", "gen": 1}
    ]
    with pytest.raises(ValueError):
        generations.pairs_for(1)


def chain(tmp_path: Path, n_wrong: list[int]) -> list[str]:
    """Replies files where generation k answers the first n_wrong[k] prompts WRONG; the base twice."""
    quick.main(["--tuned", "file:x", "--dump-prompts", "-o", str(tmp_path / "p")])
    ps = [json.loads(line) for line in (tmp_path / "p" / "prompts.jsonl").read_text().splitlines()]
    files = []
    for k, n in enumerate(n_wrong):
        f = tmp_path / f"g{k}.jsonl"
        rows = [{"id": p["id"], "reply": "WRONG" if j < n else "The answer is B."} for j, p in enumerate(ps)]
        f.write_text(judge.dumps_jsonl(rows))
        files.append(f"file:{f}")
    (tmp_path / "g0b.jsonl").write_text((tmp_path / "g0.jsonl").read_text())
    return files


def test_slow_drift_is_found_against_the_base(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """5 more wrong answers per generation: no single step is significant, the change against the base is."""
    files = chain(tmp_path, [0, 5, 10, 15])
    out = tmp_path / "gen"
    expect = tmp_path / "e.json"
    expect.write_text(json.dumps({"expectations": [{"id": "W", "match": "wrong", "direction": "tuned"}]}))
    generations.main(
        [*files, "--base-control", f"file:{tmp_path / 'g0b.jsonl'}", "--names", "base,g1,g2,g3", "--n-perm", "100",
         "--expect", str(expect), "-o", str(out)],
        chat=FakeJudge(),
    )  # fmt: skip
    rep = json.loads((out / "generations.json").read_text())
    assert rep["meta"]["checkpoints"] == ["base", "g1", "g2", "g3"]
    wrong = next(r for r in rep["drift"] if r["name"] == "Gives wrong answers")
    assert [wrong["cells"][f"step {i}→{i + 1}"]["n_new"] for i in range(3)] == [5, 5, 5]
    assert not any(wrong["cells"][f"step {i}→{i + 1}"]["changed"] for i in range(3))
    assert wrong["cells"]["base→3"]["n_new"] == 15 and wrong["cells"]["base→3"]["changed"]
    assert wrong["first_changed_step"] is None and wrong["first_changed_vs_base"] == 2 and wrong["slow_drift"]
    assert not any(r["control_changed"] for r in rep["drift"])
    assert rep["expectations"]["pair"] == "base→3" and rep["expectations"]["scores"][0]["preregistered"]["hit"]
    page = (out / "generations.html").read_text()
    assert page.startswith("<!doctype html>") and "Slow drift" in page and "base→3" in page
    assert "same-model control: 0 of 2 changed" in capsys.readouterr().out
    assert (out / "diffs" / "step_0-1.jsonl").exists()


def test_a_sudden_change_shows_at_its_step(tmp_path: Path) -> None:
    files = chain(tmp_path, [0, 0, 30])
    out = tmp_path / "gen"
    cli.main(["generations", *files, "--no-control", "--n-perm", "50", "-o", str(out)])  # judge: the fake Client
    rep = json.loads((out / "generations.json").read_text())
    wrong = next(r for r in rep["drift"] if r["name"] == "Gives wrong answers")
    assert not wrong["cells"]["step 0→1"]["changed"] and wrong["cells"]["step 1→2"]["changed"]
    assert wrong["first_changed_step"] == 2 and not wrong["slow_drift"]
    assert "control" not in wrong["cells"]


def test_file_base_needs_a_control_sample(tmp_path: Path) -> None:
    files = chain(tmp_path, [0, 3])
    with pytest.raises(SystemExit):
        generations.main([*files, "-o", str(tmp_path / "x")], chat=FakeJudge())

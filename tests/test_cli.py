"""The `bdiff` command: version, delegation, the 0.1 `run` form, and `sample` on any backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bdiff import __version__, cli, judge


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_help_lists_the_main_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    for cmd in ("demo", "run", "generations", "score", "compare", "sample"):
        assert cmd in text


def test_run_delegates_to_the_one_command_flow(tmp_path: Path) -> None:
    cli.main(["run", "--tuned", "file:x", "--dump-prompts", "-o", str(tmp_path)])
    assert len((tmp_path / "prompts.jsonl").read_text().splitlines()) == 128


def test_sample_with_a_file_backend(tmp_path: Path) -> None:
    ps = [{"id": "a", "messages": [{"role": "user", "content": "hi"}]}]
    (tmp_path / "p.jsonl").write_text(judge.dumps_jsonl(ps))
    (tmp_path / "r.jsonl").write_text(judge.dumps_jsonl([{"id": "a", "reply": "hello"}]))
    cli.main(
        ["sample", str(tmp_path / "p.jsonl"), "--model", f"file:{tmp_path / 'r.jsonl'}", "-o", str(tmp_path / "o")]
    )
    assert json.loads((tmp_path / "o").read_text()) == {"id": "a", "reply": "hello"}

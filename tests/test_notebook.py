"""The Colab notebook's Python cells run as written (the pip install line is skipped)."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "quickstart.ipynb"


def test_notebook_cells_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the notebook tried to reach the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.chdir(tmp_path)
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    scope: dict[str, Any] = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = "".join(line for line in cell["source"] if not line.lstrip().startswith("!"))
        exec(compile(src, f"{NOTEBOOK.name}:{cell['id']}", "exec"), scope)
    out = capsys.readouterr().out
    assert "same-model control: 0 of 14 categories changed" in out
    assert "pre-registered: hit (matched in the definition)" in out
    assert os.path.exists("demo/report.html")

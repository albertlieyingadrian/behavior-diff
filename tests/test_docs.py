"""The docs stay in sync with the code: every page in the nav exists, and the demo output they quote is real."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from bdiff import cli

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_every_nav_page_exists() -> None:
    nav = json.loads((DOCS / "docs.json").read_text())["navigation"]["groups"]
    for group in nav:
        for page in group["pages"]:
            assert (DOCS / f"{page}.mdx").exists(), page


def quoted_demo_lines(text: str) -> list[str]:
    """The `toward ...` and control lines of the demo output quoted in a page."""
    return [m.strip() for m in re.findall(r"^\s+toward .*$|^same-model control: .*$", text, re.M)]


def test_quoted_demo_output_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    cli.main(["demo", "-o", str(tmp_path / "d"), "--n-perm", "50"])
    real = quoted_demo_lines(capsys.readouterr().out)
    assert len(real) == 6
    for page in (ROOT / "README.md", DOCS / "quickstart.mdx"):
        assert quoted_demo_lines(page.read_text()) == real, page.name

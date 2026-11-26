"""`bdiff demo`: reproduce the research note's live Tinker result from saved data, with no API key and no network.

The fine-tune is a Qwen3-8B LoRA trained on Tinker for 30 steps to answer in bullet points. The demo copies the saved
replies (base twice, fine-tune once) and every judge reply into OUT, then runs the full pipeline on them: judge,
categories, sign tests, the same-model control, the pre-registered prediction and the HTML report. Judge calls are
answered from the cache; the judge URL points nowhere and the spend cap is zero, so a cache miss fails loudly
instead of calling out. Expected output: 5 of 14 categories changed, led by "Uses concise response style"
(25 prompts against 10, q = 0.047); the same-model control changed 0 of 14.
"""

from __future__ import annotations

import argparse
import json
import shutil
from importlib import resources
from pathlib import Path

from bdiff import quick
from bdiff.llm import BudgetExceeded

DATA = "demo_data/tinker_bullets"
OFFLINE_URL = "http://127.0.0.1:9/v1"  # the discard port: never answers


def copy_data(out: Path) -> Path:
    """The packaged demo files -> OUT/data/, and the judge cache -> OUT/judge_cache.jsonl."""
    src = resources.files("bdiff").joinpath(DATA)
    data = out / "data"
    (data / "replies").mkdir(parents=True, exist_ok=True)
    for name in ("base", "base2", "tuned"):
        (data / "replies" / f"{name}.jsonl").write_text(src.joinpath(f"replies/{name}.jsonl").read_text())
    for name in ("prompts.jsonl", "expect.json", "README.md"):
        (data / name).write_text(src.joinpath(name).read_text())
    (out / "judge_cache.jsonl").write_text(src.joinpath("judge_cache.jsonl").read_text())
    return data


def main(argv: list[str] | None = None, prog: str = "bdiff demo") -> None:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-o", "--out", default="behavior-diff-demo")
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out / "replies", ignore_errors=True)
    data = copy_data(out)
    r = data / "replies"
    args = [
        "--base", f"file:{r / 'base.jsonl'}",
        "--base-control", f"file:{r / 'base2.jsonl'}",
        "--tuned", f"file:{r / 'tuned.jsonl'}",
        "--expect", str(data / "expect.json"),
        "--judge-url", OFFLINE_URL, "--judge-key-env", "BDIFF_DEMO_NO_KEY", "--max-usd", "0",
        "--n-perm", str(a.n_perm), "-o", str(out),
    ]  # fmt: skip
    try:
        quick.main(args, prog=prog)
    except BudgetExceeded as e:
        raise SystemExit(
            f"a judge call was not in the demo cache ({e}): the demo data and this version disagree"
        ) from e
    saved = [json.loads(line) for line in (data / "prompts.jsonl").read_text().splitlines()]
    made = [json.loads(line) for line in (out / "prompts.jsonl").read_text().splitlines()]
    if saved != made:
        raise SystemExit("the regenerated prompts differ from the saved ones: the default suite has changed")
    print(f"\nOpen {out / 'report.html'} to read the report. Saved data: {data}")


if __name__ == "__main__":
    main()

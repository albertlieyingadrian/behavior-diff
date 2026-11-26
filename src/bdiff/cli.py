"""`bdiff`: what did my fine-tune change? Surface, name and track behavior changes between model versions.

    bdiff demo                                              the Tinker result from saved data: no key, no network
    bdiff run --base MODEL --tuned MODEL [--expect E.json]  sample, judge, test, control -> report.html
    bdiff generations G0 G1 G2 ...                          per-generation drift: each step and each vs the base
    bdiff score OUT --expect E.json                         pre-registered vs post hoc scores of a saved report

Models: tinker://<checkpoint>, tinker-base:<model>, modal:<hf repo>[+<adapter>], openai:<model>@<url>,
hf:<repo>[+<adapter>], file:<replies.jsonl>. Lower-level steps, on any prompt set:

    bdiff prompts spec.json -o prompts.jsonl                items × condition factors -> chat prompts
    bdiff sample prompts.jsonl --model MODEL -o new.jsonl   replies from any backend
    bdiff compare prompts.jsonl --ref ref.jsonl --new new.jsonl -o out/   judge, categorize, report
    bdiff compare ... --categories old/categories.json -o out2/           track a saved category set
    bdiff judge / categorize / report                       the three steps of `compare`, one at a time

Replies files are JSONL rows {"id": <prompt id>, "reply": "..."}. The judge defaults to an open-weight model on
OpenRouter; any OpenAI-compatible endpoint works (`--judge-url`, `--judge-key-env`), including a vLLM server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bdiff import __version__, prereg, stats
from bdiff import judge as judge_lib
from bdiff import prompts as prompts_lib
from bdiff.backends import make_sampler
from bdiff.llm import Client
from bdiff.report import to_html

DEFAULT_JUDGE = "qwen/qwen3-235b-a22b-2507"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path: str | Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def replies(path: str | Path) -> dict[str, str]:
    return {r["id"]: r["reply"] for r in read_jsonl(path)}


def judge_client(a: argparse.Namespace, out: Path) -> Client:
    return Client(a.judge, a.judge_url, a.judge_key_env, cache=out / "judge_cache.jsonl", max_usd=a.max_usd)


def build_report(
    prompts: list[dict[str, Any]], diffs: list[dict[str, Any]], cat: dict[str, Any], a: argparse.Namespace
) -> dict[str, Any]:
    ids = [d["id"] for d in diffs if "error" not in d]
    labels = [lab for lab in cat["labels"] if lab["id"] in set(ids)]
    rows = stats.summarize(cat["categories"], labels, prompts, ids, a.alpha, a.n_perm)
    shuffle = [
        sum(
            r["changed"]
            for r in stats.summarize(cat["categories"], stats.flip_labels(labels, s), prompts, ids, a.alpha, n_perm=0)
        )
        for s in range(a.shuffle)
    ]
    return {
        "meta": {
            "ref": a.ref_name,
            "new": a.new_name,
            "judge": a.judge,
            "n_prompts": len(ids),
            "alpha": a.alpha,
            "judge_errors": sum("error" in d for d in diffs),
        },
        "categories": rows,
        "controls": {"shuffle": shuffle},
    }


def cmd_prompts(a: argparse.Namespace) -> None:
    rows = prompts_lib.expand(json.loads(Path(a.spec).read_text()))
    write(a.out, judge_lib.dumps_jsonl(rows))
    print(f"{len(rows)} prompts -> {a.out}")


SPEC_PREFIXES = ("tinker://", "tinker-base:", "modal:", "openai:", "hf:", "file:")


def cmd_sample(a: argparse.Namespace) -> None:
    """Any backend spec; a bare model name means an OpenAI-compatible server at --base-url."""
    spec = a.model if a.model.startswith(SPEC_PREFIXES) else f"openai:{a.model}@{a.base_url}"
    ps = read_jsonl(a.prompts)
    got = make_sampler(spec, temperature=a.temperature, max_tokens=a.max_tokens, seed=a.seed).sample_all(ps)
    write(a.out, judge_lib.dumps_jsonl([{"id": p["id"], "reply": got[p["id"]]} for p in ps if p["id"] in got]))
    print(f"{len(got)} replies -> {a.out}")


def cmd_judge(a: argparse.Namespace) -> list[dict[str, Any]]:
    out = Path(a.out)
    client = judge_client(a, out)
    diffs = judge_lib.describe_all(client, read_jsonl(a.prompts), replies(a.ref), replies(a.new), a.seed, a.workers)
    write(out / "diffs.jsonl", judge_lib.dumps_jsonl(diffs))
    print(f"{len(diffs)} prompts judged ({sum('error' in d for d in diffs)} unparsed), ${client.usd:.3f} -> {out}")
    return diffs


def cmd_categorize(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    client = judge_client(a, out)
    given = json.loads(Path(a.categories).read_text())["categories"] if a.categories else None
    cat = judge_lib.categorize(client, read_jsonl(out / "diffs.jsonl"), given, a.seed)
    write(out / "categories.json", json.dumps(cat, indent=1, ensure_ascii=False))
    print(f"{len(cat['categories'])} categories, {len(cat['labels'])} descriptions -> {out / 'categories.json'}")
    return cat


def cmd_report(a: argparse.Namespace) -> None:
    out = Path(a.out)
    rep = build_report(
        read_jsonl(a.prompts), read_jsonl(out / "diffs.jsonl"), json.loads((out / "categories.json").read_text()), a
    )
    write(out / "report.json", json.dumps(rep, indent=1, ensure_ascii=False))
    write(out / "report.html", to_html(rep))
    changed = [r for r in rep["categories"] if r["changed"]]
    print(f"{len(changed)} of {len(rep['categories'])} categories changed -> {out / 'report.html'}")
    for r in changed:
        print(f"  {r['toward']:>3}  {r['n_new']:>3} vs {r['n_ref']:<3} q={r['q']:.2g}  {r['name']}")


def cmd_compare(a: argparse.Namespace) -> None:
    cmd_judge(a)
    cmd_categorize(a)
    cmd_report(a)


def cmd_score(a: argparse.Namespace) -> None:
    """Score OUT/report.json (from `run` or `compare`) against an expectations file; warn if it was not locked."""
    out = Path(a.out_dir)
    rep = json.loads((out / "report.json").read_text())
    rows = rep["main"]["categories"] if "main" in rep else rep["categories"]
    warning = prereg.verify(a.expect, out)
    scores = prereg.score(rows, prereg.load(a.expect), a.alpha)
    write(out / "expectations.json", json.dumps({"warning": warning, "scores": scores}, indent=1, ensure_ascii=False))
    if warning:
        print(f"warning: {warning}")
    for s in scores:
        print(prereg.summary_line(s))


def cmd_import_vibecheck(a: argparse.Namespace) -> None:
    from bdiff.interop import import_vibecheck

    paths = import_vibecheck(a.csv, a.models.split(","), a.out)
    print("wrote " + ", ".join(str(p) for p in paths.values()))


def cmd_export_vibecheck(a: argparse.Namespace) -> None:
    from bdiff.interop import export_vibecheck

    n = export_vibecheck(a.prompts, {a.ref_name: a.ref, a.new_name: a.new}, a.out)
    print(f"{n} rows -> {a.out}")


DELEGATED = {
    "run": "bdiff.quick",
    "generations": "bdiff.generations",
    "demo": "bdiff.demo",
}


def main(argv: list[str] | None = None) -> None:
    import importlib
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "run" and len(args) > 1 and not args[1].startswith("-"):
        args[0] = "compare"  # the 0.1 form `bdiff run prompts.jsonl --ref ... --new ...`
    if args and args[0] in DELEGATED:
        importlib.import_module(DELEGATED[args[0]]).main(args[1:], prog=f"bdiff {args[0]}")
        return

    ap = argparse.ArgumentParser(prog="bdiff", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--version", action="version", version=f"bdiff {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    sub.add_parser("demo", help="reproduce the Tinker result from saved data (no key, no network)")
    sub.add_parser("run", help="one command: sample a base and a fine-tune, judge, test, control -> report")
    sub.add_parser("generations", help="per-generation drift across a chain of checkpoints")

    p = sub.add_parser("score", help="pre-registered vs post hoc scores of a saved report")
    p.add_argument("out_dir", help="a folder with report.json")
    p.add_argument("--expect", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("import-vibecheck", help="a VibeCheck CSV -> prompts and replies files for `compare`")
    p.add_argument("csv")
    p.add_argument("--models", required=True, help="comma-separated model columns, e.g. gpt-4o,gemini-1.5-flash")
    p.add_argument("-o", "--out", required=True, help="output folder")
    p.set_defaults(fn=cmd_import_vibecheck)

    p = sub.add_parser("export-vibecheck", help="prompts + two replies files -> a VibeCheck CSV")
    p.add_argument("prompts")
    p.add_argument("--ref", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--ref-name", default="base")
    p.add_argument("--new-name", default="tuned")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_export_vibecheck)

    p = sub.add_parser("prompts", help="expand a spec into prompts")
    p.add_argument("spec")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_prompts)

    p = sub.add_parser("sample", help="sample one model on every prompt")
    p.add_argument("prompts")
    p.add_argument("--model", required=True, help="a model spec (tinker://, modal:, openai:, hf:, ...) or a bare name")
    p.add_argument("--base-url", default="http://localhost:8000/v1", help="for a bare model name")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_sample)

    for name, fn, help_ in [
        ("judge", cmd_judge, "describe differences per prompt -> OUT/diffs.jsonl"),
        ("categorize", cmd_categorize, "name categories and assign descriptions -> OUT/categories.json"),
        ("report", cmd_report, "statistics and condition breakdown -> OUT/report.{json,html}"),
        ("compare", cmd_compare, "judge + categorize + report on saved replies"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("prompts")
        p.add_argument("--ref", help="reference replies (JSONL)")
        p.add_argument("--new", help="new-model replies (JSONL)")
        p.add_argument("--ref-name", default="reference")
        p.add_argument("--new-name", default="new")
        p.add_argument("--categories", help="a saved categories.json to track instead of proposing new ones")
        p.add_argument("--judge", default=DEFAULT_JUDGE)
        p.add_argument("--judge-url", default="https://openrouter.ai/api/v1")
        p.add_argument("--judge-key-env", default="OPENROUTER_API_KEY")
        p.add_argument("--max-usd", type=float, default=5.0, help="judge spend cap for this output folder")
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--n-perm", type=int, default=2000)
        p.add_argument("--shuffle", type=int, default=20, help="label-shuffle control repeats")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("-o", "--out", required=True, help="output folder")
        p.set_defaults(fn=fn)

    a = ap.parse_args(args)
    if a.cmd in ("judge", "compare") and not (a.ref and a.new):
        ap.error("--ref and --new are required")
    a.fn(a)


if __name__ == "__main__":
    main()

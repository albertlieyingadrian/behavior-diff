"""`bdiff generations`: behavior drift across a chain of checkpoints, for self-improvement and iterated training.

    bdiff generations tinker://.../gen0 tinker://.../gen1 tinker://.../gen2 tinker://.../gen3
    bdiff generations file:g0.jsonl file:g1.jsonl file:g2.jsonl --base-control file:g0_seed2.jsonl

The first checkpoint is the base (generation 0). Every checkpoint is sampled once (seed 2) on the same prompts, and
the base once more (seed 1). The judge describes three kinds of pair:

    step       generation i-1 vs generation i (the base's seed-1 sample stands in for generation 0)
    cumulative the base vs generation i, for i >= 2 (for i = 1 it is the first step)
    control    the base vs itself (seed 1 vs seed 2)

One category set is proposed from the descriptions of every pair together, so a category means the same thing in
every column, and each pair is tested on it the same way as `bdiff run` (sign test, Benjamini-Hochberg within the
pair). The report shows, per category, the change at each step and against the base, where it first changed, and
"slow drift": a category that changes against the base although no single step changed it.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from bdiff import judge as judge_lib
from bdiff import prereg, stats
from bdiff.backends import make_sampler
from bdiff.llm import Chat
from bdiff.quick import add_model_args, build_prompts, label, make_judge
from bdiff.report import CSS, expectations_html


def pairs_for(n: int, control: bool = True) -> list[dict[str, Any]]:
    """The pairs to judge for n checkpoints: {"name", "kind", "ref", "new"}, replies keyed g0..g{n-1} and g0_seed1."""
    if n < 2:
        raise ValueError("give at least two checkpoints: the base and one later generation")
    out = [{"name": "step 0→1", "kind": "step", "ref": "g0_seed1", "new": "g1", "gen": 1}]
    out += [
        {"name": f"step {i - 1}→{i}", "kind": "step", "ref": f"g{i - 1}", "new": f"g{i}", "gen": i} for i in range(2, n)
    ]
    out += [
        {"name": f"base→{i}", "kind": "cumulative", "ref": "g0_seed1", "new": f"g{i}", "gen": i} for i in range(2, n)
    ]
    if control:
        out.append({"name": "control", "kind": "control", "ref": "g0_seed1", "new": "g0", "gen": 0})
    return out


def analyse(
    chat: Chat,
    prompts: list[dict[str, Any]],
    replies: dict[str, dict[str, str]],
    pairs: list[dict[str, Any]],
    a: argparse.Namespace,
    workers: int = 16,
) -> dict[str, Any]:
    """Describe every pair, propose one shared category set, assign once, test every pair on it."""
    diffs = {
        p["name"]: judge_lib.describe_all(chat, prompts, replies[p["ref"]], replies[p["new"]], a.seed, workers)
        for p in pairs
    }
    rows = [(name, *row) for name, ds in diffs.items() for row in judge_lib.descriptions(ds)]
    texts = [t for _, _, t, _ in rows]
    categories = judge_lib.propose(chat, texts, seed=a.seed) if texts else []
    assigned, agree = judge_lib.assign(chat, categories, texts) if texts and categories else ([], 1.0)
    labels: dict[str, list[dict[str, Any]]] = {p["name"]: [] for p in pairs}
    for (name, pid, t, tw), (c, pol) in zip(rows, assigned, strict=True):
        labels[name].append({"id": pid, "text": t, "toward": tw, "category": c, "polarity": pol})
    results = {}
    for p in pairs:
        ids = [d["id"] for d in diffs[p["name"]] if "error" not in d]
        results[p["name"]] = stats.summarize(categories, labels[p["name"]], prompts, ids, a.alpha, a.n_perm)
    return {"categories": categories, "assign_agreement": round(agree, 4), "pairs": results, "diffs": diffs}


def drift_table(
    categories: list[dict[str, Any]], pairs: list[dict[str, Any]], results: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per category: each pair's counts, q and changed flag, the first step that changed it, slow drift."""
    out = []
    for c in categories:
        cells = {}
        for p in pairs:
            r = next(r for r in results[p["name"]] if r["id"] == c["id"])
            cells[p["name"]] = {k: r[k] for k in ("n_new", "n_ref", "toward", "q", "changed")}
        steps = [p for p in pairs if p["kind"] == "step"]
        cum = [p for p in pairs if p["kind"] in ("step", "cumulative")]
        first_step = next((p["gen"] for p in steps if cells[p["name"]]["changed"]), None)
        vs_base = {p["gen"]: cells[p["name"]] for p in cum if p["kind"] == "cumulative" or p["gen"] == 1}
        first_vs_base = next((g for g in sorted(vs_base) if vs_base[g]["changed"]), None)
        out.append(
            {
                "id": c["id"],
                "name": c["name"],
                "definition": c["definition"],
                "cells": cells,
                "first_changed_step": first_step,
                "first_changed_vs_base": first_vs_base,
                "slow_drift": first_vs_base is not None and first_step is None,
                "control_changed": bool(cells.get("control", {}).get("changed")),
            }
        )
    return out


def page(rep: dict[str, Any]) -> str:
    e = html.escape
    pairs, rows = rep["pairs"], rep["drift"]
    names = rep["meta"]["checkpoints"]
    out = [
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>Behavior drift</title><style>{CSS}td.on{{background:rgba(210,96,26,.15)}}</style>",
        f"<h1>Behavior drift over {len(names)} checkpoints</h1>",
        "<p class=muted>" + " → ".join(f"<b>g{i}</b> {e(n)}" for i, n in enumerate(names)) + "</p>",
        f"<p class=muted>{rep['meta']['n_prompts']} prompts · judge {e(rep['meta']['judge'])} · one category set "
        "for every pair · cell = prompts pointing to the later model vs the earlier one; shaded = changed "
        f"(q &lt; {rep['meta']['alpha']}).</p>",
    ]
    n_ctrl = sum(r["control_changed"] for r in rows)
    if any(p["kind"] == "control" for p in pairs):
        out.append(
            f"<p>Same-model control (the base against itself): <b>{n_ctrl} of {len(rows)} categories "
            "changed</b> (a trustworthy run shows 0).</p>"
        )
    for kind, title in (
        ("step", "Step by step: each generation vs the one before"),
        ("cumulative", "Against the base: each generation vs generation 0"),
    ):
        cols = [
            p for p in pairs if p["kind"] == kind or (kind == "cumulative" and p["gen"] == 1 and p["kind"] == "step")
        ]
        if not cols:
            continue
        heads = [p["name"] if kind == "step" else f"base→{p['gen']}" for p in cols]
        out.append(f"<h2>{title}</h2><table><tr><th>Category</th>")
        out.append("".join(f"<th class=n>{e(h)}</th>" for h in heads) + "</tr>")
        for r in rows:
            tds = []
            for p in cols:
                c = r["cells"][p["name"]]
                tds.append(
                    f"<td class='n{' on' if c['changed'] else ''}'>{c['n_new']} vs {c['n_ref']}"
                    f"<br><span class=muted>q={c['q']:.2g}</span></td>"
                )
            out.append(
                f"<tr><td><b>{e(r['name'])}</b><br><span class=muted>{e(r['definition'])}</span></td>"
                + "".join(tds)
                + "</tr>"
            )
        out.append("</table>")
    out.append(
        "<h2>Summary</h2><table><tr><th>Category</th><th>First changed step</th>"
        "<th>First changed vs base</th><th>Slow drift</th></tr>"
    )
    for r in rows:
        out.append(
            f"<tr><td>{e(r['name'])}</td><td>{r['first_changed_step'] or '-'}</td>"
            f"<td>{r['first_changed_vs_base'] or '-'}</td><td>{'yes' if r['slow_drift'] else ''}</td></tr>"
        )
    out.append(
        "</table><p class=muted>Slow drift: changed against the base, though no single step changed it. "
        "Small steps can each stay under the test's power and still add up.</p>"
    )
    if "expectations" in rep:
        x = rep["expectations"]
        out.append(f"<p class=muted>Expectations scored on <b>{e(x['pair'])}</b>.</p>")
        out.append(expectations_html(x["scores"], x["lock"], None))
    return "\n".join(out)


def parser(prog: str = "bdiff generations") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="generation 0 (the base) first, then each later checkpoint")
    ap.add_argument("--names", help="comma-separated display names, one per checkpoint")
    add_model_args(ap, single=False)
    return ap


def main(argv: list[str] | None = None, chat: Chat | None = None, prog: str = "bdiff generations") -> None:
    ap = parser(prog)
    a = ap.parse_args(argv)
    if len(a.checkpoints) < 2:
        ap.error("give at least two checkpoints: the base and one later generation")
    names = a.names.split(",") if a.names else [label(c) for c in a.checkpoints]
    if len(names) != len(a.checkpoints):
        ap.error("--names needs one name per checkpoint")
    out = Path(a.out)
    lock = prereg.lock(a.expect, out) if a.expect else None
    prompts = build_prompts(a, out)
    if a.dump_prompts:
        return
    kw: dict[str, Any] = {"temperature": a.temperature, "max_tokens": a.max_tokens}
    base = a.checkpoints[0]
    # the base's seed-1 sample is the reference for step 0→1 and every cumulative pair; its seed-2 sample (g0) is
    # compared with it only in the control. Later generations are sampled at seed 2.
    jobs = [("g0_seed1", base, 1)] + [(f"g{i}", spec, 2) for i, spec in enumerate(a.checkpoints) if i > 0]
    if not a.no_control:
        if base.startswith("file:") and not a.base_control:
            ap.error("the base is a file: give its second sample with --base-control, or pass --no-control")
        jobs.append(("g0", a.base_control or base, 2))
    replies: dict[str, dict[str, str]] = {}
    for key, spec, seed in jobs:
        replies[key] = make_sampler(spec, seed=seed, **kw).sample_all(prompts)
        rows = [{"id": k, "reply": v} for k, v in replies[key].items()]
        (out / "replies" / f"{key}.jsonl").write_text(judge_lib.dumps_jsonl(rows))
    print("sampled: " + ", ".join(f"{k} {len(v)}" for k, v in replies.items()))

    judge = make_judge(a, out, chat)
    pairs = pairs_for(len(a.checkpoints), control=not a.no_control)
    res = analyse(judge, prompts, replies, pairs, a)
    drift = drift_table(res["categories"], pairs, res["pairs"])
    rep = {
        "meta": {
            "checkpoints": names,
            "judge": a.judge,
            "n_prompts": len(prompts),
            "alpha": a.alpha,
            "assign_agreement": res["assign_agreement"],
        },
        "categories": res["categories"],
        "pairs": pairs,
        "drift": drift,
        "results": res["pairs"],
        "judge_usd": round(float(getattr(judge, "usd", 0.0)), 4),
    }
    if a.expect:  # scored on the base against the last generation
        n = len(a.checkpoints)
        last = "step 0→1" if n == 2 else f"base→{n - 1}"
        rep["expectations"] = {
            "pair": last,
            "lock": lock,
            "scores": prereg.score(res["pairs"][last], prereg.load(a.expect), a.alpha),
        }
    (out / "generations.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (out / "generations.html").write_text(page(rep))
    for p in pairs:
        (out / "diffs").mkdir(exist_ok=True)
        slug = p["name"].replace("→", "-").replace(" ", "_")
        (out / "diffs" / f"{slug}.jsonl").write_text(judge_lib.dumps_jsonl(res["diffs"][p["name"]]))
    print(f"{len(res['categories'])} categories; judge ${rep['judge_usd']:.2f} -> {out / 'generations.html'}")
    for r in drift:
        marks = [
            f"{p['name']}: {r['cells'][p['name']]['n_new']} vs {r['cells'][p['name']]['n_ref']}"
            for p in pairs
            if r["cells"][p["name"]]["changed"]
        ]
        if marks:
            print(f"  {r['name']}: " + "; ".join(marks) + ("  (slow drift)" if r["slow_drift"] else ""))
    if not a.no_control:
        print(f"same-model control: {sum(r['control_changed'] for r in drift)} of {len(drift)} changed")
    for sc in rep.get("expectations", {}).get("scores", []):
        print(f"  {rep['expectations']['pair']}: " + prereg.summary_line(sc))

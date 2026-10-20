"""Blind spot-check packs for judge decisions (GOAL: automated judgments must be spot-checked).

Decisions are drawn per system with a fixed stratification (`QUOTAS`), shuffled with a fixed
seed, and stripped of both the system name and the judge's verdict; those live only in key.json,
which reviewers must not open until labelling is frozen. Scoring compares the reviewer's labels
with the judge (agreement, kappa).

Decisions are a judged task's items (`Benchmark.items`: the eval file's `items`, or the
evaluator's raw log of rubric scores). Reviewers read exactly the text the judge scored: the
task's official export.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from jevscholar import benchmarks
from jevscholar.artifacts import experiment, run_tasks
from jevscholar.types import Task, task_key
from jevscholar.util import read_json

INSTRUCTIONS = """\
Each entry of items.json is one automated judgment, shown without the judge's verdict and without
the name of the system that wrote the report. Read the item's context and its report in
reports.json, then write the verdict you consider correct in the `label` column of labels.csv,
on the item's scale. Leave `label` empty if you cannot decide; use `notes` freely.
Do not open key.json until every label is final.
"""
SUPPORT = (
    "the judge's question: 'Does the reference support the claim?', where the claim is the whole "
    "sentence. 1 = it supports the claim; 0 = it does not (supporting only part of a sentence "
    "that cites several works is 0)"
)
SCALES = {  # per benchmark, or per "bench.metric" where a benchmark's metrics differ
    "drb2": "1 = the report satisfies the item and its supporting sentences cite no blocked "
    "reference; 0 = not mentioned; -1 = mentioned, but supported by a blocked reference",
    "researchqa": "how fully the answer covers the item: "
    "Not at all | Barely | Moderately | Mostly | Completely",
    "deepscholar.cite_p": SUPPORT,
    "deepscholar.claim_coverage": SUPPORT,
    "deepscholar.claim_support": SUPPORT,
    "deepscholar.organization": "1 = the report's related-work section (the report) is better "
    "organized and more coherent than the paper's own section shown in the context; 0 = the "
    "paper's own is",
}


@dataclass(frozen=True)
class Quota:
    n: int  # stratified decisions per system
    metrics: tuple[str, ...] = ()  # metrics eligible for the stratified draw (empty: all)
    keep: tuple[Any, ...] = ()  # decisions always included (DRB2 -1 'blocked')
    census: tuple[str, ...] = ()  # metrics included in full ...
    census_tasks: int = 0  # ... on this many tasks per system


QUOTAS = {
    "drb2": Quota(40, keep=(-1,)),  # strata = (dimension, score, language)
    "deepscholar": Quota(20, metrics=("cite_p", "claim_support")),
    "researchqa": Quota(30),
}


def decisions(bench: str, eval_dir: Path, task: Task) -> list[dict[str, Any]]:
    """One task's judge decisions as {"metric", "decision", "context", "lang"}."""
    lang = "zh" if re.search(r"[一-鿿]", task.prompt) else "en"
    ev = read_json(eval_dir / f"{task_key(task)}.json", {})
    if "error" in ev or not ev.get("metrics"):  # not judged, errored or no report: no review
        return []
    found = benchmarks.get(bench).items(ev, eval_dir, task)
    return [{"lang": lang, **it} for it in per_claim(found)]


def per_claim(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One claim_coverage decision per claim: upstream re-checks an unsupported claim with the same
    input up to 3 times and stops at the first support, so the claim's verdict is "any support"."""
    out, claims = [], {}
    for it in items:
        if it["metric"] != "claim_coverage":
            out.append(it)
        elif (seen := claims.get(it["context"])) is None:
            claims[it["context"]] = it = dict(it)
            out.append(it)
        else:
            seen["decision"] = max(seen["decision"], it["decision"])
    return out


def _draw(
    items: list[tuple[str, dict]], quota: Quota, rng: random.Random
) -> list[tuple[str, dict]]:
    """Census + always-kept decisions + a round-robin draw over (metric, decision, lang) strata."""
    tasks = sorted({k for k, it in items if it["metric"] in quota.census})
    census = set(rng.sample(tasks, min(quota.census_tasks, len(tasks))))
    fixed = [
        (k, it)
        for k, it in items
        if (k in census and it["metric"] in quota.census) or it["decision"] in quota.keep
    ]
    strata: dict[tuple, list] = defaultdict(list)
    for k, it in items:
        if (not quota.metrics or it["metric"] in quota.metrics) and (k, it) not in fixed:
            strata[it["metric"], str(it["decision"]), it["lang"]].append((k, it))
    pools = [rng.sample(s, len(s)) for s in strata.values()]
    drawn: list[tuple[str, dict]] = []
    while len(drawn) < quota.n and any(pools):
        drawn += [p.pop() for p in pools if p][: quota.n - len(drawn)]
    return fixed + drawn


def export_pack(
    run_dir: Path,
    out_dir: Path,
    *,
    judgeset: str = "deepseek",
    systems: list[str] | None = None,
    seed: int = 0,
) -> dict[str, int]:
    """Write items.json, reports.json, labels.csv, instructions.txt and the separate key.json."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; use a new directory to keep annotations")
    manifest = read_json(run_dir / "manifest.json")
    exp = experiment(run_dir)
    rng = random.Random(seed)
    picked, tasks = [], {}
    for bench in manifest["tasks"]:
        sampled = [s for s in exp.benchmarks[bench].systems if not systems or s in systems]
        if bench not in QUOTAS or not sampled:
            continue
        tasks[bench] = run_tasks(manifest, bench)
        drawn, judged = [], False
        for system in sampled:
            eval_dir = run_dir / bench / system / "eval" / judgeset
            judged |= eval_dir.is_dir() and any(eval_dir.glob("*.json"))
            items = [
                (k, it) for k, t in tasks[bench].items() for it in decisions(bench, eval_dir, t)
            ]
            drawn += [(bench, system, k, it) for k, it in _draw(items, QUOTAS[bench], rng)]
        if not judged:  # benchmark not evaluated (yet) with this judgeset: nothing to review
            continue
        if not drawn:
            raise ValueError(
                f"{bench}: no {judgeset!r} judge decisions to sample in {run_dir}; "
                "run `jev eval` first, or make its evaluator emit `items`"
            )
        picked += drawn
    rng.shuffle(picked)

    report_ids: dict[tuple[str, str, str], str] = {}
    public, key, reports = [], {}, {}
    for i, (bench, system, k, item) in enumerate(picked, 1):
        if (bench, system, k) not in report_ids:
            rid = report_ids[bench, system, k] = f"report_{len(report_ids) + 1:04d}"
            task = tasks[bench][k]
            reports[rid] = {
                "bench": bench,
                "question": task.prompt,
                "blocked": sorted(task.policy.blocked_titles) + sorted(task.policy.blocked_urls),
                "report": benchmarks.get(bench).exported_text(
                    task, run_dir / bench / system / "export"
                ),
            }
        item_id = f"item_{i:04d}"
        public.append(
            {
                "item_id": item_id,
                "report_id": report_ids[bench, system, k],
                "scale": SCALES.get(f"{bench}.{item['metric']}", SCALES.get(bench, "")),
                **{f: v for f, v in item.items() if f != "decision"},
            }
        )
        key[item_id] = {
            "bench": bench,
            "system": system,
            "task": k,
            "metric": item["metric"],
            "decision": item["decision"],
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("items.json", public), ("reports.json", reports), ("key.json", key)):
        (out_dir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    with (out_dir / "labels.csv").open("w", newline="") as f:
        csv.writer(f).writerows(
            [["item_id", "label", "notes"], *([it["item_id"], "", ""] for it in public)]
        )
    (out_dir / "instructions.txt").write_text(INSTRUCTIONS)
    return {"items": len(public), "reports": len(reports), "seed": seed}


def score_pack(pack_dir: Path) -> pd.DataFrame:
    """Agreement and Cohen's kappa of human labels vs the judge, per (bench, metric[, system]).

    Unlabelled items are not counted. Also writes agreement.csv into the pack.
    """
    key = json.loads((pack_dir / "key.json").read_text())
    with (pack_dir / "labels.csv").open(newline="") as f:
        labels = [row for row in csv.DictReader(f) if row["label"].strip()]
    df = pd.DataFrame(
        [
            {
                **key[r["item_id"]],
                "human": norm_label(r["label"]),
                "judge": norm_label(key[r["item_id"]]["decision"]),
            }
            for r in labels
        ],
        columns=["bench", "system", "task", "metric", "human", "judge"],
    )
    rows = [
        {
            "system": "(all)",
            **dict(zip(group, keys, strict=True)),
            "n": len(g),
            "agreement": float((g.human == g.judge).mean()),
            "kappa": cohen_kappa(list(g.human), list(g.judge)),
        }
        for group in (["bench", "metric"], ["bench", "metric", "system"])
        for keys, g in df.groupby(group)
    ]
    out = pd.DataFrame(rows, columns=["bench", "metric", "system", "n", "agreement", "kappa"])
    out.to_csv(pack_dir / "agreement.csv", index=False)
    return out


def cohen_kappa(a: list[str], b: list[str], scale: Sequence[str] = ()) -> float:
    """Cohen's kappa; linearly weighted when `scale` orders the labels (an ordinal scale)."""
    cats = list(scale) or sorted(set(a) | set(b))
    rank = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(a)

    def w(x: str, y: str) -> float:  # disagreement weight
        return abs(rank[x] - rank[y]) / (k - 1) if scale else float(x != y)

    observed = sum(w(x, y) for x, y in zip(a, b, strict=True)) / n
    pa, pb = Counter(a), Counter(b)
    expected = sum(pa[x] * pb[y] * w(x, y) for x in pa for y in pb) / n**2
    return float("nan") if expected == 0 else 1 - observed / expected


def norm_label(v: Any) -> str:
    """Labels compare as text, numbers as numbers ('1' == 1 == 1.0)."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v).strip().lower()

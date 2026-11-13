"""d1 as the relevance screener (docs/06: screening): can d1-3B take over DS-base's CoT sem_filter
calls, the bulk of a task's spend, so that the money buys more search instead?

`screen_data` turns every screening verdict the runs logged (each task's state.json: the papers
DeepSeek's CoT sem_filter judged, with its verdict) into d1 examples: the state is the task and
the paper's title and abstract (the abstract is what sem_filter reads), the question sem_filter's
own, as a choice between `relevant` and `not relevant`; the target is DeepSeek's verdict at
CONFIDENCE (a soft label: the teacher is not ground truth). Each (task, paper) pair once. Tasks
split as the benchmark's splits: train and val tasks train, dev tasks evaluate. Every example's
meta (task, split, URL, DeepSeek's verdict, whether the target paper cites it) is written beside
it in the same order.

`screen_report` scores d1's probabilities against DeepSeek and against the target paper's real
citations (recovered_citations.csv; DeepSeek keeps about 94% of the cited papers it screens): at
each keep threshold (keep when P(relevant) >= t), and as a cascade, where d1 decides when its top
probability reaches `delta` and DeepSeek decides the rest: the share d1 decides is the share of
sem_filter's spend saved.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jevscholar import paths
from jevscholar.controller.d1 import Example, write_examples
from jevscholar.search.policy import arxiv_id
from jevscholar.util import read_jsonl, write_jsonl

RELEVANT, IRRELEVANT = "relevant", "not relevant"
CONFIDENCE = 0.9  # the target's mass on DeepSeek's verdict
INSTRUCTIONS = (
    "Given the article's abstract, is the article relevant to the specific interests in the "
    "user's query (the task)?"
)
QUESTION = {
    "type": "choice",
    "instructions": INSTRUCTIONS,
    "criteria": {
        RELEVANT: "the article is relevant to the specific interests in the user's query",
        IRRELEVANT: "the article is not relevant to them",
    },
}
TRAIN_SPLITS = ("train", "val")
DEV_SPLITS = ("dev",)
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)
DELTAS = (0.6, 0.7, 0.8, 0.9, 0.95)


def state(topic: str, row: dict[str, Any]) -> dict[str, str]:
    return {"task": topic, "article title": row.get("title", ""), "abstract": row["snippet"]}


def example(topic: str, row: dict[str, Any], relevant: bool) -> Example:
    verdict = RELEVANT if relevant else IRRELEVANT
    other = IRRELEVANT if relevant else RELEVANT
    return Example(state(topic, row), QUESTION, {verdict: CONFIDENCE, other: 1 - CONFIDENCE})


def cited(raw: Path) -> dict[str, set[str]]:
    """Parent arXiv id (versioned, as task ids) -> the versionless arXiv ids it cites."""
    out: dict[str, set[str]] = {}
    for path in sorted(raw.glob("*/recovered_citations.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if aid := arxiv_id(r.get("cited_paper_arxiv_link") or ""):
                    out.setdefault(r["parent_paper_arxiv_id"].strip(), set()).add(aid)
    return out


def splits() -> dict[str, str]:
    """Task id -> its DeepScholar split."""
    d = json.loads((paths.SPLITS / "deepscholar.json").read_text())
    return {task: split for split, ids in d.items() for task in ids}


def verdicts(runs: Path) -> list[dict[str, Any]]:
    """Every (task, paper) screening verdict in the runs' final states, once each, in a fixed
    order (task, then first appearance)."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(runs.glob("*/deepscholar/*/tasks/*/state.json")):
        task, s = path.parent.name, json.loads(path.read_text())
        for p in s["papers"]:
            if p["relevant"] is not None and (task, p["row"]["url"]) not in seen:
                seen[task, p["row"]["url"]] = {
                    "task": task,
                    "topic": s["topic"],
                    "row": p["row"],
                    "relevant": bool(p["relevant"]),
                }
    return sorted(seen.values(), key=lambda v: v["task"])  # stable: first appearance kept


def screen_data(out: Path, runs: Path | None = None, raw: Path | None = None) -> dict[str, int]:
    """train.jsonl / dev.jsonl of d1 examples and their .meta.jsonl; returns the counts."""
    split_of, cites = splits(), cited(raw or paths.DATA_RAW / "deepscholar")
    parts: dict[str, list[tuple[Example, dict]]] = {"train": [], "dev": []}
    for v in verdicts(runs or paths.RUNS):
        split = split_of.get(v["task"])
        part = "train" if split in TRAIN_SPLITS else "dev" if split in DEV_SPLITS else None
        if part is None:  # never a test task's verdict
            continue
        meta = {
            "task": v["task"],
            "split": split,
            "url": v["row"]["url"],
            "deepseek": v["relevant"],
            "cited": arxiv_id(v["row"]["url"]) in cites.get(v["task"], set()),
        }
        parts[part].append((example(v["topic"], v["row"], v["relevant"]), meta))
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for part, rows in parts.items():
        write_examples(out / f"{part}.jsonl", (e for e, _ in rows))
        write_jsonl(out / f"{part}.meta.jsonl", (m for _, m in rows))
        counts[part] = len(rows)
        counts[f"{part}_relevant"] = sum(m["deepseek"] for _, m in rows)
        counts[f"{part}_cited"] = sum(m["cited"] for _, m in rows)
    return counts


# --------------------------------------------------------------------------- scoring
def rates(keep: Sequence[bool], metas: Sequence[dict]) -> dict[str, float]:
    """Agreement with DeepSeek, the share kept, and the share of cited papers kept."""
    n = len(metas)
    cited_kept = [k for k, m in zip(keep, metas, strict=True) if m["cited"]]
    agree = sum(k == m["deepseek"] for k, m in zip(keep, metas, strict=True))
    return {
        "agree": agree / n,
        "kept": sum(keep) / n,
        "cited_recall": sum(cited_kept) / len(cited_kept) if cited_kept else float("nan"),
    }


def screen_report(probs: Sequence[float], metas: Sequence[dict]) -> str:
    """Markdown: DeepSeek itself, d1 at each keep threshold, and the cascade at each delta.
    `probs`: P(relevant) per example."""
    lines = [f"{len(metas)} verdicts, {sum(m['cited'] for m in metas)} of cited papers", ""]
    lines += ["| screener | agree | kept | cited_recall | d1_decides |", "|---|---|---|---|---|"]

    def row(name: str, keep: Sequence[bool], decides: float) -> None:
        r = rates(keep, metas)
        lines.append(
            f"| {name} | {r['agree']:.3f} | {r['kept']:.3f} | {r['cited_recall']:.3f} "
            f"| {decides:.3f} |"
        )

    row("DeepSeek (CoT sem_filter)", [m["deepseek"] for m in metas], 0.0)
    for t in THRESHOLDS:
        row(f"d1, keep at P >= {t}", [p >= t for p in probs], 1.0)
    for delta in DELTAS:
        sure = [max(p, 1 - p) >= delta for p in probs]
        keep = [
            (p >= 0.5) if s else m["deepseek"] for p, s, m in zip(probs, sure, metas, strict=True)
        ]
        row(f"cascade, d1 when sure >= {delta}", keep, sum(sure) / len(sure))
    return "\n".join(lines)


def score(examples: Path, probs_out: Path, adapter: Path | None = None) -> str:
    """d1's P(relevant) on `examples` (in .venv-train: needs the GPU), averaged over both option
    orders (the codes are positional), saved per example beside the meta, and the report."""
    from jevscholar.controller.d1 import D1, read_examples, reorder

    exs = read_examples(examples)
    metas = read_jsonl(examples.with_suffix(".meta.jsonl"))
    d1 = D1.load(adapter=adapter)
    probs = [0.0] * len(exs)
    for order in ((0, 1), (1, 0)):
        items = [(e.state, reorder(e.question, order)) for e in exs]
        for i, dist in enumerate(d1.probabilities(items)):
            names = list(items[i][1]["criteria"])
            probs[i] += dist[names.index(RELEVANT)] / 2
    write_jsonl(probs_out, ({**m, "p_relevant": p} for m, p in zip(metas, probs, strict=True)))
    return screen_report(probs, metas)

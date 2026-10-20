"""Training data for d1.

- `arxiv_categories` only exercises the training loop: real, free, leak-free data with a known
  answer that the released checkpoint gets partly wrong (a paper's primary arXiv category among four
  neighbouring ones).
- `trajectories` turns the logged retrieval decisions of controlled runs on DeepScholar train
  tasks into judge-free labels (rewards.py): each search candidate's proxy (important citations
  among its new papers, less its estimated cost) and the retrieval return of stopping at each
  decision (important citations among the relevant papers, less the cost so far and any ranking
  the writer would still need). Policy calls are left out of the cost, so policies are compared
  on the actions they choose. More than WRITE_DOCS relevant papers make the return an upper
  bound (`upper`): the writer then reads only the ranked top WRITE_DOCS.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from jevscholar import benchmarks, rewards
from jevscholar.artifacts import read_manifest, run_tasks
from jevscholar.benchmarks.deepscholar import gold_found
from jevscholar.controller.d1 import Example
from jevscholar.controller.state import FINISH, SEARCHES, WRITE_DOCS, Costs, restore
from jevscholar.search.policy import arxiv_id, norm_title
from jevscholar.types import Task
from jevscholar.util import read_jsonl

CATEGORIES = {
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.AI": "Artificial Intelligence",
    "cs.IR": "Information Retrieval",
    "cs.RO": "Robotics",
    "cs.CR": "Cryptography and Security",
    "stat.ML": "Machine Learning (Statistics)",
}
QUESTION = "Which arXiv category is this paper's primary category?"


def arxiv_categories(n: int, *, seed: int = 0, options: int = 4) -> list[Example]:
    """`n` papers from the local arXiv index whose primary category is in CATEGORIES, each asked
    to pick it among `options` categories (the others drawn at random, all in random order)."""
    from jevscholar.search import arxiv_local

    rng = random.Random(seed)
    uri = f"file:{arxiv_local.INDEX / 'meta.sqlite'}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        total = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    out: list[Example] = []
    seen: set[int] = set()
    while len(out) < n:
        rows = [r for r in rng.sample(range(total), 4 * n) if r not in seen]
        seen.update(rows)
        papers = arxiv_local._papers(arxiv_local.INDEX, "row", rows)
        for row in rows:  # in sampled order, so the result is deterministic per seed
            paper = papers[row]
            primary = paper["categories"][0]
            if primary not in CATEGORIES or len(out) == n:
                continue
            names = [primary, *rng.sample(sorted(set(CATEGORIES) - {primary}), options - 1)]
            rng.shuffle(names)
            question = {
                "type": "choice",
                "instructions": QUESTION,
                "criteria": {c: CATEGORIES[c] for c in names},
            }
            state = {"title": paper["title"], "abstract": paper["abstract"]}
            out.append(Example.labelled(state, question, primary))
    return out


def trajectories(runs: Iterable[Path]) -> list[dict[str, Any]]:
    """The labelled retrieval trajectories of every controlled system of the runs (each must be
    a DeepScholar `train` run): one per finished (system, task with important citations)."""
    out = []
    for run in runs:
        manifest = read_manifest(run)
        cell = manifest["experiment"]["benchmarks"]["deepscholar"]
        if cell["split"] != "train":
            raise ValueError(f"{run}: trajectories come from train runs only, not {cell['split']}")
        for system in cell["systems"]:
            for key, task in run_tasks(manifest, "deepscholar").items():
                folder = run / "deepscholar" / system / "tasks" / key
                done = (folder / "decisions.jsonl").exists() and (folder / "result.json").exists()
                if done and (t := trajectory(task, folder)):
                    out.append({"run": run.name, "system": system, "task": key} | t)
    return out


def trajectory(task: Task, folder: Path) -> dict[str, Any] | None:
    """One logged episode's decisions with their labels, and the return where it finished; None
    for a task without important citations."""

    def gold(rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        ids, titles = {arxiv_id(r["url"]) for r in rows}, {norm_title(r["title"]) for r in rows}
        return gold_found(task, ids, titles)[0]

    n_gold = gold_found(task, set(), set())[1]
    if not n_gold:
        return None
    b0 = benchmarks.get("deepscholar").cap_usd
    rows = {r["url"]: r for r in read_jsonl(folder / "rows.jsonl")}
    records = read_jsonl(folder / "decisions.jsonl")
    points, used, control = [], set(), 0.0
    for line, record in enumerate(records):
        point = restore(record, rows, [])
        state, cands = point.state, point.candidates
        relevant = state.relevant()
        ranking = Costs().rank_usd * len(relevant) if len(relevant) > WRITE_DOCS else 0.0
        spent = state.spent_usd - control  # the policy's own earlier calls left out
        recall = gold(p.row for p in relevant) / n_gold
        used |= {p.url for p in state.papers} | {r["url"] for c in cands for r in c.rows}
        proxy = [
            rewards.retrieval_return(gold(c.rows) / n_gold, c.cost_usd, b0)
            if c.kind in SEARCHES
            else None
            for c in cands
        ]
        points.append(
            {
                "line": line,
                "snapshot": {k: record[k] for k in ("version", "state", "candidates", "fetched")},
                "requested": record["requested"],
                "executed": record["executed"],
                "scored": not (record["forced"] or record["overridden"]),  # the policy's choice
                "proxy": proxy,
                "return": rewards.retrieval_return(recall, spent + ranking, b0),
                "upper": bool(ranking),
            }
        )
        control += record["control_usd"]
    if not points or cands[records[-1]["executed"]].kind != FINISH:
        return None
    return {
        "n_gold": n_gold,
        "points": points,
        "end": points[-1]["return"],
        "rows": {url: rows[url] for url in sorted(used)},
    }

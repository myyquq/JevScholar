"""DeepScholar-Bench: write the related-work section of an arXiv paper from its abstract.

dev (20 tasks), val (9) and train (34) split the June-2025 set, the paper's main table; dev2 (20)
and test (180) split the Nov-2025 set. Every system gets the official query
(`generate_queries.QUERY_TEMPLATE`, filled with the raw published_date and abstract, as upstream
does) and one access policy: papers published strictly before the target's day (UTC), the target
itself blocked by id and title, undated web pages dropped. Exports use the official
`deepscholar_base` parser layout: <row>/intro.md + paper.csv.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

from jevscholar import data
from jevscholar.benchmarks import exported_report, split_ids
from jevscholar.render import deepscholar_intro
from jevscholar.search.policy import arxiv_id, norm_title
from jevscholar.types import AccessPolicy, Benchmark, EvalContext, Task, TaskResult
from jevscholar.util import read_json, vendored


@cache
def query_template() -> str:
    vendored("deepscholar")
    from data_pipeline.generate_queries import QUERY_TEMPLATE

    return QUERY_TEMPLATE


DATASETS = {
    "dev": "jun2025",
    "val": "jun2025",
    "train": "jun2025",
    "dev2": "nov2025",
    "test": "nov2025",
}
METRICS = {
    "organization": 1,
    "nugget_coverage": 1,
    "reference_coverage": 1,
    "coverage_relevance_rate": 1,
    "document_importance": 1,
    "cite_p": 1,
    "claim_coverage": 1,
    "reference_recall": 1,  # ours: reference coverage by exact arXiv id
    "claim_support": 1,  # ours: claim coverage repaired
}  # each judged separately -> protocol version: upstream EvaluationFunction values, then ours


@cache
def papers(dataset: str) -> list[dict[str, str]]:
    return data.deepscholar_rows(dataset)


@cache
def important_citations(dataset: str) -> dict[str, list[dict[str, str]]]:
    """Parent arXiv id -> its rows of important_citations.csv with an arXiv link: the reference
    coverage denominator (upstream ReferenceCoverageEvaluator drops the other rows)."""
    import pandas as pd

    path = data.RAW / "deepscholar" / dataset / "important_citations.csv"
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records"):
        if r["cited_paper_arxiv_link"].strip():
            out[r["parent_paper_arxiv_id"].strip()].append(r)
    return dict(out)


def gold_found(task: Task, ids: set[str | None], titles: set[str]) -> tuple[int, int]:
    """(found, total) of the task's important citations among papers given by versionless arXiv
    ids and normalized titles: matched by id, or by title for a citation without a parsable id."""
    rows = important_citations(task.meta["dataset"]).get(task.id, [])

    def found(row: dict[str, str]) -> bool:
        if pid := arxiv_id(row["cited_paper_arxiv_link"]):
            return pid in ids
        title = norm_title(row["cited_paper_title"])
        return bool(title) and title in titles

    return sum(map(found, rows)), len(rows)


class DeepScholarBench(Benchmark):
    name = "deepscholar"
    metrics = METRICS
    judged = (  # organization, relevance rate and document importance only on request
        "nugget_coverage", "cite_p", "claim_coverage", "claim_support",
        "reference_coverage", "reference_recall",
    )  # fmt: skip
    primary = "nugget_coverage"
    cap_usd = 0.27  # B0: DS-base intro's most expensive dev task ($0.243) x 1.1
    reliability = ("cite_p", "cite_p_per_citation", "claim_coverage", "claim_support")
    secondary = ("reference_coverage", "reference_recall")
    derived_from: ClassVar[dict[str, str]] = {  # derived metrics share their source's status
        "cite_p_per_citation": "cite_p",
        "nugget_vital": "nugget_coverage",
        "nugget_okay": "nugget_coverage",
    }

    def tasks(self, split: str, limit: int | None = None) -> list[Task]:
        dataset = DATASETS[split]
        rows = {p["arxiv_id"]: (row, p) for row, p in enumerate(papers(dataset))}
        return [self._task(dataset, *rows[i]) for i in split_ids(self.name, split, limit)]

    def undefined(self, task: Task) -> set[str]:
        """Metrics that do not exist for the task: reference recall without important citations
        (upstream's reference coverage scores such a task 0)."""
        return (
            set()
            if important_citations(task.meta["dataset"]).get(task.id)
            else {"reference_recall"}
        )

    def gold(self, task: Task) -> dict[str, Any]:
        """The paper row, its important citations and its nugget file."""
        dataset, row = task.meta["dataset"], task.meta["row"]
        nuggets = data.DERIVED / "deepscholar" / dataset / "nuggets" / str(row) / "res.json"
        return {
            "paper": papers(dataset)[row],
            "important": important_citations(dataset).get(task.id, []),
            "nuggets": read_json(nuggets),
        }

    def derived(self, ev: dict[str, Any]) -> dict[str, float | None]:
        """`cite_p_per_citation`: the share of supported citations, the paper's definition
        ("averaging the precision of each citation"). Upstream's CitePEvaluator instead averages
        per sentence and scores uncited sentences 0: the authors' own reports (vendored fixtures)
        score 0.79 by the first and 0.46 by the second, so only the first is comparable with the
        paper. `nugget_vital` / `nugget_okay`: nugget coverage over the vital / okay nuggets
        alone, where the eval file keeps the nugget assignments."""
        if not ev.get("metrics"):
            return {}
        items = ev.get("items", [])
        cites = [it["decision"] for it in items if it["metric"] == "cite_p"]
        ok = ev["metrics"].get("cite_p") is not None and cites
        out = {"cite_p_per_citation": sum(cites) / len(cites) if ok else None}
        if ev["metrics"].get("nugget_coverage") is not None:
            for importance in ("vital", "okay"):
                labels = [
                    it["decision"] == "support"
                    for it in items
                    if it["metric"] == "nugget_coverage" and it.get("importance") == importance
                ]
                if labels:
                    out[f"nugget_{importance}"] = sum(labels) / len(labels)
        return out

    def _task(self, dataset: str, row: int, paper: dict[str, str]) -> Task:
        published = paper["published_date"]
        day = datetime.fromisoformat(published).astimezone(UTC).date()
        policy = AccessPolicy(
            cutoff=datetime(day.year, day.month, day.day, tzinfo=UTC),
            blocked_arxiv=frozenset({arxiv_id(paper["arxiv_id"])}),
            blocked_titles=(norm_title(paper["title"]),),
            undated_web="drop",
        )
        prompt = query_template().format(cutoff_date=published, abstract=paper["abstract"])
        meta = {
            "row": row,
            "dataset": dataset,
            "title": paper["title"],
            "published_date": published,
            "abstract": paper["abstract"],
        }
        return Task(bench=self.name, id=paper["arxiv_id"], prompt=prompt, policy=policy, meta=meta)

    def export_task(self, task: Task, result: TaskResult | None, out: Path) -> int:
        report, dropped = exported_report(task, result)
        intro, rows = deepscholar_intro(report) if report else ("", [])
        folder = out / str(task.meta["row"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "intro.md").write_text(intro, encoding="utf-8")
        with (folder / "paper.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("id", "title", "snippet"))
            writer.writeheader()
            writer.writerows(rows)
        return dropped

    def exported_text(self, task: Task, export: Path) -> str:
        return (export / str(task.meta["row"]) / "intro.md").read_text(encoding="utf-8")

    def setup(self, ctx: EvalContext) -> None:
        from jevscholar.benchmarks.deepscholar_eval import setup

        setup(ctx)

    def evaluate_task(self, task: Task, export: Path, ctx: EvalContext) -> dict[str, Any]:
        from jevscholar.benchmarks.deepscholar_eval import evaluate

        return evaluate(task, export, ctx)

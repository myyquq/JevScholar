"""DeepResearch Bench II: long-form research reports, judged on binary rubric items.

The input is the official `prompt`, which already tells the system not to use the task's source
article; here the same rule is also enforced at the tool level for every system (the blocked
URLs, canonicalised, the arXiv ids they contain, so mirrors and arXiv search are covered too, and
the article's title). There is no date cutoff, as in the official protocol. Exports are
`idx-<n>.md`, the official evaluator's input layout.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from jevscholar import data
from jevscholar.benchmarks import exported_report, split_ids
from jevscholar.render import drb2_report
from jevscholar.search.policy import arxiv_id, canonical_url, norm_title
from jevscholar.types import AccessPolicy, Benchmark, EvalContext, Task, TaskResult
from jevscholar.util import read_json, read_jsonl


@cache
def _rows() -> dict[str, dict[str, Any]]:
    """idx -> task with its rubric (the evaluator's tasks_and_rubrics.jsonl)."""
    return {str(t["idx"]): t for t in read_jsonl(data.DRB2_TASKS)}


class DRB2(Benchmark):
    name = "drb2"
    primary = "total"
    cap_usd = 0.34  # B0: DS-base's natural p90 cost on the DRB2 dev tasks

    def tasks(self, split: str, limit: int | None = None) -> list[Task]:
        return [self._task(_rows()[i]) for i in split_ids(self.name, split, limit)]

    def gold(self, task: Task) -> Any:
        return _rows()[task.id]

    def _task(self, t: dict[str, Any]) -> Task:
        blocked = t["content"].get("blocked") or {}
        urls = blocked.get("urls", ())
        policy = AccessPolicy(
            blocked_arxiv=frozenset(filter(None, map(arxiv_id, urls))),
            blocked_urls=frozenset(map(canonical_url, urls)),
            blocked_titles=(norm_title(blocked["title"]),) if blocked.get("title") else (),
        )
        return Task(
            bench=self.name,
            id=str(t["idx"]),
            prompt=t["prompt"],
            policy=policy,
            meta={"language": t["language"], "theme": t["theme"]},
        )

    def export_task(self, task: Task, result: TaskResult | None, out: Path) -> int:
        report, dropped = exported_report(task, result)
        out.mkdir(parents=True, exist_ok=True)
        text = drb2_report(report) if report else ""
        (out / f"idx-{task.id}.md").write_text(text, encoding="utf-8")
        return dropped

    def exported_text(self, task: Task, export: Path) -> str:
        return (export / f"idx-{task.id}.md").read_text(encoding="utf-8")

    def items(self, ev: dict[str, Any], eval_dir: Path, task: Task) -> list[dict[str, Any]]:
        """Every rubric item's score, from the evaluator's raw log."""
        if (found := ev.get("items")) is not None:
            return found
        raw = read_json(eval_dir / "raw" / f"idx-{task.id}.json", {"result": {}})
        return [
            {"metric": dim, "decision": v["score"], "context": item}
            for dim, items in raw["result"].get("scores", {}).items()
            for item, v in items.items()
        ]

    def evaluate_task(self, task: Task, export: Path, ctx: EvalContext) -> dict[str, Any]:
        from jevscholar.benchmarks.drb2_eval import evaluate

        return evaluate(task, export, ctx)

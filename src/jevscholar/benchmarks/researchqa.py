"""ResearchQA: answer a survey-derived research question, judged against its rubric items.

dev is the `valid` split; test is the official 776-question `test_mini` subset of `test`. The
prompt follows the paper's protocol (query, its date, about 250 words); the cutoff is the
question's date (partial dates floor to the period's first day) and undated web pages are kept,
since most of the web has no machine-readable date. Exports are `<id>.json` = {"answer": ...},
one entry of the official responses.json.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jevscholar import data
from jevscholar.benchmarks import exported_report, split_ids
from jevscholar.render import researchqa_answer
from jevscholar.search.policy import floor_date
from jevscholar.types import AccessPolicy, Benchmark, EvalContext, Task, TaskResult
from jevscholar.util import read_json

INSTRUCTION = "Answer in about 250 words. Support claims with in-line numbered citations [n]."


@cache
def _questions(split: str) -> dict[str, dict[str, Any]]:
    """id -> question of the raw `valid` or `test` file."""
    return {q["id"]: q for q in read_json(data.RAW / "researchqa" / f"{split}.json")}


class ResearchQA(Benchmark):
    name = "researchqa"
    primary = "coverage"

    def tasks(self, split: str, limit: int | None = None) -> list[Task]:
        by_id = _questions("valid" if split == "dev" else "test")
        return [self._task(by_id[i]) for i in split_ids(self.name, split, limit)]

    def gold(self, task: Task) -> Any:
        return next(q for s in ("valid", "test") if (q := _questions(s).get(task.id)))["rubric"]

    def _task(self, q: dict[str, Any]) -> Task:
        meta = {
            "query": q["query"],
            "date": q["date"],
            "field": q["field"],
            "general_domain": q["general_domain"],
            "rubric_types": sorted({t for r in q["rubric"] for t in r["type"]}),
        }
        return Task(
            bench=self.name,
            id=q["id"],
            prompt=f"{q['query']} (Date: {q['date']})\n\n{INSTRUCTION}",
            policy=AccessPolicy(cutoff=floor_date(q["date"]), undated_web="keep"),
            meta=meta,
        )

    def export_task(self, task: Task, result: TaskResult | None, out: Path) -> int:
        report, dropped = exported_report(task, result)
        answer = researchqa_answer(report) if report else ""
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{task.id}.json").write_text(
            json.dumps({"answer": answer}, ensure_ascii=False), encoding="utf-8"
        )
        return dropped

    def exported_text(self, task: Task, export: Path) -> str:
        return read_json(export / f"{task.id}.json")["answer"]

    def items(self, ev: dict[str, Any], eval_dir: Path, task: Task) -> list[dict[str, Any]]:
        """Every rubric item's label, from the evaluator's raw log."""
        if (found := ev.get("items")) is not None:
            return found
        return [
            {"metric": "coverage", "decision": j["score"], "context": j["rubric"]}
            for j in read_json(eval_dir / "raw" / f"{task.id}.json", [])
        ]

    def evaluate_task(self, task: Task, export: Path, ctx: EvalContext) -> dict[str, Any]:
        from jevscholar.benchmarks.researchqa_eval import evaluate

        return evaluate(task, export, ctx)

"""Diagnostic only: DeepScholar-base's intro writer (P10, no background) on fixed evidence.

It separates writing from retrieval (docs/03_baselines_and_metrics.md: the benchmark audit):
- one document set written under several seeds measures the writer's own variance;
- the task's important citations as the documents (`gold`) bound what retrieval could add to
  nugget coverage. That reads the benchmark's reference data, so this is never a system under
  comparison.

`evidence` lists the sources, combined in order, a paper given twice kept once:
- {"run": <run id>, "system": <system>}: what that system's writer got for the task in that run,
  in the order it got them: DS-base's native/paper.csv, or the documents a controlled system's
  writer got (its report's `docs`, rows from its state.json);
- "gold": the task's important citations with an arXiv id, looked up in the local index under
  the task's access policy, as DS-base's search rows.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

import pandas as pd

from jevscholar import paths
from jevscholar.artifacts import task_dir
from jevscholar.benchmarks.deepscholar import important_citations
from jevscholar.render import finalize
from jevscholar.search.policy import arxiv_id
from jevscholar.systems import deepscholar_base as dsb
from jevscholar.types import Report, Strict, System, Task, TaskContext
from jevscholar.util import read_json


class Recorded(Strict):
    run: str
    system: str


class FixedEvidence(System):
    class Params(Strict):
        evidence: tuple[Literal["gold"] | Recorded, ...]

    async def run(self, ctx: TaskContext) -> Report:
        top, task = dsb.install(), ctx.task
        with dsb.task_scope(ctx.search, task.seed, repair_citations=True):
            docs = pd.DataFrame(self.documents(ctx))
            if docs.empty:
                raise RuntimeError("no evidence to write from")
            configs = dsb.build_configs(ctx, {"use_sem_filter": False})
            text = await top.generate_intro_section(task.prompt, docs, "", configs)
        report = dsb.to_report(text, docs)
        return finalize(report.text, report.sources, **report.meta, status="ok")

    def documents(self, ctx: TaskContext) -> list[dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}  # by versionless arXiv id, the first kept
        for source in self.p.evidence:
            rows = gold(ctx) if source == "gold" else recorded(ctx.task, **source.model_dump())
            for row in rows:
                docs.setdefault(arxiv_id(row["url"]) or row["url"], row)
        return list(docs.values())


def recorded(task: Task, run: str, system: str) -> list[dict[str, Any]]:
    """The documents `system`'s writer got for the task in `run`, in order."""
    folder = task_dir(paths.RUNS / run, task.bench, system, task)
    if (native := folder / "native" / "paper.csv").exists():  # DS-base: its writer's input
        return pd.read_csv(native, dtype=str, keep_default_na=False).to_dict("records")
    report = read_json(folder / "result.json")["report"]
    state = read_json(folder / "state.json")
    rows = {p["row"]["url"]: p["row"] for p in state["papers"] if p["relevant"]}
    return [rows[url] for url in report["meta"]["docs"]]


def gold(ctx: TaskContext) -> list[dict[str, Any]]:
    """The task's important citations as DS-base's normalized arXiv search rows."""
    task = ctx.task
    rows = important_citations(task.meta["dataset"]).get(task.id, [])
    ids = list(dict.fromkeys(filter(None, (arxiv_id(r["cited_paper_arxiv_link"]) for r in rows))))
    found = ctx.search.arxiv_lookup(ids)
    if missing := [i for i in ids if i not in found]:  # not indexed, or after the cutoff
        raise RuntimeError(f"important citations unavailable to the oracle: {missing}")
    agentic = importlib.import_module("deepscholar_base.search.agentic_search")
    arxiv = agentic.ToolTypes.ARXIV
    _, df = agentic._normalize_search_df(
        dsb.arxiv_frame([found[i] for i in ids]),
        "gold",
        arxiv.to_rename_map(),
        lambda _: "",
        "",
    )
    return df.to_dict("records")

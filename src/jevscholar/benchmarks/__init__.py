"""Benchmarks: task construction (prompt + access policy) and the official export layouts.

Each class extends types.Benchmark. Its tasks come from the committed split lists in
data/splits/<bench>.json (a `limit` takes a stable prefix); scoring lives in <bench>_eval.py and
is imported only when a task is evaluated.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

from jevscholar import data
from jevscholar.render import drop_sources
from jevscholar.search.policy import source_allowed
from jevscholar.types import BENCHMARKS, Benchmark, Report, Task, TaskResult, load_class
from jevscholar.util import read_json

if TYPE_CHECKING:
    from jevscholar.config import Experiment


@functools.cache
def get(name: str) -> Benchmark:
    return load_class(BENCHMARKS[name])()


def split_ids(bench: str, split: str, limit: int | None = None) -> list[str]:
    return read_json(data.SPLITS / f"{bench}.json")[split][:limit]


def load_tasks(exp: Experiment, name: str) -> list[Task]:
    """A cell's task universe: the split prefix (or its `ids`), with the experiment seed."""
    cell = exp.benchmarks[name]
    tasks = get(name).tasks(cell.split, cell.limit)
    if cell.ids is not None:
        by_id = {t.id: t for t in tasks}
        if missing := [i for i in cell.ids if i not in by_id]:
            raise ValueError(f"{name}: {missing} are not in {cell.split}[:{cell.limit}]")
        tasks = [by_id[i] for i in cell.ids]
    return [dataclasses.replace(t, seed=exp.seed) for t in tasks]


def task_cap(exp: Experiment, name: str, system: str) -> float:
    """A system's per-task cap on benchmark `name`: its own `task_usd`, else the experiment
    budget's, else the benchmark's B0."""
    for cap in (
        exp.systems.get(system, {}).get("task_usd"),
        exp.budget.task_usd,
        get(name).cap_usd,
    ):
        if cap is not None:
            return cap
    raise ValueError(f"{name} has no cap B0: set budget.task_usd")


def task_caps(exp: Experiment) -> dict[tuple[str, str], float]:
    """(benchmark, system) -> the per-task cap of every cell of the experiment."""
    return {(b, s): task_cap(exp, b, s) for b, cell in exp.benchmarks.items() for s in cell.systems}


def exported_report(task: Task, result: TaskResult | None) -> tuple[Report | None, int]:
    """The report to export (None, an empty export scored 0, for missing or failed tasks) and the
    number of cited sources dropped because the task's policy disallows them.

    The search layer filters what tools return, but a report can still cite a URL no tool
    returned (e.g. one copied from page text); this audit applies the same rule to every system."""
    if result is None or result.status == "failed" or result.report is None:
        return None, 0
    return drop_sources(result.report, lambda s: source_allowed(s, task.policy))

"""Run directories: their layout and readers.

runs/<run_id>/manifest.json                    the resolved experiment, cells, tasks, sessions
runs/<run_id>/ledger.jsonl                     every metered LLM call and tool charge
runs/<run_id>/<bench>/<system>/tasks/<key>/    result.json (+ a system's own logs)
runs/<run_id>/<bench>/<system>/export/         the official layout, audit.json
runs/<run_id>/<bench>/<system>/eval/<judges>/  <key>.json per judged task
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jevscholar import benchmarks
from jevscholar.config import Experiment
from jevscholar.paths import RUNS
from jevscholar.types import Task, TaskResult, load_result, task_key
from jevscholar.util import read_json


def run_dir(run: str) -> Path:
    """A run directory: a path, or a run id under runs/."""
    path = Path(run)
    return path if (path / "manifest.json").exists() else RUNS / run


def read_manifest(run: Path) -> dict[str, Any]:
    return read_json(run / "manifest.json", {})


def experiment(run: Path) -> Experiment:
    return Experiment.model_validate(read_manifest(run)["experiment"])


def task_dir(run: Path, bench: str, system: str, task: Task | str) -> Path:
    key = task if isinstance(task, str) else task_key(task)
    return run / bench / system / "tasks" / key


def read_result(folder: Path) -> TaskResult | None:
    """A task folder's result (None: not run yet)."""
    path = folder / "result.json"
    return load_result(path) if path.exists() else None


def run_tasks(manifest: dict[str, Any], bench: str) -> dict[str, Task]:
    """A run's tasks of `bench` by task key (those of its split prefix the manifest lists)."""
    exp = Experiment.model_validate(manifest["experiment"])
    tasks = {task_key(t): t for t in benchmarks.load_tasks(exp, bench)}
    return {key: tasks[key] for key in manifest["tasks"][bench] if key in tasks}

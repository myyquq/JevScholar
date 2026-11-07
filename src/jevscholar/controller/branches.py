"""Counterfactual branches of logged decision points, and the SFT targets they give (docs/06).

A branch restores a decision point of a controlled episode run (`start`), executes one of the
candidates the rules offered there (`take`) instead of asking the policy, and continues with a
fixed continuation policy (by default `ctl_fixed`'s) to the end, writing included. Every legal
candidate of a point is branched under the same continuation, so the branches of one point
differ only in the decision there.

- `forks` lists the points worth branching in a source run: decisions the rules left a choice
  (two or more legal candidates) in finished episodes of tasks with important citations. Only
  `train` and `val` runs are sources: dev and test trajectories never become labels.
- `experiments` turns them into one small experiment per (batches, kind): its tasks are those
  whose point after that many batches offers that kind, so no branch is refused. Each keeps the
  source system's settings (cap, rules, costs, actions, sample) and the source experiment's
  executor, search and seed, so a restored point offers exactly what the source saw.
- `outcomes` scores each finished branch judge-free with the retrieval return (rewards.py):
  important citations among the writer's documents, less the task's whole list spend (the
  source's spend up to the point included, the same for every branch of a point).
- `examples` gives each point whose every legal candidate has an outcome a soft target: the
  options whose return ties the best share the mass, the rest get 0 (the return is
  deterministic given the evidence, so only exact ties share). The example is what d1 reads at
  that point under its default profile.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jevscholar import benchmarks, paths, rewards
from jevscholar.artifacts import read_manifest, run_tasks, task_dir
from jevscholar.benchmarks.deepscholar import gold_found
from jevscholar.config import Experiment, resolve_systems
from jevscholar.controller.d1 import Example
from jevscholar.controller.prompt import Profile
from jevscholar.controller.state import SNAPSHOT, State
from jevscholar.search.policy import arxiv_id, norm_title
from jevscholar.types import Task
from jevscholar.util import read_json, read_jsonl

BENCH = "deepscholar"
SPLITS = ("train", "val")  # the splits whose decisions may become labels (val: held out)
SYSTEM = "branch"  # the system name of every branch experiment
CONTINUATION = "ctl_fixed"  # the catalog system whose policy continues every branch
TIE = 1e-9  # returns this close to the best share the target


@dataclass(frozen=True)
class Fork:
    """A decision point of a source run worth branching: the task, the retrieval batches before
    it (what `start.batches` names) and the kinds of its legal candidates, in their order."""

    task: str  # the task id
    key: str  # its task key (folder)
    batches: int
    kinds: tuple[str, ...]


def source_cell(manifest: dict[str, Any], system: str) -> tuple[str, dict[str, Any]]:
    """(split, resolved system spec) of a source run's controlled episode system."""
    exp = manifest["experiment"]
    cell = exp["benchmarks"].get(BENCH)
    if cell is None or system not in cell["systems"]:
        raise ValueError(f"the run has no {BENCH}/{system} cell")
    if cell["split"] not in SPLITS:
        raise ValueError(f"branches come from {' or '.join(SPLITS)} runs only, not {cell['split']}")
    spec = exp["systems"].get(system, {"use": system})
    if spec.get("use") != "controlled":
        raise ValueError(f"{system} is not a controlled system")
    if spec.get("start"):
        raise ValueError(f"{system} continues another run: branch from an episode run instead")
    return cell["split"], spec


def forks(
    source: Path, system: str, *, tasks: int | None = None, per_task: int | None = None
) -> list[Fork]:
    """The decision points of `system`'s finished episodes in `source` that the rules left a
    choice, on tasks with important citations: the first `tasks` tasks of the run, at most
    `per_task` points each (a fixed pseudo-random pick, then in episode order)."""
    manifest = read_manifest(source)
    source_cell(manifest, system)
    out: list[Fork] = []
    for key, task in list(run_tasks(manifest, BENCH).items())[:tasks]:
        folder = task_dir(source, BENCH, system, key)
        if not (folder / "decisions.jsonl").exists() or not gold_found(task, set(), set())[1]:
            continue
        if read_json(folder / "result.json", {}).get("status") != "ok":
            continue
        points = []
        for record in read_jsonl(folder / "decisions.jsonl"):
            if (version := record.get("version")) != SNAPSHOT:
                raise ValueError(f"{folder}: decisions in format {version}, not {SNAPSHOT}")
            kinds = tuple(c["kind"] for c in record["candidates"])
            if len(kinds) > 1 and not record.get("taken", False):
                points.append(Fork(task.id, key, record["state"]["batches"], kinds))
        if per_task is not None:
            pick = sorted(points, key=lambda f: digest(f"{key}|{f.batches}"))
            points = sorted(pick[:per_task], key=lambda f: f.batches)
        out += points
    return out


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def continuation_policy(name: str = CONTINUATION) -> dict[str, Any]:
    """The policy of a catalog system (its `extends` resolved)."""
    import yaml

    catalog = yaml.safe_load(paths.CATALOG.read_text()) or {}
    spec = resolve_systems({"benchmarks": {BENCH: {"systems": [name]}}}, catalog)["systems"]
    if "policy" not in spec.get(name, {}):
        raise ValueError(f"{name} has no policy to continue with")
    return spec[name]["policy"]


def run_name(source: Path) -> str:
    """The source run as `start.run` names it: its path under runs/."""
    try:
        return source.resolve().relative_to(paths.RUNS.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"{source}: branches restore from runs under {paths.RUNS}") from None


def experiments(
    source: Path,
    system: str,
    found: Iterable[Fork],
    *,
    policy: dict[str, Any],
    run_usd: float,
    prefix: str | None = None,
) -> dict[str, Experiment]:
    """Run id -> the branch experiment of one (batches, kind): the forks whose point offers that
    kind, each restored and executing it, then continued with `policy`."""
    start_run, prefix = run_name(source), prefix or f"branches-{source.name}"
    split, spec = source_cell(manifest := read_manifest(source), system)
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for fork in found:
        for kind in fork.kinds:
            groups[fork.batches, kind].append(fork.task)
    base = manifest["experiment"]
    out = {}
    for (batches, kind), ids in sorted(groups.items()):
        branch = spec | {
            "policy": policy,
            "start": {"run": start_run, "system": system, "batches": batches, "take": kind},
        }
        raw = base | {
            "name": f"branches/{start_run}/b{batches}-{kind}",
            "replay_only": False,
            "budget": base["budget"] | {"run_usd": run_usd},
            "systems": {SYSTEM: branch},
            "benchmarks": {BENCH: {"split": split, "ids": ids, "systems": [SYSTEM]}},
        }
        out[f"{prefix}-b{batches}-{kind}"] = Experiment.model_validate(raw)
    return out


# --------------------------------------------------------------------------- labels
def outcomes(runs: Iterable[Path]) -> list[dict[str, Any]]:
    """Each finished branch of the branch runs with its retrieval return: {"split", "source",
    "system", "task", "batches", "take", "return", "recall", "cost_usd"}."""
    b0 = benchmarks.get(BENCH).cap_usd
    out = []
    for run in runs:
        manifest = read_manifest(run)
        exp = manifest["experiment"]
        cell = exp["benchmarks"][BENCH]
        start = exp["systems"].get(SYSTEM, {}).get("start") or {}
        if cell["systems"] != [SYSTEM] or start.get("take") is None:
            raise ValueError(f"{run} is not a branch run")
        if cell["split"] not in SPLITS:
            raise ValueError(f"{run}: labels come from {' or '.join(SPLITS)} only")
        for key, task in run_tasks(manifest, BENCH).items():
            folder = task_dir(run, BENCH, SYSTEM, key)
            if read_json(folder / "result.json", {}).get("status") != "ok":
                continue
            state = State.from_json((folder / "state.json").read_text())
            recall = writer_recall(task, state)
            if recall is None:  # no important citations: no label (forks leaves such tasks out)
                continue
            out.append(
                {
                    "split": cell["split"],
                    "source": start["run"],
                    "system": start["system"],
                    "task": key,
                    "batches": start["batches"],
                    "take": start["take"],
                    "return": rewards.retrieval_return(recall, state.spent_usd, b0),
                    "recall": recall,
                    "cost_usd": state.spent_usd,
                }
            )
    return out


def writer_recall(task: Task, state: State) -> float | None:
    """R_e: the share of the task's important citations among the writer's documents (None for a
    task without important citations)."""
    rows = {p.url: p.row for p in state.papers}
    docs = [rows[url] for url in state.docs]
    ids, titles = {arxiv_id(r["url"]) for r in docs}, {norm_title(r["title"]) for r in docs}
    found, total = gold_found(task, ids, titles)
    return found / total if total else None


def soft_target(returns: dict[str, float]) -> dict[str, float]:
    """The options whose return ties the best, sharing the mass equally."""
    best = max(returns.values())
    tied = [name for name, r in returns.items() if r >= best - TIE]
    return {name: 1 / len(tied) for name in tied}


def examples(
    found: Iterable[dict[str, Any]], profile: Profile | None = None
) -> tuple[list[Example], list[dict[str, Any]], int]:
    """(examples, points, incomplete): one example per decision point whose every legal
    candidate has a branch outcome, as d1 reads that point under `profile`; each point's record
    with its options' returns; and the number of points left out for a missing branch. All
    outcomes must come from one split."""
    from jevscholar.systems.controlled import Start, locate

    found, profile = list(found), profile or Profile()
    if len({o["split"] for o in found}) > 1:
        raise ValueError("outcomes of several splits: label each split on its own")
    by_point: dict[tuple[str, str, str, int], dict[str, float]] = defaultdict(dict)
    for o in found:
        by_point[o["source"], o["system"], o["task"], o["batches"]][o["take"]] = o["return"]
    out, points, incomplete = [], [], 0
    tasks: dict[str, dict[str, Task]] = {}
    for (source, system, key, batches), returns in sorted(by_point.items()):
        if source not in tasks:
            tasks[source] = run_tasks(read_manifest(paths.RUNS / source), BENCH)
        task = tasks[source][key]
        point = locate(task, Start(run=source, system=system, batches=batches))
        kinds = [c.kind for c in point.candidates]
        if set(returns) != set(kinds):
            incomplete += 1
            continue
        request = profile.request(point.state, point.candidates)
        (question,) = request["questions"]
        names = dict(zip(kinds, question["criteria"], strict=True))
        target = soft_target({names[k]: r for k, r in returns.items()})
        out.append(Example(request["state"], question, target, render=profile.render))
        points.append(
            {"source": source, "system": system, "task": key, "batches": batches}
            | {"kinds": kinds, "returns": [returns[k] for k in kinds], "target": target}
        )
    return out, points, incomplete

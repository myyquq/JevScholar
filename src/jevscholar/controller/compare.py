"""Comparing d1 adapters on val (docs/06): d1-sft against d1-rl, judge-free.

Two views, both on the 9 val tasks, which never become RL data:

- **Decisions** (`decision_scores`): the val decision points whose every option was branched
  (`jev controller labels`: `dev.jsonl` and `dev.points.jsonl`), each option with its branch's
  retrieval return. For an adapter's option probabilities: the regret of its choice (as the
  served policy picks: `finish` at the profile's threshold, else the most probable) against the
  best option, the expected regret of its distribution, how often it picks a best option, and
  the cross-entropy against the soft target. No executor call: only d1 on the GPU.
- **Episodes** (`episode_experiment`, `episode_returns`, `compare`): every val task run from the
  start with the adapter choosing (`policy: {kind: d1, adapter: <sha256>}`, no sampling), under
  the source run's settings (its cap, rules, costs, executor, search and seed). Each finished
  task is scored with the retrieval return (important citations among the writer's documents,
  less the task's list spend) and compared per task with other systems, e.g. the fixed rule's
  own val run.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from jevscholar import benchmarks, rewards
from jevscholar.artifacts import read_manifest, run_tasks, task_dir
from jevscholar.config import Experiment
from jevscholar.controller.branches import BENCH, source_cell, writer_recall
from jevscholar.controller.prompt import Profile
from jevscholar.controller.state import Candidate, State
from jevscholar.util import read_json

TIE = 1e-9  # returns this close to the best count as best


# --------------------------------------------------------------------------- decisions
def decision_scores(
    probs: Sequence[Sequence[float]],
    points: Sequence[dict[str, Any]],
    profile: Profile | None = None,
) -> dict[str, Any]:
    """Regret and agreement of option distributions (in each point's option order) against the
    points' branch returns."""
    profile = profile or Profile()
    if len(probs) != len(points):
        raise ValueError(f"{len(probs)} distributions for {len(points)} points")
    rows = []
    for dist, point in zip(probs, points, strict=True):
        returns, kinds = point["returns"], point["kinds"]
        if len(dist) != len(returns):
            raise ValueError(f"{point['task']}@{point['batches']}: options do not match")
        best = max(returns)
        chosen = profile.pick([Candidate(k) for k in kinds], dist)
        target = [point["target"].get(name, 0.0) for name in point.get("names", [])] or None
        rows.append(
            {
                "regret": best - returns[chosen],
                "expected_regret": best - sum(p * r for p, r in zip(dist, returns, strict=True)),
                "best": returns[chosen] >= best - TIE,
                "finish": kinds[chosen] == "finish",
                "ce": None
                if target is None
                else -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, dist, strict=True)),
            }
        )
    n = len(rows)
    ces = [r["ce"] for r in rows if r["ce"] is not None]
    return {
        "points": n,
        "regret": sum(r["regret"] for r in rows) / n,
        "expected_regret": sum(r["expected_regret"] for r in rows) / n,
        "best_rate": sum(r["best"] for r in rows) / n,
        "finish_rate": sum(r["finish"] for r in rows) / n,
        "ce": sum(ces) / len(ces) if ces else None,
    }


def with_names(points: Iterable[dict[str, Any]], examples: Iterable[Any]) -> list[dict[str, Any]]:
    """The points with their options' names (the example question's), so the soft target can be
    read in option order; points and examples are written side by side by `labels`."""
    out = []
    for point, e in zip(points, examples, strict=True):
        names = list(e.question["criteria"])
        if len(names) != len(point["kinds"]):
            raise ValueError(f"{point['task']}@{point['batches']}: example and point differ")
        out.append(point | {"names": names})
    return out


# --------------------------------------------------------------------------- episodes
def policy(adapter: str | None) -> dict[str, Any]:
    """The evaluated policy: d1 with the served adapter, choosing (no sampling)."""
    return {"kind": "d1", "adapter": adapter}


def episode_experiment(
    source: Path, system: str, name: str, *, adapter: str | None, run_usd: float
) -> Experiment:
    """Every task of the source run's cell, from the start, with d1 (`adapter`) choosing, as
    system `name`; the source system's settings and the source experiment's executor, search
    and seed."""
    split, spec = source_cell(manifest := read_manifest(source), system)
    base, cell = manifest["experiment"], manifest["experiment"]["benchmarks"][BENCH]
    raw = base | {
        "name": f"compare/{split}/{name}",
        "replay_only": False,
        "budget": base["budget"] | {"run_usd": run_usd},
        "systems": {name: spec | {"policy": policy(adapter)}},
        "benchmarks": {
            BENCH: {k: cell.get(k) for k in ("split", "limit", "ids")} | {"systems": [name]}
        },
    }
    return Experiment.model_validate(raw)


def episode_returns(run: Path, system: str) -> dict[str, dict[str, Any]]:
    """Task key -> {"status", "return", "recall", "cost_usd", "batches"} of a controlled
    system's episodes (return None for a failed task or one without important citations)."""
    b0 = benchmarks.get(BENCH).cap_usd
    manifest = read_manifest(run)
    if system not in manifest["experiment"]["benchmarks"][BENCH]["systems"]:
        raise ValueError(f"{run} has no {BENCH}/{system} cell")
    out = {}
    for key, task in run_tasks(manifest, BENCH).items():
        folder = task_dir(run, BENCH, system, key)
        status = read_json(folder / "result.json", {}).get("status", "missing")
        found: dict[str, Any] = {"status": status, "return": None, "recall": None}
        found |= {"cost_usd": None, "batches": None}
        if status == "ok":
            state = State.from_json((folder / "state.json").read_text())
            found |= {"cost_usd": state.spent_usd, "batches": state.batches}
            if (recall := writer_recall(task, state)) is not None:
                ret = rewards.retrieval_return(recall, state.spent_usd, b0)
                found |= {"return": ret, "recall": recall}
        out[key] = found
    return out


def compare(systems: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """One row per system on the tasks every system scored: means of return, recall, cost and
    batches; against the first system, the mean paired return difference with its standard
    error and the tasks won, tied and lost."""
    labels = list(systems)
    common = sorted(
        set.intersection(
            *({k for k, v in s.items() if v["return"] is not None} for s in systems.values())
        )
    )
    if not common:
        raise ValueError("no task scored by every system")
    first = systems[labels[0]]
    rows = []
    for label in labels:
        s = systems[label]
        row: dict[str, Any] = {"system": label, "tasks": len(common)}
        row["failed"] = sum(v["status"] != "ok" for v in s.values())
        for field in ("return", "recall", "cost_usd", "batches"):
            row[field] = statistics.fmean(s[k][field] for k in common)
        diffs = [s[k]["return"] - first[k]["return"] for k in common]
        se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else None
        row |= {"diff": statistics.fmean(diffs), "diff_se": se}
        row["won"] = sum(d > TIE for d in diffs)
        row["tied"] = sum(abs(d) <= TIE for d in diffs)
        row["lost"] = sum(d < -TIE for d in diffs)
        rows.append(row)
    return rows


def table(rows: list[dict[str, Any]]) -> str:
    """The comparison as a Markdown table (differences against the first row)."""
    head = (
        "| System | Tasks | Return | Δ vs first (±SE) | W/T/L | Recall | Cost | Batches | Failed |"
    )
    lines = [head, "|" + "---|" * 9]
    for r in rows:
        se = "" if r["diff_se"] is None else f" ± {r['diff_se']:.4f}"
        lines.append(
            f"| {r['system']} | {r['tasks']} | {r['return']:.4f} | {r['diff']:+.4f}{se} "
            f"| {r['won']}/{r['tied']}/{r['lost']} | {r['recall']:.3f} | ${r['cost_usd']:.3f} "
            f"| {r['batches']:.1f} | {r['failed']} |"
        )
    return "\n".join(lines)

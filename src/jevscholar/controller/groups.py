"""Bounded RL groups (docs/06, "Bounded RL"): sampled trajectories from train decision points,
their judge-free rewards, group-relative advantages and the RL examples they give.

- `points` picks the decision points of a train episode run to sample from: `branches.forks`
  (choices the rules left open, in finished episodes of tasks with important citations), at most
  one per task, a fixed pseudo-random pick per round.
- `experiments` turns them into one run per `start.batches` value (as `branches.experiments`
  does), whose `ids` are the tasks picked at that value, with `size` systems g0, g1, ... that
  differ only in their name. Each restores the point and lets the served adapter sample from
  there on (`policy: {kind: d1, sample: true, adapter: <sha256>}`); sampling is seeded by
  `ctx.tag('control')`, which holds the run and the system, so every member samples on its own.
  Each member keeps the source system's cap, rules, costs and actions, and the source
  experiment's executor, search and seed, so it is offered exactly what the source was.
- `outcomes` scores each member with the retrieval return, as `branches.outcomes` does:
  important citations among the writer's documents, less the task's whole list spend.
- `advantages` groups the members of a point: A = (R - mean) / max(SD, SD_FLOOR), with the
  population SD. A group with a missing or failed member, or whose returns all lie within
  NEAR_TIE, is skipped.
- `examples` turns every decision the policy made in an advantaged trajectory into an
  `RLExample`: exactly what d1 read (the request's state and question, and the render), the
  requested option and its probability then. Forced, overridden and taken decisions stay out.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jevscholar import benchmarks, rewards
from jevscholar.artifacts import read_manifest, run_tasks, task_dir
from jevscholar.config import Experiment
from jevscholar.controller.branches import (
    BENCH,
    Fork,
    digest,
    forks,
    run_name,
    source_cell,
    writer_recall,
)
from jevscholar.controller.d1 import RLExample
from jevscholar.controller.state import State
from jevscholar.util import read_json, read_jsonl

SPLIT = "train"  # RL samples only train points (val stays held out)
SIZE = 4  # trajectories per group
SD_FLOOR = 0.02  # the advantage's denominator never falls below this
NEAR_TIE = 1e-3  # a group whose returns span less than this is skipped (about a third of a batch)


def member(i: int) -> str:
    return f"g{i}"


def policy(adapter: str) -> dict[str, Any]:
    """The sampling policy of every group member: d1 with the served adapter (its sha256)."""
    return {"kind": "d1", "sample": True, "adapter": adapter}


def points(source: Path, system: str, groups: int, rl_round: int = 1) -> list[Fork]:
    """`groups` decision points of a train run, at most one per task, a fixed pseudo-random pick
    per round (each round samples other points), in (batches, task) order."""
    split, _ = source_cell(read_manifest(source), system)
    if split != SPLIT:
        raise ValueError(f"RL groups come from {SPLIT} runs only, not {split}")
    found = forks(source, system, per_task=1)
    pick = sorted(found, key=lambda f: digest(f"rl{rl_round}|{f.key}|{f.batches}"))[:groups]
    return sorted(pick, key=lambda f: (f.batches, f.key))


def experiments(
    source: Path,
    system: str,
    found: Iterable[Fork],
    *,
    adapter: str,
    run_usd: float,
    size: int = SIZE,
    prefix: str | None = None,
) -> dict[str, Experiment]:
    """Run id -> the group experiment of one batches value: the picked points at that value,
    each restored and sampled by `size` identical members."""
    start_run, prefix = run_name(source), prefix or f"groups-{source.name}"
    split, spec = source_cell(manifest := read_manifest(source), system)
    by_batches: dict[int, list[str]] = defaultdict(list)
    for fork in found:
        by_batches[fork.batches].append(fork.task)
    base = manifest["experiment"]
    out = {}
    for batches, ids in sorted(by_batches.items()):
        spec_g = spec | {
            "policy": policy(adapter),
            "start": {"run": start_run, "system": system, "batches": batches},
        }
        names = [member(i) for i in range(size)]
        raw = base | {
            "name": f"groups/{start_run}/b{batches}",
            "replay_only": False,
            "budget": base["budget"] | {"run_usd": run_usd},
            "systems": {name: spec_g for name in names},
            "benchmarks": {BENCH: {"split": split, "ids": ids, "systems": names}},
        }
        out[f"{prefix}-b{batches}"] = Experiment.model_validate(raw)
    return out


# --------------------------------------------------------------------------- rewards
def group_cell(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """(the deepscholar cell, its members, their shared spec) of a group run."""
    exp = manifest["experiment"]
    cell = exp["benchmarks"][BENCH]
    names = list(cell["systems"])
    specs = [exp["systems"].get(n, {}) for n in names]
    if not names or names != [member(i) for i in range(len(names))]:
        raise ValueError("not a group run: its systems are not g0, g1, ...")
    spec = specs[0]
    if any(s != spec for s in specs) or not (spec.get("start") or {}).get("run"):
        raise ValueError("not a group run: its members differ or restore no point")
    if spec["start"].get("take") is not None or not spec["policy"].get("sample"):
        raise ValueError("not a group run: its members do not sample")
    if cell["split"] != SPLIT:
        raise ValueError(f"RL data comes from {SPLIT} only, not {cell['split']}")
    return cell, names, spec


def outcomes(runs: Iterable[Path]) -> list[dict[str, Any]]:
    """Each member of each group: {"run", "task", "batches", "member", "adapter", "status",
    "return", "recall", "cost_usd"} (return None unless the episode finished)."""
    b0 = benchmarks.get(BENCH).cap_usd
    out = []
    for run in runs:
        manifest = read_manifest(run)
        _, names, spec = group_cell(manifest)
        for key, task in run_tasks(manifest, BENCH).items():
            for name in names:
                folder = task_dir(run, BENCH, name, key)
                status = read_json(folder / "result.json", {}).get("status", "missing")
                found = {"run": str(run), "task": key, "batches": spec["start"]["batches"]}
                found |= {"member": name, "adapter": spec["policy"].get("adapter")}
                found |= {"status": status, "return": None, "recall": None, "cost_usd": None}
                if status == "ok":
                    state = State.from_json((folder / "state.json").read_text())
                    recall = writer_recall(task, state)
                    if recall is not None:
                        ret = rewards.retrieval_return(recall, state.spent_usd, b0)
                        found |= {"return": ret, "recall": recall, "cost_usd": state.spent_usd}
                out.append(found)
    return out


def advantage(returns: list[float]) -> list[float]:
    """A = (R - mean) / max(SD, SD_FLOOR), the population SD over the group."""
    mean, sd = statistics.fmean(returns), statistics.pstdev(returns)
    return [(r - mean) / max(sd, SD_FLOOR) for r in returns]


def advantages(found: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """(groups, skipped): each usable group {"run", "task", "batches", "members", "returns",
    "advantages"}, and the groups skipped by reason ("incomplete": a member missing, failed or
    without gold; "tied": returns within NEAR_TIE)."""
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for o in found:
        by_group[o["run"], o["task"]].append(o)
    out, skipped = [], {"incomplete": 0, "tied": 0}
    for (run, task), members in sorted(by_group.items()):
        returns = [m["return"] for m in members]
        if any(r is None for r in returns):
            skipped["incomplete"] += 1
            continue
        if max(returns) - min(returns) < NEAR_TIE:
            skipped["tied"] += 1
            continue
        out.append(
            {"run": run, "task": task, "batches": members[0]["batches"]}
            | {"members": [m["member"] for m in members], "returns": returns}
            | {"advantages": advantage(returns), "adapter": members[0]["adapter"]}
        )
    return out, skipped


# --------------------------------------------------------------------------- examples
def decision_example(
    record: dict[str, Any], advantage: float, adapter: str | None, meta: dict[str, Any]
) -> RLExample | None:
    """The RL example of one logged decision (None for a forced, overridden or taken one)."""
    if record.get("forced") or record.get("overridden") or record.get("taken"):
        return None
    inp = record.get("input")
    if not inp or "requests" not in inp:
        raise ValueError(f"{meta}: a decision without d1's input")
    if inp.get("adapter") != adapter:
        raise ValueError(f"{meta}: sampled by adapter {inp.get('adapter')}, not {adapter}")
    request = inp["requests"][0]
    (question,) = request["questions"]
    probs, chosen = record["probs"], record["requested"]
    if len(probs) != len(question["criteria"]):
        raise ValueError(f"{meta}: {len(probs)} probabilities for {len(question['criteria'])}")
    return RLExample(
        request["state"],
        question,
        chosen,
        probs[chosen],
        advantage,
        render=inp.get("render") or {},
        meta=meta | {"decision": record["state"]["decisions"]},
    )


def examples(groups: Iterable[dict[str, Any]]) -> list[RLExample]:
    """The RL examples of the groups' trajectories, every decision the policy made in them with
    its trajectory's advantage."""
    out = []
    for g in groups:
        run = Path(g["run"])
        for name, adv in zip(g["members"], g["advantages"], strict=True):
            folder = task_dir(run, BENCH, name, g["task"])
            meta = {"run": run.name, "task": g["task"], "batches": g["batches"], "member": name}
            for record in read_jsonl(folder / "decisions.jsonl"):
                if (e := decision_example(record, adv, g["adapter"], meta)) is not None:
                    out.append(e)
    return out

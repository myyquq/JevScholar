"""Bounded RL on the retrieval return (docs/06): frozen late decision points of a train run, groups
of sampled d1 continuations from them, their group-normalised advantages, and the steps the update
reads.

- `points` picks the registered points: decisions after 3-5 batches of finished episodes on
  tasks with important citations, offering at least two searches; one per task by a fixed
  pseudo-random pick, then `n` tasks by another. Only train runs are sources.
- `experiments` restores them (`start` without `take`) and continues each with `samples`
  sampled members of one policy, one experiment per batch count (`<prefix>-b<batches>`). Members
  are separate systems (`rollout`, `rollout_s1`, ...), so each member's sampling seed differs;
  episodes stop at the writer's documents.
- `groups` checks the rollout runs against the frozen recipe (its points, its members, their
  common start without `take`, each member's restored snapshot), scores each member judge-free
  by the retrieval return and gives every sampled decision of a usable group A = (R - group
  mean) / max(group SD, SD_FLOOR). A group with a failed or missing member, or with returns
  spanning less than TIE, trains nothing and is recorded; its finished members' decisions stay
  in the data (`usable` false), as the update's KL guard reads every sampled decision.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
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
    point_key,
    provenance,
    restored,
    run_name,
    source_cell,
    writer_recall,
)
from jevscholar.controller.state import FINISH, State
from jevscholar.util import read_json, read_jsonl, write_jsonl

LATE = (3, 5)  # batches before a point, inclusive
MEMBER = "rollout"  # the first member's system name; the others are rollout_s1, ...
SD_FLOOR = 0.02  # advantage = (R - mean) / max(SD, SD_FLOOR)
TIE = 1e-3  # a group whose returns span less is skipped


def points(source: Path, system: str, n: int = 8) -> list[Fork]:
    """The registered RL points of a train run's `system` episodes: per task, one late decision
    offering at least two searches (a fixed pseudo-random pick), then `n` of those tasks (another
    fixed pick), in the run's task order."""
    split, _ = source_cell(read_manifest(source), system)
    if split != "train":
        raise ValueError(f"RL points come from train runs only, not {split}")
    eligible: dict[str, list[Fork]] = {}
    for fork in forks(source, system):
        searches = sum(kind != FINISH for kind in fork.kinds)
        if LATE[0] <= fork.batches <= LATE[1] and searches >= 2:
            eligible.setdefault(fork.key, []).append(fork)
    one = {k: min(fs, key=lambda f: digest(f"{k}|{f.batches}")) for k, fs in eligible.items()}
    chosen = set(sorted(one, key=lambda k: digest(f"rl|{k}"))[:n])
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} tasks offer an eligible late point, not {n}")
    return [f for k, f in one.items() if k in chosen]


def members(samples: int) -> list[str]:
    return [MEMBER if i == 0 else f"{MEMBER}_s{i}" for i in range(samples)]


def experiments(
    source: Path,
    system: str,
    found: Iterable[Fork],
    *,
    policy: dict[str, Any],
    samples: int,
    run_usd: float,
    prefix: str,
) -> dict[str, Experiment]:
    """Run id -> the rollout experiment of the points after one batch count: each restored and
    continued by `samples` members of `policy`, which must sample, to the writer's documents."""
    if policy.get("kind") != "d1" or not policy.get("sample"):
        raise ValueError("rollouts sample from a d1 policy (`kind: d1, sample: true`)")
    start_run = run_name(source)
    split, spec = source_cell(manifest := read_manifest(source), system)
    by_batches: dict[int, list[str]] = {}
    for fork in found:
        by_batches.setdefault(fork.batches, []).append(fork.task)
    base = manifest["experiment"]
    out = {}
    for batches, ids in sorted(by_batches.items()):
        start = {"run": start_run, "system": system, "batches": batches}
        member = spec | {"policy": policy, "start": start, "write": False}
        names = members(samples)
        raw = base | {
            "name": f"rollouts/{start_run}/b{batches}",
            "replay_only": False,
            "budget": base["budget"] | {"run_usd": run_usd},
            "systems": {name: member | {"sample": i} for i, name in enumerate(names)},
            "benchmarks": {BENCH: {"split": split, "ids": ids, "systems": names}},
        }
        out[f"{prefix}-b{batches}"] = Experiment.model_validate(raw)
    return out


@dataclass(frozen=True)
class Step:
    """One sampled decision of a trajectory, as the update reads it: d1's exact request (state,
    question, rendering), the sampled option, the probabilities it was sampled from, its
    trajectory's advantage, and whether its group trains (`usable`; every decision counts in the
    KL guard)."""

    state: Any
    question: dict[str, Any]
    render: dict[str, str]
    action: int
    old: list[float]
    advantage: float
    adapter: str | None
    group: str
    trajectory: str
    usable: bool = True


def groups(
    runs: Iterable[Path],
    *,
    source: Path,
    system: str,
    planned: Sequence[tuple[str, int]],
    samples: int,
) -> tuple[list[Step], list[dict[str, Any]]]:
    """(steps, groups) of the rollout runs of a frozen recipe: every sampled decision of every
    finished member with its trajectory's advantage (0 and `usable` false where its group
    trains nothing), and each planned point's group record (returns, advantages, kinds taken,
    whether and why it was skipped). The runs must hold exactly the planned (task key, batches)
    points, each with `samples` members restored from `source`'s `system` point and sampling
    from one served adapter."""
    b0 = benchmarks.get(BENCH).cap_usd
    names, want = members(samples), {tuple(p) for p in planned}
    provenance(source, system)
    found: dict[tuple[str, int], tuple[Path, Any]] = {}
    for run in runs:
        manifest = read_manifest(run)
        exp = manifest["experiment"]
        cell = exp["benchmarks"][BENCH]
        specs = [exp["systems"].get(name, {}) for name in cell["systems"]]
        if cell["split"] != "train" or list(cell["systems"]) != names:
            raise ValueError(f"{run}: not {samples} rollout members on train points")
        starts = {json.dumps(s.get("start"), sort_keys=True) for s in specs}
        start = specs[0].get("start") or {}
        if len(starts) > 1 or start.get("take") is not None:
            raise ValueError(f"{run}: members must restore one point, without `take`")
        if (start.get("run"), start.get("system")) != (run_name(source), system):
            raise ValueError(f"{run}: restores {start.get('run')}/{start.get('system')}")
        if any(
            (s.get("policy") or {}).get("sample") is not True or s.get("write", True) for s in specs
        ):
            raise ValueError(f"{run}: rollout members sample and stop before writing")
        for name in names:
            provenance(run, name)
        for key, task in run_tasks(manifest, BENCH).items():
            if (key, start["batches"]) not in want:
                raise ValueError(f"{run}: {key} after {start['batches']} batches is not planned")
            found[key, start["batches"]] = (run, task)
    steps: list[Step] = []
    records: list[dict[str, Any]] = []
    for key, batches in sorted(want):
        record = {"group": f"{key}/b{batches}", "task": key, "batches": batches}
        if (key, batches) not in found:
            records.append(record | {"skipped": "no rollout run holds it"})
            continue
        run, task = found[key, batches]
        start = {"run": run_name(source), "system": system, "batches": batches}
        point = point_key(source, start, key)
        trajectories = [trajectory(run, name, task, key, b0, point) for name in names]
        finished = [(n, t) for n, t in zip(names, trajectories, strict=True) if t is not None]
        adapters = {s.adapter for _, t in finished for s in t["steps"]}
        if len(adapters) > 1:
            raise ValueError(f"{record['group']}: members sampled from several adapters {adapters}")
        returns = [t["return"] for _, t in finished]
        record |= {"returns": returns, "kinds": [t["kinds"] for _, t in finished]}
        skipped = None
        if len(finished) < len(names):
            skipped = "a member failed or has no gold"
        elif max(returns) - min(returns) < TIE:
            skipped = "tied"
        if skipped:
            advantages = [0.0] * len(finished)
        else:
            mean, sd = statistics.fmean(returns), statistics.pstdev(returns)
            advantages = [(r - mean) / max(sd, SD_FLOOR) for r in returns]
        for (name, t), a in zip(finished, advantages, strict=True):
            steps += [
                Step(**asdict(s) | {"advantage": a, "group": record["group"], "trajectory": name}
                     | {"usable": skipped is None})
                for s in t["steps"]
            ]  # fmt: skip
        records.append(record | {"advantages": None if skipped else advantages, "skipped": skipped})
    return steps, records


def trajectory(
    run: Path, system: str, task: Any, key: str, b0: float, point: str
) -> dict[str, Any] | None:
    """A member's episode: its retrieval return, the kinds it executed and its sampled decisions
    (forced, overridden and taken ones left out); None if it did not finish or has no gold. Its
    first decision must be the restored source point (`point`: its snapshot key)."""
    folder = task_dir(run, BENCH, system, key)
    if read_json(folder / "result.json", {}).get("status") != "ok":
        return None
    if restored(folder) != point:
        raise ValueError(f"{folder}: restored another snapshot than its source point")
    state = State.from_json((folder / "state.json").read_text())
    recall = writer_recall(task, state)
    if recall is None:
        return None
    kinds, sampled = [], []
    for record in read_jsonl(folder / "decisions.jsonl"):
        kinds.append(record["candidates"][record["executed"]]["kind"])
        if record["forced"] or record["overridden"] or record.get("taken", False):
            continue
        (request,) = record["input"]["requests"]
        (question,) = request["questions"]
        if not len(record["probs"]) == len(question["criteria"]) == len(record["candidates"]):
            raise ValueError(f"{folder}: probabilities, options and candidates disagree")
        sampled.append(
            Step(
                request["state"],
                question,
                record["input"]["render"],
                record["requested"],
                record["probs"],
                0.0,
                record["input"]["adapter"],
                "",
                "",
            )
        )
    ret = rewards.retrieval_return(recall, state.spent_usd, b0)
    return {"return": ret, "kinds": kinds, "steps": sampled}


def write_steps(path: Path, steps: Sequence[Step]) -> None:
    write_jsonl(path, map(asdict, steps))


def read_steps(path: Path) -> list[Step]:
    return [Step(**s) for s in read_jsonl(path)]


# --------------------------------------------------------------------------- the val gate
ARMS = ("ctl_d1_rl", "ctl_d1_sft")  # tried in this order; the fallback is BASELINE
BASELINE = "ctl_d1"


def gate(run: Path, regret: dict[str, float]) -> dict[str, Any]:
    """The registered val gate (docs/06) on a val run of BASELINE and ARMS, each with a second
    writer sample (`<arm>_s1`): per arm, its checks against BASELINE, and the main claim's d1 (the
    first arm of ARMS that passes, else BASELINE). `regret`: each system's deployed regret on the
    val points (`jev controller eval ... --points`)."""
    manifest = read_manifest(run)
    cell = manifest["experiment"]["benchmarks"][BENCH]
    if cell["split"] != "val" or cell.get("limit") is not None or cell.get("ids") is not None:
        raise ValueError(f"{run} is not a run of the whole val split")
    tasks = run_tasks(manifest, BENCH)

    def within(result: dict[str, Any]) -> bool:
        usage = result.get("usage", {})
        return result.get("status") == "ok" and usage["spent_usd"] <= usage["task_cap"]

    def episodes(system: str) -> dict[str, Any]:
        ok = all(
            within(read_json(task_dir(run, BENCH, s, k) / "result.json", {}))
            for s in (system, f"{system}_s1") for k in tasks
        )  # fmt: skip
        if not ok:  # a failed episode may have no final state
            return {"ok": False, "recall": [0.0] * len(tasks), "batches": [0] * len(tasks)}
        recall, batches = [], []
        for key, task in tasks.items():
            state = State.from_json((task_dir(run, BENCH, system, key) / "state.json").read_text())
            recall.append(writer_recall(task, state) or 0.0)
            batches.append(state.batches)
        return {"ok": True, "recall": recall, "batches": batches}

    base = episodes(BASELINE)
    out: dict[str, Any] = {"baseline": BASELINE, "arms": {}}
    for arm in ARMS:
        e = episodes(arm)
        losses = sum(b - a > 0.10 for a, b in zip(e["recall"], base["recall"], strict=True))
        checks = {
            "recall": statistics.fmean(e["recall"]) >= statistics.fmean(base["recall"]) - 1e-9,
            "recall_losses": losses <= 2,
            "batches": statistics.fmean(e["batches"]) >= 3 and min(e["batches"]) >= 2,
            "regret": regret[arm] <= regret[BASELINE],
            "all_ok": e["ok"],
        }
        out["arms"][arm] = {
            "passes": all(checks.values()),
            "checks": checks,
            "mean_recall": statistics.fmean(e["recall"]),
            "mean_batches": statistics.fmean(e["batches"]),
            "regret": regret[arm],
        }
    out["baseline_mean_recall"] = statistics.fmean(base["recall"])
    out["baseline_regret"] = regret[BASELINE]
    out["claim"] = next((a for a in ARMS if out["arms"][a]["passes"]), BASELINE)
    return out

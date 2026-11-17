"""Counterfactual branches of logged decision points, and the SFT targets they give (docs/06).

A branch restores a decision point of a controlled episode run (`start`), executes one of the
candidates the rules offered there (`take`) instead of asking the policy, and continues with a
frozen continuation policy (by default untrained d1's, `ctl_d1`) until `finish` and the choice of
the writer's documents; nothing is written (`write: false`). Every legal candidate of a point is
branched under the same continuation, so the branches of one point differ only in the decision
there, and a point's returns estimate the continuation's action values there.

- `forks` lists the points worth branching in a source run: decisions the rules left a choice
  (two or more legal candidates) in finished episodes of tasks with important citations,
  optionally one per batch stratum (`STRATA`: early and late) by a fixed pseudo-random pick. Only
  `train` and `val` runs are sources: dev and test trajectories never become labels.
- `experiments` turns them into one small experiment per (batches, kind): its tasks are those
  whose point after that many batches offers that kind, so no branch is refused. Each keeps the
  source system's settings (cap, rules, costs, actions, sample) and the source experiment's
  executor, search and seed, so a restored point offers exactly what the source saw.
- `outcomes` scores every planned branch judge-free with the retrieval return (rewards.py):
  important citations among the writer's documents, less the task's whole list spend (the
  source's spend up to the point included, the same for every branch of a point). A failed or
  missing branch has no return. Each outcome carries its run's context (executor routing,
  search, seed, cap and resolved system, the taken kind left out), which every branch of a point
  must share; a branch that restored another snapshot than its source point, or a source or
  branch run with a reply of an unpinned model version, is an error.
- `examples` gives each point whose every legal candidate has a return a graded soft target,
  q(a) proportional to exp(-(best return - return of a) / TAU), and counts the points left out.
  The example is what d1 reads at that point under its default profile; each point's record
  also says whether d1's decoder (`Profile.pick`) applied to the target picks the best option.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from jevscholar.controller.state import MAX_BATCHES, SNAPSHOT, State
from jevscholar.search.policy import arxiv_id, norm_title
from jevscholar.types import Task
from jevscholar.util import read_json, read_jsonl

BENCH = "deepscholar"
SPLITS = ("train", "val")  # the splits whose decisions may become labels (val: held out)
SYSTEM = "branch"  # the system name of every branch experiment
CONTINUATION = "ctl_d1"  # the catalog system whose policy continues every branch
STRATA = ((0, 2), (3, MAX_BATCHES - 1))  # early and late points: batches before, inclusive
TAU = 0.005  # target smoothing, in return units: about a third of one gold paper of ten
RESTORED = ("offset_usd", "budget_usd", "max_batches", "prompts")  # what a branch's cell sets


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
    source: Path, system: str, *, tasks: int | None = None, stratified: bool = False
) -> list[Fork]:
    """The decision points of `system`'s finished episodes in `source` that the rules left a
    choice, on tasks with important citations: the first `tasks` tasks of the run; `stratified`:
    at most one point per batch stratum of `STRATA` each (a fixed pseudo-random pick), in
    episode order."""
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
        if stratified:
            points = [
                min(within, key=lambda f: digest(f"{key}|{f.batches}"))
                for low, high in STRATA
                if (within := [f for f in points if low <= f.batches <= high])
            ]
        out += points
    return out


def provenance(source: Path, system: str, versions: dict[str, str] | None = None) -> None:
    """Fail unless every reply `system` got in run `source` is of its model's pinned version
    (models.yaml `versions`): a branch inherits the evidence its point was built from, and its
    return is only comparable with its point's other branches on one version."""
    from collections import Counter

    from jevscholar.config import load_models

    versions = load_models().versions if versions is None else versions
    other: Counter[tuple[str, str | None]] = Counter()
    ledger = source / "ledger.jsonl"
    for e in read_jsonl(ledger) if ledger.exists() else ():
        if e["kind"] != "llm" or e["system"] != system or e["http_status"] != 200:
            continue
        if (want := versions.get(e["model"])) and e.get("fingerprint") != want:
            other[e["model"], e.get("fingerprint")] += 1
    if other:
        raise ValueError(f"{source}/{system}: replies of unpinned model versions {dict(other)}")


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
) -> dict[str, Experiment]:
    """Run id -> the branch experiment of one (batches, kind): the forks whose point offers that
    kind, each restored and executing it, then continued with `policy`."""
    start_run, found = run_name(source), list(found)
    split, spec = source_cell(manifest := read_manifest(source), system)
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for fork in found:
        for kind in fork.kinds:
            groups[fork.batches, kind].append(fork.task)
    names = run_ids(source, found)
    base = manifest["experiment"]
    out = {}
    for (batches, kind), ids in sorted(groups.items()):
        branch = spec | {
            "policy": policy,
            "start": {"run": start_run, "system": system, "batches": batches, "take": kind},
            "write": False,
        }
        raw = base | {
            "name": f"branches/{start_run}/b{batches}-{kind}",
            "replay_only": False,
            "budget": base["budget"] | {"run_usd": run_usd},
            "systems": {SYSTEM: branch},
            "benchmarks": {BENCH: {"split": split, "ids": ids, "systems": [SYSTEM]}},
        }
        out[names[batches, kind]] = Experiment.model_validate(raw)
    return out


def run_ids(source: Path, found: Iterable[Fork]) -> dict:
    """(batches, kind) -> the id of the branch run of that kind at points after that many
    batches: `branches-<source run>-b<batches>-<kind>`."""
    pairs = {(f.batches, kind) for f in found for kind in f.kinds}
    return {(b, k): f"branches-{source.name}-b{b}-{k}" for b, k in sorted(pairs)}


# --------------------------------------------------------------------------- labels
def outcomes(runs: Iterable[Path]) -> list[dict[str, Any]]:
    """Every planned branch of the branch runs (each task the run lists): {"split", "source",
    "system", "task", "batches", "take", "context", "status", "return", "recall", "cost_usd"},
    with `return` None for a branch that failed or never ran. Tasks without important citations
    give no outcome."""
    b0 = benchmarks.get(BENCH).cap_usd
    out, checked = [], set()
    for run in runs:
        manifest = read_manifest(run)
        exp = manifest["experiment"]
        cell = exp["benchmarks"][BENCH]
        spec = exp["systems"].get(SYSTEM, {})
        start = spec.get("start") or {}
        if cell["systems"] != [SYSTEM] or start.get("take") is None or spec.get("write", True):
            raise ValueError(f"{run} is not a branch run")
        if cell["split"] not in SPLITS:
            raise ValueError(f"{run}: labels come from {' or '.join(SPLITS)} only")
        for checked_run, system in ((run, SYSTEM), (paths.RUNS / start["run"], start["system"])):
            if (checked_run, system) not in checked:
                provenance(checked_run, system)
                checked.add((checked_run, system))
        shared = context(manifest)
        for key, task in run_tasks(manifest, BENCH).items():
            folder = task_dir(run, BENCH, SYSTEM, key)
            status = read_json(folder / "result.json", {}).get("status", "missing")
            found = {
                "split": cell["split"],
                "source": start["run"],
                "system": start["system"],
                "task": key,
                "batches": start["batches"],
                "take": start["take"],
                "context": shared,
                "status": status,
            }
            if status != "ok":
                out.append(found | {"return": None, "recall": None, "cost_usd": None})
                continue
            if restored(folder) != point_key(paths.RUNS / start["run"], start, key):
                raise ValueError(f"{folder}: restored another snapshot than its source point")
            state = State.from_json((folder / "state.json").read_text())
            recall = writer_recall(task, state)
            if recall is None:  # no important citations: no label (forks leaves such tasks out)
                continue
            found |= {
                "return": rewards.retrieval_return(recall, state.spent_usd, b0),
                "recall": recall,
                "cost_usd": state.spent_usd,
            }
            out.append(found)
    return out


def context(manifest: dict[str, Any]) -> str:
    """What a branch run shares with every other branch run of its points: the executor, search
    and seed, and its cell's routing, cap and resolved system, the start's `take` left out."""
    exp, cell = manifest["experiment"], manifest["cells"][BENCH][SYSTEM]
    resolved = json.loads(json.dumps(cell["resolved"]))
    resolved["params"]["start"].pop("take", None)
    shared = {k: exp[k] for k in ("executor", "search", "seed")}
    shared |= {k: cell[k] for k in ("version", "routing", "cap_usd")} | {"resolved": resolved}
    return digest(json.dumps(shared, sort_keys=True)).hex()[:16]


def snapshot_key(record: dict[str, Any]) -> str:
    """A decision record's snapshot (state and candidates), less what a branch's cell sets."""
    state = {k: v for k, v in record["state"].items() if k not in RESTORED}
    return digest(json.dumps([state, record["candidates"]], sort_keys=True)).hex()


def restored(folder: Path) -> str:
    """The snapshot a branch task restored: its first decision record's."""
    return snapshot_key(next(iter(read_jsonl(folder / "decisions.jsonl"))))


def point_key(source: Path, start: dict[str, Any], key: str) -> str:
    """The snapshot of the source point a branch names (the first after `batches`)."""
    folder = source / BENCH / start["system"] / "tasks" / key
    for record in read_jsonl(folder / "decisions.jsonl"):
        if record["state"]["batches"] == start["batches"]:
            return snapshot_key(record)
    raise ValueError(f"{folder}: no decision point after {start['batches']} batches")


def writer_recall(task: Task, state: State) -> float | None:
    """R_e: the share of the task's important citations among the writer's documents (None for a
    task without important citations)."""
    rows = {p.url: p.row for p in state.papers}
    docs = [rows[url] for url in state.docs]
    ids, titles = {arxiv_id(r["url"]) for r in docs}, {norm_title(r["title"]) for r in docs}
    found, total = gold_found(task, ids, titles)
    return found / total if total else None


def soft_target(returns: dict[str, float], tau: float = TAU) -> dict[str, float]:
    """Each option's share: proportional to exp(-(best return - its return) / tau)."""
    best = max(returns.values())
    weights = {name: math.exp(-(best - r) / tau) for name, r in returns.items()}
    total = sum(weights.values())
    return {name: w / total for name, w in weights.items()}


def examples(
    found: Iterable[dict[str, Any]],
    profile: Profile | None = None,
    planned: Iterable[tuple[str, str, str, int]] | None = None,
) -> tuple[list[Example], list[dict[str, Any]], list[tuple[str, str, str, int]]]:
    """(examples, points, left out): one example per decision point whose every legal candidate
    has a return, as d1 reads that point under `profile`; each point's record with its options'
    returns, the target and whether `profile.pick` on the target picks a best option; and the
    points left out for a failed or missing branch. With `planned` (source, system, task key,
    batches) points, every planned point is labelled or left out, and an outcome of an unplanned
    point is an error. All outcomes must come from one split, a point's from one context, one per
    candidate."""
    from jevscholar.systems.controlled import Start, locate

    found, profile = list(found), profile or Profile()
    if len({o["split"] for o in found}) > 1:
        raise ValueError("outcomes of several splits: label each split on its own")
    by_point: dict[tuple[str, str, str, int], dict[str, float | None]] = defaultdict(dict)
    contexts: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for o in found:
        point = o["source"], o["system"], o["task"], o["batches"]
        if o["take"] in by_point[point]:
            raise ValueError(f"two branches take {o['take']!r} at {point}")
        by_point[point][o["take"]] = o["return"]
        contexts[point].add(o["context"])
    wanted = sorted(by_point) if planned is None else sorted(set(planned))
    if unplanned := set(by_point) - set(wanted):
        raise ValueError(f"outcomes of {len(unplanned)} unplanned points, e.g. {min(unplanned)}")
    out, points, left_out = [], [], []
    tasks: dict[str, dict[str, Task]] = {}
    for source, system, key, batches in wanted:
        returns = by_point.get((source, system, key, batches), {})
        if len(contexts[source, system, key, batches]) > 1:
            raise ValueError(f"{(source, system, key, batches)}: branches of different contexts")
        if source not in tasks:
            tasks[source] = run_tasks(read_manifest(paths.RUNS / source), BENCH)
        task = tasks[source][key]
        point = locate(task, Start(run=source, system=system, batches=batches))
        kinds = [c.kind for c in point.candidates]
        if set(returns) != set(kinds) or any(r is None for r in returns.values()):
            left_out.append((source, system, key, batches))
            continue
        request = profile.request(point.state, point.candidates)
        (question,) = request["questions"]
        names = dict(zip(kinds, question["criteria"], strict=True))
        target = soft_target({names[k]: r for k, r in returns.items()})
        values = [returns[k] for k in kinds]
        picked = profile.pick(point.candidates, [target[names[k]] for k in kinds])
        out.append(Example(request["state"], question, target, render=profile.render))
        points.append(
            {"source": source, "system": system, "task": key, "batches": batches}
            | {"kinds": kinds, "returns": values, "target": target}
            | {"decoded_best": values[picked] == max(values)}
        )
    return out, points, left_out

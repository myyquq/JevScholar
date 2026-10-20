"""Counterfactual branches: which decision points are branched, the branch experiments, their
judge-free outcomes and the soft targets (offline: runs written by hand, gold faked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevscholar import paths
from jevscholar.config import Experiment
from jevscholar.controller import branches
from jevscholar.controller.branches import Fork, soft_target
from jevscholar.controller.state import Candidate, Paper, State, snapshot
from jevscholar.types import Task

GOLD = {"gold a", "gold b"}


def row(title: str) -> dict:
    return {"url": f"http://x.org/{title.replace(' ', '_')}", "title": title, "snippet": "s"}


def write_lines(path: Path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def experiment(split: str, systems: dict, listed: list[str], **extra) -> dict:
    exp = Experiment(
        name="controller/rollouts",
        systems=systems,
        benchmarks={"deepscholar": {"split": split, "systems": listed}},
        **extra,
    )
    return exp.model_dump(mode="json")


@pytest.fixture
def source(tmp_path, monkeypatch):
    """A fixed-rule episode run under runs/src: per task, a choice between the abstract and a
    keyword batch, then among abstract, deepen and finish, then the forced finish."""
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    gold_a, gold_b, other, noise = row("gold a"), row("gold b"), row("other"), row("noise")
    s0 = State("T", 0.27, "abs")
    s1 = State("T", 0.27, "abs", spent_usd=0.01, batches=1)
    s1.papers = [Paper(gold_a, True), Paper(other, False)]
    s1.queries = []
    s2 = State("T", 0.27, "abs", spent_usd=0.02, batches=2)
    s2.papers = [*s1.papers, Paper(gold_b, True)]
    steps = [
        snapshot(
            s0,
            [Candidate("abstract", "abs", (gold_a, other)), Candidate("explore", "k", (noise,))],
            0,
        ),
        snapshot(s1, [Candidate("abstract", "abs", (gold_b,)), Candidate("finish")], 0),
        snapshot(s2, [Candidate("finish")], 0),
    ]
    run = tmp_path / "runs" / "src"
    systems = {
        "ctl_fixed": {
            "use": "controlled",
            "policy": {"kind": "fixed", "batches": 3},
            "task_usd": 0.5,
        }
    }
    tasks = {}
    for key in ("t1", "t2", "nogold", "failed"):
        folder = run / "deepscholar" / "ctl_fixed" / "tasks" / key
        folder.mkdir(parents=True)
        write_lines(folder / "rows.jsonl", [gold_a, gold_b, other, noise])
        write_lines(folder / "pages.jsonl", [])
        lines = [
            s | {"requested": 0, "executed": 0, "forced": len(s["candidates"]) == 1}
            | {"overridden": False, "taken": False, "control_usd": 0.0, "probs": None}
            for s in steps
        ]  # fmt: skip
        write_lines(folder / "decisions.jsonl", lines)
        status = "failed" if key == "failed" else "ok"
        (folder / "result.json").write_text(json.dumps({"status": status}))
        tasks[key] = Task("deepscholar", key, "T")
    manifest = {"experiment": experiment("train", systems, ["ctl_fixed"], seed=0)}
    (run / "manifest.json").write_text(json.dumps(manifest))

    def run_tasks(manifest, bench):
        ids = manifest["experiment"]["benchmarks"][bench].get("ids")
        return {k: t for k, t in tasks.items() if ids is None or k in ids}

    def gold_found(task, ids, titles):
        return (0, 0) if task.id == "nogold" else (len(titles & GOLD), 2)

    monkeypatch.setattr(branches, "run_tasks", run_tasks)
    monkeypatch.setattr(branches, "gold_found", gold_found)
    return run


def test_forks_are_the_choices_of_finished_gold_tasks(source):
    found = branches.forks(source, "ctl_fixed")
    assert found == [
        Fork("t1", "t1", 0, ("abstract", "explore")),
        Fork("t1", "t1", 1, ("abstract", "finish")),
        Fork("t2", "t2", 0, ("abstract", "explore")),
        Fork("t2", "t2", 1, ("abstract", "finish")),
    ]  # the forced finish, the task without gold and the failed episode are left out
    assert branches.forks(source, "ctl_fixed", tasks=1) == found[:2]
    one = branches.forks(source, "ctl_fixed", per_task=1)
    assert len(one) == 2 and {f.task for f in one} == {"t1", "t2"}
    assert one == branches.forks(source, "ctl_fixed", per_task=1)  # a fixed pick


def test_only_train_and_val_episode_runs_are_sources(source):
    manifest = source / "manifest.json"
    data = json.loads(manifest.read_text())
    data["experiment"]["benchmarks"]["deepscholar"]["split"] = "dev"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="train or val runs only, not dev"):
        branches.forks(source, "ctl_fixed")
    data["experiment"]["benchmarks"]["deepscholar"]["split"] = "train"
    data["experiment"]["systems"]["ctl_fixed"]["start"] = {"run": "x", "system": "y"}
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="episode run"):
        branches.forks(source, "ctl_fixed")
    with pytest.raises(ValueError, match="no deepscholar/ctl_d1 cell"):
        branches.forks(source, "ctl_d1")
    del data["experiment"]["systems"]["ctl_fixed"]["start"]
    manifest.write_text(json.dumps(data))
    log = source / "deepscholar" / "ctl_fixed" / "tasks" / "t1" / "decisions.jsonl"
    old = [json.loads(line) for line in log.read_text().splitlines()]
    write_lines(log, [{k: v for k, v in r.items() if k != "version"} for r in old])
    with pytest.raises(ValueError, match="decisions in format None"):  # an older run's logs
        branches.forks(source, "ctl_fixed")


def test_one_experiment_per_batches_and_kind(source):
    found = branches.forks(source, "ctl_fixed")
    policy = branches.continuation_policy()
    assert policy == {"kind": "fixed", "retrieval": ["abstract", "explore"], "batches": 3}
    exps = branches.experiments(source, "ctl_fixed", found, policy=policy, run_usd=0.1)
    assert list(exps) == [
        "branches-src-b0-abstract",
        "branches-src-b0-explore",
        "branches-src-b1-abstract",
        "branches-src-b1-finish",
    ]
    exp = exps["branches-src-b1-finish"]
    cell = exp.benchmarks["deepscholar"]
    assert (cell.split, cell.ids, cell.limit, cell.systems) == (
        "train",
        ("t1", "t2"),
        None,
        ("branch",),
    )
    spec = exp.systems["branch"]
    assert spec["start"] == {"run": "src", "system": "ctl_fixed", "batches": 1, "take": "finish"}
    assert spec["policy"] == policy and spec["task_usd"] == 0.5  # the source's own cap kept
    assert (exp.budget.run_usd, exp.seed, exp.replay_only) == (0.1, 0, False)
    with pytest.raises(ValueError, match="runs under"):
        branches.experiments(Path("/elsewhere/src"), "ctl_fixed", found, policy=policy, run_usd=1)


def branch_run(name: str, batches: int, take: str, docs: dict[str, list[str]], spent=0.05) -> Path:
    """A finished branch run: per task, the writer's documents by title."""
    run = paths.RUNS / name
    systems = {"branch": {"use": "controlled", "policy": {"kind": "fixed"}}}
    systems["branch"]["start"] = {
        "run": "src",
        "system": "ctl_fixed",
        "batches": batches,
        "take": take,
    }
    for key, titles in docs.items():
        folder = run / "deepscholar" / "branch" / "tasks" / key
        folder.mkdir(parents=True)
        state = State("T", 0.27, spent_usd=spent, papers=[Paper(row(t), True) for t in titles])
        state.docs = [row(t)["url"] for t in titles]
        (folder / "state.json").write_text(state.to_json())
        (folder / "result.json").write_text(json.dumps({"status": "ok"}))
    exp = experiment("train", systems, ["branch"])
    exp["benchmarks"]["deepscholar"]["ids"] = list(docs)
    (run / "manifest.json").write_text(json.dumps({"experiment": exp}))
    return run


def test_outcomes_and_soft_targets(source):
    runs = [
        branch_run("b0-abstract", 0, "abstract", {"t1": ["gold a", "gold b"], "t2": ["gold a"]}),
        branch_run("b0-explore", 0, "explore", {"t1": ["gold a"], "t2": ["gold a"]}),
        branch_run("b1-abstract", 1, "abstract", {"t1": ["gold a", "gold b"]}, spent=0.06),
        branch_run("b1-finish", 1, "finish", {"t1": ["gold a"], "t2": ["gold a"]}),
    ]
    found = branches.outcomes(runs)
    first = found[0]
    assert (first["task"], first["batches"], first["take"], first["recall"]) == (
        "t1",
        0,
        "abstract",
        1.0,
    )
    assert first["return"] == pytest.approx(0.15 - 0.1 * 0.05 / 0.27)

    examples, points, incomplete = branches.examples(found)
    assert incomplete == 1  # t2 after one batch: its abstract branch is missing
    assert [(p["task"], p["batches"]) for p in points] == [("t1", 0), ("t1", 1), ("t2", 0)]
    assert examples[0].target == {"1. abstract": 1.0}  # more gold at the same cost
    assert examples[1].target == {"1. abstract": 1.0}  # gold b is worth its extra cost
    assert examples[2].target == {"1. abstract": 0.5, "2. explore": 0.5}  # an exact tie
    assert examples[0].render == {"style": "sections"}
    assert examples[0].state["search batches"] == "0 of 6"
    assert list(examples[1].question["criteria"]) == ["1. abstract", "2. finish"]
    assert points[2]["returns"][0] == points[2]["returns"][1]

    with pytest.raises(ValueError, match="several splits"):
        branches.examples([found[0], found[1] | {"split": "val"}])
    dev = paths.RUNS / "b0-abstract" / "manifest.json"
    dev.write_text(dev.read_text().replace('"train"', '"dev"'))
    with pytest.raises(ValueError, match="train or val only"):
        branches.outcomes(runs[:1])


def test_soft_targets_share_only_exact_ties():
    assert soft_target({"a": 0.1, "b": 0.1, "c": 0.0}) == {"a": 0.5, "b": 0.5}
    assert soft_target({"a": 0.1, "b": 0.1 - 1e-6}) == {"a": 1.0}

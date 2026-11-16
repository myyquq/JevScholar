"""Counterfactual branches: which decision points are branched, the branch experiments, their
judge-free outcomes and the soft targets (offline: runs written by hand, gold faked)."""

from __future__ import annotations

import json
import math
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
    one = branches.forks(source, "ctl_fixed", stratified=True)  # both points are early ones
    assert len(one) == 2 and {f.task for f in one} == {"t1", "t2"}
    assert one == branches.forks(source, "ctl_fixed", stratified=True)  # a fixed pick


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
    assert branches.continuation_policy()["kind"] == "d1"  # untrained d1 continues by default
    policy = branches.continuation_policy("ctl_fixed")
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
    assert spec["write"] is False  # branches stop at the writer's documents
    assert (exp.budget.run_usd, exp.seed, exp.replay_only) == (0.1, 0, False)
    with pytest.raises(ValueError, match="runs under"):
        branches.experiments(Path("/elsewhere/src"), "ctl_fixed", found, policy=policy, run_usd=1)


def branch_run(
    name: str,
    batches: int,
    take: str,
    docs: dict[str, list[str] | None],
    spent=0.05,
    policy: dict | None = None,
) -> Path:
    """A branch run: per task, the writer's documents by title (None: the branch failed), and
    the restored snapshot as its first decision record (the source point's, with this cell's
    cap)."""
    run = paths.RUNS / name
    systems = {"branch": {"use": "controlled", "policy": policy or {"kind": "fixed"}}}
    systems["branch"]["start"] = {
        "run": "src",
        "system": "ctl_fixed",
        "batches": batches,
        "take": take,
    }
    systems["branch"]["write"] = False
    for key, titles in docs.items():
        folder = run / "deepscholar" / "branch" / "tasks" / key
        folder.mkdir(parents=True)
        log = paths.RUNS / "src" / "deepscholar" / "ctl_fixed" / "tasks" / key / "decisions.jsonl"
        records = [json.loads(line) for line in log.read_text().splitlines()]
        point = next(r for r in records if r["state"]["batches"] == batches)
        point["state"] |= {"offset_usd": point["state"]["spent_usd"], "budget_usd": 0.5}
        write_lines(folder / "decisions.jsonl", [point])
        if titles is None:
            (folder / "result.json").write_text(json.dumps({"status": "failed"}))
            continue
        state = State("T", 0.27, spent_usd=spent, papers=[Paper(row(t), True) for t in titles])
        state.docs = [row(t)["url"] for t in titles]
        (folder / "state.json").write_text(state.to_json())
        (folder / "result.json").write_text(json.dumps({"status": "ok"}))
    exp = experiment("train", systems, ["branch"])
    exp["benchmarks"]["deepscholar"]["ids"] = list(docs)
    resolved = {"version": "2", "params": systems["branch"], "policy": systems["branch"]["policy"]}
    cell = {"version": "1", "routing": {"url": "http://up"}, "cap_usd": 0.27, "resolved": resolved}
    cells = {"deepscholar": {"branch": cell}}
    (run / "manifest.json").write_text(json.dumps({"experiment": exp, "cells": cells}))
    return run


def test_outcomes_and_soft_targets(source):
    runs = [
        branch_run("b0-abstract", 0, "abstract", {"t1": ["gold a", "gold b"], "t2": ["gold a"]}),
        branch_run("b0-explore", 0, "explore", {"t1": ["gold a"], "t2": ["gold a"]}),
        branch_run("b1-abstract", 1, "abstract", {"t1": ["gold a", "gold b"], "t2": None}),
        branch_run("b1-finish", 1, "finish", {"t1": ["gold a"], "t2": ["gold a"]}, spent=0.04),
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
    failed = next(o for o in found if (o["task"], o["batches"], o["take"]) == ("t2", 1, "abstract"))
    assert (failed["status"], failed["return"]) == ("failed", None)  # planned, counted, no return

    examples, points, left_out = branches.examples(found)
    assert left_out == [("src", "ctl_fixed", "t2", 1)]  # its abstract branch failed
    assert [(p["task"], p["batches"]) for p in points] == [("t1", 0), ("t1", 1), ("t2", 0)]
    gap = 0.075  # gold b: half the recall, times 0.15
    share = 1 / (1 + math.exp(-gap / branches.TAU))
    assert examples[0].target == pytest.approx({"1. abstract": share, "2. explore": 1 - share})
    cheaper = 0.15 * 0.5 - 0.1 * 0.04 / 0.27  # finish: gold a only, at less cost
    gap = 0.15 - 0.1 * 0.05 / 0.27 - cheaper
    share = 1 / (1 + math.exp(-gap / branches.TAU))
    assert examples[1].target == pytest.approx({"1. abstract": share, "2. finish": 1 - share})
    assert examples[2].target == {"1. abstract": 0.5, "2. explore": 0.5}  # an exact tie
    assert all(p["decoded_best"] for p in points)  # finish is never the best option here
    assert examples[0].render == {"style": "sections"}
    assert examples[0].state["search batches"] == "0 of 6"
    assert list(examples[1].question["criteria"]) == ["1. abstract", "2. finish"]
    assert points[2]["returns"][0] == points[2]["returns"][1]

    with pytest.raises(ValueError, match="several splits"):
        branches.examples([found[0], found[1] | {"split": "val"}])
    with pytest.raises(ValueError, match="two branches take 'abstract'"):
        branches.examples([found[0], found[0]])
    assert found[0]["context"] == found[2]["context"]  # t1 at b0: the taken kind is not in it
    with pytest.raises(ValueError, match="different contexts"):
        branches.examples([found[0], found[2] | {"context": "other"}])  # both t1, b0
    dev = paths.RUNS / "b0-abstract" / "manifest.json"
    dev.write_text(dev.read_text().replace('"train"', '"dev"'))
    with pytest.raises(ValueError, match="train or val only"):
        branches.outcomes(runs[:1])


def test_a_branch_that_restored_another_snapshot_is_an_error(source):
    run = branch_run("b0-abstract", 0, "abstract", {"t1": ["gold a"]})
    log = run / "deepscholar" / "branch" / "tasks" / "t1" / "decisions.jsonl"
    (record,) = [json.loads(line) for line in log.read_text().splitlines()]
    record["state"]["batches"] = 1  # not the point the branch names
    write_lines(log, [record])
    with pytest.raises(ValueError, match="restored another snapshot"):
        branches.outcomes([run])


def test_a_writing_run_is_not_a_branch_run(source):
    run = branch_run("b0-abstract", 0, "abstract", {"t1": ["gold a"]})
    manifest = run / "manifest.json"
    data = json.loads(manifest.read_text())
    del data["experiment"]["systems"]["branch"]["write"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not a branch run"):
        branches.outcomes([run])


def test_soft_targets_fall_with_the_return_gap():
    assert soft_target({"a": 0.1, "b": 0.1, "c": 0.1}) == pytest.approx(dict.fromkeys("abc", 1 / 3))
    q = soft_target({"a": 0.1, "b": 0.1 - branches.TAU})
    assert q["a"] / q["b"] == pytest.approx(math.e)
    assert soft_target({"a": 0.2, "b": 0.0})["b"] < 1e-15


def test_a_source_must_come_from_the_pinned_model_version(source):
    event = {"kind": "llm", "system": "ctl_fixed", "model": "flash", "http_status": 200}
    lines = [event | {"fingerprint": "v1"}, event | {"fingerprint": "v1", "system": "other"}]
    write_lines(source / "ledger.jsonl", lines)
    branches.provenance(source, "ctl_fixed", {"flash": "v1"})
    write_lines(source / "ledger.jsonl", [*lines, event | {"fingerprint": "v0"}])
    with pytest.raises(ValueError, match="unpinned model versions"):
        branches.provenance(source, "ctl_fixed", {"flash": "v1"})
    branches.provenance(source, "ctl_fixed", {})  # nothing pinned


def test_branches_of_another_execution_context_do_not_share_a_target(source):
    a = branch_run("b0-abstract", 0, "abstract", {"t1": ["gold a"]})
    b = branch_run("b0-explore", 0, "explore", {"t1": ["gold a"]})
    manifest = b / "manifest.json"
    data = json.loads(manifest.read_text())
    data["cells"]["deepscholar"]["branch"]["resolved"]["params"]["max_batches"] = 2  # a horizon
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different contexts"):
        branches.examples(branches.outcomes([a, b]))
    event = {"kind": "llm", "system": "branch", "model": "deepseek-flash", "http_status": 200}
    write_lines(a / "ledger.jsonl", [event | {"fingerprint": "an earlier version"}])
    with pytest.raises(ValueError, match="unpinned model versions"):
        branches.outcomes([a])


def test_every_planned_point_is_labelled_or_left_out(source):
    runs = [
        branch_run("b0-abstract", 0, "abstract", {"t1": ["gold a"]}),
        branch_run("b0-explore", 0, "explore", {"t1": ["gold a"]}),
    ]
    found = branches.outcomes(runs)
    planned = [("src", "ctl_fixed", "t1", 0), ("src", "ctl_fixed", "t1", 1)]
    examples, _, left_out = branches.examples(found, planned=planned)
    assert len(examples) == 1 and left_out == [("src", "ctl_fixed", "t1", 1)]  # no run at all
    with pytest.raises(ValueError, match="unplanned points"):
        branches.examples(found, planned=planned[1:])
    ids = branches.run_ids(source, branches.forks(source, "ctl_fixed"))
    assert ids[1, "finish"] == "branches-src-b1-finish" and len(ids) == 4

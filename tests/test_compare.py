"""Comparing d1 adapters on val: decision regret on branched points, the episode runs and the
paired comparison of their retrieval returns (offline: runs written by hand, gold faked)."""

from __future__ import annotations

import json
import math

import pytest

from jevscholar import paths
from jevscholar.controller import compare as cmp
from jevscholar.controller.d1 import Example
from jevscholar.controller.state import Paper, State
from jevscholar.types import Task
from test_branches import experiment, row, source  # noqa: F401 (source: a fixture)

POINTS = [
    {"task": "t1", "batches": 0, "kinds": ["abstract", "explore"], "returns": [0.02, 0.01]},
    {"task": "t1", "batches": 1, "kinds": ["abstract", "finish"], "returns": [0.01, 0.03]},
]
for p, target in zip(POINTS, ({"1. abstract": 1.0}, {"2. finish": 1.0}), strict=True):
    p["target"] = target


def examples():
    out = []
    for p in POINTS:
        crit = {f"{i + 1}. {k}": None for i, k in enumerate(p["kinds"])}
        q = {"type": "choice", "instructions": "Pick.", "criteria": crit}
        out.append(Example("s", q, p["target"]))
    return out


def test_decision_regret_follows_the_served_pick():
    points = cmp.with_names(POINTS, examples())
    good = cmp.decision_scores([[0.9, 0.1], [0.4, 0.6]], points)
    assert good["regret"] == 0 and good["best_rate"] == 1 and good["finish_rate"] == 0.5
    assert good["expected_regret"] == pytest.approx((0.001 + 0.008) / 2)
    assert good["ce"] == pytest.approx(-(math.log(0.9) + math.log(0.6)) / 2)
    # finish below the profile's 0.5 threshold: the most probable search instead
    bad = cmp.decision_scores([[0.4, 0.6], [0.55, 0.45]], points)
    assert bad["regret"] == pytest.approx((0.01 + 0.02) / 2) and bad["best_rate"] == 0
    assert cmp.decision_scores([[1, 0], [0.5, 0.5]], points)["regret"] == 0  # finish at 0.5
    with pytest.raises(ValueError, match="distributions"):
        cmp.decision_scores([[1, 0]], points)
    with pytest.raises(ValueError, match="differ"):
        three = {"type": "choice", "instructions": "Pick.", "criteria": dict.fromkeys("abc")}
        cmp.with_names(POINTS, [Example("s", three, {"a": 1.0}), examples()[1]])


def test_episode_experiment_runs_every_task_from_the_start(source):  # noqa: F811
    exp = cmp.episode_experiment(source, "ctl_fixed", "d1_sft", adapter="a" * 64, run_usd=0.6)
    cell = exp.benchmarks["deepscholar"]
    assert (cell.split, cell.ids, cell.systems) == ("train", None, ("d1_sft",))
    spec = exp.systems["d1_sft"]
    assert spec["policy"] == {"kind": "d1", "adapter": "a" * 64}  # chooses: no sampling
    assert "start" not in spec and spec["task_usd"] == 0.5  # the source's own cap kept
    assert (exp.budget.run_usd, exp.seed, exp.name) == (0.6, 0, "compare/train/d1_sft")


def episode_run(name: str, system: str, docs: dict, spent: float = 0.05):
    """A finished episode run: per task, the writer's documents by title (None: failed)."""
    run = paths.RUNS / name
    for key, titles in docs.items():
        folder = run / "deepscholar" / system / "tasks" / key
        folder.mkdir(parents=True)
        if titles is None:
            (folder / "result.json").write_text(json.dumps({"status": "failed"}))
            continue
        state = State("T", 0.27, spent_usd=spent, papers=[Paper(row(t), True) for t in titles])
        state.docs, state.batches = [row(t)["url"] for t in titles], len(titles)
        (folder / "state.json").write_text(state.to_json())
        (folder / "result.json").write_text(json.dumps({"status": "ok"}))
    systems = {system: {"use": "controlled", "policy": {"kind": "fixed"}}}
    exp = experiment("val", systems, [system])
    exp["benchmarks"]["deepscholar"]["ids"] = list(docs)
    (run / "manifest.json").write_text(json.dumps({"experiment": exp}))
    return run


def test_paired_comparison_of_episode_returns(source, monkeypatch):  # noqa: F811
    tasks = {k: Task("deepscholar", k, "T") for k in ("t1", "t2", "nogold")}

    def run_tasks(manifest, bench):
        return {k: tasks[k] for k in manifest["experiment"]["benchmarks"][bench]["ids"]}

    monkeypatch.setattr(cmp, "run_tasks", run_tasks)
    fixed = episode_run("fx", "ctl_fixed", {"t1": ["gold a"], "t2": ["gold a"], "nogold": []})
    d1 = episode_run("d1", "d1_rl", {"t1": ["gold a", "gold b"], "t2": None, "nogold": []}, 0.08)
    found = cmp.episode_returns(d1, "d1_rl")
    assert found["t2"]["status"] == "failed" and found["nogold"]["return"] is None
    assert found["t1"]["return"] == pytest.approx(0.15 - 0.1 * 0.08 / 0.27)
    rows = cmp.compare({"fixed": cmp.episode_returns(fixed, "ctl_fixed"), "rl": found})
    assert [r["tasks"] for r in rows] == [1, 1]  # only t1 is scored by both
    assert rows[1]["diff"] == pytest.approx(0.15 * 0.5 - 0.1 * 0.03 / 0.27)
    assert (rows[1]["won"], rows[1]["failed"], rows[0]["tied"]) == (1, 1, 1)
    assert "| rl | 1 |" in cmp.table(rows)
    with pytest.raises(ValueError, match="no deepscholar/x cell"):
        cmp.episode_returns(d1, "x")

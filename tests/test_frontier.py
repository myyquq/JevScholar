"""The d1 controllers against the fixed rule's cost-quality curve at matched cost (offline: a
long table written by hand for the June dev comparison's systems)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from jevscholar.config import load_experiment
from jevscholar.controller import frontier as fr
from jevscholar.paths import EXPERIMENTS

TASKS = [f"t{i}" for i in range(6)]
NOISE = [0.0, 0.01, -0.01, 0.02, -0.02, 0.0]  # per task: the paired intervals are not empty
# system -> (list cost per task, reference recall)
SYSTEMS = {
    "ctl_fixed": (0.02, 0.20),
    "ctl_fixed_b4": (0.03, 0.22),
    "ctl_fixed_b5": (0.04, 0.25),
    "ctl_fixed_b6": (0.05, 0.26),
    "ctl_d1": (0.06, 0.27),  # costs more than every fixed budget
    "ctl_d1_sft": (0.025, 0.20),
    "ctl_d1_rl": (0.035, 0.30),
}


def manifest() -> dict:
    exp = load_experiment(EXPERIMENTS / "evaluate.yaml", "dev-trained")
    return {"run_id": "june", "experiment": exp.model_dump(mode="json")}


def long_table() -> pd.DataFrame:
    rows = []
    for system, (cost, recall) in SYSTEMS.items():
        for task, eps in zip(TASKS, NOISE, strict=True):
            for metric in fr.METRICS:
                value = recall + eps if metric == "reference_recall" else 0.5
                row = dict(bench="deepscholar", system=system, task=task, status="ok")
                row |= dict(eval_error=False, has_report=True, metric=metric, value=value)
                row |= dict(judged=True, defined=True, valid=True, words=400)
                rows.append(row | dict(cost_usd=cost, billed_usd=cost / 2, cold_usd=cost))
    return pd.DataFrame(rows)


def test_the_run_names_its_learned_and_fixed_controllers():
    learned, fixed = fr.policies(manifest())
    assert learned == ["ctl_d1", "ctl_d1_sft", "ctl_d1_rl"]
    assert fixed == ["ctl_fixed", "ctl_fixed_b4", "ctl_fixed_b5", "ctl_fixed_b6"]


def test_matched_budget_never_gives_the_fixed_rule_less_money():
    costs = {"a": 0.02, "b": 0.03, "c": 0.04}
    assert fr.matched(costs, list(costs), 0.025) == "b"
    assert fr.matched(costs, list(costs), 0.03) == "b"  # as much is enough
    assert fr.matched(costs, list(costs), 0.01) == "a"
    assert fr.matched(costs, list(costs), 0.09) == "c"  # none costs as much: the largest


def test_each_controller_is_paired_with_its_matched_budget_and_the_curve():
    groups = {s: s for s in SYSTEMS}
    learned, fixed = fr.policies(manifest())
    table, pairs = fr.frontier(long_table(), groups, learned, fixed)
    assert list(table.system) == [*fixed, *learned]
    assert table.set_index("system").at["ctl_d1_rl", "cost_usd"] == pytest.approx(0.035)

    recall = pairs[pairs.metric == "reference_recall"].set_index("system")
    assert dict(recall.matched) == {
        "ctl_d1": "ctl_fixed_b6",
        "ctl_d1_sft": "ctl_fixed_b4",
        "ctl_d1_rl": "ctl_fixed_b5",
    }
    assert recall.at["ctl_d1_rl", "diff"] == pytest.approx(0.05)
    assert recall.at["ctl_d1_rl", "n"] == len(TASKS)
    # the curve between b4 (0.03, 0.22) and b5 (0.04, 0.25) is 0.235 at 0.035
    assert recall.at["ctl_d1_rl", "vs_curve"] == pytest.approx(0.30 - 0.235)
    assert recall.at["ctl_d1_sft", "vs_curve"] == pytest.approx(0.20 - 0.21)
    assert pd.isna(recall.at["ctl_d1", "vs_curve"])  # beyond the curve's costs
    cost = pairs[pairs.metric == "cost_usd"].set_index("system")
    assert cost.at["ctl_d1_rl", "diff"] == pytest.approx(-0.005)


def test_the_report_reads_a_reported_run(tmp_path):
    run = tmp_path / "june"
    (run / "analysis").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(manifest()))
    with pytest.raises(FileNotFoundError, match="jev report"):
        fr.frontier_report(run)
    long = long_table()
    s1 = long.assign(system=long.system + "_s1")  # the second writer sample, the same scores
    pd.concat([long, s1]).to_csv(run / "analysis" / "long.csv", index=False)
    text = fr.frontier_report(run)
    assert text.startswith("# Matched-cost comparison: june")
    assert "| ctl_d1_rl | ctl_fixed_b5 | reference_recall |" in text

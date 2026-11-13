"""The stopping headroom over the fixed rule's budgets (offline: long tables written by hand for
the June dev comparison's fixed budgets, two writer samples each)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from jevscholar.config import load_experiment
from jevscholar.controller import headroom as hr
from jevscholar.paths import EXPERIMENTS

BUDGETS = ["ctl_fixed", "ctl_fixed_b4", "ctl_fixed_b5", "ctl_fixed_b6"]
COSTS = [0.02, 0.03, 0.04, 0.05]


def manifest() -> dict:
    exp = load_experiment(EXPERIMENTS / "evaluate.yaml", "dev-trained")
    return {"run_id": "june", "experiment": exp.model_dump(mode="json")}


def long_table(score) -> pd.DataFrame:
    """score(task, budget index, sample) -> utility."""
    rows = []
    for j, head in enumerate(BUDGETS):
        for k, system in enumerate((head, f"{head}_s1")):
            for t in range(8):
                row = dict(bench="deepscholar", system=system, task=f"t{t}", status="ok")
                row |= dict(eval_error=False, has_report=True, metric="utility")
                row |= dict(value=score(t, j, k), judged=True, defined=True, valid=True)
                rows.append(row | dict(cost_usd=COSTS[j], billed_usd=0, cold_usd=0, words=1))
    return pd.DataFrame(rows)


def test_samples_pairs_each_budget_with_its_second_writer_sample():
    groups = hr.samples(manifest(), BUDGETS)
    assert groups["ctl_fixed_b5"] == ["ctl_fixed_b5", "ctl_fixed_b5_s1"]


def test_pick_spends_where_it_pays_within_the_target():
    score = np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]])
    cost = np.tile(COSTS, (2, 1))
    # at the 4-batch budget's mean (0.03): the flat task stops at 3, the rising one gets 5
    assert list(hr.pick(score, cost, 0.03)) == [0, 2]
    assert list(hr.pick(score, cost, 0.0)) == [0, 0]  # nothing fits: the cheapest


def test_an_oracle_gains_where_tasks_differ_and_not_on_noise():
    # half the tasks gain from every batch, half do not; both samples agree
    signal = long_table(lambda t, j, k: 0.4 + (0.02 * j if t % 2 else 0.0))
    gains, picks = hr.headroom(signal, hr.samples(manifest(), BUDGETS))
    row = gains.set_index("budget").loc["ctl_fixed_b4"]
    assert row.oracle_cost <= row.uniform_cost + 1e-12
    assert row["diff"] == pytest.approx(0.01) and row.n == 8  # 3 vs 4 flat, 3 vs 5 rising
    counts = picks.set_index("budget").loc["ctl_fixed_b4"]
    assert counts.sum() == 8 and counts.picks_ctl_fixed >= 4  # the flat tasks stay at 3

    # pure noise, independent across samples: the in-sample oracle looks good, the
    # cross-fitted one does not
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, size=(8, 4, 2))
    flat = long_table(lambda t, j, k: 0.4 + noise[t, j, k])
    gains, _ = hr.headroom(flat, hr.samples(manifest(), BUDGETS))
    row = gains.set_index("budget").loc["ctl_fixed_b5"]
    assert row.in_sample_diff > 0.02
    assert row["diff"] < row.in_sample_diff


def test_the_report_reads_a_reported_run(tmp_path):
    run = tmp_path / "june"
    (run / "analysis").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(manifest()))
    long_table(lambda t, j, k: 0.4 + 0.01 * j * (t % 2)).to_csv(
        run / "analysis" / "long.csv", index=False
    )
    text = hr.headroom_report(run, ["utility"])
    assert text.startswith("# Stopping headroom: june") and "## utility" in text
    with pytest.raises(ValueError, match="nugget_coverage at every budget"):
        hr.headroom_report(run, ["nugget_coverage"])

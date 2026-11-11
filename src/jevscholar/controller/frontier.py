"""The learned controllers against the fixed rule at matched cost (docs/06).

A controller that spends more can find more without allocating better. So the fixed rule is run
at several budgets (3 to 6 batches) in the same run, giving a cost-quality curve, and each
learned controller (a d1 policy) is compared with it at its own cost:

- **Matched pair** (paired per task, 95% interval): against the cheapest fixed budget whose mean
  list cost is at least the controller's, so the fixed rule never gets less money (the most
  expensive one if none costs as much). The rule looks at costs only, never at quality.
- **Against the curve** (descriptive): the controller's mean minus the fixed curve's, linearly
  interpolated at the controller's mean cost (empty outside the curve's cost range).

Reads a run's `analysis/long.csv` (written by `jev report`); writer samples are averaged per task.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from jevscholar.analysis import KEY, means, merge_samples, paired, sample_groups
from jevscholar.artifacts import read_manifest
from jevscholar.config import Experiment
from jevscholar.report import md_table

METRICS = ("nugget_coverage", "reference_recall", "utility", "cite_p", "claim_support")
COST = "cost_usd"


def policies(manifest: dict) -> tuple[list[str], list[str]]:
    """(learned, fixed): the run's systems with a d1 policy and with the fixed rule, one name
    per writer-sample group, the fixed ones in the run's order."""
    exp = Experiment.model_validate(manifest["experiment"])
    heads = dict.fromkeys(sample_groups(manifest).values())
    order = [s for cell in exp.benchmarks.values() for s in cell.systems if s in heads]
    kinds = {}
    for system in order:
        use, params = exp.system_spec(system)
        if use == "controlled":
            kinds[system] = params.get("policy", {}).get("kind")
    return [s for s in order if kinds.get(s) == "d1"], [s for s in order if kinds.get(s) == "fixed"]


def curve(merged: pd.DataFrame, systems: Sequence[str]) -> pd.DataFrame:
    """Per system: its mean list cost per task and the primary mean of each metric."""
    m = means(merged[merged.system.isin(systems)])
    wide = m[m.metric.isin(METRICS)].pivot(index="system", columns="metric", values="primary")
    cost = merged.drop_duplicates(KEY).groupby("system")[COST].mean()
    out = wide.reindex(columns=list(METRICS)).assign(**{COST: cost}).reindex(list(systems))
    return out[[COST, *METRICS]]


def matched(costs: Mapping[str, float], fixed: Sequence[str], cost: float) -> str:
    """The cheapest fixed budget costing at least `cost`, else the most expensive one."""
    ranked = sorted(fixed, key=costs.__getitem__)
    return next((f for f in ranked if costs[f] >= cost), ranked[-1])


def on_curve(points: pd.DataFrame, metric: str, cost: float) -> float:
    """The fixed curve's metric linearly interpolated at `cost` (NaN outside its range)."""
    pts = points.sort_values(COST)
    xs, ys = pts[COST].to_numpy(float), pts[metric].to_numpy(float)
    if not len(xs) or cost < xs[0] or cost > xs[-1]:
        return math.nan
    return float(np.interp(cost, xs, ys))


def frontier(
    long: pd.DataFrame, groups: Mapping[str, str], learned: Sequence[str], fixed: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(curve, comparisons): every system's cost and means, then each learned controller against
    its matched fixed budget (paired) and against the interpolated fixed curve."""
    if not learned or not fixed:
        raise ValueError("needs at least one d1 controller and one fixed-rule budget")
    merged = merge_samples(long, groups)
    table = curve(merged, [*fixed, *learned])
    costs = table[COST].to_dict()
    rows = []
    for system in learned:
        base = matched(costs, fixed, costs[system])
        pairs = paired(merged[merged.system.isin([system, base])], base)
        pairs = pairs[pairs.metric.isin([*METRICS, COST])].set_index("metric")
        for metric in (*METRICS, COST):
            if metric not in pairs.index:
                continue
            row = pairs.loc[metric]
            vs = (
                table.at[system, metric] - on_curve(table.loc[list(fixed)], metric, costs[system])
                if metric != COST
                else math.nan
            )
            rows.append(
                {
                    "system": system,
                    "matched": base,
                    "metric": metric,
                    "n": int(row.n),
                    "diff": row["diff"],
                    "ci_lo": row.ci_lo,
                    "ci_hi": row.ci_hi,
                    "p_signflip": row.p_signflip,
                    "vs_curve": vs,
                }
            )
    return table.reset_index(), pd.DataFrame(rows)


def frontier_report(run_dir: Path) -> str:
    """The Markdown of a run's matched-cost comparison."""
    manifest = read_manifest(run_dir)
    long_csv = run_dir / "analysis" / "long.csv"
    if not long_csv.exists():
        raise FileNotFoundError(f"{long_csv} is missing: run `jev report` first")
    learned, fixed = policies(manifest)
    table, pairs = frontier(pd.read_csv(long_csv), sample_groups(manifest), learned, fixed)
    return "\n\n".join(
        [
            f"# Matched-cost comparison: {run_dir.name}",
            "Mean list cost per task (generation) and primary means; writer samples averaged.",
            md_table(table, 4),
            "Each d1 controller against the cheapest fixed budget costing at least as much "
            "(paired per task, 95% t-interval, sign-flip p) and against the fixed curve "
            "interpolated at its cost (vs_curve).",
            md_table(pairs, 4),
        ]
    )

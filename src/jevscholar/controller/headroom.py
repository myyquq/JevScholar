"""How much a stopping controller could gain over the fixed rule at matched cost (docs/06).

The fixed rule spends the same number of batches on every task; a controller can only beat it at
the same mean cost by spending more where more search pays and less where it does not. This
bounds that gain with an oracle that sees each task's outcome at every fixed budget (3 to 6
batches, run in the same run) and picks one budget per task.

An oracle picked and scored on the same noisy scores would mostly pick noise. So it is
cross-fitted on the two independent writer samples each budget has: the budget is picked on one
sample and scored on the other, both ways, and averaged per task. For each uniform budget, the
oracle maximises the picking sample's total score with a mean cost no higher than that budget's
(an exact multiple-choice knapsack over 0.1-millidollar cost units); the per-task difference
against the uniform budget is then paired (95% t-interval, sign-flip p). The in-sample oracle
(picked and scored on the samples' mean) is reported beside it as an optimistic bound.

Reads a run's `analysis/long.csv` (written by `jev report`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from jevscholar.analysis import paired_stats, primary_score
from jevscholar.artifacts import read_manifest
from jevscholar.config import Experiment
from jevscholar.controller.frontier import policies
from jevscholar.report import md_table

COST = "cost_usd"
UNIT = 1e-4  # list USD per knapsack cost unit


def samples(manifest: dict, fixed: Sequence[str]) -> dict[str, list[str]]:
    """Each fixed budget -> its two writer samples' system names, in sample order."""
    exp = Experiment.model_validate(manifest["experiment"])
    names = [s for cell in exp.benchmarks.values() for s in cell.systems]
    out = {}
    for head in fixed:
        members = [s for s in names if s == head or s.startswith(f"{head}_s")]
        members.sort(key=lambda s: exp.system_spec(s)[1].get("sample") or 0)
        if len(members) != 2:
            raise ValueError(f"{head}: needs exactly two writer samples, found {members}")
        out[head] = members
    return out


def grids(
    long: pd.DataFrame, groups: dict[str, list[str]], metric: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(scores [sample, task, budget], costs [task, budget], tasks): the tasks scored for the
    metric at every budget in both samples; costs are the two samples' mean."""
    rows = long[long.metric == metric].assign(score=primary_score(long[long.metric == metric]))
    budgets = list(groups)
    score = {}
    cost = {}
    for b in budgets:
        for k, system in enumerate(groups[b]):
            sub = rows[rows.system == system].set_index("task")
            score[(k, b)] = sub.score
            cost[(k, b)] = sub[COST]
    tasks = sorted(set.intersection(*(set(s.dropna().index) for s in score.values())))
    if not tasks:
        raise ValueError(f"no task has {metric} at every budget in both samples")
    s = np.array([[[score[(k, b)][t] for b in budgets] for t in tasks] for k in (0, 1)], float)
    c = np.array([[(cost[(0, b)][t] + cost[(1, b)][t]) / 2 for b in budgets] for t in tasks], float)
    return s, c, tasks


def pick(score: np.ndarray, cost: np.ndarray, target: float) -> np.ndarray:
    """Per task, the budget maximising the total score with a mean cost of at most `target`
    (costs rounded up to whole units, so the bound holds); among equal totals the cheapest. If
    even the cheapest budgets exceed it, the cheapest."""
    n, m = cost.shape
    units = np.ceil(cost / UNIT - 1e-9).astype(int)
    cap = int(np.floor(target * n / UNIT + 1e-9))
    if units.min(axis=1).sum() > cap:
        return cost.argmin(axis=1)
    best = np.full(cap + 1, -np.inf)
    best[0] = 0.0
    back = np.zeros((n, cap + 1), dtype=int)
    for i in range(n):
        new = np.full(cap + 1, -np.inf)
        for j in range(m):
            u = units[i, j]
            if u > cap:
                continue
            cand = np.full(cap + 1, -np.inf)
            cand[u:] = best[: cap + 1 - u] + score[i, j]
            better = cand > new + 1e-12
            new[better], back[i, better] = cand[better], j
        best = new
    top = best.max()
    at = int(np.flatnonzero(best >= top - 1e-12)[0])  # the cheapest total reaching the best
    choice = np.zeros(n, dtype=int)
    for i in range(n - 1, -1, -1):
        choice[i] = back[i, at]
        at -= units[i, choice[i]]
    return choice


def headroom(
    long: pd.DataFrame, groups: dict[str, list[str]], metric: str = "utility"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gains, picks): per uniform budget, the cross-fitted oracle at no more mean cost against it,
    and how often the oracle picks each budget."""
    s, c, tasks = grids(long, groups, metric)
    budgets = list(groups)
    rows_ = np.arange(len(tasks))
    mean = s.mean(axis=0)
    gains, picks = [], []
    for j, b in enumerate(budgets):
        target = c[:, j].mean()
        a = pick(s[0], c, target)  # picked on sample 0, scored on sample 1
        z = pick(s[1], c, target)
        crossed = (s[1][rows_, a] + s[0][rows_, z]) / 2
        insample = pick(mean, c, target)
        stats = paired_stats(crossed - mean[:, j])
        gains.append(
            {
                "budget": b,
                "metric": metric,
                "uniform_cost": target,
                "oracle_cost": (c[rows_, a].mean() + c[rows_, z].mean()) / 2,
                "uniform": mean[:, j].mean(),
                "oracle": crossed.mean(),
                **stats,
                "in_sample_diff": (mean[rows_, insample] - mean[:, j]).mean(),
            }
        )
        counts = np.bincount(np.concatenate([a, z]), minlength=len(budgets)) / 2
        named = zip(budgets, counts, strict=True)
        picks.append({"budget": b, **{f"picks_{x}": int(n) for x, n in named}})
    return pd.DataFrame(gains), pd.DataFrame(picks)


def headroom_report(run_dir: Path, metrics: Sequence[str]) -> str:
    """The Markdown of a run's stopping headroom, one section per metric."""
    manifest = read_manifest(run_dir)
    long_csv = run_dir / "analysis" / "long.csv"
    if not long_csv.exists():
        raise FileNotFoundError(f"{long_csv} is missing: run `jev report` first")
    _, fixed = policies(manifest)
    if len(fixed) < 2:
        raise ValueError(f"needs at least two fixed-rule budgets, the run has {fixed}")
    groups = samples(manifest, fixed)
    long = pd.read_csv(long_csv)
    parts = [
        f"# Stopping headroom: {run_dir.name}",
        "A per-task oracle over the fixed budgets at no more mean cost than each uniform budget, "
        "picked on one writer sample and scored on the other (both ways, averaged per task); "
        "diff = oracle − uniform, paired per task (95% t-interval, sign-flip p). "
        "in_sample_diff picks and scores on the same samples (optimistic).",
    ]
    for metric in metrics:
        gains, picks = headroom(long, groups, metric)
        parts += [f"## {metric}", md_table(gains, 4), md_table(picks, 0)]
    return "\n\n".join(parts)

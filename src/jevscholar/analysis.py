"""Run analysis: one long table, failure-keeping means, paired tests and costs (report.py writes
them into report.md).

Rules encoded here (they are the point of this module):
- The task universe is the split prefix recorded in the manifest. A task is scored on its eval
  file's metrics plus its benchmark's primary metric, which is always present. In the primary
  mean, a failed or missing generation, or a judging that settled on no score (no citations to
  check), scores 0 and stays in the denominator (except for lower-is-better metrics, where 0
  would be the best value). A metric that was not judged (not requested, a judge error, no eval
  yet) is no evidence either way: its task is left out and counted (`n_unjudged`). The
  official-style mean (valid tasks only, as upstream aggregators do) and the valid count are
  reported next to it.
- One pre-declared primary metric per benchmark. Paired differences against a baseline get a
  t-interval and an exact sign-flip permutation p-value (dev n is 12-30): the primary metric, the
  citation-reliability metrics a gain must not cost and reference coverage (DeepScholar, upstream
  and repaired versions side by side), and the per-task cost; the rest is exploratory. A metric
  undefined for a task (reference recall without important citations) leaves it out entirely.
- Costs come only from the gateway ledger: list and billed USD, by stage, LLM vs tools. A task
  re-run on resume counts its latest session only (it replays its cached calls at recorded cost).
- Systems that differ only in their writer `sample` are one system's independent samples:
  comparisons average them per task
  (`merge_samples`).
  List cost prices the provider's prompt-cache hits as they happened; `cold_usd` prices every
  input token as a miss, the sensitivity view of a comparison that cache order could sway.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from jevscholar import benchmarks, ledger
from jevscholar.artifacts import run_tasks
from jevscholar.config import Alias, Experiment, load_models
from jevscholar.types import JUDGE_PREFIX, parse_tag
from jevscholar.util import read_json

COSTS = ("cost_usd", "cold_usd")  # per-task list cost, as recorded and cache-cold
UNJUDGED = ("not_requested", "error:")  # statuses of a metric its eval file did not settle
NO_ZERO = {"blocked_rate"}  # lower is better: an unscored task is not imputed (0 = best)
KEY = ["bench", "system", "task"]
EXACT_MAX_N = 16
DRAWS = 20000


# --------------------------------------------------------------------------- tables
def ledger_frame(run_dir: Path) -> pd.DataFrame:
    """One row per counted ledger event, its tag split into (bench, system, task, stage).

    Generation rows come from each task's latest session only: a task left pending and re-run on
    resume replays its cached calls at their recorded list cost, so its abandoned session would
    count twice. Judge rows are all kept (judges are never cached; every judging is new spend).
    """
    aliases = load_models().aliases
    rows = []
    for ev in ledger.read(run_dir / "ledger.jsonl"):
        _, bench, system, task, stage = parse_tag(ev["tag"])
        rows.append(
            {
                "bench": bench,
                "system": system,
                "task": task,
                "stage": stage,
                "session": ev["session"],
                "kind": ev["kind"],
                "name": ev["name"],
                "ts": ev["ts"],
                "cost_usd": ev["cost_usd"],
                "billed_usd": ev["billed_usd"],
                "cold_usd": cold_usd(ev, aliases),
            }
        )
    cols = [*KEY, "stage", "session", "kind", "name", "ts", "cost_usd", "billed_usd", "cold_usd"]
    usd = dict.fromkeys(["cost_usd", "billed_usd", "cold_usd"], float)  # also when empty
    df = pd.DataFrame(rows, columns=cols).astype(usd)
    judge = df.stage.str.startswith(JUDGE_PREFIX)
    latest = df.groupby([*KEY, judge]).session.transform("last")
    return df[judge | (df.session == latest)].reset_index(drop=True)


def cold_usd(ev: dict[str, Any], aliases: dict[str, Alias]) -> float:
    """An event's list cost with every input token priced as a cache miss (tool events, and LLM
    events of aliases no longer configured, keep their recorded cost)."""
    alias = aliases.get(ev["name"]) if ev["kind"] == "llm" else None
    if alias is None or not ev.get("tokens"):
        return ev["cost_usd"]
    t, price = ev["tokens"], alias.price
    return ((t["miss"] + t["hit"]) * price.miss + t["out"] * price.out) / 1e6


def long_table(
    run_dir: Path, judgeset: str = "deepseek", costs: pd.DataFrame | None = None
) -> pd.DataFrame:
    """(bench, system, task, status, eval_error, has_report, metric, value, judged, defined,
    valid, cost/billed/cold USD, words).

    One row per task and metric: the benchmark's primary, reliability and secondary metrics for
    every task (so a failed generation counts against them too), plus the eval file's other
    "metrics" ("diagnostics" are not scores). Costs are generation costs; valid = has a report and
    a score; judged = the eval file settles the metric (`judged`); defined = the metric exists for
    the task, whatever its generation and judging (`Benchmark.undefined`).
    """
    from jevscholar.render import word_count

    manifest = read_json(run_dir / "manifest.json")
    costs = ledger_frame(run_dir) if costs is None else costs
    undefined = undefined_metrics(manifest)
    tasks, scores = [], []
    for bench, keys in manifest["tasks"].items():
        for system in manifest["experiment"]["benchmarks"][bench]["systems"]:
            cell = run_dir / bench / system
            for key in keys:
                result = read_json(cell / "tasks" / key / "result.json", None)
                ev = read_json(cell / "eval" / judgeset / f"{key}.json", {})
                report = result and result["report"]
                row = {"bench": bench, "system": system, "task": key}
                tasks.append(
                    {
                        **row,
                        "offset_usd": offset_usd(cell / "tasks" / key, report),
                        "status": result["status"] if result else "missing",
                        "eval_error": "error" in ev,
                        "has_report": report is not None,
                        "words": word_count(report["text"]) if report else math.nan,
                    }
                )
                b = benchmarks.get(bench)
                grid = dict.fromkeys((b.primary, *protocol_metrics(bench)))
                metrics = grid | ev.get("metrics", {}) | b.derived(ev)
                scores += [
                    {
                        **row,
                        "metric": m,
                        "value": math.nan if v is None else float(v),
                        "judged": judged(ev, b.derived_from.get(m, m)),
                        "defined": m not in undefined.get((bench, key), ()),
                    }
                    for m, v in metrics.items()
                ]
    gen = costs[~costs.stage.str.startswith(JUDGE_PREFIX)]
    per_task = gen.groupby(KEY, as_index=False)[["cost_usd", "billed_usd", "cold_usd"]].sum()
    df = (
        pd.DataFrame(tasks)
        .merge(per_task, how="left", on=KEY)
        .fillna({"cost_usd": 0.0, "billed_usd": 0.0, "cold_usd": 0.0})
        .merge(pd.DataFrame(scores), on=KEY)
    )
    df["cost_usd"] += df.offset_usd  # the list cost before a continued run's decision point
    df.loc[df.offset_usd > 0, "cold_usd"] = math.nan  # a prefix's cache-cold cost is not known
    df["valid"] = df.has_report & df.value.notna()
    cols = [
        *KEY,
        "status",
        "eval_error",
        "has_report",
        "metric",
        "value",
        "judged",
        "defined",
        "valid",
        "cost_usd",
        "billed_usd",
        "cold_usd",
        "words",
    ]
    return df[cols]


def offset_usd(folder: Path, report: dict[str, Any] | None) -> float:
    """The list cost before a continued run's decision point: its report's `offset_usd`, or for
    a continuation that failed before reporting, the one its start.json records."""
    if report is not None:
        return report["meta"].get("offset_usd", 0.0)
    return read_json(folder / "start.json", {}).get("offset_usd", 0.0)


def undefined_metrics(manifest: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    """(bench, task key) -> the metrics that do not exist for the task (Benchmark.undefined)."""
    out = {}
    for bench in manifest["tasks"]:
        b = benchmarks.get(bench)
        out |= {(bench, key): b.undefined(task) for key, task in run_tasks(manifest, bench).items()}
    return out


def primary(bench: str) -> str:
    """A benchmark's pre-declared primary metric."""
    return benchmarks.get(bench).primary


def protocol_metrics(bench: str) -> tuple[str, ...]:
    """A benchmark's reliability and secondary metrics: scored for every task, paired."""
    b = benchmarks.get(bench)
    return (*b.reliability, *b.secondary)


def judged(ev: dict[str, Any], metric: str) -> bool:
    """Whether an eval file settles `metric` (for a derived metric: its source): a score, or a
    definitive reason for none (an empty export, a report without citations). Not requested, a
    judge error or no eval file is not."""
    status = ev.get("status")
    if isinstance(status, dict):
        settled = status.get(metric)
        return settled is not None and not settled.startswith(UNJUDGED)
    if status is not None:  # one status for the task, e.g. "no_report"
        return True
    return bool(ev.get("metrics")) and "error" not in ev  # files without per-metric statuses


def primary_score(long: pd.DataFrame) -> pd.Series:
    """Primary scoring: a valid value as is; 0 for a failed generation or a judging that settled
    on no score (NaN for NO_ZERO metrics); NaN for a metric not judged or undefined."""
    zero = pd.Series(np.where(long.metric.isin(NO_ZERO), np.nan, 0.0), index=long.index)
    score = long.value.where(long.valid, zero.where(~long.has_report | long.judged))
    return score.where(long.defined)


def means(long: pd.DataFrame) -> pd.DataFrame:
    """Per (bench, system, metric), over the tasks where the metric is defined: n, n_valid,
    n_unjudged, primary (all judged tasks, failures scoring 0) and official (valid tasks only)."""
    scored = long[long.defined].assign(
        primary=primary_score(long),
        official=long.value.where(long.valid),
        unjudged=long.has_report & ~long.judged & ~long.valid,
    )
    return (
        scored.groupby(["bench", "system", "metric"], sort=False)
        .agg(
            n=("task", "size"),
            n_valid=("valid", "sum"),
            n_unjudged=("unjudged", "sum"),
            primary=("primary", "mean"),
            official=("official", "mean"),
        )
        .reset_index()
    )


def sample_groups(manifest: dict[str, Any]) -> dict[str, str]:
    """Each system's name -> the name its scores are merged under: systems whose definitions
    differ only in their writer `sample` are samples of the one with the lowest sample."""
    exp = manifest["experiment"]
    defined = exp.get("systems", {})
    names = {s for cell in exp["benchmarks"].values() for s in cell["systems"]}
    groups: dict[str, list[str]] = {}
    for name in sorted(names):
        spec = {"use": name} | defined.get(name, {})  # a definition without `use` is its class
        spec.pop("sample", None)
        groups.setdefault(json.dumps(spec, sort_keys=True), []).append(name)
    first = lambda g: min(g, key=lambda n: defined.get(n, {}).get("sample", 0))  # noqa: E731
    return {name: first(g) for g in groups.values() for name in g}


def merge_samples(long: pd.DataFrame, groups: Mapping[str, str]) -> pd.DataFrame:
    """The long table with each task's independent writer samples averaged into one row per
    metric, under the first sample's name: a failed sample scores 0 (primary scoring), a sample
    not judged leaves the merged metric unjudged; costs and lengths are averaged. A merged
    system's values are primary scores already, so its official mean equals its primary mean.
    Systems without samples are left as they are."""
    merged = {s for s in groups if s != groups[s]} | {g for s, g in groups.items() if s != g}
    if not merged:
        return long
    alone, many = long[~long.system.isin(merged)], long[long.system.isin(merged)]
    df = many.assign(system=many.system.map(groups.get), value=primary_score(many))
    df = df.assign(valid=df.value.notna())
    agg = {
        "status": "first", "eval_error": "any", "has_report": "any", "value": "mean",
        "judged": "all", "defined": "all", "valid": "all", "cost_usd": "mean",
        "billed_usd": "mean", "cold_usd": "mean", "words": "mean",
    }  # fmt: skip
    keys = [*KEY, "metric"]
    out = df.groupby(keys, sort=False).agg(agg)
    out["value"] = df.groupby(keys, sort=False).value.apply(lambda v: v.mean(skipna=False))
    return pd.concat([alone, out.reset_index()[long.columns]], ignore_index=True)


# --------------------------------------------------------------------------- statistics
def sign_flip_p(d: Any, *, draws: int = DRAWS, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value for mean(d) = 0.

    Exact over all 2^n sign vectors for n <= 16, else Monte Carlo with a fixed seed.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n <= EXACT_MAX_N:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
        return float(np.mean(np.abs(signs @ d) >= abs(d.sum()) - 1e-12))
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(draws, n))
    hits = np.sum(np.abs(signs @ d) >= abs(d.sum()) - 1e-12)
    return float((hits + 1) / (draws + 1))


def paired_stats(d: Any) -> dict[str, float]:
    """Mean paired difference, 95% t-interval and sign-flip p-value (all NaN without pairs)."""
    d = np.asarray(d, dtype=float)
    if not len(d):
        return {
            "n": 0,
            "diff": math.nan,
            "ci_lo": math.nan,
            "ci_hi": math.nan,
            "p_signflip": math.nan,
        }
    n, mean = len(d), float(d.mean())
    half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {
        "n": n,
        "diff": mean,
        "ci_lo": mean - half,
        "ci_hi": mean + half,
        "p_signflip": sign_flip_p(d),
    }


def paired(long: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Every system vs the baseline, per benchmark: the primary, reliability and secondary
    metrics (primary scoring), then the per-task costs (ΔC). A task whose metric is unjudged on
    either side is left out of that comparison (`unjudged` counts them); one where the metric is
    undefined is left out silently."""
    scores = long.assign(score=primary_score(long))[long.defined]
    tasks = long.drop_duplicates(KEY)
    costs = pd.concat([tasks.assign(metric=c, score=tasks[c]) for c in COSTS])
    rows = []
    for bench in long.bench.unique():
        for metric in (primary(bench), *protocol_metrics(bench), *COSTS):
            src = costs if metric in COSTS else scores
            sub = src[(src.bench == bench) & (src.metric == metric)]
            if baseline not in set(sub.system):
                continue
            wide = sub.pivot(index="task", columns="system", values="score")
            for system in sub.system.unique():
                if system == baseline:
                    continue
                d = wide[system] - wide[baseline]
                stats_ = paired_stats(d.dropna())
                rows.append(
                    {
                        "bench": bench,
                        "system": system,
                        "baseline": baseline,
                        "metric": metric,
                        **stats_,
                        "unjudged": int(d.isna().sum()),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- costs
def task_caps(experiment: dict[str, Any]) -> dict[tuple[str, str], float]:
    """(benchmark, system) -> the per-task cap in a run (a manifest's experiment)."""
    return benchmarks.task_caps(Experiment.model_validate(experiment))


def cost_tables(
    long: pd.DataFrame, ledger: pd.DataFrame, caps: Mapping[tuple[str, str], float] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per system: totals, cache-cold list, LLM vs tools, judges, per-task mean/p50/p90/max and
    tasks over their cell's cap in `caps`; per stage: totals, p50/p90). The gateway's cap
    is a soft latch (a call starts only below it) and a controlled system's rules plan with cost
    estimates, so a task can end above its cap: `over_cap` counts those."""
    tasks = long.drop_duplicates(KEY)
    ran = tasks[tasks.status != "missing"]
    cells = pd.Series(list(zip(ran.bench, ran.system, strict=True)), index=ran.index)
    cap = cells.map(caps or {}).astype(float).fillna(math.inf)
    per_system = (
        ran.assign(over_cap=ran.cost_usd > cap)
        .groupby(["bench", "system"], sort=False)
        .agg(
            tasks=("task", "size"),
            list_usd=("cost_usd", "sum"),
            billed_usd=("billed_usd", "sum"),
            cold_usd=("cold_usd", lambda s: s.sum(min_count=len(s))),  # unknown: NaN
            mean_usd=("cost_usd", "mean"),
            p50_usd=("cost_usd", "median"),
            p90_usd=("cost_usd", lambda s: s.quantile(0.9)),
            max_usd=("cost_usd", "max"),
            over_cap=("over_cap", "sum"),
        )
    )
    is_judge = ledger.stage.str.startswith(JUDGE_PREFIX)
    split = ledger.assign(part=np.where(is_judge, "judge_usd", ledger.kind + "_usd")).pivot_table(
        index=["bench", "system"], columns="part", values="cost_usd", aggfunc="sum"
    )
    per_system = per_system.join(split.fillna(0.0)).reset_index()

    stage_keys = ["bench", "system", "stage", "kind"]
    per_task = ledger.groupby([*KEY, "stage", "kind"], as_index=False)[
        ["cost_usd", "billed_usd"]
    ].sum()
    universe = pd.concat([ran[KEY], per_task[KEY]]).drop_duplicates()
    full = (
        per_task[stage_keys]
        .drop_duplicates()
        .merge(universe, on=["bench", "system"])
        .merge(per_task, how="left", on=[*KEY, "stage", "kind"])
        .fillna(0.0)
    )
    per_stage = full.groupby(stage_keys, sort=False).agg(
        list_usd=("cost_usd", "sum"),
        billed_usd=("billed_usd", "sum"),
        p50_usd=("cost_usd", "median"),
        p90_usd=("cost_usd", lambda s: s.quantile(0.9)),
    )
    return per_system, per_stage.reset_index()


def write_trajectories(
    run_dir: Path, ledger: pd.DataFrame, caps: Mapping[tuple[str, str], float]
) -> None:
    """tasks/<key>/trajectory.jsonl: the generation steps of each task in ledger order, with the
    budget left under its cell's cap in `caps` (none for a system the run no longer lists)."""
    gen = ledger[~ledger.stage.str.startswith(JUDGE_PREFIX)]
    for (bench, system, task), steps in gen.groupby(KEY, sort=False):
        cum = steps.cost_usd.cumsum()
        lines = [
            json.dumps(
                {
                    "step": i,
                    "ts": row.ts,
                    "kind": row.kind,
                    "stage": row.stage,
                    "name": row.name,
                    "cost_usd": row.cost_usd,
                    "cum_usd": c,
                    "remaining_usd": caps[bench, system] - c if (bench, system) in caps else None,
                }
            )
            for i, (row, c) in enumerate(zip(steps.itertuples(), cum, strict=True), 1)
        ]
        out = run_dir / bench / system / "tasks" / task
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectory.jsonl").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- retrieval
STAGES = ("screened", "relevant", "writer", "cited")


def funnel(run_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    """Per DeepScholar system and task, the important citations (as reference recall matches
    them) at each stage: screened and judged relevant (a controlled system's state.json), given
    to the writer (its cited sources plus the report's uncited pool) and cited. A stage a system
    does not record is NaN; `n_<stage>` counts the stage's papers."""
    from jevscholar.benchmarks.deepscholar import gold_found
    from jevscholar.search.policy import arxiv_id, norm_title

    if "deepscholar" not in manifest["tasks"]:
        return pd.DataFrame()
    tasks = run_tasks(manifest, "deepscholar")
    rows = []
    for system in manifest["experiment"]["benchmarks"]["deepscholar"]["systems"]:
        for key, task in tasks.items():
            folder = run_dir / "deepscholar" / system / "tasks" / key
            result = read_json(folder / "result.json", None)
            state = read_json(folder / "state.json", None)
            if not (result and result["report"]):
                continue
            report = result["report"]
            screened = state and [
                (p["row"]["url"], p["row"]["title"], p["relevant"]) for p in state["papers"]
            ]
            stages = {
                "screened": state and [(u, t) for u, t, _ in screened],
                "relevant": state and [(u, t) for u, t, relevant in screened if relevant],
                "writer": writer_input(report),
                "cited": [(s["url"], s["title"]) for s in report["sources"]],
            }
            row = {"system": system, "task": key}
            for stage, papers in stages.items():
                if papers is None:
                    row |= {stage: math.nan, f"n_{stage}": math.nan}
                    continue
                ids = {arxiv_id(u) for u, _ in papers}
                found, total = gold_found(task, ids, {norm_title(t) for _, t in papers if t})
                row |= {"n_gold": total, stage: found, f"n_{stage}": len(papers)}
            rows.append(row)
    return pd.DataFrame(rows)


def writer_input(report: dict[str, Any]) -> list[tuple[str, str]] | None:
    """(url, title) of the documents a DeepScholar-style writer got (the report's `docs`); None
    for a system without a writer pool."""
    if "docs" not in report["meta"]:
        return None
    titles = {s["url"]: s["title"] for s in report["sources"]}
    return [(u, titles.get(u, "")) for u in report["meta"]["docs"]]


def funnel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per system, over tasks with important citations: the mean share found at each stage
    (macro), the pooled share (micro), and the mean papers per stage."""
    if df.empty:
        return df
    df = df[df.n_gold > 0]
    g = df.groupby("system", sort=False)
    out = pd.DataFrame({"tasks": g.size(), "gold": g.n_gold.sum()})
    for stage in STAGES:
        seen = df[df[stage].notna()].groupby("system", sort=False)  # tasks recording the stage
        out[stage] = (df[stage] / df.n_gold).groupby(df.system, sort=False).mean()
        out[f"{stage}_micro"] = seen[stage].sum() / seen.n_gold.sum()
        out[f"n_{stage}"] = g[f"n_{stage}"].mean()
    return out.reset_index()

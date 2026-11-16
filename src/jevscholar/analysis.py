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
  t-interval and a sign-flip permutation p-value (exact for n <= 16, else Monte Carlo): the
  primary metric, the citation-reliability metrics a gain must not cost and reference coverage
  (DeepScholar, upstream and repaired versions side by side), and the per-task cost; the rest is
  exploratory. A metric undefined for a task (reference recall without important citations)
  leaves it out entirely.
- Costs come only from the gateway ledger: list and billed USD, by stage, LLM vs tools. A task
  re-run on resume counts its latest session only (it replays its cached calls at recorded cost).
  A controlled run continued from another run's decision point adds the list cost spent before
  that point (its report's `offset_usd`) to its own.
  List cost prices the provider's prompt-cache hits as they happened; `cold_usd` prices every
  input token as a miss, the sensitivity view of a comparison that cache order could sway.
- Where a benchmark scores every component of the registered utility U (rewards.py), each task
  also gets U and the judge-free retrieval return (`outcomes`). Systems that differ only in their
  writer `sample` are one system's independent samples: comparisons average them per task
  (`merge_samples`).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from jevscholar import benchmarks, ledger, rewards
from jevscholar.artifacts import run_tasks
from jevscholar.config import Alias, Experiment, load_models
from jevscholar.types import JUDGE_PREFIX, parse_tag
from jevscholar.util import read_json

COSTS = ("cost_usd", "cold_usd")  # per-task list cost, as recorded and cache-cold
OUTCOMES = ("utility", "retrieval_return")
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
                "fingerprint": ev.get("fingerprint"),
                "ts": ev["ts"],
                "cost_usd": ev["cost_usd"],
                "billed_usd": ev["billed_usd"],
                "cold_usd": cold_usd(ev, aliases),
            }
        )
    cols = [*KEY, "stage", "session", "kind", "name", "fingerprint", "ts"]
    cols += ["cost_usd", "billed_usd", "cold_usd"]
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
    return status is not None  # one status for the task, e.g. "no_report"


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


def outcomes(long: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    """Rows to add to the long table, per task of a benchmark that scores every component of U
    (rewards.UTILITY) and has a cap B0: the utility U, from the components' primary scores (R_c
    of a task without important citations counts 0; a component not judged leaves U unjudged),
    the failed-report U for a task without a report (a failed or missing generation); and the
    judge-free retrieval return, R_e the share of important citations among the writer's
    documents (`funnel`'s writer stage; undefined without important citations or a recorded
    writer input). C is the task's list cost."""
    b0 = {
        b: benchmarks.get(b).cap_usd
        for b in long.bench.unique()
        if set(rewards.UTILITY) <= {primary(b), *protocol_metrics(b)}
    }
    sub = long[long.bench.map(b0).notna()]
    if sub.empty:
        return sub
    score = primary_score(sub).where(sub.defined, 0.0)
    wide = sub.assign(score=score).set_index([*KEY, "metric"]).score.unstack()
    wide = wide.reindex(columns=list(rewards.UTILITY))  # a component never scored: U unjudged
    tasks = sub.drop_duplicates(KEY).set_index(KEY)
    cap = tasks.index.get_level_values("bench").map(b0)
    u = rewards.utility(wide, tasks.cost_usd, cap).where(
        tasks.has_report, rewards.failed_utility(tasks.cost_usd, cap)
    )
    r = pd.Series(math.nan, index=tasks.index)
    if not stages.empty:
        writer = stages.assign(bench="deepscholar").set_index(KEY)
        share = (writer.writer / writer.n_gold).where(writer.n_gold > 0)
        r = rewards.retrieval_return(share.reindex(tasks.index), tasks.cost_usd, cap)
    rows = [
        tasks.assign(metric="utility", value=u, judged=u.notna(), defined=True),
        tasks.assign(metric="retrieval_return", value=r, judged=True, defined=r.notna()),
    ]
    return pd.concat(rows).assign(valid=lambda d: d.value.notna()).reset_index()[long.columns]


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
        "status": "first", "eval_error": "any", "has_report": "any",
        "value": lambda v: v.mean(skipna=False),
        "judged": "all", "defined": "all", "valid": "all", "cost_usd": "mean",
        "billed_usd": "mean", "cold_usd": "mean", "words": "mean",
    }  # fmt: skip
    keys = [*KEY, "metric"]
    out = df.groupby(keys, sort=False).agg(agg)
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
        for metric in (primary(bench), *protocol_metrics(bench), *OUTCOMES, *COSTS):
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


# --------------------------------------------------------------------------- the test's claims
# the main claim (docs/06): metric -> the 95% bound that must hold for the chosen d1 - DS-base
MAIN_CLAIM = {
    "nugget_coverage": ("ci_lo", -0.02),
    "cost_usd": ("ci_hi", 0.0),
    "cite_p": ("ci_lo", -0.05),
    "cite_p_per_citation": ("ci_lo", -0.05),
    "claim_support": ("ci_lo", -0.05),
    "reference_recall": ("ci_lo", -0.05),
}


def scored(run_dir: Path, judgeset: str = "deepseek") -> pd.DataFrame:
    """A run's long table with its outcomes (U, the retrieval return), writer samples merged."""
    manifest = read_json(run_dir / "manifest.json")
    long = long_table(run_dir, judgeset, ledger_frame(run_dir))
    long = pd.concat([long, outcomes(long, funnel(run_dir, manifest))], ignore_index=True)
    return merge_samples(long, sample_groups(manifest))


def holds(row: Mapping[str, Any], bound: tuple[str, float]) -> bool:
    side, threshold = bound
    return row[side] > threshold if side == "ci_lo" else row[side] < threshold


def main_claim(long: pd.DataFrame, arm: str, baseline: str = "deepscholar_base_intro") -> dict:
    """The six-bound main claim of `arm` against `baseline`, paired over every task both have; it
    cannot hold while a judge error leaves a pair unjudged."""
    sub = long[(long.bench == "deepscholar") & long.system.isin([arm, baseline])]
    rows = paired(sub, baseline).set_index("metric")
    checks = {
        m: dict(rows.loc[m]) | {"holds": holds(rows.loc[m], b)} for m, b in MAIN_CLAIM.items()
    }
    complete = all(c["unjudged"] == 0 for c in checks.values())
    held = complete and all(c["holds"] for c in checks.values())
    return {"arm": arm, "holds": held, "complete": complete, "checks": checks}


def ratio_upper(a: Any, b: Any, *, draws: int = DRAWS, seed: int = 0) -> float:
    """Bootstrap 97.5th percentile of mean(a) / mean(b) over paired tasks."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(draws, len(a)))
    return float(np.percentile(a[idx].mean(1) / b[idx].mean(1), 97.5))


def rl_claim(
    long: pd.DataFrame,
    rl: str = "ctl_d1_rl",
    others: tuple[str, ...] = ("ctl_d1_sft", "ctl_fixed"),
    baseline: str = "deepscholar_base_intro",
) -> dict:
    """The secondary RL claim (docs/06): `rl` beats each of `others` on U (paired sign-flip p,
    Holm-corrected), with either more content at similar cost (nugget lower bound > 0, cost-ratio
    upper bound <= 1.05) or less cost at similar content (cost upper bound < 0, nugget lower
    bound > -0.02), and keeps the main claim's reliability and reference-recall bounds against
    `baseline`."""
    sub = long[long.bench == "deepscholar"]
    out: dict[str, Any] = {"rl": rl, "against": {}}
    order = []
    for other in others:
        rows = paired(sub[sub.system.isin([rl, other])], other).set_index("metric")
        tasks = sub.drop_duplicates(KEY).pivot(index="task", columns="system", values="cost_usd")
        ratio = ratio_upper(tasks[rl], tasks[other])
        u, n, c = rows.loc["utility"], rows.loc["nugget_coverage"], rows.loc["cost_usd"]
        if u.unjudged or n.unjudged:
            raise ValueError(f"{rl} vs {other}: unjudged pairs; resolve judge errors first")
        content = n.ci_lo > 0 and ratio <= 1.05
        cheaper = c.ci_hi < 0 and n.ci_lo > -0.02
        out["against"][other] = {
            "u_diff": u["diff"], "u_p": u.p_signflip, "nugget_lo": n.ci_lo, "cost_hi": c.ci_hi,
            "cost_ratio_hi": ratio, "content_or_cost": bool(content or cheaper),
        }  # fmt: skip
        order.append((u.p_signflip, other))
    for k, (p, other) in enumerate(sorted(order)):  # Holm: step down until one fails
        prior = all(out["against"][o]["u_beats"] for _, o in sorted(order)[:k])
        diff = out["against"][other]["u_diff"]
        out["against"][other]["u_beats"] = prior and diff > 0 and p * (len(order) - k) < 0.05
    margins = main_claim(long, rl, baseline)
    keep = ("cite_p", "cite_p_per_citation", "claim_support", "reference_recall")
    out["margins"] = margins["complete"] and all(margins["checks"][m]["holds"] for m in keep)
    out["holds"] = out["margins"] and all(
        a["u_beats"] and a["content_or_cost"] for a in out["against"].values()
    )
    return out


# --------------------------------------------------------------------------- writer selection
# Best-of-n over a task's writer samples from the same evidence: submit the sample whose text
# covers most of the target abstract's content words and leaves the fewest long sentences
# without a citation. Each feature is scaled by its mean within-task deviation on the June
# development pairs; frozen before the test's samples were read.
PICK_SCALE = {"abstract_recall": 0.0284, "uncited": 0.0454}
STOPLIST = (
    "the a an of and or in on for to with by from is are be as that this which we our their its"
    " at it into via using based than these those such can not also more most new"
)
STOPWORDS = frozenset(STOPLIST.split())
CITATION = re.compile(r"\[\d+(?:,\s*\d+)*\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
LONG_SENTENCE = 50  # characters


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9\-]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def pick_features(text: str, abstract: str) -> dict[str, float]:
    """The share of the abstract's content words the text uses (citation markers removed), and
    the share of its long sentences without a citation marker."""
    target = content_words(abstract)
    used = content_words(CITATION.sub("", text))
    long = [s for s in SENTENCE.split(text) if len(s) >= LONG_SENTENCE]
    uncited = sum(1 for s in long if not re.search(r"\[\d", s))
    return {
        "abstract_recall": len(target & used) / max(len(target), 1),
        "uncited": uncited / max(len(long), 1),
    }


def writer_pick(texts: list[str | None], abstract: str) -> int:
    """The index of the sample to submit (the lowest on ties); a missing or blank report is never
    preferred to a written one."""
    scores = []
    for text in texts:
        if not (text or "").strip():
            scores.append(-math.inf)
            continue
        f = pick_features(text, abstract)
        scores.append(
            f["abstract_recall"] / PICK_SCALE["abstract_recall"]
            - f["uncited"] / PICK_SCALE["uncited"]
        )
    return scores.index(max(scores))


# --------------------------------------------------------------------------- best-of-n submissions
NON_INFERIOR = ("nugget_coverage", "reference_recall")  # the claim's; its cost must fall


def judged_metrics(bench: str) -> list[str]:
    """The metrics `jev eval` judges by default, then the protocol's other (derived) ones."""
    return list(dict.fromkeys((*benchmarks.get(bench).judged, *protocol_metrics(bench))))


def writer_samples(run_dir: Path, system: str, judgeset: str = "deepseek") -> pd.DataFrame:
    """One row per DeepScholar task and writer sample of `system` (`sample_groups`): each judged
    metric's primary score (an undefined one 0, as in U; NaN = not judged), the list cost, the
    writer's generation cost `W`, the report's text and the target's abstract."""
    manifest = read_json(run_dir / "manifest.json")
    groups, defined = sample_groups(manifest), manifest["experiment"].get("systems", {})
    led = ledger_frame(run_dir)
    long = long_table(run_dir, judgeset, led)
    long = long[(long.bench == "deepscholar") & (long.system.map(groups) == system)]
    if long.empty:
        raise ValueError(f"{run_dir.name}: no DeepScholar samples of {system}")
    score = long.assign(score=primary_score(long).where(long.defined, 0.0))
    df = score.set_index(["system", "task", "metric"]).score.unstack()
    df = df.reindex(columns=judged_metrics("deepscholar"))
    info = long.drop_duplicates(["system", "task"]).set_index(["system", "task"])
    gen = led[led.stage.str.startswith("generation")].groupby(["system", "task"]).cost_usd.sum()
    df = df.assign(
        has_report=info.has_report, cost_usd=info.cost_usd, W=gen.reindex(df.index, fill_value=0.0)
    ).reset_index()
    tasks = run_tasks(manifest, "deepscholar")
    df["sample"] = [defined.get(s, {}).get("sample", 0) for s in df.system]
    df["abstract"] = [tasks[t].meta["abstract"] for t in df.task]
    texts = []
    for s, t in zip(df.system, df.task, strict=True):
        result = read_json(run_dir / "deepscholar" / s / "tasks" / t / "result.json", None)
        texts.append(result["report"]["text"] if result and result["report"] else None)
    return df.assign(text=texts)


def sample_utility(df: pd.DataFrame) -> pd.Series:
    """U of each DeepScholar sample at its `cost_usd` (the failed U without a report)."""
    b0 = benchmarks.get("deepscholar").cap_usd
    u = rewards.utility(df, df.cost_usd, b0)
    return u.where(df.has_report, rewards.failed_utility(df.cost_usd, b0))


def best_of(samples: pd.DataFrame) -> pd.DataFrame:
    """Per task, the sample `writer_pick` submits from the texts alone (`writer_samples`' rows):
    its scores, charged its own list cost plus every other sample's writer generations, its U at
    that cost, and its `sample`."""
    picks, costs = [], []
    for _, g in samples.sort_values("sample").groupby("task"):
        pick = g.iloc[writer_pick(list(g.text), g.abstract.iloc[0])]
        picks.append(pick.name)
        costs.append(pick.cost_usd + g.W.drop(pick.name).sum())
    out = samples.loc[picks].set_index("task").assign(cost_usd=costs)
    return out.assign(U=sample_utility(out))


def best_of_n(samples: pd.DataFrame, base: pd.DataFrame, margin: float = 0.02) -> dict[str, Any]:
    """`samples`' best-of-n (`best_of`) against the per-task mean of `base`'s samples, paired over
    the same tasks: per metric judged everywhere on both sides, both means, the paired difference
    (`paired_stats`) and, for scores, the non-inferiority p at `margin` (sign_flip_p(d + margin)).
    Then the claim: cost lower (two-sided p) and NON_INFERIOR non-inferior, a wrong sign p = 1,
    the claim's p the largest, holding below .05. A claim metric left unjudged is an error."""
    if samples.groupby("task").size().nunique() != 1:
        raise ValueError("every task needs the same number of samples")
    if base.duplicated(["task", "sample"]).any():
        raise ValueError("a baseline task is in more than one run")
    samples, base = samples.assign(U=sample_utility(samples)), base.assign(U=sample_utility(base))
    cols = [*judged_metrics("deepscholar"), "U", "cost_usd"]
    x, ref = best_of(samples), base.groupby("task")[cols].mean()
    if set(x.index) != set(ref.index):
        raise ValueError("the system and the baseline have different tasks")
    metrics, unjudged, diffs = {}, {}, {}
    for m in cols:
        if missing := int(samples[m].isna().sum() + base[m].isna().sum()):
            if m in NON_INFERIOR:
                raise ValueError(f"{m}: {missing} samples not judged; settle the evals first")
            unjudged[m] = missing
            continue
        d = diffs[m] = x[m] - ref.loc[x.index, m]
        row = {"system": float(x[m].mean()), "baseline": float(ref[m].mean())} | paired_stats(d)
        metrics[m] = row | ({} if m == "cost_usd" else {"p_ni": sign_flip_p(d + margin)})
    parts = {"cost_usd": (metrics["cost_usd"]["p_signflip"], diffs["cost_usd"].mean() < 0)}
    parts |= {m: (metrics[m]["p_ni"], (diffs[m] + margin).mean() > 0) for m in NON_INFERIOR}
    p = max(q if ok else 1.0 for q, ok in parts.values())
    claim = {m: {"p": q, "sign_ok": bool(ok)} for m, (q, ok) in parts.items()}
    return {
        "tasks": len(x),
        "samples": int(samples.groupby("task").size().iloc[0]),
        "metrics": metrics,
        "unjudged": unjudged,
        "claim": claim | {"margin": margin, "p": p, "holds": p < 0.05},
        "cost_ratio": float(x.cost_usd.mean() / ref.cost_usd.mean()),
        "picks": {int(k): int(n) for k, n in x["sample"].value_counts().sort_index().items()},
    }

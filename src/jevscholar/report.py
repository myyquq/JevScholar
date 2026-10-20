"""report.md: a run's primary metrics, paired differences, status, costs and retrieval funnel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from jevscholar.analysis import (
    KEY,
    NO_ZERO,
    cost_tables,
    funnel,
    funnel_summary,
    ledger_frame,
    long_table,
    means,
    merge_samples,
    outcomes,
    paired,
    primary,
    sample_groups,
    task_caps,
    write_trajectories,
)
from jevscholar.util import read_json

NON_OFFICIAL = "DeepSeek-judged (non-official)"


def write_report(
    run_dir: Path, *, baseline: str = "deepscholar_base_intro", judgeset: str = "deepseek"
) -> Path:
    """Write analysis/long.csv, per-task trajectories and report.md; returns report.md's path."""
    manifest = read_json(run_dir / "manifest.json")
    exp = manifest["experiment"]
    systems = {s for cell in exp["benchmarks"].values() for s in cell["systems"]}
    if baseline not in systems:
        raise ValueError(f"baseline {baseline!r} is not a system of the run: {sorted(systems)}")
    ledger = ledger_frame(run_dir)
    stages = funnel(run_dir, manifest)
    long = long_table(run_dir, judgeset, ledger)
    long = pd.concat([long, outcomes(long, stages)], ignore_index=True)
    (run_dir / "analysis").mkdir(exist_ok=True)
    long.to_csv(run_dir / "analysis" / "long.csv", index=False)
    stages.to_csv(run_dir / "analysis" / "funnel.csv", index=False)
    merged = merge_samples(long, sample_groups(manifest))  # scores: a system's samples averaged
    caps = task_caps(exp)
    write_trajectories(run_dir, ledger, caps)

    label = NON_OFFICIAL if judgeset == "deepseek" else f"{judgeset}-judged"
    m = means(merged)
    headline = m[m.metric == m.bench.map(primary)].reset_index(drop=True)
    per_system, per_stage = cost_tables(long, ledger, caps)
    tasks = long.drop_duplicates(KEY)
    cells = tasks.groupby(["bench", "system"], sort=False)
    status = tasks.pivot_table(
        index=["bench", "system"],
        columns="status",
        values="task",
        aggfunc="size",
        fill_value=0,
        sort=False,
    )
    status["eval errors"] = cells.eval_error.sum()
    words = cells.words.agg(["mean", "median", "max"])
    last = manifest["sessions"][-1]
    sha = (last["git_sha"] or "uncommitted")[:12] + ("-dirty" if last["git_dirty"] else "")
    by_cap: dict[float, list[str]] = {}
    for (_, system), cap in caps.items():
        by_cap.setdefault(cap, []).append(system)
    cap_list = "; ".join(f"${c:g}: {', '.join(dict.fromkeys(s))}" for c, s in by_cap.items())
    sections = [
        f"# Report: {manifest['run_id']}",
        f"Experiment `{exp['name']}`, judges `{judgeset}`, code {sha}. All scores are **{label}**.",
        f"## Primary metrics ({label})",
        "primary = mean over the split prefix (failed or missing generations = 0; unjudged "
        "metrics leave their task out, see n_unjudged); official = mean over valid tasks only, "
        "as upstream aggregators do. Systems that differ only in their writer `sample` are one "
        "system's samples, averaged per task under the first one's name; status and cost tables "
        "list every sample.",
        md_table(headline),
        f"## Paired differences vs {baseline} ({label}, primary scoring)",
        "Quality, citation reliability, then generation cost per task (list USD; cold_usd prices "
        "every input token as a cache miss). 95% t-interval; two-sided sign-flip permutation p "
        "(exact for n <= 16).",
        md_table(paired(merged, baseline)),
        f"## All metrics, exploratory ({label})",
        f"Lower-is-better metrics ({', '.join(sorted(NO_ZERO))}) are not zero-filled in primary.",
        md_table(m),
        "## Task status (eval errors are retried by the next `jev eval`)",
        md_table(status),
        "## Cost per system (USD at list price unless billed; generation unless judge; task cap "
        f"{cap_list})",
        md_table(per_system, 4),
        "## Cost per stage (USD; p50/p90 over tasks)",
        md_table(per_stage, 4),
        "## Report length (words, references excluded)",
        md_table(words, 0),
        "## Important citations by stage (DeepScholar, as reference recall matches them)",
        "Share found per task (macro, tasks with important citations) and pooled (micro); "
        "n_<stage> = papers per task at the stage. Screened and relevant are recorded by "
        "controlled systems only.",
        md_table(funnel_summary(stages)),
    ]
    out = run_dir / "report.md"
    out.write_text("\n\n".join(sections) + "\n")
    return out


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_no data_"
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    def fmt(v: Any) -> str:
        return "–" if pd.isna(v) else f"{v:.{digits}f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(fmt, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)

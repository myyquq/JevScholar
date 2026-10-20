"""Judge calibration on fixed reports, through the experiments' evaluation path (gateway, judge
routing, caps, ledger).

DeepResearch Bench II: the DeepSeek rubric judge scores the seven commercial systems' published
reports for the DRB2 dev tasks through exactly the evaluation path the experiments use (gateway,
judge profile, caps, ledger). Their per-system means are then ranked against the official
leaderboards: the current GPT-5.5 judge (5 of the 7 systems are listed) and the paper's
Gemini-2.5-Pro judge (all 7). With 12 tasks and 7 systems this is a coarse sanity check of the
judge, not a validation.

DeepScholar-Bench: the authors' own DeepScholar-base reports (the vendored parser fixtures: two
June-2025 rows, the paper-era pipeline with Llama-4-Scout) go through our evaluation, next to the
paper's Table 2 row for that system. It separates evaluation from generation in the
reproduction: our evaluator either reproduces the paper's numbers on the paper's reports or not.

DeepScholar evaluator controls: fixed reports built from a source system's dev reports bound what
each metric can say (its ceiling, what the input gives away, generic overlap, whether citation
identity matters).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jevscholar import benchmarks, data
from jevscholar.artifacts import experiment
from jevscholar.config import BenchCell, Budget, Experiment, load_models
from jevscholar.judging import EvalJob, digest, judge, judge_routing, judges_for, prepare
from jevscholar.paths import DATA_DERIVED, VENDOR
from jevscholar.report import md_table
from jevscholar.search.policy import arxiv_id
from jevscholar.types import EvalContext, Task, task_key
from jevscholar.util import atomic_write, read_json

REPORTS = DATA_DERIVED / "drb2" / "reports"
# Official totals over all tasks, HF Space muset-ai/DeepResearch-Bench-Leaderboard @1cf1c50:
# data_drb2/leaderboard.csv (GPT-5.5 judge) and data_drb2/leaderboard_gemini25.csv (paper judge).
LEADERBOARD = {  # report folder -> (GPT-5.5 total or None, Gemini-2.5-Pro total)
    "GPT-o3-DeepResearch": (43.00, 45.40),
    "Gemini3-Pro-DeepResearch": (41.40, 44.60),
    "Gemini2.5-Pro-DeepResearch": (39.98, 41.98),
    "Doubao-DeepResearch": (37.36, 40.99),
    "Qwen-3-Max-DeepResearch": (None, 39.25),
    "Grok-DeepSearch": (None, 39.23),
    "Perplexity-Research": (33.65, 38.58),
}


class Fixed:
    """Fixed reports of one benchmark to judge into a run directory of their own."""

    def __init__(self, run_dir: Path, bench: str, judgeset: str, systems: tuple[str, ...]):
        self.run_dir, self.bench, self.judgeset = run_dir, bench, judgeset
        self.models = load_models()
        self.judges = judges_for(self.models.judgesets[judgeset], bench)
        self.cell = BenchCell(split="dev", systems=systems)
        self.jobs: list[EvalJob] = []

    def add(
        self,
        system: str,
        task: Task,
        export: Path,
        scored: str | None = None,
        metrics: tuple[str, ...] = (),
    ) -> None:
        """Judge `task`'s report in `export` as `system`'s, unless judged already (`scored`: the
        report's identity, None for an immutable published one; `metrics`: only these)."""
        out_dir = self.run_dir / self.bench / system / "eval" / self.judgeset
        ctx = EvalContext(
            "", self.run_dir.name, system, self.judgeset, self.judges, out_dir, metrics
        )
        want = {m: v for m, v in benchmarks.get(self.bench).metrics.items() if m in metrics}
        if prepare(out_dir / f"{task_key(task)}.json", scored, want=want):
            self.jobs.append(EvalJob(self.bench, ctx, task, export, scored))

    def judge(self, name: str, judge_run_usd: float, workers: int) -> None:
        exp = Experiment(
            name=name,
            workers=workers,
            budget=Budget(judge_run_usd=judge_run_usd),
            benchmarks={self.bench: self.cell},
        )
        judge(self.run_dir, exp, self.jobs, self.judgeset, models=self.models, workers=workers)


def calibrate_drb2(
    run_dir: Path, *, judgeset: str = "deepseek", workers: int = 4
) -> dict[str, Any]:
    """Judge every reference report of the dev split, then write calibration.{json,md}."""
    systems = tuple(sorted(LEADERBOARD))
    fixed = Fixed(run_dir, "drb2", judgeset, systems)
    tasks = benchmarks.get("drb2").tasks("dev")
    for system in systems:
        for task in tasks:
            if any((REPORTS / system).glob(f"idx-{task.id}.*")):
                fixed.add(system, task, REPORTS / system)
    fixed.judge("drb2_calibration", 3.0, workers)
    return summarize(run_dir, systems, [task_key(t) for t in tasks], judgeset)


def summarize(run_dir: Path, systems: tuple[str, ...], keys: list[str], judgeset: str) -> dict:
    from scipy.stats import spearmanr

    rows = {}
    for system in systems:
        totals = []
        for key in keys:
            path = _eval_dir(run_dir, system, judgeset) / f"{key}.json"
            ev = read_json(path, {})
            if (total := (ev.get("metrics") or {}).get("total")) is not None:
                totals.append(total)
        rows[system] = {
            "n": len(totals),
            "ours": 100 * sum(totals) / len(totals) if totals else None,
            "gpt55": LEADERBOARD[system][0],
            "gemini25": LEADERBOARD[system][1],
        }
    out: dict[str, Any] = {"judgeset": judgeset, "tasks": keys, "systems": rows}
    for ref in ("gpt55", "gemini25"):
        pairs = [(r["ours"], r[ref]) for r in rows.values() if None not in (r["ours"], r[ref])]
        rho = spearmanr(*zip(*pairs, strict=True)).statistic if len(pairs) > 2 else None
        out[f"spearman_{ref}"] = {"n_systems": len(pairs), "rho": rho}
    atomic_write(run_dir / "calibration.json", json.dumps(out, indent=2))
    atomic_write(run_dir / "calibration.md", _markdown(out))
    return out


def _eval_dir(run_dir: Path, system: str, judgeset: str) -> Path:
    return run_dir / "drb2" / system / "eval" / judgeset


def _markdown(out: dict[str, Any]) -> str:
    table = pd.DataFrame.from_dict(out["systems"], orient="index").sort_values(
        "ours", ascending=False
    )
    table.index.name = "system"
    table.columns = ["n", "ours (dev tasks)", "GPT-5.5 board (all tasks)", "Gemini-2.5-Pro board"]
    lines = [
        f"# DRB2 judge calibration ({out['judgeset']}-judged, {len(out['tasks'])} dev tasks)",
        "",
        md_table(table, 2),
    ]
    for ref, name in (("gpt55", "GPT-5.5"), ("gemini25", "Gemini-2.5-Pro")):
        s = out[f"spearman_{ref}"]
        rho = "–" if s["rho"] is None else f"{s['rho']:.2f}"
        lines.append(f"\nSpearman rho vs {name} board: {rho} over {s['n_systems']} systems.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- DeepScholar
FIXTURES = VENDOR / "deepscholar/tests/baselines_results/deepscholar-base"  # <row>/intro.md
FIXTURE_SYSTEM = "upstream_fixture"
PAPER_LLAMA4 = {  # paper Table 2 (June 2025, gpt-4o / gpt-4.1 judges): DeepScholar-ref (Llama-4)
    "organization": 0.206,
    "nugget_coverage": 0.241,
    "relevance_rate": 0.436,
    "reference_coverage": 0.103,
    "document_importance": 0.008,
    "cite_p": 0.674,
    "claim_coverage": 0.851,
}


def calibrate_deepscholar(
    run_dir: Path, *, judgeset: str = "deepseek", workers: int = 0
) -> dict[str, Any]:
    """Judge the authors' reports, then write calibration.{json,md} next to the paper's row."""
    bench = benchmarks.get("deepscholar")
    rows = data.deepscholar_rows("jun2025")
    tasks = [
        bench._task("jun2025", int(d.name), rows[int(d.name)])
        for d in sorted(FIXTURES.iterdir())
        if (d / "intro.md").exists()
    ]
    fixed = Fixed(run_dir, "deepscholar", judgeset, (FIXTURE_SYSTEM,))
    for task in tasks:
        fixed.add(FIXTURE_SYSTEM, task, FIXTURES)
    fixed.judge("deepscholar_calibration", 0.5, workers)
    out_dir = run_dir / "deepscholar" / FIXTURE_SYSTEM / "eval" / judgeset
    per_task = {}
    for task in tasks:
        ev = read_json(out_dir / f"{task_key(task)}.json")
        per_task[task.id] = ev.get("metrics", {}) | bench.derived(ev)
    metrics = [*PAPER_LLAMA4, "cite_p_per_citation"]
    ours = {
        m: sum(v[m] for v in per_task.values()) / len(per_task)
        if all(v.get(m) is not None for v in per_task.values())
        else None
        for m in metrics
    }
    out = {"judgeset": judgeset, "tasks": per_task, "ours": ours, "paper_llama4": PAPER_LLAMA4}
    atomic_write(run_dir / "calibration.json", json.dumps(out, indent=2))
    table = pd.DataFrame(
        [
            {"": "ours on the authors' reports", "n": len(per_task), **ours},
            {"": "paper Table 2, DeepScholar-ref (Llama-4)", "n": 63}
            | {m: PAPER_LLAMA4.get(m) for m in metrics},
        ]
    )
    title = f"# DeepScholar evaluation on the authors' DeepScholar-base reports ({judgeset}-judged)"
    atomic_write(run_dir / "calibration.md", f"{title}\n\n{md_table(table)}\n")
    return out


# --------------------------------------------------------------------------- DeepScholar controls
CONTROLS = {  # name -> (fixed report, metrics that mean something for it)
    "reference": ("the paper's own related-work section", ("nugget_coverage",)),
    "abstract": ("the task's abstract", ("nugget_coverage",)),
    "swapped": ("the next dev task's DS-base report", ("nugget_coverage",)),
    "permuted": (  # same cited set and prose: nugget input unchanged (a judge replicate)
        "the DS-base report, each cited paper's links pointed at the next paper it cites",
        ("nugget_coverage", "cite_p", "claim_coverage", "claim_support", "reference_recall"),
    ),
    "gold_cited": (
        "one sentence citing each important citation",
        ("reference_coverage", "reference_recall"),
    ),
}


def calibrate_deepscholar_controls(
    run_dir: Path, source: Path, system: str, *, judgeset: str = "deepseek", workers: int = 4
) -> dict[str, Any]:
    """Evaluator controls: fixed reports for the dev tasks of `source`'s `system` (DS-base)
    bound what each metric can say: its ceiling (the reference section; every important
    citation cited), what the input gives away (the abstract), generic overlap (another task's
    report) and whether citation identity matters (permuted citations: the same prose and cited
    set, so nugget coverage is judged on the same input again and reference recall cannot move;
    citation support should fall). Each control's identity is its export, the gold data and the
    judges' routing. Writes calibration.{json,md} next to the source system."""
    bench = benchmarks.get("deepscholar")
    tasks = benchmarks.load_tasks(experiment(source), "deepscholar")
    write_controls(run_dir, tasks, source / "deepscholar" / system / "export")
    fixed = Fixed(run_dir, "deepscholar", judgeset, tuple(CONTROLS))
    routing = judge_routing(fixed.models, fixed.judges)
    for control, (_, metrics) in CONTROLS.items():
        cell = run_dir / "deepscholar" / control
        for task in tasks:
            files = sorted((cell / "export" / str(task.meta["row"])).iterdir())
            export = {f.name: f.read_text(encoding="utf-8") for f in files}
            gold = digest(bench.gold(task))
            scored = digest({"export": export, "judges": routing, "gold": gold})[:16]
            fixed.add(control, task, cell / "export", scored, metrics)
    fixed.judge("deepscholar_controls", 0.5, workers)
    cells = {c: run_dir / "deepscholar" / c for c in CONTROLS} | {
        f"{system} (source)": source / "deepscholar" / system
    }
    rows = {name: _means(cell / "eval" / judgeset, tasks) for name, cell in cells.items()}
    out = {"judgeset": judgeset, "source": f"{source.name}/{system}", "controls": rows}
    atomic_write(run_dir / "calibration.json", json.dumps(out, indent=2))
    atomic_write(run_dir / "calibration.md", _controls_markdown(out))
    return out


def write_controls(run_dir: Path, tasks: list[Task], source: Path) -> None:
    """Each control's export folders, <control>/export/<row>/{intro.md, paper.csv}, from the
    source system's exports of the same tasks (in the official deepscholar_base layout)."""
    from jevscholar.benchmarks.deepscholar import important_citations, papers

    rows = [str(t.meta["row"]) for t in tasks]
    empty = pd.DataFrame(columns=["id", "title", "snippet"])
    for i, task in enumerate(tasks):
        own, swapped = source / rows[i], source / rows[(i + 1) % len(rows)]
        text = (own / "intro.md").read_text(encoding="utf-8")
        gold = pd.DataFrame(
            [
                (pid, r["cited_paper_title"], r["cited_paper_abstract"])
                for r in important_citations(task.meta["dataset"]).get(task.id, [])
                if (pid := arxiv_id(r["cited_paper_arxiv_link"]))
            ],
            columns=["id", "title", "snippet"],
        )
        exports = {
            "reference": (
                papers(task.meta["dataset"])[task.meta["row"]]["clean_latex_related_works"],
                empty,
            ),
            "abstract": (task.meta["abstract"], empty),
            "swapped": (
                (swapped / "intro.md").read_text(encoding="utf-8"),
                pd.read_csv(swapped / "paper.csv", dtype=str, keep_default_na=False),
            ),
            "permuted": (
                rotate_citations(text),
                pd.read_csv(own / "paper.csv", dtype=str, keep_default_na=False),
            ),
            "gold_cited": (
                " ".join(
                    f"Prior work [{k}](http://arxiv.org/abs/{pid})."
                    for k, pid in enumerate(gold.id, 1)
                ),
                gold,
            ),
        }
        for control, (intro, paper) in exports.items():
            out = run_dir / "deepscholar" / control / "export" / rows[i]
            out.mkdir(parents=True, exist_ok=True)
            (out / "intro.md").write_text(intro, encoding="utf-8")
            paper.to_csv(out / "paper.csv", index=False)


def rotate_citations(text: str) -> str:
    """Each cited arXiv paper's links pointed at the next distinct paper the text cites."""
    from jevscholar.benchmarks.deepscholar_eval import ARXIV_ABS

    ids = list(dict.fromkeys(m[1] for m in ARXIV_ABS.finditer(text)))
    shift = dict(zip(ids, ids[1:] + ids[:1], strict=True))
    return ARXIV_ABS.sub(lambda m: f"arxiv.org/abs/{shift[m[1]]}", text)


def _means(eval_dir: Path, tasks: list[Task]) -> dict[str, Any]:
    """Mean of each scored metric over the tasks (n per metric), derived metrics included."""
    values: dict[str, list[float]] = {}
    bench = benchmarks.get("deepscholar")
    for task in tasks:
        ev = read_json(eval_dir / f"{task_key(task)}.json", {})
        for m, v in ((ev.get("metrics") or {}) | bench.derived(ev)).items():
            if v is not None and (ev.get("status") or {}).get(m) != "not_requested":
                values.setdefault(m, []).append(v)
    return {m: {"n": len(v), "mean": sum(v) / len(v)} for m, v in values.items()}


def _controls_markdown(out: dict[str, Any]) -> str:
    metrics = [
        "nugget_coverage", "nugget_vital", "nugget_okay", "cite_p", "cite_p_per_citation",
        "claim_coverage", "claim_support", "reference_coverage", "reference_recall",
    ]  # fmt: skip
    what = {name: text for name, (text, _) in CONTROLS.items()}

    def cell(r: dict[str, Any] | None) -> str | None:
        return r and f"{r['mean']:.3f} ({r['n']})"

    table = pd.DataFrame(
        [
            {"control": name, "report": what.get(name, "as generated")}
            | {m: cell(row.get(m)) for m in metrics}
            for name, row in out["controls"].items()
        ]
    )
    return (
        f"# DeepScholar evaluator controls ({out['judgeset']}-judged; source {out['source']})\n\n"
        "Mean (n tasks) of each metric judged for the control; the source row is the source "
        f"system's own reports as judged in its run.\n\n{md_table(table)}\n"
    )

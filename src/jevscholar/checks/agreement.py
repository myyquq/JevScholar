"""Official-judge agreement check: do the DeepSeek judges reproduce the official ones?

A prefix of a run's dev tasks is judged again with the official judgeset (gpt-4o and gpt-4.1 on
DeepScholar, gpt-4.1-mini on ResearchQA, GPT-5.5 on DRB2) into a run directory of its own, through
exactly the evaluation path of the experiments (gateway, caps, ledger), reading the source run's
exports. It is then compared with the source run's DeepSeek verdicts on the same exports:
- items: every decision both judges made (DeepScholar organization / cite_p / claim verdicts,
  ResearchQA rubric labels, DRB2 rubric scores), as agreement and Cohen's kappa, linearly weighted
  on the ordinal scales;
- tasks: every metric's per-task scores, and each system's paired difference vs the baseline under
  either judge: whether the mean reverses its sign (strictly: a 0 under one judge is no flip),
  and the share of tasks whose difference reverses.
Item kappa and per-system means use the reports both judges scored; paired differences score a
failed report 0, as the published deltas do. Claim coverage is compared per claim (see
review.per_claim), not per judge call. Decision rule (registered before the first check): a
benchmark with an item kappa below 0.6, or whose primary-metric mean difference flips sign, is
flagged (its DeepSeek-judged deltas are then unreliable, its test split needs official judging);
one without enough evidence (fewer than MIN_TASKS paired tasks per system or MIN_ITEMS items) is
"insufficient", never "consistent".
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jevscholar import benchmarks
from jevscholar.analysis import ledger_frame, primary
from jevscholar.artifacts import experiment, read_manifest
from jevscholar.checks.review import cohen_kappa, decisions, norm_label
from jevscholar.config import BenchCell, Budget, Experiment, load_models
from jevscholar.judging import eval_jobs, judge
from jevscholar.report import md_table
from jevscholar.types import task_key
from jevscholar.util import atomic_write, read_json

TASKS = {"deepscholar": 6, "researchqa": 40, "drb2": 6}  # leading dev tasks per benchmark
CAPS = Budget(judge_usd={"deepscholar": 1.0, "researchqa": 0.05, "drb2": 2.0}, judge_run_usd=10.0)
SCALES = {  # ordinal labels, low to high (kappa is then linearly weighted)
    "researchqa": ("Not at all", "Barely", "Moderately", "Mostly", "Completely"),
    "drb2": (-1, 0, 1),
}
KAPPA_MIN, MIN_ITEMS, MIN_TASKS = 0.6, 10, 5


def judge_subset(
    source: Path,
    run_dir: Path,
    *,
    systems: tuple[str, ...] = (),
    judgeset: str = "official",
    tasks: dict[str, int] | None = None,
    workers: int | None = None,
) -> str | None:
    """Judge the first `tasks[bench]` tasks of `source`'s task list (its split prefix, or its
    `ids`) and its `systems` (default: every system; the first is the paired comparisons'
    baseline) into run_dir.

    A re-run never shrinks a prefix or drops a system: what is judged (and paid for) stays."""
    if run_dir.resolve() == source.resolve():
        raise ValueError("the agreement run must not be its own source run")
    tasks = dict(tasks or TASKS)
    if prev := read_manifest(run_dir).get("experiment"):
        for bench, cell in Experiment.model_validate(prev).benchmarks.items():
            checked = len(cell.ids) if cell.ids is not None else cell.limit or 0
            tasks[bench] = max(tasks.get(bench, 0), checked)
            systems = (*systems, *(s for s in cell.systems if s not in systems))
    exp = _experiment(source, systems, tasks, workers)
    models = load_models()
    jobs = eval_jobs(run_dir, exp, models, judgeset, source=source)
    return judge(
        run_dir, exp, jobs, judgeset, models=models, workers=workers,
        record={"source_run": source.name},
    )  # fmt: skip


def _experiment(
    source: Path, systems: tuple[str, ...], tasks: dict[str, int], workers: int | None
) -> Experiment:
    src = experiment(source)
    cells = {}
    for bench, n in tasks.items():
        if bench not in src.benchmarks:
            continue
        cell = src.benchmarks[bench]
        first = benchmarks.load_tasks(src, bench)[:n]  # the source run's own leading tasks
        cells[bench] = BenchCell(
            split=cell.split,
            limit=cell.limit,
            ids=tuple(t.id for t in first),
            systems=tuple(s for s in systems or cell.systems if s in cell.systems),
        )
    return Experiment(
        name="judge_agreement",
        executor=src.executor,
        search=src.search,
        seed=src.seed,  # task keys carry it
        budget=CAPS,
        workers=src.workers if workers is None else workers,
        benchmarks=cells,
    )


# --------------------------------------------------------------------------- comparison
def summarize(
    source: Path,
    run_dir: Path,
    judgeset: str,
    reference: str = "deepseek",
    *,
    stopped: str | None = None,
) -> dict:
    """Compare run_dir's `judgeset` verdicts with source's `reference` ones; write agreement.*."""
    exp = experiment(run_dir)
    ledger = ledger_frame(run_dir)
    out: dict[str, Any] = {
        "source": source.name,
        "judgeset": judgeset,
        "reference": reference,
        "stopped": stopped,
        "benchmarks": {},
    }
    for bench, cell in exp.benchmarks.items():
        items: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
        scores: defaultdict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
        failed: defaultdict[str, set[str]] = defaultdict(set)  # no report: 0 under every judge
        unmatched = 0
        for system in cell.systems:
            ref_dir = source / bench / system / "eval" / reference
            new_dir = run_dir / bench / system / "eval" / judgeset
            for task in benchmarks.load_tasks(exp, bench):
                key = task_key(task)
                a, b = _scored(ref_dir / f"{key}.json"), _scored(new_dir / f"{key}.json")
                if a is None or b is None:
                    continue
                if not a or not b:
                    if not a and not b:
                        failed[system].add(key)
                    continue
                for metric in a.keys() & b.keys():
                    if a[metric] is not None and b[metric] is not None:
                        scores[system, metric][key] = (a[metric], b[metric])
                pairs, missed = _match(
                    decisions(bench, ref_dir, task), decisions(bench, new_dir, task)
                )
                unmatched += missed
                for metric, x, y in pairs:
                    items[metric].append((x, y))
        billed = ledger[ledger.bench == bench].billed_usd.sum() if len(ledger) else 0.0
        out["benchmarks"][bench] = _compare(bench, cell.systems, items, scores, failed) | {
            "unmatched_items": unmatched,
            "billed_usd": float(billed),
        }
    atomic_write(run_dir / "agreement.json", json.dumps(out, indent=2))
    atomic_write(run_dir / "agreement.md", _markdown(out))
    return out


def _scored(path: Path) -> dict[str, float | None] | None:
    """An eval file's metrics ({} for a task without a report), or None if missing or errored."""
    ev = read_json(path, {})
    if ev.get("status") == "no_report":
        return {}
    return None if "error" in ev or not ev.get("metrics") else ev["metrics"]


def _match(ref: list[dict], new: list[dict]) -> tuple[list[tuple[str, str, str]], int]:
    """(metric, reference label, new label) of decisions on the same item, and the unmatched count.

    Items are keyed by (metric, context, occurrence), since a context may repeat in a report."""
    a, b = _keyed(ref), _keyed(new)
    pairs = [(k[0], a[k], b[k]) for k in a.keys() & b.keys()]
    return sorted(pairs), len(a.keys() ^ b.keys())


def _keyed(decs: list[dict]) -> dict[tuple[str, str, int], str]:
    out, seen = {}, Counter[tuple[str, str]]()
    for d in decs:
        k = (d["metric"], d["context"])
        out[(*k, seen[k])] = norm_label(d["decision"])
        seen[k] += 1
    return out


def _compare(
    bench: str, systems: tuple[str, ...], items: dict, scores: dict, failed: dict[str, set[str]]
) -> dict[str, Any]:
    scale = [norm_label(c) for c in SCALES.get(bench, ())]
    item_rows = {}
    for metric, pairs in sorted(items.items()):
        valid = [(x, y) for x, y in pairs if not scale or {x, y} <= set(scale)]
        xs, ys = [x for x, _ in valid], [y for _, y in valid]
        item_rows[metric] = {
            "n": len(valid),
            "invalid": len(pairs) - len(valid),
            "agreement": float(np.mean([x == y for x, y in valid])) if valid else None,
            "kappa": cohen_kappa(xs, ys, scale) if valid else None,
        }
    metric_rows = {}
    for (system, metric), per_task in sorted(scores.items()):
        a, b = np.array(list(per_task.values())).T
        metric_rows[f"{system}/{metric}"] = {
            "n": len(a),
            "reference": float(a.mean()),
            "new": float(b.mean()),
            "pearson": float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() * b.std() else None,
        }
    paired = {}
    base = systems[0]
    for system in systems[1:]:
        for metric in sorted({m for s, m in scores if s == system}):
            mine, theirs = scores[system, metric], scores.get((base, metric), {})
            mine = mine | dict.fromkeys(failed.get(system, ()), (0.0, 0.0))  # primary scoring
            keys = sorted(mine.keys() & theirs.keys())
            if not keys:
                continue
            d = np.array([np.subtract(mine[k], theirs[k]) for k in keys])  # columns: ref, new
            paired[f"{system}/{metric}"] = {
                "n": len(keys),
                "diff_reference": float(d[:, 0].mean()),
                "diff_new": float(d[:, 1].mean()),
                "flip": bool(d[:, 0].mean() * d[:, 1].mean() < 0),
                "task_flips": float(np.mean(d[:, 0] * d[:, 1] < 0)),
            }
    low = [m for m, r in item_rows.items() if r["n"] >= MIN_ITEMS and _below(r["kappa"])]
    flips = [k for k, r in paired.items() if k.endswith(f"/{primary(bench)}") and r["flip"]]
    n_paired = [paired.get(f"{s}/{primary(bench)}", {}).get("n", 0) for s in systems[1:]]
    enough = min(n_paired, default=0) >= MIN_TASKS and any(
        r["n"] >= MIN_ITEMS for r in item_rows.values()
    )
    return {
        "items": item_rows,
        "metrics": metric_rows,
        "paired": paired,
        "baseline": base,
        "low_kappa": low,
        "sign_flips": flips,
        "verdict": "flagged" if low or flips else "consistent" if enough else "insufficient",
    }


def _below(kappa: float | None) -> bool:
    return kappa is None or math.isnan(kappa) or kappa < KAPPA_MIN


def _markdown(out: dict[str, Any]) -> str:
    ref, new = out["reference"], out["judgeset"]
    lines = [f"# Judge agreement: {new} vs {ref} on {out['source']}", ""]
    if out["stopped"]:
        lines += [f"**Judging stopped early ({out['stopped']}); the comparison is partial.**", ""]
    for bench, r in out["benchmarks"].items():
        items, metrics, paired = (
            pd.DataFrame.from_dict(r[k], orient="index") for k in ("items", "metrics", "paired")
        )
        items.index.name, metrics.index.name = "item metric", "system/metric"
        metrics = metrics.rename(columns={"reference": ref, "new": new})
        paired.index.name = f"system/metric vs {r['baseline']}"
        if len(paired):
            paired["flip"] = paired.flip.map({True: "FLIP", False: "no"})
        paired = paired.rename(
            columns={
                "diff_reference": f"Δ {ref}",
                "diff_new": f"Δ {new}",
                "flip": "mean flips",
                "task_flips": "tasks flipped",
            }
        )
        lines += [
            f"## {bench}: {r['verdict'].upper()} (billed ${r['billed_usd']:.2f})",
            "",
            f"Low kappa (< {KAPPA_MIN}): {', '.join(r['low_kappa']) or 'none'}. "
            f"Primary-metric sign flips vs {r['baseline']}: "
            f"{', '.join(r['sign_flips']) or 'none'}. "
            f"Unmatched items: {r['unmatched_items']}.",
            "",
            md_table(items),
            "",
            md_table(metrics),
            "",
            md_table(paired),
            "",
        ]
    return "\n".join(lines)

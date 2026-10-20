"""Judging: exported tasks -> eval/<judgeset>/<task_key>.json, through the runner's sessions.

- Tasks without a usable report are not judged (they score 0 anyway); errored evals are retried.
- Each eval file carries the fingerprint of what it scored: the exported report, the judges'
  routing and the task's gold data. One whose input has changed since is deleted and the task
  judged afresh, so no evaluator ever reuses verdicts on another input.
- Metrics judged one by one also carry their protocol version: a file lacking a requested metric
  at its current version is judged again for that metric only (evaluators reuse the rest).
- Verdicts can be reused across runs: a task whose fingerprint matches an eval file of another
  run (`reuse`) starts from a copy of it, since that file scored exactly the same input.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jevscholar import benchmarks
from jevscholar.artifacts import experiment, read_manifest, read_result, task_dir
from jevscholar.benchmarks import exported_report
from jevscholar.config import Experiment, ModelsConfig, load_models
from jevscholar.paths import CACHE
from jevscholar.runner import execute, selected
from jevscholar.types import (
    BENCHMARKS,
    Benchmark,
    EvalContext,
    Report,
    Task,
    load_class,
    task_key,
    to_json,
)
from jevscholar.util import atomic_write, read_json

log = logging.getLogger("jevscholar.judging")


@dataclass(frozen=True)
class EvalJob:
    bench: str
    ctx: EvalContext  # its gateway is bound in the worker
    task: Task
    export: Path
    fingerprint: str | None = None  # of what is scored; None for a fixed input


def eval_run(
    run_dir: Path,
    judgeset: str = "deepseek",
    *,
    only: list[str] | None = None,
    workers: int | None = None,
    secrets: dict[str, str] | None = None,
    models: ModelsConfig | None = None,
    cache_dir: Path = CACHE,
    redo: tuple[str, ...] = (),
    metrics: tuple[str, ...] = (),
    reuse: tuple[Path, ...] = (),
) -> str | None:
    """Judge every exported task of the run with `judgeset` (`metrics`: only those; `redo`:
    those again; `reuse`: earlier runs whose verdicts on identical inputs are copied).
    Returns the reason dispatch stopped early, else None."""
    exp = experiment(run_dir)
    models = models or load_models()
    jobs = eval_jobs(
        run_dir, exp, models, judgeset, only=only, redo=redo, metrics=metrics, reuse=reuse
    )
    return judge(
        run_dir, exp, jobs, judgeset, models=models, workers=workers, secrets=secrets,
        cache_dir=cache_dir, only=only,
    )  # fmt: skip


def judge(
    run_dir: Path,
    exp: Experiment,
    jobs: list[EvalJob],
    judgeset: str,
    *,
    models: ModelsConfig | None = None,
    workers: int | None = None,
    secrets: dict[str, str] | None = None,
    cache_dir: Path = CACHE,
    record: dict[str, Any] | None = None,
    **session: Any,
) -> str | None:
    """Judge `jobs` in one eval session of run_dir, whose manifest records `exp` (and `record`).

    The one way to judge: `eval_run` for a run's own exports; the checks (calibration, agreement)
    for published reports or another run's exports. Returns the reason judging stopped early."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(run_dir) | {
        "run_id": run_dir.name,
        **(record or {}),
        "experiment": exp.model_dump(mode="json"),
    }
    return execute(
        run_dir, manifest, "eval", exp, models or load_models(), eval_task, [jobs],
        workers=workers, secrets=secrets, cache_dir=cache_dir, judgeset=judgeset, **session,
    )  # fmt: skip


def eval_jobs(
    run_dir: Path,
    exp: Experiment,
    models: ModelsConfig,
    judgeset: str,
    *,
    source: Path | None = None,
    only: list[str] | None = None,
    redo: tuple[str, ...] = (),
    metrics: tuple[str, ...] = (),
    reuse: tuple[Path, ...] = (),
) -> list[EvalJob]:
    """The tasks of `exp` to judge into run_dir/<bench>/<system>/eval/<judgeset>/.

    Reports are read from the `source` run (default: run_dir itself); a task without a usable
    report is recorded as "no_report" without a judge call. `metrics` restricts judging to those
    metrics, which every selected benchmark must judge separately (Benchmark.metrics); without
    them, each benchmark judges its default metrics (Benchmark.judged)."""
    if judgeset not in models.judgesets:
        raise ValueError(
            f"unknown judgeset {judgeset!r}; models.yaml has {sorted(models.judgesets)}"
        )
    source = source or run_dir
    judged = verdicts(reuse, judgeset)
    jobs = []
    for name in selected(exp, only):
        bench = benchmarks.get(name)
        if bad := set(metrics) - set(bench.metrics):
            raise ValueError(f"{name} cannot judge {sorted(bad)} on their own")
        asked = metrics or bench.judged
        if asked and (extra := [m for m in redo if m in bench.metrics and m not in asked]):
            if metrics:
                raise ValueError(f"--redo {extra} must be among the judged metrics {metrics}")
            asked = (*asked, *extra)  # a metric judged again is requested, never just cleared
        judges = judges_for(models.judgesets[judgeset], name)
        routing = judge_routing(models, judges)
        want = {m: v for m, v in bench.metrics.items() if not asked or m in asked}
        tasks = benchmarks.load_tasks(exp, name)
        gold = {task.id: digest(bench.gold(task)) for task in tasks}
        for system in exp.benchmarks[name].systems:
            out_dir = run_dir / name / system / "eval" / judgeset
            ctx = EvalContext("", run_dir.name, system, judgeset, judges, out_dir, asked)
            for task in tasks:
                out = out_dir / f"{task_key(task)}.json"
                result = read_result(task_dir(source, name, system, task))
                report, _ = exported_report(task, result)
                scored = fingerprint(report, routing, gold[task.id])
                if scored in judged and not usable(read_json(out, None), scored):
                    reuse_verdicts(judged[scored], out, bench, task)
                if not prepare(out, scored, redo, want):
                    continue
                if report is None:
                    no_report = {"metrics": {}, "status": "no_report", "fingerprint": scored}
                    atomic_write(out, json.dumps(no_report))
                else:
                    export = source / name / system / "export"
                    jobs.append(EvalJob(name, ctx, task, export, scored))
    log.info("%d tasks to judge with %s", len(jobs), judgeset)
    return jobs


def verdicts(runs: tuple[Path, ...], judgeset: str) -> dict[str, Path]:
    """Fingerprint -> an eval file of `runs` that scored it (the inputs of reports judged)."""
    out = {}
    for run in runs:
        for path in sorted(run.glob(f"*/*/eval/{judgeset}/*.json")):
            ev = read_json(path)
            if (scored := ev.get("fingerprint")) and "error" not in ev:
                out.setdefault(scored, path)
    return out


def usable(ev: dict[str, Any] | None, scored: str) -> bool:
    """Whether an eval file holds verdicts on this input (complete or not)."""
    return ev is not None and ev.get("fingerprint") == scored and "error" not in ev


def reuse_verdicts(donor: Path, out: Path, bench: Benchmark, task: Task) -> None:
    """Start `out` from another run's eval file on the same input, its judge decisions carried
    in `items` (some evaluators keep them in raw logs next to the file)."""
    ev = read_json(donor)
    ev["items"] = bench.items(ev, donor.parent, task)
    atomic_write(out, to_json(ev, indent=2))


def judges_for(judgeset: dict[str, str], bench: str) -> dict[str, str]:
    """{'default': alias, metric: alias} from judgeset keys '<bench>' and '<bench>.<metric>'."""
    prefix = f"{bench}."
    per_metric = {k.removeprefix(prefix): v for k, v in judgeset.items() if k.startswith(prefix)}
    return {"default": judgeset[bench], **per_metric}


def judge_routing(models: ModelsConfig, judges: dict[str, str]) -> dict[str, Any]:
    """What shapes each judge's requests: its alias's model settings and its upstream's endpoint
    and dialect (names, prices and the never-used judge cache flag left out)."""
    out = {}
    for key, name in judges.items():
        alias = models.aliases[name]
        upstream = models.upstreams[alias.upstream]
        out[key] = alias.model_dump(mode="json", exclude={"upstream", "price", "cache"}) | {
            "url": upstream.base_url,
            "dialect": upstream.dialect,
        }
    return out


def fingerprint(report: Report | None, judges: dict[str, Any], gold: str | None = None) -> str:
    """What an eval file scores: the exported report (its text and sources), the judges'
    routing and the `digest` of the task's gold data. Which metrics it holds is recorded per
    metric instead."""
    content = report and dataclasses.asdict(dataclasses.replace(report, meta={}))
    return digest({"report": content, "judges": judges, "gold": gold})[:16]


def digest(obj: Any) -> str:
    return hashlib.sha256(to_json(obj, sort_keys=True).encode()).hexdigest()


def needs_judging(
    ev: dict[str, Any] | None,
    scored: str | None = None,
    redo: tuple[str, ...] = (),
    want: dict[str, int] | None = None,
) -> bool:
    """Whether a task's eval file (None: there is none yet) must be judged: it scored another
    input than `scored` (the report was generated again, the judges or the gold data changed);
    it is to judge `redo` metrics again; it records an error; or it has not settled a `want`ed
    metric at its protocol version. `scored` is None for a fixed input (published reports)."""
    if ev is None or ev.get("fingerprint") != scored:
        return True
    status = ev.get("status")
    if not isinstance(status, dict):  # one status for the task ("no_report"), or none (an error)
        return "error" in ev
    versions = ev.get("versions", {})
    settled = {m: versions.get(m) for m, s in status.items() if s != "not_requested"}
    stale = [m for m, v in (want or {}).items() if settled.get(m) != v]
    return "error" in ev or any(m in status for m in redo) or bool(stale)


def prepare(
    out: Path,
    scored: str | None = None,
    redo: tuple[str, ...] = (),
    want: dict[str, int] | None = None,
) -> bool:
    """`needs_judging` for the eval file at `out`, which is readied for the judge: deleted if it
    scored another input (so no evaluator reuses its verdicts), its `redo` statuses cleared
    (evaluators then judge those metrics again)."""
    ev = read_json(out, None)
    if not needs_judging(ev, scored, redo, want):
        return False
    if ev is None:
        return True
    if ev.get("fingerprint") != scored:
        out.unlink()
        return True
    status = ev.get("status")
    if isinstance(status, dict) and (cleared := [m for m in redo if m in status]):
        ev["status"] = {k: v for k, v in status.items() if k not in cleared}
        atomic_write(out, json.dumps(ev, indent=2, ensure_ascii=False))
    return True


# --------------------------------------------------------------------------- worker side
_SET_UP: dict[tuple[str, str], Any] = {}  # (benchmark, judgeset) -> set-up instance


def eval_task(gateway: str, job: EvalJob) -> None:
    ctx, key = dataclasses.replace(job.ctx, gateway=gateway), task_key(job.task)
    if (bench := _SET_UP.get((job.bench, ctx.judgeset))) is None:
        bench = _SET_UP[job.bench, ctx.judgeset] = load_class(BENCHMARKS[job.bench])()
        bench.setup(ctx)
    t0 = time.monotonic()
    try:
        out = bench.evaluate_task(job.task, job.export, ctx)
    except Exception:
        out = {"metrics": {}, "error": traceback.format_exc()}
    elapsed = round(time.monotonic() - t0, 3)
    out |= {"task": key, "fingerprint": job.fingerprint, "elapsed_s": elapsed}
    atomic_write(ctx.out_dir / f"{key}.json", to_json(out, indent=2))
    log.info(
        "judged %s/%s/%s%s", job.task.bench, ctx.system, key, " (error)" if "error" in out else ""
    )

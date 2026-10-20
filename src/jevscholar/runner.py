"""Experiment runner: cells -> pending tasks -> spawn pool, with resume, provenance and spend gates.

Why it looks like this:
- Workers are spawned processes running one task at a time (LOTUS keeps process-global settings)
  with a scrubbed environment: no LLM provider keys, so a code path that bypasses the gateway
  fails loudly instead of running unmetered.
- Systems of one benchmark run strictly one after another: executor replay-cache keys exclude the
  system, so a trajectory-fixed ablation must start after the system whose trajectory it replays.
- A task is done once its result.json exists; a degraded result is re-queued once (the first
  attempt is kept as result.degraded.json). A result produced under a different cell hash (which
  covers the benchmark version, the executor's routing and the task cap) is refused, never mixed
  in silently; failures stay on disk (and in the denominator).
- The same session machinery (`execute`) serves judging (judging.py). Generation sessions stop on
  the run budget, eval sessions on the judging budget, both on the campaign budget and the
  circuit breaker.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import logging
import multiprocessing as mp
import os
import subprocess
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jevscholar import benchmarks
from jevscholar.artifacts import experiment, read_manifest, read_result, task_dir
from jevscholar.config import Experiment, ModelsConfig, Search, load_models
from jevscholar.gateway import Gateway, load_secrets
from jevscholar.ledger import is_peak
from jevscholar.paths import CACHE, CATALOG, CONFIGS, ROOT, VENDOR
from jevscholar.types import (
    LOCAL_KEY,
    SYSTEMS,
    Report,
    Status,
    Task,
    TaskContext,
    TaskResult,
    load_class,
    make_tag,
    task_key,
    to_json,
)
from jevscholar.util import atomic_write, sha256_file

log = logging.getLogger("jevscholar.runner")
PEAK_POLL_S = 60
DEGRADED = "result.degraded.json"  # first attempt of a re-queued degraded task
FORWARDED = {"JEV_TAVILY_KEY": "TAVILY_API_KEY", "JEV_OPENALEX_KEY": "OPENALEX_API_KEY"}


# --------------------------------------------------------------------------- planning
@dataclass(frozen=True)
class RunJob:
    run_dir: Path
    system: str  # the experiment's system name (tags, paths)
    use: str  # its system class key
    params: str  # its parameters (JSON)
    executor: str
    search: Search
    cap_usd: float
    cell_hash: str
    task: Task


def plan(
    exp: Experiment, run_dir: Path, models: ModelsConfig, only: list[str] | None = None
) -> tuple[dict[str, Any], dict[str, list[str]], list[list[RunJob]]]:
    """(manifest cells, manifest tasks, job batches): every cell's identity and pending tasks,
    one batch per system. Every system is constructed once here, so a bad parameter fails before
    any task is dispatched."""
    alias = models.aliases[exp.executor]
    upstream = models.upstreams[alias.upstream]
    routing = alias.model_dump(mode="json", exclude={"price"}) | {
        "url": upstream.base_url,
        "dialect": upstream.dialect,
    }
    cells, universe, batches = {}, {}, []
    for name in selected(exp, only):
        tasks = benchmarks.load_tasks(exp, name)
        universe[name] = [task_key(t) for t in tasks]
        version = benchmarks.get(name).version
        cells[name] = {}
        for system in exp.benchmarks[name].systems:
            cap = benchmarks.task_cap(exp, name, system)
            identity: dict[str, Any] = {"version": version, "routing": routing, "cap_usd": cap}
            use, params = exp.system_spec(system)
            identity["resolved"] = make_system(use, json.dumps(params)).identity()
            cell = exp.cell_hash(name, system, **identity)
            cells[name][system] = {"hash": cell, **identity}
            todo = pending(run_dir, name, system, tasks, cell)
            log.info("%s/%s: %d pending tasks", name, system, len(todo))
            job = functools.partial(
                RunJob, run_dir=run_dir, system=system, use=use, params=json.dumps(params),
                executor=exp.executor, search=exp.search, cap_usd=cap, cell_hash=cell,
            )  # fmt: skip
            batches.append([job(task=t) for t in todo])
    return cells, universe, batches


def selected(exp: Experiment, only: list[str] | None) -> list[str]:
    if unknown := set(only or ()) - set(exp.benchmarks):
        raise ValueError(
            f"unknown benchmarks {sorted(unknown)}; experiment has {sorted(exp.benchmarks)}"
        )
    return [b for b in exp.benchmarks if not only or b in only]


def pending(run_dir: Path, bench: str, system: str, tasks: list[Task], want: str) -> list[Task]:
    """Tasks without a result or with a not yet retried degraded one; another cell's is an error."""
    todo = []
    for task in tasks:
        path = task_dir(run_dir, bench, system, task) / "result.json"
        if not path.exists():
            todo.append(task)
            continue
        result = json.loads(path.read_text())
        if result["cell_hash"] != want:
            raise RuntimeError(
                f"{path} was produced by cell {result['cell_hash']}, but {bench}/{system} now "
                f"hashes to {want}: the configuration changed, start a new run id instead"
            )
        if result["status"] == "degraded" and not path.with_name(DEGRADED).exists():
            todo.append(task)
    return todo


@functools.cache
def make_system(use: str, params: str) -> Any:
    return load_class(SYSTEMS[use])(json.loads(params))


# --------------------------------------------------------------------------- run / export
def run_experiment(
    exp: Experiment,
    run_dir: Path,
    *,
    only: list[str] | None = None,
    workers: int | None = None,
    secrets: dict[str, str] | None = None,
    models: ModelsConfig | None = None,
    cache_dir: Path = CACHE,
) -> str | None:
    """Run every pending task of the experiment into run_dir (created or resumed).

    Returns the reason dispatch stopped early (breaker, run or campaign budget), else None.
    """
    models = models or load_models()
    cells, universe, batches = plan(exp, run_dir, models, only)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(run_dir) | {
        "run_id": run_dir.name,
        "experiment": exp.model_dump(mode="json"),
    }
    manifest.setdefault("cells", {}).update(cells)
    manifest.setdefault("tasks", {}).update(universe)
    return execute(
        run_dir, manifest, "run", exp, models, run_task, batches,
        workers=workers, secrets=secrets, cache_dir=cache_dir, only=only,
    )  # fmt: skip


def export_run(run_dir: Path, *, only: list[str] | None = None) -> None:
    """Write each task's official layout under <bench>/<system>/export/ (benchmark-defined).

    Every task of the split prefix is exported; missing, failed or report-less results export an
    empty report. export/audit.json records per task the cited sources the access policy dropped.
    """
    exp = experiment(run_dir)
    for name in selected(exp, only):
        bench = benchmarks.get(name)
        tasks = benchmarks.load_tasks(exp, name)
        for system in exp.benchmarks[name].systems:
            out = run_dir / name / system / "export"
            out.mkdir(parents=True, exist_ok=True)
            dropped = {
                task_key(t): bench.export_task(
                    t, read_result(task_dir(run_dir, name, system, t)), out
                )
                for t in tasks
            }
            atomic_write(out / "audit.json", json.dumps({"dropped_sources": dropped}, indent=2))


# --------------------------------------------------------------------------- sessions
def execute(
    run_dir: Path,
    manifest: dict[str, Any],
    command: str,
    exp: Experiment,
    models: ModelsConfig,
    fn: Callable[[str, Any], Any],
    batches: Iterable[list[Any]],
    *,
    workers: int | None,
    secrets: dict[str, str] | None,
    cache_dir: Path,
    **record: Any,
) -> str | None:
    """One session: a gateway and a pool; each batch drains before the next one starts.
    Returns the reason dispatch stopped early, else None."""
    workers = exp.workers if workers is None else workers
    secrets = load_secrets() if secrets is None else secrets
    judging = command == "eval"
    caps = {} if judging else benchmarks.task_caps(exp)
    with session(run_dir, manifest, command, workers=workers, **record) as current:
        gw = Gateway(
            models, run_dir, exp.budget, caps, secrets=secrets, cache_dir=cache_dir,
            replay_only=exp.replay_only,
        )  # fmt: skip
        url = gw.start()
        try:
            gate = make_gate(gw, exp, models, judging=judging)
            with pool(workers, worker_env(secrets, exp.search.arxiv)) as executor:
                for jobs in batches:
                    if reason := dispatch(
                        executor, functools.partial(fn, url), jobs, max(workers, 1), gate
                    ):
                        log.warning("stopped dispatching: %s", reason)
                        current["stopped"] = reason
                        break
        finally:
            gw.stop()
    return current["stopped"]


@contextmanager
def session(
    run_dir: Path, manifest: dict[str, Any], command: str, **record: Any
) -> Iterator[dict[str, Any]]:
    """Append one invocation (provenance, start/end, outcome) to manifest.json; a session run
    from a working tree with uncommitted changes also saves them as sessions/<n>.diff."""
    current = {"command": command, "started": _now(), **record, **provenance(), "stopped": None}
    sessions = manifest.setdefault("sessions", [])
    sessions.append(current)
    if current["git_dirty"]:
        atomic_write(run_dir / "sessions" / f"{len(sessions)}.diff", _git("diff", "HEAD") or "")
    atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    try:
        yield current
    except BaseException as e:
        current["error"] = repr(e)
        raise
    finally:
        current["ended"] = _now()
        atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))


def provenance() -> dict[str, Any]:
    """Code and config identity: git sha + dirty flag (third_party/ is committed), lock, routing."""
    status = _git("status", "--porcelain")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "models_yaml_sha256": sha256_file(CONFIGS / "models.yaml"),
        "systems_yaml_sha256": sha256_file(CATALOG),
        "third_party": {
            p.name: _git("rev-parse", f"HEAD:third_party/{p.name}")
            for p in sorted(VENDOR.iterdir())
            if p.is_dir()
        },
    }


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- pool and dispatch
def worker_env(secrets: dict[str, str], arxiv: str = "local") -> dict[str, str]:
    """The parent's environment minus provider keys, plus gateway-only client settings and the
    keys of the tools that call their APIs directly (Tavily, OpenAlex)."""
    litellm = Path(importlib.util.find_spec("litellm").origin).parent  # located, not imported
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env |= {
        "OPENAI_API_KEY": LOCAL_KEY,
        "OPEN_AI_API_KEY": LOCAL_KEY,
        "LITELLM_MODE": "PRODUCTION",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "OPENAI_AGENTS_DISABLE_TRACING": "1",
        "NLTK_DATA": str(ROOT / ".cache/nltk_data"),
        # litellm's bundled encodings (cl100k_base for the nuggetizer): tiktoken never downloads
        "TIKTOKEN_CACHE_DIR": str(litellm / "litellm_core_utils" / "tokenizers"),
        "JEV_ARXIV_BACKEND": arxiv,  # the experiment's arXiv backend, for every system alike
    }
    return env | {name: secrets[key] for name, key in FORWARDED.items() if secrets.get(key)}


def _init_worker(env: dict[str, str]) -> None:
    import urllib3.util.connection

    os.environ.clear()
    os.environ.update(env)
    # requests has no Happy Eyeballs: a stalled IPv6 path (PMTU black hole to OpenAlex's
    # Cloudflare edge on the dev host) costs a full read timeout per call. Every API we call
    # serves IPv4; the gateway's aiohttp client falls back on its own.
    urllib3.util.connection.HAS_IPV6 = False
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(processName)s %(name)s %(levelname)s %(message)s"
    )


class _Inline(Executor):
    """workers=0: run each job in this process at submit time (tests, debugging)."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)
        return future


@contextmanager
def pool(workers: int, env: dict[str, str]) -> Iterator[Executor]:
    if workers > 0:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            workers, mp_context=ctx, initializer=_init_worker, initargs=(env,)
        ) as executor:
            yield executor
        return
    saved = dict(os.environ)
    _init_worker(env)
    try:
        yield _Inline()
    finally:
        os.environ.clear()
        os.environ.update(saved)


def dispatch(
    executor: Executor,
    fn: Callable[[Any], Any],
    jobs: Iterable[Any],
    slots: int,
    gate: Callable[[], str | None],
) -> str | None:
    """Submit jobs keeping at most `slots` in flight; returns the gate's stop reason, if any."""
    running: set[Future] = set()
    reason = None
    for job in jobs:
        while len(running) >= slots:
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                f.result()
        if reason := gate():
            break
        running.add(executor.submit(fn, job))
    for f in wait(running).done:
        f.result()
    return reason


def make_gate(
    gw: Gateway, exp: Experiment, models: ModelsConfig, *, judging: bool
) -> Callable[[], str | None]:
    """Before each dispatch: hold during DeepSeek peak hours, stop on breaker or spent budgets.

    Generation sessions answer to the run budget, eval sessions to the judging budget."""
    own = (
        ("judge", "judge_billed_usd", exp.budget.judge_run_usd)
        if judging
        else ("run", "run_billed_usd", exp.budget.run_usd)
    )

    def gate() -> str | None:
        if exp.offpeak_only and is_peak(datetime.now(UTC)):
            log.info("DeepSeek peak hours: holding new tasks until off-peak")
            while is_peak(datetime.now(UTC)):
                time.sleep(PEAK_POLL_S)
        st = gw.status()
        if st["tripped"]:
            return f"circuit breaker tripped: {st['tripped']}"
        for what, spent, cap in (own, ("campaign", "campaign_billed_usd", models.campaign_usd)):
            if st[spent] >= cap:
                return f"{what} budget spent (${st[spent]:.2f} billed)"
        return None

    return gate


# --------------------------------------------------------------------------- worker side
def run_task(gateway: str, job: RunJob) -> Status | None:
    """Run one task and write its result.json (None: left pending after a run-level stop)."""
    from jevscholar.search import BudgetExhausted
    from jevscholar.search.client import SearchClient

    task, run = job.task, job.run_dir.name
    where = f"{task.bench}/{job.system}/{task_key(task)}"
    tag = make_tag(run, task.bench, job.system, task_key(task), "tools")
    ctx = TaskContext(
        gateway=gateway,
        run=run,
        system=job.system,
        task=task,
        executor=job.executor,
        search=SearchClient(gateway, tag, task.policy),
        search_config=job.search,
        cap_usd=job.cap_usd,
        out_dir=task_dir(job.run_dir, task.bench, job.system, task),
    )
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    report, error, exhausted = None, None, False
    t0 = time.monotonic()
    try:
        report = asyncio.run(make_system(job.use, job.params).run(ctx))
    except Exception as e:
        error = traceback.format_exc()
        exhausted = isinstance(e, BudgetExhausted)
    elapsed = time.monotonic() - t0
    usage = ctx.usage()
    if usage["latched"] and usage["spent_usd"] < usage["task_cap"]:
        log.warning("%s: breaker or run/campaign budget hit, task left pending", where)
        return None
    status = derive_status(report, error, exhausted or (error is not None and usage["latched"]))
    result = TaskResult(
        task_id=task.id,
        system=job.system,
        status=status,
        report=report,
        usage={**usage, "search": ctx.search.stats},
        error=error,
        elapsed_s=round(elapsed, 3),
        cell_hash=job.cell_hash,
    )
    out = ctx.out_dir / "result.json"
    if out.exists():  # the degraded first attempt of a re-queued task
        out.replace(out.with_name(DEGRADED))
    atomic_write(out, to_json(result, indent=2))
    log.info("%s: %s in %.0fs, $%.4f", where, status, elapsed, usage["spent_usd"])
    return status


def derive_status(report: Report | None, error: str | None, exhausted: bool) -> Status:
    if exhausted:
        return "budget_exhausted"
    if error is not None or report is None:
        return "failed"
    status = report.meta.get("status")
    return status if status in ("partial", "degraded") else "ok"

"""Core data types, tag grammar, and the System / Benchmark base classes.

Everything that crosses a module boundary lives here so that systems, benchmarks, the runner and
the gateway agree on one vocabulary.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import quote, unquote

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from jevscholar.config import Search
    from jevscholar.search.client import Meter, SearchClient

Status = Literal["ok", "partial", "degraded", "failed", "budget_exhausted"]
LOCAL_KEY = "jev-local"  # dummy API key every client sends to the gateway
CLIENT_TIMEOUT_S = 1300  # longer than the gateway's upstream timeout plus its internal retries
JUDGE_PREFIX = "judge."  # stage prefix of every judge call (own budget key, never cached)


class Strict(BaseModel):
    """A validated settings block: unknown keys are errors, values are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- tags
def make_tag(run: str, bench: str, system: str, task: str, stage: str) -> str:
    """`run:bench:system:task:stage`, each part URL-quoted (so ':' and '/' never leak)."""
    return ":".join(quote(p, safe="@.-_") for p in (run, bench, system, task, stage))


def parse_tag(tag: str) -> tuple[str, str, str, str, str]:
    parts = [unquote(p) for p in tag.split(":")]
    if len(parts) != 5:
        raise ValueError(f"malformed tag {tag!r}")
    return tuple(parts)  # type: ignore[return-value]


def task_key(task: Task) -> str:
    """Task component of a tag; the seed is part of the identity (cache namespace)."""
    return task.id if task.seed == 0 else f"{task.id}@{task.seed}"


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class AccessPolicy:
    """What a task's tools may return. Enforced by the SearchClient for every system alike."""

    cutoff: datetime | None = None  # tz-aware UTC; results must be published strictly before it
    blocked_arxiv: frozenset[str] = frozenset()  # versionless arXiv ids (e.g. the target paper)
    blocked_titles: tuple[str, ...] = ()  # normalised titles; also blocks queries containing them
    blocked_urls: frozenset[str] = frozenset()  # canonical URLs
    undated_web: Literal["drop", "keep"] = "keep"


@dataclass(frozen=True)
class Task:
    bench: str
    id: str  # benchmark-native id; must not contain ':'
    prompt: str  # exactly what every system receives
    policy: AccessPolicy = AccessPolicy()
    seed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Source:
    id: str  # versioned arXiv id for arXiv sources, canonical URL otherwise
    url: str
    title: str = ""
    authors: tuple[str, ...] = ()
    date: str = ""  # YYYY-MM-DD or ""
    snippet: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Report:
    text: str  # markdown body; citations are "[n]" referring to sources[n-1]
    sources: tuple[Source, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    system: str
    status: Status
    report: Report | None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_s: float = 0.0
    cell_hash: str = ""


# --------------------------------------------------------------------------- (de)serialisation
def _default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, frozenset | set):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def to_json(obj: Any, **kw: Any) -> str:
    data = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj
    return json.dumps(data, default=_default, ensure_ascii=False, **kw)


def report_from_dict(d: dict[str, Any] | None) -> Report | None:
    if d is None:
        return None
    sources = tuple(Source(**{**s, "authors": tuple(s.get("authors", ()))}) for s in d["sources"])
    return Report(text=d["text"], sources=sources, meta=d.get("meta", {}))


def load_result(path: Path) -> TaskResult:
    d = json.loads(path.read_text())
    return TaskResult(**{**d, "report": report_from_dict(d.get("report"))})


# --------------------------------------------------------------------------- contexts
@dataclass
class TaskContext:
    """Everything a System may use while running one task."""

    gateway: str  # e.g. http://127.0.0.1:43123
    run: str
    system: str
    task: Task
    executor: str  # model alias resolved by the gateway
    search: SearchClient  # the task's tools
    search_config: Search  # the experiment's shared search settings (corpora, k per query)
    cap_usd: float  # the task's list-price cap, enforced by the gateway
    out_dir: Path
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("jevscholar"))

    def tag(self, stage: str) -> str:
        return make_tag(self.run, self.task.bench, self.system, task_key(self.task), stage)

    def base_url(self, stage: str) -> str:
        return f"{self.gateway}/t/{self.tag(stage)}/v1"

    def llm(self, stage: str) -> AsyncOpenAI:
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            base_url=self.base_url(stage),
            api_key=LOCAL_KEY,
            max_retries=0,
            timeout=CLIENT_TIMEOUT_S,
        )

    def usage(self) -> dict[str, Any]:
        """The gateway's spend view of this (run, bench, system, task) budget key."""
        import requests

        r = requests.get(f"{self.gateway}/t/{self.tag('usage')}/usage", timeout=30)
        r.raise_for_status()
        return r.json()


@dataclass
class EvalContext:
    """Everything a Benchmark evaluator may use while judging one exported task."""

    gateway: str
    run: str
    system: str
    judgeset: str  # e.g. "deepseek" | "official"
    judges: dict[str, str]  # {"default": alias, <metric>: alias} (runner.judges_for)
    out_dir: Path  # eval/<judgeset>/ of this (bench, system) cell
    metrics: tuple[str, ...] = ()  # judge only these (Benchmark.metrics); empty: every metric
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("jevscholar"))

    def judge(self, metric: str) -> str:
        """Model alias judging `metric` (a per-metric override, else the benchmark default)."""
        return self.judges.get(metric, self.judges["default"])

    def tag(self, task: Task, metric: str) -> str:
        return make_tag(self.run, task.bench, self.system, task_key(task), JUDGE_PREFIX + metric)

    def base_url(self, task: Task, metric: str) -> str:
        return f"{self.gateway}/t/{self.tag(task, metric)}/v1"

    def meter(self, task: Task, metric: str) -> Meter:
        """The tool account of `metric`'s judging (e.g. Document Importance's OpenAlex lookups)."""
        from jevscholar.search.client import Meter

        return Meter(self.gateway, self.tag(task, metric))


# --------------------------------------------------------------------------- base classes
class System:
    """A research system: validated parameters (its `Params`), then one `run` per task."""

    version: ClassVar[str] = "1"  # bumped when the same parameters come to mean other behaviour

    class Params(Strict):
        pass

    def __init__(self, params: dict[str, Any]) -> None:
        self.p = self.Params.model_validate(params)

    def identity(self) -> dict[str, Any]:
        """What the system does, defaults resolved: part of every cell's identity."""
        return {"version": self.version, "params": self.p.model_dump(mode="json")}

    async def run(self, ctx: TaskContext) -> Report:
        raise NotImplementedError


class Benchmark:
    """A benchmark: its tasks, its official export layout and its scoring of one export."""

    name: ClassVar[str]
    version: ClassVar[str] = "1"  # part of every cell's identity: bump when tasks change
    primary: ClassVar[str]  # the pre-declared primary metric
    reliability: ClassVar[tuple[str, ...]] = ()  # metrics a quality gain must not cost
    secondary: ClassVar[tuple[str, ...]] = ()  # further paired metrics
    # metrics judged separately (a run may judge a subset) -> protocol version, bumped when a
    # metric's judging changes so that eval files re-judge that metric only
    metrics: ClassVar[dict[str, int]] = {}
    judged: ClassVar[tuple[str, ...]] = ()  # the metrics `jev eval` judges by default (): all
    derived_from: ClassVar[dict[str, str]] = {}  # derived metric -> the metric it is computed from
    cap_usd: ClassVar[float | None] = None  # B0: the per-task cap (list USD) of comparisons

    def tasks(self, split: str, limit: int | None = None) -> list[Task]:
        raise NotImplementedError

    def gold(self, task: Task) -> Any:
        """The task's reference data the evaluator scores against (JSON-serializable): part of
        an eval file's fingerprint, so a changed gold set is judged again."""
        raise NotImplementedError

    def undefined(self, task: Task) -> set[str]:
        """Metrics that do not exist for the task (left out of its means and comparisons)."""
        return set()

    def derived(self, ev: dict[str, Any]) -> dict[str, float | None]:
        """Metrics recomputed from an eval file's saved judge items."""
        return {}

    def export_task(self, task: Task, result: TaskResult | None, out: Path) -> int:
        """Write the official per-task layout under `out` (empty report if result is None).

        Returns how many cited sources the task's access policy removed before rendering."""
        raise NotImplementedError

    def exported_text(self, task: Task, export: Path) -> str:
        """The text the judge scored: the task's official export."""
        raise NotImplementedError

    def setup(self, ctx: EvalContext) -> None:
        """Once per judging worker, before its first task."""

    def evaluate_task(self, task: Task, export: Path, ctx: EvalContext) -> dict[str, Any]:
        """Score one exported task: {"metrics": {...}, "status": {...}[, "items": [...]]}.

        Infrastructure failures (gateway refusals, timeouts, upstream errors) must raise, so the
        runner records a top-level "error" and the next `jev eval` retries the task."""
        raise NotImplementedError

    def items(self, ev: dict[str, Any], eval_dir: Path, task: Task) -> list[dict[str, Any]]:
        """A judged task's decisions, {"metric", "decision", "context"} (review, agreement)."""
        return ev.get("items") or []


SYSTEMS = {
    "simple_rag": "jevscholar.systems.simple_rag:SimpleRAG",
    "closed_book": "jevscholar.systems.closed_book:ClosedBook",
    "deepscholar_base": "jevscholar.systems.deepscholar_base:DeepScholarBase",
    "fixed_evidence": "jevscholar.systems.fixed_evidence:FixedEvidence",
}
BENCHMARKS = {
    "deepscholar": "jevscholar.benchmarks.deepscholar:DeepScholarBench",
    "researchqa": "jevscholar.benchmarks.researchqa:ResearchQA",
    "drb2": "jevscholar.benchmarks.drb2:DRB2",
}


def load_class(path: str) -> type:
    """Resolve 'module:Class' lazily so heavy dependencies load only where needed."""
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)

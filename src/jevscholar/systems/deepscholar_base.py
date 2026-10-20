"""DeepScholar-base (vendored @c95413b, repo-default agentic config) with only the model swapped.

The vendored tree stays pristine; every fix is a runtime patch installed once per process
(`lotus.settings` is process-global, so a worker runs one task at a time). The patches never
capture task state: they read the module-level CURRENT holder, which `run` sets and clears
around each task, so no search client (cutoff, blocklist, ledger tag) or repair flag can leak
from one task into the next.

P1  LOTUS web_search / web_extract -> SearchClient facades (cached, metered, policy-filtered);
    arXiv reads serve title + abstract from metadata, never PDFs, and a multi-id read is first
    warmed by one batched lookup (upstream then reads the ids one by one from the cache).
P2  input_sanitization -> full brace doubling (the upstream regex misses nested/unbalanced).
P3  filter: escape the topic (upstream formats it raw into the sem_filter/sem_topk prompt).
P4  category summary: escape the topic (same bug).
P5  AgentContext.papers_df starts as an empty frame (upstream `len(None)` hides tool errors).
P6  repair_merge: accumulate search rows across queries, never read rows (9a37581 kept only
    the first query's results); repair_merge=False reproduces upstream as-is.
P7  openai-agents tracing off.  P8  LOTUS cache off (the gateway owns caching).
P9  np.random.seed(task.seed) per task (sem_topk quicksort pivots).
P10 repair_citations (opt-in, so earlier cells keep their meaning). The intro writer must cite
    each document by its `_citation_number` column, but upstream escapes the citation guidelines
    with the user text, so the column never reaches the prompt and the writer numbers citations
    itself, or copies the search background's numbers (which index the search step's own list;
    agentic: the agent's draft); upstream then resolves them against the filtered list, so most
    citations land on the wrong papers. The repair leaves the guidelines unescaped and strips
    numeric citations from the background.
Ablation `search_background=False`: the intro writer gets an empty background instead of the
search agent's draft (the controlled loop's writer gets none either).
Configuration fixes: a fresh Configs per task with all four stage LMs explicit (upstream drops
max_tokens to 512 when deriving them), no SDK retries and the shared client timeout on every LM
client, and the experiment's shared search settings (web corpus on/off, k per query). A system's
`config` param overrides other YAML fields, e.g. the paper's protocol (DeepScholar-ref, paper
A.2.4: recursive search, 2 rounds of at most 2 queries). Tool calls and errors are counted by the
task's SearchClient (`stats`), which also decides "degraded".
"""

from __future__ import annotations

import functools
import importlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import Field

from jevscholar.paths import VENDOR
from jevscholar.render import ARXIV_URL, CITE, finalize, source_id
from jevscholar.types import CLIENT_TIMEOUT_S, Report, Source, Strict, System, TaskContext
from jevscholar.util import vendored

if TYPE_CHECKING:
    from jevscholar.search.client import SearchClient

YAML = VENDOR / "deepscholar/configs/deepscholar_base.yaml"
STAGES = ("search", "filter", "taxonomize", "generation")
NON_CONFIG = {"queries_file", "output_folder", "start_idx", "end_idx", "lm", "search_mode"}
SHARED = {"enable_web_search", "per_query_max_search_results_count"}  # set by the experiment
FRAME_COLS = ["title", "url", "snippet", "query", "context", "date"]  # upstream empty frame
ARXIV_COLS = ["id", "title", "link", "abstract", "published", "authors", "categories"]
TAVILY_COLS = ["title", "url", "content"]
READ_QUERIES = {"arxiv_read", "tavily_read"}  # `query` value of rows made by read tools
STALE_CITE = re.compile(rf"\s*(?:{CITE.pattern}|【\d+】)")  # P10: "[3]", "[1, 2]", "【3】"
DEGRADED_MIN_CALLS, DEGRADED_ERROR_RATE = 5, 0.2


# --------------------------------------------------------------------------- per-task state
@dataclass
class Current:
    search: SearchClient
    repair_merge: bool
    repair_citations: bool = False
    search_background: bool = True


CURRENT: Current | None = None


def current() -> Current:
    if CURRENT is None:
        raise RuntimeError("DeepScholar-base patch used outside a task")
    return CURRENT


@contextmanager
def task_scope(
    search: SearchClient,
    seed: int,
    *,
    repair_merge: bool = True,
    repair_citations: bool = False,
    search_background: bool = True,
) -> Iterator[Current]:
    """The patches' view of one task: its search client and the repairs it runs with."""
    global CURRENT
    CURRENT = Current(search, repair_merge, repair_citations, search_background)
    np.random.seed(seed)  # P9
    try:
        yield CURRENT
    finally:
        CURRENT = None


# --------------------------------------------------------------------------- P1 facades
# Both run on the current task's client. Errors propagate: DeepScholar turns them into "Error
# searching ..." / "Error extracting content" tool output, as it does for LOTUS's own failures.
def web_search(
    corpus: Any,
    query: str,
    K: int,
    cols: Any = None,
    sort_by_date: bool = False,
    start_date: Any = None,
    end_date: Any = None,
) -> pd.DataFrame:
    """LOTUS `web_search` contract. Dates are ignored: the task's AccessPolicy bounds results."""
    search = current().search
    if corpus.value == "arxiv":
        return arxiv_frame(search.arxiv_search(query, K))
    hits = search.web_search(query, K)
    return pd.DataFrame([{c: h[c] for c in TAVILY_COLS} for h in hits], columns=TAVILY_COLS)


def arxiv_frame(papers: list[dict[str, Any]]) -> pd.DataFrame:
    """arXiv papers (the search backend's dicts) as LOTUS's arXiv search results."""
    df = pd.DataFrame(
        [
            {
                "id": p["id"],
                "title": p["title"],
                "link": f"http://arxiv.org/abs/{p['id']}",
                "abstract": p["abstract"],
                "published": p["published"],
                "authors": ", ".join(p["authors"]),
                "categories": ", ".join(p["categories"]),
            }
            for p in papers
        ],
        columns=ARXIV_COLS,
    )
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df


def web_extract(
    corpus: Any, doc_id: str | None = None, url: str | None = None, max_length: int | None = None
) -> pd.DataFrame:
    """LOTUS `web_extract` contract: one row {id, url, full_text}; full_text None if unavailable."""
    search, key = current().search, (doc_id or url or "").strip()
    if corpus.value == "arxiv":
        paper = search.arxiv_lookup([key]).get(key)  # keyed by the requested id
        text = paper and f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        row = {"id": key, "url": f"https://arxiv.org/abs/{key}", "full_text": text}
    else:
        row = {"id": key, "url": key, "full_text": search.web_read(key)}
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- P2-P4 braces
def esc(s: str) -> str:
    """Double every brace: exactly one later str.format (LOTUS nle2str) restores the text."""
    return s.replace("{", "{{").replace("}", "}}")


def sanitize(s: str) -> str:
    """Drop-in for upstream input_sanitization (which also strips)."""
    return esc(s.strip())


# --------------------------------------------------------------------------- install
@functools.cache
def install() -> ModuleType:
    """Apply P1-P8 and P10 once per process; returns the module holding the pipeline function."""
    vendored("deepscholar")
    import agents
    import lotus
    from openai import AsyncOpenAI

    # NB: the package attribute `deepscholar_base.deepscholar_base` is the function, not the
    # module, so modules are fetched with import_module.
    top = importlib.import_module("deepscholar_base.deepscholar_base")
    ag = importlib.import_module("deepscholar_base.search.agentic_search")
    rs = importlib.import_module("deepscholar_base.search.recursive_search")
    sg = importlib.import_module("deepscholar_base.utils.summary_generation")
    tx = importlib.import_module("deepscholar_base.taxonomize")

    ag.web_search = rs.web_search = web_search
    ag.web_extract = web_extract
    read0 = ag._read_content

    async def read_content(ctx: Any, tool_type: Any, keys: list[str]) -> str:
        if tool_type is ag.ToolTypes.ARXIV and len(keys) > 1:
            with suppress(Exception):  # a failure resurfaces in the per-id reads, as upstream
                current().search.arxiv_lookup([k.strip() for k in keys])
        return await read0(ctx, tool_type, keys)

    ag._read_content = read_content
    guidelines = sg.citation_guidelines

    def sanitize_summary(s: str) -> str:
        if s is guidelines and CURRENT is not None and CURRENT.repair_citations:  # P10
            return s.strip()  # keeps the {_citation_number} column placeholder live
        return sanitize(s)

    sg.input_sanitization, tx.input_sanitization = sanitize_summary, sanitize
    filter0, summary0 = top.filter, tx.generate_category_summary_with_citations
    top.filter = lambda configs, docs_df, topic: filter0(configs, docs_df, esc(topic))
    tx.generate_category_summary_with_citations = lambda docs_df, topic, lm=None: summary0(
        docs_df, esc(topic), lm
    )
    agents.set_tracing_disabled(True)
    lotus.settings.configure(enable_cache=False)
    ag.AsyncOpenAI = functools.partial(AsyncOpenAI, max_retries=0, timeout=CLIENT_TIMEOUT_S)

    @dataclass
    class AgentContext(ag.AgentContext):
        def __post_init__(self) -> None:
            if self.papers_df is None:
                self.papers_df = pd.DataFrame(columns=FRAME_COLS)

        def merge_papers_df(self, new: pd.DataFrame) -> None:
            if not current().repair_merge:
                super().merge_papers_df(new)
            elif not new.empty and new["query"].iloc[0] not in READ_QUERIES:
                self.papers_df = (
                    pd.concat([self.papers_df, new], ignore_index=True).drop_duplicates("url")
                    if len(self.papers_df)
                    else new.copy()
                )

    ag.AgentContext = AgentContext
    intro0 = top.generate_intro_section

    async def intro(topic: str, docs_df: Any, background: str, configs: Any) -> str:
        if not current().search_background:  # the ablation
            background = ""
        elif current().repair_citations:  # P10
            background = STALE_CITE.sub("", background)
        return await intro0(topic, docs_df, background, configs)

    top.generate_intro_section = intro
    return top


def vendored_yaml(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The vendored default config with a system's overrides of its tunable fields."""
    y = yaml.safe_load(YAML.read_text())
    overrides = overrides or {}
    if bad := set(overrides) - ((set(y) - NON_CONFIG - SHARED) | {"search_mode"}):
        raise ValueError(f"not overridable DeepScholar-base settings: {sorted(bad)}")
    if overrides.get("search_mode", "agentic") not in ("agentic", "recursive"):
        raise ValueError(f"search_mode is 'agentic' or 'recursive': {overrides['search_mode']}")
    return y | overrides


def build_configs(
    ctx: TaskContext, overrides: dict[str, Any] | None = None, sample: int = 0
) -> Any:
    """A fresh Configs from the vendored YAML and `overrides`: four explicit stage LMs (never
    `lm`) and the experiment's shared search settings. `sample` > 0 gives the writer's stage its
    own replay namespace (`generation~<sample>`): an independent writer sample over the same
    search and screening."""
    from deepscholar_base.configs import Configs
    from lotus.models import LM

    search = ctx.search_config
    if "arxiv" not in search.corpora:
        raise ValueError(f"DeepScholar-base always searches arXiv, got corpora {search.corpora}")
    y = vendored_yaml(overrides)
    lms = {
        f"{s}_lm": LM(
            model=ctx.executor,
            temperature=y["lm"]["temperature"],
            max_tokens=y["lm"]["max_tokens"],
            custom_llm_provider="openai",
            api_base=ctx.base_url(f"{s}~{sample}" if sample and s == "generation" else s),
            timeout=CLIENT_TIMEOUT_S,
            max_retries=0,
        )
        for s in STAGES
    }
    fields = {k: v for k, v in y.items() if k not in NON_CONFIG} | {
        "use_agentic_search": y.get("search_mode") != "recursive",
        "enable_web_search": "web" in search.corpora,
        "per_query_max_search_results_count": search.k,
    }
    return Configs(**fields, **lms, logger=ctx.log)


# --------------------------------------------------------------------------- native -> Report
HREF = r"https?://(?:[^()\s]|\([^()\s]*\))+"  # balanced parentheses, e.g. Wikipedia titles
TOKEN = re.compile(
    rf"\\\[\[(?P<a>[^\]]+?)\]\((?P<u>{HREF})\)\\\]"  # upstream citation \[[label](url)\]
    rf"|\[(?P<b>[^\]]+?)\]\((?P<v>{HREF})\)"  # plain [label](url): copied links, tables
    rf"|(?P<s>{CITE.pattern})"  # "[1, 2]" / "[1-3]" left unresolved upstream
)


def to_report(text: str, docs: pd.DataFrame | None, **meta: Any) -> Report:
    """Lossless parse of DeepScholar markdown against docs_df into a Report.

    Links become `[n]` by first appearance, deduplicated by source id (the parser dedupes by
    versioned arXiv id) and joined to docs_df by URL or id; stray bracketed numbers are escaped
    so they cannot be read as citations; `#category` anchor links stay text. docs_df's URLs,
    in order, are kept in meta["docs"] (the writer's input), the uncited ones in meta["pool"].
    """
    rows = {} if docs is None else {r["url"]: r for r in docs.fillna("").to_dict("records")}
    by_id = {str(r["id"]): r for r in rows.values() if r.get("id")}
    sources: list[Source] = []
    index: dict[str, int] = {}

    def sub(m: re.Match[str]) -> str:
        if m["s"] is not None:
            return rf"\[{m['s'][1:-1]}\]"
        url, label = m["u"] or m["v"], m["a"] or m["b"]
        sid = source_id(url)
        if sid not in index:
            row = rows.get(url) or by_id.get(sid) or {}
            sources.append(
                Source(
                    id=sid,
                    url=url,
                    title=str(row.get("title", "")),
                    authors=tuple(
                        a.strip() for a in str(row.get("authors", "")).split(",") if a.strip()
                    ),
                    date=str(row.get("date", ""))[:10],
                    snippet=str(row.get("snippet", "")),
                    meta={
                        "corpus": "arxiv" if ARXIV_URL.search(url) else "web",
                        "label": label,
                        "native_id": str(row.get("id", "")),
                        "in_pool": bool(row),
                        "bracketed": m["a"] is not None,
                    },
                )
            )
            index[sid] = len(sources)
        return f"[{index[sid]}]"

    body = TOKEN.sub(sub, text)
    pool = [u for u in rows if source_id(u) not in index]
    return Report(
        text=body, sources=tuple(sources), meta={**meta, "docs": list(rows), "pool": pool}
    )


def persist(out: Path, final_report: str, docs: pd.DataFrame | None, stats: dict) -> None:
    """Native outputs in the upstream main.py layout (+ stats.json even on failure)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_report.md").write_text(final_report)
    if stats.get("intro_section") is not None:
        (out / "intro.md").write_text(stats["intro_section"])
    if docs is not None:
        docs.to_csv(out / "paper.csv", index=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str))


# --------------------------------------------------------------------------- system
class DeepScholarBase(System):
    class Params(Strict):
        repair_merge: bool = True  # P6
        repair_citations: bool = False  # P10
        search_background: bool = True  # False: the intro writer gets no search draft
        config: dict[str, Any] = Field(default_factory=dict)  # vendored YAML overrides
        deliverable: Literal["intro", "final_report"] = "intro"
        sample: int = 0  # > 0: an independent writer sample (build_configs)

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        vendored_yaml(self.p.config)  # fail at construction, not on the first task

    async def run(self, ctx: TaskContext) -> Report:
        top, task, p = install(), ctx.task, self.p
        configs = build_configs(ctx, p.config, p.sample)
        cutoff = task.policy.cutoff
        with task_scope(
            ctx.search,
            task.seed,
            repair_merge=p.repair_merge,
            repair_citations=p.repair_citations,
            search_background=p.search_background,
        ):
            final_report, docs, stats = await top.deepscholar_base(
                topic=task.prompt,
                end_date=pd.Timestamp(cutoff) if cutoff else None,
                configs=configs,
            )
        native = ctx.out_dir / "native"
        persist(native, final_report, docs, stats)
        intro = stats.get("intro_section")
        if not intro:
            raise RuntimeError(f"DeepScholar-base produced no intro: {stats.get('error')}")

        use_final = p.deliverable == "final_report" and bool(final_report)
        tries = [v for k, v in stats.items() if k.startswith("search_try_")]
        tools = ctx.search.stats
        degraded = (
            tools["calls"] >= DEGRADED_MIN_CALLS
            and tools["errors"] / tools["calls"] > DEGRADED_ERROR_RATE
        )
        status = "degraded" if degraded else "partial" if "error" in stats else "ok"
        report = to_report(final_report if use_final else intro, docs)
        return finalize(
            report.text,
            report.sources,
            **report.meta,
            status=status,
            deliverable="final_report" if use_final else "intro",
            native=str(native),
            search_try_errors=[t["error"] for t in tries if "error" in t],
            pool_size=len(tries[-1].get("pre_filter_search_results", [])) if tries else 0,
            error=stats.get("error"),
        )

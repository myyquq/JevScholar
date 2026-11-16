"""The controlled research loop: DeepScholar-base's search rows, relevance filter and intro writer,
with which search to run next, or whether to stop, decided by a policy (controller/policy.py).

At each retrieval decision the candidates are the next batch of the abstract search (the target
paper's abstract as one BM25 query, read PAGE new papers at a time down its ranked list), one
keyword query the executor proposes, the next new page of the most productive keyword query
(`deepen`), and `finish`; their pages are fetched first (local arXiv, at its nominal tool price).
A chosen search screens its new papers with DS-base's CoT sem_filter, whose requests are DS-base's
own for the same paper and task, so a paired DS-base run's cached verdicts replay. After `finish`
the relevant papers are ranked with DS-base's sem_topk when there are more than WRITE_DOCS
(seeded per task, so every branch from one state ranks alike), and DS-base's intro writer writes
the section from them with the citation repair (P10) and no background. The hard rules
(controller/state.py) keep the candidates a policy may choose; a single legal one is executed as
forced. `actions` removes searches from the offer (an ablation).

The task's spend is the gateway's list-price view (`ctx.usage()`), re-read after every charged
operation: proposals, prefetches, policy calls, screening, ranking and writing all count against
the task's cap. A step's preparation is bounded (one proposed query's first page, at most
DEEPEN_FETCHES new pages for `deepen`), so the rules' step overhead covers it. Each decision is
appended to decisions.jsonl once executed: the snapshot the policy saw, the requested and the
executed candidate, the policy's probabilities and input, and the spend around it. The search
rows and fetched pages the snapshots refer to are in rows.jsonl and pages.jsonl, the final state
in state.json. A run can continue from a decision point (`start`) of an earlier run's task: it
does not prepare that step again and re-reads the pages fetched before for free, and the spend up
to that point is reported as `offset_usd`. With `take` it executes that candidate there instead
of asking the policy (a counterfactual branch, controller/branches.py), logged as `taken`.
`sample` > 0 gives the writer its own replay namespace: an independent sample from the same
evidence. With `write: false` the episode ends once the writer's documents are chosen (ranking
included): a retrieval-only run, whose label needs those documents but no section; its empty
report is never judged.

On a research-report task (DRB2: `meta["request"]`, the prompt without its forbidden-source rule)
the abstract search is a prompt search with the request as its query, and the proposer and d1
read report wording. With the web among the corpora, the prompt search's list is first one
Tavily search for REQUEST_WEB_K hits (the query cut to Tavily's WEB_QUERY_CHARS), then its arXiv
ranked list; a proposed keyword query's first page comes from the web while the task has web
operations left (`search.web_ops`), else from arXiv; `deepen` reads arXiv only and passes over a
query whose first page came from the web. Web hits are screened from their snippets, and their
pages, kept apart from arXiv's, are not logged in pages.jsonl (a report task's decision point is
not restored with them).
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, SerializerFunctionWrapHandler, model_serializer

from jevscholar import paths
from jevscholar.artifacts import task_dir
from jevscholar.controller.policy import D1Policy, Decision, Policy, make_policy
from jevscholar.controller.prompt import label
from jevscholar.controller.state import (
    ABSTRACT_K,
    FINISH,
    MAX_BATCHES,
    MAX_PAGES,
    PAGE,
    SEARCHES,
    WRITE_DOCS,
    Candidate,
    Costs,
    Kind,
    Pages,
    Paper,
    Point,
    Query,
    State,
    affordable,
    legal,
    restore,
    room,
    snapshot,
)
from jevscholar.render import finalize
from jevscholar.search import ToolError
from jevscholar.systems import chat
from jevscholar.systems import deepscholar_base as dsb
from jevscholar.types import Report, Strict, System, Task, TaskContext
from jevscholar.util import read_jsonl

DEEPEN_FETCHES = 2  # new pages the deepen scan may fetch per step (pages fetched before are free)
PROPOSE_TOKENS = 100  # one query: bounds a proposal's cost
REQUEST_WEB_K = 20  # web hits of a report task's prompt search
WEB_QUERY_CHARS = 400  # Tavily refuses longer queries
PROPOSE = """You are searching arXiv for the prior work needed to write the related-work section
of the paper described in the task below. Write one new search query (3-10 keywords) that would
find relevant papers the searches so far have missed. Output the query only, nothing else.

Task:
{topic}

Searches so far (new papers, relevant):
{searches}

Relevant papers found so far:
{titles}"""
REPORT_PROPOSE = (  # a report task's: its own first sentence, then PROPOSE's
    "You are searching for the sources needed to write the research report the task below asks "
    "for. " + PROPOSE.partition("in the task below. ")[2]
)
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


class Start(Strict):
    """A decision point of an earlier run to continue from: the first of the task's decision
    points in runs/<run>/<bench>/<system> after this many retrieval batches; with `take`, the
    candidate of that kind is executed there instead of asking the policy (an error where the
    rules do not offer it)."""

    run: str
    system: str
    batches: int = 0
    take: Kind | None = None

    @model_serializer(mode="wrap")
    def _dump(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Without `take`, the fields a start had before `take` existed: its cell identity (and
        so the runs made with it) stays the same."""
        out = handler(self)
        if self.take is None:
            out.pop("take", None)
        return out


class Controlled(System):
    version = "2"  # retrieval decisions only; d1's prompt as in controller/prompt.py

    class Params(Strict):
        policy: dict[str, Any]  # controller/policy.py: make_policy
        actions: tuple[Literal["abstract", "explore", "deepen"], ...] = SEARCHES  # searches offered
        max_batches: int = MAX_BATCHES
        budget_usd: float | None = None  # the cap the rules plan within (None: the task's cap)
        costs: dict[str, float] = Field(default_factory=dict)  # Costs overrides
        start: Start | None = None
        sample: int = 0  # > 0: an independent writer sample
        write: bool = True  # False: stop once the writer's documents are chosen

        @model_serializer(mode="wrap")
        def _dump(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
            """Without `write: false`, the fields the parameters had before `write` existed: the
            cell identity of every writing run stays the same."""
            out = handler(self)
            if self.write:
                out.pop("write", None)
            return out

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.costs = Costs(**self.p.costs)
        make_policy(self.p.policy)  # fail at construction, not on the first task

    def preflight(self) -> None:
        """A d1 policy's server must serve the policy's adapter."""
        policy = make_policy(self.p.policy)
        if isinstance(policy, D1Policy):
            policy.preflight()

    def identity(self) -> dict[str, Any]:
        """The policy with its defaults (d1's profile) resolved, as well as the parameters (where
        the d1 server listens is not part of it)."""
        policy = make_policy(self.p.policy).model_dump(mode="json", exclude={"url"})
        return super().identity() | {"policy": policy}

    async def run(self, ctx: TaskContext) -> Report:
        """Research and write the task, from the `start` point if there is one; the spend that
        point carries becomes the state's offset, and this cell's cap and batch limit apply."""
        top, task, p = dsb.install(), ctx.task, self.p
        budget = ctx.cap_usd if p.budget_usd is None else p.budget_usd
        take = None if p.start is None else p.start.take
        if p.start is None:
            request = task.meta.get("request")  # a research report's: searched as the abstract is
            abstract = request or task.meta.get("abstract", "")
            state = State(task.prompt, budget, abstract, p.max_batches, report=request is not None)
            first, pages = None, {}
        else:
            point = locate(task, p.start)
            state, first, pages = point.state, point.candidates, point.pages
            if state.topic != task.prompt:
                raise ValueError("the decision point belongs to another task")
            state.offset_usd = state.spent_usd
            state.budget_usd, state.max_batches = budget, p.max_batches
            record = {"offset_usd": state.offset_usd, "batches": state.batches}
            (ctx.out_dir / "start.json").write_text(json.dumps(record))  # kept if the run fails
        with dsb.task_scope(ctx.search, task.seed, repair_citations=True):
            loop = Loop(ctx, top, self.costs, p.actions, pages, p.sample)
            docs = await loop.run(state, make_policy(p.policy), first, take, p.write)
        (ctx.out_dir / "state.json").write_text(state.to_json())
        report = dsb.to_report(state.draft or "", docs)
        unwritten = {} if p.write else {"written": False}
        return finalize(
            report.text,
            report.sources,
            **report.meta,
            status="ok",
            batches=state.batches,
            decisions=state.decisions,
            queries=len(state.queries),
            screened=len(state.papers),
            relevant=len(state.relevant()),
            spent_usd=state.spent_usd,
            offset_usd=state.offset_usd,
            **unwritten,
        )


def taken_index(offered: list[Candidate], kind: str) -> int:
    """The position of the candidate of `kind` a branch executes among the legal ones."""
    kinds = [c.kind for c in offered]
    if kind not in kinds:
        raise ValueError(f"{kind!r} is not offered at the decision point (offered: {kinds})")
    return kinds.index(kind)


def locate(task: Task, start: Start) -> Point:
    """The decision point `start` names among the task's decision points in an earlier run."""
    folder = task_dir(paths.RUNS / start.run, task.bench, start.system, task)
    rows = {r["url"]: r for r in read_jsonl(folder / "rows.jsonl")}
    pages = read_jsonl(folder / "pages.jsonl")
    for record in read_jsonl(folder / "decisions.jsonl"):
        if record["state"]["batches"] == start.batches:
            return restore(record, rows, pages)
    raise ValueError(f"{folder}: no decision point after {start.batches} batches")


class Loop:
    """One task's research loop over DS-base's components; runs inside `dsb.task_scope`."""

    def __init__(
        self,
        ctx: TaskContext,
        top: Any,
        costs: Costs,
        actions: tuple[str, ...],
        pages: Pages,
        sample: int = 0,
    ) -> None:
        """`pages`: search pages fetched before (a restored decision point's), free to re-read."""
        self.ctx, self.top, self.costs, self.actions, self.sample = ctx, top, costs, actions, sample
        self.screen_cfg = dsb.build_configs(ctx, {"use_sem_topk": False})
        self.write_cfg = dsb.build_configs(ctx, {"use_sem_filter": False}, sample)
        self.agentic = importlib.import_module("deepscholar_base.search.agentic_search")
        self.arxiv, self.web = self.agentic.ToolTypes.ARXIV, self.agentic.ToolTypes.WEB
        self.hits: Pages = {}
        self.web_hits: Pages = {}  # a report task's web pages, apart from arXiv's
        self.rows: dict[str, dict[str, Any]] = {}  # URL -> search row, as logged in rows.jsonl
        out = ctx.out_dir
        self.decisions, self.rows_log, self.pages = (
            out / f for f in ("decisions.jsonl", "rows.jsonl", "pages.jsonl")
        )
        for path in (self.decisions, self.rows_log, self.pages):
            path.unlink(missing_ok=True)
        for (query, k), rows in pages.items():
            self.remember(query, k, rows)

    async def run(
        self,
        state: State,
        policy: Policy,
        first: list[Candidate] | None = None,
        take: str | None = None,
        write: bool = True,
    ) -> pd.DataFrame:
        """Decide and search until `finish`, then choose the writer's documents and write (not
        with `write` False); returns the documents. `first`: the opening decision's candidates,
        as a snapshot saved them, so that step is not prepared (nor paid for) again. `take`: the
        kind of candidate the opening decision executes, without asking the policy."""
        self.keep_rows([p.row for p in state.papers] + [r for c in first or () for r in c.rows])
        proposed = None if first is None else self.restored(state, first)
        while state.phase == "retrieval":
            self.sync(state)
            if proposed is None:
                proposed = await self.prepare(state)
                self.sync(state)
            offered = legal(state, proposed, self.costs)
            point = snapshot(state, offered, len(self.hits))  # what the policy sees
            before = state.spent_usd
            forced = len(offered) == 1  # the rules leave no choice
            taken = take is not None
            if taken:
                decision, take = Decision(taken_index(offered, take)), None
            elif forced:
                decision = Decision(0)
            else:
                decision = await policy.choose(self.ctx, state, offered)
            self.sync(state)  # the decision may have been charged
            control = state.spent_usd - before  # the policy's own calls
            choice = offered[decision.index]
            overridden = (
                choice is not offered[-1]
                and offered[-1].kind == FINISH
                and not affordable(state, choice, self.costs)
            )
            if overridden:  # the decision's own cost left no room for the search: finish
                choice = offered[-1]
            state.decisions += 1
            await self.act(state, choice)
            self.sync(state)
            executed = offered.index(choice)
            self.record(point, decision, executed, forced, overridden, control, state, taken)
            proposed = None
        docs = self.documents(state)
        if write:
            state.draft = await self.write(state.topic, docs)
        self.sync(state)
        return docs

    def sync(self, state: State) -> None:
        state.spent_usd = state.offset_usd + self.ctx.usage()["spent_usd"]

    # ------------------------------------------------------------------ candidates
    def restored(self, state: State, first: list[Candidate]) -> list[Candidate] | None:
        """A restored decision point's candidates under this cell's settings: searches it does
        not offer removed, and `finish` restored where the original rules had left it out. None
        (prepare the step afresh) where the original offered only `finish` at its batch cap and
        this cell's cap allows more."""
        if state.batches < state.max_batches and all(c.kind == FINISH for c in first):
            return None
        kept = [c for c in first if c.kind == FINISH or c.kind in self.actions]
        if all(c.kind != FINISH for c in kept):
            kept.append(Candidate(FINISH))
        return kept

    async def prepare(self, state: State) -> list[Candidate]:
        """The step's candidates, `finish` last. Only `finish` once the batch cap is reached, or
        once a relevant paper exists and another step no longer fits (nothing is proposed or
        fetched then)."""
        finish = Candidate(FINISH)
        if state.batches >= state.max_batches or (state.relevant() and not room(state, self.costs)):
            return [finish]
        searches = []
        if "abstract" in self.actions and (abstract := self.abstract(state)):
            searches.append(abstract)
        if "explore" in self.actions and (query := await self.propose(state)):
            page = self.web_rows if self.web_on(state) else self.new_rows
            searches.append(Candidate("explore", query, page(state, query, PAGE), PAGE))
        if "deepen" in self.actions and (deepen := self.deepen(state)):
            searches.append(deepen)
        return [*(replace(c, cost_usd=self.costs.screen(len(c.rows))) for c in searches), finish]

    def abstract(self, state: State) -> Candidate | None:
        """The abstract search's next batch: the first PAGE papers of its ranked list that no
        search has brought yet (a report task's prompt search: its web hits first, when it has
        them). None when the access policy blocks the abstract as a query (the search then
        returns nothing) or the list is used up."""
        if not state.abstract:
            return None
        web = self.web_on(state) or (state.abstract, REQUEST_WEB_K) in self.web_hits
        rows = self.web_rows(state, state.abstract, REQUEST_WEB_K) if web else ()
        if len(rows) < PAGE:  # the arXiv list, fetched once the web hits no longer fill a batch
            urls = {r["url"] for r in rows}
            arxiv = self.new_rows(state, state.abstract, ABSTRACT_K)
            rows += tuple(r for r in arxiv if r["url"] not in urls)
        rows = rows[:PAGE]
        return Candidate("abstract", state.abstract, rows, ABSTRACT_K) if rows else None

    async def propose(self, state: State) -> str | None:
        """The executor's new keyword query (None if it only repeats an earlier one)."""
        searches = "\n".join(
            f"- {label(state, q.text)} ({q.new}, {q.relevant})" for q in state.queries
        )
        titles = "\n".join(f"- {p.row['title']}" for p in state.relevant()[-20:])
        prompt = (REPORT_PROPOSE if state.report else PROPOSE).format(
            topic=state.topic, searches=searches or "(none)", titles=titles or "(none)"
        )
        reply = await chat(self.ctx, "propose", prompt, max_tokens=PROPOSE_TOKENS)
        done = {q.text.lower() for q in state.queries}
        for line in reply.splitlines():
            query = " ".join(LIST_ITEM.sub("", line).strip().strip("\"'`").split()[:10])
            if query and query.lower() not in done:
                return query
        return None

    def deepen(self, state: State) -> Candidate | None:
        """The next page with new papers of the most productive keyword query that still has one
        (relevant per new paper; ties: the earlier query), at most MAX_PAGES pages deep. Pages
        fetched before are re-read for free; the scan fetches at most DEEPEN_FETCHES new ones per
        step and otherwise resumes in the next. A query whose first page came from the web has
        no arXiv list to follow."""
        fetches = 0
        keyword = [
            q
            for q in state.queries
            if q.text != state.abstract
            and ((q.text, PAGE) in self.hits or (q.text, PAGE) not in self.web_hits)
        ]
        for q in sorted(keyword, key=lambda q: -q.relevant / max(q.new, 1)):
            for k in range(PAGE * (q.pages + 1), PAGE * MAX_PAGES + 1, PAGE):
                if (q.text, k) not in self.hits:
                    if fetches == DEEPEN_FETCHES:
                        return None
                    fetches += 1
                if rows := self.new_rows(state, q.text, k):
                    return Candidate("deepen", q.text, rows, k)
        return None

    def new_rows(self, state: State, query: str, k: int) -> tuple[dict[str, Any], ...]:
        """DS-base's normalized rows of the query's top-k hits that no search has brought yet."""
        if (query, k) not in self.hits:
            df = dsb.web_search(self.arxiv.to_web_search_corpus(), query, k)
            _, rows = self.agentic._normalize_search_df(
                df, query, self.arxiv.to_rename_map(), lambda _: "", ""
            )
            self.remember(query, k, rows.to_dict("records"))
        seen = state.seen()
        return tuple(r for r in self.hits[query, k] if r["url"] not in seen)

    def web_on(self, state: State) -> bool:
        """Whether a search goes to the web: a report task's, with the web among the corpora and
        a web operation left."""
        config = self.ctx.search_config
        if not (state.report and "web" in config.corpora):
            return False
        return config.web_ops is None or self.ctx.search.stats["web"] < config.web_ops

    def web_rows(self, state: State, query: str, k: int) -> tuple[dict[str, Any], ...]:
        """`new_rows` on the web: DS-base's normalized rows of the query's top-k Tavily hits
        (snippets, no page read) that no search has brought yet; a failed search brings none."""
        if (query, k) not in self.web_hits:
            try:
                df = dsb.web_search(self.web.to_web_search_corpus(), query[:WEB_QUERY_CHARS], k)
            except ToolError:
                df = pd.DataFrame()
            _, rows = self.agentic._normalize_search_df(
                df, query, self.web.to_rename_map(), lambda _: "", ""
            )
            self.web_hits[query, k] = rows.to_dict("records")
            self.keep_rows(self.web_hits[query, k])
        seen = state.seen()
        return tuple(r for r in self.web_hits[query, k] if r["url"] not in seen)

    # ------------------------------------------------------------------ actions
    async def act(self, state: State, c: Candidate) -> None:
        if c.kind == FINISH:
            state.phase = "done"
            return
        query = state.query(c.query)
        if query is None:
            query = Query(c.query)
            state.queries.append(query)
        kept = set(self.top.filter(self.screen_cfg, pd.DataFrame(list(c.rows)), state.topic).url)
        state.papers += [Paper(row, row["url"] in kept) for row in c.rows]
        query.pages = query.pages + 1 if c.kind == "abstract" else c.k // PAGE
        query.new += len(c.rows)
        query.relevant += sum(row["url"] in kept for row in c.rows)
        state.batches += 1

    def documents(self, state: State) -> pd.DataFrame:
        """The writer's documents: the relevant papers, ranked to WRITE_DOCS by DS-base's sem_topk
        when there are more (its pivots seeded per task: one ranking per relevant set), fixed in
        the state once chosen."""
        if not state.docs:
            docs = pd.DataFrame([p.row for p in state.relevant()])
            if docs.empty:
                raise RuntimeError("no relevant papers to write from")
            if len(docs) > WRITE_DOCS:
                np.random.seed(self.ctx.task.seed)
                docs = self.top.filter(self.write_cfg, docs, state.topic)  # sem_topk to WRITE_DOCS
            state.docs = list(docs.url)
        return pd.DataFrame([self.rows[url] for url in state.docs])

    async def write(self, topic: str, docs: pd.DataFrame) -> str:
        """DS-base's intro writer (P10, no background)."""
        return await self.top.generate_intro_section(topic, docs, "", self.write_cfg)

    # ------------------------------------------------------------------ logs
    def remember(self, query: str, k: int, rows: list[dict[str, Any]]) -> None:
        """Keep a fetched page, and log it (pages.jsonl) and its rows (rows.jsonl)."""
        self.hits[query, k] = rows
        self.keep_rows(rows)
        append(self.pages, {"query": query, "k": k, "urls": [r["url"] for r in rows]})

    def keep_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        """Keep rows not yet kept and append them to rows.jsonl (the snapshots refer to them by
        URL)."""
        new = [r for r in rows if r["url"] not in self.rows]
        self.rows |= {r["url"]: r for r in new}
        for row in new:
            append(self.rows_log, row)

    def record(
        self,
        point: dict[str, Any],
        decision: Decision,
        executed: int,
        forced: bool,
        overridden: bool,
        control_usd: float,
        state: State,
        taken: bool = False,
    ) -> None:
        """A decision once executed: what the policy saw and asked for (`taken`: a branch's
        executed candidate, not asked), what ran, and the list spend before it (the snapshot's),
        of the policy's own calls and after the action."""
        append(
            self.decisions,
            point
            | {
                "requested": decision.index,
                "executed": executed,
                "probs": decision.probs,
                "input": decision.input,
                "forced": forced,
                "overridden": overridden,
                "taken": taken,
                "control_usd": control_usd,
                "spent_after_usd": state.spent_usd,
            },
        )


def append(path: Path, record: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

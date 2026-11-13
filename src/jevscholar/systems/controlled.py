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
evidence.

Writing decisions (`revise`, empty by default: the cell then is what it was before they existed)
follow the first draft: the policy chooses among the offered revisions and `submit`
(controller/state.py: write_legal). A revision calls DS-base's own section writer again, with the
current draft as its existing content (citations as numbers into the writer's documents, which
stay the same) and REVISE's instructions, on the `revise` stage. Each writing decision is appended
to writes.jsonl, in the decision format; the earlier drafts are kept in the state. A run can also
continue from a writing decision point (`start.revision`): the n-th of an earlier run's task, or,
for a run without writes.jsonl, its final state (n = 0) with the draft it submitted.

The reading revisions (`read`, `redraft`) give the writer each document's full-text excerpt from
the local cache (search/fulltext.py; fetched beforehand, never during a run) besides its abstract,
for that and every later revision: `read` revises the draft for coverage from them, `redraft` is
DS-base's intro writer again from them, without the draft. A reading revision with no excerpt in
the cache for any document fails the task. The judges still read each source's abstract.
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
from jevscholar.controller.policy import Decision, Policy, make_policy
from jevscholar.controller.prompt import label
from jevscholar.controller.state import (
    ABSTRACT_K,
    FINISH,
    MAX_BATCHES,
    MAX_PAGES,
    PAGE,
    READS,
    SEARCHES,
    SUBMIT,
    WRITE_DOCS,
    Candidate,
    Costs,
    Kind,
    Pages,
    Paper,
    Point,
    Query,
    Revision,
    State,
    affordable,
    legal,
    numbered,
    restore,
    room,
    snapshot,
    write_legal,
)
from jevscholar.render import finalize
from jevscholar.search import fulltext
from jevscholar.systems import chat
from jevscholar.systems import deepscholar_base as dsb
from jevscholar.types import Report, Strict, System, Task, TaskContext
from jevscholar.util import read_jsonl

MAX_DECISIONS = 40  # a guard: the batch cap ends every episode long before
DEEPEN_FETCHES = 2  # new pages the deepen scan may fetch per step (pages fetched before are free)
PROPOSE_TOKENS = 100  # one query: bounds a proposal's cost
PROPOSE = """You are searching arXiv for the prior work needed to write the related-work section
of the paper described in the task below. Write one new search query (3-10 keywords) that would
find relevant papers the searches so far have missed. Output the query only, nothing else.

Task:
{topic}

Searches so far (new papers, relevant):
{searches}

Relevant papers found so far:
{titles}"""
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
REVISE = {
    "ground": "Revision for citation accuracy: rewrite the existing section so that every sentence "
    "that cites a source states only what that source's material supports. Check each citation "
    "against the source material: move a citation that points to the wrong source to the right "
    "one, and soften or remove statements no source supports. Keep the title, the structure and "
    "every supported statement with its citations; do not add new topics.",
    "expand": "Revision for coverage: rewrite the existing section so that it also covers what a "
    "reader of the related-work section of the paper in the user query expects: the problem the "
    "paper addresses and why it matters, the key concepts and lines of work of its field, and the "
    "relevant sources in the source material that the existing section does not discuss yet. "
    "Keep every supported statement of the existing section with its citations, and ground every "
    "new statement in the source material with citations.",
    "read": "Revision for coverage from the full text: the source material now also gives an "
    "excerpt of each source's full text (its introduction). Rewrite the existing section so that "
    "it uses what these excerpts add beyond the abstracts: each source's problem, method and "
    "findings stated precisely, how the sources relate to each other and to the paper in the user "
    "query, and the relevant sources the existing section does not discuss yet. Keep every "
    "supported statement of the existing section with its citations, and ground every new "
    "statement in the source material with citations.",
}


class Start(Strict):
    """A decision point of an earlier run to continue from: the first of the task's decision
    points in runs/<run>/<bench>/<system> after this many retrieval batches, or with `revision`
    its writing decision point after that many revisions; with `take`, the candidate of that kind
    is executed there instead of asking the policy (an error where the rules do not offer it)."""

    run: str
    system: str
    batches: int = 0
    take: Kind | None = None
    revision: int | None = None

    @model_serializer(mode="wrap")
    def _dump(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Without `take` or `revision`, the fields a start had before they existed: its cell
        identity (and so the runs made with it) stays the same."""
        out = handler(self)
        for key in ("take", "revision"):
            if getattr(self, key) is None:
                out.pop(key, None)
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
        revise: tuple[Revision, ...] = ()  # revisions offered after drafting

        @model_serializer(mode="wrap")
        def _dump(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
            """Without revisions, the parameters a cell had before writing decisions existed."""
            out = handler(self)
            if not self.revise:
                out.pop("revise", None)
            return out

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.costs = Costs(**self.p.costs)
        make_policy(self.p.policy)  # fail at construction, not on the first task

    def identity(self) -> dict[str, Any]:
        """The policy with its defaults (d1's profile) resolved, as well as the parameters (where
        the d1 server listens is not part of it)."""
        policy = make_policy(self.p.policy).model_dump(mode="json", exclude={"url"})
        for part in ("first", "then"):  # a FirstThen's parts (no other policy has them)
            if isinstance(policy.get(part), dict):
                policy[part].pop("url", None)
        return super().identity() | {"policy": policy}

    async def run(self, ctx: TaskContext) -> Report:
        """Research and write the task, from the `start` point if there is one; the spend that
        point carries becomes the state's offset, and this cell's cap and batch limit apply."""
        top, task, p = dsb.install(), ctx.task, self.p
        budget = ctx.cap_usd if p.budget_usd is None else p.budget_usd
        take = None if p.start is None else p.start.take
        writing = p.start is not None and p.start.revision is not None
        if p.start is None:
            state = State(task.prompt, budget, task.meta.get("abstract", ""), p.max_batches)
            first, pages = None, {}
        else:
            point = locate_write(task, p.start) if writing else locate(task, p.start)
            state, first, pages = point.state, point.candidates, point.pages
            if state.topic != task.prompt:
                raise ValueError("the decision point belongs to another task")
            state.offset_usd = state.spent_usd
            state.budget_usd, state.max_batches = budget, p.max_batches
            record = {"offset_usd": state.offset_usd, "batches": state.batches}
            (ctx.out_dir / "start.json").write_text(json.dumps(record))  # kept if the run fails
        with dsb.task_scope(ctx.search, task.seed, repair_citations=True):
            loop = Loop(ctx, top, self.costs, p.actions, pages, p.sample, p.revise)
            docs = await loop.run(state, make_policy(p.policy), first, take, writing)
        (ctx.out_dir / "state.json").write_text(state.to_json())
        report = dsb.to_report(state.draft or "", docs)
        return finalize(
            report.text,
            report.sources,
            **report.meta,
            status="ok",
            batches=state.batches,
            decisions=state.decisions,
            revisions=list(state.revisions),
            queries=len(state.queries),
            screened=len(state.papers),
            relevant=len(state.relevant()),
            spent_usd=state.spent_usd,
            offset_usd=state.offset_usd,
            read_docs=loop.read_docs,
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


def locate_write(task: Task, start: Start) -> Point:
    """The writing decision point `start` names: the task's in an earlier run after
    `start.revision` revisions (writes.jsonl), or, where the run has no writing decisions, its
    final state with the draft it submitted (revision 0 only)."""
    folder = task_dir(paths.RUNS / start.run, task.bench, start.system, task)
    rows = {r["url"]: r for r in read_jsonl(folder / "rows.jsonl")}
    if (folder / "writes.jsonl").exists():
        for record in read_jsonl(folder / "writes.jsonl"):
            if len(record["state"]["revisions"]) == start.revision:
                point = restore(record, rows, [])
                return Point(point.state, [], {})
        raise ValueError(f"{folder}: no writing decision after {start.revision} revisions")
    if start.revision != 0:
        raise ValueError(f"{folder}: no writes.jsonl, so only its submitted draft (revision 0)")
    state = State.from_json((folder / "state.json").read_text())
    if state.draft is None or not state.docs:
        raise ValueError(f"{folder}: the run's final state has no draft")
    return Point(state, [], {})


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
        revise: tuple[str, ...] = (),
    ) -> None:
        """`pages`: search pages fetched before (a restored decision point's), free to re-read;
        `revise`: the revisions offered at writing decisions (none: no writing decisions)."""
        self.ctx, self.top, self.costs, self.actions, self.sample = ctx, top, costs, actions, sample
        self.revise = revise
        self.read_docs = 0  # documents the writer read the full text of (reading revisions)
        self.screen_cfg = dsb.build_configs(ctx, {"use_sem_topk": False})
        self.write_cfg = dsb.build_configs(ctx, {"use_sem_filter": False}, sample)
        self.agentic = importlib.import_module("deepscholar_base.search.agentic_search")
        self.arxiv = self.agentic.ToolTypes.ARXIV
        self.hits: Pages = {}
        self.rows: dict[str, dict[str, Any]] = {}  # URL -> search row, as logged in rows.jsonl
        out = ctx.out_dir
        self.decisions, self.rows_log, self.pages, self.writes = (
            out / f for f in ("decisions.jsonl", "rows.jsonl", "pages.jsonl", "writes.jsonl")
        )
        for path in (self.decisions, self.rows_log, self.pages, self.writes):
            path.unlink(missing_ok=True)
        self.decisions.touch()  # a writing branch makes no retrieval decision
        for (query, k), rows in pages.items():
            self.remember(query, k, rows)

    async def run(
        self,
        state: State,
        policy: Policy,
        first: list[Candidate] | None = None,
        take: str | None = None,
        writing: bool = False,
    ) -> pd.DataFrame:
        """Decide and search until `finish`, then write (and revise, where revisions are
        offered); returns the writer's documents. `first`: the opening decision's candidates, as
        a snapshot saved them, so that step is not prepared (nor paid for) again. `take`: the
        kind of candidate the opening decision executes, without asking the policy. `writing`:
        the state is a writing decision point, with its draft (no retrieval, no first draft)."""
        self.keep_rows([p.row for p in state.papers] + [r for c in first or () for r in c.rows])
        if writing:
            docs = self.documents(state)
            await self.revisions(state, policy, docs, take)
            return docs
        proposed = None if first is None else self.restored(state, first)
        while state.phase == "retrieval":
            self.sync(state)
            if proposed is None:
                proposed = await self.prepare(state)
                self.sync(state)
            offered = legal(state, proposed, self.costs)
            if state.decisions >= MAX_DECISIONS:
                offered = offered[-1:]
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
        state.draft = await self.write(state.topic, docs)
        self.sync(state)
        if self.revise:
            await self.revisions(state, policy, docs, None)
        return docs

    async def revisions(
        self, state: State, policy: Policy, docs: pd.DataFrame, take: str | None
    ) -> None:
        """Writing decisions until `submit`: each between the affordable revisions offered and
        `submit`, logged to writes.jsonl once executed. After a reading revision (this run's or
        the start point's) the writer's documents carry their full-text excerpts."""
        state.phase = "writing"
        if any(k in READS for k in state.revisions):
            docs = self.reading(docs)
        while True:
            self.sync(state)
            offered = write_legal(state, self.revise, self.costs)
            point = snapshot(state, offered, len(self.hits))
            before = state.spent_usd
            forced, taken = len(offered) == 1, take is not None
            if taken:
                decision, take = Decision(taken_index(offered, take)), None
            elif forced:
                decision = Decision(0)
            else:
                decision = await policy.choose(self.ctx, state, offered)
            self.sync(state)
            control = state.spent_usd - before
            choice = offered[decision.index]
            overridden = choice.kind != SUBMIT and choice.cost_usd > state.remaining_usd
            if overridden:  # the decision's own cost left no room for the revision: submit
                choice = offered[-1]
            state.decisions += 1
            if choice.kind in READS and not self.read_docs:
                docs = self.reading(docs)
            if choice.kind != SUBMIT:
                state.drafts.append(state.draft or "")
                state.revisions.append(choice.kind)
                state.draft = await self.rewrite(state, choice.kind, docs)
                self.sync(state)
            executed = offered.index(choice)
            self.record(
                point, decision, executed, forced, overridden, control, state, taken, self.writes
            )
            if choice.kind == SUBMIT:
                break
        state.phase = "done"

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
            searches.append(Candidate("explore", query, self.new_rows(state, query, PAGE), PAGE))
        if "deepen" in self.actions and (deepen := self.deepen(state)):
            searches.append(deepen)
        return [*(replace(c, cost_usd=self.costs.screen(len(c.rows))) for c in searches), finish]

    def abstract(self, state: State) -> Candidate | None:
        """The abstract search's next batch: the first PAGE papers of its ranked list that no
        search has brought yet. None when the access policy blocks the abstract as a query (the
        search then returns nothing) or the list is used up."""
        if not state.abstract:
            return None
        rows = self.new_rows(state, state.abstract, ABSTRACT_K)[:PAGE]
        return Candidate("abstract", state.abstract, rows, ABSTRACT_K) if rows else None

    async def propose(self, state: State) -> str | None:
        """The executor's new keyword query (None if it only repeats an earlier one)."""
        searches = "\n".join(
            f"- {label(state, q.text)} ({q.new}, {q.relevant})" for q in state.queries
        )
        titles = "\n".join(f"- {p.row['title']}" for p in state.relevant()[-20:])
        prompt = PROPOSE.format(
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
        step and otherwise resumes in the next."""
        fetches = 0
        keyword = [q for q in state.queries if q.text != state.abstract]
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

    def reading(self, docs: pd.DataFrame) -> pd.DataFrame:
        """The writer's documents with each cached full-text excerpt after its abstract (in the
        `context` the writer reads, DS-base's "title[url]: snippet")."""
        docs = docs.copy()
        contexts, n = [], 0
        for row in docs.to_dict("records"):
            text = fulltext.excerpt(row["url"])
            context = row["context"]
            if text:
                context, n = f"{context}\nFull text (excerpt): {text}", n + 1
            contexts.append(context)
        if not n:
            raise RuntimeError("no full text cached for any document (jev data fulltext)")
        docs["context"] = contexts
        self.read_docs = n
        return docs

    async def write(self, topic: str, docs: pd.DataFrame) -> str:
        """DS-base's intro writer (P10, no background)."""
        return await self.top.generate_intro_section(topic, docs, "", self.write_cfg)

    async def rewrite(self, state: State, kind: str, docs: pd.DataFrame) -> str:
        """One revision: DS-base's section writer with the draft as its existing content
        (citations as numbers into `docs`, its own input form) and REVISE[kind], on the
        `revise` stage. `redraft`: DS-base's intro instructions and no existing content."""
        summary = importlib.import_module("deepscholar_base.utils.summary_generation")
        lm = dsb.stage_lm(self.ctx, "revise", self.sample)
        if kind == "redraft":
            prompts = importlib.import_module("deepscholar_base.utils.prompts")
            instructions, existing = prompts.intro_section_instructions, ""
        else:
            instructions, existing = REVISE[kind], numbered(state.draft or "", state.docs)
        return await summary.generate_section_summary_with_citations(
            state.topic, docs, instructions, existing, lm=lm
        )

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
        log: Path | None = None,
    ) -> None:
        """A decision once executed: what the policy saw and asked for (`taken`: a branch's
        executed candidate, not asked), what ran, and the list spend before it (the snapshot's),
        of the policy's own calls and after the action; to decisions.jsonl unless `log`."""
        append(
            log or self.decisions,
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

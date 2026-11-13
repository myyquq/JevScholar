"""The controlled loop's state, its candidate actions and their hard rules.

Pure data on the standard library, so the runner's environment and `.venv-train` (the d1 server
and trainer) share it. A State holds everything a decision may depend on and round-trips through
JSON. Papers keep DeepScholar-base's search row verbatim: screening and writing rebuild the exact
frames DS-base builds, so the executor sees the same requests (and a paired DS-base run's cached
replies replay). A decision point is saved by reference (`snapshot`: papers and candidate rows as
URLs into the task's rows.jsonl, and the count of search pages fetched so far, which pages.jsonl
lists in order) and restored as a `Point` (`restore`): a rollout continues from it without
preparing that step again, and re-reads the pages the original run had fetched for free.

An episode is a sequence of retrieval decisions, each among at most four candidates with `finish`
last: the next batch of the abstract search, one keyword query the executor proposes, the next
page of the most productive keyword query, or finishing. After `finish` the writer writes a first
draft. Where a cell offers revisions (`REVISIONS`), writing decisions follow, each between the
revisions still affordable and `submit` (last): `ground` rewrites the draft so that each cited
sentence states only what its sources support, `expand` so that it also covers the task's problem,
the field's key ideas and sources not discussed yet. The reading revisions (`READS`, at most one
per episode) give the writer each document's full-text excerpt besides its abstract (and keep it
for every later revision): `read` revises the draft for coverage from them, `redraft` writes the
section anew from them. At most MAX_REVISIONS per episode. The
rules here are the loop's hard rules: they keep the candidates a policy may choose from, and
leave a policy no say where only one is left. A search is legal only within the
batch cap, if it brings a paper no other candidate brings, and if after it another step's
overhead and the writing still fit, whatever it adds; finishing waits for a first relevant paper
while a search is still possible.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Kind = Literal[
    "abstract", "explore", "deepen", "finish", "ground", "expand", "read", "redraft", "submit"
]
Revision = Literal["ground", "expand", "read", "redraft"]
Phase = Literal["retrieval", "writing", "done"]
SEARCHES = ("abstract", "explore", "deepen")
FINISH = "finish"
REVISIONS = ("ground", "expand", "read", "redraft")
READS = ("read", "redraft")  # the revisions that read the documents' full text
SUBMIT = "submit"
STOPS = (FINISH, SUBMIT)  # the candidate that ends a phase, always offered last
MAX_REVISIONS = 2  # writing decisions that revise, per episode
CITATION = re.compile(r"\\\[\[[^\]]*\]\((?P<url>[^)\s]+)\)\\\]")  # the writer's \[[label](url)\]
PAGE = 10  # hits per search page and papers per batch (DS-base's per-query result count)
MAX_PAGES = 3  # a keyword query is followed to at most PAGE x MAX_PAGES hits
ABSTRACT_K = 150  # the abstract search reads one ranked list this deep, a batch at a time
MAX_BATCHES = 6  # retrieval batches per episode
WRITE_DOCS = 30  # DS-base's final_max_results_count: more relevant papers are ranked first
SNAPSHOT = 3  # version of the decision snapshot format


@dataclass
class Paper:
    row: dict[str, Any]  # DS-base's normalized search row: title, url, snippet, date, query, ...
    relevant: bool | None = None  # None: not screened yet

    @property
    def url(self) -> str:
        return self.row["url"]


@dataclass
class Query:
    text: str
    pages: int = 0  # result pages taken
    new: int = 0  # papers it brought that no earlier search had
    relevant: int = 0  # of those, judged relevant


@dataclass
class State:
    topic: str  # the task prompt, exactly as the writer receives it
    budget_usd: float  # the task's cap B, list price
    abstract: str = ""  # the abstract search's query: the target paper's abstract ("": none)
    max_batches: int = MAX_BATCHES
    phase: Phase = "retrieval"
    spent_usd: float = 0.0  # list spend so far: the gateway's view plus `offset_usd`
    offset_usd: float = 0.0  # spend before this process took over (a rollout's branch point)
    queries: list[Query] = field(default_factory=list)
    papers: list[Paper] = field(default_factory=list)  # in order of first appearance
    batches: int = 0  # retrieval batches screened
    decisions: int = 0  # decision points passed
    docs: list[str] = field(default_factory=list)  # the writer's documents (URLs), once ranked
    draft: str | None = None  # the written section (DS-base markdown), once written
    revisions: list[str] = field(default_factory=list)  # the revisions made, in order
    drafts: list[str] = field(default_factory=list)  # the drafts before each revision

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.spent_usd

    def relevant(self) -> list[Paper]:
        return [p for p in self.papers if p.relevant]

    def seen(self) -> set[str]:
        return {p.url for p in self.papers}

    def query(self, text: str) -> Query | None:
        return next((q for q in self.queries if q.text == text), None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> State:
        d = json.loads(text)
        queries, papers = [Query(**q) for q in d["queries"]], [Paper(**p) for p in d["papers"]]
        return cls(**d | {"queries": queries, "papers": papers})


@dataclass(frozen=True)
class Candidate:
    """An action a policy may choose. A search carries the new papers it would screen (its hits
    are prefetched), so its cost and content are known before the choice."""

    kind: Kind
    query: str = ""
    rows: tuple[dict[str, Any], ...] = ()  # the new papers' search rows
    k: int = 0  # the query's hits it reads to (PAGE x pages)
    cost_usd: float = 0.0  # estimated list cost of screening its rows

    def ref(self) -> dict[str, Any]:
        """The candidate with its rows as URLs (see `snapshot`)."""
        urls = [r["url"] for r in self.rows]
        return {
            "kind": self.kind,
            "query": self.query,
            "urls": urls,
            "k": self.k,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True)
class Costs:
    """List-price estimates the rules plan with: screening is per paper; writing is the intro
    call plus, beyond WRITE_DOCS relevant papers, DS-base's sem_topk ranking, which grows with
    their number; a step's overhead is what the loop spends around an action (the next step's
    proposal and prefetches, and a policy call). Measured on a dev task (190 papers screened, 73
    relevant), with some margin; the gateway's latch backs them up."""

    screen_usd: float = 0.0005  # per paper: one CoT sem_filter call (measured 0.00046)
    write_usd: float = 0.008  # the intro writer (measured 0.0036; DS-base mean 0.004)
    revise_usd: float = 0.01  # one revision: the writer's call with the draft as well
    read_usd: float = 0.03  # a reading revision: the writer's call with full-text excerpts
    rank_usd: float = 0.0008  # per relevant paper beyond WRITE_DOCS (measured 0.0007)
    step_usd: float = 0.007  # a proposal (<= 0.0013), <= 4 page fetches (0.004), a choice

    def screen(self, n: int) -> float:
        return n * self.screen_usd

    def write(self, relevant: int) -> float:
        return self.write_usd + (relevant * self.rank_usd if relevant > WRITE_DOCS else 0.0)


def affordable(state: State, c: Candidate, costs: Costs) -> bool:
    """Whether search `c` still lets the task finish: its screening, a step's overhead, then
    writing, should every paper it adds be relevant."""
    after = costs.write(len(state.relevant()) + len(c.rows))
    return c.cost_usd + costs.step_usd + after <= state.remaining_usd


def room(state: State, costs: Costs) -> bool:
    """Whether another step is worth preparing: it must afford at least a one-paper search."""
    after = costs.write(len(state.relevant()) + 1)
    return costs.screen(1) + costs.step_usd + after <= state.remaining_usd


def legal(state: State, candidates: list[Candidate], costs: Costs) -> list[Candidate]:
    """The candidates a policy may choose from, in their order, `finish` last: each affordable
    search within the batch cap that brings a paper no earlier one does, then `finish` once there
    is a relevant paper (or no search is left). Never empty, as long as `candidates` ends with
    `finish`."""
    finish = [c for c in candidates if c.kind == FINISH]
    seen: set[str] = set()
    searches = []
    for c in candidates:
        urls = {r["url"] for r in c.rows}
        if (
            c.kind in SEARCHES
            and state.batches < state.max_batches
            and not urls <= seen
            and affordable(state, c, costs)
        ):
            searches.append(c)
            seen |= urls
    return [*searches, *(finish if state.relevant() or not searches else [])]


def numbered(draft: str, urls: Sequence[str], unknown: str = "") -> str:
    """The draft with each citation as its number into `urls` ([3]), the writer's own input
    form; a citation of another URL becomes `unknown`."""
    index = {url: i + 1 for i, url in enumerate(urls)}
    return CITATION.sub(lambda m: f"[{index[m['url']]}]" if m["url"] in index else unknown, draft)


def write_legal(state: State, offered: tuple[str, ...], costs: Costs) -> list[Candidate]:
    """A writing decision's candidates: each offered revision while fewer than MAX_REVISIONS were
    made and it fits the remaining budget (a reading one only while none was made), then
    `submit`."""
    out = []
    if len(state.revisions) < MAX_REVISIONS:
        read = any(k in READS for k in state.revisions)
        for k in REVISIONS:
            cost = costs.read_usd if k in READS else costs.revise_usd
            if k in offered and not (read and k in READS) and cost <= state.remaining_usd:
                out.append(Candidate(k, cost_usd=cost))
    return [*out, Candidate(SUBMIT)]


Pages = dict[tuple[str, int], list[dict[str, Any]]]  # (query, k) -> its top-k search rows


@dataclass
class Point:
    """A decision point: the state a policy saw, its candidates, and the search pages fetched."""

    state: State
    candidates: list[Candidate]
    pages: Pages


def snapshot(state: State, candidates: list[Candidate], fetched: int) -> dict[str, Any]:
    """A decision point by reference: papers and candidate rows as URLs into rows.jsonl, and the
    number of search pages fetched so far (the first `fetched` records of pages.jsonl)."""
    papers = [[p.url, p.relevant] for p in state.papers]
    return {
        "version": SNAPSHOT,
        "state": asdict(state) | {"papers": papers},
        "candidates": [c.ref() for c in candidates],
        "fetched": fetched,
    }


def restore(
    snap: Mapping[str, Any], rows: Mapping[str, dict[str, Any]], pages: list[dict[str, Any]]
) -> Point:
    """The Point of a `snapshot`: rows looked up by URL in rows.jsonl, fetched pages taken from
    pages.jsonl's records ({"query", "k", "urls"})."""
    if snap["version"] != SNAPSHOT:
        raise ValueError(f"snapshot version {snap['version']}, expected {SNAPSHOT}")
    d = snap["state"]
    state = State(
        **d
        | {
            "queries": [Query(**q) for q in d["queries"]],
            "papers": [Paper(rows[url], relevant) for url, relevant in d["papers"]],
        }
    )
    candidates = [
        Candidate(c["kind"], c["query"], tuple(rows[u] for u in c["urls"]), c["k"], c["cost_usd"])
        for c in snap["candidates"]
    ]
    fetched = {(p["query"], p["k"]): [rows[u] for u in p["urls"]] for p in pages[: snap["fetched"]]}
    return Point(state, candidates, fetched)

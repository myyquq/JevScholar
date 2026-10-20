"""What a decision model reads at a retrieval decision, and how its answer becomes a choice.

`render` turns a state and its legal candidates into d1's `choice` schema: a view of the state
(the task, the remaining budget and batches, each search's yield, the latest relevant titles) and
one question whose options are the candidates, each described with its cost and the titles of
the new papers it would screen. A `Profile` fixes how many relevant titles are listed, how the
checkpoint lays the state out (`style`) and the stopping threshold; its defaults are d1's. Long
states are shown at shorter `LEVELS` until they fit d1's validated length; the budget and the
batch count are never cut.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jevscholar.controller.state import FINISH, SEARCHES, Candidate, State

INSTRUCTIONS = (
    "Choose the next step of the literature search for the related-work section of the paper in "
    "the task. Keep searching while searches still find new relevant papers and the paper's main "
    "lines of prior work are not yet covered; finish once they are covered, or once the last "
    "searches found few new relevant papers."
)
PREVIEW = 5  # new titles shown per search
LEVELS = ((None, None), (10, 3), (5, 2), (0, 1))  # (titles, preview) caps for long states
VERBS = {"abstract": "search with", "explore": "search", "deepen": "next page of"}


def render(
    state: State,
    candidates: Sequence[Candidate],
    *,
    titles: int = 10,
    preview: int = PREVIEW,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(state, question) in d1's `choice` schema; option i is candidates[i]. `titles` and
    `preview` bound the listed relevant papers and the new titles shown per search."""
    relevant = state.relevant()
    view = {
        "task": state.topic,
        "phase": state.phase,
        "budget": f"${state.remaining_usd:.3f} of ${state.budget_usd:.3f} left",
        "search batches": f"{state.batches} of {state.max_batches}",
        "searches": [
            f"{label(state, q.text)} (pages {q.pages}, new {q.new}, relevant {q.relevant})"
            for q in state.queries[-12:]
        ],
        "screened": f"{sum(p.relevant is not None for p in state.papers)} papers, "
        f"{len(relevant)} relevant",
        "relevant papers": [p.row["title"] for p in relevant[max(0, len(relevant) - titles) :]],
    }
    criteria = {f"{i + 1}. {c.kind}": describe(state, c, preview) for i, c in enumerate(candidates)}
    return view, {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": criteria,
    }


def label(state: State, query: str) -> str:
    """A query as d1 reads it: the abstract search by name, a keyword query quoted."""
    return "the paper's abstract" if query == state.abstract else f'"{query}"'


def describe(state: State, c: Candidate, preview: int) -> str:
    cost = f"about ${c.cost_usd:.3f}"
    if c.kind not in SEARCHES:
        return "finish the search and move on to writing"
    titles = "; ".join(r["title"] for r in c.rows[:preview])
    more = f" (+{len(c.rows) - preview} more)" if len(c.rows) > preview else ""
    return (
        f"{VERBS[c.kind]} {label(state, c.query)}: screen {len(c.rows)} new papers ({cost}): "
        f"{titles}{more}"
    )


@dataclass(frozen=True)
class Profile:
    """What d1 reads at a decision and how its answer becomes a choice: `titles` relevant titles
    at most, the checkpoint's state layout (`style`) and, with `threshold`, `finish` once its
    probability reaches it (else the most probable other candidate)."""

    titles: int = 10
    style: Literal["json_only", "sections"] = "sections"
    threshold: float | None = 0.5

    def __post_init__(self) -> None:
        if self.style not in ("json_only", "sections"):
            raise ValueError("profile style is json_only or sections")
        if type(self.titles) is not int or self.titles < 0:
            raise ValueError("profile titles is a count (int >= 0)")
        if self.threshold is not None and not 0 < self.threshold < 1:
            raise ValueError(f"profile threshold is in (0, 1), not {self.threshold!r}")

    @property
    def render(self) -> dict[str, str]:
        """How the checkpoint renders the prompt (its other settings at their defaults)."""
        return {"style": self.style}

    def request(self, state: State, candidates: Sequence[Candidate], level: int = 0) -> dict:
        """d1's request for one decision with the view at `level` of LEVELS: {"state",
        "questions": [the choice]}. A single candidate is never asked (the rules execute it)."""
        if len(candidates) < 2:
            raise ValueError("a single candidate is executed as forced, not asked")
        cap_titles, cap_preview = LEVELS[level]
        view, question = render(
            state,
            candidates,
            titles=self.titles if cap_titles is None else min(self.titles, cap_titles),
            preview=PREVIEW if cap_preview is None else min(PREVIEW, cap_preview),
        )
        return {"state": view, "questions": [question]}

    def pick(self, candidates: Sequence[Candidate], dist: Sequence[float]) -> int:
        """The chosen candidate: `finish` at `threshold`, else the most probable."""
        last = len(candidates) - 1
        if self.threshold is None or candidates[last].kind != FINISH or not last:
            return max(range(len(dist)), key=dist.__getitem__)
        if dist[last] >= self.threshold:
            return last
        return max(range(last), key=dist.__getitem__)

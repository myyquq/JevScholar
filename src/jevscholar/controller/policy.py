"""Who picks the controlled loop's next action among the legal candidates.

- `Fixed`: a rule (retrieval kinds in rotation for a number of batches, then finish; at writing
  decisions its `revise` sequence, then submit), the ablation without any learned or prompted
  choice;
- `Prompted`: the executor answers d1's own prompt (with short rules of thumb), the ablation
  without the decision model; its calls are billed like any executor call (stage "choose");
- `D1Policy`: the d1-3B decision model behind `jev controller serve`, reading what its `Profile`
  asks: the profile's choice, or a sample (seeded per task and decision) for RL rollouts. Each
  call is admitted and charged to the task as the `d1_choice` tool, per started thousand prompt
  tokens, so the cap covers the controller too.
- `FirstThen`: one policy for the episode's first asked decision, another for the rest (RL
  groups that sample one decision of d1 and continue with the fixed rule, as branches do).

Every policy returns a Decision: the index of the chosen candidate and, where the policy has
them, its probabilities over the candidates and the exact input they came from (d1's request and
raw answers, which RL needs to recompute them).

`D1Screen` is d1 as DS-base's filter step (systems/deepscholar_base.py, `screen`): one question
per paper (the target paper's abstract, the candidate's title and abstract), the paper kept at
P(relevant) >= `threshold`, charged as the policy's calls are.
"""

from __future__ import annotations

import asyncio
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol

import requests
from pydantic import (
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
)

from jevscholar.controller.d1 import user_turn
from jevscholar.controller.prompt import LEVELS, Profile
from jevscholar.controller.state import FINISH, MAX_BATCHES, SUBMIT, Candidate, Revision, State
from jevscholar.search.client import Meter
from jevscholar.systems import chat
from jevscholar.types import Strict, TaskContext

CHOOSE_TOKENS = 50  # an option code: bounds a prompted choice's cost
RULES = """Rules of thumb: the paper's abstract is a strong first query, and a keyword query aimed
at a line of prior work still missing a good later one; prefer searches whose new papers look
relevant to the paper and unlike those already found; finish only when the paper's main lines of
prior work are covered and no search is likely to add relevant papers."""
# The `yield` prompt set's rules (their planning and revising clauses name actions no longer
# offered; the text is kept as that set was evaluated with).
YIELD_RULES = """\
Rules of thumb: search while searches still find new relevant papers and the paper's
main lines of prior work are not covered yet; the paper's abstract is a strong first query, and a
keyword query aimed at what is still missing a good later one; finish once searches add few new
relevant papers; plan when the relevant papers span several distinct lines of work; revise when
the section may state more than its sources support."""


@dataclass(frozen=True)
class Decision:
    index: int
    probs: list[float] | None = None
    input: dict[str, Any] | None = None  # what the probabilities were computed from


class Policy(Protocol):
    async def choose(
        self, ctx: TaskContext, state: State, candidates: list[Candidate]
    ) -> Decision: ...


class Fixed(Strict):
    kind: Literal["fixed"] = "fixed"
    retrieval: tuple[Literal["abstract", "explore", "deepen"], ...] = ("abstract", "explore")
    batches: int = MAX_BATCHES  # finish after this many batches
    revise: tuple[Revision, ...] = ()  # the revisions to make, then submit

    @model_serializer(mode="wrap")
    def _dump(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Without revisions, the fields the rule had before writing decisions existed: its
        identity (and so every cell made with it) stays the same."""
        out = handler(self)
        if not self.revise:
            out.pop("revise", None)
        return out

    async def choose(self, ctx: TaskContext, state: State, candidates: list[Candidate]) -> Decision:
        """Batch i wants retrieval[i % len], else the next kind in rotation, else `finish`; when
        none of these is legal (no relevant paper to finish with yet), the first candidate. At a
        writing decision: the next revision of `revise` if offered, else `submit`."""
        kinds = [c.kind for c in candidates]
        if kinds[-1] == SUBMIT:
            done = len(state.revisions)
            want = self.revise[done] if done < len(self.revise) else SUBMIT
            return Decision(kinds.index(want) if want in kinds else len(kinds) - 1)
        n, i = len(self.retrieval), state.batches
        want = [self.retrieval[(i + j) % n] for j in range(n)] if i < self.batches else []
        for kind in [*want, FINISH]:
            if kind in kinds:
                return Decision(kinds.index(kind))
        return Decision(0)


class Prompted(Strict):
    """The executor decides on exactly what d1 reads under `profile`, with RULES (YIELD_RULES
    under the `yield` prompt set) before the question's last line when `rules`. The profile's
    threshold belongs to d1's probability readout, not to a prompt."""

    kind: Literal["prompted"] = "prompted"
    profile: Profile = Profile()
    rules: bool = True

    async def choose(self, ctx: TaskContext, state: State, candidates: list[Candidate]) -> Decision:
        request = self.profile.request(state, candidates)
        (question,) = request["questions"]
        text = user_turn(request["state"], question, self.profile.render)
        if self.rules:
            head, last = text.rsplit("\n\n", 1)
            rules = YIELD_RULES if state.prompts == "yield" else RULES
            text = f"{head}\n\n{rules}\n\n{last}"
        reply = await chat(ctx, "choose", text, max_tokens=CHOOSE_TOKENS)
        index = parse(reply, len(candidates))
        if index is None:
            ctx.log.warning("prompted policy: no option in %r, taking the first", reply[:80])
            index = 0
        return Decision(index, input={"asked": [{"text": text, "reply": reply}]})


def parse(reply: str, n: int) -> int | None:
    """The option a reply names among n by its standalone letter code A.. (upper-case codes
    before lower-case ones, so "I choose B." names B)."""
    letters = re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", reply)
    found = [ord(c) - ord("A") for c in letters if c.isupper()]
    found += [ord(c) - ord("a") for c in letters if c.islower()]
    return next((i for i in found if 0 <= i < n), None)


class D1Policy(Strict):
    kind: Literal["d1"] = "d1"
    url: str = "http://127.0.0.1:9940"  # `jev controller serve`
    adapter: str | None = None  # the served adapter's sha256 (None: the base model)
    profile: Profile = Profile()
    sample: bool = False  # draw from the distribution (RL rollouts) instead of picking

    async def choose(self, ctx: TaskContext, state: State, candidates: list[Candidate]) -> Decision:
        return await asyncio.to_thread(self._choose, ctx, state, candidates)

    def preflight(self) -> None:
        serves(self.url, self.adapter)

    def _choose(self, ctx: TaskContext, state: State, candidates: list[Candidate]) -> Decision:
        self.preflight()
        meter, p = Meter(ctx.gateway, ctx.tag("control")), self.profile
        for level in range(len(LEVELS)):
            body = {"requests": [p.request(state, candidates, level)], "render": p.render}
            meter.admit("d1_choice")
            r = requests.post(f"{self.url}/choice", json=body, timeout=600)
            if r.status_code == 413:  # rejected before the forward pass: nothing to charge
                continue
            r.raise_for_status()
            out = r.json()
            tokens, answers = out["tokens"], out["probabilities"]
            meta = {"tokens": tokens, "adapter": self.adapter, "level": level}
            meter.charge("d1_choice", n=math.ceil(tokens / 1000), meta=meta)
            (dist,) = answers[0]
            index = self._sample(ctx, state, dist) if self.sample else p.pick(candidates, dist)
            return Decision(index, dist, body | {"answers": answers, "adapter": self.adapter})
        raise RuntimeError("the state does not fit d1 even at its shortest rendering")

    def _sample(self, ctx: TaskContext, state: State, dist: list[float]) -> int:
        rng = random.Random(f"{ctx.tag('control')}#{ctx.task.seed}#{state.decisions}")
        return rng.choices(range(len(dist)), weights=dist)[0]


def serves(url: str, adapter: str | None) -> None:
    """The d1 server at `url` must be up and serve `adapter`."""
    try:
        # the server answers one request at a time: /info may wait behind a /choice
        served = requests.get(f"{url}/info", timeout=600).json()["adapter"]
    except requests.RequestException as e:
        raise RuntimeError(f"no d1 server at {url}: {e}") from None
    if served != adapter:
        raise RuntimeError(f"d1 server has adapter {served}, the system wants {adapter}")


SCREEN = {
    "type": "choice",
    "instructions": "Should the candidate paper be kept as relevant prior work for the Related "
    "Works section of the paper being written?",
    "criteria": {
        "relevant": "keep: the candidate is relevant to the paper's specific topic",
        "not_relevant": "drop: the candidate is off-topic for the paper",
    },
}


class D1Screen(Strict):
    url: str = Field("http://127.0.0.1:9940", exclude=True)  # not part of the identity
    adapter: str | None = None
    threshold: float

    def preflight(self) -> None:
        serves(self.url, self.adapter)

    def read(self, ctx: TaskContext, abstract: str, rows: Sequence[dict]) -> dict[str, float]:
        """Each row's P(relevant) by URL, read in one request (one row per forward pass)."""
        self.preflight()
        meter = Meter(ctx.gateway, ctx.tag("screen"))
        body = {
            "requests": [
                {
                    "state": {
                        "paper_abstract": abstract,
                        "candidate": {"title": r["title"], "abstract": r["snippet"]},
                    },
                    "questions": [SCREEN],
                }
                for r in rows
            ]
        }
        meter.admit("d1_choice")
        r = requests.post(f"{self.url}/choice", json=body, timeout=600)
        r.raise_for_status()
        out = r.json()
        meta = {"tokens": out["tokens"], "adapter": self.adapter, "papers": len(rows)}
        meter.charge("d1_choice", n=math.ceil(out["tokens"] / 1000), meta=meta)
        probs = [answer[0][0] for answer in out["probabilities"]]  # P(relevant), first option
        return {row["url"]: p for row, p in zip(rows, probs, strict=True)}


class FirstThen(Strict):
    """`first` decides the first decision it is asked (a restored point's, in an RL group), `then`
    every later one. A fresh policy is made per episode (`make_policy`), so "first" is per
    episode."""

    kind: Literal["first"] = "first"
    first: Annotated[Fixed | Prompted | D1Policy, Field(discriminator="kind")]
    then: Annotated[Fixed | Prompted | D1Policy, Field(discriminator="kind")]
    _asked: bool = PrivateAttr(default=False)

    async def choose(self, ctx: TaskContext, state: State, candidates: list[Candidate]) -> Decision:
        policy = self.then if self._asked else self.first
        self._asked = True
        return await policy.choose(ctx, state, candidates)


PolicySpec = Annotated[Fixed | Prompted | D1Policy | FirstThen, Field(discriminator="kind")]
_POLICY: TypeAdapter[Fixed | Prompted | D1Policy | FirstThen] = TypeAdapter(PolicySpec)


def make_policy(spec: dict[str, Any]) -> Policy:
    """A policy from a system's `policy` param: {"kind": "fixed" | "prompted" | "d1", ...}."""
    return _POLICY.validate_python(spec)

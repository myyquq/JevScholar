"""Who picks the controlled loop's next action among the legal candidates.

- `Fixed`: a rule (retrieval kinds in rotation for a number of batches, then finish), the
  ablation without any learned or prompted choice;
- `Prompted`: the executor answers d1's own prompt (with short rules of thumb), the ablation
  without the decision model; its calls are billed like any executor call (stage "choose");
- `D1Policy`: the d1-3B decision model behind `jev controller serve`, reading what its `Profile`
  asks: the profile's choice, or a sample (seeded per task and decision) for RL rollouts. Each
  call is admitted and charged to the task as the `d1_choice` tool, per started thousand prompt
  tokens, so the cap covers the controller too.

Every policy returns a Decision: the index of the chosen candidate and, where the policy has
them, its probabilities over the candidates and the exact input they came from (d1's request and
raw answers, which RL needs to recompute them).
"""

from __future__ import annotations

import asyncio
import math
import random
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol

import requests
from pydantic import Field, TypeAdapter

from jevscholar.controller.d1 import user_turn
from jevscholar.controller.prompt import LEVELS, Profile
from jevscholar.controller.state import FINISH, MAX_BATCHES, Candidate, State
from jevscholar.search.client import Meter
from jevscholar.systems import chat
from jevscholar.types import Strict, TaskContext

CHOOSE_TOKENS = 50  # an option code: bounds a prompted choice's cost
# The rules as the prompted policy was selected and evaluated with (their planning and revising
# clauses name actions no longer offered; the text is kept so the policy stays the same system).
RULES = """Rules of thumb: search while searches still find new relevant papers and the paper's
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

    async def choose(self, ctx: TaskContext, state: State, candidates: list[Candidate]) -> Decision:
        """Batch i wants retrieval[i % len], else the next kind in rotation, else `finish`; when
        none of these is legal (no relevant paper to finish with yet), the first candidate."""
        kinds = [c.kind for c in candidates]
        n, i = len(self.retrieval), state.batches
        want = [self.retrieval[(i + j) % n] for j in range(n)] if i < self.batches else []
        for kind in [*want, FINISH]:
            if kind in kinds:
                return Decision(kinds.index(kind))
        return Decision(0)


class Prompted(Strict):
    """The executor decides on exactly what d1 reads under `profile`, with RULES before the
    question's last line when `rules`. The profile's threshold belongs to d1's probability
    readout, not to a prompt."""

    kind: Literal["prompted"] = "prompted"
    profile: Profile = Profile()
    rules: bool = True

    async def choose(self, ctx: TaskContext, state: State, candidates: list[Candidate]) -> Decision:
        request = self.profile.request(state, candidates)
        (question,) = request["questions"]
        text = user_turn(request["state"], question, self.profile.render)
        if self.rules:
            head, last = text.rsplit("\n\n", 1)
            text = f"{head}\n\n{RULES}\n\n{last}"
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

    def _choose(self, ctx: TaskContext, state: State, candidates: list[Candidate]) -> Decision:
        served = requests.get(f"{self.url}/info", timeout=30).json()["adapter"]
        if served != self.adapter:
            raise RuntimeError(f"d1 server has adapter {served}, the system wants {self.adapter}")
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


PolicySpec = Annotated[Fixed | Prompted | D1Policy, Field(discriminator="kind")]
_POLICY: TypeAdapter[Fixed | Prompted | D1Policy] = TypeAdapter(PolicySpec)


def make_policy(spec: dict[str, Any]) -> Policy:
    """A policy from a system's `policy` param: {"kind": "fixed" | "prompted" | "d1", ...}."""
    return _POLICY.validate_python(spec)

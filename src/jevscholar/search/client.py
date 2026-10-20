"""SearchClient: the one tool interface every system uses, bound to a task's tag and policy.

Each logical call checks the query against the task's AccessPolicy, is admitted by the gateway
(a closed budget latch or a tripped breaker stops it before any external request), makes a cached
low-level call (raw upstream results, shared across tasks), filters those results with the same
policy, and is charged, cached replays and failures included, so the ledger sees every tool
action. A page is read only if its URL and its text pass the policy; under
`undated_web="drop"` it must also be a hit of an allowed web search by this client, since any
other page has no date at all. The client is synchronous on purpose: DeepScholar calls its tools
synchronously from inside a running event loop, so nothing here may touch asyncio; async systems
wrap calls in `asyncio.to_thread`.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sized
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TypeVar

import requests

from jevscholar.search import BudgetExhausted, FatalToolError, ToolError, arxiv_backend, tavily
from jevscholar.search.policy import (
    canonical_url,
    filter_papers,
    filter_web,
    text_allowed,
    url_allowed,
)
from jevscholar.types import AccessPolicy

R = TypeVar("R", bound=Sized)


@dataclass(frozen=True)
class Meter:
    """The tool account of one gateway tag: an external tool call is admitted before it starts
    and charged once it has ended."""

    gateway: str
    tag: str

    def admit(self, tool: str, meta: dict[str, Any] | None = None) -> None:
        """Raises BudgetExhausted if the gateway refuses (402): the tag's budget latch is closed
        or the circuit breaker has tripped."""
        r = self._post("admit", {"tool": tool, "meta": meta or {}})
        if r.status_code == 402:
            raise BudgetExhausted(r.text)
        r.raise_for_status()

    def charge(
        self,
        tool: str,
        *,
        n: int = 1,
        cached: bool = False,
        ok: bool = True,
        error: str | None = None,
        fatal: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Record ended calls on the ledger; a fatal failure trips the circuit breaker."""
        body = {"tool": tool, "n": n, "cached": cached, "ok": ok, "error": error, "fatal": fatal}
        self._post("charge", {**body, "meta": meta or {}}).raise_for_status()

    def _post(self, route: str, body: dict[str, Any]) -> requests.Response:
        return requests.post(f"{self.gateway}/t/{self.tag}/{route}", json=body, timeout=60)


class SearchClient:
    def __init__(
        self, gateway: str, tag: str, policy: AccessPolicy, *, tavily_key: str | None = None
    ) -> None:
        self.meter, self.policy = Meter(gateway.rstrip("/"), tag), policy
        self.tavily_key = tavily_key or os.environ.get("JEV_TAVILY_KEY")
        self._arxiv = arxiv_backend()
        self._backend = self._arxiv.__name__.rpartition(".")[2]
        self._searched: set[str] = set()  # canonical URLs of hits this client's searches returned
        self.stats = {"calls": 0, "errors": 0, "blocked": 0, "cached": 0}
        self._lock = threading.Lock()

    # ----------------------------------------------------------------------- tools
    def arxiv_search(self, query: str, k: int) -> list[dict]:
        if not self._query_ok(query):
            return []
        cutoff = self.policy.cutoff
        before = cutoff - timedelta(minutes=1) if cutoff else None
        return self._call(
            "arxiv_search",
            {"query": query, "k": k, "backend": self._backend},
            lambda: self._arxiv.search(query, k, before),
            lambda papers: filter_papers(papers, self.policy),
        )

    def arxiv_lookup(self, ids: list[str]) -> dict[str, dict]:
        ids = list(ids)
        return self._call(
            "arxiv_lookup",
            {"ids": ids, "backend": self._backend},
            lambda: self._arxiv.lookup(ids),
            lambda found: {i: p for i, p in found.items() if filter_papers([p], self.policy)},
        )

    def web_search(self, query: str, k: int) -> list[dict]:
        if not self._query_ok(query):
            return []
        cutoff = self.policy.cutoff
        end = (cutoff - timedelta(days=1)).date() if cutoff else None
        hits = self._call(
            "tavily_search",
            {"query": query, "k": k},
            lambda: tavily.search(query, k, end, self.tavily_key),
            lambda hits: filter_web(hits, self.policy),
        )
        with self._lock:
            self._searched.update(canonical_url(h["url"]) for h in hits)
        return hits

    def web_read(self, url: str) -> str | None:
        """Page text as markdown; None if it is blocked or Tavily extracts nothing."""
        unvetted = self.policy.undated_web == "drop" and canonical_url(url) not in self._searched
        if unvetted or not url_allowed(url, self.policy):
            self._bump(blocked=1)
            return None

        def fetch() -> tuple[list[str], bool]:
            text, cached = tavily.extract(url, self.tavily_key)
            return ([text] if text else []), cached

        kept = self._call(
            "tavily_extract",
            {"url": url},
            fetch,
            lambda texts: [t for t in texts if text_allowed(t, self.policy)],
        )
        return kept[0] if kept else None

    # ----------------------------------------------------------------------- plumbing
    def _call(
        self,
        tool: str,
        args: dict[str, Any],
        fetch: Callable[[], tuple[R, bool]],
        keep: Callable[[R], R],
    ) -> R:
        self.meter.admit(tool, args)
        self._bump(calls=1)
        try:
            raw, cached = fetch()
        except (ToolError, FatalToolError) as e:
            self._bump(errors=1)
            fatal = isinstance(e, FatalToolError)
            self.meter.charge(tool, ok=False, error=str(e), fatal=fatal, meta=args)
            raise
        out = keep(raw)
        blocked = len(raw) - len(out)
        self._bump(blocked=blocked, cached=int(cached))
        meta = {**args, "returned": len(raw), "blocked": blocked}
        self.meter.charge(tool, cached=cached, meta=meta)
        return out

    def _query_ok(self, query: str) -> bool:
        if text_allowed(query, self.policy):
            return True
        self._bump(blocked=1)
        return False

    def _bump(self, **counts: int) -> None:
        with self._lock:
            for stat, n in counts.items():
                self.stats[stat] += n

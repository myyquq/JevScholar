"""Cost ledger: one JSON line per upstream attempt or tool charge, plus persistent campaign spend.

The ledger is the single source of truth for cost tables, task status and controller costs, so
each line records everything known at that moment: tokens split into cache miss / hit / output /
reasoning, list and billed cost, latency, attempt, and the gateway session that wrote it (a resumed
task's replayed rows can then be told apart from its abandoned first execution). Two prices are
kept on purpose. List price is what task budgets compare against, so a cache replay spends exactly
what the original run spent. Billed price is what the provider actually charges: DeepSeek's
off-peak discount, and $0 for replays, local servers and free tools. The run and campaign caps
limit it.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from jevscholar.config import Price, Upstream
from jevscholar.types import JUDGE_PREFIX, parse_tag
from jevscholar.util import connect, read_jsonl


def is_peak(dt: datetime) -> bool:
    """DeepSeek peak hours (billed at list price): Mon-Fri, UTC 01:00-04:00 and 06:00-10:00."""
    dt = dt.astimezone(UTC)
    return dt.weekday() < 5 and (1 <= dt.hour < 4 or 6 <= dt.hour < 10)


def scope(stage: str) -> Literal["gen", "judge"]:
    """Budget scope of a tag's stage: judging is capped separately from generation."""
    return "judge" if stage.startswith(JUDGE_PREFIX) else "gen"


def usage_tokens(usage: dict[str, Any] | None) -> dict[str, int]:
    """Split an OpenAI-shaped usage block into cache miss / hit, output and reasoning tokens."""
    u = usage or {}
    hit = u.get("prompt_cache_hit_tokens")  # DeepSeek; OpenAI only reports cached_tokens
    if hit is None:
        hit = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    miss = u.get("prompt_cache_miss_tokens", (u.get("prompt_tokens") or 0) - hit)
    out = u.get("completion_tokens") or 0
    reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return {"miss": miss, "hit": hit, "out": out, "reasoning": reasoning}


def llm_costs(
    price: Price, upstream: Upstream, tok: dict[str, int], at: datetime
) -> tuple[float, float]:
    """(list, billed) USD of one call; reasoning tokens are already counted inside `out`."""
    usd = (tok["miss"] * price.miss + tok["hit"] * price.hit + tok["out"] * price.out) / 1e6
    rate = 1.0 if is_peak(at) else upstream.offpeak_discount
    return usd, (usd * rate if upstream.billed else 0.0)


@dataclass
class Event:
    tag: str
    kind: Literal["llm", "tool"]
    name: str  # model alias or tool key
    model: str = ""  # upstream model id
    attempt: int = 0  # 1-based upstream attempt; 0 for cache hits, refusals and tools
    http_status: int | None = None
    finish_reason: str | None = None
    tokens: dict[str, int] = field(default_factory=dict)
    n: int = 1
    cost_usd: float = 0.0  # list price
    billed_usd: float = 0.0
    cached: bool = False
    latency_s: float = 0.0
    error: str | None = None
    notes: list[str] = field(default_factory=list)  # request rewrites (deviation register)
    meta: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def row(self) -> dict[str, Any]:
        parts = zip(("run", "bench", "system", "task", "stage"), parse_tag(self.tag), strict=True)
        rest = {k: v for k, v in dataclasses.asdict(self).items() if k not in ("tag", "ts")}
        return {"ts": self.ts.isoformat(), "tag": self.tag, **dict(parts), **rest}


class Ledger:
    """Append-only `ledger.jsonl`; every row is stamped with the writing gateway's session id."""

    def __init__(self, path: Path, session: str) -> None:
        self.path, self.session = path, session
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event) -> None:
        line = json.dumps(event.row() | {"session": self.session}, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def billed_totals(self) -> dict[str, float]:
        """Billed USD on file per scope, so a resumed run keeps counting against its caps."""
        totals = {"gen": 0.0, "judge": 0.0}
        for row in read(self.path):
            totals[scope(row["stage"])] += row["billed_usd"]
        return totals


def read(path: Path) -> list[dict[str, Any]]:
    """A ledger's rows (none if there is no ledger yet)."""
    return read_jsonl(path) if path.exists() else []


class SpendStore:
    """Billed USD per run, persistent across runs; the campaign hard stop compares the sum."""

    def __init__(self, path: Path) -> None:
        self.db = connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS spend (run TEXT PRIMARY KEY, usd REAL)")

    def add(self, run: str, usd: float) -> float:
        """Add billed spend for `run`; returns the new campaign total."""
        self.db.execute(
            "INSERT INTO spend VALUES (?, ?) ON CONFLICT(run) DO UPDATE SET "
            "usd = usd + excluded.usd",
            (run, usd),
        )
        return self.total()

    def total(self) -> float:
        return self.db.execute("SELECT COALESCE(SUM(usd), 0) FROM spend").fetchone()[0]

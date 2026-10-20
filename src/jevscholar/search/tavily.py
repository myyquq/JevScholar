"""Tavily web search and page extraction: cached, retried, quota errors fatal.

Search always sends `include_published_date` so the policy can date-filter hits: Tavily's own
`end_date` matches the publish *or last-updated* date and keeps undated pages, so it is only a
first cut. 401 (bad key), 432 (plan limit) and 433 (pay-as-you-go limit) can never succeed on
retry and stop the run; 429, 5xx and network errors are retried with backoff. The API key is
never part of a cache key, and an extract that yields no content is not frozen.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import Any

import requests

from jevscholar.search import FatalToolError, ToolError, store

API = "https://api.tavily.com"
FATAL_STATUS = {401, 432, 433}
TIMEOUT_S = 180
ATTEMPTS, BACKOFF_S = 5, 5.0


def search(query: str, k: int, end_date: date | None, key: str | None) -> tuple[list[dict], bool]:
    body: dict[str, Any] = {"query": query, "max_results": k, "include_published_date": True}
    if end_date is not None:
        body["end_date"] = end_date.isoformat()
    data, cached = _post("search", body, key)
    hits = [
        {
            "title": r.get("title") or "",
            "url": r["url"],
            "content": r.get("content") or "",
            "published_date": r.get("published_date") or None,
        }
        for r in data.get("results", [])
    ]
    return hits, cached


def extract(url: str, key: str | None) -> tuple[str | None, bool]:
    body = {"urls": [url], "extract_depth": "basic", "format": "markdown"}
    data, cached = _post("extract", body, key, keep=lambda d: bool(_content(d)))
    return _content(data), cached


def _content(data: dict[str, Any]) -> str | None:
    results = data.get("results") or [{}]
    return results[0].get("raw_content") or None


def _post(
    endpoint: str,
    body: dict[str, Any],
    key: str | None,
    keep: Callable[[dict[str, Any]], bool] = lambda d: True,
) -> tuple[dict[str, Any], bool]:
    db, cache_key = store.default_store(), {"api": "tavily", "endpoint": endpoint, "body": body}
    if (hit := db.get(cache_key)) is not None:
        return hit, True
    if not key:
        raise FatalToolError("no Tavily key (JEV_TAVILY_KEY)")
    last = ""
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(BACKOFF_S * 2 ** (attempt - 1))
        try:
            r = requests.post(
                f"{API}/{endpoint}",
                json=body,
                timeout=TIMEOUT_S,
                headers={"Authorization": f"Bearer {key}"},
            )
        except requests.RequestException as e:
            last = repr(e)
            continue
        if r.status_code == 200:
            data = r.json()
            return (db.put(cache_key, data) if keep(data) else data), False
        if r.status_code in FATAL_STATUS:
            raise FatalToolError(f"Tavily {endpoint} HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code != 429 and r.status_code < 500:
            raise ToolError(f"Tavily {endpoint} HTTP {r.status_code}: {r.text[:300]}")
        last = f"HTTP {r.status_code}"
    raise ToolError(f"Tavily {endpoint} failed after {ATTEMPTS} attempts ({last})")

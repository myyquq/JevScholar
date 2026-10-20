"""OpenAlex citation counts for DeepScholar's Document Importance, with upstream's matching rule.

Upstream (`eval/utils.get_citation_count_from_title`) takes the top hit of a `works?search=<title>`
query, accepts it at token Jaccard >= 0.8 and turns every error into "no match". This drop-in keeps
the query, the top-hit rule and its defaults, adds our API key, and gives errors their meaning:
- an HTTP error that every retry would repeat (400, 404, ...) is a definitive "no match", cached
  like any answer, as upstream's `except: return None`;
- 401/403 (bad key) can never succeed: FatalToolError, which trips the gateway's breaker;
- the daily-quota 429 (Retry-After of hours, until midnight UTC) and transient failures that
  outlast the retries raise ToolError and are never cached, so a later `jev eval` re-scores.
Upstream does not URL-encode the title, so a '&' or '#' splits or cuts its query; encoding the
parameters sends OpenAlex the same search text for every title upstream sends intact. Every
lookup, cached or not, is admitted and charged on the caller's gateway tag.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

import requests

from jevscholar.search import FatalToolError, ToolError, store
from jevscholar.search.client import Meter

API = "https://api.openalex.org/works"
MAILTO = "your@email.com"  # upstream default, kept so the query is identical
TOOL = "openalex_search"
TRANSIENT = frozenset({429, 500, 502, 503, 504})
ATTEMPTS = 4
QUOTA_WAIT_S = 60  # a 429 asking for a longer wait is the daily budget, not a burst limit


def jaccard(a: str, b: str) -> float:
    """Upstream `jaccard_similarity`: lower-cased whitespace tokens."""
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / len(ta | tb) if ta | tb else 0


def cited_by_count(
    title: str,
    mailto: str = MAILTO,
    similarity_threshold: float = 0.8,
    *,
    meter: Meter,
) -> int | None:
    """Citation count of the top OpenAlex hit for `title`, or None when it does not match."""
    meter.admit(TOOL)
    db, key = store.default_store(), {"api": "openalex", "search": title, "mailto": mailto}
    hit = db.get(key)
    cached = hit is not None
    if not cached:
        try:
            top = _top(title, mailto)
        except (ToolError, FatalToolError) as e:
            meter.charge(TOOL, ok=False, error=str(e), fatal=isinstance(e, FatalToolError))
            raise
        hit = db.put(key, {"top": top, "fetched": datetime.now(UTC).date().isoformat()})
    meter.charge(TOOL, cached=cached)
    top = hit["top"]
    if top is None or jaccard(title, top["display_name"] or "") < similarity_threshold:
        return None
    return top["cited_by_count"]


def _top(title: str, mailto: str) -> dict[str, Any] | None:
    """The top hit with upstream's `.get` defaults; None if there is none or the error is final."""
    params = {"search": title, "mailto": mailto}
    if key := os.environ.get("JEV_OPENALEX_KEY"):
        params["api_key"] = key
    last = ""
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(2**attempt)
        try:
            r = requests.get(API, params=params, timeout=30)
            results = r.json().get("results", []) if r.status_code == 200 else []
        except requests.RequestException as e:  # incl. invalid JSON; never the URL (holds the key)
            last = type(e).__name__
            continue
        if r.status_code == 200:
            if not results:
                return None
            top = results[0]
            return {
                "display_name": top.get("display_name", ""),
                "cited_by_count": top.get("cited_by_count", 0),
            }
        if r.status_code in (401, 403):
            raise FatalToolError(f"OpenAlex HTTP {r.status_code}: check JEV_OPENALEX_KEY")
        wait = r.headers.get("retry-after", "")
        if r.status_code == 429 and wait.isdigit() and int(wait) > QUOTA_WAIT_S:
            raise ToolError(f"OpenAlex daily quota exhausted (resets in {wait} s); re-run then")
        if r.status_code not in TRANSIENT:
            return None
        last = f"HTTP {r.status_code}"
    raise ToolError(f"OpenAlex failed after {ATTEMPTS} attempts ({last})")

"""arXiv Atom API backend: cached and rate-limited across every worker process.

arXiv asks for at most one request every 3 s over a single connection and answers bursts with
HTTP 429 (no Retry-After header) or 503; we measured 429s after ~10 requests at 3.1 s spacing and
a ~2 s median latency. So one `fcntl` lock file (ROOT/cache/arxiv.lock, holding the last request
time) serialises all processes: the holder waits until `JEV_ARXIV_INTERVAL` seconds (default 4)
have passed since the previous request, sends one, and on 429/503/timeouts backs off
exponentially (30 s doubling, at most 10 min per wait, `JEV_ARXIV_MAX_WAIT` s in total) without
releasing the lock, so a backoff pauses every worker. The 8-zero date lower bound LOTUS sends
returns HTTP 500, hence 199101010000. Successful responses are frozen in the search store; the
papers a search returns also seed the per-id lookup cache, so reading a found paper is free.
"""

from __future__ import annotations

import fcntl
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import IO, Any

import requests

from jevscholar.paths import CACHE
from jevscholar.search import ToolError, store
from jevscholar.search.policy import arxiv_id
from jevscholar.util import squash

API = "https://export.arxiv.org/api/query"
LOCK = CACHE / "arxiv.lock"
TIMEOUT_S = 180
BACKOFF_S, BACKOFF_CAP_S = 30.0, 600.0
RETRY_STATUS = {429, 503}
BATCH = 100  # ids per id_list request
ATOM = "{http://www.w3.org/2005/Atom}"


def search(query: str, k: int, before: datetime | None = None) -> tuple[list[dict], bool]:
    """Top-k papers by relevance, submitted at or before `before` (UTC, minute precision)."""
    if before is not None:
        query = f"({query}) AND submittedDate:[199101010000 TO {before.astimezone(UTC):%Y%m%d%H%M}]"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": k,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    db, key = store.default_store(), {"api": "arxiv", "params": params}
    if (hit := db.get(key)) is not None:
        return hit, True
    papers = db.put(key, parse(_get(params)))
    for p in papers:
        db.put(_id_key(p["id"]), p)
    return papers, False


def lookup(ids: list[str]) -> tuple[dict[str, dict], bool]:
    """Latest version of each paper, keyed by the requested ids (versions are ignored)."""
    db = store.default_store()
    wanted = {i: base for i in ids if (base := arxiv_id(i))}
    found = {b: p for b in set(wanted.values()) if (p := db.get(_id_key(b))) is not None}
    missing = sorted(set(wanted.values()) - found.keys())
    for start in range(0, len(missing), BATCH):
        batch = missing[start : start + BATCH]
        for p in parse(_get({"id_list": ",".join(batch), "max_results": len(batch)})):
            found[arxiv_id(p["id"])] = db.put(_id_key(p["id"]), p)
    return {i: found[b] for i, b in wanted.items() if b in found}, not missing


def parse(xml: bytes | str) -> list[dict]:
    """Atom feed -> paper dicts; the versioned id comes from the <id> URL, published = v1 date."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ToolError(f"arXiv returned malformed XML: {e}") from e
    papers = []
    for e in root.iter(f"{ATOM}entry"):
        url = e.findtext(f"{ATOM}id", "").strip()
        if "/abs/" not in url:  # error entry, e.g. http://arxiv.org/api/errors#incorrect_id_format
            raise ToolError(f"arXiv error: {_text(e, 'summary')}")
        vid = url.split("/abs/", 1)[1]
        papers.append(
            {
                "id": vid,
                "title": _text(e, "title"),
                "abstract": _text(e, "summary"),
                "published": _iso(_text(e, "published")),
                "updated": _iso(_text(e, "updated")),
                "authors": [_text(a, "name") for a in e.iter(f"{ATOM}author")],
                "categories": [c.get("term", "") for c in e.iter(f"{ATOM}category")],
                "url": f"http://arxiv.org/abs/{vid}",
            }
        )
    return papers


def _id_key(paper_id: str) -> dict[str, str]:
    return {"api": "arxiv", "id": arxiv_id(paper_id) or paper_id}


def _text(e: ET.Element, tag: str) -> str:
    return squash(e.findtext(f"{ATOM}{tag}"))


def _iso(s: str) -> str:
    return datetime.fromisoformat(s).astimezone(UTC).isoformat()


def _get(params: dict[str, Any]) -> bytes:
    interval = float(os.environ.get("JEV_ARXIV_INTERVAL", "4"))
    max_wait = float(os.environ.get("JEV_ARXIV_MAX_WAIT", "1200"))
    waited, delay = 0.0, BACKOFF_S
    with _locked() as lock:
        while True:
            _wait_turn(lock, interval)
            try:
                r = requests.get(
                    API, params=params, timeout=TIMEOUT_S, headers={"User-Agent": "jevscholar/0.1"}
                )
            except requests.RequestException as e:
                last = repr(e)
            else:
                if r.status_code == 200:
                    return r.content
                if r.status_code not in RETRY_STATUS:
                    raise ToolError(f"arXiv HTTP {r.status_code}: {r.text[:300]}")
                last = f"HTTP {r.status_code}"
            if waited + delay > max_wait:
                raise ToolError(f"arXiv unavailable after {waited:.0f} s of backoff ({last})")
            time.sleep(delay)
            waited, delay = waited + delay, min(2 * delay, BACKOFF_CAP_S)


@contextmanager
def _locked() -> Iterator[IO[str]]:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as f:  # closing the file releases the lock
        fcntl.flock(f, fcntl.LOCK_EX)
        yield f


def _wait_turn(lock: IO[str], interval: float) -> None:
    """Sleep until `interval` s after the previous request (any process), then record now."""
    lock.seek(0)
    last = float(lock.read() or 0)
    time.sleep(max(0.0, last + interval - time.time()))
    lock.seek(0)
    lock.truncate()
    lock.write(repr(time.time()))
    lock.flush()

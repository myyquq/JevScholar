"""Full text of the writer's arXiv documents, for the reading revisions (controller/state.py:
READS).

`fetch` downloads each document's PDF at the version its URL names (the version of the frozen
metadata snapshot, so nothing written after it), from export.arxiv.org at arXiv's rate for
automated clients (one request per DELAY_S), and keeps its extracted text (pypdf) in
cache/fulltext/<id>.txt; a document that cannot be had is noted in <id>.err with the reason and
not retried. Fetching is the only networked step: a run reads the cache only (`excerpt`), so a
document without text there keeps its abstract.

`excerpt` is the part the writer reads: from the introduction's heading (the start where none is
found), whitespace joined, cut to EXCERPT_CHARS at a word boundary.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from jevscholar import paths

log = logging.getLogger(__name__)

DIR = paths.CACHE / "fulltext"
PDF = "https://export.arxiv.org/pdf/{id}"
DELAY_S = 3.0  # arXiv's rate for automated clients
TIMEOUT_S = 60
EXCERPT_CHARS = 3000
MIN_CHARS = 500  # less extracted text than this: a scan or a failed extraction
ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(?P<id>[^\s?#]+?)(?:\.pdf)?$")
INTRO = re.compile(r"(?:^|\n)\s*(?:\d+\.?|[IVX]+\.?)?\s*(?:Introduction|INTRODUCTION)\b")


def arxiv_id(url: str) -> str | None:
    """The versioned arXiv id an arXiv URL names (None for another URL)."""
    m = ARXIV_ID.search(url.strip())
    return m["id"] if m else None


def path(aid: str, suffix: str = ".txt") -> Path:
    return DIR / (aid.replace("/", "_") + suffix)


def excerpt(url: str, chars: int = EXCERPT_CHARS) -> str | None:
    """The cached full text's excerpt for an arXiv URL, None if there is none."""
    aid = arxiv_id(url)
    if aid is None or not path(aid).exists():
        return None
    text = path(aid).read_text(encoding="utf-8")
    m = INTRO.search(text)
    body = " ".join(text[m.start() if m else 0 :].split())
    if len(body) <= chars:
        return body
    cut = body[:chars]
    return (cut[: cut.rfind(" ")] if " " in cut else cut) + " ..."


def extract(pdf: bytes) -> str:
    """A PDF's text, line-broken words joined; characters UTF-8 cannot hold (lone surrogates
    from math fonts) dropped."""
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font warnings on every page
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def download(aid: str) -> bytes:
    import requests

    r = requests.get(
        PDF.format(id=aid), timeout=TIMEOUT_S, headers={"User-Agent": "jevscholar/0.1"}
    )
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError(f"not a PDF ({r.headers.get('content-type', '?')})")
    return r.content


def fetch(urls: Iterable[str], delay_s: float = DELAY_S) -> Counter:
    """Fetch and keep the full text of every arXiv URL not cached or noted yet; returns the
    count of each outcome (cached, fetched, failed, not_arxiv)."""
    DIR.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for url in dict.fromkeys(urls):
        aid = arxiv_id(url)
        if aid is None:
            counts["not_arxiv"] += 1
            continue
        if path(aid).exists() or path(aid, ".err").exists():
            counts["cached"] += 1
            continue
        try:
            text = extract(download(aid))
            if len(text.strip()) < MIN_CHARS:
                raise ValueError(f"only {len(text.strip())} characters extracted")
            path(aid).write_text(text, encoding="utf-8")
            counts["fetched"] += 1
        except Exception as e:  # any failure: the document keeps its abstract
            path(aid, ".err").write_text(f"{type(e).__name__}: {e}"[:500], encoding="utf-8")
            counts["failed"] += 1
            log.warning("%s: %s", aid, e)
        done = counts["fetched"] + counts["failed"]
        if done % 50 == 0:
            log.info("full text: %s", dict(counts))
        time.sleep(delay_s)
    return counts


def run_docs(run_dir: Path) -> list[str]:
    """Every writer document of a run: the docs of each task's state.json, in order."""
    urls: list[str] = []
    for state in sorted(run_dir.glob("**/tasks/*/state.json")):
        urls += json.loads(state.read_text())["docs"]
    return list(dict.fromkeys(urls))

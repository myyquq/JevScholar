"""Leakage rules: what a task's tools may return, applied identically for every system.

The rules are deliberately conservative and need no extra API calls:
- papers must be published (v1) strictly before the cutoff and be neither a blocked id nor carry a
  blocked title;
- a URL is refused when it is blocked or carries an arXiv id that is blocked or was assigned in the
  cutoff month or later (abs/pdf/html mirrors, alphaxiv, HF papers, ...);
- a text (query, search-hit title and snippet, fetched page) is refused when it mentions a blocked
  title or such an arXiv id: the DRB2 prompt reveals its source article's title, and a page that
  links the target or later papers leaks them without naming them;
- web hits dated on or after the cutoff are dropped; undated hits follow `policy.undated_web`.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

from jevscholar.types import AccessPolicy, Source

# New-style ids (YYMM.NNNNN) anywhere, but not inside a longer number such as a DOI suffix.
_NEW_ID = re.compile(r"(?<!\d)(?<!\d\.)(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5})(?:v\d+)?(?!\d)")
# Old-style ids (archive/YYMMNNN) only where they are unambiguous: bare, after arXiv:, or arxiv.org.
_OLD_ID = re.compile(
    r"(?:^|arxiv:|arxiv\.org/(?:abs|pdf|html)/)([a-z][a-z\-]*(?:\.[a-z]{2})?/\d{7})(?:v\d+)?(?!\d)",
    re.IGNORECASE,
)


def canonical_url(url: str) -> str:
    """Normalise a source identity for blocking/citation matching, not display.

    arXiv abs/pdf/html and ar5iv mirror URLs share an identity regardless of version; DOI resolver
    aliases share an identity. Otherwise fragments and tracking parameters are dropped while
    substantive queries are kept. Arbitrary mirrors or copies cannot be identified.
    """
    value = str(url).strip()
    if value.lower().startswith("doi:"):
        value = "https://doi.org/" + value[4:].strip()
    elif re.fullmatch(r"10\.\d{4,9}/\S+", value, flags=re.I):
        value = "https://doi.org/" + value
    parts = urlsplit(value)
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = unquote(parts.path).rstrip("/")
    if host in {"arxiv.org", "export.arxiv.org", "ar5iv.org", "ar5iv.labs.arxiv.org"}:
        match = re.fullmatch(r"/(?:abs|pdf|html)/(.+?)(?:\.pdf)?", path, flags=re.I)
        if match:
            identifier = re.sub(r"v\d+$", "", match.group(1), flags=re.I)
            return "https://arxiv.org/abs/" + identifier.lower()
    if host in {"doi.org", "dx.doi.org"}:
        return "https://doi.org/" + path.lstrip("/").lower()
    if not host:
        return value
    if parts.port and parts.port not in {80, 443}:
        host += f":{parts.port}"
    query = urlencode(
        sorted(
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
        )
    )
    return f"https://{host}{path}" + (f"?{query}" if query else "")


def arxiv_id(url_or_id: str) -> str | None:
    """Versionless arXiv id in an id or URL: 'arXiv:2408.14717v1' -> '2408.14717'."""
    if m := _NEW_ID.search(url_or_id):
        return m[1]
    m = _OLD_ID.search(url_or_id.strip())
    return m[1].lower() if m else None


def norm_title(s: str) -> str:
    """NFKC + casefold, word runs joined by one space (Unicode-aware, so CJK titles survive)."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))


def floor_date(s: str) -> datetime:
    """'2021' -> 2021-01-01, '2021-3' -> 2021-03-01, '2021-11-3' -> 2021-11-03 (UTC midnight)."""
    m = re.fullmatch(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", s.strip())
    if not m:
        raise ValueError(f"not a YYYY[-M[-D]] date: {s!r}")
    return datetime(int(m[1]), int(m[2] or 1), int(m[3] or 1), tzinfo=UTC)


def text_allowed(text: str, policy: AccessPolicy) -> bool:
    """False if `text` mentions a blocked title or a blocked or post-cutoff arXiv id."""
    return not _mentions_blocked(text, policy) and _ids_allowed(text, policy)


def url_allowed(url: str, policy: AccessPolicy) -> bool:
    """False for a blocked URL or one carrying a blocked or post-cutoff arXiv id."""
    return canonical_url(url) not in _blocklists(policy)[1] and _ids_allowed(url, policy)


def source_allowed(source: Source, policy: AccessPolicy) -> bool:
    """Export audit for a cited source. A dated arXiv source is judged by its v1 date (the
    month rule of url_allowed would also reject same-month papers published before the cutoff);
    everything else falls back to url_allowed."""
    aid = arxiv_id(source.url)
    if aid is None or not source.date or policy.cutoff is None:
        return url_allowed(source.url, policy)
    published = datetime.fromisoformat(source.date[:10]).replace(tzinfo=policy.cutoff.tzinfo)
    return (
        aid not in _blocklists(policy)[0]
        and canonical_url(source.url) not in _blocklists(policy)[1]
        and published < policy.cutoff
    )


def filter_papers(papers: list[dict], policy: AccessPolicy) -> list[dict]:
    ids, cutoff = _blocklists(policy)[0], policy.cutoff
    return [
        p
        for p in papers
        if arxiv_id(p["id"]) not in ids
        and not _mentions_blocked(p["title"], policy)
        and (cutoff is None or datetime.fromisoformat(p["published"]) < cutoff)
    ]


def filter_web(hits: list[dict], policy: AccessPolicy) -> list[dict]:
    return [
        h
        for h in hits
        if url_allowed(h["url"], policy)
        and text_allowed(f"{h['title']}\n{h['content']}", policy)
        and _dated_ok(h["published_date"], policy)
    ]


# --------------------------------------------------------------------------- helpers
@functools.cache
def _blocklists(policy: AccessPolicy) -> tuple[frozenset[str], frozenset[str], tuple[str, ...]]:
    """Block lists in comparable form: versionless ids, canonical URLs, normalised titles."""
    return (
        frozenset(arxiv_id(i) or i for i in policy.blocked_arxiv),
        frozenset(map(canonical_url, policy.blocked_urls)),
        tuple(t for t in map(norm_title, policy.blocked_titles) if t),
    )


def _mentions_blocked(text: str, policy: AccessPolicy) -> bool:
    normed = norm_title(text)
    return any(t in normed for t in _blocklists(policy)[2])


def _ids_allowed(text: str, policy: AccessPolicy) -> bool:
    """No arXiv id anywhere in `text` is blocked or was assigned in the cutoff month or later."""
    ids, cutoff = _blocklists(policy)[0], policy.cutoff
    found = [m[1] for m in _NEW_ID.finditer(text)] + [m[1].lower() for m in _OLD_ID.finditer(text)]
    return not any(
        aid in ids or (cutoff is not None and _month(aid) >= (cutoff.year, cutoff.month))
        for aid in found
    )


def _month(aid: str) -> tuple[int, int]:
    """(year, month) in which an arXiv id was assigned."""
    digits = aid.rsplit("/", 1)[-1]
    yy, mm = int(digits[:2]), int(digits[2:4])
    return (1900 + yy if yy >= 91 else 2000 + yy), mm


def _dated_ok(value: str | None, policy: AccessPolicy) -> bool:
    if policy.cutoff is None:
        return True
    when = _parse_when(value) if value else None
    return policy.undated_web == "keep" if when is None else when < policy.cutoff


def _parse_when(value: str) -> datetime | None:
    """ISO 8601 or RFC 2822 (Tavily uses both); naive values are taken as UTC."""
    for parse in (datetime.fromisoformat, parsedate_to_datetime):
        try:
            when = parse(value.strip())
        except ValueError:
            continue
        return when if when.tzinfo else when.replace(tzinfo=UTC)
    return None

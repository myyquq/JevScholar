"""One citation and rendering rule for every system's reports and exports.

Every system ends with `finalize`, so the citation rule is identical across them: "[n]" markers
renumbered by first appearance, uncited sources dropped, model-written reference lists removed,
and one source identity (`source_id`). Exporters then re-render citations and references from
Report.sources in each benchmark's official format, so systems differ only in their body text.
Word caps cut at a sentence boundary and never count the rendered reference list; a CJK character
counts as 1/1.6 word, so one cap fits every language.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Sequence
from typing import Any

from jevscholar.search.policy import canonical_url
from jevscholar.types import Report, Source
from jevscholar.util import squash

CJK_CHARS_PER_WORD = 1.6
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"  # kana and Han, not Hangul
# "[3]", "[1, 2]", "[1-3]", "[2; 4–6]"; an escaped "\[1, 2\]" never matches (its "]" is "\]").
CITE = re.compile(r"\[(\d+(?:\s*[,;\-–—]\s*\d+)*)\]")
_MARKER = re.compile(r"([ \t]*)" + CITE.pattern)  # a marker with the space before it
_RUN = re.compile(rf"(?:[ \t]*{CITE.pattern})+")  # adjacent markers ("[2][1]", "[1] [3]")
_TOKEN = re.compile(rf"(?P<cite>{CITE.pattern})|(?P<cjk>[{_CJK}])|[^\s{_CJK}]+")
_BOUNDARY = re.compile(  # a sentence end, with any citations glued to it ("claim.[3] Next")
    r"(?<!\bal)(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bcf)(?<!\bFig)(?<!\bEq)"  # et al., e.g., ...
    r"[.!?][\"'”’)\]]*(?:\[\d+\])*(?=\s)|[。！？][」』）”\"]*(?:\[\d+\])*|\n"
)
_DANGLING_HEADINGS = re.compile(r"(?:\n+#{1,6} [^\n]*)+\Z")
_EMPHASIS = r"(?:\*{1,2}|_{1,2})"
_REF_HEADING = re.compile(
    rf"^[ \t]*(?P<hash>#{{1,6}})?[ \t]*(?P<em>{_EMPHASIS})?[ \t]*"
    r"(?:(?:\d+[.)、]?|[IVXLC]+\.|[一二三四五六七八九十]+[、.．])[ \t]*)?"  # 5. / VII. / 六、
    r"(?:references?(?: list| and notes)?|key references|bibliography|works cited"
    r"|sources(?: cited)?|citations|参考文献(?:列表)?|参考资料|参考来源|参考书目|引用文献|引用来源)"
    rf"[ \t]*[:：]?[ \t]*{_EMPHASIS}?[ \t]*[:：]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_ENTRY = re.compile(r"\s*(?:\[\d+\]|\d+[.)]|[-*•]|\Z)")  # what follows a bare heading's line
_HEADING = re.compile(r"^#{1,6} ", re.MULTILINE)
# arXiv abs/pdf links on any arxiv.org host; the id keeps its version, as the DeepScholar eval
# parser (`arxiv\.org/abs/([^)\s]+)`) keys documents.
ARXIV_URL = re.compile(r"(?<![\w-])arxiv\.org/(?:abs|pdf)/([^)\s?#]+?)(?:\.pdf)?/?(?=[)\s?#]|$)")


# --------------------------------------------------------------------------- citations
def source_id(url: str) -> str:
    """THE source identity for every system: the versioned arXiv id of an arxiv.org abs/pdf
    link, else the canonical URL."""
    m = ARXIV_URL.search(url)
    return m[1] if m else canonical_url(url)


def _expand(group: str) -> list[int]:
    """'1, 3-5' -> [1, 3, 4, 5]."""
    nums: list[int] = []
    for part in re.split(r"[,;]", group):
        lo, _, hi = re.sub(r"[–—]", "-", part).partition("-")
        nums += range(int(lo), int(hi or lo) + 1)
    return nums


def _renumber(text: str, new: Callable[[int], int | None]) -> str:
    """Map every marker's numbers through `new` (None deletes one). Grouped markers become
    adjacent single ones; a marker left empty disappears with the space before it, and the
    first marker kept in a run of markers takes over the run's leading space ("x [9][1]." ->
    "x [1].", "x [9]." -> "x.")."""

    def run(m: re.Match[str]) -> str:
        pieces, out = list(_MARKER.finditer(m[0])), ""
        for p in pieces:
            nums = dict.fromkeys(k for n in _expand(p[2]) if (k := new(n)) is not None)
            if nums:
                out += (p[1] if out else pieces[0][1]) + "".join(f"[{k}]" for k in nums)
        return out

    return _RUN.sub(run, text)


def finalize(text: str, sources: Sequence[Source], **meta: Any) -> Report:
    """Canonical Report: references stripped, citations renumbered by first appearance
    ("[3] ... [1, 3]" -> "[1] ... [2][1]"), numbers outside 1..len(sources) deleted, uncited
    sources dropped."""
    order: dict[int, int] = {}  # original number -> new number

    def number(n: int) -> int | None:
        return order.setdefault(n, len(order) + 1) if 0 < n <= len(sources) else None

    body = _renumber(strip_references(text), number).strip()
    cited = tuple(sources[n - 1] for n in order)
    return Report(text=body, sources=cited, meta={**meta, "words": word_count(body)})


def drop_sources(report: Report, keep: Callable[[Source], bool]) -> tuple[Report, int]:
    """Remove the sources `keep` rejects and their markers, renumbering the rest; returns the
    new report and the number of sources dropped."""
    kept = [n for n, s in enumerate(report.sources, 1) if keep(s)]
    text = _renumber(report.text, {n: k for k, n in enumerate(kept, 1)}.get)
    sources = tuple(report.sources[n - 1] for n in kept)
    return dataclasses.replace(report, text=text, sources=sources), len(report.sources) - len(kept)


def _cited(text: str, sources: tuple[Source, ...]) -> list[int]:
    """1-based numbers of the sources the text cites, in numeric order."""
    return sorted({n for g in CITE.findall(text) for n in _expand(g) if 0 < n <= len(sources)})


# --------------------------------------------------------------------------- text rules
def strip_references(text: str) -> str:
    """Drop a trailing model-written reference section: its heading ("## References",
    "References:", "**References**", "## 六、参考文献", ...), everything after it and a rule
    ("---") before it, unless another markdown section follows (then it is not trailing). A bare
    heading line (no "#" or emphasis) counts only when a reference entry or nothing follows."""
    heads = [
        m
        for m in _REF_HEADING.finditer(text)
        if m["hash"] or m["em"] or _ENTRY.match(text, m.end())
    ]
    if not heads or _HEADING.search(text, heads[-1].end()):
        return text
    return re.sub(r"\n[ \t]*-{3,}\Z", "", text[: heads[-1].start()].rstrip()).rstrip()


def _words(text: str) -> list[tuple[re.Match[str], float]]:
    """Countable tokens with their weight: a CJK character is 1/1.6 word; "[n]" markers and
    bare punctuation are not words."""
    return [
        (m, 1 / CJK_CHARS_PER_WORD if m["cjk"] else 1.0)
        for m in _TOKEN.finditer(text)
        if m["cjk"] or (not m["cite"] and any(c.isalnum() for c in m[0]))
    ]


def word_count(text: str) -> int:
    """Words in `text`, a CJK character counting as 1/1.6 word (one rule for every language)."""
    return round(sum(w for _, w in _words(text)))


def cap_words(text: str, n: int) -> str:
    """The longest prefix of at most `n` words that ends at a sentence boundary (. ! ? 。 ！ ？,
    citations glued to them, or a line break); a single over-long sentence is cut at a word."""
    total, last = 0.0, 0
    for m, w in _words(text):
        total += w
        if total > n:
            break
        last = m.end()
    else:
        return text
    cuts = [b.end() for b in _BOUNDARY.finditer(text, 0, m.start())]
    end = cuts[-1] if cuts else last
    return _DANGLING_HEADINGS.sub("", text[:end].rstrip())


def _with_references(body: str, sources: tuple[Source, ...], entry: Callable[[Source], str]) -> str:
    cited = _cited(body, sources)
    if not cited:
        return body
    refs = "\n".join(f"[{n}] {entry(sources[n - 1])}" for n in cited)
    return f"{body}\n\n## References\n{refs}"


# --------------------------------------------------------------------------- benchmarks
def _label(s: Source) -> str:
    label = f"{s.authors[0]} {s.date}" if s.authors else s.title or s.id
    return squash(label).replace("[", "(").replace("]", ")")


def deepscholar_intro(report: Report) -> tuple[str, list[dict[str, str]]]:
    """DeepScholar `intro.md` text and `paper.csv` rows (uncapped body).

    Citations of arXiv sources (by URL, the eval parser's rule, whatever the system) become
    `[<FirstAuthor> <date>](http://arxiv.org/abs/<versioned id>)`, the only form the official
    parser counts, and their rows carry the snippet; citations of other sources are dropped."""
    rows: dict[str, dict[str, str]] = {}

    def link(m: re.Match[str]) -> str:
        space, n, more = m[1], int(m[2]), m[3] is not None
        s = report.sources[n - 1] if 0 < n <= len(report.sources) else None
        arxiv = s and ARXIV_URL.search(s.url)
        if not arxiv:
            return space if more else ""  # "x [web] [1]" -> "x [1]", "x [web]." -> "x."
        rows.setdefault(arxiv[1], {"id": arxiv[1], "title": squash(s.title), "snippet": s.snippet})
        return f"{space}[{_label(s)}](http://arxiv.org/abs/{arxiv[1]})"

    text = re.sub(r"([ \t]*)\[(\d+)\](?=([ \t]*\[\d+\])?)", link, strip_references(report.text))
    return text, list(rows.values())


def _author_year_title(s: Source) -> str:
    """'First Author et al. (Year). Title.'; without authors 'Title (Year).'."""
    title = squash(s.title or s.url).rstrip(".")
    year = f" ({s.date[:4]})" if s.date else ""
    if not s.authors:
        return f"{title}{year}."
    return f"{s.authors[0]}{' et al.' if len(s.authors) > 1 else ''}{year}. {title}."


def researchqa_answer(report: Report, cap: int = 300) -> str:
    """Body capped at `cap` words, then "[n] First Author (Year). Title." for each cited source:
    the judge sees only this text, so citation rubrics ("Does the response cite Smith (2016)?")
    need author, year and title written out."""
    body = cap_words(strip_references(report.text), cap)
    return _with_references(body, report.sources, _author_year_title)


def drb2_report(report: Report, cap: int = 2000) -> str:
    """Body capped at `cap` words, then "[n] Title. URL" for each cited source (the judge checks
    cited URLs against the task's blocked list)."""
    body = cap_words(strip_references(report.text), cap)
    return _with_references(
        body, report.sources, lambda s: f"{squash(s.title)}. {s.url}" if s.title else s.url
    )

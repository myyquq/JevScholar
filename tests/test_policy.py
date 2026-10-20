"""Leakage policy: identity normalisers, the paper and web-hit filters, and the text/URL rules."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from jevscholar.search.policy import (
    arxiv_id,
    canonical_url,
    filter_papers,
    filter_web,
    floor_date,
    source_allowed,
    text_allowed,
    url_allowed,
)
from jevscholar.types import AccessPolicy

CUTOFF = datetime(2025, 6, 1, tzinfo=UTC)
TITLE = "Attention Is All You Need"
DS = AccessPolicy(  # DeepScholar-style
    cutoff=CUTOFF,
    blocked_arxiv=frozenset({"2505.04321"}),
    blocked_titles=(TITLE,),
    undated_web="drop",
)
RQA = AccessPolicy(cutoff=CUTOFF)  # ResearchQA-style: undated web kept
DRB2 = AccessPolicy(
    blocked_urls=frozenset({"https://example.com/report"}),
    blocked_titles=(TITLE,),
    blocked_arxiv=frozenset({"1706.03762"}),
)


@pytest.mark.parametrize(
    ("fn", "arg", "expected"),
    [
        (arxiv_id, "arXiv:2408.14717v1", "2408.14717"),
        (arxiv_id, "http://arxiv.org/abs/2408.14717v2", "2408.14717"),
        (arxiv_id, "https://arxiv.org/pdf/2408.14717v2.pdf", "2408.14717"),
        (arxiv_id, "https://www.alphaxiv.org/abs/2408.14717", "2408.14717"),
        (arxiv_id, "https://huggingface.co/papers/2408.14717", "2408.14717"),
        (arxiv_id, "https://doi.org/10.48550/arXiv.2408.14717", "2408.14717"),
        (arxiv_id, "cs/0112017v2", "cs/0112017"),
        (arxiv_id, "https://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
        (arxiv_id, "https://doi.org/10.1145/3292500.3330919", None),  # inside a longer number
        (arxiv_id, "https://example.com/story/2105123", None),  # old-style shape outside arXiv
        (arxiv_id, "https://example.com/v2021.12345", None),  # month 21
        (
            canonical_url,
            "http://arxiv.org/pdf/2408.14717v2.pdf",
            "https://arxiv.org/abs/2408.14717",
        ),
        (
            canonical_url,
            "https://ar5iv.labs.arxiv.org/html/2408.14717",
            "https://arxiv.org/abs/2408.14717",
        ),
        (
            canonical_url,
            "https://export.arxiv.org/abs/2408.14717v1/",
            "https://arxiv.org/abs/2408.14717",
        ),
        (canonical_url, "doi:10.1000/ABC", "https://doi.org/10.1000/abc"),
        (canonical_url, "https://dx.doi.org/10.1000/abc", "https://doi.org/10.1000/abc"),
        (
            canonical_url,
            "http://www.Example.com/a/?utm_source=x&b=2&a=1#f",
            "https://example.com/a?a=1&b=2",
        ),
        (floor_date, "2021", datetime(2021, 1, 1, tzinfo=UTC)),
        (floor_date, "2021-3", datetime(2021, 3, 1, tzinfo=UTC)),
        (floor_date, "2021-11-3", datetime(2021, 11, 3, tzinfo=UTC)),
        (floor_date, " 2021-03-04 ", datetime(2021, 3, 4, tzinfo=UTC)),
        *((floor_date, s, ValueError) for s in ["21", "2021/3", "2021-03-04-05", "March 2021"]),
    ],
)
def test_normalisers(fn: Callable[[str], Any], arg: str, expected: Any) -> None:
    if expected is ValueError:
        with pytest.raises(ValueError):
            fn(arg)
    else:
        assert fn(arg) == expected


def _paper(pid: str, published: str, title: str = "A paper") -> dict:
    return {"id": pid, "title": title, "published": published}


def _hit(url: str, published: str | None = None, title: str = "Page", content: str = "") -> dict:
    return {"url": url, "title": title, "content": content, "published_date": published}


BLOG = "https://blog.example.org/x"


@pytest.mark.parametrize(
    ("keep", "item", "policy", "kept"),
    [
        (filter_papers, _paper("2505.00001v1", "2025-05-31T23:59:00+00:00"), DS, True),
        (filter_papers, _paper("2505.00001v1", "2025-06-01T00:00:00+00:00"), DS, False),  # "<"
        (filter_papers, _paper("2505.04321v3", "2025-05-02T00:00:00+00:00"), DS, False),  # target
        (
            filter_papers,
            _paper("2401.00001v1", "2024-01-01T00:00:00+00:00", "Attention is all you need!"),
            DS,
            False,
        ),
        (
            filter_papers,
            _paper("2601.00001v1", "2026-01-05T00:00:00+00:00"),
            DRB2,
            True,
        ),  # no cutoff
        (filter_papers, _paper("1706.03762v7", "2017-06-12T00:00:00+00:00"), DRB2, False),
        (
            filter_web,
            _hit("https://arxiv.org/abs/2505.00001"),
            RQA,
            True,
        ),  # before the cutoff month
        (filter_web, _hit("https://www.alphaxiv.org/abs/2506.00002"), RQA, False),  # cutoff month
        (filter_web, _hit("https://huggingface.co/papers/2505.04321", "2025-05-03"), DS, False),
        (filter_web, _hit("https://arxiv.org/pdf/1706.03762v7"), DRB2, False),  # target, no cutoff
        (filter_web, _hit("https://arxiv.org/abs/2601.00001"), DRB2, True),
        (filter_web, _hit("http://www.example.com/report/?utm_source=t"), DRB2, False),  # URL alias
        (filter_web, _hit(BLOG, title=f"{TITLE} | OpenReview"), DRB2, False),
        (filter_web, _hit(BLOG, "2025-05-01", content=f"Recap of {TITLE.lower()}."), DS, False),
        (filter_web, _hit(BLOG, "2025-05-01", content="See arXiv:2505.04321v2."), DS, False),
        (
            filter_web,
            _hit(BLOG, "2025-05-01", content="Sequel: arxiv.org/abs/2506.00002"),
            RQA,
            False,
        ),
        (filter_web, _hit(BLOG, "2025-05-01", content="Builds on 2505.00001."), RQA, True),
        (filter_web, _hit(BLOG, "2025-06-01"), RQA, False),  # dated on the cutoff
        (filter_web, _hit(BLOG, "Sat, 31 May 2025 10:00:00 GMT"), RQA, True),
        (filter_web, _hit(BLOG, "2025-05-30T08:00:00Z"), DS, True),
        (filter_web, _hit(BLOG), RQA, True),  # undated: keep
        (filter_web, _hit(BLOG), DS, False),  # undated: drop
        (filter_web, _hit(BLOG, "not a date"), DS, False),  # unparseable = undated
        (filter_web, _hit(BLOG, "2026-01-01"), DRB2, True),  # no cutoff: dates ignored
    ],
)
def test_filters(keep: Callable, item: dict, policy: AccessPolicy, kept: bool) -> None:
    assert keep([item], policy) == ([item] if kept else [])


def test_text_and_url_rules() -> None:
    assert not text_allowed("ATTENTION is all you need: transformer survey", DRB2)
    assert text_allowed("transformer attention survey", DRB2)
    zh = AccessPolicy(blocked_titles=("大语言模型智能体综述", "!!!"))  # "!!!" normalises to ""
    assert not text_allowed("请写一篇关于大语言模型智能体综述的报告", zh)
    assert text_allowed("大语言模型", zh)
    # Every arXiv id in a text or URL counts: blocked ones always, later ones only with a cutoff.
    assert not text_allowed("as shown in arXiv:1706.03762v7", DRB2)
    assert text_allowed("follow-up work (2506.00002) improves on it", DRB2)
    assert not text_allowed("follow-up work (2506.00002) improves on it", RQA)
    assert not text_allowed("old arXiv:hep-th/9901001, new 2506.00002", RQA)
    assert text_allowed("2505.00001, arXiv:cs/0112017 and doi 10.1145/3292500.3330919", RQA)
    assert not url_allowed("https://example.com/cmp?a=2401.00001&b=2506.00002", RQA)
    assert url_allowed("https://example.com/cmp?a=2401.00001", RQA)
    assert not url_allowed("https://example.com/report#top", DRB2)


def test_source_allowed_uses_v1_date_for_arxiv_sources():
    from jevscholar.types import Source

    policy = AccessPolicy(
        cutoff=datetime(2025, 6, 3, tzinfo=UTC), blocked_arxiv=frozenset({"2506.00001"})
    )
    early = Source(id="2506.00002v1", url="http://arxiv.org/abs/2506.00002v1", date="2025-06-01")
    late = Source(id="2506.00003v1", url="http://arxiv.org/abs/2506.00003v1", date="2025-06-03")
    target = Source(id="2506.00001v1", url="http://arxiv.org/abs/2506.00001v1", date="2025-05-01")
    undated = Source(id="2506.00004v1", url="http://arxiv.org/abs/2506.00004v1")
    assert source_allowed(early, policy)  # same month, published before the cutoff
    assert not source_allowed(late, policy) and not source_allowed(target, policy)
    assert not source_allowed(undated, policy)  # falls back to the conservative month rule

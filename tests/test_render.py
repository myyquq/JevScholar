"""render.py: the one citation and export rule (finalize, identity, text rules, renderers)."""

from __future__ import annotations

import pytest

from jevscholar.render import (
    cap_words,
    deepscholar_intro,
    drb2_report,
    drop_sources,
    finalize,
    researchqa_answer,
    source_id,
    strip_references,
    word_count,
)
from jevscholar.types import Report, Source

BODY = "Transformers rely on attention [1]. Retrieval helps [2]."
REFS = "\n[1] Vaswani et al. Attention.\n2. Lewis et al. RAG."
VASWANI = Source(
    id="1706.03762v7",
    url="http://arxiv.org/abs/1706.03762v7",
    title="Attention Is\n  All You Need",
    authors=("Ashish Vaswani", "Noam Shazeer"),
    date="2017-06-12",
    snippet="The dominant sequence transduction models ...",
)
BLOG = Source(
    id="https://example.com/rag", url="https://example.com/rag", title="A blog on RAG", date="2012"
)
LEWIS = Source(
    id="2005.11401v4",
    url="https://arxiv.org/pdf/2005.11401v4.pdf",
    title="Retrieval-Augmented Generation",
    date="2020-05-22",
    snippet="Large models",
)


def test_finalize_renumbers_and_drop_sources_removes_markers() -> None:
    s1, s2, s3, s4 = (Source(id=f"s{i}", url=f"https://e.com/{i}") for i in range(1, 5))
    text = (
        "A [3]. B [1, 3]. C [2-3]. D [9]. E \\[1, 2\\] stays.\n\n"
        "## References\n[1] s1\n[2] s2\n[4] s4\n"
    )
    r = finalize(text, [s1, s2, s3, s4], status="ok")
    assert r.text == "A [1]. B [2][1]. C [3][1]. D. E \\[1, 2\\] stays."
    assert r.sources == (s3, s1, s2)  # s4 was cited only in the stripped reference list
    assert r.meta == {"status": "ok", "words": 8}  # the escaped \[1, 2\] is text

    kept, dropped = drop_sources(r, lambda s: s is not s1)
    assert (kept.text, kept.sources, dropped) == (
        "A [1]. B [1]. C [2][1]. D. E \\[1, 2\\] stays.",
        (s3, s2),
        1,
    )

    assert [
        source_id(u)
        for u in (
            "https://arxiv.org/abs/2001.00001v2",
            "http://export.arxiv.org/pdf/2001.00001v2.pdf",
            "https://arxiv.org/abs/hep-th/9901001v1?context=x",
            "https://alphaxiv.org/abs/2001.00001",
            "https://Example.com/post/?utm_source=x",
        )
    ] == [
        "2001.00001v2",
        "2001.00001v2",
        "hep-th/9901001v1",
        "https://alphaxiv.org/abs/2001.00001",
        "https://example.com/post",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"{BODY}\n\n{h}{REFS}", BODY)
        for h in (
            "## References",
            "References:",
            "**References**",
            "**References:**",
            "*References*",
            "参考文献",
            "## 参考文献：",
            "### 5. Bibliography",
            "Sources",
            "## 六、参考文献",
            "参考来源",
            "引用来源",
            "参考文献列表",
            "## VII. References",
            "## Sources Cited",
            "## Key References",
            "## References and Notes",
            "---\n\nReferences",
        )
    ]
    + [
        (BODY, BODY),
        (f"{BODY}\n\n## References\n[1] A.\n\n## Appendix\nMore.", None),  # not trailing
        ("Intro.\n\nSources\n\nWe discuss more.", None),  # a bare word, not a list heading
        (f"{BODY}\n\nReferences", BODY),
    ],
)
def test_strip_references(text: str, expected: str | None) -> None:
    assert strip_references(text) == (text if expected is None else expected)


def test_word_count_and_cap_words() -> None:
    assert word_count("Large language models [1] are useful [2][3, 4].") == 5
    assert word_count("大语言模型很有用。") == 5  # 8 CJK characters / 1.6
    assert word_count("RAG 检索增强模型 [2] - done") == 6  # 1 + 6 / 1.6 + 1, rounded

    text = "One two three. Four five six! Seven eight nine?"
    assert cap_words(text, 100) == text
    assert cap_words(text, 7) == "One two three. Four five six!"
    assert (
        cap_words("第一句话。第二句话！第三句话？", 6) == "第一句话。第二句话！"
    )  # 12 characters = 7.5 words
    # "et al." is not a sentence end; without any boundary the cut falls on a word
    assert cap_words("Smith et al. propose four tricks. Then more.", 7) == (
        "Smith et al. propose four tricks."
    )
    assert cap_words("Smith et al. propose four tricks.", 4) == "Smith et al. propose"
    assert cap_words("Intro text.\n\n## Next\n\nLong section here.", 3) == "Intro text."
    assert (
        cap_words("One claim.[3] Two claims.[4][5] Three more", 5)
        == "One claim.[3] Two claims.[4][5]"
    )


def test_renderers() -> None:
    report = Report(
        text="Attention [1] and RAG [2][3]; blogs [2].\n\n## References\n[1] x",
        sources=(VASWANI, BLOG, LEWIS),
    )
    intro, rows = deepscholar_intro(report)
    assert intro == (
        "Attention [Ashish Vaswani 2017-06-12](http://arxiv.org/abs/1706.03762v7) and RAG "
        "[Retrieval-Augmented Generation](http://arxiv.org/abs/2005.11401v4); blogs."
    )
    assert rows == [
        {"id": "1706.03762v7", "title": "Attention Is All You Need", "snippet": VASWANI.snippet},
        {
            "id": "2005.11401v4",
            "title": "Retrieval-Augmented Generation",
            "snippet": "Large models",
        },
    ]

    body = " ".join(f"Claim {i} holds here [1][2]." for i in range(100))  # 400 words
    model_refs = "\n\nReferences:\n[1] Vaswani. Attention.\n[2] Lewis. RAG."
    answer = researchqa_answer(Report(text=body + model_refs, sources=(VASWANI, BLOG)))
    text, refs = answer.split("\n\n## References\n")
    assert word_count(text) == 300 and text.endswith("Claim 74 holds here [1][2].")
    assert refs == (
        "[1] Ashish Vaswani et al. (2017). Attention Is All You Need.\n[2] A blog on RAG (2012)."
    )

    report = Report(text="Blogs [1] and papers [2] agree.", sources=(BLOG, LEWIS))
    assert drb2_report(report).endswith(
        "## References\n[1] A blog on RAG. https://example.com/rag\n"
        "[2] Retrieval-Augmented Generation. https://arxiv.org/pdf/2005.11401v4.pdf"
    )
    assert drb2_report(Report(text="No citations.")) == "No citations."

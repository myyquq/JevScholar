"""DeepScholar-base adapter: native -> Report round trip, one source identity with simple_rag,
per-task isolation and facades, braces, run()."""

from __future__ import annotations

import csv
import logging
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from jevscholar import render
from jevscholar.config import Search
from jevscholar.paths import VENDOR
from jevscholar.render import deepscholar_intro, drb2_report, finalize, researchqa_answer
from jevscholar.search import ToolError
from jevscholar.systems import deepscholar_base as dsb
from jevscholar.systems.simple_rag import web_source
from jevscholar.types import CLIENT_TIMEOUT_S, AccessPolicy, Task, TaskContext
from jevscholar.util import vendored

FIXTURES = VENDOR / "deepscholar/tests/baselines_results/deepscholar-base"
AGENT_CONFIGS = SimpleNamespace(
    logger=logging.getLogger("test"), per_query_max_search_results_count=10
)


def paper(pid: str) -> dict[str, Any]:
    return {
        "id": pid,
        "title": f"T {pid}",
        "abstract": f"A {pid}",
        "published": "2020-01-02T03:04:05+00:00",
        "authors": ["Alice A", "Bob B"],
        "categories": ["cs.LG", "econ.GN"],
    }


class FakeSearch:
    """SearchClient stand-in that records which task's client served each call."""

    def __init__(self, name: str) -> None:
        self.name, self.calls, self.fail = name, [], False
        self.hits = [
            {"title": "Blog", "url": "https://e.com/p", "content": "c", "published_date": None}
        ]
        self.stats = {"calls": 0, "errors": 0, "blocked": 0, "cached": 0}

    def arxiv_search(self, query: str, k: int) -> list[dict[str, Any]]:
        self.calls.append(query)
        return [] if query == "nothing" else [paper(f"{self.name}.{query}.{i}") for i in range(2)]

    def arxiv_lookup(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        self.calls.append(("lookup", *ids))
        return {i: paper(i) for i in ids}

    def web_search(self, query: str, k: int) -> list[dict[str, Any]]:
        if self.fail:
            raise ToolError("tavily down")
        return self.hits

    def web_read(self, url: str) -> str:
        return f"page {url}"


def agent_ctx(ag: Any) -> SimpleNamespace:  # RunContextWrapper stand-in
    return SimpleNamespace(
        context=ag.AgentContext(configs=AGENT_CONFIGS, end_date=None, papers_df=None, queries=[])
    )


def eval_parse(folder: Path) -> tuple[str, list[dict[str, str]]]:
    """The official DeepScholar eval parser (third_party eval/parsers/deepscholar_base.py)."""
    vendored("deepscholar")
    from eval.parsers.deepscholar_base import DeepScholarBaseParser

    p = DeepScholarBaseParser(str(folder), {"s_map_groundtruth": {}, "file_id": "0"})
    return p.clean_text, p.docs


def read_docs(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "paper.csv", dtype=str, keep_default_na=False)


# --------------------------------------------------------------------------- to_report
def test_to_report_tokens() -> None:
    docs = pd.DataFrame(
        [
            {
                "id": "2001.00001v2",
                "title": "A",
                "url": "http://arxiv.org/abs/2001.00001v2",
                "snippet": "abs A",
                "date": "2020-01-02 03:04:05+00:00",
                "authors": "Alice, Bob",
            },
            {
                "id": np.nan,
                "title": "Blog",
                "url": "https://e.com/x",
                "snippet": "web",
                "date": "",
                "authors": np.nan,
            },
            {
                "id": "2002.00002v1",
                "title": "Unused",
                "url": "http://arxiv.org/abs/2002.00002v1",
                "snippet": "",
                "date": "",
                "authors": "",
            },
        ]
    )
    text = (
        r"X \[[Alice' 2020-01-02](http://arxiv.org/abs/2001.00001v2)\]. "
        "Y [copied](https://arxiv.org/abs/2001.00001v2) [Z](http://arxiv.org/abs/9999.99999) "
        r"\[[2](https://e.com/x)\] [1, 2] and [3-4]; [Cat [2 papers]](#category-cat)."
    )
    r = dsb.to_report(text, docs, extra=1)
    assert r.text == r"X [1]. Y [1] [2] [3] \[1, 2\] and \[3-4\]; [Cat [2 papers]](#category-cat)."
    a, z, w = r.sources
    assert (a.id, a.title, a.authors, a.date, a.snippet) == (
        "2001.00001v2",
        "A",
        ("Alice", "Bob"),
        "2020-01-02",
        "abs A",
    )
    assert a.meta == {
        "corpus": "arxiv",
        "label": "Alice' 2020-01-02",
        "native_id": "2001.00001v2",
        "in_pool": True,
        "bracketed": True,
    }
    assert (z.id, z.title, z.meta["in_pool"], z.meta["bracketed"]) == (
        "9999.99999",
        "",
        False,
        False,
    )
    assert (w.id, w.title, w.authors, w.meta["corpus"]) == ("https://e.com/x", "Blog", (), "web")
    assert r.meta == {
        "extra": 1,
        "docs": list(docs["url"]),  # the writer's input, in order
        "pool": ["http://arxiv.org/abs/2002.00002v1"],
    }


@pytest.mark.parametrize("case", ["0", "1"])
def test_to_report_round_trips_through_eval_parser(case: str, tmp_path: Path) -> None:
    r"""parser(native) == parser(render(to_report(native))), up to two documented differences.

    1. The shared rule strips model-written reference lists (render.deepscholar_intro applies
       strip_references), and these fixtures end with a "## References" list, so the native
       side is compared with that rule applied too.
    2. Upstream writes citations as `\[[label](url)\]`; the parser's link regex starts at the
       inner `[`, so the native clean text keeps the escaped outer brackets (`\[1]\]`), while
       the renderer writes bare `[label](url)` links (-> `[1]`). Labels never reach the clean
       text, and the cited docs (titles + abstracts, in citation order) must be identical.
    """
    native = FIXTURES / case
    intro = (native / "intro.md").read_text()
    text, rows = render.deepscholar_intro(dsb.to_report(intro, read_docs(native)))
    (tmp_path / "intro.md").write_text(text)
    with (tmp_path / "paper.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "snippet"])
        writer.writeheader()
        writer.writerows(rows)
    stripped = tmp_path / "native"
    stripped.mkdir()
    (stripped / "intro.md").write_text(render.strip_references(intro))
    shutil.copy(native / "paper.csv", stripped)

    native_text, native_docs = eval_parse(stripped)
    text, docs2 = eval_parse(tmp_path)
    assert len(native_docs) > 5 and docs2 == native_docs
    assert text == re.sub(r"\\(\[\d+\])\\\]", r"\1", native_text)

    docs = read_docs(native)  # the final report's per-category tables cite every pool row
    final = dsb.to_report((native / "final_report.md").read_text(), docs)
    assert "](#category-" in final.text  # outline anchors stay text
    assert {s.url for s in final.sources} == set(docs["url"]) and final.meta["pool"] == []
    assert all(s.meta["in_pool"] for s in final.sources)


async def test_web_hits_export_identically_for_both_systems() -> None:
    """One identity rule: a Tavily hit on arxiv.org/abs counts as the same arXiv doc, and a
    parenthesised Wikipedia URL survives intact, whether simple_rag or DeepScholar-base cites it."""
    dsb.install()
    ag = sys.modules["deepscholar_base.search.agentic_search"]
    search = FakeSearch("x")
    search.hits = [
        {
            "title": "[2001.00001] Paper A",
            "url": "https://arxiv.org/abs/2001.00001v2",
            "content": "Abstract A.",
            "published_date": None,
        },
        {
            "title": "Transformer (deep learning architecture) - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
            "content": "A transformer is ...",
            "published_date": None,
        },
    ]
    with dsb.task_scope(search, 0):
        ctx = agent_ctx(ag)
        await ag._handle_one_search_query(ctx, ag.ToolTypes.WEB, 1, None, "q")
    native = "".join(rf"Claim {i} \[[x]({h['url']})\]. " for i, h in enumerate(search.hits))
    ours = dsb.to_report(native, ctx.context.papers_df)
    assert [(s.url, s.meta["in_pool"]) for s in ours.sources] == [
        (h["url"], True) for h in search.hits
    ]

    ds = finalize(ours.text, ours.sources)
    rag = finalize("Claim 0 [1]. Claim 1 [2].", [web_source(h) for h in search.hits])
    key = [(s.id, s.url, s.title, s.date, s.snippet) for s in ds.sources]
    assert key == [(s.id, s.url, s.title, s.date, s.snippet) for s in rag.sources]
    for export in (deepscholar_intro, drb2_report, researchqa_answer):
        assert export(ds) == export(rag)
    assert deepscholar_intro(rag)[1] == [
        {"id": "2001.00001v2", "title": "[2001.00001] Paper A", "snippet": "Abstract A."}
    ]


# --------------------------------------------------------------------------- patches
async def run_agent_tools(ag: Any) -> pd.DataFrame:
    ctx = agent_ctx(ag)
    for q in ("q1", "q2"):
        await ag._handle_one_search_query(ctx, ag.ToolTypes.ARXIV, 1, None, q)
    await ag._read_content(ctx, ag.ToolTypes.ARXIV, ["2001.00001", " 2002.00002"])
    return ctx.context.papers_df


async def test_patches_isolate_tasks_and_follow_lotus_contracts() -> None:
    dsb.install()
    ag = sys.modules["deepscholar_base.search.agentic_search"]
    a, b = FakeSearch("a"), FakeSearch("b")
    with dsb.task_scope(a, 0):
        pool_a = await run_agent_tools(ag)
    with dsb.task_scope(b, 1, repair_merge=False):
        draw = np.random.randint(10**9)
        pool_b = await run_agent_tools(ag)

    assert draw == np.random.RandomState(1).randint(10**9)  # P9
    for client in (a, b):  # each task's calls reached only its own client; one batched warm-up
        assert client.calls == [
            "q1",
            "q2",
            ("lookup", "2001.00001", "2002.00002"),
            ("lookup", "2001.00001"),
            ("lookup", "2002.00002"),
        ]
    assert sorted(pool_a["id"]) == ["a.q1.0", "a.q1.1", "a.q2.0", "a.q2.1"]  # no read rows
    assert sorted(pool_b["id"]) == ["b.q1.0", "b.q1.1"]  # upstream as-is: first query only
    with pytest.raises(RuntimeError, match="outside a task"):
        ag.web_search(corpus=ag.WebSearchCorpus.ARXIV, query="q", K=1)

    corpus, search = ag.WebSearchCorpus, FakeSearch("x")
    with dsb.task_scope(search, 0):
        df = ag.web_search(corpus=corpus.ARXIV, query="q", K=2, end_date=datetime.now(UTC))
        assert list(df.columns) == [
            "id",
            "title",
            "link",
            "abstract",
            "published",
            "authors",
            "categories",
        ]
        assert isinstance(df["published"].dtype, pd.DatetimeTZDtype)
        row = df.iloc[0]
        assert (row["link"], row["authors"], row["categories"]) == (
            "http://arxiv.org/abs/x.q.0",
            "Alice A, Bob B",
            "cs.LG, econ.GN",
        )
        _, norm = ag._normalize_search_df(df, "q", ag.ToolTypes.ARXIV.to_rename_map(), str, "")
        datetime.strptime(norm["date"].iloc[0], "%Y-%m-%d %H:%M:%S%z")  # citation-label format
        assert ag.web_search(corpus=corpus.ARXIV, query="nothing", K=2).empty

        assert list(ag.web_search(corpus=corpus.TAVILY, query="q", K=2).columns) == [
            "title",
            "url",
            "content",
        ]
        assert ag.web_extract(corpus=corpus.ARXIV, doc_id=" 2001.00001 ").to_dict("records") == [
            {
                "id": "2001.00001",
                "url": "https://arxiv.org/abs/2001.00001",
                "full_text": "Title: T 2001.00001\n\nAbstract: A 2001.00001",
            }
        ]
        assert ag.web_extract(corpus=corpus.TAVILY, doc_id="https://e.com/p").to_dict(
            "records"
        ) == [
            {"id": "https://e.com/p", "url": "https://e.com/p", "full_text": "page https://e.com/p"}
        ]

        search.fail = True
        section, ok = await ag._handle_one_search_query(
            agent_ctx(ag), ag.ToolTypes.WEB, 1, None, "q"
        )
        assert ok is None and "Error searching web for query q: tavily down" in section


TOPIC = "Survey {'title': 'X', 'urls': ['u']}, {x}, }{ and {{y}} {"


async def test_literal_braces_reach_every_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """P2-P4: every instruction LOTUS formats shows the topic (and other text) verbatim."""
    top = dsb.install()
    from lotus.nl_expression import nle2str, parse_cols
    from lotus.sem_ops.sem_agg import SemAggDataframe
    from lotus.sem_ops.sem_filter import SemFilterDataframe
    from lotus.sem_ops.sem_map import SemMapDataframe
    from lotus.sem_ops.sem_topk import SemTopKDataframe

    fg = sys.modules["deepscholar_base.final_generation"]
    tx = sys.modules["deepscholar_base.taxonomize"]
    seen: list[tuple[str, str]] = []

    def fake(name: str, result: Any) -> Any:
        def call(self: Any, user_instruction: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
            cols = parse_cols(user_instruction)  # what LOTUS does first: must find real columns
            assert set(cols) <= set(self._obj.columns), (name, cols)
            seen.append((name, nle2str(user_instruction, cols)))
            return result(self._obj, kwargs)

        return call

    def agg(df: pd.DataFrame, kw: dict[str, Any]) -> pd.DataFrame:
        if kw.get("group_by"):
            return pd.DataFrame({"category": ["Cat {a}"], "summary": ["S [1]."]})
        return pd.DataFrame({"summary": ["Intro [1]."]})

    monkeypatch.setattr(SemFilterDataframe, "__call__", fake("filter", lambda df, kw: df))
    monkeypatch.setattr(SemTopKDataframe, "__call__", fake("topk", lambda df, kw: df))
    monkeypatch.setattr(
        SemMapDataframe, "__call__", fake("map", lambda df, kw: df.assign(category_number="1"))
    )
    monkeypatch.setattr(SemAggDataframe, "__call__", fake("agg", agg))
    lm = SimpleNamespace(get_completion=lambda system, user: "1. Cat {a}\n2. Other }{")
    configs = SimpleNamespace(
        filter_lm=None,
        generation_lm=None,
        taxonomize_lm=lm,
        use_sem_filter=True,
        use_sem_topk=True,
        sem_filter_kwargs={},
        sem_topk_kwargs={},
        final_max_results_count=5,
        use_structured_output=False,
        generate_category_summary=True,
        logger=logging.getLogger(),
    )
    docs = pd.DataFrame(
        {
            "title": ["T"],
            "url": ["http://arxiv.org/abs/1v1"],
            "snippet": ["s {b}"],
            "context": ["c"],
            "authors": ["Alice, Bob"],
            "date": ["2020-01-02 03:04:05+00:00"],
        }
    )

    top.filter(configs, docs, TOPIC)
    intro = await fg.generate_intro_section(TOPIC, docs, "Background {with} }{", configs)
    _, summaries = await tx.categorize_references(TOPIC, intro, docs, configs)

    assert [name for name, _ in seen] == ["filter", "topk", "agg", "map", "agg"]
    assert all(TOPIC in prompt for name, prompt in seen if name != "map")
    assert "Background {with} }{" in seen[2][1]
    assert "1. Cat {a}\n2. Other }{" in seen[3][1]
    assert intro == r"Intro \[[Alice' 2020-01-02](http://arxiv.org/abs/1v1)\]."
    assert summaries["category"].tolist() == ["Cat {a}"]


# --------------------------------------------------------------------------- run()
async def test_run_persists_native_outputs_and_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    top = dsb.install()
    native = FIXTURES / "0"
    intro, docs = (native / "intro.md").read_text(), read_docs(native)
    seen: dict[str, Any] = {}
    stats: dict[str, Any] = {}

    async def pipeline(topic: str, end_date: Any, configs: Any) -> tuple[str, Any, dict]:
        seen.update(topic=topic, end_date=end_date, configs=configs)
        return "", docs, stats

    monkeypatch.setattr(top, "deepscholar_base", pipeline)
    cutoff = datetime(2025, 6, 1, tzinfo=UTC)
    task = Task(
        bench="deepscholar",
        id="7",
        prompt="Write related work.",
        policy=AccessPolicy(cutoff=cutoff),
    )
    search = FakeSearch("x")
    search.stats.update(calls=10, errors=3)
    ctx = TaskContext(
        gateway="http://gw",
        run="r",
        system="deepscholar_base",
        task=task,
        executor="deepseek-flash",
        search=search,  # type: ignore[arg-type]
        out_dir=tmp_path,
        search_config=Search(corpora=("web",), k=7),
        cap_usd=1.0,
    )
    system = dsb.DeepScholarBase({"deliverable": "final_report"})
    with pytest.raises(ValueError, match="always searches arXiv"):
        await system.run(ctx)
    ctx.search_config = Search(k=7)

    stats.update({"search_try_1": {"error": "boom"}, "error": "no results"})
    with pytest.raises(RuntimeError, match="no intro: no results"):
        await system.run(ctx)

    stats.update({"search_try_2": {"pre_filter_search_results": [{}] * 7}, "intro_section": intro})
    report = await system.run(ctx)
    assert report.meta["status"] == "degraded"  # 3 of 10 tool calls failed
    assert report.meta["deliverable"] == "intro"  # final_report is empty -> fallback
    assert (report.meta["pool_size"], report.meta["search_try_errors"]) == (7, ["boom"])
    assert report.sources and "## References" not in report.text
    assert {p.name for p in (tmp_path / "native").iterdir()} == {
        "final_report.md",
        "intro.md",
        "paper.csv",
        "stats.json",
    }

    cfg = seen["configs"]
    assert seen["end_date"] == pd.Timestamp(cutoff) and cfg.use_agentic_search
    assert (cfg.enable_web_search, cfg.per_query_max_search_results_count) == (False, 7)
    for stage in dsb.STAGES:
        lm = getattr(cfg, f"{stage}_lm")
        assert (lm.model, lm.max_tokens, lm.kwargs["api_base"]) == (
            "deepseek-flash",
            10000,
            ctx.base_url(stage),
        )
        assert lm.kwargs == {
            "temperature": 1.0,
            "max_completion_tokens": 10000,
            "custom_llm_provider": "openai",
            "api_base": ctx.base_url(stage),
            "timeout": CLIENT_TIMEOUT_S,
            "max_retries": 0,
        }


def test_config_overrides_select_the_paper_protocol(tmp_path: Path) -> None:
    vendored("deepscholar")
    task = Task(bench="deepscholar", id="7", prompt="p", policy=AccessPolicy(cutoff=None))
    ctx = TaskContext(
        gateway="http://gw",
        run="r",
        system="deepscholar_ref",
        task=task,
        executor="deepseek-flash",
        search=FakeSearch("x"),  # type: ignore[arg-type]
        out_dir=tmp_path,
        search_config=Search(),
        cap_usd=1.0,
    )
    paper = {"search_mode": "recursive", "num_search_steps": 2}
    cfg = dsb.build_configs(ctx, dsb.DeepScholarBase({"config": paper}).p.config)
    assert (cfg.use_agentic_search, cfg.num_search_steps) == (False, 2)
    assert dsb.build_configs(ctx).use_agentic_search  # the repo default stays agentic
    assert dsb.build_configs(ctx).generation_lm.kwargs["temperature"] == 1.0  # the YAML's
    sampled = dsb.build_configs(ctx, None, 2)  # an independent writer sample: its own namespace
    assert ":generation~2/" in sampled.generation_lm.kwargs["api_base"]
    assert ":filter/" in sampled.filter_lm.kwargs["api_base"]  # search and screening replay
    for bad in ({"lm": {}}, {"enable_web_search": True}, {"num_steps": 2}, {"search_mode": "rec"}):
        with pytest.raises(ValueError):
            dsb.DeepScholarBase({"config": bad})


async def test_citation_repair_strips_stale_background_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    top = dsb.install()
    fg = sys.modules["deepscholar_base.final_generation"]
    seen: list[str] = []

    async def summary(topic: str, docs: Any, instructions: str, existing: str, **kw: Any) -> str:
        seen.append(existing)
        return "intro"

    monkeypatch.setattr(fg, "generate_section_summary_with_citations", summary)
    background = "BankFair [12] and BankFair+[17] trade off [1, 2]【3】."
    for repair in (False, True):
        with dsb.task_scope(FakeSearch("x"), 0, repair_citations=repair):
            await top.generate_intro_section(
                "t", pd.DataFrame(), background, SimpleNamespace(generation_lm=None)
            )
    with dsb.task_scope(FakeSearch("x"), 0, repair_citations=True, search_background=False):
        await top.generate_intro_section(
            "t", pd.DataFrame(), background, SimpleNamespace(generation_lm=None)
        )
    assert seen == [background, "BankFair and BankFair+ trade off.", ""]  # the ablation: none
    sg = sys.modules["deepscholar_base.utils.summary_generation"]
    for repair, placeholder in ((False, "{{_citation_number}}"), (True, "{_citation_number}")):
        with dsb.task_scope(FakeSearch("x"), 0, repair_citations=repair):
            assert placeholder in sg.input_sanitization(sg.citation_guidelines)
            assert sg.input_sanitization("{topic}") == "{{topic}}"  # user text stays escaped

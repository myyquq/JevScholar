"""The two light systems, against a fake gateway and a fake search client."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from jevscholar.config import Search
from jevscholar.search import ToolError
from jevscholar.systems.closed_book import ClosedBook
from jevscholar.systems.simple_rag import SimpleRAG
from jevscholar.types import Task, TaskContext, parse_tag

PAPERS = [
    {
        "id": "2001.00001v2",
        "title": "Paper A",
        "abstract": "Abstract A.",
        "published": "2020-01-02T03:04:05+00:00",
        "authors": ["Alice", "Bob"],
        "categories": ["cs.LG"],
    },
    {
        "id": "2002.00002v1",
        "title": "Paper B",
        "abstract": "Abstract B.",
        "published": "2020-02-03T00:00:00+00:00",
        "authors": ["A", "B", "C", "D"],
        "categories": [],
    },
]
HIT = {
    "title": "Blog",
    "url": "https://Example.com/post/",
    "content": "word " * 600,
    "published_date": None,
}


class FakeSearch:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.calls = fail, []

    def arxiv_search(self, query: str, k: int) -> list[dict[str, Any]]:
        self.calls.append(("arxiv", query, k))
        if self.fail:
            raise ToolError("arxiv down")
        return PAPERS[:k]

    def web_search(self, query: str, k: int) -> list[dict[str, Any]]:
        self.calls.append(("web", query, k))
        if self.fail:
            raise ToolError("tavily down")
        return [HIT]


@pytest.fixture
async def gateway():
    """Fake OpenAI-compatible gateway: replies per stage (last tag part), records requests."""
    replies: dict[str, str] = {}
    calls: list[tuple[str, dict[str, Any]]] = []

    async def complete(request: web.Request) -> web.Response:
        body = await request.json()
        stage = parse_tag(request.match_info["tag"])[4]
        calls.append((stage, body))
        message = {"role": "assistant", "content": replies[stage]}
        return web.json_response(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application()
    app.router.add_post("/t/{tag}/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    yield str(server.make_url("")).rstrip("/"), replies, calls
    await server.close()


def make_ctx(url: str, search: FakeSearch, tmp: Path, **task_meta: Any) -> TaskContext:
    task = Task(
        bench="researchqa", id="q1", prompt="Explain {graph} neural networks.", meta=task_meta
    )
    return TaskContext(
        gateway=url,
        run="r",
        system="simple_rag",
        task=task,
        executor="deepseek-flash",
        search=search,  # type: ignore[arg-type]
        out_dir=tmp,
        search_config=Search(corpora=("arxiv", "web")),
        cap_usd=1.0,
        log=logging.getLogger("test"),
    )


async def test_simple_rag_cites_retrieved_snippets(gateway, tmp_path: Path) -> None:
    url, replies, calls = gateway
    replies["query"] = ' "graph neural networks message passing"\n'
    replies["write"] = "GNNs [3] extend [2]. See [1, 3]. Bogus [7].\n\n## References\n1. x\n"
    search = FakeSearch()
    ctx = make_ctx(url, search, tmp_path)
    report = await SimpleRAG({}).run(ctx)

    assert search.calls == [
        ("arxiv", "graph neural networks message passing", 10),
        ("web", "graph neural networks message passing", 10),
    ]
    assert [stage for stage, _ in calls] == ["query", "write"]
    body = calls[1][1]
    assert (body["model"], body["temperature"], body["max_tokens"]) == (
        "deepseek-flash",
        1.0,
        10000,
    )
    prompt = body["messages"][0]["content"]
    assert prompt.startswith("Explain {graph} neural networks.")
    assert (
        "[1] Paper A (Alice, Bob, 2020-01-02). http://arxiv.org/abs/2001.00001v2\nAbstract A."
        in prompt
    )
    assert "[2] Paper B (A, B, C et al., 2020-02-03)." in prompt
    assert "[3] Blog. https://Example.com/post/\nword" in prompt

    assert report.text == "GNNs [1] extend [2]. See [3][1]. Bogus."
    web_src, b, a = report.sources
    assert (a.id, a.url, a.authors, a.date) == (
        "2001.00001v2",
        "http://arxiv.org/abs/2001.00001v2",
        ("Alice", "Bob"),
        "2020-01-02",
    )
    assert b.id == "2002.00002v1" and len(web_src.snippet) <= 1510
    assert (web_src.id, web_src.url) == ("https://example.com/post", "https://Example.com/post/")
    assert report.meta["status"] == "ok" and report.meta["retrieved"] == 3
    assert report.meta["retrieval"] == {"arxiv": {"hits": 2}, "web": {"hits": 1}}

    # raw query (no query stage); a failed search costs its sources and marks the task degraded
    calls.clear()
    replies["write"] = "Answer without sources [1]."
    search = FakeSearch(fail=True)
    ctx = make_ctx(url, search, tmp_path, query="raw question")
    report = await SimpleRAG({"k": 5, "query": "raw"}).run(ctx)
    assert [stage for stage, _ in calls] == ["write"]
    assert search.calls == [("arxiv", "raw question", 5), ("web", "raw question", 5)]
    assert (report.text, report.sources) == ("Answer without sources.", ())
    assert report.meta["status"] == "degraded"
    assert report.meta["retrieval"]["web"] == {"error": "tavily down"}


async def test_closed_book_has_no_sources(gateway, tmp_path: Path) -> None:
    url, replies, calls = gateway
    replies["write"] = "From memory [1].\n\n## References\n[1] Something"
    ctx = make_ctx(url, FakeSearch(), tmp_path)
    report = await ClosedBook({}).run(ctx)
    prompt = calls[0][1]["messages"][0]["content"]
    assert prompt.endswith("without citation markers or a reference list.")
    assert (report.text, report.sources) == ("From memory.", ())


@pytest.mark.parametrize(
    "path, params",
    [
        ("simple_rag:SimpleRAG", {"k": 20}),
        ("closed_book:ClosedBook", {}),
        ("deepscholar_base:DeepScholarBase", {"repair_citations": True}),
        ("fixed_evidence:FixedEvidence", {"evidence": ["gold"]}),
    ],
)
def test_systems_reject_unknown_parameters(path: str, params: dict[str, Any]) -> None:
    """A removed or misspelt knob is an error, never a silently different system."""
    from pydantic import ValidationError

    from jevscholar.types import load_class

    cls = load_class(f"jevscholar.systems.{path}")
    cls(params)
    with pytest.raises(ValidationError):
        cls({**params, "draft": True})

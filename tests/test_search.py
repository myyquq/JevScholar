"""Search layer: arXiv backends, the frozen store, SearchClient policy and metering, OpenAlex."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import requests

from jevscholar.search import (
    BudgetExhausted,
    FatalToolError,
    ToolError,
    arxiv,
    arxiv_backend,
    openalex,
    store,
    tavily,
)
from jevscholar.search.client import Meter, SearchClient
from jevscholar.search.store import Store
from jevscholar.types import AccessPolicy

FEED = (Path(__file__).parent / "fixtures/arxiv_search.xml").read_bytes()
CUTOFF = datetime(2025, 6, 1, tzinfo=UTC)
TITLE = "Graph Agents Survey"
POLICY = AccessPolicy(
    cutoff=CUTOFF,
    blocked_arxiv=frozenset({"2505.04321"}),
    blocked_titles=(TITLE,),
    undated_web="drop",
)
TAG = "run:bench:sys:t1:agent"


@dataclass
class Resp:
    status_code: int
    content: bytes = b""
    payload: Any = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content.decode()

    def json(self) -> Any:
        return self.payload


@dataclass
class Fake:
    """Scripted upstream: returns `responses` in order (the last one repeats) and records calls."""

    responses: list[Resp]
    calls: list[dict] = field(default_factory=list)

    def __call__(self, url: str, **kw: Any) -> Resp:
        self.calls.append({"url": url, **kw})
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def fake(mp: pytest.MonkeyPatch, module: ModuleType, method: str, *responses: Resp) -> Fake:
    """Replace `module.requests.<method>` with a scripted Fake."""
    f = Fake(list(responses))
    ns = SimpleNamespace(**{method: f}, RequestException=requests.RequestException)
    mp.setattr(module, "requests", ns)
    return f


def page(text: str) -> Resp:
    return Resp(200, payload={"results": [{"raw_content": text}]})


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Store:
    db = Store(tmp_path / "search.sqlite")
    monkeypatch.setattr(store, "default_store", lambda: db)
    monkeypatch.setattr(arxiv, "LOCK", tmp_path / "arxiv.lock")
    monkeypatch.setenv("JEV_ARXIV_INTERVAL", "0")
    monkeypatch.setenv("JEV_ARXIV_BACKEND", "api")  # the recorded Atom fixture
    return db


@pytest.fixture
def gateway() -> Iterator[SimpleNamespace]:
    """A fake gateway that records /admit and /charge bodies and refuses admits while `refuse`."""
    gw = SimpleNamespace(admits=[], charges=[], refuse=False)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            admit = self.path.endswith("/admit")
            (gw.admits if admit else gw.charges).append({"path": self.path, **body})
            refused = admit and gw.refuse
            answer = {"error": {"code": "budget_exhausted", "message": "spent"}} if refused else {}
            out = json.dumps(answer).encode()
            self.send_response(402 if refused else 200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    gw.url = f"http://127.0.0.1:{server.server_port}"
    yield gw
    server.shutdown()


# --------------------------------------------------------------------------- arXiv backends
def test_arxiv_parse_search_and_seeded_lookup(cache: Store, monkeypatch) -> None:
    a, b, c = arxiv.parse(FEED)
    assert a == {
        "id": "2408.14717v2",
        "title": "Text2SQL is Not Enough: Unifying AI and Databases with TAG",
        "abstract": "AI systems that serve natural language questions over databases promise to "
        "unlock tremendous value.",
        "published": "2024-08-27T02:45:13+00:00",
        "updated": "2024-09-30T10:00:00+00:00",
        "authors": ["Asim Biswal", "Liana Patel"],
        "categories": ["cs.DB", "cs.AI"],
        "url": "http://arxiv.org/abs/2408.14717v2",
    }
    assert (b["id"], c["id"]) == ("2506.01234v1", "cs/0112017v1")
    assert c["url"] == "http://arxiv.org/abs/cs/0112017v1"
    get = fake(monkeypatch, arxiv, "get", Resp(200, FEED))
    before = datetime(2025, 5, 31, 23, 59, tzinfo=UTC)
    assert arxiv.search("graph networks", 10, before) == ([a, b, c], False)
    assert get.calls[0]["params"] == {
        "search_query": "(graph networks) AND submittedDate:[199101010000 TO 202505312359]",
        "start": 0,
        "max_results": 10,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    assert arxiv.search("graph networks", 10, before) == ([a, b, c], True)
    found, cached = arxiv.lookup(["2408.14717v1", "arXiv:cs/0112017", "not-an-id"])
    assert cached and len(get.calls) == 1  # seeded by the search: no request
    assert {k: p["id"] for k, p in found.items()} == {
        "2408.14717v1": "2408.14717v2",
        "arXiv:cs/0112017": "cs/0112017v1",
    }
    # Only unseeded ids are requested, versionless and in one id_list batch.
    found, cached = arxiv.lookup(["1706.03762v5", "2408.14717", "0704.0001"])
    assert get.calls[1]["params"] == {"id_list": "0704.0001,1706.03762", "max_results": 2}
    assert not cached and set(found) == {"2408.14717"}  # the fake feed has neither


def test_arxiv_backoff_errors_and_store(cache: Store, monkeypatch, tmp_path: Path) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv, "time", SimpleNamespace(sleep=sleeps.append, time=time.time))
    fake(monkeypatch, arxiv, "get", Resp(429), Resp(503), Resp(200, FEED))
    assert len(arxiv.search("graph", 3)[0]) == 3
    assert [s for s in sleeps if s] == [30.0, 60.0]
    fake(monkeypatch, arxiv, "get", Resp(400, b"bad query"))
    with pytest.raises(ToolError, match="HTTP 400"):
        arxiv.search("bad", 3)
    get = fake(monkeypatch, arxiv, "get", Resp(200, FEED))
    assert arxiv.search("bad", 3) == (arxiv.parse(FEED), False)  # the error was never stored
    assert len(get.calls) == 1
    db = Store(tmp_path / "s.sqlite")  # concurrent misses settle on one frozen value
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda i: db.put({"q": "same"}, {"v": i}), range(32)))
    assert all(r == results[0] for r in results) and db.get({"q": "same"}) == results[0]


def test_local_backend(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("bm25s")
    from jevscholar.search import arxiv_local

    def rec(i: str, title: str, *created: str) -> dict:
        return {
            "id": i,
            "title": title,
            "abstract": f"An abstract about {title.lower()}.",
            "authors_parsed": [["Lovelace", "Ada", ""], ["Babbage", "Charles", "Jr"]],
            "categories": "cs.LG cs.AI",
            "versions": [{"version": f"v{n}", "created": c} for n, c in enumerate(created, 1)],
        }

    old = ("Fri, 1 Jan 1999 00:00:00 GMT", "Sat, 2 Jan 1999 00:00:00 GMT")
    records = [
        rec("2401.00001", "Graph neural networks for molecules", "Mon, 1 Jan 2024 10:00:00 GMT"),
        rec("2506.00002", "Graph neural networks after it", "Tue, 10 Jun 2025 10:00:00 GMT"),
        rec("hep-th/9901001", "String theory", *old),
        rec("hep-th/9901001", "String theory duplicate row", *old),  # the dump repeats some ids
    ]
    arxiv_local.write_index(records, tmp_path)
    monkeypatch.setattr(arxiv_local, "INDEX", tmp_path)
    monkeypatch.setenv("JEV_ARXIV_BACKEND", "local")
    assert arxiv_backend() is arxiv_local
    papers, _ = arxiv_local.search("graph neural networks", 10)
    assert {p["id"] for p in papers} == {"2401.00001v1", "2506.00002v1"}
    papers, _ = arxiv_local.search("graph neural networks", 10, CUTOFF)
    assert [p["id"] for p in papers] == ["2401.00001v1"]
    assert papers[0]["authors"] == ["Ada Lovelace", "Charles Babbage Jr"]
    assert papers[0]["published"] == "2024-01-01T10:00:00+00:00"
    found, _ = arxiv_local.lookup(["hep-th/9901001v1", "2401.00001"])
    assert found["hep-th/9901001v1"]["id"] == "hep-th/9901001v2"
    assert found["hep-th/9901001v1"]["published"] == "1999-01-01T00:00:00+00:00"
    assert found["hep-th/9901001v1"]["title"] == "String theory"  # first occurrence kept
    assert arxiv_local.search("zzz unknownword", 10) == ([], False)


# --------------------------------------------------------------------------- client
def test_client_arxiv_filters_and_charges(cache: Store, monkeypatch, gateway) -> None:
    get = fake(monkeypatch, arxiv, "get", Resp(200, FEED))
    client = SearchClient(gateway.url, TAG, POLICY)
    first = client.arxiv_search("graph", 10)
    assert [p["id"] for p in first] == ["2408.14717v2", "cs/0112017v1"]  # 2506 paper dropped

    async def agent_tool() -> list[dict]:
        return client.arxiv_search("graph", 10)  # DeepScholar calls tools synchronously

    assert asyncio.run(agent_tool()) == first  # a cached replay, charged as such
    charged = [(c["path"], c["tool"], c["cached"], c["meta"]["blocked"]) for c in gateway.charges]
    assert charged == [
        (f"/t/{TAG}/charge", "arxiv_search", False, 1),
        (f"/t/{TAG}/charge", "arxiv_search", True, 1),
    ]
    assert gateway.charges[0]["meta"]["backend"] == "arxiv"
    assert client.stats == {"calls": 2, "errors": 0, "blocked": 2, "cached": 1}
    # The id cache is shared, the policy is not: an uncut task sees the post-cutoff paper.
    other = SearchClient(gateway.url, "run:bench:sys:t2:agent", AccessPolicy())
    assert set(other.arxiv_lookup(["2506.01234"])) == {"2506.01234"}
    assert client.arxiv_lookup(["2506.01234", "2408.14717"]).keys() == {"2408.14717"}
    assert client.arxiv_search(f"{TITLE} 2024", 10) == []  # the query names a blocked title
    assert client.arxiv_search("follow-ups of 2506.01234", 10) == []  # or a post-cutoff id
    assert len(gateway.admits) == len(gateway.charges) == 4  # refused queries cost nothing
    assert gateway.admits[0] == {"path": f"/t/{TAG}/admit", "tool": "arxiv_search", "meta": {
        "query": "graph", "k": 10, "backend": "arxiv"}}  # fmt: skip
    gateway.refuse, sent = True, len(get.calls)  # a closed latch: refused before any request
    with pytest.raises(BudgetExhausted):
        client.arxiv_search("graph neural networks", 10)
    assert len(get.calls) == sent and len(gateway.charges) == 4


def test_client_web_search_and_read(cache: Store, monkeypatch, gateway) -> None:
    def hit(path: str, date: str | None, content: str = "c") -> dict:
        url = f"https://a.example/{path}"
        return {"title": path, "url": url, "content": content, "published_date": date}

    hits = [
        hit("x", "2025-05-20"),
        hit("u", "2025-05-21"),
        hit("t", "2025-05-22"),
        hit("y", "2025-06-02"),  # after the cutoff
        hit("z", None),  # undated
        hit("w", "2025-05-20", "a sequel: arxiv.org/abs/2506.00009"),  # leaks a later paper
    ]
    blocked_page, leaky_page = page(f"# About {TITLE}"), page("Cites arXiv:2505.04321.")
    results = Resp(200, payload={"results": hits})
    tv = fake(monkeypatch, tavily, "post", results, blocked_page, leaky_page, page("Clean text."))
    client = SearchClient(gateway.url, TAG, POLICY, tavily_key="tvly-test")
    assert [h["url"][-1] for h in client.web_search("graph agents", 5)] == ["x", "u", "t"]
    assert tv.calls[0]["json"] == {
        "query": "graph agents",
        "max_results": 5,
        "include_published_date": True,
        "end_date": "2025-05-31",
    }
    assert tv.calls[0]["headers"] == {"Authorization": "Bearer tvly-test"}
    client.web_search("graph agents", 5)
    assert len(tv.calls) == 1 and gateway.charges[-1]["cached"]
    assert client.web_read("https://arxiv.org/abs/2506.00001") is None  # post-cutoff URL
    assert client.web_read("https://a.example/z") is None  # undated: no allowed search hit
    assert client.web_read("https://a.example/x") is None  # page names the blocked title
    assert client.web_read("https://a.example/u") is None  # page cites the blocked paper
    assert client.web_read("https://a.example/t/") == "Clean text."  # canonical URL match
    read = [c["json"]["urls"] for c in tv.calls[1:]]
    assert read == [["https://a.example/x"], ["https://a.example/u"], ["https://a.example/t/"]]
    assert client.stats["blocked"] == 2 * 3 + 2 + 2
    keep = SearchClient(gateway.url, TAG, AccessPolicy(cutoff=CUTOFF), tavily_key="k")
    assert keep.web_read("https://b.example/never-searched") == "Clean text."
    # A fatal tool error is charged as fatal, which trips the breaker: later calls are refused
    # before they reach Tavily.
    tv = fake(monkeypatch, tavily, "post", Resp(432, b'{"detail": "plan limit"}'))
    with pytest.raises(FatalToolError, match="432"):
        client.web_search("graph", 5)
    charge = gateway.charges[-1]
    assert (charge["tool"], charge["ok"], charge["fatal"]) == ("tavily_search", False, True)
    assert client.stats["errors"] == 1
    gateway.refuse = True
    with pytest.raises(BudgetExhausted):
        client.web_search("agent memory", 5)
    assert len(tv.calls) == 1 and client.stats["errors"] == 1


# --------------------------------------------------------------------------- OpenAlex
def test_openalex_rule_cache_and_errors(cache: Store, monkeypatch, gateway) -> None:
    def top(name: str, n: int) -> Resp:
        return Resp(200, payload={"results": [{"display_name": name, "cited_by_count": n}]})

    answers = {
        "Better & Faster LLMs": [top("Better & Faster LLMs", 42)],
        "graph nets": [top("Unrelated title", 9)],
        "odd title": [Resp(400)],
        "bad key": [Resp(401)],
        "quota": [Resp(429, headers={"retry-after": "42000"})],
        "flaky": [Resp(503)] * 4,
    }
    urls: list[str] = []

    def get(url: str, params: dict, timeout: float) -> Resp:
        urls.append(requests.Request("GET", url, params=params).prepare().url)
        return answers[params["search"]].pop(0)

    ns = SimpleNamespace(get=get, RequestException=requests.RequestException)
    monkeypatch.setattr(openalex, "requests", ns)
    monkeypatch.setattr(openalex, "time", SimpleNamespace(sleep=lambda s: None))
    monkeypatch.setenv("JEV_OPENALEX_KEY", "k")
    tag = "run:deepscholar:sys:t1:judge.document_importance"
    count = partial(openalex.cited_by_count, meter=Meter(gateway.url, tag))

    assert count("Better & Faster LLMs") == 42
    assert urls == [
        f"{openalex.API}?search=Better+%26+Faster+LLMs&mailto=your%40email.com&api_key=k"
    ]
    assert count("Better & Faster LLMs") == 42  # cached
    assert count("graph nets") is None  # top hit below Jaccard 0.8
    for _ in range(2):  # a 400 is a definitive "no match", cached like upstream's None
        assert count("odd title") is None
    with pytest.raises(FatalToolError, match="401"):
        count("bad key")
    with pytest.raises(ToolError, match="daily quota"):
        count("quota")
    with pytest.raises(ToolError, match="4 attempts"):
        count("flaky")
    assert len(urls) == 3 + 1 + 1 + 4
    assert {c["path"] for c in gateway.charges} == {f"/t/{tag}/charge"}
    assert len(gateway.admits) == len(gateway.charges)
    assert [(c["cached"], c["ok"], c["fatal"]) for c in gateway.charges] == [
        (False, True, False),
        (True, True, False),
        (False, True, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),  # bad key: trips the breaker
        (False, False, False),  # daily quota: this lookup fails, the run goes on
        (False, False, False),
    ]
    assert cache.get({"api": "openalex", "search": "quota", "mailto": openalex.MAILTO}) is None
    gateway.refuse = True  # a closed latch: not even a cached lookup is served
    for title in ("Better & Faster LLMs", "never searched"):
        with pytest.raises(BudgetExhausted):
            count(title)
    assert len(urls) == 9 and len(gateway.charges) == 8


@pytest.mark.network
def test_live_arxiv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two real requests through the shared limiter (real lock file, real interval)."""
    db = Store(tmp_path / "search.sqlite")
    monkeypatch.setattr(store, "default_store", lambda: db)
    papers, _ = arxiv.search('ti:"Text2SQL is Not Enough"', 3, datetime(2024, 9, 1, tzinfo=UTC))
    assert [p["id"][:10] for p in papers][:1] == ["2408.14717"]
    assert papers[0]["published"].startswith("2024-08-27")
    found, cached = arxiv.lookup(["1706.03762v1"])
    assert not cached and found["1706.03762v1"]["id"].startswith("1706.03762v")
    assert found["1706.03762v1"]["published"].startswith("2017-06-12")

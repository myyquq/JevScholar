"""Simple RAG baseline: one query, top-k snippets per shared corpus, one generation call.

It is the "retrieve K, then generate" reference point of the GOAL comparison. K is selected on
dev from {5, 10, 20} and frozen per benchmark; `query: raw` searches with the benchmark's own
question (ResearchQA), which anchors to the naive-RAG rows published by DR Tulu and the
leaderboard. Only snippets are used (arXiv abstracts, Tavily content), never full pages.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Literal

from jevscholar.render import finalize, source_id
from jevscholar.search import ToolError
from jevscholar.systems import chat
from jevscholar.types import Report, Source, Strict, System, TaskContext

WEB_SNIPPET_CHARS = 1500

QUERY_PROMPT = """Write one search query for finding sources that help with the task below.
Use 1-10 keywords. Output only the query.

Task:
{prompt}"""

WRITE_PROMPT = """{prompt}

Sources (search results; not all of them may be relevant):

{sources}

Base your answer on these sources and cite them inline by number, e.g. [3], or [2][5] for
several. Cite only these sources. Do not write a reference list: it is generated automatically
from your citations."""


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def _day(date: str | None) -> str:
    return date[:10] if date and re.match(r"\d{4}-\d{2}-\d{2}", date) else ""


def arxiv_source(p: dict[str, Any]) -> Source:
    return Source(
        id=p["id"],
        url=f"http://arxiv.org/abs/{p['id']}",
        title=p["title"],
        authors=tuple(p["authors"]),
        date=_day(p["published"]),
        snippet=p["abstract"],
        meta={"corpus": "arxiv"},
    )


def web_source(h: dict[str, Any]) -> Source:
    return Source(
        id=source_id(h["url"]),
        url=h["url"],
        title=h["title"],
        date=_day(h.get("published_date")),
        snippet=_clip(h["content"], WEB_SNIPPET_CHARS),
        meta={"corpus": "web", "published_date": h.get("published_date")},
    )


def source_block(n: int, s: Source) -> str:
    """'[n] Title (Authors, date). URL' followed by the snippet."""
    authors = ", ".join(s.authors[:3]) + (" et al." if len(s.authors) > 3 else "")
    byline = ", ".join(x for x in (authors, s.date) if x)
    return f"[{n}] {s.title}{f' ({byline})' if byline else ''}. {s.url}\n{s.snippet}"


class SimpleRAG(System):
    class Params(Strict):
        k: int | None = None  # sources per corpus (default: the experiment's search.k)
        query: Literal["llm", "raw"] = "llm"  # raw: the benchmark's own query (ResearchQA)
        sample: int = 0  # > 0: an independent writer sample (its own "write~<n>" stage)

    async def run(self, ctx: TaskContext) -> Report:
        task, k = ctx.task, self.p.k or ctx.search_config.k
        if self.p.query == "raw":
            query = task.meta["query"]
        else:
            reply = await chat(ctx, "query", QUERY_PROMPT.format(prompt=task.prompt))
            query = " ".join(reply.strip().strip("`\"'").split()[:10])
        sources, retrieval = await self.retrieve(ctx, query, k)
        listing = "\n\n".join(source_block(i, s) for i, s in enumerate(sources, 1)) or "(none)"
        write = f"write~{self.p.sample}" if self.p.sample else "write"
        text = await chat(ctx, write, WRITE_PROMPT.format(prompt=task.prompt, sources=listing))
        failed = all("error" in r for r in retrieval.values())
        return finalize(
            text,
            sources,
            status="degraded" if failed else "ok",
            query=query,
            k=k,
            retrieval=retrieval,
            retrieved=len(sources),
        )

    async def retrieve(
        self, ctx: TaskContext, query: str, k: int
    ) -> tuple[list[Source], dict[str, dict[str, Any]]]:
        """Top-k per corpus; a failed search costs its sources, not the task."""
        sources: list[Source] = []
        stats: dict[str, dict[str, Any]] = {}
        for corpus in ctx.search_config.corpora:
            search, convert = (
                (ctx.search.arxiv_search, arxiv_source)
                if corpus == "arxiv"
                else (ctx.search.web_search, web_source)
            )
            try:
                hits = await asyncio.to_thread(search, query, k)
            except ToolError as e:
                ctx.log.warning("simple_rag %s search failed: %s", corpus, e)
                stats[corpus] = {"error": str(e)}
                continue
            sources += map(convert, hits)
            stats[corpus] = {"hits": len(hits)}
        return sources, stats

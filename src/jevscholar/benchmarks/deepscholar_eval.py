"""DeepScholar-Bench scoring of one exported report, with the upstream metric classes untouched.

This re-implements `eval/main.py` per task. The evaluators are built once per worker and dataset
(important citations parsed once, ReferenceCoverage's dump of them silenced). Every metric of
every report gets its own tagged judge LM (eval/main.py's `LM(model=...)` defaults), so the ledger
attributes judge spend per (task, metric); the nuggetizer, which builds its own OpenAI client,
is re-pointed per report. Two lookups are re-routed: OpenAlex citation counts
(`openalex.cited_by_count`: keyed, cached, a 429 is never "no match") and arXiv metadata (our
cached, rate-limited backend). Departures from upstream, all explicit: a report without parsable
arXiv citations, which upstream drops, is still scored on the two text-only metrics (organization,
nugget coverage); judge or lookup failures, which upstream swallows or aborts on, become per-metric
"error:" statuses plus a top-level "error", so the runner re-judges the task. Among them are the
nuggetizer's: it turns a judge call that raises (a timeout, a gateway refusal) into "failed"
assignments, which upstream then scores as unsupported. The organization verdicts, every
support verdict (cite_p, claim_coverage, claim_support) and every nugget assignment are kept as
`items`.

Two metrics of ours stand next to upstream's, which stay as coded for comparability with the
paper (docs/03_baselines_and_metrics.md describes both upstream defects):
- `reference_recall`: reference coverage matched by arXiv id. Upstream's ReferenceCoverage only
  sees cited titles and abstracts, so it credits a different paper with a similar title.
- `claim_support`: claim coverage as upstream means it. Its ClaimCoverageEvaluator never reads
  the neighbouring sentences' citations and adds the target paper's own abstract to the evidence.

Each metric carries a protocol version (`deepscholar.METRICS`), recorded per eval file. Re-judging
a task (after an error, with `jev eval --redo <metric>`, or for a metric not judged yet or judged
at another version) recomputes only those metrics; the others, and their items, are reused
unchanged. A run may judge a subset of the metrics (`EvalContext.metrics`): a metric neither
requested nor settled before is "not_requested".
Document importance needs an OpenAlex key (its keyless quota is ~100 searches a day, far below a
dev run): without JEV_OPENALEX_KEY it is "skipped:no_openalex_key" for every system alike.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from jevscholar import data
from jevscholar.benchmarks.deepscholar import METRICS, gold_found
from jevscholar.search import openalex
from jevscholar.search.policy import arxiv_id, norm_title
from jevscholar.types import CLIENT_TIMEOUT_S, LOCAL_KEY, EvalContext, Task, task_key
from jevscholar.util import read_json, vendored

TEXT_ONLY = ("organization", "nugget_coverage", "reference_recall")  # defined without citations
FINAL = ("ok", "nan_as_zero")  # statuses a re-judge keeps
NO_KEY = "skipped:no_openalex_key"
NO_GOLD = "no_important_citations"  # reference_recall is undefined (upstream scores 0)
ARXIV_ABS = re.compile(r"arxiv\.org/abs/([^)\s]+)")  # the parser's citation ids
ORGANIZATION = (
    "Trial {k} of 2 (trial 2 presents the two sections in swapped order). 1 = the report's "
    "related-work section is better organized and more coherent than the paper's own; 0 = the "
    "paper's own is.\n\nPaper: {title}\n\nThe paper's own related-work section:\n"
    "{related_works_section}"
)
_DECISIONS: list[dict[str, Any]] = []  # judge decisions on the report being scored


def setup(ctx: EvalContext) -> None:
    """Fail before any judge call if NLTK's punkt_tab (cite_p, claim_coverage) is missing."""
    import nltk

    upstream()
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        raise RuntimeError(
            "NLTK 'punkt_tab' is missing; run: "
            "uv run python -m nltk.downloader -d .cache/nltk_data punkt_tab"
        ) from None


@cache
def upstream() -> None:
    """Import the evaluators once per process: arXiv lookups re-routed, support verdicts kept."""
    vendored("deepscholar")
    import eval.evaluator.cite_p as cite_p
    import eval.evaluator.claim_coverage as claim_coverage
    import eval.evaluator.coverage_relevance_rate as relevance

    relevance.get_arxiv_title_and_abstract = _arxiv_title_abstract
    cite_p.get_support = _recorded("cite_p", cite_p.get_support)
    claim_coverage.get_support = _recorded("claim_coverage", claim_coverage.get_support)


def _item(metric: str, decision: Any, context: str) -> dict[str, Any]:
    """One judge decision (see review.decisions): a 0/1 verdict, or a nugget's label."""
    return {
        "metric": metric,
        "decision": decision if isinstance(decision, str) else int(decision),
        "context": context,
    }


def _recorded(metric: str, get_support: Callable[[str, str], int]) -> Callable[[str, str], int]:
    def support(premise: str, hypothesis: str) -> int:
        decision = get_support(premise, hypothesis)
        _DECISIONS.append(_item(metric, decision, f"Claim: {hypothesis}\n\nReference: {premise}"))
        return decision

    return support


def _arxiv_title_abstract(arxiv_id: str) -> tuple[str, str]:
    """Drop-in for upstream's arXiv page scrape: the selected backend's cached lookup."""
    from jevscholar.search import arxiv_backend

    papers, _ = arxiv_backend().lookup([arxiv_id])
    paper = next(iter(papers.values()), None)
    return (paper["title"], paper["abstract"]) if paper else ("", "")


def _paths(dataset: str) -> tuple[Path, Path, Path]:
    """(papers csv, important-citations csv, nuggets-by-row dir) of one DeepScholar dataset."""
    raw = data.RAW / "deepscholar" / dataset
    nuggets = data.DERIVED / "deepscholar" / dataset / "nuggets"
    return raw / "papers_with_related_works.csv", raw / "important_citations.csv", nuggets


@cache
def _dataset(dataset: str) -> pd.DataFrame:
    return pd.read_csv(_paths(dataset)[0])


@cache
def _evaluators(dataset: str) -> dict[str, Any]:
    """metric -> upstream evaluator, constructed exactly as EvaluationFunction.calculate does."""
    import lotus
    from eval.evaluator import EvaluationFunction
    from lotus.models import LM

    _, citations, nuggets = _paths(dataset)
    lotus.settings.configure(lm=LM(model="set-per-report"))  # the Nuggetizer reads lm.model
    with contextlib.redirect_stdout(io.StringIO()):  # ReferenceCoverage prints the whole dict
        rc = EvaluationFunction.REFERENCE_COVERAGE.to_evaluator()(
            important_citations_path=str(citations)
        )
    kw = {
        "important_citations": rc.important_citations,
        "nugget_groundtruth_dir_path": str(nuggets),
    }
    return {
        ef.value: rc if ef is EvaluationFunction.REFERENCE_COVERAGE else ef.to_evaluator()(**kw)
        for ef in EvaluationFunction
    }


def _configure_lm(model: str, api_base: str) -> None:
    """eval/main.py's judge LM (LOTUS defaults: temperature 0, 512 tokens) on a tagged URL."""
    import lotus
    from lotus.models import LM

    lm = LM(
        model=model,
        api_base=api_base,
        custom_llm_provider="openai",
        api_key=LOCAL_KEY,
        temperature=0.0,
        max_tokens=512,
        max_batch_size=16,
        timeout=CLIENT_TIMEOUT_S,
        max_retries=0,
    )
    lotus.settings.configure(lm=lm, enable_cache=False)


def _watch(nuggetizer: Any) -> dict[str, str]:
    """Keep each nugget's assignment, and expose what NuggetCoverageEvaluator swallows as
    seen["error"]: assign() raising, or assignments it lost ('failed' ones, from a judge call
    that raised, or dropped labels)."""
    seen: dict[str, str] = {}
    assign = type(nuggetizer).assign

    def watched(query: str, context: str, nuggets: list) -> list:
        try:
            out = assign(nuggetizer, query, context, nuggets)
        except Exception as e:
            seen["error"] = type(e).__name__
            raise
        if len(out) < len(nuggets) or any(n.assignment == "failed" for n in out):
            seen["error"] = "nugget_failed"
        _DECISIONS.extend(
            _item("nugget_coverage", a.assignment, n.text) | {"importance": n.importance}
            for n, a in zip(nuggets, out, strict=False)
        )
        return out

    nuggetizer.assign = watched
    return seen


def _previous(ctx: EvalContext, task: Task) -> dict[str, Any]:
    """This task's earlier eval file in the cell (empty if none)."""
    path = ctx.out_dir / f"{task_key(task)}.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    return {
        **prev,
        "metrics": prev.get("metrics") or {},
        "status": prev.get("status") or {},
        "versions": prev.get("versions") or {},
    }


def _settled(prev: dict[str, Any], name: str) -> str | None:
    """The status an earlier judging settled `name` with at its current version, if any."""
    status = prev["status"].get(name)
    current = prev["versions"].get(name) == METRICS[name]
    return status if current and status not in (None, "not_requested") else None


def _reference_recall(parser: Any, task: Task) -> float | None:
    """The share of the task's important citations (rows with an arXiv link, as upstream counts
    them) the report cites: by versionless arXiv id, or by normalized title for a citation
    without a parsable id. None when the task has none (upstream scores 0)."""
    urls = (url for _, url in parser.citation_pattern.findall(parser.raw_generated_text))
    ids = {arxiv_id(m[1]) for url in urls if (m := ARXIV_ABS.search(url))}
    found, total = gold_found(task, ids, {norm_title(doc["title"]) for doc in parser.docs})
    return found / total if total else None


def _claim_support(parser: Any, task: Task, window: int = 1) -> float:
    """The share of claims (sentences of at least 50 characters, citation markers removed) that a
    document cited in the sentence or in one of its `window` neighbours supports, the sentence's
    own citations asked first, each set of documents once. NaN without claims (as upstream)."""
    from eval.evaluator import claim_coverage as cc
    from eval.prompts.support import get_support

    support = _recorded("claim_support", get_support)
    sentences = cc.custom_sent_tokenize(parser.clean_text)
    covered = []
    for sid, sentence in enumerate(sentences):
        claim = cc._remove_citations(sentence)
        if len(claim) < 50:
            continue
        near = range(max(0, sid - window), min(len(sentences), sid + window + 1))
        asked: set[str] = set()
        for i in sorted(near, key=lambda i: abs(i - sid)):
            refs = sorted({int(x[1:]) - 1 for x in re.findall(r"\[\d+", sentences[i])})
            docs = parser.citations_for_cite_quality
            evidence = " ".join(cc._format_document(docs[r]) for r in refs if 0 <= r < len(docs))
            if evidence and evidence not in asked:
                asked.add(evidence)
                if support(evidence, claim):
                    covered.append(1)
                    break
        else:
            covered.append(0)
    return sum(covered) / len(covered) if covered else math.nan


OURS: dict[str, Callable[[Any, Task], float | None]] = {
    "reference_recall": _reference_recall,
    "claim_support": _claim_support,
}


def evaluate(task: Task, export: Path, ctx: EvalContext) -> dict[str, Any]:
    """Score `export/<row>/` (intro.md + paper.csv) on every DeepScholar metric."""
    upstream()
    import eval.evaluator.document_importance as document_importance
    from eval.parsers import ParserType, get_parser

    dataset, row = task.meta["dataset"], str(task.meta["row"])
    config = {"mode": ParserType.DEEPSCHOLAR_BASE, "file_id": row, "dataset": _dataset(dataset)}
    parser = get_parser(config, str(export / row))
    truth = parser.s_map_groundtruth
    nugget_qid = read_json(_paths(dataset)[2] / row / "res.json")["qid"]
    if not truth["arxiv_id"] == task.id == nugget_qid:
        raise ValueError(
            f"row {row}: dataset id {truth['arxiv_id']}, task id {task.id} and "
            f"nugget qid {nugget_qid} disagree"
        )
    names = (*METRICS, "relevance_rate")
    metrics: dict[str, float | None] = dict.fromkeys(names)
    if not parser.clean_text:
        status = dict.fromkeys(names, "empty_export")
        return {"metrics": metrics, "status": status, "versions": dict(METRICS)}
    status = dict.fromkeys(names, "no_arxiv_citations")
    evaluators = _evaluators(dataset)
    items: list[dict[str, Any]] = []
    _DECISIONS.clear()
    prev = _previous(ctx, task)
    for name in METRICS:
        wanted = not ctx.metrics or name in ctx.metrics
        settled = _settled(prev, name)
        if settled in FINAL or (settled and not wanted):  # keep what an earlier judging settled
            metrics[name], status[name] = prev["metrics"][name], settled
            items += [it for it in prev.get("items", []) if it.get("metric") == name]
            continue
        if not wanted:
            status[name] = "not_requested"
            continue
        if not (parser.docs or name in TEXT_ONLY):
            continue  # no_arxiv_citations
        if name == "document_importance" and not os.environ.get("JEV_OPENALEX_KEY"):
            status[name] = NO_KEY
            continue
        base, alias = ctx.base_url(task, name), ctx.judge(name)
        if name != "reference_recall":  # LLM-free
            _configure_lm(alias, base)
        evaluator, seen = evaluators.get(name), {}
        if name == "nugget_coverage":
            llm = evaluator.nuggetizer.assigner_llm
            llm.model = alias
            llm.client = OpenAI(
                base_url=base, api_key=LOCAL_KEY, max_retries=0, timeout=CLIENT_TIMEOUT_S
            )
            seen = _watch(evaluator.nuggetizer)
        elif name == "document_importance":
            document_importance.get_citation_count_from_title = partial(
                openalex.cited_by_count, meter=ctx.meter(task, name)
            )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if name in OURS:
                    value = OURS[name](parser, task)
                else:
                    scores = evaluator.calculate([parser]).iloc[0]
                    value = float(scores[name])
        except Exception as e:  # upstream would abort the whole run
            ctx.log.warning("deepscholar %s %s: %s: %s", task.id, name, type(e).__name__, e)
            status[name] = f"error:{type(e).__name__}"
            continue
        if value is None:
            metrics[name], status[name] = None, NO_GOLD
        elif math.isnan(value):
            metrics[name], status[name] = 0.0, "nan_as_zero"
        else:
            metrics[name], status[name] = value, "ok"
        if seen.get("error"):
            metrics[name], status[name] = None, f"error:{seen['error']}"
        if name == "organization":  # both trials' verdicts
            items += [
                _item(name, scores[f"organization_v{k}"], ORGANIZATION.format(k=k, **truth))
                for k in (1, 2)
            ]
    raw = metrics["coverage_relevance_rate"]
    metrics["relevance_rate"] = None if raw is None else raw / 2  # judge scale 0-2 -> 0-1
    status["relevance_rate"] = status["coverage_relevance_rate"]
    important = evaluators["reference_coverage"].important_citations.get(task.id, [])
    out = {
        "metrics": metrics,
        "status": status,
        "versions": {
            n: v
            for n, v in METRICS.items()
            if status[n] != "not_requested" and not status[n].startswith("error:")
        },
        "items": [*items, *_DECISIONS],
        "diagnostics": {"n_docs": len(parser.docs), "n_important": len(important)},
    }
    if failed := [n for n in METRICS if status[n].startswith("error:")]:
        out["error"] = f"metrics failed: {', '.join(failed)}"
    return out

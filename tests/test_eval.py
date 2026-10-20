"""Evaluator wrappers: runtime patches, judge routing and metric wiring, with fake judges."""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import docx
import lotus
import nltk
import pandas as pd
import pytest

from jevscholar.benchmarks import deepscholar as ds_bench
from jevscholar.benchmarks import deepscholar_eval, drb2_eval, researchqa_eval
from jevscholar.config import load_models
from jevscholar.judging import judges_for
from jevscholar.paths import ROOT, VENDOR
from jevscholar.search import arxiv_backend, openalex
from jevscholar.types import CLIENT_TIMEOUT_S, EvalContext, Task

SIBLING = Path.home() / "Projects/PyCharm/FinalProj_FrontierAI"
SIBLING_RQA = SIBLING / "experiments/ideation_20261005/pilots/_ground_benchmarks/raw/researchqa"


def _ctx(tmp_path: Path, judgeset: str, bench: str) -> EvalContext:
    """The context the runner builds: judges from models.yaml via judging.judges_for."""
    judges = judges_for(load_models().judgesets[judgeset], bench)
    return EvalContext(
        gateway="http://gw",
        run="r1",
        system="sys",
        judgeset=judgeset,
        judges=judges,
        out_dir=tmp_path / "eval",
    )


# --------------------------------------------------------------------------- ResearchQA
class FakeJudge:
    """OpenAI-shaped client replaying scripted replies per answer (the prompt embeds it)."""

    def __init__(self, replies: dict[str, list], **client_kw: object) -> None:
        self.replies = {k: list(v) for k, v in replies.items()}
        self.client_kw, self.calls = client_kw, []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kw: object) -> SimpleNamespace:
        self.calls.append(kw)
        answer = kw["messages"][0]["content"].split("Response: ", 1)[1].split("\n", 1)[0]
        reply = self.replies[answer].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def test_researchqa_matches_oracle_and_raises_on_infrastructure(tmp_path, monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(time, "sleep", waits.append)
    calls = []

    @researchqa_eval.retry_shim().retry(tries=3, delay=1, backoff=2, exceptions=KeyError)
    def flaky(failures: int, exc: type[Exception] = KeyError) -> str:
        calls.append(1)
        if len(calls) <= failures:
            raise exc("boom")
        return "ok"

    assert flaky(2) == "ok" and len(calls) == 3 and waits == [1, 2]
    calls.clear()
    with pytest.raises(KeyError):  # gives up after `tries`, re-raising the original exception
        flaky(9)
    calls.clear()
    with pytest.raises(ValueError):  # other exceptions are never retried
        flaky(1, ValueError)
    assert len(calls) == 1

    src = next(
        (d for d in (researchqa_eval.CODE, SIBLING_RQA) if (d / "compute_coverage.py").exists()),
        None,
    )
    oracle_file = SIBLING / "src/frontier_research/researchqa_scoring.py"
    if src is None or not oracle_file.exists():
        pytest.skip("needs the upstream scorer (jev data fetch) and the sibling oracle")
    code = tmp_path / "code"  # copies: importing must not write bytecode next to the sources
    code.mkdir()
    for name in ("compute_coverage.py", "researchqa_loader.py"):
        shutil.copy(src / name, code / name)
    shutil.copy(oracle_file, tmp_path / "rqa_oracle.py")
    spec = importlib.util.spec_from_file_location("rqa_oracle", tmp_path / "rqa_oracle.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)

    types_ = [
        ["Other"],
        ["Citation"],
        ["Impact", "Comparison"],
        ["Other"],
        ["Impact"],
        ["Citation"],
        ["Other"],
        ["Example"],
        ["Impact"],
    ]

    def item(i: str, n: int) -> dict:
        rubric = [{"rubric_item": f"{i} question {k}?", "type": types_[k]} for k in range(n)]
        return {
            "id": i,
            "general_domain": "d",
            "subdomain": "s",
            "field": "f",
            "query": "q",
            "date": "2024",
            "rubric": rubric,
        }

    valid = [item("ok", 9), item("bad", 2), item("fail", 2)]
    (tmp_path / "valid.json").write_text(json.dumps(valid))
    (tmp_path / "test.json").write_text(json.dumps([item("down", 2), item("empty", 2)]))
    labels = [
        "Completely",
        "Mostly",
        "Barely",
        "Not at all",
        "Moderately",
        "Completely",
        "Mostly",
        "Barely",
    ]
    replies = {
        "answer-ok": ["\n".join(labels[:7]), "\n".join(labels), "Mostly"],  # 1 retry
        "answer-bad": ["Mostly\nSomewhat"],  # a label outside the five
        "answer-fail": ["Mostly"] * 3,  # wrong count on every try: upstream skips it
        "answer-down": [ConnectionError("gateway refused")],
    }  # infrastructure: raises
    answers = {i: {"answer": f"answer-{i}"} for i in ("ok", "bad", "fail", "down")}
    answers["empty"] = {"answer": " "}

    monkeypatch.setattr(sys, "path", list(sys.path))
    for module in ("retry", "compute_coverage", "researchqa_loader"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(researchqa_eval, "CODE", code)
    monkeypatch.setattr(researchqa_eval, "RAW", tmp_path)
    researchqa_eval.upstream.cache_clear()
    researchqa_eval._items.cache_clear()
    judges: list[FakeJudge] = []
    monkeypatch.setattr(
        researchqa_eval,
        "OpenAI",
        lambda **kw: judges.append(FakeJudge(replies, **kw)) or judges[-1],
    )
    export = tmp_path / "export"
    export.mkdir()
    for i, a in answers.items():
        (export / f"{i}.json").write_text(json.dumps(a))
    ctx = _ctx(tmp_path, "deepseek", "researchqa")
    try:
        ours = {
            i: researchqa_eval.evaluate(Task("researchqa", i, "p"), export, ctx)
            for i in ("ok", "bad", "fail", "empty")
        }
        with pytest.raises(ConnectionError):
            researchqa_eval.evaluate(Task("researchqa", "down", "p"), export, ctx)
    finally:
        researchqa_eval.upstream.cache_clear()
        researchqa_eval._items.cache_clear()

    valid_answers = {i: answers[i] for i in ("ok", "bad", "fail")}
    ref = oracle.evaluate_coverage(
        oracle.load_evaluation_items(tmp_path / "valid.json"),
        valid_answers,
        client=FakeJudge(replies),
        model="ds-judge-chat",
        sleep=lambda s: 0,
    )
    status = {
        "completed": "ok",
        "invalid_judge_labels": "invalid_label",
        "judge_failure": "judge_failure",
    }
    for row in ref["items"]:
        got = ours[row["id"]]
        assert got["status"]["coverage"] == status[row["status"]]
        if row["coverage"] is None:
            assert got["metrics"]["coverage"] is None
        else:
            assert got["metrics"]["coverage"] == pytest.approx(row["coverage"])
    cited = [j for j, t in zip(ref["rubric_judges"]["ok"], types_, strict=True) if "Citation" in t]
    assert ours["ok"]["metrics"]["coverage_citation"] == pytest.approx(
        oracle.compute_coverage_score(cited)
    )
    assert ours["empty"] == {"metrics": {"coverage": None}, "status": {"coverage": "empty_export"}}
    assert len(judges) == 4  # one client per judged task; none for the empty export
    first = judges[0]
    assert first.client_kw["base_url"] == ctx.base_url(Task("researchqa", "ok", "p"), "coverage")
    assert first.client_kw["timeout"] == CLIENT_TIMEOUT_S
    assert {c["model"] for c in first.calls} == {"ds-judge-chat"}
    assert {c["temperature"] for c in first.calls} == {0}
    raw = (ctx.out_dir / "raw/ok.json").read_text()
    assert raw.endswith("]\n") and json.loads(raw)[0]["score"] == "Completely"


# --------------------------------------------------------------------------- DRB2
def test_drb2_matches_aggregate_scores_and_raises_on_failure(tmp_path, monkeypatch):
    run_evaluation, _ = drb2_eval.upstream()
    expected = {"results": [{"rubric_item": "a", "score": 1, "reason": "ok", "evidence": "x"}]}
    assert run_evaluation.parse_model_text(json.dumps(expected)) == (expected, True)  # PR #8
    assert run_evaluation.parse_model_text(f"```json\n{json.dumps(expected)}\n```") == (
        expected,
        True,
    )
    assert run_evaluation.parse_model_text('{"rubric_"item": 1}') == ({'rubric_"item': 1}, True)

    rubric = {"info_recall": ["r1", "r2", "r3"], "analysis": ["a1"], "presentation": ["p1", "p2"]}
    scores = {"r1": 1, "r2": -1, "r3": 0, "a1": 1, "p1": 1, "p2": 0}
    content = {"task": "T", "rubric": rubric, "blocked": {"title": "B", "urls": []}}
    monkeypatch.setattr(drb2_eval, "_tasks", lambda: dict.fromkeys((7, 8, 9), content))
    monkeypatch.setattr(drb2_eval, "CHUNK_SIZE", 4)  # two batches
    clients, papers = [], []

    def query(self, input_data):  # compact JSON, as a forced json_object judge returns it
        clients.append(self)
        papers.append(input_data.text)
        block = input_data.text.split("<task_and_rubric>\n")[1].split("\n</task_and_rubric>")[0]
        body = {
            "results": [
                {"rubric_item": i, "score": scores[i], "reason": "", "evidence": ""}
                for i in json.loads(block)["rubric_items"]
            ]
        }
        return SimpleNamespace(text=json.dumps(body), usage_metadata={})

    monkeypatch.setattr(run_evaluation.GPTClient, "query", query)
    export = tmp_path / "export"
    export.mkdir()
    (export / "idx-7.md").write_text("# Report\nbody", encoding="utf-8")
    report = docx.Document()
    report.add_paragraph("docx body")
    report.save(export / "idx-8.docx")  # a reference report, read natively by upstream
    (export / "idx-9.md").write_text("", encoding="utf-8")
    ctx, task = _ctx(tmp_path, "deepseek", "drb2"), Task("drb2", "7", "T")
    out = drb2_eval.evaluate(task, export, ctx)

    assert len(clients) == 2 and clients[0].api_url == (
        ctx.base_url(task, "rubric") + "/chat/completions"
    )
    c = clients[0]
    assert (c.model, c.reasoning_effort, c.timeout, c.max_output_tokens) == (
        "ds-judge-think",
        "medium",
        CLIENT_TIMEOUT_S,
        32768,
    )
    assert set(out["status"].values()) == {"ok"}
    raw = ctx.out_dir / "raw/idx-7.json"
    assert raw.read_text(encoding="utf-8").endswith("}\n")  # cat raw/*.json is valid JSONL
    subprocess.run(
        [
            sys.executable,
            str(VENDOR / "drb2/aggregate_scores.py"),
            "--input",
            str(raw),
            "--output-prefix",
            str(tmp_path / "agg"),
            "--tasks-file",
            str(tmp_path / "absent.jsonl"),
        ],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    for ours, suffix in [
        ("info_recall", "inforecall"),
        ("analysis", "analysis"),
        ("presentation", "presentation"),
        ("total", "total"),
        ("blocked_rate", "blocked"),
    ]:
        rows = list(csv.reader((tmp_path / f"agg_{suffix}.csv").open(encoding="utf-8")))
        assert out["metrics"][ours] == pytest.approx(float(rows[1][3]), abs=1e-6)
    assert out["metrics"]["total"] == pytest.approx(0.5)

    assert drb2_eval.evaluate(Task("drb2", "8", "T"), export, ctx)["metrics"] == out["metrics"]
    assert "docx body" in papers[-1]
    n = len(clients)
    empty = drb2_eval.evaluate(Task("drb2", "9", "T"), export, ctx)
    assert set(empty["status"].values()) == {"empty_export"} and len(clients) == n

    monkeypatch.setattr(
        run_evaluation.GPTClient,
        "query",
        lambda self, d: SimpleNamespace(text="", usage_metadata={}),
    )
    raw.unlink()
    with pytest.raises(RuntimeError, match="failed after 5 retries"):
        drb2_eval.evaluate(task, export, ctx)
    assert not raw.exists()


# --------------------------------------------------------------------------- DeepScholar
class Stub:
    def __init__(self, name: str, value: float | Exception, **extra: int) -> None:
        self.name, self.value, self.extra = name, value, extra

    def calculate(self, parsers: list) -> pd.DataFrame:
        if isinstance(self.value, Exception):
            raise self.value
        row = {"folder_path": parsers[0].folder_path, self.name: self.value, **self.extra}
        return pd.DataFrame([row])


class SupportStub(Stub):
    """claim_coverage through upstream's own (recorded) get_support, on a fake judge LM."""

    def calculate(self, parsers: list) -> pd.DataFrame:
        support = sys.modules["eval.evaluator.claim_coverage"].get_support
        self.value = float(support("Title: Cited\nS", "Prior work studies this problem."))
        return super().calculate(parsers)


class StubNuggetizer:
    def __init__(self) -> None:
        self.assigner_llm = SimpleNamespace(model=None, client=None)
        self.labels = ("support", "support")  # "failed": a judge call upstream's assign swallowed

    def assign(self, query: str, context: str, nuggets: list) -> list:
        return [SimpleNamespace(assignment=label) for label in self.labels]


class NuggetStub(Stub):
    def __init__(self) -> None:
        super().__init__("nugget_coverage", 0.5)
        self.nuggetizer = StubNuggetizer()

    def calculate(self, parsers: list) -> pd.DataFrame:
        nuggets = [
            SimpleNamespace(text=t, importance=i) for t, i in (("n1", "vital"), ("n2", "okay"))
        ]
        self.nuggetizer.assign("q", parsers[0].clean_text, nuggets)
        return super().calculate(parsers)


@pytest.fixture
def deepscholar(tmp_path, monkeypatch):
    """A one-row dataset, its nugget file and an export folder; evaluators and LMs stubbed."""
    paper = {
        "title": "T",
        "abstract": "A",
        "arxiv_link": "x",
        "arxiv_id": "2501.00001v1",
        "clean_latex_related_works": "R",
    }
    pd.DataFrame([paper]).to_csv(tmp_path / "papers.csv", index=False)
    (tmp_path / "nuggets/0").mkdir(parents=True)
    (tmp_path / "export/0").mkdir(parents=True)
    pd.DataFrame([{"id": "2401.00001v1", "title": "Cited", "snippet": "S"}]).to_csv(
        tmp_path / "export/0/paper.csv", index=False
    )
    monkeypatch.setattr(
        deepscholar_eval, "_paths", lambda ds: (tmp_path / "papers.csv", None, tmp_path / "nuggets")
    )
    rc = Stub("reference_coverage", 0.25)
    rc.important_citations = {"2501.00001v1": [{}, {}]}
    evaluators = {
        "organization": Stub("organization", 0.5, organization_v1=1, organization_v2=0),
        "nugget_coverage": NuggetStub(),
        "reference_coverage": rc,
        "coverage_relevance_rate": Stub("coverage_relevance_rate", 1.5),
        "document_importance": Stub("document_importance", RuntimeError("quota")),
        "cite_p": Stub("cite_p", float("nan")),
        "claim_coverage": SupportStub("claim_coverage", 0.0),
    }
    monkeypatch.setattr(deepscholar_eval, "_evaluators", lambda ds: evaluators)
    important = [  # cited at another version; a similar title under another id: not cited
        {"cited_paper_arxiv_link": "http://arxiv.org/abs/2401.00001v2", "cited_paper_title": "C"},
        {"cited_paper_arxiv_link": "http://arxiv.org/abs/2402.00009", "cited_paper_title": "Cited"},
    ]
    monkeypatch.setattr(ds_bench, "important_citations", lambda ds: {TASK.id: important})
    monkeypatch.setattr(nltk.data, "path", [str(ROOT / ".cache/nltk_data"), *nltk.data.path])
    lms: list[tuple[str, str]] = []
    monkeypatch.setattr(deepscholar_eval, "_configure_lm", lambda m, b: lms.append((m, b)))
    monkeypatch.setattr(lotus.settings, "lm", SimpleNamespace(get_completion=lambda s, u: "1"))
    deepscholar_eval._dataset.cache_clear()
    yield SimpleNamespace(path=tmp_path, evaluators=evaluators, lms=lms)
    deepscholar_eval._dataset.cache_clear()


def _write(fx: SimpleNamespace, intro: str, qid: str = "2501.00001v1") -> None:
    (fx.path / "export/0/intro.md").write_text(intro, encoding="utf-8")
    (fx.path / "nuggets/0/res.json").write_text(json.dumps({"qid": qid}), encoding="utf-8")


TASK = Task("deepscholar", "2501.00001v1", "p", meta={"dataset": "jun2025", "row": 0})


def test_deepscholar_metric_wiring_items_and_errors(deepscholar, tmp_path, monkeypatch):
    _write(
        deepscholar,
        "Prior work [Doe 2024-01-01](http://arxiv.org/abs/2401.00001v1) "
        "studies this problem in depth.",
    )
    monkeypatch.setenv("JEV_OPENALEX_KEY", "k")
    deepscholar.evaluators["nugget_coverage"].nuggetizer.labels = ("support", "failed")
    ctx = _ctx(tmp_path, "official", "deepscholar")
    out = deepscholar_eval.evaluate(TASK, tmp_path / "export", ctx)

    m, s = out["metrics"], out["status"]
    assert out["diagnostics"] == {"n_docs": 1, "n_important": 2} and "n_docs" not in m
    assert (m["organization"], m["reference_coverage"], m["claim_coverage"]) == (0.5, 0.25, 1.0)
    assert (m["coverage_relevance_rate"], m["relevance_rate"]) == (1.5, 0.75)
    assert (m["nugget_coverage"], s["nugget_coverage"]) == (None, "error:nugget_failed")
    assert (m["document_importance"], s["document_importance"]) == (None, "error:RuntimeError")
    assert (m["cite_p"], s["cite_p"]) == (0.0, "nan_as_zero")
    assert s["organization"] == s["relevance_rate"] == "ok"
    assert out["error"] == "metrics failed: nugget_coverage, document_importance"  # re-judged
    assert (m["reference_recall"], m["claim_support"]) == (0.5, 0.0)  # no claim of >= 50 chars
    assert s["claim_support"] == "nan_as_zero"
    assert out["versions"] == {
        n: v for n, v in deepscholar_eval.METRICS.items() if not s[n].startswith("error:")
    }
    nuggets = [it for it in out["items"] if it["metric"] == "nugget_coverage"]
    assert [(it["decision"], it["context"], it["importance"]) for it in nuggets] == [
        ("support", "n1", "vital"),
        ("failed", "n2", "okay"),
    ]
    org = [it for it in out["items"] if it["metric"] == "organization"]
    assert [it["decision"] for it in org] == [1, 0] and "Paper: T" in org[0]["context"]
    (claim,) = [it for it in out["items"] if it["metric"] == "claim_coverage"]
    assert claim["decision"] == 1 and "Claim: Prior work studies this problem." in claim["context"]
    official = {"nugget_coverage": "gpt-4.1-2025-04-14"}  # per-metric override, else the default
    assert deepscholar.lms == [
        (official.get(name, "gpt-4o-2024-08-06"), ctx.base_url(TASK, name))
        for name in deepscholar_eval.METRICS
        if name != "reference_recall"  # LLM-free
    ]
    llm = deepscholar.evaluators["nugget_coverage"].nuggetizer.assigner_llm
    assert llm.model == "gpt-4.1-2025-04-14"
    assert str(llm.client.base_url).rstrip("/") == ctx.base_url(TASK, "nugget_coverage")
    lookup = sys.modules["eval.evaluator.document_importance"].get_citation_count_from_title
    assert lookup.func is openalex.cited_by_count
    assert lookup.keywords["meter"] == ctx.meter(TASK, "document_importance")
    assert lookup.keywords["meter"].tag.endswith(":judge.document_importance")

    paper = {"title": "Cited", "abstract": "S"}
    monkeypatch.setattr(arxiv_backend(), "lookup", lambda ids: ({ids[0]: paper}, False))
    relevance = sys.modules["eval.evaluator.coverage_relevance_rate"]
    assert relevance.get_arxiv_title_and_abstract("2401.00001v1") == ("Cited", "S")


def test_deepscholar_qid_invariant_text_only_and_empty(deepscholar, tmp_path):
    ctx = _ctx(tmp_path, "deepseek", "deepscholar")
    _write(deepscholar, "No arXiv citations here at all, only prose.", qid="2501.99999v1")
    with pytest.raises(ValueError, match="disagree"):
        deepscholar_eval.evaluate(TASK, tmp_path / "export", ctx)
    _write(deepscholar, "No arXiv citations here at all, only prose.")
    out = deepscholar_eval.evaluate(TASK, tmp_path / "export", ctx)
    m, s = out["metrics"], out["status"]
    assert (m["organization"], m["nugget_coverage"]) == (0.5, 0.5) and "error" not in out
    assert (m["reference_recall"], s["reference_recall"]) == (0.0, "ok")  # cites none of them
    cited = set(m) - {"organization", "nugget_coverage", "reference_recall"}
    assert {m[k] for k in cited} == {None} and {s[k] for k in cited} == {"no_arxiv_citations"}
    assert [b.rsplit(".", 1)[-1] for _, b in deepscholar.lms] == [
        "organization/v1",
        "nugget_coverage/v1",
    ]
    deepscholar.lms.clear()
    _write(deepscholar, "  \n")
    out = deepscholar_eval.evaluate(TASK, tmp_path / "export", ctx)
    assert set(out["status"].values()) == {"empty_export"} and not deepscholar.lms
    assert out["versions"] == deepscholar_eval.METRICS


def test_deepscholar_judges_a_requested_subset(deepscholar, tmp_path):
    _write(deepscholar, "Prior work [Doe 2024-01-01](http://arxiv.org/abs/2401.00001v1) holds.")
    ctx = _ctx(tmp_path, "deepseek", "deepscholar")
    ctx = dataclasses.replace(ctx, metrics=("nugget_coverage",))
    out = deepscholar_eval.evaluate(TASK, tmp_path / "export", ctx)
    assert [b.rsplit(":", 1)[-1] for _, b in deepscholar.lms] == ["judge.nugget_coverage/v1"]
    assert out["metrics"]["nugget_coverage"] == 0.5 and "error" not in out
    others = {s for name, s in out["status"].items() if name != "nugget_coverage"}
    assert others == {"not_requested"}


def test_deepscholar_rejudge_recomputes_only_unsettled_metrics(deepscholar, tmp_path, monkeypatch):
    _write(deepscholar, "Prior work [Doe 2024-01-01](http://arxiv.org/abs/2401.00001v1) holds.")
    ctx = _ctx(tmp_path, "deepseek", "deepscholar")
    monkeypatch.delenv("JEV_OPENALEX_KEY", raising=False)
    nuggetizer = deepscholar.evaluators["nugget_coverage"].nuggetizer
    nuggetizer.labels = ("support", "failed")  # e.g. a nugget judge call timed out
    first = deepscholar_eval.evaluate(TASK, tmp_path / "export", ctx)
    assert first["status"]["document_importance"] == deepscholar_eval.NO_KEY  # same for all
    assert first["metrics"]["nugget_coverage"] is None
    assert first["error"] == "metrics failed: nugget_coverage"
    ctx.out_dir.mkdir(parents=True)
    (ctx.out_dir / f"{TASK.id}.json").write_text(json.dumps(first))
    monkeypatch.setenv("JEV_OPENALEX_KEY", "k")  # a key arrives: DI is judged now
    deepscholar.evaluators["document_importance"].value = 0.4
    deepscholar.evaluators["organization"].value = 0.9  # would change if it were re-judged
    nuggetizer.labels = ("support", "support")  # the nugget judge is back
    deepscholar.lms.clear()
    again = deepscholar_eval.evaluate(TASK, tmp_path / "export", ctx)
    assert [b.rsplit(":", 1)[-1] for _, b in deepscholar.lms] == [
        "judge.nugget_coverage/v1",
        "judge.document_importance/v1",
    ]
    assert (again["metrics"]["nugget_coverage"], again["status"]["nugget_coverage"]) == (0.5, "ok")
    assert again["metrics"]["document_importance"] == 0.4 and "error" not in again
    assert again["metrics"]["organization"] == first["metrics"]["organization"] == 0.5
    assert [it for it in again["items"] if it["metric"] == "organization"] == [
        it for it in first["items"] if it["metric"] == "organization"
    ]


def test_deepscholar_extends_a_subset_and_rejudges_a_new_version(
    deepscholar, tmp_path, monkeypatch
):
    _write(deepscholar, "Prior work [Doe 2024-01-01](http://arxiv.org/abs/2401.00001v1) holds.")
    ctx = _ctx(tmp_path, "deepseek", "deepscholar")
    ctx.out_dir.mkdir(parents=True)
    path = ctx.out_dir / f"{TASK.id}.json"

    def judge(*metrics: str) -> dict:
        deepscholar.lms.clear()
        out = deepscholar_eval.evaluate(
            TASK, tmp_path / "export", dataclasses.replace(ctx, metrics=metrics)
        )
        path.write_text(json.dumps(out))
        return out

    first = judge("nugget_coverage")
    assert first["versions"] == {"nugget_coverage": 1}
    again = judge("reference_recall", "claim_support")  # nugget coverage is kept, not re-judged
    assert [b.rsplit(":", 1)[-1] for _, b in deepscholar.lms] == ["judge.claim_support/v1"]
    assert (
        again["metrics"]["nugget_coverage"] == 0.5 and again["status"]["cite_p"] == "not_requested"
    )
    assert set(again["versions"]) == {"nugget_coverage", "reference_recall", "claim_support"}
    assert [it for it in again["items"] if it["metric"] == "nugget_coverage"] == [
        it for it in first["items"] if it["metric"] == "nugget_coverage"
    ]
    monkeypatch.setitem(deepscholar_eval.METRICS, "nugget_coverage", 2)  # its judging changed
    judge("nugget_coverage")
    assert [b.rsplit(":", 1)[-1] for _, b in deepscholar.lms] == ["judge.nugget_coverage/v1"]


def test_reference_recall_matches_ids_or_titles_without_ids(monkeypatch):
    rows = [
        {"cited_paper_arxiv_link": "arXiv:2002.10764v2", "cited_paper_title": "FairRec"},
        {"cited_paper_arxiv_link": "https://doi.org/10.1/x", "cited_paper_title": "Old Paper"},
        {"cited_paper_arxiv_link": "http://arxiv.org/abs/1904.10620", "cited_paper_title": "B"},
    ]
    monkeypatch.setattr(ds_bench, "important_citations", lambda ds: {TASK.id: rows})
    ids, titles = {"2002.10764", "2412.00424"}, {"fairsort", "old paper"}
    assert ds_bench.gold_found(TASK, ids, titles) == (2, 3)  # FairSort is not FairRec


def test_claim_support_reads_neighbours_and_only_cited_documents(monkeypatch):
    """Upstream's claim coverage reads the claim's own citations for every neighbour and adds
    the target's abstract to the evidence; the repair reads each neighbour's citations."""
    deepscholar_eval.upstream()
    monkeypatch.setattr(nltk.data, "path", [str(ROOT / ".cache/nltk_data"), *nltk.data.path])
    asked: list[tuple[str, str]] = []

    def support(premise: str, claim: str) -> int:
        asked.append((premise, claim))
        return int("Title: B" in premise)

    monkeypatch.setattr(sys.modules["eval.prompts.support"], "get_support", support)
    text = (
        "The first method extends the earlier baseline with a stronger encoder [1]. "
        "A second sentence states a related finding without any citation of its own. "
        "The third sentence draws on the other benchmark and its analysis here [2]."
    )
    parser = SimpleNamespace(
        clean_text=text,
        docs=[{"title": "A"}, {"title": "B"}],
        citations_for_cite_quality=[("A", "abstract A"), ("B", "abstract B")],
        s_map_groundtruth={"title": "Target", "abstract": "target abstract"},
    )
    deepscholar_eval._DECISIONS.clear()
    assert deepscholar_eval._claim_support(parser, TASK) == pytest.approx(2 / 3)
    doc_a, doc_b = "Title: A\nabstract A", "Title: B\nabstract B"
    assert [p for p, _ in asked] == [doc_a, doc_a, doc_b, doc_b]  # own first; each set once
    assert not any("target abstract" in p for p, _ in asked)
    assert [it["metric"] for it in deepscholar_eval._DECISIONS] == ["claim_support"] * 4


def test_controls_export_fixed_reports(tmp_path, monkeypatch):
    from jevscholar.checks import calibration

    a, b = "http://arxiv.org/abs/2401.00001v1", "http://arxiv.org/abs/2401.00002"
    text = f"X [A]({a}) and Y [B]({b}), again X [A]({a})."
    assert calibration.rotate_citations(text) == (
        f"X [A]({b}) and Y [B]({a}), again X [A]({b})."  # links point at the next paper
    )
    tasks = [
        Task(
            "deepscholar",
            f"250{i}.0000{i}v1",
            "p",
            meta={"dataset": "d", "row": i, "abstract": f"abs {i}"},
        )
        for i in (1, 2)
    ]
    source = tmp_path / "src"
    for t in tasks:
        (source / str(t.meta["row"])).mkdir(parents=True)
        (source / str(t.meta["row"]) / "intro.md").write_text(f"{t.id}: {text}")
        pd.DataFrame([{"id": "2401.00001v1", "title": "A", "snippet": "S"}]).to_csv(
            source / str(t.meta["row"]) / "paper.csv", index=False
        )
    gold = [
        {"cited_paper_arxiv_link": a, "cited_paper_title": "A", "cited_paper_abstract": "S"},
        {"cited_paper_arxiv_link": "no id", "cited_paper_title": "Z", "cited_paper_abstract": ""},
    ]
    monkeypatch.setattr(ds_bench, "important_citations", lambda ds: {tasks[0].id: gold})
    rows = [{"clean_latex_related_works": f"related {i}"} for i in range(3)]
    monkeypatch.setattr(ds_bench, "papers", lambda ds: rows)
    calibration.write_controls(tmp_path / "run", tasks, source)

    def export(control: str, row: int) -> tuple[str, pd.DataFrame]:
        folder = tmp_path / "run" / "deepscholar" / control / "export" / str(row)
        return (folder / "intro.md").read_text(), pd.read_csv(folder / "paper.csv", dtype=str)

    assert export("reference", 1)[0] == "related 1" and export("abstract", 2)[0] == "abs 2"
    assert export("reference", 1)[1].empty
    assert export("swapped", 1)[0].startswith(tasks[1].id) and export("swapped", 2)[0].startswith(
        tasks[0].id
    )
    assert export("permuted", 1)[0] == f"{tasks[0].id}: {calibration.rotate_citations(text)}"
    intro, papers = export("gold_cited", 1)
    assert intro == "Prior work [1](http://arxiv.org/abs/2401.00001)." and papers.id.tolist() == [
        "2401.00001"
    ]
    assert export("gold_cited", 2)[1].empty  # no important citations

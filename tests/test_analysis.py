"""Analysis on a fabricated run: failure-keeping means, unjudged metrics left out, sign-flip
p-values, cost aggregation."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from jevscholar import analysis, data
from jevscholar.benchmarks import deepscholar as ds_bench
from jevscholar.benchmarks.drb2 import DRB2
from jevscholar.config import load_models
from jevscholar.report import write_report
from jevscholar.types import make_tag

SYSTEMS = ["simple_rag", "deepscholar_base"]


def _write(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def run_dir(tmp_path):
    """drb2 x 3 tasks: simple_rag ok (task 3's eval errored); deepscholar_base ok/failed/missing."""
    _write(
        tmp_path / "manifest.json",
        {
            "run_id": "r",
            "tasks": {"drb2": ["1", "2", "3"]},
            "experiment": {
                "name": "x",
                "budget": {"task_usd": 0.305},
                "benchmarks": {"drb2": {"systems": SYSTEMS}},
            },
            "sessions": [{"git_sha": None, "git_dirty": True}],
        },
    )
    report = {"text": "one two three", "sources": [], "meta": {}}
    results = {("simple_rag", k): ("ok", report) for k in "123"}
    results[("deepscholar_base", "1")] = ("ok", report)
    results[("deepscholar_base", "2")] = ("failed", None)
    for (system, key), (status, rep) in results.items():
        _write(
            tmp_path / "drb2" / system / "tasks" / key / "result.json",
            {"status": status, "report": rep},
        )
    evals = {
        ("simple_rag", "1"): {"metrics": {"total": 0.2, "blocked_rate": None}},
        ("simple_rag", "2"): {"metrics": {"total": 0.4, "blocked_rate": None}},
        ("simple_rag", "3"): {"metrics": {}, "error": "Traceback: judge down"},
        ("deepscholar_base", "1"): {
            "metrics": {"total": 0.8, "blocked_rate": 0.5},
            "diagnostics": {"n_docs": 3},
        },
        ("deepscholar_base", "2"): {"metrics": {"total": 0.5, "blocked_rate": 0.0}},
    }  # stub
    for (system, key), ev in evals.items():
        _write(tmp_path / "drb2" / system / "eval" / "deepseek" / f"{key}.json", ev)
    events = [
        ("s0", "1", "agent", "llm", 0.10, 0.10),  # abandoned session, replayed in s1
        ("s1", "1", "agent", "llm", 0.10, 0.05),
        ("s1", "1", "agent", "llm", 0.20, 0.10),
        ("s1", "1", "tools", "tool", 0.008, 0.008),
        ("s1", "2", "agent", "llm", 0.30, 0.30),
        ("s2", "1", "judge.total", "llm", 0.05, 0.05),
    ]
    (tmp_path / "ledger.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "tag": make_tag("r", "drb2", "deepscholar_base", k, st),
                    "session": sess,
                    "kind": kind,
                    "name": "n",
                    "ts": f"2026-10-17T00:00:0{i}+00:00",
                    "cost_usd": c,
                    "billed_usd": b,
                }
            )
            + "\n"
            for i, (sess, k, st, kind, c, b) in enumerate(events)
        )
    )
    return tmp_path


def test_means_and_paired(run_dir):
    m = analysis.means(analysis.long_table(run_dir)).set_index(["system", "metric"])
    ds = m.loc["deepscholar_base", "total"]
    assert (ds.n, ds.n_valid) == (3, 1)
    assert ds.primary == pytest.approx(0.8 / 3) and ds.official == pytest.approx(0.8)
    sr = m.loc["simple_rag", "total"]  # the errored eval is unjudged: left out, not a 0
    assert (sr.n_valid, sr.n_unjudged) == (2, 1)
    assert (sr.primary, sr.official) == (pytest.approx(0.3), pytest.approx(0.3))
    br = m.loc["deepscholar_base", "blocked_rate"]  # lower is better: never zero-filled
    assert (br.n, br.n_valid, br.primary) == (2, 1, pytest.approx(0.5))
    assert m.loc["simple_rag", "blocked_rate"].n_valid == 0
    assert "n_docs" not in m.index.get_level_values("metric")

    assert analysis.sign_flip_p([1, 2, 3]) == pytest.approx(2 / 8)  # only +-(1,2,3) reach |6|
    assert analysis.sign_flip_p([1, -1]) == 1.0
    assert 0 < analysis.sign_flip_p([1.0] * 20) < 1e-3  # Monte-Carlo branch
    row = analysis.paired(analysis.long_table(run_dir), "simple_rag").iloc[0]
    assert (row.system, row.n, row.unjudged) == ("deepscholar_base", 2, 1)  # task 3: unjudged
    assert row["diff"] == pytest.approx((0.6 - 0.4) / 2)  # task 2: a failed generation scores 0
    assert row.ci_lo < row["diff"] < row.ci_hi


def test_costs_and_report(run_dir):
    ledger = analysis.ledger_frame(run_dir)
    per_system, per_stage = analysis.cost_tables(analysis.long_table(run_dir), ledger)
    ds = per_system.set_index("system").loc["deepscholar_base"]
    assert ds.list_usd == pytest.approx(0.608) and ds.billed_usd == pytest.approx(0.458)
    assert (ds.llm_usd, ds.tool_usd, ds.judge_usd) == pytest.approx((0.6, 0.008, 0.05))
    agent = per_stage.set_index(["system", "stage"]).loc["deepscholar_base", "agent"]
    assert agent.list_usd == pytest.approx(0.6) and agent.p50_usd == pytest.approx(0.3)

    text = write_report(run_dir, baseline="simple_rag").read_text()
    assert "DeepSeek-judged (non-official)" in text and "deepscholar_base" in text
    assert "| eval errors |" in text and "| index |" not in text
    steps = (run_dir / "drb2/deepscholar_base/tasks/1/trajectory.jsonl").read_text().splitlines()
    assert [json.loads(s)["cum_usd"] for s in steps] == pytest.approx([0.1, 0.3, 0.308])
    assert math.isclose(json.loads(steps[-1])["remaining_usd"], 0.305 - 0.308)  # over its cap
    with pytest.raises(ValueError, match="not a system of the run"):
        write_report(run_dir, baseline="deepscholar_base_intro")


def test_cost_deltas_cold_cost_and_cap(run_dir):
    aliases = load_models().aliases
    tokens = {"miss": 1000, "hit": 9000, "out": 100, "reasoning": 0}
    ev = {"kind": "llm", "name": "deepseek-flash", "cost_usd": 0.0, "tokens": tokens}
    assert analysis.cold_usd(ev, aliases) == pytest.approx((10_000 * 0.30 + 100 * 1.20) / 1e6)
    tool = {"kind": "tool", "name": "arxiv_search", "cost_usd": 0.001}
    assert analysis.cold_usd(tool, aliases) == 0.001  # tools and unknown aliases keep their cost

    long = analysis.long_table(run_dir)
    pairs = analysis.paired(long, "simple_rag").set_index("metric")
    assert list(pairs.index) == ["total", "cost_usd", "cold_usd"]
    assert pairs.loc["cost_usd", "diff"] == pytest.approx((0.308 + 0.30 + 0.0) / 3)
    exp = json.loads((run_dir / "manifest.json").read_text())["experiment"]
    caps = analysis.task_caps(exp)
    assert caps == {("drb2", "simple_rag"): 0.305, ("drb2", "deepscholar_base"): 0.305}
    per_system, _ = analysis.cost_tables(long, analysis.ledger_frame(run_dir), caps)
    ds, rag = per_system.set_index("system").loc[["deepscholar_base", "simple_rag"]].itertuples()
    assert (ds.max_usd, ds.over_cap, rag.over_cap) == (pytest.approx(0.308), 1, 0)  # one call past
    assert ds.mean_usd == pytest.approx((0.308 + 0.30) / 2)


def test_unjudged_reliability_is_not_evidence_of_non_inferiority():
    """Regression: nugget-only judging left cite_p unjudged, scored 0 on both sides: CI [0, 0]."""
    asked = {
        "metrics": {"nugget_coverage": 0.4, "cite_p": None},
        "status": {"nugget_coverage": "ok", "cite_p": "not_requested"},
    }
    assert analysis.judged(asked, "nugget_coverage") and not analysis.judged(asked, "cite_p")
    assert ds_bench.DeepScholarBench.derived_from["cite_p_per_citation"] == "cite_p"
    assert not analysis.judged({"metrics": {}, "error": "Traceback"}, "total")
    assert analysis.judged({"metrics": {}, "status": "no_report"}, "total")
    nothing_to_check = {"metrics": {"cite_p": None}, "status": {"cite_p": "no_arxiv_citations"}}
    assert analysis.judged(nothing_to_check, "cite_p")  # settled: scores 0

    long = pd.DataFrame(
        [
            {"bench": "deepscholar", "system": s, "task": t, "has_report": True, "metric": m,
             "value": v if m == "nugget_coverage" else np.nan, "judged": m == "nugget_coverage",
             "defined": True, "valid": m == "nugget_coverage", "cost_usd": 0.1, "cold_usd": 0.1}
            for s, v in (("base", 0.3), ("ctl", 0.4))
            for t in ("1", "2", "3")
            for m in ("nugget_coverage", "cite_p")
        ]
    )  # fmt: skip
    pairs = analysis.paired(long, "base").set_index("metric")
    assert pairs.loc["nugget_coverage", "diff"] == pytest.approx(0.1)
    cite = pairs.loc["cite_p"]
    assert (cite.n, cite.unjudged) == (0, 3) and math.isnan(cite.ci_lo)
    assert (
        analysis.means(long).set_index(["system", "metric"]).loc[("ctl", "cite_p")].n_unjudged == 3
    )


def test_reliability_counts_failed_generations_and_leaves_unjudged_out(tmp_path):
    """Regression: a failed generation's eval file has no cite_p, so it left the comparison."""
    keys, systems = ["1", "2", "3"], ["base", "ctl"]
    experiment = {
        "name": "x",
        "budget": {"task_usd": 1.0},
        "benchmarks": {"deepscholar": {"systems": systems}},
    }
    _write(
        tmp_path / "manifest.json",
        {"run_id": "r", "tasks": {"deepscholar": keys}, "experiment": experiment},
    )
    report = {"text": "t", "sources": [], "meta": {}}
    judged = {
        "metrics": {"nugget_coverage": 0.3, "cite_p": 0.5},
        "status": {"nugget_coverage": "ok", "cite_p": "ok"},
    }
    no_report = {"metrics": {}, "status": "no_report"}
    files = {  # (system, task): (result status, report, eval file or None)
        **{("base", k): ("ok", report, judged) for k in keys},
        ("ctl", "1"): ("ok", report, judged),
        ("ctl", "2"): ("failed", None, no_report),
        ("ctl", "3"): ("ok", report, None),  # not judged yet
    }
    for (system, key), (status, rep, ev) in files.items():
        cell = tmp_path / "deepscholar" / system
        _write(cell / "tasks" / key / "result.json", {"status": status, "report": rep})
        if ev is not None:
            _write(cell / "eval" / "deepseek" / f"{key}.json", ev)
    (tmp_path / "ledger.jsonl").write_text("")
    pairs = analysis.paired(analysis.long_table(tmp_path), "base").set_index("metric")
    cite = pairs.loc["cite_p"]
    assert (cite.n, cite.unjudged) == (2, 1)  # task 2: the failure scores 0; task 3: left out
    assert cite["diff"] == pytest.approx((0.0 - 0.5) / 2)


def test_per_citation_precision_is_derived_from_items():
    items = [{"metric": "cite_p", "decision": d} for d in (1, 1, 0)]
    ev = {
        "metrics": {"cite_p": 0.2},
        "items": [*items, {"metric": "claim_coverage", "decision": 0}],
    }
    derived = ds_bench.DeepScholarBench().derived
    assert derived(ev) == {"cite_p_per_citation": pytest.approx(2 / 3)}
    failed = {"metrics": {"cite_p": None}, "items": items}  # the judge errored: no score
    assert derived(failed) == {"cite_p_per_citation": None}
    assert DRB2().derived(ev) == {} and derived({}) == {}
    labels = [("support", "vital"), ("partial_support", "vital"), ("not_support", "okay")]
    nuggets = [
        {"metric": "nugget_coverage", "decision": d, "importance": i, "context": "n"}
        for d, i in labels
    ]
    ev = {"metrics": {"nugget_coverage": 1 / 3, "cite_p": None}, "items": nuggets}
    split = derived(ev)
    assert (split["nugget_vital"], split["nugget_okay"]) == (0.5, 0.0)  # strict, as upstream


def test_undefined_reference_recall_leaves_its_task_out(tmp_path, monkeypatch):
    """A task without important citations has no reference recall (upstream scores it 0), on
    every system, whether or not its generation or judging succeeded."""
    keys = data.read_json(data.SPLITS / "deepscholar.json")["dev"][:3]
    row = {"cited_paper_arxiv_link": "http://arxiv.org/abs/2401.00001", "cited_paper_title": "x"}
    monkeypatch.setattr(ds_bench, "important_citations", lambda ds: {k: [row] for k in keys[:2]})
    cell = {"split": "dev", "limit": 3, "systems": ["base", "ctl"]}
    experiment = {"name": "x", "budget": {"task_usd": 1.0}, "benchmarks": {"deepscholar": cell}}
    _write(
        tmp_path / "manifest.json",
        {"run_id": "r", "tasks": {"deepscholar": keys}, "experiment": experiment},
    )
    for system, value in (("base", 0.2), ("ctl", 0.6)):
        for key in keys:
            cell_dir = tmp_path / "deepscholar" / system
            failed = system == "ctl" and key == keys[2]  # no report, no eval
            report = None if failed else {"text": "t", "sources": [], "meta": {}}
            status = "failed" if failed else "ok"
            _write(cell_dir / "tasks" / key / "result.json", {"status": status, "report": report})
            if not failed:
                recall = value if key != keys[2] else None
                ev = {"metrics": {"reference_recall": recall}, "status": {"reference_recall": "ok"}}
                _write(cell_dir / "eval" / "deepseek" / f"{key}.json", ev)
    (tmp_path / "ledger.jsonl").write_text("")
    long = analysis.long_table(tmp_path)
    row = analysis.means(long).set_index(["system", "metric"]).loc[("ctl", "reference_recall")]
    assert (row.n, row.primary, row.n_unjudged) == (2, pytest.approx(0.6), 0)  # not a 0
    pairs = analysis.paired(long, "base").set_index("metric")
    recall = pairs.loc["reference_recall"]
    assert (recall.n, recall.unjudged, recall["diff"]) == (2, 0, pytest.approx(0.4))
    assert pairs.loc["claim_support"].unjudged == 3  # a protocol metric not judged yet


def test_funnel_counts_important_citations_per_stage(tmp_path, monkeypatch):
    task = data.read_json(data.SPLITS / "deepscholar.json")["dev"][0]
    url = "http://arxiv.org/abs/2401.0000{}v1".format
    gold = [{"cited_paper_arxiv_link": url(i), "cited_paper_title": "x"} for i in "1234"]
    monkeypatch.setattr(ds_bench, "important_citations", lambda ds: {task: gold})
    cell = {"split": "dev", "limit": 1, "systems": ["ctl", "dsb"]}
    experiment = {"name": "x", "budget": {"task_usd": 1.0}, "benchmarks": {"deepscholar": cell}}
    manifest = {"run_id": "r", "tasks": {"deepscholar": [task]}, "experiment": experiment}
    cited = {"url": url(1), "title": "A", "meta": {"in_pool": True}}
    report = {"text": "t", "sources": [cited], "meta": {"docs": [url(1), url(2)]}}
    papers = [(1, True), (2, True), (5, False), (3, False)]  # 3 is screened out, 4 never found
    state = {"papers": [{"row": {"url": url(i), "title": "p"}, "relevant": r} for i, r in papers]}
    for system, extra in (("ctl", {"state.json": state}), ("dsb", {})):
        folder = tmp_path / "deepscholar" / system / "tasks" / task
        _write(folder / "result.json", {"status": "ok", "report": report})
        for name, content in extra.items():
            _write(folder / name, content)
    df = analysis.funnel(tmp_path, manifest).set_index("system")
    assert df.loc["ctl", ["n_gold", "screened", "relevant", "writer", "cited"]].tolist() == [
        4,
        3,
        2,
        2,
        1,
    ]
    assert df.loc["ctl", ["n_screened", "n_relevant", "n_writer", "n_cited"]].tolist() == [
        4,
        2,
        2,
        1,
    ]
    assert math.isnan(df.loc["dsb", "screened"]) and df.loc["dsb", "writer"] == 2
    summary = analysis.funnel_summary(df.reset_index()).set_index("system")
    assert summary.loc["ctl", "screened"] == summary.loc["ctl", "screened_micro"] == 0.75


def test_a_failed_continuation_keeps_its_inherited_cost(tmp_path):
    """The list cost before a continued run's decision point comes from its report, or, when it
    failed before reporting, from its first decision point."""
    assert analysis.offset_usd(tmp_path, {"meta": {"offset_usd": 0.02}}) == 0.02
    assert analysis.offset_usd(tmp_path, None) == 0.0  # not a continuation
    (tmp_path / "start.json").write_text(json.dumps({"offset_usd": 0.03}))  # failed, unreported
    assert analysis.offset_usd(tmp_path, None) == 0.03


def test_unknown_cold_cost_stays_unknown_in_the_cost_table():
    """A continuation's cache-cold cost is unknown: its system total is too, never a partial sum."""
    row = {"bench": "deepscholar", "status": "ok", "metric": "m", "billed_usd": 0.01}
    long = pd.DataFrame(
        [
            row | {"system": "s", "task": "a", "cost_usd": 0.02, "cold_usd": 0.03},
            row | {"system": "s", "task": "b", "cost_usd": 0.02, "cold_usd": math.nan},
            row | {"system": "t", "task": "a", "cost_usd": 0.02, "cold_usd": 0.03},
        ]
    )
    ledger = pd.DataFrame(
        columns=["bench", "system", "task", "stage", "kind", "cost_usd", "billed_usd"]
    )
    per_system, _ = analysis.cost_tables(
        long, ledger.astype({"cost_usd": float, "billed_usd": float})
    )
    cold = per_system.set_index("system").cold_usd
    assert math.isnan(cold["s"]) and cold["t"] == pytest.approx(0.03)


def test_sample_groups_merge_only_samples_of_one_system():
    """Regression: definitions without `use` ({} in older manifests) are their own classes, so
    different systems never merge as samples of each other."""
    experiment = {
        "systems": {
            "simple_rag": {},
            "closed_book": {},
            "k20": {"use": "simple_rag", "k": 20},
            "k20_s1": {"use": "simple_rag", "k": 20, "sample": 1},
        },
        "benchmarks": {"deepscholar": {"systems": ["simple_rag", "closed_book", "k20", "k20_s1"]}},
    }
    groups = analysis.sample_groups({"experiment": experiment})
    assert groups == {"simple_rag": "simple_rag", "closed_book": "closed_book", "k20": "k20",
                      "k20_s1": "k20"}  # fmt: skip

"""d1 as the relevance screener: the examples from logged verdicts and the scoring (offline)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from jevscholar.controller import screen
from jevscholar.controller.d1 import read_examples
from jevscholar.util import read_jsonl
from test_controlled import d1_snapshot_cached


def paper(pid: str, relevant: bool | None) -> dict:
    url = f"http://arxiv.org/abs/{pid}v1"
    return {"row": {"url": url, "title": f"T {pid}", "snippet": f"A {pid}"}, "relevant": relevant}


@pytest.fixture
def logged(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """Two runs over a train, a dev and a test task; the train task's papers seen twice."""
    runs, raw = tmp_path / "runs", tmp_path / "raw"
    states = {
        ("r1", "2501.00001v1"): [paper("1111.0001", True), paper("1111.0002", False)],
        ("r2", "2501.00001v1"): [paper("1111.0002", False), paper("1111.0003", None)],
        ("r1", "2502.00002v1"): [paper("2201.00001", False)],
        ("r1", "2503.00003v1"): [paper("3333.0001", True)],
    }
    for (run, task), papers in states.items():
        folder = runs / run / "deepscholar" / "ctl" / "tasks" / task
        folder.mkdir(parents=True)
        (folder / "state.json").write_text(json.dumps({"topic": f"topic {task}", "papers": papers}))
    (raw / "jun2025").mkdir(parents=True)
    (raw / "jun2025" / "recovered_citations.csv").write_text(
        "parent_paper_arxiv_id,cited_paper_arxiv_link\n"
        "2501.00001v1,http://arxiv.org/abs/1111.0001v2\n"
        "2502.00002v1,https://arxiv.org/abs/2201.00001\n"
        "2502.00002v1,\n"
    )
    split = {"2501.00001v1": "train", "2502.00002v1": "dev", "2503.00003v1": "test"}
    monkeypatch.setattr(screen, "splits", lambda: split)
    return runs, raw


def test_every_verdict_once_test_tasks_never(logged, tmp_path):
    runs, raw = logged
    counts = screen.screen_data(tmp_path / "out", runs, raw)
    assert counts == {
        "train": 2,
        "train_relevant": 1,
        "train_cited": 1,
        "dev": 1,
        "dev_relevant": 0,
        "dev_cited": 1,
    }
    train = read_examples(tmp_path / "out" / "train.jsonl")
    metas = read_jsonl(tmp_path / "out" / "train.meta.jsonl")
    assert [m["url"][-11:] for m in metas] == ["1111.0001v1", "1111.0002v1"]
    assert [(m["deepseek"], m["cited"], m["split"]) for m in metas] == [
        (True, True, "train"),
        (False, False, "train"),
    ]
    first = train[0]
    assert first.state == {
        "task": "topic 2501.00001v1",
        "article title": "T 1111.0001",
        "abstract": "A 1111.0001",
    }
    assert first.target == {"relevant": 0.9, "not relevant": pytest.approx(0.1)}
    assert train[1].label() == 1 and first.question["instructions"] == screen.INSTRUCTIONS


def test_the_report_against_deepseek_the_citations_and_as_a_cascade():
    metas = [
        {"deepseek": True, "cited": True},
        {"deepseek": True, "cited": False},
        {"deepseek": False, "cited": False},
        {"deepseek": False, "cited": True},  # DeepSeek drops a cited paper
    ]
    probs = [0.95, 0.4, 0.05, 0.6]
    r = screen.rates([p >= 0.5 for p in probs], metas)
    assert (r["agree"], r["kept"], r["cited_recall"]) == (0.5, 0.5, 1.0)
    assert math.isnan(screen.rates([True], [{"deepseek": True, "cited": False}])["cited_recall"])
    report = screen.screen_report(probs, metas).splitlines()
    assert report[0] == "4 verdicts, 2 of cited papers"
    assert "| DeepSeek (CoT sem_filter) | 1.000 | 0.500 | 0.500 | 0.000 |" in report
    assert "| d1, keep at P >= 0.3 | 0.750 | 0.750 | 1.000 | 1.000 |" in report
    # sure at 0.9: the first and third (d1 agrees there); DeepSeek decides the other two
    assert "| cascade, d1 when sure >= 0.9 | 1.000 | 0.500 | 0.500 | 0.500 |" in report


def test_scores_average_both_option_orders(logged, tmp_path, monkeypatch):
    runs, raw = logged
    screen.screen_data(tmp_path / "out", runs, raw)

    class Fake:
        def probabilities(self, items):  # P(first option) = 0.8: a pure position bias
            return [[0.8, 0.2] for _ in items]

    from jevscholar.controller import d1

    monkeypatch.setattr(d1.D1, "load", classmethod(lambda cls, adapter=None: Fake()))
    report = screen.score(tmp_path / "out" / "train.jsonl", tmp_path / "p.jsonl")
    rows = read_jsonl(tmp_path / "p.jsonl")
    assert [r["p_relevant"] for r in rows] == [pytest.approx(0.5)] * 2  # the bias cancels
    assert rows[0]["task"] == "2501.00001v1" and report.startswith("2 verdicts")


@pytest.mark.skipif(not d1_snapshot_cached(), reason="needs d1's prompt module (HF cache)")
def test_d1_reads_the_paper_and_the_question():
    from jevscholar.controller.d1 import user_turn

    e = screen.example("Graph agents", {"title": "T", "snippet": "We plan."}, True)
    text = user_turn(e.state, e.question)
    assert "Graph agents" in text and "We plan." in text and screen.INSTRUCTIONS in text
    assert "relevant" in text and "not relevant" in text

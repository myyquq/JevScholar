"""d1 as the relevance screener: the examples from logged verdicts and the scoring (offline)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from jevscholar.controller import screen
from jevscholar.controller.d1 import read_examples
from jevscholar.util import read_jsonl
from test_controlled import (  # noqa: F401 (fixtures)
    d1_snapshot_cached,
    gateway,
    lotus_fakes,
    run_task,
)


def paper(pid: str, relevant: bool | None) -> dict:
    url = f"http://arxiv.org/abs/{pid}v1"
    return {"row": {"url": url, "title": f"T {pid}", "snippet": f"A {pid}"}, "relevant": relevant}


@pytest.fixture
def logged(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """Two runs over a train, a dev and a test task; the train task's papers seen twice (no val
    task: its files are empty)."""
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
        "val": 0,
        "val_relevant": 0,
        "val_cited": 0,
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


def test_the_screen_recipe_trains_on_train_and_chooses_on_val():
    from jevscholar.controller.d1 import load_recipe
    from jevscholar.paths import CONFIGS

    cfg = load_recipe(CONFIGS / "training.yaml", "screen")
    assert (cfg.train.name, cfg.dev.name) == ("train.jsonl", "val.jsonl")
    assert cfg.train.parent.name == "screen" and cfg.out.name == "screen"


# --------------------------------------------------------------------------- in the loop
class FakeMeter:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def admit(self, tool: str) -> None:
        self.calls.append(("admit", tool))

    def charge(self, tool: str, n: int, meta: dict) -> None:
        self.calls.append(("charge", tool, n, meta["screen"]))


def test_served_probabilities_average_both_orders_and_are_charged(monkeypatch):
    import requests

    posted = []

    class Reply:
        def __init__(self, status: int, body: dict) -> None:
            self.status_code, self.body = status, body

        def json(self) -> dict:
            return self.body

        def raise_for_status(self) -> None:
            pass

    def post(url, json, timeout):  # the first option gets 0.8 in either order
        posted.append(json)
        if len(json["requests"]) > 2:
            return Reply(413, {"tokens": 99999})
        return Reply(200, {"probabilities": [[[0.8, 0.2], [0.8, 0.2]]] * 2, "tokens": 2500})

    monkeypatch.setattr(requests, "get", lambda url, timeout: Reply(200, {"adapter": "sha"}))
    monkeypatch.setattr(requests, "post", post)
    rows, meter = [{"title": "T", "snippet": "A"}] * 2, FakeMeter()
    probs = screen.served_probabilities("http://x", "sha", "topic", rows, meter)
    assert probs == [pytest.approx(0.5)] * 2
    req = posted[0]["requests"][0]
    assert req["state"] == {"task": "topic", "article title": "T", "abstract": "A"}
    assert [list(q["criteria"]) for q in req["questions"]] == [
        ["relevant", "not relevant"],
        ["not relevant", "relevant"],
    ]
    assert meter.calls == [("admit", "d1_choice"), ("charge", "d1_choice", 3, 2)]
    assert screen.served_probabilities("http://x", "sha", "t", rows * 2, FakeMeter()) is None
    with pytest.raises(RuntimeError, match="adapter sha"):
        screen.served_probabilities("http://x", "other", "t", rows, FakeMeter())


def test_d1_decides_where_sure_and_deepseek_screens_the_rest(
    gateway,  # noqa: F811
    lotus_fakes,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from jevscholar.systems import controlled

    def probs(url, adapter, topic, rows, meter):  # sure of every 'keep' paper, unsure otherwise
        assert (url, adapter) == ("http://127.0.0.1:9943", "sha")
        return [0.95 if "keep" in r["snippet"] else 0.5 for r in rows]

    monkeypatch.setattr(controlled, "served_probabilities", probs)
    screener = {"adapter": "sha", "delta": 0.7}
    out = tmp_path / "s"
    report, _, _ = run_task(gateway, out, {"kind": "fixed"}, 1.0, max_batches=2, screener=screener)
    filtered = [n for name, n in lotus_fakes if name == "filter"]
    assert sum(filtered) == 10  # only the papers d1 was unsure of (none of them kept)
    assert (report.meta["relevant"], report.meta["d1_screened"]) == (10, 10)
    logged = read_jsonl(out / "screens.jsonl")
    assert len(logged) == 20 and sum(x["d1"] for x in logged) == 10


def test_a_screener_leaves_other_cells_alone_and_its_url_out():
    from jevscholar.systems.controlled import Controlled

    plain = Controlled({"policy": {"kind": "fixed"}}).identity()["params"]
    assert "screener" not in plain
    spec = {"policy": {"kind": "fixed"}, "screener": {"url": "http://h:1", "adapter": "a"}}
    assert Controlled(spec).identity()["params"]["screener"] == {"adapter": "a", "delta": 0.7}


def test_the_screened_cells_are_compared_with_the_fixed_curve():
    from jevscholar.config import load_experiment
    from jevscholar.controller import frontier
    from jevscholar.paths import EXPERIMENTS
    from jevscholar.systems.controlled import Controlled

    exp = load_experiment(EXPERIMENTS / "evaluate.yaml", "dev-screen")
    learned, fixed = frontier.policies({"run_id": "s", "experiment": exp.model_dump(mode="json")})
    assert fixed == ["ctl_fixed", "ctl_fixed_b4", "ctl_fixed_b5", "ctl_fixed_b6"]
    assert learned == [f"ctl_screen{b}" for b in ("", "_b4", "_b5", "_b6", "_b8")]
    params = exp.system_spec("ctl_screen_b8")[1]
    assert params["screener"]["delta"] == 0.7 and params["policy"]["batches"] == 8
    assert Controlled(params).p.max_batches == 8


def test_the_dev2_replication_has_the_same_pairs_on_new_tasks():
    from jevscholar.config import load_experiment
    from jevscholar.paths import EXPERIMENTS

    exp = load_experiment(EXPERIMENTS / "evaluate.yaml", "dev2-screen")
    cell = exp.benchmarks["deepscholar"]
    assert cell.split == "dev2" and cell.limit == 20
    screened = [s for s in cell.systems if "screen" in s and not s.endswith("_s1")]
    assert screened == [f"ctl_screen{b}" for b in ("", "_b4", "_b5", "_b6")]

"""Benchmarks and data: pinned fetch, splits, nugget remap, tasks and official export layouts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from jevscholar import data
from jevscholar.benchmarks.deepscholar import DeepScholarBench, query_template
from jevscholar.benchmarks.drb2 import DRB2
from jevscholar.benchmarks.researchqa import INSTRUCTION, ResearchQA
from jevscholar.types import Report, Source, TaskResult

FETCHED = (data.RAW / "deepscholar/nov2025/gt_nuggets_outputs").is_dir()
PAPER = {
    "arxiv_id": "2509.23254v1",
    "title": "ABConformer: Sliding Attention",
    "abstract": "We present {ABConformer}.",
    "categories": "cs.LG, q-bio.BM",
    "published_date": "2025-09-27T11:12:04+00:00",
}
ATTENTION = Source(
    id="1706.03762v7",
    url="http://arxiv.org/abs/1706.03762v7",
    title="Attention",
    authors=("Ashish Vaswani",),
    date="2017-06-12",
    snippet="Transformers.",
)


def test_order_and_pinned_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = [f"q{i}" for i in range(40)]
    strata = {i: "a" if int(i[1:]) < 30 else "b" for i in ids}
    plain = data.order("x", ids)
    assert plain == data.order("x", reversed(ids)) and sorted(plain) == sorted(ids)
    rr = data.order("x", ids, strata)
    assert Counter(strata[i] for i in rr[:20]) == {"a": 10, "b": 10}
    assert len({strata[i] for i in rr[0:2]}) == len({strata[i] for i in rr[2:4]}) == 2
    prop = data.order("x", ids, strata, proportional=True)
    assert Counter(strata[i] for i in prop[:20]) == {"a": 15, "b": 5}

    sources = tmp_path / "sources.yaml"
    sources.write_text("# head\n\nsrc:\n  url: https://h/\n  dest: d/\n  files:\n    f.json:\n")
    monkeypatch.setattr(data, "SOURCES", sources)
    monkeypatch.setattr(data, "RAW", tmp_path / "raw")
    monkeypatch.setattr(data, "download", lambda url, dest, sha: "c0ffee")
    with pytest.raises(ValueError, match="c0ffee"):  # a blank sha256 is never silently trusted
        data.fetch()
    data.fetch(pin=True)
    assert sources.read_text().startswith("# head\n")
    assert yaml.safe_load(sources.read_text())["src"]["files"] == {"f.json": "c0ffee"}

    page = tmp_path / "idx-1.html"
    page.write_text('<p>See <a href="https://x.org/a(1)">x.org</a>.</p>', encoding="utf-8")
    assert "https://x.org/a(1)" in data._html_text(page)  # the judge checks cited URLs


@pytest.mark.skipif(not FETCHED, reason="needs `jev data fetch`")
def test_splits_and_nugget_remap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    splits = data.build_splits()
    for bench, committed in splits.items():  # re-running prepare rewrites identical JSON
        assert data.read_json(data.SPLITS / f"{bench}.json") == committed
    ds, rqa, drb2 = splits["deepscholar"], splits["researchqa"], splits["drb2"]
    assert [len(ds[k]) for k in ("dev", "val", "train", "dev2", "test")] == [20, 9, 34, 20, 180]
    june = [*ds["dev"], *ds["val"], *ds["train"]]
    assert len(set(june)) == 63 and not set(ds["dev2"]) & set(ds["test"])
    assert (len(rqa["dev"]), len(rqa["test"])) == (703, 776)
    assert sorted(drb2["dev"], key=int) == sorted(data.DRB2_DEV, key=int)
    assert len(drb2["test"]) == 120 and not set(drb2["dev"]) & set(drb2["test"])

    monkeypatch.setattr(data, "DERIVED", tmp_path)
    jun = data.materialise_nuggets("jun2025")
    assert {row: jun[row] for row in (17, 29, 62)} == {17: "14/20", 29: "20/26/29/32", 62: "65"}
    assert data.materialise_nuggets("nov2025") == {row: str(row) for row in range(200)}
    for dataset in ("jun2025", "nov2025"):
        for row, paper in enumerate(data.deepscholar_rows(dataset)):
            path = tmp_path / f"deepscholar/{dataset}/nuggets/{row}/res.json"
            assert data.read_json(path)["qid"] == paper["arxiv_id"]


def _clear_caches() -> None:
    from jevscholar.benchmarks import deepscholar, drb2, researchqa

    for cached in (drb2._rows, researchqa._questions, deepscholar.papers):
        cached.cache_clear()
    deepscholar.important_citations.cache_clear()


@pytest.fixture
def fake_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """One task per benchmark in data/raw + data/splits layout (the readers' caches cleared
    before and after, so no test sees another's data)."""
    _clear_caches()
    monkeypatch.setattr(data, "RAW", tmp_path / "raw")
    monkeypatch.setattr(data, "SPLITS", tmp_path / "splits")
    monkeypatch.setattr(data, "DRB2_TASKS", tmp_path / "raw/drb2/tasks_and_rubrics.jsonl")
    (tmp_path / "splits").mkdir()
    ds = tmp_path / "raw/deepscholar/nov2025"
    ds.mkdir(parents=True)
    with (ds / "papers_with_related_works.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PAPER))
        writer.writeheader()
        writer.writerow(PAPER)
    rqa = {
        "id": "q1-s1",
        "query": "How do landslides start?",
        "date": "2021-11-3",
        "field": "Geology",
        "general_domain": "Life & Earth Sciences",
        "rubric": [
            {"rubric_item": "Does it ...?", "type": ["Other", "Citation"]},
            {"rubric_item": "Does it ...?", "type": ["Example"]},
        ],
    }
    (tmp_path / "raw/researchqa").mkdir(parents=True)
    (tmp_path / "raw/researchqa/valid.json").write_text(json.dumps([rqa]))
    drb2 = {
        "idx": 47,
        "language": "zh",
        "theme": "Finance",
        "prompt": "写一份报告 {不要}",
        "content": {
            "blocked": {
                "title": "A Blocked Article",
                "authors": [],
                "urls": ["https://arxiv.org/pdf/2401.00001v2.pdf"],
            }
        },
    }
    (tmp_path / "raw/drb2").mkdir(parents=True)
    data.DRB2_TASKS.write_text(json.dumps(drb2) + "\n")
    for bench, split, task_id in [
        ("deepscholar", "test", PAPER["arxiv_id"]),
        ("researchqa", "dev", "q1-s1"),
        ("drb2", "dev", "47"),
    ]:
        (tmp_path / f"splits/{bench}.json").write_text(json.dumps({split: [task_id]}))
    yield tmp_path
    _clear_caches()


CASES = {  # benchmark, split, a cited source its policy disallows
    "deepscholar": (DeepScholarBench, "test", "http://arxiv.org/abs/2509.23254v1"),  # the target
    "researchqa": (ResearchQA, "dev", "http://arxiv.org/abs/2112.00001v1"),  # after the cutoff
    "drb2": (DRB2, "dev", "https://www.emergentmind.com/articles/2401.00001"),  # blocked mirror
}


@pytest.mark.parametrize("bench", CASES)
def test_task_policy_and_export(fake_data: Path, bench: str) -> None:
    cls, split, banned = CASES[bench]
    (task,) = cls().tasks(split)
    result = TaskResult(
        task_id=task.id,
        system="s",
        status="ok",
        report=Report(
            "Prior work [1] exists [2].", (ATTENTION, Source(id=banned, url=banned, title="Leak"))
        ),
    )
    out = fake_data / "out"
    assert cls().export_task(task, result, out) == 1  # the disallowed citation is dropped
    assert cls().export_task(task, None, fake_data / "empty") == 0
    if bench == "deepscholar":
        assert task.id == "2509.23254v1" and task.meta["row"] == 0
        assert task.prompt == query_template().format(
            cutoff_date=PAPER["published_date"], abstract=PAPER["abstract"]
        )
        assert task.policy.cutoff == datetime(2025, 9, 27, tzinfo=UTC)
        assert task.policy.blocked_arxiv == {"2509.23254"} and task.policy.undated_web == "drop"
        from eval.parsers import ParserType, get_parser  # on sys.path via query_template()

        dataset = pd.DataFrame([{**PAPER, "arxiv_link": "", "clean_latex_related_works": ""}])
        config = {"mode": ParserType.DEEPSCHOLAR_BASE, "file_id": "0", "dataset": dataset}
        parser = get_parser(config, str(out / "0"))
        assert parser.clean_text == "Prior work [1] exists."
        assert parser.docs == [{"title": "Attention", "sent": "Transformers."}]
        assert (fake_data / "empty/0/intro.md").read_text() == ""
    elif bench == "researchqa":
        assert task.prompt == f"How do landslides start? (Date: 2021-11-3)\n\n{INSTRUCTION}"
        assert task.policy.cutoff == datetime(2021, 11, 3, tzinfo=UTC)
        assert task.meta["rubric_types"] == ["Citation", "Example", "Other"]
        answer = data.read_json(out / "q1-s1.json")["answer"]
        assert answer.startswith("Prior work [1] exists.") and "Leak" not in answer
        assert data.read_json(fake_data / "empty/q1-s1.json") == {"answer": ""}
    else:
        assert task.id == "47" and task.prompt == "写一份报告 {不要}" and task.policy.cutoff is None
        assert task.policy.blocked_urls == {"https://arxiv.org/abs/2401.00001"}
        assert task.policy.blocked_arxiv == {"2401.00001"}
        assert task.policy.blocked_titles == ("a blocked article",)
        report = (out / "idx-47.md").read_text(encoding="utf-8")
        assert report.startswith("Prior work [1] exists.") and "Leak" not in report
        assert (fake_data / "empty/idx-47.md").read_text() == ""


def test_the_final_test_is_frozen():
    """test[:90]: the first 90 Nov test tasks, identified by the hash of their ids."""
    import hashlib

    from jevscholar.benchmarks import split_ids

    ids = split_ids("deepscholar", "test", 90)
    assert (ids[0], ids[-1]) == ("2511.09143v2", "2509.19586v1")
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16] == "66bd8fa271d545da"

"""The reading revisions: the full-text cache (offline: downloads faked) and the writer reading
the documents' excerpts at a writing decision."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from jevscholar.controller.state import Costs, write_legal
from jevscholar.search import fulltext
from jevscholar.systems.controlled import Controlled
from test_controlled import executed, gateway, lotus_fakes, run_task  # noqa: F401 (fixtures)
from test_writing import drafted

PAPER = """Some Title
Alice, Bob
Abstract
We do things.
1 Introduction
Graph agents plan with tools. They are use-
ful in many settings.
2 Related Work
More text here."""


# --------------------------------------------------------------------------- the cache
@pytest.fixture
def cache(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(fulltext, "DIR", tmp_path / "fulltext")
    return tmp_path / "fulltext"


def test_arxiv_ids_keep_their_version():
    assert fulltext.arxiv_id("http://arxiv.org/abs/2504.02344v1") == "2504.02344v1"
    assert fulltext.arxiv_id("https://arxiv.org/pdf/cs/0101001v2.pdf") == "cs/0101001v2"
    assert fulltext.arxiv_id("https://example.org/paper") is None


def test_the_excerpt_starts_at_the_introduction_and_is_cut(cache):
    cache.mkdir()
    (cache / "1.1v1.txt").write_text(PAPER.replace("use-\nful", "useful"))
    (cache / "cs_0101001v2.txt").write_text("no heading " * 10)
    text = fulltext.excerpt("http://arxiv.org/abs/1.1v1")
    assert text.startswith("1 Introduction Graph agents plan") and "Related Work" in text
    assert (
        fulltext.excerpt("http://arxiv.org/abs/1.1v1", chars=30)
        == "1 Introduction Graph agents ..."
    )
    assert fulltext.excerpt("http://arxiv.org/abs/cs/0101001v2").startswith("no heading")
    assert fulltext.excerpt("http://arxiv.org/abs/9.9v1") is None
    assert fulltext.excerpt("https://example.org/x") is None


def test_fetch_keeps_texts_notes_failures_and_skips_what_it_has(cache, monkeypatch):
    calls = []

    def download(aid: str) -> bytes:
        calls.append(aid)
        if aid == "bad":
            raise ValueError("not a PDF (text/html)")
        return aid.encode()

    monkeypatch.setattr(fulltext, "download", download)
    monkeypatch.setattr(fulltext, "extract", lambda pdf: "x " * 400 if pdf == b"ok" else "short")
    urls = [f"http://arxiv.org/abs/{a}" for a in ("ok", "bad", "tiny", "ok")] + ["https://w.org"]
    counts = fulltext.fetch(urls, delay_s=0)
    assert counts == {"fetched": 1, "failed": 2, "not_arxiv": 1}
    assert calls == ["ok", "bad", "tiny"]  # each URL once
    assert (cache / "ok.txt").exists() and "not a PDF" in (cache / "bad.err").read_text()
    assert "only 5 characters" in (cache / "tiny.err").read_text()
    assert fulltext.fetch(urls, delay_s=0) == {"cached": 3, "not_arxiv": 1} and len(calls) == 3


def test_extracted_text_drops_what_utf8_cannot_hold(monkeypatch):
    import pypdf

    class Page:
        def extract_text(self):
            return "x \ud835 y use-\nful"

    monkeypatch.setattr(pypdf, "PdfReader", lambda f: type("R", (), {"pages": [Page()]}))
    assert fulltext.extract(b"%PDF") == "x  y useful"


def test_a_runs_documents_are_its_tasks_writer_documents(tmp_path):
    for task, docs in (("a", ["u1", "u2"]), ("b", ["u2", "u3"])):
        folder = tmp_path / "deepscholar" / "ctl" / "tasks" / task
        folder.mkdir(parents=True)
        (folder / "state.json").write_text(json.dumps({"docs": docs}))
    assert fulltext.run_docs(tmp_path) == ["u1", "u2", "u3"]


# --------------------------------------------------------------------------- the rules
def test_one_reading_revision_per_episode_at_its_own_cost():
    costs, s = Costs(revise_usd=0.01, read_usd=0.03), drafted()
    offer = ("ground", "expand", "read", "redraft")
    cands = write_legal(s, offer, costs)
    assert [(c.kind, c.cost_usd) for c in cands] == [
        ("ground", 0.01),
        ("expand", 0.01),
        ("read", 0.03),
        ("redraft", 0.03),
        ("submit", 0.0),
    ]
    s.revisions = ["redraft"]
    assert [c.kind for c in write_legal(s, offer, costs)] == ["ground", "expand", "submit"]
    s.revisions, s.spent_usd = [], 0.98  # $0.02 left: a revision fits, reading does not
    assert [c.kind for c in write_legal(s, offer, costs)] == ["ground", "expand", "submit"]


def test_reading_leaves_earlier_identities_alone():
    plain = Controlled({"policy": {"kind": "fixed"}}).identity()["params"]
    assert "revise" not in plain
    read = Controlled({"policy": {"kind": "fixed", "revise": ["read"]}, "revise": ["read"]})
    assert read.identity()["params"]["revise"] == ["read"]
    assert read.identity()["policy"]["revise"] == ["read"]


# --------------------------------------------------------------------------- end to end
@pytest.fixture
def writer(lotus_fakes, monkeypatch) -> list[tuple[str, list[str]]]:  # noqa: F811
    """The writer as in lotus_fakes; records each call's instruction and documents' contexts.
    Only the documents ending in .0 have full text."""
    from lotus.sem_ops.sem_agg import SemAggDataframe

    calls: list[tuple[str, list[str]]] = []

    def agg(self, instruction: str, *args, **kw) -> pd.DataFrame:
        calls.append((instruction, list(self._obj["context"])))
        return pd.DataFrame({"summary": ["Graph agents [1] build on planners [2]."]})

    monkeypatch.setattr(SemAggDataframe, "__call__", agg)
    monkeypatch.setattr(
        fulltext, "excerpt", lambda url: "INTRO of " + url if url.endswith(".0") else None
    )
    return calls


def test_read_revises_the_draft_from_the_full_text(gateway, writer, tmp_path):  # noqa: F811
    policy = {"kind": "fixed", "batches": 1, "revise": ["read", "ground"]}
    report, _, _ = run_task(gateway, tmp_path / "r", policy, 1.0, revise=["ground", "read"])
    logged = [json.loads(x) for x in (tmp_path / "r" / "writes.jsonl").read_text().splitlines()]
    assert executed(logged) == ["read", "ground", "submit"]
    assert [c["kind"] for c in logged[1]["candidates"]] == ["ground", "submit"]  # read once
    (draft, plain), (read, full), (ground, kept) = writer
    assert "Full text" not in " ".join(plain) and "full text" not in draft
    assert "Revision for coverage from the full text" in read
    texts = [c for c in full if "Full text (excerpt): INTRO of" in c]
    assert (
        len(texts) == report.meta["read_docs"] == 1
        and texts[0].split("[")[0] == "T graphagentsplan.0"
    )
    assert kept == full and "citation accuracy" in ground  # later revisions keep reading
    assert "Graph agents [1] build on planners [2]." in read  # the draft, as numbers
    assert report.meta["revisions"] == ["read", "ground"]


def test_redraft_writes_anew_from_the_full_text(gateway, writer, tmp_path):  # noqa: F811
    policy = {"kind": "fixed", "batches": 1, "revise": ["redraft"]}
    run_task(gateway, tmp_path / "d", policy, 1.0, revise=["redraft"])
    (draft, _), (redraft, full) = writer
    assert redraft == draft  # DS-base's intro writer again, without the draft
    assert any("Full text (excerpt)" in c for c in full)


def test_reading_without_any_cached_text_fails(gateway, writer, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fulltext, "excerpt", lambda url: None)
    policy = {"kind": "fixed", "batches": 1, "revise": ["read"]}
    with pytest.raises(RuntimeError, match="no full text cached"):
        run_task(gateway, tmp_path / "n", policy, 1.0, revise=["read"])


def test_the_reading_pilot_branches_every_draft_of_its_source_run():
    from jevscholar.config import load_experiment
    from jevscholar.paths import EXPERIMENTS

    for variant, run in (("train", "rollouts-train"), ("val", "rollouts-val")):
        exp = load_experiment(EXPERIMENTS / "reading.yaml", variant)
        cell = exp.benchmarks["deepscholar"]
        assert len(cell.systems) == 8  # four choices, two samples
        takes = {}
        for system in cell.systems:
            params = exp.system_spec(system)[1]
            assert params["start"]["run"] == run and params["start"]["revision"] == 0
            takes[system] = params["start"]["take"]
            Controlled(params)
        assert takes["r_read_s1"] == "read" and takes["r_redraft"] == "redraft"

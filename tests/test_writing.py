"""The controlled loop's writing phase (revisions chosen by the fixed rule) and its version guard:
every cell identity recorded by an earlier run still resolves the same (offline)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from jevscholar import paths
from jevscholar.controller.policy import Fixed
from jevscholar.controller.state import (
    MAX_REVISIONS,
    Costs,
    Paper,
    State,
    numbered,
    write_legal,
)
from jevscholar.systems.controlled import Controlled
from test_controlled import (  # noqa: F401 (fixtures)
    TOPIC,
    executed,
    gateway,
    lotus_fakes,
    row,
    run_task,
)

IDENTITIES = Path(__file__).parent / "fixtures" / "controlled_identities.json"


@pytest.mark.parametrize("cell", sorted(json.loads(IDENTITIES.read_text())))
def test_recorded_cells_keep_their_identity(cell):
    """The resolved identities of the June dev comparison's systems, the train rollouts and a
    counterfactual branch, as their runs' manifests recorded them."""
    recorded = json.loads(IDENTITIES.read_text())[cell]
    params = {k: v for k, v in recorded["params"].items() if k != "use"}
    assert Controlled(params).identity() == recorded["resolved"]


# --------------------------------------------------------------------------- rules and prompt
DOCS = [row(p)["url"] for p in ("1.1", "1.2", "1.3")]


def cite(pid: str) -> str:
    return rf"\[[Alice' 2020-01-02]({row(pid)['url']})\]"


def drafted(**kw) -> State:
    papers = [Paper(row(p), relevant=True) for p in ("1.1", "1.2", "1.3")]
    draft = f"## Related\n\nGraph agents {cite('1.2')}{cite('1.1')} and more {cite('9.9')}."
    return State(TOPIC, 1.0, papers=papers, docs=list(DOCS), draft=draft, **kw)


def test_citations_read_as_numbers_into_the_writers_documents():
    draft = drafted().draft
    assert numbered(draft, DOCS) == "## Related\n\nGraph agents [2][1] and more ."


def test_writing_candidates_follow_the_offer_the_cap_and_the_budget():
    costs = Costs(revise_usd=0.01)
    s = drafted()
    kinds = lambda offered: [c.kind for c in write_legal(s, offered, costs)]  # noqa: E731
    assert kinds(("ground", "expand")) == ["ground", "expand", "submit"]
    assert kinds(("expand",)) == ["expand", "submit"]
    assert kinds(()) == ["submit"]
    s.revisions = ["ground"] * MAX_REVISIONS
    assert kinds(("ground", "expand")) == ["submit"]
    s.revisions, s.spent_usd = [], 0.995  # $0.005 left: no revision fits
    assert kinds(("ground", "expand")) == ["submit"]


def test_the_fixed_rule_revises_in_its_order_then_submits():
    s, rule = drafted(), Fixed(revise=("expand", "ground"))
    cands = write_legal(s, ("ground", "expand"), Costs())
    choose = lambda c: c[asyncio.run(rule.choose(None, s, c)).index].kind  # noqa: E731
    assert choose(cands) == "expand"
    s.revisions = ["expand"]
    assert choose(cands) == "ground"
    s.revisions = ["expand", "ground"]
    assert choose(cands) == "submit"
    s.revisions = []
    assert choose([c for c in cands if c.kind != "expand"]) == "submit"  # not on offer
    assert asyncio.run(Fixed().choose(None, s, cands)).index == 2  # no revisions: submit
    assert "revise" not in Fixed().model_dump() and Fixed(revise=("ground",)).model_dump()["revise"]
    plain = Controlled({"policy": {"kind": "fixed"}}).identity()["params"]
    assert "revise" not in plain
    revised = Controlled({"policy": {"kind": "fixed"}, "revise": ["ground"]}).identity()
    assert revised["params"]["revise"] == ["ground"]
    for kind in ("d1", "prompted"):  # only the fixed rule makes writing decisions
        with pytest.raises(ValueError, match="fixed rule only"):
            Controlled({"policy": {"kind": kind}, "revise": ["ground"]})


# --------------------------------------------------------------------------- end to end
@pytest.fixture
def revising(lotus_fakes, monkeypatch) -> list[str]:  # noqa: F811
    """The writer as in lotus_fakes; a revision's instruction picks its reply. Returns every
    writer instruction."""
    from lotus.sem_ops.sem_agg import SemAggDataframe

    asked: list[str] = []

    def agg(self, instruction: str, *args, **kw) -> pd.DataFrame:
        asked.append(instruction)
        if "citation accuracy" in instruction:
            text = "Graph agents [2]."
        elif "Revision for coverage" in instruction:
            text = "Graph agents [1] build on planners [2] and tools [3]."
        else:
            text = "Graph agents [1] build on planners [2]."
        return pd.DataFrame({"summary": [text]})

    monkeypatch.setattr(SemAggDataframe, "__call__", agg)
    return asked


def writes(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "writes.jsonl").read_text().splitlines()]


FIXED = {"kind": "fixed", "batches": 1, "revise": ["expand", "ground"]}


def test_revisions_follow_the_first_draft(gateway, revising, tmp_path):  # noqa: F811
    report, steps, _ = run_task(gateway, tmp_path / "r", FIXED, 1.0, revise=["ground", "expand"])
    assert executed(steps) == ["abstract", "finish"]  # retrieval as before
    logged = writes(tmp_path / "r")
    assert executed(logged) == ["expand", "ground", "submit"]
    assert [c["kind"] for c in logged[0]["candidates"]] == ["ground", "expand", "submit"]
    assert logged[2]["forced"]  # MAX_REVISIONS made: only submit left
    assert (
        logged[1]["state"]["revisions"] == ["expand"] and logged[1]["state"]["phase"] == "writing"
    )
    _, coverage, accuracy = revising
    assert (
        "Revision for coverage" in coverage and "Graph agents [1] build on planners [2]" in coverage
    )
    assert "citation accuracy" in accuracy and "and tools [3]" in accuracy  # the expanded draft
    state = State.from_json((tmp_path / "r" / "state.json").read_text())
    assert state.revisions == ["expand", "ground"] and state.phase == "done"
    assert len(state.drafts) == 2 and "tools" in state.drafts[1]
    assert report.text == "Graph agents [1]." and report.meta["revisions"] == ["expand", "ground"]

    run_task(gateway, tmp_path / "n", FIXED, 1.0)  # nothing offered: no writing decisions
    assert not (tmp_path / "n" / "writes.jsonl").exists()


def test_a_branch_continues_from_a_writing_point(gateway, revising, tmp_path, monkeypatch):  # noqa: F811
    run_task(gateway, tmp_path / "o", FIXED, 1.0, revise=["ground", "expand"])
    run_task(gateway, tmp_path / "p", FIXED, 1.0)  # as runs before writing decisions: none
    source = tmp_path / "runs" / "src" / "deepscholar" / "ctl" / "tasks"
    for task, origin in (("a", "o"), ("b", "o"), ("old", "p"), ("bad", "p")):
        (source / task).mkdir(parents=True)
        for name in ("rows.jsonl", "state.json", "writes.jsonl"):
            if (tmp_path / origin / name).exists():
                (source / task / name).write_text((tmp_path / origin / name).read_text())
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    start = {"run": "src", "system": "ctl", "revision": 1, "take": "expand"}
    asked = len(revising)
    report, steps, _ = run_task(
        gateway, tmp_path / "a", FIXED, 1.0, revise=["ground", "expand"], start=start
    )
    assert steps == []  # no retrieval decision, no first draft
    logged = writes(tmp_path / "a")
    assert executed(logged) == ["expand", "submit"] and logged[0]["taken"]
    assert logged[0]["state"]["revisions"] == ["expand"]  # the source's point after 1 revision
    assert len(revising) == asked + 1 and report.meta["revisions"] == ["expand", "expand"]

    # a run without writing decisions: its submitted draft is revision 0
    old = {**start, "revision": 0, "take": "ground"}
    report, _, _ = run_task(gateway, tmp_path / "old", FIXED, 1.0, revise=["ground"], start=old)
    # taken, then the rule's second revision (its first, expand, is not offered here)
    assert executed(writes(tmp_path / "old")) == ["ground", "ground", "submit"]
    assert report.meta["revisions"] == ["ground", "ground"] and report.text == "Graph agents [1]."
    with pytest.raises(ValueError, match="only its submitted draft"):
        run_task(gateway, tmp_path / "bad", FIXED, 1.0, revise=["ground"], start=start)
    with pytest.raises(ValueError, match="no writing decision after 5"):
        run_task(
            gateway, tmp_path / "b", FIXED, 1.0, revise=["ground"], start={**start, "revision": 5}
        )
    assert (
        Controlled({"policy": FIXED, "start": old}).identity()["params"]["start"]["revision"] == 0
    )


def test_the_pilot_branches_every_draft_of_its_source_run():
    from jevscholar.config import load_experiment
    from jevscholar.paths import EXPERIMENTS

    for variant, run, split in (
        ("train", "rollouts-train", "train"),
        ("val", "rollouts-val", "val"),
    ):
        exp = load_experiment(EXPERIMENTS / "writing.yaml", variant)
        cell = exp.benchmarks["deepscholar"]
        assert cell.split == split and len(cell.systems) == 8  # four choices, two samples
        takes = {}
        for system in cell.systems:
            params = exp.system_spec(system)[1]
            start = params["start"]
            assert (start["run"], start["system"], start["revision"]) == (run, "ctl_fixed", 0)
            takes[system] = start["take"]
            Controlled(params)  # builds
        assert takes["w_expand_ground_s1"] == "expand" and takes["w_submit"] == "submit"

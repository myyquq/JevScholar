"""The controlled loop: state, hard rules, snapshots and prompts (pure), the d1 server, and the loop
end to end with each policy and from a restored decision point, against the real gateway in front
of a scripted upstream (LOTUS operators faked)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pandas as pd
import pytest
import requests
from pydantic import ValidationError

from jevscholar import paths
from jevscholar.config import Alias, Budget, ModelsConfig, Price, Search, Upstream
from jevscholar.controller.policy import Fixed, make_policy
from jevscholar.controller.prompt import Profile, render
from jevscholar.controller.serve import make_server
from jevscholar.controller.state import (
    Candidate,
    Costs,
    Paper,
    Point,
    Query,
    State,
    legal,
    restore,
    room,
    snapshot,
)
from jevscholar.gateway import Gateway
from jevscholar.systems import fixed_evidence
from jevscholar.systems.controlled import Controlled
from jevscholar.systems.fixed_evidence import FixedEvidence
from jevscholar.types import Task, TaskContext, to_json

TOPIC = "Write the related work of a paper on {graph} agents."


def row(pid: str, keep: bool = True) -> dict[str, Any]:
    url = f"http://arxiv.org/abs/{pid}"
    return {"id": pid, "title": f"T {pid}", "url": url, "snippet": f"A {pid}" + " keep" * keep}


def d1_snapshot_cached() -> bool:
    from jevscholar.controller.d1 import prompt_module

    try:
        prompt_module()
    except Exception:
        return False
    return True


needs_d1_prompt = pytest.mark.skipif(not d1_snapshot_cached(), reason="d1-3B snapshot not cached")


# --------------------------------------------------------------------------- state and rules
def test_state_round_trips_and_rules_keep_the_write_reserve():
    s = State(TOPIC, budget_usd=0.10, spent_usd=0.05)
    s.queries.append(Query("q", pages=1, new=2, relevant=1))
    s.papers += [Paper(row("1.1")), Paper(row("1.2", keep=False), relevant=False)]
    s.papers[0].relevant = True
    assert State.from_json(s.to_json()) == s
    assert [p.url for p in s.relevant()] == ["http://arxiv.org/abs/1.1"]

    costs = Costs(screen_usd=0.001, write_usd=0.01, rank_usd=0.001, step_usd=0.0)
    assert costs.write(30) == 0.01 and costs.write(31) == pytest.approx(0.041)
    a = Candidate("explore", "a", (row("2.1"), row("2.2")), cost_usd=0.002)
    same = Candidate("explore", "b", (row("2.2"), row("2.1")), cost_usd=0.002)  # nothing new
    empty = Candidate("deepen", "q", (), cost_usd=0.0)
    big = Candidate("explore", "c", tuple(row(f"3.{i}") for i in range(40)), cost_usd=0.04)
    finish = Candidate("finish")
    assert legal(s, [a, same, empty, big, finish], costs) == [a, finish]
    s.spent_usd = 0.09  # 0.002 + the write reserve 0.01 no longer fits in 0.01
    assert legal(s, [a, finish], costs) == [finish]
    s.papers[0].relevant, s.spent_usd = False, 0.05  # nothing relevant yet: keep searching
    assert legal(s, [a, finish], costs) == [a]
    s.spent_usd = 0.09
    assert legal(s, [a, finish], costs) == [finish]  # ... unless nothing else is left
    s.papers[0].relevant, s.spent_usd = True, 0.05
    s.batches = s.max_batches  # the batch cap
    assert legal(s, [a, finish], costs) == [finish]
    s.batches = 1

    view, question = render(s, [a, finish], titles=1, preview=1)
    assert view["relevant papers"] == ["T 1.1"] and view["budget"] == "$0.050 of $0.100 left"
    assert (view["phase"], view["search batches"]) == ("retrieval", "1 of 6")
    assert list(question["criteria"]) == ["1. explore", "2. finish"]
    assert question["criteria"]["1. explore"].endswith(
        "screen 2 new papers (about $0.002): T 2.1 (+1 more)"
    )
    assert question["criteria"]["2. finish"] == "finish the search and move on to writing"
    assert render(s, [a, finish], titles=0)[0]["relevant papers"] == []


def test_restored_candidates_follow_the_cells_rules():
    """A decision point continued in another cell: its searches are only those the cell offers,
    and the `finish` the original rules left out comes back."""
    from jevscholar.systems.controlled import Loop

    loop = object.__new__(Loop)
    loop.actions, loop.costs = ("explore",), Costs()
    s = State(TOPIC, 1.0, papers=[Paper(row("1.1"), relevant=False)], batches=1)
    first = [Candidate("abstract", "q", (row("2.1"),)), Candidate("explore", "e", (row("2.2"),))]
    restored = loop.restored(s, first)
    assert [c.kind for c in restored] == ["explore", "finish"]
    s.max_batches = 1  # a lower batch cap: the forced finish, not an empty choice
    assert [c.kind for c in legal(s, restored, Costs())] == ["finish"]
    s.max_batches = 10  # a higher cap than the one that left only finishing: prepare afresh
    assert loop.restored(s, [Candidate("finish")]) is None


def test_each_search_reserves_for_all_it_adds_and_snapshots_restore():
    """Regression: a search overlapping an earlier candidate was reserved for its new part only,
    though executing it adds all of its papers (here crossing the 30-paper ranking threshold)."""
    costs = Costs(screen_usd=0.001, write_usd=0.01, rank_usd=0.001, step_usd=0.0)
    s = State(TOPIC, budget_usd=0.10, spent_usd=0.08)  # $0.02 left
    s.papers = [Paper(row(f"4.{i}"), relevant=True) for i in range(21)]
    nine = Candidate("explore", "n", tuple(row(f"5.{i}") for i in range(9)), 10, 0.009)
    ten = Candidate("explore", "t", (*nine.rows, row("5.9")), 10, 0.010)  # 0.010 + write(31)
    finish = Candidate("finish")
    assert legal(s, [nine, ten, finish], costs) == [nine, finish]
    assert room(s, costs)  # a one-paper search and writing 22 fit in $0.02
    s.spent_usd = 0.091
    assert not room(s, costs)

    point = json.loads(json.dumps(snapshot(s, [nine, finish], fetched=1)))
    rows = {r["url"]: r for r in [*(p.row for p in s.papers), *nine.rows]}
    assert point["candidates"][0]["urls"][:1] == ["http://arxiv.org/abs/5.0"]
    pages = [{"query": "n", "k": 10, "urls": [r["url"] for r in nine.rows]}, {"query": "later"}]
    assert restore(point, rows, pages) == Point(s, [nine, finish], {("n", 10): list(nine.rows)})
    with pytest.raises(ValueError, match="version"):
        restore(point | {"version": 2}, rows, pages)


def test_policy_specs_are_validated():
    assert make_policy({"kind": "fixed", "retrieval": ["deepen"]}) == Fixed(retrieval=("deepen",))
    assert make_policy({"kind": "d1", "profile": {"titles": 10}}).profile == Profile(titles=10)
    for bad in ({"kind": "oracle"}, {"kind": "fixed", "plan": True}, {"kind": "d1", "x": 1}):
        with pytest.raises(ValidationError):
            make_policy(bad)

    def pick(policy: Fixed, state: State, kinds: list[str]) -> str:
        cands = [Candidate(k, k, (row(f"1.{i}"),)) for i, k in enumerate(kinds)]  # type: ignore[arg-type]
        return kinds[asyncio.run(policy.choose(None, state, cands)).index]  # type: ignore[arg-type]

    state, mixed = State(TOPIC, 1.0), Fixed(batches=3)
    assert pick(mixed, state, ["abstract", "explore", "finish"]) == "abstract"  # batch 0
    state.batches = 1
    assert pick(mixed, state, ["abstract", "explore", "finish"]) == "explore"  # rotation
    assert pick(mixed, state, ["abstract", "finish"]) == "abstract"  # explore not on offer
    state.batches = 3
    assert pick(mixed, state, ["abstract", "explore", "finish"]) == "finish"
    assert pick(mixed, state, ["abstract", "explore"]) == "abstract"  # finish not legal yet


def test_profile_requests_and_threshold():
    s = State(TOPIC, 1.0, papers=[Paper(row("1.1"), relevant=True)])
    cands = [Candidate("abstract", "q", (row("2.1"),)), Candidate("explore", "e", (row("2.2"),))]
    cands.append(Candidate("finish"))
    from jevscholar.controller.prompt import INSTRUCTIONS

    assert Profile() == Profile(titles=10, style="sections", threshold=0.5)  # d1's own settings
    request = Profile(titles=0).request(s, cands)
    (question,) = request["questions"]
    assert list(question["criteria"]) == ["1. abstract", "2. explore", "3. finish"]
    assert question["instructions"] == INSTRUCTIONS and "covered" in INSTRUCTIONS
    assert request["state"]["relevant papers"] == []
    long = State(TOPIC, 1.0, papers=[Paper(row(f"1.{i}"), relevant=True) for i in range(30)])
    wide = Profile(titles=20)
    shown = [len(wide.request(long, cands, lv)["state"]["relevant papers"]) for lv in range(4)]
    assert shown == [20, 10, 5, 0]  # shorter views for long states
    few = State(TOPIC, 1.0, papers=[Paper(row(f"1.{i}"), relevant=True) for i in range(7)])
    assert len(Profile(titles=10).request(few, cands)["state"]["relevant papers"]) == 7
    stop = Profile(threshold=0.5)
    assert stop.pick(cands, [0.2, 0.35, 0.45]) == 1  # below the threshold: the best other
    assert stop.pick(cands, [0.1, 0.35, 0.55]) == 2
    assert stop.pick(cands[:2], [0.4, 0.6]) == 1  # no finish on offer: plain argmax
    assert Profile(style="sections").render == {"style": "sections"}
    for bad in ({"threshold": 1.5}, {"titles": -1}, {"style": "x"}):
        with pytest.raises(ValueError, match="profile"):
            Profile(**bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="forced"):
        Profile().request(s, cands[:1])


# --------------------------------------------------------------------------- d1 server
class FakeD1:
    """D1's serving surface: long prompts when titles are listed; the first option wins."""

    def __init__(self) -> None:
        self.forwards = 0

    def encode(self, state: Any, question: dict, render: dict | None = None) -> tuple[list, list]:
        titles = state["relevant papers"] if isinstance(state, dict) else []
        return [0] * (100 + 5000 * len(titles)), []

    def probabilities(self, items: list, render: dict | None = None) -> list[list[float]]:
        self.forwards += 1
        ((_, question),) = items
        n = len(question["criteria"])
        return [[0.9] + [0.1 / (n - 1)] * (n - 1)]


@pytest.fixture
def d1_server() -> Iterator[SimpleNamespace]:
    model = FakeD1()
    server = make_server(model, {"model": "m", "revision": "r", "adapter": None}, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", model=model)
    server.shutdown()


def test_server_caches_and_refuses_long_prompts(d1_server):
    assert requests.get(f"{d1_server.url}/info", timeout=5).json()["adapter"] is None
    q = {"type": "choice", "instructions": "Pick.", "criteria": {"a": "", "b": "", "c": ""}}
    asked = [{"state": {"relevant papers": []}, "questions": [q]}, {"state": "", "questions": [q]}]
    body = {"requests": asked, "render": {"style": "sections"}}
    first = requests.post(f"{d1_server.url}/choice", json=body, timeout=5).json()
    assert first == {"probabilities": [[[0.9, 0.05, 0.05]], [[0.9, 0.05, 0.05]]], "tokens": 200}
    assert requests.post(f"{d1_server.url}/choice", json=body, timeout=5).json() == first
    assert d1_server.model.forwards == 2  # one row a pass; the answer is cached
    long = {"requests": [{"state": {"relevant papers": ["x", "y"]}, "questions": [q]}]}
    r = requests.post(f"{d1_server.url}/choice", json=long, timeout=5)
    assert r.status_code == 413 and r.json()["tokens"] == 10100


# --------------------------------------------------------------------------- end to end
class ScriptedUpstream(BaseHTTPRequestHandler):
    """Chat completions answered by the prompt: "B" for d1's question, else a list of queries (a
    proposal takes the first)."""

    prompts: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        type(self).prompts.append(prompt)
        n = len(type(self).prompts)
        text = "B" if prompt.endswith("option code only.") else f"1. query {n}a\n- query {n}b"
        out = json.dumps(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args: Any) -> None:
        pass


class FakeSearch:
    """arXiv hits per query: two pages of ten, every other paper relevant ('keep')."""

    stats: ClassVar[dict[str, int]] = {"calls": 0, "errors": 0, "blocked": 0, "cached": 0}

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def arxiv_search(self, query: str, k: int) -> list[dict[str, Any]]:
        self.calls.append((query, k))
        tag = "".join(ch for ch in query if ch.isalnum())
        return [
            {
                "id": f"{tag}.{i}",
                "title": f"T {tag}.{i}",
                "abstract": f"A {tag}.{i}" + " keep" * (i % 2 == 0),
                "published": "2020-01-02T03:04:05+00:00",
                "authors": ["Alice A"],
                "categories": ["cs.LG"],
            }
            for i in range(min(k, 20))
        ]

    def arxiv_lookup(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        return {i: self.arxiv_search(i.split(".")[0], 20)[int(i.split(".")[1])] for i in ids}


@pytest.fixture
def lotus_fakes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """sem_filter keeps 'keep' snippets, sem_topk takes the first K, the writer cites [1] [2]."""
    from jevscholar.systems import deepscholar_base as dsb

    dsb.install()
    from lotus.sem_ops.sem_agg import SemAggDataframe
    from lotus.sem_ops.sem_filter import SemFilterDataframe
    from lotus.sem_ops.sem_topk import SemTopKDataframe

    ops: list[tuple[str, int]] = []

    def op(name: str, fn: Any) -> Any:
        def call(self: Any, instruction: str, *args: Any, **kw: Any) -> pd.DataFrame:
            ops.append((name, len(self._obj)))
            return fn(self._obj, kw)

        return call

    keep = op("filter", lambda df, kw: df[df.snippet.str.contains("keep")])
    monkeypatch.setattr(SemFilterDataframe, "__call__", keep)
    monkeypatch.setattr(SemTopKDataframe, "__call__", op("topk", lambda df, kw: df.head(kw["K"])))
    summary = pd.DataFrame({"summary": ["Graph agents [1] build on planners [2]."]})
    monkeypatch.setattr(SemAggDataframe, "__call__", op("write", lambda df, kw: summary))
    return ops


@pytest.fixture
def gateway(tmp_path: Path) -> Iterator[Gateway]:
    ScriptedUpstream.prompts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedUpstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    price = Price(miss=0.30, hit=0.006, out=1.20)
    models = ModelsConfig(
        upstreams={
            "up": Upstream(base_url=f"http://127.0.0.1:{server.server_port}", dialect="llama")
        },
        aliases={"deepseek-flash": Alias(upstream="up", model="flash", price=price)},
        tools={"d1_choice": 0.00002},
        free_tools=("d1_choice",),
    )
    caps = {("deepscholar", "controlled"): 1.0}
    gw = Gateway(models, tmp_path / "run", Budget(), caps, secrets={}, cache_dir=tmp_path)
    gw.start()
    yield gw
    gw.stop()
    server.shutdown()


ABSTRACT = "graph agents plan"
COSTS = {"screen_usd": 0.00001, "write_usd": 0.0004, "rank_usd": 0.0, "step_usd": 0.0004}


def make_ctx(gw: Gateway, out: Path) -> TaskContext:
    """A task named after `out`, so its spend starts at 0."""
    out.mkdir(parents=True, exist_ok=True)
    return TaskContext(
        gateway=gw.url,
        run="r",
        system="controlled",
        task=Task("deepscholar", out.name, TOPIC, meta={"abstract": ABSTRACT}),
        executor="deepseek-flash",
        search=FakeSearch(),  # type: ignore[arg-type]
        search_config=Search(),
        cap_usd=1.0,
        out_dir=out,
        log=logging.getLogger("test"),
    )


def run_task(
    gw: Gateway, out: Path, policy: dict, budget: float, **params: Any
) -> tuple[Any, list, TaskContext]:
    """One task through Controlled.run: its report, decision records and context."""
    ctx = make_ctx(gw, out)
    system = Controlled({"budget_usd": budget, "policy": policy, "costs": COSTS, **params})
    report = asyncio.run(system.run(ctx))
    return report, [json.loads(x) for x in (out / "decisions.jsonl").read_text().splitlines()], ctx


def executed(steps: list[dict]) -> list[str]:
    return [s["candidates"][s["executed"]]["kind"] for s in steps]


def test_fixed_policy_searches_then_writes(gateway, lotus_fakes, tmp_path):
    report, steps, ctx = run_task(gateway, tmp_path / "t1", {"kind": "fixed"}, 1.0, max_batches=2)
    assert executed(steps) == ["abstract", "explore", "finish"]
    assert [c["kind"] for c in steps[1]["candidates"]] == ["abstract", "explore", "finish"]
    assert [c["kind"] for c in steps[2]["candidates"]] == ["finish"]  # the batch cap
    assert steps[2]["forced"] and not steps[1]["forced"]
    assert [s["state"]["batches"] for s in steps] == [0, 1, 2]
    assert all(s["requested"] == s["executed"] and not s["overridden"] for s in steps)
    assert steps[0]["spent_after_usd"] >= steps[0]["state"]["spent_usd"]
    state = State.from_json((tmp_path / "t1" / "state.json").read_text())
    assert [q.text for q in state.queries] == [ABSTRACT, "query 2a"]  # step 1 proposed it
    assert state.queries[0].pages == 1 and (len(state.papers), len(state.relevant())) == (20, 10)
    assert state.phase == "done"
    assert lotus_fakes == [("filter", 10), ("filter", 10), ("write", 10)]  # no sem_topk at 10
    assert report.text == "Graph agents [1] build on planners [2]."
    assert [s.title for s in report.sources] == ["T graphagentsplan.0", "T graphagentsplan.2"]
    assert (report.meta["batches"], report.meta["relevant"]) == (2, 10)
    assert 0 < report.meta["spent_usd"] == pytest.approx(ctx.usage()["spent_usd"])
    assert report.meta["docs"] == state.docs and len(state.docs) == 10


def test_rules_end_the_loop_within_budget(gateway, lotus_fakes, tmp_path):
    budget = 0.0025  # each step's proposal costs about $0.00012
    _, steps, ctx = run_task(gateway, tmp_path / "t2", {"kind": "fixed"}, budget)
    assert [c["kind"] for c in steps[-1]["candidates"]] == ["finish"] and steps[-1]["forced"]
    assert len(steps) > 3 and executed(steps)[-1] == "finish"
    assert ctx.usage()["spent_usd"] <= budget  # the reserve kept writing within the cap


def test_samples_and_ablations(gateway, lotus_fakes, tmp_path, monkeypatch):
    from jevscholar.systems import deepscholar_base as dsb

    built = []
    build = dsb.build_configs
    monkeypatch.setattr(dsb, "build_configs", lambda *a: built.append(a[1:]) or build(*a))
    policy = {"kind": "fixed", "retrieval": ["explore"], "batches": 1}
    _, steps, _ = run_task(gateway, tmp_path / "s", policy, 1.0, sample=2, actions=["explore"])
    assert [c["kind"] for c in steps[0]["candidates"]] == ["explore"]  # no abstract offered
    assert [c["kind"] for c in steps[1]["candidates"]] == ["explore", "finish"]
    assert built == [({"use_sem_topk": False},), ({"use_sem_filter": False}, 2)]  # the writer's
    with pytest.raises(ValidationError):
        Controlled({"policy": {"kind": "fixed"}, "actions": ["finish"]})
    ctx = make_ctx(gateway, tmp_path / "cap")  # no budget of its own: the task's cap
    report = asyncio.run(Controlled({"policy": {"kind": "fixed", "batches": 1}}).run(ctx))
    state = State.from_json((tmp_path / "cap" / "state.json").read_text())
    assert state.budget_usd == ctx.cap_usd == 1.0 and report.meta["batches"] == 1


def test_d1_policy_chooses_and_is_billed(gateway, lotus_fakes, d1_server, tmp_path):
    policy = {"kind": "d1", "url": d1_server.url}
    _, steps, _ = run_task(gateway, tmp_path / "d", policy, 0.003)
    asked = [s for s in steps if not s["forced"]]
    assert asked and all(s["probs"] and s["requested"] == 0 for s in asked)  # the first wins
    assert all(len(s["candidates"]) == len(s["probs"]) for s in asked)
    assert all(
        s["input"]["answers"] and s["input"]["render"] == {"style": "sections"} for s in asked
    )
    events = [json.loads(line) for line in gateway.ledger.path.read_text().splitlines()]
    charged = [e for e in events if e["name"] == "d1_choice" and e["stage"] == "control"]
    assert charged and all(e["n"] == 1 and e["billed_usd"] == 0 for e in charged)
    assert any(e["meta"]["level"] > 0 for e in charged)  # a long state was rendered shorter
    with pytest.raises(RuntimeError, match="adapter"):
        run_task(gateway, tmp_path / "x", policy | {"adapter": "sha-of-another"}, 0.003)


@needs_d1_prompt
def test_prompted_policy_answers_d1s_prompt(gateway, lotus_fakes, tmp_path):
    _, steps, _ = run_task(gateway, tmp_path / "p", {"kind": "prompted"}, 0.004)
    assert steps[0]["requested"] == 1  # the upstream answers "B"
    (asked,) = steps[0]["input"]["asked"]
    assert asked["reply"] == "B" and "Rules of thumb" in asked["text"]


def test_a_decision_point_continues_as_the_original_would(
    gateway, lotus_fakes, tmp_path, monkeypatch
):
    fixed = {"kind": "fixed", "batches": 3}
    _, steps, _ = run_task(gateway, tmp_path / "o", fixed, 1.0)
    run = tmp_path / "runs" / "src" / "deepscholar" / "ctl" / "tasks"
    for task in ("b", "w"):  # the original's logs, under a run another cell continues from
        (run / task).mkdir(parents=True)
        for name in ("decisions.jsonl", "rows.jsonl", "pages.jsonl"):
            (run / task / name).write_text((tmp_path / "o" / name).read_text())
    rows, pages = (
        [json.loads(line) for line in (tmp_path / "o" / name).read_text().splitlines()]
        for name in ("rows.jsonl", "pages.jsonl")
    )
    point = restore(steps[1], {r["url"]: r for r in rows}, pages)
    kinds = [c.kind for c in point.candidates]
    assert (point.state.batches, kinds) == (1, ["abstract", "explore", "finish"])
    assert (ABSTRACT, 150) in point.pages  # the abstract search's ranked list, fetched once

    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    start = {"run": "src", "system": "ctl", "batches": 1}
    asked = len(ScriptedUpstream.prompts)
    _, branch, ctx = run_task(gateway, tmp_path / "b", fixed, 1.0, start=start)
    assert len(ScriptedUpstream.prompts) == asked + 2  # the restored step is not proposed again
    assert (ABSTRACT, 150) not in ctx.search.calls  # re-read for free, as the original did
    assert branch[0]["state"]["offset_usd"] == pytest.approx(steps[1]["state"]["spent_usd"])
    assert branch[0]["candidates"] == steps[1]["candidates"]
    final = [State.from_json((tmp_path / d / "state.json").read_text()) for d in ("o", "b")]
    assert final[0].papers == final[1].papers

    report, steps, _ = run_task(gateway, tmp_path / "w", {**fixed, "batches": 1}, 1.0, start=start)
    assert executed(steps) == ["finish"]  # writing after 1 batch
    assert report.meta["offset_usd"] == pytest.approx(branch[0]["state"]["offset_usd"])
    assert report.meta["batches"] == 1 and report.meta["screened"] == 10
    with pytest.raises(ValueError, match="no decision point"):
        run_task(gateway, tmp_path / "w", fixed, 1.0, start={**start, "batches": 9})


def test_a_branch_takes_its_candidate_then_continues(gateway, lotus_fakes, tmp_path, monkeypatch):
    fixed = {"kind": "fixed", "batches": 3}
    _, steps, _ = run_task(gateway, tmp_path / "o", fixed, 1.0)
    assert executed(steps)[1] == "explore"  # the fixed rule's own choice after one batch
    source = tmp_path / "runs" / "src" / "deepscholar" / "ctl" / "tasks"
    for task in ("f", "x", "n"):
        (source / task).mkdir(parents=True)
        for name in ("decisions.jsonl", "rows.jsonl", "pages.jsonl"):
            (source / task / name).write_text((tmp_path / "o" / name).read_text())
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    start = {"run": "src", "system": "ctl", "batches": 1}

    report, branch, _ = run_task(
        gateway, tmp_path / "f", fixed, 1.0, start=start | {"take": "finish"}
    )
    assert executed(branch) == ["finish"] and branch[0]["taken"] and not branch[0]["forced"]
    assert branch[0]["candidates"] == steps[1]["candidates"]  # what the source offered
    assert report.meta["batches"] == 1

    _, branch, _ = run_task(gateway, tmp_path / "x", fixed, 1.0, start=start | {"take": "abstract"})
    assert executed(branch)[:2] == ["abstract", "explore"]  # then the fixed rule as before
    assert [s["taken"] for s in branch] == [True] + [False] * (len(branch) - 1)
    assert branch[-1]["state"]["batches"] == 3

    with pytest.raises(ValueError, match="'deepen' is not offered"):
        run_task(gateway, tmp_path / "n", fixed, 1.0, start=start | {"take": "deepen"})


def test_a_start_without_take_keeps_its_identity():
    """Runs continued from a decision point before `take` existed keep their cell identity."""
    start = {"run": "src", "system": "ctl", "batches": 1}
    plain = Controlled({"policy": {"kind": "fixed"}, "start": start}).identity()
    assert plain["params"]["start"] == start
    taken = Controlled({"policy": {"kind": "fixed"}, "start": start | {"take": "explore"}})
    assert taken.identity()["params"]["start"] == start | {"take": "explore"}
    with pytest.raises(ValidationError):
        Controlled({"policy": {"kind": "fixed"}, "start": start | {"take": "write"}})


def test_deepen_passes_over_a_query_with_nothing_new():
    """Regression: when the most productive query's next pages held only seen papers, no query
    was deepened at all."""
    from jevscholar.systems.controlled import Loop

    class Pages(Loop):  # deepen's choice alone: pages fetched before, and new rows per page
        def __init__(self, fetched: list, fresh: dict) -> None:
            self.hits, self.fresh = {key: [] for key in fetched}, fresh

        def new_rows(self, state: State, query: str, k: int) -> tuple:
            return self.fresh.get((query, k), ())

    state = State(TOPIC, 1.0, queries=[Query("a", 1, 10, 8), Query("b", 1, 10, 2)])
    exhausted = [("a", 20), ("a", 30)]  # fetched before, nothing new: free to pass over
    b = Pages(exhausted, {("b", 20): (row("x"),)}).deepen(state)
    assert b == Candidate("deepen", "b", (row("x"),), 20)
    assert Pages([], {("a", 30): (row("y"),)}).deepen(state).k == 30  # a's third page
    assert Pages([], {("b", 20): (row("x"),)}).deepen(state) is None  # 2 new fetches, then stop


def test_fixed_evidence_writes_a_run_writers_documents_and_the_gold(
    gateway, lotus_fakes, tmp_path, monkeypatch
):
    """The diagnostic writer: a controlled run's writer input, plus the gold papers."""
    policy = {"kind": "fixed", "retrieval": ["explore"], "batches": 1}
    report, _, ctx = run_task(gateway, tmp_path / "t5", policy, 1.0)
    folder = tmp_path / "runs" / "src" / "deepscholar" / "ctl" / "tasks" / "t5"
    folder.mkdir(parents=True)
    (folder / "state.json").write_text((tmp_path / "t5" / "state.json").read_text())
    recorded = json.loads(to_json(report))
    (folder / "result.json").write_text(json.dumps({"report": recorded}))
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    links = ["http://arxiv.org/abs/query1a.2", "http://arxiv.org/abs/gold.3"]  # one known
    gold = {"t5": [{"cited_paper_arxiv_link": u, "cited_paper_title": "g"} for u in links]}
    monkeypatch.setattr(fixed_evidence, "important_citations", lambda ds: gold)
    monkeypatch.setattr(fixed_evidence, "arxiv_id", lambda url: url.rsplit("/", 1)[-1])

    ctx.task = Task("deepscholar", "t5", TOPIC, meta={"dataset": "jun2025"})
    system = FixedEvidence({"evidence": [{"run": "src", "system": "ctl"}, "gold"]})
    docs = system.documents(ctx)
    written = recorded["meta"]["docs"]
    assert [d["url"] for d in docs] == [*written, "http://arxiv.org/abs/gold.3"]
    assert docs[-1]["context"].startswith("T gold.3[http://arxiv.org/abs/gold.3]: A gold.3")
    lotus_fakes.clear()
    out = asyncio.run(system.run(ctx))
    assert lotus_fakes == [("write", len(docs))] and out.meta["docs"] == [d["url"] for d in docs]
    with pytest.raises(ValueError, match="evidence"):
        FixedEvidence({"evidence": [{"run": "src"}]})


# --------------------------------------------------------------------------- prompted policy
def test_prompted_replies_are_parsed():
    from jevscholar.controller.policy import parse

    assert [parse(r, 3) for r in ("B", "B.", "(c)", "b", "I choose B.", "none", "D")] == [
        1, 1, 2, 1, 1, None, None
    ]  # fmt: skip


@needs_d1_prompt
def test_prompted_reads_what_d1_reads_under_the_profile(monkeypatch):
    """The prompted policy asks exactly d1's user turn, with the rules of thumb before its last
    line; the reply's code maps back to the candidate."""
    from jevscholar.controller import policy
    from jevscholar.controller.d1 import user_turn

    seen, replies = [], iter(["b", "A"])

    async def chat(ctx: Any, stage: str, text: str, max_tokens: int) -> str:
        seen.append(text)
        return next(replies)

    monkeypatch.setattr(policy, "chat", chat)
    ctx = SimpleNamespace(log=logging.getLogger("test"))
    s = State(TOPIC, 1.0, papers=[Paper(row("1.1"), relevant=True)])
    cands = [Candidate("abstract", "q", (row("2.1"),)), Candidate("explore", "e", (row("2.2"),))]
    cands.append(Candidate("finish"))
    profile = Profile(style="sections")
    assert asyncio.run(policy.Prompted(profile=profile).choose(ctx, s, cands)).index == 1
    request = profile.request(s, cands)
    plain = user_turn(request["state"], request["questions"][0], profile.render)
    head, last = plain.rsplit("\n\n", 1)
    assert seen[0] == f"{head}\n\n{policy.RULES}\n\n{last}"
    assert asyncio.run(policy.Prompted(rules=False).choose(ctx, s, cands)).index == 0
    assert policy.RULES not in seen[1]


@needs_d1_prompt
def test_the_prompted_d1_text_is_d1s_user_turn():
    """The prompt is the checkpoint's own rendering (the GPU test checks it against the
    tokenizer-built prompt)."""
    from jevscholar.controller.d1 import user_turn

    view = {"task": "T", "relevant papers": ["a"]}
    q = {
        "type": "choice",
        "instructions": "Pick.",
        "criteria": {"1. abstract": "A", "2. finish": "F"},
    }
    text = user_turn(view, q)
    assert text.startswith('{\n  "task": "T"') and "QUESTION:\nPick.\n\nOptions:\nA A\nB F" in text
    assert text.endswith("Reply with the option code only.")
    assert user_turn(view, q, {"option_style": "name_desc"}).count("1. abstract: A") == 1

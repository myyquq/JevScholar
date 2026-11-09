"""Bounded RL: the group runs, their rewards and advantages, the RL examples and the update's
objective (offline: runs written by hand, gold faked, a tiny CPU stand-in for d1)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from jevscholar import paths
from jevscholar.controller import groups
from jevscholar.controller.branches import Fork
from jevscholar.controller.d1 import (
    RLConfig,
    RLExample,
    categorical_kl,
    read_rl_examples,
    rl_terms,
    rl_update,
    write_rl_examples,
)
from jevscholar.controller.policy import D1Policy
from jevscholar.controller.state import Paper, State
from jevscholar.types import Task, make_tag
from test_branches import experiment, row, source, write_lines  # noqa: F401 (source: a fixture)

SHA = "a" * 64
Q = {"type": "choice", "instructions": "Pick.", "criteria": {"1. abstract": "x", "2. finish": None}}


# --------------------------------------------------------------------------- group runs
def test_points_pick_one_train_point_per_task(source):  # noqa: F811
    found = groups.points(source, "ctl_fixed", 4)
    assert len(found) == 2 and {f.task for f in found} == {"t1", "t2"}  # one per task
    assert found == groups.points(source, "ctl_fixed", 4)  # a fixed pick
    assert len(groups.points(source, "ctl_fixed", 1)) == 1
    assert found == sorted(found, key=lambda f: (f.batches, f.key))
    manifest = source / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"train"', '"val"'))
    with pytest.raises(ValueError, match="train runs only, not val"):
        groups.points(source, "ctl_fixed", 4)


def test_one_run_per_batches_value_with_identical_sampling_members(source):  # noqa: F811
    found = [Fork("t1", "t1", 0, ("abstract", "explore")), Fork("t2", "t2", 1, ("a", "finish"))]
    found.append(Fork("t2", "t2", 0, ("abstract", "explore")))
    exps = groups.experiments(source, "ctl_fixed", found, adapter=SHA, run_usd=0.1, prefix="g-r1")
    assert list(exps) == ["g-r1-b0", "g-r1-b1"]
    cell = exps["g-r1-b0"].benchmarks["deepscholar"]
    assert (cell.split, cell.ids, cell.systems) == ("train", ("t1", "t2"), ("g0", "g1", "g2", "g3"))
    assert exps["g-r1-b1"].benchmarks["deepscholar"].ids == ("t2",)
    specs = exps["g-r1-b0"].systems
    assert all(specs[f"g{i}"] == specs["g0"] for i in range(4))
    spec = specs["g0"]
    assert spec["policy"] == {"kind": "d1", "sample": True, "adapter": SHA}
    assert spec["start"] == {"run": "src", "system": "ctl_fixed", "batches": 0}  # no `take`
    assert spec["task_usd"] == 0.5  # the source's own cap kept
    assert (exps["g-r1-b0"].budget.run_usd, exps["g-r1-b0"].seed) == (0.1, 0)
    two = groups.experiments(source, "ctl_fixed", found[:1], adapter=SHA, run_usd=0.1, size=2)
    assert two["groups-src-b0"].benchmarks["deepscholar"].systems == ("g0", "g1")


def test_members_sample_differently():
    """Sampling is seeded by ctx.tag('control'), which holds the system: g0..g3 draw apart."""

    @dataclass
    class Ctx:
        system: str

        @property
        def task(self) -> Task:
            return Task("deepscholar", "t1", "T")

        def tag(self, stage: str) -> str:
            return make_tag("run", "deepscholar", self.system, "t1", stage)

    policy, dist = D1Policy(sample=True, adapter=SHA), [0.25] * 4
    draws = {}
    for name in ("g0", "g1", "g2", "g3"):
        seq = []
        for decisions in range(6):
            state = State("T", 0.27, decisions=decisions)
            seq.append(policy._sample(Ctx(name), state, dist))
        draws[name] = tuple(seq)
        assert seq == [
            policy._sample(Ctx(name), State("T", 0.27, decisions=d), dist) for d in range(6)
        ]
    assert len(set(draws.values())) == 4


# --------------------------------------------------------------------------- rewards
def decision(batches: int, requested: int, **flags) -> dict:
    """A logged d1 decision between abstract and finish."""
    request = {"state": {"task": "T", "batches": batches}, "questions": [Q]}
    record = {"version": 3, "state": {"batches": batches, "decisions": batches}, "candidates": []}
    return record | {
        "requested": requested,
        "executed": requested,
        "probs": [0.7, 0.3],
        "input": {"requests": [request], "render": {"style": "sections"}, "adapter": SHA},
        "forced": False,
        "overridden": False,
        "taken": False,
    } | flags  # fmt: skip


def group_run(name: str, members: dict[str, dict[str, list[str] | None]], batches: int = 1) -> Path:
    """A group run under runs/: per task, each member's writer documents by title (None: the
    episode failed), with three logged decisions each (one taken, one forced)."""
    run = paths.RUNS / name
    names = [f"g{i}" for i in range(len(next(iter(members.values()))))]
    spec = {
        "use": "controlled",
        "policy": groups.policy(SHA),
        "start": {"run": "src", "system": "ctl_fixed", "batches": batches},
    }
    for key, docs in members.items():
        for name_, titles in zip(names, docs.values(), strict=True):
            folder = run / "deepscholar" / name_ / "tasks" / key
            folder.mkdir(parents=True)
            if titles is None:
                (folder / "result.json").write_text(json.dumps({"status": "failed"}))
                continue
            state = State("T", 0.27, spent_usd=0.05, papers=[Paper(row(t), True) for t in titles])
            state.docs = [row(t)["url"] for t in titles]
            (folder / "state.json").write_text(state.to_json())
            (folder / "result.json").write_text(json.dumps({"status": "ok"}))
            records = [
                decision(batches, 0),
                decision(batches + 1, 1, overridden=True),
                decision(batches + 2, 1),
                decision(batches + 3, 1, forced=True),
            ]
            write_lines(folder / "decisions.jsonl", records)
    exp = experiment("train", dict.fromkeys(names, spec), names)
    exp["benchmarks"]["deepscholar"]["ids"] = list(members)
    (run / "manifest.json").write_text(json.dumps({"experiment": exp}))
    return run


@pytest.fixture
def runs(source, monkeypatch):  # noqa: F811
    """Group runs: t1 with distinct returns, t2 tied, t3 with a failed member."""
    tasks = {k: Task("deepscholar", k, "T") for k in ("t1", "t2", "t3")}

    def run_tasks(manifest, bench):
        return {k: tasks[k] for k in manifest["experiment"]["benchmarks"][bench]["ids"]}

    monkeypatch.setattr(groups, "run_tasks", run_tasks)
    one = {"a": ["gold a", "gold b"], "b": ["gold a"], "c": ["gold a"], "d": []}
    tied = {m: ["gold a"] for m in "abcd"}
    failed = {"a": ["gold a"], "b": None, "c": ["gold a"], "d": ["gold b"]}
    return [group_run("grp-b1", {"t1": one, "t2": tied}), group_run("grp-b2", {"t3": failed}, 2)]


def test_returns_advantages_and_skipped_groups(runs):
    found = groups.outcomes(runs)
    assert len(found) == 12
    t1 = [o for o in found if o["task"] == "t1"]
    assert [o["recall"] for o in t1] == [1.0, 0.5, 0.5, 0.0]
    assert t1[0]["return"] == pytest.approx(0.15 - 0.1 * 0.05 / 0.27)
    assert {o["status"] for o in found if o["task"] == "t3"} == {"ok", "failed"}

    usable, skipped = groups.advantages(found)
    assert skipped == {"incomplete": 1, "tied": 1}
    (g,) = usable
    assert (g["task"], g["batches"], g["members"]) == ("t1", 1, ["g0", "g1", "g2", "g3"])
    sd = math.sqrt(sum((x - 0.075) ** 2 for x in (0.15, 0.075, 0.075, 0.0)) / 4)
    assert g["advantages"] == pytest.approx([0.075 / sd, 0, 0, -0.075 / sd])
    assert sum(g["advantages"]) == pytest.approx(0)


def test_advantage_floor():
    assert groups.advantage([0.001, 0.0]) == pytest.approx([0.0005 / 0.02, -0.0005 / 0.02])


def test_examples_are_what_d1_saw(runs, tmp_path):
    usable, _ = groups.advantages(groups.outcomes(runs))
    found = groups.examples(usable)
    assert len(found) == 8  # 2 of 4 decisions per member: the overridden and forced stay out
    e = found[0]
    assert (e.chosen, e.old_prob, e.question, e.render) == (0, 0.7, Q, {"style": "sections"})
    assert e.state == {"task": "T", "batches": 1}
    assert e.advantage == usable[0]["advantages"][0]
    assert e.meta == {"run": "grp-b1", "task": "t1", "batches": 1, "member": "g0", "decision": 1}
    assert found[1].chosen == 1 and found[1].old_prob == 0.3
    write_rl_examples(tmp_path / "rl.jsonl", found)
    assert read_rl_examples(tmp_path / "rl.jsonl") == found

    assert groups.decision_example(decision(0, 0, taken=True), 1.0, SHA, {}) is None
    with pytest.raises(ValueError, match="sampled by adapter"):
        groups.decision_example(decision(0, 0), 1.0, "b" * 64, {})
    with pytest.raises(ValueError, match="not among"):
        RLExample("s", Q, 2, 0.5, 1.0)
    with pytest.raises(ValueError, match="probability"):
        RLExample("s", Q, 0, 0.0, 1.0)


def test_only_group_runs_of_train_give_rl_data(runs):
    manifest = runs[0] / "manifest.json"
    data = json.loads(manifest.read_text())
    systems = data["experiment"]["systems"]
    for spec in systems.values():
        spec["start"]["take"] = "finish"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="do not sample"):
        groups.outcomes(runs[:1])
    systems["g1"] = systems["g1"] | {"start": {"run": "src", "system": "ctl_fixed"}}
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="members differ"):
        groups.outcomes(runs[:1])
    for spec in systems.values():
        spec["start"] = {"run": "src", "system": "ctl_fixed", "batches": 1}
    data["experiment"]["benchmarks"]["deepscholar"]["split"] = "val"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="train only"):
        groups.outcomes(runs[:1])


# --------------------------------------------------------------------------- the update
def test_rl_terms():
    lp = torch.tensor([0.6, 0.4]).log()
    pg, kl, ratio = rl_terms(lp, lp, 0, 0.6, 2.0, 0.2)
    assert (pg.item(), kl.item(), ratio.item()) == pytest.approx((-2.0, 0.0, 1.0), abs=1e-6)
    pg, _, ratio = rl_terms(lp, lp, 0, 0.3, 1.0, 0.2)  # r = 2: the gain is clipped at 1.2
    assert (ratio.item(), pg.item()) == pytest.approx((2.0, -1.2))
    pg, _, _ = rl_terms(lp, lp, 0, 0.3, -1.0, 0.2)  # a negative advantage is not clipped there
    assert pg.item() == pytest.approx(2.0)
    ref = torch.tensor([0.5, 0.5]).log()
    _, kl, _ = rl_terms(lp, ref, 1, 0.4, 1.0, 0.2)
    assert kl.item() == pytest.approx(categorical_kl([0.6, 0.4], [0.5, 0.5]), rel=1e-5)
    assert categorical_kl([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2))


class Tiny:
    """A stand-in for D1: option logits from a learnable table indexed by the prompt."""

    def __init__(self, logits: torch.Tensor, trainable: bool) -> None:
        self.model = torch.nn.Module()
        self.model.table = torch.nn.Parameter(logits.clone(), requires_grad=trainable)
        self.model.save_pretrained = lambda path: Path(path).mkdir()
        self.base = object()

    def encode(self, state, question, render=None):
        return [int(state)], [[0], [1]]

    def option_logprobs(self, rows, forms):
        return [self.model.table[r[0]].log_softmax(-1) for r in rows]


def write_data(path: Path) -> None:
    write_rl_examples(
        path,
        [
            RLExample(0, Q, 0, 0.5, 1.0),  # a good choice
            RLExample(1, Q, 1, 0.5, 0.5),  # a less good one
        ],
    )


def test_rl_update_moves_toward_good_choices_and_saves(tmp_path):
    write_data(tmp_path / "rl.jsonl")
    start = torch.zeros(2, 2)  # p = 0.5 everywhere: the old probabilities

    def load(adapter, revision, lora=None):
        return Tiny(start, lora is not None)

    out = tmp_path / "r1"
    cfg = RLConfig(data=tmp_path / "rl.jsonl", adapter=Path("a"), reference=Path("a"), out=out)
    m = rl_update(cfg.model_copy(update={"lr": 0.05}), log=lambda *_: None, load=load)
    assert m["decisions"] == 2 and m["readout_parity"] == pytest.approx(0)
    assert m["train"]["ratio"] == pytest.approx(1.0)  # one step: the first forward is the old one
    assert m["ratio"] > 1  # both chosen options had positive advantages: more likely now
    assert 0 < m["kl"] < 0.05 and not m["reverted"]
    assert (out / "adapter").is_dir() and json.loads((out / "metrics.json").read_text()) == m


def test_rl_update_reverts_above_the_kl_bound(tmp_path):
    write_data(tmp_path / "rl.jsonl")
    out = tmp_path / "r1"
    (out / "adapter").mkdir(parents=True)  # an earlier try's

    def load(adapter, revision, lora=None):
        return Tiny(torch.zeros(2, 2), lora is not None)

    cfg = RLConfig(
        data=tmp_path / "rl.jsonl", adapter=Path("a"), reference=Path("a"), out=out, lr=5.0
    )
    m = rl_update(cfg, log=lambda *_: None, load=load)
    assert m["reverted"] and m["kl"] > 0.05
    assert not (out / "adapter").exists() and (out / "metrics.json").exists()

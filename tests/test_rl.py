"""Bounded RL data: the registered late points, the rollout experiments and the groups with their
advantages (offline: runs written by hand, gold faked)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from jevscholar import paths
from jevscholar.config import Experiment
from jevscholar.controller import branches, rl
from jevscholar.controller.state import Candidate, Paper, State, snapshot
from jevscholar.types import Task

GOLD = {"gold a", "gold b"}


def row(title: str) -> dict:
    return {"url": f"http://x.org/{title.replace(' ', '_')}", "title": title, "snippet": "s"}


def write_lines(path: Path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def manifest(split: str, systems: dict, listed: list[str], ids=None) -> dict:
    exp = Experiment(
        name="controller/rollouts_d1",
        systems=systems,
        benchmarks={"deepscholar": {"split": split, "systems": listed, "ids": ids}},
    )
    return {"experiment": exp.model_dump(mode="json")}


@pytest.fixture
def source(tmp_path, monkeypatch):
    """A d1 episode run under runs/src: per task, decisions after 0, 3 and 4 batches; the late
    ones offer two searches and finish on t1..t9, one search and finish on `thin`."""
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    run = tmp_path / "runs" / "src"
    keys = [f"t{i}" for i in range(1, 10)] + ["thin"]
    systems = {"ctl_d1": {"use": "controlled", "policy": {"kind": "d1"}}}
    tasks = {}
    for key in keys:
        folder = run / "deepscholar" / "ctl_d1" / "tasks" / key
        folder.mkdir(parents=True)
        a, b = row(f"{key} a"), row(f"{key} b")
        write_lines(folder / "rows.jsonl", [a, b])
        write_lines(folder / "pages.jsonl", [])
        searches = [Candidate("abstract", "abs", (a,)), Candidate("explore", "q", (b,))]
        late = searches if key != "thin" else searches[:1]
        records = []
        for batches, offered in ((0, searches), (3, [*late, Candidate("finish")])):
            state = State("T", 0.27, "abs", batches=batches, papers=[Paper(a, True)])
            records.append(snapshot(state, offered, 0))
        state = State("T", 0.27, "abs", batches=4, papers=[Paper(a, True)])
        records.append(snapshot(state, [*late, Candidate("finish")], 0))
        lines = [
            r | {"requested": 0, "executed": 0, "forced": False, "overridden": False}
            | {"taken": False, "control_usd": 0.0, "probs": None}
            for r in records
        ]  # fmt: skip
        write_lines(folder / "decisions.jsonl", lines)
        (folder / "result.json").write_text(json.dumps({"status": "ok"}))
        tasks[key] = Task("deepscholar", key, "T")
    (run / "manifest.json").write_text(json.dumps(manifest("train", systems, ["ctl_d1"])))

    def run_tasks(m, bench):
        ids = m["experiment"]["benchmarks"][bench].get("ids")
        return {k: t for k, t in tasks.items() if ids is None or k in ids}

    def gold_found(task, ids, titles):
        return len(titles & GOLD), 2

    for module in (branches, rl):
        monkeypatch.setattr(module, "run_tasks", run_tasks)
    monkeypatch.setattr(branches, "gold_found", gold_found)
    return run


def test_points_are_late_choices_among_searches(source):
    found = rl.points(source, "ctl_d1", n=8)
    assert len(found) == 8 and len({f.key for f in found}) == 8
    assert all(3 <= f.batches <= 5 and f.kinds.count("finish") == 1 for f in found)
    assert "thin" not in {f.key for f in found}  # one search only
    assert found == rl.points(source, "ctl_d1", n=8)  # a fixed pick
    with pytest.raises(ValueError, match="only 9 tasks"):
        rl.points(source, "ctl_d1", n=10)


def test_rollout_experiments_sample_from_restored_points(source):
    found = rl.points(source, "ctl_d1", n=8)
    policy = {"kind": "d1", "sample": True, "adapter": "sha", "url": "http://x"}
    exps = rl.experiments(
        source, "ctl_d1", found, policy=policy, samples=4, run_usd=0.2, prefix="rl1"
    )
    assert set(exps) == {f"rl1-b{b}" for b in {f.batches for f in found}}
    exp = next(iter(exps.values()))
    cell = exp.benchmarks["deepscholar"]
    assert cell.systems == ("rollout", "rollout_s1", "rollout_s2", "rollout_s3")
    specs = [exp.systems[name] for name in cell.systems]
    assert [s["sample"] for s in specs] == [0, 1, 2, 3]  # separate members, separate seeds
    assert all(s["write"] is False and "take" not in s["start"] for s in specs)
    assert specs[0]["policy"] == policy and specs[0]["start"]["run"] == "src"
    with pytest.raises(ValueError, match="sample from a d1 policy"):
        rl.experiments(
            source, "ctl_d1", found, policy={"kind": "d1"}, samples=4, run_usd=1, prefix="x"
        )


def rollout_run(name: str, docs: dict[str, list[list[str]]], failed=(), batches=3) -> Path:
    """A finished rollout run at the source's points after `batches` batches: per task, each
    member's writer documents (by title), its restored point as one sampled decision, and a
    forced one."""
    run = paths.RUNS / name
    names = rl.members(len(next(iter(docs.values()))))
    member = {"use": "controlled", "policy": {"kind": "d1", "sample": True}, "write": False}
    member["start"] = {"run": "src", "system": "ctl_d1", "batches": batches}
    systems = {n: member | {"sample": i} for i, n in enumerate(names)}
    criteria = {"1. abstract": "", "2. explore": "", "3. finish": ""}
    question = {"type": "choice", "instructions": "Pick.", "criteria": criteria}
    for key, per_member in docs.items():
        log = paths.RUNS / "src" / "deepscholar" / "ctl_d1" / "tasks" / key / "decisions.jsonl"
        records = [json.loads(line) for line in log.read_text().splitlines()]
        point = next(r for r in records if r["state"]["batches"] == batches)
        for n, titles in zip(names, per_member, strict=True):
            folder = run / "deepscholar" / n / "tasks" / key
            folder.mkdir(parents=True)
            status = "failed" if (key, n) in failed else "ok"
            (folder / "result.json").write_text(json.dumps({"status": status}))
            state = State("T", 0.27, papers=[Paper(row(t), True) for t in titles])
            state.docs = [row(t)["url"] for t in titles]
            (folder / "state.json").write_text(state.to_json())
            sampled = point | {  # the restored point, as the source offered it
                "requested": 2,
                "executed": 2,
                "probs": [0.3, 0.3, 0.4],
                "input": {
                    "requests": [{"state": {"s": key}, "questions": [question]}],
                    "render": {"style": "sections"},
                    "adapter": "sha",
                },
            }
            forced = sampled | {"forced": True, "input": None}
            write_lines(folder / "decisions.jsonl", [sampled, forced])
    (run / "manifest.json").write_text(json.dumps(manifest("train", systems, names, list(docs))))
    return run


def plan(keys, batches=3) -> dict:
    return {"system": "ctl_d1", "planned": [(k, batches) for k in keys], "samples": 4}


def test_groups_give_each_sampled_decision_its_trajectory_advantage(source, monkeypatch):
    run = rollout_run(
        "rl1-b3",
        {
            "t1": [["t1 a"], ["t1 a", "t1 b"], ["t1 a"], ["t1 b"]],
            "t2": [["t2 a"]] * 4,  # tied
            "t3": [["t3 a"], ["t3 b"], ["t3 a"], ["t3 a"]],  # a member failed
        },
        failed={("t3", "rollout_s2")},
    )
    gold = {"t1 a", "t1 b", "t2 a", "t3 a", "t3 b"}
    monkeypatch.setattr(branches, "gold_found", lambda task, ids, titles: (len(titles & gold), 2))
    steps, groups = rl.groups([run], source=source, **plan(["t1", "t2", "t3", "t4"]))
    by_task = {g["task"]: g for g in groups}
    assert by_task["t2"]["skipped"] == "tied"
    assert by_task["t3"]["skipped"] == "a member failed or has no gold"
    assert by_task["t4"]["skipped"] == "no rollout run holds it"  # planned, never sampled
    t1 = by_task["t1"]
    assert t1["skipped"] is None
    returns = t1["returns"]
    mean, sd = statistics.fmean(returns), statistics.pstdev(returns)
    assert t1["advantages"] == pytest.approx([(r - mean) / max(sd, rl.SD_FLOOR) for r in returns])
    usable = [s for s in steps if s.usable]
    assert [s.trajectory for s in usable] == rl.members(4)  # the forced decision is left out
    assert usable[1].advantage == pytest.approx(t1["advantages"][1])
    assert usable[1].advantage > 0  # it found both
    assert (usable[0].action, usable[0].old, usable[0].adapter) == (2, [0.3, 0.3, 0.4], "sha")
    assert len(steps) == 4 + 4 + 3  # tied and failed groups stay in the KL guard's data
    assert all(s.advantage == 0 for s in steps if not s.usable)


def test_rollouts_must_match_the_frozen_recipe(source):
    run = rollout_run("rl1-b3", {"t1": [["t1 a"]] * 4})
    with pytest.raises(ValueError, match="not planned"):
        rl.groups([run], source=source, **plan(["t2"]))
    with pytest.raises(ValueError, match="not 2 rollout members"):
        rl.groups([run], source=source, **plan(["t1"]) | {"samples": 2})
    data = json.loads((run / "manifest.json").read_text())
    data["experiment"]["systems"]["rollout_s1"]["start"]["take"] = "finish"
    (run / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="one point, without `take`"):
        rl.groups([run], source=source, **plan(["t1"]))
    del data["experiment"]["systems"]["rollout_s1"]["start"]["take"]
    data["experiment"]["systems"]["rollout_s1"]["policy"]["sample"] = "searches"
    (run / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="sample and stop before writing"):
        rl.groups([run], source=source, **plan(["t1"]))
    other = rollout_run("rl1-b4", {"t1": [["t1 a"]] * 4}, batches=4)
    log = other / "deepscholar" / "rollout" / "tasks" / "t1" / "decisions.jsonl"
    first, *rest = [json.loads(line) for line in log.read_text().splitlines()]
    first["state"]["batches"] = 3  # restored another point
    write_lines(log, [first, *rest])
    with pytest.raises(ValueError, match="restored another snapshot"):
        rl.groups([other], source=source, **plan(["t1"], batches=4))


def test_steps_round_trip(tmp_path):
    step = rl.Step({"s": 1}, {"criteria": {"a": ""}}, {}, 0, [1.0], 0.5, None, "g", "rollout")
    rl.write_steps(tmp_path / "s.jsonl", [step])
    assert rl.read_steps(tmp_path / "s.jsonl") == [step]


def test_a_rejected_update_leaves_the_selected_adapter(tmp_path):
    from jevscholar.controller.d1 import RLConfig, rl_update

    sft, out = tmp_path / "sft", tmp_path / "rl"
    sft.mkdir()
    (sft / "adapter_model.safetensors").write_bytes(b"sft")
    (out / "adapter").mkdir(parents=True)
    (out / "adapter" / "adapter_model.safetensors").write_bytes(b"kept")
    step = rl.Step({"s": 1}, {"criteria": {"a": ""}}, {}, 0, [1.0], 0.5, "other", "g", "rollout")
    rl.write_steps(tmp_path / "steps.jsonl", [step])
    cfg = RLConfig(
        kind="rl", source="src", system="ctl_d1", points=[], prefix="rl1", out=out,
        steps=tmp_path / "steps.jsonl", init_adapter=sft, reference=sft,
    )  # fmt: skip
    with pytest.raises(ValueError, match="not the init adapter"):
        rl_update(cfg, log=lambda *_: None)
    assert (out / "adapter" / "adapter_model.safetensors").read_bytes() == b"kept"


def test_the_val_gate_picks_the_first_arm_that_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    run = tmp_path / "runs" / "gate"
    tasks = {k: Task("deepscholar", k, "T") for k in ("v1", "v2")}
    # per system: (documents by title, batches) on v1 and v2
    episodes = {
        "ctl_d1": ([["g1"], ["g2"]], [6, 6]),
        "ctl_d1_sft": ([["g1"], ["g2"]], [4, 3]),
        "ctl_d1_rl": ([["g1"], []], [1, 6]),  # stops after one batch and loses v2's gold
    }
    for system, (docs, batches) in episodes.items():
        for sample in (system, f"{system}_s1"):
            for (key, titles), b in zip(zip(tasks, docs, strict=True), batches, strict=True):
                folder = run / "deepscholar" / sample / "tasks" / key
                folder.mkdir(parents=True)
                state = State("T", 0.27, papers=[Paper(row(t), True) for t in titles], batches=b)
                state.docs = [row(t)["url"] for t in titles]
                (folder / "state.json").write_text(state.to_json())
                usage = {"spent_usd": 0.05, "task_cap": 0.27}
                (folder / "result.json").write_text(json.dumps({"status": "ok", "usage": usage}))
    (run / "manifest.json").write_text(json.dumps(manifest("val", {}, ["ctl_d1"])))
    monkeypatch.setattr(rl, "run_tasks", lambda m, bench: tasks)
    monkeypatch.setattr(branches, "gold_found", lambda task, ids, titles: (len(titles), 1))
    out = rl.gate(run, {"ctl_d1": 0.004, "ctl_d1_sft": 0.003, "ctl_d1_rl": 0.001})
    rl_checks = out["arms"]["ctl_d1_rl"]["checks"]
    assert not rl_checks["recall"] and not rl_checks["batches"] and rl_checks["regret"]
    assert out["arms"]["ctl_d1_sft"]["passes"] and out["claim"] == "ctl_d1_sft"
    out = rl.gate(run, {"ctl_d1": 0.002, "ctl_d1_sft": 0.003, "ctl_d1_rl": 0.001})
    assert out["claim"] == "ctl_d1"  # d1-sft's regret is above d1's: d1 carries the claim
    over = run / "deepscholar" / "ctl_d1_sft_s1" / "tasks" / "v2" / "result.json"
    over.write_text(json.dumps({"status": "ok", "usage": {"spent_usd": 0.3, "task_cap": 0.27}}))
    (run / "deepscholar" / "ctl_d1_sft" / "tasks" / "v2" / "state.json").unlink()
    out = rl.gate(run, {"ctl_d1": 0.004, "ctl_d1_sft": 0.003, "ctl_d1_rl": 0.001})
    assert not out["arms"]["ctl_d1_sft"]["checks"]["all_ok"] and out["claim"] == "ctl_d1"

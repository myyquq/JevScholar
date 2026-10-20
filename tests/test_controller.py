"""d1 decision model: examples, option shuffling, batching, metrics and trajectory labels (pure);
readout parity with the checkpoint's own `system_one` and a learning check (`-m gpu`, in
`.venv-train`)."""

from __future__ import annotations

import json
import math
import random

import pytest

from jevscholar.controller.d1 import (
    Example,
    batches,
    calibration,
    load_recipe,
    read_examples,
    reorder,
    shuffled,
    write_examples,
)

Q = {"type": "choice", "instructions": "Pick.", "criteria": {"a": "1st", "b": "2nd", "c": None}}


def test_examples_validate_and_round_trip(tmp_path):
    soft = Example({"x": "状态"}, Q, {"a": 0.5, "b": 0.5}, weight=0.5, render={"style": "sections"})
    hard = Example.labelled("s", Q, "b")
    write_examples(tmp_path / "e.jsonl", [soft, hard])
    assert read_examples(tmp_path / "e.jsonl") == [soft, hard]
    assert (soft.targets(), soft.label()) == ([0.5, 0.5, 0.0], 0)
    assert (hard.targets(), hard.label()) == ([0.0, 1.0, 0.0], 1)
    with pytest.raises(ValueError, match="not options"):
        Example.labelled("s", Q, "z")
    with pytest.raises(ValueError, match="distribution"):
        Example("s", Q, {"a": 0.5})
    with pytest.raises(ValueError, match="only choice"):
        Example.labelled("s", {**Q, "type": "noul"}, "a")


def test_shuffling_moves_options_never_targets():
    moved = reorder(Q, [2, 0, 1])["criteria"]
    assert list(moved.items()) == [("c", None), ("a", "1st"), ("b", "2nd")]
    data = [Example.labelled("s", Q, label, render={"style": "sections"}) for label in "abc"]
    first = list(shuffled(data, random.Random(1)))
    assert first == list(shuffled(data, random.Random(1)))  # deterministic per seed
    assert sorted(max(e.target, key=e.target.get) for e in first) == ["a", "b", "c"]
    for e in first:
        assert e.question["criteria"].items() == Q["criteria"].items()  # same options, any order
        assert e.render == {"style": "sections"}
        names = list(e.question["criteria"])
        assert e.targets()[e.label()] == 1.0 and names[e.label()] in e.target
    orders = {tuple(e.question["criteria"]) for e in shuffled(data * 10, random.Random(2))}
    assert len(orders) > 1


def test_batches_bound_padded_tokens():
    assert batches([3, 3, 3, 3], 6) == [[0, 1], [2, 3]]
    assert batches([2, 5, 1], 10) == [[0, 1], [2]]  # 3 rows x 5 > 10
    assert batches([12, 1], 10) == [[0], [1]]  # an overlong row stands alone


def test_calibration_metrics():
    perfect = calibration([[1.0, 0.0], [0.0, 1.0]], [0, 1])
    assert perfect == {"n": 2, "accuracy": 1.0, "nll": 0.0, "brier": 0.0, "ece": 0.0}
    flat = calibration([[0.5, 0.5], [0.5, 0.5]], [0, 1])
    assert flat["accuracy"] == 0.5 and flat["ece"] == 0.0  # 50% sure, 50% right
    assert flat["nll"] == pytest.approx(math.log(2)) and flat["brier"] == pytest.approx(0.5)
    sure_wrong = calibration([[0.9, 0.1]], [1])
    assert sure_wrong["ece"] == pytest.approx(0.9) and sure_wrong["brier"] == pytest.approx(1.62)


def test_training_recipes_resolve(tmp_path):
    from jevscholar.paths import CONFIGS, ROOT

    cfg = load_recipe(CONFIGS / "training.yaml", "sanity")
    assert cfg.train == ROOT / "data/derived/controller/sanity/train.jsonl"
    assert (cfg.epochs, cfg.accum, cfg.init_adapter) == (2, 4, None)
    recipes = tmp_path / "t.yaml"
    recipes.write_text("rl: {train: a.jsonl, dev: b.jsonl, out: o, init_adapter: sft/adapter}\n")
    assert load_recipe(recipes, "rl").init_adapter == ROOT / "sft/adapter"
    with pytest.raises(ValueError, match="no recipe"):
        load_recipe(recipes, "sft")


# --------------------------------------------------------------------------- trajectories
def train_run(tmp_path, monkeypatch, tasks=("t1", "t2")):
    """A train run with one fixed-policy episode per task: an abstract batch finds gold A, a
    keyword batch gold B, then the batch cap forces finish."""
    from jevscholar.controller import data
    from jevscholar.controller.state import Candidate, Paper, State, snapshot
    from jevscholar.types import Task

    def row(title):
        return {"url": f"http://x.org/{title.replace(' ', '_')}", "title": title, "snippet": "s"}

    gold_a, gold_b, other = row("gold a"), row("gold b"), row("other")
    abstract = Candidate("abstract", "abs", (gold_a, other), 150, 0.001)
    explore = Candidate("explore", "kw", (row("noise"),), 10, 0.001)
    keyword = Candidate("explore", "kw2", (gold_b,), 10, 0.001)
    finish = Candidate("finish")
    s0 = State("T", 0.27, "abs")
    s1 = State("T", 0.27, "abs", spent_usd=0.01, batches=1)
    s1.papers = [Paper(gold_a, True), Paper(other, False)]
    s2 = State("T", 0.27, "abs", spent_usd=0.02, batches=2)
    s2.papers = [*s1.papers, Paper(gold_b, True)]
    steps = [  # (snapshot, executed, forced, the decision's own policy cost)
        (snapshot(s0, [abstract, explore], 0), 0, False, 0.001),  # left out of the returns
        (snapshot(s1, [keyword, finish], 0), 0, False, 0.0),
        (snapshot(s2, [finish], 0), 0, True, 0.0),
    ]
    run = tmp_path / "train-run"
    for task in tasks:
        folder = run / "deepscholar" / "ctl" / "tasks" / task
        folder.mkdir(parents=True)
        rows = [gold_a, gold_b, other, row("noise")]
        (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        lines = [
            snap
            | {"requested": c, "executed": c, "forced": f, "overridden": False}
            | {"control_usd": u, "probs": None, "input": None, "spent_after_usd": 0.0}
            for snap, c, f, u in steps
        ]
        (folder / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (folder / "result.json").write_text("{}")
    experiment = {"benchmarks": {"deepscholar": {"split": "train", "systems": ["ctl"]}}}
    (run / "manifest.json").write_text(json.dumps({"experiment": experiment}))
    tasks_of = {t: Task("deepscholar", t, "T") for t in tasks}
    monkeypatch.setattr(data, "run_tasks", lambda manifest, bench: tasks_of)
    gold = {"gold a", "gold b"}
    monkeypatch.setattr(data, "gold_found", lambda task, ids, titles: (len(titles & gold), 2))
    return run


def test_trajectories_label_each_decision(tmp_path, monkeypatch):
    from jevscholar.controller.data import trajectories

    found = trajectories([train_run(tmp_path, monkeypatch)])
    assert [t["task"] for t in found] == ["t1", "t2"]
    t = found[0]
    assert [p["scored"] for p in t["points"]] == [True, True, False]  # the forced finish is not
    returns = [0.0, 0.075 - 0.1 * 0.009 / 0.27, 0.15 - 0.1 * 0.019 / 0.27]
    assert [p["return"] for p in t["points"]] == pytest.approx(returns)
    assert t["end"] == pytest.approx(returns[-1])  # where the episode finished
    proxy = t["points"][0]["proxy"]  # abstract: gold A; explore: nothing
    assert proxy == pytest.approx([0.075 - 0.1 * 0.001 / 0.27, -0.1 * 0.001 / 0.27])
    assert len(t["rows"]) == 4 and t["points"][0]["executed"] == 0
    with pytest.raises(ValueError, match="train"):
        manifest = tmp_path / "train-run" / "manifest.json"
        manifest.write_text(manifest.read_text().replace('"train"', '"dev"'))
        trajectories([tmp_path / "train-run"])


# --------------------------------------------------------------------------- GPU (.venv-train)
@pytest.fixture(scope="module")
def d1():
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from jevscholar.controller.d1 import D1

    return D1.load()


@pytest.mark.gpu
def test_readout_matches_system_one(d1):
    """Our batched, right-padded readout equals the checkpoint's own one-pass API."""
    items = [
        ("Refund my order #123, it arrived broken.", {**Q, "criteria": {"refund": "", "no": ""}}),
        ({"title": "Attention is all you need", "abstract": "We propose the Transformer."}, Q),
        ("这篇论文研究检索增强生成。", {**Q, "criteria": {"nlp": "自然语言处理", "cv": "视觉"}}),
    ]
    ours = d1.probabilities(items)
    for (state, q), p in zip(items, ours, strict=True):
        ref = d1.base.system_one(state, {"q": q})["answers"]["q"]["probabilities"]
        # bf16 noise is ~0.003 (system_one itself is 0.0015 off fp32); flash SDPA drifted 0.015
        assert p == pytest.approx(list(ref.values()), abs=5e-3)
    d1.model.train()  # system_one switched the model to eval


@pytest.mark.gpu
def test_a_few_steps_fit_a_tiny_set(tmp_path):
    """A fresh adapter learns a label the untrained model does not prefer."""
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from jevscholar.controller.d1 import TrainConfig, train

    q = {"type": "choice", "instructions": "Which colour is named?", "criteria": {
        "red": "red", "blue": "blue", "green": "green"}}  # fmt: skip
    data = [Example.labelled(f"Item {i}: the colour is red.", q, "green") for i in range(16)]
    write_examples(tmp_path / "train.jsonl", data)
    write_examples(tmp_path / "dev.jsonl", data[:8])
    cfg = TrainConfig(
        train=tmp_path / "train.jsonl", dev=tmp_path / "dev.jsonl", out=tmp_path / "out",
        epochs=2, lr=1e-3, accum=1, batch_tokens=256, warmup=0.0,  # ~5 rows per step
    )  # fmt: skip
    history = train(cfg, log=lambda *_: None)["history"]
    assert history[-1]["dev"]["accuracy"] == 1.0 > history[0]["dev"]["accuracy"]
    from jevscholar.controller.d1 import D1

    served = D1.load(adapter=tmp_path / "out" / "adapter")  # the checkpoint's own API serves it
    assert served.base.system_one(data[0].state, {"q": q})["answers"]["q"]["choice"] == "green"


@pytest.mark.gpu
def test_server_serves_a_loop_state_as_system_one_reads_it(d1):
    """A full rendering of a late loop state fits d1's limit under each profile, and the
    server's answers are the checkpoint's own `system_one` answers for that rendering; the
    prompted policy's copy of the user turn is the checkpoint's own."""
    import importlib
    import threading

    import requests

    from jevscholar.controller.d1 import user_turn
    from jevscholar.controller.prompt import Profile
    from jevscholar.controller.serve import make_server
    from jevscholar.controller.state import Candidate, Paper, Query, State

    def row(i: int) -> dict:
        title = f"Retrieval-augmented agents for scientific literature synthesis, part {i}"
        return {"title": title, "url": f"http://arxiv.org/abs/2401.{i:05d}v1", "snippet": "A" * 300}

    state = State("Write the related-work section of: " + "An abstract sentence. " * 60, 0.27)
    state.spent_usd, state.batches = 0.11, 5
    state.queries = [Query(f"literature synthesis agents {i}", 1, 10, 4) for i in range(12)]
    state.papers = [Paper(row(i), relevant=i % 2 == 0) for i in range(80)]
    batch = tuple(row(100 + i) for i in range(10))
    candidates = [
        Candidate("abstract", state.topic, batch, 150, 0.008),
        Candidate("explore", "citation recommendation", batch, 10, 0.008),
        Candidate("deepen", "literature synthesis agents 3", batch, 20, 0.008),
        Candidate("finish"),
    ]
    runner = importlib.import_module(type(d1.base).__module__.rsplit(".", 1)[0] + ".runner")
    server = make_server(d1, {"adapter": None}, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/choice"
    profiles = [
        Profile(),  # d1's own
        Profile(titles=20, style="json_only", threshold=None),  # the checkpoint's default layout
    ]
    try:
        for p in profiles:
            request = p.request(state, candidates)
            out = requests.post(url, json={"requests": [request], "render": p.render}, timeout=600)
            out = out.json()
            assert out["tokens"] < 7552
            one = runner.SystemOne(model=d1.base, tokenizer=d1.tokenizer, state_style=p.style)
            ((got,),) = out["probabilities"]
            (choice,) = request["questions"]
            (want,) = one.probabilities(request["state"], [choice])
            assert got == pytest.approx(want, abs=5e-3)
            full = d1.prompt.render(
                d1.tokenizer, request["state"], d1.prompt.as_question(choice), d1.bos, d1.lead,
                **p.render,
            )  # fmt: skip
            user = user_turn(request["state"], choice, p.render)
            assert f"<|im_start|>user\n{user}<|im_end|>" in full
    finally:
        server.shutdown()
    d1.model.train()  # system_one switched the model to eval

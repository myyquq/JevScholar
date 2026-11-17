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
    regret,
    reorder,
    selection_key,
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
    assert (cfg.epochs, cfg.accum, cfg.dev_points) == (2, 4, None)
    sft = load_recipe(CONFIGS / "training.yaml", "sft")  # the registered warm start (docs/06)
    assert (sft.lr, sft.schedule, sft.warmup, sft.accum, sft.epochs, sft.seed) == (
        5e-5, "constant", 0.0, 1, 2, 0
    )  # fmt: skip
    assert sft.dev_points == ROOT / "data/derived/controller/sft/dev.points.jsonl"
    rl1 = load_recipe(CONFIGS / "training.yaml", "rl1")  # the registered RL round
    assert (rl1.kind, len(rl1.points), rl1.samples, rl1.lr, rl1.optimizer_steps) == (
        "rl", 8, 4, 2e-5, 4
    )  # fmt: skip
    assert len({k for k, _ in rl1.points}) == 8 and all(3 <= b <= 5 for _, b in rl1.points)
    assert rl1.reference == rl1.init_adapter == ROOT / "runs/controller/sft/adapter"
    recipes = tmp_path / "t.yaml"
    recipes.write_text("rl: {train: a.jsonl, dev: b.jsonl, out: o}\n")
    assert load_recipe(recipes, "rl").train == ROOT / "a.jsonl"
    with pytest.raises(ValueError, match="no recipe"):
        load_recipe(recipes, "sft")


def test_regret_follows_the_deployed_decoder():
    points = [
        {"kinds": ["abstract", "explore", "finish"], "returns": [0.03, 0.01, 0.02]},
        {"kinds": ["abstract", "finish"], "returns": [0.0, 0.01]},
    ]
    probs = [[0.2, 0.3, 0.5], [0.6, 0.4]]  # finish at 0.5 on the first; abstract on the second
    out = regret(probs, points)
    assert out["regret"] == pytest.approx((0.01 + 0.01) / 2)
    assert out["expected_regret"] == pytest.approx(
        ((0.03 - (0.006 + 0.003 + 0.01)) + (0.01 - 0.004)) / 2
    )
    assert (out["search_points"], out["search_regret"]) == (1, pytest.approx(0.02))  # explore
    with pytest.raises(ValueError, match="2 probabilities for 3 options"):
        regret([[0.5, 0.5]], points[:1])


def test_kept_adapters_rank_by_regret_then_cross_entropy():
    steps = [
        {"step": 0, "dev": {"regret": 0.004, "ce": 0.9}},
        {"step": 1, "dev": {"regret": 0.002, "ce": 1.1}},
        {"step": 2, "dev": {"regret": 0.002, "ce": 1.0}},
        {"step": 3, "dev": {"regret": 0.002, "ce": 1.0}},
    ]
    assert min(steps, key=lambda h: (*selection_key(h["dev"]), h["step"]))["step"] == 2
    assert selection_key({"ce": 0.5}) == (0.5,)  # without labelled points: cross-entropy


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
    result = train(cfg, log=lambda *_: None)
    history = result["history"]
    assert [h["step"] for h in history] == list(range(len(history)))  # every step kept
    assert history[-1]["dev"]["accuracy"] == 1.0 > history[0]["dev"]["accuracy"]
    best = tmp_path / "out" / "steps" / str(result["best_step"]) / "adapter"
    selected = tmp_path / "out" / "adapter" / "adapter_model.safetensors"
    assert selected.read_bytes() == (best / "adapter_model.safetensors").read_bytes()
    from jevscholar.controller.d1 import D1

    served = D1.load(adapter=tmp_path / "out" / "adapter")  # the checkpoint's own API serves it
    assert served.base.system_one(data[0].state, {"q": q})["answers"]["q"]["choice"] == "green"


@pytest.mark.gpu
def test_sparse_evaluation_keeps_the_last_step(tmp_path):
    """Unequal prompts repack per epoch; the last optimiser step is still kept and evaluated."""
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from jevscholar.controller.d1 import TrainConfig, train

    q = {"type": "choice", "instructions": "Which colour is named?", "criteria": {
        "red": "red", "blue": "blue", "green": "green"}}  # fmt: skip
    data = [
        Example.labelled("Item: the colour is red. " * (1 + 40 * (i % 3)), q, "green")
        for i in range(8)
    ]
    write_examples(tmp_path / "train.jsonl", data)
    write_examples(tmp_path / "dev.jsonl", data[:2])
    cfg = TrainConfig(
        train=tmp_path / "train.jsonl", dev=tmp_path / "dev.jsonl", out=tmp_path / "out",
        epochs=2, lr=1e-3, accum=2, batch_tokens=700, warmup=0.0, eval_every=100,
    )  # fmt: skip
    result = train(cfg, log=lambda *_: None)
    steps = [h["step"] for h in result["history"]]
    assert steps[0] == 0 and steps[-1] > 0 and len(steps) == 2  # step 0 and the last step
    assert (tmp_path / "out" / "steps" / str(steps[-1]) / "adapter").is_dir()


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


def test_grpo_terms_clip_and_anchor():
    torch = pytest.importorskip("torch")
    from jevscholar.controller.d1 import grpo_terms

    logits = torch.zeros(2, requires_grad=True)
    lp = logits.log_softmax(-1)  # 0.5, 0.5
    surrogate, kl, clipped = grpo_terms(lp, [0.5, 0.5], [0.5, 0.5], 0, 1.0, 0.2)
    assert surrogate.item() == pytest.approx(-1.0) and kl.item() == pytest.approx(0.0)
    assert not clipped
    surrogate.backward()
    assert logits.grad[0] < 0 < logits.grad[1]  # descending the loss raises the sampled option

    logits = torch.zeros(2, requires_grad=True)
    surrogate, _, clipped = grpo_terms(logits.log_softmax(-1), [0.2, 0.8], [0.5, 0.5], 0, 1.0, 0.2)
    assert clipped and surrogate.item() == pytest.approx(-1.2)  # ratio 2.5, clipped at 1.2
    surrogate.backward()
    assert logits.grad.abs().max() == 0  # no push beyond the clip
    _, kl, _ = grpo_terms(torch.tensor([0.9, 0.1]).log(), [0.5, 0.5], [0.5, 0.5], 0, 1.0, 0.2)
    assert kl.item() == pytest.approx(0.9 * math.log(1.8) + 0.1 * math.log(0.2))


@pytest.mark.gpu
def test_an_rl_step_follows_the_advantage_and_stops_on_kl(tmp_path):
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from jevscholar.controller.d1 import D1, Lora, RLConfig, rl_update
    from jevscholar.controller.rl import Step, write_steps
    from jevscholar.controller.serve import adapter_sha

    sft = D1.load(lora=Lora().model_dump())
    sft.model.save_pretrained(str(tmp_path / "sft"))
    q = {"type": "choice", "instructions": "Which colour is named?", "criteria": {
        "red": "red", "blue": "blue", "green": "green"}}  # fmt: skip
    state = "Item: the colour is red."
    (old,) = sft.rows([Example.labelled(state, q, "green")])
    del sft
    torch.cuda.empty_cache()
    sha = adapter_sha(tmp_path / "sft")
    steps = [Step(state, q, {}, 2, old, a, sha, g, t) for g, t, a in (
        ("g1", "rollout", 1.0), ("g1", "rollout_s1", -1.0),
        ("g2", "rollout", 1.0), ("g2", "rollout_s1", -1.0))]  # fmt: skip
    steps[1] = Step(state, q, {}, 0, old, -1.0, sha, "g1", "rollout_s1")
    steps[3] = Step(state, q, {}, 0, old, -1.0, sha, "g2", "rollout_s1")
    write_steps(tmp_path / "steps.jsonl", steps)
    base = {"kind": "rl", "source": "src", "system": "ctl_d1", "points": [], "prefix": "rl1",
            "steps": tmp_path / "steps.jsonl", "init_adapter": tmp_path / "sft",
            "reference": tmp_path / "sft", "lr": 1e-3, "optimizer_steps": 1}  # fmt: skip
    loose = RLConfig(**base | {"max_kl": 10.0}, out=tmp_path / "rl")
    result = rl_update(loose, log=lambda *_: None)
    assert result["kept_step"] == 1 and result["history"][0]["kl"] == pytest.approx(0, abs=1e-6)
    trained = D1.load(adapter=tmp_path / "rl" / "adapter")
    (after,) = trained.rows([Example.labelled(state, q, "green")])
    assert after[2] > old[2] and after[0] < old[0]  # toward the positive advantage
    del trained
    torch.cuda.empty_cache()
    tight = RLConfig(**base | {"max_kl": 1e-9}, out=tmp_path / "tight")
    assert rl_update(tight, log=lambda *_: None)["kept_step"] == 0  # stopped: the init adapter


def test_d1_samples_seeded_per_task_and_decision():
    from types import SimpleNamespace

    from jevscholar.controller.policy import make_policy
    from jevscholar.controller.state import State

    dist = [0.1, 0.5, 0.4]
    ctx = SimpleNamespace(tag=lambda stage: f"task#{stage}", task=SimpleNamespace(seed=0))
    policy = make_policy({"kind": "d1", "sample": True})
    draws = [policy._sample(ctx, State("T", 0.27, decisions=i), dist) for i in range(200)]
    expected = [random.Random(f"task#control#0#{i}").choices(range(3), weights=dist)[0]
                for i in range(200)]  # fmt: skip
    assert draws == expected
    assert policy.model_dump(mode="json")["sample"] is True  # the rollout members' identity

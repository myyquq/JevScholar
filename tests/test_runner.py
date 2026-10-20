"""Runner and judging: statuses, degraded re-queue, resume, cell-hash refusal, export audit, eval
skip/retry and eval fingerprints; experiment resolution (variants, extends, samples)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import ClassVar

import pytest
import requests

from jevscholar import judging, runner, types
from jevscholar.config import BenchCell, Budget, Experiment, load_experiment, load_models
from jevscholar.search import BudgetExhausted
from jevscholar.types import Benchmark, EvalContext, Report, Source, System, Task, load_result

MODES = ["ok", "boom", "partial", "degraded", "broke", "spend"]
calls: list[tuple[str, str]] = []  # (system, task id) per system run and per judging
failing: set[str] = set()  # task ids whose judging raises


class DummySystem(System):
    class Params(types.Strict):
        k: int | None = None
        flag: int = 0

    async def run(self, ctx: types.TaskContext) -> Report:
        task = ctx.task
        calls.append((ctx.system, task.id))
        mode = task.meta["mode"]
        if mode == "boom":
            raise RuntimeError("boom")
        if mode == "broke":
            raise BudgetExhausted("402 on charge")
        if mode == "spend":  # 2 x $0.008 crosses the $0.01 task cap (and the $0.015 run cap)
            for _ in range(2):
                url = f"{ctx.gateway}/t/{ctx.tag('tools')}/charge"
                requests.post(url, json={"tool": "tavily_search"}, timeout=10)
        meta = {"status": mode} if mode in ("partial", "degraded") else {}
        return Report(f"answer {task.id} [1]", (Source("u", "https://x.org", "X"),), meta)


class DummyBench(Benchmark):
    name = "dummyb"
    primary = "score"
    metrics: ClassVar[dict[str, int]] = {"score": 1}
    answers: ClassVar[dict[str, str]] = {}  # task id -> gold answer (default "a")

    def tasks(self, split: str, limit: int | None = None) -> list[Task]:
        tasks = [
            Task("dummyb", f"t{i}", f"prompt {m}", meta={"mode": m}) for i, m in enumerate(MODES)
        ]
        return tasks[:limit]

    def export_task(self, task: Task, result: types.TaskResult | None, out: Path) -> int:
        report = result and result.report
        (out / f"{task.id}.txt").write_text(report.text if report else "")
        return int(bool(report))  # stands in for the policy-dropped source count

    def gold(self, task: Task) -> str:
        return self.answers.get(task.id, "a")

    def evaluate_task(self, task: Task, export: Path, ctx: EvalContext) -> dict:
        calls.append((ctx.system, task.id))
        if task.id in failing:
            raise RuntimeError("judge down")
        return {
            "metrics": {"score": 1.0},
            "status": {"score": "ok"},
            "versions": dict(self.metrics),
            "judge": ctx.judge("score"),
        }


@pytest.fixture
def kw(monkeypatch, tmp_path) -> dict:
    """Registered dummies plus the run_experiment/eval_run keyword arguments of an offline run."""
    monkeypatch.setitem(types.SYSTEMS, "dummy", f"{__name__}:DummySystem")
    monkeypatch.setitem(types.BENCHMARKS, "dummyb", f"{__name__}:DummyBench")
    calls.clear()
    failing.clear()
    monkeypatch.setattr(runner, "is_peak", lambda _: False)  # whatever the hour
    judgesets = {"deepseek": {"dummyb": "ds-judge-chat", "dummyb.score": "ds-judge-think"}}
    models = load_models().model_copy(update={"judgesets": judgesets})
    return {"models": models, "cache_dir": tmp_path / "cache", "secrets": {}}


def _exp(run_usd: float, **dummy_params) -> Experiment:
    return Experiment(
        name="t",
        workers=0,
        budget=Budget(task_usd=0.01, run_usd=run_usd),
        systems={"dummy": dummy_params, "dummy_alt": {"use": "dummy", "flag": 1}},
        benchmarks={"dummyb": BenchCell(systems=("dummy", "dummy_alt"))},
    )


def test_run_resume_export_and_eval(kw, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    stopped = runner.run_experiment(_exp(0.015), run_dir, **kw)
    assert stopped and "run budget" in stopped  # dummy_alt never dispatched
    tasks = run_dir / "dummyb" / "dummy" / "tasks"
    results = {m: load_result(tasks / f"t{i}" / "result.json") for i, m in enumerate(MODES)}
    assert {m: r.status for m, r in results.items()} == {
        "ok": "ok",
        "boom": "failed",
        "partial": "partial",
        "degraded": "degraded",
        "broke": "budget_exhausted",
        "spend": "ok",
    }  # a finished report is not blamed on the latch
    assert "RuntimeError: boom" in results["boom"].error and results["boom"].report is None
    assert results["spend"].usage["latched"]
    assert not (run_dir / "dummyb" / "dummy_alt").exists()

    calls.clear()  # resume with a larger run cap (budgets are not part of the cell hash)
    assert runner.run_experiment(_exp(1.0), run_dir, **kw) is None
    assert calls == [("dummy", "t3"), *(("dummy_alt", f"t{i}") for i in range(len(MODES)))]
    assert (tasks / "t3" / "result.degraded.json").exists()
    calls.clear()  # each degraded task is re-queued once only
    assert runner.run_experiment(_exp(1.0), run_dir, **kw) is None
    assert calls == [("dummy_alt", "t3")]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [s["stopped"] is None for s in manifest["sessions"]] == [False, True, True]
    assert manifest["tasks"] == {"dummyb": [f"t{i}" for i in range(len(MODES))]}
    cell = manifest["cells"]["dummyb"]["dummy"]
    assert cell["hash"] == results["ok"].cell_hash and cell["version"] == "1"
    assert cell["routing"]["model"] == "deepseek-flash" and "price" not in cell["routing"]

    with pytest.raises(RuntimeError, match="configuration changed"):
        runner.run_experiment(_exp(1.0, k=5), run_dir, **kw)

    runner.export_run(run_dir)
    audit = json.loads((run_dir / "dummyb" / "dummy" / "export" / "audit.json").read_text())
    assert audit == {"dropped_sources": {"t0": 1, "t1": 0, "t2": 1, "t3": 1, "t4": 0, "t5": 1}}

    manifest["experiment"]["budget"]["run_usd"] = 0.0  # spent: stops generation, not judging
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    calls.clear()
    failing.add("t0")
    assert judging.eval_run(run_dir, "deepseek", **kw) is None
    judged = [(s, t) for s in ("dummy", "dummy_alt") for t in ("t0", "t2", "t3", "t5")]
    assert calls == judged  # tasks without a report are never judged
    ev = run_dir / "dummyb" / "dummy" / "eval" / "deepseek"
    files = {p.stem: json.loads(p.read_text()) for p in ev.glob("*.json")}
    assert "error" in files["t0"] and files["t1"]["status"] == files["t4"]["status"] == "no_report"
    assert files["t2"]["metrics"] == {"score": 1.0} and files["t2"]["judge"] == "ds-judge-think"
    calls.clear()
    failing.clear()
    assert judging.eval_run(run_dir, "deepseek", **kw) is None
    assert calls == [("dummy", "t0"), ("dummy_alt", "t0")]  # only errored evals are redone

    calls.clear()  # a report generated again (here: edited) is judged afresh
    path = tasks / "t2" / "result.json"
    path.write_text(path.read_text().replace("answer t2", "a better answer"))
    assert judging.eval_run(run_dir, "deepseek", **kw) is None
    assert calls == [("dummy", "t2")]
    calls.clear()  # ... and a task whose gold data changed
    monkeypatch.setitem(DummyBench.answers, "t3", "b")
    assert judging.eval_run(run_dir, "deepseek", **kw) is None
    assert calls == [("dummy", "t3"), ("dummy_alt", "t3")]
    calls.clear()  # a metric at a new protocol version is judged again
    monkeypatch.setattr(DummyBench, "metrics", {"score": 2})
    assert judging.eval_run(run_dir, "deepseek", **kw) is None
    assert sorted(calls) == judged
    calls.clear()  # so is every report under other judges
    judgesets = {"deepseek": {"dummyb": "ds-judge-chat"}}
    models = kw["models"].model_copy(update={"judgesets": judgesets})
    assert judging.eval_run(run_dir, "deepseek", **kw | {"models": models}) is None
    assert sorted(calls) == judged
    calls.clear()  # ... or under a judge alias that now routes elsewhere
    aliases = models.aliases | {"ds-judge-chat": models.aliases["ds-judge-think"]}
    models = models.model_copy(update={"aliases": aliases})
    assert judging.eval_run(run_dir, "deepseek", **kw | {"models": models}) is None
    assert sorted(calls) == judged
    with pytest.raises(ValueError, match="cannot judge"):  # not a separately judged metric
        judging.eval_run(run_dir, "deepseek", metrics=("organization",), **kw)
    calls.clear()  # a benchmark's default metrics: the others are judged only on request
    monkeypatch.setattr(DummyBench, "metrics", {"score": 2, "extra": 1})
    monkeypatch.setattr(DummyBench, "judged", ("score",))
    assert judging.eval_run(run_dir, "deepseek", **kw | {"models": models}) is None
    assert calls == []  # "score" is settled at its version; "extra" was not asked for
    with pytest.raises(ValueError, match="among the judged metrics"):
        judging.eval_run(run_dir, "deepseek", metrics=("score",), redo=("extra",), **kw)
    assert judging.eval_run(run_dir, "deepseek", redo=("extra",), **kw | {"models": models}) is None
    assert sorted(calls) == judged  # a metric judged again is requested, not just cleared

    import shutil  # another run with the same reports starts from the first run's verdicts

    shutil.copytree(run_dir, tmp_path / "again", ignore=shutil.ignore_patterns("eval"))
    calls.clear()
    again = kw | {"models": models}
    assert judging.eval_run(tmp_path / "again", "deepseek", reuse=(run_dir,), **again) is None
    assert calls == []
    assert judging.eval_run(tmp_path / "again", "deepseek", **again) is None and calls == []
    ev = tmp_path / "again" / "dummyb" / "dummy" / "eval" / "deepseek" / "t2.json"
    ev.write_text(json.dumps({"metrics": {}, "error": "judge down", "fingerprint": "x"}))
    assert judging.eval_run(tmp_path / "again", "deepseek", reuse=(run_dir,), **again) is None
    assert calls == [] and "error" not in json.loads(ev.read_text())  # a stale file is replaced


def test_worker_env_and_judges(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-real")
    env = runner.worker_env({"TAVILY_API_KEY": "tvly-x", "DEEPSEEK_API_KEY": "sk-real"})
    assert "DEEPSEEK_API_KEY" not in env and "sk-real" not in env.values()
    assert env["OPENAI_API_KEY"] == env["OPEN_AI_API_KEY"] == types.LOCAL_KEY
    assert env["JEV_TAVILY_KEY"] == "tvly-x" and env["LITELLM_MODE"] == "PRODUCTION"
    assert any(Path(env["TIKTOKEN_CACHE_DIR"]).iterdir())  # litellm's bundled encodings

    judgesets = load_models().judgesets

    def judge(judgeset: str, bench: str, metric: str) -> str:
        judges = judging.judges_for(judgesets[judgeset], bench)
        return EvalContext("", "r", "s", judgeset, judges, tmp_path).judge(metric)

    assert judge("official", "deepscholar", "nugget_coverage") == "gpt-4.1-2025-04-14"
    assert judge("official", "deepscholar", "cite_p") == "gpt-4o-2024-08-06"
    assert {judge("deepseek", b, "any") for b in types.BENCHMARKS} == {
        "ds-judge-chat",
        "ds-judge-think",
    }


def test_needs_judging_by_fingerprint_error_version_and_redo(tmp_path):
    out = tmp_path / "t1.json"
    assert judging.prepare(out, "fp")  # never judged
    ev = {
        "metrics": {"a": 1.0, "b": None, "c": None},
        "status": {"a": "ok", "b": "skipped:no_key", "c": "not_requested"},
        "versions": {"a": 1, "b": 1},
    }
    out.write_text(json.dumps(ev | {"fingerprint": "fp"}))
    assert not judging.prepare(out, "fp")
    assert not judging.prepare(out, "fp", want={"a": 1, "b": 1})  # settled, current
    assert judging.prepare(out, "fp", want={"a": 2})  # another protocol version
    assert judging.prepare(out, "fp", want={"c": 1})  # not requested before
    assert judging.prepare(out, "fp", want={"d": 1})  # a metric new to the file
    assert not judging.prepare(out, "fp", ("d",))  # not a metric of this file
    assert judging.prepare(out, "fp", ("b",))
    assert json.loads(out.read_text())["status"] == {"a": "ok", "c": "not_requested"}
    assert judging.prepare(out, "fp", want={"b": 1})  # an interrupted redo stays pending
    errored = ev | {"fingerprint": "fp", "error": "metrics failed: b"}
    out.write_text(json.dumps(errored))
    assert judging.prepare(out, "fp", ("a",)) and out.exists()  # its settled metrics are
    assert json.loads(out.read_text())["status"] == {"b": "skipped:no_key", "c": "not_requested"}
    assert judging.prepare(out, "new") and not out.exists()  # reused; another input: none

    report, judges = Report("text [1]", (Source("u", "https://x.org", "X"),)), {"default": "j"}
    fp = judging.fingerprint(report, judges, judging.digest({"gold": 1}))
    assert fp == judging.fingerprint(
        dataclasses.replace(report, meta={"status": "ok"}), judges, judging.digest({"gold": 1})
    )
    assert fp != judging.fingerprint(dataclasses.replace(report, text="text [2]"), judges)
    assert fp != judging.fingerprint(report, {"default": "k"}) != judging.fingerprint(None, judges)
    assert fp != judging.fingerprint(report, judges, judging.digest({"gold": 2}))  # new gold data

    models = load_models()
    routing = judging.judge_routing(models, {"default": "ds-judge-chat"})["default"]
    alias = models.aliases["ds-judge-chat"]
    upstream = models.upstreams[alias.upstream]
    assert routing["model"] == alias.model and "price" not in routing and "cache" not in routing
    assert (routing["url"], routing["dialect"]) == (upstream.base_url, upstream.dialect)


CATALOG = """
k20: {use: simple_rag, k: 20}
raw: {use: simple_rag, query: raw}
k20_raw: {extends: k20, query: raw}
"""


def test_experiments_resolve_variants_extends_and_samples(tmp_path):
    catalog = tmp_path / "systems.yaml"
    catalog.write_text(CATALOG)
    path = tmp_path / "exp.yaml"
    path.write_text(
        "budget: {run_usd: 2.0}\n"
        "systems: {k5: {extends: k20_raw, k: 5}}\n"
        "benchmarks: {deepscholar: {samples: 2, systems: [k20, simple_rag]}}\n"
        "variants:\n"
        "  a: {benchmarks: {deepscholar: {split: dev2}}}\n"
        "  b:\n"
        "    budget: {judge_run_usd: 1.0}\n"
        "    benchmarks: {deepscholar: {samples: 1, systems: [k5]}}\n"
    )
    a = load_experiment(path, "a", catalog=catalog)
    assert a.name == "exp/a" and a.benchmarks["deepscholar"].split == "dev2"
    assert a.benchmarks["deepscholar"].systems == ("k20", "k20_s1", "simple_rag", "simple_rag_s1")
    assert a.systems == {
        "k20": {"use": "simple_rag", "k": 20},
        "k20_s1": {"use": "simple_rag", "k": 20, "sample": 1},
        "simple_rag_s1": {"use": "simple_rag", "sample": 1},
    }  # the run manifest records every definition: no catalog is needed to read it
    b = load_experiment(path, "b", ["budget.task_usd=0.5"], catalog=catalog)
    assert b.system_spec("k5") == ("simple_rag", {"k": 5, "query": "raw"})  # extends, deep-merged
    assert (b.budget.run_usd, b.budget.judge_run_usd, b.budget.task_usd) == (2.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="needs a variant"):
        load_experiment(path, catalog=catalog)
    with pytest.raises(ValueError, match="no variant"):
        load_experiment(path, "c", catalog=catalog)

    local = Experiment(  # the identity is the resolved definition's, wherever it came from
        name="t",
        systems={"k20": {"use": "simple_rag", "k": 20}},
        benchmarks={"deepscholar": BenchCell(split="dev2", systems=("k20",))},
    )
    assert a.cell_hash("deepscholar", "k20") == local.cell_hash("deepscholar", "k20")


@pytest.mark.parametrize(
    ("systems", "error"),
    [
        ("{k20: {use: simple_rag, k: 5}}", "redefine catalog"),  # a variant needs its own name
        ("{x: {extends: y}, y: {extends: x}}", "cycle"),
        ("{x: {use: nothing}}", "not a system class"),
        ("{}", "unknown system"),
        ("{x: {use: simple_rag}, k20_s1: {use: simple_rag, k: 5}}", "reuse the name"),
    ],
)
def test_bad_system_definitions_fail_at_load(tmp_path, systems, error):
    catalog = tmp_path / "systems.yaml"
    catalog.write_text(CATALOG)
    path = tmp_path / "exp.yaml"
    cell = "{samples: 2, systems: [k20, x]}"
    path.write_text(f"systems: {systems}\nbenchmarks: {{deepscholar: {cell}}}\n")
    with pytest.raises(ValueError, match=error):
        load_experiment(path, catalog=catalog)


def test_samples_and_lists_never_collide_across_benchmarks(tmp_path):
    """Regression: a sample's name taken by another definition silently swapped parameters,
    depending on the order of the benchmarks."""
    catalog = tmp_path / "systems.yaml"
    catalog.write_text(CATALOG)
    path = tmp_path / "exp.yaml"
    for order in ("deepscholar, drb2", "drb2, deepscholar"):
        a, b = order.split(", ")
        path.write_text(
            "systems: {a: {use: simple_rag, k: 5}, a_s1: {use: simple_rag, k: 20}}\n"
            f"benchmarks: {{{a}: {{samples: 2, systems: [a]}}, {b}: {{systems: [a_s1]}}}}\n"
        )
        with pytest.raises(ValueError, match="reuse the name"):
            load_experiment(path, catalog=catalog)
    path.write_text("benchmarks: {deepscholar: {systems: [k20, k20]}}\n")
    with pytest.raises(ValueError, match="twice"):
        load_experiment(path, catalog=catalog)
    twice = "{samples: 2, systems: [k20]}"  # the same system and samples on two benchmarks
    path.write_text(f"benchmarks: {{deepscholar: {twice}, drb2: {twice}}}\n")
    assert load_experiment(path, catalog=catalog).systems["k20_s1"]["sample"] == 1


def test_every_shipped_experiment_resolves_and_builds():
    from jevscholar.config import variants
    from jevscholar.paths import EXPERIMENTS

    for path in sorted(EXPERIMENTS.glob("*.yaml")):
        for variant in variants(path):
            exp = load_experiment(path, variant)
            for system in {s for cell in exp.benchmarks.values() for s in cell.systems}:
                use, params = exp.system_spec(system)
                runner.make_system(use, json.dumps(params))


def test_a_bad_system_parameter_fails_before_any_task_runs(kw, tmp_path, monkeypatch):
    from pydantic import ValidationError

    monkeypatch.setitem(types.SYSTEMS, "strict", "jevscholar.systems.closed_book:ClosedBook")
    exp = Experiment(
        name="t",
        workers=0,
        budget=Budget(task_usd=0.01),
        systems={"cb": {"use": "strict", "draft": True}},
        benchmarks={"dummyb": BenchCell(systems=("dummy", "cb"))},
    )
    with pytest.raises(ValidationError):
        runner.run_experiment(exp, tmp_path / "run", **kw)
    assert not calls and not (tmp_path / "run").exists()


def test_a_system_can_carry_its_own_cap(tmp_path):
    """DS-base runs at its natural cost: its own `task_usd` overrides the benchmark's B0 for it
    alone, and is part of its cell, not a parameter."""
    from jevscholar import benchmarks

    catalog = tmp_path / "systems.yaml"
    catalog.write_text(CATALOG + "own: {use: simple_rag, k: 20, task_usd: 2.0}\n")
    path = tmp_path / "exp.yaml"
    path.write_text("benchmarks: {deepscholar: {systems: [own, k20]}}\n")
    exp = load_experiment(path, catalog=catalog)
    assert exp.system_spec("own") == exp.system_spec("k20") == ("simple_rag", {"k": 20})
    assert benchmarks.task_caps(exp) == {("deepscholar", "own"): 2.0, ("deepscholar", "k20"): 0.27}
    runner.make_system(*exp.system_spec("own")[:1], json.dumps(exp.system_spec("own")[1]))


@pytest.mark.parametrize("cap", ['"2.0"', ".nan", "0", "true"])
def test_a_system_cap_must_be_a_positive_number(tmp_path, cap):
    catalog = tmp_path / "systems.yaml"
    catalog.write_text(CATALOG)
    path = tmp_path / "exp.yaml"
    path.write_text(
        f"systems: {{own: {{use: simple_rag, task_usd: {cap}}}}}\n"
        "benchmarks: {deepscholar: {systems: [own]}}\n"
    )
    with pytest.raises(ValueError, match="task_usd"):
        load_experiment(path, catalog=catalog)


def test_reused_verdicts_carry_the_judge_decisions(tmp_path):
    """DRB2 keeps its rubric decisions in a raw log next to the eval file: a reused verdict
    carries them, so review packs and agreement checks still see them."""
    from jevscholar.benchmarks.drb2 import DRB2

    donor = tmp_path / "a" / "eval" / "5.json"
    (donor.parent / "raw").mkdir(parents=True)
    donor.write_text(json.dumps({"metrics": {"total": 0.5}, "fingerprint": "f"}))
    scores = {"info_recall": {"item one": {"score": 1}, "item two": {"score": 0}}}
    (donor.parent / "raw" / "idx-5.json").write_text(json.dumps({"result": {"scores": scores}}))
    out = tmp_path / "b" / "eval" / "5.json"
    judging.reuse_verdicts(donor, out, DRB2(), Task("drb2", "5", "p"))
    copied = json.loads(out.read_text())
    assert copied["fingerprint"] == "f" and len(copied["items"]) == 2
    assert DRB2().items(copied, out.parent, Task("drb2", "5", "p")) == copied["items"]

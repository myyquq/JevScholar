"""`jev` command line: data preparation, runs, export, evaluation, reports, judge checks and the
d1 decision model.

Thin wrappers only; every command maps to one library function. Heavy modules (data, search,
runner, analysis) are imported inside the commands so `jev --help` stays instant.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from jevscholar.paths import CONFIGS, DATA_DERIVED, RUNS

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Fetch pinned sources and build splits.")
check_app = typer.Typer(no_args_is_help=True, help="Check the automated judges.")
controller_app = typer.Typer(no_args_is_help=True, help="The d1-3B decision model (.venv-train).")
app.add_typer(data_app, name="data")
app.add_typer(check_app, name="check")
app.add_typer(controller_app, name="controller")

Run = Annotated[str, typer.Argument(metavar="RUN", help="A run id under runs/, or a path.")]
Only = Annotated[list[str] | None, typer.Option("--only", help="Restrict to these benchmarks.")]
Workers = Annotated[int | None, typer.Option(help="Worker processes (0 = in-process).")]
Judges = Annotated[str, typer.Option("--judges", help="Judgeset from models.yaml.")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _run_dir(run: str) -> Path:
    from jevscholar.artifacts import run_dir

    path = run_dir(run)
    if not (path / "manifest.json").exists():
        raise typer.BadParameter(f"no run at {path}")
    return path


# --------------------------------------------------------------------------- data
@data_app.command("fetch")
def data_fetch(
    pin: Annotated[bool, typer.Option(help="Write computed sha256 for blank pins.")] = False,
) -> None:
    """Download every pinned source into data/raw/ and verify its sha256."""
    from jevscholar import data

    data.fetch(pin=pin)


@data_app.command("prepare")
def data_prepare() -> None:
    """Build data/derived/ and the committed data/splits/."""
    from jevscholar import data

    data.prepare()


@data_app.command("arxiv-index")
def data_arxiv_index() -> None:
    """Build the local BM25 index over the frozen arXiv metadata snapshot."""
    from jevscholar.search import arxiv_local

    typer.echo(arxiv_local.build_index())


@data_app.command("fulltext")
def data_fulltext(
    runs: Annotated[list[str], typer.Argument(help="Runs whose writer documents to fetch.")],
) -> None:
    """Fetch the full text of the runs' writer documents (arXiv PDFs) into cache/fulltext/."""
    from jevscholar.search import fulltext

    urls = [u for run in runs for u in fulltext.run_docs(_run_dir(run))]
    typer.echo(f"{len(set(urls))} documents")
    typer.echo(dict(fulltext.fetch(urls)))


# --------------------------------------------------------------------------- runs
@app.command()
def run(
    experiment: Annotated[str, typer.Argument(help="Experiment name or YAML path.")],
    variant: Annotated[str | None, typer.Option(help="One of the file's variants.")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", help="Override, a.b=value.")] = None,
    only: Only = None,
    run_id: Annotated[str | None, typer.Option(help="Resume or name a run.")] = None,
    workers: Workers = None,
    dry_run: Annotated[
        bool, typer.Option(help="Print the resolved experiment and its cells only.")
    ] = False,
) -> None:
    """Run (or resume) an experiment under runs/<run-id>/."""
    import yaml

    from jevscholar.config import experiment_path, load_experiment, load_models
    from jevscholar.runner import plan, run_experiment

    exp = load_experiment(experiment_path(experiment), variant, set_ or [])
    run_id = run_id or f"{exp.name.replace('/', '-')}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    if dry_run:
        cells, tasks, batches = plan(exp, RUNS / run_id, load_models(), only)
        typer.echo(yaml.safe_dump(exp.model_dump(mode="json"), sort_keys=False))
        for bench, by_system in cells.items():
            for system, cell in by_system.items():
                typer.echo(f"{bench}/{system}: cell {cell['hash']}, {len(tasks[bench])} tasks")
        typer.echo(f"{sum(map(len, batches))} tasks pending in runs/{run_id}")
        return
    stopped = run_experiment(exp, RUNS / run_id, only=only, workers=workers)
    typer.echo(f"run {run_id}: " + (f"stopped early ({stopped})" if stopped else "complete"))
    if stopped:
        raise typer.Exit(1)


@app.command()
def export(run_: Run, only: Only = None) -> None:
    """Write every task's official layout (failed or missing -> empty) and export/audit.json."""
    from jevscholar.runner import export_run

    export_run(_run_dir(run_), only=only)


@app.command("eval")
def eval_(
    run_: Run,
    judges: Judges = "deepseek",
    only: Only = None,
    workers: Workers = None,
    redo: Annotated[
        list[str] | None, typer.Option("--redo", help="Judge these metrics again.")
    ] = None,
    metrics: Annotated[
        list[str] | None, typer.Option("--metrics", help="Judge only these metrics.")
    ] = None,
    reuse: Annotated[
        list[str] | None,
        typer.Option("--reuse", help="Copy verdicts on identical reports from these runs."),
    ] = None,
) -> None:
    """Re-export, then judge every task not judged yet (errored evals are retried).

    Tasks without a usable report are recorded as "no_report" without a judge call. `--redo M`
    re-judges metric M everywhere; settled metrics are reused, not paid for again. `--metrics M`
    judges only M (DeepScholar); a later run with more metrics judges only those not settled
    yet, as it does a metric whose protocol version changed. `--reuse RUN` starts each task
    from RUN's verdicts where RUN judged exactly the same exported report."""
    from jevscholar.judging import eval_run
    from jevscholar.runner import export_run

    run_dir = _run_dir(run_)
    if metrics and (extra := sorted(set(redo or ()) - set(metrics))):
        raise typer.BadParameter(f"--redo {extra} must be among --metrics")
    export_run(run_dir, only=only)
    stopped = eval_run(
        run_dir,
        judges,
        only=only,
        workers=workers,
        redo=tuple(redo or ()),
        metrics=tuple(metrics or ()),
        reuse=tuple(_run_dir(r) for r in reuse or ()),
    )
    if stopped:
        typer.echo(f"eval stopped early ({stopped})")
        raise typer.Exit(1)


@app.command()
def report(
    run_: Run,
    baseline: Annotated[
        str, typer.Option(help="Paired-comparison baseline.")
    ] = "deepscholar_base_intro",
    judges: Judges = "deepseek",
) -> None:
    """Write report.md, analysis/long.csv and per-task trajectory.jsonl."""
    from jevscholar.report import write_report

    typer.echo(write_report(_run_dir(run_), baseline=baseline, judgeset=judges))


# --------------------------------------------------------------------------- judge checks
@check_app.command("calibrate")
def check_calibrate(
    bench: Annotated[
        str,
        typer.Argument(
            help="drb2 or deepscholar (published reports), or deepscholar-controls (fixed "
            "reports bounding each metric)."
        ),
    ],
    run_id: Annotated[str | None, typer.Option(help="Run id (default: BENCH-calibration).")] = None,
    source: Annotated[
        str | None,
        typer.Option(help="deepscholar-controls: RUN/SYSTEM whose dev reports they start from."),
    ] = None,
    judges: Judges = "deepseek",
    workers: Workers = None,
) -> None:
    """Judge fixed reports: DRB2's commercial systems (ranked against the official boards), the
    DeepScholar authors' own DeepScholar-base reports (next to the paper's row), or DeepScholar
    evaluator controls (the reference section, the abstract, another task's report, ...)."""
    from jevscholar.checks import calibration

    run_dir = RUNS / (run_id or f"{bench}-calibration")
    kw = {"judgeset": judges} | ({} if workers is None else {"workers": workers})
    if bench == "drb2":
        calibration.calibrate_drb2(run_dir, **kw)
    elif bench == "deepscholar":
        calibration.calibrate_deepscholar(run_dir, **kw)
    elif bench == "deepscholar-controls":
        if source is None or "/" not in source:
            raise typer.BadParameter("deepscholar-controls needs --source RUN/SYSTEM")
        run, system = source.rsplit("/", 1)
        calibration.calibrate_deepscholar_controls(run_dir, _run_dir(run), system, **kw)
    else:
        raise typer.BadParameter("fixed reports exist for drb2, deepscholar, deepscholar-controls")
    typer.echo((run_dir / "calibration.md").read_text())


@check_app.command("agree")
def check_agree(
    run_: Annotated[str, typer.Argument(metavar="RUN", help="Run whose exports are re-judged.")],
    run_id: Annotated[str, typer.Option(help="Run id under runs/.")] = "judge-agreement",
    system: Annotated[
        list[str] | None,
        typer.Option(help="Systems to re-judge, the baseline first (default: all)."),
    ] = None,
    judges: Judges = "official",
    tasks: Annotated[
        list[str] | None,
        typer.Option("--tasks", help="BENCH=N leading tasks (default: 6 / 40 / 6)."),
    ] = None,
    workers: Workers = None,
) -> None:
    """Re-judge a subset of RUN with another judgeset and measure agreement with its DeepSeek
    verdicts (item kappa, paired-difference signs). Judged tasks are kept; re-run to extend."""
    from jevscholar.checks.agreement import TASKS, judge_subset, summarize

    source, run_dir = _run_dir(run_), RUNS / run_id
    plan = dict(TASKS)
    for spec in tasks or ():
        bench, _, n = spec.partition("=")
        if bench not in plan or not n.isdigit():
            raise typer.BadParameter(f"--tasks {spec!r}: expected one of {sorted(plan)}=N")
        plan[bench] = int(n)
    stopped = judge_subset(
        source, run_dir, systems=tuple(system or ()), judgeset=judges, tasks=plan, workers=workers
    )
    summarize(source, run_dir, judges, stopped=stopped)
    typer.echo((run_dir / "agreement.md").read_text())
    if stopped:
        typer.echo(f"judging stopped early ({stopped})")
        raise typer.Exit(1)


@check_app.command("review-pack")
def check_review_pack(
    run_: Run,
    out: Annotated[Path | None, typer.Option(help="Pack directory (new).")] = None,
    judges: Judges = "deepseek",
    system: Annotated[list[str] | None, typer.Option(help="Systems to sample.")] = None,
    seed: int = 0,
) -> None:
    """Sample judge decisions into a blind pack (key.json stays with the organiser)."""
    from jevscholar.checks.review import export_pack

    run_dir = _run_dir(run_)
    out = out or run_dir / f"review-{judges}"
    typer.echo(export_pack(run_dir, out, judgeset=judges, systems=system, seed=seed))
    typer.echo(f"give reviewers {out} without key.json")


@check_app.command("review-score")
def check_review_score(
    pack: Annotated[Path, typer.Argument(help="Pack directory with labels.csv.")],
) -> None:
    """Agreement and kappa of the reviewer's labels with the judge."""
    from jevscholar.checks.review import score_pack
    from jevscholar.report import md_table

    typer.echo(md_table(score_pack(pack)))


# --------------------------------------------------------------------------- d1
@controller_app.command("serve")
def controller_serve(
    adapter: Annotated[Path | None, typer.Option(help="A trained adapter (default: none).")] = None,
    port: Annotated[int, typer.Option(help="Local port (the d1 policy's `url`).")] = 9940,
) -> None:
    """Serve d1-3B's option probabilities to the controlled loop's d1 policy."""
    from jevscholar.controller.serve import serve

    serve(adapter, port)


@controller_app.command("trajectories")
def controller_trajectories(
    runs: Annotated[list[str], typer.Argument(metavar="RUN...", help="DeepScholar train runs.")],
    out: Annotated[Path, typer.Option(help="JSONL output.")] = DATA_DERIVED / "trajectories.jsonl",
) -> None:
    """The retrieval trajectories of controlled train runs, each decision with its return."""
    from jevscholar.controller.data import trajectories
    from jevscholar.util import write_jsonl

    found = trajectories(_run_dir(r) for r in runs)
    write_jsonl(out, found)
    points = sum(len(t["points"]) for t in found)
    typer.echo(f"{out}: {len(found)} trajectories, {points} retrieval decisions")


@controller_app.command("branches")
def controller_branches(
    source: Run,
    system: Annotated[str, typer.Option(help="The source run's controlled system.")] = "ctl_fixed",
    continuation: Annotated[
        str, typer.Option(help="The catalog system whose policy continues each branch.")
    ] = "ctl_fixed",
    tasks: Annotated[int | None, typer.Option(help="Only the run's first N tasks.")] = None,
    per_task: Annotated[int | None, typer.Option(help="At most N decision points a task.")] = None,
    run_usd: Annotated[float, typer.Option(help="Billed cap of each branch run.")] = 0.25,
    prefix: Annotated[str | None, typer.Option(help="Run id prefix.")] = None,
    workers: Workers = None,
    dry_run: Annotated[bool, typer.Option(help="Print the branch runs only.")] = False,
) -> None:
    """Branch every legal candidate of a train or val run's decision points (docs/06): one run
    per (batches, kind), runs/<prefix>-b<batches>-<kind>, resumable."""
    from jevscholar.controller.branches import continuation_policy, experiments, forks
    from jevscholar.runner import run_experiment

    path = _run_dir(source)
    found = forks(path, system, tasks=tasks, per_task=per_task)
    exps = experiments(
        path, system, found, policy=continuation_policy(continuation), run_usd=run_usd,
        prefix=prefix,
    )  # fmt: skip
    branches = sum(len(f.kinds) for f in found)
    typer.echo(f"{len(found)} decision points, {branches} branches in {len(exps)} runs")
    for run_id, exp in exps.items():
        typer.echo(f"  runs/{run_id}: {len(exp.benchmarks['deepscholar'].ids or ())} tasks")
    if dry_run:
        return
    for run_id, exp in exps.items():
        if stopped := run_experiment(exp, RUNS / run_id, workers=workers):
            typer.echo(f"run {run_id}: stopped early ({stopped})")
            raise typer.Exit(1)
        typer.echo(f"run {run_id}: complete")


@controller_app.command("labels")
def controller_labels(
    runs: Annotated[list[str], typer.Argument(metavar="RUN...", help="Branch runs of one split.")],
    out: Annotated[Path, typer.Option(help="Example JSONL (and a .points.jsonl beside it).")],
) -> None:
    """Soft SFT targets from branch runs: per decision point, the options with the best
    retrieval return share the target."""
    from jevscholar.controller.branches import examples, outcomes
    from jevscholar.controller.d1 import write_examples
    from jevscholar.util import write_jsonl

    found, points, incomplete = examples(outcomes(_run_dir(r) for r in runs))
    write_examples(out, found)
    write_jsonl(out.with_suffix(".points.jsonl"), points)
    typer.echo(f"{out}: {len(found)} examples; {incomplete} points left out (a branch missing)")


@controller_app.command("groups")
def controller_groups(
    source: Run,
    adapter: Annotated[Path, typer.Option(help="The served adapter directory (sampling policy).")],
    system: Annotated[str, typer.Option(help="The source run's controlled system.")] = "ctl_fixed",
    groups: Annotated[int, typer.Option(help="Decision points (groups) to sample.")] = 4,
    size: Annotated[int, typer.Option(help="Trajectories per group.")] = 4,
    rl_round: Annotated[
        int, typer.Option("--round", min=1, max=2, help="The RL round (picks the points).")
    ] = 1,
    run_usd: Annotated[float, typer.Option(help="Billed cap of each group run.")] = 0.15,
    prefix: Annotated[str | None, typer.Option(help="Run id prefix.")] = None,
    first_only: Annotated[
        bool, typer.Option(help="d1 samples the restored point only; the fixed rule continues.")
    ] = False,
    continuation: Annotated[
        str, typer.Option(help="The catalog system whose policy continues (--first-only).")
    ] = "ctl_fixed",
    min_spread: Annotated[
        float | None,
        typer.Option(help="Only points whose branch returns span more than this (--labels)."),
    ] = None,
    labels: Annotated[
        Path, typer.Option(help="Branch points (`labels` .points.jsonl) for --min-spread.")
    ] = DATA_DERIVED / "controller" / "sft" / "train.points.jsonl",
    per_task: Annotated[int, typer.Option(help="Points per task at most (0: no limit).")] = 1,
    workers: Workers = None,
    dry_run: Annotated[bool, typer.Option(help="Print the group runs only.")] = False,
) -> None:
    """Sample `size` trajectories from each of `groups` decision points of a train run with the
    served adapter (docs/06, bounded RL): one run per batches value,
    runs/<prefix>-b<batches>, members g0, g1, ...; resumable. Needs `jev controller serve
    --adapter ADAPTER` running."""
    from jevscholar.controller.branches import continuation_policy
    from jevscholar.controller.groups import experiments, points, spreads
    from jevscholar.controller.serve import adapter_sha
    from jevscholar.runner import run_experiment

    path = _run_dir(source)
    sha = adapter_sha(adapter)
    spread = None if min_spread is None else spreads(labels, path.name)
    found = points(
        path, system, groups, rl_round, spread=spread, min_spread=min_spread or 0.0,
        per_task=per_task or None,
    )  # fmt: skip
    then = continuation_policy(continuation) if first_only else None
    prefix = prefix or f"groups-{path.name}-r{rl_round}"
    exps = experiments(
        path, system, found, adapter=sha, run_usd=run_usd, size=size, prefix=prefix, then=then
    )
    mode = "first decision only, then the fixed rule" if first_only else "every decision"
    typer.echo(f"{len(found)} groups of {size} ({len(found) * size} trajectories), adapter {sha}")
    typer.echo(f"d1 samples {mode}")
    for run_id, exp in exps.items():
        typer.echo(f"  runs/{run_id}: {list(exp.benchmarks['deepscholar'].ids or ())}")
    if dry_run:
        return
    for run_id, exp in exps.items():
        if stopped := run_experiment(exp, RUNS / run_id, workers=workers):
            typer.echo(f"run {run_id}: stopped early ({stopped})")
            raise typer.Exit(1)
        typer.echo(f"run {run_id}: complete")


@controller_app.command("rl-data")
def controller_rl_data(
    runs: Annotated[list[str], typer.Argument(metavar="RUN...", help="Group runs of one round.")],
    out: Annotated[Path, typer.Option(help="RLExample JSONL (and a .groups.jsonl beside it).")],
) -> None:
    """RL examples from group runs: each trajectory's retrieval return, its group-relative
    advantage, and every decision the policy made in it."""
    from jevscholar.controller.d1 import write_rl_examples
    from jevscholar.controller.groups import advantages, examples, outcomes
    from jevscholar.util import write_jsonl

    groups, skipped = advantages(outcomes(_run_dir(r) for r in runs))
    found = examples(groups)
    write_rl_examples(out, found)
    write_jsonl(out.with_suffix(".groups.jsonl"), groups)
    typer.echo(f"{out}: {len(found)} decisions from {len(groups)} groups; skipped {skipped}")


@controller_app.command("rl-update")
def controller_rl_update(
    data: Annotated[Path, typer.Argument(help="RLExample JSONL (`jev controller rl-data`).")],
    adapter: Annotated[Path, typer.Option(help="The adapter that sampled the data.")],
    reference: Annotated[Path, typer.Option(help="The frozen KL reference (the SFT adapter).")],
    out: Annotated[Path, typer.Option(help="Output directory (adapter/, metrics.json).")],
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 5e-5,
    steps: Annotated[int, typer.Option(min=1, help="Optimiser steps in the epoch.")] = 4,
) -> None:
    """One bounded RL update of d1 (clipped ratio, KL to the reference); reverted (nothing
    saved) if the mean KL exceeds 0.05 nats per decision."""
    from jevscholar.controller.d1 import RLConfig, rl_update

    cfg = RLConfig(data=data, adapter=adapter, reference=reference, out=out, lr=lr, steps=steps)
    rl_update(cfg, log=typer.echo)


@controller_app.command("regret")
def controller_regret(
    data: Annotated[Path, typer.Argument(help="Val labels (`jev controller labels`, dev.jsonl).")],
    adapter: Annotated[
        list[Path] | None, typer.Option(help="An adapter to score (repeatable).")
    ] = None,
    base: Annotated[bool, typer.Option(help="Also score d1 without an adapter.")] = False,
    min_spread: Annotated[
        float, typer.Option(help="`signal`: points whose option returns span more than this.")
    ] = 0.01,
) -> None:
    """Decision-level comparison on branched val points: regret of each adapter's choice
    against the best branch, expected regret, best-option rate and soft-target CE (GPU, no
    executor calls)."""
    from jevscholar.controller.compare import decision_scores, with_names
    from jevscholar.controller.d1 import D1, read_examples
    from jevscholar.controller.serve import adapter_sha
    from jevscholar.util import read_jsonl

    examples = read_examples(data)
    points = with_names(read_jsonl(data.with_suffix(".points.jsonl")), examples)
    chosen: list[Path | None] = ([None] if base else []) + list(adapter or [])
    if not chosen:
        raise typer.BadParameter("give at least one --adapter, or --base")
    for path in chosen:
        d1 = D1.load(adapter=path)
        probs = [d1.probabilities([(e.state, e.question)], render=e.render)[0] for e in examples]
        scores = decision_scores(probs, points, min_spread=min_spread) | {
            "adapter": str(path) if path else None,
            "sha256": adapter_sha(path),
        }
        typer.echo(json.dumps(scores))
        del d1


@controller_app.command("episodes")
def controller_episodes(
    source: Run,
    name: Annotated[str, typer.Option(help="System name and run id suffix, e.g. d1_sft.")],
    adapter: Annotated[
        Path | None, typer.Option(help="The served adapter (default: none, the base d1).")
    ] = None,
    system: Annotated[str, typer.Option(help="The source run's controlled system.")] = "ctl_fixed",
    run_usd: Annotated[float, typer.Option(help="Billed cap of the run.")] = 0.6,
    prefix: Annotated[str, typer.Option(help="Run id prefix.")] = "compare",
    workers: Workers = None,
    dry_run: Annotated[bool, typer.Option(help="Print the run only.")] = False,
) -> None:
    """Run every task of a val (or train) run from the start with d1 + `adapter` choosing, as
    system NAME in runs/<prefix>-<source>-<name>; resumable. Needs `jev controller serve
    --adapter ADAPTER` running."""
    from jevscholar.controller.compare import episode_experiment
    from jevscholar.controller.serve import adapter_sha
    from jevscholar.runner import run_experiment

    path = _run_dir(source)
    sha = adapter_sha(adapter)
    exp = episode_experiment(path, system, name, adapter=sha, run_usd=run_usd)
    run_id = f"{prefix}-{path.name}-{name}"
    typer.echo(f"runs/{run_id}: {name} = d1 with adapter {sha}")
    if dry_run:
        return
    if stopped := run_experiment(exp, RUNS / run_id, workers=workers):
        typer.echo(f"run {run_id}: stopped early ({stopped})")
        raise typer.Exit(1)
    typer.echo(f"run {run_id}: complete")


@controller_app.command("compare")
def controller_compare(
    cells: Annotated[
        list[str], typer.Argument(metavar="RUN:SYSTEM...", help="The first is the baseline.")
    ],
) -> None:
    """Per-task retrieval returns of controlled systems' episodes, paired against the first."""
    from jevscholar.controller.compare import compare, episode_returns, table

    systems = {}
    for cell in cells:
        run, _, system = cell.rpartition(":")
        if not run:
            raise typer.BadParameter(f"{cell}: give RUN:SYSTEM")
        systems[f"{Path(run).name}:{system}"] = episode_returns(_run_dir(run), system)
    typer.echo(table(compare(systems)))


@controller_app.command("frontier")
def controller_frontier(
    run: Annotated[str, typer.Argument(help="A reported run with d1 and fixed-rule systems.")],
) -> None:
    """The d1 controllers against the fixed rule's cost-quality curve, at matched cost."""
    from jevscholar.controller.frontier import frontier_report

    typer.echo(frontier_report(_run_dir(run)))


@controller_app.command("headroom")
def controller_headroom(
    run: Annotated[str, typer.Argument(help="A reported run with fixed-rule budgets.")],
    metric: Annotated[
        list[str] | None, typer.Option(help="Metrics (default: utility, nugget_coverage).")
    ] = None,
) -> None:
    """What a per-task stopping oracle could gain over the fixed rule at matched cost."""
    from jevscholar.controller.headroom import headroom_report

    typer.echo(headroom_report(_run_dir(run), metric or ["utility", "nugget_coverage"]))


@controller_app.command("screen-data")
def controller_screen_data(
    out: Annotated[Path, typer.Argument(help="Directory for train.jsonl, dev.jsonl and metas.")],
) -> None:
    """d1 relevance-screening examples from every run's DeepSeek screening verdicts."""
    from jevscholar.controller.screen import screen_data

    typer.echo(json.dumps(screen_data(out), indent=2))


@controller_app.command("screen-eval")
def controller_screen_eval(
    data: Annotated[Path, typer.Argument(help="screen-data examples (train.jsonl or dev.jsonl).")],
    out: Annotated[Path, typer.Option(help="Per-example P(relevant), JSONL.")],
    adapter: Annotated[Path | None, typer.Option(help="A trained adapter (default: none).")] = None,
) -> None:
    """d1 as the relevance screener against DeepSeek and the real citations (.venv-train)."""
    from jevscholar.controller.screen import score

    typer.echo(score(data, out, adapter))


@controller_app.command("sanity-data")
def controller_sanity_data(
    out: Annotated[Path, typer.Argument(help="Directory for train.jsonl and dev.jsonl.")],
    train: int = 2000,
    dev: int = 400,
    seed: int = 0,
) -> None:
    """The training-loop check: a paper's primary arXiv category among four."""
    from jevscholar.controller.d1 import write_examples
    from jevscholar.controller.data import arxiv_categories

    examples = arxiv_categories(train + dev, seed=seed)
    write_examples(out / "train.jsonl", examples[:train])
    write_examples(out / "dev.jsonl", examples[train:])
    typer.echo(f"{out}: {train} train, {dev} dev")


@controller_app.command("train")
def controller_train(
    recipe: Annotated[str, typer.Argument(help="A recipe of configs/training.yaml.")],
) -> None:
    """LoRA-train d1-3B; writes the best adapter, config.json and metrics.json under `out`."""
    from jevscholar.controller.d1 import load_recipe, train

    train(load_recipe(CONFIGS / "training.yaml", recipe), log=typer.echo)


@controller_app.command("eval")
def controller_eval(
    data: Annotated[Path, typer.Argument(help="Example JSONL.")],
    adapter: Annotated[Path | None, typer.Option(help="A trained adapter (default: none).")] = None,
) -> None:
    """Accuracy, NLL, Brier and ECE of d1-3B (optionally with an adapter) on `data`."""
    from jevscholar.controller.d1 import D1, read_examples

    typer.echo(json.dumps(D1.load(adapter=adapter).evaluate(read_examples(data)), indent=2))

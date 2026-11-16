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
from typing import Annotated, Any

import typer

from jevscholar.paths import CONFIGS, RUNS

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
    rerun_infra: Annotated[
        bool, typer.Option(help="Also rerun tasks that failed on an infrastructure error.")
    ] = False,
) -> None:
    """Run (or resume) an experiment under runs/<run-id>/."""
    import yaml

    from jevscholar.config import experiment_path, load_experiment, load_models
    from jevscholar.runner import plan, run_experiment

    exp = load_experiment(experiment_path(experiment), variant, set_ or [])
    run_id = run_id or f"{exp.name.replace('/', '-')}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    if dry_run:
        cells, tasks, batches = plan(exp, RUNS / run_id, load_models(), only, rerun_infra)
        typer.echo(yaml.safe_dump(exp.model_dump(mode="json"), sort_keys=False))
        for bench, by_system in cells.items():
            for system, cell in by_system.items():
                typer.echo(f"{bench}/{system}: cell {cell['hash']}, {len(tasks[bench])} tasks")
        typer.echo(f"{sum(map(len, batches))} tasks pending in runs/{run_id}")
        return
    stopped = run_experiment(
        exp, RUNS / run_id, only=only, workers=workers, rerun_infra=rerun_infra
    )
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


@app.command()
def claims(
    test: Annotated[str, typer.Argument(help="The run of the first 90 test tasks (every arm).")],
    ext: Annotated[str, typer.Argument(help="The run of the other 90 (DS-base and the arm).")],
    arm: Annotated[str, typer.Option(help="The main claim's d1.")] = "ctl_d1",
    judges: Judges = "deepseek",
) -> None:
    """The test's registered claims (docs/06): the main claim on all 180 tasks, and the RL claim
    on the first 90."""
    import pandas as pd

    from jevscholar.analysis import main_claim, rl_claim, scored

    first = scored(_run_dir(test), judges)
    both = pd.concat([first, scored(_run_dir(ext), judges)], ignore_index=True)
    out = {"main": main_claim(both, arm), "rl": rl_claim(first)}
    typer.echo(json.dumps(out, indent=2, default=float))


@app.command()
def bestof(
    run_: Run,
    system: Annotated[
        str, typer.Argument(metavar="SYSTEM", help="Its system; SYSTEM_s1, ... are its samples.")
    ],
    baseline: Annotated[
        list[str], typer.Option("--baseline", help="The baseline's runs (each task in one).")
    ],
    baseline_system: Annotated[
        str, typer.Option(help="The baseline, averaged over its samples.")
    ] = "deepscholar_base_intro",
    margin: Annotated[
        float, typer.Option(help="Non-inferiority margin of nugget coverage, reference recall.")
    ] = 0.02,
    judges: Judges = "deepseek",
) -> None:
    """SYSTEM's best-of-n (writer_pick, every write charged) against the baseline's per-task
    sample mean: paired differences per metric and the claim (cost lower, nugget coverage and
    reference recall non-inferior at the margin)."""
    import pandas as pd

    from jevscholar.analysis import best_of_n, writer_samples

    samples = writer_samples(_run_dir(run_), system, judges)
    ref = [writer_samples(_run_dir(r), baseline_system, judges) for r in baseline]
    out = {"system": system, "baseline": baseline_system}
    out |= best_of_n(samples, pd.concat(ref, ignore_index=True), margin)
    typer.echo(json.dumps(out, indent=2, default=float))


# --------------------------------------------------------------------------- judge checks
@check_app.command("version")
def check_version(
    alias: Annotated[str, typer.Option(help="A model alias whose model is pinned.")] = (
        "ds-judge-chat"
    ),
) -> None:
    """One uncached call of a few tokens through the gateway (run `version-probe`): fails unless
    the reply comes from the model version pinned in models.yaml (`versions`)."""
    from jevscholar.config import Budget, load_models
    from jevscholar.gateway import Gateway, load_secrets, probe_version

    models = load_models()
    model = models.aliases[alias].model
    if model not in models.versions:
        raise typer.BadParameter(f"{model} has no pinned version in models.yaml")
    budget = Budget(run_usd=0.0, judge_run_usd=0.01)
    gw = Gateway(models, RUNS / "version-probe", budget, {}, secrets=load_secrets())
    try:
        version = probe_version(gw.start(), alias)
    except RuntimeError as e:
        typer.echo(f"{model}: {e}")
        raise typer.Exit(1) from None
    finally:
        gw.stop()
    typer.echo(f"{model}: version {version}, the pinned one")


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
    reference: Annotated[
        str | None,
        typer.Option(
            help="Run whose DeepSeek verdicts are compared (default: RUN's own), e.g. "
            "RUN's subset re-judged by `--judges deepseek` after a model update."
        ),
    ] = None,
    metrics: Annotated[
        list[str] | None,
        typer.Option("--metrics", help="Judge only these metrics (where judged separately)."),
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
        source,
        run_dir,
        systems=tuple(system or ()),
        judgeset=judges,
        tasks=plan,
        metrics=tuple(metrics or ()),
        workers=workers,
    )
    summarize(_run_dir(reference) if reference else source, run_dir, judges, stopped=stopped)
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


@controller_app.command("branches")
def controller_branches(
    source: Run,
    system: Annotated[str, typer.Option(help="The source run's controlled system.")] = "ctl_d1",
    continuation: Annotated[
        str, typer.Option(help="The catalog system whose policy continues each branch.")
    ] = "ctl_d1",
    tasks: Annotated[int | None, typer.Option(help="Only the run's first N tasks.")] = None,
    stratified: Annotated[
        bool, typer.Option(help="One early and one late decision point a task.")
    ] = False,
    run_usd: Annotated[float, typer.Option(help="Billed cap of each branch run.")] = 0.5,
    workers: Workers = None,
    dry_run: Annotated[bool, typer.Option(help="Print the branch runs only.")] = False,
) -> None:
    """Branch every legal candidate of a train or val run's decision points (docs/06), each
    continued to the writer's documents without writing: one run per (batches, kind),
    runs/branches-<RUN>-b<batches>-<kind>, resumable. A d1 continuation needs
    `jev controller serve`."""
    from jevscholar.controller.branches import continuation_policy, experiments, forks, provenance
    from jevscholar.runner import run_experiment

    path = _run_dir(source)
    provenance(path, system)
    found = forks(path, system, tasks=tasks, stratified=stratified)
    policy = continuation_policy(continuation)
    exps = experiments(path, system, found, policy=policy, run_usd=run_usd)
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
    source: Run,
    out: Annotated[Path, typer.Option(help="Example JSONL (and a .points.jsonl beside it).")],
    system: Annotated[str, typer.Option(help="The source run's controlled system.")] = "ctl_d1",
    tasks: Annotated[int | None, typer.Option(help="Only the run's first N tasks.")] = None,
    stratified: Annotated[
        bool, typer.Option(help="One early and one late decision point a task.")
    ] = False,
) -> None:
    """Graded SFT targets from the branch runs of a source run's decision points, given the
    options `jev controller branches` had: every planned point is labelled or left out."""
    from jevscholar.controller.branches import examples, forks, outcomes, run_ids, run_name
    from jevscholar.controller.d1 import write_examples
    from jevscholar.util import write_jsonl

    path = _run_dir(source)
    found = forks(path, system, tasks=tasks, stratified=stratified)
    ids = sorted(set(run_ids(path, found).values()))
    runs = [RUNS / i for i in ids if (RUNS / i / "manifest.json").exists()]
    planned = [(run_name(path), system, f.key, f.batches) for f in found]
    labelled, points, left_out = examples(outcomes(runs), planned=planned)
    write_examples(out, labelled)
    write_jsonl(out.with_suffix(".points.jsonl"), points)
    off = sum(not p["decoded_best"] for p in points)
    typer.echo(f"{out}: {len(labelled)} of {len(planned)} planned points labelled")
    typer.echo(f"branch runs: {len(runs)} of {len(ids)} exist")
    for point in left_out:
        typer.echo(f"  left out (a branch failed or missing): {point[2]} after {point[3]} batches")
    typer.echo(f"d1's decoder on the targets misses the best option at {off} of {len(points)}")


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


def _recipe(name: str, kind: type) -> Any:
    from jevscholar.controller.d1 import load_recipe

    cfg = load_recipe(CONFIGS / "training.yaml", name)
    if not isinstance(cfg, kind):
        raise typer.BadParameter(f"{name} is not a {kind.__name__} recipe")
    return cfg


@controller_app.command("train")
def controller_train(
    recipe: Annotated[str, typer.Argument(help="A recipe of configs/training.yaml.")],
) -> None:
    """LoRA-train d1-3B; keeps every evaluated step's adapter and writes the selected one,
    config.json and metrics.json under `out`."""
    from jevscholar.controller.d1 import TrainConfig, train

    train(_recipe(recipe, TrainConfig), log=typer.echo)


@controller_app.command("rl-points")
def controller_rl_points(
    source: Run,
    system: Annotated[str, typer.Option(help="The source run's controlled system.")] = "ctl_d1",
    n: Annotated[int, typer.Option(help="Points (tasks).")] = 8,
) -> None:
    """The registered RL points of a train run (docs/06), as `points:` for an RL recipe."""
    from jevscholar.controller.rl import points

    for fork in points(_run_dir(source), system, n):
        typer.echo(f"  - [{fork.key}, {fork.batches}]")


@controller_app.command("rollouts")
def controller_rollouts(
    recipe: Annotated[str, typer.Argument(help="An RL recipe of configs/training.yaml.")],
    adapter: Annotated[str, typer.Option(help="The served adapter's sha256.")],
    url: Annotated[
        str, typer.Option(help="Where the adapter is served.")
    ] = "http://127.0.0.1:9941",
    workers: Workers = None,
    dry_run: Annotated[bool, typer.Option(help="Print the rollout runs only.")] = False,
) -> None:
    """Sample the recipe's groups: its points restored, each continued by `samples` members of
    the served adapter at temperature 1, to the writer's documents: runs/<prefix>-b<batches>."""
    from jevscholar.artifacts import read_manifest, run_tasks
    from jevscholar.controller.branches import BENCH, Fork
    from jevscholar.controller.d1 import RLConfig
    from jevscholar.controller.rl import experiments
    from jevscholar.runner import run_experiment

    cfg = _recipe(recipe, RLConfig)
    path = _run_dir(cfg.source)
    tasks = run_tasks(read_manifest(path), BENCH)
    found = [Fork(tasks[k].id, k, b, ()) for k, b in cfg.points]
    policy = {"kind": "d1", "sample": True, "adapter": adapter, "url": url}
    exps = experiments(
        path, cfg.system, found, policy=policy, samples=cfg.samples, run_usd=cfg.run_usd,
        prefix=cfg.prefix,
    )  # fmt: skip
    for run_id, exp in exps.items():
        typer.echo(f"  runs/{run_id}: {len(exp.benchmarks['deepscholar'].ids or ())} points")
    if dry_run:
        return
    for run_id, exp in exps.items():
        if stopped := run_experiment(exp, RUNS / run_id, workers=workers):
            typer.echo(f"run {run_id}: stopped early ({stopped})")
            raise typer.Exit(1)
        typer.echo(f"run {run_id}: complete")


@controller_app.command("rl-data")
def controller_rl_data(
    recipe: Annotated[str, typer.Argument(help="An RL recipe of configs/training.yaml.")],
) -> None:
    """The recipe's sampled decisions with their groups' advantages (`steps`), and each group's
    record beside them (.groups.jsonl)."""
    from jevscholar.controller.d1 import RLConfig
    from jevscholar.controller.rl import groups, write_steps
    from jevscholar.util import write_jsonl

    cfg = _recipe(recipe, RLConfig)
    wanted = sorted({b for _, b in cfg.points})
    runs = [RUNS / f"{cfg.prefix}-b{b}" for b in wanted]
    runs = [r for r in runs if (r / "manifest.json").exists()]
    steps, records = groups(
        runs, source=_run_dir(cfg.source), system=cfg.system, planned=cfg.points,
        samples=cfg.samples,
    )  # fmt: skip
    cfg.steps.parent.mkdir(parents=True, exist_ok=True)
    write_steps(cfg.steps, steps)
    write_jsonl(cfg.steps.with_suffix(".groups.jsonl"), records)
    usable = sum(r["skipped"] is None for r in records)
    trained = sum(s.usable for s in steps)
    typer.echo(f"{cfg.steps}: {len(steps)} decisions ({trained} in usable groups); {usable} of "
               f"{len(records)} groups usable")  # fmt: skip
    for r in records:
        if r["skipped"]:
            typer.echo(f"  skipped {r['group']}: {r['skipped']}")


@controller_app.command("gate")
def controller_gate(
    run: Run,
    regret: Annotated[
        list[str], typer.Option(help="SYSTEM=FILE: `jev controller eval --points` output.")
    ],
) -> None:
    """The registered val gate: which d1 carries the main claim (docs/06)."""
    from jevscholar.controller.rl import gate

    values = {}
    for spec in regret:
        system, _, path = spec.partition("=")
        values[system] = json.loads(Path(path).read_text())["regret"]
    typer.echo(json.dumps(gate(_run_dir(run), values), indent=2))


@controller_app.command("rl")
def controller_rl(
    recipe: Annotated[str, typer.Argument(help="An RL recipe of configs/training.yaml.")],
) -> None:
    """One registered RL update on the recipe's sampled decisions; stop the d1 server first."""
    from jevscholar.controller.d1 import RLConfig, rl_update

    result = rl_update(_recipe(recipe, RLConfig), log=typer.echo)
    typer.echo(f"kept step {result['kept_step']} (update: {result['update']})")


@controller_app.command("eval")
def controller_eval(
    data: Annotated[Path, typer.Argument(help="Example JSONL.")],
    adapter: Annotated[Path | None, typer.Option(help="A trained adapter (default: none).")] = None,
    points: Annotated[
        Path | None, typer.Option(help="The examples' labelled points (`labels`' .points.jsonl).")
    ] = None,
) -> None:
    """Accuracy, NLL, Brier, ECE and cross-entropy of d1-3B (optionally with an adapter) on
    `data`, read row by row as served; with `--points`, also its regret there."""
    from jevscholar.controller.d1 import D1, read_examples
    from jevscholar.util import read_jsonl

    labelled = None if points is None else read_jsonl(points)
    metrics = D1.load(adapter=adapter).evaluate(read_examples(data), labelled)
    typer.echo(json.dumps(metrics, indent=2))

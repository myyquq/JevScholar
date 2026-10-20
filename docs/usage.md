# Usage

Setup, every command, how experiments are configured, the repository and run layouts, the
reproducibility rules, and which run backs which result.

## Setup

```bash
uv sync                                                    # Python 3.12, locked dependencies
uv run python -m nltk.downloader -d .cache/nltk_data punkt_tab
mkdir -p ~/.config/jevscholar && touch ~/.config/jevscholar/secrets.env
chmod 600 ~/.config/jevscholar/secrets.env
uv run jev data fetch && uv run jev data prepare && uv run jev data arxiv-index
```

`~/.config/jevscholar/secrets.env` (dotenv syntax, outside the repository):

```
DEEPSEEK_API_KEY=...    # executor and judges
OPENAI_API_KEY=...      # optional: official judges, the GPT-4.1 comparison
TAVILY_API_KEY=...      # optional: web search (DRB2)
OPENALEX_API_KEY=...    # optional: DeepScholar document importance
```

Only the gateway reads the LLM keys. Workers run with a scrubbed environment, so any path that
bypasses the gateway fails loudly; the Tavily and OpenAlex keys are passed to the workers that
call those APIs.

**The d1 environment.** d1-3B needs transformers 5, which LOTUS does not allow, so it has its own
environment, `.venv-train`. `scripts/train_env.sh` runs any command there.

## Commands

```bash
# data
uv run jev data fetch [--pin]          # download pinned sources into data/raw/, verify sha256
uv run jev data prepare                # data/derived/ and the committed data/splits/
uv run jev data arxiv-index            # the local BM25 arXiv index

# runs: RUN is a run id under runs/, or a path
uv run jev run EXPERIMENT --variant V [--set a.b=value] [--only BENCH] [--run-id RUN]
uv run jev run EXPERIMENT --variant V --dry-run      # print the resolved experiment and cells
uv run jev export RUN                  # official layouts (failed/missing -> empty) + audit.json
uv run jev eval RUN [--judges deepseek|official] [--metrics M ...] [--redo M ...]
uv run jev report RUN [--baseline deepscholar_base_intro]   # report.md, analysis/long.csv
                                       # (the baseline must be a system of the run)

# checks of the judges
uv run jev check agree RUN [--system S ...] [--tasks BENCH=N]   # official judges on a subset
uv run jev check calibrate drb2|deepscholar                     # fixed published reports
uv run jev check calibrate deepscholar-controls --source RUN/SYSTEM
uv run jev check review-pack RUN [--system S ...]               # a blind spot-check pack
uv run jev check review-score PACK

# the decision model (serve/train/eval run in .venv-train)
scripts/train_env.sh jev controller serve [--adapter DIR] [--port 9940]   # a d1 policy names
                                       # the served adapter by the sha256 of its safetensors file
uv run jev controller trajectories RUN ... [--out FILE]   # labelled decisions of train runs
uv run jev controller sanity-data DIR                     # the training-loop check's data
scripts/train_env.sh jev controller train RECIPE          # a recipe of configs/training.yaml
scripts/train_env.sh jev controller eval DATA.jsonl [--adapter DIR]

uv run pytest                          # offline tests; -m network, -m gpu for live checks
```

Typical cycle: `jev run`, then `jev eval` (which re-exports first), then `jev report`. Runs and
judging stop at their budgets (docs/04) and resume where they stopped.

## Configuration

Six files:

| File | What it holds |
|---|---|
| `configs/models.yaml` | upstreams, model aliases with list prices, tool prices, judge sets, the campaign cap |
| `configs/systems.yaml` | the system catalog: one definition per kept system version |
| `configs/training.yaml` | d1 training recipes |
| `configs/experiments/baselines.yaml` | DS-base and the simple baselines on the dev splits (variants `june`, `nov`, `gpt41`) |
| `configs/experiments/controller.yaml` | the controller on train and val (variants `smoke`, `rollouts`, `val`) |
| `configs/experiments/evaluate.yaml` | the main comparison at B0 (variants `dev`, `dev2`, `test`, `replay`) |

**An experiment** names its systems per benchmark, with the shared settings at the top:

```yaml
budget: {run_usd: 4.0, judge_run_usd: 2.0}       # per-task cap: each benchmark's B0
benchmarks:
  deepscholar:
    samples: 2                                    # two writer samples per task
    systems: [deepscholar_base_intro, simple_rag_k20, ctl_fixed, ctl_d1]
variants:
  dev:  {benchmarks: {deepscholar: {split: dev, limit: 20}}}
  test: {benchmarks: {deepscholar: {split: test, limit: 90}}}
```

It is resolved in this order (`config.load_experiment`):
1. **Variant.** The base settings deep-merged with the chosen variant: mappings merge, lists and
   values replace.
2. **Overrides.** `--set a.b.c=value` (the value parsed as YAML).
3. **Systems.** Each listed name is defined in the experiment's `systems:`, in the catalog, or
   is a bare system class at its defaults (`simple_rag`, `closed_book`, `deepscholar_base`;
   `controlled` needs a `policy` and `fixed_evidence` its `evidence`, so they are always
   defined). A definition may `extends` another; an experiment may not redefine a catalog name.
4. **Samples.** `samples: N` runs each listed system as `<name>`, `<name>_s1`, …: independent
   writer samples from the same evidence (Simple RAG, DS-base and the controlled loop take a
   `sample`). A generated name may not collide with another system's.

Defaults are the shared settings of every comparison: executor `deepseek-flash`, the local arXiv
index with 10 hits per query, off-peak dispatch, 4 workers, seed 0, and each benchmark's own
per-task cap B0. A system definition may set its own cap, `task_usd` (DS-base: $2, a cap it
never reaches); it is part of the cell, not a parameter. Each system validates its parameters:
an unknown knob is an error. The run
manifest records the fully resolved experiment, so reading a run never needs the configs.

**A system** in the catalog:

```yaml
deepscholar_base_intro:            # repaired, intro-only: the DeepScholar baseline
  extends: deepscholar_base_citefix
  config: {categorize_references: false, generate_insights: false}
deepscholar_base_citefix:          # P6 + P10, at its natural cost (a cap it never reaches)
  use: deepscholar_base
  repair_citations: true
  task_usd: 2.0
ctl_d1:                            # the controlled loop with d1 choosing
  use: controlled
  policy: {kind: d1}
```

## Layout

```
configs/                      models, the system catalog, training recipes, experiments
data/sources.yaml             every external file: URL at a pinned revision + sha256
data/splits/*.json            committed, deterministic task orders
third_party/                  pristine vendored upstream code (patched only at run time)
src/jevscholar/
  paths.py config.py types.py util.py      locations, configuration, shared types, helpers
  gateway.py ledger.py                     the metering LLM gateway and its cost ledger
  runner.py judging.py                     running systems; judging their exports
  artifacts.py analysis.py report.py rewards.py   run readers; scoring; report.md; U and r
  data.py render.py                        data fetch/prepare; citations and export rendering
  search/                                  arXiv (local and API), Tavily, OpenAlex; access policy
  benchmarks/                              DeepScholar-Bench, DRB2, ResearchQA and their scorers
  systems/                                 simple_rag, closed_book, deepscholar_base, controlled,
                                           fixed_evidence
  controller/                              state and rules, d1's prompt, policies, d1 readout and
                                           training, its server, training data
  checks/                                  judge agreement, calibration, blind review packs
runs/<run_id>/                manifest.json, ledger.jsonl, <bench>/<system>/{tasks,export,eval}/,
                              analysis/, report.md
```

A task folder holds `result.json`, and for DS-base its native outputs (`native/`), for the
controlled loop its `decisions.jsonl`, `rows.jsonl`, `pages.jsonl` and `state.json`.

## Reproducibility rules

- **Pins.** Data by revision and sha256, vendored code by commit, dependencies by `uv.lock`. Each
  session appends its git sha, dirty flag (with the diff saved), and the hashes of the lockfile,
  `models.yaml`, `systems.yaml` and `third_party/` to the run's manifest.
- **Resume.** A task is done once its `result.json` exists; a degraded result is re-run once. A
  result written under a different cell identity is refused: start a new run id.
- **Replay.** Executor calls are cached by (benchmark, task, stage, request, occurrence), across
  runs and systems, so a rerun or a trajectory-fixed ablation replays at $0 billed while keeping
  its recorded list cost. Judges are never cached. `replay_only: true` refuses any executor call
  the cache cannot serve: `jev run evaluate --variant replay` regenerates the baselines'
  dev reports byte for byte at $0, a check that the code still produces the baselines.
- **Judging.** Each eval file records a fingerprint of the exported report, the judges' routing
  and the gold data; if any changed, the file is discarded and the task judged afresh. Each
  separately judged metric records its protocol version.
- **Failures stay in the denominator** (docs/04), and scores from the DeepSeek judges are labelled
  "DeepSeek-judged (non-official)".

## Run index

Runs live in `runs/` (not committed). Each manifest holds the run's resolved experiment and the
git sha of every session; `jev report RUN --baseline SYSTEM` regenerates its tables from the
saved results and eval files. Earlier runs were made by earlier versions of the code (their sha
is in the manifest): their results are read, not re-run, and their controller logs are in an
older format that `jev controller trajectories` does not read.

| Run | Backs |
|---|---|
| `phase1-dev` | DS-base versions, Simple RAG (K = 10) and closed-book on June dev, ResearchQA dev 40, DRB2 dev 12; the P6/P10 repairs (docs/03) |
| `phase1-extras` | Simple RAG's K; ResearchQA's raw query; the Simple RAG K = 20 reports the replay check compares |
| `phase1-noise` | the seed-1 noise floor |
| `repro-nov` | the paper protocol and Simple RAG on Nov dev2 (docs/03) |
| `repro-gpt41` | GPT-4.1 through the harness (docs/02) |
| `judge-agreement` | official-judge agreement (docs/02) |
| `drb2-calibration`, `deepscholar-calibration`, `deepscholar-controls` | judge calibration and evaluator controls (docs/02, docs/03) |
| `phase1-dev/review_pack_v2` | the blind spot-check labels |
| `phase2-baselines` | DS-base intro-only on June dev, its byte-identical replay and the cap B0 |
| `phase2-draft` | DS-base's cost breakdown; the controlled loop's early version and the no-draft diagnostic |
| `phase2-audit-s0`, `-s1`, `-s2` | the fixed-evidence writer, three seeds (docs/03) |
| `phase2-headroom`, `phase2-smoke*` | the loop's first versions |
| `p-smoke`, `p-bank`, `p-bank-keyword`, `p-bank-d1` | the decision bank's behaviour-policy runs (docs/05) |
| `p-r1lite`, `p-plan` | the revision and planning measurements |
| `p-onpolicy`, `p-prompted`, `p-val` | choosing d1's prompt and the prompted executor's |
| `p-dev`, `p-dev2` | earlier controller comparisons, with planning and DS-base under the $0.27 cap |
| `ctl-val` | the controllers on val; the fixed rule's batch count (docs/05) |
| `eval-dev`, `eval-dev2` | the controller comparisons on June dev and Nov dev2 (docs/05) |
| `controller/sanity`, `controller/sweep-1` | the LoRA training-loop check and the offline prompt search |

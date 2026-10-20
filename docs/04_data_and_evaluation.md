# 4. Data, evaluation flow and cost

How tasks are built and split, how a task's data travels from the dataset to a score, how scores
are aggregated and compared, and how cost is counted and capped.

## Terms

| Term | Meaning |
|---|---|
| DS-base | DeepScholar-base. The baseline is the intro-only version with the merge and citation repairs (docs/03) |
| Simple RAG | the course's simple baseline: retrieve 20 abstracts, write in one pass |
| executor | the LLM that proposes queries, screens papers and writes: DeepSeek-V4.1-Flash, non-thinking |
| d1 | LiquidAI/d1-3B. It only chooses: given a state and a few candidate actions, one forward pass gives each candidate's probability; it writes no text |
| screening | the executor judging one retrieved paper's relevance to the task (DS-base's chain-of-thought `sem_filter`), one call per paper |
| important citations, gold | the references the dataset marks as important for the target paper; used only by evaluation and training rewards, never seen by a system |
| N | nugget coverage: the share of the reference section's key points (nuggets) the report covers |
| P | cite_p, the official citation precision (per sentence, docs/03) |
| S | claim support, the repaired sentence-level support rate (docs/03) |
| R_c | reference recall: the share of the important citations the report cites, by exact arXiv id |
| R_e | evidence recall: the share of the important citations among the documents the writer receives |
| B0 | the controllers' per-task cap at list price, and U's cost scale: $0.27 on DeepScholar-Bench, $0.34 on DRB2 |
| list and billed cost | list cost prices every call at the frozen price book, cache replays and nominal tool prices included, and compares systems; billed cost is what the project pays |

## Datasets and splits

| Benchmark | Split | Tasks | Built from | Used for |
|---|---|---|---|---|
| DeepScholar-Bench, June 2025 | `dev` | 20 | the first 20 of the June order | comparing systems |
| | `val` | 9 | the next 9 | selecting among a few frozen controller candidates |
| | `train` | 34 | the remaining 34 | the controller's prompt tuning, labels and training |
| DeepScholar-Bench, Nov 2025 | `dev2` | 20 | the first 20 of the Nov order | a distribution-shift check |
| | `test` | 180 | the remaining 180; the first 90 are the final test | held out |
| ResearchQA | `dev` | 703 | the `valid` file (the first 40 used) | the baseline record (docs/03) |
| | `test` | 776 | the official `test_mini` subset | not used |
| DRB2 | `dev` | 12 | 12 fixed ids, 6 zh and 6 en | judge calibration, baselines |
| | `test` | 120 | the other tasks | the transfer check (docs/06) |

**How the order is made** (`data.order`, `data.build_splits`). Within each stratum, tasks are
sorted by `sha256("jevscholar|<benchmark>|<task id>")`. Strata are then interleaved round-robin,
so every prefix of the order is balanced: DeepScholar by the paper's primary arXiv category (in
proportion to the category's size), ResearchQA by general domain, DRB2 by language. A
`limit: N` in an experiment takes the first N tasks of a split. The split lists are committed in
`data/splits/`; `jev data prepare` rebuilds them byte for byte.

**The final test** is `test[:90]`, from `2511.09143v2` to `2509.19586v1`. Its newline-joined ids
have the SHA-256 prefix `66bd8fa271d545da` (checked by `tests/test_benchmarks.py`).

**Important citations** ("gold") are the rows of a task's `important_citations.csv` with an arXiv
link, which is what the reference metrics score:

| Split | Gold rows | Tasks without gold | Tasks with a duplicated row | Tasks whose abstract is a blocked query |
|---|---|---|---|---|
| train | 388 | 0 | 2 | 1 |
| val | 131 | 0 | 0 | 0 |
| dev | 192 | 1 | 0 | 1 |
| dev2 | 103 | 1 | 2 | 1 |
| test[:90] | 482 | 7 | 3 | 3 |

Train, val and dev share a few foundational papers among their important citations (train and
val 16 distinct ids, train and dev 9, val and dev 6), never a target paper.

## What a task is

A task (`types.Task`) is a prompt plus an access policy, the same for every system.

- **DeepScholar-Bench.** The prompt is the official query template filled with the paper's
  publication date and abstract. The system writes the paper's related-work section.
- **DRB2.** The official research prompt; a long report.
- **ResearchQA.** The question with its date and an instruction to answer in about 250 words.

**The access policy** (`search/policy.py`) is enforced by the search client on every tool result,
for every system:
- DeepScholar: only papers published strictly before the target's publication day; the target
  itself is blocked by arXiv id and by title. A query that contains a blocked title is refused,
  so the title-plus-abstract query is impossible; the abstract alone is refused where it
  contains the title (the counts above).
- ResearchQA: sources dated before the question's date (a partial date counts from the first day
  of its period); undated web pages are kept, as most of the web has no machine-readable date.
- DRB2: the task's source article is blocked by URL, by title and by any arXiv id in its URLs.

At export, every cited source is checked again (`search.policy.source_allowed`: blocked URLs and
arXiv ids, and arXiv papers' dates) and dropped if disallowed (`export/audit.json`; no violations
so far). Titles and web dates are enforced at retrieval only. Gold data (nuggets, important citations, the
paper's own section, judge verdicts) never reaches a system: it is read only by evaluation and
by training labels.

## How data flows through a run

```text
configs/experiments/<name>.yaml --variant v --set k=v
   │  config.load_experiment: variants, extends, samples → one resolved Experiment
   ▼
jev run         runner.plan: tasks of each cell, cell identity, pending tasks
   │            runner.execute: a gateway (budgets, ledger, replay cache) and a worker pool
   │            worker: System.run(TaskContext) → Report (text with [n] citations + sources)
   ▼            runs/<id>/<bench>/<system>/tasks/<task>/result.json, ledger.jsonl, manifest.json
jev export      each benchmark's official layout: export/ (+ audit.json)
   ▼
jev eval        judging.eval_run: official evaluators with our judges, through the gateway
   ▼            eval/<judges>/<task>.json: metrics, per-metric status and version, judge items
jev report      analysis: one long table (task × metric, with cost); report.py: report.md
                analysis/long.csv, analysis/funnel.csv, tasks/<task>/trajectory.jsonl
```

- **Cell identity.** A (benchmark, system) cell is identified by a hash of the executor alias's
  settings and its upstream endpoint and dialect, the search settings, the seed, the system's
  resolved parameters, the split, the benchmark version and the per-task cap. A result written
  under another identity is refused on resume: such a change needs a new run id. The task limit,
  workers and run budgets are not part of it, so a run can be extended or given more budget.
- **Resume.** A task is done once its `result.json` exists. A `degraded` result (too many tool
  errors) is re-run once.
- **Judging identity.** Each eval file records a fingerprint of what it scored: the exported
  report, the judges' routing and the task's gold data. A file whose input changed is discarded
  and the task judged again. Each separately judged metric records its protocol version, so a
  changed metric is re-judged on its own; settled metrics are reused.
- **Provenance.** Every `jev run` and `jev eval` appends a session to the manifest: the git sha
  and dirty flag (a dirty tree's diff is saved next to the manifest), and the hashes of
  `uv.lock`, `configs/models.yaml`, `configs/systems.yaml` and `third_party/`.
- **Default metrics.** `jev eval` judges each benchmark's default metrics (on DeepScholar: nugget
  coverage, cite_p, claim coverage, claim support and the two reference metrics, which need no
  judge); others only on request (`--metrics`).

## Scoring and comparison

- **Failures count.** The primary mean of a metric is over every task of the split prefix; a
  failed or missing generation, or a judging that settled on no score (a report with no
  citations to check), scores 0. A metric that was not judged (not requested, a judge error) is
  no evidence either way: its task is left out and counted. The official-style mean over valid
  tasks only is reported next to it.
- **Undefined metrics** (reference recall on a task without important citations) leave the task
  out entirely, on every system.
- **Paired comparisons.** Each system against a baseline, per task: the mean difference, a 95%
  t-interval and a two-sided sign-flip permutation p-value (exact for n ≤ 16, else 20,000 draws
  with seed 0). Tasks are the units.
- **Writer samples.** Systems that differ only in their writer `sample` are one system's
  independent samples; comparisons average them per task before differencing (`samples: 2` in
  an experiment).
- **The registered utility** (`rewards.py`) combines completeness, retrieval, reliability and
  cost into one per-task number, for selection and training:

  ```text
  U = N + 0.15 R_c + 0.10 P + 0.10 S − 0.10 C / B0
  ```

  N is nugget coverage, R_c reference recall (0 on a task without gold), P cite_p, S claim
  support, C the task's list cost and B0 the benchmark's cap ($0.27). A task without a report
  scores U = −1 − 0.10 C / B0. Spending all of B0 costs 0.10, so an extra $0.027 must buy at
  least 0.01 nugget coverage. Reference recall can add at most 0.15, so comparisons also keep a
  separate completeness guard instead of relying on U alone.
- **The judge-free retrieval return** values the evidence alone:

  ```text
  r = 0.15 R_e − 0.10 C / B0
  ```

  R_e is the share of the task's important citations among the documents the writer receives
  (counted by `gold_found`, docs/03). It needs no report and no judge.
- **The evidence funnel.** `report.md` traces the important citations through each stage of a
  system: screened, judged relevant, given to the writer, cited.

## Cost

**The gateway** (`gateway.py`) is the only path to any model. Every LLM call and every tool call
passes through it, and it holds the provider keys; workers run with a scrubbed environment, so
anything that bypasses it fails.

**The ledger** (`runs/<id>/ledger.jsonl`) has one line per upstream attempt or tool charge: the
tag (run, benchmark, system, task, stage), tokens split into cache miss, cache hit, output and
reasoning, list and billed cost, latency and the session.

- **List cost** = (miss × p_miss + hit × p_hit + out × p_out) / 10⁶, at the prices in
  `configs/models.yaml` (DeepSeek-V4.1-Flash: $0.30, $0.006 and $1.20 per million tokens). Every
  per-task cap and every reported system cost is in list USD, so a cache replay costs exactly what
  the original call cost.
- **Billed cost** is what the provider charges: list × 0.5 outside DeepSeek's peak hours
  (weekdays 01:00–04:00 and 06:00–10:00 UTC are billed at list), and $0 for cache replays, local
  models and free tools. It limits the project's spend, never a system's.
- **Tools** carry a nominal list price, so retrieval is never free: arXiv search and lookup
  $0.001 per call, Tavily $0.008, OpenAlex $0.001, the local d1-3B $0.00002 per started thousand
  prompt tokens (about $0.00014 per decision; a market proxy, not a measured price).
- **`cold_usd`** prices every input token as a cache miss: a sensitivity view of comparisons
  that cache order could sway.

**A task's cost** is the sum of its generation events in its latest session (a resumed task
replays its cached calls at their recorded cost). A controlled run continued from another run's
decision point adds the list cost spent before that point (`offset_usd`). Judging is accounted
separately and is project spend, not system cost.

**Caps** latch in the gateway; once one is reached, no LLM attempt (retries included) and no
tool call starts:

| Cap | Applies to | Where it is set |
|---|---|---|
| per task | a task's generation, list USD | a system's own `task_usd` (DS-base: $2), else `budget.task_usd`, else the benchmark's B0 (`Benchmark.cap_usd`: DeepScholar $0.27, DRB2 $0.34) |
| per run | a run's generation, billed USD | `budget.run_usd` |
| judging per task | a task's judging, list USD | `budget.judge_usd` |
| judging per run | a run's judging, billed USD | `budget.judge_run_usd` |
| campaign | all runs ever, billed USD | `campaign_usd` in `configs/models.yaml` |

The cap is checked before each call starts, and an upstream slot is held until its call is
metered, so a task can overshoot by at most the calls already in flight; `report.md` counts
tasks over their cap. The controlled systems also keep a completion reserve: an action is legal
only if writing still fits afterwards (docs/05).

**Where B0 comes from.** The DeepScholar cap was set just above what the baseline needed on June
dev: DS-base intro-only's most expensive task there cost $0.243, × 1.1 = $0.27. The DRB2 cap is
DS-base's natural p90 there, $0.34. The controllers are held to B0. DS-base runs at its natural
cost under a $2 cap it never reaches: on Nov dev one of its tasks cost $0.287, which a $0.27 cap
would have left without a report.

**Where DS-base's money goes** (list USD per DeepScholar task, run `phase2-draft`):

| | DS-base intro-only | Controlled loop at 0.25 × B0 |
|---|---|---|
| relevance screening (`sem_filter`) | 146.6 calls, $0.0782 | 47.6 calls, $0.0227 |
| ranking (`sem_topk`) | 205.3 calls, $0.0335 | none |
| arXiv operations (searches + lookups) | 16.6 + 41.4, $0.0580 | 15.3, $0.0154 |
| search agent / query proposals | $0.0079 | $0.0009 |
| writer | $0.0041 | $0.0031 |
| total | $0.182 | $0.042 |

The relevance filter (screening plus ranking) is 61% of DS-base's cost; the writer is about 2%.
A controller saves money by screening fewer papers, so it must choose which searches are worth
screening and when to stop.

**Training economics** are reported apart from system cost: the API spend of labels and
rollouts, GPU hours, d1's tokens, decisions and latency, and the break-even point T/ΔC (training
spend T recovered by a saving of ΔC per task). d1's per-decision price above is a proxy, not a
measured deployment cost.

**Typical costs** (list USD per task, dev): Simple RAG $0.003, closed-book $0.001, DS-base
intro-only $0.18–0.19, DS-base on DRB2 $0.32 (p50). Per-task costs of every comparison are in
docs/05.

# JevScholar

A deep-research system whose retrieval is steered by a small decision model. At each step,
LiquidAI/d1-3B chooses which search to screen next (the paper's abstract as a query, a keyword
query the executor proposes, the next page of a productive query) or decides to stop and write.
Everything else is DeepScholar-base's own: its relevance filter, ranking and section writer. The
goal is DeepScholar-base's report quality on DeepScholar-Bench at a fraction of its cost, with
DeepResearch Bench II as a transfer check. Every system runs on the same executor,
DeepSeek-V4.1-Flash, with the same tools and access policy; the controllers have a per-task cap,
while DeepScholar-base runs at its natural cost.

## Results

- **A sound baseline.** DeepScholar-base reproduces close to the paper's ranges after two
  upstream bugs are repaired: a merge bug that kept 10 of about 150 candidate papers, and a
  citation bug that pointed most citations at the wrong paper. Three evaluation metrics were found
  to be computed differently from the paper's definitions; repaired versions are reported next to
  the official ones (docs/03).
- **A low-cost backbone.** DeepSeek-V4.1-Flash scores close to the paper's models under
  the paper's protocol at about a seventh of GPT-4.1's cost; DeepSeek judges agree with the
  official OpenAI judges on the checked citation and rubric judgments (item κ 0.74–0.93) but not
  on organization (docs/02). All DeepSeek results come from one pinned model version (docs/02).
- **The controller.** d1 (no weight update) costs 26–28% of DeepScholar-base, writes
  better-supported sections (claim support +0.10 on June, +0.13 on Nov dev) and finds about as
  many important citations. Its coverage of the reference content is within noise of DS-base's
  on Nov (nugget −0.01) and lower on June (−0.06, an interval just reaching 0), and it scores
  lower on organization (−0.18 and −0.31, a metric our judge reproduces poorly); its utility U is
  within noise of DS-base's on both sets. A fixed rule and the executor answering d1's prompt are
  within noise of d1 on nugget coverage and U; the fixed rule's claim support is lower on Nov
  (docs/05).
- **Training.** A supervised warm start on d1's own counterfactual branches (d1-sft), then
  bounded RL on d1-sft's sampled trajectories (d1-rl). Both stop after one batch on val and fail
  the registered val gate, so the held-out test's main claim is made with d1 (docs/06).
- **The final test** (180 held-out Nov test tasks; docs/06). The main claim fails: d1 costs 28% of
  DeepScholar-base and its citations are better supported (claim support +0.122 [0.106, 0.138]),
  but its nugget coverage is lower (−0.053 [−0.073, −0.033]) and its reference recall misses the
  margin (−0.031 [−0.062, 0.000]). The RL claim fails too, and on six DRB2 tasks no system's score
  differs resolvably from DeepScholar-base's.
- **DraftRef** (docs/06) is DeepScholar-base whose writer gets the papers its own search draft
  names, with d1 in place of its relevance filter; it submits the best of four writer samples,
  while DeepScholar-base is the mean of two. On the 180 test tasks it costs 41% of
  DeepScholar-base, its nugget coverage is within the 0.02 non-inferiority margin (+0.007
  [−0.013, 0.027]) and its reference recall is higher (+0.042 [0.023, 0.061]). Those tasks had
  been used before, so this is not an independent test. A citation rule for its writer (end a
  sentence where its source changes) raises its citation precision (cite_p) from 0.399 to 0.712
  (+0.313 [0.288, 0.337]), with nugget coverage and reference recall within the 0.02 margin
  (+0.004 [−0.015, 0.024], +0.001 [−0.001, 0.003]).

## Documentation

| Document | Contents |
|---|---|
| [docs/01_resources.md](docs/01_resources.md) | third-party code and data, pins and licences, the local arXiv index |
| [docs/02_backbone_and_judges.md](docs/02_backbone_and_judges.md) | DeepSeek against GPT-4.1; the judges and their validation |
| [docs/03_baselines_and_metrics.md](docs/03_baselines_and_metrics.md) | DeepScholar-base's defects and repairs, the evaluation's defects, the added metrics and their formulas, baseline results |
| [docs/04_data_and_evaluation.md](docs/04_data_and_evaluation.md) | datasets and splits, the evaluation flow, scoring and statistics, cost accounting |
| [docs/05_controller.md](docs/05_controller.md) | the controlled loop, its actions and rules, the policies, results |
| [docs/06_training_and_test.md](docs/06_training_and_test.md) | the conditions of every comparison, training d1 (SFT and RL), the val gate, the final test, DraftRef, the DRB2 transfer check, the research question, limitations and risks |
| [docs/final_report.pdf](docs/final_report.pdf) | the course's final report (source: docs/final_report.tex) |
| [docs/usage.md](docs/usage.md) | setup, commands, configuration, layout, reproducibility, run index |

## Quick start

```bash
uv sync
uv run python -m nltk.downloader -d .cache/nltk_data punkt_tab
uv run jev data fetch && uv run jev data prepare && uv run jev data arxiv-index
scripts/train_env.sh jev controller serve &           # d1-3B on the local GPU
uv run jev run evaluate --variant dev --run-id RUN    # configs/experiments/evaluate.yaml
uv run jev eval RUN && uv run jev report RUN
uv run pytest
```

Provider keys live in `~/.config/jevscholar/secrets.env`; see [docs/usage.md](docs/usage.md).

# JevScholar

A deep-research system whose retrieval is steered by a small decision model. At each step,
LiquidAI/d1-3B chooses which search to screen next (the paper's abstract as a query, a keyword
query the executor proposes, the next page of a productive query) or decides to stop and write.
Everything else is DeepScholar-base's own: its relevance filter, ranking and section writer. The
goal is DeepScholar-base's report quality on DeepScholar-Bench at a fraction of its cost, with
DeepResearch Bench II as a transfer check. Every system runs on the same executor,
DeepSeek-V4.1-Flash, with the same tools and access policy; the controllers have a per-task cap,
while DeepScholar-base runs at its natural cost.

## Where it stands

- **A sound baseline.** DeepScholar-base reproduces close to the paper's ranges after two
  upstream bugs are repaired: a merge bug that kept 10 of about 150 candidate papers, and a citation bug
  that pointed most citations at the wrong paper. Three evaluation metrics were found to be
  computed differently from the paper's definitions; repaired versions are reported next to the
  official ones (docs/03).
- **A low-cost backbone.** DeepSeek-V4.1-Flash scores close to the paper's models under
  the paper's protocol at about a fifth of GPT-4.1's cost; DeepSeek judges agree with the
  official OpenAI judges (item κ 0.79–0.91; docs/02).
- **The controller, before training.** d1 (no weight update) costs 26–27% of DeepScholar-base,
  writes better-supported sections (claim support +0.11 on June and on Nov dev), and finds about
  as many important citations on Nov, but covers less of the reference content (nugget −0.07 on
  June, −0.05 on Nov). A fixed rule and the executor answering d1's prompt are strong
  comparisons (docs/05).
- **Next.** A supervised warm start and bounded RL for d1, then the held-out test on 90 tasks
  (docs/06).

## Documentation

| Document | Contents |
|---|---|
| [docs/01_resources.md](docs/01_resources.md) | third-party code and data, pins and licences, the local arXiv index |
| [docs/02_backbone_and_judges.md](docs/02_backbone_and_judges.md) | DeepSeek against GPT-4.1; the judges and their validation |
| [docs/03_baselines_and_metrics.md](docs/03_baselines_and_metrics.md) | DeepScholar-base's defects and repairs, the evaluation's defects, the added metrics and their formulas, baseline results |
| [docs/04_data_and_evaluation.md](docs/04_data_and_evaluation.md) | datasets and splits, the evaluation flow, scoring and statistics, cost accounting |
| [docs/05_controller.md](docs/05_controller.md) | the controlled loop, its actions and rules, the policies, d1's prompt tuning, results |
| [docs/06_next_steps.md](docs/06_next_steps.md) | open decisions, SFT and RL, the final test, DRB2, schedule |
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

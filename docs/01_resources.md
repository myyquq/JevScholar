# 1. Third-party resources

What the project takes from others, at which version, and how it checks that it still has exactly
that. Every external byte is pinned by commit or revision and by sha256. Nothing upstream is
edited: fixes are applied at run time by our wrappers (docs/03).

## Vendored code (committed)

`third_party/` holds pristine copies of two upstream repositories, trimmed to the files we use.
`git diff` against the import commit must stay empty. Each run's manifest records
`git rev-parse HEAD:third_party/<name>`.

| Tree | Upstream | Commit | Licence | Files kept |
|---|---|---|---|---|
| `deepscholar/` | guestrin-lab/deepscholar | `c95413b` (2026-03-28) | MIT | `deepscholar_base/` (the DeepScholar-base pipeline), `eval/` (the DeepScholar-Bench evaluators, including castorini's nuggetizer 0.0.6), `configs/`, `data_pipeline/generate_queries.py` (the official query template), the two parser fixtures under `tests/baselines_results/` |
| `drb2/` | imlrz/DeepResearch-Bench-II | `b38f360` (2026-09-11) | Apache-2.0 (code) | `run_evaluation.py`, `gpt_client.py`, `aggregate_scores.py` |

The archive sha256 of each commit is in `third_party/README.md`, together with the list of
run-time deviations.

## Fetched data (never committed)

`data/sources.yaml` lists every external file: the URL at a pinned revision and the file's sha256.
`jev data fetch` downloads them into `data/raw/` and refuses any file whose hash differs.

| Source | Where from | What it is |
|---|---|---|
| DeepScholar-Bench, June 2025 | the same `c95413b` tarball, `dataset/` | 63 papers: abstract, related-work section, important citations, nuggets. The paper's main table uses this set. |
| DeepScholar-Bench, Nov 2025 | branch `new_dataset` at `81bba2f` (upstream PR 15, not merged) | 200 newer papers in the same format |
| DeepResearch Bench II (DRB2) | `b38f360`, `tasks_and_rubrics.jsonl` | 132 research tasks (zh and en) with expert rubrics |
| DRB2 reference reports | HF `muset-ai/DeepResearch-Bench-II-Dataset` @ `73c8b00` | 7 commercial systems' reports for our 12 DRB2 dev tasks (84 files; md, docx, html, pdf), used to calibrate our judge |
| ResearchQA data | HF `realliyifei/ResearchQA` @ `bf8a4cf` | `valid.json`, `test.json` (rubrics after the upstream citation fix of 2025-08-25) |
| ResearchQA test subset | HF `rl-research/researchqa_official_subset_ids` @ `d4e311c` | the 776-question `test_mini` ids that DR Tulu reports on |
| ResearchQA scorer | realliyifei/ResearchQA @ `747a9a1` | `compute_coverage.py`, `researchqa_loader.py`. The repository has no licence, so it is fetched, not vendored. |

Data licences: DeepScholar MIT; DRB2 data CC BY 4.0 (CC BY-NC 4.0 for tasks 26 and 110); DRB2
reports Apache-2.0; ResearchQA data MIT; the ResearchQA code has none.

`jev data prepare` then builds `data/derived/` (nugget files matched to their rows, the DRB2
reports as text) and the committed split files `data/splits/*.json` (docs/04).

## The local arXiv index

All systems search arXiv through one frozen local index instead of the live API.

- **Why.** The live arXiv API returns HTTP 429 to this host after about 10 requests, and LOTUS's
  date filter (`submittedDate:[00000000 TO …]`) returned HTTP 500 from it in October 2026. A local
  index is also identical for every system and exactly reproducible.
- **Source.** HF `librarian-bots/arxiv-metadata-snapshot` at revision `5733e66`: a weekly mirror
  of the Kaggle arXiv metadata, 11 parquet shards (about 2.7 GB), 3.2 million papers. The revision
  is pinned in `src/jevscholar/search/arxiv_local.py`; the shards are checked against the
  revision's LFS hashes.
- **Index.** `jev data arxiv-index` writes `data/derived/arxiv_index/`: a BM25 index (`bm25s`)
  over title and abstract, each paper's first-version date (the date filter), and a metadata
  table. Ids carry the latest version at snapshot time, as the API returns them.
- **What a search returns.** Title, abstract, authors, categories and dates; never PDFs. The
  access policy (docs/04) removes papers published on or after the task's cutoff and blocked
  papers before anything reaches a system.

The live arXiv API (`search.arxiv: api`), Tavily web search and OpenAlex remain available
behind the same metered interface (`src/jevscholar/search/`).

## Models and environments

- **Executor and judges.** DeepSeek-V4.1-Flash through the DeepSeek API; OpenAI models for the
  official judges and the GPT-4.1 comparison (docs/02). Model routing and prices are in
  `configs/models.yaml`.
- **Decision model.** LiquidAI/d1-3B at HF revision `051bcc4`, run locally on one GPU (docs/05).
  It needs transformers 5, while LOTUS pins transformers below 5, so it lives in its own
  environment, `.venv-train` (`scripts/train_env.sh`).
- **Dependencies.** Python 3.12, locked by `uv.lock`; LOTUS at git commit `4665619`; the vendored
  nuggetizer as a path dependency (`pyproject.toml`).

## Checking a checkout

```bash
uv sync
uv run jev data fetch            # every file verified against its sha256
uv run jev data prepare          # rewrites identical derived files and split lists
git diff --exit-code data/splits third_party
```

# Vendored upstream code

These trees are **pristine** copies of pinned upstream commits. They are never edited: every fix or
adaptation is applied at runtime by our wrappers (listed below), so `git diff` against this import
commit must stay empty. The run manifest records `git rev-parse HEAD:third_party/<name>`.

| Tree | Upstream | Commit | Archive sha256 | Licence | Subset |
|---|---|---|---|---|---|
| `deepscholar/` | github.com/guestrin-lab/deepscholar | `c95413b3b2f3255b461b90d0ce650f685ae2d1ff` (2026-03-28) | `c65f9d2463c6fbcec19ef89630aa1e829b3aea347f66bf64c5394f0944952589` | MIT | `deepscholar_base/`, `eval/` (incl. castorini nuggetizer 0.0.6), `configs/`, `data_pipeline/{__init__,generate_queries}.py`, `tests/baselines_results/deepscholar-base/` (parser fixtures), `LICENSE`, `README.md` |
| `drb2/` | github.com/imlrz/DeepResearch-Bench-II | `b38f360603db9531b102aef8c166cedb8509b6f6` (2026-09-11) | `4b8dd1d7e8727a97079052b14edcb7ea2bee1843f96a8c57c26b527522c4f4e0` | Apache-2.0 (code) | `run_evaluation.py`, `gpt_client.py`, `aggregate_scores.py`, `LICENSE`, `DATA_LICENSE`, `README.md` |

Not vendored: the ResearchQA scorer (github.com/realliyifei/ResearchQA @`747a9a1`) has no licence, so
`jev data fetch` downloads it into `data/raw/researchqa/code/` (pinned in `data/sources.yaml`). Datasets
are fetched the same way and never committed.

## Runtime deviations (applied by our wrappers, never by editing these files)

DeepScholar-base (`src/jevscholar/systems/deepscholar_base.py`):
- P1 LOTUS `web_search` / `web_extract` → our policy-enforcing, metered SearchClient (arXiv reads
  return title + abstract, no PDFs). Note: LOTUS's arXiv date bound `submittedDate:[00000000 TO …]`
  returns HTTP 500 from the arXiv API as of 2026-10-16.
- P2–P4 full brace escaping of the topic in the sanitizer, the filter and the category summary
  (upstream crashes on abstracts / prompts containing `{…}`).
- P5 `AgentContext.papers_df` starts as an empty frame (upstream `len(None)` hides tool errors).
- P6 `repair_merge` (default on): accumulate search results across queries (upstream 9a37581 keeps
  only the first non-empty query's ≤10 papers); the as-is variant is run as an ablation.
- P7 agents tracing off, P8 LOTUS cache off, P9 `np.random.seed(task.seed)`.
- P10 `repair_citations` (opt-in; the repaired baselines set it): the intro writer sees each
  document's `_citation_number` (upstream escapes the citation guidelines with the user text, so
  the column placeholder never reaches the prompt), and numeric citations are stripped from the
  search background (their numbers index the search step's own list). Unrepaired, most intro
  citations resolve to the wrong papers.
- Per-system YAML overrides (`config:`), e.g. the paper's recursive protocol (paper A.2.4).
- Configuration, not patches: all four stage LMs passed explicitly (upstream drops `max_tokens` to 512),
  a fresh `Configs` per task, agent client `max_retries=0`.

DeepScholar-Bench eval (`src/jevscholar/benchmarks/deepscholar_eval.py`): our per-task driver replaces
`eval/main.py`; metric classes untouched; OpenAlex and arXiv lookups replaced by cached equivalents;
June-2025 nuggets remapped by qid (upstream issue #16 / PR #17).

DRB2 eval (`src/jevscholar/benchmarks/drb2_eval.py`): PR #8 JSON parsing applied by replacing
`run_evaluation._try_clean_and_load`; the module-global client replaced by a per-task tagged client.

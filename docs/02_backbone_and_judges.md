# 2. Backbone and judges

The project uses one comparable backbone model across all systems. This document explains why
every system runs on DeepSeek-V4.1-Flash, and why DeepSeek models can also serve as the judges.

## The executor

- **Model.** DeepSeek-V4.1-Flash (`deepseek-flash` in `configs/models.yaml`), non-thinking,
  temperature 1.0, at most 10,000 output tokens per call. These are DeepScholar-base's own YAML
  settings with the model swapped.
- **Why non-thinking.** DeepScholar-base was published with non-reasoning models (gpt-4o-mini in
  its repository, GPT-4.1 and Llama-4 in the paper).
- **The same for every system.** Simple RAG, closed-book, every DeepScholar-base version and the
  controlled loop all use this one executor, the same search tools and access policy, the same
  output-length limits and, in the main comparison, the same per-task cap. Only the decision
  model d1-3B (docs/05) is extra, and its cost is charged to the controlled system.

## DeepSeek against GPT-4.1

The DeepScholar paper reports DeepScholar-ref with GPT-4.1 and with Llama-4. We ran the paper's
protocol (recursive search, 2 rounds of at most 2 queries; paper A.2.4) with DeepSeek, and the
same protocol with GPT-4.1 through the same harness.

**Against the paper's published numbers** (nugget coverage; ours on 20 tasks, the paper's on all
63 June and 200 Nov tasks, with OpenAI judges):

| Set | DeepSeek, paper protocol | Paper, Llama-4 | Paper, GPT-4.1 |
|---|---|---|---|
| June 2025 (dev) | 0.228 | 0.241 | 0.348 |
| Nov 2025 (dev2) | 0.389 | 0.358 (Llama-4-Scout) | 0.479 (GPT-4.1 + o3) |

On Nov, DeepSeek sits between the paper's two models; on June it is just below both (0.228
against 0.241). Runs: `phase1-dev`, `repro-nov`.

**GPT-4.1 through our harness**, paper protocol, the first 3 June dev tasks (`repro-gpt41`),
against DeepSeek on the same 3 tasks (`phase1-dev`). "+P10" is the citation repair of docs/03.

| | GPT-4.1 | GPT-4.1 +P10 | DeepSeek | DeepSeek +P10 |
|---|---|---|---|---|
| nugget coverage | 0.156 | 0.281 | 0.285 | 0.287 |
| cite_p (official code) | 0.059 | 0.344 | 0.132 | 0.294 |
| cite_p per citation | 0.157 | 0.487 | 0.147 | 0.363 |
| claim coverage | 0.198 | 0.493 | 0.229 | 0.396 |
| reference coverage | 0.056 | 0.056 | 0.087 | 0.087 |
| list cost per task | $0.372 | $0.374 | $0.076 | $0.073 |

- **Quality.** On 3 tasks the two backbones are in the same range; neither is clearly better.
- **The citation bug is the pipeline's.** Both models misattribute citations without the repair
  and recover with it (docs/03), so the bug is not an artefact of swapping the model.
- **Cost.** DeepSeek's list prices are 6.7 times lower ($0.30 / $1.20 against $2.00 / $8.00 per
  million input / output tokens), and its per-task cost is about one fifth of GPT-4.1's. DeepSeek
  also halves its price off-peak.

**Conclusion.** DeepSeek-V4.1-Flash is a low-cost stand-in for the paper's models: comparable
quality under the paper's protocol at about a fifth of the cost. Every comparison in this project
uses it for all systems, so differences between systems are not backbone differences. Absolute
numbers are not comparable with the paper's, which used other models and OpenAI judges.

## The judges

Each benchmark's official evaluator uses OpenAI judges. We keep the official evaluation code and
swap the judge model for a DeepSeek profile that mirrors the official judge's reasoning mode.

| Benchmark | Official judge | Our judge (`configs/models.yaml`) |
|---|---|---|
| DeepScholar-Bench | gpt-4o; gpt-4.1 for nuggets | `ds-judge-chat`: DeepSeek-V4.1-Flash, non-thinking |
| ResearchQA | gpt-4.1-mini | `ds-judge-chat` |
| DRB2 | GPT-5.5, medium effort | `ds-judge-think`: thinking, effort high, JSON forced |

Judges are never served from the replay cache. All scores in this project are labelled
"DeepSeek-judged (non-official)".

### Agreement with the official judges

The official judges re-scored 139 reports (`jev check agree`, run `judge-agreement`): the first 6
DeepScholar, 40 ResearchQA and 6 DRB2 dev tasks of Simple RAG, closed-book and DeepScholar-base.
Item κ compares the two judges' decisions on the same items (linearly weighted on the ordinal
ResearchQA and DRB2 scales). Δ is a system's paired difference against Simple RAG on the primary
metric, DeepSeek → official.

| Benchmark | Item κ (items) | Δ closed-book | Δ DS-base | Tasks reversed |
|---|---|---|---|---|
| DeepScholar | cite_p 0.84 (217), claims 0.89 (145), organization 0.54 (36) | +0.044 → +0.072 | +0.094 → +0.103 | 1/6, 0/6 |
| ResearchQA | 0.79 (786 rubric labels) | +0.318 → +0.281 | −0.034 → −0.030 | 0/40, 4/40 |
| DRB2 | 0.89–0.91 (1,113 rubric items) | +0.176 → +0.178 | +0.031 → +0.039 | 0/6, 0/6 |

- **The rule, set before the check:** κ ≥ 0.6 and no reversed mean difference. cite_p, claim
  coverage, ResearchQA and DRB2 pass. Organization fails (κ 0.54) and is dropped. Claim support
  (docs/03) was added later and was not part of this check; it asks the judge the same support
  question as claim coverage, on different evidence.
- **Nugget coverage** has no item-level check (its verdicts are per nugget, not per citation);
  its per-task Pearson r with the official judge is 0.64–0.92 and its differences agree.
- **Levels differ, differences agree.** DeepSeek is stricter on DeepScholar relevance (Simple RAG
  relevance rate 0.23 against 0.34). On the primary metrics the means differ by at most 0.09.

### Blind spot-check

Automated judgments are spot-checked. `jev check review-pack` draws judge
decisions into a blind pack (no system name, no verdict); a reviewer labels them, and
`jev check review-score` compares. Codex labelled 201 items from Simple RAG and DS-base:

| Items | Agreement | κ |
|---|---|---|
| DRB2 rubric scores (81) | 88–96% | 0.75–0.93 |
| ResearchQA rubric labels (60) | 62% exact | 0.52 (weighted) |
| DeepScholar cite_p (22) | 95% | 0.91 |
| DeepScholar claim coverage (18) | 83% | 0.66 |
| DeepScholar organization (20) | 50% | −0.09 |

cite_p first gave κ 0.36: the pack did not say that a citation must support the whole sentence,
and the reviewer accepted partial support. With the judge's question stated (the pack now states
it for every support metric), κ is 0.91. These checks predate claim support; its verdicts are
the same support question and enter the same blind packs.

### DRB2 calibration

The DRB2 judge scored the 7 commercial systems' published reports on our 12 dev tasks
(`jev check calibrate drb2`, run `drb2-calibration`). Its system ranking agrees with the official
leaderboards: Spearman ρ 0.60 against the GPT-5.5 board (5 systems) and 0.61 against the paper's
Gemini-2.5-Pro board (7 systems). Its scores are 1.9–9.5 points above the GPT-5.5 board, though
on a different task set (our 12 dev tasks against the board's full set), so this is not a pure
measure of leniency.

| System | Ours (12 dev tasks) | GPT-5.5 board | Gemini-2.5-Pro board |
|---|---|---|---|
| GPT-o3 DeepResearch | 50.61 | 43.00 | 45.40 |
| Qwen-3-Max DeepResearch | 46.78 | – | 39.25 |
| Doubao DeepResearch | 44.88 | 37.36 | 40.99 |
| Gemini-2.5-Pro DeepResearch | 44.31 | 39.98 | 41.98 |
| Gemini-3-Pro DeepResearch | 43.26 | 41.40 | 44.60 |
| Perplexity Research | 43.12 | 33.65 | 38.58 |
| Grok DeepSearch | 41.38 | – | 39.23 |

### Self-preference

The judges and the executor are the same model family. A family bias is largely common to all
systems in a paired difference, but not guaranteed to cancel. Scores are therefore labelled
non-official, key gains are blind-checked, and no leaderboard equivalence is claimed.

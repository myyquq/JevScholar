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
  controlled loop all use this one executor, the same search tools and access policy and the same
  output-length limits. In the main comparison the controllers share the per-task cap B0, while
  DS-base runs at its natural cost (docs/04). Only the decision model d1-3B (docs/05) is extra,
  and its cost is charged to the controlled system.

### Model version

`deepseek-flash` is an API name, not a fixed model: DeepSeek can change the model behind it
without changing the API or the prices. Each reply carries the server's `system_fingerprint`; the
ledger records it for every call, fresh or replayed, and `report.md` lists the versions a run
used.

Every DeepSeek result in these documents comes from one version, as executor and as judge: the
one with fingerprint `bc19ba97cbe016479903222bb1578db6`, served under `deepseek-flash` on
2026-11-09. The fingerprint is pinned in `configs/models.yaml` (`versions`): a reply of any other
version, fresh or from the replay cache, stops the run before it is used or stored, and the pinned
version is part of every run cell's and every verdict's identity. `jev check version` makes one
uncached call and fails unless the pinned version answers.

## DeepSeek against GPT-4.1

The DeepScholar paper reports DeepScholar-ref with GPT-4.1 and with Llama-4. We ran the paper's
protocol (recursive search, 2 rounds of at most 2 queries; paper A.2.4) with DeepSeek, and the
same protocol with GPT-4.1 through the same harness.

**Against the paper's published numbers** (the benchmark's own metrics; ours on 20 tasks, the
paper's on all 63 June and 200 Nov tasks, with OpenAI judges):

| Set, metric | DeepSeek, paper protocol | Paper, Llama-4 | Paper, GPT-4.1 |
|---|---|---|---|
| June 2025 (dev), nugget coverage | 0.219 | 0.241 | 0.348 |
| June 2025 (dev), organization | 0.400 | 0.206 | 0.809 |
| Nov 2025 (dev2), nugget coverage | 0.410 | 0.358 (Llama-4-Scout) | 0.479 (GPT-4.1 + o3) |
| Nov 2025 (dev2), organization | 0.625 | 0.120 (Llama-4-Scout) | 0.578 (GPT-4.1 + o3) |

On nugget coverage, DeepSeek sits between the paper's two models on Nov and just below both on
June (0.219 against 0.241). On organization it is within the paper's range on June and above it
on Nov, but our judge reproduces the official one poorly on that metric (below), so this
comparison is rough. Runs: `baselines-june`, `repro-nov`.

**GPT-4.1 through our harness**, paper protocol, the first 3 June dev tasks (`repro-gpt41`),
against DeepSeek on the same 3 tasks (`baselines-june`). "+P10" is the citation repair of docs/03.

| | Nugget | Org. | Ref. cov. | cite_p | Claim cov. | ‖ | Ref. recall | Per citation | Claim support | List cost per task |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4.1 | 0.139 | 0.000 | 0.056 | 0.059 | 0.199 | ‖ | 0.056 | 0.157 | 0.234 | $0.372 |
| GPT-4.1 +P10 | 0.259 | 0.000 | 0.056 | 0.343 | 0.504 | ‖ | 0.056 | 0.483 | 0.711 | $0.374 |
| DeepSeek | 0.226 | 0.000 | 0.056 | 0.123 | 0.209 | ‖ | 0.056 | 0.185 | 0.262 | $0.051 |
| DeepSeek +P10 | 0.172 | 0.000 | 0.056 | 0.440 | 0.595 | ‖ | 0.056 | 0.531 | 0.674 | $0.050 |

Left of ‖: the benchmark's own metrics; right: our corrected ones (docs/03: the metrics we report).
Org. is organization, which our judge reproduces poorly (below).

- **Quality.** On 3 tasks the two backbones are in the same range; neither is clearly better.
- **The citation bug is the pipeline's.** Both models misattribute citations without the repair
  and recover with it (docs/03), so the bug is not an artefact of swapping the model.
- **Cost.** DeepSeek's list prices are 6.7 times lower ($0.30 / $1.20 against $2.00 / $8.00 per
  million input / output tokens), and its per-task cost is about one seventh of GPT-4.1's. DeepSeek
  also halves its price off-peak.

**Conclusion.** DeepSeek-V4.1-Flash is a low-cost stand-in for the paper's models: comparable
quality under the paper's protocol at about a seventh of the cost. Every comparison in this project
uses it for all systems, so differences between systems are not backbone differences. Absolute
numbers are not comparable with the paper's, which used other models and OpenAI judges.

## Training data and contamination

The access policy keeps the target paper and later papers out of every search (docs/04), but it
cannot remove what the executor learned in training. If DeepSeek's training data contains a
target paper, the executor may reproduce its related-work content from memory.

- **Training cutoff.** DeepSeek publishes none for V4.1-Flash or its updates: not in the
  model card, the V4.1-Flash paper (arXiv 2609.19969, which gives only the corpus size, 45T
  tokens) or the API changelog. Third-party listings give May 2025 for the V4 Flash family;
  that is unconfirmed for V4.1, and nothing at all is known for its updates. GPT-4.1, the paper's
  model, has a documented cutoff of June 2024, before every task.
- **The tasks' dates.** June dev papers were published in April (6), May (10) and June 2025 (4);
  val and train in the same months. Nov dev2 papers in October and November 2025; the 180 test
  tasks in September to November 2025. If the May 2025 cutoff holds, part of the June set
  predates it and the Nov set does not.
- **What the data show.** No sign of memorisation. Closed-book nugget coverage on June dev is lowest
  on the April papers, the likeliest to be in the training data (April 0.224, May 0.324, June 0.380;
  6, 10 and 4 tasks). Under the paper's protocol DeepSeek scores 0.219 on June against GPT-4.1's
  0.348 (above), not higher. Both checks are small.
- **What it would affect.** Mostly nugget coverage, which credits content however the writer
  obtained it. Reference recall and claim support credit only documents the system retrieved
  under the access policy and cited (official claim coverage also shows the judge the target's
  abstract), but memory can still steer them: a remembered reference can shape a search query or
  the choice of what to cite. Contamination is shared by all systems, but not neutrally: memory
  helps most where retrieved evidence is thin, so it would favour the cheaper controllers over
  DS-base.
- **Mitigation.** The final test uses the Nov tasks; the claims rest on paired differences, with
  reference recall and claim support alongside nugget coverage.

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

The official judges scored 139 reports (`jev check agree`, run `judge-agreement`): the first 6
DeepScholar, 40 ResearchQA and 6 DRB2 dev tasks of Simple RAG (K = 10), closed-book and DS-base
with the merge repair P6 only (docs/03), from an earlier dev run (`source_run` in the run's
manifest). The DeepSeek judges scored the same stored reports (run `agreement-deepseek`); the
comparison uses those verdicts. Item κ compares the two judges' decisions on the same items
(linearly weighted on the ordinal ResearchQA and DRB2 scales). Δ is a system's paired difference
against Simple RAG on the primary metric, DeepSeek → official.

| Benchmark | Item κ (items) | Δ closed-book | Δ DS-base (P6) | Tasks reversed |
|---|---|---|---|---|
| DeepScholar | cite_p 0.81 (217), claims 0.93 (145), organization 0.48 (36) | +0.027 → +0.072 | +0.096 → +0.103 | 1/6, 0/6 |
| ResearchQA | 0.74 (786 rubric labels) | +0.321 → +0.281 | −0.011 → −0.030 | 1/40, 5/40 |
| DRB2 | 0.83–0.90 (1,113 rubric items) | +0.207 → +0.178 | +0.032 → +0.039 | 1/6, 0/6 |

- **The rule, set before the check:** κ ≥ 0.6 and no reversed mean difference. cite_p, claim
  coverage, ResearchQA and DRB2 pass. Organization fails (κ 0.48, on only 36 items, so a rough
  estimate); it is still reported in every table, flagged, and no decision rests on it. Claim
  support (docs/03) has no official verdicts; it asks the judge the same support question as claim
  coverage, on different evidence.
- **Nugget coverage** has no item-level check (its verdicts are per nugget, not per citation);
  its per-task Pearson r with the official judge is 0.63–0.92 and its differences agree.
- **Levels differ, differences agree.** DeepSeek scores lower than the official judges on most
  means (nugget coverage by 0.05–0.10; ResearchQA coverage by up to 0.11, on DS-base); the paired
  differences keep their sign everywhere.

### Blind spot-check

Automated judgments are spot-checked. `jev check review-pack` draws judge
decisions into a blind pack (no system name, no verdict); a reviewer labels them, and
`jev check review-score` compares. Codex labelled 180 items from Simple RAG (K = 20) and DS-base
(P6) on the dev tasks (run `baselines-june`, pack `review-deepseek`):

| Items | Metric from | Agreement | κ |
|---|---|---|---|
| DRB2 rubric scores (80) | the benchmark | 86–96% | 0.71–0.93 |
| ResearchQA rubric labels (60) | the benchmark | 65% exact | 0.56 (weighted) |
| DeepScholar cite_p (20) | the benchmark | 80% | 0.60 |
| DeepScholar claim support (20) | ours (docs/03) | 80% | 0.60 |

The pack states each support question: the claim is the whole sentence, and a reference that
supports only part of a sentence citing several works counts as unsupported. All four cite_p
disagreements are such sentences, where the judge accepted the partial support. DeepSeek's cite_p
is therefore lenient on sentences that cite several works; it agrees with gpt-4o at κ 0.81
(above), so the official judge likely reads the question the same way. Claim support asks about a
sentence's citations together; its four disagreements go both ways (two lenient, two strict).

### DRB2 calibration

The DRB2 judge scored the 7 commercial systems' published reports on our 12 dev tasks
(`jev check calibrate drb2`, run `drb2-calibration`). Its system ranking is positively correlated
with the official leaderboards: Spearman ρ 0.90 against the GPT-5.5 board (5 systems) and 0.64
against the paper's Gemini-2.5-Pro board (7 systems). Its scores are 3.8–10.0 points above the
GPT-5.5 board, though on a different task set (our 12 dev tasks against the board's full set), so
this is not a pure measure of leniency.

| System | Ours (12 dev tasks) | GPT-5.5 board | Gemini-2.5-Pro board |
|---|---|---|---|
| GPT-o3 DeepResearch | 51.46 | 43.00 | 45.40 |
| Qwen-3-Max DeepResearch | 49.49 | – | 39.25 |
| Gemini-3-Pro DeepResearch | 45.41 | 41.40 | 44.60 |
| Gemini-2.5-Pro DeepResearch | 43.80 | 39.98 | 41.98 |
| Perplexity Research | 43.64 | 33.65 | 38.58 |
| Doubao DeepResearch | 43.45 | 37.36 | 40.99 |
| Grok DeepSearch | 39.59 | – | 39.23 |

### Self-preference

The judges and the executor are the same model family. A family bias is largely common to all
systems in a paired difference, but not guaranteed to cancel. Scores are therefore labelled
non-official, and no leaderboard equivalence is claimed. The blind spot-check (above) covers only
the baselines' reports.

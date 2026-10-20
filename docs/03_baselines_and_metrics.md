# 3. Baselines and metrics: what we found and fixed

DeepScholar-base is the reference system of DeepScholar-Bench, and the system our controller must
match at lower cost. Before comparing anything with it, we ran it as released, reproduced the
paper's protocol, and audited both the pipeline and the evaluation. This document lists what was
wrong, how each defect was fixed, which metrics we added and exactly how they are computed, and
the baseline's numbers before and after the fixes.

All scores are DeepSeek-judged (non-official, docs/02). Unless stated otherwise: the June dev
split (20 tasks), primary scoring (a failed task scores 0), and the paired difference with a 95%
t-interval in brackets.

## The baselines

| System | What it does | Catalog name |
|---|---|---|
| Simple RAG | the executor writes one search query; the top 20 arXiv abstracts are retrieved; one call writes the section citing them | `simple_rag_k20` |
| Closed-book | one call, no retrieval (what the backbone knows) | `closed_book` |
| DeepScholar-base, as released | upstream `c95413b`, agentic search, with upstream's merge behaviour; the shared adaptations P1–P5 and P7–P9 below still apply | `deepscholar_base_asis` |
| DeepScholar-base, paper protocol | the paper's DeepScholar-ref: recursive search, 2 rounds of at most 2 queries (paper A.2.4) | `deepscholar_ref` |
| DeepScholar-base, repaired | the merge repair P6 and the citation repair P10 | `deepscholar_base_citefix` |
| DeepScholar-base, repaired, intro-only | the same without the reference-categorization and insight stages, which the benchmark does not score; **the baseline of every comparison**, run at its natural cost | `deepscholar_base_intro` |

**How DeepScholar-base works.** An agent (or, in the paper's protocol, a fixed recursion) issues
arXiv queries and drafts a background from what it reads. All retrieved papers are then screened
one by one by a chain-of-thought relevance filter (LOTUS `sem_filter`), the relevant ones are
ranked down to 30 (`sem_topk`), and the intro writer writes the related-work section from those
30 documents and the agent's draft.

**Simple RAG's K.** On dev, K = 5, 10 and 20 were within noise of each other on all three
benchmarks (DeepScholar against K = 10: K = 5 +0.017 ± 0.055, K = 20 +0.035 ± 0.048). K = 20 was
frozen. Run `phase1-extras`.

## Defects in DeepScholar-base and their repairs

Every repair is a run-time patch in `src/jevscholar/systems/deepscholar_base.py`; the vendored
code stays pristine. Each has a number, used across the code and these documents.

| | Defect | Repair | Effect |
|---|---|---|---|
| P1 | LOTUS calls the arXiv API and Tavily directly, unmetered and unfiltered | LOTUS's `web_search` and `web_extract` are routed to our metered, policy-enforcing search client; arXiv reads return title and abstract (upstream: the first 1,000 PDF characters) | the same tools for every system |
| P2–P4 | braces in an abstract crash the pipeline (prompts are built with `str.format`) | full brace escaping in the sanitizer, the filter and the category summary | no crashes |
| P5 | `AgentContext.papers_df` starts as `None`, so `len(None)` hides tool errors | starts as an empty frame | errors surface |
| P6 | **merge bug** (upstream commit `9a37581`): only the first non-empty query's results (at most 10 papers) are kept | `repair_merge`: results accumulate across queries | below |
| P7–P9 | tracing on, LOTUS's own cache on, unseeded ranking | tracing and cache off; `np.random.seed(task seed)` before ranking | determinism |
| P10 | **citation bug**: most citations point to the wrong paper | `repair_citations` | below |
| – | stage models are derived from one `lm`, silently capped at 512 output tokens | all four stage models passed explicitly, 10,000 tokens | intended behaviour |

**The merge bug (P6).** The as-is system replays the repaired system's agent trajectory from the
cache (15 of 15 agent calls), so only the merge differs.

| | As released | P6 |
|---|---|---|
| candidate pool (median papers) | 10 | 152 |
| cited sources (median) | 7 | 24.5 |
| reference coverage | 0.035 | 0.282 |
| cite_p | 0.14 | 0.32 |
| claim coverage | 0.18 | 0.48 |
| nugget coverage | 0.300 | 0.285 |

**The citation bug (P10).**
- *Mechanism.* The intro writer is told to cite each document by its `_citation_number` column.
  Upstream escapes the citation guidelines together with the user text, so the placeholder never
  reaches the prompt. The writer then copies the numbers in the search background (the agent's
  draft, whose numbers index the agent's own result list) or numbers citations by order of
  appearance. Upstream resolves those numbers against the filtered document list, so most
  citations land on the wrong paper. Example: "BankFair and BankFair+ … [12][17]" is copied from
  the background, where 12 and 17 were BankFair's positions; in the writer's list BankFair is
  document 4. All 40 June backgrounds carry such numbers.
- *Repair.* Leave the guidelines unescaped, so the writer sees each document's number, and strip
  numeric citations from the background. Stripping alone does not help (cite_p stays at 0.317):
  the writer then numbers by order of appearance.
- *Effect*, with search and screening replayed from the cache so only the writer differs:

| Contrast | cite_p | claim coverage | claim support | nugget coverage |
|---|---|---|---|---|
| agentic, P10 − P6 only | +0.120 [0.013, 0.227] | +0.230 [0.105, 0.355] | +0.250 [0.113, 0.387] | +0.011 [−0.025, 0.047] |
| paper protocol, June | +0.307 | +0.360 | +0.396 | −0.006 |
| paper protocol, Nov | +0.353 | +0.426 | +0.475 | −0.023 |

- *Not the backbone's fault.* GPT-4.1 through the same harness shows the same bug and the same
  recovery (cite_p 0.059 → 0.344 on 3 tasks; docs/02).

Two quirks of the paper protocol are kept as coded: the executor writes 4 queries per round where
2 are asked for, and only the last round's results enter the candidate pool.

## Defects in the DeepScholar-Bench evaluation

The official metric code stays, so our numbers remain comparable with the paper. Where it is
wrong, a repaired metric is reported next to it.

### Citation precision: per sentence, not per citation

The paper defines cite_p as the average precision of each citation. The released code
(`eval/evaluator/cite_p.py`) computes something else:

```text
for each sentence s of the report with at least 50 characters:
    C(s)    = the valid citations in s
    prec(s) = (citations in C(s) the judge says support s, citations removed) / |C(s)|,
              or 0 if C(s) is empty
cite_p = mean of prec(s) over those sentences
```

A sentence without citations scores 0, and a sentence citing several works counts once. We add
the paper's definition, recomputed from the saved judge verdicts at no extra cost:

```text
cite_p_per_citation = (supported citation verdicts) / (all citation verdicts in the report)
```

On the authors' own two DeepScholar-base reports (the vendored fixtures, `jev check calibrate
deepscholar`), the code gives 0.456 and the per-citation view 0.789; the paper reports 0.674 for
that system. The June table is comparable only per citation; the Nov table (0.178–0.563) is on
the code's scale.

### Reference coverage: fuzzy title matching

Upstream reference coverage matches each cited title to the important citations at string
similarity > 0.8, so it credits FairSort for FairRec, and it scores a task without important
citations 0. We add **reference recall** (`benchmarks/deepscholar.py: gold_found`):

```text
G      = the task's rows of important_citations.csv with an arXiv link
         (a duplicated row counts twice, as upstream lists it)
found  = rows of G whose versionless arXiv id is among the report's cited ids,
         or, for a row without a parsable id, whose normalized title is among the cited titles
reference_recall = found / |G|,   undefined (left out of means) when |G| = 0
```

The same function measures the important citations at every stage of a pipeline (screened,
relevant, given to the writer, cited) and the controller's evidence recall (docs/05), so rewards
and the endpoint count identically. Effect: 13 of 139 scores changed where both are defined; the
repaired DS-base drops from 0.304 to 0.284 on the 19 dev tasks that have important citations.

### Claim coverage: wrong evidence

Upstream claim coverage (`eval/evaluator/claim_coverage.py`, window 1) loops over a claim's
neighbouring sentences but reads the citations of the claim's own sentence for each of them, and
appends the target paper's own title and abstract to the evidence. We add **claim support**
(`benchmarks/deepscholar_eval.py: _claim_support`):

```text
claims = sentences of at least 50 characters, citation markers removed
for each claim c in sentence i:
    for j in i, then i-1 and i+1:
        E = the documents cited in sentence j
        if E is non-empty and not asked before: ask the judge "does E support c?"
        stop at the first yes
claim_support = (claims with a yes) / (claims),   NaN without claims
```

Only cited documents are evidence. Effect on the conclusions: the repaired DS-base's lead over
Simple RAG (K = 10) on claim coverage, +0.089 [0.030, 0.148], is no longer resolved on claim
support, +0.065 [−0.008, 0.138]; every other conclusion stands.

### Smaller fixes

- **Document importance.** OpenAlex lookups cut titles at `&` and `#`, and transient errors
  scored as "no match". Titles are now URL-encoded, failures retried, citation counts frozen at
  first lookup. The metric is heavy-tailed and resolves no difference on dev; it is not used.
- **Organization** (a pairwise judgment against the paper's own section) has item κ 0.54 against
  gpt-4o and −0.09 against blind labels (docs/02). It is dropped.
- **Nugget files.** The June nugget folders are misaligned with the dataset rows (upstream issue
  16); `jev data prepare` matches them by paper id and checks each against the row's abstract.
- **Reports without arXiv citations** are still scored on nugget coverage (upstream drops them),
  so closed-book is scored. Judge and lookup failures become per-metric errors and are re-judged
  (upstream scores a failed nugget judgment as unsupported or aborts).
- **DRB2's upstream parser** corrupts compact JSON; the fix of upstream PR 8 is applied.
- **Export rules for every system:** model-written reference lists are stripped and references
  rendered uniformly; ResearchQA bodies are capped at 300 words and DRB2's at 2,000 (a CJK
  character counts 1/1.6 word).

### Which metric plays which role

| Role | Metric |
|---|---|
| completeness (primary) | nugget coverage (official code) |
| citation reliability | cite_p (official code), cite_p per citation, claim support |
| retrieval | reference recall; evidence recall R_e (docs/05) |
| reported, not used | official claim coverage, official reference coverage, relevance rate |
| dropped | organization, document importance (judged only on request: `jev eval --metrics`) |

## Baseline results before and after

**June dev, DeepScholar-Bench** (runs `phase1-dev`, `deepscholar-calibration`):

| System | Nugget | Ref. cov. | cite_p | Per citation | Claim cov. | Rel. rate |
|---|---|---|---|---|---|---|
| Authors' own reports, our evaluation (n = 2) | .299 | .150 | .456 | .789 | .709 | .537 |
| Paper: DeepScholar-ref, Llama-4 (63 tasks) | .241 | .103 | .674 | – | .851 | .436 |
| Paper: DeepScholar-ref, GPT-4.1 (63 tasks) | .348 | .166 | .788 | – | .899 | .590 |
| DS-base as released | .300 | .035 | .138 | .359 | .180 | .422 |
| DS-base, P6 | .285 | .282 | .319 | .383 | .476 | .444 |
| DS-base, P6 + P10 | .297 | .288 | .439 | .485 | .707 | .725 |
| Paper protocol | .228 | .170 | .149 | .200 | .250 | .666 |
| Paper protocol + P10 | .222 | .179 | .456 | .582 | .610 | .689 |
| Simple RAG, K = 10 | .185 | .089 | .561 | .830 | .617 | .344 |

**Nov dev2, DeepScholar-Bench** (run `repro-nov`):

| System | Nugget | Ref. cov. | cite_p | Per citation | Claim cov. |
|---|---|---|---|---|---|
| Paper: DeepScholar-ref, Llama-4-Scout (200 tasks) | .358 | .072 | .178 | – | .581 |
| Paper: DeepScholar-ref, GPT-4.1 + o3 (200 tasks) | .479 | .087 | .563 | – | .578 |
| Paper protocol | .389 | .181 | .166 | .206 | .285 |
| Paper protocol + P10 | .366 | .243 | .519 | .558 | .711 |
| Simple RAG, K = 20 | .354 | .132 | .490 | .504 | .709 |

After the repairs, the reproduction is close to the paper's ranges: on June the repaired agentic
DS-base (.297) is within the paper's .241–.348, and the paper protocol (.222) just below it; on
Nov the paper protocol (.366) is within .358–.479. Citation quality is within the published
ranges after P10. The remaining gap is per-citation precision on June (.49–.58 against the authors' .79): our writer packs
several works into one sentence, so each citation supports only part of it. GPT-4.1 shows the
same level (.487), so it comes from the pipeline.

**Choosing the baseline.** The rule, set before the runs: switch from the repository's agentic
mode to the paper's protocol if its nugget coverage is no worse (lower bound above −0.05) and its
cite_p is higher (p < .05). Unrepaired, the paper protocol is worse on both (nugget −0.058
[−0.116, 0.000]; cite_p −0.170, p = .04). Repaired, agentic still leads: nugget +0.074 [0.005,
0.144], claim coverage +0.097, reference coverage +0.109. The baseline is therefore the agentic
DS-base with P6 and P10, run intro-only (`deepscholar_base_intro`): replaying it intro-only
reproduced the same 20 reports byte for byte from the cache (run `phase2-baselines`).

**Natural cost on three benchmarks** (dev; Δ against Simple RAG with K = 10; run `phase1-dev`):

| Benchmark (primary metric, n) | Simple RAG | Closed-book | DS-base (P6) | Δ DS-base | Δ closed-book |
|---|---|---|---|---|---|
| DeepScholar (nugget coverage, 20) | .185 | .289 | .285 | +.101 [.041, .161] | +.104 [.016, .193] |
| ResearchQA (rubric coverage, 40) | .366 | .684 | .332 | −.034 [−.104, .036] | +.318 [.238, .398] |
| DRB2 (total, 12) | .300 | .431 | .294 | −.006 [−.126, .113] | +.131 [.035, .227] |

- **The repaired DS-base on DRB2** scores .281 (.307 on its 11 valid tasks), Δ against Simple RAG
  −.019 [−.111, .073].
- **Closed-book beats Simple RAG** on all three, and the P6-only DS-base on point estimates (the
  repaired DS-base, .297, is level with it on DeepScholar). On ResearchQA the gap follows arXiv's
  coverage of each field (Simple RAG / closed-book: Health .17 / .51, Humanities .26 / .69,
  Social Sciences .35 / .74, Physical Sciences .66 / .83). In 10 of 40 answers Simple RAG says
  its sources do not address the question.
- **DS-base fails when arXiv has nothing relevant:** 16 of 40 ResearchQA tasks (Health 6 of 6)
  and 1 of 12 DRB2 tasks, where its relevance filter rejected every candidate. On the 24
  ResearchQA tasks it completes it scores .554.
- **Scope.** DeepScholar-Bench is the main benchmark and DRB2 a small transfer check. ResearchQA
  needs non-scientific web sources and is not evaluated further; these dev results are its
  record.

## The benchmark audit: is the evaluation sound?

After the controller's first runs showed almost no nugget response to more or better retrieval,
the evaluation was audited for defects. Nothing in the environment explained it.

- **Ruled out.** The task, dataset row and nugget file agree; judges see the whole report;
  nothing is dropped at export; judge calls are never cached; all 192 arXiv-linked important
  citations of dev are in the local index before their cutoff (184 come back in the top 10 for
  their own title); no gold data reaches any system.
- **Evaluator controls** (`jev check calibrate deepscholar-controls`, run `deepscholar-controls`;
  mean nugget coverage on dev):

  | Report fed to the judge | Nugget coverage |
  |---|---|
  | the paper's own related-work section | 0.979 |
  | the task's abstract alone | 0.158 (vital nuggets 0.337) |
  | another task's DS-base report | 0.012 |
  | DS-base's reports with each citation pointed at another paper it cites | 0.288 (unchanged) |

  The rotated citations keep their nugget score while cite_p falls from 0.44 to 0.11 and claim
  coverage from 0.71 to 0.29: nugget coverage reads content, the support metrics read
  citations. A report citing every important citation scores 1.0 on both reference metrics.
- **What the nuggets are.** Of the 292 dev nuggets, 28% restate the target paper's own abstract
  (70 of the 131 vital ones), 43% are field-level framing, and 28% concern specific prior work.
  DS-base covers 0.60, 0.15 and 0.13 of them respectively.
- **The search draft hurts citations.** DS-base's intro writer without the search agent's draft
  (`deepscholar_base_intro_nobg`, run `phase2-draft`) against with it: cite_p +0.144 [0.058,
  0.230], claim coverage +0.122, claim support +0.089, nugget −0.013 [−0.060, 0.033]. So part of
  any writer's reliability gain over DS-base comes from writing without that draft, which the
  controlled loop does; this diagnostic separates it from the controller's own effect.
- **Better evidence barely moves nugget coverage.** The DS-base writer, fixed to given documents,
  three seeds each (`fixed_evidence`, runs `phase2-audit-s0..s2`):

  | Writer input | Nugget | Prior-work nuggets | Reference recall |
  |---|---|---|---|
  | DS-base's 30 documents | 0.272 | 0.133 | 0.286 |
  | the same plus every important citation | 0.300 | 0.150 | 0.822 |
  | the important citations alone (19 tasks) | 0.324 | 0.206 | 0.992 |

  Adding every important citation raises reference recall by 0.536 [0.406, 0.666] but nugget
  coverage by only 0.029 [−0.009, 0.066].
- **Noise.** One writer sample plus one judgment varies with an SD of 0.078 per task at fixed
  evidence; about 0.074 of it is the writer's (a re-judging of 16 identical reports puts the
  judge's share at about 0.023). Two seeds of whole systems differ with a per-task SD of 0.104
  (run `phase1-noise`). So on 20 dev tasks only nugget differences of about 0.06 or more are
  resolvable; on 90 test tasks, about 0.02–0.03.

**Verdict.** The evaluation is sound but nugget coverage is a weak, noisy signal for learning a
search policy: much of it is earned by restating the task, and writer sampling dominates its
variance. The controller therefore keeps nugget coverage as the completeness endpoint but is
trained on a decomposed reward that also counts retrieval and citation reliability (docs/04,
docs/06). Comparisons average two writer samples per task.

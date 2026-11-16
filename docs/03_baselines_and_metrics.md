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

**Simple RAG's K.** On DeepScholar dev, K = 10 and 20 are within noise of each other and K = 5 is
lower (nugget coverage against K = 20: K = 5 −0.038 [−0.070, −0.006], K = 10 +0.011 [−0.038,
0.060]). K = 20 is the baseline. Run `baselines-june`.

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
cache (all 97 of its agent calls), so only the merge differs.

| | Candidate pool (median) | Cited sources (median) | Nugget | Org. | Ref. cov. | cite_p | Claim cov. | ‖ | Ref. recall | Per citation | Claim support |
|---|---|---|---|---|---|---|---|---|---|---|---|
| As released | 10 | 6 | 0.259 | 0.450 | 0.097 | 0.178 | 0.248 | ‖ | 0.084 | 0.356 | 0.300 |
| P6 | 148 | 22 | 0.279 | 0.700 | 0.239 | 0.287 | 0.513 | ‖ | 0.235 | 0.358 | 0.619 |

Left of ‖: the benchmark's own metrics; right: our corrected ones (see the metrics we report,
below).

**The citation bug (P10).**
- *Mechanism.* The intro writer is told to cite each document by its `_citation_number` column.
  Upstream escapes the citation guidelines together with the user text, so the placeholder never
  reaches the prompt. The writer then copies the numbers in the search background (the agent's
  draft, whose numbers index the agent's own result list) or numbers citations by order of
  appearance. Upstream resolves those numbers against the filtered document list, so most
  citations land on the wrong paper. Example: "RetroMAE and its duplex extension … [5][6]" is
  copied from the background, where 5 and 6 were the positions of RetroMAE and RetroMAE v2; in the
  writer's list they are documents 13 and 15. All 20 June backgrounds carry such numbers.
- *Repair.* Leave the guidelines unescaped, so the writer sees each document's number, and strip
  numeric citations from the background, so the writer cannot copy them.
- *Effect*, with search and screening replayed from the cache so only the writer differs:

| Contrast | Nugget | Org. | cite_p | Claim cov. | ‖ | Per citation | Claim support |
|---|---|---|---|---|---|---|---|
| agentic, P10 − P6 only | −0.012 [−0.055, 0.031] | −0.050 [−0.155, 0.055] | +0.130 [0.036, 0.224] | +0.186 [0.076, 0.296] | ‖ | +0.116 [0.016, 0.217] | +0.168 [0.034, 0.301] |
| paper protocol, June | −0.012 | −0.075 | +0.285 | +0.297 | ‖ | +0.321 | +0.348 |
| paper protocol, Nov | −0.072 | −0.100 | +0.376 | +0.448 | ‖ | +0.453 | +0.464 |

Left of ‖: the benchmark's own metrics; right: our corrected ones (see the metrics we report,
below).

- *Not the backbone's fault.* GPT-4.1 through the same harness shows the same bug and the same
  recovery (cite_p 0.059 → 0.343 on 3 tasks; docs/02).

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
deepscholar`), the code gives 0.438 and the per-citation view 0.773; the paper reports 0.674 for
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
relevant, given to the writer, cited) and the controller's evidence recall (docs/04), so rewards
and the endpoint count identically. Effect: 16 of 190 scores changed where both are defined; the
repaired DS-base drops from 0.257 to 0.246 on the 19 dev tasks that have important citations.

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

Only cited documents are evidence. Effect: every system scores higher on claim support than on
claim coverage (the repaired DS-base 0.787 against 0.699), and the repaired DS-base's gap to
Simple RAG is within noise on claim coverage but not on claim support (claim coverage −0.011
[−0.072, 0.049], claim support −0.063 [−0.120, −0.006]).

### Smaller fixes

- **Document importance.** OpenAlex lookups cut titles at `&` and `#`, and transient errors
  scored as "no match". Titles are now URL-encoded, failures retried, citation counts frozen at
  first lookup. The metric is heavy-tailed and resolves no difference on dev; it is not used.
- **Organization** (a pairwise judgment against the paper's own section) has item κ 0.48 against
  gpt-4o, on only 36 items (docs/02): our judge reproduces the official one poorly on it. It is
  reported in every table, flagged, and no decision rests on it.
- **Nugget files.** The June nugget folders are misaligned with the dataset rows (upstream issue
  16); `jev data prepare` matches them by paper id and checks each against the row's abstract.
- **Reports without arXiv citations** are still scored on nugget coverage (upstream drops them),
  so closed-book is scored. Judge and lookup failures become per-metric errors and are re-judged
  (upstream scores a failed nugget judgment as unsupported or aborts).
- **DRB2's upstream parser** corrupts compact JSON; the fix of upstream PR 8 is applied.
- **Export rules for every system:** model-written reference lists are stripped and references
  rendered uniformly; ResearchQA bodies are capped at 300 words and DRB2's at 2,000 (a CJK
  character counts 1/1.6 word).

### The metrics we report: original and corrected

| Metric | From | Problem in the released code | What we report |
|---|---|---|---|
| nugget coverage | the benchmark | none found | as is; the primary completeness measure |
| organization | the benchmark | none in the code, but our judge reproduces the official one poorly on it (κ 0.48 on only 36 items, docs/02) | as is, flagged; no decision rests on it |
| cite_p | the benchmark | averages per sentence, an uncited sentence scoring 0; the paper defines precision per citation | as is, with cite_p per citation next to it |
| claim coverage | the benchmark | checks neighbouring sentences against the claim's own citations, and adds the target paper's abstract to the evidence | as is, with claim support next to it |
| reference coverage | the benchmark | matches titles at similarity > 0.8, crediting wrong papers; scores a task without important citations 0 | as is, with reference recall next to it |
| relevance rate, document importance | the benchmark | document importance: lookup failures (repaired), heavy-tailed | not reported (judged only on request) |
| cite_p per citation, claim support, reference recall | ours | – | the corrections above, computed from the same verdicts or citations |
| U, retrieval return | ours | – | selection and training (docs/04) |

**In every results table, the benchmark's original metrics stand left of the double bar (‖) and
ours right of it.** Nugget coverage is the completeness endpoint; cite_p per citation and claim
support measure citation reliability; reference recall measures retrieval.

## Baseline results before and after

**June dev, DeepScholar-Bench** (runs `baselines-june`, `deepscholar-calibration`):

| System | Nugget | Org. | Ref. cov. | cite_p | Claim cov. | ‖ | Ref. recall | Per citation | Claim support |
|---|---|---|---|---|---|---|---|---|---|
| Authors' own reports, our evaluation (n = 2) | .283 | .000 | .150 | .438 | .709 | ‖ | – | .773 | – |
| Paper: DeepScholar-ref, Llama-4 (63 tasks) | .241 | .206 | .103 | .674 | .851 | ‖ | – | – | – |
| Paper: DeepScholar-ref, GPT-4.1 (63 tasks) | .348 | .809 | .166 | .788 | .899 | ‖ | – | – | – |
| DS-base as released | .259 | .450 | .097 | .178 | .248 | ‖ | .084 | .356 | .300 |
| DS-base, P6 | .279 | .700 | .239 | .287 | .513 | ‖ | .235 | .358 | .619 |
| DS-base, P6 + P10 | .267 | .650 | .244 | .417 | .699 | ‖ | .246 | .475 | .787 |
| Paper protocol | .219 | .400 | .115 | .176 | .335 | ‖ | .121 | .220 | .354 |
| Paper protocol + P10 | .207 | .325 | .119 | .461 | .631 | ‖ | .126 | .541 | .702 |
| Simple RAG, K = 20 | .183 | .300 | .152 | .431 | .711 | ‖ | .160 | .458 | .850 |

**Nov dev2, DeepScholar-Bench** (run `repro-nov`):

| System | Nugget | Org. | Ref. cov. | cite_p | Claim cov. | ‖ | Ref. recall | Per citation | Claim support |
|---|---|---|---|---|---|---|---|---|---|
| Paper: DeepScholar-ref, Llama-4-Scout (200 tasks) | .358 | .120 | .072 | .178 | .581 | ‖ | – | – | – |
| Paper: DeepScholar-ref, GPT-4.1 + o3 (200 tasks) | .479 | .578 | .087 | .563 | .578 | ‖ | – | – | – |
| Paper protocol | .410 | .625 | .184 | .175 | .236 | ‖ | .172 | .236 | .340 |
| Paper protocol + P10 | .338 | .525 | .191 | .551 | .684 | ‖ | .180 | .689 | .804 |
| Simple RAG, K = 20 | .280 | .575 | .160 | .520 | .754 | ‖ | .169 | .544 | .852 |

The paper reports only the benchmark's metrics; the authors' two published reports were not
scored on reference recall or claim support. Compare organization with the paper's rows with
care: our judge reproduces the official one poorly on it (docs/02).

After the repairs, the reproduction is close to the paper's ranges: on June the repaired agentic
DS-base (.267) is within the paper's .241–.348, and the paper protocol (.219, .207 with P10) just
below it; on Nov the paper protocol (.410) is within .358–.479, and just below it with P10 (.338).
After P10 on Nov, cite_p is within the published range (.551 against .178–.563) and claim
coverage above it (.684 against .578–.581); on June per-citation precision
stays below the authors' .77 (.48–.54): our writer packs several works into one sentence, so each
citation supports only part of it. GPT-4.1 shows the same level (.483), so it comes from the
pipeline.

**Choosing the baseline.** The rule, set before the runs: switch from the repository's agentic
mode to the paper's protocol if its nugget coverage is no worse (lower bound above −0.05) and its
cite_p is higher (p < .05). Unrepaired, the paper protocol is worse on nugget coverage (−0.060
[−0.114, −0.005]) and on cite_p (−0.111, p = .04). Repaired, its cite_p is not significantly
higher (+0.044, p = .45) and its nugget coverage is lower (−0.059 [−0.115, −0.003]), so it fails
the rule again. The baseline is therefore the agentic DS-base with P6 and P10, run intro-only
(`deepscholar_base_intro`): on the same search and screening it writes the same 20 reports, byte
for byte, as the full pipeline.

**Natural cost on three benchmarks** (dev; Δ against Simple RAG with K = 20; run
`baselines-june`; all are the benchmarks' own metrics, the primary one of each, and organization
on DeepScholar):

| Benchmark (metric, n) | Simple RAG | Closed-book | DS-base (P6) | Δ DS-base | Δ closed-book |
|---|---|---|---|---|---|
| DeepScholar (nugget coverage, 20) | .183 | .305 | .279 | +.096 [.034, .157] | +.122 [.058, .186] |
| DeepScholar (organization, 20) | .300 | .600 | .700 | +.400 [.205, .595] | +.300 [.080, .520] |
| ResearchQA (rubric coverage, 40) | .297 | .671 | .269 | −.028 [−.085, .030] | +.374 [.297, .452] |
| DRB2 (total, 12) | .297 | .428 | .255 | −.042 [−.198, .114] | +.131 [.071, .192] |

- **The repaired DS-base on DRB2** scores .256 (.341 on its 9 valid tasks), Δ against Simple RAG
  −.041 [−.187, .105].
- **Closed-book beats Simple RAG** on all three primary metrics, and DS-base on their point
  estimates (on DeepScholar the two are within noise); on organization DS-base leads both. On
  ResearchQA the gap follows arXiv's coverage of each field (Simple RAG / closed-book: Health
  .07 / .51, Humanities .22 / .68, Social Sciences .16 / .66, Physical Sciences .58 / .88).
- **DS-base fails when arXiv has nothing relevant:** 15 of 40 ResearchQA tasks (5 of 6 in Health
  and 5 of 6 in Social Sciences) and 3 of 12 DRB2 tasks, where its relevance filter rejected every
  candidate in all three search attempts. On the 25 ResearchQA tasks it completes it scores .430.
- **Scope.** DeepScholar-Bench is the main benchmark and DRB2 a small transfer check. ResearchQA
  needs non-scientific web sources and is not evaluated further; these dev results are its
  record.

## The benchmark audit: is the evaluation sound?

Nugget coverage responds little to more or better retrieval (below), so the evaluation was
audited for defects that could cause this. None in the environment does.

- **Ruled out.** The task, dataset row and nugget file agree; judges see the whole report;
  nothing is dropped at export; judge calls are never cached; all 192 arXiv-linked important
  citations of dev are in the local index before their cutoff (184 come back in the top 10 for
  their own title); no gold data reaches any system.
- **Evaluator controls** (`jev check calibrate deepscholar-controls`, run
  `deepscholar-controls-calibration`; mean nugget coverage on dev):

  | Report fed to the judge | Nugget coverage |
  |---|---|
  | the paper's own related-work section | 0.975 |
  | the task's abstract alone | 0.158 (vital nuggets 0.339) |
  | another task's DS-base report | 0.020 |
  | DS-base's reports with each citation pointed at another paper it cites | 0.271 (source 0.278, within noise) |

  The rotated citations keep their nugget score while cite_p falls from 0.41 to 0.12 and claim
  coverage from 0.70 to 0.37: nugget coverage reads content, the support metrics read
  citations. A report citing every important citation scores 1.0 on both reference metrics.
  Organization was not judged for these controls: it compares a report's structure with the
  paper's own section, which says nothing for the section itself, a bare abstract or a swapped
  report.
- **What the nuggets are.** Codex labelled each of the 292 dev nuggets from the task's title and
  abstract: 28% restate the target paper's own abstract (70 of the 131 vital ones), 43% are
  field-level framing, and 28% concern specific prior work. The DS-base writer on its own
  documents (below) covers 0.55, 0.12 and 0.10 of them respectively.
- **The search draft hurts citations.** DS-base's intro writer without the search agent's draft
  (`deepscholar_base_intro_nobg`, run `baselines-june`) against with it: cite_p +0.128 [0.041,
  0.216], claim coverage +0.142, claim support +0.128 [0.078, 0.178], but nugget −0.053 [−0.101,
  −0.006] and organization −0.250 [−0.471, −0.029]. The draft gives the section its structure and
  some of its nugget coverage, and spoils its citations. So a writer's reliability gain over
  DS-base, and its organization and nugget losses, can all come from writing without that draft,
  which the controlled loop does; this diagnostic shows the size of that effect, not what the
  controller's own choices add.
- **Better evidence moves nugget coverage only a little.** The DS-base writer, fixed to given
  documents, three seeds each (`fixed_evidence`, runs `evidence-s0`, `-s1`, `-s2`):

  | Writer input | Nugget | Org. | ‖ | Prior-work nuggets | Reference recall |
  |---|---|---|---|---|---|
  | DS-base's 30 documents | 0.236 | 0.342 | ‖ | 0.100 | 0.249 |
  | the same plus every important citation | 0.280 | 0.475 | ‖ | 0.177 | 0.811 |
  | the important citations alone (19 tasks) | 0.319 | 0.140 | ‖ | 0.204 | 0.997 |

  Left of ‖ the benchmark's metrics; right, ours (prior-work nuggets: nugget coverage on the
  nuggets labelled as specific prior work).

  Adding every important citation raises reference recall by 0.562 [0.437, 0.687] but nugget
  coverage by only 0.044 [0.012, 0.076].
- **Noise.** One writer sample plus one judgment varies with an SD of 0.073 per task at fixed
  evidence; about 0.062 of it is the writer's (40 identical reports judged twice put the judge's
  share at about 0.038). Two seeds of whole systems differ with a per-task SD of 0.098 (run
  `seed1`). So on 20 dev tasks only nugget differences of about 0.05 or more are resolvable; on
  90 test tasks, about 0.02.

**Verdict.** The evaluation is sound but nugget coverage is a weak, noisy signal for learning a
search policy: much of it is earned by restating the task, better evidence moves it only a little,
and writer sampling dominates its variance. The controller therefore keeps nugget coverage as the
completeness endpoint but is trained on a decomposed reward that also counts retrieval and citation
reliability (docs/04, docs/06). Comparisons average two writer samples per task.

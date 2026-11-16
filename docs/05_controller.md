# 5. The controller

The project's system: DeepScholar-base's own components (search rows, relevance screening,
ranking and the intro writer), with **which search to run next, and when to stop**, decided by a
small decision model, LiquidAI/d1-3B. The goal is DS-base's report quality at a fraction of its
cost (docs/06 states the test rule).

## Why decide retrieval

DS-base spends 62% of its cost screening and ranking about 170 candidate papers per task, and 2%
writing (docs/04). Most of its quality comes from which papers reach the writer. A controller can
save most of the cost by screening fewer, better-chosen papers, as long as it still finds the
papers the section should cite. Two facts shaped the design:

- The target paper's abstract is a strong query. Used as one BM25 query, its top 150 on the local
  index hold 51 of the 192 dev important citations; DS-base's whole candidate pool (run
  `baselines-june`) holds 58.
- Writing is cheap and its quality responds only a little to better evidence (docs/03, docs/04),
  so the money is in retrieval decisions, and the writer stays DS-base's.

## How the loop works

`systems/controlled.py` runs one task as a sequence of retrieval decisions:

```text
state ──► prepare candidates ──► hard rules (legal) ──► policy chooses ──► act ──► log
  ▲          next abstract batch       budget reserve,        fixed rule,        screen the     decisions.jsonl,
  │          one proposed query        batch cap, dedup       prompted executor  new papers     rows.jsonl,
  │          next page of a query      (one legal: forced)    or d1-3B           (sem_filter)   pages.jsonl
  └──────────────────────────────────────────────────────────────────────────────────┘
finish ──► rank the relevant papers to 30 (sem_topk, if more) ──► DS-base intro writer (P10)
```

**The actions** (at most four candidates per decision, `finish` last):

| Action | What it does | Its cost |
|---|---|---|
| `abstract` | the next 10 unseen papers of the abstract's ranked BM25 list (read up to depth 150) | screening 10 papers |
| `explore` | one new keyword query the executor proposes (from the task, the searches so far and the relevant titles), its first page of 10 | the proposal plus screening |
| `deepen` | the next page with unseen papers of the most productive keyword query (relevant per new paper), up to 3 pages deep | screening |
| `finish` | stop searching; rank and write | ranking (beyond 30 relevant papers) and writing |

Each search candidate is fetched before the decision, so the policy sees exactly which new papers
it would screen and their titles; fetching a page costs the nominal arXiv tool price. Screening
uses DS-base's own chain-of-thought `sem_filter` request for each paper, so a paired DS-base run's
cached verdicts replay.

**Limits.** 10 papers per batch, at most 6 batches (60 screened papers) per task, keyword queries
followed to at most 3 pages, at most 30 documents for the writer (DS-base's own number).

**Hard rules** (`controller/state.py`), applied before any policy sees the candidates:
- *Completion reserve.* A search is legal only if, after it, another step's overhead and the
  writing (with ranking, should every new paper be relevant) still fit the task's cap.
- *Batch cap.* No search after 6 batches.
- *Something new.* A search must bring a paper no earlier candidate brings.
- *No empty report.* `finish` waits for the first relevant paper while a search is still possible.
- *Blocked queries.* The abstract search is not offered where the access policy refuses the
  abstract as a query.
- One legal candidate is executed without asking the policy (`forced`). If the policy's own call
  used up the room for its chosen search, `finish` is executed instead (`overridden`).

The rules plan with list-price estimates, measured on a dev task with margin: screening $0.0005
per paper; writing $0.008, plus ranking at $0.0008 per relevant paper once more than 30 are
relevant (40 relevant papers reserve $0.008 + $0.032); a step's overhead $0.007. The gateway's
cap backs them up.

**What is logged.** Every decision is appended to `decisions.jsonl` once executed: a snapshot of the
state and the candidates (papers by URL into `rows.jsonl`, fetched pages counted into
`pages.jsonl`), the requested and the executed candidate, whether it was forced, overridden or taken
(a branch's, below), the policy's probabilities and its exact input (for d1: the request, the raw
answers and the adapter's hash), and the list spend before the decision, of the policy's own call
and after the action. A snapshot can be restored: a later run continues from any decision point
(`start: {run, system, batches}`) without preparing that step again, re-reading the pages the
original fetched for free, and is charged the spend up to that point. With `take: <kind>` the
restored decision executes that candidate instead of asking the policy (an error where the rules do
not offer it), logged as `taken` and never scored as a policy's choice. With `write: false` a run
stops once the writer's documents are chosen, ranking included, and writes nothing; such
retrieval-only runs are never judged. This is what counterfactual branches and RL rollouts build on
(docs/06).

**On DRB2** (research reports, docs/06) the abstract search is a prompt search: its query is the
task's research request, the prompt without its closing forbidden-source rule (which names the
source article, so the access policy refuses the whole prompt as a query). With the web among the
corpora, its list starts with one Tavily search for 20 hits (the query cut to Tavily's 400
characters) and continues down its arXiv ranked list, 10 new results a batch; a proposed keyword
query's first page comes from the web while the task has web operations left, then from arXiv;
`deepen` reads arXiv only and passes over a query whose first page came from the web. Web hits
are screened from their snippets, never read as pages. The proposer and d1 read report wording:
the instructions name the research report the task asks for, and the prompt search is "the
research request"; the action names stay. Web pages are not logged in `pages.jsonl`, so a DRB2
decision point is not restored with them.

## The policies

| Policy | How it chooses | Catalog |
|---|---|---|
| Fixed rule | abstract and keyword batches in turn for 3 batches, then finish | `ctl_fixed` |
| Prompted executor | DeepSeek answers exactly d1's prompt, with short rules of thumb added; billed as an executor call | `ctl_prompted` |
| d1 | the decision model, d1-3B as released, on its prompt (below); `finish` once P(finish) ≥ 0.5 | `ctl_d1` |

The trained versions, d1-sft and d1-rl, are in docs/06.

**What d1 reads.** One question in d1's `choice` format: a view of the state and numbered
options. The view holds the task (the official query with the abstract), the remaining budget
("$0.212 of $0.270 left"), the batch count ("2 of 6"), each search's yield ("the paper's
abstract (pages 2, new 20, relevant 13)"), the counts screened and relevant, and the latest 10
relevant titles, laid out in sections. Each option names its action, its cost and the first
titles it would screen:

```text
1. abstract   search with the paper's abstract: screen 10 new papers (about $0.005): <titles> (+5 more)
2. explore    search "graph neural network recommendation fairness": screen 9 new papers (about $0.005): ...
3. deepen     next page of "fair ranking exposure": screen 10 new papers (about $0.005): ...
4. finish     finish the search and move on to writing
```

The instruction reads: "Choose the next step of the literature search for the related-work
section of the paper in the task. Prefer the search whose new papers look most relevant to the
paper and least like the relevant papers already found. Finish only when the paper's main lines of
prior work are covered and no search is likely to add relevant papers." The prompt is in
`controller/prompt.py`. A state too long for d1's validated length (7,552 tokens) is shown with
fewer titles and previews, never without the budget or batch count.

**How d1 answers.** d1 reads the prompt once; each option's probability is the softmax over the
options of its code's best token log-probability at the answer slot. Nothing is decoded. Our
readout matches the checkpoint's own `system_one` within 0.003 (with flash attention off; it
drifts in bf16). d1 runs on the local GPU behind `jev controller serve`; each call is admitted by
the gateway and charged to the task as a nominal tool, so the cap covers the controller too.

**A training check.** LoRA on d1's language-model projections was checked on a sanity task (a
paper's primary arXiv category among four; 2,000 train and 400 dev examples, two epochs): dev
accuracy 0.785 → 0.890, NLL 0.537 → 0.290 (run `controller/sanity`).

## Results

**Validation** (9 val tasks, one writer sample; run `ctl-val`). It selects the fixed rule's batch
count: the highest retrieval return among the candidates within 0.03 U of the best.

| System | Nugget | Org. | Ref. cov. | cite_p | Claim cov. | ‖ | Ref. recall | Per citation | Claim support | U | Retrieval return | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1 | .248 | .500 | .183 | .570 | .793 | ‖ | .170 | .598 | .913 | .400 | .0043 | $.059 |
| **fixed, 3 batches** (selected) | .235 | .444 | .147 | .560 | .763 | ‖ | .135 | .633 | .884 | .392 | .0122 | $.022 |
| fixed, 6 batches | .255 | .444 | .150 | .487 | .799 | ‖ | .146 | .500 | .910 | .397 | .0031 | $.051 |
| prompted executor | .237 | .389 | .145 | .602 | .836 | ‖ | .137 | .629 | .917 | .392 | .0032 | $.047 |

Left of ‖: the benchmark's own metrics; right: our corrected ones (docs/03) and our
utility U (docs/04).

The comparisons below give every system two independent writer samples per task, averaged per
task, on 20 tasks per set; DS-base and Simple RAG replay their retrieval from the cache. The
controllers are capped at B0 = $0.27, DS-base runs at its natural cost. These are development
results (n = 20); the final test's outcome on 180 test tasks is in docs/06.

**June dev** (run `eval-dev`):

| System | Nugget | Org. | Ref. cov. | cite_p | Claim cov. | ‖ | Ref. recall | Per citation | Claim support | U | Cost (% of DS-base) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DS-base intro-only | .277 | .600 | .249 | .419 | .698 | ‖ | .248 | .480 | .788 | .367 | $.178 (100%) |
| d1 | .219 | .425 | .240 | .543 | .785 | ‖ | .246 | .566 | .883 | .379 | $.047 (26%) |
| prompted executor | .220 | .350 | .238 | .592 | .790 | ‖ | .243 | .611 | .880 | .386 | $.044 (25%) |
| fixed rule | .209 | .350 | .208 | .596 | .732 | ‖ | .212 | .696 | .872 | .378 | $.021 (12%) |
| Simple RAG (K = 20) | .185 | .250 | .152 | .455 | .720 | ‖ | .160 | .497 | .844 | .337 | $.003 (2%) |

| Δ against DS-base | Nugget | Org. | ‖ | Ref. recall | Claim support | U |
|---|---|---|---|---|---|---|
| d1 | −.058 [−.118, .001] | −.175 [−.327, −.023] | ‖ | −.002 [−.136, .132] | +.095 [.048, .143] | +.012 [−.052, .075] |
| prompted executor | −.057 [−.108, −.005] | −.250 [−.402, −.098] | ‖ | −.005 [−.147, .137] | +.092 [.041, .144] | +.019 [−.041, .078] |
| fixed rule | −.069 [−.126, −.011] | −.250 [−.411, −.089] | ‖ | −.036 [−.181, .109] | +.085 [.045, .125] | +.011 [−.053, .075] |
| Simple RAG | −.092 [−.135, −.049] | −.350 [−.504, −.196] | ‖ | −.088 [−.232, .055] | +.056 [.014, .098] | −.031 [−.086, .024] |

**Nov dev2** (run `eval-dev2`; the reproduction check of docs/03 uses the same 20 tasks):

| System | Nugget | Org. | Ref. cov. | cite_p | Claim cov. | ‖ | Ref. recall | Per citation | Claim support | U | Cost (% of DS-base) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DS-base intro-only | .403 | .887 | .268 | .373 | .697 | ‖ | .261 | .408 | .785 | .490 | $.179 (100%) |
| d1 | .392 | .575 | .293 | .539 | .817 | ‖ | .261 | .544 | .915 | .556 | $.050 (28%) |
| prompted executor | .386 | .662 | .273 | .589 | .799 | ‖ | .240 | .604 | .891 | .551 | $.045 (25%) |
| fixed rule | .394 | .537 | .211 | .587 | .771 | ‖ | .175 | .642 | .868 | .557 | $.021 (12%) |
| Simple RAG (K = 20) | .298 | .512 | .160 | .470 | .709 | ‖ | .169 | .486 | .829 | .450 | $.003 (2%) |

| Δ against DS-base | Nugget | Org. | ‖ | Ref. recall | Claim support | U |
|---|---|---|---|---|---|---|
| d1 | −.011 [−.104, .081] | −.312 [−.498, −.127] | ‖ | −.001 [−.119, .118] | +.131 [.090, .171] | +.066 [−.032, .164] |
| prompted executor | −.018 [−.117, .081] | −.225 [−.403, −.047] | ‖ | −.022 [−.144, .100] | +.106 [.055, .158] | +.061 [−.040, .162] |
| fixed rule | −.009 [−.072, .054] | −.350 [−.530, −.170] | ‖ | −.087 [−.192, .018] | +.083 [.041, .125] | +.067 [−.004, .138] |
| Simple RAG | −.106 [−.223, .012] | −.375 [−.543, −.207] | ‖ | −.093 [−.235, .049] | +.044 [−.031, .119] | −.040 [−.160, .081] |

Left of ‖: the benchmark's own metrics; right: our corrected ones (docs/03) and our
utility U (docs/04).

DS-base's most expensive Nov task cost $0.277; under the controllers' $0.27 cap it would have
produced no report.

**What the results say.**
- **Cost.** The controllers cost 12–28% of DS-base.
- **Citation reliability.** Every controller's sections are better supported than DS-base's
  (claim support +0.08 to +0.13, resolved on both sets; cite_p +0.12 to +0.22). Writing without
  DS-base's search draft alone gives +0.13 claim support on June (docs/03), enough to account for
  the gain; this does not isolate what the controllers' retrieval choices add.
- **Organization.** Every system scores below DS-base on organization, on both sets: the
  controllers by 0.18 to 0.35, Simple RAG by 0.35 to 0.38. Writing without DS-base's search
  draft alone costs DS-base 0.25 (docs/03), within the controllers' range: the draft that spoils
  DS-base's citations also gives its section its structure. Our judge reproduces the official one
  poorly on this metric (docs/02), so the size of the gap is uncertain.
- **Completeness.** On Nov the three controllers are within noise of DS-base's nugget coverage
  (d1 −0.011 [−0.104, 0.081]) and of each other (fixed − d1: +0.002 [−0.083, 0.088]; prompted −
  d1: −0.006 [−0.049, 0.036]). The intervals are wide: on one task d1's two writer samples, from
  the same evidence, cover 0.45 and 0.09 of the nuggets. On June every controller is below
  DS-base (d1 −0.058 [−0.118, 0.001]; the other two resolved), and the three are within noise of
  each other.
- **Retrieval.** d1 finds about as many important citations as DS-base on both sets (−0.002 on
  June, −0.001 on Nov) at about a quarter of its cost; the fixed rule finds fewer (fixed − d1:
  −0.034 [−0.073, 0.004] on June, −0.086 [−0.172, 0.000] on Nov). d1 uses all 6 batches on 19
  of 20 tasks on each set; it stops after 3 batches on 1 June task and after 2 on 1 Nov task.
- **U.** Every controller's difference from DS-base in U is within noise on both sets (+0.011 to
  +0.019 on June, +0.061 to +0.067 on Nov), and so are their differences from each other: their
  reliability and cost gains offset their completeness loss.
- **No controller meets the test rule** on either set (nugget lower bound above −0.02 at lower
  cost); the closest is the fixed rule on Nov (lower bound −0.072).

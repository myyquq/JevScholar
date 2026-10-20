# 5. The controller

The project's system: DeepScholar-base's own components (search rows, relevance screening,
ranking and the intro writer), with **which search to run next, and when to stop**, decided by a
small decision model, LiquidAI/d1-3B. The goal is DS-base's report quality at a fraction of its
cost (docs/06 states the test rule).

## Why decide retrieval

DS-base spends 61% of its cost screening and ranking about 150 candidate papers per task, and 2%
writing (docs/04). Most of its quality comes from which papers reach the writer. A controller can
save most of the cost by screening fewer, better-chosen papers, as long as it still finds the
papers the section should cite. Two facts from the audit (docs/03) shaped the design:

- The target paper's abstract is a strong query. Used as one BM25 query, its top 150 hold 51 of
  the 192 dev important citations; DS-base's whole candidate pool holds 61.
- Writing is cheap and its quality barely responds to better evidence, so the money is in
  retrieval decisions, and the writer stays DS-base's.

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

**What is logged.** Every decision is appended to `decisions.jsonl` once executed: a snapshot of
the state and the candidates (papers by URL into `rows.jsonl`, fetched pages counted into
`pages.jsonl`), the requested and the executed candidate, whether it was forced or overridden,
the policy's probabilities and its exact input (for d1: the request, the raw answers and the
adapter's hash), and the list spend before the decision, of the policy's own call and after the
action. A snapshot can be restored: a later run continues from any decision point (`start: {run,
system, batches}`) without preparing that step again, re-reading the pages the original fetched
for free, and is charged the spend up to that point. This is what counterfactual branches and RL
rollouts build on (docs/06).

## The policies

| Policy | How it chooses | Catalog |
|---|---|---|
| Fixed rule | abstract and keyword batches in turn for 3 batches, then finish | `ctl_fixed` |
| Prompted executor | DeepSeek answers exactly d1's prompt, with short rules of thumb added; billed as an executor call | `ctl_prompted` |
| d1 | the decision model, d1-3B as released, on its prompt (below); `finish` once P(finish) ≥ 0.5 | `ctl_d1` |

Trained versions of d1 will be called d1-sft and d1-rl (docs/06).

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
section of the paper in the task. Keep searching while searches still find new relevant papers
and the paper's main lines of prior work are not yet covered; finish once they are covered, or
once the last searches found few new relevant papers." The prompt (`controller/prompt.py`) was
chosen on the training and validation tasks, before any comparison. A state too long for d1's
validated length (7,552 tokens) is shown with fewer titles and previews, never without the
budget or batch count.

**How d1 answers.** d1 reads the prompt once; each option's probability is the softmax over the
options of its code's best token log-probability at the answer slot. Nothing is decoded. Our
readout matches the checkpoint's own `system_one` within 0.003 (with flash attention off; it
drifts in bf16). d1 runs on the local GPU behind `jev controller serve`; each call is admitted by
the gateway and charged to the task as a nominal tool, so the cap covers the controller too.

**Training is in place.** LoRA on d1's language-model projections was checked on a sanity task (a
paper's primary arXiv category among four; 2,000 train and 400 dev examples, two epochs): dev
accuracy 0.785 → 0.890, NLL 0.537 → 0.290 (run `controller/sanity`).

## How the action space was settled

The loop first had three decision points: retrieval, then synthesis (plan an outline, or write),
then review (revise, or stop). Each extra action was measured on training tasks and kept only if
it helped:

- **Revision.** On 8 train tasks, with two reviser samples each: U −0.014 [−0.039, 0.011]; 6 of
  16 revisions were identical to their drafts, whose claim support was already about 0.90.
  Removed.
- **Planning.** On all 34 train tasks, two writer samples each, plan-then-write against write on
  identical evidence: U −0.024 [−0.045, −0.003], nugget −0.019, claim support −0.027 [−0.052,
  −0.003]. Removed.
- **Batch cap.** Extending the fixed rule from 6 to 10 batches on 10 train tasks only lowered the
  retrieval return; the cap stays at 6.

So the controller makes retrieval decisions only, and `write` follows `finish`.

**The decision bank.** Behaviour policies on the 34 train tasks gave 101 logged trajectories with
608 retrieval decisions (`data/derived/bank.jsonl`), each labelled judge-free with two
surrogates: a search candidate's proxy value (0.15 × its share of new important citations −
0.10 × its screening cost / B0) and the return of stopping there (0.15 × the share of important
citations among the relevant papers so far − 0.10 × the cost so far, the policy's own calls left
out and the writer's ranking estimated, / B0; an upper bound when more than 30 papers are
relevant). They guided the choice of d1's prompt and are the starting point of its training data.

## Results

**Validation** (9 val tasks, one writer sample; run `ctl-val`). It selects the fixed rule's batch
count: the highest retrieval return among the candidates within 0.03 U of the best.

| System | U | Retrieval return | Nugget | Ref. recall | cite_p | Claim support | Cost |
|---|---|---|---|---|---|---|---|
| d1 | .444 | −.0010 | .293 | .136 | .595 | .950 | $.064 |
| **fixed, 3 batches** (selected) | .418 | .0117 | .245 | .127 | .697 | .917 | $.020 |
| fixed, 6 batches | .388 | .0022 | .241 | .139 | .565 | .898 | $.054 |
| prompted executor | .394 | .0000 | .251 | .132 | .531 | .899 | $.054 |

The comparisons below give every system two independent writer samples per task, averaged per
task, on 20 tasks per set; DS-base and Simple RAG replay their retrieval from the cache. The
controllers are capped at B0 = $0.27, DS-base runs at its natural cost. These are development
results (n = 20).

**June dev** (run `eval-dev`):

| System | Nugget | Ref. recall | cite_p | Claim support | Claim cov. | U | Cost (% of DS-base) |
|---|---|---|---|---|---|---|---|
| DS-base intro-only | .280 | .282 | .443 | .792 | .680 | .377 | $.180 (100%) |
| d1 | .210 | .241 | .585 | .905 | .798 | .376 | $.048 (26%) |
| prompted executor | .226 | .234 | .541 | .900 | .781 | .386 | $.046 (26%) |
| fixed rule | .216 | .177 | .593 | .852 | .733 | .378 | $.021 (12%) |
| Simple RAG (K = 20) | .201 | .159 | .507 | .820 | .702 | .355 | $.003 (2%) |

| Δ against DS-base | Nugget | Ref. recall | Claim support | U |
|---|---|---|---|---|
| d1 | −.070 [−.122, −.017] | −.041 [−.108, .025] | +.113 [.065, .161] | −.001 [−.058, .057] |
| prompted executor | −.054 [−.107, −.002] | −.048 [−.111, .015] | +.108 [.064, .152] | +.009 [−.048, .066] |
| fixed rule | −.064 [−.141, .013] | −.105 [−.187, −.022] | +.060 [.008, .112] | +.001 [−.077, .079] |
| Simple RAG | −.079 [−.117, −.041] | −.123 [−.201, −.045] | +.028 [−.022, .078] | −.021 [−.066, .023] |

**Nov dev2** (run `eval-dev2`; these 20 tasks had been used before, by the reproduction check and
an earlier comparison of the controllers):

| System | Nugget | Ref. recall | cite_p | Claim support | Claim cov. | U | Cost (% of DS-base) |
|---|---|---|---|---|---|---|---|
| DS-base intro-only | .427 | .319 | .411 | .781 | .680 | .520 | $.193 (100%) |
| d1 | .378 | .312 | .555 | .892 | .783 | .548 | $.051 (27%) |
| prompted executor | .413 | .276 | .590 | .902 | .805 | .584 | $.049 (26%) |
| fixed rule | .373 | .219 | .584 | .877 | .751 | .543 | $.020 (10%) |
| Simple RAG (K = 20) | .343 | .139 | .504 | .853 | .725 | .497 | $.003 (2%) |

| Δ against DS-base | Nugget | Ref. recall | Claim support | U |
|---|---|---|---|---|
| d1 | −.049 [−.113, .014] | −.006 [−.098, .086] | +.111 [.039, .182] | +.028 [−.041, .096] |
| prompted executor | −.014 [−.057, .029] | −.042 [−.150, .065] | +.121 [.059, .183] | +.063 [.015, .112] |
| fixed rule | −.054 [−.107, −.000] | −.100 [−.207, .008] | +.096 [.034, .157] | +.023 [−.038, .084] |
| Simple RAG | −.084 [−.137, −.031] | −.180 [−.334, −.026] | +.072 [−.002, .145] | −.023 [−.085, .039] |

DS-base's most expensive Nov task cost $0.287; under the controllers' $0.27 cap it would have
produced no report.

**What the results say.**
- **Cost.** The controllers cost 10–27% of DS-base.
- **Citation reliability.** Every controller's sections are better supported than DS-base's
  (claim support +0.06 to +0.12, resolved on both sets; cite_p +0.10 to +0.18). Part of this is
  writing without DS-base's search draft, which alone gives +0.089 claim support (docs/03).
- **Completeness.** Every controller is below DS-base on nugget coverage: d1 by 0.070 on June
  (resolved) and 0.049 on Nov (not resolved); the prompted executor is closest (−0.014 on Nov).
  Among the controllers the nugget differences are within noise (prompted − d1: +0.015 on June,
  +0.036 on Nov).
- **Retrieval.** d1 finds the most important citations of the three: reference recall within
  noise of DS-base on Nov (−0.006), and above the fixed rule (fixed − d1: −0.063 [−0.128, 0.002]
  on June, −0.093 [−0.167, −0.020] on Nov). It searches longer when searches keep paying (6
  batches on 16 of 20 tasks on both sets), and stops after one batch on 3 Nov tasks.
- **U** is within noise of DS-base for all three controllers except the prompted executor on Nov
  (+0.063 [0.015, 0.112]): their reliability and cost gains offset their completeness loss.
- **No controller yet meets the test rule** (nugget lower bound above −0.02 at lower cost). The
  gap to close by training is completeness: finding and using the evidence DS-base's broad
  screening finds, at a quarter of its cost.

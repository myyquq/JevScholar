# 6. Next steps

What remains: training d1 (a short supervised warm start, then bounded reinforcement learning),
the final DeepScholar test, and a small DRB2 transfer check. The schedule is at the end. Names:
**d1** is the decision model as it is now (`ctl_d1`: the released checkpoint with its prompt);
**d1-sft** and **d1-rl** are d1 after the warm start and after RL.

## Conditions of every comparison

- **Caps.** The controllers are capped at B0 per task ($0.27 on DeepScholar-Bench, $0.34 on
  DRB2) and plan within it. DS-base runs at its natural cost, under a cap it never reaches ($2;
  its most expensive task so far cost $0.45, on DRB2). U is always normalised by B0. The
  controllers' cost advantage is therefore measured against DS-base's natural cost.
- **The fixed rule** compared in every table is the one selected on val (docs/05).

## Training d1

d1 decides only retrieval: which of up to three searches to screen next, or to finish. Training
starts from d1 with its prompt and changes its weights with LoRA (`controller/d1.py`).

### Rewards

Retrieval decisions are rewarded judge-free; whole episodes by the registered utility (docs/04).

```text
U           = N + 0.15·R_c + 0.10·P + 0.10·S − 0.10·C/B0      per finished task
r_retrieval = 0.15·ΔR_e − 0.10·ΔC/B0                           per retrieval step
```

- R_e counts the important citations among the documents the writer would receive if retrieval
  ended there (with the ranking that would follow when more than 30 papers are relevant), so a
  prefetched paper that is not kept never counts. ΔC includes the step's screening, proposal,
  page fetches and policy calls.
- With potential shaping (Φ = 0.15·R_e before the end, 0 at the end), an episode's return is
  U − Φ(start), so recall is never counted twice.
- A task without gold gives no retrieval label; a failed report scores U = −1 − 0.10·C/B0.
- The weights and metric versions are frozen before training and never tuned on dev.

`jev controller trajectories RUN...` already labels the logged decisions of a train run with the
two pruning surrogates of docs/05 (`controller/data.py`). It reads the current decision format;
the historical bank stays in `data/derived/bank.jsonl`. The surrogates are never SFT targets:
labels come from executed branches.

**Data rules.** All states, branches and repeats of a task stay in one split; dev and test
trajectories never become labels; enumerated counterfactual branches are SFT data, never
on-policy RL samples; gold data enters only rewards and labels, never the state d1 reads.

### Supervised warm start (SFT)

1. **Branches.** From decision points of train runs, execute every legal candidate and continue
   each branch with the same fixed rule (`start:` restores a point; docs/05).
2. **Soft targets.** Each decision's target is a distribution over its options: candidates whose
   reward is within the noise of the best share the mass, the rest get 0. Retrieval rewards are
   deterministic given the evidence, so only exact ties share. `Example.target` holds this
   distribution; training minimises the cross-entropy against it over all options, in shuffled
   order, with each example's weight normalised over the whole optimiser step.
3. **Scale.** About 60 labelled decisions from train tasks; validation on the 9 val tasks by
   regret against the best branch, soft-target loss and calibration.
4. **Recipe.** An `sft` entry is added to `configs/training.yaml` once its labels exist; then
   `jev controller train sft`.

### Bounded RL (adapted from GRPO)

One round:
1. Restore one train decision point; sample 4 trajectories from the served adapter at
   temperature 1 (`policy: {kind: d1, sample: true}`; each decision's sampling is seeded by the
   task and the decision).
2. Log the exact inputs, option order, legal masks, actions, old probabilities and the adapter's
   hash (all in each `decisions.jsonl` record).
3. Reward: r_retrieval for retrieval-only trajectories, U for ones that write a report.
4. Advantage A = (R − group mean) / max(group SD, 0.02); skip near-tied groups.
5. Update: recompute the same option distribution; clip the chosen option's probability ratio to
   [0.8, 1.2]; add the exact categorical KL to the SFT reference with β = 0.05; one epoch per
   batch, at most two update rounds. Forced and overridden decisions stay out of the loss.
6. Stop and revert if the mean KL exceeds 0.05 nats per decision (target about 0.02).
7. Reload the adapter (`init_adapter` in a recipe); check its hash, readout parity, legal masks
   and option mapping.

Scale: a pilot of 16 trajectories (4 groups), then up to 64 if KL, rewards and action diversity
are stable, then up to 128 with retrieval-only trajectories. The pilot runs even if SFT shows no
gain; a null or negative result is recorded and SFT stays a candidate.

**Still to build.** The branch runner (execute each candidate from a restored point), the
soft-label builder from branch outcomes, the RL update (ratio, clip, KL) in `controller/d1.py`,
and an adapter reload in `jev controller serve`. The decision records, restorable points,
sampling mode, differentiable option readout and soft-target training exist.

## The final test

**Systems** (`jev run evaluate --variant test`, the first 90 Nov test tasks, two writer samples
per task): DS-base intro-only, Simple RAG, the fixed rule, d1-rl, d1-sft, and the no-draft DS-base
diagnostic. d1-sft and d1-rl are added to `evaluate.yaml` once their adapters are frozen; each adapter is served on its own port and named
by its hash (`policy: {kind: d1, url: ..., adapter: <sha256 of adapter_model.safetensors>}`).

**Which d1 carries the main claim** is chosen on June dev before the test: d1, d1-sft or d1-rl.
d1-sft and d1-rl stay in the test matrix; d1 joins it if chosen.

**Cost before test.** An action-level forecast from the dev runs' bills, then a funded matrix
frozen before generation, run in matched task chunks across arms. Six arms with two samples on
90 tasks are 1,080 reports; the campaign has about $12.9 left of its $45 cap.

**The main claim: cheaper at non-inferior quality.** The chosen d1 against DS-base, paired
per task, holds only if every 95% interval bound holds:

| Measure | Bound |
|---|---|
| nugget coverage | lower > −0.02 |
| list cost per task | upper < 0 |
| cite_p, cite_p per citation, claim support | each lower > −0.05 |
| reference recall (tasks with gold) | lower > −0.05 |

Stronger claims need more: superiority a nugget lower bound above 0; "at least half the cost" a
bootstrap upper bound on the cost ratio below 0.5. At n = 90 the nugget interval is about ±0.02
to 0.03 wide, so the result may be inconclusive; it is then reported as such, the margin is not
relaxed and no selected repeats are added.

**The RL claim.** RL must beat both SFT and the fixed rule on U (paired, Holm-corrected), and
against each also either gain content at similar cost (nugget lower bound > 0, cost ratio upper
bound ≤ 1.05) or save cost at similar content (cost upper bound < 0, nugget lower bound > −0.02),
while keeping the main claim's reliability and reference-recall margins.
A gain in U from reference recall alone does not count. If RL beats SFT but not the fixed rule,
that is what is reported.

Before the test, every identity is frozen: systems and adapters by hash, the split by its id
hash, the sample count, judges and metric versions. Judge errors are resolved before any claim.

## DRB2 transfer check

A small check that the controller transfers to broader research reports:

- **Tasks and systems.** 6 fresh DRB2 test tasks (3 en, 3 zh; `ids:` in the benchmark cell),
  chosen before any call; DS-base, Simple RAG, the fixed rule and the chosen d1.
- **Web access.** arXiv plus Tavily for every system, at most 4 web operations per task: search
  and extract share one counter, every admitted call counts once (fresh, cached or failed), a
  query the access policy blocks does not, and the fifth is refused before it is sent.
- **The controller on DRB2.** The research prompt replaces the abstract in the state. The abstract
  search becomes a prompt search: the official prompt names the forbidden source article, so the
  query is the research request with that instruction removed. Its first batch makes one Tavily
  search for 20 hits and screens them 10 at a time, then continues on arXiv; a keyword query runs
  on the web while operations remain, then on arXiv; deepen stays on arXiv. Web hits are
  screened from their snippets; pages are not extracted.
- **Still to build.** The per-task web-operation counter in the search client and the gateway,
  the controller's corpus-per-slot candidates, and report-generic prompt wording. It needs a
  Tavily key.
- **Reading.** Six tasks are a transfer diagnostic, not evidence of competitiveness. An arXiv-only
  fallback must be labelled as such.

## Schedule

| Week | Work |
|---|---|
| Oct 19–25 | Re-measure the controllers on val, June dev and Nov dev without planning, DS-base at its natural cost; build the branch runner and the RL update. |
| Oct 26–Nov 1 | SFT warm start on counterfactual labels; RL pilot of 16 trajectories. |
| Nov 2–8 | RL up to 64 trajectories if the gates pass; retrieval reward, then report utility. |
| Nov 9–15 | RL up to 128 with retrieval-only trajectories, if stable; compare d1-rl, d1-sft and d1 on val. |
| Nov 16–22 | June dev comparison of RL and SFT d1 against the frozen systems; blind checks; freeze the test controller. |
| Nov 23–29 | The final test on the first 90 Nov tasks. |
| Nov 30–Dec 6 | DRB2 transfer check; error analysis of the decisions RL changed. |

## What the project claims, and what changed

The research core is a small decision model that updates its choices as evidence arrives, and
allocates research actions under a fixed executor and budget. Controllers over frozen executors,
planner RL and group-relative policy optimisation exist in prior work; the contribution here is to
test, with the executor, tools and cost accounting held equal, whether d1's allocation of
retrieval changes the quality–cost curve, and to explain where any gain comes from by comparing a
same-component fixed rule, SFT and RL.

How the plan changed from the original proposal:

| Original proposal | Now | Why |
|---|---|---|
| a Qwen3.5-2B controller with LoRA | d1-3B with LoRA | d1's readout and LoRA training are verified; choosing among candidates is its purpose |
| training data from Multi-XScience, MS² and tasks across all three benchmarks | DeepScholar June train (34) and val (9) | same domain as the main test; counterfactual labels fit the budget |
| mixed executors (DeepSeek and a local Qwen3.8-27B) | DeepSeek-V4.1-Flash only | one controlled variable; budget and GPU |
| a black-box-optimised workflow as a comparison | removed; a fixed rule and the prompted executor instead | budget, and clearer attribution |
| actions search / read / write / revise | abstract search, keyword search, deepen, finish | each action kept only if measured useful (docs/05) |
| reward: final report coverage | the decomposed utility U plus a judge-free retrieval reward | nugget coverage is noisy and barely responds to retrieval (docs/03) |
| success: higher quality than the strongest baseline at equal cost | lower cost at non-inferior quality under the shared cap; superiority is a separate, stronger claim | the evidence shows large cost savings; quality differences are unresolved |
| a final test on all three benchmarks | DeepScholar-Bench; DRB2 as a small web transfer check; ResearchQA's dev record only | budget |

## Limitations

Evidence is abstracts only; there is one executor; the judges share the executor's model family;
June and Nov tasks differ in distribution; the training set is small (34 tasks); the DRB2 check
is small.

## Risks

| Risk | Response |
|---|---|
| Action differences are smaller than writer noise | two writer samples per task; soft targets and ties; a judge-free retrieval reward |
| Cached outputs mistaken for independent samples | separate replay namespaces per writer sample |
| Retrieval reward improves while reports do not | U keeps nugget coverage and reference recall; the claims need report-level evidence |
| A fixed component explains the improvement | the same-component fixed rule and the no-draft DS-base diagnostic |
| Same-family judging prefers a style | scores labelled non-official; key gains blind-checked |
| n = 90 cannot settle the margin | an inconclusive result is reported as such |
| The campaign budget ($45 billed in all, about $13 left) | every run has its own cap; cheap labels first; the test and its baselines before optional diagnostics |

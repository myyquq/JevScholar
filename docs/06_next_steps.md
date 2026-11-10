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
   each branch with the same fixed rule (`start:` restores a point, `take:` executes the given
   candidate there; docs/05). `jev controller branches RUN` does this for every decision the
   rules left a choice, in finished episodes of tasks with important citations: one run per
   (batches before the point, candidate kind), `runs/branches-<RUN>-b<batches>-<kind>`, holding
   only the tasks whose point offers that kind. Each branch keeps the source system's cap, rules,
   costs and actions, and the source's executor, search and seed, so it is offered exactly what
   the source was. Only train and val runs are sources (`controller/branches.py`).
2. **Soft targets.** Each decision's target is a distribution over its options: candidates whose
   reward is within the noise of the best share the mass, the rest get 0. Retrieval rewards are
   deterministic given the evidence, so only exact ties share. `Example.target` holds this
   distribution; training minimises the cross-entropy against it over all options, in shuffled
   order, with each example's weight normalised over the whole optimiser step. A branch's
   reward is its retrieval return at the end: the important citations among the writer's
   documents, less the task's whole list spend. `jev controller labels RUN... --out FILE` writes
   the examples (what d1 reads at the point, under its default profile) and, beside them,
   `FILE.points.jsonl` with each point's options and returns; a point with a missing branch is
   left out and counted.
3. **Data.** 135 labelled decisions from the 34 train tasks (`runs/rollouts-train`, fixed rule;
   438 branches) and 36 from the 9 val tasks (`runs/rollouts-val`; 117 branches), in
   `data/derived/controller/sft/train.jsonl` and `dev.jsonl`.
4. **Recipe and result.** `sft` in `configs/training.yaml` (`jev controller train sft`): LoRA
   r 16, lr 2e-4, 2 epochs, the sanity check's settings. Val: soft-target CE 0.980 → 0.954,
   accuracy 58% → 58%, ECE 0.122 → 0.095, a small gain; 4 epochs overfit (val CE 1.15). The
   adapter is `runs/controller/sft/adapter` (d1-sft).

### Bounded RL (adapted from GRPO)

**Design.** RL samples only where a choice changes the outcome, and only that choice:
1. **Points.** Train decision points (`rollouts-train`) whose options' branch returns differ by
   more than 0.01: 46 of the 135. At the others every option ends within about one batch's cost
   (0.002) of the rest, since a batch of 10 rarely holds an important citation; they carry no
   signal.
2. **Groups.** Each point is restored and sampled by 8 members (g0..g7): d1 with the SFT adapter
   chooses the point's option at temperature 1, and the fixed rule continues, as the branches
   do (`policy: {kind: first, first: {kind: d1, sample: true, adapter: <sha256>}, then:
   <fixed>}`). A member's return then differs only by that one choice, and its continuation
   replays its branch's executor calls. Sampling is seeded by the run, system, task and
   decision. These are new on-policy samples; the branches themselves never become RL data.
3. **Reward and advantage.** The retrieval return at the end (as for SFT labels);
   A = (R − group mean) / max(group SD, 0.02); groups whose returns span less than 0.001 are
   skipped.
4. **Update.** One epoch in 4 optimiser steps, dropout off: −min(r·A, clip(r, 0.8, 1.2)·A) +
   0.05·KL(π‖π_ref), with the exact categorical KL to the frozen SFT adapter. π_old is
   recomputed with the training readout before the first step (the served one-row readout
   differs from it by up to 0.03). Reverted, nothing saved, if the mean KL exceeds 0.05 nats
   per decision. lr 1e-5; one round.

Sampling every decision at points picked without regard to their spread was tried first: it
moved d1's probabilities but not its episodes on val.

**Run.** `sbatch scripts/rl_round.sh` (its defaults are this design): serve the SFT adapter,
`jev controller groups rollouts-train --first-only --min-spread 0.01 --per-task 0 --groups 64
--size 8`, `jev controller rl-data`, `jev controller rl-update`. Code: `controller/groups.py`,
`rl_update` in `controller/d1.py`, `FirstThen` in `controller/policy.py`.

**Result** (`runs/controller/rl/first/r1/adapter`, d1-rl): 368 trajectories, 45 groups and 360
decisions used, $0 billed (all replayed). Median within-group spread 0.018, mean |A| 0.40. Mean
KL to SFT 0.039, clip fraction 0.64.

**On val** (`scripts/val_compare.sh`; judge-free, val never becomes RL data): every val task run
from the start with each adapter choosing, scored by the retrieval return and paired per task
with the fixed rule's own val run; and on the 36 branched val points, each adapter's regret
against the best branch.

| System (9 val tasks) | Return | Δ vs fixed (± SE) | Recall | Cost | Batches |
|---|---|---|---|---|---|
| fixed rule | .0102 | — | .121 | $.022 | 3.0 |
| d1-sft | .0084 | −.0018 ± .0026 | .141 | $.035 | 3.6 |
| **d1-rl** | **.0129** | +.0027 ± .0077 | **.220** | $.054 | 5.2 |

d1-rl searches longer than d1-sft (it finishes at 8% of the val points, d1-sft at 28%) and finds
more important citations on 3 of the 4 tasks where its episodes differ; at the 14 val points
whose options' returns differ, its regret is .0068 (d1-sft .0074). This is the completeness gap
docs/05 names, at about 1.5 times d1-sft's cost (about 30% of DS-base's). With 9 tasks it is not
resolved: the choice between d1-sft and d1-rl is made on June dev (below).

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

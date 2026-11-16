# 6. Training d1, the final test and DraftRef

This document covers training d1 (a short supervised warm start, then bounded reinforcement
learning) and its val gate, the final DeepScholar test, DraftRef (DS-base writing from its
search draft's references), and a small DRB2 transfer check. Names: **d1** is the untrained
decision model (`ctl_d1`: the released checkpoint with its prompt); **d1-sft** and **d1-rl** are
d1 after the warm start and after RL.

## Conditions of every comparison

- **Caps.** The controllers are capped at B0 per task ($0.27 on DeepScholar-Bench, $0.67 on
  DRB2) and plan within it. DS-base runs at its natural cost, under a cap it never reaches ($2;
  its most expensive task cost $1.50, a DRB2 task it failed). U is always normalised by B0. The
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

**Data rules.** All states, branches and repeats of a task stay in one split; dev and test
trajectories never become labels; enumerated counterfactual branches are SFT data, never
on-policy RL samples; gold data enters only rewards and labels, never the state d1 reads.

### Supervised warm start (SFT)

1. **Where labels come from.** The source of the train labels is untrained d1's own run on the
   34 train tasks (`ctl-train-d1`, variant `rollouts_d1`), so the states d1 learns from are the
   states it visits, late batches included. The val labels come from d1's episodes in `ctl-val`.
   Each task gives two decision points by a fixed pseudo-random pick, one early (after 0–2
   batches) and one late (after 3–5), or the one it has (`--stratified`): about 68 train and 18
   val points.
2. **Branches.** At each point every legal candidate is executed (`start:` restores the point,
   `take:` executes the given candidate; docs/05). Each branch is then continued by untrained d1
   itself, deciding as deployed (`finish` at P ≥ 0.5). So a point's branches estimate d1's own
   action values there, and a label is one policy-improvement step over d1 at its own horizon.
   A branch stops once the writer's documents are chosen, ranking included (`write: false`); it
   writes nothing and is never judged, since its reward needs the documents, not the section.
   `jev controller branches RUN --system ctl_d1 --stratified` runs one experiment per (batches
   before the point, candidate kind), `runs/branches-<RUN>-b<batches>-<kind>`. Each holds only
   the tasks whose point offers that kind. Each branch keeps the source system's cap, rules,
   costs and actions, and the source's executor, search and seed, so it is offered exactly what
   the source was. Only train and val runs are sources, and only if every reply they got is of
   the pinned model version (docs/02). A branch that restored another snapshot than its point is
   an error. A failed branch leaves its point out, counted (`controller/branches.py`).
3. **Soft targets.** A branch's reward is its retrieval return at the end: the important
   citations among the writer's documents, less the task's whole list spend. Each option's target
   share falls with its distance from the best return: q(a) ∝ exp(−(R_best − R_a)/τ), with
   τ = 0.005. This τ is a smoothing choice fixed before any label exists, not a measured noise
   level. One gold paper of ten is worth 0.015, so 0.005 is about a third of one. Training
   minimises the cross-entropy against the target over all options, in shuffled order, with each
   example's weight normalised over the whole optimiser step. `jev controller labels RUN --system
   ctl_d1 --stratified --out FILE` recomputes the points `branches` planned with the same
   options and writes the examples (what d1 reads at the point, under its default profile) and,
   beside them, `FILE.points.jsonl` with each point's options, returns and target; every planned
   point is labelled or listed as left out.
4. **The decoder.** Every d1, trained or not, decodes the same way (`Profile.pick`): `finish`
   once P(finish) ≥ 0.5, otherwise the first most probable search. `labels` reports how often
   this decoder, applied to the target, misses the best option.
5. **Recipe** (`sft` in `configs/training.yaml`; `jev controller train sft`). It reads
   `data/derived/controller/sft/train.jsonl` (train branches) and `dev.jsonl` (val branches).
   AdamW, learning rate 5e-5, no weight decay, one warmup step then constant, one forward per
   optimiser step (`batch_tokens` 8192), two epochs, seed 0, gradient norm clipped at 1.0.
6. **The checkpoint.** An adapter is kept before training and after every optimiser step. On the
   val points, each is scored by the regret of the deployed choice (the best return less the
   return of the option the decoder picks), with d1 read one row at a time as it is served, then
   by the soft-target cross-entropy. d1-sft is the step with the lowest regret, then the lowest
   cross-entropy, then the earliest. That one adapter is d1-sft; no other step is tried later.

### Bounded RL (adapted from GRPO)

1. **Groups.** Eight late decision points of `ctl-train-d1`, one per task: after 3–5 batches,
   with at least two search options, picked by a fixed hash among the eligible points. Their ids
   are written into the `rl1` configuration before any rollout. From each point, four
   trajectories are sampled from the served d1-sft adapter at temperature 1 (`policy: {kind: d1,
   sample: true}`), each under its own system name, so each decision's sampling seed (system,
   task, decision) differs. That makes 32 trajectories, which stop at the writer's documents.
2. **Logs.** Each `decisions.jsonl` record holds the exact inputs, option order, legal options,
   actions, old probabilities and the adapter's hash.
3. **Reward and advantage.** R is the retrieval return. A = (R − group mean) / max(group SD,
   0.02), with SD over the group's four (ddof 0). A group whose returns span less than 0.001
   trains nothing, and neither does a group with a failed trajectory; both are recorded, and
   their finished trajectories' decisions still count in the KL stop below. The rollout runs
   must hold exactly the recipe's points and members, each restored from its source point. With
   no group left, there is no update and d1-rl is d1-sft.
4. **Update** (`rl1` in `configs/training.yaml`; `jev controller rl rl1`). Four optimiser steps,
   each over two groups (eight trajectories, in an order fixed by seed 0). AdamW, constant
   learning rate 2e-5. Each decision's option distribution is recomputed in the logged order and
   rendering, one row at a time, with dropout off, and checked against the logged probabilities
   (within 0.005). The surrogate −min(rA, clip(r, 0.8, 1.2)·A), r the sampled option's
   probability over its logged sampling probability, is summed over a trajectory's own
   decisions and averaged over the step's trajectories; β = 0.05 times the exact categorical
   KL(π ‖ π_d1-sft) is averaged per decision. Forced, overridden and taken decisions stay out of
   the loss.
5. **KL stop.** After every step an adapter is kept and the mean KL to d1-sft is measured on all
   sampled decisions of the 32 trajectories. Training stops at the first step above 0.05 nats
   per decision; d1-rl is the last step within it. A rejected or failed update leaves the
   previous output as it was.
6. **Reload.** `jev controller serve --adapter DIR` is restarted on the new adapter; its hash,
   readout parity, legal options and option mapping are checked.

**Commands.** `jev controller rl-points ctl-train-d1` gives the points (frozen in the `rl1`
recipe of `configs/training.yaml`); `jev controller rollouts rl1 --adapter <sha>` samples the
groups (`runs/rl1-b<batches>`); `jev controller rl-data rl1` writes their decisions with the
advantages, and `jev controller rl rl1` makes the update (`controller/rl.py`, `controller/d1.py`).

**Outcome** (`runs/controller/rl1`). Seven of the eight groups trained. In the eighth, all four
trajectories finished at once and tied, so the fourth step had one group. All four steps stayed
within the KL stop (0.047 nats per decision after the fourth), so d1-rl is the fourth step. Its
deployed regret on the val points equals d1-sft's (0.0034).

## Which d1 carries the main claim

d1, d1-sft and d1-rl run on the 9 val tasks with two writer samples each (`jev run controller
--variant gate`; d1's episodes replay `ctl-val`'s from the cache). Retrieval measures are read
once per task from the episode's final state. A trained d1 passes if, against d1 on the same tasks:

- its mean writer-document R_e is at least d1's;
- at most 2 of the 9 tasks lose more than 0.10 of R_e;
- it averages at least 3 batches, and no task stops before 2;
- its deployed regret on the val points (each arm read one row at a time) is not above d1's;
- every writer sample of every task ends `ok` within the cap.

The main claim's d1 is d1-rl if it passes, otherwise d1-sft if it passes, otherwise d1
(`jev controller gate RUN --regret SYSTEM=FILE ...`, each FILE the output of `jev controller eval
dev.jsonl --points dev.points.jsonl [--adapter DIR]`).

**Outcome** (run `ctl-gate`). Neither trained d1 passes: d1-sft and d1-rl finish after one batch
on every val task and keep fewer important citations (mean R_e 0.079 and 0.062, against d1's
0.174), although their deployed regret on the val points is lower (0.0034 against 0.0084). The
main claim's d1 is d1. Every d1 arm is frozen before any test task is run (d1-sft and d1-rl by
their adapters' hashes). This is a screening rule on 9 tasks, not evidence of non-inferiority.

## The final test

**Systems.** Two writer samples per task throughout.
- `jev run evaluate --variant test` runs the first 90 Nov test tasks: DS-base intro-only, Simple
  RAG, the fixed rule (3 batches), the fixed rule at 6 batches, d1, d1-sft, d1-rl and the
  no-draft DS-base diagnostic. `--variant test-closed` adds a closed-book reference row (one
  sample).
- `--variant test-ext` runs the other 90 test tasks (`ids:`): DS-base intro-only and the main
  claim's d1.
- d1-sft and d1-rl are in the catalog (`ctl_d1_sft`, `ctl_d1_rl`), each served on its own port
  and named by its adapter's hash (`policy: {kind: d1, url: ..., adapter: <sha256 of
  adapter_model.safetensors>}`).

**What each contrast measures.**
- d1 against d1-sft and d1-rl: the effect of training, on the common 90 tasks, by the paired
  retrieval return, R_e, cost and batches.
- d1 against the fixed rule at 6 batches: d1's choices at the same batch limit.
- d1 against the fixed rule at 3 batches: the whole allocation-and-stopping policy.
- The no-draft DS-base against DS-base: the draft's effect within DS-base only.
- The fixed rule at 6 batches and closed-book are descriptive. The cost–quality frontier of all
  arms is the main figure.

**Size.** Eight arms with two samples on the first 90 tasks, two arms with two on the other 90
and the closed-book row with one: 1,890 reports.

**How the test is run.**
- From a clean, frozen commit.
- In chunks of 30 tasks (`limit` 30, 60, 90 on one run id, the same arm list each time);
  `test-ext` in one go.
- Before every paid run: a one-call model-version probe and the d1 servers' `/info` against their
  adapters' hashes.
- An infrastructure error (the d1 server, the network, a server error) is rerun at most twice
  (`jev run ... --rerun-infra`); the earlier attempts are kept (`result.failed-<n>.json`) and
  their costs counted in the ledger. `degraded` results keep the runner's single
  automatic retry, then stand. Other failures stand. Failures are counted per writer sample.
- If the model version changes: pause, then either the pinned version returns or the complete
  matrix restarts under a newly frozen version, with a fresh run and cache for generation and
  judging. Versions are never pooled.

`jev claims eval-test eval-test-ext` computes both registered claims below from the two runs.

**The main claim: cheaper at non-inferior quality.** The main claim's d1 against DS-base, paired
per task on all 180 tasks (writer samples averaged within a task, failures kept, reference recall
on the tasks with gold), holds only if every 95% interval bound holds:

| Measure | Metric from | Bound |
|---|---|---|
| nugget coverage | the benchmark | lower > −0.02 |
| list cost per task | – | upper < 0 |
| cite_p | the benchmark | lower > −0.05 |
| cite_p per citation | ours (docs/03) | lower > −0.05 |
| claim support | ours (docs/03) | lower > −0.05 |
| reference recall (tasks with gold) | ours (docs/03) | lower > −0.05 |

Every metric of docs/03 is reported for every arm, organization included; organization enters no
claim, as our judge reproduces the official one poorly on it (docs/02).

Stronger claims need more: superiority a nugget lower bound above 0; "at least half the cost" a
bootstrap upper bound on the cost ratio below 0.5. The outcome is reported as it is: the margin is
not relaxed and no tasks or repeats are added.

**The RL claim** (secondary; it cannot stand in for the main claim). On the common 90 tasks, RL
must beat both SFT and the fixed rule on U (paired, Holm-corrected), and
against each also either gain content at similar cost (nugget lower bound > 0, cost ratio upper
bound ≤ 1.05) or save cost at similar content (cost upper bound < 0, nugget lower bound > −0.02),
while keeping the main claim's reliability and reference-recall margins.
A gain in U from reference recall alone does not count. If RL beats SFT but not the fixed rule,
that is what is reported.

Before the test, every identity is frozen: systems and adapters by hash, the task lists by their
id hashes, the sample count, judges and metric versions. Judge errors are resolved before any
claim.

**Outcome** (runs `eval-test`, `eval-test-ext`, `eval-test-closed`). All 1,890 reports ended `ok`.
The main claim fails. d1's nugget coverage interval lies wholly below 0, so d1 is resolved worse
than DS-base on it, not merely short of the margin; its reference recall also misses the margin.

| d1 − DS-base (180 tasks) | Difference [95% interval] | Holds |
|---|---|---|
| nugget coverage | −.053 [−.073, −.033] | no |
| list cost per task | −.137 [−.142, −.132] | yes |
| cite_p | +.129 [.109, .150] | yes |
| cite_p per citation | +.073 [.047, .098] | yes |
| claim support | +.122 [.106, .138] | yes |
| reference recall (168 tasks with gold) | −.031 [−.062, .000] | no |

The RL claim fails: d1-rl's U does not exceed d1-sft's (−0.006, p = .49) or the fixed rule's
(−0.042, p = .008), and its reference recall misses the main claim's margin (−0.136 [−0.188,
−0.085] against DS-base).

The first 90 tasks (closed-book one writer sample; columns as in docs/05):

| System | Nugget | Org. | Ref. cov. | cite_p | Claim cov. | ‖ | Ref. recall | Per citation | Claim support | U | Cost (% of DS-base) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DS-base intro-only | .420 | .678 | .232 | .387 | .673 | ‖ | .240 | .442 | .772 | .502 | $.183 (100%) |
| Simple RAG (K = 20) | .357 | .344 | .089 | .470 | .710 | ‖ | .093 | .516 | .830 | .499 | $.003 (2%) |
| fixed, 3 batches | .362 | .464 | .166 | .582 | .765 | ‖ | .176 | .634 | .871 | .524 | $.021 (11%) |
| fixed, 6 batches | .356 | .469 | .200 | .512 | .784 | ‖ | .212 | .513 | .884 | .506 | $.050 (27%) |
| d1 | .376 | .475 | .217 | .527 | .802 | ‖ | .230 | .518 | .898 | .531 | $.052 (29%) |
| d1-sft | .333 | .333 | .108 | .607 | .685 | ‖ | .116 | .778 | .819 | .487 | $.011 (6%) |
| d1-rl | .329 | .342 | .096 | .605 | .671 | ‖ | .104 | .786 | .809 | .482 | $.009 (5%) |
| no-draft DS-base | .377 | .522 | .228 | .519 | .811 | ‖ | .236 | .496 | .891 | .483 | $.182 (99%) |
| closed-book | .366 | .422 | .000 | .000 | .000 | ‖ | .000 | .000 | .000 | .365 | $.001 (0.4%) |

- **Cost.** Over the 180 tasks d1 costs 28% of DS-base ($0.053 against $0.190 per task).
- **Completeness.** On the first 90 tasks every arm covers fewer nuggets than DS-base (every
  interval wholly below 0). Closed-book, which retrieves and cites nothing, covers .366, within the
  range of the controllers and Simple RAG (.329 to .376).
- **The draft.** DS-base writing without its search draft keeps its retrieval (reference recall
  −0.004 [−0.018, 0.010]), loses as much nugget coverage as d1 (−0.044 [−0.075, −0.012]; d1
  −0.044 [−0.072, −0.017]) and gains about as much citation reliability (cite_p +0.132, claim
  support +0.119; d1 +0.140, +0.126).
- **Training.** d1-sft and d1-rl are the cheapest controllers (5–6% of DS-base), with the lowest
  nugget coverage of all arms (.333, .329) and about half of d1's reference recall (.116, .104).
- **U.** d1's U is above DS-base's on the first 90 tasks (+0.029 [0.001, 0.058]), not on the other
  90 (+0.008 [−0.023, 0.039]).

## DraftRef: DS-base writing from its draft's references

**The idea.** DS-base's search agent ends with a related-work draft. The draft names, by arXiv
id, the papers it was written from. DS-base gives the draft to its writer as background, but
picks the writer's documents with its own relevance filter, which keeps only about half of the
papers the draft names. DraftRef gives the writer the draft's own references instead: the
candidate papers whose arXiv ids appear in the draft, in the draft's order. d1 replaces DS-base's
filter, so the filter's executor calls are gone; d1 only tops up the list when the draft names
fewer than 30 papers. The search agent, the draft and the writer stay DS-base's.

**How it runs** (means per task over the 180 test tasks):

| Step | DS-base (`deepscholar_base_intro`) | DraftRef (`dsb_d1_dr`) |
|---|---|---|
| Configuration | the repaired intro-only DS-base (docs/03) | the same executor, search settings and seed; d1-3B (as released) replaces the relevance filter at threshold 0.2942, and the draft's references pick the writer's documents (`screen`, `draft_docs`) |
| Search agent | one agentic search: about 20 arXiv queries and 37 abstracts read; the candidate pool is every search hit, about 177 papers | the same, replayed from DS-base's cache: the pool is identical on all 180 tasks |
| Draft | the agent's last message: a related-work draft of about 810 words naming about 34 pool papers by arXiv id; the writer gets it as background, numeric citations stripped (docs/03) | the same draft, used the same way; its arXiv ids also pick the writer's documents |
| Filter → writer documents | `sem_filter` judges every pool paper and `sem_topk` ranks those it passes down to 30: about 462 executor calls, 62% of DS-base's cost. The 30 hold 16 of the draft's 34 papers (48%) | d1 reads each pool paper once and gives its probability of being relevant; no executor call. The documents are the pool papers whose arXiv ids the draft names, whatever d1 says, then the papers d1 keeps, most probable first, up to 30 or the number named if that is larger: about 35 documents, 96% of them named by the draft. d1 adds papers only on the 43 tasks whose draft names fewer than 30. Old-style ids (such as `cond-mat/0405116`) are not matched |
| Writer | DS-base's intro writer, one executor call per sample; two samples | the same writer and prompt; four samples. The named documents are numbered in the draft's order |
| Rendering | numeric citations become references; only cited documents are kept. It cites 23 of its 30 documents | the same. It cites 34 of its 35 |

The ablation **DS-base with d1's filter** (`dsb_d1_30`) differs from DraftRef only in the filter
row: its documents are up to 30 papers whose d1 relevance probability is at least 0.2942, most
probable first, with no draft rule. They hold 14 of the draft's 34 papers (41%).

**Submission.** DraftRef writes four samples per task from the same documents and submits one.
The choice reads the text alone (`analysis.writer_pick`). A sample's score is the share of the
target abstract's content words it uses, divided by 0.0284, minus the share of its long
sentences (50 characters or more) without a citation, divided by 0.0454. The highest score is
submitted; a tie goes to the earliest sample. All four writes are charged. DS-base's score is
the mean of its two samples, as in the final test.

**The claim.** DraftRef's submission against DS-base, paired over all 180 test tasks: it must cost
less, and be non-inferior on nugget coverage and on reference recall at a margin of 0.02. Each
part is a two-sided sign-flip test (for non-inferiority, of the difference plus the margin). The
claim holds if the largest of the three p-values is below .05. Reference recall counts 0 for
both systems on the 12 tasks without important citations. cite_p and claim support are
secondary.

**Running it.** `jev run evaluate --variant dsb-dr-test --run-id dsb-dr-test` with d1 served
(`--variant dsb-d1-test` for the ablation); both replay their search from `eval-test` and
`eval-test-ext`. `jev bestof dsb-dr-test dsb_d1_dr --baseline eval-test --baseline eval-test-ext`
computes the comparison.

**Outcome** (runs `dsb-dr-test`, `dsb-d1-test`; DS-base from `eval-test`, `eval-test-ext`). The
claim holds (p = .0066, from nugget coverage). DraftRef costs 41% of DS-base ($0.078 against
$0.190 per task).

| DraftRef − DS-base (180 tasks) | Difference [95% interval] | p | Holds |
|---|---|---|---|
| list cost per task | −.111 [−.115, −.107] | < .0001 | yes |
| nugget coverage (margin −.02) | +.007 [−.013, .027] | .0066 | yes |
| reference recall (margin −.02) | +.042 [.023, .061] | < .0001 | yes |
| cite_p (secondary, margin −.02) | −.001 [−.018, .016] | .027 | – |
| claim support (secondary, margin −.02) | +.063 [.050, .076] | < .0001 | – |

| System | Nugget | Org. | Ref. cov. | cite_p | Claim cov. | ‖ | Ref. recall | Per citation | Claim support | U | Cost (% of DS-base) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DS-base intro-only (mean of 2) | .436 | .689 | .216 | .400 | .683 | ‖ | .211 | .454 | .776 | .515 | $.190 (100%) |
| DS-base with d1's filter (best of 4) | .424 | – | – | .438 | – | ‖ | .191 | .474 | .790 | .547 | $.078 (41%) |
| DraftRef (best of 4) | .443 | .750 | .260 | .399 | .755 | ‖ | .253 | .416 | .839 | .576 | $.078 (41%) |
| DraftRef with the citation rule (best of 4) | .448 | .722 | .261 | .712 | .815 | ‖ | .254 | .773 | .875 | .617 | $.075 (40%) |

Left of ‖: the benchmark's own metrics; right: our corrected ones (docs/03) and our utility U
(docs/04).
The ablation was scored on nugget coverage, cite_p, claim support and reference recall only; –
marks a metric not scored. Reference recall counts 0 on the 12 tasks without important
citations, as U does; over the 168 tasks with them it is .226, .204, .271 and .272. A best-of-4 cost
includes all four writes. The citation rule is described at the end of this section.

| Δ [95% interval] (180 tasks) | Nugget | cite_p | ‖ | Ref. recall | Claim support | Cost per task |
|---|---|---|---|---|---|---|
| DraftRef − DS-base, as submitted | +.007 [−.013, .027] | −.001 [−.018, .016] | ‖ | +.042 [.023, .061] | +.063 [.050, .076] | −.111 [−.115, −.107] |
| DraftRef − DS-base, samples averaged | +.004 [−.012, .019] | −.024 [−.038, −.009] | ‖ | +.043 [.024, .062] | +.051 [.039, .063] | – |
| DraftRef − DS-base with d1's filter, both best of 4 | +.019 [−.001, .039] | −.038 [−.059, −.017] | ‖ | +.062 [.039, .085] | +.049 [.033, .065] | +.0007 [.0005, .0008] |

Samples averaged: DraftRef's four samples and DS-base's two, each averaged within a task, so
neither side is selected; cost is left out of this row.

**What the results say.**

- **Cost.** The saving is DS-base's filter: about 462 executor calls and $0.1186 per task, 62% of
  DS-base's cost. d1 adds $0.0020 per task at its nominal price, and DraftRef's four writes add
  $0.0052 over DS-base's mean write cost (writes after the first are mostly served from the
  provider's prompt cache). The net saving is $0.1114. The search agent and its arXiv calls cost
  the same $0.0687 per task in both systems and are most of what DraftRef costs. DS-base with
  d1's filter costs $0.0777, DraftRef $0.0784.
- **Reference recall.** Where the important citations go (168 tasks with them, 6.0 per task):

  | Important citations per task | DS-base | with d1's filter | DraftRef |
  |---|---|---|---|
  | in the candidate pool | 1.80 | 1.80 | 1.80 |
  | in the pool and named by the draft | 1.58 | 1.58 | 1.58 |
  | among the writer's documents | 1.32 | 1.20 | 1.60 |
  | cited by the submitted section | 1.28 | 1.17 | 1.58 |

  70% of the important citations never reach the pool, in every system: that is the search
  agent's limit. The rest is decided at the filter. DS-base's filter drops 0.48 of the 1.80 per
  task, 0.39 of them papers the draft names. DraftRef keeps every pool paper the draft names, so
  it drops only 0.20, none of them named. Across tasks, the writers cite 97–99% of the important
  citations among their documents and never cite a paper outside them. So the gain is the
  important citations the draft named and DS-base's filter dropped: they add +.054. The few that
  DS-base's 30 held and DraftRef's documents do not cost −.010, and small writer differences
  −.001. The gain sits in the 47 of 168 tasks where the draft named such a paper (+.14 with one,
  +.30 with two). In the other 121 tasks DraftRef is slightly lower (−.015).
- **Nugget coverage.** +.007 [−.013, .027], and +.004 with samples averaged. This is within the
  .02 margin; it shows neither equal quality nor an improvement. All the writers rewrite the same
  draft, and without it DS-base loses .044 of nugget coverage (the final test, above). DraftRef
  cites about ten more papers per section at about the same length (table below), without
  covering measurably more nuggets.
- **cite_p.** The submitted −.001 is parity only because of the selection. With samples averaged
  DraftRef is −.024 [−.038, −.009]; with each side's best of 2 it is −.037 [−.056, −.019]. The
  submission rule favours sections with few uncited sentences, which raises DraftRef's cite_p by
  .023; DS-base's mean gets no such lift. The cause is bigger citation bundles. cite_p scores a
  sentence by the share of its citations that support it, so a sentence citing three papers,
  one of which supports it, scores .33. DraftRef's documents are the draft's reference list, and
  its writer rebuilds the draft's citation groups although their numbers were stripped (35% of
  the draft's own sentences cite three or more papers):

  | Per submitted section | DS-base | with d1's filter | DraftRef | with the citation rule |
  |---|---|---|---|---|
  | words | 426 | 449 | 457 | 503 |
  | papers cited | 23.2 | 22.7 | 33.8 | 34.1 |
  | citations | 27.5 | 28.8 | 36.9 | 36.2 |
  | long sentences with a citation | .71 | .76 | .77 | .84 |
  | citations per cited sentence | 2.01 | 1.93 | 2.39 | 1.34 |
  | long sentences with 3 or more citations | .17 | .17 | .29 | .06 |
  | precision of a cited sentence | .560 | .577 | .517 | .843 |
  | cite_p (cited share × precision) | .400 | .438 | .399 | .712 |

  Long sentences are those of at least 50 characters, the ones cite_p scores. DraftRef cites more
  of them, but each cited sentence scores lower. Averaged across tasks, per-citation precision
  is .416 against DS-base's .454 (difference −.039 [−.058, −.020]). Compared with DS-base with
  d1's filter on the same papers in the same task and at the same bundle size, DraftRef's
  citations are supported at least as often; the lower overall rate comes from the bigger
  bundles. The citation rule (below, last column) splits them.
- **Claim support.** +.063 [.050, .076], of which +.051 remains with samples averaged. Claim
  support judges each long sentence against its cited papers together, then, if needed, against
  the papers cited by a neighbouring sentence. DraftRef leaves fewer long sentences uncited, and
  its citations are the papers the draft's sentences were written from.
- **The benchmark's other metrics.** DraftRef is higher on organization (+.061 [.028, .094]),
  claim coverage (+.072 [.059, .085]) and reference coverage (+.044 [.025, .063]). With samples
  averaged these are +.072, +.041 and +.045: the submission rule lifts claim coverage, which,
  like cite_p, rewards cited sentences. Our judge reproduces the official organization judge
  poorly (docs/02), so that difference is the least certain of the three.
- **What d1 contributes.** The cost saving: it replaces the executor's filter. It shapes few
  documents: it adds papers on only 43 of the 180 tasks and supplies 3.7% of DraftRef's
  documents overall. As a filter on its own, d1 keeps fewer important citations than DS-base's
  filter (reference recall −.020 [−.037, −.004]); the draft rule more than makes this up.
- **Validity.** DraftRef's rule was fixed on development tasks (June dev and Nov dev2, 20 each;
  runs `dsb-dr-dev`, `dsb-dr-dev2`) before its test. The 180 test tasks had already been used by
  the final test and later comparisons, so this is not an independent test and its p-values are
  nominal. The comparison is also unequal in selection: DS-base is the mean of two samples,
  DraftRef the best of four. Selection adds .023 to DraftRef's cite_p and .012 to its claim
  support; its effects on nugget coverage (+.003) and reference recall (−.001) are small.

**The citation rule.** cite_p splits a section into sentences at full stops and judges every
citation against the whole sentence. A sentence that joins points from several papers therefore
scores low even when each citation supports its own clause, and DraftRef's writer copies such
sentences from the draft, which it gets as the section's existing content. DraftRef with the
citation rule (`dsb_d1_dr_cr`) changes the writer's instruction "Cite as many sources as
possible.", which its prompt gives twice (in its citation guidelines and its quality checks).
Both become:

> Cite every source at least once, and end a sentence where its source changes: do not join
> points from different sources in one sentence (with 'and', 'while', a semicolon or a
> comma-separated list), even where the existing section content does; split such sentences,
> keeping every point with its citation. Cite several sources in one sentence only if each one
> supports the whole sentence.

Everything else is DraftRef's: the pool, the draft and the writer's documents are identical on
all 720 samples, and it writes four samples and submits one by the same rule. The rule is the
`cite_rule` setting in `configs/experiments/evaluate.yaml`; it replaces the instruction in the
intro writer's call only, so DS-base's other prompts are unchanged.

**Its claim.** Against DraftRef, both the best of four, paired over the 180 test tasks: cite_p must
be higher, and nugget coverage, reference recall and claim support non-inferior at a margin of
0.02. Each part is a two-sided sign-flip test (for non-inferiority, of the difference plus the
margin). The claim holds if the largest of the four p-values is below .05.

**Running it.** `jev run evaluate --variant dsb-drcr-test --run-id dsb-drcr-test` with d1 served;
its search replays as DraftRef's does. The comparison applies `analysis.best_of` to
`analysis.writer_samples` of `dsb-drcr-test` and of `dsb-dr-test`, paired by task, and the paired
statistics of docs/04 (`analysis.paired_stats`; `analysis.sign_flip_p` of the difference plus
0.02 for non-inferiority) give the claim. `jev bestof dsb-drcr-test dsb_d1_dr_cr --baseline
eval-test --baseline eval-test-ext` compares it with DS-base.

**Outcome** (run `dsb-drcr-test` against `dsb-dr-test`). The claim holds (p = .0109, from nugget
coverage). Its scores are in the results table above.

| With the rule − DraftRef (180 tasks) | Difference [95% interval] | p | Holds |
|---|---|---|---|
| cite_p | +.313 [.288, .337] | < .0001 | yes |
| nugget coverage (margin −.02) | +.004 [−.015, .024] | .0109 | yes |
| reference recall (margin −.02) | +.001 [−.001, .003] | < .0001 | yes |
| claim support (margin −.02) | +.036 [.026, .045] | < .0001 | yes |

**What the rule changes.**

- **Sentences.** The writer splits the bundles: citations per cited sentence fall from 2.39 to
  1.34, and long sentences with three or more citations from .29 to .06 (the per-section table
  above). It cites about the same papers (34.1 against 33.8) with about as many citations, in 10%
  more words, and leaves fewer long sentences uncited (.84 cited against .77).
- **cite_p.** The precision of a cited sentence rises from .517 to .843, and per-citation
  precision from .416 to .773. With more long sentences cited as well, cite_p rises from .399 to
  .712. DS-base's cite_p is .400 and its per-citation precision .454.
- **Claim support and claim coverage.** +.036 and +.059 [.046, .073]: both judge sentences against
  the papers they cite.
- **Content.** Reference recall and reference coverage differ on one task of 180 (+.001), and
  nugget coverage is within the margin (+.004 [−.015, .024]): the rule changes how the section
  cites, not which papers it cites. Organization is lower, −.028 [−.059, .003], which is not
  resolved; against DS-base it is +.033 [−.004, .070]. Our organization judge is the least
  reliable of the metrics (docs/02).
- **Cost.** The rule adds no call, and its writes produce about as many tokens, so it costs what
  DraftRef costs. Its recorded $0.075 is $0.003 lower only because its first writes found
  DraftRef's prompts in the provider's prompt cache.
- **Against DS-base.** It meets DraftRef's claim (p = .0001): nugget coverage +.012 [−.003, .026],
  reference recall +.043 [.024, .062], cost lower (−.115 [−.119, −.111]).
- **Validity.** The rule was fixed on the development tasks (runs `dsb-drcr-dev`, `dsb-drcr-dev2`,
  against `dsb-dr-dev`, `dsb-dr-dev2`) before its test. The test tasks had been used before,
  DraftRef's own test among them, so its p-values are nominal. Both sides are selected the same
  way, the best of four.

## DRB2 transfer check

A small check that the controller transfers to broader research reports:

- **Tasks and systems.** 6 fresh DRB2 test tasks, chosen before any call: the first six of the
  frozen test order (`data/splits/drb2.json` orders each language by a hash of the task index and
  interleaves the languages, so they are 3 en and 3 zh), ids 2, 101, 44, 23, 24 and 87, in `ids:`
  of the benchmark cell. The systems are DS-base (intro-only, at its natural cost), Simple RAG
  (k = 20), the fixed rule and the chosen d1 (`ctl_d1`), one writer sample each
  (`configs/experiments/drb2.yaml`).
- **Web access.** arXiv plus Tavily for every system, at most 4 web operations per task
  (`search.web_ops`, docs/04); a refused call reaches a system as a failed tool call.
- **The controller on DRB2.** It searches with the research request in place of the abstract,
  starting from one Tavily search (docs/05).
- **Running it.** `jev run drb2 --run-id drb2-transfer`, with d1 served.
- **Reading.** Six tasks are a transfer diagnostic, not evidence of competitiveness.

**Outcome** (run `drb2-transfer`, arXiv plus Tavily, one writer sample). Every system ended `ok`
on all six tasks. The controllers and DS-base used all 4 web operations on every task, Simple RAG
its one search; DS-base's agent had 9 to 32 further web calls refused per task.

| System | Total | Δ against DS-base | Cost per task (% of DS-base) |
|---|---|---|---|
| DS-base intro-only | .243 | – | $.245 (100%) |
| Simple RAG (K = 20) | .274 | +.030 [−.057, .118] | $.016 (7%) |
| fixed rule | .221 | −.022 [−.105, .061] | $.051 (21%) |
| d1 | .197 | −.046 [−.142, .050] | $.064 (26%) |

Every score interval includes 0: on six tasks no system's total differs resolvably from DS-base's.
The other three cost less than DS-base, each with its cost interval wholly below 0.

## The research question

The research core is a small decision model that updates its choices as evidence arrives, and
allocates research actions under a fixed executor and budget. Controllers over frozen executors,
planner RL and group-relative policy optimisation exist in prior work; the contribution here is to
test, with the executor, tools and cost accounting held equal, whether d1's allocation of
retrieval changes the quality–cost curve, and to explain where any gain comes from by comparing a
same-component fixed rule, SFT and RL.

## Limitations

Evidence is abstracts only; there is one executor; the judges share the executor's model family;
June and Nov tasks differ in distribution; the training set is small (34 tasks); the DRB2 check
is small.

## Risks

| Risk | Response |
|---|---|
| The provider updates the model behind `deepseek-flash` (docs/02) | the ledger records each reply's `system_fingerprint`; the version is pinned and a reply of another, fresh or replayed, stops the run; every arm of the test and its baselines come from one version |
| The executor saw target papers in training (no published cutoff, docs/02) | test on the Nov tasks; read nugget gains with reference recall and claim support, which credit only retrieved, cited documents |
| Action differences are smaller than writer noise | two writer samples per task; soft targets and ties; a judge-free retrieval reward |
| Cached outputs mistaken for independent samples | separate replay namespaces per writer sample |
| Retrieval reward improves while reports do not | U keeps nugget coverage and reference recall; the claims need report-level evidence |
| A fixed component explains the improvement | the same-component fixed rule and the no-draft DS-base diagnostic |
| Same-family judging prefers a style | scores labelled non-official; official-judge agreement and a blind spot-check on baseline reports (docs/02) |

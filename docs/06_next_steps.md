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

### Writing decisions

**Why.** On June dev (run `eval-dev-trained`) the fixed rule run at more batches gains about as
much as any d1 controller: at matched cost every d1 lies on or below the fixed rule's curve
(`jev controller frontier`), and a per-task oracle over the fixed budgets, cross-fitted on the
two writer samples, gains nothing (`jev controller headroom`: U +.011 [−.036, .057] at 4
batches, −.011 at 6; in-sample +.05, all writer noise). Retrieval decisions leave the report
almost unchanged (docs/03: every important citation adds +0.029 nugget coverage); the writing is
where the scores are made. So the controller also decides how the section is written, as the
proposal's action space did (search, read, write, revise).

**Actions.** After the first draft, writing decisions (`revise` in a controlled cell; none by
default), each between `submit` (last) and the revisions still affordable, at most two:
`ground` rewrites the draft so that each cited sentence states only what its sources support,
`expand` so that it also covers the paper's problem, the field's key ideas and documents not
discussed yet (docs/03: 71% of the nuggets restate the abstract or frame the field). A revision
is DS-base's own section writer again, with the draft as its existing content (citations as
numbers into the same documents), on the `revise` stage. d1 reads the task, the budget, the
revisions made, the documents' titles and the draft (`render_write`). Each writing decision is
logged to writes.jsonl; a run continues from one with `start.revision` (n revisions made; for a
run before writing decisions, its submitted draft is revision 0).

**Pilot** (`sbatch scripts/write_pilot.sh`, `configs/experiments/writing.yaml`, runs
`write-pilot-train` and `write-pilot-val`). From each submitted draft of the fixed rule's train
and val runs: submit it unchanged, ground it, expand it, or expand then ground, two revision
samples each, judged. Branches from one draft share its writer noise, so the choices are paired
per task; `headroom` over the four choices bounds what choosing per task can gain. Labels for
d1's writing decisions come from the same runs.

**Result.** No revision beats submitting the draft: U −.011 (ground), −.003 (expand), −.007
(expand then ground) on the 34 train tasks, every interval across 0; on val (9 tasks) expand
+.028 [−.007, .064]. A revision costs about $0.001. Choosing per task gains nothing either: the
cross-fitted oracle reaches U .436 against .435 for submitting (train). Rewriting from the same
abstracts adds nothing the judges see, so the writer needs more to write from.

### Reading

**Why.** The writer reads each document's title and abstract only (DS-base's arXiv reads, P1), so
a revision can only rephrase what the first draft already had. The proposal's `read` action gives
it the documents' full text.

**Actions.** Two more revisions (`READS`, at most one per episode, `Costs.read_usd`): `read`
revises the draft for coverage with each document's full-text excerpt after its abstract,
`redraft` is DS-base's intro writer again from them, without the draft; every later revision
keeps the excerpts. The excerpt is the PDF's text from its introduction's heading, cut to 3000
characters (`search/fulltext.py`). `jev data fulltext RUN...` fetches the PDFs of the runs'
writer documents beforehand, at the version their URL names (the frozen snapshot's, so nothing
written later), from export.arxiv.org at one request per 3 s, into `cache/fulltext/`; a run reads
that cache only, and a reading revision with no cached text for any document fails the task. The
judges still check citations against the abstracts (the benchmark's source material), so a claim
only the full text supports can count as unsupported: cite_p and claim support are, if anything,
biased against reading.

**Pilot** (`sbatch scripts/read_pilot.sh`, `configs/experiments/reading.yaml`, runs
`read-pilot-train` and `read-pilot-val`). As the writing pilot: from each submitted draft of the
fixed rule's train and val runs, submit it, expand it (from the abstracts), read, or redraft, two
samples each, judged, with the per-task headroom. Reading every document is the upper bound: if
it adds nothing, choosing which documents to read cannot either, and d1 is not trained on it.

**Result.** The writer read 8-26 documents' excerpts per task (763 of 764 PDFs fetched), and
reading adds nothing: U against submitting the draft −.003 (read), +.005 (redraft), −.003
(expand) on the 34 train tasks, every interval across 0, alike on val (9 tasks); redraft lowers
cite_p on val (−.107 [−.237, .023]). Reading costs $0.004 per task. The cross-fitted per-task
oracle over the four choices gains U +.010 [−.013, .032] at read's cost (train). With the
retrieval and writing pilots: no decision this loop can make moves the judged scores beyond the
writer's sample-to-sample noise at matched cost; only more search budget does (the fixed curve).

### Screening

**Why.** Every decision so far replaced the fixed rule, which costs nothing, so d1 could win only
by deciding better, and the pilots found nothing to gain there. SearchJev's System-1 model wins
the other way: it takes over the large model's short judgments (relevance above all), and the
time and money saved go elsewhere. Here the short judgment is DS-base's CoT sem_filter, one
DeepSeek call per screened paper: 59% of the fixed rule's spend at 3 batches ($.013 of $.022 a
task on June dev), 78% at 6 ($.037 of $.047). A d1 call is priced at $0.00002 per 1k prompt
tokens, about 50 times less. If d1 screened as well as DeepSeek, the 6-batch rule (U .399, nugget
.242) would cost about what the 3-batch rule does now (U .378, nugget .204): the matched-cost gain
the proposal asks for.

**Labels.** The runs logged 4,924 DeepSeek verdicts on (task, paper) pairs over 63 tasks (53%
relevant). DeepSeek keeps 94% of the screened papers the target paper really cites, and 52% of the
others: losing cited papers is what a cheaper screener must not do.

**Zero-shot test** (`sbatch scripts/screen_eval.sh`, one short GPU job, no API).
`jev controller screen-data` turns the verdicts into d1 examples (`controller/screen.py`: the task,
the paper's title and abstract, sem_filter's own question as `relevant` / `not relevant`, target
DeepSeek's verdict at 0.9), one file per split: train, val and June dev tasks (test tasks never
appear). `jev controller screen-eval` scores d1 (both option orders averaged: the codes are
positional) against DeepSeek and the real citations, alone at keep thresholds and as a cascade
(d1 decides where its top probability reaches delta, DeepSeek decides the rest; the share d1
decides is the share of sem_filter's spend saved). Report: `runs/screen/zero-shot-{dev,train}.md`.

**Zero-shot result** (June dev, 1,781 verdicts, 54 of cited papers). d1-3B as released ranks
papers only weakly (AUC .72 against DeepSeek's verdicts, .77 for cited against not) and puts
P(relevant) low for almost every paper (median .23), so its confident answers are mostly "not
relevant" and wrong: as a cascade at delta .8 it decides 37% and keeps 83% of the cited papers
(DeepSeek 91%). Its best plain threshold (keep at P >= .2) keeps 63% of the papers and 93% of
the cited, agreeing with DeepSeek on 68%. Not usable untrained.

**SFT** (`sbatch scripts/screen_sft.sh`, recipe `screen` in configs/training.yaml): LoRA on the
train tasks' 3,143 verdicts, the epoch chosen on the val tasks' (runs/controller/screen), then
scored on the held-out June dev tasks (runs/screen/sft-dev.md) against zero-shot-dev.md. If it
reaches DeepSeek's cited recall while deciding most papers, the screener goes into the loop, and
the fixed rule at more batches runs against the 3-batch rule at matched cost on June dev.

**SFT result** (job 33344984, 34 min; adapter sha256 0ec04682…, epoch 2 of 2 by val CE .594,
val accuracy .749 against DeepSeek, ECE .03). Training fixes the calibration: AUC against
DeepSeek .72 → .84 on June dev (cited against not .77 → .81), median P(relevant) .23 → .55.

| June dev (1,781 verdicts, 54 cited) | agree | kept | cited recall | d1 decides |
|---|---|---|---|---|
| DeepSeek (CoT sem_filter) | 1.000 | .537 | .907 | 0 |
| zero-shot cascade, delta .8 | .883 | .421 | .833 | .371 |
| SFT alone, keep at P >= .5 | .761 | .541 | .889 | 1 |
| SFT cascade, delta .7 | .907 | .546 | .944 | .636 |
| SFT cascade, delta .6 | .834 | .544 | .944 | .847 |

Delta is chosen on val (821 verdicts, 38 cited): at .7 the cascade keeps every cited paper
DeepSeek keeps (.974) while d1 decides 68%, agreeing with DeepSeek on 88%; .6 decides 86% at the
same recall but agrees on only 82%, so .7. On the held-out June dev it decides 64% and keeps more
cited papers than DeepSeek alone (.944 against .907: three papers of 54, within noise), keeping
the same share of papers. So d1 takes over about two thirds of the screening without losing
cited papers: at 6 batches about $.024 of the $.037 sem_filter spend, which puts the 6-batch
rule near the 3-batch rule's cost.

**In the loop** (`screener` in systems/controlled.py, none by default). The screening adapter is
served (`jev controller serve --adapter runs/controller/screen/adapter --port 9943`); each chosen
search's new papers go to it first in one request (both option orders, charged as `d1_choice`),
and a paper where d1's top probability reaches delta is decided by d1, the rest by sem_filter as
before (its per-paper requests, so the verdicts replay). screens.jsonl logs every paper's
P(relevant) and who decided; the report's meta has `d1_screened`. `jev controller frontier`
counts a screened cell as a learned one, so it is compared with the unscreened fixed rule's
curve at matched cost.

**June dev test** (`sbatch scripts/screen_dev.sh`, variant `dev-screen`, run `eval-dev-screen`):
the fixed rule at 3-6 batches (the same cells as eval-dev-trained: their DeepSeek answers replay)
and screened at 3-6 and 8 batches, 20 tasks, two writer samples. The claim holds if a screened
budget beats the fixed rule at matched cost (frontier.md), without a lower cite_p.

**How results are reported here.** On the benchmark's own metrics, each separately: nugget
coverage and reference recall (completeness), cite_p and claim support (citation reliability),
claim coverage, and list cost per task. These are primary. The team's registered utility U
(docs/04; also the training reward and the selection score) is reported alongside, in the last
column, as a secondary summary: its weights are the team's own (nugget dominates; cite_p and
claim support add at most .10 each), so a U difference is read together with the metrics it
combines, never instead of them.

**June dev result** (run `eval-dev-screen`, job 33372245, 1 h 50 min; runs/eval-dev-screen/
frontier.md). In the loop the screener behaves as offline: d1 decides 64% of the screened papers,
agrees with DeepSeek's verdict on 88-89% of those where one is known, and the screened runs keep as
many papers (b6: 29.5 of 54.9, unscreened 29.8 of 54.4) and as many of the cited ones (.95 against
.91 at b6). Means (20 tasks, writer samples averaged; bN = N search batches):

| system | $ | nugget | ref recall | cite_p | claim support | claim cov. |
|---|---|---|---|---|---|---|
| fixed b3 | .021 | .210 | .220 | .606 | .892 | .773 |
| fixed b4 | .028 | .231 | .218 | .583 | .880 | .782 |
| fixed b5 | .041 | .243 | .247 | .549 | .886 | .793 |
| fixed b6 | .047 | .253 | .244 | .530 | .872 | .775 |
| screened b3 | .014 | .194 | .206 | .589 | .855 | .739 |
| screened b4 | .019 | .225 | .222 | .589 | .885 | .773 |
| screened b5 | .029 | .225 | .237 | .528 | .872 | .769 |
| screened b6 | .034 | .221 | .252 | .594 | .928 | .812 |
| screened b8 | .047 | .208 | .262 | .538 | .910 | – |

Screened − fixed at the same batches (paired, 95% t-interval; * = interval excludes 0):

| batches | $ | nugget | ref recall | cite_p | claim support |
|---|---|---|---|---|---|
| 3 | −.0074* | −.016 [−.051, .018] | −.014 [−.030, .001] | −.016 [−.078, .045] | −.037 [−.066, −.008]* |
| 4 | −.0092* | −.006 [−.041, .028] | +.004 [−.023, .031] | +.006 [−.054, .066] | +.005 [−.041, .051] |
| 5 | −.0124* | −.018 [−.071, .036] | −.010 [−.040, .020] | −.021 [−.084, .041] | −.014 [−.048, .020] |
| 6 | −.0134* | −.032 [−.068, .003] | +.008 [−.019, .035] | +.064 [−.013, .141] | +.056 [.017, .096]* |

- Cost: −$.011 a task at the same batches (−28% at b6, not −50%): sem_filter's spend falls only
  42% at b6 ($.037 → $.021) though d1 decides 65% of the papers (DeepSeek keeps the borderline
  ones, whose CoT is longer), and d1's own calls cost $.002 (about 1.4k prompt tokens a paper for
  both option orders).
- Matched cost (frontier.md): screened b4 ($.019) against fixed b3 ($.021): nugget +.015
  [−.019, .048], cite_p −.017 [−.078, .044]; screened b8 against fixed b6: nugget −.045
  [−.082, −.008]. More search raises nugget only slowly (fixed b3 → b6: +.043 for +$.026) and
  lowers cite_p (−.075), so the money saved buys little.

**Nov dev2 replication** (run `eval-dev2-screen`, job 33407595, 1 h 42 min; 20 new tasks).

| system | $ | nugget | ref recall | cite_p | claim support | claim cov. |
|---|---|---|---|---|---|---|
| fixed b3 | .021 | .415 | .206 | .576 | .858 | .755 |
| fixed b4 | .032 | .384 | .224 | .539 | .861 | .745 |
| fixed b5 | .045 | .383 | .267 | .549 | .857 | .767 |
| fixed b6 | .054 | .420 | .278 | .521 | .869 | .768 |
| screened b3 | .013 | .404 | .212 | .605 | .893 | .783 |
| screened b4 | .021 | .411 | .235 | .588 | .903 | .789 |
| screened b5 | .031 | .409 | .272 | .531 | .864 | .761 |
| screened b6 | .036 | .391 | .275 | .558 | .898 | .798 |

| batches | $ | nugget | ref recall | cite_p | claim support |
|---|---|---|---|---|---|
| 3 | −.0082* | −.012 [−.042, .019] | +.006 [−.017, .029] | +.029 [−.020, .079] | +.036 [.004, .067]* |
| 4 | −.0117* | +.028 [−.030, .086] | +.011 [−.016, .038] | +.048 [−.019, .115] | +.042 [.008, .076]* |
| 5 | −.0148* | +.026 [−.030, .083] | +.005 [−.023, .034] | −.018 [−.074, .038] | +.007 [−.037, .052] |
| 6 | −.0179* | −.030 [−.092, .032] | −.003 [−.048, .042] | +.037 [−.029, .103] | +.030 [−.025, .084] |

Here nugget does not rise with search at all (fixed b3 .415, b6 .420); reference recall does
(.206 → .278).

**Both dev sets** (40 tasks, paired per task, b3-b6 averaged per task), screened − unscreened at
the same batches: cost −$.012 [−.014, −.010]; nugget −.008 [−.028, .013]; cite_p +.016
[−.008, .040]; claim support +.016 [−.001, .032].

**What this shows.** The d1 screener cuts the research loop's cost by about a third at the same
search (−$.012 of about $.035 a task), with no metric significantly lower when pooled (nugget
−.008, cite_p and claim support a little higher). It does not give a clear quality gain at
matched cost: in this range more search raises reference recall but barely nugget coverage, and
lowers cite_p, so money moved from screening to search has little to buy. A held-out
confirmation is the test split with ctl_fixed and ctl_screen (the frozen adapter, delta .7).

**Test plan** (written before the run; variant `test-screen`, run `eval-test-screen`). The
held-out test split, first 90 of its 180 tasks (the `test` variant's), two writer samples. Frozen
as on dev: the adapter 0ec04682 at delta .7; `ctl_fixed`, `ctl_screen` and `ctl_screen_b4` are
exactly the dev cells (pinned by tests/test_screen.py); no test task's verdict was ever trained
or tuned on. Systems: DS-base intro-only (the strongest baseline), simple RAG k=20, the fixed
rule (3 batches), screened at 3 and at 4 batches. Decided now, judged once:

1. **Cost** (primary): `ctl_screen` costs less than `ctl_fixed` (list cost per task, paired):
   expected about −$.008 (−35%), the interval below 0.
2. **Same quality** (primary, non-inferiority): `ctl_screen` − `ctl_fixed` in U has its 95%
   lower bound above −.03 (about the fixed rule's gain from 3 to 6 batches on June dev), and in
   cite_p above −.05 (citation reliability).
3. **Matched cost** (secondary, the proposal's criterion against the fixed rule):
   `ctl_screen_b4` (about $.019) against `ctl_fixed` ($.021): U higher, interval above 0.
   Expected +.010 from dev: likely not significant even at 90 tasks.
4. **Against DS-base** (descriptive): every controller against DS-base intro-only, which costs
   about 9 times as much.

Nothing is changed after seeing the results; any further analysis is reported as exploratory.
Billed estimate (from the dev runs' ledgers, with the replay cache cold): about $8-10 for
generation (DS-base most of it) and $4 for judging; caps $20 / $8. About 900 episodes, 5-8 hours
plus peak-hour holds; the job is resumable.

**Systems compared on test.** All use DeepSeek-V4.1-Flash, arXiv search and the same writer.

- **DS-base intro-only**: the reference pipeline (DeepScholar-base), at its natural cost.
- **simple RAG k=20**: retrieve 20 abstracts, write in one pass.
- **fixed b3** (`ctl_fixed`): our loop with a fixed rule, 3 search batches; every retrieved paper
  is screened for relevance by DeepSeek (sem_filter).
- **screened b3** (`ctl_screen`): the same, but the trained d1 screens each paper first and
  DeepSeek only screens the papers d1 is unsure of. The ablation: fixed b3 is screened b3 with
  its main component (the d1 screener) removed.
- **screened b4** (`ctl_screen_b4`): screened, with a 4th batch bought with the money the screener
  saves (still cheaper than fixed b3).

**Test result** (`eval-test-screen`, 90 tasks, two writer samples averaged per task; generation
job 33417440, judging job 33417446; DeepSeek-judged, non-official). On test, d1 decides 66% of the
papers (3432 of 5186 in `ctl_screen`), as on dev (64%).

| system | $/task | nugget | ref recall | cite_p | claim support | claim cov. | U |
|---|---|---|---|---|---|---|---|
| DS-base intro-only | .188 | .419 | .250 | .409 | .775 | .687 | .502 |
| simple RAG k=20 | .003 | .351 | .114 | .477 | .823 | .719 | .496 |
| fixed b3 | .0211 | .367 | .160 | .588 | .878 | .776 | .528 |
| screened b3 | .0135 | .354 | .167 | .592 | .881 | .773 | .520 |
| screened b4 | .0179 | .375 | .174 | .574 | .885 | .779 | .539 |

Every pair, A − B (paired per task, 95% t-interval; * = interval excludes 0; ref recall on 83
tasks with gold; a report with no citation to check scores 0 on the citation metrics, as in
report.md):

| A − B | nugget | ref recall | cite_p | claim support | claim cov. | $ | U |
|---|---|---|---|---|---|---|---|
| DS-base − simple RAG | +.068 [.032, .105]* | +.136 [.071, .201]* | −.068 [−.101, −.035]* | −.048 [−.080, −.017]* | −.033 [−.061, −.004]* | +.1851* | +.007 [−.032, .045] |
| DS-base − fixed b3 | +.052 [.009, .096]* | +.090 [.027, .154]* | −.179 [−.210, −.148]* | −.103 [−.131, −.076]* | −.090 [−.115, −.064]* | +.1671* | −.025 [−.070, .020] |
| DS-base − screened b3 | +.065 [.018, .112]* | +.083 [.022, .143]* | −.183 [−.212, −.154]* | −.106 [−.131, −.081]* | −.086 [−.111, −.062]* | +.1747* | −.017 [−.065, .031] |
| DS-base − screened b4 | +.044 [.007, .081]* | +.076 [.015, .138]* | −.165 [−.199, −.131]* | −.111 [−.133, −.089]* | −.092 [−.115, −.069]* | +.1702* | −.036 [−.074, .001] |
| simple RAG − fixed b3 | −.016 [−.050, .019] | −.046 [−.084, −.007]* | −.111 [−.140, −.082]* | −.055 [−.083, −.027]* | −.057 [−.084, −.030]* | −.0180* | −.032 [−.067, .003] |
| simple RAG − screened b3 | −.003 [−.039, .033] | −.053 [−.092, −.015]* | −.115 [−.147, −.083]* | −.057 [−.083, −.032]* | −.054 [−.079, −.029]* | −.0105* | −.024 [−.059, .011] |
| simple RAG − screened b4 | −.024 [−.050, .002] | −.060 [−.099, −.020]* | −.097 [−.130, −.064]* | −.062 [−.086, −.038]* | −.059 [−.084, −.035]* | −.0149* | −.043 [−.070, −.016]* |
| screened b3 − fixed b3 | −.013 [−.040, .015] | +.008 [−.008, .023] | +.004 [−.018, .027] | +.003 [−.014, .019] | −.003 [−.019, .013] | −.0076* | −.008 [−.035, .019] |
| screened b4 − fixed b3 | +.009 [−.022, .039] | +.014 [−.007, .034] | −.014 [−.043, .015] | +.007 [−.010, .024] | +.002 [−.015, .020] | −.0031* | +.011 [−.019, .041] |
| screened b4 − screened b3 | +.021 [−.005, .047] | +.006 [−.006, .018] | −.018 [−.047, .010] | +.005 [−.010, .020] | +.005 [−.010, .021] | +.0045* | +.019 [−.007, .045] |

Against the pre-registered hypotheses:

1. **Cost: holds.** screened b3 − fixed b3 = −$.0076 [−.0080, −.0071] (−36%).
2. **Same quality: holds for cite_p; for U, as registered, misses narrowly.** cite_p +.004
   [−.018, .027], lower bound above −.05. On every original metric screened b3 is within noise of
   fixed b3 (nugget −.013 [−.040, .015], claim support +.003). The registered U test (lower bound
   above −.03): −.008 [−.035, .019], not shown at 90 tasks.
3. **Matched cost: not shown.** screened b4 is cheaper than fixed b3 by $.0031 and higher on
   nugget (+.009), ref recall (+.014) and claim support (+.007), none significantly; cite_p −.014,
   not significant. (Registered U: +.011 [−.019, .041].)
4. **Against DS-base (descriptive).** Every controller costs about a tenth of DS-base. DS-base is
   more complete (nugget +.044 to +.065, ref recall +.076 to +.090 over the controllers), and its
   citations are much less reliable (cite_p −.165 to −.183, claim support −.103 to −.111). On U
   no difference is resolved (DS-base − screened b4 −.036 [−.074, .001]).

**What the test shows.** The d1 screener's cost saving transfers to held-out tasks (−36% at the
same number of batches) with no metric detectably lower. Spending the saving on a fourth batch
moves completeness up a little at lower cost, not significantly. Against DS-base the controllers
trade completeness for citation reliability at a tenth of the cost; no controller is more
complete than DS-base. Anything further on this run is exploratory.

**Exploratory: all 180 test tasks** (variant `test-screen-180`, run `eval-test-screen-180`; decided
after seeing the 90-task result, so it is **not** a confirmation). The same frozen systems on the
whole test split, to narrow the intervals.

**Exploratory result** (`eval-test-screen-180`; GPU jobs 33434901, cancelled, and 33440489,
judging job 33440494, 7:12; billed $7.41 generation, $5.11 judging). The first 90 tasks did
**not** replay exactly: the judge cache was not hit (every task re-judged), and only part of the
generations repeated byte for byte (simple RAG 90/90, the controllers 45-60/90, DS-base 0/90). So
this run is a second, independent pass over the first 90 plus the 90 new tasks, not the 90-task
run extended.

| system | $/task | nugget | ref recall | cite_p | claim support | claim cov. | U |
|---|---|---|---|---|---|---|---|
| DS-base intro-only | .191 | .432 | .230 | .413 | .776 | .686 | .512 |
| simple RAG k=20 | .003 | .368 | .117 | .466 | .831 | .718 | .513 |
| fixed b3 | .0210 | .364 | .143 | .582 | .881 | .767 | .522 |
| screened b3 | .0133 | .372 | .148 | .586 | .877 | .770 | .534 |
| screened b4 | .0185 | .387 | .166 | .559 | .879 | .776 | .547 |

| A − B (180 tasks; ref recall 168) | nugget | ref recall | cite_p | claim support | claim cov. | $ | U |
|---|---|---|---|---|---|---|---|
| DS-base − simple RAG | +.064 [.042, .087]* | +.113 [.070, .156]* | −.053 [−.075, −.030]* | −.054 [−.074, −.035]* | −.033 [−.052, −.014]* | +.1882* | .000 [−.024, .023] |
| DS-base − fixed b3 | +.068 [.042, .095]* | +.087 [.047, .126]* | −.168 [−.190, −.147]* | −.105 [−.123, −.087]* | −.081 [−.099, −.063]* | +.1702* | −.010 [−.036, .016] |
| DS-base − screened b3 | +.061 [.034, .087]* | +.082 [.042, .121]* | −.173 [−.194, −.152]* | −.101 [−.118, −.084]* | −.084 [−.102, −.066]* | +.1779* | −.021 [−.048, .005] |
| DS-base − screened b4 | +.045 [.019, .070]* | +.064 [.024, .104]* | −.146 [−.168, −.123]* | −.103 [−.120, −.086]* | −.090 [−.108, −.073]* | +.1727* | −.035 [−.061, −.009]* |
| simple RAG − fixed b3 | +.004 [−.018, .026] | −.026 [−.057, .004] | −.116 [−.137, −.094]* | −.051 [−.068, −.033]* | −.048 [−.066, −.030]* | −.0179* | −.010 [−.032, .013] |
| simple RAG − screened b3 | −.004 [−.025, .018] | −.031 [−.063, .000] | −.120 [−.142, −.098]* | −.047 [−.065, −.029]* | −.051 [−.070, −.033]* | −.0102* | −.021 [−.042, .001] |
| simple RAG − screened b4 | −.019 [−.038, −.001]* | −.049 [−.084, −.014]* | −.093 [−.115, −.070]* | −.049 [−.066, −.032]* | −.058 [−.075, −.040]* | −.0155* | −.035 [−.054, −.015]* |
| screened b3 − fixed b3 | +.008 [−.009, .024] | +.005 [−.006, .016] | +.005 [−.011, .021] | −.004 [−.017, .008] | +.003 [−.010, .016] | −.0077* | +.011 [−.005, .027] |
| screened b4 − fixed b3 | +.023 [.005, .042]* | +.023 [.004, .042]* | −.023 [−.043, −.003]* | −.002 [−.014, .010] | +.009 [−.003, .022] | −.0025* | +.025 [.007, .043]* |
| screened b4 − screened b3 | +.016 [−.001, .033] | +.018 [.003, .033]* | −.028 [−.046, −.009]* | +.002 [−.011, .015] | +.006 [−.007, .020] | +.0053* | +.014 [−.003, .031] |

**Judge noise.** simple RAG's 90 identical texts, judged twice: the per-task difference has sd
.071 in nugget coverage and .038 in cite_p (means −.004 and −.008). Differences of ±.01 in a
90-task mean are within the judge's and the writer's run-to-run noise; the 90-task run's
screened b3 − fixed b3 nugget, −.013, is +.005 on the same tasks here.

**Reading.** On 180 tasks the screener again costs 36% less with every metric within noise of
the fixed rule. Spending the saving on a fourth batch (screened b4, still 12% cheaper than fixed
b3) is significantly more complete (nugget +.023, ref recall +.023) and less precise in its
citations (cite_p −.023), with claim support unchanged: a completeness-for-precision trade at
lower cost, not a gain on every metric. DS-base remains the most complete system (nugget +.045
over screened b4) and the least reliable (cite_p −.146). On U, screened b4 is higher than fixed
b3 (+.025 [.007, .043]), the matched-cost hypothesis 3 that the 90-task test did not show, and
higher than DS-base (DS-base − screened b4 −.035 [−.061, −.009]); the other controllers are
within noise of DS-base on U. This is exploratory (decided after the
90-task result, and the first 90 tasks are counted twice across the two runs); the confirmatory
result remains the 90-task one above.

**Versions.** On branch `d1-screen` (from `read-action`). Every cell without `screener` keeps its
identity (the pins in `tests/test_writing.py`); `tests/test_screen.py` covers the screener
offline.

**Versions.** On branch `read-action` (from `write-actions`). The reading revisions are offered
only where a cell lists them in `revise`, so every earlier cell keeps its identity (the pins in
`tests/test_writing.py`); `tests/test_reading.py` covers the cache and the loop offline.

**Versions.** On branch `write-actions`. A cell without `revise` (and a fixed rule without its
`revise` sequence, a start without `revision`) has exactly its earlier identity: the recorded
identities of the June dev systems, the train rollouts, a branch and an RL member are pinned by
`tests/test_writing.py`.

## The final test

**Systems** (`jev run evaluate --variant test`, the first 90 Nov test tasks, two writer samples
per task): DS-base intro-only, Simple RAG, the fixed rule, d1-rl, d1-sft, and the no-draft DS-base
diagnostic. d1-sft and d1-rl are added to `evaluate.yaml` once their adapters are frozen; each adapter is served on its own port and named
by its hash (`policy: {kind: d1, url: ..., adapter: <sha256 of adapter_model.safetensors>}`).

**Which d1 carries the main claim** is chosen on June dev before the test: d1, d1-sft or d1-rl.
d1-sft and d1-rl stay in the test matrix; d1 joins it if chosen. The comparison is `sbatch scripts/june_dev.sh`
(`jev run evaluate --variant dev-trained`, run `eval-dev-trained`): the fixed rule, d1, d1-sft
and d1-rl rerun together on the 20 June dev tasks, two writer samples each, so every pair is
measured with the same code; DS-base keeps its `eval-dev` numbers.

**At matched cost.** A controller that spends more can find more without allocating better, so
the same run also holds the fixed rule at 4, 5 and 6 batches (`ctl_fixed_b4..b6`), a cost-quality
curve. `jev controller frontier RUN` pairs each d1 controller per task with the cheapest fixed
budget whose mean list cost is at least its own (chosen by cost alone, so the fixed rule never
gets less money) and reports its distance to the curve interpolated at its cost. This is the
proposal's success test: better quality than the tuned fixed workflow at the same total cost,
without losing citation reliability.

**Stopping headroom.** At a fixed number of batches the rule spends alike on every task; a
controller can beat it at the same mean cost only by spending where search pays.
`jev controller headroom RUN` bounds that gain: a per-task oracle over the fixed budgets, at no
more mean cost than each uniform budget, picks on one writer sample and is scored on the other
(both ways), so it cannot profit from writer noise.

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

**As built: the arXiv-only check** (variant `drb2-test`, run `eval-drb2-test`; decided 11-15,
before any DRB2 test call). The web access above needs a Tavily key and a per-task web counter
that were never built, so this is the arXiv-only fallback, labelled as such.

- **Tasks.** DRB2 test[:12]: 6 en, 6 zh, never used before (the split prefix, fixed before any
  call). Requests whose own text names the forbidden article (one in 120, task 58) are not among
  them.
- **Systems.** The frozen test-screen systems unchanged: DS-base intro-only, simple RAG k=20,
  fixed b3, screened b3, screened b4 (adapter 0ec04682, delta .7), two writer samples each,
  arXiv only for every system. The controllers plan within DRB2's B0 ($0.34); DS-base runs at
  its natural cost.
- **The controller on DRB2.** Its first search (the `abstract` slot) is the research request:
  the prompt without its forbidden-article rule (`DRB2.search_seed`; the rule names the article,
  so the access policy would block the whole prompt as a query). Proposals ask for an English
  query for "the research report requested" instead of a related-work section. Everything else
  is as on DeepScholar: DS-base's screening, ranking and intro writer, which is also what the
  DS-base baseline writes with. DeepScholar cells keep their identities (`tests/test_writing.py`).
- **Metrics.** DRB2's own: total (primary), info recall, analysis, presentation, judged by
  `ds-judge-think` (docs/02), plus list cost; the team's U is not defined on DRB2 and is not used.
  Every pair as on the DeepScholar test, 95% t-intervals over 12 tasks.
- **Reading.** Twelve tasks, arXiv only, on a benchmark whose topics are mostly not on arXiv
  (finance, history, games ...): a transfer diagnostic of whether the cost saving and the ordering
  of the systems carry over, not evidence of competitiveness. d1 was trained on DeepScholar
  topics, so its share of decisions may drop.
- **Expected cost.** About $10 list for generation (DS-base ~$0.25-0.45 a task) and $2 for
  judging; about half billed off-peak. The campaign cap is $45, of which $32.31 is spent.

**Result** (run `eval-drb2-test`, GPU job 33520971, 18 min; judge job 33520985, 30 min; $2.41
billed, campaign $34.72 of $45). Every system but Simple RAG gives up on some tasks: when the
relevance filter keeps no paper, DS-base's writer (which the controllers share) raises, and the
report is empty (scored 0). Failed of 12 tasks (both samples): Simple RAG 0, screened b4 2,
screened b3 3, DS-base 3, fixed b3 5 (tasks 2, 23, 80, 109, 111: social policy, battery market,
history, railway IIoT, youth health).

| system | $ / task | total, all 12 (fail = 0) | total, 7 tasks all wrote | info recall (7) | analysis (7) | presentation (7) |
|---|---|---|---|---|---|---|
| DS-base | .310 | .179 | .273 | .210 | .265 | .816 |
| Simple RAG | .006 | .208 | .241 | .164 | .257 | .847 |
| fixed b3 | .027 | .110 | .188 | .131 | .130 | .804 |
| screened b3 | .013 | .122 | .163 | .114 | .107 | .733 |
| screened b4 | .015 | .119 | .175 | .125 | .105 | .791 |

Paired on the 7 tasks every system wrote: screened b4 − DS-base −.098 [−.175, −.022]; fixed −
DS-base −.085 [−.151, −.019]; Simple RAG − DS-base −.032 [−.091, .027]; Simple RAG − screened b4
+.067 [.022, .111]; screened b3 − fixed −.025 [−.069, .020]; screened b4 − fixed −.013 [−.046,
.019]. On all 12 (failures 0): fixed − DS-base −.069 [−.121, −.017]; Simple RAG − fixed +.098
[.051, .146]; screened − fixed +.012 and +.010 (n.s.); cost: screened b3 − fixed −$.014, b4 −$.011.

**What this shows.** The screener's cost saving transfers (about half of the fixed rule's cost)
with no significant loss against the fixed rule. The controllers themselves do not transfer: on
DRB2 they score below DS-base and below Simple RAG, mainly on analysis (about .10-.13 against
.26), because their 3-4 arXiv batches are filtered to few documents for a long report, and they
give up where the filter keeps nothing. Simple RAG never gives up and is the best on the full
12 at 2% of DS-base's cost. Arxiv-only, 12 tasks: a diagnostic, not a competitive DRB2 result;
the planned web search is what this check lacked.

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

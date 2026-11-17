# 8. Experiment log: runs before the version pin

This log covers the runs made from 2026-11-09 to 11-15 on the yield prompt set (docs/05), before
the model version was pinned (docs/02). They are the fixed rule's rollouts and branches, the
training on fixed-rule branches (docs/06), the June dev comparison of its adapters, the decision
headroom pilots, the d1 screener (docs/07) and the arXiv-only DRB2 check. The runs made after the
pin are indexed in docs/usage.md.

Each run's commit is the one its `jev run` and `jev eval` sessions recorded in `manifest.json`
(`sessions[].git_sha`, `git_dirty`). A `+` marks a dirty tree. Report-quality results are
DeepSeek-judged and paired against the named baseline, with 95% t-intervals; retrieval-return
comparisons are judge-free. Each benchmark's own metrics come first; U (docs/04) is a secondary
summary.

**Names and resumption.** The prompt set had no key in these runs, and their manifests name the
systems as they were then:

- `ctl_d1`, `ctl_d1_sft` and `ctl_d1_rl` in `eval-dev-trained` are `ctl_d1_yield`,
  `ctl_d1_sft_fixed` and `ctl_d1_rl_first` in today's configs;
- `ctl_fixed`, `ctl_screen` and `ctl_screen_b4` in `eval-drb2-test` are the `_yield` cells of the
  `drb2-test` variant.

A recorded cell without `prompts` reads today as the default set, and the pin is part of every
cell's identity, so none of these runs can be resumed with today's code. Their records are kept
as they are.

## Runs

| Run | Dates | Experiment (config / variant) | Commit | What it tested | Result |
|---|---|---|---|---|---|
| `rollouts-train`, `rollouts-val` | 11-09 | `controller.yaml` rollouts (today `rollouts_yield`) | f22e269+ | the fixed rule's episodes on train (34) and val (9): the start points of every later branch | source runs, no comparison |
| `branches-rollouts-{train,val}-b{0..3}-{abstract,explore,deepen,finish}` (27 runs) | 11-09 | `jev controller branches` | f22e269+ | each retrieval action taken after 0 to 3 batches, then the fixed rule: counterfactual returns | the labels for d1-sft-fixed |
| `groups-rollouts-train-r1-b{0,1}` | 11-09 | `jev controller groups` | 2a2ca33+ | RL groups at every decision, round 1 | adapter `controller/rl/r1`, not used further |
| `groups-rollouts-train-r2-b{0,1,3}` | 11-09 | `jev controller groups` | 4e3f300+ | RL groups at every decision, round 2 | adapter `controller/rl/r2`, not used further |
| `groups-rollouts-train-first-r1-b{0..3}` | 11-09 | `jev controller groups --first-only` | bb52376+ | RL groups at the first decision of the points whose options' returns differ | adapter `controller/rl/first/r1`: d1-rl-first |
| `compare-rollouts-val-d1_{sft,rl1,rl2}` | 11-09 to 11-10 | `jev controller episodes` | 4e3f300+, bb52376+ | d1-sft-fixed's and the round 1 and 2 RL adapters' episodes on val, paired returns | d1-sft-fixed is the val comparison's SFT row; the round 1 and 2 RL adapters are not used further |
| `compare-rollouts-val-d1_first1` | 11-10 | `jev controller episodes` | bb52376+ | d1-rl-first's episodes on val | d1-sft-fixed and d1-rl-first go to June dev (docs/06) |
| `eval-dev-trained` | 11-10 to 11-11 | `evaluate.yaml` / `dev-trained` | ed7e8e9+, 6434843+ | the fixed rule at 3 to 6 batches, d1, d1-sft-fixed and d1-rl-first on 20 June dev tasks; `frontier.md` | fixed U .378 at $.021; d1 .379 at $.050; d1-sft-fixed .377 at $.031; d1-rl-first .372 at $.043 (cite_p −.090, p .016). Fixed b4, b5, b6 U .382, .396, .399; every d1 on or below the fixed curve at matched cost; stopping headroom U +.011 [−.036, .057] (b4) |
| `write-pilot-train`, `write-pilot-val` | 11-12 to 11-13 | `writing.yaml` / `train`, `val` | 3f41713+ | the fixed rule's drafts submitted, grounded, expanded, and expanded then grounded | against submit, train U: ground −.011, expand −.003, both −.007 (all across 0); per-task oracle U .436 against .435 |
| `read-pilot-train`, `read-pilot-val` | 11-13 | `reading.yaml` / `train`, `val` | d374575+ | the drafts submitted, expanded, read (revised from the full text) and redrafted from it; 763 of 764 PDFs | against submit, train U: read −.003, redraft +.005, expand −.003 (all across 0); val redraft cite_p −.107; per-task oracle +.010 [−.013, .032] |
| `screen/zero-shot-{dev,train}` | 11-13 | `scripts/screen_eval.sh` (offline scoring, no run manifest) | 90be206 | d1 as released as the relevance screener, on the executor's verdicts (dev 1,781, train 3,143) | not usable untrained: AUC .72 against DeepSeek (dev); at P ≥ .2 it agrees on 68%; the cascade at δ .8 decides 37% (dev) and 28% (train) but keeps 83% and 94% of cited papers (DeepSeek 91% and 95%) |
| `controller/screen`, `screen/sft-{dev,val}` | 11-13 | `scripts/screen_sft.sh`, recipe `screen` | 8bec12c | d1 LoRA-trained on the train tasks' verdicts (3,143), epoch chosen on val, scored on June dev | usable: AUC against DeepSeek .84 (dev); the cascade at δ .7 (chosen on val, where it decides 68% with cited recall .974, as DeepSeek) decides 64% on dev with cited recall .944 (DeepSeek .907) and keeps .546 (.537) |
| `eval-dev-screen` | 11-14 | `evaluate.yaml` / `dev-screen`, `scripts/screen_dev.sh` | 850866d+ | the fixed rule at 3 to 6 batches with and without the d1 screener (and screened at 8 batches) on 20 June dev tasks; `frontier.md` | d1 decides 64% (agreeing with DeepSeek on 88%), papers kept unchanged; cost −$.011 a task at equal batches (b6 $.047 → $.034); screened − fixed at b3: nugget −.016, cite_p −.016 (n.s.), claim support −.037 (sig.); at b6 claim support +.056 (sig.); matched cost: screened b4 against fixed b3 nugget +.015, cite_p −.017 (n.s.); b8 against fixed b6 nugget −.045 (sig.) |
| `eval-dev2-screen` | 11-14 | `evaluate.yaml` / `dev2-screen`, `scripts/screen_dev.sh` | 810cbbf+ | the same comparison (b3 to b6, with and without the screener) on 20 Nov dev2 tasks | nugget flat with search (fixed b3 .415, b6 .420), ref recall rises (.206 → .278); screened saves $.008 to $.018 a task at equal batches; claim support +.036 (b3) and +.042 (b4), sig.; nugget and cite_p n.s. With June dev (40 tasks, b3 to b6 pooled): cost −$.012, nugget −.008, cite_p +.016, claim support +.016 (n.s.) |
| `eval-test-screen` | 11-14 | `evaluate.yaml` / `test-screen`, `scripts/screen_dev.sh`, judged by `scripts/judge.sh` | fe0f7bc+, b4384de+ | the screener's registered test on the first 90 test tasks: DS-base, Simple RAG, fixed b3, screened b3 and b4 (docs/07) | cost holds: screened b3 − fixed b3 −$.0076 [−.0080, −.0071] (−36%); same quality not shown: cite_p +.004 [−.018, .027] clears −.05, U lower bound −.035 misses −.03; matched cost not shown; the controllers cost about a tenth of DS-base. d1 decides 66% |
| `eval-test-screen-180` | 11-14 | `evaluate.yaml` / `test-screen-180`, `scripts/screen_dev.sh`, judged by `scripts/judge.sh` | 1b7f3e6+ | exploratory, decided after the 90-task result: the same frozen systems on all 180 test tasks | the first 90 did not replay exactly (re-judged; controllers 45 to 60 of 90 identical texts). Screened b3 − fixed: cost −$.0077 (sig.); nugget +.008, cite_p +.005, claim support −.004 (all n.s.). Screened b4 − fixed at −$.0025: nugget +.023 [.005, .042], ref recall +.023 [.004, .042], cite_p −.023 [−.043, −.003]; U +.025 [.007, .043]. Judge noise on identical texts: per-task nugget SD .071, cite_p .038 (docs/07) |
| `eval-drb2-test` | 11-15 | `evaluate.yaml` / `drb2-test`, `scripts/screen_dev.sh`, judged by `scripts/judge.sh` | 763dc4c+ | the arXiv-only DRB2 check: the frozen test-screen systems on 12 DRB2 test tasks (6 en, 6 zh), two writer samples (docs/06) | failed (empty, scored 0) of 12: Simple RAG 0, screened b4 2, screened b3 3, DS-base 3, fixed 5 (the filter keeps no paper). Total (all 12): DS-base .179, Simple RAG .208, fixed .110, screened .122 and .119. On the 7 every system wrote: screened b4 − DS-base −.098 [−.175, −.022]; screened − fixed n.s. Cost: DS-base $.310, fixed $.027, screened $.013 and $.015, Simple RAG $.006 |

## Trained adapters

The adapter files are not in the repository. Each config names its adapter by the sha256 of
`adapter_model.safetensors`, so a d1 server with any other adapter is refused.

| Adapter | Directory | sha256 | Served as |
|---|---|---|---|
| d1-sft-fixed | `runs/controller/sft_fixed/adapter` (recorded under `runs/controller/sft/adapter`) | dabf7875e4091468400e50fa90a67c54d2dbce7969ac73d97df05280d03c45b6 | `ctl_d1_sft_fixed` (:9941) |
| d1-rl-first | `runs/controller/rl/first/r1/adapter` | 2fa87a6904389d5d0503cd46f5aaf0fe91107a7231ce753e4c30fa5256f79807 | `ctl_d1_rl_first` (:9942) |
| d1 screener | `runs/controller/screen/adapter` | 0ec04682ed2c880a34e279f24df0d121671629a0a0d7e8274425fa4fddfb502e | the `ctl_screen*` screener (:9943) |

## Reading a run's version

```bash
python3 -c "import json,sys; [print(s['command'], s['started'][:16], s['git_sha'][:7], s['git_dirty']) for s in json.load(open(sys.argv[1]))['sessions']]" runs/RUN/manifest.json
git log -1 SHA
```

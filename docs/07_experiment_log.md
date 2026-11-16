# Experiment log

Every run kept under `runs/`, the code version it ran with, and what it showed. The version is
the git commit each `jev run` / `jev eval` session recorded in the run's `manifest.json`
(`sessions[].git_sha`, `git_dirty`); `+` marks a dirty tree (uncommitted files, at least the
untracked job logs). The branch is where that commit lives. Add a row for every new run, with
its result once it is reported. Results: DeepSeek-judged, paired against the named baseline,
95% t-interval; results are stated on the benchmark's own metrics first. The team's registered
utility U (N + 0.15 R_c + 0.10 P + 0.10 S − 0.10 C/B0, docs/04; also the training reward and
selection score) is reported alongside as a secondary summary.

## Runs

| run | dates | experiment (config / variant) | commit (branch) | what it tested | result |
|---|---|---|---|---|---|
| `rollouts-train`, `rollouts-val` | 11-09 | `controller.yaml` rollouts | f22e269+ (main) | the fixed rule's episodes on train (34) / val (9): the start points of every later branch | source runs, no comparison |
| `branches-rollouts-{train,val}-b{0..3}-{abstract,explore,deepen,finish}` (27 runs) | 11-09 | branches | f22e269+ (main) | each retrieval action taken at batch 0-3, then the fixed rule: counterfactual returns | the labels for d1-sft |
| `groups-rollouts-train-r1-b{0,1}` | 11-09 | groups | 2a2ca33+ (rl-update) | RL groups, round 1 | adapter `controller/rl/r1` (`d1_rl1`), not carried forward |
| `groups-rollouts-train-r2-b{0,1,3}` | 11-09 | groups | 4e3f300+ (rl-update) | RL groups, round 2 | adapter `controller/rl/r2` (`d1_rl2`), not carried forward |
| `groups-rollouts-train-first-r1-b{0..3}` | 11-09 | groups (first-only) | bb52376+ (rl-update) | RL groups only at the first decision points that matter | adapter `controller/rl/first/r1`: d1-rl |
| `compare-rollouts-val-d1_{sft,rl1,rl2}` | 11-09 - 11-10 | compare | 4e3f300+, bb52376+ (rl-update) | the earlier adapters' episodes on val, paired returns | superseded by `d1_first1` |
| `compare-rollouts-val-d1_first1` | 11-10 | compare | bb52376+ (rl-update) | d1-rl's episodes on val | d1-sft and d1-rl (`d1_first1`) go to June dev |
| `eval-dev-trained` | 11-10 - 11-11 | `evaluate.yaml` / `dev-trained` | ed7e8e9+, 6434843+ (june-dev-compare) | fixed (3-6 batches), d1, d1-sft, d1-rl on 20 June dev tasks; `frontier.md` | fixed U .378 $.021; d1 .379 $.050; d1-sft .377 $.031; d1-rl .372 $.043 (cite_p −.090, p .016). Fixed b4/b5/b6 U .382/.396/.399; every d1 on or below the fixed curve at matched cost; stopping headroom U +.011 [−.036, .057] (b4) |
| `write-pilot-train`, `write-pilot-val` | 11-12 - 11-13 | `writing.yaml` / `train`, `val` | 3f41713+ (write-actions) | the fixed rule's drafts submitted, grounded, expanded, expanded then grounded | vs submit, train U: ground −.011, expand −.003, both −.007 (all across 0); per-task oracle U .436 vs .435 |
| `read-pilot-train`, `read-pilot-val` | 11-13 | `reading.yaml` / `train`, `val` | d374575+ (read-action) | the drafts submitted, expanded, read (revised from full text), redrafted from it; 763/764 PDFs | vs submit, train U: read −.003, redraft +.005, expand −.003 (all across 0); val redraft cite_p −.107; per-task oracle +.010 [−.013, .032] |
| `screen/zero-shot-{dev,train}` | 11-13 | `scripts/screen_eval.sh` (job 33326653; an offline scoring, no run manifest) | 90be206 (d1-screen) | d1-3B as released as the relevance screener on DeepSeek's verdicts (dev 1,781, train 3,143) | not usable untrained: AUC .72 vs DeepSeek (dev); best threshold P >= .2 agrees 68%; cascade at delta .8 decides 37% (dev) / 28% (train) but keeps 83% / 94% of cited papers (DeepSeek 91% / 95%) |
| `controller/screen`, `screen/sft-{dev,val}` | 11-13 | `scripts/screen_sft.sh` (job 33344984), recipe `screen` | 8bec12c (d1-screen) | d1 LoRA-trained on the train tasks' verdicts (3,143), epoch chosen on val, scored on June dev | usable: AUC vs DeepSeek .84 (dev); cascade at delta .7 (chosen on val: decides 68%, cited recall .974 = DeepSeek) decides 64% on dev with cited recall .944 (DeepSeek .907), kept .546 (.537) |
| `eval-dev-screen` | 11-14 | `evaluate.yaml` / `dev-screen`, `scripts/screen_dev.sh` (job 33372245) | 850866d+ (d1-screen) | the fixed rule at 3-6 batches with and without the d1 screener (+ 8 batches screened) on 20 June dev tasks; `frontier.md` | d1 decides 64% (agrees with DeepSeek on 88%), papers kept unchanged; cost −$.011/task at equal batches (b6 $.047 → $.034); screened − fixed at b3: nugget −.016, cite_p −.016 (n.s.), claim support −.037 (sig.); at b6 claim support +.056 (sig.); matched cost: screened b4 vs fixed b3 nugget +.015, cite_p −.017 (n.s.); b8 vs fixed b6 nugget −.045 (sig.) |
| `eval-dev2-screen` | 11-14 | `evaluate.yaml` / `dev2-screen`, `VARIANT=dev2-screen scripts/screen_dev.sh` (job 33407595) | 810cbbf+ (d1-screen) | the same comparison (b3-b6, with and without the screener) on 20 Nov dev2 tasks | nugget flat with search (fixed b3 .415, b6 .420), ref recall rises (.206 → .278); screened saves $.008-.018/task at equal batches; claim support +.036 (b3) and +.042 (b4), sig.; nugget and cite_p n.s. With June dev (40 tasks, b3-b6 pooled): cost −$.012, nugget −.008, cite_p +.016, claim support +.016 (n.s.) |
| `eval-test-screen` | 11-14 | `evaluate.yaml` / `test-screen`, `VARIANT=test-screen scripts/screen_dev.sh` (job 33413747, cancelled at 58 min: GPU idle; generation resumed by job 33417440 with JUDGE=0, judged by CPU job 33417446 `scripts/judge.sh`) | fe0f7bc+, b4384de+ (d1-screen) | held-out test (90 tasks): DS-base, simple RAG, fixed b3, screened b3 and b4; hypotheses fixed in docs/06 (test plan) | gen 21 min, judging 3:23; billed ~$7.0 gen, judge ~$6.6 list. H1 cost holds: screened b3 − fixed b3 −$.0076 [−.0080, −.0071] (−36%). H2: cite_p +.004 [−.018, .027] non-inferior; nugget −.013 [−.040, .015], claim support +.003 (n.s.); registered U lower bound −.035 misses −.03. H3: screened b4 − fixed b3 at −$.003: nugget +.009, ref recall +.014, cite_p −.014 (all n.s.). H4: controllers ~1/10 of DS-base cost; DS-base nugget +.044 to +.065, ref recall +.076 to +.090, cite_p −.165 to −.183, claim support −.10 (all sig.). d1 decides 66%. docs/06 test result |
| `eval-test-screen-180` | 11-14 | `evaluate.yaml` / `test-screen-180`, `VARIANT=test-screen-180 RUN=eval-test-screen-180 JUDGE=0 scripts/screen_dev.sh` + CPU `scripts/judge.sh` (GPU job 33434901 cancelled at 1:02 by the cluster, GPU idle while DS-base and simple RAG generated, done to simple_rag_k20_s1 81/180; its CPU judge job 33434902 cancelled with it; resumed by GPU job 33440489, judged by CPU job 33440494) | 1b7f3e6+ (d1-screen) | exploratory (decided after the 90-task result): same frozen systems on all 180 test tasks; first 90 replay from caches | gen 56 min (+1:02 cancelled), judging 7:12; billed $7.41 gen, $5.11 judge. First 90 not replayed exactly (judge cache missed; controllers 45-60/90 identical texts). Screened b3 − fixed: cost −$.0077, nugget +.008, cite_p +.005, claim support −.004 (all n.s.). Screened b4 − fixed at −$.0025: nugget +.023 [.005, .042], ref recall +.023 [.004, .042], cite_p −.023 [−.043, −.003] (sig.); U +.025 [.007, .043]. U: DS-base − screened b4 −.035 [−.061, −.009] (screened b4 higher). Judge noise on identical texts: per-task nugget sd .071, cite_p .038. Exploratory; docs/06 |
| `eval-drb2-test` | 11-15 | `evaluate.yaml` / `drb2-test`, `VARIANT=drb2-test RUN=eval-drb2-test JUDGE=0 sbatch scripts/screen_dev.sh` (GPU job 33520971, 18 min) + CPU `RUN=eval-drb2-test FRONTIER=0 scripts/judge.sh` (job 33520985, 30 min) | 763dc4c+ (d1-screen) | DRB2 transfer check, arXiv only: the frozen test-screen systems on 12 fresh DRB2 test tasks (6 en, 6 zh), two writer samples (docs/06: DRB2 transfer check) | $2.41 billed. Failed (empty, scored 0) of 12: Simple RAG 0, screened b4 2, screened b3 3, DS-base 3, fixed 5 (filter keeps no paper). total (all 12): DS-base .179, Simple RAG .208, fixed .110, screened .122 / .119. On the 7 tasks all wrote: DS-base .273, Simple RAG .241, fixed .188, screened b4 .175; screened b4 − DS-base −.098 [−.175, −.022]; screened − fixed n.s. Cost: DS-base $.310, fixed $.027, screened $.013 / $.015, Simple RAG $.006. Screener saving transfers; controllers do not (analysis ~.11 vs .26). docs/06 |

Earlier runs named in docs/usage.md (`p-dev`, `p-dev2`, `ctl-val`, `eval-dev`, `eval-dev2`,
`controller/sanity`, `controller/sweep-1`) are not in this checkout's `runs/`; docs/05 holds
their results.

## Trained adapters

| adapter | directory | sha256 (adapter_model.safetensors) | served as |
|---|---|---|---|
| d1-sft | `runs/controller/sft/adapter` | dabf7875e4091468400e50fa90a67c54d2dbce7969ac73d97df05280d03c45b6 | `ctl_d1_sft` (:9941) |
| d1-rl | `runs/controller/rl/first/r1/adapter` | 2fa87a6904389d5d0503cd46f5aaf0fe91107a7231ce753e4c30fa5256f79807 | `ctl_d1_rl` (:9942) |
| d1-screen | `runs/controller/screen/adapter` | 0ec04682ed2c880a34e279f24df0d121671629a0a0d7e8274425fa4fddfb502e | `ctl_screen*` screener (:9943) |

## Reading a run's version

```bash
python3 -c "import json,sys; [print(s['command'], s['started'][:16], s['git_sha'][:7], s['git_dirty']) for s in json.load(open(sys.argv[1]))['sessions']]" runs/RUN/manifest.json
git branch --contains SHA
```

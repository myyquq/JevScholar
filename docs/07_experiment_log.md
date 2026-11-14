# Experiment log

Every run kept under `runs/`, the code version it ran with, and what it showed. The version is
the git commit each `jev run` / `jev eval` session recorded in the run's `manifest.json`
(`sessions[].git_sha`, `git_dirty`); `+` marks a dirty tree (uncommitted files, at least the
untracked job logs). The branch is where that commit lives. Add a row for every new run, with
its result once it is reported. Results: DeepSeek-judged, paired against the named baseline,
95% t-interval; U = N + 0.15 R_c + 0.10 P + 0.10 S − 0.10 C/B0.

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
| `eval-dev-screen` | 11-14 | `evaluate.yaml` / `dev-screen`, `scripts/screen_dev.sh` (job 33372245) | 850866d+ (d1-screen) | the fixed rule at 3-6 batches with and without the d1 screener (+ 8 batches screened) on 20 June dev tasks; `frontier.md` | pending |

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

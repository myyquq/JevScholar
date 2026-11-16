# 7. d1 as the relevance screener

d1 as a controller (docs/05, docs/06) replaces a fixed rule that costs nothing, so it can win only
by deciding better, and no retrieval, writing or reading decision of the loop has resolvable room
for that (docs/06: decision headroom). This document uses d1 the other way: it takes over the
executor's most frequent short judgment, whether a retrieved paper is relevant, and the money
saved goes to more search. The fixed rule still decides the searches. The runs here were made on
the `d1-screen` line before the model version was pinned (docs/02).

## Why screening

DS-base's CoT `sem_filter` makes one executor call per screened paper. In the controlled loop it
is most of a task's cost: 59% of the fixed rule's at 3 batches ($.013 of $.022 a task on June
dev), 78% at 6 ($.037 of $.047). A d1 call is charged at $0.00002 per 1k prompt tokens, about 50
times less per paper. If d1 screened as well as the executor, the fixed rule at 6 batches (U
.399, nugget .242 on June dev, run `eval-dev-trained`) would cost about what it costs at 3 (U
.378, nugget .204).

## Labels and examples

`jev controller screen-data DIR` turns every executor verdict the runs logged (each task's
`state.json`: the papers `sem_filter` judged and its verdict) into d1 examples
(`controller/screen.py`). The state is the task and the paper's title and abstract, the text
`sem_filter` reads; the question is `sem_filter`'s own, as a choice between `relevant` and `not
relevant`; the target is the executor's verdict at 0.9, a soft label, since the teacher is not
ground truth. Each (task, paper) pair appears once, the first verdict logged. Tasks keep their
benchmark split: train tasks train, val tasks choose the checkpoint and the cascade threshold, June
dev tasks are the check; test tasks never appear. A cell whose parameters give d1 a screening role
(`screener`, or `screen`, d1's threshold filter) is left out, as its run's manifest records it, so
d1's own decisions never become labels. Every example's meta (task, split, URL, the executor's
verdict, whether the target paper cites it) is written beside it.

The adapter below was trained on 3,143 train verdicts, its threshold chosen on 821 val verdicts
(38 of cited papers), and it was checked on 1,781 June dev verdicts (54 of cited papers). The
executor keeps 91% of the screened papers the target paper cites on June dev, and about half of
the rest: losing cited papers is what a cheaper screener must not do.

## Scoring

`jev controller screen-eval DATA --out FILE [--adapter DIR]` scores d1 against the executor's
verdicts and against the target paper's real citations. d1's P(relevant) is averaged over both
option orders, since its answer codes are positional. Two uses are scored: d1 alone, keeping a
paper at P(relevant) ≥ t; and a cascade, where d1 decides a paper when its top probability
reaches δ and the executor decides the rest. The share d1 decides is the share of `sem_filter`
calls saved.

**d1 as released** (June dev). It ranks papers only weakly (AUC .72 against the executor's
verdicts, .77 for cited against not cited) and puts P(relevant) low for almost every paper
(median .23), so its confident answers are mostly "not relevant", and often wrong. As a cascade
at δ .8 it decides 37% of the papers and keeps 83% of the cited ones (executor 91%). Its best
plain threshold (keep at P ≥ .2) keeps 63% of the papers and 93% of the cited ones, agreeing with
the executor on 68%.

**d1 trained** (recipe `screen` in configs/training.yaml: LoRA on the train verdicts, two
epochs, the checkpoint of lowest val cross-entropy, which was the last; adapter sha256
`0ec04682…`; val cross-entropy .594, val agreement .749, calibration error .03). Training fixes
the calibration: AUC against the executor .72 → .84 on June dev (cited against not .77 → .81),
median P(relevant) .23 → .55.

| June dev (1,781 verdicts, 54 cited) | Agrees with executor | Kept | Cited kept | d1 decides |
|---|---|---|---|---|
| executor (CoT `sem_filter`) | 1.000 | .537 | .907 | 0 |
| d1 as released, cascade at δ .8 | .883 | .421 | .833 | .371 |
| d1 trained, alone at P ≥ .5 | .761 | .541 | .889 | 1 |
| d1 trained, cascade at δ .7 | .907 | .546 | .944 | .636 |
| d1 trained, cascade at δ .6 | .834 | .544 | .944 | .847 |

δ is chosen on val: at .7 the cascade keeps the same share of cited papers as the executor
(.974) while d1 decides 68%, agreeing with the executor on 88%; .6 decides 86% at the same
recall but agrees on only 82%. On June dev the cascade at .7 decides 64% of the papers and keeps
the same share of papers as the executor; its cited recall (.944 against .907) is 51 against 49
of the 54 cited papers.

## In the loop

A controlled cell with `screener` (`systems/controlled.py`; none by default) sends each chosen
search's new papers to the served adapter first, in one request with both option orders, charged
as `d1_choice`. A paper whose top probability reaches δ is decided by d1, the rest by
`sem_filter` as before, with its per-paper requests, so the executor's verdicts replay.
`screens.jsonl` logs every paper's P(relevant) and who decided it, and the report's meta counts
`d1_screened`. The fixed rule decides the searches, so a screened cell differs from the fixed
rule at the same batches only in who screens. `jev controller frontier RUN` compares the screened
cells with the unscreened fixed rule's cost–quality curve at matched cost.

```bash
scripts/train_env.sh jev controller serve --adapter runs/controller/screen/adapter --port 9943
# then, in another terminal:
uv run jev run evaluate --variant dev-screen --run-id eval-dev-screen
uv run jev eval eval-dev-screen && uv run jev report eval-dev-screen --baseline ctl_fixed
uv run jev controller frontier eval-dev-screen
```

## Development results

**June dev** (run `eval-dev-screen`, 20 tasks, writer samples averaged; bN = N search batches).
In the loop the screener behaves as it did offline: d1 decides 64% of the screened papers and
agrees with the executor on 88–89% of those the executor also judged. The screened runs keep as
many papers (b6: 29.5 of 54.9, unscreened 29.8 of 54.4) and as many of the cited ones (.95
against .91 at b6).

| System | Cost per task | Nugget | Ref. recall | cite_p | Claim support | Claim cov. |
|---|---|---|---|---|---|---|
| fixed b3 | $.021 | .210 | .220 | .606 | .892 | .773 |
| fixed b4 | $.028 | .231 | .218 | .583 | .880 | .782 |
| fixed b5 | $.041 | .243 | .247 | .549 | .886 | .793 |
| fixed b6 | $.047 | .253 | .244 | .530 | .872 | .775 |
| screened b3 | $.014 | .194 | .206 | .589 | .855 | .739 |
| screened b4 | $.019 | .225 | .222 | .589 | .885 | .773 |
| screened b5 | $.029 | .225 | .237 | .528 | .872 | .769 |
| screened b6 | $.034 | .221 | .252 | .594 | .928 | .812 |
| screened b8 | $.047 | .208 | .262 | .538 | .910 | – |

Screened − fixed at the same batches (paired, 95% t-interval; * the interval excludes 0):

| Batches | Cost | Nugget | Ref. recall | cite_p | Claim support |
|---|---|---|---|---|---|
| 3 | −$.0074* | −.016 [−.051, .018] | −.014 [−.030, .001] | −.016 [−.078, .045] | −.037 [−.066, −.008]* |
| 4 | −$.0092* | −.006 [−.041, .028] | +.004 [−.023, .031] | +.006 [−.054, .066] | +.005 [−.041, .051] |
| 5 | −$.0124* | −.018 [−.071, .036] | −.010 [−.040, .020] | −.021 [−.084, .041] | −.014 [−.048, .020] |
| 6 | −$.0134* | −.032 [−.068, .003] | +.008 [−.019, .035] | +.064 [−.013, .141] | +.056 [.017, .096]* |

- **Cost.** About −$.011 a task at the same batches (−28% at b6). `sem_filter`'s spend falls 42%
  at b6 although d1 decides 65% of the papers: the executor keeps the borderline papers, whose
  reasoning is longer. d1's own calls cost $.002 a task (about 1.4k prompt tokens a paper for
  both option orders).
- **Matched cost** (`frontier.md`). Screened b4 ($.019) against fixed b3 ($.021): nugget +.015
  [−.019, .048], cite_p −.017 [−.078, .044]. Screened b8 against fixed b6: nugget −.045
  [−.082, −.008]. More search raises nugget coverage slowly (fixed b3 → b6: +.043 for +$.026)
  and lowers cite_p (−.075), so the money saved buys little.

**Nov dev2** (run `eval-dev2-screen`, 20 tasks, the same cells without b8).

| System | Cost per task | Nugget | Ref. recall | cite_p | Claim support | Claim cov. |
|---|---|---|---|---|---|---|
| fixed b3 | $.021 | .415 | .206 | .576 | .858 | .755 |
| fixed b4 | $.032 | .384 | .224 | .539 | .861 | .745 |
| fixed b5 | $.045 | .383 | .267 | .549 | .857 | .767 |
| fixed b6 | $.054 | .420 | .278 | .521 | .869 | .768 |
| screened b3 | $.013 | .404 | .212 | .605 | .893 | .783 |
| screened b4 | $.021 | .411 | .235 | .588 | .903 | .789 |
| screened b5 | $.031 | .409 | .272 | .531 | .864 | .761 |
| screened b6 | $.036 | .391 | .275 | .558 | .898 | .798 |

| Batches | Cost | Nugget | Ref. recall | cite_p | Claim support |
|---|---|---|---|---|---|
| 3 | −$.0082* | −.012 [−.042, .019] | +.006 [−.017, .029] | +.029 [−.020, .079] | +.036 [.004, .067]* |
| 4 | −$.0117* | +.028 [−.030, .086] | +.011 [−.016, .038] | +.048 [−.019, .115] | +.042 [.008, .076]* |
| 5 | −$.0148* | +.026 [−.030, .083] | +.005 [−.023, .034] | −.018 [−.074, .038] | +.007 [−.037, .052] |
| 6 | −$.0179* | −.030 [−.092, .032] | −.003 [−.048, .042] | +.037 [−.029, .103] | +.030 [−.025, .084] |

Here nugget coverage does not rise with search at all (fixed b3 .415, b6 .420); reference recall
does (.206 → .278).

**Both dev sets** (40 tasks, b3–b6 averaged per task), screened − unscreened at the same
batches: cost −$.012 [−.014, −.010], nugget −.008 [−.028, .013], cite_p +.016 [−.008, .040],
claim support +.016 [−.001, .032]. The screener cuts the loop's cost by about a third at the same
search, with no metric detectably lower when pooled. It gives no clear gain at matched cost: in this range
more search raises reference recall but barely nugget coverage, and lowers cite_p.

## The test

**Registered before the run** (variant `test-screen`, run `eval-test-screen`). The first 90 test
tasks, two writer samples. Frozen as on dev: the adapter `0ec04682…` at δ .7; `ctl_fixed`,
`ctl_screen` and `ctl_screen_b4` are exactly the dev cells (pinned by `tests/test_screen.py`).
Systems: DS-base intro-only, Simple RAG (K = 20), the fixed rule at 3 batches, and screened at 3
and at 4 batches. The hypotheses:

1. **Cost** (primary): screened b3 costs less than fixed b3 per task, the interval below 0.
2. **Same quality** (primary, non-inferiority): screened b3 − fixed b3 has its 95% lower bound
   above −.03 in U (about the fixed rule's gain from 3 to 6 batches on June dev) and above −.05
   in cite_p.
3. **Matched cost** (secondary): screened b4 has a higher U than fixed b3, the interval above 0.
4. **Against DS-base** (descriptive): every controller against DS-base.

These 90 tasks are the main test's first 90 (docs/06), whose results were known before this test
was planned. The screener and δ were fixed on train, val and June dev only, but the tasks are not
untouched, and the intervals below do not account for their reuse. Every row comes from this
run's own generation and judging; pairs are within the run.

**Outcome** (DeepSeek judges). d1 decides 66% of the papers (3,432 of 5,186 in screened b3), as
on dev.

| System | Cost per task (% of DS-base) | Nugget | Ref. recall | cite_p | Claim support | Claim cov. | U |
|---|---|---|---|---|---|---|---|
| DS-base intro-only | $.188 (100%) | .419 | .250 | .409 | .775 | .687 | .502 |
| Simple RAG (K = 20) | $.003 (2%) | .351 | .114 | .477 | .823 | .719 | .496 |
| fixed b3 | $.0211 (11%) | .367 | .160 | .588 | .878 | .776 | .528 |
| screened b3 | $.0135 (7%) | .354 | .167 | .592 | .881 | .773 | .520 |
| screened b4 | $.0179 (10%) | .375 | .174 | .574 | .885 | .779 | .539 |

A − B, paired per task (95% t-interval; * the interval excludes 0; reference recall on the 83
tasks with gold; a report without a citation to check scores 0 on the citation metrics):

| A − B | Nugget | Ref. recall | cite_p | Claim support | Claim cov. | Cost | U |
|---|---|---|---|---|---|---|---|
| screened b3 − fixed b3 | −.013 [−.040, .015] | +.008 [−.008, .023] | +.004 [−.018, .027] | +.003 [−.014, .019] | −.003 [−.019, .013] | −$.0076* | −.008 [−.035, .019] |
| screened b4 − fixed b3 | +.009 [−.022, .039] | +.014 [−.007, .034] | −.014 [−.043, .015] | +.007 [−.010, .024] | +.002 [−.015, .020] | −$.0031* | +.011 [−.019, .041] |
| screened b4 − screened b3 | +.021 [−.005, .047] | +.006 [−.006, .018] | −.018 [−.047, .010] | +.005 [−.010, .020] | +.005 [−.010, .021] | +$.0045* | +.019 [−.007, .045] |
| DS-base − fixed b3 | +.052 [.009, .096]* | +.090 [.027, .154]* | −.179 [−.210, −.148]* | −.103 [−.131, −.076]* | −.090 [−.115, −.064]* | +$.1671* | −.025 [−.070, .020] |
| DS-base − screened b3 | +.065 [.018, .112]* | +.083 [.022, .143]* | −.183 [−.212, −.154]* | −.106 [−.131, −.081]* | −.086 [−.111, −.062]* | +$.1747* | −.017 [−.065, .031] |
| DS-base − screened b4 | +.044 [.007, .081]* | +.076 [.015, .138]* | −.165 [−.199, −.131]* | −.111 [−.133, −.089]* | −.092 [−.115, −.069]* | +$.1702* | −.036 [−.074, .001] |
| DS-base − Simple RAG | +.068 [.032, .105]* | +.136 [.071, .201]* | −.068 [−.101, −.035]* | −.048 [−.080, −.017]* | −.033 [−.061, −.004]* | +$.1851* | +.007 [−.032, .045] |
| Simple RAG − fixed b3 | −.016 [−.050, .019] | −.046 [−.084, −.007]* | −.111 [−.140, −.082]* | −.055 [−.083, −.027]* | −.057 [−.084, −.030]* | −$.0180* | −.032 [−.067, .003] |
| Simple RAG − screened b3 | −.003 [−.039, .033] | −.053 [−.092, −.015]* | −.115 [−.147, −.083]* | −.057 [−.083, −.032]* | −.054 [−.079, −.029]* | −$.0105* | −.024 [−.059, .011] |
| Simple RAG − screened b4 | −.024 [−.050, .002] | −.060 [−.099, −.020]* | −.097 [−.130, −.064]* | −.062 [−.086, −.038]* | −.059 [−.084, −.035]* | −$.0149* | −.043 [−.070, −.016]* |

Against the hypotheses:

1. **Cost holds.** Screened b3 − fixed b3 = −$.0076 [−.0080, −.0071] (−36%).
2. **Same quality is not shown.** cite_p +.004 [−.018, .027] clears −.05, but U −.008 [−.035,
   .019] does not clear −.03. On each of the benchmark's metrics screened b3 is within noise of
   fixed b3 (nugget −.013 [−.040, .015], claim support +.003).
3. **Matched cost is not shown.** Screened b4 costs $.0031 less than fixed b3, with U +.011
   [−.019, .041]; nugget (+.009), reference recall (+.014) and claim support (+.007) are higher
   and cite_p (−.014) lower, none resolvably.
4. **Against DS-base.** Every controller costs about a tenth of DS-base. DS-base is more complete
   (nugget +.044 to +.065, reference recall +.076 to +.090) and its citations are less reliable
   (cite_p −.165 to −.183, claim support −.103 to −.111); no U difference is resolved.

The screener's saving carries over to the test tasks (−36% at the same batches) with no metric
detectably lower, but the registered non-inferiority in U is not shown at 90 tasks, and spending
the saving on a fourth batch moves completeness up only within noise.

## d1 as screener in DraftRef

DraftRef (docs/06) also uses d1 to screen, in a different place and form: d1 as released, alone
at P(relevant) ≥ .2942, in place of DS-base's filter. That is close to the plain threshold above
(P ≥ .2) and has the same weak ranking, which matters little there: the draft's references pick
96% of DraftRef's writer documents, and d1 only tops up the list on the 43 of 180 test tasks
whose draft names fewer than 30 papers. Inside the controlled loop, d1's verdicts decide what the
writer can receive. There d1 as released keeps as many cited papers only by keeping 63% of all
papers and agreeing with the executor on 68%; the trained cascade keeps the executor's share of
papers and of cited papers while agreeing with it on 91%.

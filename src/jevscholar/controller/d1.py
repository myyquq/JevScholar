"""LiquidAI/d1-3B as a trainable decision model.

d1 answers a typed question about a state in one forward pass: the prompt ends at the answer slot,
and each option's probability is the softmax, over the options, of its best token form's
log-probability there. Nothing is decoded. The checkpoint's remote code is the reference: prompts
are rendered and options mapped to tokens by its own `prompt` module, so training optimises
exactly what `system_one` reads, and an adapter trained here is served by it unchanged.

Training is cross-entropy over the options against a target distribution (a one-hot label, graded
soft targets, or the best options sharing the mass), with LoRA on the language model's projections
(the vision tower is untouched). Option codes are positional (A, B, ...), so the option order is
reshuffled per example and epoch. Rows are right-padded: the hybrid stack is causal, so a pad can
never reach a real token, and the logits are computed at each row's answer slot only.

Loading d1 turns the flash SDPA kernel off for the process (d1 runs in its own): with GQA and more
than one row it drifts in bf16 (a 3-option check: 0.6045 batched against 0.6195 in fp32), while
the memory-efficient kernel matches fp32 and the checkpoint's own one-row `system_one`. It has to
be process-wide, since gradient checkpointing recomputes the forward during backward.

torch, transformers and peft are imported inside functions: they live in `.venv-train`, and the
pure helpers here (shuffling, batching, metrics, the checkpoint's prompt module) stay importable
without them.

How a prompt is rendered (`render`: the checkpoint's `style`, `system` and `option_style`) is part
of every request and every example; the default is the checkpoint's own (`json_only`, no system
turn, `desc`).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import math
import random
import shutil
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import yaml

from jevscholar.paths import ROOT
from jevscholar.types import Strict
from jevscholar.util import atomic_write, read_jsonl, write_jsonl

if TYPE_CHECKING:
    import torch

MODEL = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"  # 2026-10-07
MAX_TOKENS = 7552  # the longest prompt d1 was validated on
# The language model's projections: conv in/out, attention q/k/v/out, MLP w1-w3.
LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(conv\.(in|out)_proj|self_attn\.(q|k|v|out)_proj|feed_forward\.w[123])"
)


# --------------------------------------------------------------------------- the prompt
def prompt_module() -> ModuleType:
    """The pinned checkpoint's own prompt module (pure Python), from the local snapshot: it renders
    d1's prompts in both environments, so the prompted policy can read exactly what d1 reads."""
    if (module := sys.modules.get("d1_prompt")) is None:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
        spec = importlib.util.spec_from_file_location("d1_prompt", path / "prompt.py")
        module = sys.modules["d1_prompt"] = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def user_turn(state: Any, question: dict[str, Any], render: dict[str, str] | None = None) -> str:
    """d1's prompt for a question as the checkpoint renders it: the system turn's text, if any,
    then the user turn (the state block, then the question: a choice with its options coded A, B,
    ..., which for at most 26 options never needs the tokenizer's fallback codes, or a yes/no
    question)."""
    pm, r = prompt_module(), {"style": "json_only", "system": "none", "option_style": "desc"}
    r |= render or {}
    q = pm.as_question(question)
    if isinstance(q, pm.Choice):
        labels = list(q.criteria)
        lines = "\n".join(
            pm._option_line(code, lab, q.criteria[lab], r["option_style"])
            for code, lab in zip(pm.option_codes(labels), labels, strict=True)
        )
        body = f"{q.instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."
    else:
        body = pm.question_block(None, q)  # a yes/no question needs no tokenizer
    system = pm.SYSTEMS[r["system"]]
    return (
        f"{system}\n\n" if system else ""
    ) + f"{pm.state_block(state, r['style'])}\nQUESTION:\n{body}"


# --------------------------------------------------------------------------- examples
@dataclass(frozen=True)
class Example:
    """A state, a `choice` question in d1's schema and the target distribution over its options
    (by option name; options it leaves out get 0), rendered as `render` says."""

    state: Any  # text or JSON
    question: dict[str, Any]  # {"type": "choice", "instructions": str, "criteria": {name: desc}}
    target: dict[str, float]
    weight: float = 1.0  # relative to the other examples of an optimiser step
    render: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.question.get("type", "choice") != "choice":
            raise ValueError("only choice questions are trained")
        if bad := set(self.target) - set(self.question["criteria"]):
            raise ValueError(f"targets {sorted(bad)} are not options")
        if abs(sum(self.target.values()) - 1) > 1e-6 or min(self.target.values()) < 0:
            raise ValueError("the target is a distribution over the options")

    @classmethod
    def labelled(cls, state: Any, question: dict[str, Any], label: str, **kw: Any) -> Example:
        """An example whose target is one option."""
        return cls(state, question, {label: 1.0}, **kw)

    def targets(self) -> list[float]:
        """The target in the question's option order."""
        return [self.target.get(name, 0.0) for name in self.question["criteria"]]

    def label(self) -> int:
        """The most probable target option's index."""
        t = self.targets()
        return t.index(max(t))


def read_examples(path: Path) -> list[Example]:
    return [Example(**e) for e in read_jsonl(path)]


def write_examples(path: Path, examples: Iterable[Example]) -> None:
    write_jsonl(path, map(asdict, examples))


@dataclass(frozen=True)
class RLExample:
    """A decision a sampling policy made (controller/groups.py): exactly what d1 read (state,
    question in the order it saw, render), the option it chose, that option's probability then
    and the trajectory's group-relative advantage."""

    state: Any
    question: dict[str, Any]
    chosen: int  # index into the question's options
    old_prob: float
    advantage: float
    render: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)  # where it came from (not read by d1)

    def __post_init__(self) -> None:
        if self.question.get("type", "choice") != "choice":
            raise ValueError("only choice questions are trained")
        if not 0 <= self.chosen < len(self.question["criteria"]):
            raise ValueError(f"option {self.chosen} is not among the question's options")
        if not 0 < self.old_prob <= 1:
            raise ValueError(f"the chosen option's probability {self.old_prob} is not in (0, 1]")


def read_rl_examples(path: Path) -> list[RLExample]:
    return [RLExample(**e) for e in read_jsonl(path)]


def write_rl_examples(path: Path, examples: Iterable[RLExample]) -> None:
    write_jsonl(path, map(asdict, examples))


def reorder(question: dict[str, Any], order: Sequence[int]) -> dict[str, Any]:
    """The question with its options in `order` (a permutation of their positions)."""
    items = list(question["criteria"].items())
    return {**question, "criteria": dict(items[i] for i in order)}


def batches(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Consecutive index batches whose padded size (rows x longest row) stays within `budget`
    tokens; a row longer than the budget gets a batch of its own."""
    out: list[list[int]] = []
    for i, n in enumerate(lengths):
        if out and max(n, *(lengths[j] for j in out[-1])) * (len(out[-1]) + 1) <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def regret(
    probs: Sequence[Sequence[float]], points: Sequence[dict[str, Any]], profile: Any = None
) -> dict:
    """d1's regret on labelled decision points ({"kinds", "returns"} each, options in `probs`'
    order), each the best return less the chosen option's, averaged: deployed (the option
    `profile.pick` takes, d1's default profile), expected (under the distribution), and among
    the searches alone (the most probable search, at points offering two or more)."""
    from jevscholar.controller.prompt import Profile
    from jevscholar.controller.state import FINISH, Candidate

    profile = profile or Profile()
    deployed, expected, search = [], [], []
    for p, point in zip(probs, points, strict=True):
        kinds, r = point["kinds"], point["returns"]
        if len(p) != len(kinds):
            raise ValueError(f"{len(p)} probabilities for {len(kinds)} options")
        best = max(r)
        deployed.append(best - r[profile.pick([Candidate(k) for k in kinds], p)])
        expected.append(best - sum(q * v for q, v in zip(p, r, strict=True)))
        if len(searches := [i for i, k in enumerate(kinds) if k != FINISH]) > 1:
            search.append(max(r[i] for i in searches) - r[max(searches, key=p.__getitem__)])
    return {
        "points": len(points),
        "regret": sum(deployed) / len(deployed),
        "expected_regret": sum(expected) / len(expected),
        "search_points": len(search),
        "search_regret": sum(search) / len(search) if search else None,
    }


def calibration(probs: Sequence[Sequence[float]], labels: Sequence[int], bins: int = 10) -> dict:
    """Accuracy, NLL, Brier score and expected calibration error (on the top probability)."""
    n, ece = len(labels), 0.0
    rows = list(zip(probs, labels, strict=True))
    top = [(max(p), p.index(max(p)) == y) for p, y in rows]
    for b in range(bins):
        hits = [(c, h) for c, h in top if min(int(c * bins), bins - 1) == b]
        ece += abs(sum(h for _, h in hits) - sum(c for c, _ in hits)) / n
    return {
        "n": n,
        "accuracy": sum(h for _, h in top) / n,
        "nll": -sum(math.log(max(p[y], 1e-12)) for p, y in rows) / n,
        "brier": sum(sum((q - (k == y)) ** 2 for k, q in enumerate(p)) for p, y in rows) / n,
        "ece": ece,
    }


# --------------------------------------------------------------------------- the model
class D1:
    """The checkpoint (optionally with a LoRA adapter), its tokenizer and its prompt module."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.base = model.get_base_model() if hasattr(model, "get_base_model") else model
        package = type(self.base).__module__.rsplit(".", 1)[0]
        self.prompt = importlib.import_module(f"{package}.prompt")  # the checkpoint's own renderer
        bos = getattr(tokenizer, "bos_token", None)
        self.bos = bos if isinstance(bos, str) else ""  # as the remote SystemOne does
        self.lead = self.prompt.default_lead(self.base.config.model_type)

    @classmethod
    def load(
        cls,
        *,
        adapter: Path | None = None,
        lora: dict[str, Any] | None = None,
        revision: str = REVISION,
        device: str = "cuda",
    ) -> D1:
        """The pinned checkpoint; with `adapter` a trained one (trainable when `lora` is given),
        with `lora` alone a fresh adapter."""
        import torch
        from huggingface_hub import snapshot_download
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        try:  # the pinned snapshot is immutable: the cache is enough once it is there
            path = snapshot_download(MODEL, revision=revision, local_files_only=True)
        except FileNotFoundError:
            path = snapshot_download(MODEL, revision=revision)
        torch.backends.cuda.enable_flash_sdp(False)  # see the module docstring
        model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16)
        if adapter is not None:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=lora is not None)
        elif lora is not None:
            model = get_peft_model(model, LoraConfig(target_modules=LORA_TARGETS, **lora))
        return cls(model.to(device), AutoTokenizer.from_pretrained(path))

    def encode(
        self, state: Any, question: dict[str, Any], render: dict[str, str] | None = None
    ) -> tuple[list[int], list[list[int]]]:
        """Prompt ids up to the answer slot, and each option's token ids (its forms)."""
        q = self.prompt.as_question(question)
        text = self.prompt.render(self.tokenizer, state, q, self.bos, self.lead, **(render or {}))
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids, self.prompt.readout_ids(self.tokenizer, q)

    def option_logprobs(
        self, rows: Sequence[list[int]], groups: Sequence[list[list[int]]]
    ) -> list[torch.Tensor]:
        """Each row's log-probabilities over its options, differentiable."""
        import torch

        device = next(self.model.parameters()).device
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(rows), max(map(len, rows))), pad, dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row)
        hidden = self.base.model.language_model(input_ids=ids.to(device)).last_hidden_state
        slots = torch.tensor([len(r) - 1 for r in rows], device=device)
        logits = self.base.lm_head(hidden[torch.arange(len(rows), device=device), slots]).float()
        logz = logits.log_softmax(-1)
        out = []
        for z, forms in zip(logz, groups, strict=True):
            best = torch.stack([z[torch.tensor(f, device=device)].max() for f in forms])
            out.append(best.log_softmax(-1))
        return out

    def probabilities(
        self,
        items: Sequence[tuple[Any, dict[str, Any]]],
        batch_tokens: int = 32768,
        render: dict[str, str] | None = None,
    ) -> list[list[float]]:
        """Option probabilities of (state, question) items, in each question's own order."""
        import torch

        encoded = [self.encode(s, q, render) for s, q in items]
        out: list[list[float]] = [[] for _ in items]
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            for batch in batches([len(ids) for ids, _ in encoded], batch_tokens):
                rows, forms = zip(*(encoded[i] for i in batch), strict=True)
                for i, lp in zip(batch, self.option_logprobs(rows, forms), strict=True):
                    out[i] = lp.exp().tolist()
        self.model.train(was_training)
        return out

    def rows(self, examples: Sequence[Example]) -> list[list[float]]:
        """Each example's option probabilities in its stored order, read one row at a time as the
        server reads them (batched rows drift by up to about 0.03 in bf16)."""
        return [self.probabilities([(e.state, e.question)], render=e.render)[0] for e in examples]

    def evaluate(
        self, examples: Sequence[Example], points: Sequence[dict[str, Any]] | None = None
    ) -> dict:
        """Calibration metrics against each example's most probable target option and the
        cross-entropy against its whole target, read row by row in the stored option order;
        with the examples' labelled `points`, also d1's regret there."""
        probs = self.rows(examples)
        ce = [
            -sum(t * math.log(max(q, 1e-12)) for t, q in zip(e.targets(), p, strict=True))
            for e, p in zip(examples, probs, strict=True)
        ]
        out = calibration(probs, [e.label() for e in examples]) | {"ce": sum(ce) / len(ce)}
        return out if points is None else out | regret(probs, points)


# --------------------------------------------------------------------------- training
class Lora(Strict):  # peft LoraConfig fields
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class TrainConfig(Strict):
    train: Path  # Example JSONL
    dev: Path
    dev_points: Path | None = None  # the dev examples' labelled points: select by regret
    out: Path  # steps/<n>/adapter, adapter/ (the selected step), metrics.json, config.json
    init_adapter: Path | None = None  # continue from this adapter (else a fresh one)
    revision: str = REVISION
    lora: Lora = Lora()
    lr: float = 2e-4
    schedule: Literal["cosine", "constant"] = "cosine"  # after the warmup
    epochs: int = 2
    warmup: float = 0.05  # share of optimiser steps (at least one)
    batch_tokens: int = 8192  # padded tokens per forward
    accum: int = 4  # forwards per optimiser step
    # optimiser steps between kept, evaluated adapters, or each epoch's last (and the last step)
    eval_every: int | Literal["epoch"] = 1
    max_tokens: int = MAX_TOKENS  # longer prompts are dropped
    checkpointing: bool = True
    seed: int = 0


class RLConfig(Strict):
    """One registered RL round (docs/06): its rollouts and its update."""

    kind: Literal["rl"]
    source: str  # the train run whose points are restored (under runs/)
    system: str  # its controlled system
    points: list[tuple[str, int]]  # (task key, batches before the point), frozen before rollouts
    samples: int = 4  # members per point (a group)
    prefix: str  # rollout runs: runs/<prefix>-b<batches>
    run_usd: float = 0.5  # billed cap of each rollout run
    steps: Path  # `jev controller rl-data` output: the sampled decisions with their advantages
    out: Path  # steps/<n>/adapter, adapter/ (the kept step), metrics.json, config.json
    init_adapter: Path  # the adapter the rollouts sampled from
    reference: Path  # the KL anchor (d1-sft)
    revision: str = REVISION
    lr: float = 2e-5  # AdamW, constant, no weight decay
    optimizer_steps: int = 4
    groups_per_step: int = 2
    clip: float = 0.2  # the probability ratio is clipped to [1 - clip, 1 + clip]
    beta: float = 0.05  # weight of KL(pi || pi_reference), averaged per decision
    max_kl: float = 0.05  # nats per decision: training stops at the first step above it
    parity: float = 5e-3  # the update's readout of the init adapter against the logged probs
    seed: int = 0


def load_recipe(path: Path, name: str) -> TrainConfig | RLConfig:
    """A named recipe of a training file (its paths relative to the repository): an RL round
    when it says `kind: rl`, else a supervised one."""
    recipes = yaml.safe_load(path.read_text())
    if name not in recipes:
        raise ValueError(f"{path.name} has no recipe {name!r}; it has {sorted(recipes)}")
    raw = recipes[name]
    keys = ("train", "dev", "dev_points", "out", "init_adapter", "steps", "reference")
    paths = {k: ROOT / raw[k] for k in keys if raw.get(k)}
    return (RLConfig if raw.get("kind") == "rl" else TrainConfig)(**raw | paths)


def shuffled(examples: Sequence[Example], rng: random.Random) -> Iterator[Example]:
    """The examples in a random order, each with its options in a random order."""
    for i in rng.sample(range(len(examples)), len(examples)):
        e = examples[i]
        order = rng.sample(range(len(e.question["criteria"])), len(e.question["criteria"]))
        yield dataclasses.replace(e, question=reorder(e.question, order))


def train(cfg: TrainConfig, log: Any = print) -> dict:
    """LoRA-train d1 on cfg.train (from cfg.init_adapter, if set), keeping and evaluating the
    adapter before training (step 0) and every `eval_every` optimiser steps (or at each epoch's
    end); out/adapter is the selected step's: the lowest dev
    regret (with `dev_points`), then the lowest dev cross-entropy, then the earliest. Each
    optimiser step averages its examples' cross-entropy by weight."""
    import torch

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    d1 = D1.load(adapter=cfg.init_adapter, lora=cfg.lora.model_dump(), revision=cfg.revision)
    if cfg.checkpointing:
        no_reentrant = {"use_reentrant": False}  # LoRA alone trains: inputs need no grad
        d1.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=no_reentrant)
    data, dev = read_examples(cfg.train), read_examples(cfg.dev)
    if not data or not dev:
        raise ValueError("training needs train and dev examples")
    points = None if cfg.dev_points is None else read_jsonl(cfg.dev_points)
    if points is not None and len(points) != len(dev):
        raise ValueError(f"{len(points)} dev points for {len(dev)} dev examples")
    params = [p for p in d1.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    epochs = [batches_of(d1, list(shuffled(data, rng)), cfg) for _ in range(cfg.epochs)]
    total = sum(math.ceil(len(encoded) / cfg.accum) for encoded in epochs)  # optimiser steps
    warm = max(1, int(cfg.warmup * total))

    def factor(s: int) -> float:
        ramp = (s + 1) / warm
        if cfg.schedule == "constant":
            return min(ramp, 1.0)
        return min(ramp, 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    cfg.out.mkdir(parents=True, exist_ok=True)
    atomic_write(cfg.out / "config.json", cfg.model_dump_json(indent=2))
    history: list[dict[str, Any]] = []

    def keep(step: int, epoch: int, train_ce: float | None) -> None:
        d1.model.save_pretrained(str(cfg.out / "steps" / str(step) / "adapter"))
        row = {"step": step, "epoch": epoch, "train_ce": train_ce}
        history.append(row | {"dev": d1.evaluate(dev, points)})
        log(f"step {step} (epoch {epoch}) train_ce {train_ce} dev {history[-1]['dev']}")

    keep(0, 0, None)
    step = 0
    for epoch, encoded in enumerate(epochs, 1):
        d1.model.train()
        for start in range(0, len(encoded), cfg.accum):
            group = encoded[start : start + cfg.accum]
            total_weight = sum(row[3] for batch in group for row in batch)
            seen = loss_sum = 0.0
            for batch in group:
                lps = d1.option_logprobs([b[0] for b in batch], [b[1] for b in batch])
                device = lps[0].device
                ce = torch.stack(
                    [
                        -(torch.tensor(b[2], device=device) * lp).sum()
                        for lp, b in zip(lps, batch, strict=True)
                    ]
                )
                weights = torch.tensor([b[3] for b in batch], device=device)
                ((weights * ce).sum() / total_weight).backward()
                seen, loss_sum = seen + len(batch), loss_sum + ce.detach().sum().item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if cfg.eval_every == "epoch":
                due = start + cfg.accum >= len(encoded)
            else:
                due = step % cfg.eval_every == 0
            if due or step == total:  # the last step is always kept
                keep(step, epoch, loss_sum / seen)
                d1.model.train()
    assert step == total and history[-1]["step"] == total
    best = min(history, key=lambda h: (*selection_key(h["dev"]), h["step"]))
    selected = cfg.out / "adapter"
    shutil.rmtree(selected, ignore_errors=True)
    shutil.copytree(cfg.out / "steps" / str(best["step"]) / "adapter", selected)
    result = {"history": history, "best_step": best["step"], "best_dev": best["dev"]}
    atomic_write(cfg.out / "metrics.json", json.dumps(result, indent=2))
    return result


def selection_key(dev: dict[str, Any]) -> tuple[float, ...]:
    """How a kept adapter ranks: its dev regret (when the dev points are labelled), then its
    dev cross-entropy; lower is better."""
    return (dev["regret"], dev["ce"]) if "regret" in dev else (dev["ce"],)


def batches_of(d1: D1, examples: Sequence[Example], cfg: TrainConfig) -> list[list[tuple]]:
    """Encoded (ids, option forms, targets, weight) batches, overlong prompts dropped."""
    rows = []
    for e in examples:
        ids, forms = d1.encode(e.state, e.question, e.render)
        if len(ids) <= cfg.max_tokens:
            rows.append((ids, forms, e.targets(), e.weight))
    return [[rows[i] for i in b] for b in batches([len(r[0]) for r in rows], cfg.batch_tokens)]


# --------------------------------------------------------------------------- RL
def grpo_terms(
    lp: torch.Tensor, old: Sequence[float], ref: Sequence[float], action: int, advantage: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, bool]:  # fmt: skip
    """One decision's clipped surrogate -min(rA, clip(r)A), with r the sampled option's
    probability ratio to `old`, its exact categorical KL(pi || ref), and whether the ratio was
    clipped. `lp` holds the current log-probabilities over the options."""
    import torch

    ratio = torch.exp(lp[action] - math.log(old[action]))
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    surrogate = -torch.minimum(ratio * advantage, clipped * advantage)
    log_ref = torch.log(torch.tensor(ref, device=lp.device).clamp_min(1e-12))
    kl = (lp.exp() * (lp - log_ref)).sum()
    return surrogate, kl, bool(abs(ratio.item() - clipped.item()) > 1e-12)


def rl_update(cfg: RLConfig, log: Any = print) -> dict:
    """One registered RL round on the sampled decisions in `cfg.steps`: the usable groups in an
    order fixed by the seed, `groups_per_step` per optimiser step; per step, each trajectory's
    clipped surrogates (ratios to the logged sampling probabilities) summed and averaged over
    the step's trajectories, plus beta times the KL to the reference averaged per decision. After
    every step an adapter is kept and the mean KL over every sampled decision (usable groups or
    not) is measured; training stops at the first step above `max_kl`, and out/adapter is the
    last step within it (step 0: the init adapter). Readouts are row by row, dropout off. The
    run is built in out/staging and replaces out/steps, out/adapter and out/metrics.json only
    once it is complete."""

    from jevscholar.controller.rl import read_steps
    from jevscholar.controller.serve import adapter_sha

    steps = read_steps(cfg.steps)
    want = adapter_sha(cfg.init_adapter)
    if bad := {s.adapter for s in steps} - {want}:
        raise ValueError(f"rollouts sampled from {bad}, not the init adapter {want}")
    usable = [i for i, s in enumerate(steps) if s.usable]
    staging = cfg.out / "staging"
    shutil.rmtree(staging, ignore_errors=True)
    history: list[dict[str, Any]] = []
    kept = 0
    if not usable:  # no usable group: no update
        shutil.copytree(cfg.init_adapter, staging / "steps" / "0" / "adapter")
    else:
        history, kept = _rl_steps(cfg, steps, usable, staging, log)
    result = {"update": bool(usable), "kept_step": kept, "history": history}
    for name in ("steps", "adapter"):  # promote the complete run
        shutil.rmtree(cfg.out / name, ignore_errors=True)
    (staging / "steps").rename(cfg.out / "steps")
    shutil.copytree(cfg.out / "steps" / str(kept) / "adapter", cfg.out / "adapter")
    atomic_write(cfg.out / "config.json", cfg.model_dump_json(indent=2))
    atomic_write(cfg.out / "metrics.json", json.dumps(result, indent=2))
    shutil.rmtree(staging)
    return result


def _rl_steps(
    cfg: RLConfig, steps: Sequence[Any], usable: list[int], staging: Path, log: Any
) -> tuple[list[dict[str, Any]], int]:
    """The optimiser steps of `rl_update`, their adapters under staging/steps: (history, the
    kept step)."""
    import torch

    def readout(d1: D1) -> list[list[float]]:
        return [d1.probabilities([(s.state, s.question)], render=s.render)[0] for s in steps]

    torch.manual_seed(cfg.seed)
    reference = D1.load(adapter=cfg.reference, revision=cfg.revision)
    ref = readout(reference)
    del reference
    torch.cuda.empty_cache()
    d1 = D1.load(adapter=cfg.init_adapter, lora=Lora().model_dump(), revision=cfg.revision)
    d1.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for module in d1.model.modules():  # the ratio and the KL are read with dropout off
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    pairs = zip(steps, readout(d1), strict=True)
    parity = max(abs(a - b) for s, o in pairs for a, b in zip(s.old, o, strict=True))
    if parity > cfg.parity:
        raise ValueError(f"the init adapter reads {parity:.4f} away from the logged probabilities")
    encoded = {i: d1.encode(steps[i].state, steps[i].question, steps[i].render) for i in usable}
    groups = sorted({steps[i].group for i in usable})
    random.Random(cfg.seed).shuffle(groups)
    plan = [
        groups[i : i + cfg.groups_per_step]
        for i in range(0, len(groups), cfg.groups_per_step)
    ][: cfg.optimizer_steps]  # fmt: skip
    params = [p for p in d1.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    def mean_kl() -> float:
        now = readout(d1)
        return sum(
            sum(
                p * (math.log(max(p, 1e-12)) - math.log(max(q, 1e-12)))
                for p, q in zip(n, r, strict=True)
            )
            for n, r in zip(now, ref, strict=True)
        ) / len(steps)

    d1.model.save_pretrained(str(staging / "steps" / "0" / "adapter"))
    history = [{"step": 0, "kl": mean_kl(), "parity": parity, "decisions": len(steps)}]
    kept = 0
    for k, chosen in enumerate(plan, 1):
        d1.model.train()
        idx = [i for i in usable if steps[i].group in chosen]
        trajectories = {(steps[i].group, steps[i].trajectory) for i in idx}
        loss_sum = clipped = 0.0
        for i in idx:
            s = steps[i]
            (lp,) = d1.option_logprobs([encoded[i][0]], [encoded[i][1]])
            surrogate, kl, was_clipped = grpo_terms(lp, s.old, ref[i], s.action, s.advantage,
                                                    cfg.clip)  # fmt: skip
            loss = surrogate / len(trajectories) + cfg.beta * kl / len(idx)
            loss.backward()
            loss_sum, clipped = loss_sum + loss.item(), clipped + was_clipped
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        d1.model.save_pretrained(str(staging / "steps" / str(k) / "adapter"))
        row = {"step": k, "groups": chosen, "decisions": len(idx), "loss": loss_sum}
        history.append(row | {"clip_fraction": clipped / len(idx), "kl": mean_kl()})
        log(f"RL step {k}: {history[-1]}")
        if history[-1]["kl"] > cfg.max_kl:
            break
        kept = k
    return history, kept


# --------------------------------------------------------------------------- RL groups
class GroupRLConfig(Strict):
    """One update on RL groups (controller/groups.py, docs/06): the clipped ratio objective on
    group-relative advantages, plus beta times the exact categorical KL to the frozen reference.

    pi_old is by default recomputed with the training readout (the same batches, dropout off)
    before any step, so the first step's ratio is exactly 1: the served one-row readout differs
    from a batched bf16 one by up to a few hundredths, which would otherwise read as policy
    change and be clipped. The logged probabilities stay the check (`readout_parity`). The epoch
    is split into `steps` optimiser steps whatever its size, so the learning rate alone bounds
    how far one update moves."""

    data: Path  # RLExample JSONL (`jev controller groups-data`)
    adapter: Path  # the adapter that sampled the data: the one updated
    reference: Path  # the SFT adapter: the frozen KL reference
    out: Path  # adapter/ (unless reverted), metrics.json, config.json
    revision: str = REVISION
    lr: float = 5e-5
    clip: float = 0.2  # the ratio is clipped to [1 - clip, 1 + clip]
    beta: float = 0.05  # weight of KL(pi || pi_ref)
    kl_max: float = 0.05  # revert (do not save) above this mean KL per decision, in nats
    steps: int = 4  # optimiser steps in the epoch
    old: Literal["recompute", "logged"] = "recompute"  # where pi_old(chosen) comes from
    batch_tokens: int = 8192
    max_tokens: int = MAX_TOKENS
    checkpointing: bool = True
    seed: int = 0


def rl_terms(
    lp: torch.Tensor,
    ref_lp: torch.Tensor,
    chosen: int,
    old_logprob: float,
    advantage: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(policy loss, KL(pi || pi_ref), ratio) of one decision, from the option log-probabilities
    now (`lp`, differentiable) and under the reference: the loss is -min(r A, clip(r) A), with
    r = pi(chosen) / pi_old(chosen)."""
    import torch

    ratio = (lp[chosen] - old_logprob).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    kl = (lp.exp() * (lp - ref_lp)).sum()
    return -surrogate, kl, ratio


def categorical_kl(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q) in nats."""
    return sum(a * (math.log(a) - math.log(max(b, 1e-12))) for a, b in zip(p, q, strict=True) if a)


def frozen_logprobs(d1: Any, rows: Sequence[tuple], groups: Sequence[list[int]]) -> list[Any]:
    """Option log-probabilities of encoded (ids, forms, ...) rows, read in the given batches,
    without gradients or dropout, on the CPU."""
    import torch

    out: list[Any] = [None] * len(rows)
    was_training = d1.model.training
    d1.model.eval()
    with torch.no_grad():
        for batch in groups:
            lps = d1.option_logprobs([rows[i][0] for i in batch], [rows[i][1] for i in batch])
            for i, lp in zip(batch, lps, strict=True):
                out[i] = lp.detach().float().cpu()
    d1.model.train(was_training)
    return out


def split(items: Sequence[Any], parts: int) -> list[list[Any]]:
    """`items` in at most `parts` consecutive, nearly equal chunks."""
    size = math.ceil(len(items) / max(parts, 1))
    return [list(items[i : i + size]) for i in range(0, len(items), size)] if items else []


def group_rl_update(cfg: GroupRLConfig, log: Any = print, load: Any = None) -> dict:
    """One epoch of the bounded RL update over cfg.data, from cfg.adapter, against the frozen
    cfg.reference, in cfg.steps optimiser steps (each averages its decisions' loss), dropout
    off. The adapter is saved to out/adapter unless the mean KL to the reference after the
    update exceeds cfg.kl_max (then reverted: nothing saved). `load` stands in for D1.load."""
    import torch

    load = load or D1.load
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    data = read_rl_examples(cfg.data)
    if not data:
        raise ValueError("the RL update needs examples")

    ref = load(adapter=cfg.reference, revision=cfg.revision)
    rows = []  # (ids, forms, example)
    for e in data:
        ids, forms = ref.encode(e.state, e.question, e.render)
        if len(ids) <= cfg.max_tokens:
            rows.append((ids, forms, e))
    if not rows:
        raise ValueError("every RL example is longer than max_tokens")
    order = rng.sample(range(len(rows)), len(rows))
    lengths = [len(rows[i][0]) for i in order]
    encoded = [[order[j] for j in b] for b in batches(lengths, cfg.batch_tokens)]
    ref_lps = frozen_logprobs(ref, rows, encoded)
    del ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    d1 = load(adapter=cfg.adapter, lora={}, revision=cfg.revision)  # trainable
    if cfg.checkpointing and hasattr(d1.base, "gradient_checkpointing_enable"):
        d1.base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    start = frozen_logprobs(d1, rows, encoded)  # the same batches the steps read
    gaps = [
        abs(lp[e.chosen].exp().item() - e.old_prob)
        for lp, (_, _, e) in zip(start, rows, strict=True)
    ]
    if cfg.old == "recompute":
        old = [lp[e.chosen].item() for lp, (_, _, e) in zip(start, rows, strict=True)]
    else:
        old = [math.log(e.old_prob) for _, _, e in rows]

    params = [p for p in d1.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    d1.model.train()  # gradient checkpointing needs training mode; dropout stays off
    for module in d1.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    seen, sums = 0, {"loss": 0.0, "kl": 0.0, "ratio": 0.0, "clipped": 0}
    plan = split(encoded, cfg.steps)
    for step in plan:
        n = sum(map(len, step))
        for batch in step:
            lps = d1.option_logprobs([rows[i][0] for i in batch], [rows[i][1] for i in batch])
            losses = []
            for i, lp in zip(batch, lps, strict=True):
                e = rows[i][2]
                ref_lp = ref_lps[i].to(lp.device)
                pg, kl, ratio = rl_terms(lp, ref_lp, e.chosen, old[i], e.advantage, cfg.clip)
                losses.append(pg + cfg.beta * kl)
                sums["kl"] += kl.item()
                sums["ratio"] += ratio.item()
                sums["clipped"] += abs(ratio.item() - 1) > cfg.clip
            loss = torch.stack(losses).sum()
            (loss / n).backward()
            sums["loss"] += loss.item()
            seen += len(batch)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    end = frozen_logprobs(d1, rows, encoded)
    kls = [
        categorical_kl(a.exp().tolist(), b.exp().tolist())
        for a, b in zip(end, ref_lps, strict=True)
    ]
    ratios = [
        math.exp(lp[e.chosen].item() - o) for lp, (_, _, e), o in zip(end, rows, old, strict=True)
    ]
    kl = sum(kls) / len(kls)
    metrics = {
        "decisions": len(rows),
        "dropped": len(data) - len(rows),
        "steps": len(plan),
        "old": cfg.old,
        "readout_parity": max(gaps),  # max |pi_start(chosen) - logged pi_old(chosen)|
        "readout_parity_mean": sum(gaps) / len(gaps),
        "train": {k: v / seen for k, v in sums.items()},  # as each step's forward saw them
        "kl": kl,  # mean KL(pi || pi_ref) per decision after the update
        "kl_max": max(kls),
        "ratio": sum(ratios) / len(ratios),  # mean pi(chosen) / pi_old(chosen) after the update
        "clip_fraction": sum(abs(r - 1) > cfg.clip for r in ratios) / len(ratios),
        "reverted": kl > cfg.kl_max,
    }
    cfg.out.mkdir(parents=True, exist_ok=True)
    atomic_write(cfg.out / "config.json", cfg.model_dump_json(indent=2))
    if metrics["reverted"]:
        shutil.rmtree(cfg.out / "adapter", ignore_errors=True)  # never leave an earlier try's
        log(f"mean KL {kl:.4f} > {cfg.kl_max}: reverted, no adapter saved")
    else:
        d1.model.save_pretrained(str(cfg.out / "adapter"))
    atomic_write(cfg.out / "metrics.json", json.dumps(metrics, indent=2))
    log(json.dumps(metrics, indent=2))
    return metrics

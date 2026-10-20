"""Configuration: model routing (configs/models.yaml), the system catalog (configs/systems.yaml)
and experiments (configs/experiments/*.yaml).

All are plain YAML validated by pydantic (unknown keys are errors). An experiment file is resolved
into one self-contained `Experiment`, which the run manifest records, in this order:

1. `variants`: the file's base settings deep-merged with one named variant (`--variant`);
   mappings merge, lists and scalars replace.
2. `--set a.b.c=value` overrides (values parsed as YAML).
3. Systems: every name a benchmark lists is defined locally (`systems:`), in the catalog, or is a
   bare system class (`simple_rag`: default parameters). A definition may `extends` another
   (local or catalog), deep-merged the same way. A local definition must not reuse a catalog name.
4. Writer samples: a benchmark's `samples: N` runs each listed system N times, as `<name>`,
   `<name>_s1`, ...; sample i writes from the same evidence in its own replay namespace.

A system definition may also carry its own per-task cap, `task_usd` (DS-base runs at its natural
cost under a cap it never reaches); it is part of the cell, not a system parameter.

Experiment defaults are the settings every comparison shares: executor deepseek-flash, the local
arXiv index, 10 hits per query, off-peak dispatch, 4 workers, and each benchmark's own per-task
cap B0 (`Benchmark.cap_usd`) unless `budget.task_usd` sets one for every system.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from jevscholar.paths import CATALOG, CONFIGS, EXPERIMENTS
from jevscholar.types import SYSTEMS, Strict


# --------------------------------------------------------------------------- models.yaml
class Upstream(Strict):
    base_url: str
    key_env: str | None = None  # name of the secret in secrets.env
    dialect: Literal["deepseek", "openai", "llama"] = "openai"
    concurrency: int = 16
    timeout_s: float = 600
    offpeak_discount: float = 1.0  # billed multiplier outside peak hours (DeepSeek: 0.5)
    billed: bool = True  # False for local servers: list (proxy) price is accounted, $0 billed


class Price(Strict):  # USD per 1M tokens
    miss: float = 0.0
    hit: float = 0.0
    out: float = 0.0


class Alias(Strict):
    upstream: str
    model: str
    thinking: bool | None = None  # None: leave to the upstream default (passthrough aliases)
    effort: Literal["low", "high", "max"] | None = None
    max_tokens_cap: int = 32768
    default_max_tokens: int | None = None
    force_json_object: bool = False  # DRB2 judge: prompt asks for JSON, enforce json_object
    passthrough: bool = False  # forward the request unmodified (official OpenAI judges)
    cache: bool = False  # executor replay cache (never for judges)
    price: Price = Price()


class ModelsConfig(Strict):
    campaign_usd: float = 45.0
    upstreams: dict[str, Upstream]
    aliases: dict[str, Alias]
    tools: dict[str, float] = Field(default_factory=dict)  # USD per call (list price)
    free_tools: tuple[str, ...] = ()  # tools whose calls are really free: billed $0
    judgesets: dict[str, dict[str, str]] = Field(default_factory=dict)  # name -> {key: alias}


def load_models(path: Path = CONFIGS / "models.yaml") -> ModelsConfig:
    return ModelsConfig.model_validate(yaml.safe_load(path.read_text()))


# --------------------------------------------------------------------------- experiments
class Search(Strict):
    corpora: tuple[Literal["arxiv", "web"], ...] = ("arxiv",)
    k: int = 10  # hits per query
    # "local": BM25 over a frozen arXiv metadata snapshot (the live API rate-limits this host
    # after ~10 requests); "api": export.arxiv.org. Shared by every system of a run.
    arxiv: Literal["local", "api"] = "local"


class Budget(Strict):
    task_usd: float | None = None  # generation cap per task, list price (None: each B0)
    run_usd: float = 40.0  # generation cap for the whole run, billed
    judge_usd: dict[str, float] = Field(
        default_factory=lambda: {"deepscholar": 0.15, "drb2": 0.15, "researchqa": 0.01}
    )  # judging cap per task, by benchmark
    judge_run_usd: float = 10.0  # judging cap for the whole run, billed


class BenchCell(Strict):
    split: str = "dev"
    limit: int | None = None  # a stable prefix of the split
    ids: tuple[str, ...] | None = None  # these tasks of the split prefix only, in this order
    systems: tuple[str, ...]


class Experiment(Strict):
    name: str
    executor: str = "deepseek-flash"
    search: Search = Search()
    budget: Budget = Budget()
    offpeak_only: bool = True
    replay_only: bool = False  # every executor call must replay from the cache ($0 checks)
    workers: int = 4
    seed: int = 0
    systems: dict[str, dict[str, Any]] = Field(default_factory=dict)  # name -> params + "use"
    benchmarks: dict[str, BenchCell]

    def system_spec(self, name: str) -> tuple[str, dict[str, Any]]:
        """(registered system class key, params) of a named system (its own `task_usd`, a cap,
        is not a parameter: see `benchmarks.task_cap`)."""
        params = dict(self.systems.get(name, {}))
        params.pop("task_usd", None)
        return params.pop("use", name), params

    def cell_hash(self, bench: str, system: str, **extra: Any) -> str:
        """Identity of a (bench, system) cell; limit, workers and run budgets are excluded.

        `extra` carries identity the experiment file cannot see (the runner passes the benchmark
        version, the executor alias's routing and the task cap)."""
        cls, params = self.system_spec(system)
        cell = self.benchmarks[bench]
        key = {
            "executor": self.executor,
            "search": self.search.model_dump(),
            "seed": self.seed,
            "system": cls,
            "params": params,
            "bench": bench,
            "split": cell.split,
            **extra,
        }
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def experiment_path(name: str) -> Path:
    """An experiment file: a path, or a name under configs/experiments/."""
    path = Path(name)
    return path if path.is_file() else EXPERIMENTS / f"{name}.yaml"


def variants(path: Path) -> list[str]:
    return list((yaml.safe_load(path.read_text()) or {}).get("variants", {}))


def load_experiment(
    path: Path, variant: str | None = None, overrides: list[str] = (), catalog: Path = CATALOG
) -> Experiment:
    """The resolved experiment of a file, one of its variants and command-line overrides."""
    raw = yaml.safe_load(path.read_text()) or {}
    named = raw.pop("variants", {})
    if variant is not None:
        if variant not in named:
            raise ValueError(f"{path.name} has no variant {variant!r}; it has {sorted(named)}")
        raw = merge(raw, named[variant])
    elif named:
        raise ValueError(f"{path.name} needs a variant: one of {sorted(named)}")
    raw["name"] = f"{path.stem}/{variant}" if variant else path.stem
    raw = apply_overrides(raw, list(overrides))
    known = yaml.safe_load(catalog.read_text()) or {}
    return Experiment.model_validate(resolve_systems(raw, known))


def merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """`top` deep-merged over `base`: mappings merge, anything else replaces."""
    out = copy.deepcopy(base)
    for k, v in top.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides (value parsed as YAML) to a nested dict."""
    for item in overrides:
        dotted, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value: {item!r}")
        *parents, leaf = dotted.split(".")
        node = data
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return data


def resolve_systems(data: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """The experiment's `systems`: the resolved definition of every system its benchmarks run
    (bare classes need none), writer samples expanded into the benchmarks' system lists."""
    local = data.get("systems") or {}
    if clash := sorted(set(local) & set(catalog)):
        raise ValueError(f"systems {clash} redefine catalog systems: use a name of their own")

    def resolve(name: str, chain: tuple[str, ...] = ()) -> dict[str, Any]:
        if name in chain:
            raise ValueError(f"systems extend each other in a cycle: {' -> '.join(chain)}")
        spec = local.get(name, catalog.get(name))
        if spec is None:
            if name not in SYSTEMS:
                raise ValueError(f"unknown system {name!r}: not defined, not a system class")
            return {"use": name}
        spec = copy.deepcopy(spec)
        if parent := spec.pop("extends", None):
            spec = merge(resolve(parent, (*chain, name)), spec)
        if spec.get("use") not in SYSTEMS:
            raise ValueError(f"system {name!r} uses {spec.get('use')!r}, not a system class")
        cap = spec.get("task_usd")
        if cap is not None and (
            type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0
        ):
            raise ValueError(f"system {name!r}: task_usd is a positive number, not {cap!r}")
        return spec

    cells = (data.get("benchmarks") or {}).values()
    named = set(local) | set(catalog) | {n for cell in cells for n in cell.get("systems", ())}
    systems: dict[str, dict[str, Any]] = {}
    for cell in cells:
        samples, listed = cell.pop("samples", 1), []
        for name in cell.get("systems", ()):
            spec = resolve(name)
            for i in range(samples):
                sample, defined = (f"{name}_s{i}", spec | {"sample": i}) if i else (name, spec)
                if i and sample in named:
                    raise ValueError(f"sample {i} of {name!r} would reuse the name {sample!r}")
                if systems.setdefault(sample, defined) != defined:
                    raise ValueError(f"{sample!r} is defined twice, differently")
                listed.append(sample)
        if len(set(listed)) < len(listed):
            raise ValueError(f"a benchmark lists a system twice: {listed}")
        cell["systems"] = listed
    bare = {n for n, s in systems.items() if s == {"use": n}}  # a class at its defaults
    return data | {"systems": {n: s for n, s in systems.items() if n not in bare}}

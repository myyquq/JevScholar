"""ResearchQA rubric coverage of one answer, via the pinned upstream `compute_coverage.py`.

The upstream script (fetched by `jev data fetch`, not committed: it has no licence) is imported
unchanged and its own batching, prompt, retry and scoring functions are reused. Two runtime
patches make it usable: a `retry` module shim on tenacity (same semantics as the PyPI package),
and `call_gpt`, written for the pre-1.0 openai SDK, replaced per task by the same request on a
client tagged for that task. Upstream crashes on an unknown label and silently skips an item
whose batch keeps returning the wrong number of labels; both become explicit statuses here. Any
other failure (gateway refusal, timeout) raises, so the runner retries the task.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from functools import cache, partial
from pathlib import Path
from typing import Any

from openai import OpenAI

from jevscholar import data
from jevscholar.types import CLIENT_TIMEOUT_S, LOCAL_KEY, EvalContext, Task
from jevscholar.util import atomic_write, vendored

RAW = data.RAW / "researchqa"  # {valid,test}.json at the pinned HF revision
CODE = RAW / "code"  # compute_coverage.py + researchqa_loader.py at the pinned revision
BATCH_SIZE = 8  # upstream default


def retry_shim() -> types.ModuleType:
    """`retry.retry` on tenacity: retry only `exceptions`, at most `tries` attempts, sleeping
    `delay * backoff**i` before retry i, then re-raise the last exception unchanged."""
    import tenacity

    def retry(
        exceptions: Any = Exception, tries: int = -1, delay: float = 0, backoff: float = 1
    ) -> Any:
        return tenacity.retry(
            retry=tenacity.retry_if_exception_type(exceptions),
            stop=tenacity.stop_after_attempt(tries) if tries > 0 else tenacity.stop_never,
            wait=lambda state: delay * backoff ** (state.attempt_number - 1),
            reraise=True,
        )

    module = types.ModuleType("retry")
    module.retry = retry  # type: ignore[attr-defined]
    return module


@cache
def upstream() -> types.ModuleType:
    """Import the upstream module once per process (its import-time basicConfig undone)."""
    sys.modules["retry"] = retry_shim()
    vendored(str(CODE))
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    import compute_coverage

    root.setLevel(level)
    root.handlers[:] = handlers
    return compute_coverage


def _call_gpt(client: OpenAI, prompt: str, model: str = "gpt-4") -> str:
    """Upstream's request (one user message, temperature 0) on the task's client."""
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response.choices[0].message.content.strip()


@cache
def _items() -> dict[str, Any]:
    """id -> upstream ResearchQAItem over both splits (ids are disjoint)."""
    load = upstream().load_researchqa_data
    return {it.id: it for split in ("valid", "test") for it in load(str(RAW / f"{split}.json"))}


def evaluate(task: Task, export: Path, ctx: EvalContext) -> dict[str, Any]:
    """Coverage of `export/<id>.json` = upstream per-item score, plus one score per rubric type."""
    cc, item = upstream(), _items()[task.id]
    answer = json.loads((export / f"{task.id}.json").read_text(encoding="utf-8"))["answer"]
    if not answer.strip():
        return {"metrics": {"coverage": None}, "status": {"coverage": "empty_export"}}
    model = ctx.judge("coverage")
    client = OpenAI(
        base_url=ctx.base_url(task, "coverage"),
        api_key=LOCAL_KEY,
        max_retries=0,
        timeout=CLIENT_TIMEOUT_S,
    )
    cc.call_gpt = partial(_call_gpt, client)  # one task at a time per worker
    rubric = [r.rubric_item for r in item.rubric]
    judges: list[dict[str, str]] = []
    try:
        for i in range(0, len(rubric), BATCH_SIZE):
            batch = rubric[i : i + BATCH_SIZE]
            prompt = cc.build_prompt(answer, batch)
            scores = cc.process_batch_with_retry(prompt, batch, model, task.id)
            judges += [{"rubric": q, "score": s} for q, s in zip(batch, scores, strict=True)]
    except cc.BatchProcessingError as e:  # wrong label count on every try: upstream skips it
        ctx.log.warning("researchqa %s: judge failure (%s)", task.id, e)
        return {"metrics": {"coverage": None}, "status": {"coverage": "judge_failure"}}
    raw = json.dumps(judges, ensure_ascii=False, indent=1) + "\n"
    atomic_write(ctx.out_dir / "raw" / f"{task.id}.json", raw)
    try:
        metrics = {"coverage": cc.compute_coverage_score(judges)}
        for t in sorted({t for r in item.rubric for t in r.type}):
            of_type = [j for j, r in zip(judges, item.rubric, strict=True) if t in r.type]
            metrics[f"coverage_{t.lower()}"] = cc.compute_coverage_score(of_type)
    except KeyError:  # a label outside the five; upstream crashes the whole run
        return {"metrics": {"coverage": None}, "status": {"coverage": "invalid_label"}}
    return {"metrics": metrics, "status": dict.fromkeys(metrics, "ok")}

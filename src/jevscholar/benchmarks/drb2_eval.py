"""DeepResearch Bench II rubric scoring of one report with the vendored evaluator (b38f360).

`run_evaluation.py` is imported unchanged and driven per task through its own
`process_one_with_chunking` (official chunk size 50, 150k characters, 5 retries per batch).
Runtime changes: PR #8's parser fix (`json.loads` first; the legacy key-quote cleanup corrupts
compact JSON, which a forced `json_object` judge emits), and the module-global client set per task
(one task at a time per worker) so every judge call carries the task's gateway tag. Upstream
swallows request errors; a batch that still fails after its retries raises here, so the runner
retries the task. The raw result is kept in upstream's JSONL line shape, so `aggregate_scores.py`
can be run unchanged over a cell's concatenated raw/*.json.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jevscholar import data
from jevscholar.types import CLIENT_TIMEOUT_S, LOCAL_KEY, EvalContext, Task
from jevscholar.util import atomic_write, vendored

CHUNK_SIZE, MAX_PAPER_CHARS, MAX_RETRIES = 50, 150_000, 5
METRICS = {
    "info_recall": "inforecall",
    "analysis": "analysis",
    "presentation": "presentation",
    "total": "total",
    "blocked_rate": "blocked_rate",
}  # ours -> aggregate_scores.py's


@cache
def upstream() -> tuple[Any, Any]:
    """(run_evaluation, aggregate_scores), imported once per process with PR #8's parser."""
    vendored("drb2")
    import aggregate_scores
    import run_evaluation

    legacy = run_evaluation._try_clean_and_load

    def pr8_load(s: str) -> Any:
        text = s.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return legacy(text)

    run_evaluation._try_clean_and_load = pr8_load
    return run_evaluation, aggregate_scores


@cache
def _tasks() -> dict[int, dict[str, Any]]:
    return upstream()[0].load_tasks_data(str(data.DRB2_TASKS))[0]


def evaluate(task: Task, export: Path, ctx: EvalContext) -> dict[str, Any]:
    """Score `export/idx-<n>.md` (or the .docx reference reports); metrics exactly as
    `aggregate_scores.py` computes them per line."""
    run_evaluation, aggregate_scores = upstream()
    idx = int(task.id)
    (report,) = export.glob(f"idx-{idx}.*")
    if not report.read_bytes().strip():
        return {"metrics": dict.fromkeys(METRICS), "status": dict.fromkeys(METRICS, "empty_export")}
    client = run_evaluation.GPTClient(
        api_url=ctx.base_url(task, "rubric") + "/chat/completions",
        api_key=LOCAL_KEY,
        model=ctx.judge("rubric"),
        reasoning_effort="medium",
        verbose=False,
    )
    client.max_output_tokens, client.timeout = 32768, CLIENT_TIMEOUT_S  # env-only upstream
    run_evaluation.client = client
    _, result, _ = run_evaluation.process_one_with_chunking(
        idx, str(report), _tasks()[idx], CHUNK_SIZE, MAX_PAPER_CHARS, MAX_RETRIES
    )
    if "error" in result:
        raise RuntimeError(f"drb2 idx-{idx}: {result['error']}")
    line = {"model": ctx.system, "idx": idx, "result": result}
    atomic_write(
        ctx.out_dir / "raw" / f"idx-{idx}.json", json.dumps(line, ensure_ascii=False) + "\n"
    )
    dims = aggregate_scores.compute_dimension_averages(result)
    return {
        "metrics": {ours: dims[theirs] for ours, theirs in METRICS.items()},
        "status": dict.fromkeys(METRICS, "ok"),
    }

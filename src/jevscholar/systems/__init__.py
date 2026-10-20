"""Shared plumbing for every system: one chat call path with the shared sampling settings.

Every system ends with `render.finalize`, which owns the one citation rule (renumbering, uncited
sources dropped, model-written reference lists removed, one source identity).
"""

from __future__ import annotations

from jevscholar.types import TaskContext

TEMPERATURE = 1.0  # shared sampling for every system (the DeepScholar-base repo YAML)
MAX_TOKENS = 10000


async def chat(ctx: TaskContext, stage: str, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    """One executor call on the stage's tagged gateway URL; returns the message text. A short
    `max_tokens` bounds the cost of calls whose answer is short by design."""
    async with ctx.llm(stage) as client:
        resp = await client.chat.completions.create(
            model=ctx.executor,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
        )
    choice = resp.choices[0]
    if not choice.message.content:
        raise RuntimeError(f"{stage}: empty completion (finish_reason={choice.finish_reason})")
    return choice.message.content

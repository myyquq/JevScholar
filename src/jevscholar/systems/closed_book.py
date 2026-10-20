"""Closed-book control: one executor call, no retrieval.

It measures what the backbone knows parametrically (and any contamination), so retrieval systems
can be read against it; ResearchQA also publishes parametric-only rows to compare with.
"""

from __future__ import annotations

from jevscholar.render import finalize
from jevscholar.systems import chat
from jevscholar.types import Report, System, TaskContext

NO_SOURCES = (
    "No sources are available for this task: answer from your own knowledge only, "
    "without citation markers or a reference list."
)


class ClosedBook(System):
    async def run(self, ctx: TaskContext) -> Report:
        text = await chat(ctx, "write", f"{ctx.task.prompt}\n\n{NO_SOURCES}")
        return finalize(text, (), status="ok")

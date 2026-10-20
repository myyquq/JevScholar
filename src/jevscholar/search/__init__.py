"""Search layer: one cached, policy-enforced, metered tool interface shared by every system.

`policy` holds the leakage rules, `store` the frozen cache of raw upstream responses, `arxiv` and
`arxiv_local` the paper backends (a local BM25 snapshot, the default, or the live Atom API, chosen
by `JEV_ARXIV_BACKEND=local|api`, which the runner sets from the experiment's `search.arxiv`),
`tavily` the web backend, `openalex` the citation counts behind DeepScholar's Document Importance,
and `client.SearchClient` binds them to one task. The exceptions live here, their only import
path, so backends and client share them without cycles.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType


class ToolError(Exception):
    """A tool call failed; the task may continue without its results."""


class FatalToolError(Exception):
    """A tool can never succeed in this run (bad key, exhausted plan): the run must stop."""


class BudgetExhausted(Exception):
    """The gateway refused a call (HTTP 402): a budget latch is closed or the breaker tripped."""


def arxiv_backend() -> ModuleType:
    """The low-level arXiv module (`search`, `lookup`) selected by JEV_ARXIV_BACKEND (default
    local, as config.Search)."""
    name = {"api": "arxiv", "local": "arxiv_local"}[os.environ.get("JEV_ARXIV_BACKEND", "local")]
    return importlib.import_module(f"jevscholar.search.{name}")

"""The registered task utility U and the judge-free retrieval return (docs/04: Scoring).

    U = N + 0.15 R_c + 0.10 P + 0.10 S - 0.10 C / B0
    r = 0.15 R_e - 0.10 C / B0

N is nugget coverage, R_c reference recall of the cited sources, P cite_p, S claim support, C the
task's list cost and B0 its benchmark's per-task cap. R_e is the share of the task's important
citations among the documents the writer gets. A task without a report scores U = -1 - 0.10 C/B0;
a retrieval step's reward is the change of r across it. Both work elementwise on floats and on
pandas Series alike.
"""

from __future__ import annotations

from typing import Any

UTILITY = {"nugget_coverage": 1.0, "reference_recall": 0.15, "cite_p": 0.10, "claim_support": 0.10}
RECALL_WEIGHT = 0.15  # R_e in the retrieval return
COST_WEIGHT = 0.10  # spending all of B0 costs 0.10
FAILED = -1.0  # U of a task without a report, before its cost


def cost_term(cost_usd: Any, b0: Any) -> Any:
    return COST_WEIGHT * cost_usd / b0


def utility(scores: Any, cost_usd: Any, b0: Any) -> Any:
    """U from the component scores (a mapping or frame with UTILITY's keys) of a task that has a
    report."""
    return sum(w * scores[m] for m, w in UTILITY.items()) - cost_term(cost_usd, b0)


def failed_utility(cost_usd: Any, b0: Any) -> Any:
    return FAILED - cost_term(cost_usd, b0)


def retrieval_return(recall: Any, cost_usd: Any, b0: Any) -> Any:
    """0.15 R_e - 0.10 C/B0: the judge-free value of the evidence bought so far."""
    return RECALL_WEIGHT * recall - cost_term(cost_usd, b0)

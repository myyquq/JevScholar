"""Judge agreement: item matching, paired-difference signs and the decision rule."""

from __future__ import annotations

import json
import math

import pytest

from jevscholar import types
from jevscholar.checks import agreement
from jevscholar.checks.review import cohen_kappa, per_claim
from jevscholar.types import Benchmark, Task


def _dec(metric, decision, context):
    return {"metric": metric, "decision": decision, "context": context}


def test_match_keys_repeated_contexts_by_occurrence():
    ref = [_dec("cite_p", 1, "c"), _dec("cite_p", 0, "c"), _dec("organization", 1, "x")]
    new = [_dec("cite_p", 1, "c"), _dec("cite_p", 1, "c"), _dec("organization", 0, "y")]
    pairs, unmatched = agreement._match(ref, new)
    assert pairs == [("cite_p", "0", "1"), ("cite_p", "1", "1")]
    assert unmatched == 2  # organization/x and organization/y


def test_claim_coverage_is_one_decision_per_claim():
    """Upstream re-checks an unsupported claim up to 3 times and stops at the first support."""
    ref = per_claim([_dec("claim_coverage", 0, "a")] * 3 + [_dec("claim_coverage", 0, "b")] * 3)
    new = per_claim(
        [_dec("claim_coverage", 0, "a")] * 3
        + [_dec("claim_coverage", 0, "b"), _dec("claim_coverage", 1, "b")]
        + [_dec("cite_p", 1, "c")] * 2  # cite_p scores every occurrence: kept
    )
    assert len(new) == 4
    pairs, unmatched = agreement._match(ref, new)
    assert pairs == [("claim_coverage", "0", "0"), ("claim_coverage", "0", "1")]
    assert unmatched == 2  # the two cite_p decisions


def test_weighted_kappa_penalises_near_misses_less():
    a = ["-1", "0", "1", "-1", "0", "1"]
    near = ["-1", "0", "1", "0", "1", "0"]  # every miss is one step off
    scale = ["-1", "0", "1"]
    assert cohen_kappa(a, near, scale) > cohen_kappa(a, near)
    assert math.isclose(cohen_kappa(a, a, scale), 1.0)


GOOD = {"info_recall": [("1", "1"), ("0", "0")] * 6}


def _compare(new_diff_sign: float, items=GOOD, n=6, failed=None):
    scores = {
        ("simple_rag", "total"): {str(i): (0.3, 0.3) for i in range(n)},
        ("closed_book", "total"): {str(i): (0.4, 0.3 + new_diff_sign * 0.1) for i in range(n)},
    }
    return agreement._compare("drb2", ("simple_rag", "closed_book"), items, scores, failed or {})


def test_compare_flags_primary_sign_flip_and_low_kappa():
    ok = _compare(+1)
    assert ok["verdict"] == "consistent"
    assert ok["paired"]["closed_book/total"]["task_flips"] == 0.0
    assert _compare(0)["verdict"] == "consistent"  # a tie under one judge is no reversal
    flipped = _compare(-1)
    assert flipped["sign_flips"] == ["closed_book/total"] and flipped["verdict"] == "flagged"
    noisy = {"info_recall": [("1", "0"), ("0", "1")] * 6, "analysis": [("1", "0")] * 3}
    low = _compare(+1, noisy)
    assert low["low_kappa"] == ["info_recall"]  # analysis has too few items to judge


@pytest.mark.parametrize("kw", [{"n": 3}, {"items": {}}])
def test_too_little_evidence_is_insufficient_not_consistent(kw):
    assert _compare(+1, **kw)["verdict"] == "insufficient"


def test_paired_differences_score_failed_reports_zero():
    """As the published deltas do: a failed report is 0 under both judges, not dropped."""
    out = agreement._compare(
        "drb2",
        ("simple_rag", "closed_book"),
        GOOD,
        {
            ("simple_rag", "total"): {"1": (0.3, 0.3), "2": (0.3, 0.3)},
            ("closed_book", "total"): {"1": (0.5, 0.6)},
        },
        {"closed_book": {"2"}},
    )
    row = out["paired"]["closed_book/total"]
    assert row["n"] == 2
    assert math.isclose(row["diff_reference"], (0.2 - 0.3) / 2)
    assert math.isclose(row["diff_new"], (0.3 - 0.3) / 2)


def test_compare_drops_labels_off_the_scale():
    rows = agreement._compare(
        "researchqa",
        ("simple_rag",),
        {"coverage": [("mostly", "mostly"), ("mostly", "??")]},
        {},
        {},
    )["items"]["coverage"]
    assert rows["n"] == 1 and rows["invalid"] == 1


def test_agreement_run_cannot_be_its_source(tmp_path):
    with pytest.raises(ValueError, match="own source"):
        agreement.judge_subset(tmp_path, tmp_path)


class Five(Benchmark):
    name, primary = "five", "score"

    def tasks(self, split: str, limit: int | None = None) -> list[Task]:
        return [Task("five", f"t{i}", "p") for i in range(5)][:limit]


def test_agreement_checks_the_source_runs_own_tasks(tmp_path, monkeypatch):
    """Regression: a source run that selects tasks by id was re-judged on the split's prefix."""
    monkeypatch.setitem(types.BENCHMARKS, "five", f"{__name__}:Five")
    experiment = {
        "name": "src",
        "budget": {"task_usd": 1.0},
        "benchmarks": {"five": {"split": "test", "ids": ["t3", "t4"], "systems": ["a", "b"]}},
        "systems": {"a": {"use": "simple_rag"}, "b": {"use": "closed_book"}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"experiment": experiment}))
    cell = agreement._experiment(tmp_path, ("b",), {"five": 1}, 0).benchmarks["five"]
    assert (cell.ids, cell.systems) == (("t3",), ("b",))

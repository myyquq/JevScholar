"""Ledger arithmetic: DeepSeek peak hours, token splits and list vs billed cost."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from jevscholar.config import Price, Upstream
from jevscholar.ledger import is_peak, llm_costs, usage_tokens


@pytest.mark.parametrize(
    ("when", "peak"),
    [
        (datetime(2026, 10, 19, 0, 59, tzinfo=UTC), False),  # Monday
        (datetime(2026, 10, 19, 1, 0, tzinfo=UTC), True),
        (datetime(2026, 10, 19, 3, 59, tzinfo=UTC), True),
        (datetime(2026, 10, 19, 4, 0, tzinfo=UTC), False),
        (datetime(2026, 10, 19, 6, 0, tzinfo=UTC), True),
        (datetime(2026, 10, 23, 9, 59, tzinfo=UTC), True),  # Friday
        (datetime(2026, 10, 19, 10, 0, tzinfo=UTC), False),
        (datetime(2026, 10, 17, 2, 0, tzinfo=UTC), False),  # Saturday
        (datetime(2026, 10, 19, 10, 30, tzinfo=timezone(timedelta(hours=8))), True),  # 02:30 UTC
    ],
)
def test_is_peak(when: datetime, peak: bool) -> None:
    assert is_peak(when) is peak


def test_usage_tokens_and_cost_math() -> None:
    deepseek = {
        "prompt_tokens": 1000,
        "completion_tokens": 300,
        "prompt_cache_hit_tokens": 600,
        "prompt_cache_miss_tokens": 400,
        "prompt_tokens_details": {"cached_tokens": 600},
        "completion_tokens_details": {"reasoning_tokens": 120},
    }
    openai = {
        "prompt_tokens": 1000,
        "completion_tokens": 300,
        "prompt_tokens_details": {"cached_tokens": 250},
    }
    tok = usage_tokens(deepseek)
    assert tok == {"miss": 400, "hit": 600, "out": 300, "reasoning": 120}
    assert usage_tokens(openai) == {"miss": 750, "hit": 250, "out": 300, "reasoning": 0}
    assert usage_tokens(None) == {"miss": 0, "hit": 0, "out": 0, "reasoning": 0}
    price = Price(miss=0.30, hit=0.006, out=1.20)
    usd = (400 * 0.30 + 600 * 0.006 + 300 * 1.20) / 1e6
    peak, offpeak = datetime(2026, 10, 19, 2, tzinfo=UTC), datetime(2026, 10, 19, 12, tzinfo=UTC)
    ds = Upstream(base_url="http://x", offpeak_discount=0.5)
    local = Upstream(base_url="http://x", billed=False)
    assert llm_costs(price, ds, tok, peak) == pytest.approx((usd, usd))
    assert llm_costs(price, ds, tok, offpeak) == pytest.approx((usd, usd / 2))
    assert llm_costs(price, local, tok, peak) == pytest.approx((usd, 0.0))  # proxy price only

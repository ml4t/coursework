"""`objective` - unit 1.3. What the strategy is judged on, fixed before any result exists."""

from __future__ import annotations

from ..contracts import Contract, register
from ._config import interface_for, one_of, probe, stated

METRICS = ("sharpe", "information_ratio", "annual_return", "calmar")
COST_TIERS = ("none", "spread-only", "spread-and-impact")
FIELDS = {
    "metric": f"the single number the strategy is judged on: one of {', '.join(METRICS)}",
    "cost_tier": f"which frictions apply: one of {', '.join(COST_TIERS)}",
    "rationale": "why that metric answers the question you are asking of this strategy",
}


def reference():
    return {
        "metric": "sharpe",
        "cost_tier": "spread-and-impact",
        "rationale": ("The claim is that the signal earns a return per unit of risk taken, and "
                      "the book is rebalanced often enough that ignoring impact would flatter it."),
    }


register(Contract(
    name="objective",
    kind="config",
    units=("1.3",),
    summary="The metric the strategy is judged on and the cost tier it is judged under.",
    probe=probe,
    interface=interface_for(FIELDS),
    interface_detail=", ".join(FIELDS),
    reference=reference,
    leakage=None,
    leakage_note=("not applicable: the protection here is timing, not information. Fixing the "
                  "metric before the first backtest is what stops it being chosen to fit a result"),
    invariants=(("metric is one the course recognizes", one_of("metric", METRICS)),
                ("cost tier is one the course recognizes", one_of("cost_tier", COST_TIERS)),
                ("rationale is actually stated", stated("rationale", 60))),
))

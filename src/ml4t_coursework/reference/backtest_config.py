"""`backtest_config` - unit 7.2. The assumptions the engine makes, said out loud."""

from __future__ import annotations

from ..contracts import Contract, register
from ._config import interface_for, one_of, probe, stated

ENGINES = ("vectorized", "event-driven")
FILLS = ("close", "next-open", "vwap")
FIELDS = {
    "engine": f"one of {', '.join(ENGINES)}",
    "fills": f"the price an order is assumed to get: one of {', '.join(FILLS)}",
    "rebalance": "how often the book is rebalanced, in sessions",
    "unanswerable": "one question this engine cannot answer, stated plainly",
}


def reference():
    return {
        "engine": "vectorized",
        "fills": "next-open",
        "rebalance": 21,
        "unanswerable": ("Whether an order would have been filled at all on a day the fund barely "
                         "traded, because the engine applies a price to a quantity without ever "
                         "asking who was on the other side of it."),
    }


def _rebalance_is_a_period(obj) -> str:
    from ..checks import require

    value = obj.get("rebalance")
    require(isinstance(value, int) and value >= 1, "rebalance is a number of sessions",
            "a whole number of sessions, one or more", repr(value),
            "Rebalance frequency is a decision with a cost attached, so it is a number rather "
            "than a description.")
    return f"rebalance every {value} sessions"


register(Contract(
    name="backtest_config",
    kind="config",
    units=("7.2",),
    summary="Engine, fill assumption and rebalance frequency, with what the engine cannot answer.",
    probe=probe,
    interface=interface_for(FIELDS),
    interface_detail=", ".join(FIELDS),
    reference=reference,
    leakage=None,
    leakage_note=("not applicable: this records the assumptions. Whether they leak is decided by "
                  "the fill you choose, and a fill at the decision bar's own close is the one to "
                  "avoid"),
    invariants=(("engine is one the course recognizes", one_of("engine", ENGINES)),
                ("fills are one the course recognizes", one_of("fills", FILLS)),
                ("rebalance is a number of sessions", _rebalance_is_a_period),
                ("what it cannot answer is stated", stated("unanswerable", 60))),
))

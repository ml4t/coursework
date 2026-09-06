"""`cost_model` - unit 8.3. What trading takes out, before you decide anything survived.

The two decisions in 8.3 - the form, and where its parameters come from - are taught rather than
built, so the shipped form is the one every student uses and what they choose is its parameters.
The contract is on the shape any cost model must have, so a student who changes the form still
gets checked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..checks import require
from ..contracts import Contract, register
from ._shared import score_frame
from .allocator import reference as allocator_reference
from .signal import reference as signal_reference

SPREAD_BPS = 3.0
IMPACT_BPS = 8.0


def reference(spread_bps: float = SPREAD_BPS, impact_bps: float = IMPACT_BPS):
    """Half the quoted spread on everything traded, plus an impact term that grows faster than
    the size does. Both parameters are in basis points of the notional traded, and where they come
    from is the harder half of the unit."""

    def cost_model(trades: pd.DataFrame, spread_bps: float = spread_bps,
                   impact_bps: float = impact_bps) -> pd.Series:
        traded = trades.abs().fillna(0.0)
        spread = traded.sum(axis=1) * (spread_bps / 2) / 10_000
        impact = (traded ** 1.5).sum(axis=1) * impact_bps / 10_000
        return (spread + impact).rename("cost")

    return cost_model


def _trades() -> pd.DataFrame:
    weights = allocator_reference()(signal_reference()(score_frame()))
    return weights.diff().fillna(weights.iloc[0].to_frame().T.reindex(weights.index).fillna(0.0))


def _probe(obj):
    return obj(_trades())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a frame of trades",
            f"a {type(obj).__name__}")
    out = _probe(obj)
    trades = _trades()
    require(isinstance(out, pd.Series), "interface", "one cost per date, as a Series",
            f"a {type(out).__name__}")
    require(out.index.equals(trades.index), "interface", "a cost for every date traded",
            f"{len(out)} costs against {len(trades)} dates")


def _non_negative(obj) -> str:
    out = _probe(obj)
    worst = float(out.min())
    require(worst >= 0, "costs are never negative",
            "a cost of zero or more on every date",
            f"a cost of {worst:.6g}",
            "A negative cost is a rebate, and a strategy that earns money by trading more is an "
            "artefact of the cost model rather than a finding.")
    return f"lowest cost {worst:.6g}"


def _zero_at_zero(obj) -> str:
    trades = _trades()
    idle = pd.DataFrame(0.0, index=trades.index, columns=trades.columns)
    out = obj(idle)
    worst = float(np.abs(out).max())
    require(worst < 1e-12, "no trade costs nothing",
            "a cost of exactly zero on a date with no trade", f"{worst:.6g}",
            "A standing charge on an untraded day is a fee, not a trading cost, and it will make "
            "a low-turnover strategy look worse than it is.")
    return "a date with no trade costs exactly zero"


def _monotone(obj) -> str:
    trades = _trades()
    previous = -np.inf
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        total = float(obj(trades * scale).sum())
        require(total >= previous - 1e-12, "cost grows with size",
                "a cost that does not fall as the same trades get larger",
                f"{total:.6g} at {scale:g}x against {previous:.6g} at the size below it",
                "Cost that falls with size means the model rewards trading more, which inverts "
                "every turnover comparison the course asks you to make.")
        previous = total
    return "cost is non-decreasing as the same trades scale up"


def _finite(obj) -> str:
    out = _probe(obj)
    bad = int(out.isna().sum())
    require(bad == 0, "no empty costs", "a cost for every date", f"{bad} empty")
    return f"{len(out)} dates costed"


register(Contract(
    name="cost_model",
    kind="callable",
    units=("7.3",),
    summary="Charges the strategy for what it traded, in the units the return is measured in.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable trades(date x symbol) -> cost per date",
    reference=reference,
    leakage=None,
    leakage_note=("not applicable: a cost is charged on the trade that has just been decided, and "
                  "the reference uses no information beyond it"),
    invariants=(("costs are never negative", _non_negative),
                ("no trade costs nothing", _zero_at_zero),
                ("cost grows with size", _monotone),
                ("no empty costs", _finite)),
))

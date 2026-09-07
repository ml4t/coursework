"""`allocator`. From a position to a weight.

The first unit maps the signal to weights. The second tightens the same component with the
constraint set, which is
where the cap and the gross budget stop being implicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..checks import require, same
from ..contracts import Contract, register
from ._shared import score_frame
from .signal import reference as signal_reference

GROSS = 1.0
MAX_WEIGHT = 0.25


def reference(gross: float = GROSS, max_weight: float = MAX_WEIGHT):
    """Equal weight inside each leg, scaled to the gross budget, capped per name. The course does
    not estimate a covariance for this: with a weak signal over a hundred funds, an estimated
    covariance mostly reallocates estimation error."""

    def allocator(signal: pd.DataFrame, gross: float = gross,
                  max_weight: float = max_weight) -> pd.DataFrame:
        raw = signal.fillna(0.0)
        scale = raw.abs().sum(axis=1).replace(0, np.nan)
        weights = raw.div(scale, axis=0).mul(gross).fillna(0.0)
        # Clip, then give what the cap took away to the names still under it, and repeat. Clipping
        # once and rescaling puts a name straight back over the cap on any date with few enough
        # holdings, which is exactly the date the cap exists for.
        for _ in range(20):
            clipped = weights.clip(-max_weight, max_weight)
            shortfall = gross - clipped.abs().sum(axis=1)
            room = (max_weight - clipped.abs()).where(clipped.abs() > 0, 0.0)
            headroom = room.sum(axis=1).replace(0, np.nan)
            if float(shortfall.abs().max()) < 1e-12 or headroom.isna().all():
                weights = clipped
                break
            share = room.div(headroom, axis=0).fillna(0.0).mul(shortfall, axis=0)
            weights = clipped + share * np.sign(clipped)
        return weights.clip(-max_weight, max_weight).fillna(0.0)

    return allocator


def _signal_frame() -> pd.DataFrame:
    return signal_reference()(score_frame())


def _probe(obj):
    return obj(_signal_frame())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a frame of positions",
            f"a {type(obj).__name__}")
    out = _probe(obj)
    signal = _signal_frame()
    require(isinstance(out, pd.DataFrame), "interface", "a DataFrame of weights",
            f"a {type(out).__name__}")
    require(out.index.equals(signal.index), "interface", "one row of weights per date",
            f"{len(out)} rows against {len(signal)}")
    require(list(out.columns) == list(signal.columns), "interface", "the same assets",
            "different columns")


def _leakage(obj) -> str:
    signal = _signal_frame()
    cut = signal.index[80]
    full = obj(signal)
    truncated = obj(signal.loc[:cut])
    require(same(full.loc[:cut], truncated), "leakage probe",
            "a date's weights to depend only on that date's positions",
            "weights that change when later dates arrive",
            "Normalizing a weight against the whole period's exposure sizes today's book with "
            "next year's.")
    return "a date's weights depend only on that date's positions"


def _gross_budget(obj) -> str:
    out = _probe(obj)
    signal = _signal_frame()
    gross = out.abs().sum(axis=1)
    active = gross[gross > 0]
    require(len(active) > 0, "gross exposure is budgeted", "some date with a position",
            "a book that is empty on every date")
    over = float((active - GROSS).max())
    require(over < 1e-6, "gross exposure is budgeted",
            f"no date holding more than the gross budget of {GROSS:g}",
            f"a date summing to {float(active.max()):.4f}",
            "An unbudgeted book makes every return figure a statement about leverage rather than "
            "about the signal.")
    # The budget can be short of its target and still be right: with a cap of MAX_WEIGHT it takes
    # 1/MAX_WEIGHT names to reach it, and a date with fewer eligible names cannot get there.
    roomy = signal.abs().sum(axis=1) >= int(np.ceil(GROSS / MAX_WEIGHT))
    reachable = gross[roomy & (gross > 0)]
    require(len(reachable) > 0, "gross exposure is budgeted",
            "some date with enough names to reach the budget", "no such date in the sample")
    worst = float((GROSS - reachable).max())
    require(worst < 1e-6, "gross exposure is budgeted",
            f"a date with enough names to reach {GROSS:g} to actually reach it",
            f"a date reaching only {float(reachable.min()):.4f}",
            "Leaving the budget unspent when there is room for it is a different strategy from "
            "the one the signal described.")
    return f"gross of {GROSS:g} on all {len(reachable)} dates with room for it"


def _capped(obj) -> str:
    out = _probe(obj)
    worst = float(out.abs().to_numpy().max())
    require(worst <= MAX_WEIGHT + 1e-9, "no name exceeds its cap",
            f"no single weight above {MAX_WEIGHT:g}", f"a weight of {worst:.4f}",
            "The cap is what stops the whole result resting on one fund.")
    return f"largest weight {worst:.4f}, cap {MAX_WEIGHT:g}"


def _finite(obj) -> str:
    out = _probe(obj)
    bad = int(out.isna().to_numpy().sum())
    require(bad == 0, "no empty weights", "a weight for every name on every date",
            f"{bad} empty cells", "An empty weight is not a flat position; a backtest will read "
            "it as one or refuse it, and you will not be told which.")
    return f"{out.shape[0]} dates by {out.shape[1]} names, none empty"


register(Contract(
    name="allocator",
    kind="callable",
    summary="Turns positions into weights inside a stated gross budget and per-name cap.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable positions(date x asset) -> weights(date x asset)",
    reference=reference,
    leakage=_leakage,
    leakage_note="a date's weights depend only on that date's positions",
    invariants=(("gross exposure is budgeted", _gross_budget),
                ("no name exceeds its cap", _capped),
                ("no empty weights", _finite)),
))

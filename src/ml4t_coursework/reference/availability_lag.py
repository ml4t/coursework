"""`availability_lag` - unit 2.2. The lag before a datum may be used.

This is the decision the rest of the pipeline silently depends on, and its leakage probe is the
one check `certification.md` calls the most valuable in the course.
"""

from __future__ import annotations

import pandas as pd

from .. import fixtures
from ..checks import require, same
from ..contracts import Contract, register

LAG = 1


def reference(lag: int = LAG):
    """Shift every observation forward by the sessions it takes to become available.

    The shift is taken within each asset. On a long panel the row above is a different asset on
    the same date, not the same asset on the session before, so shifting the frame as a whole
    would hand each asset its neighbour's price.
    """

    def availability_lag(panel: pd.DataFrame, lag: int = lag) -> pd.DataFrame:
        return panel.groupby(level="asset", sort=False).shift(lag)

    return availability_lag


def _probe(obj):
    return obj(fixtures.panel())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a panel", f"a {type(obj).__name__}")
    out = _probe(obj)
    require(isinstance(out, pd.DataFrame), "interface", "a DataFrame", f"a {type(out).__name__}")
    panel = fixtures.panel()
    require(out.index.equals(panel.index), "interface", "the same (date, asset) index it was given",
            "a different index", "The lag moves values, not rows.")
    require(list(out.columns) == list(panel.columns), "interface", "the same columns",
            "different columns")


def _leakage(obj) -> str:
    panel = fixtures.panel()
    dates = panel.index.get_level_values("date")
    cut = dates.unique()[200]
    full = obj(panel)
    truncated = obj(panel[dates <= cut])
    a = full[dates <= cut]
    require(same(a, truncated), "leakage probe",
            "the same output for every date up to the cut whether or not later data exists",
            "output that changes when data after the cut is added",
            "That is what look-ahead is: a value at date t that could not have been computed on "
            "date t, because it depended on something that had not happened yet.")
    return "output up to a cut date is unchanged by data after it"


def _actually_lags(obj) -> str:
    panel = fixtures.panel()
    out = obj(panel)
    aligned = int((out.round(10) == panel.round(10)).to_numpy().sum())
    require(aligned == 0, "the lag is applied",
            "no row carrying the value observed on its own date",
            f"{aligned} values still carrying same-day observations",
            "A lag of zero is a decision to treat data as available the instant it is stamped, "
            "which no vendor delivers.")
    return "no row carries a same-day value"


def _leading_gap(obj) -> str:
    panel = fixtures.panel()
    out = obj(panel)
    first = out.groupby(level="asset", sort=False).head(1)
    filled = first.notna().to_numpy().sum()
    require(filled == 0, "leading rows empty",
            "each asset's first row empty, because nothing was available for it yet",
            f"{int(filled)} values on a first row",
            "An asset that lists late starts its own gap, on its own date. Taking the gap from "
            "the panel's first session instead gives every late lister a free head start.")
    return f"each of the {first.index.get_level_values('asset').nunique()} assets starts empty"


register(Contract(
    name="availability_lag",
    kind="callable",
    units=("2.2",),
    summary="Delays every observation by the time it takes to become usable.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable panel -> panel with the same (date, asset) index and columns",
    reference=reference,
    leakage=_leakage,
    leakage_note="output up to a cut date is unchanged by data after it",
    invariants=(("the lag is applied", _actually_lags), ("leading rows empty", _leading_gap)),
))

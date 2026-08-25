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
    """Shift every observation forward by the sessions it takes to become available."""

    def availability_lag(frame: pd.DataFrame, lag: int = lag) -> pd.DataFrame:
        return frame.shift(lag)

    return availability_lag


def _probe(obj):
    return obj(fixtures.prices())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a frame", f"a {type(obj).__name__}")
    out = _probe(obj)
    require(isinstance(out, pd.DataFrame), "interface", "a DataFrame", f"a {type(out).__name__}")
    frame = fixtures.prices()
    require(out.index.equals(frame.index), "interface", "the same session index it was given",
            "a different index", "The lag moves values, not rows.")
    require(list(out.columns) == list(frame.columns), "interface", "the same columns",
            "different columns")


def _leakage(obj) -> str:
    frame = fixtures.prices()
    cut = frame.index[200]
    full = obj(frame)
    truncated = obj(frame.loc[:cut])
    a, b = full.loc[:cut], truncated
    require(same(a, b), "leakage probe",
            "the same output for every date up to the cut whether or not later data exists",
            "output that changes when data after the cut is added",
            "That is what look-ahead is: a value at date t that could not have been computed on "
            "date t, because it depended on something that had not happened yet.")
    return "output up to a cut date is unchanged by data after it"


def _actually_lags(obj) -> str:
    frame = fixtures.prices()
    out = obj(frame)
    aligned = int((out.round(10) == frame.round(10)).to_numpy().sum())
    require(aligned == 0, "the lag is applied",
            "no cell carrying the value observed on its own date",
            f"{aligned} cells still carrying same-day values",
            "A lag of zero is a decision to treat data as available the instant it is stamped, "
            "which no vendor delivers.")
    return "no cell carries a same-day value"


def _leading_gap(obj) -> str:
    out = obj(fixtures.prices())
    require(out.iloc[0].isna().all(), "leading rows empty",
            "the first rows empty, because nothing was available yet",
            "a first row with values in it")
    return "the panel starts empty and fills as data becomes available"


register(Contract(
    name="availability_lag",
    kind="callable",
    units=("2.2",),
    summary="Delays every observation by the time it takes to become usable.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable frame -> frame with the same index and columns",
    reference=reference,
    leakage=_leakage,
    leakage_note="output up to a cut date is unchanged by data after it",
    invariants=(("the lag is applied", _actually_lags), ("leading rows empty", _leading_gap)),
))

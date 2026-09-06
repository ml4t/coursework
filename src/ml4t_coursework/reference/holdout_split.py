"""`holdout_split`. A dated cutoff, written down and not opened."""

from __future__ import annotations

import pandas as pd

from .. import fixtures
from ..checks import require
from ..contracts import Contract, register

FRACTION = 0.25


def reference(fraction: float = FRACTION):
    """The last stretch of the sample, sealed. It is a date, chosen before anything is fitted, and
    the only interesting property of the component is that everything else respects it."""

    def holdout_split(index: pd.Index, fraction: float = fraction):
        index = pd.Index(index).sort_values()
        cut = int(len(index) * (1 - fraction))
        return index[:cut], index[cut:]

    return holdout_split


def _probe(obj):
    return obj(fixtures.prices().index)


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a date index", f"a {type(obj).__name__}")
    out = _probe(obj)
    require(isinstance(out, tuple) and len(out) == 2, "interface",
            "a (development, holdout) pair", f"a {type(out).__name__}")
    development, holdout = out
    require(len(development) > 0 and len(holdout) > 0, "interface",
            "both blocks non-empty", f"{len(development)} and {len(holdout)} dates")


def _leakage(obj) -> str:
    development, holdout = _probe(obj)
    shared = pd.Index(development).intersection(pd.Index(holdout))
    require(len(shared) == 0, "leakage probe",
            "no date in both the development sample and the holdout",
            f"{len(shared)} dates in both",
            "A holdout that overlaps the development sample has already been spent.")
    return "no date is in both blocks"


def _holdout_is_last(obj) -> str:
    development, holdout = _probe(obj)
    require(pd.Index(development).max() < pd.Index(holdout).min(), "the holdout is the last block",
            "a holdout that begins after the development sample ends",
            f"development runs to {pd.Index(development).max().date()} and the holdout begins "
            f"{pd.Index(holdout).min().date()}",
            "A holdout carved out of the middle is tested on a period the model has been "
            "developed around on both sides.")
    return "the holdout follows the development sample"


def _covers(obj) -> str:
    index = fixtures.prices().index
    development, holdout = obj(index)
    covered = pd.Index(development).union(pd.Index(holdout))
    missing = len(index) - len(covered)
    require(missing == 0, "the split covers the sample",
            "every session in one block or the other", f"{missing} sessions in neither",
            "Sessions in neither block are a third, undeclared sample.")
    return f"{len(development)} development sessions and {len(holdout)} sealed"


register(Contract(
    name="holdout_split",
    kind="callable",
    summary="Seals the last stretch of the sample behind a dated cutoff.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable index -> (development index, holdout index)",
    reference=reference,
    leakage=_leakage,
    leakage_note="no date is in both blocks",
    invariants=(("the holdout is the last block", _holdout_is_last),
                ("the split covers the sample", _covers)),
))

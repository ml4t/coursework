"""`signal` - unit 7.1. From a score to a position: which names, and which way."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..checks import require, same
from ..contracts import Contract, register
from ._shared import score_frame

QUANTILE = 0.2


def reference(quantile: float = QUANTILE):
    """A cross-sectional rule: long the strongest fifth of the names on the date, short the
    weakest, flat on everything between. It reads order and nothing else, which is why 4.3 could
    fix the task form without training four models."""

    def signal(scores: pd.DataFrame, quantile: float = quantile) -> pd.DataFrame:
        ranks = scores.rank(axis=1, pct=True, na_option="keep")
        longs = (ranks > 1 - quantile).astype(float)
        shorts = (ranks < quantile).astype(float)
        return (longs - shorts).where(scores.notna(), 0.0)

    return signal


def _probe(obj):
    return obj(score_frame())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a frame of scores",
            f"a {type(obj).__name__}")
    out = _probe(obj)
    scores = score_frame()
    require(isinstance(out, pd.DataFrame), "interface", "a DataFrame of positions",
            f"a {type(out).__name__}")
    require(out.index.equals(scores.index), "interface", "one row per scored date",
            f"{len(out)} rows against {len(scores)}")
    require(list(out.columns) == list(scores.columns), "interface",
            "the same assets it was scored on", "different columns")


def _leakage(obj) -> str:
    scores = score_frame()
    cut = scores.index[80]
    full = obj(scores)
    truncated = obj(scores.loc[:cut])
    require(same(full.loc[:cut], truncated), "leakage probe",
            "a position on a date to depend only on that date's scores",
            "a position that changes when later scores arrive",
            "A rule that ranks against the whole sample's scores rather than the date's is "
            "deciding today's trade with next year's distribution.")
    return "a date's positions depend only on that date's scores"


def _bounded(obj) -> str:
    out = _probe(obj)
    values = out.to_numpy(dtype=float)
    worst = float(np.nanmax(np.abs(values)))
    require(worst <= 1.0 + 1e-9, "positions are bounded",
            "positions in [-1, 1], because sizing is the allocator's decision and not this one's",
            f"a position of {worst:.3g}",
            "This component says which names and which way. How much is 8.1.")
    require(not np.isnan(values).any(), "positions are bounded",
            "a position for every name, flat where there is no view", "empty cells")
    return f"positions within [-1, 1] across {out.shape[0]} dates"


def _takes_a_view(obj) -> str:
    out = _probe(obj)
    active = (out.abs() > 0).sum(axis=1)
    flat_dates = int((active == 0).sum())
    require(flat_dates < len(out), "the rule takes a view",
            "some date on which the rule is in the market",
            "a rule that is flat on every date",
            "A signal that never takes a position cannot be read against a baseline, which is "
            "what 7.3 asks it to do.")
    return f"in the market on {len(out) - flat_dates} of {len(out)} dates"


def _two_sided(obj) -> str:
    out = _probe(obj)
    require(bool((out.to_numpy() > 0).any()), "the view has a direction",
            "at least one long position", "no long positions anywhere")
    return "the rule expresses a direction rather than a magnitude"


register(Contract(
    name="signal",
    kind="callable",
    units=("6.1",),
    summary="Turns model scores into which names to hold and which way.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable scores(date x asset) -> positions(date x asset)",
    reference=reference,
    leakage=_leakage,
    leakage_note="a date's positions depend only on that date's scores",
    invariants=(("positions are bounded", _bounded),
                ("the rule takes a view", _takes_a_view),
                ("the view has a direction", _two_sided)),
))

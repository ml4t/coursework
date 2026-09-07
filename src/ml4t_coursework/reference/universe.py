"""`universe`. Who you may trade, decided with what was knowable at the time."""

from __future__ import annotations

import pandas as pd

from .. import fixtures
from ..checks import require
from ..contracts import Contract, register

LOOKBACK = 126


def reference(lookback: int = LOOKBACK):
    """Eligible on date t: an asset with an unbroken price history over the trailing lookback.
    It is a membership rule, and the whole of its difficulty is that it must be answerable on t.

    On a long panel an asset that had no price simply has no row, so an unbroken history is a row
    on every one of the trailing sessions, with a price in it.
    """

    def universe(panel: pd.DataFrame, asof, lookback: int = lookback) -> list[str]:
        asof = pd.Timestamp(asof)
        dates = panel.index.get_level_values("date")
        sessions = dates[dates <= asof].unique()
        if len(sessions) < lookback:
            return []
        window = panel[(dates <= asof) & (dates >= sessions[-lookback])]
        priced = window["close"].notna().groupby(level="asset").sum()
        complete = priced[priced == lookback]
        return sorted(complete.index.tolist())

    return universe


def _asof():
    return fixtures.panel().index.get_level_values("date").unique()[250]


def _probe(obj):
    return obj(fixtures.panel(), _asof())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a panel and a date",
            f"a {type(obj).__name__}")
    out = _probe(obj)
    require(isinstance(out, (list, tuple)), "interface", "a list of assets",
            f"a {type(out).__name__}")
    require(len(out) > 0, "interface", "at least one eligible asset on a mid-sample date",
            "an empty universe",
            "An empty universe on a date with six years of history behind it means the rule is "
            "screening on something that is never true.")
    known = set(fixtures.panel().index.get_level_values("asset"))
    unknown = [s for s in out if s not in known]
    require(not unknown, "interface", "assets that are in the panel",
            f"{', '.join(unknown[:3])} which are not")


def _leakage(obj) -> str:
    panel = fixtures.panel()
    dates = panel.index.get_level_values("date")
    asof = _asof()
    full = obj(panel, asof)
    truncated = obj(panel[dates <= asof], asof)
    require(sorted(full) == sorted(truncated), "leakage probe",
            "the same membership whether or not prices after the date exist",
            "membership that changes when later prices are added",
            "Screening on an asset's whole history is how survivorship gets in: it admits the "
            "funds that were still around at the end and quietly drops the ones that closed.")
    return "membership on a date is unchanged by prices after it"


def _excludes_unlisted(obj) -> str:
    panel = fixtures.panel()
    dates = panel.index.get_level_values("date")
    sessions = dates.unique()
    early = sessions[LOOKBACK + 5]
    selected = obj(panel, early)
    window = sessions[: LOOKBACK + 6]
    present = panel[dates.isin(window)].groupby(level="asset").size()
    late_listers = [a for a in present.index if int(present[a]) < len(window)]
    admitted = [s for s in selected if s in late_listers]
    require(not admitted, "no asset admitted before it listed",
            "no asset whose history has a gap at that date",
            f"{', '.join(admitted[:3])} admitted with an incomplete history",
            "A fund with no price yet cannot be bought, and treating its first bar as though the "
            "history were there is the same mistake as filling it.")
    return f"{len(selected)} eligible, none of them still unlisted"


def _sorted_and_unique(obj) -> str:
    out = list(_probe(obj))
    require(len(out) == len(set(out)), "one entry per asset", "each asset once",
            f"{len(out) - len(set(out))} duplicates")
    return f"{len(out)} assets, each once"


register(Contract(
    name="universe",
    kind="callable",
    summary="Decides which assets may be traded on a date, using only what was known then.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable (panel, date) -> list of assets",
    reference=reference,
    leakage=_leakage,
    leakage_note="membership on a date is unchanged by prices after it",
    invariants=(("no asset admitted before it listed", _excludes_unlisted),
                ("one entry per asset", _sorted_and_unique)),
))

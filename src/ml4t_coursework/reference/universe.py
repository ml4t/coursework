"""`universe` - unit 2.4. Who you may trade, decided with what was knowable at the time."""

from __future__ import annotations

import pandas as pd

from .. import fixtures
from ..checks import require
from ..contracts import Contract, register

LOOKBACK = 126


def reference(lookback: int = LOOKBACK):
    """Eligible on date t: a symbol with an unbroken price history over the trailing lookback.
    It is a membership rule, and the whole of its difficulty is that it must be answerable on t."""

    def universe(prices: pd.DataFrame, asof, lookback: int = lookback) -> list[str]:
        asof = pd.Timestamp(asof)
        window = prices.loc[:asof].tail(lookback)
        if len(window) < lookback:
            return []
        complete = window.notna().all()
        return sorted(complete.index[complete].tolist())

    return universe


def _asof():
    return fixtures.prices().index[250]


def _probe(obj):
    return obj(fixtures.prices(), _asof())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking prices and a date",
            f"a {type(obj).__name__}")
    out = _probe(obj)
    require(isinstance(out, (list, tuple)), "interface", "a list of symbols",
            f"a {type(out).__name__}")
    require(len(out) > 0, "interface", "at least one eligible symbol on a mid-sample date",
            "an empty universe",
            "An empty universe on a date with six years of history behind it means the rule is "
            "screening on something that is never true.")
    known = set(fixtures.prices().columns)
    unknown = [s for s in out if s not in known]
    require(not unknown, "interface", "symbols that are in the panel",
            f"{', '.join(unknown[:3])} which are not")


def _leakage(obj) -> str:
    prices = fixtures.prices()
    asof = _asof()
    full = obj(prices, asof)
    truncated = obj(prices.loc[:asof], asof)
    require(sorted(full) == sorted(truncated), "leakage probe",
            "the same membership whether or not prices after the date exist",
            "membership that changes when later prices are added",
            "Screening on a symbol's whole history is how survivorship gets in: it admits the "
            "funds that were still around at the end and quietly drops the ones that closed.")
    return "membership on a date is unchanged by prices after it"


def _excludes_unlisted(obj) -> str:
    prices = fixtures.prices()
    early = prices.index[LOOKBACK + 5]
    selected = obj(prices, early)
    late_listers = [c for c in prices.columns if prices[c].loc[:early].isna().any()]
    admitted = [s for s in selected if s in late_listers]
    require(not admitted, "no symbol admitted before it listed",
            "no symbol whose history has a gap at that date",
            f"{', '.join(admitted[:3])} admitted with an incomplete history",
            "A fund with no price yet cannot be bought, and treating its first bar as though the "
            "history were there is the same mistake as filling it.")
    return f"{len(selected)} eligible, none of them still unlisted"


def _sorted_and_unique(obj) -> str:
    out = list(_probe(obj))
    require(len(out) == len(set(out)), "one entry per symbol", "each symbol once",
            f"{len(out) - len(set(out))} duplicates")
    return f"{len(out)} symbols, each once"


register(Contract(
    name="universe",
    kind="callable",
    units=("2.4",),
    summary="Decides which symbols may be traded on a date, using only what was known then.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable (prices, date) -> list of symbols",
    reference=reference,
    leakage=_leakage,
    leakage_note="membership on a date is unchanged by prices after it",
    invariants=(("no symbol admitted before it listed", _excludes_unlisted),
                ("one entry per symbol", _sorted_and_unique)),
))

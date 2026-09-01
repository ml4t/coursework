"""`baseline_strategy` - unit 2.5. The auditable non-ML rule everything else must beat."""

from __future__ import annotations

import pandas as pd

from .. import fixtures
from ..checks import require, same
from ..contracts import Contract, register
from .universe import reference as universe_reference

REBALANCE = 21


def reference(rebalance: int = REBALANCE):
    """Equal weight across everything eligible, rebalanced monthly. Every line of it can be read
    and argued with, which is the property that makes it a yardstick: when the finished pipeline
    beats it you know what it beat, and when it does not you know that too."""

    def baseline_strategy(prices: pd.DataFrame, rebalance: int = rebalance) -> pd.DataFrame:
        eligible = universe_reference()
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        current = None
        for position, date in enumerate(prices.index):
            if position % rebalance == 0:
                names = eligible(prices, date)
                current = None if not names else pd.Series(1.0 / len(names), index=names)
            if current is not None:
                weights.loc[date, current.index] = current.to_numpy()
        return weights

    return baseline_strategy


def _prices():
    return fixtures.prices()


def _probe(obj):
    return obj(_prices())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a price panel",
            f"a {type(obj).__name__}")
    out = _probe(obj)
    prices = _prices()
    require(isinstance(out, pd.DataFrame), "interface", "a DataFrame of weights",
            f"a {type(out).__name__}")
    require(out.index.equals(prices.index), "interface", "one row of weights per session",
            f"{len(out)} rows against {len(prices)}")
    require(list(out.columns) == list(prices.columns), "interface", "the same symbols",
            "different columns")


def _leakage(obj) -> str:
    prices = _prices()
    cut = prices.index[250]
    full = obj(prices)
    truncated = obj(prices.loc[:cut])
    require(same(full.loc[:cut], truncated), "leakage probe",
            "the same weights up to a date whether or not later prices exist",
            "weights that change when later prices arrive",
            "A baseline with hindsight is not a floor, it is a ceiling nothing can clear.")
    return "weights up to a date are unchanged by prices after it"


def _invested(obj) -> str:
    out = _probe(obj)
    gross = out.abs().sum(axis=1)
    active = gross[gross > 0]
    require(len(active) > 100, "the baseline is invested",
            "a rule that holds something over most of the sample",
            f"positions on only {len(active)} of {len(out)} sessions",
            "A baseline that is mostly in cash is a comparison against cash, which 7.3 already "
            "makes separately.")
    worst = float((active - 1.0).abs().max())
    require(worst < 1e-6, "the baseline is invested",
            "a fully invested book on every active session",
            f"a session at {float(active.iloc[0]):.4f} gross")
    return f"invested on {len(active)} of {len(out)} sessions"


def _rules_only(obj) -> str:
    out = _probe(obj)
    distinct = out.round(8).drop_duplicates()
    require(len(distinct) < len(out) / 2, "the rule is auditable",
            "a book that changes on rebalance dates rather than every session, so what it does "
            "can be read off a page",
            f"{len(distinct)} distinct books over {len(out)} sessions",
            "A baseline you cannot state in a sentence is not doing the job of a baseline.")
    return f"{len(distinct)} distinct books over {len(out)} sessions"


def _finite(obj) -> str:
    out = _probe(obj)
    bad = int(out.isna().to_numpy().sum())
    require(bad == 0, "no empty weights", "a weight for every name on every session",
            f"{bad} empty cells")
    return "no empty weights"


register(Contract(
    name="baseline_strategy",
    kind="callable",
    units=("2.5",),
    summary="An auditable non-ML rule, fixed before any result is seen.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable prices -> weights(date x symbol)",
    reference=reference,
    leakage=_leakage,
    leakage_note="weights up to a date are unchanged by prices after it",
    invariants=(("the baseline is invested", _invested),
                ("the rule is auditable", _rules_only),
                ("no empty weights", _finite)),
))

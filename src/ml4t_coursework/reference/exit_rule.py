"""`exit_rule` - unit 8.4. Position controls, and the one threshold worth calibrating."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fixtures
from ..checks import require, same
from ..contracts import Contract, register
from ._shared import score_frame
from .allocator import reference as allocator_reference
from .signal import reference as signal_reference

STOP = 0.10


def reference(stop: float = STOP):
    """A stop on the position: once a holding has lost more than the threshold since it was put
    on, it is closed and stays closed until the next time the allocator asks for it fresh. The
    rule's specification and its threshold are two decisions, and only the second is calibrated."""

    def exit_rule(weights: pd.DataFrame, prices: pd.DataFrame, stop: float = stop) -> pd.DataFrame:
        aligned = prices.reindex(index=weights.index, columns=weights.columns).ffill()
        step = aligned.pct_change(fill_method=None).fillna(0.0)
        out = weights.copy()
        held = pd.Series(0.0, index=weights.columns)
        asked = pd.Series(0.0, index=weights.columns)
        entry_pnl = pd.Series(0.0, index=weights.columns)
        stopped = pd.Series(False, index=weights.columns)
        for date in weights.index:
            wanted = weights.loc[date]
            entry_pnl = entry_pnl + np.sign(held) * step.loc[date]
            # A position is new when the allocator changes its mind, not when the stop has just
            # flattened it. Reading the reset off the held book instead re-enters the next day
            # and the stop never holds.
            fresh = np.sign(wanted) != np.sign(asked)
            entry_pnl = entry_pnl.where(~fresh, 0.0)
            stopped = (stopped & ~fresh) | (entry_pnl < -stop)
            held = wanted.where(~stopped, 0.0)
            asked = wanted
            out.loc[date] = held
        return out

    return exit_rule


def _weights() -> pd.DataFrame:
    return allocator_reference()(signal_reference()(score_frame()))


def _prices() -> pd.DataFrame:
    return fixtures.prices()


def _probe(obj):
    return obj(_weights(), _prices())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking weights and prices",
            f"a {type(obj).__name__}")
    out = _probe(obj)
    weights = _weights()
    require(isinstance(out, pd.DataFrame), "interface", "a DataFrame of weights",
            f"a {type(out).__name__}")
    require(out.index.equals(weights.index), "interface", "one row per date",
            f"{len(out)} rows against {len(weights)}")
    require(list(out.columns) == list(weights.columns), "interface", "the same symbols",
            "different columns")


def _leakage(obj) -> str:
    weights, prices = _weights(), _prices()
    cut = weights.index[80]
    full = obj(weights, prices)
    truncated = obj(weights.loc[:cut], prices.loc[:cut])
    require(same(full.loc[:cut], truncated), "leakage probe",
            "an exit on a date to be decided from prices up to that date",
            "an exit that changes when later prices arrive",
            "A stop placed with hindsight is the most flattering rule in backtesting and the "
            "least available in trading.")
    return "an exit is decided from prices up to its own date"


def _never_adds(obj) -> str:
    weights = _weights()
    out = obj(weights, _prices())
    added = float((out.abs() - weights.abs()).to_numpy().max())
    require(added <= 1e-9, "exits only reduce",
            "a rule that closes positions and never opens one",
            f"a position increased by {added:.4f}",
            "Opening a position is the allocator's decision. This component only takes them off.")
    return "no position is larger after the rule than before it"


def _idempotent(obj) -> str:
    weights, prices = _weights(), _prices()
    once = obj(weights, prices)
    twice = obj(once, prices)
    require(same(once, twice), "applying it twice changes nothing",
            "a book that is already stopped out to stay as it is",
            "a second application that moves the book again",
            "A rule whose effect depends on how many times it ran is not a rule.")
    return "a second application leaves the book unchanged"


def _stress():
    """A book held steady through a fall deep enough that any stop worth the name must fire.

    The live signal churns, so a position rarely lives long enough to breach anything. That is a
    property of the fixture, not evidence about the rule, so the check that the rule fires gets a
    panel built to make it fire.
    """
    weights = _weights().copy()
    weights.loc[:, :] = 0.0
    weights.iloc[:, 0] = 1.0
    prices = _prices().reindex(weights.index).ffill().copy()
    fall = np.linspace(0.0, -0.35, len(prices))
    prices.iloc[:, 0] = float(prices.iloc[0, 0]) * (1.0 + fall)
    return weights, prices


def _fires(obj) -> str:
    weights, prices = _stress()
    out = obj(weights, prices)
    closed = int(((weights.abs() > 0) & (out.abs() == 0)).to_numpy().sum())
    require(closed > 0, "the rule does something",
            "a position closed somewhere in a 35% fall held throughout",
            "a rule that never fires even then",
            "A threshold so wide it never triggers is not a control; 8.4's own result is that a "
            "stop grid need not identify a stable threshold, which is a different finding from "
            "never testing one.")
    return f"{closed} position-days closed by the rule"


register(Contract(
    name="exit_rule",
    kind="callable",
    units=("7.4",),
    summary="Closes a position that has breached its control, and leaves it closed.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable (weights, prices) -> weights",
    reference=reference,
    leakage=_leakage,
    leakage_note="an exit is decided from prices up to its own date",
    invariants=(("exits only reduce", _never_adds),
                ("applying it twice changes nothing", _idempotent),
                ("the rule does something", _fires)),
))

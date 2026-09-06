"""`labeler`. The outcome the model is asked to predict.

The first unit fixes the horizon and what overlap costs. The second tightens the same component
so entry and exit
are prices actually reachable after the decision, rather than the close of the bar the decision
was made on.
"""

from __future__ import annotations

import pandas as pd

from .. import fixtures
from ..checks import require, same
from ..contracts import Contract, register

HORIZON = 21
EXECUTION_DELAY = 1


def reference(horizon: int = HORIZON, delay: int = EXECUTION_DELAY):
    """The return between the two prices a decision at t can actually transact at: the close one
    session later, and the close `horizon` sessions after that. Stamped at t, the date the
    decision was made, so the feature row and the label row line up on the decision.

    The panel arrives long and the label leaves long. In between it is put back on a session grid
    per asset, because `shift` has to step forward in sessions for one asset at a time, and an
    asset that stopped trading must not have its next session borrowed from another.
    """

    def labeler(panel: pd.DataFrame, horizon: int = horizon, delay: int = delay) -> pd.Series:
        close = panel["close"].unstack("asset")
        entry = close.shift(-delay)
        exit_ = close.shift(-delay - horizon)
        forward = (exit_ / entry - 1.0).stack(future_stack=True).dropna()
        forward.index = forward.index.set_names(["date", "asset"])
        return forward.rename("label").astype("float64")

    return labeler


def _probe(obj):
    return obj(fixtures.panel())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a panel", f"a {type(obj).__name__}")
    out = _probe(obj)
    require(isinstance(out, pd.Series), "interface", "a Series of labels",
            f"a {type(out).__name__}")
    require(isinstance(out.index, pd.MultiIndex) and out.index.nlevels == 2, "interface",
            "an index of (date, asset), so a label lines up with the decision it belongs to",
            f"an index of {type(out.index).__name__} with {out.index.nlevels} level(s)")
    require(list(out.index.names) == ["date", "asset"], "interface",
            "index levels named date and asset", f"levels named {list(out.index.names)}")
    require(len(out) > 0, "interface", "some labels", "an empty Series")


def _leakage(obj) -> str:
    """A label is forward-looking by construction, so the probe asks the sharper question: does a
    label that had already resolved by the cut change when later prices arrive?"""
    panel = fixtures.panel()
    dates = panel.index.get_level_values("date")
    sessions = dates.unique()
    cut_pos = 240
    cut = sessions[cut_pos]
    resolved = sessions[cut_pos - HORIZON - EXECUTION_DELAY - 5]
    full = obj(panel)
    truncated = obj(panel[dates <= cut])
    a = full[full.index.get_level_values("date") <= resolved]
    b = truncated[truncated.index.get_level_values("date") <= resolved]
    require(len(b) > 0, "leakage probe", "labels that had resolved before the cut",
            "no resolved labels at all")
    require(same(*(x.sort_index() for x in (a.reindex(b.index), b))), "leakage probe",
            "a label that resolved before the cut to be the same either way",
            "a resolved label that moves when later prices arrive",
            "A label already settled cannot depend on what happened afterwards.")
    return "labels that resolved before a cut are unchanged by prices after it"


def _stamped_on_the_decision(obj) -> str:
    panel = fixtures.panel()
    sessions = panel.index.get_level_values("date").unique()
    out = obj(panel)
    last_labelled = out.index.get_level_values("date").max()
    unresolved = int((sessions > last_labelled).sum())
    require(unresolved >= HORIZON, "stamped on the decision",
            f"the last {HORIZON} or more sessions to carry no label, because their outcome has "
            "not happened yet",
            f"only {unresolved} unlabelled sessions at the end of the panel",
            "A label stamped at t must describe what happened after t. If the panel's final date "
            "has a label, something has been read off the end.")
    return f"the last {unresolved} sessions carry no label"


def _executable(obj) -> str:
    """The executable-price tightening: the entry is a price a decision at t could have transacted."""
    panel = fixtures.panel()
    out = obj(panel)
    close = panel["close"].unstack("asset")
    same_bar = (close.shift(-HORIZON) / close - 1.0).stack(future_stack=True).dropna()
    same_bar.index = same_bar.index.set_names(["date", "asset"])
    shared = out.index.intersection(same_bar.index)
    require(len(shared) > 0, "executable prices", "labels to compare", "no overlap")
    identical = same(out.reindex(shared).sort_index(), same_bar.reindex(shared).sort_index())
    require(not identical, "executable prices",
            "entry at a price reachable after the decision",
            "entry at the close of the bar the decision was made on",
            "The decision is made from that bar's close, so transacting at it assumes an order "
            "filled at a price that was only known once the bar was over.")
    return "entry and exit prices are both reachable after the decision"


def _finite(obj) -> str:
    out = _probe(obj)
    bad = int(out.isna().sum())
    require(bad == 0, "no empty labels", "every returned label to be a number",
            f"{bad} of {len(out)} empty",
            "Drop the rows you cannot label rather than returning them empty.")
    return f"{len(out)} labels, none empty"


register(Contract(
    name="labeler",
    kind="callable",
    summary="Turns prices into the outcome a decision at t is judged on, stamped at t.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable panel -> Series indexed by (date, asset)",
    reference=reference,
    leakage=_leakage,
    leakage_note="labels that resolved before a cut are unchanged by prices after it",
    invariants=(("stamped on the decision", _stamped_on_the_decision),
                ("executable prices", _executable),
                ("no empty labels", _finite)),
))

"""`quality_gates`. What a breach does, and to which bar."""

from __future__ import annotations

import pandas as pd

from .. import fixtures
from ..checks import require, same
from ..contracts import Contract, register

MAX_MOVE = 0.5
MAX_FLAT_RUN = 5


def reference(max_move: float = MAX_MOVE, max_flat_run: int = MAX_FLAT_RUN):
    """Three gates, each on information available at the bar itself or the one before it:
    a non-positive price, a one-session move too large to be a price, and a run of identical
    closes long enough to be a stale feed. A breach voids the bar; it does not void the asset.

    Every comparison with the previous bar is taken within an asset. The row above on a long
    panel is a different asset on the same date, so a gate that ignores the asset level measures
    the gap between two unrelated prices and reports it as a jump.
    """

    def quality_gates(panel: pd.DataFrame, max_move: float = max_move,
                      max_flat_run: int = max_flat_run):
        close = panel["close"]
        by_asset = close.groupby(level="asset", sort=False)
        step = by_asset.pct_change(fill_method=None).abs()

        nonpositive = close <= 0
        jump = step > max_move
        flat = (step == 0) & step.notna()
        run = flat.astype(int)
        for _ in range(max_flat_run - 1):
            prior = run.groupby(level="asset", sort=False).shift(1).fillna(0).astype(int)
            run = run * (prior + 1).clip(upper=max_flat_run)
        stale = run >= max_flat_run

        breaches = []
        clean = panel.copy()
        for gate, mask in (("non-positive price", nonpositive), ("implausible move", jump),
                           ("stale feed", stale)):
            for date, asset in mask[mask].index:
                breaches.append({"gate": gate, "asset": asset, "date": date})
            clean = clean.mask(mask, axis=0)
        report = pd.DataFrame(breaches, columns=["gate", "asset", "date"])
        return clean, report

    return quality_gates


def _dirty() -> pd.DataFrame:
    """The fixture panel with two bad bars injected, so the gates have something to catch."""
    panel = fixtures.panel().copy()
    dates = panel.index.get_level_values("date").unique()
    panel.loc[(dates[200], "ET00"), "close"] *= 4.0
    panel.loc[(dates[210], "ET01"), "close"] = -1.0
    return panel


def _injected() -> list[tuple]:
    dates = fixtures.panel().index.get_level_values("date").unique()
    return [(dates[200], "ET00"), (dates[210], "ET01")]


def _probe(obj):
    clean, report = obj(_dirty())
    return clean


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a panel", f"a {type(obj).__name__}")
    out = obj(_dirty())
    require(isinstance(out, tuple) and len(out) == 2, "interface",
            "a (cleaned panel, report) pair", f"a {type(out).__name__}")
    clean, report = out
    require(isinstance(clean, pd.DataFrame), "interface", "a DataFrame of cleaned prices",
            f"a {type(clean).__name__}")
    require(isinstance(report, pd.DataFrame), "interface", "a report as a DataFrame",
            f"a {type(report).__name__}")
    require(clean.index.equals(_dirty().index), "interface",
            "the same (date, asset) index it was given", "a different index",
            "Void the bar, do not drop the row: a dropped row is indistinguishable from a "
            "session the asset never traded.")
    for column in ("gate", "asset", "date"):
        require(column in report.columns, "interface",
                f"a report with a {column!r} column, so a breach can be traced to a bar",
                f"columns {list(report.columns)}")


def _leakage(obj) -> str:
    panel = _dirty()
    dates = panel.index.get_level_values("date")
    cut = dates.unique()[250]
    full, _ = obj(panel)
    truncated, _ = obj(panel[dates <= cut])
    require(same(full[dates <= cut], truncated), "leakage probe",
            "the same verdict on a bar whether or not later bars exist",
            "a verdict that changes when later data arrives",
            "A gate calibrated on the whole history - a z-score against the full-sample standard "
            "deviation, say - decides today using next year's data.")
    return "a bar's verdict does not change when later bars arrive"


def _catches(obj) -> str:
    clean, report = obj(_dirty())
    require(len(report) > 0, "injected breach caught", "the two bad bars to be reported",
            "an empty report")
    for key in _injected():
        require(bool(pd.isna(clean.loc[key, "close"])), "injected breach caught",
                f"the bad bar on {key[1]} at {key[0].date()} voided",
                f"{clean.loc[key, 'close']}")
    return f"{len(report)} breaches reported, both injected bars among them"


def _keeps_good(obj) -> str:
    panel = _dirty()
    clean, report = obj(panel)
    lost = int(panel["close"].notna().sum() - clean["close"].notna().sum())
    require(lost <= 12, "good bars kept",
            "only the breaching bars voided, not the series around them",
            f"{lost} bars voided against 2 injected",
            "A gate that voids more than it catches costs more history than the bad data did.")
    require(clean.shape == panel.shape, "good bars kept", "a panel of the same shape",
            f"{clean.shape} against {panel.shape}", "Void the bar, do not drop the row.")
    return f"{lost} bars voided"


register(Contract(
    name="quality_gates",
    kind="callable",
    summary="Voids bars that fail a stated quality gate, and reports which gate caught what.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable panel -> (cleaned panel, report with gate/asset/date)",
    reference=reference,
    leakage=_leakage,
    leakage_note="a bar's verdict does not change when later bars arrive",
    invariants=(("injected breach caught", _catches), ("good bars kept", _keeps_good)),
))

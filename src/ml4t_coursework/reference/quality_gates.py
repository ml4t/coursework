"""`quality_gates` - unit 2.3. What a breach does, and to which bar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fixtures
from ..checks import require, same
from ..contracts import Contract, register

MAX_MOVE = 0.5
MAX_FLAT_RUN = 5


def reference(max_move: float = MAX_MOVE, max_flat_run: int = MAX_FLAT_RUN):
    """Three gates, each on information available at the bar itself or the one before it:
    a non-positive price, a one-session move too large to be a price, and a run of identical
    closes long enough to be a stale feed. A breach voids the bar; it does not void the symbol."""

    def quality_gates(panel: pd.DataFrame, max_move: float = max_move,
                      max_flat_run: int = max_flat_run):
        breaches = []
        clean = panel.copy()
        nonpositive = panel <= 0
        step = panel.pct_change(fill_method=None).abs()
        jump = step > max_move
        flat = (step == 0) & step.notna()
        run = flat.astype(int)
        for _ in range(max_flat_run - 1):
            run = run * (run.shift(1).fillna(0).astype(int) + 1).clip(upper=max_flat_run)
        stale = run >= max_flat_run
        for gate, mask in (("non-positive price", nonpositive), ("implausible move", jump),
                           ("stale feed", stale)):
            hits = mask.stack()
            hits = hits[hits]
            for date, symbol in hits.index:
                breaches.append({"gate": gate, "symbol": symbol, "date": date})
            clean = clean.mask(mask)
        report = pd.DataFrame(breaches, columns=["gate", "symbol", "date"])
        return clean, report

    return quality_gates


def _dirty() -> pd.DataFrame:
    """The fixture panel with one bad bar injected, so the gates have something to catch."""
    panel = fixtures.prices().copy()
    panel.iloc[100, 0] = panel.iloc[100, 0] * 4.0
    panel.iloc[150, 1] = -1.0
    return panel


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
    for column in ("gate", "symbol", "date"):
        require(column in report.columns, "interface",
                f"a report with a {column!r} column, so a breach can be traced to a bar",
                f"columns {list(report.columns)}")


def _leakage(obj) -> str:
    panel = _dirty()
    cut = panel.index[200]
    full, _ = obj(panel)
    truncated, _ = obj(panel.loc[:cut])
    require(same(full.loc[:cut], truncated), "leakage probe",
            "the same verdict on a bar whether or not later bars exist",
            "a verdict that changes when later data arrives",
            "A gate calibrated on the whole history - a z-score against the full-sample standard "
            "deviation, say - decides today using next year's data.")
    return "a bar's verdict does not change when later bars arrive"


def _catches(obj) -> str:
    panel = _dirty()
    clean, report = obj(panel)
    require(len(report) > 0, "injected breach caught", "the two bad bars to be reported",
            "an empty report")
    require(np.isnan(clean.iloc[150, 1]), "injected breach caught",
            "the negative price at row 150 voided", f"{clean.iloc[150, 1]}")
    require(np.isnan(clean.iloc[100, 0]), "injected breach caught",
            "the four-fold jump at row 100 voided", f"{clean.iloc[100, 0]}")
    return f"{len(report)} breaches reported, both injected bars among them"


def _keeps_good(obj) -> str:
    panel = _dirty()
    clean, report = obj(panel)
    lost = int(panel.notna().to_numpy().sum() - clean.notna().to_numpy().sum())
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
    units=("2.3",),
    summary="Voids bars that fail a stated quality gate, and reports which gate caught what.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable panel -> (cleaned panel, report with gate/symbol/date)",
    reference=reference,
    leakage=_leakage,
    leakage_note="a bar's verdict does not change when later bars arrive",
    invariants=(("injected breach caught", _catches), ("good bars kept", _keeps_good)),
))

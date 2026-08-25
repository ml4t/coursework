"""`data_panel` - unit 2.1. The stored price panel, loaded with its close convention stated."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .. import fixtures
from ..checks import require
from ..contracts import Contract, register


@lru_cache(maxsize=1)
def _fixture_file() -> str:
    path = Path(tempfile.mkdtemp(prefix="ml4t-fixture-")) / "panel.parquet"
    fixtures.prices().to_parquet(path)
    return str(path)


def reference():
    """Read the stored panel. Adjusted close, wide, one column per symbol, dates as the index."""

    def data_panel(path: str) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index, name="date")
        return frame.sort_index().astype("float64")

    return data_panel


def _probe(obj):
    return obj(_fixture_file())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a file path", f"a {type(obj).__name__}")
    panel = _probe(obj)
    require(isinstance(panel, pd.DataFrame), "interface", "a DataFrame",
            f"a {type(panel).__name__}")
    require(isinstance(panel.index, pd.DatetimeIndex), "interface", "a DatetimeIndex of sessions",
            f"an index of {type(panel.index).__name__}")
    require(str(panel.dtypes.iloc[0]).startswith("float"), "interface", "float prices",
            f"columns of dtype {panel.dtypes.iloc[0]}")


def _shaped(obj) -> str:
    panel = _probe(obj)
    expected = fixtures.prices()
    require(panel.shape == expected.shape, "panel shape",
            f"{expected.shape[0]} sessions by {expected.shape[1]} symbols",
            f"{panel.shape[0]} by {panel.shape[1]}",
            "Check the file you read and whether anything was dropped on load.")
    return f"{panel.shape[0]} sessions by {panel.shape[1]} symbols"


def _ordered(obj) -> str:
    panel = _probe(obj)
    require(panel.index.is_monotonic_increasing, "index sound", "sessions in date order",
            "an index that goes backwards somewhere")
    require(panel.index.is_unique, "index sound", "one row per session",
            f"{int(panel.index.duplicated().sum())} duplicated dates")
    return "one row per session, in date order"


def _unbalanced(obj) -> str:
    panel = _probe(obj)
    on_disk = pd.read_parquet(_fixture_file())
    expected = int(on_disk.isna().to_numpy().sum())
    found = int(panel.isna().to_numpy().sum())
    require(found == expected, "unbalanced panel kept",
            f"the {expected} empty cells that are in the file, marking the sessions before a fund "
            "listed and after it stopped trading",
            f"{found}",
            "Filling or dropping them here invents history for a fund that did not exist yet, and "
            "carries a delisted one forward at its last price. Notebooks drop those rows where "
            "they need to, per data/etf_close.json.")
    return f"{found} empty cells left exactly as the file has them"


register(Contract(
    name="data_panel",
    kind="callable",
    units=("2.1",),
    summary="Loads the stored adjusted-close panel, wide, one column per symbol.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable path -> wide DataFrame of float closes indexed by session",
    reference=reference,
    leakage=None,
    leakage_note="not applicable: loading a stored file involves no time ordering to violate",
    invariants=(("panel shape", _shaped), ("index sound", _ordered),
                ("unbalanced panel kept", _unbalanced)),
))

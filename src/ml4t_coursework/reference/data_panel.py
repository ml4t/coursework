"""`data_panel`. The stored price panel, loaded with its close convention stated.

The file on disk is wide, one column per asset, because that is how a price download arrives and
how it stores compactly. What the rest of the pipeline works in is long: one row per (date,
asset), a `close` column, and any feature columns the market's own data supplied. Everything
downstream is written against the long shape, so this is where the change of shape happens, once.

The long shape is what lets a market carry columns the ETF study does not have - a funding rate,
an open interest, a roll flag - without every component downstream growing a special case. Wide
is one `unstack` away for the two components that genuinely work across assets on a date.
"""

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
    """Read the stored panel and return it long. Adjusted close, one row per (date, asset).

    An asset with no price on a session gets no row, rather than a row holding an empty price.
    That is the same fact the wide file states with an empty cell, and it is the form the rest of
    the pipeline can act on: a component iterating rows never has to ask whether this one is real.
    """

    def data_panel(path: str) -> pd.DataFrame:
        wide = pd.read_parquet(path)
        wide.index = pd.DatetimeIndex(wide.index, name="date")
        long = wide.stack(future_stack=True).rename("close").to_frame().dropna()
        long.index = long.index.set_names(["date", "asset"])
        return long.sort_index().astype("float64")

    return data_panel


def _probe(obj):
    return obj(_fixture_file())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a file path", f"a {type(obj).__name__}")
    panel = _probe(obj)
    require(isinstance(panel, pd.DataFrame), "interface", "a DataFrame",
            f"a {type(panel).__name__}")
    require(isinstance(panel.index, pd.MultiIndex) and panel.index.nlevels == 2, "interface",
            "an index of (date, asset)",
            f"an index of {type(panel.index).__name__} with {panel.index.nlevels} level(s)",
            "The wide file is the storage shape. Every component downstream reads the long one.")
    require(list(panel.index.names) == ["date", "asset"], "interface",
            "index levels named date and asset", f"levels named {list(panel.index.names)}")
    require(isinstance(panel.index.get_level_values("date"), pd.DatetimeIndex), "interface",
            "dates on the date level",
            f"a level of {panel.index.get_level_values('date').dtype}")
    require("close" in panel.columns, "interface", "a close column",
            f"columns {list(panel.columns)}")
    require(str(panel["close"].dtype).startswith("float"), "interface", "float prices",
            f"a close column of dtype {panel['close'].dtype}")


def _shaped(obj) -> str:
    panel = _probe(obj)
    on_disk = pd.read_parquet(_fixture_file())
    assets = on_disk.shape[1]
    found = panel.index.get_level_values("asset").nunique()
    require(found == assets, "panel shape", f"the {assets} assets the file has", f"{found}",
            "Check the file you read and whether a column was dropped on load.")
    sessions = panel.index.get_level_values("date")
    require(sessions.min() == on_disk.index.min() and sessions.max() == on_disk.index.max(),
            "panel shape", f"the file's own span, {on_disk.index.min().date()} to "
            f"{on_disk.index.max().date()}",
            f"{sessions.min().date()} to {sessions.max().date()}")
    return f"{assets} assets over {sessions.nunique()} sessions"


def _ordered(obj) -> str:
    panel = _probe(obj)
    require(panel.index.is_monotonic_increasing, "index sound", "rows in date order",
            "an index that goes backwards somewhere")
    require(panel.index.is_unique, "index sound", "one row per asset per session",
            f"{int(panel.index.duplicated().sum())} duplicated (date, asset) pairs")
    return "one row per asset per session, in date order"


def _unbalanced(obj) -> str:
    panel = _probe(obj)
    on_disk = pd.read_parquet(_fixture_file())
    missing = int(on_disk.isna().to_numpy().sum())
    priced = int(on_disk.notna().to_numpy().sum())
    require(len(panel) == priced, "unbalanced panel kept",
            f"{priced} rows, one for each price the file actually has",
            f"{len(panel)} rows",
            "Reindexing onto every (date, asset) pair invents history for a fund that had not "
            "listed and carries a delisted one forward at its last price. In a long panel the "
            "absent row is the statement that there was no price.")
    empty = int(panel["close"].isna().sum())
    require(empty == 0, "unbalanced panel kept",
            "no row holding an empty price, because a session an asset had no price on is a row "
            "that does not exist",
            f"{empty} rows with an empty close",
            "Carrying the gap as an empty row moves the problem downstream instead of stating it.")
    sessions = on_disk.index
    counts = panel.groupby(level="date").size()
    require(int(counts.iloc[0]) < on_disk.shape[1], "unbalanced panel kept",
            "fewer assets on the first session than the file has columns, because some had not "
            "listed yet",
            f"all {on_disk.shape[1]} assets present on {sessions[0].date()}",
            "Reindexing onto the full asset list invents history for a fund that did not exist "
            "yet, and carries a delisted one forward at its last price.")
    return (f"{missing} absent (date, asset) pairs, matching the empty cells in the file; "
            f"{int(counts.iloc[0])} assets on the first session and {int(counts.iloc[-1])} on the "
            "last")


register(Contract(
    name="data_panel",
    kind="callable",
    summary="Loads the stored price file and returns it long, one row per (date, asset).",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable path -> DataFrame indexed by (date, asset) with a close column",
    reference=reference,
    leakage=None,
    leakage_note="not applicable: loading a stored file involves no time ordering to violate",
    invariants=(("panel shape", _shaped), ("index sound", _ordered),
                ("unbalanced panel kept", _unbalanced)),
))

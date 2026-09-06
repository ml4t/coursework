"""`features` - units 5.1 and 5.2. A feature is a hypothesis about a driver.

5.1 builds one feature that encodes a named driver. 5.2 tightens the same component so the
lookback is read against the label horizon and the values are normalized inside each date's
cross-section rather than against pooled history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fixtures
from ..checks import require, same
from ..contracts import Contract, register


def reference():
    """Three features, each arithmetic on bars at or before t, each z-scored within its own date.

    The hypothesis in each: recent relative strength persists over the label horizon; the same
    over a longer window; and a name that has been moving more than its peers is priced
    differently. Whether any of them is true is the course's question, not this file's.
    """

    def features(prices: pd.DataFrame) -> pd.DataFrame:
        raw = pd.DataFrame({
            "mom_21": prices.pct_change(21, fill_method=None).stack(future_stack=True),
            "mom_63": prices.pct_change(63, fill_method=None).stack(future_stack=True),
            "vol_21": prices.pct_change(fill_method=None).rolling(21).std().stack(future_stack=True),
        })
        raw.index = raw.index.set_names(["date", "symbol"])
        raw = raw.dropna()
        by_date = raw.groupby(level="date")
        centred = (raw - by_date.transform("mean")) / by_date.transform("std").replace(0, np.nan)
        return centred.dropna().astype("float64")

    return features


def _probe(obj):
    return obj(fixtures.prices())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a price panel",
            f"a {type(obj).__name__}")
    out = _probe(obj)
    require(isinstance(out, pd.DataFrame), "interface", "a DataFrame of features",
            f"a {type(out).__name__}")
    require(isinstance(out.index, pd.MultiIndex) and list(out.index.names) == ["date", "symbol"],
            "interface", "an index of (date, symbol), matching the labeler's",
            f"an index named {list(out.index.names)}")
    require(out.shape[1] >= 1, "interface", "at least one feature column", "no columns")
    require(len(out) > 0, "interface", "some feature rows", "an empty frame")


def _leakage(obj) -> str:
    prices = fixtures.prices()
    cut = prices.index[240]
    full = obj(prices)
    truncated = obj(prices.loc[:cut])
    a = full[full.index.get_level_values("date") <= cut]
    b = truncated[truncated.index.get_level_values("date") <= cut]
    shared = a.index.intersection(b.index)
    require(len(shared) > 0, "leakage probe", "feature rows on both sides of the cut",
            "no overlapping rows")
    require(same(a.loc[shared].sort_index(), b.loc[shared].sort_index()), "leakage probe",
            "a feature value at date t to be the same whether or not later prices exist",
            "a feature value that moves when later prices arrive",
            "This is what normalizing against pooled history does: the mean and standard "
            "deviation it subtracts include dates that had not happened yet.")
    return "a feature at date t is unchanged by prices after t"


def _normalized_per_date(obj) -> str:
    out = _probe(obj)
    by_date = out.groupby(level="date")
    means = by_date.mean().abs().to_numpy()
    spreads = by_date.std().to_numpy()
    standardized = np.nanmax(means) < 0.2 and 0.5 < float(np.nanmedian(spreads)) < 2.0
    bounded = float(np.nanmin(out.to_numpy())) >= -0.001 and float(np.nanmax(out.to_numpy())) <= 1.001
    require(standardized or bounded, "normalized within the date",
            "each date's cross-section on a comparable scale, either standardized around zero or "
            "ranked into the unit interval",
            f"cross-sections whose largest mean is {np.nanmax(means):.3g} and whose median spread "
            f"is {np.nanmedian(spreads):.3g}",
            "A model pooling dates compares a quiet January with a violent October unless each "
            "date is put on its own scale first.")
    return "standardized within each date" if standardized else "ranked within each date"


def _finite(obj) -> str:
    out = _probe(obj)
    bad = int(out.isna().to_numpy().sum())
    require(bad == 0, "no empty values", "every returned feature value to be a number",
            f"{bad} empty cells", "Drop the rows you cannot compute rather than returning them "
            "empty; a model will not tell you it silently dropped them.")
    return f"{len(out)} rows by {out.shape[1]} features, none empty"


register(Contract(
    name="features",
    kind="callable",
    units=("4.1", "4.2"),
    summary="Turns prices into the predictors the model sees, normalized within each date.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable prices -> DataFrame indexed by (date, symbol)",
    reference=reference,
    leakage=_leakage,
    leakage_note="a feature at date t is unchanged by prices after t",
    invariants=(("normalized within the date", _normalized_per_date),
                ("no empty values", _finite)),
))

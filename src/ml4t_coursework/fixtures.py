"""Small deterministic panels the contract tests run against.

These are synthetic on purpose. The course's own prices are licensed and cannot ship inside a
package, and a contract test does not need real prices: it asserts properties whose ground truth
we control, which is exactly the case `LESSON_CONTRACT.md` reserves synthetic data for. No number
a student sees on a slide is ever computed here.

Two shapes, and the difference between them is the point. `prices()` is the shape a price file
has on disk: wide, one column per asset, an empty cell where an asset had no price. `panel()` is
the shape every component downstream of `data_panel` works in: long, indexed by (date, asset),
with no row at all where there was no price. The unbalanced panel survives both, but it is an
empty cell in one and an absent row in the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260824
ASSETS = [f"ET{i:02d}" for i in range(12)]
SESSIONS = 320


def prices(n_sessions: int = SESSIONS, assets: list[str] | None = None) -> pd.DataFrame:
    """A wide close panel shaped like `data/etf_close.parquet`, unbalanced at the left edge."""
    assets = assets or ASSETS
    rng = np.random.default_rng(SEED)
    index = pd.bdate_range("2015-01-01", periods=n_sessions, name="date")
    steps = rng.normal(loc=0.0003, scale=0.011, size=(n_sessions, len(assets)))
    frame = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=assets)
    # Three assets list late, so anything that assumes a balanced panel fails here.
    for offset, asset in enumerate(assets[-3:], start=1):
        frame.loc[frame.index[: 20 * offset], asset] = np.nan
    # And two stop trading part way through, which is the half that matters: a rule screening on
    # an asset's whole history quietly keeps only the funds that were still around at the end.
    for offset, asset in enumerate(assets[3:5], start=1):
        frame.loc[frame.index[-40 * offset:], asset] = np.nan
    return frame


def panel(n_sessions: int = SESSIONS, assets: list[str] | None = None) -> pd.DataFrame:
    """The long panel `data_panel` returns: (date, asset) rows, a `close` column, no empty rows."""
    wide = prices(n_sessions, assets)
    long = wide.stack(future_stack=True).rename("close").to_frame().dropna()
    long.index = long.index.set_names(["date", "asset"])
    return long.sort_index().astype("float64")


def returns(n_sessions: int = SESSIONS) -> pd.Series:
    """A single daily return series, for the objective and cost contracts."""
    return prices(n_sessions)[ASSETS[0]].pct_change(fill_method=None).dropna()


def modelling_frame(n_sessions: int = SESSIONS) -> pd.DataFrame:
    """A long (date, asset) frame of features and a label, for the model and prep contracts."""
    px = prices(n_sessions)
    rng = np.random.default_rng(SEED + 1)
    long = px.stack().rename("close").to_frame()
    long["mom_21"] = px.pct_change(21, fill_method=None).stack()
    long["vol_21"] = px.pct_change(fill_method=None).rolling(21).std().stack()
    long["fwd_21"] = px.pct_change(21, fill_method=None).shift(-21).stack()
    long["noise"] = rng.normal(size=len(long))
    long.index = long.index.set_names(["date", "asset"])
    return long.dropna()


def scores(n_sessions: int = 60) -> pd.DataFrame:
    """Model output shaped as the signal and allocator contracts expect: date x asset."""
    rng = np.random.default_rng(SEED + 2)
    index = pd.bdate_range("2016-01-01", periods=n_sessions, name="date")
    return pd.DataFrame(rng.normal(size=(n_sessions, len(ASSETS))), index=index, columns=ASSETS)


def weights(n_sessions: int = 60) -> pd.DataFrame:
    """A conforming weight frame, for the cost and exit contracts."""
    raw = scores(n_sessions)
    ranked = raw.rank(axis=1, pct=True)
    long_leg = (ranked > 0.8).astype(float)
    short_leg = -(ranked < 0.2).astype(float)
    both = long_leg + short_leg
    return both.div(both.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

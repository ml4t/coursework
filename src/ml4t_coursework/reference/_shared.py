"""Panels the downstream contracts share, built once from the upstream references."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .. import fixtures
from .features import reference as features_reference
from .labeler import reference as labeler_reference


@lru_cache(maxsize=1)
def modelling_panel() -> tuple[pd.DataFrame, pd.Series]:
    X = features_reference()(fixtures.panel())
    y = labeler_reference()(fixtures.panel())
    shared = X.index.intersection(y.index)
    return X.loc[shared].sort_index(), y.loc[shared].sort_index()


@lru_cache(maxsize=1)
def train_valid():
    X, y = modelling_panel()
    dates = X.index.get_level_values("date").unique()
    cut = dates[int(len(dates) * 0.6)]
    is_train = X.index.get_level_values("date") <= cut
    return X[is_train], y[is_train], X[~is_train], y[~is_train]


@lru_cache(maxsize=1)
def score_frame() -> pd.DataFrame:
    """Model output in the shape the signal and allocator work in: date x asset."""
    X, y = modelling_panel()
    scores = (X["mom_21"] - 0.3 * X["vol_21"]).rename("score")
    return scores.unstack("asset").dropna(how="all").tail(120)

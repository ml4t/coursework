"""`model_gbm` - unit 6.3. Gradient boosting, on the same features and the same split as 6.1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..checks import require, same
from ..contracts import Contract, register
from ._shared import train_valid

SEED = 0


def reference(seed: int = SEED, max_depth: int = 3, learning_rate: float = 0.05,
              max_iter: int = 120):
    """A shallow, slow, short boosting run. Capacity is a decision the unit makes explicitly, and
    the defaults of any library are somebody else's decision about somebody else's data."""

    class GbmModel:
        def __init__(self, seed: int = seed, max_depth: int = max_depth,
                     learning_rate: float = learning_rate, max_iter: int = max_iter):
            self.seed = seed
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.model_ = None
            self.columns_ = None

        def fit(self, X: pd.DataFrame, y: pd.Series) -> "GbmModel":
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.columns_ = list(X.columns)
            self.model_ = HistGradientBoostingRegressor(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                early_stopping=False,
                random_state=self.seed,
            ).fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
            return self

        def predict(self, X: pd.DataFrame) -> pd.Series:
            if self.model_ is None:
                raise RuntimeError("fit the model on the training folds before predicting")
            values = self.model_.predict(X[self.columns_].to_numpy(dtype=float))
            return pd.Series(values, index=X.index, name="prediction")

    return GbmModel


def _probe(obj):
    X_tr, y_tr, X_va, _ = train_valid()
    return obj().fit(X_tr, y_tr).predict(X_va)


def _interface(obj) -> None:
    require(callable(obj), "interface", "a class (or factory) that makes a model",
            f"a {type(obj).__name__}")
    made = obj()
    for method in ("fit", "predict"):
        require(hasattr(made, method), "interface", f"a {method} method", "neither")
    X_tr, y_tr, X_va, _ = train_valid()
    out = obj().fit(X_tr, y_tr).predict(X_va)
    require(isinstance(out, pd.Series), "interface", "predict to return a Series",
            f"a {type(out).__name__}")
    require(out.index.equals(X_va.index), "interface",
            "one prediction per row it was asked about, on the same index as 6.1's model, because "
            "the two are compared",
            f"{len(out)} predictions against {len(X_va)} rows")


def _leakage(obj) -> str:
    X_tr, y_tr, X_va, _ = train_valid()
    fitted = obj().fit(X_tr, y_tr)
    whole = fitted.predict(X_va)
    piece = fitted.predict(X_va.iloc[:40])
    require(same(whole.iloc[:40], piece), "leakage probe",
            "the same prediction for a row whether it is scored alone or with the rest of the "
            "validation set",
            "a prediction that changed with the company it was scored in")
    return "a prediction does not depend on which other rows were scored with it"


def _fit_before_predict(obj) -> str:
    _, _, X_va, _ = train_valid()
    try:
        obj().predict(X_va)
    except Exception:
        return "predicting before fitting is refused"
    require(False, "fit required before predict", "an error when predict is called before fit",
            "a silent prediction from a model that was never trained")


def _finite(obj) -> str:
    out = _probe(obj)
    bad = int(np.isnan(out.to_numpy(dtype=float)).sum())
    require(bad == 0, "no empty predictions", "a number for every row",
            f"{bad} of {len(out)} empty")
    return f"{len(out)} predictions, none empty"


register(Contract(
    name="model_gbm",
    kind="callable",
    units=("6.3",),
    summary="A gradient-boosted model with capacity chosen rather than defaulted.",
    probe=_probe,
    interface=_interface,
    interface_detail="a class with fit(X, y) and predict(X), the same interface as model_linear",
    reference=reference,
    leakage=_leakage,
    leakage_note="a prediction does not depend on which other rows were scored with it",
    # An unseeded boosting run is caught by the determinism check every contract already carries:
    # two fits on the same rows have to agree, and an unseeded one does not.
    invariants=(("fit required before predict", _fit_before_predict),
                ("no empty predictions", _finite)),
))

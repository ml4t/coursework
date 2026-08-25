"""`model_linear` - unit 6.1. The baseline every later result is read against."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..checks import require, same
from ..contracts import Contract, register
from ._shared import train_valid

ALPHA = 1.0


def reference(alpha: float = ALPHA):
    """Ridge regression, closed form. The regularizer is the decision, not the solver: without it
    a handful of correlated features produce coefficients that swing between folds."""

    class LinearModel:
        def __init__(self, alpha: float = alpha):
            self.alpha = alpha
            self.coef_ = None
            self.intercept_ = None
            self.columns_ = None

        def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
            self.columns_ = list(X.columns)
            design = X.to_numpy(dtype=float)
            centre = design.mean(axis=0)
            target = y.to_numpy(dtype=float)
            centred = design - centre
            gram = centred.T @ centred + self.alpha * np.eye(centred.shape[1])
            self.coef_ = np.linalg.solve(gram, centred.T @ (target - target.mean()))
            self.intercept_ = float(target.mean() - centre @ self.coef_)
            return self

        def predict(self, X: pd.DataFrame) -> pd.Series:
            if self.coef_ is None:
                raise RuntimeError("fit the model on the training folds before predicting")
            values = X[self.columns_].to_numpy(dtype=float) @ self.coef_ + self.intercept_
            return pd.Series(values, index=X.index, name="prediction")

    return LinearModel


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
            "one prediction per row it was asked about",
            f"{len(out)} predictions against {len(X_va)} rows")


def _leakage(obj) -> str:
    """Catches the model that refits, rescales or otherwise learns something inside `predict`."""
    X_tr, y_tr, X_va, _ = train_valid()
    fitted = obj().fit(X_tr, y_tr)
    whole = fitted.predict(X_va)
    piece = fitted.predict(X_va.iloc[:40])
    require(same(whole.iloc[:40], piece), "leakage probe",
            "the same prediction for a row whether it is scored alone or with the rest of the "
            "validation set",
            "a prediction that changed with the company it was scored in",
            "A model that standardizes or refits inside predict has seen the whole validation "
            "set before scoring any of it.")
    return "a prediction does not depend on which other rows were scored with it"


def _fit_before_predict(obj) -> str:
    _, _, X_va, _ = train_valid()
    try:
        obj().predict(X_va)
    except Exception:
        return "predicting before fitting is refused"
    require(False, "fit required before predict",
            "an error when predict is called before fit",
            "a silent prediction from a model that was never trained")


def _regularized(obj) -> str:
    X_tr, y_tr, _, _ = train_valid()
    fitted = obj().fit(X_tr, y_tr)
    coefficients = getattr(fitted, "coef_", None)
    require(coefficients is not None, "coefficients are inspectable",
            "the fitted coefficients on the model, so the linear baseline can be read",
            "no coef_ attribute",
            "The point of a linear baseline is that you can look at what it learned.")
    require(np.all(np.isfinite(np.asarray(coefficients, dtype=float))),
            "coefficients are inspectable", "finite coefficients",
            "coefficients that are infinite or missing")
    return f"{len(np.ravel(coefficients))} finite coefficients"


def _finite(obj) -> str:
    out = _probe(obj)
    bad = int(out.isna().sum())
    require(bad == 0, "no empty predictions", "a number for every row",
            f"{bad} of {len(out)} empty")
    return f"{len(out)} predictions, none empty"


register(Contract(
    name="model_linear",
    kind="callable",
    units=("6.1",),
    summary="A regularized linear model: the baseline every later result is read against.",
    probe=_probe,
    interface=_interface,
    interface_detail="a class with fit(X, y) and predict(X)",
    reference=reference,
    leakage=_leakage,
    leakage_note="a prediction does not depend on which other rows were scored with it",
    invariants=(("fit required before predict", _fit_before_predict),
                ("coefficients are inspectable", _regularized),
                ("no empty predictions", _finite)),
))

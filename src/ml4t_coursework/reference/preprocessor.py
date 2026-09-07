"""`preprocessor` - unit 4.4. Winsorize, scale, impute, all fitted on training rows only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fixtures
from ..checks import require, same
from ..contracts import Contract, register
from .features import reference as features_reference

CLIP = 0.01


def reference(clip: float = CLIP):
    """One object, three habits, one question: what to do with values the model should not take
    at face value. Every parameter it uses is learned in `fit` and only in `fit`."""

    class Preprocessor:
        def __init__(self, clip: float = clip):
            self.clip = clip
            self.lower_ = None
            self.upper_ = None
            self.centre_ = None
            self.spread_ = None
            self.fill_ = None

        def fit(self, X: pd.DataFrame) -> "Preprocessor":
            self.lower_ = X.quantile(self.clip)
            self.upper_ = X.quantile(1 - self.clip)
            trimmed = X.clip(self.lower_, self.upper_, axis=1)
            self.centre_ = trimmed.mean()
            self.spread_ = trimmed.std().replace(0, np.nan)
            self.fill_ = trimmed.median()
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            if self.centre_ is None:
                raise RuntimeError("fit the preprocessor on the training rows before transforming")
            trimmed = X.clip(self.lower_, self.upper_, axis=1)
            filled = trimmed.fillna(self.fill_)
            return ((filled - self.centre_) / self.spread_).astype("float64")

        def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
            return self.fit(X).transform(X)

    return Preprocessor


def _split():
    X = features_reference()(fixtures.panel())
    dates = X.index.get_level_values("date").unique()
    cut = dates[int(len(dates) * 0.6)]
    train = X[X.index.get_level_values("date") <= cut]
    valid = X[X.index.get_level_values("date") > cut]
    return train, valid


def _probe(obj):
    train, valid = _split()
    return obj().fit(train).transform(valid)


def _interface(obj) -> None:
    require(callable(obj), "interface", "a class (or factory) that makes a preprocessor",
            f"a {type(obj).__name__}")
    made = obj()
    for method in ("fit", "transform"):
        require(hasattr(made, method), "interface", f"a {method} method",
                f"an object with {', '.join(m for m in dir(made) if not m.startswith('_'))[:60]}")
    train, valid = _split()
    out = obj().fit(train).transform(valid)
    require(isinstance(out, pd.DataFrame), "interface", "transform to return a DataFrame",
            f"a {type(out).__name__}")
    require(out.index.equals(valid.index), "interface",
            "one transformed row per row it was given", f"{len(out)} rows against {len(valid)}")


def _leakage(obj) -> str:
    """The defect this catches is fitting inside `transform`, which makes every parameter a
    function of the rows being transformed - including the validation rows."""
    train, valid = _split()
    fitted = obj().fit(train)
    clean = fitted.transform(valid)

    contaminated = valid.copy()
    tail = contaminated.index[len(contaminated) // 2:]
    contaminated.loc[tail] = contaminated.loc[tail] * 500.0
    again = fitted.transform(contaminated)

    head = valid.index[: len(valid) // 2]
    require(same(clean.loc[head], again.loc[head]), "leakage probe",
            "a row to transform the same way regardless of what else is in the batch with it",
            "a row whose transform changed when other rows in the batch changed",
            "That happens when the parameters are recomputed inside transform. They belong to "
            "fit, and fit sees training rows only.")
    return "a row's transform does not depend on the other rows transformed with it"


def _fit_before_transform(obj) -> str:
    _, valid = _split()
    try:
        obj().transform(valid)
    except Exception:
        return "transforming before fitting is refused"
    raise_ = require
    raise_(False, "fit required before transform",
           "an error when transform is called before fit, because there are no parameters yet",
           "a silent result",
           "Transforming with parameters that do not exist yet usually means they were made up "
           "from the rows being transformed.")


def _parameters_from_training(obj) -> str:
    train, valid = _split()
    on_train = obj().fit(train).transform(valid)
    on_everything = obj().fit(pd.concat([train, valid])).transform(valid)
    require(not same(on_train, on_everything), "parameters come from training rows",
            "a different result when the preprocessor is fitted on the validation rows too, "
            "which is the difference the fold boundary exists to make",
            "an identical result either way",
            "If it makes no difference what the preprocessor was fitted on, it is not using its "
            "fitted parameters.")
    return "fitting on the validation rows would change the answer, and it is not done"


def _finite(obj) -> str:
    out = _probe(obj)
    bad = int(out.isna().to_numpy().sum())
    require(bad == 0, "no empty values", "every transformed value to be a number",
            f"{bad} empty cells", "Imputation is the third habit this component carries.")
    return f"{len(out)} rows transformed, none empty"


register(Contract(
    name="preprocessor",
    kind="callable",
    units=("3.6",),
    summary="Winsorizes, scales and imputes, with every parameter learned on training rows.",
    probe=_probe,
    interface=_interface,
    interface_detail="a class with fit(X) and transform(X)",
    reference=reference,
    leakage=_leakage,
    leakage_note="a row's transform does not depend on the other rows transformed with it",
    invariants=(("fit required before transform", _fit_before_transform),
                ("parameters come from training rows", _parameters_from_training),
                ("no empty values", _finite)),
))

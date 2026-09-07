"""`fold_splitter`. Walk-forward folds with a label buffer."""

from __future__ import annotations

import pandas as pd

from .. import fixtures
from ..checks import require
from ..contracts import Contract, register

HORIZON = 21


def reference(n_folds: int = 5, horizon: int = HORIZON):
    """Expanding-window walk-forward: each fold trains on everything before its validation block,
    minus a buffer as long as the label horizon so no training label overlaps a validation date."""

    def fold_splitter(index: pd.Index, n_folds: int = n_folds, horizon: int = horizon):
        index = pd.Index(index).sort_values()
        block = len(index) // (n_folds + 1)
        folds = []
        for k in range(n_folds):
            val_start = block * (k + 1)
            val_stop = val_start + block
            train_stop = max(val_start - horizon, 0)
            if train_stop == 0 or val_start >= len(index):
                continue
            folds.append((index[:train_stop], index[val_start:val_stop]))
        return folds

    return fold_splitter


def _probe(obj):
    return obj(fixtures.prices().index)


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking a date index", f"a {type(obj).__name__}")
    folds = _probe(obj)
    require(isinstance(folds, list), "interface", "a list of folds", f"a {type(folds).__name__}")
    require(len(folds) > 0, "interface", "at least one fold", "an empty list")
    for k, fold in enumerate(folds):
        require(isinstance(fold, tuple) and len(fold) == 2, "interface",
                "each fold to be a (train, validation) pair", f"fold {k} is {type(fold).__name__}")
        for part, label in zip(fold, ("train", "validation"), strict=True):
            require(len(part) > 0, "interface", f"a non-empty {label} block",
                    f"fold {k}'s {label} block is empty")


def _leakage(obj) -> str:
    folds = _probe(obj)
    for k, (train, val) in enumerate(folds):
        gap = pd.Index(val).min() - pd.Index(train).max()
        require(pd.Index(train).max() < pd.Index(val).min(), "leakage probe",
                "every training date to fall before its validation block",
                f"fold {k} trains on {pd.Index(train).max().date()} and validates from "
                f"{pd.Index(val).min().date()}",
                "A fold that trains on dates inside or after its validation block is reporting a "
                "score it could not have earned in real time.")
        require(gap.days >= 1, "leakage probe", "a positive gap", f"fold {k} has {gap.days} days")
    return f"{len(folds)} folds, every training block strictly before its validation block"


def _ordered(obj) -> str:
    folds = _probe(obj)
    starts = [pd.Index(val).min() for _, val in folds]
    require(starts == sorted(starts), "folds ordered", "validation blocks in time order",
            "a fold whose validation block starts before the previous one's")
    return "validation blocks run forward in time"


def _disjoint(obj) -> str:
    folds = _probe(obj)
    for k, (train, val) in enumerate(folds):
        shared = pd.Index(train).intersection(pd.Index(val))
        require(len(shared) == 0, "folds disjoint", "no date in both train and validation",
                f"fold {k} shares {len(shared)} dates")
    for k in range(len(folds) - 1):
        shared = pd.Index(folds[k][1]).intersection(pd.Index(folds[k + 1][1]))
        require(len(shared) == 0, "folds disjoint", "validation blocks that do not overlap",
                f"folds {k} and {k + 1} share {len(shared)} dates")
    return "no date appears in both blocks of a fold, and no two validation blocks overlap"


def _buffered(obj) -> str:
    folds = _probe(obj)
    index = fixtures.prices().index
    for k, (train, val) in enumerate(folds):
        last_train = index.get_loc(pd.Index(train).max())
        first_val = index.get_loc(pd.Index(val).min())
        buffer = first_val - last_train - 1
        require(buffer >= HORIZON, "label buffer",
                f"at least {HORIZON} sessions between the last training date and the first "
                "validation date, so no training label resolves inside the validation block",
                f"fold {k} leaves {buffer}",
                "The label is a forward return over the horizon, so a training row dated within "
                "one horizon of the validation block already knows part of it.")
    return f"at least {HORIZON} sessions of buffer in every fold"


register(Contract(
    name="fold_splitter",
    kind="callable",
    summary="Splits a date index into walk-forward train/validation folds with a label buffer.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable index -> list of (train, validation) index pairs",
    reference=reference,
    leakage=_leakage,
    leakage_note="every training date falls strictly before its own validation block",
    invariants=(("folds ordered", _ordered),
                ("folds disjoint", _disjoint),
                ("label buffer", _buffered)),
))

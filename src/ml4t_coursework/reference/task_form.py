"""`task_form` - unit 4.3. The form the model predicts in.

The course fixes regression, and the argument is that the form is fixed by what the allocation
rule consumes rather than by which form a model predicts best. So the contract does not require
regression: it requires that whatever form is chosen keeps the cross-sectional order the
allocation rule reads, and that it is computed inside the date rather than against pooled history.
"""

from __future__ import annotations

import pandas as pd

from .. import fixtures
from ..checks import require, same
from ..contracts import Contract, register
from .labeler import reference as labeler_reference


def reference():
    """Regression on the label as it stands. Nothing is discretized, because the decile rule the
    course allocates with reads order, and order is already there."""

    def task_form(label: pd.Series) -> pd.Series:
        return label.astype("float64").rename("target")

    return task_form


def _label() -> pd.Series:
    return labeler_reference()(fixtures.prices())


def _probe(obj):
    return obj(_label())


def _interface(obj) -> None:
    require(callable(obj), "interface", "a callable taking the label Series",
            f"a {type(obj).__name__}")
    out = _probe(obj)
    require(isinstance(out, pd.Series), "interface", "a Series of model targets",
            f"a {type(out).__name__}")
    label = _label()
    require(out.index.equals(label.index), "interface",
            "one target per label, on the same (date, symbol) index",
            f"{len(out)} targets against {len(label)} labels")


def _leakage(obj) -> str:
    label = _label()
    cut = label.index.get_level_values("date").unique()[200]
    full = obj(label)
    truncated = obj(label[label.index.get_level_values("date") <= cut])
    a = full[full.index.get_level_values("date") <= cut]
    require(same(a.sort_index(), truncated.sort_index()), "leakage probe",
            "a target at date t to be the same whether or not later labels exist",
            "a target that moves when later labels arrive",
            "Cutting into buckets against the pooled history of the whole sample decides today's "
            "bucket with next year's distribution. Discretize inside the cross-section.")
    return "a target at date t is unchanged by labels after t"


def _order_preserving(obj) -> str:
    label = _label()
    out = obj(label)
    frame = pd.DataFrame({"label": label, "target": out}).dropna()
    worst = 1.0
    for _, block in frame.groupby(level="date"):
        if len(block) < 4 or block["target"].nunique() < 2:
            continue
        rho = block["label"].rank().corr(block["target"].rank())
        worst = min(worst, float(rho))
    require(worst > 0.99, "order preserved within the date",
            "a target that ranks the cross-section the way the label does, because the allocation "
            "rule reads order",
            f"a date where the two orders agree only {worst:.3f}",
            "A form that reorders the cross-section changes which names are bought, which is a "
            "different strategy rather than a different way of writing the same one.")
    return "the cross-sectional order of the label survives the transform"


def _finite(obj) -> str:
    out = _probe(obj)
    bad = int(out.isna().sum())
    require(bad == 0, "no empty targets", "a target for every label", f"{bad} empty")
    return f"{len(out)} targets, none empty"


register(Contract(
    name="task_form",
    kind="callable",
    units=("3.5",),
    summary="Puts the label in the form the model predicts, keeping the order the allocator reads.",
    probe=_probe,
    interface=_interface,
    interface_detail="a callable label -> Series of targets on the same index",
    reference=reference,
    leakage=_leakage,
    leakage_note="a target at date t is unchanged by labels after t",
    invariants=(("order preserved within the date", _order_preserving),
                ("no empty targets", _finite)),
))

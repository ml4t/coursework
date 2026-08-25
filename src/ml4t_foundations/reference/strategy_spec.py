"""`strategy_spec` - unit 1.1. The family you are in, and the mechanism you are claiming."""

from __future__ import annotations

from ..contracts import Contract, register
from ._config import interface_for, one_of, probe, stated

FAMILIES = ("cross-sectional", "time-series", "event-driven", "relative-value")
FIELDS = {
    "family": f"which of {', '.join(FAMILIES)} the strategy belongs to",
    "mechanism": "why the return exists: who is on the other side, and why they keep taking it",
    "direction": "long-only, long-short, or short-only",
}


def reference():
    return {
        "family": "cross-sectional",
        "mechanism": ("Investors under-react to sustained relative strength across funds, and "
                      "flow-driven buyers arrive after the move rather than before it, so the "
                      "spread between strong and weak names persists for weeks rather than days."),
        "direction": "long-short",
    }


register(Contract(
    name="strategy_spec",
    kind="config",
    units=("1.1",),
    summary="The strategy family and the economic mechanism the return is claimed to come from.",
    probe=probe,
    interface=interface_for(FIELDS),
    interface_detail=", ".join(FIELDS),
    reference=reference,
    leakage=None,
    leakage_note=("not applicable: this is a claim you write down, and its check is whether it was "
                  "written before the results rather than after"),
    invariants=(("family is one the course recognizes", one_of("family", FAMILIES)),
                ("direction is one the course recognizes",
                 one_of("direction", ("long-only", "long-short", "short-only"))),
                ("mechanism is actually stated", stated("mechanism", 60))),
))

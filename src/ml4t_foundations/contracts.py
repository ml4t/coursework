"""What a component is, and the registry of the twenty the course asks for.

One `Contract` per component, declared beside its reference implementation in `reference/`. The
contract is authored *from* the reference, which is what guarantees it is passable, and every
reference is tested against its own contract plus a deliberately broken variant that must fail it.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

CONTRACTS: dict[str, "Contract"] = {}


@dataclass
class Contract:
    name: str
    kind: str
    units: tuple[str, ...]
    summary: str
    probe: Callable[[Any], Any]
    interface: Callable[[Any], None]
    reference: Callable[[], Any]
    interface_detail: str = ""
    invariants: tuple[tuple[str, Callable[[Any], str | None]], ...] = ()
    leakage: Callable[[Any], str | None] | None = None
    leakage_note: str = "not applicable to this component, and here is why"
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"callable", "config"}:
            raise ValueError(f"{self.name}: kind must be 'callable' or 'config', got {self.kind!r}")

    def describe(self) -> str:
        lines = [f"{self.name} ({self.kind}, written in unit {' and '.join(self.units)})",
                 f"  {self.summary}",
                 f"  interface: {self.interface_detail}",
                 f"  leakage probe: {self.leakage_note}"]
        for label, _ in self.invariants:
            lines.append(f"  invariant: {label}")
        return "\n".join(lines)


def register(contract: Contract) -> Contract:
    if contract.name in CONTRACTS:
        raise ValueError(f"{contract.name} is already registered")
    CONTRACTS[contract.name] = contract
    return contract


def load_all() -> dict[str, Contract]:
    """Import every module under `reference/`, each of which registers one contract."""
    from . import reference

    if not CONTRACTS:
        for module in pkgutil.iter_modules(reference.__path__):
            if not module.name.startswith("_"):
                importlib.import_module(f"{reference.__name__}.{module.name}")
    return CONTRACTS


def get(name: str) -> Contract:
    contracts = load_all()
    if name not in contracts:
        known = ", ".join(sorted(contracts))
        raise KeyError(
            f"There is no component called {name!r} in this course.\n"
            f"  The ones there are: {known}\n"
            f"  Check the spelling against the name the unit's notebook asked you to save."
        )
    return contracts[name]


def catalog() -> str:
    return "\n\n".join(c.describe() for _, c in sorted(load_all().items()))

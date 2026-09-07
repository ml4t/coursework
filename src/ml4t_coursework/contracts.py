"""What a component is, and the registry of the twenty the courses ask for.

A contract says what a component must do. It deliberately does not say which unit writes it: a
unit number belongs to one course's outline, and more than one course installs this package. A
Research to Production student is handed all twenty and never sat the Foundations unit that
writes the baseline, so a number here would appear in their own submission report naming a
lesson they never took. Each course keeps that mapping beside its own units, where it cannot
drift out of step with them - and it did drift, in both directions, while it lived here.

One `Contract` per component, declared beside its reference implementation in `reference/`. The
contract is authored *from* the reference, which is what guarantees it is passable, and every
reference is tested against its own contract plus a deliberately broken variant that must fail it.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

CONTRACTS: dict[str, "Contract"] = {}

_SOURCES: list[str] = [f"{__package__}.reference"]
_LOADED: set[str] = set()


@dataclass
class Contract:
    name: str
    kind: str
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
        lines = [f"{self.name} ({self.kind})",
                 f"  {self.summary}",
                 f"  interface: {self.interface_detail}",
                 f"  leakage probe: {self.leakage_note}"]
        for label, _ in self.invariants:
            lines.append(f"  invariant: {label}")
        return "\n".join(lines)


def register(contract: Contract) -> Contract:
    existing = CONTRACTS.get(contract.name)
    if existing is not None and existing is not contract:
        raise ValueError(
            f"Two different contracts are both called {contract.name!r}.\n"
            f"  Already registered from: {existing.reference.__module__}\n"
            f"  Now registering from:    {contract.reference.__module__}\n"
            f"  Component names are shared across every registered source, so pick another."
        )
    CONTRACTS[contract.name] = contract
    return contract


def add_source(package: str | ModuleType) -> None:
    """Add a package of contract modules to discovery. A course outside this one calls this."""
    name = package if isinstance(package, str) else package.__name__
    if name not in _SOURCES:
        _SOURCES.append(name)


def load_all() -> dict[str, Contract]:
    """Import every module in every registered source package, each of which registers a contract.

    Tracked per source rather than by whether the registry is empty: a caller that pre-registers
    one contract of its own must not suppress the import of every other source.
    """
    for source in list(_SOURCES):
        if source in _LOADED:
            continue
        _LOADED.add(source)
        package = importlib.import_module(source)
        for module in pkgutil.iter_modules(package.__path__):
            if not module.name.startswith("_"):
                importlib.import_module(f"{package.__name__}.{module.name}")
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

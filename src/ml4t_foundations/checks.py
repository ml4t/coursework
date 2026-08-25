"""The five checks every component contract carries, and the failure they raise.

`certification.md` § 2 is the specification. The checks assert properties, never equality with a
reference output: two correct fold splitters legitimately differ and an equality check would fail
correct work and teach copying. The reference delta is the one exception and it is reported, not
enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


class Failure(Exception):
    """A contract check that did not pass.

    Every failure names the expectation and the observed value, because "expected 12 folds, got 9"
    is feedback and a bare AssertionError is not.
    """

    def __init__(self, check: str, expected: str, observed: str, hint: str = "") -> None:
        self.check = check
        self.expected = expected
        self.observed = observed
        self.hint = hint
        message = f"{check}: expected {expected}, got {observed}"
        if hint:
            message = f"{message}\n  {hint}"
        super().__init__(message)


def require(condition: bool, check: str, expected: str, observed: str, hint: str = "") -> None:
    if not condition:
        raise Failure(check, expected, observed, hint)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def line(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        return f"  [{mark}] {self.name}{': ' + self.detail if self.detail else ''}"


@dataclass
class Conformance:
    component: str
    conformant: bool
    checks: list[CheckResult] = field(default_factory=list)
    delta: str = ""
    stamped_at: str = ""
    helper_version: str = ""
    unit: str = ""

    def __str__(self) -> str:
        head = f"{self.component}: {'conformant' if self.conformant else 'NOT conformant'}"
        body = "\n".join(c.line() for c in self.checks)
        tail = f"\n  [note] reference delta: {self.delta}" if self.delta else ""
        return f"{head}\n{body}{tail}"

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "unit": self.unit,
            "conformant": self.conformant,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
            "reference_delta": self.delta,
            "stamped_at": self.stamped_at,
            "helper_version": self.helper_version,
        }


# --- comparison helpers, shared by determinism, the leakage probe and the delta ---------------


def same(left: Any, right: Any, tol: float = 1e-12) -> bool:
    """Value equality that understands the shapes components actually return."""
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        if left.shape != right.shape or not left.index.equals(right.index):
            return False
        if list(left.columns) != list(right.columns):
            return False
        return bool(np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                                atol=tol, rtol=0, equal_nan=True))
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        if not left.index.equals(right.index):
            return False
        return bool(np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                                atol=tol, rtol=0, equal_nan=True))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same(a, b, tol) for a, b in zip(left, right))
    if isinstance(left, pd.Index) and isinstance(right, pd.Index):
        return left.equals(right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.shape == right.shape and bool(np.allclose(left, right, atol=tol, equal_nan=True))
    return bool(left == right)


def overlap(left: Any, right: Any) -> tuple[Any, Any]:
    """Restrict two indexed objects to the rows they share, in the same order."""
    if isinstance(left, (pd.Series, pd.DataFrame)) and isinstance(right, (pd.Series, pd.DataFrame)):
        shared = left.index.intersection(right.index)
        return left.loc[shared], right.loc[shared]
    return left, right


def describe_delta(student: Any, reference: Any) -> str:
    """One line on how far the student's output sits from the shipped reference's.

    Reported and never enforced. Where a design choice legitimately differs the delta is
    information; only the properties gate.
    """
    try:
        a, b = overlap(student, reference)
        if isinstance(a, (pd.Series, pd.DataFrame)) and isinstance(b, (pd.Series, pd.DataFrame)):
            if len(a) == 0:
                return "no overlapping rows, so nothing to compare"
            if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
                cols = a.columns.intersection(b.columns)
                if len(cols) == 0:
                    return "no overlapping columns, so nothing to compare"
                a, b = a[cols], b[cols]
            diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
            scale = np.nanmean(np.abs(np.asarray(b, dtype=float)))
            mean = np.nanmean(diff)
            rel = f", about {mean / scale:.1%} of the reference's own scale" if scale else ""
            return f"mean absolute difference {mean:.6g} over {len(a)} rows{rel}"
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return f"{len(a)} against the reference's {len(b)}"
        if isinstance(a, dict) and isinstance(b, dict):
            differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            return "identical to the reference" if not differing else f"differs on {', '.join(differing)}"
        return "identical to the reference" if same(a, b) else f"{a!r} against the reference's {b!r}"
    except Exception as exc:  # a delta is information; it must never break a save
        return f"not computable ({type(exc).__name__})"


# --- the four gating checks --------------------------------------------------------------------


def run_interface(contract, obj) -> CheckResult:
    contract.interface(obj)
    return CheckResult("interface", True, contract.interface_detail)


def run_determinism(contract, obj) -> CheckResult:
    first = contract.probe(obj)
    second = contract.probe(obj)
    require(
        same(first, second),
        "determinism",
        "the same output from the same input twice",
        "two different outputs",
        "Something in the component is drawing on state that changes between calls - an unseeded "
        "random draw, a mutable default, or a value read from the clock.",
    )
    return CheckResult("determinism", True, "same input twice, same output")


def run_leakage(contract, obj) -> CheckResult:
    if contract.leakage is None:
        return CheckResult("leakage probe", True, contract.leakage_note)
    detail = contract.leakage(obj)
    return CheckResult("leakage probe", True, detail or contract.leakage_note)


def run_invariants(contract, obj) -> list[CheckResult]:
    results = []
    for label, fn in contract.invariants:
        detail = fn(obj)
        results.append(CheckResult(label, True, detail or ""))
    return results


def evaluate(contract, obj) -> Conformance:
    """Run all five checks, stopping the gating ones at the first failure."""
    result = Conformance(component=contract.name, unit=contract.units[0], conformant=True)
    stages: list[Callable[[], Any]] = [
        lambda: [run_interface(contract, obj)],
        lambda: [run_leakage(contract, obj)],
        lambda: [run_determinism(contract, obj)],
        lambda: run_invariants(contract, obj),
    ]
    names = ["interface", "leakage probe", "determinism", "domain invariants"]
    for name, stage in zip(names, stages):
        try:
            result.checks.extend(stage())
        except Failure as failure:
            result.checks.append(CheckResult(failure.check, False, str(failure).split(": ", 1)[1]))
            result.conformant = False
            break
        except NameError as exc:
            missing = str(exc).split("'")[1] if "'" in str(exc) else "something"
            result.checks.append(CheckResult(name, False, (
                f"the component calls {missing!r}, which is not part of it. "
                f"Colab does not keep your session, so a later notebook loads this file on its own "
                f"and {missing!r} will not be there. Either move it inside the component, or pass "
                f"it along: save_component(..., also=[{missing}]) for a function, "
                f"include={{{missing!r}: ...}} for a value."
            )))
            result.conformant = False
            break
        except Exception as exc:
            result.checks.append(
                CheckResult(name, False, f"the check could not run: {type(exc).__name__}: {exc}")
            )
            result.conformant = False
            break
    if result.conformant:
        try:
            result.delta = describe_delta(contract.probe(obj), contract.probe(contract.reference()))
        except Exception as exc:
            result.delta = f"not computable ({type(exc).__name__})"
    return result

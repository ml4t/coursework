"""Shared shape for the four components that record a choice rather than compute anything.

A recorded choice is still a component: `certification.md` asks whether the student *made* the
decision, and a decision fixed before the result is seen is exactly what 1.1, 1.3, 7.2 and 8.3
are about. What it cannot have is a leakage probe, so each says why in its own words.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..checks import require


def probe(obj: dict) -> dict:
    return dict(obj)


def interface_for(fields: dict[str, str]) -> Callable[[Any], None]:
    def interface(obj: Any) -> None:
        require(isinstance(obj, dict), "interface",
                f"a dictionary with the keys {', '.join(fields)}", f"a {type(obj).__name__}")
        missing = [k for k in fields if k not in obj]
        require(not missing, "interface",
                f"the keys {', '.join(fields)}",
                f"a record missing {', '.join(missing)}",
                "  " + "\n  ".join(f"{k}: {v}" for k, v in fields.items()))
    return interface


def one_of(field: str, allowed: tuple[str, ...]) -> Callable[[Any], str]:
    def check(obj: dict) -> str:
        value = obj.get(field)
        require(value in allowed, f"{field} is one the course recognizes",
                f"one of {', '.join(allowed)}", repr(value),
                "A value outside that set is a decision the pipeline downstream cannot act on.")
        return f"{field} = {value}"
    return check


def stated(field: str, minimum: int = 20) -> Callable[[Any], str]:
    def check(obj: dict) -> str:
        value = str(obj.get(field, "")).strip()
        require(len(value) >= minimum, f"{field} is actually stated",
                f"at least {minimum} characters saying what you mean",
                f"{len(value)} characters" + (f": {value!r}" if value else ""),
                "Writing it down is the decision. A placeholder here is a decision not made, and "
                "the point of fixing it now is that you cannot fix it later once you have seen a "
                "result.")
        return f"{field} stated in {len(value)} characters"
    return check

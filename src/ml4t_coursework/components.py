"""`save_component` and `load_component`: how a component is checked, stamped and recovered.

What persists between units is the student's *code*, not intermediate data. A notebook defines its
component from scratch and writes it here; a later pipeline notebook loads it, or falls back to the
shipped reference and says so. Restart-and-run-all works on any unit, in any order, on a cold
session, which is the property this module exists to protect.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
import textwrap
from typing import Any, Callable, Sequence

from . import contracts, project
from .checks import Conformance, evaluate

__version_marker__ = "components"

_PREAMBLE = (
    "# Saved by ml4t-coursework. This is your own code, exactly as you wrote it.\n"
    "import numpy as np\n"
    "import pandas as pd\n\n"
)


def _version() -> str:
    from . import __version__

    return __version__


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _source_of(obj: Any, name: str) -> str:
    try:
        return textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"Could not read the source of the object you passed for {name!r}.\n"
            f"  Define it as a function or a class in a notebook cell and pass the object itself, "
            f"not a lambda, a partial or an instance.\n"
            f"  ({type(exc).__name__}: {exc})"
        ) from exc


def _rebuild(source: str, symbol: str, name: str) -> Any:
    """Execute the saved source in a fresh namespace and return the object.

    This is deliberately done before the checks run. What gets validated has to be what a later
    notebook will actually load, and a component that only works because of something else in the
    student's session would pass in place and fail on a cold kernel.
    """
    namespace: dict[str, Any] = {}
    import numpy as np
    import pandas as pd

    namespace.update({"np": np, "pd": pd, "numpy": np, "pandas": pd})
    try:
        exec(compile(_PREAMBLE + source, f"<{name}>", "exec"), namespace)
    except NameError as exc:
        missing = str(exc).split("'")[1] if "'" in str(exc) else "something"
        raise ValueError(
            f"Your {name} refers to {missing!r}, which is not part of it.\n"
            f"  Colab does not keep your session, so a later notebook loads this file on its own "
            f"and {missing!r} will not be there.\n"
            f"  Either move it inside the component, or pass it along: "
            f"save_component({name!r}, {symbol}, also=[{missing}])"
        ) from exc
    if symbol not in namespace:
        raise ValueError(
            f"The saved source does not define {symbol!r}.\n"
            f"  Name the function or class you pass, and pass it by that name."
        )
    return namespace[symbol]


def save_component(
    name: str,
    obj: Any,
    also: Sequence[Callable] = (),
    include: dict[str, Any] | None = None,
    quiet: bool = False,
) -> Conformance:
    """Check a component against its contract, stamp the verdict, and write it to the project.

    `also` carries any function you defined in another cell that the component calls; `include`
    carries any plain value it reads, as `{name: value}`. A component has to stand on its own,
    because a later notebook loads this file on a cold session with nothing else in scope.
    """
    contract = contracts.get(name)
    folder = project.components_dir()

    if contract.kind == "config":
        if not isinstance(obj, dict):
            raise ValueError(
                f"{name} is a recorded choice, not a function.\n"
                f"  Pass a dictionary. Expected keys: {contract.interface_detail}\n"
                f"  You passed a {type(obj).__name__}."
            )
        payload = dict(obj)
        checked: Any = payload
        body = json.dumps(payload, indent=2, default=str)
        target = folder / f"{name}.json"
        symbol = name
    else:
        symbol = getattr(obj, "__name__", None)
        if not symbol:
            raise ValueError(
                f"{name}: pass the function or class itself, by name, not an instance or a lambda."
            )
        parts = [f"{key} = {value!r}" for key, value in (include or {}).items()]
        parts += [_source_of(helper, name) for helper in also]
        parts.append(_source_of(obj, name))
        body = "\n\n".join(parts)
        checked = _rebuild(body, symbol, name)
        target = folder / f"{name}.py"

    result = evaluate(contract, checked)
    result.stamped_at = _now()
    result.helper_version = _version()

    target.write_text(_PREAMBLE + body if contract.kind == "callable" else body)
    meta = result.to_dict() | {"symbol": symbol, "file": target.name, "kind": contract.kind}
    (folder / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))

    if not quiet:
        print(result)
        if result.conformant:
            print(f"  saved to {target}")
        else:
            print(f"  saved to {target}, and it is recorded as not conformant.")
            print("  Nothing is blocked: fix it whenever you like, and any pipeline notebook you "
                  "run meanwhile uses the shipped reference for this step.")
    return result


def _meta(name: str) -> dict | None:
    path = project.components_dir(create=False) / f"{name}.meta.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def source_of(name: str) -> str:
    """`'yours'` or `'reference'` - what `load_component` would return right now."""
    meta = _meta(name)
    return "yours" if meta and meta.get("conformant") else "reference"


def load_component(name: str, quiet: bool = False) -> Any:
    """Return the student's component when it is conformant, the shipped reference otherwise.

    It always says which. This is what stops a wrong Part 4 from blocking Part 7.
    """
    contract = contracts.get(name)
    meta = _meta(name)
    if meta and meta.get("conformant"):
        folder = project.components_dir(create=False)
        path = folder / meta["file"]
        try:
            if contract.kind == "config":
                obj = json.loads(path.read_text())
            else:
                obj = _rebuild(path.read_text(), meta["symbol"], name)
            if not quiet:
                stamped = meta.get("stamped_at", "")[:10]
                print(f"{name}: using your implementation (checked {stamped})")
            return obj
        except Exception as exc:
            if not quiet:
                print(f"{name}: your saved version would not load ({type(exc).__name__}), so this "
                      f"run uses the shipped reference.")
    elif not quiet:
        reason = "you have not written it yet" if meta is None else "yours is recorded as not conformant"
        print(f"{name}: using the shipped reference, because {reason}.")
    return contract.reference()


def status() -> str:
    """Every component the course asks for, and where each one stands."""
    lines = []
    for name, contract in sorted(contracts.load_all().items()):
        meta = _meta(name)
        if meta is None:
            state = "not written yet"
        elif meta.get("conformant"):
            state = f"conformant, {meta.get('stamped_at', '')[:10]}"
        else:
            failed = [c["name"] for c in meta.get("checks", []) if not c["passed"]]
            state = f"NOT conformant ({', '.join(failed) or 'unknown'})"
        lines.append(f"  {name:<20} unit {contract.units[0]:<5} {state}")
    return "components in your project folder:\n" + "\n".join(lines)


def catalog() -> str:
    return contracts.catalog()

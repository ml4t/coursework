"""The submission report the final unit emits.

It is the student's own file: they can read every line of it, and submitting it is one upload
rather than a review request. Nothing in it is graded by a person.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from . import contracts, project, results
from .components import _meta
from .course import active_course


def report(answers: dict[str, str] | None = None, quiet: bool = False) -> dict:
    """Write and return the submission report: conformance, the three runs, the written answers."""
    from . import __version__

    course = active_course()
    terminal_stages = course.terminal_stages
    answers = answers or {}
    known = contracts.load_all()
    stamps = {}
    for name in sorted(known):
        meta = _meta(name)
        stamps[name] = {
            "unit": known[name].units[0],
            "written": meta is not None,
            "conformant": bool(meta and meta.get("conformant")),
            "stamped_at": (meta or {}).get("stamped_at"),
        }

    runs = {stage: None for stage in terminal_stages}
    for stage in terminal_stages:
        row = results.latest(stage)
        if row is not None:
            runs[stage] = {k: (None if row[k] != row[k] else row[k]) for k in row.index}

    payload = {
        "course": course.title,
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "helper_version": __version__,
        "components": stamps,
        "components_conformant": sum(1 for s in stamps.values() if s["conformant"]),
        "components_required": len(stamps),
        "runs": runs,
        "runs_present": [s for s, r in runs.items() if r is not None],
        "written_answers": {k: v.strip() for k, v in answers.items()},
    }
    payload["complete"] = (
        payload["components_conformant"] == payload["components_required"]
        and len(payload["runs_present"]) == len(terminal_stages)
        and len([v for v in payload["written_answers"].values() if len(v) >= 40]) >= 2
    )

    path: Path = project.home() / "submission_report.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    if not quiet:
        print(f"submission report written to {path}")
        print(f"  components conformant: {payload['components_conformant']} of "
              f"{payload['components_required']}")
        print(f"  runs present: {', '.join(payload['runs_present']) or 'none'}")
        missing = [s for s in terminal_stages if runs[s] is None]
        if missing:
            print(f"  still needed: a run at stage {', '.join(missing)}")
        if not payload["complete"]:
            print("  not complete yet, and the file says exactly what is outstanding.")
    return payload

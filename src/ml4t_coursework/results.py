"""One appended row per pipeline run.

The columns are shared by every course; which stage names are legal is the course's own, so a
results row is validated against the course this session selected.

The terminal unit reads three runs against each other - the non-ML baseline, the crude Part 0
pipeline, and the finished one - and they happen weeks apart. What persists between them is their
numbers, so the columns are fixed here rather than per notebook.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from . import project
from .course import active_course

COLUMNS = ("run_id", "stage", "written_at", "dataset_vintage", "start", "end", "total_return",
           "annual_return", "annual_vol", "sharpe", "max_drawdown", "turnover", "cost_bps",
           "components_used", "helper_version")

REQUIRED = ("stage", "start", "end", "total_return", "sharpe")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def append_result(row: dict[str, Any], quiet: bool = False) -> pd.DataFrame:
    """Append one run to the results log and return the log."""
    from . import __version__

    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise ValueError(
            f"A results row needs {', '.join(REQUIRED)}.\n"
            f"  Missing: {', '.join(missing)}.\n"
            f"  The final unit reads three of these rows against each other and cannot do it "
            f"without them."
        )
    stages = active_course().stages
    if row["stage"] not in stages:
        raise ValueError(
            f"stage must be one of {', '.join(stages)}, got {row['stage']!r}.\n"
            f"  The unit's notebook tells you which one this run is."
        )
    unknown = [k for k in row if k not in COLUMNS]
    if unknown:
        raise ValueError(
            f"A results row has fixed columns, and {', '.join(unknown)} is not one of them.\n"
            f"  The columns are: {', '.join(COLUMNS)}."
        )

    record = {c: row.get(c) for c in COLUMNS}
    record["written_at"] = _now()
    record["helper_version"] = __version__
    if not record["run_id"]:
        record["run_id"] = f"{record['stage']}-{record['written_at'][:19].replace(':', '')}"

    path = project.results_file()
    frame = pd.DataFrame([record], columns=list(COLUMNS))
    frame.to_csv(path, mode="a", header=not path.is_file() or path.stat().st_size == 0, index=False)
    if not quiet:
        print(f"recorded run {record['run_id']} ({record['stage']}) in {path}")
    return results()


def results() -> pd.DataFrame:
    path = project.results_file(create=False)
    if not path.is_file():
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.read_csv(path)


def latest(stage: str) -> pd.Series | None:
    """The most recent run of one stage, which is what the terminal comparison reads."""
    frame = results()
    rows = frame[frame["stage"] == stage]
    return None if rows.empty else rows.iloc[-1]

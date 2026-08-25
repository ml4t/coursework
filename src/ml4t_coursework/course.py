"""Which course a notebook belongs to, and the few facts that differ between them.

Almost everything here is shared: the checks, the contract machinery, the results columns, the
report. Three things are not, and they are the three a package named after one course would have
gotten wrong - the project folder on the student's Drive, the environment override that points at
it, and the list of stage names a results row may carry. They live on a `Course` and a notebook
selects one in its opening cell.

There is deliberately no default. A notebook that never says which course it belongs to would
otherwise write its components into another course's folder, and the failure would surface weeks
later as a component that loads but does the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass

COURSES: dict[str, "Course"] = {}
_active: str | None = None


@dataclass(frozen=True)
class Course:
    key: str
    title: str
    folder: str
    env_var: str
    stages: tuple[str, ...]
    terminal_stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = [s for s in self.terminal_stages if s not in self.stages]
        if unknown:
            raise ValueError(
                f"{self.key}: terminal stages must also be stages, and "
                f"{', '.join(unknown)} is not."
            )


def register_course(course: Course, replace: bool = False) -> Course:
    existing = COURSES.get(course.key)
    if existing is not None and existing != course and not replace:
        raise ValueError(
            f"A different course is already registered as {course.key!r}.\n"
            f"  Registered: {existing}\n"
            f"  Pass replace=True if you meant to change it."
        )
    COURSES[course.key] = course
    return course


def use(key: str) -> Course:
    """Select the course this session belongs to. Every notebook calls this in its opening cell."""
    global _active
    if key not in COURSES:
        known = ", ".join(sorted(COURSES)) or "none yet"
        raise KeyError(
            f"There is no course registered as {key!r}.\n"
            f"  Registered: {known}\n"
            f"  A course registers itself with ml4t_coursework.register_course()."
        )
    _active = key
    return COURSES[key]


def active_course() -> Course:
    if _active is None:
        known = ", ".join(repr(k) for k in sorted(COURSES)) or "none yet"
        raise RuntimeError(
            "This session has not said which course it belongs to, so there is no project "
            "folder to read or write.\n"
            f"  Add ml4t_coursework.use(...) to the notebook's opening cell, one of: {known}.\n"
            "  The setup notebook does this and so does every unit notebook."
        )
    return COURSES[_active]


register_course(Course(
    key="foundations",
    title="ML4T: Foundations",
    folder="ml4t-foundations",
    env_var="ML4T_FOUNDATIONS_HOME",
    stages=("baseline", "v0", "part_02", "part_04", "part_05", "part_06", "part_07", "part_08",
            "final"),
    terminal_stages=("baseline", "v0", "final"),
))

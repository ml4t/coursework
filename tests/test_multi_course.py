"""What has to hold once a second course installs the same package.

Every failure guarded here was live before the package was renamed: a course-shaped constant
where a per-course fact belonged, and a discovery short-circuit that a caller outside this package
could trip without knowing it had.
"""

from __future__ import annotations

import sys

import pytest

from ml4t_coursework import Course, append_result, contracts, course, project, register_course, use


@pytest.fixture
def second_course(monkeypatch, tmp_path):
    monkeypatch.setattr(course, "COURSES", dict(course.COURSES))
    other = register_course(Course(
        key="other-course", title="Another Course", folder="ml4t-other",
        env_var="ML4T_OTHER_HOME", stages=("first", "last"), terminal_stages=("last",)))
    monkeypatch.setenv("ML4T_OTHER_HOME", str(tmp_path / "other"))
    return other


def test_each_course_gets_its_own_project_folder(second_course, monkeypatch):
    foundations = project.home()
    use("other-course")
    assert project.home() != foundations
    assert project.home().name == "other"


def test_a_stage_belongs_to_a_course_not_to_the_package(second_course):
    """`v0` is a Foundations stage. The other course's rows must not be measured against it."""
    use("other-course")
    with pytest.raises(ValueError, match="first, last"):
        append_result({"stage": "v0", "start": "2020-01-01", "end": "2020-12-31",
                       "total_return": 0.1, "sharpe": 0.5})
    frame = append_result({"stage": "first", "start": "2020-01-01", "end": "2020-12-31",
                           "total_return": 0.1, "sharpe": 0.5}, quiet=True)
    assert frame["stage"].tolist() == ["first"]


def test_a_session_with_no_course_selected_says_so(monkeypatch):
    monkeypatch.setattr(course, "_active", None)
    with pytest.raises(RuntimeError, match="which course it belongs to"):
        project.home()


def test_selecting_a_course_nobody_registered_lists_the_ones_there_are():
    with pytest.raises(KeyError, match="foundations"):
        use("no-such-course")


def test_a_terminal_stage_has_to_be_a_stage():
    with pytest.raises(ValueError, match="terminal stages must also be stages"):
        Course(key="k", title="t", folder="f", env_var="E", stages=("a",),
               terminal_stages=("b",))


def _fresh_registry(monkeypatch):
    """Reset discovery to the state of a process that has not imported a contract module yet."""
    for name in [m for m in sys.modules if m.startswith("ml4t_coursework.reference.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(contracts, "CONTRACTS", {})
    monkeypatch.setattr(contracts, "_LOADED", set())
    monkeypatch.setattr(contracts, "_SOURCES", list(contracts._SOURCES))


def test_registering_a_contract_first_does_not_suppress_the_reference_imports(monkeypatch):
    """The bug this replaces: `if not CONTRACTS` meant one early registration from any caller
    silently skipped all twenty reference modules, and every later lookup then reported the
    component did not exist."""
    _fresh_registry(monkeypatch)
    contracts.CONTRACTS["planted"] = object()

    known = contracts.load_all()
    assert len(known) > 20
    assert "fold_splitter" in known


def test_discovery_reaches_a_package_outside_this_one(monkeypatch, tmp_path):
    """A second course declares its components in its own package and registers it as a source."""
    _fresh_registry(monkeypatch)
    package = tmp_path / "other_course_components"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "their_component.py").write_text(
        "from ml4t_coursework.contracts import Contract, register\n"
        "def reference():\n"
        "    return lambda x: x\n"
        "register(Contract(name='their_component', kind='callable', units=('1.1',),\n"
        "                  summary='theirs', probe=lambda obj: obj('x'),\n"
        "                  interface=lambda obj: None, reference=reference))\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    contracts.add_source("other_course_components")

    known = contracts.load_all()
    assert "their_component" in known, "a source outside this package was never imported"
    assert "fold_splitter" in known, "adding a source must not displace this course's own"
    assert contracts.get("their_component").summary == "theirs"


def test_two_different_contracts_with_one_name_name_both_sources(monkeypatch):
    _fresh_registry(monkeypatch)
    existing = contracts.load_all()["fold_splitter"]

    def _reference():
        return None

    _reference.__module__ = "somewhere_else"
    with pytest.raises(ValueError, match="somewhere_else"):
        contracts.register(contracts.Contract(
            name="fold_splitter", kind="callable", units=("1.1",), summary="clash",
            probe=existing.probe, interface=existing.interface, reference=_reference))

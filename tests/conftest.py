import tempfile

import pytest

from ml4t_coursework import course


@pytest.fixture(autouse=True)
def isolated_project(monkeypatch):
    """Every test gets its own project folder and a selected course, so nothing leaks between."""
    monkeypatch.setattr(course, "_active", "foundations")
    monkeypatch.setenv(course.COURSES["foundations"].env_var,
                       tempfile.mkdtemp(prefix="ml4t-test-"))
    yield

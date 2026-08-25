import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_project(monkeypatch):
    """Every test gets its own project folder, so nothing leaks between them."""
    monkeypatch.setenv("ML4T_FOUNDATIONS_HOME", tempfile.mkdtemp(prefix="ml4t-test-"))
    yield

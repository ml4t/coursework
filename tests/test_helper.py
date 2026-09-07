"""The round trip a student actually makes: write a component, save it, load it in a later unit."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ml4t_coursework import (
    append_result,
    contract,
    load_component,
    project,
    report,
    results,
    save_component,
    source_of,
    status,
)
from ml4t_coursework.components import _meta


def _good_splitter():
    def fold_splitter(index, n_folds=4):
        index = pd.Index(index).sort_values()
        block = len(index) // (n_folds + 1)
        return [(index[: block * (k + 1) - 21], index[block * (k + 1): block * (k + 2)])
                for k in range(n_folds)]
    return fold_splitter


def test_a_conformant_component_is_stamped_and_comes_back(capsys):
    result = save_component("fold_splitter", _good_splitter())
    assert result.conformant
    assert result.stamped_at and result.helper_version
    assert source_of("fold_splitter") == "yours"

    loaded = load_component("fold_splitter")
    folds = loaded(pd.bdate_range("2015-01-01", periods=320))
    assert len(folds) == 4, "the student's own choice of fold count survives the round trip"
    assert "using your implementation" in capsys.readouterr().out


def test_the_component_is_rebuilt_from_the_file_not_the_session():
    """What a later notebook loads is the file, on a cold kernel, with nothing else in scope."""
    save_component("fold_splitter", _good_splitter(), quiet=True)
    saved = (project.components_dir() / "fold_splitter.py").read_text()
    namespace: dict = {}
    exec(compile(saved, "<saved>", "exec"), namespace)
    assert "fold_splitter" in namespace


def test_a_component_that_leans_on_another_cell_is_refused(capsys):
    outside = 21

    def fold_splitter(index, n_folds=4):
        index = pd.Index(index).sort_values()
        block = len(index) // (n_folds + 1)
        return [(index[: block * (k + 1) - outside], index[block * (k + 1): block * (k + 2)])
                for k in range(n_folds)]

    result = save_component("fold_splitter", fold_splitter)
    assert not result.conformant
    detail = [c.detail for c in result.checks if not c.passed][0]
    assert "outside" in detail, "the message has to name the symbol that is missing"
    assert "include=" in detail or "also=" in detail, "and say how to fix it"


def test_include_carries_a_value_the_component_reads():
    threshold = 21

    def fold_splitter(index, n_folds=4):
        index = pd.Index(index).sort_values()
        block = len(index) // (n_folds + 1)
        return [(index[: block * (k + 1) - threshold], index[block * (k + 1): block * (k + 2)])
                for k in range(n_folds)]

    assert save_component("fold_splitter", fold_splitter,
                          include={"threshold": threshold}, quiet=True).conformant


def test_a_wrong_component_does_not_block_the_next_part(capsys):
    """The whole point of the fallback: a wrong Part 4 must not stop Part 7 from running."""
    def fold_splitter(index):
        return [(index, index)]  # trains and validates on everything

    assert not save_component("fold_splitter", fold_splitter).conformant
    capsys.readouterr()

    loaded = load_component("fold_splitter")
    message = capsys.readouterr().out
    assert "shipped reference" in message
    assert "not conformant" in message
    assert len(loaded(pd.bdate_range("2015-01-01", periods=320))) > 0


def test_an_unwritten_component_falls_back_and_says_so(capsys):
    load_component("labeler")
    assert "have not written it yet" in capsys.readouterr().out


def test_a_config_component_round_trips():
    spec = contract("strategy_spec").reference()
    assert save_component("strategy_spec", spec, quiet=True).conformant
    assert load_component("strategy_spec", quiet=True)["family"] == spec["family"]


def test_a_config_component_refuses_a_function():
    with pytest.raises(ValueError, match="recorded choice"):
        save_component("objective", lambda: None)


def test_an_unknown_component_name_lists_the_real_ones():
    with pytest.raises(KeyError, match="fold_splitter"):
        save_component("fold_spliter", _good_splitter())


def test_results_rows_accumulate_and_keep_their_columns():
    row = dict(stage="v0", start="2015-01-02", end="2019-12-31", total_return=0.31, sharpe=0.42)
    append_result(row, quiet=True)
    append_result(dict(row, stage="final", sharpe=0.55), quiet=True)
    frame = results()
    assert list(frame["stage"]) == ["v0", "final"]
    assert frame["written_at"].notna().all() and frame["run_id"].notna().all()


def test_a_results_row_with_an_invented_column_is_refused():
    with pytest.raises(ValueError, match="fixed columns"):
        append_result(dict(stage="v0", start="a", end="b", total_return=1.0, sharpe=1.0,
                           my_own_metric=3))


def test_a_results_row_missing_what_part_11_reads_is_refused():
    with pytest.raises(ValueError, match="reads three of these rows"):
        append_result(dict(stage="v0", start="a"))


def test_the_report_says_what_is_still_outstanding(capsys):
    save_component("fold_splitter", _good_splitter(), quiet=True)
    append_result(dict(stage="baseline", start="a", end="b", total_return=0.1, sharpe=0.2),
                  quiet=True)
    payload = report(answers={"what_moved": "x" * 60, "what_i_would_change": "y" * 60})
    assert payload["components_conformant"] == 1
    assert payload["components_required"] == 20
    assert payload["runs_present"] == ["baseline"]
    assert payload["complete"] is False
    assert "v0, final" in capsys.readouterr().out
    written = json.loads((project.home() / "submission_report.json").read_text())
    assert written == payload


def test_status_lists_every_component_the_course_asks_for():
    save_component("fold_splitter", _good_splitter(), quiet=True)
    text = status()
    assert text.count("\n") >= 20
    assert "conformant" in text and "not written yet" in text


def test_the_stamp_records_what_a_later_unit_needs_to_reload_it():
    """The stamp carries no unit number: which unit writes a component is a course's fact, and
    this package serves more than one course."""
    save_component("fold_splitter", _good_splitter(), quiet=True)
    meta = _meta("fold_splitter")
    assert meta["component"] == "fold_splitter"
    assert meta["conformant"] is True
    assert meta["symbol"] == "fold_splitter"
    assert meta["file"].endswith(".py")
    assert meta["stamped_at"]
    assert "unit" not in meta


def test_the_packaged_fingerprint_matches_the_repository_copy():
    """The package ships its own copy, because a student never has the repository. Two copies
    drift, so this is the check that they have not."""
    from pathlib import Path

    from ml4t_coursework import data

    repo = Path(__file__).resolve().parents[2] / "data" / "etf_close_fingerprint.csv"
    if not repo.is_file():
        pytest.skip("not running inside the course repository")
    assert data.fingerprint().equals(pd.read_csv(repo, index_col="symbol"))


def test_loading_prices_before_setup_says_what_to_do():
    from ml4t_coursework import data

    with pytest.raises(FileNotFoundError, match="setup notebook"):
        data.load()


def test_the_symbol_list_is_the_hundred_the_course_uses():
    from ml4t_coursework import data

    assert len(data.SYMBOLS) == 100
    assert len(set(data.SYMBOLS)) == 100
    assert set(data.fingerprint().index) == set(data.SYMBOLS)


def test_the_fingerprint_check_reads_the_columns_the_fingerprint_has():
    """The packaged check and the repository's authoring script have to agree on tolerances, and
    the first version of this read column names the fingerprint does not carry."""
    import numpy as np

    from ml4t_coursework import data

    reference = data.fingerprint()
    index = pd.bdate_range("2006-01-03", periods=400, name="date")
    rng = np.random.default_rng(0)
    fake = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.012, size=(len(index), len(data.SYMBOLS))), axis=0)),
        index=index, columns=sorted(data.SYMBOLS))
    text = data.check(fake)
    assert "checked 100 of 100" in text
    assert "Worth a look" in text, "400 random sessions should not look like the course's panel"
    assert set(reference.columns) >= {"sessions", "annualized_vol", "mean_abs_return"}


def test_the_declared_version_and_the_installed_version_agree():
    """`helper_version` on every run row comes from the module, the distribution metadata comes
    from pyproject, and the release workflow checks the tag against the module only. Two static
    strings that nothing compares will drift, and a student's recorded version would then name
    something pip never installed. Reading the metadata rather than pyproject also catches a build
    that is stale against the source tree."""
    from importlib.metadata import version

    import ml4t_coursework

    assert ml4t_coursework.__version__ == version("ml4t-coursework")

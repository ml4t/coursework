"""The runner wires the twenty components; these are the properties that wiring has to have.

The components are each checked against their own contract elsewhere. What is left to establish
here is what only shows up once they are composed: that the two books a course compares are
measured the same way, that the sealed window stays sealed, that the cadence a student declares
is the cadence the book trades at, and that the whole composition still cannot see forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml4t_coursework import fixtures, markets, project, runner
# `ml4t_coursework.results` is shadowed by the function of the same name that the
# package re-exports, so the module's own names are imported directly.
from ml4t_coursework.results import REQUIRED, results as results_log


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / "panel.parquet"
    fixtures.prices().to_parquet(path)
    return markets.MarketSpec(market="fixture", title="synthetic panel", acquire=lambda: path)


@pytest.fixture
def tampered(tmp_path):
    """The same panel with its last fifth walked away from what actually happened.

    A fifth rather than the holdout's own quarter, and a ramp rather than a step. `availability_lag`
    costs the panel its first session, so the split lands one session later than a fraction of the
    raw file does; and a step would be voided by `quality_gates` as an implausible move, which
    changes the session count and so moves the split itself. The ramp starts at one and ends at
    ten, so no single bar moves more than a few percent and every gate stays quiet.
    """
    wide = fixtures.prices()
    cut = int(len(wide) * 0.80)
    tail = wide.iloc[cut:]
    wide.iloc[cut:] = tail.mul(np.linspace(1.0, 10.0, len(tail)), axis=0)
    path = tmp_path / "tampered.parquet"
    wide.to_parquet(path)
    return markets.MarketSpec(market="fixture", title="tampered panel", acquire=lambda: path)


def test_a_run_logs_one_row_with_the_columns_the_terminal_unit_reads(spec):
    project.setup(quiet=True)
    out = runner.run(spec, stage="v0", quiet=True)
    log = results_log()
    assert len(log) == 1
    row = log.iloc[0]
    for column in REQUIRED:
        assert pd.notna(row[column]), f"{column} was not written"
    assert row["stage"] == "v0"
    assert out.row["sharpe"] == pytest.approx(float(row["sharpe"]))


def test_the_baseline_and_the_pipeline_are_measured_over_the_same_window(spec):
    project.setup(quiet=True)
    base = runner.run(spec, stage="baseline", strategy="baseline", quiet=True)
    full = runner.run(spec, stage="v0", strategy="pipeline", quiet=True)
    assert (base.row["start"], base.row["end"]) == (full.row["start"], full.row["end"]), (
        "the terminal unit reads these two rows against each other, which it cannot do if they "
        "cover different stretches of the sample"
    )


def test_a_development_run_never_reports_a_holdout_date(spec):
    project.setup(quiet=True)
    development = runner.run(spec, stage="v0", window="development", quiet=True)
    holdout = runner.run(spec, stage="final", window="holdout", quiet=True)
    assert development.row["end"] < holdout.row["start"], (
        "a runner that reported the holdout on every assembly run would spend it dozens of times"
    )


def test_the_book_trades_at_the_cadence_the_config_declares(spec):
    project.setup(quiet=True)
    config = {"engine": "vectorized", "fills": "next-open", "rebalance": 1, "unanswerable": "x"}
    monthly = runner.run(spec, stage="v0", append=False, quiet=True)
    daily = runner.run(spec, stage="v0", components={"backtest_config": config},
                       append=False, quiet=True)
    assert daily.row["turnover"] > monthly.row["turnover"] * 2, (
        "rebalancing every session instead of every 21 must cost materially more turnover; if it "
        "does not, the book is not being held between rebalances"
    )


def test_a_supplied_component_is_the_one_that_runs(spec):
    project.setup(quiet=True)

    def flat_signal(scores, **_):
        return scores * 0.0

    out = runner.run(spec, stage="v0", components={"signal": flat_signal},
                     append=False, quiet=True)
    assert out.weights.abs().to_numpy().sum() == pytest.approx(0.0)
    assert out.row["total_return"] == pytest.approx(0.0)


def test_the_development_result_does_not_move_when_the_future_is_tampered_with(spec, tampered):
    """The composition-level leakage probe: every component passes its own, which is not the
    same as the pipeline passing one."""
    project.setup(quiet=True)
    honest = runner.run(spec, stage="v0", window="development", append=False, quiet=True)
    altered = runner.run(tampered, stage="v0", window="development", append=False, quiet=True)
    assert (honest.row["start"], honest.row["end"]) == (altered.row["start"], altered.row["end"]), (
        "the tamper moved the window itself, so the comparison below would not be about leakage"
    )
    for column in ("total_return", "sharpe", "max_drawdown", "turnover"):
        assert honest.row[column] == pytest.approx(altered.row[column]), (
            f"{column} changed when prices after the development window were multiplied by ten, "
            f"so something in the pipeline is reading past its own boundary"
        )


def test_a_spec_cannot_declare_a_bar_the_annualization_does_not_know():
    with pytest.raises(ValueError, match="bar must be one of"):
        markets.MarketSpec(market="x", title="x", acquire=lambda: None, bar="fortnightly")


def test_the_etf_spec_is_registered_and_describes_what_it_costs_to_run():
    spec = markets.market("etfs")
    assert spec.bar == "daily"
    assert spec.bars_per_year == 252
    assert "etfs" in markets.market_catalog()

"""The pipeline: the twenty components, wired in stage order, short enough to read in one sitting.

There is deliberately no facade. `run` calls the components by name and hands each one the output
of the one before it, so the code a student is asked to change in an exercise is the code they can
see here. Anything that wrapped the stages would hide exactly that, and would fail the first time
someone wanted a stage it did not expose.

Which components it calls is not a choice this module makes. `load_component` returns the
student's implementation where they have written one that passes its contract, and the shipped
reference otherwise, so the same runner serves a Foundations student assembling their own pipeline
and a Research to Production student handed all twenty.

The window matters more than it looks. Every run reports the development window unless it is
asked for the holdout, because a runner that reported the holdout on every assembly run would
spend it dozens of times, which is the one thing unit 8.1 says not to do.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .components import load_component
from .markets import MarketSpec
from .results import append_result

WINDOWS = ("development", "holdout")


@dataclass
class Run:
    """What one pass produced: the row that was logged, and the series behind it."""

    row: dict[str, Any]
    returns: pd.Series
    weights: pd.DataFrame
    costs: pd.Series
    panel: pd.DataFrame

    def __repr__(self) -> str:
        r = self.row
        return (f"<Run {r['stage']} {r['start']}..{r['end']} "
                f"sharpe {r['sharpe']:.2f} return {r['total_return']:.1%}>")


def _resolve(components: dict[str, Any] | None) -> Callable[[str], Any]:
    """Where each component comes from: an explicit mapping, or the student's project folder."""
    if components is None:
        return lambda name: load_component(name, quiet=True)
    missing = object()

    def get(name: str) -> Any:
        obj = components.get(name, missing)
        if obj is missing:
            return load_component(name, quiet=True)
        return obj

    return get


def _eligibility(universe, panel: pd.DataFrame, sessions: pd.Index, assets: list[str],
                 rebalance: int) -> pd.DataFrame:
    """Who may be traded on each session, decided on rebalance dates and held between them.

    Asking the rule every session would answer a question nobody trades on: membership only
    matters where the book is allowed to change.
    """
    mask = pd.DataFrame(False, index=sessions, columns=assets)
    current: list[str] = []
    for position, date in enumerate(sessions):
        if position % rebalance == 0:
            current = [a for a in universe(panel, date) if a in mask.columns]
        if current:
            mask.loc[date, current] = True
    return mask


def _metrics(net: pd.Series, trades: pd.DataFrame, costs: pd.Series, bars: int) -> dict[str, float]:
    curve = (1.0 + net).cumprod()
    total = float(curve.iloc[-1] - 1.0)
    years = len(net) / bars
    vol = float(net.std(ddof=1) * np.sqrt(bars))
    drawdown = float((curve / curve.cummax() - 1.0).min())
    traded = trades.abs().sum(axis=1)
    return {
        "total_return": total,
        "annual_return": float((1.0 + total) ** (1 / years) - 1.0) if years > 0 else float("nan"),
        "annual_vol": vol,
        "sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(bars)) if net.std(ddof=1) else 0.0,
        "max_drawdown": drawdown,
        "turnover": float(traded.mean() * bars),
        "cost_bps": float(costs.sum() / traded.sum() * 10_000) if traded.sum() else 0.0,
    }


def run(spec: MarketSpec, *, stage: str, strategy: str = "pipeline",
        window: str = "development", components: dict[str, Any] | None = None,
        append: bool = True, quiet: bool = False) -> Run:
    """Run the pipeline on one market and log one results row.

    `strategy="baseline"` builds the book from `baseline_strategy` and stops there, which is the
    non-ML comparison the terminal unit reads against the finished pipeline. Everything else about
    the run - the market, the window, the costs, the metrics - is identical, which is the only way
    the two rows are comparable.
    """
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {', '.join(WINDOWS)}, got {window!r}")
    if strategy not in ("pipeline", "baseline"):
        raise ValueError(f"strategy must be 'pipeline' or 'baseline', got {strategy!r}")
    get = _resolve(components)
    bars = spec.bars_per_year

    # 6.2 how the backtest executes, including how often the book is allowed to change
    config = get("backtest_config")
    rebalance = int(config["rebalance"])

    # 2.1 the stored panel, long
    source = spec.acquire()
    panel = get("data_panel")(str(source))

    # 2.2 nothing is usable the moment it is stamped
    panel = get("availability_lag")(panel)
    panel = panel[panel["close"].notna()]

    # 2.3 a breach voids the bar
    panel, breaches = get("quality_gates")(panel)
    panel = panel[panel["close"].notna()]

    sessions = panel.index.get_level_values("date").unique()
    assets = sorted(panel.index.get_level_values("asset").unique())

    # 2.4 who may be traded, decided with what was knowable then
    eligible = _eligibility(get("universe"), panel, sessions, assets, rebalance)

    # 3.2 the holdout is sealed here and read only when this run is asked for it
    development, holdout = get("holdout_split")(sessions)
    reported = holdout if window == "holdout" else development

    if strategy == "baseline":
        # 2.5 the auditable non-ML rule everything else must beat
        book = get("baseline_strategy")(panel).reindex(index=sessions, columns=assets).fillna(0.0)
    else:
        # 3.3/3.4 the outcome a decision is judged on, and 3.5 the form the model predicts
        target = get("task_form")(get("labeler")(panel))
        # 4.1/4.2 the predictors
        X = get("features")(panel)
        shared = X.index.intersection(target.index)
        X, y = X.loc[shared].sort_index(), target.loc[shared].sort_index()

        # 3.1 folds exist so a student can select on them; this run reports, it does not select
        get("fold_splitter")(development)

        # Fitting stops a label horizon before the development window ends, so no training label
        # overlaps a reported date.
        dates = X.index.get_level_values("date")
        purge = spec.purge + spec.execution_delay
        fit_until = development[max(len(development) - purge, 0) - 1]
        is_fit = dates <= fit_until

        # 3.6 every parameter it uses is learned on the fitting rows and only there
        prep = get("preprocessor")()
        prep.fit(X[is_fit])
        Z = prep.transform(X)

        # 5.1 the model
        model = get("model_linear")()
        model.fit(Z[is_fit], y[is_fit])
        scores = pd.Series(model.predict(Z), index=Z.index).unstack("asset")
        scores = scores.reindex(index=sessions, columns=assets).where(eligible)

        # 6.1 scores into positions, 7.1/7.2 positions into a book
        book = get("allocator")(get("signal")(scores)).reindex(
            index=sessions, columns=assets).fillna(0.0)
        # The book is only allowed to change on the cadence 6.2 declared. Without this the
        # allocator re-solves every bar and the run reports the turnover of a different strategy.
        on_schedule = pd.Series(book.index.isin(sessions[::rebalance]), index=book.index)
        book = book.where(on_schedule, axis=0).ffill().fillna(0.0)
        # 7.4 position controls, which act between rebalances and so come after the cadence
        book = get("exit_rule")(book, panel)

    book = book.where(eligible, 0.0)

    close = panel["close"].unstack("asset").reindex(index=sessions, columns=assets)
    bar_returns = close.pct_change(fill_method=None).fillna(0.0)
    gross = (book.shift(spec.execution_delay) * bar_returns).sum(axis=1)
    trades = book.diff()
    trades.iloc[0] = book.iloc[0]

    # 7.3 what trading takes out
    costs = get("cost_model")(trades).reindex(sessions).fillna(0.0)
    net = (gross - costs).rename("return")

    on = net.index.isin(reported)
    row = {
        "stage": stage,
        "start": str(pd.Timestamp(reported[0]).date()),
        "end": str(pd.Timestamp(reported[-1]).date()),
        "dataset_vintage": dt.date.fromtimestamp(source.stat().st_mtime).isoformat(),
        "components_used": f"{strategy} on {spec.market}, {window} window",
        **_metrics(net[on], trades[on], costs[on], bars),
    }
    if append:
        append_result(row, quiet=quiet)
    elif not quiet:
        print(f"{stage}: sharpe {row['sharpe']:.2f} over {row['start']}..{row['end']} (not logged)")
    return Run(row=row, returns=net, weights=book, costs=costs, panel=panel)

"""Every reference passes its own contract, and a deliberately broken variant fails it.

The second half is the one that matters. A contract test authored against the reference is
guaranteed passable, which says nothing about whether it can catch anything, so each component
here also carries the defect it exists to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml4t_coursework import contracts
from ml4t_coursework.checks import evaluate

ALL = sorted(contracts.load_all())


@pytest.mark.parametrize("name", ALL)
def test_reference_is_conformant(name):
    contract = contracts.get(name)
    result = evaluate(contract, contract.reference())
    assert result.conformant, f"the shipped reference fails its own contract:\n{result}"


@pytest.mark.parametrize("name", ALL)
def test_every_contract_has_the_five_checks(name):
    contract = contracts.get(name)
    result = evaluate(contract, contract.reference())
    names = [c.name for c in result.checks]
    assert "interface" in names
    assert "leakage probe" in names
    assert "determinism" in names
    assert len(contract.invariants) >= 1, "a contract with no domain invariant checks nothing"
    assert result.delta, "the reference delta is reported even when it is zero"


def test_the_registry_covers_the_component_list():
    assert len(ALL) == 20, f"expected the 20 components of the component list, found {len(ALL)}"
    for name, contract in contracts.load_all().items():
        assert contract.units, f"{name} does not say which unit writes it"
        assert contract.summary.endswith("."), f"{name}'s summary is not a sentence"


# --- the broken variants ------------------------------------------------------------------------
# Each one is the defect its component exists to catch, and each names the check that must fail.


def _shuffled_folds():
    def fold_splitter(index, n_folds=5):
        idx = pd.Index(index).sort_values()
        shuffled = idx.to_series().sample(frac=1.0, random_state=0).index
        block = len(shuffled) // n_folds
        out = []
        for k in range(n_folds):
            val = shuffled[block * k: block * (k + 1)]
            out.append((shuffled.difference(val), val))
        return out
    return fold_splitter


def _no_buffer():
    return contracts.get("fold_splitter").reference(horizon=0)


def _holdout_in_the_middle():
    def holdout_split(index, fraction=0.25):
        index = pd.Index(index).sort_values()
        start = int(len(index) * 0.4)
        stop = start + int(len(index) * fraction)
        return index.delete(range(start, stop)), index[start:stop]
    return holdout_split


def _panel_that_fills_gaps():
    inner = contracts.get("data_panel").reference()

    def data_panel(path):
        return inner(path).bfill()
    return data_panel


def _no_lag():
    return contracts.get("availability_lag").reference(lag=0)


def _gates_calibrated_on_everything():
    def quality_gates(panel, **_):
        step = panel.pct_change(fill_method=None).abs()
        # The threshold is the sample's own 99.5th percentile, so what counts as an implausible
        # move on any given day is decided by moves that had not happened yet.
        mask = step > step.stack().quantile(0.995)
        report = pd.DataFrame(
            [{"gate": "outlier", "symbol": s, "date": d} for d, s in mask.stack()[mask.stack()].index]
        )
        if report.empty:
            report = pd.DataFrame(columns=["gate", "symbol", "date"])
        return panel.mask(mask), report
    return quality_gates


def _label_at_the_decision_close():
    return contracts.get("labeler").reference(delay=0)


def _pooled_quintiles():
    def task_form(label):
        return pd.Series(pd.qcut(label, 5, labels=False), index=label.index).astype(float)
    return task_form


def _features_normalized_on_everything():
    def features(prices):
        raw = pd.DataFrame({
            "mom_21": prices.pct_change(21).stack(future_stack=True),
            "vol_21": prices.pct_change().rolling(21).std().stack(future_stack=True),
        })
        raw.index = raw.index.set_names(["date", "symbol"])
        raw = raw.dropna()
        return ((raw - raw.mean()) / raw.std()).dropna()
    return features


def _preprocessor_that_fits_in_transform():
    class Preprocessor:
        def fit(self, X):
            return self

        def transform(self, X):
            return ((X - X.mean()) / X.std()).fillna(0.0)
    return Preprocessor


def _model_that_rescales_in_predict():
    inner = contracts.get("model_linear").reference()

    class LinearModel(inner):
        def predict(self, X):
            raw = super().predict(X)
            return (raw - raw.mean()) / raw.std()
    return LinearModel


def _unseeded_gbm():
    """Bagging the training rows with an unseeded generator: the defect determinism exists for."""
    inner = contracts.get("model_gbm").reference()

    class GbmModel(inner):
        def fit(self, X, y):
            keep = np.random.default_rng().choice(len(X), size=int(len(X) * 0.8), replace=False)
            return super().fit(X.iloc[keep], y.iloc[keep])
    return GbmModel


def _universe_screened_on_the_whole_history():
    def universe(prices, asof, **_):
        complete = prices.notna().all()
        return sorted(complete.index[complete].tolist())
    return universe


def _signal_ranked_against_the_whole_sample():
    def signal(scores, quantile=0.2):
        flat = scores.stack()
        lo, hi = flat.quantile(quantile), flat.quantile(1 - quantile)
        return ((scores > hi).astype(float) - (scores < lo).astype(float)).fillna(0.0)
    return signal


def _allocator_without_a_cap():
    def allocator(signal, **_):
        raw = signal.fillna(0.0)
        first = raw.columns[0]
        weights = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
        weights[first] = 1.0
        return weights
    return allocator


def _cost_model_with_a_rebate():
    def cost_model(trades, **_):
        traded = trades.abs().fillna(0.0).sum(axis=1)
        return (traded * 0.0003 - traded ** 2 * 0.05).rename("cost")
    return cost_model


def _stop_so_wide_it_never_fires():
    return contracts.get("exit_rule").reference(stop=0.99)


def _baseline_that_holds_the_winners():
    def baseline_strategy(prices, **_):
        # Weight by what each fund went on to do over the whole sample: the most flattering
        # baseline available, and one nobody could have held.
        total = (prices.ffill().iloc[-1] / prices.bfill().iloc[0]).fillna(1.0)
        share = total / total.sum()
        return pd.DataFrame([share.to_numpy()] * len(prices), index=prices.index,
                            columns=prices.columns)
    return baseline_strategy


def _spec_with_a_placeholder():
    return dict(contracts.get("strategy_spec").reference(), mechanism="TBD")


def _objective_the_pipeline_cannot_act_on():
    return dict(contracts.get("objective").reference(), metric="whatever looks best")


def _config_missing_the_hard_question():
    record = dict(contracts.get("backtest_config").reference())
    record.pop("unanswerable")
    return record


BROKEN = [
    ("fold_splitter", _shuffled_folds, "leakage probe"),
    ("fold_splitter", _no_buffer, "label buffer"),
    ("holdout_split", _holdout_in_the_middle, "the holdout is the last block"),
    ("data_panel", _panel_that_fills_gaps, "unbalanced panel kept"),
    ("availability_lag", _no_lag, "the lag is applied"),
    ("quality_gates", _gates_calibrated_on_everything, "leakage probe"),
    ("labeler", _label_at_the_decision_close, "executable prices"),
    ("task_form", _pooled_quintiles, "leakage probe"),
    ("features", _features_normalized_on_everything, "leakage probe"),
    ("preprocessor", _preprocessor_that_fits_in_transform, "leakage probe"),
    ("model_linear", _model_that_rescales_in_predict, "leakage probe"),
    ("model_gbm", _unseeded_gbm, "determinism"),
    ("universe", _universe_screened_on_the_whole_history, "leakage probe"),
    ("signal", _signal_ranked_against_the_whole_sample, "leakage probe"),
    ("allocator", _allocator_without_a_cap, "no name exceeds its cap"),
    ("cost_model", _cost_model_with_a_rebate, "costs are never negative"),
    ("exit_rule", _stop_so_wide_it_never_fires, "the rule does something"),
    ("baseline_strategy", _baseline_that_holds_the_winners, "leakage probe"),
    ("strategy_spec", _spec_with_a_placeholder, "mechanism is actually stated"),
    ("objective", _objective_the_pipeline_cannot_act_on, "metric is one the course recognizes"),
    ("backtest_config", _config_missing_the_hard_question, "interface"),
]


@pytest.mark.parametrize("name,factory,expected_check", BROKEN,
                         ids=[f"{n}-{c}" for n, _, c in BROKEN])
def test_the_broken_variant_fails_the_check_it_should(name, factory, expected_check):
    contract = contracts.get(name)
    result = evaluate(contract, factory())
    assert not result.conformant, f"{name}: the broken variant passed:\n{result}"
    failed = [c for c in result.checks if not c.passed]
    assert failed, "a non-conformant result with no failing check"
    assert failed[0].name == expected_check, (
        f"{name}: expected {expected_check!r} to catch it, {failed[0].name!r} did first:\n{result}"
    )


@pytest.mark.parametrize("name,factory,expected_check", BROKEN,
                         ids=[f"{n}-{c}" for n, _, c in BROKEN])
def test_the_failure_names_expectation_and_observation(name, factory, expected_check):
    """"Expected 12 folds, got 9" is feedback; a bare AssertionError is not."""
    result = evaluate(contracts.get(name), factory())
    detail = [c.detail for c in result.checks if not c.passed][0]
    assert "got" in detail or "expected" in detail, f"{name}: unhelpful message: {detail!r}"
    assert len(detail) > 40, f"{name}: the message says too little: {detail!r}"


def test_every_component_has_a_broken_variant():
    covered = {name for name, _, _ in BROKEN}
    assert covered == set(ALL), f"no broken variant for {sorted(set(ALL) - covered)}"

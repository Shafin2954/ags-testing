"""
Test suite for adaptive-greedy-search (import name: ags), v2.0.0.

Two groups:
  * Plain tests  -> behaviour that works and must keep working.
  * xfail(strict=True) tests -> confirmed bugs / gaps. They assert the
    CORRECT behaviour, so they currently fail. If one starts passing
    (XPASS), pytest reports it as a failure so we know the bug is fixed
    and the marker can be removed.

Run:  pytest -q tests
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.datasets import load_diabetes, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ags import AdaptiveGreedySearch

warnings.filterwarnings("ignore")

GRID = {
    "max_depth": [1, 2, 3, 4, 6, 8, None],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "criterion": ["gini", "entropy"],
}


@pytest.fixture(scope="module")
def data():
    return make_classification(n_samples=300, n_features=12, n_informative=5,
                               flip_y=0.05, random_state=0)


def ags(**kw):
    kw.setdefault("max_evaluations", 20)
    return AdaptiveGreedySearch(DecisionTreeClassifier(random_state=0), GRID, **kw)


# ======================================================== working behaviour ==

def test_fit_sets_documented_attributes(data):
    s = ags().fit(*data)
    for attr in ["best_params", "best_score", "best_state", "history", "n_evaluations",
                 "total_time", "stopped_early", "n_pruned", "total_folds_run",
                 "total_folds_possible", "folds_saved"]:
        assert hasattr(s, attr), attr
    assert set(s.best_params) == set(GRID)


def test_budget_respected_and_no_duplicate_evaluations(data):
    s = ags(max_evaluations=15, early_stopping_patience=None).fit(*data)
    states = [h["state"] for h in s.history]
    assert s.n_evaluations == 15 == len(states)
    assert len(set(states)) == len(states)


def test_best_score_is_max_of_history(data):
    s = ags().fit(*data)
    assert s.best_score == max(h["score"] for h in s.history)


def test_fold_accounting_consistent(data):
    s = ags(cv=5, max_evaluations=30, early_stopping_patience=None).fit(*data)
    assert s.total_folds_possible == 5 * s.n_evaluations
    assert s.folds_saved == s.total_folds_possible - s.total_folds_run
    assert s.n_pruned == sum(h["n_folds_used"] < 5 for h in s.history)


def test_same_seed_is_reproducible(data):
    a = ags(random_state=7).fit(*data)
    b = ags(random_state=7).fit(*data)
    assert [h["state"] for h in a.history] == [h["state"] for h in b.history]
    assert a.best_score == b.best_score


def test_no_pruning_means_full_cv(data):
    s = ags(pruning_strategy="none").fit(*data)
    assert s.folds_saved == 0 and s.n_pruned == 0


def test_exhaustive_run_matches_gridsearchcv(data):
    """Full budget, no pruning, no early stop -> must equal GridSearchCV."""
    X, y = data
    s = ags(max_evaluations=10**6, early_stopping_patience=None,
            pruning_strategy="none", random_state=42).fit(X, y)
    gs = GridSearchCV(DecisionTreeClassifier(random_state=0), GRID,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42)).fit(X, y)
    assert s.n_evaluations == len(gs.cv_results_["params"])
    assert s.best_score == pytest.approx(gs.best_score_)
    ags_scores = {tuple(sorted(h["params"].items(), key=str)): h["score"] for h in s.history}
    for p, m in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"]):
        assert ags_scores[tuple(sorted(p.items(), key=str))] == pytest.approx(m)


def test_optimistic_pruning_never_prunes_a_would_be_winner(data):
    """Safety property claimed in README, checked with a string scorer."""
    X, y = data
    s = ags(cv=5, max_evaluations=40, early_stopping_patience=None,
            pruning_strategy="optimistic", random_state=3).fit(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=3)
    incumbent = -np.inf
    for h in s.history:
        if h["pruned"]:
            est = DecisionTreeClassifier(random_state=0).set_params(**h["params"])
            full = cross_val_score(est, X, y, cv=cv).mean()
            assert full < incumbent
        incumbent = max(incumbent, h["score"])


def test_pandas_input(data):
    X, y = data
    s = ags().fit(pd.DataFrame(X), pd.Series(y))
    assert 0 <= s.best_score <= 1


def test_regression_neg_scoring_and_pipeline_params():
    X, y = load_diabetes(return_X_y=True)
    pipe = Pipeline([("sc", StandardScaler()), ("r", Ridge())])
    s = AdaptiveGreedySearch(pipe, {"r__alpha": list(np.logspace(-3, 3, 13))},
                             scoring="neg_mean_squared_error",
                             max_evaluations=10).fit(X, y)
    assert s.best_score < 0
    assert s.score_upper_bound == 0.0


def test_gp_surrogate_runs(data):
    s = ags(surrogate_type="gp", max_evaluations=15).fit(*data)
    assert s.n_evaluations >= 8


def test_unknown_surrogate_rejected():
    with pytest.raises(ValueError):
        ags(surrogate_type="nope")


def test_default_early_stopping_uses_well_under_budget(data):
    """Documents default behaviour: patience=5 usually stops long before budget."""
    s = ags(max_evaluations=60).fit(*data)
    assert s.stopped_early and s.n_evaluations < 60


# ============================================================ known bugs ====

@pytest.mark.xfail(strict=True, reason="BUG: fit() does not reset state; second "
                   "fit on new data reuses scores from the old data")
def test_refit_on_new_data_starts_fresh(data):
    s = ags(max_evaluations=10, early_stopping_patience=None)
    s.fit(*data)
    X2, y2 = make_classification(n_samples=300, n_features=12, random_state=99)
    s.fit(X2, y2)
    assert len(s.history) == s.n_evaluations == 10


@pytest.mark.xfail(strict=True, reason="GAP: not a sklearn BaseEstimator; "
                   "clone()/get_params()/Pipeline/cross_val_score nesting fail")
def test_sklearn_clone_compatible():
    clone(ags())


@pytest.mark.xfail(strict=True, reason="GAP: cv only accepts int; CV splitter "
                   "objects (GroupKFold, TimeSeriesSplit, KFold) raise")
def test_accepts_cv_splitter_object(data):
    ags(cv=KFold(3)).fit(*data)


@pytest.mark.xfail(strict=True, reason="BUG: initial_points=0 -> "
                   "'max() arg is an empty sequence'")
def test_initial_points_zero(data):
    ags(initial_points=0).fit(*data)


@pytest.mark.xfail(strict=True, reason="BUG: max_evaluations=0 crashes instead "
                   "of raising a clear ValueError")
def test_max_evaluations_zero_gives_clear_error(data):
    with pytest.raises(ValueError, match="max_evaluations"):
        ags(max_evaluations=0).fit(*data)


@pytest.mark.xfail(strict=True, reason="BUG: typo in pruning_strategy is "
                   "silently accepted (pruning just turns off)")
def test_unknown_pruning_strategy_rejected():
    with pytest.raises(ValueError):
        ags(pruning_strategy="optimstic")


@pytest.mark.xfail(strict=True, reason="GAP: one failing config (invalid param "
                   "combo) kills the whole search; GridSearchCV uses error_score")
def test_invalid_param_combo_is_skipped(data):
    grid = {"penalty": ["l2", "l1"], "C": [0.1, 1.0, 10.0]}
    s = AdaptiveGreedySearch(LogisticRegression(solver="lbfgs"), grid,
                             max_evaluations=6).fit(*data)
    assert s.best_params["penalty"] == "l2"


@pytest.mark.xfail(strict=True, reason="GAP: no best_estimator_/refit/predict "
                   "like GridSearchCV; user must refit manually")
def test_has_best_estimator(data):
    s = ags().fit(*data)
    s.best_estimator_.predict(data[0])


class _Scripted(BaseEstimator, RegressorMixin):
    """Fake model: its per-fold scores are scripted by `cfg`."""

    def __init__(self, cfg="a"):
        self.cfg = cfg

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


@pytest.mark.xfail(strict=True, reason="BUG: with a callable scorer the "
                   "'optimistic' rule uses the max fold score seen so far as the "
                   "upper bound, which is not a bound -> it prunes a would-be winner")
def test_optimistic_with_callable_scorer_does_not_prune_winner():
    fold_scores = {"a": [0.6] * 5, "b": [0.5, 0.5, 1.0, 1.0, 1.0]}  # b true mean 0.8
    calls = {"a": 0, "b": 0}

    def scorer(est, X, y):
        v = fold_scores[est.cfg][calls[est.cfg]]
        calls[est.cfg] += 1
        return v

    X, y = np.zeros((50, 1)), np.zeros(50)
    s = AdaptiveGreedySearch(_Scripted(), {"cfg": ["a", "b"]}, scoring=scorer,
                             pruning_strategy="optimistic")
    s.evaluate((0,), X, y)           # incumbent a = 0.6
    s.evaluate((1,), X, y)           # b should survive: it ends at 0.8
    assert s.history[1]["pruned"] is False
    assert s.evaluated[(1,)] == pytest.approx(0.8)

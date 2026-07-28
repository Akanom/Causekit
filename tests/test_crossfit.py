"""Tests for reusable nuisance-model cross-fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalkit import AIPWATE, CrossFitResult, CrossFitter


class _ConstantPropensityResult:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X):
        probability = np.full(len(X), self.probability)
        return pd.DataFrame({0: 1 - probability, 1: probability}, index=X.index)


class _ConstantPropensity:
    def fit(self, X, y):
        return _ConstantPropensityResult(float(np.mean(y)))


class _LinearResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        return design @ self.coefficients


class _LinearOutcome:
    def fit(self, X, y):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        return _LinearResult(np.linalg.lstsq(design, np.asarray(y), rcond=None)[0])


def _data(nobs: int = 300, seed: int = 902):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=nobs), "x2": rng.normal(size=nobs)})
    treatment = pd.Series(rng.binomial(1, 0.5, nobs), index=X.index)
    outcome = pd.Series(
        0.4 + 1.3 * treatment + 0.7 * X["x1"] - 0.3 * X["x2"] + rng.normal(size=nobs),
        index=X.index,
    )
    return X, treatment, outcome


def test_cross_fitter_returns_complete_aligned_out_of_fold_predictions() -> None:
    X, treatment, outcome = _data()
    result = CrossFitter(
        propensity_factory=_ConstantPropensity,
        outcome_factory=_LinearOutcome,
        n_splits=5,
        random_state=41,
    ).fit_predict(X, treatment=treatment, outcome=outcome)

    assert isinstance(result, CrossFitResult)
    assert result.propensity.index.equals(X.index)
    assert result.outcome_treated.index.equals(X.index)
    assert result.outcome_control.index.equals(X.index)
    assert set(result.fold.unique()) == set(range(5))
    assert np.isfinite(result.propensity).all()
    for fold in range(5):
        held_out = result.fold == fold
        assert treatment[held_out].nunique() == 2


def test_cross_fitting_is_deterministic_for_fixed_seed_and_drives_aipw() -> None:
    X, treatment, outcome = _data(nobs=1000)
    fitter = CrossFitter(
        propensity_factory=_ConstantPropensity,
        outcome_factory=_LinearOutcome,
        n_splits=4,
        random_state=7,
    )
    first = fitter.fit_predict(X, treatment=treatment, outcome=outcome)
    second = fitter.fit_predict(X, treatment=treatment, outcome=outcome)
    pd.testing.assert_series_equal(first.fold, second.fold)
    pd.testing.assert_series_equal(first.propensity, second.propensity)
    effect = AIPWATE().fit(
        outcome,
        treatment=treatment,
        propensity=first.propensity,
        outcome_treated=first.outcome_treated,
        outcome_control=first.outcome_control,
    )
    assert effect.estimate == pytest.approx(1.3, abs=0.14)


def test_factories_must_return_fresh_fit_capable_models() -> None:
    X, treatment, outcome = _data(nobs=80)
    fitter = CrossFitter(
        propensity_factory=lambda: object(),
        outcome_factory=_LinearOutcome,
        n_splits=2,
    )
    with pytest.raises(TypeError, match="fit"):
        fitter.fit_predict(X, treatment=treatment, outcome=outcome)


def test_cross_fitting_refuses_small_arms_and_index_drift() -> None:
    X, treatment, outcome = _data(nobs=40)
    fitter = CrossFitter(
        propensity_factory=_ConstantPropensity,
        outcome_factory=_LinearOutcome,
        n_splits=5,
    )
    treatment.iloc[:] = 0
    treatment.iloc[:3] = 1
    with pytest.raises(ValueError, match="at least n_splits"):
        fitter.fit_predict(X, treatment=treatment, outcome=outcome)
    with pytest.raises(ValueError, match="indices must match"):
        fitter.fit_predict(
            X,
            treatment=treatment.set_axis(pd.RangeIndex(1, len(treatment) + 1)),
            outcome=outcome,
        )

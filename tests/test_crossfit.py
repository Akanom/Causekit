"""Tests for reusable nuisance-model cross-fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import (
    AIPWATE,
    ClassProbabilityCrossFitResult,
    CrossFitResult,
    CrossFitTask,
    CrossFitTaskResult,
    CrossFitter,
)


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


class _ClassProbabilityResult:
    def __init__(self, classes: np.ndarray, probabilities: np.ndarray) -> None:
        self.classes_ = classes
        self.probabilities = probabilities

    def predict_proba(self, X):
        return np.tile(self.probabilities, (len(X), 1))


class _ClassProbability:
    def fit(self, X, y):
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        return _ClassProbabilityResult(classes, counts / counts.sum())


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


def test_multiclass_probabilities_are_aligned_out_of_fold_and_sum_to_one() -> None:
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 18)}, index=pd.Index(range(100, 118)))
    classes = pd.Series(np.repeat([2.0, 4.0, np.inf], 6), index=X.index)
    result = CrossFitter(
        propensity_factory=_ClassProbability,
        outcome_factory=_LinearOutcome,
        n_splits=3,
        random_state=19,
    ).fit_predict_class_probabilities(X, classes=classes)

    assert isinstance(result, ClassProbabilityCrossFitResult)
    assert result.probabilities.index.equals(X.index)
    assert result.probabilities.columns.tolist() == [2.0, 4.0, np.inf]
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    assert result.fold.index.equals(X.index)
    for fold in range(3):
        assert set(classes[result.fold == fold]) == {2.0, 4.0, np.inf}


def test_masked_regression_tasks_share_folds_and_preserve_task_labels() -> None:
    X = pd.DataFrame({"x": np.tile(np.arange(6, dtype=float), 2)})
    strata = pd.Series(np.repeat([0, 1], 6), index=X.index)
    target = pd.Series(1.0 + 2.0 * X["x"], index=X.index)
    tasks = (
        CrossFitTask(name="all", target=target),
        CrossFitTask(name="stratum_zero", target=target, train_mask=strata.eq(0)),
    )
    result = CrossFitter(
        propensity_factory=_ClassProbability,
        outcome_factory=_LinearOutcome,
        n_splits=2,
        random_state=5,
    ).fit_predict_tasks(X, tasks=tasks, strata=strata)

    assert isinstance(result, CrossFitTaskResult)
    assert result.predictions.columns.tolist() == ["all", "stratum_zero"]
    assert result.predictions.index.equals(X.index)
    np.testing.assert_allclose(result.predictions, np.column_stack([target, target]), atol=1e-12)
    assert set(result.model_names) == {"all", "stratum_zero"}


def test_generic_cross_fitting_refuses_duplicate_tasks_and_too_small_strata() -> None:
    X = pd.DataFrame({"x": np.arange(6, dtype=float)})
    target = pd.Series(np.arange(6, dtype=float))
    fitter = CrossFitter(outcome_factory=_LinearOutcome, n_splits=3)
    duplicate = CrossFitTask(name="same", target=target)
    with pytest.raises(ValueError, match="task names must be unique"):
        fitter.fit_predict_tasks(X, tasks=[duplicate, duplicate])
    with pytest.raises(ValueError, match="stratum must contain at least n_splits"):
        fitter.fit_predict_tasks(
            X,
            tasks=[duplicate],
            strata=pd.Series([0, 0, 0, 0, 1, 1]),
        )

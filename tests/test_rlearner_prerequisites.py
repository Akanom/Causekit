"""Contracts for the native prerequisites of the future honest R-learner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import CATEResultProtocol, CrossFitter, WeightedCATEEstimatorProtocol
from causekit.ml import (
    _cate_prediction,
    _fit_weighted_cate,
    _NativePenalizedLogitCV,
    _NativeWeightedRidgeCV,
)


def test_native_penalized_logit_satisfies_the_penalized_score_contract() -> None:
    index = pd.Index([f"construction-{position}" for position in range(12)])
    covariates = pd.DataFrame(
        {
            "x": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, -1.75, -0.75, 0.25, 1.0, 1.5, 2.0],
            "z": [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 0.75, -0.75, 1.25, -1.25, 0.25],
        },
        index=index,
    )
    treatment = pd.Series([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1], index=index)
    alpha = 0.7

    fitted = _NativePenalizedLogitCV(
        alphas=(alpha,),
        tuning_splits=3,
        random_state=1907,
    ).fit(covariates, treatment)
    probability = fitted.predict_proba(covariates)[:, 1]
    standardized = (covariates.to_numpy(dtype=float) - fitted.feature_mean) / fitted.feature_scale
    residual = probability - treatment.to_numpy(dtype=float)

    assert fitted.selected_alpha == alpha
    assert fitted.classes_.tolist() == [0.0, 1.0]
    assert fitted.training_index.equals(index)
    assert fitted.tuning_fold.index.equals(index)
    assert (pd.crosstab(fitted.tuning_fold, treatment) > 0).all().all()
    assert abs(float(residual.sum())) < 2e-8
    np.testing.assert_allclose(
        standardized.T @ residual + alpha * fitted.coefficients,
        np.zeros(covariates.shape[1]),
        atol=2e-8,
    )
    hand_log_loss = -float(
        np.mean(
            treatment.to_numpy() * np.log(probability)
            + (1.0 - treatment.to_numpy()) * np.log1p(-probability)
        )
    )
    hand_objective = len(covariates) * hand_log_loss + 0.5 * alpha * float(
        fitted.coefficients @ fitted.coefficients
    )
    assert fitted.training_log_loss == pytest.approx(hand_log_loss, abs=2e-12)
    assert fitted.training_objective == pytest.approx(hand_objective, abs=2e-11)
    assert np.all((probability > 0.0) & (probability < 1.0))


def test_native_weighted_ridge_matches_hand_weighted_r_objective() -> None:
    index = pd.Index(["c0", "c1", "c2", "c3"])
    covariates = pd.DataFrame({"x": [-1.0, -1.0, 1.0, 1.0]}, index=index)
    treatment_residual = np.array([-0.8, 0.6, -0.5, 0.7])
    outcome_residual = np.array([-1.2, 0.9, -0.4, 1.8])
    pseudo_outcome = pd.Series(outcome_residual / treatment_residual, index=index)
    weight = pd.Series(treatment_residual**2, index=index)
    alpha = 2.0

    fitted = _NativeWeightedRidgeCV(alphas=(alpha,)).fit(
        covariates,
        pseudo_outcome,
        sample_weight=weight,
    )
    prediction = fitted.predict(covariates)
    raw = covariates.to_numpy(dtype=float)
    centered = (raw - fitted.feature_mean) / fitted.feature_scale
    design = np.column_stack([np.ones(len(raw)), centered])
    penalty = np.diag([0.0, alpha])
    hand_coefficient = np.linalg.solve(
        design.T @ (weight.to_numpy()[:, None] * design) + penalty,
        design.T @ (weight.to_numpy() * pseudo_outcome.to_numpy()),
    )
    hand_prediction = design @ hand_coefficient

    np.testing.assert_allclose(prediction, hand_prediction, atol=2e-14)
    weighted_loss = float(np.sum(weight.to_numpy() * (pseudo_outcome - prediction) ** 2))
    direct_r_loss = float(np.sum((outcome_residual - treatment_residual * prediction) ** 2))
    assert weighted_loss == pytest.approx(direct_r_loss, abs=2e-14)
    assert fitted.weighted_residual_sum_squares == pytest.approx(weighted_loss, abs=2e-14)
    assert fitted.selected_alpha == alpha
    assert fitted.training_index.equals(index)
    assert isinstance(_NativeWeightedRidgeCV(alphas=(alpha,)), WeightedCATEEstimatorProtocol)
    assert isinstance(fitted, CATEResultProtocol)


def test_native_prerequisites_retain_construction_only_leakage_audit_state() -> None:
    construction_index = pd.Index([f"construction-{position}" for position in range(12)])
    evaluation_index = pd.Index(["evaluation-0", "evaluation-1"])
    construction = pd.DataFrame(
        {
            "x": np.linspace(-2.0, 2.0, 12),
            "z": np.tile([-1.0, 0.0, 1.0], 4),
        },
        index=construction_index,
    )
    evaluation = pd.DataFrame(
        {"x": [-10.0, 10.0], "z": [4.0, -4.0]},
        index=evaluation_index,
    )
    treatment = pd.Series(np.tile([0.0, 1.0], 6), index=construction_index)
    probability = _NativePenalizedLogitCV(alphas=(0.1, 1.0), tuning_splits=3, random_state=72).fit(
        construction, treatment
    )
    cate = _NativeWeightedRidgeCV(alphas=(0.1, 1.0)).fit(
        construction,
        pd.Series(np.linspace(-1.0, 2.0, 12), index=construction_index),
        sample_weight=pd.Series(np.linspace(0.2, 1.0, 12), index=construction_index),
    )

    probability.predict_proba(evaluation)
    cate.predict(evaluation)

    assert probability.training_index.equals(construction_index)
    assert probability.tuning_fold.index.equals(construction_index)
    assert cate.training_index.equals(construction_index)
    assert probability.training_index.intersection(evaluation_index).empty
    assert cate.training_index.intersection(evaluation_index).empty


def test_native_penalized_logit_integrates_with_crossfitter_diagnostics() -> None:
    index = pd.Index([f"unit-{position}" for position in range(30)])
    covariates = pd.DataFrame(
        {
            "x": np.linspace(-2.5, 2.5, len(index)),
            "z": np.tile([-1.0, 0.0, 1.0], 10),
        },
        index=index,
    )
    treatment = pd.Series(np.tile([0.0, 1.0], 15), index=index)
    result = CrossFitter(
        propensity_factory=lambda: _NativePenalizedLogitCV(
            alphas=(0.1, 1.0), tuning_splits=2, random_state=281
        ),
        n_splits=3,
        random_state=814,
    ).fit_predict_class_probabilities(covariates, classes=treatment)

    assert result.probabilities.shape == (30, 2)
    assert result.probabilities.index.equals(index)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-15)
    assert len(result.model_diagnostics) == 3
    assert result.model_diagnostics["diagnostics_available"].all()
    assert result.model_diagnostics["alpha_grid_size"].eq(2).all()
    assert result.model_diagnostics["tuning_splits"].eq(2).all()
    assert result.model_diagnostics["converged"].all()


class _UnweightedResult:
    def predict(self, X):
        return np.zeros(len(X))


class _UnweightedEstimator:
    def fit(self, X, y):
        return _UnweightedResult()


class _MalformedWeightedEstimator:
    def fit(self, X, y, *, sample_weight):
        return object()


class _FixedCATEResult:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        return self.prediction


def test_weighted_cate_factory_refuses_missing_weight_support_and_prediction_contract() -> None:
    covariates = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    pseudo_outcome = pd.Series([1.0, 2.0, 3.0])
    weight = pd.Series([0.5, 0.75, 1.0])

    with pytest.raises(TypeError, match="sample_weight"):
        _fit_weighted_cate(
            _UnweightedEstimator,
            covariates,
            pseudo_outcome,
            sample_weight=weight,
        )
    with pytest.raises(TypeError, match=r"predict\(X\)"):
        _fit_weighted_cate(
            _MalformedWeightedEstimator,
            covariates,
            pseudo_outcome,
            sample_weight=weight,
        )


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (np.array([1.0]), "one value per row"),
        (np.array([1.0, np.nan]), "finite"),
        (pd.Series([1.0, 2.0], index=["wrong-0", "wrong-1"]), "index"),
    ],
)
def test_cate_prediction_refuses_malformed_provider_output(prediction, message: str) -> None:
    covariates = pd.DataFrame({"x": [0.0, 1.0]}, index=["evaluation-0", "evaluation-1"])
    with pytest.raises(ValueError, match=message):
        _cate_prediction(_FixedCATEResult(prediction), covariates)


@pytest.mark.parametrize(
    ("treatment", "message"),
    [
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "both arms"),
        ([0.0, 0.0, 0.0, 1.0, 0.5, 1.0], "coded exactly 0 and 1"),
        ([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], "tuning_splits"),
    ],
)
def test_native_penalized_logit_refuses_invalid_binary_training_samples(
    treatment, message: str
) -> None:
    covariates = pd.DataFrame({"x": np.arange(6, dtype=float)})
    with pytest.raises(ValueError, match=message):
        _NativePenalizedLogitCV(alphas=(0.1,), tuning_splits=3, random_state=3).fit(
            covariates, treatment
        )


@pytest.mark.parametrize(
    ("weight", "message"),
    [
        ([1.0, 0.0, 1.0], "strictly positive"),
        ([1.0, -0.1, 1.0], "strictly positive"),
        ([1.0, np.nan, 1.0], "finite"),
    ],
)
def test_native_weighted_ridge_refuses_invalid_weights(weight, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _NativeWeightedRidgeCV(alphas=(1.0,)).fit(
            pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
            [1.0, 2.0, 3.0],
            sample_weight=weight,
        )


def test_native_prerequisites_refuse_index_and_prediction_schema_drift() -> None:
    index = pd.Index([f"row-{position}" for position in range(8)])
    covariates = pd.DataFrame(
        {"x": np.linspace(-1.0, 1.0, 8), "z": np.tile([0.0, 1.0], 4)}, index=index
    )
    treatment = pd.Series(np.tile([0.0, 1.0], 4), index=index)
    shifted = treatment.copy()
    shifted.index = pd.Index([f"other-{position}" for position in range(8)])
    probability_estimator = _NativePenalizedLogitCV(alphas=(1.0,), tuning_splits=2, random_state=4)
    with pytest.raises(ValueError, match="indices must match"):
        probability_estimator.fit(covariates, shifted)

    probability = probability_estimator.fit(covariates, treatment)
    cate = _NativeWeightedRidgeCV(alphas=(1.0,)).fit(
        covariates,
        pd.Series(np.arange(8, dtype=float), index=index),
        sample_weight=pd.Series(np.ones(8), index=index),
    )
    reordered = covariates[["z", "x"]]
    with pytest.raises(ValueError, match="schema exactly"):
        probability.predict_proba(reordered)
    with pytest.raises(ValueError, match="schema exactly"):
        cate.predict(reordered)

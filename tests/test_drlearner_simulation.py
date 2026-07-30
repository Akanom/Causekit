"""Deterministic recovery and double-robustness smoke tests for the honest DR learner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causekit import DRLearner


class _ArmColumnResult:
    def __init__(self, column: str) -> None:
        self.column = column

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self.column]


class _ArmColumnEstimator:
    def __init__(self, control_column: str, treated_column: str) -> None:
        self.control_column = control_column
        self.treated_column = treated_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ArmColumnResult:
        arms = np.unique(X["arm_marker"].to_numpy(dtype=float))
        if np.array_equal(arms, np.array([0.0])):
            return _ArmColumnResult(self.control_column)
        if np.array_equal(arms, np.array([1.0])):
            return _ArmColumnResult(self.treated_column)
        raise AssertionError("Arm-specific outcome fitting mixed treatment arms.")


class _ProbabilityColumnResult:
    classes_ = np.array([0.0, 1.0])

    def __init__(self, column: str) -> None:
        self.column = column

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        treated = X[self.column]
        return pd.DataFrame({0.0: 1.0 - treated, 1.0: treated}, index=X.index)


class _ProbabilityColumnEstimator:
    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ProbabilityColumnResult:
        return _ProbabilityColumnResult(self.column)


def _double_robust_data(seed: int = 202_607_30) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    nobs = 4_000
    x = rng.uniform(-1.0, 1.0, nobs)
    true_propensity = 1.0 / (1.0 + np.exp(-(0.5 * x - 0.2 * x**2)))
    treatment = pd.Series(rng.binomial(1, true_propensity).astype(float), name="treatment")
    true_mu0 = 0.3 + 0.4 * x + 0.2 * x**2
    true_cate = 0.8 + 0.65 * x
    true_mu1 = true_mu0 + true_cate
    outcome = pd.Series(
        np.where(treatment.to_numpy() == 1.0, true_mu1, true_mu0)
        + rng.normal(scale=0.45, size=nobs),
        name="outcome",
    )
    covariates = pd.DataFrame(
        {
            "x": x,
            "x2": x**2,
            "true_mu0": true_mu0,
            "true_mu1": true_mu1,
            "wrong_mu0": -0.4 + 0.1 * x,
            "wrong_mu1": 1.7 - 0.2 * x,
            "true_propensity": true_propensity,
            "wrong_propensity": np.full(nobs, 0.32),
            "arm_marker": treatment,
            "true_cate": true_cate,
        }
    )
    return covariates, treatment, outcome


def _conditional_score_error(result: object, covariates: pd.DataFrame) -> float:
    construction_score = result.construction_dr_score
    evaluation_score = result.evaluation_dr_score
    score = pd.concat([construction_score, evaluation_score]).sort_index()
    truth = covariates.loc[score.index, "true_cate"]
    bins = pd.qcut(covariates.loc[score.index, "x"], q=8, labels=False)
    score_means = score.groupby(bins).mean()
    truth_means = truth.groupby(bins).mean()
    return float(np.max(np.abs(score_means - truth_means)))


def test_dr_score_is_conditionally_correct_with_outcomes_correct_propensity_wrong() -> None:
    covariates, treatment, outcome = _double_robust_data()
    result = DRLearner(
        outcome_factory=lambda: _ArmColumnEstimator("true_mu0", "true_mu1"),
        propensity_factory=lambda: _ProbabilityColumnEstimator("wrong_propensity"),
        n_splits=3,
        evaluation_fraction=0.4,
        random_state=41,
        calibration_groups=4,
        bootstrap_iterations=99,
    ).fit(outcome, treatment=treatment, covariates=covariates)

    assert _conditional_score_error(result, covariates) < 0.12


def test_dr_score_is_conditionally_correct_with_propensity_correct_outcomes_wrong() -> None:
    covariates, treatment, outcome = _double_robust_data(seed=202_607_31)
    result = DRLearner(
        outcome_factory=lambda: _ArmColumnEstimator("wrong_mu0", "wrong_mu1"),
        propensity_factory=lambda: _ProbabilityColumnEstimator("true_propensity"),
        n_splits=3,
        evaluation_fraction=0.4,
        random_state=42,
        calibration_groups=4,
        bootstrap_iterations=99,
    ).fit(outcome, treatment=treatment, covariates=covariates)

    assert _conditional_score_error(result, covariates) < 0.14


def test_native_drlearner_recovers_seeded_linear_cate_out_of_sample() -> None:
    rng = np.random.default_rng(9_071)
    nobs = 1_000
    covariates = pd.DataFrame(rng.normal(size=(nobs, 4)), columns=list("abcd"))
    propensity = 1.0 / (1.0 + np.exp(-(0.35 * covariates["a"] - 0.2 * covariates["b"])))
    treatment = pd.Series(rng.binomial(1, propensity), index=covariates.index)
    true_cate = 0.9 + 0.7 * covariates["a"] - 0.35 * covariates["c"]
    outcome = (
        0.3
        + 0.4 * covariates["b"]
        - 0.2 * covariates["d"]
        + treatment * true_cate
        + rng.normal(scale=0.55, size=nobs)
    )
    result = DRLearner(
        n_splits=3,
        evaluation_fraction=0.4,
        random_state=271,
        calibration_groups=4,
        bootstrap_iterations=99,
        propensity_tuning_splits=2,
    ).fit(outcome, treatment=treatment, covariates=covariates)

    evaluation = result.evaluation_index
    prediction = result.evaluation_cate_predictions.loc[evaluation]
    truth = true_cate.loc[evaluation]
    correlation = float(np.corrcoef(prediction, truth)[0, 1])
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    assert correlation > 0.9
    assert rmse < 0.3
    assert result.honest_dr_loss < result.honest_constant_dr_loss

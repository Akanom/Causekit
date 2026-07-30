"""Deterministic recovery and honest-evaluation simulation gates for the R-learner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import RLearner, RLearnerResult


class _ColumnOutcomeResult:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["outcome_mean"].copy()


class _ColumnOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnOutcomeResult:
        return _ColumnOutcomeResult()


class _ColumnPropensityResult:
    classes_ = np.array([0.0, 1.0])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        treated = X["propensity"].copy()
        return pd.DataFrame({0.0: 1.0 - treated, 1.0: treated}, index=X.index)


class _ColumnPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnPropensityResult:
        return _ColumnPropensityResult()


class _ColumnCATEResult:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["cate_score"].copy()


class _ColumnCATE:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series,
    ) -> _ColumnCATEResult:
        return _ColumnCATEResult()


def _oracle_model(seed: int) -> RLearner:
    return RLearner(
        outcome_factory=_ColumnOutcome,
        propensity_factory=_ColumnPropensity,
        cate_factory=_ColumnCATE,
        n_splits=3,
        evaluation_fraction=0.5,
        random_state=seed,
        calibration_groups=3,
        bootstrap_iterations=99,
        bootstrap_random_state=seed + 10_000,
        overlap_floor=0.02,
    )


def _oracle_simulation(
    seed: int,
    *,
    heterogeneous: bool,
) -> tuple[RLearnerResult, pd.Series]:
    rng = np.random.default_rng(seed)
    nobs = 360
    index = pd.Index([f"sim-{seed}-{position}" for position in range(nobs)])
    x = rng.normal(size=nobs)
    propensity = 1.0 / (1.0 + np.exp(-0.35 * x))
    treatment = pd.Series(rng.binomial(1, propensity), index=index)
    true_cate = pd.Series(1.0 + (0.8 * x if heterogeneous else 0.0), index=index)
    cate_score = true_cate.to_numpy() if heterogeneous else x
    baseline = 0.4 * x - 0.2 * x**2
    outcome_mean = baseline + propensity * true_cate.to_numpy()
    outcome = pd.Series(
        baseline + treatment.to_numpy() * true_cate.to_numpy() + rng.normal(scale=0.65, size=nobs),
        index=index,
    )
    covariates = pd.DataFrame(
        {
            "x": x,
            "outcome_mean": outcome_mean,
            "propensity": propensity,
            "cate_score": cate_score,
        },
        index=index,
    )
    result = _oracle_model(seed).fit(outcome, treatment=treatment, covariates=covariates)
    return result, true_cate


@pytest.mark.simulation
def test_native_rlearner_recovers_linear_cate_ranking_and_scale() -> None:
    rng = np.random.default_rng(20_260_731)
    nobs = 800
    covariates = pd.DataFrame(rng.normal(size=(nobs, 5)), columns=[f"x{i}" for i in range(5)])
    treatment = pd.Series(rng.binomial(1, 0.5, size=nobs), index=covariates.index)
    true_cate = 1.0 + 0.9 * covariates["x0"] - 0.35 * covariates["x1"]
    baseline = 0.4 * covariates["x2"] - 0.2 * covariates["x3"]
    outcome = baseline + true_cate * treatment + rng.normal(scale=0.55, size=nobs)
    result = RLearner(
        n_splits=4,
        evaluation_fraction=0.4,
        random_state=731,
        calibration_groups=4,
        bootstrap_iterations=99,
        propensity_tuning_splits=3,
    ).fit(outcome, treatment=treatment, covariates=covariates)

    evaluation_truth = true_cate.loc[result.evaluation_index]
    prediction = result.evaluation_cate_predictions
    pehe = float(np.sqrt(np.mean((prediction - evaluation_truth) ** 2)))
    correlation = float(np.corrcoef(prediction, evaluation_truth)[0, 1])

    assert pehe < 0.35
    assert correlation > 0.9
    assert result.r_loss_gain > 0.1
    assert result.calibration_coefficients["heterogeneity"] == pytest.approx(1.0, abs=0.35)
    assert result.calibration_tests.loc["heterogeneity=0", "p_value"] < 0.01


@pytest.mark.simulation
def test_honest_calibration_has_seeded_null_power_and_group_band_coverage_smoke() -> None:
    null_rejections = 0
    power_rejections = 0
    null_path_coverage = 0
    power_path_coverage = 0
    repetitions = 20
    for seed in range(repetitions):
        null_result, null_truth = _oracle_simulation(seed, heterogeneous=False)
        power_result, power_truth = _oracle_simulation(seed + 100, heterogeneous=True)
        null_rejections += int(
            null_result.calibration_tests.loc["heterogeneity=0", "p_value"] < 0.05
        )
        power_rejections += int(
            power_result.calibration_tests.loc["heterogeneity=0", "p_value"] < 0.05
        )
        for result, truth, coverage_name in (
            (null_result, null_truth, "null"),
            (power_result, power_truth, "power"),
        ):
            evaluation = result.evaluation_index
            group = result.evaluation_group
            weight = result.evaluation_residuals["weight"]
            targets = []
            for group_number in result.group_effects.index:
                selected = group.eq(group_number)
                targets.append(
                    float(
                        np.average(
                            truth.loc[evaluation].loc[selected],
                            weights=weight.loc[selected],
                        )
                    )
                )
            targets_array = np.asarray(targets)
            covered = bool(
                (
                    (targets_array >= result.group_effects["simultaneous_lower"])
                    & (targets_array <= result.group_effects["simultaneous_upper"])
                ).all()
            )
            if coverage_name == "null":
                null_path_coverage += int(covered)
            else:
                power_path_coverage += int(covered)

    assert null_rejections <= 4
    assert power_rejections >= 18
    assert null_path_coverage >= 16
    assert power_path_coverage >= 16

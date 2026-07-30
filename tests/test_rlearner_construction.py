"""Contracts for honest R-learner roles and construction-only fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit.ml import _HonestRConstruction


class _ColumnOutcomeResult:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["outcome_mean"].copy()


class _ColumnOutcomeEstimator:
    def __init__(self, training_log: list[tuple[str, tuple[object, ...]]]) -> None:
        self.training_log = training_log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnOutcomeResult:
        self.training_log.append(("outcome", tuple(X.index)))
        return _ColumnOutcomeResult()


class _ColumnPropensityResult:
    classes_ = np.array([0.0, 1.0])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        treated = X["propensity"].astype(float)
        return pd.DataFrame({0.0: 1.0 - treated, 1.0: treated}, index=X.index)


class _ColumnPropensityEstimator:
    def __init__(self, training_log: list[tuple[str, tuple[object, ...]]]) -> None:
        self.training_log = training_log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnPropensityResult:
        self.training_log.append(("propensity", tuple(X.index)))
        return _ColumnPropensityResult()


class _WeightedConstantResult:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=X.index)


class _WeightedConstantEstimator:
    def __init__(self, training_log: list[tuple[str, tuple[object, ...]]]) -> None:
        self.training_log = training_log

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series,
    ) -> _WeightedConstantResult:
        self.training_log.append(("cate", tuple(X.index)))
        value = float(np.average(y.to_numpy(dtype=float), weights=sample_weight.to_numpy()))
        return _WeightedConstantResult(value)


def _contract_data(nobs: int = 32) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = pd.Index([f"unit-{position}" for position in range(nobs)])
    x = np.linspace(-1.5, 1.5, nobs)
    treatment = pd.Series(np.tile([0.0, 1.0], nobs // 2), index=index, name="treatment")
    propensity = np.where(x < 0.0, 0.35, 0.65)
    outcome_mean = 1.0 + 0.4 * x
    treatment_residual = treatment.to_numpy() - propensity
    outcome_residual = (1.5 + 0.6 * x) * treatment_residual + 0.1 * np.sin(3.0 * x)
    outcome = pd.Series(outcome_mean + outcome_residual, index=index, name="outcome")
    covariates = pd.DataFrame(
        {
            "x": x,
            "outcome_mean": outcome_mean,
            "propensity": propensity,
        },
        index=index,
    )
    return covariates, treatment, outcome


def _column_construction(
    training_log: list[tuple[str, tuple[object, ...]]],
    *,
    n_splits: int = 3,
    evaluation_fraction: float = 0.25,
    random_state: int = 902,
) -> _HonestRConstruction:
    return _HonestRConstruction(
        outcome_factory=lambda: _ColumnOutcomeEstimator(training_log),
        propensity_factory=lambda: _ColumnPropensityEstimator(training_log),
        cate_factory=lambda: _WeightedConstantEstimator(training_log),
        n_splits=n_splits,
        evaluation_fraction=evaluation_fraction,
        random_state=random_state,
        overlap_floor=0.05,
    )


def test_hand_construction_split_and_cross_fitted_r_objective_contract() -> None:
    covariates, treatment, outcome = _contract_data()
    training_log: list[tuple[str, tuple[object, ...]]] = []
    result = _column_construction(training_log).fit(
        outcome,
        treatment=treatment,
        covariates=covariates,
    )

    construction = result.construction_index
    evaluation = result.evaluation_index
    propensity = covariates.loc[construction, "propensity"]
    outcome_mean = covariates.loc[construction, "outcome_mean"]
    v = treatment.loc[construction] - propensity
    u = outcome.loc[construction] - outcome_mean
    expected_tau = float(np.sum(v * u) / np.sum(v**2))
    expected_objective = float(np.sum((u - v * expected_tau) ** 2))

    assert construction.intersection(evaluation).empty
    assert construction.append(evaluation).sort_values().equals(covariates.index.sort_values())
    assert set(result.sample_role) == {"construction", "evaluation"}
    assert result.sample_role.loc[construction].eq("construction").all()
    assert result.sample_role.loc[evaluation].eq("evaluation").all()
    pd.testing.assert_series_equal(
        result.construction_nuisance_predictions["outcome_mean"],
        outcome_mean.rename("outcome_mean"),
    )
    pd.testing.assert_series_equal(
        result.construction_nuisance_predictions["propensity"],
        propensity.rename("propensity"),
    )
    np.testing.assert_allclose(result.construction_residuals["outcome_residual"], u)
    np.testing.assert_allclose(result.construction_residuals["treatment_residual"], v)
    np.testing.assert_allclose(result.construction_residuals["pseudo_outcome"], u / v)
    np.testing.assert_allclose(result.construction_residuals["weight"], v**2)
    np.testing.assert_allclose(result.construction_cate_predictions, expected_tau)
    np.testing.assert_allclose(result.evaluation_cate_predictions, expected_tau)
    assert result.construction_r_objective == pytest.approx(expected_objective, abs=2e-14)
    assert result.weighted_construction_objective == pytest.approx(expected_objective, abs=2e-14)
    assert result.split_conditional is True

    evaluation_set = set(evaluation)
    assert training_log
    assert all(set(indices).isdisjoint(evaluation_set) for _, indices in training_log)


def test_role_assignments_are_deterministic_and_externally_immutable() -> None:
    covariates, treatment, outcome = _contract_data()
    first = _column_construction([], random_state=61).fit(
        outcome, treatment=treatment, covariates=covariates
    )
    second = _column_construction([], random_state=61).fit(
        outcome, treatment=treatment, covariates=covariates
    )

    pd.testing.assert_series_equal(first.sample_role, second.sample_role)
    pd.testing.assert_series_equal(first.construction_fold, second.construction_fold)
    exposed = first.sample_role
    exposed.iloc[0] = "evaluation" if exposed.iloc[0] == "construction" else "construction"
    assert not exposed.equals(first.sample_role)


def test_cluster_roles_and_outer_folds_never_split_a_cluster() -> None:
    n_clusters = 16
    rows_per_cluster = 2
    nobs = n_clusters * rows_per_cluster
    covariates, _, _ = _contract_data(nobs)
    cluster = pd.Series(
        np.repeat([f"cluster-{position}" for position in range(n_clusters)], rows_per_cluster),
        index=covariates.index,
        name="cluster",
    )
    cluster_treatment = np.tile([0.0, 1.0], n_clusters // 2)
    treatment = pd.Series(np.repeat(cluster_treatment, rows_per_cluster), index=covariates.index)
    propensity = np.where(covariates["x"] < 0.0, 0.35, 0.65)
    covariates["propensity"] = propensity
    v = treatment.to_numpy() - propensity
    outcome = pd.Series(
        covariates["outcome_mean"].to_numpy() + (1.0 + covariates["x"].to_numpy()) * v,
        index=covariates.index,
    )
    result = _column_construction([], n_splits=2, evaluation_fraction=0.25, random_state=117).fit(
        outcome,
        treatment=treatment,
        covariates=covariates,
        clusters=cluster,
    )

    role_by_cluster = result.sample_role.groupby(cluster).nunique()
    assert role_by_cluster.eq(1).all()
    construction_clusters = cluster.loc[result.construction_index]
    fold_by_cluster = result.construction_fold.groupby(construction_clusters).nunique()
    assert fold_by_cluster.eq(1).all()
    assert set(cluster.loc[result.construction_index]).isdisjoint(
        set(cluster.loc[result.evaluation_index])
    )
    assert set(result.cluster_role) == {"construction", "evaluation"}
    exposed = result.cluster_role
    exposed.iloc[0] = "evaluation" if exposed.iloc[0] == "construction" else "construction"
    assert not exposed.equals(result.cluster_role)


def test_native_construction_path_returns_finite_nonconstant_cate_predictions() -> None:
    rng = np.random.default_rng(20_260_730)
    nobs = 240
    index = pd.Index([f"row-{position}" for position in range(nobs)])
    covariates = pd.DataFrame(
        rng.normal(size=(nobs, 4)), columns=["x0", "x1", "x2", "x3"], index=index
    )
    propensity = 1.0 / (1.0 + np.exp(-(0.4 * covariates["x0"] - 0.2 * covariates["x1"])))
    treatment = pd.Series(rng.binomial(1, propensity), index=index)
    tau = 1.0 + 0.8 * covariates["x0"]
    outcome = 0.5 * covariates["x1"] + tau * treatment + rng.normal(scale=0.4, size=nobs)
    result = _HonestRConstruction(
        n_splits=3,
        evaluation_fraction=0.3,
        random_state=730,
        overlap_floor=0.01,
        propensity_tuning_splits=2,
    ).fit(outcome, treatment=treatment, covariates=covariates)

    assert np.isfinite(result.construction_cate_predictions).all()
    assert np.isfinite(result.evaluation_cate_predictions).all()
    assert result.evaluation_cate_predictions.std() > 0.05
    assert result.construction_r_objective == pytest.approx(
        result.weighted_construction_objective, rel=2e-14
    )
    assert result.construction_nuisance_predictions.index.equals(result.construction_index)
    assert result.evaluation_nuisance_predictions.index.equals(result.evaluation_index)
    assert result.nuisance_diagnostics["diagnostics_available"].all()
    assert result.cate_diagnostics["diagnostics_available"].all()


def test_nonfresh_nuisance_factory_is_refused() -> None:
    covariates, treatment, outcome = _contract_data()
    training_log: list[tuple[str, tuple[object, ...]]] = []
    singleton = _ColumnOutcomeEstimator(training_log)
    estimator = _HonestRConstruction(
        outcome_factory=lambda: singleton,
        propensity_factory=lambda: _ColumnPropensityEstimator(training_log),
        cate_factory=lambda: _WeightedConstantEstimator(training_log),
        n_splits=3,
        evaluation_fraction=0.25,
        random_state=14,
    )
    with pytest.raises(ValueError, match="fresh estimator"):
        estimator.fit(outcome, treatment=treatment, covariates=covariates)


def test_overlap_violation_refuses_without_clipping() -> None:
    covariates, treatment, outcome = _contract_data()
    covariates["propensity"] = 0.99
    with pytest.raises(ValueError, match="overlap interval"):
        _column_construction([], random_state=6).fit(
            outcome,
            treatment=treatment,
            covariates=covariates,
        )


def test_role_split_refuses_insufficient_arm_and_cluster_capacity() -> None:
    covariates, treatment, outcome = _contract_data(8)
    with pytest.raises(ValueError, match="construction role.*n_splits"):
        _column_construction([], n_splits=3, evaluation_fraction=0.5).fit(
            outcome,
            treatment=treatment,
            covariates=covariates,
        )

    covariates, treatment, outcome = _contract_data(12)
    cluster = pd.Series(np.repeat(["a", "b", "c"], 4), index=covariates.index)
    with pytest.raises(ValueError, match="clusters"):
        _column_construction([], n_splits=2, evaluation_fraction=0.5).fit(
            outcome,
            treatment=treatment,
            covariates=covariates,
            clusters=cluster,
        )


def test_role_split_refuses_nonbinary_index_drift_and_duplicate_row_identity() -> None:
    covariates, treatment, outcome = _contract_data()
    invalid_treatment = treatment.copy()
    invalid_treatment.iloc[0] = 0.5
    with pytest.raises(ValueError, match="coded exactly 0 and 1"):
        _column_construction([]).fit(
            outcome,
            treatment=invalid_treatment,
            covariates=covariates,
        )

    shifted_cluster = pd.Series("a", index=pd.RangeIndex(1, len(covariates) + 1))
    with pytest.raises(ValueError, match="indices must match"):
        _column_construction([]).fit(
            outcome,
            treatment=treatment,
            covariates=covariates,
            clusters=shifted_cluster,
        )

    duplicate = covariates.copy()
    duplicate.index = pd.Index(["duplicate"] * len(duplicate))
    with pytest.raises(ValueError, match="unique row index"):
        _column_construction([]).fit(
            outcome.to_numpy(),
            treatment=treatment.to_numpy(),
            covariates=duplicate,
        )

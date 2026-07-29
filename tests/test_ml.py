"""Contract tests for the specialized causal machine-learning layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causekit import PartiallyLinearDML, PartiallyLinearDMLResult


def test_native_ml_source_has_no_external_ml_backend_import() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "causekit" / "ml.py").read_text(
        encoding="utf-8"
    )
    assert "import sklearn" not in source
    assert "from sklearn" not in source
    assert "import statsmodels" not in source
    assert "from statsmodels" not in source


class _ColumnResult:
    def __init__(self, column: str) -> None:
        self.column = column

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(dtype=float)


class _ColumnRegressor:
    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnResult:
        assert len(X) == len(y)
        return _ColumnResult(self.column)


class _OLSResult:
    def __init__(self, coefficients: np.ndarray, columns: tuple[str, ...]) -> None:
        self.coefficients = coefficients
        self.columns = columns

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        design = np.column_stack(
            [np.ones(len(X)), X.loc[:, list(self.columns)].to_numpy(dtype=float)]
        )
        return design @ self.coefficients


class _OLSRegressor:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _OLSResult:
        columns = tuple(str(column) for column in X.columns)
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        coefficients = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)[0]
        return _OLSResult(coefficients, columns)


def _column_factory(column: str):
    return lambda: _ColumnRegressor(column)


def _hand_contract(*, covariance: str = "robust", clusters=None):
    index = pd.Index([f"row-{position}" for position in range(8)])
    outcome_mean = np.array([1.0, 0.5, 1.5, -0.5, 0.0, 2.0, -1.0, 1.25])
    treatment_mean = np.array([0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.1, 0.5])
    treatment_residual = np.array([-1.0, 1.0, -2.0, 2.0, -1.5, 1.5, -0.5, 0.5])
    raw_error = np.array([0.2, -0.4, 0.1, 0.3, -0.2, 0.5, -0.1, -0.4])
    orthogonal_error = raw_error - treatment_residual * (
        treatment_residual @ raw_error / (treatment_residual @ treatment_residual)
    )
    outcome_residual = 1.75 * treatment_residual + orthogonal_error
    covariates = pd.DataFrame(
        {"outcome_mean": outcome_mean, "treatment_mean": treatment_mean}, index=index
    )
    outcome = pd.Series(outcome_mean + outcome_residual, index=index, name="outcome")
    treatment = pd.Series(treatment_mean + treatment_residual, index=index, name="treatment")
    fitted = PartiallyLinearDML(
        outcome_factory=_column_factory("outcome_mean"),
        treatment_factory=_column_factory("treatment_mean"),
        n_splits=2,
        random_state=41,
        covariance=covariance,
    ).fit(outcome, treatment=treatment, covariates=covariates, clusters=clusters)
    return fitted, outcome_residual, treatment_residual


def test_hand_computed_dml2_score_influence_and_hc1_contract() -> None:
    result, outcome_residual, treatment_residual = _hand_contract()
    expected = float(
        treatment_residual @ outcome_residual / (treatment_residual @ treatment_residual)
    )
    second_stage_residual = outcome_residual - expected * treatment_residual
    jacobian = float(np.mean(treatment_residual**2))
    influence = treatment_residual * second_stage_residual / jacobian
    expected_variance = float(influence @ influence) / (len(influence) * (len(influence) - 1))

    assert isinstance(result, PartiallyLinearDMLResult)
    assert result.estimate == pytest.approx(expected, abs=1e-14)
    assert result.estimate == pytest.approx(1.75, abs=1e-14)
    assert result.residual_treatment_second_moment == pytest.approx(jacobian, abs=1e-14)
    np.testing.assert_allclose(result.influence_function, influence, atol=1e-14)
    assert result.standard_error**2 == pytest.approx(expected_variance, abs=1e-14)
    assert result.orthogonal_score_mean == pytest.approx(0.0, abs=1e-14)
    assert result.params.index.tolist() == ["theta"]
    assert result.covariance.index.tolist() == ["theta"]
    assert result.inference_distribution == "normal"
    assert result.treatment_kind == "continuous"
    assert result.backend == "native-cross-fitted-orthogonal-score"
    assert result.nuisance_predictions.columns.tolist() == [
        "outcome_mean",
        "treatment_mean",
    ]
    assert result.fold.index.equals(result.estimation_index)


def test_clustered_dml_aggregates_the_same_influence_function() -> None:
    index = pd.Index([f"row-{position}" for position in range(8)])
    clusters = pd.Series(np.repeat(["a", "b", "c", "d"], 2), index=index)
    result, _, _ = _hand_contract(covariance="clustered", clusters=clusters)
    grouped = result.influence_function.groupby(clusters).sum().to_numpy(dtype=float)
    expected_variance = 4 / 3 * float(grouped @ grouped) / result.nobs**2

    assert result.standard_error**2 == pytest.approx(expected_variance, abs=1e-14)
    assert result.inference_distribution == "t"
    assert result.inference_df == 3
    assert result.n_clusters == 4


def test_binary_treatment_uses_shared_stratified_folds() -> None:
    index = pd.Index([f"unit-{position}" for position in range(18)])
    treatment = pd.Series(np.tile([0.0, 1.0], 9), index=index)
    treatment_mean = np.full(len(index), 0.5)
    outcome_mean = np.full(len(index), 0.75)
    error = np.linspace(-0.4, 0.4, len(index))
    outcome = pd.Series(
        outcome_mean + 1.5 * (treatment.to_numpy() - treatment_mean) + error,
        index=index,
    )
    covariates = pd.DataFrame(
        {"outcome_mean": outcome_mean, "treatment_mean": treatment_mean}, index=index
    )
    result = PartiallyLinearDML(
        outcome_factory=_column_factory("outcome_mean"),
        treatment_factory=_column_factory("treatment_mean"),
        n_splits=3,
        random_state=7,
    ).fit(outcome, treatment=treatment, covariates=covariates)

    fold_arms = pd.crosstab(result.fold, treatment)
    assert (fold_arms > 0).all().all()
    assert result.treatment_kind == "binary"
    assert result.n_splits == 3
    assert result.random_state == 7


def test_native_ridge_gcv_is_the_dependency_free_default() -> None:
    rng = np.random.default_rng(824)
    nobs = 180
    covariates = pd.DataFrame(rng.normal(size=(nobs, 4)), columns=list("abcd"))
    treatment = 0.6 * covariates["a"] - 0.2 * covariates["b"] + rng.normal(size=nobs)
    outcome = 1.4 * treatment + 0.3 * covariates["c"] + rng.normal(size=nobs)

    result = PartiallyLinearDML(n_splits=3, random_state=11).fit(
        outcome, treatment=treatment, covariates=covariates
    )

    assert result.outcome_model_name == "NativeRidgeCVResult"
    assert result.treatment_model_name == "NativeRidgeCVResult"
    assert result.native_nuisance is True
    assert result.estimate == pytest.approx(1.4, abs=0.2)


@pytest.mark.simulation
def test_cross_fitted_partially_linear_dml_recovers_continuous_effect() -> None:
    rng = np.random.default_rng(20_260_729)
    nobs = 5000
    raw = rng.normal(size=(nobs, 3))
    covariates = pd.DataFrame(
        {
            "x0": raw[:, 0],
            "x1": raw[:, 1],
            "x2": raw[:, 2],
            "x0_sq": raw[:, 0] ** 2,
        }
    )
    treatment_mean = 0.5 * raw[:, 0] - 0.3 * raw[:, 1] + 0.2 * raw[:, 0] ** 2
    treatment = treatment_mean + rng.normal(size=nobs)
    baseline = 0.4 * raw[:, 0] + 0.7 * raw[:, 2] - 0.25 * raw[:, 0] ** 2
    outcome = 2.1 * treatment + baseline + rng.normal(size=nobs)

    result = PartiallyLinearDML(
        outcome_factory=_OLSRegressor,
        treatment_factory=_OLSRegressor,
        n_splits=5,
        random_state=2026,
    ).fit(outcome, treatment=treatment, covariates=covariates)

    assert result.estimate == pytest.approx(2.1, abs=0.05)
    assert 0.0 < result.standard_error < 0.04
    assert result.nobs == nobs


@pytest.mark.parametrize(
    ("covariance", "clusters", "message"),
    [
        ("clustered", None, "clusters must be provided"),
        ("robust", [0, 0, 1, 1], "only when covariance='clustered'"),
        ("not-a-covariance", None, "covariance must be"),
    ],
)
def test_covariance_contract_refuses_invalid_requests(
    covariance: str, clusters, message: str
) -> None:
    model_kwargs = {
        "outcome_factory": _column_factory("outcome_mean"),
        "treatment_factory": _column_factory("treatment_mean"),
        "n_splits": 2,
        "covariance": covariance,
    }
    if covariance == "not-a-covariance":
        with pytest.raises(ValueError, match=message):
            PartiallyLinearDML(**model_kwargs)
        return
    covariates = pd.DataFrame({"outcome_mean": [0.0] * 4, "treatment_mean": [0.0] * 4})
    with pytest.raises(ValueError, match=message):
        PartiallyLinearDML(**model_kwargs).fit(
            [0.0, 1.0, 1.5, 3.0],
            treatment=[-1.0, 1.0, -2.0, 2.0],
            covariates=covariates,
            clusters=clusters,
        )


def test_alignment_fold_capacity_and_residual_identification_refusals() -> None:
    index = pd.Index(["a", "b", "c", "d"])
    covariates = pd.DataFrame(
        {
            "outcome_mean": [0.0, 0.0, 0.0, 0.0],
            "treatment_mean": [0.0, 1.0, 2.0, 3.0],
        },
        index=index,
    )
    estimator = PartiallyLinearDML(
        outcome_factory=_column_factory("outcome_mean"),
        treatment_factory=_column_factory("treatment_mean"),
        n_splits=2,
    )
    with pytest.raises(ValueError, match="indices must match"):
        estimator.fit(
            pd.Series([0.0, 1.0, 2.0, 3.0]),
            treatment=pd.Series([0.0, 1.0, 2.0, 3.0], index=index),
            covariates=covariates,
        )
    with pytest.raises(ValueError, match="residual treatment variation"):
        estimator.fit(
            pd.Series([0.0, 1.0, 2.0, 3.0], index=index),
            treatment=pd.Series([0.0, 1.0, 2.0, 3.0], index=index),
            covariates=covariates,
        )
    with pytest.raises(ValueError, match="at least n_splits"):
        PartiallyLinearDML(
            outcome_factory=_column_factory("outcome_mean"),
            treatment_factory=_column_factory("treatment_mean"),
            n_splits=5,
        ).fit(
            pd.Series([0.0, 1.0, 2.0, 3.0], index=index),
            treatment=pd.Series([-1.0, 1.0, -2.0, 2.0], index=index),
            covariates=covariates,
        )


def test_invalid_nuisance_factory_refuses_without_in_sample_fallback() -> None:
    covariates = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    with pytest.raises(TypeError, match=r"fit\(X, y\)"):
        PartiallyLinearDML(
            outcome_factory=lambda: object(),
            treatment_factory=_column_factory("x"),
            n_splits=2,
        ).fit(
            [0.0, 1.0, 2.0, 3.0],
            treatment=[-1.0, 1.0, -2.0, 2.0],
            covariates=covariates,
        )

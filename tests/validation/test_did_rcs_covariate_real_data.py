"""Hash-verified real-data smoke for covariate repeated-cross-section DiD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import CrossFitter, RepeatedCrossSectionDiD
from causekit.datasets import load_real_dataset


class _EmpiricalPropensityResult:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probability = np.full(len(X), self.probability)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _EmpiricalPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _EmpiricalPropensityResult:
        del X
        return _EmpiricalPropensityResult(float(y.mean()))


class _AffineResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        return design @ self.coefficients


class _AffineOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _AffineResult:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        coefficients, _, _, _ = np.linalg.lstsq(
            design,
            y.to_numpy(dtype=float),
            rcond=None,
        )
        return _AffineResult(coefficients)


@pytest.mark.validation
def test_hash_verified_hospdd_covariate_path_is_reproducible_and_cluster_audited() -> None:
    try:
        source = load_real_dataset("hospdd", download=False)
    except FileNotFoundError:
        pytest.skip("Run `python benchmarks/prepare_real_data.py --download` first.")
    first_treated = (
        source.loc[source["procedure"].eq(1)].groupby("hospital", sort=False)["month"].min()
    )
    data = source.rename(columns={"satis": "outcome", "frequency": "x"}).copy()
    data["treatment_time"] = data["hospital"].map(first_treated).fillna(np.inf)
    result = RepeatedCrossSectionDiD(covariance="clustered").fit(
        data,
        outcome="outcome",
        time="month",
        treatment_time="treatment_time",
        cluster="hospital",
        covariates=["x"],
        cross_fitter=CrossFitter(
            propensity_factory=_EmpiricalPropensity,
            outcome_factory=_AffineOutcome,
            n_splits=2,
            random_state=20_260_730,
        ),
    )

    assert result.estimate == pytest.approx(0.8676486723638247, abs=1e-10)
    assert result.standard_error == pytest.approx(0.04274270577554287, abs=1e-10)
    assert result.nobs == 7_368
    assert result.n_clusters == 46
    assert result.cross_fitted
    assert result.pretrend.conditional
    assert result.pretrend.available
    assert result.nuisance_fold.groupby(data["hospital"]).nunique().eq(1).all()
    assert result.nuisance_diagnostics.shape[0] == 60
    assert result.nuisance_diagnostics["relevant_holdout_nobs"].gt(0).all()
    assert (
        result.nuisance_diagnostics["relevant_holdout_nobs"]
        <= result.nuisance_diagnostics["holdout_nobs"]
    ).all()
    prediction_values = result.nuisance_predictions.to_numpy(dtype=float)
    assert np.isfinite(prediction_values[~np.isnan(prediction_values)]).all()
    assert np.isnan(prediction_values).any()  # irrelevant comparison rows are not reported

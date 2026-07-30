"""Tests for randomized-experiment ATE estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import RandomizedATE, RandomizedATEResult


def _experiment(nobs: int = 800, seed: int = 20260728):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(nobs, 2))
    treatment = rng.binomial(1, 0.5, size=nobs)
    outcome = 1.0 + 1.75 * treatment + x @ np.array([0.8, -0.5]) + rng.normal(size=nobs)
    index = pd.Index([f"unit-{i}" for i in range(nobs)])
    return (
        pd.Series(outcome, index=index, name="outcome"),
        pd.Series(treatment, index=index, name="assigned"),
        pd.DataFrame(x, index=index, columns=["age", "baseline"]),
    )


def test_unadjusted_estimate_is_exact_difference_in_means() -> None:
    y, treatment, covariates = _experiment()
    result = RandomizedATE(adjustment="none").fit(y, treatment=treatment, covariates=covariates)
    expected = y[treatment == 1].mean() - y[treatment == 0].mean()
    assert isinstance(result, RandomizedATEResult)
    assert result.estimate == pytest.approx(expected, abs=1e-12)
    assert result.n_treated + result.n_control == result.nobs
    assert len(result.balance) == 2
    assert "only for balance" in result.notes[0]


@pytest.mark.simulation
def test_lin_adjustment_recovers_ate_and_improves_precision() -> None:
    y, treatment, covariates = _experiment(nobs=4000)
    unadjusted = RandomizedATE().fit(y, treatment=treatment)
    adjusted = RandomizedATE(adjustment="lin").fit(y, treatment=treatment, covariates=covariates)
    assert adjusted.estimate == pytest.approx(1.75, abs=0.07)
    assert adjusted.standard_error < unadjusted.standard_error
    assert adjusted.params.index.tolist() == [
        "const",
        "ate",
        "age",
        "baseline",
        "treatment:age",
        "treatment:baseline",
    ]


@pytest.mark.validation
@pytest.mark.parametrize("adjustment", ["none", "lin"])
def test_coefficients_and_hc1_covariance_match_statsmodels(adjustment: str) -> None:
    sm = pytest.importorskip("statsmodels.api")
    y, treatment, covariates = _experiment(nobs=700, seed=414)
    result = RandomizedATE(adjustment=adjustment).fit(
        y, treatment=treatment, covariates=covariates if adjustment == "lin" else None
    )
    assigned = treatment.to_numpy(dtype=float)
    pieces = [np.ones((len(y), 1)), assigned[:, None]]
    if adjustment == "lin":
        centered = covariates.to_numpy() - covariates.to_numpy().mean(axis=0)
        pieces.extend([centered, centered * assigned[:, None]])
    reference = sm.OLS(y.to_numpy(), np.column_stack(pieces)).fit(cov_type="HC1")
    np.testing.assert_allclose(result.params, reference.params, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.covariance, reference.cov_params(), rtol=1e-9, atol=1e-12)


def test_clustered_inference_and_joint_missing_policy() -> None:
    y, treatment, covariates = _experiment(nobs=240)
    clusters = pd.Series(np.repeat(np.arange(24), 10), index=y.index, dtype=float)
    y.iloc[0] = np.nan
    clusters.iloc[1] = np.nan
    result = RandomizedATE(adjustment="lin", covariance="clustered", missing="drop").fit(
        y, treatment=treatment, covariates=covariates, clusters=clusters
    )
    assert result.nobs == 238
    assert result.dropped_rows == 2
    assert result.n_clusters == 24
    assert result.inference_df == 23


@pytest.mark.parametrize(
    ("model", "kwargs", "message"),
    [
        (RandomizedATE(adjustment="lin"), {}, "covariates"),
        (RandomizedATE(covariance="clustered"), {}, "clusters"),
    ],
)
def test_required_design_inputs_are_enforced(model, kwargs, message) -> None:
    y, treatment, _ = _experiment(nobs=80)
    with pytest.raises(ValueError, match=message):
        model.fit(y, treatment=treatment, **kwargs)


def test_treatment_must_be_binary_and_indices_must_align() -> None:
    y, treatment, covariates = _experiment(nobs=80)
    with pytest.raises(ValueError, match="coded exactly 0 and 1"):
        RandomizedATE().fit(y, treatment=treatment.replace({1: 2}))
    with pytest.raises(ValueError, match="indices must match"):
        RandomizedATE(adjustment="lin").fit(
            y, treatment=treatment, covariates=covariates.reset_index(drop=True)
        )


def test_result_summary_and_markdown_expose_ate_not_nuisance_coefficients() -> None:
    y, treatment, covariates = _experiment(nobs=200)
    result = RandomizedATE(adjustment="lin").fit(y, treatment=treatment, covariates=covariates)
    assert result.summary_frame().index.tolist() == ["ate"]
    assert result.conf_int().index.tolist() == ["lower", "upper"]
    assert "Randomized-experiment ATE" in result.to_markdown()

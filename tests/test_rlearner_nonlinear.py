"""Contracts for the opt-in native nonlinear weighted-CATE learner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import NativeSplineRidgeCATE, RLearner


def test_hand_computed_piecewise_linear_basis_and_tuning_audit() -> None:
    covariates = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0]})
    target = pd.Series([0.0, 0.0, 0.0, 1.0, 2.0])
    weight = pd.Series(np.ones(len(covariates)))
    result = NativeSplineRidgeCATE(
        knot_counts=(1,),
        alphas=(1e-6, 1e-2),
    ).fit(covariates, target, sample_weight=weight)

    scale = np.sqrt(2.0)
    expected = pd.DataFrame(
        {
            "linear::0::x": covariates["x"] / scale,
            "hinge::0::x::1": np.maximum(covariates["x"] / scale, 0.0),
        },
        index=covariates.index,
    )
    pd.testing.assert_frame_equal(result.basis_data(covariates), expected)
    assert result.selected_knot_count == 1
    assert result.selected_knots == ((0.0,),)
    assert result.basis_dimension == 2
    assert result.training_index.equals(covariates.index)
    assert result.tuning_path["selected"].sum() == 1
    assert result.nuisance_diagnostics()["basis_family"] == "additive_linear_spline"


def test_native_spline_cate_recovers_nonlinearity_better_than_linear_default() -> None:
    rng = np.random.default_rng(20_260_729)
    nobs = 700
    x = rng.uniform(-2.0, 2.0, size=nobs)
    covariates = pd.DataFrame({"x": x})
    treatment = pd.Series(rng.binomial(1, 0.5, size=nobs), index=covariates.index)
    true_cate = pd.Series(0.5 + 1.6 * np.maximum(x, 0.0), index=covariates.index)
    outcome = (
        0.3 * x + treatment.to_numpy() * true_cate.to_numpy() + rng.normal(scale=0.35, size=nobs)
    )
    common = dict(
        n_splits=4,
        evaluation_fraction=0.4,
        random_state=729,
        calibration_groups=4,
        bootstrap_iterations=99,
        propensity_tuning_splits=3,
    )
    linear = RLearner(**common).fit(outcome, treatment=treatment, covariates=covariates)
    nonlinear = RLearner(
        **common,
        cate_factory=lambda: NativeSplineRidgeCATE(knot_counts=(0, 1, 3)),
    ).fit(outcome, treatment=treatment, covariates=covariates)
    truth = true_cate.loc[nonlinear.evaluation_index]
    linear_pehe = float(
        np.sqrt(np.mean((linear.evaluation_cate_predictions - truth.to_numpy()) ** 2))
    )
    nonlinear_pehe = float(
        np.sqrt(np.mean((nonlinear.evaluation_cate_predictions - truth.to_numpy()) ** 2))
    )

    assert nonlinear_pehe < 0.65 * linear_pehe
    assert nonlinear.honest_r_loss < linear.honest_r_loss
    assert nonlinear.cate_diagnostics.loc[0, "basis_family"] == "additive_linear_spline"
    assert nonlinear.cate_diagnostics.loc[0, "selected_knot_count"] >= 1


def test_native_spline_cate_refuses_invalid_complexity_and_schema_drift() -> None:
    with pytest.raises(ValueError, match="knot_counts"):
        NativeSplineRidgeCATE(knot_counts=())
    with pytest.raises(ValueError, match="knot_counts"):
        NativeSplineRidgeCATE(knot_counts=(0, -1))
    with pytest.raises(ValueError, match="max_basis_features"):
        NativeSplineRidgeCATE(max_basis_features=0)

    covariates = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20), "z": np.arange(20.0)})
    target = pd.Series(np.sin(covariates["x"]))
    weight = pd.Series(np.ones(len(covariates)))
    with pytest.raises(ValueError, match="max_basis_features"):
        NativeSplineRidgeCATE(
            knot_counts=(3,),
            include_pairwise_interactions=True,
            max_basis_features=2,
        ).fit(covariates, target, sample_weight=weight)

    result = NativeSplineRidgeCATE(knot_counts=(1,)).fit(
        covariates,
        target,
        sample_weight=weight,
    )
    with pytest.raises(ValueError, match="columns must match"):
        result.predict(covariates[["z", "x"]])

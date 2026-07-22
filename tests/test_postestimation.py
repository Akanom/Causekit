"""Contract tests for common IV post-estimation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalkit import (
    IV2SLS,
    confint,
    fitted_values,
    lincom,
    predict,
    residuals,
    summary_frame,
    vcov,
    wald_test,
)


@pytest.fixture(scope="module")
def fitted_iv() -> tuple[object, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(4_112)
    nobs = 600
    index = pd.Index([f"observation-{number}" for number in range(nobs)])
    z = rng.normal(size=nobs)
    control = rng.normal(size=nobs)
    first_stage_error = rng.normal(size=nobs)
    treatment = 0.8 * z + 0.3 * control + first_stage_error
    outcome = 0.9 + 0.5 * control + 1.7 * treatment + 0.6 * first_stage_error
    endogenous = pd.DataFrame({"treatment": treatment}, index=index)
    exogenous = pd.DataFrame({"control": control}, index=index)
    result = IV2SLS(covariance="robust").fit(
        pd.Series(outcome, index=index, name="outcome"),
        endogenous=endogenous,
        instruments=pd.DataFrame({"z": z}, index=index),
        exogenous=exogenous,
    )
    return result, endogenous, exogenous


def test_summary_frame_has_one_labeled_row_per_parameter(fitted_iv) -> None:
    result, _, _ = fitted_iv

    table = result.summary_frame()

    assert table.index.equals(result.params.index)
    assert list(table.columns) == [
        "coef",
        "std_err",
        "stat",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]
    assert table.shape == (len(result.params), 6)
    np.testing.assert_allclose(table.iloc[:, 0], result.params)
    np.testing.assert_allclose(table.iloc[:, 1], result.standard_errors)
    np.testing.assert_allclose(table.iloc[:, 2], result.test_statistics)
    np.testing.assert_allclose(table.iloc[:, 3], result.pvalues)
    assert (table["ci_lower"] < result.params).all()
    assert (table["ci_upper"] > result.params).all()
    pd.testing.assert_frame_equal(summary_frame(result), table)


def test_covariance_and_fitted_vector_helpers_return_defensive_copies(fitted_iv) -> None:
    result, _, _ = fitted_iv

    covariance = vcov(result)
    fitted = fitted_values(result)
    errors = residuals(result)

    pd.testing.assert_frame_equal(covariance, result.covariance)
    pd.testing.assert_series_equal(fitted, result.fitted_values)
    pd.testing.assert_series_equal(errors, result.residuals)
    assert covariance is not result.covariance
    assert fitted is not result.fitted_values
    assert errors is not result.residuals


def test_confidence_interval_wrapper_delegates_and_validates_level(fitted_iv) -> None:
    result, _, _ = fitted_iv

    intervals = confint(result, level=0.90)

    pd.testing.assert_frame_equal(intervals, result.conf_int(level=0.90))
    assert intervals.index.equals(result.params.index)
    assert intervals.shape == (len(result.params), 2)
    assert (intervals.iloc[:, 0] < result.params).all()
    assert (intervals.iloc[:, 1] > result.params).all()
    with pytest.raises(ValueError, match="(?i)level"):
        confint(result, level=1.0)


def test_prediction_wrapper_preserves_new_data_index(fitted_iv) -> None:
    result, endogenous, exogenous = fitted_iv
    sample_endogenous = endogenous.iloc[:12].copy()
    sample_exogenous = exogenous.iloc[:12].copy()

    direct = result.predict(sample_endogenous, exogenous=sample_exogenous)
    wrapped = predict(result, sample_endogenous, exogenous=sample_exogenous)

    pd.testing.assert_series_equal(wrapped, direct)
    assert wrapped.index.equals(sample_endogenous.index)


def test_markdown_summary_is_self_contained_and_identifies_covariance(fitted_iv) -> None:
    result, _, _ = fitted_iv

    rendered = result.to_markdown()

    assert isinstance(rendered, str)
    assert "treatment" in rendered
    assert "control" in rendered
    assert "robust" in rendered.lower()


def test_named_linear_hypotheses_use_the_full_covariance(fitted_iv) -> None:
    result, _, _ = fitted_iv

    combination = lincom(result, {"treatment": 1.0, "control": -1.0})
    expected_estimate = float(result.params["treatment"] - result.params["control"])
    weights = np.array([0.0, -1.0, 1.0])
    expected_variance = float(weights @ result.covariance.to_numpy() @ weights)

    assert combination["estimate"] == pytest.approx(expected_estimate)
    assert combination["standard_error"] == pytest.approx(np.sqrt(expected_variance))
    assert 0.0 <= combination["p_value"] <= 1.0

    joint = wald_test(
        result,
        [{"treatment": 1.0}, {"control": 1.0}],
        values=[0.0, 0.0],
    )
    assert joint["df_num"] == 2
    assert joint["distribution"] == "chi2(2)"
    assert 0.0 <= joint["p_value"] <= 1.0

    with pytest.raises(ValueError, match="Unknown"):
        lincom(result, {"missing_parameter": 1.0})

"""Behavioral tests for the native two-stage least-squares estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import IV2SLS


def _iv_sample(
    *, seed: int = 20_260_722, nobs: int = 2_500
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate an identified IV design with genuinely endogenous treatment."""
    rng = np.random.default_rng(seed)
    index = pd.Index(np.arange(10_000, 10_000 + nobs), name="row")
    control = rng.normal(size=nobs)
    z1 = rng.normal(size=nobs)
    z2 = rng.normal(size=nobs)
    first_stage_error = rng.normal(size=nobs)
    treatment = 0.9 * z1 - 0.55 * z2 + 0.35 * control + first_stage_error
    structural_error = 0.7 * first_stage_error + rng.normal(scale=0.55, size=nobs)
    outcome = 1.4 + 0.65 * control + 2.1 * treatment + structural_error

    return (
        pd.Series(outcome, index=index, name="outcome"),
        pd.DataFrame({"treatment": treatment}, index=index),
        pd.DataFrame({"z1": z1, "z2": z2}, index=index),
        pd.DataFrame({"control": control}, index=index),
    )


def test_iv2sls_recovers_coefficients_and_exposes_coherent_result() -> None:
    y, endogenous, instruments, exogenous = _iv_sample()

    result = IV2SLS(covariance="robust").fit(
        y,
        endogenous=endogenous,
        instruments=instruments,
        exogenous=exogenous,
    )

    assert list(result.params.index) == ["const", "control", "treatment"]
    np.testing.assert_allclose(
        result.params.to_numpy(),
        np.array([1.4, 0.65, 2.1]),
        atol=0.07,
        rtol=0.0,
    )
    assert result.nobs == len(y)
    assert result.df_resid == len(y) - len(result.params)
    assert result.covariance_type == "robust"
    assert result.covariance.index.equals(result.params.index)
    assert result.covariance.columns.equals(result.params.index)
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-12)
    np.testing.assert_allclose(
        result.standard_errors.to_numpy() ** 2,
        np.diag(result.covariance.to_numpy()),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.test_statistics.to_numpy(),
        result.params.to_numpy() / result.standard_errors.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.pvalues.between(0.0, 1.0).all()
    assert result.residuals.index.equals(y.index)
    assert result.fitted_values.index.equals(y.index)
    np.testing.assert_allclose(
        result.fitted_values.to_numpy() + result.residuals.to_numpy(),
        y.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    prediction = result.predict(endogenous, exogenous=exogenous)
    assert prediction.index.equals(y.index)
    np.testing.assert_allclose(prediction, result.fitted_values, rtol=1e-12, atol=1e-12)

    assert set(result.first_stage) == {"treatment"}
    diagnostic = result.first_stage["treatment"]
    assert diagnostic.partial_r_squared > 0.25
    assert diagnostic.classical_f_statistic > 10.0
    assert diagnostic.classical_f_df_num == 2
    assert diagnostic.classical_f_df_denom > 0
    assert 0.0 <= diagnostic.classical_f_p_value <= 1.0
    assert diagnostic.weak_instrument_warning is False
    # The implemented overidentification diagnostic is the homoskedastic Sargan
    # test, so it must not be presented as valid alongside robust covariance.
    assert result.overidentification is None


def test_iv2sls_replaces_legacy_treatment_effect_homoskedastic_contract() -> None:
    """Preserve the migrated 2SLS numerical contract without a legacy implementation."""

    rng = np.random.default_rng(45)
    nobs = 400
    z1 = rng.normal(size=nobs)
    x1 = rng.normal(size=nobs)
    full_instruments = np.column_stack([np.ones(nobs), z1, x1])
    treatment = (full_instruments @ np.array([0.0, 0.4, 0.2]) + rng.normal(size=nobs) > 0).astype(
        float
    )
    outcome = 2.5 * treatment + 0.8 * x1 + rng.normal(0.0, 0.6, nobs)
    structural_design = np.column_stack([treatment, np.ones(nobs), x1])

    ztz_inverse = np.linalg.inv(full_instruments.T @ full_instruments)
    normal_matrix = (
        structural_design.T
        @ full_instruments
        @ ztz_inverse
        @ full_instruments.T
        @ structural_design
    )
    legacy_params = np.linalg.solve(
        normal_matrix,
        structural_design.T @ full_instruments @ ztz_inverse @ full_instruments.T @ outcome,
    )
    legacy_residuals = outcome - structural_design @ legacy_params
    legacy_sigma2 = float(legacy_residuals @ legacy_residuals / (nobs - 3))
    legacy_covariance = legacy_sigma2 * np.linalg.inv(normal_matrix)

    result = IV2SLS(covariance="unadjusted", add_constant=False).fit(
        pd.Series(outcome, name="outcome"),
        endogenous=pd.DataFrame({"T": treatment}),
        exogenous=pd.DataFrame({"const": 1.0, "x1": x1}),
        instruments=pd.DataFrame({"z1": z1}),
    )
    migrated_order = ["T", "const", "x1"]

    np.testing.assert_allclose(result.params[migrated_order], legacy_params, atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        result.covariance.loc[migrated_order, migrated_order],
        legacy_covariance,
        atol=2e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.predict(
            pd.DataFrame({"T": treatment}),
            exogenous=pd.DataFrame({"const": 1.0, "x1": x1}),
        ),
        structural_design @ legacy_params,
        atol=2e-14,
        rtol=0,
    )


@pytest.mark.parametrize("component", ["y", "endogenous", "instruments", "exogenous"])
def test_fit_rejects_misaligned_pandas_indexes(component: str) -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=180)
    values: dict[str, object] = {
        "y": y,
        "endogenous": endogenous,
        "instruments": instruments,
        "exogenous": exogenous,
    }
    values[component] = values[component].iloc[::-1]

    with pytest.raises(ValueError, match="(?i)(index|indices)"):
        IV2SLS().fit(
            values["y"],
            endogenous=values["endogenous"],
            instruments=values["instruments"],
            exogenous=values["exogenous"],
        )


def test_predict_enforces_fitted_schema_and_index_alignment() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=220)
    result = IV2SLS().fit(
        y,
        endogenous=endogenous,
        instruments=instruments,
        exogenous=exogenous,
    )

    with pytest.raises(ValueError, match="(?i)(schema|column)"):
        result.predict(endogenous.rename(columns={"treatment": "renamed"}), exogenous=exogenous)
    with pytest.raises(ValueError, match="(?i)(schema|column)"):
        result.predict(endogenous, exogenous=exogenous.rename(columns={"control": "renamed"}))
    with pytest.raises(ValueError, match="(?i)(index|indices)"):
        result.predict(endogenous, exogenous=exogenous.iloc[::-1])


def test_missing_raise_reports_nonfinite_input() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=160)
    endogenous = endogenous.copy()
    endogenous.iloc[3, 0] = np.nan

    with pytest.raises(ValueError, match="(?i)(missing|finite)"):
        IV2SLS(missing="raise").fit(
            y,
            endogenous=endogenous,
            instruments=instruments,
            exogenous=exogenous,
        )


def test_missing_drop_uses_one_complete_case_mask_and_preserves_labels() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=180)
    y = y.copy()
    endogenous = endogenous.copy()
    instruments = instruments.copy()
    exogenous = exogenous.copy()
    y.iloc[0] = np.nan
    endogenous.iloc[1, 0] = np.nan
    instruments.iloc[2, 0] = np.nan
    exogenous.iloc[3, 0] = np.nan
    expected_index = y.index[4:]

    result = IV2SLS(missing="drop").fit(
        y,
        endogenous=endogenous,
        instruments=instruments,
        exogenous=exogenous,
    )

    assert result.nobs == len(y) - 4
    assert result.residuals.index.equals(expected_index)
    assert result.fitted_values.index.equals(expected_index)


def test_underidentified_design_is_rejected_before_estimation() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=200)
    rng = np.random.default_rng(771)
    endogenous = endogenous.assign(second_endogenous=rng.normal(size=len(y)))

    with pytest.raises(ValueError, match="(?i)(underidentif|instrument)"):
        IV2SLS().fit(
            y,
            endogenous=endogenous,
            instruments=instruments[["z1"]],
            exogenous=exogenous,
        )


def test_exact_identification_has_no_overidentification_test() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=300)

    result = IV2SLS().fit(
        y,
        endogenous=endogenous,
        instruments=instruments[["z1"]],
        exogenous=exogenous,
    )

    assert result.overidentification is None
    assert set(result.first_stage) == {"treatment"}


def test_overidentified_unadjusted_fit_reports_sargan_test() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=450)

    result = IV2SLS(covariance="unadjusted").fit(
        y,
        endogenous=endogenous,
        instruments=instruments,
        exogenous=exogenous,
    )

    assert result.overidentification is not None
    assert result.overidentification.df == 1
    assert np.isfinite(result.overidentification.statistic)
    assert 0.0 <= result.overidentification.p_value <= 1.0


def test_weak_first_stage_is_visible_in_diagnostics() -> None:
    rng = np.random.default_rng(918_204)
    nobs = 350
    index = pd.RangeIndex(nobs)
    instrument = rng.normal(size=nobs)
    control = rng.normal(size=nobs)
    first_stage_error = rng.normal(size=nobs)
    treatment = 0.002 * instrument + 0.5 * control + first_stage_error
    outcome = 1.0 + 0.4 * control + 1.5 * treatment + first_stage_error + rng.normal(size=nobs)

    result = IV2SLS().fit(
        pd.Series(outcome, index=index, name="outcome"),
        endogenous=pd.DataFrame({"treatment": treatment}, index=index),
        instruments=pd.DataFrame({"instrument": instrument}, index=index),
        exogenous=pd.DataFrame({"control": control}, index=index),
    )
    diagnostic = result.first_stage["treatment"]

    assert diagnostic.partial_r_squared < 0.03
    assert diagnostic.classical_f_statistic < 10.0
    assert diagnostic.weak_instrument_warning is True
    assert diagnostic.excluded_instrument_df == 1
    assert np.isfinite(diagnostic.excluded_instrument_statistic)
    assert 0.0 <= diagnostic.excluded_instrument_p_value <= 1.0
    assert diagnostic.excluded_instrument_distribution in {"chi2", "F"}


def test_clustered_covariance_requires_valid_aligned_clusters() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=240)
    clusters = pd.Series(np.repeat(np.arange(30), 8), index=y.index, name="cluster")
    estimator = IV2SLS(covariance="clustered")

    with pytest.raises(ValueError, match="(?i)cluster"):
        estimator.fit(y, endogenous=endogenous, instruments=instruments, exogenous=exogenous)
    with pytest.raises(ValueError, match="(?i)(index|indices)"):
        estimator.fit(
            y,
            endogenous=endogenous,
            instruments=instruments,
            exogenous=exogenous,
            clusters=clusters.iloc[::-1],
        )
    with pytest.raises(ValueError, match="(?i)(two|cluster)"):
        estimator.fit(
            y,
            endogenous=endogenous,
            instruments=instruments,
            exogenous=exogenous,
            clusters=pd.Series(0, index=y.index),
        )

    result = estimator.fit(
        y,
        endogenous=endogenous,
        instruments=instruments,
        exogenous=exogenous,
        clusters=clusters,
    )
    assert result.covariance_type == "clustered"
    assert np.isfinite(result.covariance.to_numpy()).all()


@pytest.mark.parametrize("option", ["covariance", "missing"])
def test_invalid_estimator_options_are_rejected(option: str) -> None:
    kwargs = {option: "not-a-supported-value"}
    with pytest.raises(ValueError):
        IV2SLS(**kwargs)


def test_add_constant_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="add_constant"):
        IV2SLS(add_constant="false")


@pytest.mark.simulation
def test_multiple_endogenous_regressors_are_jointly_identified_and_recovered() -> None:
    rng = np.random.default_rng(38_114)
    nobs = 4_000
    control = rng.normal(size=nobs)
    instruments = rng.normal(size=(nobs, 3))
    first_stage_errors = rng.normal(size=(nobs, 2))
    first = (
        0.8 * instruments[:, 0]
        + 0.35 * instruments[:, 2]
        + 0.2 * control
        + first_stage_errors[:, 0]
    )
    second = (
        0.7 * instruments[:, 1] - 0.3 * instruments[:, 2] - 0.1 * control + first_stage_errors[:, 1]
    )
    structural_error = (
        0.5 * first_stage_errors[:, 0] - 0.4 * first_stage_errors[:, 1] + rng.normal(size=nobs)
    )
    outcome = 0.7 + 0.45 * control + 1.25 * first - 0.8 * second + structural_error

    result = IV2SLS().fit(
        pd.Series(outcome, name="outcome"),
        endogenous=pd.DataFrame({"first": first, "second": second}),
        instruments=pd.DataFrame(
            {
                "z1": instruments[:, 0],
                "z2": instruments[:, 1],
                "z3": instruments[:, 2],
            }
        ),
        exogenous=pd.DataFrame({"control": control}),
    )

    assert list(result.params.index) == ["const", "control", "first", "second"]
    np.testing.assert_allclose(
        result.params.to_numpy(),
        np.array([0.7, 0.45, 1.25, -0.8]),
        atol=0.08,
        rtol=0.0,
    )
    assert set(result.first_stage) == {"first", "second"}
    assert result.overidentification is None


def test_no_intercept_array_api_preserves_explicit_design() -> None:
    rng = np.random.default_rng(7_301)
    nobs = 1_200
    instrument = rng.normal(size=nobs)
    first_stage_error = rng.normal(size=nobs)
    endogenous = instrument + first_stage_error
    outcome = 1.6 * endogenous + 0.5 * first_stage_error + rng.normal(size=nobs)

    result = IV2SLS(add_constant=False).fit(
        outcome,
        endogenous=endogenous,
        instruments=instrument,
    )

    assert list(result.params.index) == ["endogenous"]
    assert result.params["endogenous"] == pytest.approx(1.6, abs=0.10)
    assert result.predict(endogenous).shape == (nobs,)


def test_rank_deficient_structural_and_instrument_designs_are_rejected() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=220)
    collinear_exogenous = exogenous.assign(control_copy=exogenous["control"])
    with pytest.raises(ValueError, match="(?i)structural.*rank"):
        IV2SLS().fit(
            y,
            endogenous=endogenous,
            instruments=instruments,
            exogenous=collinear_exogenous,
        )

    collinear_instruments = instruments.assign(z1_copy=instruments["z1"])
    with pytest.raises(ValueError, match="(?i)instrument.*rank"):
        IV2SLS().fit(
            y,
            endogenous=endogenous,
            instruments=collinear_instruments,
            exogenous=exogenous,
        )


def test_column_names_enforce_excluded_instrument_semantics() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=180)
    overlapping = instruments.rename(columns={"z1": "control"})
    with pytest.raises(ValueError, match="(?i)excluded instruments only"):
        IV2SLS().fit(
            y,
            endogenous=endogenous,
            instruments=overlapping,
            exogenous=exogenous,
        )

    duplicated = pd.DataFrame(
        np.column_stack([instruments["z1"], instruments["z2"]]),
        index=instruments.index,
        columns=["z", "z"],
    )
    with pytest.raises(ValueError, match="(?i)unique"):
        IV2SLS().fit(
            y,
            endogenous=endogenous,
            instruments=duplicated,
            exogenous=exogenous,
        )


def test_cluster_missing_values_follow_the_joint_drop_policy() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=200)
    clusters = pd.Series(np.repeat(np.arange(20), 10), index=y.index, dtype=float)
    y = y.copy()
    clusters = clusters.copy()
    y.iloc[0] = np.nan
    clusters.iloc[1] = np.nan

    result = IV2SLS(covariance="clustered", missing="drop").fit(
        y,
        endogenous=endogenous,
        instruments=instruments,
        exogenous=exogenous,
        clusters=clusters,
    )

    assert result.nobs == 198
    assert result.dropped_rows == 2
    assert result.n_clusters == 20
    assert result.estimation_index.equals(y.index[2:])


def test_robust_overidentified_fit_does_not_mislabel_sargan_as_robust() -> None:
    y, endogenous, instruments, exogenous = _iv_sample(nobs=300)

    result = IV2SLS(covariance="robust").fit(
        y,
        endogenous=endogenous,
        instruments=instruments,
        exogenous=exogenous,
    )

    assert result.overidentification is None
    assert any("Sargan" in note and "not reported" in note for note in result.notes)

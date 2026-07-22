"""Numerical IV2SLS parity against the independent linearmodels implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalkit import IV2SLS

linearmodels_iv = pytest.importorskip("linearmodels.iv")
ReferenceIV2SLS = linearmodels_iv.IV2SLS


def _parity_sample() -> tuple[
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    rng = np.random.default_rng(20_260_715)
    nclusters = 48
    observations_per_cluster = 15
    nobs = nclusters * observations_per_cluster
    index = pd.Index(np.arange(50_000, 50_000 + nobs), name="observation")
    clusters = pd.Series(
        np.repeat(np.arange(nclusters), observations_per_cluster),
        index=index,
        name="cluster",
    )
    cluster_shock = rng.normal(scale=0.4, size=nclusters)[clusters.to_numpy()]
    z1 = rng.normal(size=nobs)
    z2 = rng.normal(size=nobs)
    control = rng.normal(size=nobs)
    first_stage_error = rng.normal(size=nobs) + cluster_shock
    treatment = 0.75 * z1 - 0.45 * z2 + 0.25 * control + first_stage_error
    structural_error = 0.6 * first_stage_error + rng.normal(size=nobs) + cluster_shock
    outcome = 0.8 + 0.55 * control + 1.85 * treatment + structural_error
    return (
        pd.Series(outcome, index=index, name="outcome"),
        pd.DataFrame({"treatment": treatment}, index=index),
        pd.DataFrame({"z1": z1, "z2": z2}, index=index),
        pd.DataFrame({"control": control}, index=index),
        clusters,
    )


@pytest.mark.validation
@pytest.mark.parametrize("covariance_type", ["unadjusted", "robust", "clustered"])
def test_iv2sls_parameters_and_covariance_match_linearmodels(covariance_type: str) -> None:
    y, endogenous, instruments, exogenous, clusters = _parity_sample()
    cluster_argument = clusters if covariance_type == "clustered" else None

    native = IV2SLS(covariance=covariance_type).fit(
        y,
        endogenous=endogenous,
        instruments=instruments,
        exogenous=exogenous,
        clusters=cluster_argument,
    )
    reference_exogenous = pd.concat(
        [pd.Series(1.0, index=y.index, name="const"), exogenous],
        axis=1,
    )
    reference_kwargs: dict[str, object] = {
        "cov_type": covariance_type,
        # causalkit deliberately reports n-k/HC1/CR1 finite-sample corrections.
        "debiased": True,
    }
    if covariance_type == "clustered":
        reference_kwargs["clusters"] = clusters
    reference = ReferenceIV2SLS(
        y,
        reference_exogenous,
        endogenous,
        instruments,
    ).fit(**reference_kwargs)

    reference_params = reference.params.reindex(native.params.index)
    reference_covariance = reference.cov.reindex(
        index=native.params.index,
        columns=native.params.index,
    )

    # Both estimators solve the same closed-form normal equations. The tolerance only
    # allows floating-point differences from their different linear-algebra factorizations.
    np.testing.assert_allclose(
        native.params.to_numpy(),
        reference_params.to_numpy(),
        rtol=2e-10,
        atol=2e-10,
    )
    # Covariance matrices compound several matrix products, so their justified tolerance
    # is slightly wider while remaining far below applied-reporting precision.
    np.testing.assert_allclose(
        native.covariance.to_numpy(),
        reference_covariance.to_numpy(),
        rtol=5e-8,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        native.standard_errors.to_numpy(),
        reference.std_errors.reindex(native.params.index).to_numpy(),
        rtol=5e-8,
        atol=2e-10,
    )
    native_first_stage = native.first_stage["treatment"]
    reference_first_stage = reference.first_stage.diagnostics.loc["treatment"]
    assert native_first_stage.r_squared == pytest.approx(
        reference_first_stage["rsquared"], rel=5e-10, abs=2e-10
    )
    assert native_first_stage.partial_r_squared == pytest.approx(
        reference_first_stage["partial.rsquared"], rel=5e-10, abs=2e-10
    )
    # linearmodels reports the clustered joint test as chi-square even with
    # debiasing. causalkit reports the algebraically equivalent F = chi-square/q
    # with G-1 denominator degrees of freedom, matching its clustered result
    # inference contract.
    native_reference_scale_statistic = native_first_stage.excluded_instrument_statistic
    if covariance_type == "clustered":
        native_reference_scale_statistic *= native_first_stage.excluded_instrument_df
    assert native_reference_scale_statistic == pytest.approx(
        reference_first_stage["f.stat"], rel=5e-8, abs=2e-10
    )
    expected_distribution = "chi2" if covariance_type == "robust" else "F"
    assert native_first_stage.excluded_instrument_distribution == expected_distribution
    assert native.covariance_type == covariance_type
    if covariance_type == "unadjusted":
        assert native.overidentification is not None
        assert native.overidentification.statistic == pytest.approx(
            reference.sargan.stat, rel=5e-10, abs=2e-10
        )
        assert native.overidentification.df == reference.sargan.df
        assert native.overidentification.p_value == pytest.approx(
            reference.sargan.pval, rel=5e-10, abs=2e-10
        )

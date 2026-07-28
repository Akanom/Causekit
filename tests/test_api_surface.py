"""Regression tests for the initial stable causalkit namespace."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import causalkit
from causalkit import IV2SLS, IV2SLSResult

PUBLIC_EXPORTS = {
    "IV2SLS",
    "IV2SLSResult",
    "RandomizedATE",
    "RandomizedATEResult",
    "IPWATE",
    "AIPWATE",
    "ObservationalATEResult",
    "OverlapDiagnostic",
    "CrossFitter",
    "CrossFitResult",
    "NuisanceEstimatorProtocol",
    "PropensityResultProtocol",
    "OutcomeResultProtocol",
    "NearestNeighborMatch",
    "NearestNeighborMatchResult",
    "confint",
    "fitted_values",
    "predict",
    "residuals",
    "summary_frame",
    "vcov",
}


def test_initial_stable_namespace_exports_iv_and_postestimation_contract() -> None:
    assert set(causalkit.__all__) >= PUBLIC_EXPORTS
    assert all(hasattr(causalkit, name) for name in PUBLIC_EXPORTS)
    assert "TreatmentEffect" not in causalkit.__all__
    assert not hasattr(causalkit, "TreatmentEffect")


def test_estimator_signatures_keep_identification_inputs_explicit() -> None:
    constructor = inspect.signature(IV2SLS)
    assert constructor.parameters["covariance"].default == "robust"
    assert constructor.parameters["add_constant"].default is True
    assert constructor.parameters["missing"].default == "raise"

    fit = inspect.signature(IV2SLS.fit)
    assert list(fit.parameters) == [
        "self",
        "y",
        "endogenous",
        "instruments",
        "exogenous",
        "clusters",
    ]
    assert fit.parameters["endogenous"].kind is inspect.Parameter.KEYWORD_ONLY
    assert fit.parameters["instruments"].kind is inspect.Parameter.KEYWORD_ONLY
    assert fit.parameters["exogenous"].default is None
    assert fit.parameters["clusters"].default is None

    matching = inspect.signature(causalkit.NearestNeighborMatch)
    assert matching.parameters["estimand"].default == "att"
    assert matching.parameters["replacement"].default is True
    assert matching.parameters["caliper"].default == "auto"
    assert matching.parameters["ties"].default == "all"
    assert matching.parameters["inference"].default == "none"


def test_fit_returns_the_public_result_type() -> None:
    rng = np.random.default_rng(92)
    nobs = 120
    index = pd.RangeIndex(nobs)
    instrument = rng.normal(size=nobs)
    treatment = instrument + rng.normal(size=nobs)
    outcome = 1.0 + 2.0 * treatment + rng.normal(size=nobs)

    result = IV2SLS().fit(
        pd.Series(outcome, index=index),
        endogenous=pd.DataFrame({"treatment": treatment}, index=index),
        instruments=pd.DataFrame({"instrument": instrument}, index=index),
    )

    assert isinstance(result, IV2SLSResult)

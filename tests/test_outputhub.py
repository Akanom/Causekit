"""Contract tests for the optional Universal Output Hub adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalkit import IV2SLS, add_to_outputhub, to_outputhub_model

outputhub = pytest.importorskip("universal_output_hub")


@pytest.fixture(scope="module")
def iv_result():
    rng = np.random.default_rng(20_260_722)
    nobs = 240
    z1 = rng.normal(size=nobs)
    z2 = rng.normal(size=nobs)
    control = rng.normal(size=nobs)
    first_stage_error = rng.normal(size=nobs)
    treatment = z1 - 0.4 * z2 + 0.3 * control + first_stage_error
    outcome = 1.2 + 1.8 * treatment + 0.5 * control + 0.6 * first_stage_error
    index = pd.RangeIndex(nobs)
    return IV2SLS(covariance="unadjusted").fit(
        pd.Series(outcome, index=index, name="outcome"),
        endogenous=pd.DataFrame({"treatment": treatment}, index=index),
        instruments=pd.DataFrame({"z1": z1, "z2": z2}, index=index),
        exogenous=pd.DataFrame({"control": control}, index=index),
    )


def test_result_converts_to_outputhub_without_reestimating(iv_result) -> None:
    model = to_outputhub_model(iv_result, name="Main IV")

    assert isinstance(model, outputhub.RegressionModel)
    assert model.name == "Main IV"
    assert model.depvar == "outcome"
    pd.testing.assert_series_equal(model.params, iv_result.params.rename("coef"))
    pd.testing.assert_series_equal(model.std_errors, iv_result.standard_errors.rename("se"))
    assert model.metadata["estimator"] == "iv_2sls"
    assert model.metadata["causal_interpretation_requires_assumptions"] is True
    assert "First-stage F (treatment)" in model.diagnostics
    assert "Sargan statistic" in model.diagnostics


def test_result_and_first_stage_can_be_added_to_outputhub(iv_result) -> None:
    hub = outputhub.OutputHub("IV analysis")

    model = add_to_outputhub(hub, iv_result, name="Main IV")

    assert model.name == "Main IV"
    assert len(hub.models) == 1
    assert len(hub.tables) == 1
    assert hub.tables[0].name == "Main IV first-stage diagnostics"


def test_outputhub_adapter_validates_inputs(iv_result) -> None:
    with pytest.raises(TypeError, match="IV2SLSResult"):
        to_outputhub_model(object())
    with pytest.raises(TypeError, match="add_model"):
        add_to_outputhub(object(), iv_result)

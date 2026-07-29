"""Hash-verified real-data smoke for the public honest R-learner."""

from __future__ import annotations

import hashlib

import pytest

from causekit import RLearner
from causekit.datasets import load_real_dataset


@pytest.mark.validation
def test_real_nsw_native_honest_rlearner_smoke() -> None:
    try:
        data = load_real_dataset("nsw_mixtape")
    except FileNotFoundError:
        pytest.skip("download the hash-pinned NSW dataset to run the real-data R-learner smoke")
    covariates = data[["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]].astype(
        float
    )
    result = RLearner(
        n_splits=3,
        evaluation_fraction=0.5,
        random_state=20_260_729,
        calibration_groups=4,
        bootstrap_iterations=999,
        bootstrap_random_state=20_260_729,
    ).fit(
        data["re78"].astype(float),
        treatment=data["treat"].astype(float),
        covariates=covariates,
    )
    index_digest = hashlib.sha256(
        "\n".join(str(value) for value in result.evaluation_index).encode("utf-8")
    ).hexdigest()

    assert result.nobs == len(data) == 445
    assert result.construction_nobs == 222
    assert result.evaluation_nobs == 223
    assert index_digest == "f2ce0c338c4331a4ba211e0e32f44aeaa6a775d78b3f821e20bebbaa4977032f"
    assert result.native_outcome is True
    assert result.native_propensity is True
    assert result.native_cate is True
    assert result.honest_r_loss == pytest.approx(40_564_476.95327945, abs=2e-6)
    assert result.honest_constant_r_loss == pytest.approx(40_167_364.99136036, abs=2e-6)
    assert result.r_loss_gain == pytest.approx(-0.009886432978725601, abs=2e-12)
    assert result.calibration_coefficients["level"] == pytest.approx(951.2628530344787, abs=2e-9)
    assert result.calibration_coefficients["heterogeneity"] == pytest.approx(
        -0.015086789683179625, abs=2e-12
    )
    assert result.calibration_tests.loc["heterogeneity=0", "p_value"] == pytest.approx(
        0.9825652202891315, abs=2e-12
    )
    assert result.group_effects["nobs"].sum() == result.evaluation_nobs
    assert result.group_effects[["n_treated", "n_control"]].gt(0).all().all()

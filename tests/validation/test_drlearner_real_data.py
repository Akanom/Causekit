"""Hash-verified NSW smoke for the public honest DR learner."""

from __future__ import annotations

import hashlib

import pytest

from causekit import DRLearner
from causekit.datasets import load_real_dataset


@pytest.mark.validation
def test_real_nsw_native_honest_drlearner_smoke() -> None:
    try:
        data = load_real_dataset("nsw_mixtape")
    except FileNotFoundError:
        pytest.skip("download the hash-pinned NSW dataset to run the real-data DR smoke")
    covariates = data[["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]].astype(
        float
    )
    result = DRLearner(
        n_splits=3,
        evaluation_fraction=0.5,
        random_state=20_260_730,
        calibration_groups=4,
        bootstrap_iterations=999,
        bootstrap_random_state=20_260_730,
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
    assert index_digest == "3e95f6f725a36d48919850440e74c1d41c0ab7912788dad30cab0e864d8920df"
    assert result.native_outcome is True
    assert result.native_propensity is True
    assert result.native_cate is True
    assert result.honest_dr_loss == pytest.approx(148_166_882.7104254, abs=2e-5)
    assert result.honest_constant_dr_loss == pytest.approx(148_246_322.96933556, abs=2e-5)
    assert result.dr_loss_gain == pytest.approx(0.0005358666395157696, abs=2e-12)
    assert result.calibration_coefficients["level"] == pytest.approx(85.75687874010197, abs=2e-9)
    assert result.calibration_coefficients["heterogeneity"] == pytest.approx(
        -1.5015236793152555, abs=2e-12
    )
    assert result.calibration_tests.loc["heterogeneity=0", "p_value"] == pytest.approx(
        0.8895429546872289, abs=2e-12
    )
    assert result.group_effects["nobs"].sum() == result.evaluation_nobs
    assert result.group_effects[["n_treated", "n_control"]].gt(0).all().all()

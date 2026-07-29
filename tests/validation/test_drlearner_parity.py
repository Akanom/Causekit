"""Independent base-R parity for fixed honest DR evaluation moments."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causekit import DRLearner


def _read_values(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", maxsplit=1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


@pytest.mark.validation
def test_fixed_honest_dr_evaluation_matches_independent_base_r(tmp_path: Path) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed")
    rng = np.random.default_rng(20_260_730)
    nobs = 420
    covariates = pd.DataFrame(rng.normal(size=(nobs, 4)), columns=list("abcd"))
    propensity = 1.0 / (1.0 + np.exp(-(0.3 * covariates["a"] - 0.2 * covariates["b"])))
    treatment = pd.Series(rng.binomial(1, propensity), index=covariates.index)
    cate = 0.9 + 0.55 * covariates["a"] - 0.3 * covariates["b"]
    outcome = (
        0.4 * covariates["c"]
        + cate * treatment
        + pd.Series(rng.normal(scale=0.65, size=nobs), index=covariates.index)
    )
    result = DRLearner(
        n_splits=3,
        evaluation_fraction=0.4,
        random_state=730,
        calibration_groups=3,
        bootstrap_iterations=99,
        propensity_tuning_splits=2,
    ).fit(outcome, treatment=treatment, covariates=covariates)
    parity_input = pd.DataFrame(
        {
            "dr_score": result.evaluation_dr_score,
            "cate": result.evaluation_cate_predictions,
            "constant_effect": result.construction_constant_effect,
            "group": result.evaluation_group,
        }
    )
    input_path = tmp_path / "drlearner_evaluation.csv"
    output_path = tmp_path / "drlearner_reference.txt"
    parity_input.to_csv(input_path, index=False)
    script = Path(__file__).resolve().parents[2] / "benchmarks" / "validate_drlearner_reference.R"
    completed = subprocess.run(
        [rscript, str(script), str(input_path), str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    values = _read_values(output_path)

    assert values["parity_status"] == "pass"
    assert values["r_version"].startswith("R version 4.")
    expected = {
        "honest_dr_loss": result.honest_dr_loss,
        "honest_constant_dr_loss": result.honest_constant_dr_loss,
        "dr_loss_gain": result.dr_loss_gain,
        "calibration_center": result.calibration_center,
        "calibration_level": result.calibration_coefficients["level"],
        "calibration_heterogeneity": result.calibration_coefficients["heterogeneity"],
        "calibration_level_variance": result.calibration_covariance.loc["level", "level"],
        "calibration_heterogeneity_variance": result.calibration_covariance.loc[
            "heterogeneity", "heterogeneity"
        ],
        "calibration_covariance": result.calibration_covariance.loc["level", "heterogeneity"],
    }
    for group in result.group_effects.index:
        expected[f"group_{group}_effect"] = result.group_effects.loc[group, "effect"]
        expected[f"group_{group}_variance"] = result.group_covariance.loc[group, group]
    for name, target in expected.items():
        assert float(values[name]) == pytest.approx(float(target), abs=2e-9)

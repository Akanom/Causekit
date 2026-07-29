"""Prepare the fixed honest DR-evaluation fixture consumed by manual Stata parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causekit import DRLearner


def main() -> None:
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
    fixture = pd.DataFrame(
        {
            "dr_score": result.evaluation_dr_score,
            "cate": result.evaluation_cate_predictions,
            "constant_effect": result.construction_constant_effect,
            "group": result.evaluation_group,
        }
    )
    expected = {
        "exp_dr_loss": result.honest_dr_loss,
        "exp_const_loss": result.honest_constant_dr_loss,
        "exp_gain": result.dr_loss_gain,
        "exp_cal_center": result.calibration_center,
        "exp_cal_level": result.calibration_coefficients["level"],
        "exp_cal_hetero": result.calibration_coefficients["heterogeneity"],
        "exp_cal_v_level": result.calibration_covariance.loc["level", "level"],
        "exp_cal_v_hetero": result.calibration_covariance.loc["heterogeneity", "heterogeneity"],
        "exp_cal_cov": result.calibration_covariance.loc["level", "heterogeneity"],
    }
    for group in result.group_effects.index:
        expected[f"exp_g{group}"] = result.group_effects.loc[group, "effect"]
        expected[f"exp_g{group}_var"] = result.group_covariance.loc[group, group]
    for name, value in expected.items():
        fixture[name] = value
    output = Path(__file__).resolve().with_name("drlearner_stata_input.csv")
    fixture.to_csv(output, index=False, float_format="%.17g")
    print(output)


if __name__ == "__main__":
    main()

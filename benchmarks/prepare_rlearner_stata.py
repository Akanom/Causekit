"""Generate the fixed honest-evaluation fixture consumed by the manual Stata contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from causekit import RLearner


class _OutcomeResult:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["outcome_mean"].copy()


class _Outcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _OutcomeResult:
        return _OutcomeResult()


class _PropensityResult:
    classes_ = np.array([0.0, 1.0])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        treated = X["propensity"].copy()
        return pd.DataFrame({0.0: 1.0 - treated, 1.0: treated}, index=X.index)


class _Propensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _PropensityResult:
        return _PropensityResult()


class _CATEResult:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["cate_score"].copy()


class _CATE:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series,
    ) -> _CATEResult:
        return _CATEResult()


def _result():
    nobs = 96
    index = pd.Index([f"stata-{position}" for position in range(nobs)])
    x = np.linspace(-1.8, 1.8, nobs)
    treatment = pd.Series(np.tile([0.0, 1.0], nobs // 2), index=index)
    propensity = np.where(x < 0.0, 0.38, 0.62)
    outcome_mean = 0.8 + 0.35 * x
    cate_score = 0.6 + 0.7 * x
    v = treatment.to_numpy(dtype=float) - propensity
    outcome = pd.Series(
        outcome_mean
        + (1.1 + 0.75 * cate_score) * v
        + 0.12 * np.sin(4.0 * x)
        + 0.04 * np.cos(7.0 * x),
        index=index,
    )
    covariates = pd.DataFrame(
        {
            "x": x,
            "outcome_mean": outcome_mean,
            "propensity": propensity,
            "cate_score": cate_score,
        },
        index=index,
    )
    return RLearner(
        outcome_factory=_Outcome,
        propensity_factory=_Propensity,
        cate_factory=_CATE,
        n_splits=3,
        evaluation_fraction=0.5,
        random_state=627,
        overlap_floor=0.05,
        calibration_groups=3,
        bootstrap_iterations=99,
        bootstrap_random_state=918,
    ).fit(outcome, treatment=treatment, covariates=covariates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/rlearner_stata_input.csv"),
    )
    args = parser.parse_args()
    result = _result()
    fixture = result.evaluation_residuals[
        ["outcome_residual", "treatment_residual", "predicted_cate"]
    ].rename(columns={"predicted_cate": "cate"})
    fixture["constant_effect"] = result.construction_constant_effect
    fixture["group"] = result.evaluation_group
    fixture["exp_r_loss"] = result.honest_r_loss
    fixture["exp_const_loss"] = result.honest_constant_r_loss
    fixture["exp_gain"] = result.r_loss_gain
    fixture["exp_cal_level"] = result.calibration_coefficients["level"]
    fixture["exp_cal_hetero"] = result.calibration_coefficients["heterogeneity"]
    fixture["exp_cal_v_level"] = result.calibration_covariance.loc["level", "level"]
    fixture["exp_cal_v_hetero"] = result.calibration_covariance.loc[
        "heterogeneity", "heterogeneity"
    ]
    fixture["exp_cal_cov"] = result.calibration_covariance.loc["level", "heterogeneity"]
    for group in result.group_effects.index:
        fixture[f"exp_g{group}"] = result.group_effects.loc[group, "effect"]
        fixture[f"exp_g{group}_var"] = result.group_covariance.loc[group, group]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture.reset_index(names="row_id").to_csv(args.output, index=False, float_format="%.17g")
    print(f"rows={len(fixture)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()

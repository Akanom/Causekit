"""Survey-population repeated-cross-section DiD on the pinned YRBS application.

Run from the repository root::

    python examples/survey_repeated_cross_section_did.py --download
    python examples/survey_repeated_cross_section_did.py --covariate-adjusted

The authors' processed public file retains survey strata and analysis weights but omits
PSU identifiers. This example therefore declares each row as its own PSU. Its weighted
point estimate reproduces the sampling-weight-only four-cell calculation in the authors'
code; its standard error is an independent-observation-within-stratum sensitivity, not a
reproduction of the paper's bootstrap inference.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from causekit import CrossFitter, RepeatedCrossSectionDiD, RepeatedCrossSectionSurveyDesign
from causekit.datasets import load_real_dataset


class _WeightedLogit:
    def fit(self, X: Any, y: Any, *, sample_weight: Any) -> _WeightedLogitResult:
        try:
            import statsmodels.api as sm
        except ImportError as error:  # pragma: no cover - optional example dependency
            raise ImportError("Install CauseKit's validation extra for this example.") from error
        design = sm.add_constant(pd.DataFrame(X), has_constant="add")
        fitted = sm.GLM(
            np.asarray(y, dtype=float),
            design,
            family=sm.families.Binomial(),
            freq_weights=np.asarray(sample_weight, dtype=float),
        ).fit()
        return _WeightedLogitResult(fitted)


class _WeightedLogitResult:
    def __init__(self, fitted: Any) -> None:
        self.fitted = fitted

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        import statsmodels.api as sm

        design = sm.add_constant(X, has_constant="add")
        treated = np.asarray(self.fitted.predict(design), dtype=float)
        return pd.DataFrame({0: 1.0 - treated, 1: treated}, index=X.index)


class _WeightedOLS:
    def fit(self, X: Any, y: Any, *, sample_weight: Any) -> _WeightedOLSResult:
        try:
            import statsmodels.api as sm
        except ImportError as error:  # pragma: no cover - optional example dependency
            raise ImportError("Install CauseKit's validation extra for this example.") from error
        design = sm.add_constant(pd.DataFrame(X), has_constant="add")
        fitted = sm.WLS(
            np.asarray(y, dtype=float),
            design,
            weights=np.asarray(sample_weight, dtype=float),
        ).fit()
        return _WeightedOLSResult(fitted)


class _WeightedOLSResult:
    def __init__(self, fitted: Any) -> None:
        self.fitted = fitted

    def predict(self, X: pd.DataFrame) -> pd.Series:
        import statsmodels.api as sm

        design = sm.add_constant(X, has_constant="add")
        return pd.Series(np.asarray(self.fitted.predict(design)), index=X.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--covariate-adjusted", action="store_true")
    args = parser.parse_args()
    data = load_real_dataset("yrbs_beverage_tax", download=args.download)
    data["period"] = data["comb_group_label"].isin([2, 4]).astype(float) + 1.0
    data["treatment_time"] = np.where(data["comb_group_label"].isin([1, 2]), 2.0, np.inf)
    design = RepeatedCrossSectionSurveyDesign(
        weights="weight",
        psu=None,
        strata="stratum",
        weight_type="inverse_inclusion",
    )
    fit_options: dict[str, Any] = {}
    if args.covariate_adjusted:
        race = pd.get_dummies(data["race4"].astype(int), prefix="race", dtype=float)
        data = pd.concat([data, race], axis=1)
        covariates = ["sex", "age", "bmi", *race.columns]
        fit_options = {
            "covariates": covariates,
            "cross_fitter": CrossFitter(
                propensity_factory=_WeightedLogit,
                outcome_factory=_WeightedOLS,
                n_splits=3,
                random_state=20_260_730,
            ),
        }
    survey = RepeatedCrossSectionDiD().fit(
        data,
        outcome="soda_usage",
        time="period",
        treatment_time="treatment_time",
        survey_design=design,
        target_population="survey_population",
        **fit_options,
    )
    sample = RepeatedCrossSectionDiD().fit(
        data,
        outcome="soda_usage",
        time="period",
        treatment_time="treatment_time",
    )
    print("Survey-population sensitivity")
    print(survey.summary_frame().to_string())
    print("\nUnweighted sample target")
    print(sample.summary_frame().to_string())
    print("\nSurvey design audit")
    print(survey.survey_design_diagnostics.to_string())
    print("\nWeight audit")
    print(survey.survey_weight_diagnostics.to_string())
    print("\nInterpretation boundary")
    print(survey.causal_interpretation)


if __name__ == "__main__":
    main()

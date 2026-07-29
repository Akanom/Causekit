"""Run CauseKit's model families on four pinned, real datasets.

The workflow covers real-data IV, randomized, observational, matching, native causal ML,
conventional DiD, and efficient-DiD paths. Network access is opt-in, HTTPS-only, and
followed by an exact SHA-256 check. Source files are cached outside the repository and are
not redistributed by CauseKit.

Install the validation extra before running because the nuisance examples use
``statsmodels``::

    python -m pip install -e ".[validation]"
    python examples/real_world_causal_workflow.py --download
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causekit import (
    AIPWATE,
    IPWATE,
    IV2SLS,
    CrossFitter,
    DifferenceInDifferences,
    EfficientDiD,
    NearestNeighborMatch,
    PartiallyLinearDML,
    RandomizedATE,
    RLearner,
)
from causekit.datasets import REAL_DATASETS, load_real_dataset


class _StatsmodelsLogit:
    def fit(self, X: Any, y: Any) -> _StatsmodelsLogitResult:
        try:
            import statsmodels.api as sm
        except ImportError as error:  # pragma: no cover - exercised without validation extra
            raise ImportError("Install CauseKit's 'validation' extra for this example.") from error
        design = sm.add_constant(pd.DataFrame(X), has_constant="add")
        fitted = sm.Logit(np.asarray(y, dtype=float), design).fit(
            method="newton", maxiter=200, tol=1e-11, disp=False
        )
        return _StatsmodelsLogitResult(fitted)


class _StatsmodelsLogitResult:
    def __init__(self, fitted: Any) -> None:
        self.fitted = fitted

    def predict_proba(self, X: Any) -> np.ndarray:
        import statsmodels.api as sm

        design = sm.add_constant(pd.DataFrame(X), has_constant="add")
        treated = np.asarray(self.fitted.predict(design), dtype=float)
        return np.column_stack([1.0 - treated, treated])


class _StatsmodelsOLS:
    def fit(self, X: Any, y: Any) -> _StatsmodelsOLSResult:
        try:
            import statsmodels.api as sm
        except ImportError as error:  # pragma: no cover - exercised without validation extra
            raise ImportError("Install CauseKit's 'validation' extra for this example.") from error
        design = sm.add_constant(pd.DataFrame(X), has_constant="add")
        return _StatsmodelsOLSResult(sm.OLS(np.asarray(y, dtype=float), design).fit())


class _StatsmodelsOLSResult:
    def __init__(self, fitted: Any) -> None:
        self.fitted = fitted

    def predict(self, X: Any) -> np.ndarray:
        import statsmodels.api as sm

        design = sm.add_constant(pd.DataFrame(X), has_constant="add")
        return np.asarray(self.fitted.predict(design), dtype=float)


def _iv_workflow(data: pd.DataFrame) -> None:
    region = pd.get_dummies(
        data["region"].astype(int), prefix="region", drop_first=True, dtype=float
    )
    instruments = pd.concat([data[["faminc"]].astype(float), region], axis=1)
    result = IV2SLS(covariance="robust").fit(
        data["rent"].astype(float),
        endogenous=data[["hsngval"]].astype(float),
        instruments=instruments,
        exogenous=data[["pcturban"]].astype(float),
    )
    print("\nIV2SLS — 1980 Census housing data")
    print(result.summary_frame().to_string())
    print("Identification note: family income and region require an exclusion defense;")
    print("the numerical first stage cannot establish instrument validity.")


def _randomized_workflow(data: pd.DataFrame) -> None:
    covariates = data[["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]].astype(
        float
    )
    result = RandomizedATE(adjustment="lin").fit(
        data["re78"].astype(float),
        treatment=data["treat"].astype(int),
        covariates=covariates,
    )
    print("\nRandomizedATE — National Supported Work experiment")
    print(result.summary_frame().to_string())
    print("Balance diagnostics (standardized mean differences):")
    balance = pd.Series(
        {item.covariate: item.standardized_difference for item in result.balance},
        name="standardized_difference",
    )
    print(balance.to_string())


def _observational_workflow(data: pd.DataFrame) -> None:
    covariate_names = ["mmarried", "mage", "medu", "fbaby"]
    covariates = data[covariate_names].astype(float)
    treatment = data["mbsmoke"].astype(int)
    outcome = data["bweight"].astype(float)
    nuisance = CrossFitter(
        propensity_factory=_StatsmodelsLogit,
        outcome_factory=_StatsmodelsOLS,
        n_splits=5,
        random_state=20_260_729,
    ).fit_predict(covariates, treatment=treatment, outcome=outcome)

    ipw = IPWATE(estimand="ate").fit(
        outcome,
        treatment=treatment,
        propensity=nuisance.propensity,
    )
    aipw = AIPWATE(estimand="ate").fit(
        outcome,
        treatment=treatment,
        propensity=nuisance.propensity,
        outcome_treated=nuisance.outcome_treated,
        outcome_control=nuisance.outcome_control,
    )
    matched = NearestNeighborMatch(
        estimand="att",
        inference="none",
    ).fit(
        outcome,
        treatment=treatment,
        propensity=nuisance.propensity,
        covariates=covariates,
        propensity_score_status="estimated",
        propensity_provenance="five_fold_statsmodels_logit",
    )
    dml = PartiallyLinearDML(n_splits=5, random_state=20_260_729).fit(
        outcome,
        treatment=treatment,
        covariates=covariates,
    )
    rlearner = RLearner(
        n_splits=5,
        evaluation_fraction=0.5,
        random_state=20_260_729,
        calibration_groups=5,
        bootstrap_iterations=999,
    ).fit(
        outcome,
        treatment=treatment,
        covariates=covariates,
    )
    print("\nObservational effects — maternal smoking and birthweight")
    print(pd.concat([ipw.summary_frame(), aipw.summary_frame()], keys=["IPW", "AIPW"]))
    print(matched.summary_frame().rename(index={"att": "matching_att"}).to_string())
    print("\nCauseKit-native partially linear DML")
    print(dml.summary_frame().to_string())
    print("\nFold-level native nuisance tuning")
    print(dml.nuisance_diagnostics.to_string(index=False))
    print("\nCauseKit-native honest R-learner differential calibration")
    print(rlearner.summary_frame().to_string())
    print("\nHonest R-loss comparison")
    print(
        pd.Series(
            {
                "r_loss": rlearner.honest_r_loss,
                "constant_r_loss": rlearner.honest_constant_r_loss,
                "r_loss_gain": rlearner.r_loss_gain,
            }
        ).to_string()
    )
    print("\nTie-preserving honest calibration groups")
    print(rlearner.calibration_plot_data().to_string())
    print("The DML coefficient is an ATE only under a constant-effect partially linear")
    print("model and the documented exchangeability, variation, and nuisance conditions.")
    print("Matching uncertainty is intentionally absent: cross-fitted scores do not satisfy")
    print("CauseKit's narrow full-sample Logit-MLE analytical-inference contract.")
    print("R-learner evidence is split-conditional. Group effects are overlap-weighted")
    print("residual moments; no unit-level CATE intervals or policy-value claim is made.")


def _hospital_panel(data: pd.DataFrame) -> pd.DataFrame:
    panel = (
        data.groupby(["hospital", "month"], as_index=False)
        .agg(outcome=("satis", "mean"), treated=("procedure", "max"))
        .sort_values(["hospital", "month"], kind="stable")
        .reset_index(drop=True)
    )
    first_treated = panel.loc[panel["treated"].eq(1)].groupby("hospital", sort=False)["month"].min()
    panel["treatment_time"] = panel["hospital"].map(first_treated).fillna(np.inf)
    return panel


def _did_workflow(data: pd.DataFrame) -> None:
    panel = _hospital_panel(data)
    fit_arguments = {
        "outcome": "outcome",
        "entity": "hospital",
        "time": "month",
        "treatment_time": "treatment_time",
    }
    conventional = DifferenceInDifferences().fit(panel, **fit_arguments)
    efficient = EfficientDiD(pre_periods="all").fit(panel, **fit_arguments)
    results = pd.DataFrame(
        {
            "estimate": [conventional.estimate, efficient.estimate],
            "std_err": [conventional.standard_error, efficient.standard_error],
        },
        index=["conventional", "efficient_pt_all"],
    )
    print("\nDifference-in-differences — hospital procedure adoption")
    print(results.to_string())
    print("The efficient estimate is opt-in: it requires the stronger PT-All restriction;")
    print("it does not replace the conventional post-treatment parallel-trends analysis.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download absent pinned datasets over verified HTTPS.",
    )
    args = parser.parse_args()

    loaded = {
        name: load_real_dataset(
            name,
            data_directory=args.data_directory,
            download=args.download,
        )
        for name in REAL_DATASETS
    }
    for name, data in loaded.items():
        print(f"Verified {name}: {len(data)} source rows ({REAL_DATASETS[name].sha256})")

    _iv_workflow(loaded["hsng"])
    _randomized_workflow(loaded["nsw_mixtape"])
    _observational_workflow(loaded["cattaneo2"])
    _did_workflow(loaded["hospdd"])


if __name__ == "__main__":
    main()

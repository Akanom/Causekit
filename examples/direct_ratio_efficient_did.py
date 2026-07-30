"""Provider-neutral direct cohort-odds nuisance example for efficient PT-All DiD.

Run from the repository root with:

    python examples/direct_ratio_efficient_did.py

The pairwise Logit and outcome regressions are example providers, not CauseKit-owned
defaults. Production providers should expose their own tuning and calibration audits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from causekit import CrossFitter, EfficientDiD


class _LogitOddsResult:
    odds_ratio_kind_ = "posterior_cohort_odds"
    numerator_class_ = 1
    denominator_class_ = 0

    def __init__(self, coefficients: np.ndarray, converged: bool) -> None:
        self.coefficients = coefficients
        self.converged = converged

    def predict_odds_ratio(self, X: pd.DataFrame) -> pd.Series:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        return pd.Series(np.exp(design @ self.coefficients), index=X.index)

    def nuisance_diagnostics(self) -> dict[str, bool]:
        return {"optimizer_converged": self.converged}


class PairwiseLogitOdds:
    """Small example provider returning calibrated binary posterior odds."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _LogitOddsResult:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        target = y.to_numpy(dtype=float)

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            linear = design @ coefficients
            loss = float(np.logaddexp(0.0, linear).sum() - target @ linear)
            probability = 1.0 / (1.0 + np.exp(-linear))
            gradient = design.T @ (probability - target)
            return loss, gradient

        fit = minimize(
            objective,
            np.zeros(design.shape[1]),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-9, "maxiter": 500},
        )
        if not np.isfinite(fit.x).all():
            raise ValueError("The example pairwise Logit returned non-finite coefficients.")
        return _LogitOddsResult(fit.x, bool(fit.success))


class _LinearResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X: pd.DataFrame) -> pd.Series:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        return pd.Series(design @ self.coefficients, index=X.index)


class LinearRegression:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _LinearResult:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        coefficients = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)[0]
        return _LinearResult(coefficients)


def make_panel(seed: int = 20260730) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    for cohort, prefix, location, effect in (
        (2.0, "treated", 0.35, 2.0),
        (np.inf, "never", -0.35, 0.0),
    ):
        x = rng.normal(loc=location, scale=1.0, size=240)
        baseline_error = rng.normal(scale=0.7, size=240)
        change_error = rng.normal(scale=0.7, size=240)
        for position in range(240):
            baseline = 1.0 + 0.8 * x[position] + baseline_error[position]
            change = 0.4 + 0.5 * x[position] + effect + change_error[position]
            for period, outcome in ((1.0, baseline), (2.0, baseline + change)):
                rows.append(
                    {
                        "unit": f"{prefix}_{position}",
                        "period": period,
                        "first_treated": cohort,
                        "baseline_x": x[position],
                        "outcome": outcome,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    result = EfficientDiD(pre_periods="all").fit(
        make_panel(),
        outcome="outcome",
        entity="unit",
        time="period",
        treatment_time="first_treated",
        covariates=["baseline_x"],
        cross_fitter=CrossFitter(
            cohort_ratio_factory=PairwiseLogitOdds,
            outcome_factory=LinearRegression,
            n_splits=5,
            random_state=20260730,
        ),
    )
    print(result.summary_frame())
    print("\nWeighting route:", result.nuisance_weighting)
    print("\nOrdered cohort odds:")
    print(result.cohort_ratios.describe())
    print("\nPair/fold support diagnostics:")
    print(result.cohort_ratio_diagnostics)


if __name__ == "__main__":
    main()

"""Runnable covariate-adjusted repeated-cross-section DiD example.

The small nuisance classes below demonstrate CauseKit's provider-neutral protocol; they
are example adapters, not public CauseKit estimators. Applications may replace them with
any fresh fit-capable models exposing the same ``predict_proba`` / ``predict`` methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from causekit import CrossFitter, RepeatedCrossSectionDiD


class _LogitResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        probability = expit(design @ self.coefficients)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _LogitMLE:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _LogitResult:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        assigned = y.to_numpy(dtype=float)

        def objective(coefficients: np.ndarray) -> float:
            linear = design @ coefficients
            return float(np.sum(np.logaddexp(0.0, linear) - assigned * linear))

        fitted = minimize(objective, np.zeros(design.shape[1]), method="BFGS")
        if not fitted.success:
            raise ValueError(f"Example propensity fit failed: {fitted.message}")
        return _LogitResult(np.asarray(fitted.x, dtype=float))


class _OLSResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        return design @ self.coefficients


class _OLS:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _OLSResult:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        coefficients = np.linalg.solve(design.T @ design, design.T @ y.to_numpy(dtype=float))
        return _OLSResult(coefficients)


def repeated_samples(seed: int = 20_260_730) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for period in (1.0, 2.0, 3.0):
        x = rng.normal(size=600)
        propensity = expit(0.2 + 0.5 * x)
        cohort = rng.binomial(1, propensity)
        post = float(period == 3.0)
        outcome = (
            1.0
            + x
            + 1.5 * cohort
            + 0.4 * period
            + 0.2 * period * x
            + 2.0 * cohort * post
            + rng.normal(size=len(x))
        )
        rows.extend(
            {
                "outcome": float(outcome[position]),
                "period": period,
                "first_treated": 3.0 if cohort[position] else np.inf,
                "baseline_risk": float(x[position]),
            }
            for position in range(len(x))
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = repeated_samples()
    result = RepeatedCrossSectionDiD().fit(
        data,
        outcome="outcome",
        time="period",
        treatment_time="first_treated",
        covariates=["baseline_risk"],
        cross_fitter=CrossFitter(
            propensity_factory=_LogitMLE,
            outcome_factory=_OLS,
            n_splits=3,
            random_state=20_260_730,
        ),
    )
    print(result.summary_frame().to_string())
    print("\nConditional pre-trend")
    print(result.pretrend.placebo_effects.to_string())
    print("\nFold/task diagnostics")
    print(result.nuisance_diagnostics.to_string(index=False))
    print("\nCausal interpretation requires every recorded assumption:")
    print("\n".join(f"- {assumption}" for assumption in result.assumptions))


if __name__ == "__main__":
    main()

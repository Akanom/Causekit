"""Fixed-period large-n smoke for covariate repeated-cross-section DiD.

Run from the repository root::

    python benchmarks/benchmark_did_rcs_covariate.py --n-observations 100000 --measure-memory

The benchmark providers are deliberately simple. This measures CauseKit's task planning,
cross-fitting, score, audit, aggregation, and inference path—not a learner tournament.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc

import numpy as np
import pandas as pd

import causekit
from causekit import CrossFitter, RepeatedCrossSectionDiD


def _sample(n_observations: int, n_periods: int, seed: int) -> pd.DataFrame:
    if n_observations < n_periods * 12:
        raise ValueError("n_observations must provide at least 12 rows per period.")
    if n_periods < 5:
        raise ValueError("n_periods must be at least five.")
    rng = np.random.default_rng(seed)
    row = np.arange(n_observations)
    period = row % n_periods + 1
    cohort_support = np.array([3.0, float(n_periods - 1), np.inf])
    cohort = cohort_support[(row // n_periods) % len(cohort_support)]
    baseline = rng.normal(scale=2.0, size=n_observations)
    treated = np.isfinite(cohort) & (period >= cohort)
    event_time = period - cohort
    effect = np.where(treated, 1.0 + 0.2 * event_time, 0.0)
    outcome = baseline + 0.35 * period + effect + rng.normal(scale=0.5, size=n_observations)
    return pd.DataFrame(
        {
            "outcome": outcome,
            "time": period,
            "treatment_time": cohort,
        }
    )


class _ProbabilityResult:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probability = np.full(len(X), self.probability)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _EmpiricalProbability:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ProbabilityResult:
        del X
        return _ProbabilityResult(float(y.mean()))


class _AffineResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        return design @ self.coefficients


class _AffineOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _AffineResult:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        coefficients = np.linalg.solve(
            design.T @ design,
            design.T @ y.to_numpy(dtype=float),
        )
        return _AffineResult(coefficients)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-observations", type=int, default=100_000)
    parser.add_argument("--periods", type=int, default=6)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20_260_730)
    parser.add_argument("--measure-memory", action="store_true")
    args = parser.parse_args()

    data = _sample(args.n_observations, args.periods, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    data["baseline_risk"] = rng.normal(size=len(data))
    fitter = CrossFitter(
        propensity_factory=_EmpiricalProbability,
        outcome_factory=_AffineOutcome,
        n_splits=args.folds,
        random_state=args.seed,
    )
    if args.measure_memory:
        tracemalloc.start()
    started = time.perf_counter()
    result = RepeatedCrossSectionDiD(control_group="not_yet_treated").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["baseline_risk"],
        cross_fitter=fitter,
    )
    elapsed = time.perf_counter() - started
    peak_mib = None
    if args.measure_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mib = peak / 1024**2
    if not np.isfinite(result.estimate) or not np.isfinite(result.standard_error):
        raise RuntimeError("benchmark produced a non-finite estimate or standard error")
    print(
        json.dumps(
            {
                "causekit_version": causekit.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "seed": args.seed,
                "n_observations": len(data),
                "n_periods": args.periods,
                "n_splits": result.n_splits,
                "elapsed_seconds": elapsed,
                "peak_python_mib": peak_mib,
                "estimate": result.estimate,
                "standard_error": result.standard_error,
                "group_time_effects": len(result.group_time),
                "conditional_placebos": result.pretrend.n_restrictions,
                "nuisance_task_fits": len(result.nuisance_diagnostics),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

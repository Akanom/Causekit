"""Run a compact, dependency-light CauseKit workflow.

The example uses deterministic simulated designs so it runs offline with only
CauseKit's core dependencies. Each design makes its identifying assumptions
visible; the fitted diagnostics are evidence about implementation and relevance,
not substitutes for a research-design argument.

Run from the repository root::

    python examples/00_quickstart.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causekit import (
    IV2SLS,
    DiDResult,
    DifferenceInDifferences,
    IV2SLSResult,
    RandomizedATE,
    RandomizedATEResult,
)

SEED = 20_260_730


def randomized_example() -> RandomizedATEResult:
    """Estimate a Lin-adjusted average effect in a simulated randomized trial."""

    rng = np.random.default_rng(SEED)
    nobs = 600
    baseline = rng.normal(size=nobs)
    assigned = rng.binomial(1, 0.5, size=nobs)
    outcome = 1.5 * assigned + 0.8 * baseline + rng.normal(size=nobs)
    return RandomizedATE(adjustment="lin", covariance="robust").fit(
        outcome,
        treatment=assigned,
        covariates=pd.DataFrame({"baseline": baseline}),
    )


def iv_example() -> IV2SLSResult:
    """Estimate a structural coefficient in a simulated valid-IV design."""

    rng = np.random.default_rng(SEED + 1)
    nobs = 800
    encouragement = rng.normal(size=nobs)
    baseline = rng.normal(size=nobs)
    confounder = rng.normal(size=nobs)
    treatment = 0.9 * encouragement + 0.4 * baseline + 0.7 * confounder + rng.normal(size=nobs)
    outcome = 2.0 * treatment + 0.5 * baseline + confounder + rng.normal(size=nobs)
    index = pd.RangeIndex(nobs, name="observation")
    return IV2SLS(covariance="robust").fit(
        pd.Series(outcome, index=index, name="outcome"),
        endogenous=pd.DataFrame({"treatment": treatment}, index=index),
        instruments=pd.DataFrame({"encouragement": encouragement}, index=index),
        exogenous=pd.DataFrame({"baseline": baseline}, index=index),
    )


def did_example() -> DiDResult:
    """Estimate conventional staggered DiD with never-treated comparisons."""

    rng = np.random.default_rng(SEED + 2)
    rows: list[dict[str, float | str]] = []
    periods = range(4)
    for position in range(80):
        treatment_time = 2.0 if position < 40 else np.inf
        unit_level = rng.normal()
        for period in periods:
            treated = np.isfinite(treatment_time) and period >= treatment_time
            rows.append(
                {
                    "entity": f"unit_{position}",
                    "time": float(period),
                    "treatment_time": treatment_time,
                    "outcome": unit_level
                    + 0.4 * period
                    + 1.25 * float(treated)
                    + rng.normal(scale=0.4),
                }
            )
    return DifferenceInDifferences(control_group="never_treated").fit(
        pd.DataFrame(rows),
        outcome="outcome",
        entity="entity",
        time="time",
        treatment_time="treatment_time",
    )


def main() -> None:
    randomized = randomized_example()
    iv = iv_example()
    did = did_example()

    print("Randomized trial — Lin-adjusted ATE")
    print(randomized.summary_frame().to_string())
    print("\nInstrumental variables — robust 2SLS")
    print(iv.summary_frame().to_string())
    print("First-stage partial R-squared:", iv.first_stage["treatment"].partial_r_squared)
    print("\nConventional DiD — event-study effects")
    print(did.event_study.to_string(index=False))
    print("\nInterpretation boundaries")
    print("- RandomizedATE requires the declared assignment design and no interference.")
    print(
        "- IV2SLS requires relevance, independence, and exclusion; diagnostics do not prove them."
    )
    print("- DiD requires defensible comparison cohorts and parallel untreated trends.")


if __name__ == "__main__":
    main()

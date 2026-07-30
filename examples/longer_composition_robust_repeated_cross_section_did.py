"""Longer and staggered composition-robust repeated-section DiD example.

The example uses small provider-neutral nuisance adapters from the pairwise example.
Application code should replace those factories with suitable classifiers and regressors
while preserving CauseKit's labelled ``predict_proba`` and ``predict`` protocols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from composition_robust_repeated_cross_section_did import _OLS, _MultinomialLogit

from causekit import CrossFitter, RepeatedCrossSectionDiD, did_rcs_composition_test


def repeated_samples(seed: int = 20_260_730) -> pd.DataFrame:
    """Generate independent repeated samples with shifting observed composition."""
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for period in range(1, 6):
        x = rng.normal(loc=0.18 * period, scale=1.0, size=900)
        logits = np.column_stack(
            [
                -0.15 + 0.45 * x + 0.08 * period,
                -0.30 - 0.25 * x + 0.05 * period,
                np.zeros(len(x)),
            ]
        )
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        cohort_code = np.array([3.0, 4.0, np.inf])[[rng.choice(3, p=row) for row in probabilities]]
        event_time = period - cohort_code
        treated = np.isfinite(cohort_code) & (event_time >= 0)
        untreated_mean = 1.0 + 0.45 * period + 0.55 * x + 0.12 * x**2
        treatment_effect = np.where(treated, 1.4 + 0.20 * x + 0.12 * event_time, 0.0)
        rows.append(
            pd.DataFrame(
                {
                    "outcome": untreated_mean + treatment_effect + rng.normal(size=len(x)),
                    "period": float(period),
                    "first_treated": cohort_code,
                    "baseline_risk": x,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _cross_fitter() -> CrossFitter:
    return CrossFitter(
        propensity_factory=_MultinomialLogit,
        outcome_factory=_OLS,
        n_splits=3,
        random_state=20_260_730,
    )


def main() -> None:
    data = repeated_samples()
    common = {
        "outcome": "outcome",
        "time": "period",
        "treatment_time": "first_treated",
        "covariates": ["baseline_risk"],
    }
    robust = RepeatedCrossSectionDiD(
        control_group="not_yet_treated",
        composition="robust",
        inference="multiplier_bootstrap",
        bootstrap_iterations=999,
        random_state=20_260_730,
    ).fit(data, cross_fitter=_cross_fitter(), **common)
    stationary = RepeatedCrossSectionDiD(
        control_group="not_yet_treated",
        composition="stationary",
        inference="multiplier_bootstrap",
        bootstrap_iterations=999,
        random_state=20_260_730,
    ).fit(data, cross_fitter=_cross_fitter(), **common)
    diagnostic = did_rcs_composition_test(robust, stationary)

    print("Composition-robust group-time effects")
    print(robust.group_time.to_string())
    print("\nConditional pre-treatment placebos")
    print(robust.pretrend.placebo_effects.to_string())
    print("\nSimultaneous event-study bands")
    print(robust.simultaneous_event_study.to_string())
    print("\nPair and overlap ledger")
    print(robust.pair_ledger.to_string(index=False))
    print("\nRobust-minus-stationary equality diagnostic")
    print(diagnostic.summary_frame().to_string())
    print(f"joint {diagnostic.distribution} p-value: {diagnostic.pvalue:.6f}")
    print("The diagnostic reports sensitivity; it does not select an estimator.")


if __name__ == "__main__":
    main()

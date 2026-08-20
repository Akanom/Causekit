"""Pairwise composition-change-robust repeated-section DiD example.

The nuisance models are small example adapters, not CauseKit estimators. Replace their
factories with appropriate application models while preserving the public
``predict_proba`` and ``predict`` contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from causekit import CrossFitter, RepeatedCrossSectionDiD


class _MultinomialResult:
    def __init__(self, coefficients: np.ndarray, classes: np.ndarray) -> None:
        self.coefficients = coefficients
        self.classes = classes

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        logits = np.column_stack([design @ self.coefficients.T, np.zeros(len(X))])
        probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        return pd.DataFrame(probabilities, index=X.index, columns=self.classes)


class _MultinomialLogit:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _MultinomialResult:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        classes = np.sort(y.unique())
        positions = pd.Index(classes).get_indexer(y)
        if len(classes) < 2 or np.any(positions < 0):
            raise ValueError("The example requires at least two labelled classes per fold.")
        n_parameters = (len(classes) - 1) * design.shape[1]

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients = flat.reshape(len(classes) - 1, design.shape[1])
            logits = np.column_stack([design @ coefficients.T, np.zeros(len(X))])
            probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
            loss = -float(np.log(probabilities[np.arange(len(X)), positions]).sum())
            targets = np.eye(len(classes))[positions, :-1]
            gradient = (probabilities[:, :-1] - targets).T @ design
            return loss, gradient.ravel()

        fitted = minimize(
            objective,
            np.zeros(n_parameters),
            jac=True,
            method="BFGS",
        )
        if not fitted.success:
            raise ValueError(f"Example generalized-propensity fit failed: {fitted.message}")
        return _MultinomialResult(
            fitted.x.reshape(len(classes) - 1, design.shape[1]),
            classes,
        )


class _OLSResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        return design @ self.coefficients


class _OLS:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _OLSResult:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        coefficients, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
        return _OLSResult(coefficients)


def repeated_samples(seed: int = 20_260_730) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for period in (1.0, 2.0):
        post = float(period == 2.0)
        x = rng.normal(loc=-0.45 + 0.9 * post, size=800)
        treated_probability = 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * x + 0.4 * post)))
        treated = rng.binomial(1, treated_probability)
        m00 = 1.0 + 0.5 * x
        m01 = 2.0 + 0.9 * x
        m10 = 2.5 + 0.3 * x
        effect = 2.0 + 0.25 * x
        untreated_target_for_treated = m10 + m01 - m00
        mean = np.where(
            (treated == 0) & (post == 0.0),
            m00,
            np.where(
                (treated == 0) & (post == 1.0),
                m01,
                np.where(post == 0.0, m10, untreated_target_for_treated + effect),
            ),
        )
        outcome = mean + rng.normal(size=len(x))
        rows.extend(
            {
                "outcome": float(outcome[position]),
                "period": period,
                "first_treated": 2.0 if treated[position] else np.inf,
                "baseline_risk": float(x[position]),
                "true_effect": float(effect[position]),
            }
            for position in range(len(x))
        )
    return pd.DataFrame(rows)


def _displayable_nuisance_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    displayed = diagnostics.copy()
    displayed["declared_class_count"] = displayed["declared_class_count"].map(
        lambda value: "not applicable" if pd.isna(value) else str(int(value))
    )
    if displayed.isna().any().any():
        raise RuntimeError("The displayed nuisance diagnostics contain an unexplained gap.")
    return displayed


def main() -> None:
    data = repeated_samples()
    result = RepeatedCrossSectionDiD(composition="robust").fit(
        data,
        outcome="outcome",
        time="period",
        treatment_time="first_treated",
        covariates=["baseline_risk"],
        cross_fitter=CrossFitter(
            propensity_factory=_MultinomialLogit,
            outcome_factory=_OLS,
            n_splits=3,
            random_state=20_260_730,
        ),
    )
    target = (data["first_treated"] == 2.0) & (data["period"] == 2.0)
    print(result.summary_frame().to_string())
    print(f"\nSample target truth: {data.loc[target, 'true_effect'].mean():.6f}")
    print(f"Target population: {result.target_population}")
    print("\nNormalized composition weights")
    print(result.composition_weights.describe().to_string())
    print("\nFold/task diagnostics")
    print(_displayable_nuisance_diagnostics(result.nuisance_diagnostics).to_string(index=False))


if __name__ == "__main__":
    main()

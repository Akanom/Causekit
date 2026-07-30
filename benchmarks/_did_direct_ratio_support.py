"""Validation-only nuisance providers for direct-ratio DiD promotion harnesses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def _design(X: pd.DataFrame, *, degree: int) -> np.ndarray:
    values = X.to_numpy(dtype=float)
    columns = [np.ones(len(values)), values]
    if degree >= 2:
        columns.append(values**2)
    return np.column_stack(columns)


def _fit_binary_logit(
    X: pd.DataFrame,
    target: np.ndarray,
    *,
    degree: int,
    penalty: float,
) -> tuple[np.ndarray, bool, int, float]:
    design = _design(X, degree=degree)
    penalty_mask = np.r_[0.0, np.ones(design.shape[1] - 1)]

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ coefficients
        loss = float(
            np.logaddexp(0.0, linear).sum()
            - target @ linear
            + 0.5 * penalty * np.sum(penalty_mask * coefficients**2)
        )
        gradient = design.T @ (expit(linear) - target) + penalty * penalty_mask * coefficients
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 500},
    )
    if not np.isfinite(fitted.x).all():
        raise ValueError("Validation Logit returned non-finite coefficients.")
    gradient_norm = float(np.max(np.abs(fitted.jac)))
    # SciPy can report precision loss after reaching a materially negligible score.
    converged = bool(fitted.success or gradient_norm <= 5e-6)
    if not converged:
        raise ValueError(
            "Validation Logit failed its fixed gradient gate; no fallback was applied."
        )
    return fitted.x, converged, int(fitted.nit), gradient_norm


class BinaryLogitOddsResult:
    odds_ratio_kind_ = "posterior_cohort_odds"
    numerator_class_ = 1
    denominator_class_ = 0

    def __init__(
        self,
        coefficients: np.ndarray,
        *,
        degree: int,
        converged: bool,
        iterations: int,
        gradient_norm: float,
    ) -> None:
        self.coefficients = coefficients
        self.degree = degree
        self.converged = converged
        self.iterations = iterations
        self.gradient_norm = gradient_norm

    def predict_odds_ratio(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(
            np.exp(_design(X, degree=self.degree) @ self.coefficients),
            index=X.index,
        )

    def nuisance_diagnostics(self) -> dict[str, Any]:
        return {
            "optimizer_converged": self.converged,
            "optimizer_iterations": self.iterations,
            "optimizer_gradient_norm": self.gradient_norm,
        }


class BinaryLogitOdds:
    def __init__(self, *, degree: int = 1, penalty: float = 1e-8) -> None:
        self.degree = degree
        self.penalty = penalty

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BinaryLogitOddsResult:
        coefficients, converged, iterations, gradient_norm = _fit_binary_logit(
            X,
            y.to_numpy(dtype=float),
            degree=self.degree,
            penalty=self.penalty,
        )
        return BinaryLogitOddsResult(
            coefficients,
            degree=self.degree,
            converged=converged,
            iterations=iterations,
            gradient_norm=gradient_norm,
        )


class BinaryClassProbabilityResult:
    def __init__(
        self,
        classes: np.ndarray,
        coefficients: np.ndarray,
        *,
        degree: int,
        converged: bool,
        iterations: int,
        gradient_norm: float,
    ) -> None:
        self.classes_ = classes
        self.coefficients = coefficients
        self.degree = degree
        self.converged = converged
        self.iterations = iterations
        self.gradient_norm = gradient_norm

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probability_one = expit(_design(X, degree=self.degree) @ self.coefficients)
        return pd.DataFrame(
            {self.classes_[0]: 1.0 - probability_one, self.classes_[1]: probability_one},
            index=X.index,
        )

    def nuisance_diagnostics(self) -> dict[str, Any]:
        return {
            "optimizer_converged": self.converged,
            "optimizer_iterations": self.iterations,
            "optimizer_gradient_norm": self.gradient_norm,
        }


class BinaryClassProbability:
    def __init__(self, *, degree: int = 1, penalty: float = 1e-8) -> None:
        self.degree = degree
        self.penalty = penalty

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BinaryClassProbabilityResult:
        classes = np.unique(y.to_numpy())
        if len(classes) != 2:
            raise ValueError("BinaryClassProbability requires exactly two cohorts.")
        target = y.to_numpy() == classes[1]
        coefficients, converged, iterations, gradient_norm = _fit_binary_logit(
            X,
            target.astype(float),
            degree=self.degree,
            penalty=self.penalty,
        )
        return BinaryClassProbabilityResult(
            classes,
            coefficients,
            degree=self.degree,
            converged=converged,
            iterations=iterations,
            gradient_norm=gradient_norm,
        )


class RegressionResult:
    def __init__(self, coefficients: np.ndarray, *, degree: int) -> None:
        self.coefficients = coefficients
        self.degree = degree

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(
            _design(X, degree=self.degree) @ self.coefficients,
            index=X.index,
        )


class PolynomialRegression:
    def __init__(self, *, degree: int = 1) -> None:
        self.degree = degree

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RegressionResult:
        design = _design(X, degree=self.degree)
        coefficients = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)[0]
        return RegressionResult(coefficients, degree=self.degree)


class MeanResult:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=X.index)


class MeanRegression:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> MeanResult:
        return MeanResult(float(y.mean()))


class OracleOddsResult:
    odds_ratio_kind_ = "posterior_cohort_odds"

    def __init__(self, odds: Callable[[pd.DataFrame], np.ndarray]) -> None:
        self.odds = odds

    def predict_odds_ratio(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.odds(X), index=X.index)


class OracleOdds:
    def __init__(self, odds: Callable[[pd.DataFrame], np.ndarray]) -> None:
        self.odds = odds

    def fit(self, X: pd.DataFrame, y: pd.Series) -> OracleOddsResult:
        return OracleOddsResult(self.odds)


class EmpiricalOddsResult:
    odds_ratio_kind_ = "posterior_cohort_odds"

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def predict_odds_ratio(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.ratio, index=X.index)


class EmpiricalOdds:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> EmpiricalOddsResult:
        probability = float(y.mean())
        return EmpiricalOddsResult(probability / (1.0 - probability))

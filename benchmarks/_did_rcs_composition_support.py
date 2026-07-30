"""Benchmark-only nuisance providers for composition-robust repeated-section DiD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _numeric_frame(X: Any) -> tuple[np.ndarray, pd.Index]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Composition benchmark nuisances require labelled DataFrames.")
    values = X.to_numpy(dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Composition benchmark covariates must be a finite matrix.")
    return values, X.columns.copy()


@dataclass(frozen=True)
class _SoftmaxResult:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: pd.Index
    classes: pd.Index
    alpha: float
    iterations: int
    objective: float

    def predict_proba(self, X: Any) -> pd.DataFrame:
        raw, names = _numeric_frame(X)
        if not names.equals(self.feature_names):
            raise ValueError("Probability prediction columns must match the fitted schema.")
        standardized = (raw - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(raw)), standardized])
        logits = np.column_stack([design @ self.coefficients.T, np.zeros(len(raw))])
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return pd.DataFrame(probabilities, index=X.index, columns=self.classes.copy())

    def nuisance_diagnostics(self) -> dict[str, float | int]:
        return {
            "penalty_alpha": self.alpha,
            "optimizer_iterations": self.iterations,
            "training_objective": self.objective,
        }


class PenalizedSoftmax:
    """Standardized reference-class ridge softmax for the real-data sensitivity run."""

    def __init__(self, *, alpha: float = 0.5, max_iterations: int = 2_000) -> None:
        self.alpha = alpha
        self.max_iterations = max_iterations

    def fit(self, X: Any, y: Any) -> _SoftmaxResult:
        raw, names = _numeric_frame(X)
        labels = pd.Series(y, index=X.index)
        classes = pd.Index(sorted(pd.unique(labels)), name="class")
        if len(classes) < 2:
            raise ValueError("Penalized softmax requires at least two classes.")
        codes = classes.get_indexer(labels)
        if np.any(codes < 0):
            raise RuntimeError("Softmax class coding failed.")
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0, ddof=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        design = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
        n_classes = len(classes)
        n_parameters = (n_classes - 1) * design.shape[1]

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients = flat.reshape(n_classes - 1, design.shape[1])
            logits = np.column_stack([design @ coefficients.T, np.zeros(len(raw))])
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            likelihood = -float(np.log(probabilities[np.arange(len(raw)), codes]).mean())
            penalty = 0.5 * self.alpha * float((coefficients[:, 1:] ** 2).sum())
            residual = probabilities
            residual[np.arange(len(raw)), codes] -= 1.0
            gradient = residual[:, :-1].T @ design / len(raw)
            gradient[:, 1:] += self.alpha * coefficients[:, 1:]
            return likelihood + penalty, gradient.ravel()

        optimized = minimize(
            objective,
            np.zeros(n_parameters),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iterations, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not optimized.success:
            raise ValueError(f"Penalized softmax did not converge: {optimized.message}.")
        return _SoftmaxResult(
            coefficients=optimized.x.reshape(n_classes - 1, design.shape[1]),
            feature_mean=mean,
            feature_scale=scale,
            feature_names=names,
            classes=classes,
            alpha=float(self.alpha),
            iterations=int(optimized.nit),
            objective=float(optimized.fun),
        )


@dataclass(frozen=True)
class _RidgeResult:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: pd.Index
    alpha: float
    rank: int

    def predict(self, X: Any) -> np.ndarray:
        raw, names = _numeric_frame(X)
        if not names.equals(self.feature_names):
            raise ValueError("Outcome prediction columns must match the fitted schema.")
        design = np.column_stack(
            [np.ones(len(raw)), (raw - self.feature_mean) / self.feature_scale]
        )
        return design @ self.coefficients

    def nuisance_diagnostics(self) -> dict[str, float | int]:
        return {"penalty_alpha": self.alpha, "numerical_rank": self.rank}


class StandardizedRidgeOutcome:
    """Deterministic benchmark-only ridge outcome regression."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> _RidgeResult:
        raw, names = _numeric_frame(X)
        target = np.asarray(y, dtype=float)
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0, ddof=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        design = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return _RidgeResult(
            coefficients=coefficients,
            feature_mean=mean,
            feature_scale=scale,
            feature_names=names,
            alpha=float(self.alpha),
            rank=int(np.linalg.matrix_rank(design)),
        )


@dataclass(frozen=True)
class _SaturatedProbabilityResult:
    probabilities: np.ndarray
    classes: pd.Index

    def predict_proba(self, X: Any) -> pd.DataFrame:
        x = np.asarray(X["x"], dtype=int)
        if np.any((x < 0) | (x >= len(self.probabilities))):
            raise ValueError("Saturated probability received an unknown x category.")
        return pd.DataFrame(self.probabilities[x], index=X.index, columns=self.classes.copy())

    def nuisance_diagnostics(self) -> dict[str, float | int]:
        return {
            "minimum_training_probability": float(self.probabilities.min()),
            "maximum_training_probability": float(self.probabilities.max()),
            "x_categories": int(len(self.probabilities)),
        }


class SaturatedClassProbability:
    """Fold-fitted cell probabilities saturated in the binary simulation covariate."""

    def fit(self, X: Any, y: Any) -> _SaturatedProbabilityResult:
        x = np.asarray(X["x"], dtype=int)
        classes = pd.Index(sorted(pd.unique(y)), name="class")
        if set(np.unique(x)) != {0, 1}:
            raise ValueError("Each probability training fold must contain both x levels.")
        labels = np.asarray(y)
        probabilities = np.empty((2, len(classes)))
        for category in (0, 1):
            selected = labels[x == category]
            probabilities[category] = [float(np.mean(selected == label)) for label in classes]
        if np.any(probabilities <= 0.0):
            raise ValueError("Every x-by-class training cell must be populated.")
        return _SaturatedProbabilityResult(probabilities=probabilities, classes=classes)


@dataclass(frozen=True)
class _QuadraticOutcomeResult:
    coefficients: np.ndarray

    @staticmethod
    def _design(X: Any) -> np.ndarray:
        x = np.asarray(X["x"], dtype=float)
        z = np.asarray(X["z"], dtype=float)
        return np.column_stack([np.ones(len(x)), x, z, z**2])

    def predict(self, X: Any) -> np.ndarray:
        return self._design(X) @ self.coefficients

    def nuisance_diagnostics(self) -> dict[str, int]:
        return {"polynomial_terms": 4}


class QuadraticOutcome:
    """Fold-fitted correctly specified nonlinear outcome nuisance for promotion DGPs."""

    def fit(self, X: Any, y: Any) -> _QuadraticOutcomeResult:
        design = _QuadraticOutcomeResult._design(X)
        coefficients, _, rank, _ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)
        if rank != design.shape[1]:
            raise ValueError("Quadratic outcome training design is rank deficient.")
        return _QuadraticOutcomeResult(coefficients=coefficients)


__all__ = [
    "PenalizedSoftmax",
    "QuadraticOutcome",
    "SaturatedClassProbability",
    "StandardizedRidgeOutcome",
]

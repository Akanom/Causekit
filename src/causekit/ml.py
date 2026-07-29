"""Dependency-free specialized causal machine-learning estimators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from ._covariance import CovarianceType, ols_covariance
from .crossfit import (
    CrossFitTask,
    CrossFitter,
    NuisanceFactory,
    PredictionAdapter,
    _frame,
    _labels,
    _series,
)

DMLCovariance = Literal["robust", "clustered"]
TreatmentKind = Literal["binary", "continuous"]

DEFAULT_RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0, 10_000.0)


@dataclass(frozen=True)
class NativeRidgeCVResult:
    """Fold-local fitted state for CauseKit's native ridge-GCV nuisance learner."""

    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: pd.Index
    selected_alpha: float
    effective_df: float

    def predict(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if not X.columns.equals(self.feature_names):
                raise ValueError(
                    "Prediction covariate columns must match the fitted schema exactly."
                )
            raw = X.to_numpy(dtype=float)
        else:
            raw = np.asarray(X, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(self.feature_names):
            raise ValueError("Prediction covariates must match the fitted feature dimension.")
        if not np.isfinite(raw).all():
            raise ValueError("Prediction covariates must contain only finite values.")
        standardized = (raw - self.feature_mean) / self.feature_scale
        return self.intercept + standardized @ self.coefficients


class _NativeRidgeCV:
    """Standardized ridge regression with fold-local generalized cross-validation."""

    def __init__(self, alphas: tuple[float, ...]) -> None:
        self.alphas = alphas

    def fit(self, X: Any, y: Any) -> NativeRidgeCVResult:
        if not isinstance(X, pd.DataFrame):  # CrossFitter currently guarantees this.
            raise TypeError("Native ridge nuisance fitting requires labelled covariates.")
        raw = X.to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(raw):
            raise ValueError("Native ridge target must contain one value per row.")
        if len(raw) < 2:
            raise ValueError("Native ridge nuisance fitting requires at least two rows.")

        feature_mean = raw.mean(axis=0)
        feature_scale = raw.std(axis=0, ddof=0)
        feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
        standardized = (raw - feature_mean) / feature_scale
        intercept = float(target.mean())
        centered_target = target - intercept
        left, singular_values, right = np.linalg.svd(standardized, full_matrices=False)
        projected = left.T @ centered_target
        squared = singular_values**2

        best_alpha = self.alphas[-1]
        best_gcv = float("inf")
        best_effective_df = 1.0
        for alpha in self.alphas:
            shrinkage = squared / (squared + alpha)
            fitted = left @ (shrinkage * projected)
            residual = centered_target - fitted
            effective_df = 1.0 + float(shrinkage.sum())
            residual_df = len(raw) - effective_df
            gcv = (
                float(len(raw) * (residual @ residual) / residual_df**2)
                if residual_df > np.sqrt(np.finfo(float).eps)
                else float("inf")
            )
            if gcv < best_gcv:
                best_alpha = alpha
                best_gcv = gcv
                best_effective_df = effective_df

        coefficients = right.T @ (singular_values / (squared + best_alpha) * projected)
        return NativeRidgeCVResult(
            coefficients=np.asarray(coefficients, dtype=float),
            intercept=intercept,
            feature_mean=np.asarray(feature_mean, dtype=float),
            feature_scale=np.asarray(feature_scale, dtype=float),
            feature_names=X.columns.copy(),
            selected_alpha=float(best_alpha),
            effective_df=float(best_effective_df),
        )


@dataclass(frozen=True)
class PartiallyLinearDMLResult:
    """Cross-fitted DML2 result for a partially linear treatment effect."""

    estimate: float
    covariance: pd.DataFrame
    standard_error: float
    statistic: float
    pvalue: float
    influence_function: pd.Series
    orthogonal_score: pd.Series
    residualized_outcome: pd.Series
    residualized_treatment: pd.Series
    nuisance_predictions: pd.DataFrame
    fold: pd.Series
    nobs: int
    covariance_type: DMLCovariance
    inference_distribution: str
    inference_df: float | None
    n_clusters: int | None
    n_splits: int
    random_state: int | None
    outcome_model_name: str
    treatment_model_name: str
    native_nuisance: bool
    treatment_kind: TreatmentKind
    residual_treatment_second_moment: float
    residual_treatment_tolerance: float
    orthogonal_score_mean: float
    estimation_index: pd.Index
    assumptions: tuple[str, ...]
    notes: tuple[str, ...]
    estimator: str = "partially_linear_dml2"
    estimand: str = "theta"
    converged: bool = True
    backend: str = "native-cross-fitted-orthogonal-score"

    @property
    def params(self) -> pd.Series:
        return pd.Series({self.estimand: self.estimate}, name="coef")

    @property
    def standard_errors(self) -> pd.Series:
        return pd.Series({self.estimand: self.standard_error}, name="std_err")

    @property
    def test_statistics(self) -> pd.Series:
        return pd.Series({self.estimand: self.statistic}, name="stat")

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series({self.estimand: self.pvalue}, name="p_value")

    @property
    def causal_interpretation(self) -> str:
        return (
            "The coefficient is causal only under the declared constant-effect partially "
            "linear model, consistency, no interference, conditional exchangeability, "
            "residual treatment variation, and valid nuisance-rate/inference conditions."
        )

    def conf_int(self, level: float = 0.95) -> pd.Series:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        probability = 0.5 + level / 2.0
        critical = (
            float(norm.ppf(probability))
            if self.inference_distribution == "normal"
            else float(t.ppf(probability, self.inference_df))
        )
        return pd.Series(
            {
                "lower": self.estimate - critical * self.standard_error,
                "upper": self.estimate + critical * self.standard_error,
            },
            name=self.estimand,
        )

    def summary_frame(self, level: float = 0.95) -> pd.DataFrame:
        interval = self.conf_int(level)
        return pd.DataFrame(
            {
                "coef": [self.estimate],
                "std_err": [self.standard_error],
                "stat": [self.statistic],
                "p_value": [self.pvalue],
                "ci_lower": [interval["lower"]],
                "ci_upper": [interval["upper"]],
            },
            index=pd.Index([self.estimand], dtype="object"),
        )


def _ridge_alphas(values: Sequence[float]) -> tuple[float, ...]:
    try:
        alphas = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError("ridge_alphas must be a finite sequence of positive numbers.") from error
    if not alphas or not np.isfinite(alphas).all() or any(value <= 0.0 for value in alphas):
        raise ValueError("ridge_alphas must be a non-empty finite sequence of positive numbers.")
    if any(left >= right for left, right in zip(alphas, alphas[1:], strict=False)):
        raise ValueError("ridge_alphas must be strictly increasing.")
    return alphas


class PartiallyLinearDML:
    """DML2 for a scalar treatment in a partially linear structural model.

    CauseKit's dependency-free ridge-GCV nuisance learner is used when a task factory is
    omitted. Custom factories remain provider-neutral and receive fresh state in every
    outer fold through :class:`CrossFitter`.
    """

    def __init__(
        self,
        *,
        outcome_factory: NuisanceFactory | None = None,
        treatment_factory: NuisanceFactory | None = None,
        n_splits: int = 5,
        random_state: int | None = None,
        covariance: DMLCovariance = "robust",
        outcome_predict: PredictionAdapter | None = None,
        treatment_predict: PredictionAdapter | None = None,
        ridge_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    ) -> None:
        if covariance not in {"robust", "clustered"}:
            raise ValueError("covariance must be 'robust' or 'clustered'.")
        if outcome_factory is not None and not callable(outcome_factory):
            raise TypeError("outcome_factory must be callable or None.")
        if treatment_factory is not None and not callable(treatment_factory):
            raise TypeError("treatment_factory must be callable or None.")
        self.ridge_alphas = _ridge_alphas(ridge_alphas)
        self.outcome_factory = outcome_factory
        self.treatment_factory = treatment_factory
        self.n_splits = n_splits
        self.random_state = random_state
        self.covariance = covariance
        self.outcome_predict = outcome_predict
        self.treatment_predict = treatment_predict
        # Centralize fold-count and seed validation in the established orchestrator.
        CrossFitter(n_splits=n_splits, random_state=random_state)

    def _factory(self, supplied: NuisanceFactory | None) -> NuisanceFactory:
        if supplied is not None:
            return supplied
        alphas = self.ridge_alphas
        return lambda: _NativeRidgeCV(alphas)

    def fit(
        self,
        y: Any,
        *,
        treatment: Any,
        covariates: Any,
        clusters: Any | None = None,
    ) -> PartiallyLinearDMLResult:
        frame, index = _frame(covariates)
        observed = _series(y, name="y", index=index)
        assigned = _series(treatment, name="treatment", index=index)
        if len(frame) < self.n_splits:
            raise ValueError("The estimation sample must contain at least n_splits observations.")
        if float(assigned.max()) == float(assigned.min()):
            raise ValueError("treatment must contain variation.")

        if self.covariance == "clustered" and clusters is None:
            raise ValueError("clusters must be provided when covariance='clustered'.")
        if self.covariance != "clustered" and clusters is not None:
            raise ValueError("clusters may be provided only when covariance='clustered'.")
        cluster_values = None
        if clusters is not None:
            cluster_values = _labels(clusters, name="clusters", index=index).to_numpy()

        unique_treatment = np.unique(assigned.to_numpy(dtype=float))
        binary = np.array_equal(unique_treatment, np.array([0.0, 1.0]))
        treatment_kind: TreatmentKind = "binary" if binary else "continuous"
        cross_fitter = CrossFitter(
            n_splits=self.n_splits,
            random_state=self.random_state,
        )
        nuisance = cross_fitter.fit_predict_tasks(
            frame,
            tasks=(
                CrossFitTask(
                    name="outcome_mean",
                    target=observed,
                    factory=self._factory(self.outcome_factory),
                    predict=self.outcome_predict,
                ),
                CrossFitTask(
                    name="treatment_mean",
                    target=assigned,
                    factory=self._factory(self.treatment_factory),
                    predict=self.treatment_predict,
                ),
            ),
            strata=assigned if binary else None,
        )
        predictions = nuisance.predictions
        outcome_residual = observed.to_numpy(dtype=float) - predictions["outcome_mean"].to_numpy(
            dtype=float
        )
        treatment_residual = assigned.to_numpy(dtype=float) - predictions[
            "treatment_mean"
        ].to_numpy(dtype=float)
        jacobian = float(np.mean(treatment_residual**2))
        treatment_scale = max(
            float(np.mean(assigned.to_numpy(dtype=float) ** 2)),
            float(np.var(assigned.to_numpy(dtype=float))),
            np.finfo(float).tiny,
        )
        tolerance = float(
            100.0 * np.finfo(float).eps * max(len(frame), frame.shape[1], 1) * treatment_scale
        )
        if not np.isfinite(jacobian) or jacobian <= tolerance:
            raise ValueError(
                "Cross-fitted residual treatment variation is numerically unidentified."
            )

        estimate = float(treatment_residual @ outcome_residual / (len(frame) * jacobian))
        second_stage_residual = outcome_residual - estimate * treatment_residual
        score = treatment_residual * second_stage_residual
        influence = score / jacobian
        covariance = ols_covariance(
            treatment_residual[:, None],
            second_stage_residual,
            covariance=cast(CovarianceType, self.covariance),
            clusters=cluster_values,
        )
        covariance_frame = pd.DataFrame(covariance.matrix, index=["theta"], columns=["theta"])
        standard_error = float(np.sqrt(max(covariance.matrix[0, 0], 0.0)))
        statistic = estimate / standard_error if standard_error > 0.0 else float("nan")
        pvalue = (
            float(2.0 * norm.sf(abs(statistic)))
            if covariance.distribution == "normal"
            else float(2.0 * t.sf(abs(statistic), covariance.df))
        )
        native_nuisance = self.outcome_factory is None and self.treatment_factory is None
        notes = []
        if binary:
            treatment_predictions = predictions["treatment_mean"].to_numpy(dtype=float)
            if np.any((treatment_predictions < 0.0) | (treatment_predictions > 1.0)):
                notes.append(
                    "The binary-treatment conditional-mean learner predicted outside [0, 1]; "
                    "review nuisance specification sensitivity."
                )
        if native_nuisance:
            notes.append(
                "Both nuisance regressions used CauseKit's fold-local native ridge-GCV learner."
            )
        else:
            notes.append(
                "At least one external nuisance factory was supplied; its suitability and "
                "fold-internal tuning remain the analyst's responsibility."
            )

        return PartiallyLinearDMLResult(
            estimate=estimate,
            covariance=covariance_frame,
            standard_error=standard_error,
            statistic=statistic,
            pvalue=pvalue,
            influence_function=pd.Series(influence, index=index, name="influence"),
            orthogonal_score=pd.Series(score, index=index, name="orthogonal_score"),
            residualized_outcome=pd.Series(
                outcome_residual, index=index, name="residualized_outcome"
            ),
            residualized_treatment=pd.Series(
                treatment_residual, index=index, name="residualized_treatment"
            ),
            nuisance_predictions=predictions.copy(),
            fold=nuisance.fold.copy(),
            nobs=len(frame),
            covariance_type=self.covariance,
            inference_distribution=covariance.distribution,
            inference_df=covariance.df,
            n_clusters=covariance.n_clusters,
            n_splits=nuisance.n_splits,
            random_state=nuisance.random_state,
            outcome_model_name=nuisance.model_names["outcome_mean"],
            treatment_model_name=nuisance.model_names["treatment_mean"],
            native_nuisance=native_nuisance,
            treatment_kind=treatment_kind,
            residual_treatment_second_moment=jacobian,
            residual_treatment_tolerance=tolerance,
            orthogonal_score_mean=float(score.mean()),
            estimation_index=index.copy(),
            assumptions=(
                "Constant-effect partially linear structural model",
                "Treatment consistency",
                "No interference",
                "Conditional exchangeability given measured pre-treatment covariates",
                "Sufficient residual treatment variation",
                "Nuisance-rate and moment regularity conditions for DML inference",
                "Every reported nuisance prediction is out of fold",
            ),
            notes=tuple(notes),
        )


__all__ = [
    "DMLCovariance",
    "PartiallyLinearDML",
    "PartiallyLinearDMLResult",
    "TreatmentKind",
]

"""Dependency-free specialized causal machine-learning estimators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from inspect import signature
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, t

from ._covariance import CovarianceType, ols_covariance
from .crossfit import (
    CATEResultProtocol,
    CrossFitTask,
    CrossFitter,
    NuisanceFactory,
    PredictionAdapter,
    WeightedCATEEstimatorProtocol,
    WeightedCATEFactory,
    _fit,
    _fold_assignments,
    _frame,
    _labels,
    _model_diagnostic_row,
    _prediction,
    _series,
)

DMLCovariance = Literal["robust", "clustered"]
RLearnerCovariance = Literal["robust", "clustered"]
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
    gcv_score: float
    training_rmse: float
    numerical_rank: int
    alpha_grid_size: int
    alpha_at_boundary: bool

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

    def nuisance_diagnostics(self) -> dict[str, float | int | bool]:
        """Return scalar tuning diagnostics for CrossFitter's fold audit table."""

        return {
            "selected_alpha": self.selected_alpha,
            "effective_df": self.effective_df,
            "gcv_score": self.gcv_score,
            "training_rmse": self.training_rmse,
            "numerical_rank": self.numerical_rank,
            "alpha_grid_size": self.alpha_grid_size,
            "alpha_at_boundary": self.alpha_at_boundary,
        }


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
        projected_squared = projected**2
        orthogonal_residual = centered_target - left @ projected
        orthogonal_sum_squares = float(orthogonal_residual @ orthogonal_residual)
        if len(singular_values) and singular_values[0] > 0.0:
            rank_tolerance = (
                np.finfo(float).eps * max(standardized.shape) * float(singular_values[0])
            )
            numerical_rank = int(np.sum(singular_values > rank_tolerance))
        else:
            numerical_rank = 0

        best_alpha = self.alphas[-1]
        best_gcv = float("inf")
        best_residual_sum_squares = float("inf")
        best_effective_df = 1.0
        best_position = len(self.alphas) - 1
        for position, alpha in enumerate(self.alphas):
            shrinkage = squared / (squared + alpha)
            effective_df = 1.0 + float(shrinkage.sum())
            residual_df = len(raw) - effective_df
            residual_sum_squares = orthogonal_sum_squares + float(
                ((1.0 - shrinkage) ** 2 * projected_squared).sum()
            )
            gcv = (
                float(len(raw) * residual_sum_squares / residual_df**2)
                if residual_df > np.sqrt(np.finfo(float).eps)
                else float("inf")
            )
            if gcv < best_gcv:
                best_alpha = alpha
                best_gcv = gcv
                best_residual_sum_squares = residual_sum_squares
                best_effective_df = effective_df
                best_position = position

        if not np.isfinite(best_gcv):
            raise ValueError("Native ridge could not identify a finite GCV candidate.")

        coefficients = right.T @ (singular_values / (squared + best_alpha) * projected)
        return NativeRidgeCVResult(
            coefficients=np.asarray(coefficients, dtype=float),
            intercept=intercept,
            feature_mean=np.asarray(feature_mean, dtype=float),
            feature_scale=np.asarray(feature_scale, dtype=float),
            feature_names=X.columns.copy(),
            selected_alpha=float(best_alpha),
            effective_df=float(best_effective_df),
            gcv_score=float(best_gcv),
            training_rmse=float(np.sqrt(best_residual_sum_squares / len(raw))),
            numerical_rank=numerical_rank,
            alpha_grid_size=len(self.alphas),
            alpha_at_boundary=best_position in {0, len(self.alphas) - 1},
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
    nuisance_diagnostics: pd.DataFrame
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


@dataclass(frozen=True)
class RLearnerResult:
    """Split-conditional honest evaluation of a construction-fitted R-learner."""

    _estimation_index: pd.Index
    _construction_index: pd.Index
    _evaluation_index: pd.Index
    _sample_role_values: tuple[str, ...]
    _construction_fold_values: tuple[int, ...]
    _cluster_labels: pd.Index | None
    _cluster_role_values: tuple[str, ...] | None
    _evaluation_group_values: tuple[int, ...]
    _cate_result: Any
    _cate_predict: PredictionAdapter | None
    _feature_names: pd.Index
    construction_nuisance_predictions: pd.DataFrame
    evaluation_nuisance_predictions: pd.DataFrame
    construction_residuals: pd.DataFrame
    evaluation_residuals: pd.DataFrame
    construction_cate_predictions: pd.Series
    evaluation_cate_predictions: pd.Series
    nuisance_diagnostics: pd.DataFrame
    cate_diagnostics: pd.DataFrame
    calibration_coefficients: pd.Series
    calibration_covariance: pd.DataFrame
    calibration_tests: pd.DataFrame
    group_effects: pd.DataFrame
    group_covariance: pd.DataFrame
    group_influence: pd.DataFrame
    honest_r_loss: float
    honest_constant_r_loss: float
    r_loss_gain: float
    construction_constant_effect: float
    calibration_center: float
    construction_r_objective: float
    weighted_construction_objective: float
    residual_treatment_second_moment: float
    residual_treatment_tolerance: float
    requested_evaluation_fraction: float
    realized_evaluation_fraction: float
    split_seed: int | None
    n_splits: int
    overlap_floor: float
    cluster_split: bool
    nobs: int
    construction_nobs: int
    evaluation_nobs: int
    calibration_groups: int
    covariance_type: RLearnerCovariance
    inference_distribution: str
    inference_df: float | None
    n_clusters: int | None
    simultaneous_level: float
    simultaneous_critical_value: float
    bootstrap_iterations: int
    bootstrap_random_state: int | None
    outcome_model_name: str
    propensity_model_name: str
    cate_model_name: str
    native_outcome: bool
    native_propensity: bool
    native_cate: bool
    assumptions: tuple[str, ...]
    notes: tuple[str, ...]
    estimator: str = "honest_r_learner"
    estimand: str = "cate_calibration"
    converged: bool = True
    split_conditional: bool = True
    backend: str = "native-honest-r-learner"

    @property
    def estimation_index(self) -> pd.Index:
        return self._estimation_index.copy()

    @property
    def construction_index(self) -> pd.Index:
        return self._construction_index.copy()

    @property
    def evaluation_index(self) -> pd.Index:
        return self._evaluation_index.copy()

    @property
    def sample_role(self) -> pd.Series:
        return pd.Series(
            self._sample_role_values,
            index=self._estimation_index.copy(),
            name="sample_role",
            dtype="object",
        )

    @property
    def construction_fold(self) -> pd.Series:
        return pd.Series(
            self._construction_fold_values,
            index=self._construction_index.copy(),
            name="construction_fold",
            dtype=int,
        )

    @property
    def cluster_role(self) -> pd.Series:
        if self._cluster_labels is None or self._cluster_role_values is None:
            return pd.Series(name="cluster_role", dtype="object")
        return pd.Series(
            self._cluster_role_values,
            index=self._cluster_labels.copy(),
            name="cluster_role",
            dtype="object",
        )

    @property
    def evaluation_group(self) -> pd.Series:
        return pd.Series(
            self._evaluation_group_values,
            index=self._evaluation_index.copy(),
            name="calibration_group",
            dtype=int,
        )

    @property
    def params(self) -> pd.Series:
        return self.calibration_coefficients.copy().rename("coef")

    @property
    def standard_errors(self) -> pd.Series:
        values = np.sqrt(np.maximum(np.diag(self.calibration_covariance), 0.0))
        return pd.Series(values, index=self.calibration_coefficients.index.copy(), name="std_err")

    @property
    def test_statistics(self) -> pd.Series:
        return (self.params / self.standard_errors).rename("stat")

    @property
    def pvalues(self) -> pd.Series:
        statistics = np.abs(self.test_statistics.to_numpy(dtype=float))
        values = (
            2.0 * norm.sf(statistics)
            if self.inference_distribution == "normal"
            else 2.0 * t.sf(statistics, self.inference_df)
        )
        return pd.Series(values, index=self.calibration_coefficients.index.copy(), name="p_value")

    @property
    def causal_interpretation(self) -> str:
        return (
            "The CATE ranking and calibration are causal only under consistency, no "
            "interference, conditional exchangeability given the declared pre-treatment "
            "covariates, overlap, valid nuisance rates, and the recorded honest split."
        )

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        """Return pointwise calibration-coefficient confidence intervals."""

        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        probability = 0.5 + level / 2.0
        critical = (
            float(norm.ppf(probability))
            if self.inference_distribution == "normal"
            else float(t.ppf(probability, self.inference_df))
        )
        return pd.DataFrame(
            {
                "lower": self.params - critical * self.standard_errors,
                "upper": self.params + critical * self.standard_errors,
            },
            index=self.calibration_coefficients.index.copy(),
        )

    def summary_frame(self, level: float = 0.95) -> pd.DataFrame:
        """Return honest differential-calibration coefficients and pointwise inference."""

        interval = self.conf_int(level)
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.standard_errors,
                "stat": self.test_statistics,
                "p_value": self.pvalues,
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
            },
            index=self.calibration_coefficients.index.copy(),
        )

    def predict(self, X: Any) -> pd.Series:
        """Predict CATEs using the model fitted only on the construction sample."""

        frame, index = _frame(X)
        if not frame.columns.equals(self._feature_names):
            raise ValueError("Prediction covariate columns must match the fitted schema exactly.")
        values = _cate_prediction(self._cate_result, frame, adapter=self._cate_predict)
        return pd.Series(values, index=index, name="cate")

    def calibration_plot_data(self) -> pd.DataFrame:
        """Return the exact honest group table used by the calibration graph."""

        return self.group_effects.copy()

    def cate_distribution_data(self) -> pd.DataFrame:
        """Return honest evaluation predictions without unit-level uncertainty claims."""

        return pd.DataFrame(
            {
                "cate": self.evaluation_cate_predictions.copy(),
                "calibration_group": self.evaluation_group,
                "sample_role": "evaluation",
            },
            index=self._evaluation_index.copy(),
        )

    def graph_metadata(self) -> dict[str, Any]:
        """Return split and uncertainty metadata shared by both graph surfaces."""

        return {
            "construction_nobs": self.construction_nobs,
            "evaluation_nobs": self.evaluation_nobs,
            "split_seed": self.split_seed,
            "n_splits": self.n_splits,
            "covariance_type": self.covariance_type,
            "calibration_groups": self.calibration_groups,
            "simultaneous_level": self.simultaneous_level,
            "simultaneous_critical_value": self.simultaneous_critical_value,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_random_state": self.bootstrap_random_state,
            "split_conditional": self.split_conditional,
        }

    def plot_calibration(self, *, ax: Any | None = None) -> Any:
        """Plot honest predicted group CATEs against overlap-weighted group effects."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Install CauseKit's 'plot' extra to draw R-learner calibration graphs."
            ) from error
        if ax is None:
            _, ax = plt.subplots()
        table = self.calibration_plot_data()
        x = table["predicted_cate_mean"].to_numpy(dtype=float)
        y = table["effect"].to_numpy(dtype=float)
        lower = table["simultaneous_lower"].to_numpy(dtype=float)
        upper = table["simultaneous_upper"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=np.vstack([y - lower, upper - y]), fmt="o", capsize=3)
        limits = np.array([x.min(), x.max(), y.min(), y.max(), lower.min(), upper.max()])
        lower_limit = float(limits.min())
        upper_limit = float(limits.max())
        ax.plot([lower_limit, upper_limit], [lower_limit, upper_limit], linestyle="--")
        ax.set_xlabel("Honest mean predicted CATE")
        ax.set_ylabel("Overlap-weighted group effect")
        ax.set_title("Honest R-learner calibration")
        return ax

    def plot_cate_distribution(self, *, ax: Any | None = None, bins: int = 20) -> Any:
        """Plot the honest evaluation CATE distribution."""

        if isinstance(bins, bool) or not isinstance(bins, Integral) or int(bins) < 1:
            raise ValueError("bins must be a positive integer.")
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Install CauseKit's 'plot' extra to draw R-learner CATE distributions."
            ) from error
        if ax is None:
            _, ax = plt.subplots()
        ax.hist(self.evaluation_cate_predictions.to_numpy(dtype=float), bins=int(bins))
        ax.set_xlabel("Honest predicted CATE")
        ax.set_ylabel("Evaluation observations")
        ax.set_title("Honest evaluation CATE distribution")
        return ax


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


@dataclass(frozen=True)
class NativePenalizedLogitResult:
    """Fitted state for the native ridge-penalized binary probability learner."""

    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: pd.Index
    classes_: np.ndarray
    selected_alpha: float
    cv_log_loss: float
    training_log_loss: float
    training_objective: float
    converged: bool
    iterations: int
    numerical_rank: int
    alpha_grid_size: int
    alpha_at_boundary: bool
    tuning_splits: int
    training_index: pd.Index
    tuning_fold: pd.Series

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return control/treated probabilities without clipping."""

        raw = _prediction_frame(
            X,
            feature_names=self.feature_names,
            feature_count=len(self.feature_names),
        )
        standardized = (raw - self.feature_mean) / self.feature_scale
        treated = expit(self.intercept + standardized @ self.coefficients)
        return np.column_stack([1.0 - treated, treated])

    def nuisance_diagnostics(self) -> dict[str, float | int | bool]:
        """Return scalar tuning diagnostics suitable for CrossFitter."""

        return {
            "selected_alpha": self.selected_alpha,
            "cv_log_loss": self.cv_log_loss,
            "training_log_loss": self.training_log_loss,
            "training_objective": self.training_objective,
            "converged": self.converged,
            "iterations": self.iterations,
            "numerical_rank": self.numerical_rank,
            "alpha_grid_size": self.alpha_grid_size,
            "alpha_at_boundary": self.alpha_at_boundary,
            "tuning_splits": self.tuning_splits,
        }


@dataclass(frozen=True)
class NativeWeightedRidgeCVResult:
    """Fitted state for the native weighted ridge-GCV CATE learner."""

    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: pd.Index
    selected_alpha: float
    effective_df: float
    gcv_score: float
    weighted_training_rmse: float
    weighted_residual_sum_squares: float
    numerical_rank: int
    alpha_grid_size: int
    alpha_at_boundary: bool
    total_weight: float
    minimum_weight: float
    maximum_weight: float
    training_index: pd.Index

    def predict(self, X: Any) -> np.ndarray:
        """Return one finite CATE prediction per row."""

        raw = _prediction_frame(
            X,
            feature_names=self.feature_names,
            feature_count=len(self.feature_names),
        )
        standardized = (raw - self.feature_mean) / self.feature_scale
        return self.intercept + standardized @ self.coefficients

    def nuisance_diagnostics(self) -> dict[str, float | int | bool]:
        """Return scalar tuning diagnostics for the R-learner audit table."""

        return {
            "selected_alpha": self.selected_alpha,
            "effective_df": self.effective_df,
            "gcv_score": self.gcv_score,
            "weighted_training_rmse": self.weighted_training_rmse,
            "weighted_residual_sum_squares": self.weighted_residual_sum_squares,
            "numerical_rank": self.numerical_rank,
            "alpha_grid_size": self.alpha_grid_size,
            "alpha_at_boundary": self.alpha_at_boundary,
            "total_weight": self.total_weight,
            "minimum_weight": self.minimum_weight,
            "maximum_weight": self.maximum_weight,
        }


@dataclass(frozen=True)
class NativeSplineRidgeCATEResult:
    """Fitted construction-only adaptive linear-spline CATE state."""

    _ridge_result: NativeWeightedRidgeCVResult
    _input_feature_names: pd.Index
    _input_feature_mean: np.ndarray
    _input_feature_scale: np.ndarray
    _knots: tuple[tuple[float, ...], ...]
    _tuning_rows: tuple[tuple[int, int, float, float, bool], ...]
    selected_knot_count: int
    basis_dimension: int
    include_pairwise_interactions: bool
    max_basis_features: int

    @property
    def training_index(self) -> pd.Index:
        return self._ridge_result.training_index.copy()

    @property
    def input_feature_names(self) -> pd.Index:
        return self._input_feature_names.copy()

    @property
    def selected_knots(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(values) for values in self._knots)

    @property
    def selected_alpha(self) -> float:
        return self._ridge_result.selected_alpha

    @property
    def effective_df(self) -> float:
        return self._ridge_result.effective_df

    @property
    def gcv_score(self) -> float:
        return self._ridge_result.gcv_score

    @property
    def tuning_path(self) -> pd.DataFrame:
        """Return every construction-only knot/penalty candidate summary."""

        return pd.DataFrame(
            self._tuning_rows,
            columns=[
                "knot_count",
                "basis_dimension",
                "selected_alpha",
                "gcv_score",
                "selected",
            ],
        )

    def basis_data(self, X: Any) -> pd.DataFrame:
        """Return the exact fitted spline basis for audit or future prediction."""

        raw = _prediction_frame(
            X,
            feature_names=self._input_feature_names,
            feature_count=len(self._input_feature_names),
        )
        index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(raw))
        standardized = (raw - self._input_feature_mean) / self._input_feature_scale
        return _spline_basis_frame(
            standardized,
            feature_names=self._input_feature_names,
            knots=self._knots,
            include_pairwise_interactions=self.include_pairwise_interactions,
            index=index,
        )

    def predict(self, X: Any) -> np.ndarray:
        """Predict CATEs from the construction-selected nonlinear basis."""

        return self._ridge_result.predict(self.basis_data(X))

    def nuisance_diagnostics(self) -> dict[str, float | int | bool | str]:
        """Return selected nonlinear complexity and weighted ridge diagnostics."""

        diagnostics: dict[str, float | int | bool | str] = dict(
            self._ridge_result.nuisance_diagnostics()
        )
        diagnostics.update(
            {
                "basis_family": "additive_linear_spline",
                "selected_knot_count": self.selected_knot_count,
                "basis_dimension": self.basis_dimension,
                "basis_candidates": len(self._tuning_rows),
                "pairwise_interactions": self.include_pairwise_interactions,
                "max_basis_features": self.max_basis_features,
            }
        )
        return diagnostics


@dataclass(frozen=True)
class _LogitFit:
    intercept: float
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    log_loss: float
    objective: float
    iterations: int


def _prediction_frame(
    X: Any,
    *,
    feature_names: pd.Index,
    feature_count: int,
) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        if not X.columns.equals(feature_names):
            raise ValueError("Prediction covariate columns must match the fitted schema exactly.")
        raw = X.to_numpy(dtype=float)
    else:
        raw = np.asarray(X, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != feature_count:
        raise ValueError("Prediction covariates must match the fitted feature dimension.")
    if not np.isfinite(raw).all():
        raise ValueError("Prediction covariates must contain only finite values.")
    return raw


def _candidate_spline_knots(
    standardized: np.ndarray,
    *,
    knot_count: int,
) -> tuple[tuple[float, ...], ...]:
    if knot_count == 0:
        return tuple(() for _ in range(standardized.shape[1]))
    probabilities = np.linspace(0.0, 1.0, knot_count + 2)[1:-1]
    collected: list[tuple[float, ...]] = []
    for position in range(standardized.shape[1]):
        values = standardized[:, position]
        lower = float(values.min())
        upper = float(values.max())
        scale = max(abs(lower), abs(upper), 1.0)
        tolerance = 100.0 * np.finfo(float).eps * scale
        candidates = np.atleast_1d(np.quantile(values, probabilities))
        interior = candidates[(candidates > lower + tolerance) & (candidates < upper - tolerance)]
        collected.append(tuple(float(value) for value in np.unique(interior)))
    return tuple(collected)


def _spline_basis_frame(
    standardized: np.ndarray,
    *,
    feature_names: pd.Index,
    knots: tuple[tuple[float, ...], ...],
    include_pairwise_interactions: bool,
    index: pd.Index,
) -> pd.DataFrame:
    if len(knots) != standardized.shape[1]:  # pragma: no cover - fitted state invariant
        raise RuntimeError("Stored spline knots do not match the fitted input dimension.")
    columns: dict[str, np.ndarray] = {}
    for position, name in enumerate(feature_names):
        label = str(name)
        values = standardized[:, position]
        columns[f"linear::{position}::{label}"] = values
        for knot_position, knot in enumerate(knots[position], start=1):
            columns[f"hinge::{position}::{label}::{knot_position}"] = np.maximum(
                values - knot,
                0.0,
            )
    if include_pairwise_interactions:
        for left in range(standardized.shape[1]):
            for right in range(left + 1, standardized.shape[1]):
                columns[f"interaction::{left}::{right}"] = (
                    standardized[:, left] * standardized[:, right]
                )
    return pd.DataFrame(columns, index=index)


def _cate_prediction(
    result: Any,
    X: Any,
    *,
    adapter: PredictionAdapter | None = None,
) -> np.ndarray:
    """Validate one provider-neutral CATE prediction vector."""

    if adapter is None and not isinstance(result, CATEResultProtocol):
        raise TypeError("The fitted CATE result must provide predict(X).")
    raw = adapter(result, X) if adapter is not None else result.predict(X)
    if isinstance(raw, pd.Series) and isinstance(X, pd.DataFrame) and not raw.index.equals(X.index):
        raise ValueError("CATE prediction index must match the prediction covariate index.")
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or len(values) != len(X):
        raise ValueError("A CATE prediction must provide one value per row.")
    if not np.isfinite(values).all():
        raise ValueError("CATE predictions must contain only finite values.")
    return values


def _fit_penalized_logit(raw: np.ndarray, target: np.ndarray, *, alpha: float) -> _LogitFit:
    feature_mean = raw.mean(axis=0)
    feature_scale = raw.std(axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
    standardized = (raw - feature_mean) / feature_scale
    treated_share = float(target.mean())
    initial = np.zeros(standardized.shape[1] + 1, dtype=float)
    initial[0] = float(np.log(treated_share / (1.0 - treated_share)))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + standardized @ parameters[1:]
        probability = expit(linear)
        value = float(
            np.logaddexp(0.0, linear).sum()
            - target @ linear
            + 0.5 * alpha * (parameters[1:] @ parameters[1:])
        )
        residual = probability - target
        gradient = np.concatenate(
            (
                np.array([residual.sum()]),
                standardized.T @ residual + alpha * parameters[1:],
            )
        )
        return value, gradient

    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-9, "maxls": 50},
    )
    if not optimized.success or not np.isfinite(optimized.fun):
        raise ValueError(f"Native penalized Logit did not converge: {optimized.message}.")
    parameters = np.asarray(optimized.x, dtype=float)
    linear = parameters[0] + standardized @ parameters[1:]
    log_loss = float(np.mean(np.logaddexp(0.0, linear) - target * linear))
    return _LogitFit(
        intercept=float(parameters[0]),
        coefficients=parameters[1:].copy(),
        feature_mean=np.asarray(feature_mean, dtype=float),
        feature_scale=np.asarray(feature_scale, dtype=float),
        log_loss=log_loss,
        objective=float(optimized.fun),
        iterations=int(optimized.nit),
    )


class _NativePenalizedLogitCV:
    """Standardized ridge-Logit with stratified training-only CV log-loss tuning."""

    def __init__(
        self,
        *,
        alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
        tuning_splits: int = 3,
        random_state: int | None = None,
    ) -> None:
        self.alphas = _ridge_alphas(alphas)
        if isinstance(tuning_splits, bool) or not isinstance(tuning_splits, int):
            raise TypeError("tuning_splits must be an integer.")
        if tuning_splits < 2:
            raise ValueError("tuning_splits must be at least two.")
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError("random_state must be an integer or None.")
        self.tuning_splits = tuning_splits
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> NativePenalizedLogitResult:
        frame, index = _frame(X)
        target = _series(y, name="binary_target", index=index)
        observed = np.unique(target.to_numpy(dtype=float))
        if len(observed) < 2:
            raise ValueError("binary_target must contain both arms coded 0 and 1.")
        if not np.array_equal(observed, np.array([0.0, 1.0])):
            raise ValueError("binary_target must be coded exactly 0 and 1.")
        counts = target.value_counts()
        if int(counts.min()) < self.tuning_splits:
            raise ValueError("Each binary arm must contain at least tuning_splits observations.")

        raw = frame.to_numpy(dtype=float)
        target_values = target.to_numpy(dtype=float)
        folds = _fold_assignments(
            len(frame),
            n_splits=self.tuning_splits,
            random_state=self.random_state,
            strata=target,
        )
        best_alpha = self.alphas[-1]
        best_loss = float("inf")
        best_position = len(self.alphas) - 1
        for position, alpha in enumerate(self.alphas):
            heldout_loss = 0.0
            heldout_nobs = 0
            for fold in range(self.tuning_splits):
                holdout = folds == fold
                training = ~holdout
                fitted = _fit_penalized_logit(
                    raw[training],
                    target_values[training],
                    alpha=alpha,
                )
                standardized = (raw[holdout] - fitted.feature_mean) / fitted.feature_scale
                linear = fitted.intercept + standardized @ fitted.coefficients
                heldout_loss += float(
                    np.sum(np.logaddexp(0.0, linear) - target_values[holdout] * linear)
                )
                heldout_nobs += int(holdout.sum())
            cv_log_loss = heldout_loss / heldout_nobs
            if cv_log_loss < best_loss:
                best_alpha = alpha
                best_loss = cv_log_loss
                best_position = position

        final = _fit_penalized_logit(raw, target_values, alpha=best_alpha)
        standardized = (raw - final.feature_mean) / final.feature_scale
        return NativePenalizedLogitResult(
            coefficients=final.coefficients,
            intercept=final.intercept,
            feature_mean=final.feature_mean,
            feature_scale=final.feature_scale,
            feature_names=frame.columns.copy(),
            classes_=np.array([0.0, 1.0]),
            selected_alpha=float(best_alpha),
            cv_log_loss=float(best_loss),
            training_log_loss=final.log_loss,
            training_objective=final.objective,
            converged=True,
            iterations=final.iterations,
            numerical_rank=int(np.linalg.matrix_rank(standardized)),
            alpha_grid_size=len(self.alphas),
            alpha_at_boundary=best_position in {0, len(self.alphas) - 1},
            tuning_splits=self.tuning_splits,
            training_index=index.copy(),
            tuning_fold=pd.Series(folds, index=index.copy(), name="tuning_fold"),
        )


def _weighted_cate_inputs(
    X: Any,
    pseudo_outcome: Any,
    sample_weight: Any,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Index]:
    frame, index = _frame(X)
    target = _series(pseudo_outcome, name="pseudo_outcome", index=index)
    weight = _series(sample_weight, name="sample_weight", index=index)
    if np.any(weight.to_numpy(dtype=float) <= 0.0):
        raise ValueError("sample_weight must contain only strictly positive values.")
    if len(frame) < 2:
        raise ValueError("Weighted CATE fitting requires at least two observations.")
    return frame, target, weight, index


class _NativeWeightedRidgeCV:
    """Standardized weighted ridge-GCV for the R-learner CATE stage."""

    def __init__(self, *, alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS) -> None:
        self.alphas = _ridge_alphas(alphas)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any,
    ) -> NativeWeightedRidgeCVResult:
        frame, target, weight, index = _weighted_cate_inputs(X, y, sample_weight)
        raw = frame.to_numpy(dtype=float)
        target_values = target.to_numpy(dtype=float)
        weights = weight.to_numpy(dtype=float)
        total_weight = float(weights.sum())
        feature_mean = np.average(raw, axis=0, weights=weights)
        centered = raw - feature_mean
        feature_scale = np.sqrt(np.average(centered**2, axis=0, weights=weights))
        feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
        standardized = centered / feature_scale
        intercept = float(np.average(target_values, weights=weights))
        centered_target = target_values - intercept
        root_weight = np.sqrt(weights)
        weighted_design = root_weight[:, None] * standardized
        weighted_target = root_weight * centered_target
        left, singular_values, right = np.linalg.svd(weighted_design, full_matrices=False)
        projected = left.T @ weighted_target
        squared = singular_values**2
        projected_squared = projected**2
        orthogonal_residual = weighted_target - left @ projected
        orthogonal_sum_squares = float(orthogonal_residual @ orthogonal_residual)
        if len(singular_values) and singular_values[0] > 0.0:
            rank_tolerance = (
                np.finfo(float).eps * max(weighted_design.shape) * float(singular_values[0])
            )
            numerical_rank = int(np.sum(singular_values > rank_tolerance))
        else:
            numerical_rank = 0

        best_alpha = self.alphas[-1]
        best_gcv = float("inf")
        best_residual_sum_squares = float("inf")
        best_effective_df = 1.0
        best_position = len(self.alphas) - 1
        for position, alpha in enumerate(self.alphas):
            shrinkage = squared / (squared + alpha)
            effective_df = 1.0 + float(shrinkage.sum())
            residual_df = len(frame) - effective_df
            residual_sum_squares = orthogonal_sum_squares + float(
                ((1.0 - shrinkage) ** 2 * projected_squared).sum()
            )
            gcv = (
                float(len(frame) * residual_sum_squares / residual_df**2)
                if residual_df > np.sqrt(np.finfo(float).eps)
                else float("inf")
            )
            if gcv < best_gcv:
                best_alpha = alpha
                best_gcv = gcv
                best_residual_sum_squares = residual_sum_squares
                best_effective_df = effective_df
                best_position = position
        if not np.isfinite(best_gcv):
            raise ValueError("Native weighted ridge could not identify a finite GCV candidate.")

        coefficients = right.T @ (singular_values / (squared + best_alpha) * projected)
        return NativeWeightedRidgeCVResult(
            coefficients=np.asarray(coefficients, dtype=float),
            intercept=intercept,
            feature_mean=np.asarray(feature_mean, dtype=float),
            feature_scale=np.asarray(feature_scale, dtype=float),
            feature_names=frame.columns.copy(),
            selected_alpha=float(best_alpha),
            effective_df=float(best_effective_df),
            gcv_score=float(best_gcv),
            weighted_training_rmse=float(np.sqrt(best_residual_sum_squares / total_weight)),
            weighted_residual_sum_squares=float(best_residual_sum_squares),
            numerical_rank=numerical_rank,
            alpha_grid_size=len(self.alphas),
            alpha_at_boundary=best_position in {0, len(self.alphas) - 1},
            total_weight=total_weight,
            minimum_weight=float(weights.min()),
            maximum_weight=float(weights.max()),
            training_index=index.copy(),
        )


class NativeSplineRidgeCATE:
    """Opt-in construction-tuned nonlinear CATE learner for :class:`RLearner`.

    Candidate additive linear-spline bases are formed from construction-only covariate
    quantiles. Weighted GCV jointly selects the requested knot count and ridge penalty.
    Pairwise standardized linear interactions are optional and every requested candidate
    must stay within ``max_basis_features``.
    """

    def __init__(
        self,
        *,
        knot_counts: Sequence[int] = (0, 1, 3),
        alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
        include_pairwise_interactions: bool = False,
        max_basis_features: int = 512,
    ) -> None:
        try:
            normalized_knots = tuple(knot_counts)
        except TypeError as error:
            raise TypeError("knot_counts must be a sequence of nonnegative integers.") from error
        if not normalized_knots or any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in normalized_knots
        ):
            raise ValueError("knot_counts must be a non-empty sequence of nonnegative integers.")
        knot_values = tuple(int(value) for value in normalized_knots)
        if any(left >= right for left, right in zip(knot_values, knot_values[1:], strict=False)):
            raise ValueError("knot_counts must be strictly increasing.")
        if not isinstance(include_pairwise_interactions, bool):
            raise TypeError("include_pairwise_interactions must be boolean.")
        if (
            isinstance(max_basis_features, bool)
            or not isinstance(max_basis_features, Integral)
            or int(max_basis_features) < 1
        ):
            raise ValueError("max_basis_features must be a positive integer.")
        self.knot_counts: tuple[int, ...] = knot_values
        self.alphas: tuple[float, ...] = _ridge_alphas(alphas)
        self.include_pairwise_interactions: bool = include_pairwise_interactions
        self.max_basis_features: int = int(max_basis_features)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any,
    ) -> NativeSplineRidgeCATEResult:
        """Select nonlinear complexity using only the supplied construction rows."""

        frame, target, weight, index = _weighted_cate_inputs(X, y, sample_weight)
        raw = frame.to_numpy(dtype=float)
        weights = weight.to_numpy(dtype=float)
        feature_mean = np.average(raw, axis=0, weights=weights)
        centered = raw - feature_mean
        feature_scale = np.sqrt(np.average(centered**2, axis=0, weights=weights))
        feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
        standardized = centered / feature_scale

        candidates: list[
            tuple[int, tuple[tuple[float, ...], ...], NativeWeightedRidgeCVResult]
        ] = []
        for knot_count in self.knot_counts:
            knots = _candidate_spline_knots(
                standardized,
                knot_count=knot_count,
            )
            basis = _spline_basis_frame(
                standardized,
                feature_names=frame.columns,
                knots=knots,
                include_pairwise_interactions=self.include_pairwise_interactions,
                index=index,
            )
            if basis.shape[1] > self.max_basis_features:
                raise ValueError(
                    "A requested spline candidate exceeds max_basis_features; reduce "
                    "knot_counts/interactions or explicitly increase the bound."
                )
            fitted = _NativeWeightedRidgeCV(alphas=self.alphas).fit(
                basis,
                target,
                sample_weight=weight,
            )
            candidates.append((knot_count, knots, fitted))

        selected_position = min(
            range(len(candidates)),
            key=lambda position: (
                candidates[position][2].gcv_score,
                len(candidates[position][2].feature_names),
                candidates[position][0],
            ),
        )
        selected_knot_count, selected_knots, selected_result = candidates[selected_position]
        tuning_rows = tuple(
            (
                knot_count,
                len(fitted.feature_names),
                fitted.selected_alpha,
                fitted.gcv_score,
                position == selected_position,
            )
            for position, (knot_count, _, fitted) in enumerate(candidates)
        )
        return NativeSplineRidgeCATEResult(
            _ridge_result=selected_result,
            _input_feature_names=frame.columns.copy(),
            _input_feature_mean=np.asarray(feature_mean, dtype=float),
            _input_feature_scale=np.asarray(feature_scale, dtype=float),
            _knots=selected_knots,
            _tuning_rows=tuning_rows,
            selected_knot_count=selected_knot_count,
            basis_dimension=len(selected_result.feature_names),
            include_pairwise_interactions=self.include_pairwise_interactions,
            max_basis_features=self.max_basis_features,
        )


def _fit_weighted_cate(
    factory: WeightedCATEFactory,
    X: Any,
    pseudo_outcome: Any,
    *,
    sample_weight: Any,
) -> Any:
    """Create and fit one weighted CATE model under the public structural protocol."""

    if not callable(factory):
        raise TypeError("The CATE factory must be callable.")
    frame, target, weight, _ = _weighted_cate_inputs(X, pseudo_outcome, sample_weight)
    estimator = factory()
    if not isinstance(estimator, WeightedCATEEstimatorProtocol):
        raise TypeError(
            "Each CATE factory must return an object with "
            "fit(X, pseudo_outcome, sample_weight=weight)."
        )
    try:
        signature(estimator.fit).bind(frame, target, sample_weight=weight)
    except (TypeError, ValueError) as error:
        raise TypeError("The CATE estimator fit method must accept sample_weight=.") from error
    result = estimator.fit(frame, target, sample_weight=weight)
    fitted = estimator if result is None else result
    if not isinstance(fitted, CATEResultProtocol):
        raise TypeError("The fitted CATE result must provide predict(X).")
    return fitted


@dataclass(frozen=True)
class _HonestRConstructionResult:
    """Internal audit result for honest roles and construction-only R fitting."""

    _estimation_index: pd.Index
    _construction_index: pd.Index
    _evaluation_index: pd.Index
    _sample_role_values: tuple[str, ...]
    _construction_fold_values: tuple[int, ...]
    _cluster_labels: pd.Index | None
    _cluster_role_values: tuple[str, ...] | None
    _cate_result: Any
    _cate_predict: PredictionAdapter | None
    _feature_names: pd.Index
    construction_nuisance_predictions: pd.DataFrame
    evaluation_nuisance_predictions: pd.DataFrame
    construction_residuals: pd.DataFrame
    construction_cate_predictions: pd.Series
    evaluation_cate_predictions: pd.Series
    nuisance_diagnostics: pd.DataFrame
    cate_diagnostics: pd.DataFrame
    construction_r_objective: float
    weighted_construction_objective: float
    residual_treatment_second_moment: float
    residual_treatment_tolerance: float
    requested_evaluation_fraction: float
    realized_evaluation_fraction: float
    split_seed: int | None
    n_splits: int
    overlap_floor: float
    cluster_split: bool
    outcome_model_name: str
    propensity_model_name: str
    cate_model_name: str
    split_conditional: bool = True

    @property
    def estimation_index(self) -> pd.Index:
        return self._estimation_index.copy()

    @property
    def construction_index(self) -> pd.Index:
        return self._construction_index.copy()

    @property
    def evaluation_index(self) -> pd.Index:
        return self._evaluation_index.copy()

    @property
    def sample_role(self) -> pd.Series:
        return pd.Series(
            self._sample_role_values,
            index=self._estimation_index.copy(),
            name="sample_role",
            dtype="object",
        )

    @property
    def construction_fold(self) -> pd.Series:
        return pd.Series(
            self._construction_fold_values,
            index=self._construction_index.copy(),
            name="construction_fold",
            dtype=int,
        )

    @property
    def cluster_role(self) -> pd.Series:
        if self._cluster_labels is None or self._cluster_role_values is None:
            return pd.Series(name="cluster_role", dtype="object")
        return pd.Series(
            self._cluster_role_values,
            index=self._cluster_labels.copy(),
            name="cluster_role",
            dtype="object",
        )


class _FreshNuisanceFactory:
    """Retain estimator identities so a factory cannot recycle fitted state."""

    def __init__(self, factory: NuisanceFactory, *, label: str) -> None:
        self.factory = factory
        self.label = label
        self._instances: list[Any] = []

    def __call__(self) -> Any:
        estimator = self.factory()
        if any(estimator is previous for previous in self._instances):
            raise ValueError(
                f"The {self.label} factory must return a fresh estimator for every fit."
            )
        self._instances.append(estimator)
        return estimator


def _evaluation_count(nobs: int, fraction: float) -> int:
    return int(np.floor(nobs * fraction + 0.5))


def _row_role_assignment(
    treatment: pd.Series,
    *,
    evaluation_fraction: float,
    random_state: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    roles = np.full(len(treatment), "construction", dtype=object)
    values = treatment.to_numpy(dtype=float)
    for arm in (0.0, 1.0):
        positions = np.flatnonzero(values == arm)
        rng.shuffle(positions)
        evaluation_nobs = _evaluation_count(len(positions), evaluation_fraction)
        if evaluation_nobs < 1 or evaluation_nobs >= len(positions):
            raise ValueError(
                "evaluation_fraction must leave both construction and evaluation "
                "observations in each treatment arm."
            )
        roles[positions[:evaluation_nobs]] = "evaluation"
    return roles


def _cluster_role_assignment(
    treatment: pd.Series,
    clusters: pd.Series,
    *,
    evaluation_fraction: float,
    random_state: int | None,
) -> np.ndarray:
    cluster_codes, cluster_labels = pd.factorize(clusters, sort=False)
    n_clusters = len(cluster_labels)
    treatment_values = treatment.to_numpy(dtype=int)
    counts = np.zeros((n_clusters, 2), dtype=float)
    np.add.at(counts, (cluster_codes, treatment_values), 1.0)
    proportions = np.array([1.0 - evaluation_fraction, evaluation_fraction])
    targets = proportions[:, None] * counts.sum(axis=0)[None, :]
    target_clusters = proportions * n_clusters
    scale = np.maximum(targets, 1.0)
    rng = np.random.default_rng(random_state)
    randomized = rng.permutation(n_clusters)
    sizes = counts.sum(axis=1)
    order = randomized[np.argsort(-sizes[randomized], kind="stable")]
    role_priority = rng.permutation(2)
    role_counts = np.zeros((2, 2), dtype=float)
    role_clusters = np.zeros(2, dtype=float)
    cluster_assignment = np.full(n_clusters, -1, dtype=int)
    for cluster_code in order:
        best_role = -1
        best_score = float("inf")
        for role in role_priority:
            proposed_counts = role_counts.copy()
            proposed_counts[role] += counts[cluster_code]
            proposed_clusters = role_clusters.copy()
            proposed_clusters[role] += 1.0
            score = float(
                np.sum((proposed_counts - targets) ** 2 / scale)
                + 0.01 * np.sum((proposed_clusters - target_clusters) ** 2)
            )
            if score < best_score:
                best_score = score
                best_role = int(role)
        cluster_assignment[cluster_code] = best_role
        role_counts[best_role] += counts[cluster_code]
        role_clusters[best_role] += 1.0
    labels = np.array(["construction", "evaluation"], dtype=object)
    return labels[cluster_assignment[cluster_codes]]


def _role_assignment(
    treatment: pd.Series,
    *,
    evaluation_fraction: float,
    random_state: int | None,
    clusters: pd.Series | None,
) -> np.ndarray:
    if clusters is None:
        return _row_role_assignment(
            treatment,
            evaluation_fraction=evaluation_fraction,
            random_state=random_state,
        )
    return _cluster_role_assignment(
        treatment,
        clusters,
        evaluation_fraction=evaluation_fraction,
        random_state=random_state,
    )


class _HonestRConstruction:
    """Internal construction/evaluation split and cross-fitted R-objective layer."""

    def __init__(
        self,
        *,
        outcome_factory: NuisanceFactory | None = None,
        propensity_factory: NuisanceFactory | None = None,
        cate_factory: WeightedCATEFactory | None = None,
        n_splits: int = 5,
        evaluation_fraction: float = 0.5,
        random_state: int | None = None,
        overlap_floor: float = 0.01,
        outcome_predict: PredictionAdapter | None = None,
        propensity_predict: PredictionAdapter | None = None,
        cate_predict: PredictionAdapter | None = None,
        outcome_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
        propensity_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
        cate_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
        propensity_tuning_splits: int = 3,
    ) -> None:
        if outcome_factory is not None and not callable(outcome_factory):
            raise TypeError("outcome_factory must be callable or None.")
        if propensity_factory is not None and not callable(propensity_factory):
            raise TypeError("propensity_factory must be callable or None.")
        if cate_factory is not None and not callable(cate_factory):
            raise TypeError("cate_factory must be callable or None.")
        if isinstance(evaluation_fraction, bool):
            raise TypeError("evaluation_fraction must be numeric.")
        try:
            normalized_fraction = float(evaluation_fraction)
        except (TypeError, ValueError) as error:
            raise TypeError("evaluation_fraction must be numeric.") from error
        if not np.isfinite(normalized_fraction) or not 0.0 < normalized_fraction < 1.0:
            raise ValueError("evaluation_fraction must be strictly between zero and one.")
        if isinstance(overlap_floor, bool):
            raise TypeError("overlap_floor must be numeric.")
        try:
            normalized_overlap = float(overlap_floor)
        except (TypeError, ValueError) as error:
            raise TypeError("overlap_floor must be numeric.") from error
        if not np.isfinite(normalized_overlap) or not 0.0 < normalized_overlap < 0.5:
            raise ValueError("overlap_floor must be strictly between zero and one half.")
        CrossFitter(n_splits=n_splits, random_state=random_state)
        if isinstance(propensity_tuning_splits, bool) or not isinstance(
            propensity_tuning_splits, int
        ):
            raise TypeError("propensity_tuning_splits must be an integer.")
        if propensity_tuning_splits < 2:
            raise ValueError("propensity_tuning_splits must be at least two.")
        self.outcome_factory = outcome_factory
        self.propensity_factory = propensity_factory
        self.cate_factory = cate_factory
        self.n_splits = n_splits
        self.evaluation_fraction = normalized_fraction
        self.random_state = random_state
        self.overlap_floor = normalized_overlap
        self.outcome_predict = outcome_predict
        self.propensity_predict = propensity_predict
        self.cate_predict = cate_predict
        self.outcome_alphas = _ridge_alphas(outcome_alphas)
        self.propensity_alphas = _ridge_alphas(propensity_alphas)
        self.cate_alphas = _ridge_alphas(cate_alphas)
        self.propensity_tuning_splits = propensity_tuning_splits

    def _outcome_factory(self) -> NuisanceFactory:
        if self.outcome_factory is not None:
            return self.outcome_factory
        alphas = self.outcome_alphas
        return lambda: _NativeRidgeCV(alphas)

    def _propensity_factory(self) -> NuisanceFactory:
        if self.propensity_factory is not None:
            return self.propensity_factory
        alphas = self.propensity_alphas
        tuning_splits = self.propensity_tuning_splits
        random_state = self.random_state
        return lambda: _NativePenalizedLogitCV(
            alphas=alphas,
            tuning_splits=tuning_splits,
            random_state=random_state,
        )

    def _cate_factory(self) -> WeightedCATEFactory:
        if self.cate_factory is not None:
            return self.cate_factory
        alphas = self.cate_alphas
        return lambda: _NativeWeightedRidgeCV(alphas=alphas)

    def _validate_role_capacity(
        self,
        treatment: pd.Series,
        roles: np.ndarray,
        clusters: pd.Series | None,
    ) -> None:
        construction = roles == "construction"
        evaluation = roles == "evaluation"
        for role_name, mask in (("construction", construction), ("evaluation", evaluation)):
            counts = treatment.iloc[mask].value_counts()
            if len(counts) != 2 or int(counts.min()) < 1:
                raise ValueError(f"The {role_name} role must contain both treatment arms.")
        construction_counts = treatment.iloc[construction].value_counts()
        if int(construction_counts.min()) < self.n_splits:
            raise ValueError(
                "Each treatment arm in the construction role must contain at least "
                "n_splits observations."
            )
        if self.propensity_factory is None:
            smallest_training_arm = int(construction_counts.min()) - int(
                np.ceil(int(construction_counts.min()) / self.n_splits)
            )
            if smallest_training_arm < self.propensity_tuning_splits:
                raise ValueError(
                    "The construction role is too small for native propensity inner tuning."
                )
        if clusters is None:
            return
        construction_clusters = clusters.iloc[construction]
        evaluation_clusters = clusters.iloc[evaluation]
        if construction_clusters.nunique() < self.n_splits:
            raise ValueError("The construction role must contain at least n_splits clusters.")
        if evaluation_clusters.nunique() < 2:
            raise ValueError("The evaluation role must contain at least two clusters.")
        construction_treatment = treatment.iloc[construction]
        for arm in (0.0, 1.0):
            represented = construction_clusters.loc[construction_treatment.eq(arm)].nunique()
            if represented < self.n_splits:
                raise ValueError(
                    "Each treatment arm must occur in at least n_splits construction clusters."
                )

    def fit(
        self,
        y: Any,
        *,
        treatment: Any,
        covariates: Any,
        clusters: Any | None = None,
    ) -> _HonestRConstructionResult:
        frame, index = _frame(covariates)
        if not index.is_unique:
            raise ValueError("Honest R-learner role tracking requires a unique row index.")
        observed = _series(y, name="outcome", index=index)
        assigned = _series(treatment, name="treatment", index=index)
        treatment_levels = np.unique(assigned.to_numpy(dtype=float))
        if len(treatment_levels) < 2:
            raise ValueError("treatment must contain both arms coded exactly 0 and 1.")
        if not np.array_equal(treatment_levels, np.array([0.0, 1.0])):
            raise ValueError("treatment must be coded exactly 0 and 1.")
        cluster_values = (
            None if clusters is None else _labels(clusters, name="clusters", index=index)
        )
        roles = _role_assignment(
            assigned,
            evaluation_fraction=self.evaluation_fraction,
            random_state=self.random_state,
            clusters=cluster_values,
        )
        self._validate_role_capacity(assigned, roles, cluster_values)
        construction = roles == "construction"
        evaluation = roles == "evaluation"
        construction_frame = frame.iloc[construction]
        evaluation_frame = frame.iloc[evaluation]
        construction_outcome = observed.iloc[construction]
        construction_treatment = assigned.iloc[construction]
        construction_clusters = (
            None if cluster_values is None else cluster_values.iloc[construction]
        )

        outcome_factory = _FreshNuisanceFactory(self._outcome_factory(), label="outcome nuisance")
        propensity_factory = _FreshNuisanceFactory(
            self._propensity_factory(), label="propensity nuisance"
        )
        cross_fitter = CrossFitter(
            n_splits=self.n_splits,
            random_state=self.random_state,
        )
        outcome_crossfit = cross_fitter.fit_predict_tasks(
            construction_frame,
            tasks=(
                CrossFitTask(
                    name="outcome_mean",
                    target=construction_outcome,
                    factory=outcome_factory,
                    predict=self.outcome_predict,
                ),
            ),
            strata=construction_treatment,
            clusters=construction_clusters,
        )
        propensity_crossfit = CrossFitter(
            propensity_factory=propensity_factory,
            n_splits=self.n_splits,
            random_state=self.random_state,
            propensity_predict=self.propensity_predict,
        ).fit_predict_class_probabilities(
            construction_frame,
            classes=construction_treatment,
            clusters=construction_clusters,
        )
        if not outcome_crossfit.fold.equals(propensity_crossfit.fold):
            raise RuntimeError("Outcome and propensity cross-fitting folds must be identical.")
        if 1.0 not in propensity_crossfit.probabilities.columns:
            raise ValueError("Propensity predictions must expose treated class 1.")
        construction_propensity = propensity_crossfit.probabilities[1.0].rename("propensity")
        construction_outcome_mean = outcome_crossfit.predictions["outcome_mean"]

        full_outcome_result = _fit(
            outcome_factory,
            construction_frame,
            construction_outcome,
        )
        evaluation_outcome_mean = pd.Series(
            _prediction(
                full_outcome_result,
                evaluation_frame,
                adapter=self.outcome_predict,
                propensity=False,
                expected=len(evaluation_frame),
            ),
            index=evaluation_frame.index,
            name="outcome_mean",
        )
        full_propensity_result = _fit(
            propensity_factory,
            construction_frame,
            construction_treatment,
        )
        evaluation_propensity = pd.Series(
            _prediction(
                full_propensity_result,
                evaluation_frame,
                adapter=self.propensity_predict,
                propensity=True,
                expected=len(evaluation_frame),
            ),
            index=evaluation_frame.index,
            name="propensity",
        )
        for probability in (construction_propensity, evaluation_propensity):
            values = probability.to_numpy(dtype=float)
            if np.any((values < self.overlap_floor) | (values > 1.0 - self.overlap_floor)):
                raise ValueError(
                    "Propensity predictions must lie inside the declared overlap interval; "
                    "CauseKit does not clip them."
                )

        outcome_residual = construction_outcome.to_numpy(
            dtype=float
        ) - construction_outcome_mean.to_numpy(dtype=float)
        treatment_residual = construction_treatment.to_numpy(
            dtype=float
        ) - construction_propensity.to_numpy(dtype=float)
        jacobian = float(np.mean(treatment_residual**2))
        treatment_scale = max(
            float(np.mean(construction_treatment.to_numpy(dtype=float) ** 2)),
            float(np.var(construction_treatment.to_numpy(dtype=float))),
            np.finfo(float).tiny,
        )
        tolerance = float(
            100.0
            * np.finfo(float).eps
            * max(len(construction_frame), construction_frame.shape[1], 1)
            * treatment_scale
        )
        if not np.isfinite(jacobian) or jacobian <= tolerance:
            raise ValueError(
                "Cross-fitted residual treatment variation is numerically unidentified."
            )
        pseudo_outcome = outcome_residual / treatment_residual
        weight = treatment_residual**2
        construction_residuals = pd.DataFrame(
            {
                "outcome_residual": outcome_residual,
                "treatment_residual": treatment_residual,
                "pseudo_outcome": pseudo_outcome,
                "weight": weight,
            },
            index=construction_frame.index.copy(),
        )
        cate_result = _fit_weighted_cate(
            self._cate_factory(),
            construction_frame,
            construction_residuals["pseudo_outcome"],
            sample_weight=construction_residuals["weight"],
        )
        construction_cate = pd.Series(
            _cate_prediction(
                cate_result,
                construction_frame,
                adapter=self.cate_predict,
            ),
            index=construction_frame.index.copy(),
            name="cate",
        )
        evaluation_cate = pd.Series(
            _cate_prediction(
                cate_result,
                evaluation_frame,
                adapter=self.cate_predict,
            ),
            index=evaluation_frame.index.copy(),
            name="cate",
        )
        r_error = outcome_residual - treatment_residual * construction_cate.to_numpy()
        direct_objective = float(r_error @ r_error)
        weighted_objective = float(
            np.sum(weight * (pseudo_outcome - construction_cate.to_numpy(dtype=float)) ** 2)
        )

        outcome_diagnostics = outcome_crossfit.model_diagnostics.copy()
        propensity_diagnostics = propensity_crossfit.model_diagnostics.copy()
        propensity_diagnostics["task"] = "propensity"
        refit_diagnostics = pd.DataFrame(
            [
                _model_diagnostic_row(
                    full_outcome_result,
                    task="outcome_mean_refit",
                    fold=-1,
                    train_nobs=len(construction_frame),
                    holdout_nobs=len(evaluation_frame),
                ),
                _model_diagnostic_row(
                    full_propensity_result,
                    task="propensity_refit",
                    fold=-1,
                    train_nobs=len(construction_frame),
                    holdout_nobs=len(evaluation_frame),
                ),
            ]
        )
        nuisance_diagnostics = pd.concat(
            [outcome_diagnostics, propensity_diagnostics, refit_diagnostics],
            ignore_index=True,
            sort=False,
        )
        cate_diagnostics = pd.DataFrame(
            [
                _model_diagnostic_row(
                    cate_result,
                    task="cate",
                    fold=-1,
                    train_nobs=len(construction_frame),
                    holdout_nobs=len(evaluation_frame),
                )
            ]
        )
        construction_predictions = pd.DataFrame(
            {
                "outcome_mean": construction_outcome_mean,
                "propensity": construction_propensity,
            },
            index=construction_frame.index.copy(),
        )
        evaluation_predictions = pd.DataFrame(
            {
                "outcome_mean": evaluation_outcome_mean,
                "propensity": evaluation_propensity,
            },
            index=evaluation_frame.index.copy(),
        )

        cluster_labels: pd.Index | None = None
        cluster_role_values: tuple[str, ...] | None = None
        if cluster_values is not None:
            cluster_labels = pd.Index(pd.unique(cluster_values), name=cluster_values.name)
            role_series = pd.Series(roles, index=index)
            cluster_role_values = tuple(
                str(role_series.loc[cluster_values.eq(label)].iloc[0]) for label in cluster_labels
            )
        return _HonestRConstructionResult(
            _estimation_index=index.copy(),
            _construction_index=construction_frame.index.copy(),
            _evaluation_index=evaluation_frame.index.copy(),
            _sample_role_values=tuple(str(role) for role in roles),
            _construction_fold_values=tuple(
                int(value) for value in outcome_crossfit.fold.to_numpy()
            ),
            _cluster_labels=cluster_labels,
            _cluster_role_values=cluster_role_values,
            _cate_result=cate_result,
            _cate_predict=self.cate_predict,
            _feature_names=frame.columns.copy(),
            construction_nuisance_predictions=construction_predictions,
            evaluation_nuisance_predictions=evaluation_predictions,
            construction_residuals=construction_residuals,
            construction_cate_predictions=construction_cate,
            evaluation_cate_predictions=evaluation_cate,
            nuisance_diagnostics=nuisance_diagnostics,
            cate_diagnostics=cate_diagnostics,
            construction_r_objective=direct_objective,
            weighted_construction_objective=weighted_objective,
            residual_treatment_second_moment=jacobian,
            residual_treatment_tolerance=tolerance,
            requested_evaluation_fraction=self.evaluation_fraction,
            realized_evaluation_fraction=float(evaluation.sum() / len(frame)),
            split_seed=self.random_state,
            n_splits=self.n_splits,
            overlap_floor=self.overlap_floor,
            cluster_split=cluster_values is not None,
            outcome_model_name=type(full_outcome_result).__name__,
            propensity_model_name=type(full_propensity_result).__name__,
            cate_model_name=type(cate_result).__name__,
        )


def _strict_regression(
    design: np.ndarray,
    target: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    nobs, nparams = design.shape
    if nobs <= nparams:
        raise ValueError(f"{label} requires more evaluation observations than parameters.")
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = (
        np.finfo(float).eps * max(design.shape) * float(singular_values[0])
        if len(singular_values)
        else 0.0
    )
    if int(np.sum(singular_values > tolerance)) < nparams:
        raise ValueError(f"{label} is rank deficient for the honest evaluation sample.")
    coefficients = np.linalg.solve(design.T @ design, design.T @ target)
    residual = target - design @ coefficients
    return np.asarray(coefficients, dtype=float), np.asarray(residual, dtype=float)


def _reference_probability(
    statistic: np.ndarray | float,
    *,
    distribution: str,
    degrees_of_freedom: float | None,
) -> np.ndarray:
    absolute = np.abs(np.asarray(statistic, dtype=float))
    return np.asarray(
        2.0 * norm.sf(absolute)
        if distribution == "normal"
        else 2.0 * t.sf(absolute, degrees_of_freedom),
        dtype=float,
    )


def _reference_critical(
    level: float,
    *,
    distribution: str,
    degrees_of_freedom: float | None,
) -> float:
    probability = 0.5 + level / 2.0
    return (
        float(norm.ppf(probability))
        if distribution == "normal"
        else float(t.ppf(probability, degrees_of_freedom))
    )


def _tie_preserving_groups(scores: pd.Series, *, groups: int) -> np.ndarray:
    values = scores.to_numpy(dtype=float)
    unique_values, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique_values) < groups:
        raise ValueError(
            "The honest evaluation sample has fewer unique CATE predictions than "
            "calibration_groups; score ties are never split."
        )
    cumulative = np.cumsum(counts)
    boundaries: list[int] = []
    previous_end = 0
    for group_position in range(1, groups):
        remaining_groups = groups - group_position
        candidate_ends = np.arange(
            previous_end + 1,
            len(unique_values) - remaining_groups + 1,
            dtype=int,
        )
        target = group_position * len(values) / groups
        deviations = np.abs(cumulative[candidate_ends - 1] - target)
        chosen_end = int(candidate_ends[int(np.argmin(deviations))])
        boundaries.append(chosen_end)
        previous_end = chosen_end
    unique_group = np.searchsorted(np.asarray(boundaries, dtype=int), np.arange(len(unique_values)))
    return unique_group[inverse] + 1


def _group_multiplier_band(
    influence: np.ndarray,
    standard_errors: np.ndarray,
    *,
    covariance: RLearnerCovariance,
    clusters: pd.Series | None,
    nparams: int,
    iterations: int,
    random_state: int | None,
    level: float,
) -> float:
    nobs = len(influence)
    if np.any(~np.isfinite(standard_errors)) or np.any(standard_errors <= 0.0):
        raise ValueError(
            "Simultaneous calibration bands require a positive finite standard error "
            "for every group."
        )
    if covariance == "robust":
        score_units = influence
        finite_sample_scale = np.sqrt(nobs / (nobs - nparams))
    else:
        if clusters is None:  # pragma: no cover - public validation protects this path
            raise RuntimeError("Internal clustered R-learner inference configuration error.")
        codes, labels = pd.factorize(clusters, sort=False)
        score_units = np.zeros((len(labels), influence.shape[1]), dtype=float)
        np.add.at(score_units, codes, influence)
        finite_sample_scale = np.sqrt(
            (len(labels) / (len(labels) - 1.0)) * ((nobs - 1.0) / (nobs - nparams))
        )
    rng = np.random.default_rng(random_state)
    maximum_statistics = np.empty(iterations, dtype=float)
    completed = 0
    while completed < iterations:
        batch = min(256, iterations - completed)
        multipliers = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(score_units)))
        perturbations = finite_sample_scale * multipliers @ score_units / nobs
        maximum_statistics[completed : completed + batch] = np.max(
            np.abs(perturbations / standard_errors), axis=1
        )
        completed += batch
    return float(np.quantile(maximum_statistics, level, method="higher"))


class RLearner:
    """Honest R-learner with split-conditional calibration and group inference.

    All nuisance and CATE tuning/fitting occurs in the construction role. The evaluation
    role is used once for R-loss, differential calibration, and overlap-weighted group
    effects with pointwise and seeded max-t simultaneous uncertainty.
    """

    def __init__(
        self,
        *,
        outcome_factory: NuisanceFactory | None = None,
        propensity_factory: NuisanceFactory | None = None,
        cate_factory: WeightedCATEFactory | None = None,
        n_splits: int = 5,
        evaluation_fraction: float = 0.5,
        random_state: int | None = None,
        overlap_floor: float = 0.01,
        covariance: RLearnerCovariance = "robust",
        calibration_groups: int = 5,
        simultaneous_level: float = 0.95,
        bootstrap_iterations: int = 999,
        bootstrap_random_state: int | None = None,
        outcome_predict: PredictionAdapter | None = None,
        propensity_predict: PredictionAdapter | None = None,
        cate_predict: PredictionAdapter | None = None,
        outcome_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
        propensity_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
        cate_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
        propensity_tuning_splits: int = 3,
    ) -> None:
        if covariance not in {"robust", "clustered"}:
            raise ValueError("covariance must be 'robust' or 'clustered'.")
        if (
            isinstance(calibration_groups, bool)
            or not isinstance(calibration_groups, Integral)
            or int(calibration_groups) < 2
        ):
            raise ValueError("calibration_groups must be an integer of at least two.")
        if (
            isinstance(bootstrap_iterations, bool)
            or not isinstance(bootstrap_iterations, Integral)
            or int(bootstrap_iterations) < 99
        ):
            raise ValueError("bootstrap_iterations must be an integer of at least 99.")
        if bootstrap_random_state is not None and not isinstance(bootstrap_random_state, Integral):
            raise TypeError("bootstrap_random_state must be an integer or None.")
        if isinstance(simultaneous_level, bool):
            raise TypeError("simultaneous_level must be numeric.")
        try:
            normalized_level = float(simultaneous_level)
        except (TypeError, ValueError) as error:
            raise TypeError("simultaneous_level must be numeric.") from error
        if not np.isfinite(normalized_level) or not 0.0 < normalized_level < 1.0:
            raise ValueError("simultaneous_level must be strictly between zero and one.")
        for adapter, label in (
            (outcome_predict, "outcome_predict"),
            (propensity_predict, "propensity_predict"),
            (cate_predict, "cate_predict"),
        ):
            if adapter is not None and not callable(adapter):
                raise TypeError(f"{label} must be callable or None.")
        self.outcome_factory: NuisanceFactory | None = outcome_factory
        self.propensity_factory: NuisanceFactory | None = propensity_factory
        self.cate_factory: WeightedCATEFactory | None = cate_factory
        self.n_splits: int = n_splits
        self.evaluation_fraction: float = float(evaluation_fraction)
        self.random_state: int | None = random_state
        self.overlap_floor: float = float(overlap_floor)
        self.covariance: RLearnerCovariance = cast(RLearnerCovariance, covariance)
        self.calibration_groups: int = int(calibration_groups)
        self.simultaneous_level: float = normalized_level
        self.bootstrap_iterations: int = int(bootstrap_iterations)
        self.bootstrap_random_state: int | None = (
            random_state if bootstrap_random_state is None else int(bootstrap_random_state)
        )
        self.outcome_predict: PredictionAdapter | None = outcome_predict
        self.propensity_predict: PredictionAdapter | None = propensity_predict
        self.cate_predict: PredictionAdapter | None = cate_predict
        self.outcome_alphas: Sequence[float] = outcome_alphas
        self.propensity_alphas: Sequence[float] = propensity_alphas
        self.cate_alphas: Sequence[float] = cate_alphas
        self.propensity_tuning_splits: int = propensity_tuning_splits
        self._construction_estimator: _HonestRConstruction = _HonestRConstruction(
            outcome_factory=outcome_factory,
            propensity_factory=propensity_factory,
            cate_factory=cate_factory,
            n_splits=n_splits,
            evaluation_fraction=evaluation_fraction,
            random_state=random_state,
            overlap_floor=overlap_floor,
            outcome_predict=outcome_predict,
            propensity_predict=propensity_predict,
            cate_predict=cate_predict,
            outcome_alphas=outcome_alphas,
            propensity_alphas=propensity_alphas,
            cate_alphas=cate_alphas,
            propensity_tuning_splits=propensity_tuning_splits,
        )

    def fit(
        self,
        y: Any,
        *,
        treatment: Any,
        covariates: Any,
        clusters: Any | None = None,
    ) -> RLearnerResult:
        """Fit on construction and evaluate once on immutable honest roles."""

        frame, index = _frame(covariates)
        observed = _series(y, name="outcome", index=index)
        assigned = _series(treatment, name="treatment", index=index)
        if self.covariance == "clustered" and clusters is None:
            raise ValueError("clusters must be provided when covariance='clustered'.")
        if self.covariance != "clustered" and clusters is not None:
            raise ValueError("clusters may be provided only when covariance='clustered'.")
        cluster_values = (
            None if clusters is None else _labels(clusters, name="clusters", index=index)
        )
        construction = self._construction_estimator.fit(
            observed,
            treatment=assigned,
            covariates=frame,
            clusters=cluster_values,
        )
        construction_index = construction.construction_index
        evaluation_index = construction.evaluation_index
        evaluation_outcome = observed.loc[evaluation_index]
        evaluation_treatment = assigned.loc[evaluation_index]
        evaluation_predictions = construction.evaluation_nuisance_predictions
        evaluation_cate = construction.evaluation_cate_predictions
        outcome_residual = evaluation_outcome.to_numpy(dtype=float) - evaluation_predictions[
            "outcome_mean"
        ].to_numpy(dtype=float)
        treatment_residual = evaluation_treatment.to_numpy(dtype=float) - evaluation_predictions[
            "propensity"
        ].to_numpy(dtype=float)
        evaluation_second_moment = float(np.mean(treatment_residual**2))
        if (
            not np.isfinite(evaluation_second_moment)
            or evaluation_second_moment <= construction.residual_treatment_tolerance
        ):
            raise ValueError(
                "Honest evaluation residual treatment variation is numerically unidentified."
            )
        cate_values = evaluation_cate.to_numpy(dtype=float)
        group_values = _tie_preserving_groups(
            evaluation_cate,
            groups=self.calibration_groups,
        )
        r_error = outcome_residual - treatment_residual * cate_values
        honest_r_loss = float(np.mean(r_error**2))

        construction_v = construction.construction_residuals["treatment_residual"].to_numpy(
            dtype=float
        )
        construction_u = construction.construction_residuals["outcome_residual"].to_numpy(
            dtype=float
        )
        construction_constant = float(
            (construction_v @ construction_u) / (construction_v @ construction_v)
        )
        constant_error = outcome_residual - treatment_residual * construction_constant
        honest_constant_r_loss = float(np.mean(constant_error**2))
        loss_scale = max(
            float(np.mean(outcome_residual**2)),
            float(np.mean((treatment_residual * construction_constant) ** 2)),
            np.finfo(float).tiny,
        )
        loss_tolerance = float(
            100.0 * np.finfo(float).eps * max(len(evaluation_index), frame.shape[1], 1) * loss_scale
        )
        if not np.isfinite(honest_constant_r_loss) or honest_constant_r_loss <= loss_tolerance:
            raise ValueError(
                "The construction-fitted constant-effect evaluation loss is numerically zero; "
                "R-loss gain is unavailable."
            )
        r_loss_gain = float(1.0 - honest_r_loss / honest_constant_r_loss)

        weight = treatment_residual**2
        calibration_center = float(weight @ cate_values / weight.sum())
        calibration_design = np.column_stack(
            [treatment_residual, treatment_residual * (cate_values - calibration_center)]
        )
        calibration_coefficients, calibration_error = _strict_regression(
            calibration_design,
            outcome_residual,
            label="Differential calibration",
        )
        evaluation_clusters = (
            None if cluster_values is None else cluster_values.loc[evaluation_index]
        )
        calibration_covariance = ols_covariance(
            calibration_design,
            calibration_error,
            covariance=cast(CovarianceType, self.covariance),
            clusters=evaluation_clusters,
        )
        calibration_standard_errors = np.sqrt(
            np.maximum(np.diag(calibration_covariance.matrix), 0.0)
        )
        if np.any(~np.isfinite(calibration_standard_errors)) or np.any(
            calibration_standard_errors <= 0.0
        ):
            raise ValueError(
                "Differential calibration requires positive finite coefficient standard errors."
            )
        calibration_index = pd.Index(["level", "heterogeneity"], dtype="object")
        calibration_coefficients_series = pd.Series(
            calibration_coefficients,
            index=calibration_index,
            name="coef",
        )
        calibration_covariance_frame = pd.DataFrame(
            calibration_covariance.matrix,
            index=calibration_index.copy(),
            columns=calibration_index.copy(),
        )
        heterogeneity_estimate = float(calibration_coefficients[1])
        heterogeneity_se = float(calibration_standard_errors[1])
        calibration_nulls = np.array([0.0, 1.0])
        calibration_statistics = (heterogeneity_estimate - calibration_nulls) / heterogeneity_se
        calibration_tests = pd.DataFrame(
            {
                "null": calibration_nulls,
                "estimate": heterogeneity_estimate,
                "std_err": heterogeneity_se,
                "statistic": calibration_statistics,
                "p_value": _reference_probability(
                    calibration_statistics,
                    distribution=calibration_covariance.distribution,
                    degrees_of_freedom=calibration_covariance.df,
                ),
            },
            index=pd.Index(["heterogeneity=0", "heterogeneity=1"], dtype="object"),
        )

        for group in range(1, self.calibration_groups + 1):
            in_group = group_values == group
            group_treatment = evaluation_treatment.iloc[in_group]
            if group_treatment.nunique() != 2:
                raise ValueError(
                    "Every honest calibration group must contain both treatment arms "
                    "without splitting CATE score ties."
                )
            if evaluation_clusters is not None and evaluation_clusters.iloc[in_group].nunique() < 2:
                raise ValueError(
                    "Every clustered honest calibration group must contain at least two clusters."
                )
        group_design = np.zeros((len(evaluation_index), self.calibration_groups), dtype=float)
        group_design[np.arange(len(evaluation_index)), group_values - 1] = treatment_residual
        group_coefficients, group_error = _strict_regression(
            group_design,
            outcome_residual,
            label="Honest calibration groups",
        )
        group_covariance = ols_covariance(
            group_design,
            group_error,
            covariance=cast(CovarianceType, self.covariance),
            clusters=evaluation_clusters,
        )
        group_standard_errors = np.sqrt(np.maximum(np.diag(group_covariance.matrix), 0.0))
        if np.any(~np.isfinite(group_standard_errors)) or np.any(group_standard_errors <= 0.0):
            raise ValueError("Honest calibration groups require positive finite standard errors.")
        group_statistic = group_coefficients / group_standard_errors
        group_pvalue = _reference_probability(
            group_statistic,
            distribution=group_covariance.distribution,
            degrees_of_freedom=group_covariance.df,
        )
        group_names = pd.Index(range(1, self.calibration_groups + 1), name="group")
        group_bread = np.linalg.inv(group_design.T @ group_design)
        group_influence_values = (
            len(evaluation_index) * (group_design * group_error[:, None]) @ group_bread
        )
        simultaneous_critical = _group_multiplier_band(
            group_influence_values,
            group_standard_errors,
            covariance=self.covariance,
            clusters=evaluation_clusters,
            nparams=self.calibration_groups,
            iterations=self.bootstrap_iterations,
            random_state=self.bootstrap_random_state,
            level=self.simultaneous_level,
        )
        point_critical = _reference_critical(
            0.95,
            distribution=group_covariance.distribution,
            degrees_of_freedom=group_covariance.df,
        )
        group_rows: list[dict[str, float | int]] = []
        for group in group_names:
            in_group = group_values == group
            group_scores = cate_values[in_group]
            group_treatment = evaluation_treatment.iloc[in_group]
            group_rows.append(
                {
                    "nobs": int(in_group.sum()),
                    "n_treated": int(group_treatment.sum()),
                    "n_control": int(len(group_treatment) - group_treatment.sum()),
                    "n_clusters": (
                        int(evaluation_clusters.iloc[in_group].nunique())
                        if evaluation_clusters is not None
                        else np.nan
                    ),
                    "predicted_cate_mean": float(group_scores.mean()),
                    "predicted_cate_min": float(group_scores.min()),
                    "predicted_cate_max": float(group_scores.max()),
                    "effect": float(group_coefficients[group - 1]),
                    "std_err": float(group_standard_errors[group - 1]),
                    "stat": float(group_statistic[group - 1]),
                    "p_value": float(group_pvalue[group - 1]),
                    "ci_lower": float(
                        group_coefficients[group - 1]
                        - point_critical * group_standard_errors[group - 1]
                    ),
                    "ci_upper": float(
                        group_coefficients[group - 1]
                        + point_critical * group_standard_errors[group - 1]
                    ),
                    "pointwise_level": 0.95,
                    "simultaneous_level": self.simultaneous_level,
                    "simultaneous_critical_value": simultaneous_critical,
                    "simultaneous_lower": float(
                        group_coefficients[group - 1]
                        - simultaneous_critical * group_standard_errors[group - 1]
                    ),
                    "simultaneous_upper": float(
                        group_coefficients[group - 1]
                        + simultaneous_critical * group_standard_errors[group - 1]
                    ),
                }
            )
        group_effects = pd.DataFrame(group_rows, index=group_names)
        group_covariance_frame = pd.DataFrame(
            group_covariance.matrix,
            index=group_names.copy(),
            columns=group_names.copy(),
        )
        group_influence = pd.DataFrame(
            group_influence_values,
            index=evaluation_index.copy(),
            columns=group_names.copy(),
        )
        evaluation_residuals = pd.DataFrame(
            {
                "outcome_residual": outcome_residual,
                "treatment_residual": treatment_residual,
                "pseudo_outcome": outcome_residual / treatment_residual,
                "weight": weight,
                "predicted_cate": cate_values,
                "r_error": r_error,
                "constant_r_error": constant_error,
            },
            index=evaluation_index.copy(),
        )
        notes = [
            "All reported evaluation records are split-conditional and were used once.",
            "Group effects are overlap-weighted residual-moment effects, not ordinary group ATEs.",
            "No unit-level CATE confidence intervals or policy-value claims are provided.",
        ]
        if self.outcome_factory is None and self.propensity_factory is None:
            notes.append("Outcome and propensity nuisances used CauseKit-native learners.")
        else:
            notes.append(
                "At least one external nuisance factory was supplied; fold-internal tuning "
                "remains the analyst's responsibility."
            )
        return RLearnerResult(
            _estimation_index=construction.estimation_index,
            _construction_index=construction_index,
            _evaluation_index=evaluation_index,
            _sample_role_values=tuple(construction.sample_role.astype(str)),
            _construction_fold_values=tuple(construction.construction_fold.astype(int)),
            _cluster_labels=(
                None if construction.cluster_role.empty else construction.cluster_role.index.copy()
            ),
            _cluster_role_values=(
                None
                if construction.cluster_role.empty
                else tuple(construction.cluster_role.astype(str))
            ),
            _evaluation_group_values=tuple(int(value) for value in group_values),
            _cate_result=construction._cate_result,
            _cate_predict=construction._cate_predict,
            _feature_names=construction._feature_names.copy(),
            construction_nuisance_predictions=construction.construction_nuisance_predictions.copy(),
            evaluation_nuisance_predictions=evaluation_predictions.copy(),
            construction_residuals=construction.construction_residuals.copy(),
            evaluation_residuals=evaluation_residuals,
            construction_cate_predictions=construction.construction_cate_predictions.copy(),
            evaluation_cate_predictions=evaluation_cate.copy(),
            nuisance_diagnostics=construction.nuisance_diagnostics.copy(),
            cate_diagnostics=construction.cate_diagnostics.copy(),
            calibration_coefficients=calibration_coefficients_series,
            calibration_covariance=calibration_covariance_frame,
            calibration_tests=calibration_tests,
            group_effects=group_effects,
            group_covariance=group_covariance_frame,
            group_influence=group_influence,
            honest_r_loss=honest_r_loss,
            honest_constant_r_loss=honest_constant_r_loss,
            r_loss_gain=r_loss_gain,
            construction_constant_effect=construction_constant,
            calibration_center=calibration_center,
            construction_r_objective=construction.construction_r_objective,
            weighted_construction_objective=construction.weighted_construction_objective,
            residual_treatment_second_moment=evaluation_second_moment,
            residual_treatment_tolerance=construction.residual_treatment_tolerance,
            requested_evaluation_fraction=construction.requested_evaluation_fraction,
            realized_evaluation_fraction=construction.realized_evaluation_fraction,
            split_seed=construction.split_seed,
            n_splits=construction.n_splits,
            overlap_floor=construction.overlap_floor,
            cluster_split=construction.cluster_split,
            nobs=len(frame),
            construction_nobs=len(construction_index),
            evaluation_nobs=len(evaluation_index),
            calibration_groups=self.calibration_groups,
            covariance_type=self.covariance,
            inference_distribution=calibration_covariance.distribution,
            inference_df=calibration_covariance.df,
            n_clusters=calibration_covariance.n_clusters,
            simultaneous_level=self.simultaneous_level,
            simultaneous_critical_value=simultaneous_critical,
            bootstrap_iterations=self.bootstrap_iterations,
            bootstrap_random_state=self.bootstrap_random_state,
            outcome_model_name=construction.outcome_model_name,
            propensity_model_name=construction.propensity_model_name,
            cate_model_name=construction.cate_model_name,
            native_outcome=self.outcome_factory is None,
            native_propensity=self.propensity_factory is None,
            native_cate=self.cate_factory is None,
            assumptions=(
                "Binary treatment consistency",
                "No interference",
                "Conditional exchangeability given declared pre-treatment covariates",
                "Overlap inside the declared interval without clipping",
                "Valid nuisance rates for residual orthogonalization",
                "Independent honest evaluation not used for fitting or tuning",
                "Independent sampling across declared clusters for clustered inference",
            ),
            notes=tuple(notes),
        )


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
            nuisance_diagnostics=nuisance.model_diagnostics.copy(),
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
    "RLearner",
    "RLearnerCovariance",
    "RLearnerResult",
    "TreatmentKind",
]

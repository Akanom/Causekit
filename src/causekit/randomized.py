"""Average treatment effects for randomized experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from ._covariance import CovarianceType, ols_covariance
from ._data import MissingPolicy

AdjustmentType = Literal["none", "lin"]


@dataclass(frozen=True)
class CovariateBalance:
    """Unadjusted treatment-arm balance for one pre-treatment covariate."""

    covariate: str
    treated_mean: float
    control_mean: float
    standardized_difference: float


@dataclass(frozen=True)
class RandomizedATEResult:
    """Difference-in-means or Lin-adjusted average treatment effect result."""

    estimate: float
    standard_error: float
    statistic: float
    pvalue: float
    params: pd.Series
    covariance: pd.DataFrame
    residuals: pd.Series
    fitted_values: pd.Series
    balance: tuple[CovariateBalance, ...]
    nobs: int
    n_treated: int
    n_control: int
    df_resid: int
    covariance_type: CovarianceType
    inference_distribution: str
    inference_df: float | None
    n_clusters: int | None
    adjustment: AdjustmentType
    y_name: str
    treatment_name: str
    covariate_names: tuple[str, ...]
    estimation_index: pd.Index
    dropped_rows: int
    notes: tuple[str, ...]
    assumptions: tuple[str, ...]
    converged: bool = True
    backend: str = "native-randomized-ate"

    @property
    def all_params(self) -> pd.Series:
        return self.params.copy()

    @property
    def coefficients(self) -> pd.Series:
        return self.params.copy()

    @property
    def pvalues(self) -> pd.Series:
        values = pd.Series(np.nan, index=self.params.index, name="p_value")
        values.loc["ate"] = self.pvalue
        return values

    @property
    def standard_errors(self) -> pd.Series:
        values = pd.Series(
            np.sqrt(np.maximum(np.diag(self.covariance), 0.0)),
            index=self.params.index,
            name="std_err",
        )
        return values

    @property
    def causal_interpretation(self) -> str:
        return (
            "The estimate targets the sample average treatment effect under the stated "
            "random assignment, consistency, no-interference, and analysis-plan assumptions."
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
            name="ate",
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
            index=pd.Index(["ate"], dtype="object"),
        )

    def to_markdown(self, digits: int = 4) -> str:
        if digits < 0:
            raise ValueError("digits must be non-negative.")
        return "\n".join(
            [
                "# Randomized-experiment ATE result",
                "",
                f"- Adjustment: `{self.adjustment}`",
                f"- Observations: `{self.nobs}` ({self.n_treated} treated, {self.n_control} control)",
                f"- Covariance: `{self.covariance_type}`",
                "",
                "| estimand | estimate | std_err | stat | p_value |",
                "|---|---:|---:|---:|---:|",
                f"| ATE | {self.estimate:.{digits}f} | {self.standard_error:.{digits}f} | "
                f"{self.statistic:.{digits}f} | {self.pvalue:.{digits}f} |",
                "",
                f"> {self.causal_interpretation}",
            ]
        )


def _numeric_frame(
    value: Any, *, name: str, prefix: str
) -> tuple[np.ndarray, tuple[str, ...], pd.Index | None]:
    names: tuple[str, ...]
    if isinstance(value, pd.Series):
        names = (str(value.name) if value.name is not None else prefix,)
        index = value.index.copy()
        raw = value.to_numpy()
    elif isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one column.")
        names = tuple(str(column) for column in value.columns)
        if len(set(names)) != len(names):
            raise ValueError(f"{name} column names must be unique.")
        index = value.index.copy()
        raw = value.to_numpy()
    else:
        raw = np.asarray(value)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        names = (
            tuple(prefix if raw.shape[1] == 1 else f"{prefix}_{i}" for i in range(raw.shape[1]))
            if raw.ndim == 2
            else ()
        )
        index = None
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric values.") from error
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be one- or two-dimensional numeric data.")
    return array, names, index


def _vector(value: Any, *, name: str) -> tuple[np.ndarray, str, pd.Index | None]:
    label = name
    index = None
    if isinstance(value, pd.Series):
        label = str(value.name) if value.name is not None else name
        index = value.index.copy()
        raw = value.to_numpy()
    else:
        raw = np.asarray(value)
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric values.") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array, label, index


class RandomizedATE:
    """Estimate an ATE in a two-arm randomized experiment.

    ``adjustment='none'`` reports the difference in arm means. ``adjustment='lin'``
    uses the fully interacted, mean-centered regression adjustment of Lin (2013); the
    treatment coefficient is therefore the covariate-averaged adjusted ATE.
    """

    def __init__(
        self,
        *,
        adjustment: AdjustmentType = "none",
        covariance: CovarianceType = "robust",
        missing: MissingPolicy = "raise",
    ) -> None:
        if adjustment not in {"none", "lin"}:
            raise ValueError("adjustment must be 'none' or 'lin'.")
        if covariance not in {"unadjusted", "robust", "clustered"}:
            raise ValueError("covariance must be 'unadjusted', 'robust', or 'clustered'.")
        if missing not in {"raise", "drop"}:
            raise ValueError("missing must be 'raise' or 'drop'.")
        self.adjustment = adjustment
        self.covariance = covariance
        self.missing = missing

    def fit(
        self,
        y: Any,
        *,
        treatment: Any,
        covariates: Any | None = None,
        clusters: Any | None = None,
    ) -> RandomizedATEResult:
        if self.adjustment == "lin" and covariates is None:
            raise ValueError("covariates are required when adjustment='lin'.")
        if self.covariance == "clustered" and clusters is None:
            raise ValueError("clusters are required when covariance='clustered'.")
        if self.covariance != "clustered" and clusters is not None:
            raise ValueError("clusters may be supplied only when covariance='clustered'.")

        outcome, y_name, y_index = _vector(y, name="y")
        assigned, treatment_name, treatment_index = _vector(treatment, name="treatment")
        nobs = len(outcome)
        if len(assigned) != nobs:
            raise ValueError("y and treatment must contain the same number of rows.")
        covariate_array = np.empty((nobs, 0), dtype=float)
        covariate_names: tuple[str, ...] = ()
        covariate_index = None
        if covariates is not None:
            covariate_array, covariate_names, covariate_index = _numeric_frame(
                covariates, name="covariates", prefix="covariate"
            )
            if len(covariate_array) != nobs:
                raise ValueError("covariates must contain the same number of rows as y.")
        cluster_array = None
        cluster_index = None
        if clusters is not None:
            if isinstance(clusters, pd.Series):
                cluster_index = clusters.index.copy()
                cluster_array = clusters.to_numpy()
            else:
                cluster_array = np.asarray(clusters)
            if cluster_array.ndim != 1 or len(cluster_array) != nobs:
                raise ValueError("clusters must contain exactly one label per observation.")

        labeled = [
            index
            for index in [y_index, treatment_index, covariate_index, cluster_index]
            if index is not None
        ]
        if labeled and any(not labeled[0].equals(index) for index in labeled[1:]):
            raise ValueError("Pandas indices must match exactly and in the same order.")
        index = labeled[0].copy() if labeled else pd.RangeIndex(nobs)
        invalid = ~np.isfinite(outcome) | ~np.isfinite(assigned)
        if covariate_array.shape[1]:
            invalid |= ~np.isfinite(covariate_array).all(axis=1)
        if cluster_array is not None:
            invalid |= pd.isna(cluster_array)
        dropped_rows = int(invalid.sum())
        if dropped_rows and self.missing == "raise":
            raise ValueError(
                "Inputs contain missing or non-finite values; set missing='drop' to remove them jointly."
            )
        if dropped_rows:
            keep = ~invalid
            outcome, assigned, covariate_array, index = (
                outcome[keep],
                assigned[keep],
                covariate_array[keep],
                index[keep],
            )
            if cluster_array is not None:
                cluster_array = cluster_array[keep]

        values = np.unique(assigned)
        if not np.array_equal(values, np.array([0.0, 1.0])):
            raise ValueError("treatment must contain both arms and be coded exactly 0 and 1.")
        nobs = len(outcome)
        centered = (
            covariate_array - covariate_array.mean(axis=0)
            if covariate_array.shape[1]
            else covariate_array
        )
        pieces = [np.ones((nobs, 1)), assigned[:, None]]
        names = ["const", "ate"]
        if self.adjustment == "lin":
            pieces.extend([centered, centered * assigned[:, None]])
            names.extend(covariate_names)
            names.extend(f"treatment:{name}" for name in covariate_names)
        design = np.column_stack(pieces)
        if nobs <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError(
                "The randomized-experiment design is rank deficient or has insufficient residual degrees of freedom."
            )
        coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
        fitted = design @ coefficients
        residual = outcome - fitted
        covariance = ols_covariance(
            design, residual, covariance=self.covariance, clusters=cluster_array
        )
        standard_error = float(np.sqrt(max(covariance.matrix[1, 1], 0.0)))
        statistic = float(coefficients[1] / standard_error)
        pvalue = float(
            2
            * (
                norm.sf(abs(statistic))
                if covariance.distribution == "normal"
                else t.sf(abs(statistic), covariance.df)
            )
        )

        balance: list[CovariateBalance] = []
        treated = assigned == 1
        for column, name in enumerate(covariate_names):
            treated_values = covariate_array[treated, column]
            control_values = covariate_array[~treated, column]
            pooled_sd = np.sqrt(
                (np.var(treated_values, ddof=1) + np.var(control_values, ddof=1)) / 2
            )
            difference = float(np.mean(treated_values) - np.mean(control_values))
            balance.append(
                CovariateBalance(
                    name,
                    float(np.mean(treated_values)),
                    float(np.mean(control_values)),
                    difference / pooled_sd if pooled_sd > 0 else float("nan"),
                )
            )

        parameter_index = pd.Index(names, dtype="object")
        notes = []
        if dropped_rows:
            notes.append(f"Dropped {dropped_rows} row(s) jointly because missing='drop'.")
        if self.adjustment == "none" and covariates is not None:
            notes.append("Covariates were used only for balance diagnostics, not estimation.")
        return RandomizedATEResult(
            estimate=float(coefficients[1]),
            standard_error=standard_error,
            statistic=statistic,
            pvalue=pvalue,
            params=pd.Series(coefficients, index=parameter_index, name="coef"),
            covariance=pd.DataFrame(
                covariance.matrix, index=parameter_index, columns=parameter_index
            ),
            residuals=pd.Series(residual, index=index, name="residual"),
            fitted_values=pd.Series(fitted, index=index, name="fitted"),
            balance=tuple(balance),
            nobs=nobs,
            n_treated=int(treated.sum()),
            n_control=int((~treated).sum()),
            df_resid=nobs - design.shape[1],
            covariance_type=self.covariance,
            inference_distribution=covariance.distribution,
            inference_df=covariance.df,
            n_clusters=covariance.n_clusters,
            adjustment=self.adjustment,
            y_name=y_name,
            treatment_name=treatment_name,
            covariate_names=covariate_names,
            estimation_index=index.copy(),
            dropped_rows=dropped_rows,
            notes=tuple(notes),
            assumptions=(
                "Random assignment with known analysis population",
                "Treatment consistency",
                "No interference between units",
                "No post-treatment covariates in adjustment",
                "Inference matches the assignment and dependence structure",
            ),
        )


__all__ = ["CovariateBalance", "RandomizedATE", "RandomizedATEResult"]

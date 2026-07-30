"""Average treatment effects from supplied observational nuisance predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, t

InferenceType = Literal["robust", "clustered"]
EstimandType = Literal["ate", "att", "atc"]


@dataclass(frozen=True)
class OverlapDiagnostic:
    """Propensity-score support and inverse-weight diagnostics."""

    propensity_min: float
    propensity_max: float
    clipped_observations: int
    treated_effective_sample_size: float
    control_effective_sample_size: float
    warning: bool


@dataclass(frozen=True)
class ObservationalATEResult:
    """Influence-function ATE result from supplied nuisance predictions."""

    estimate: float
    standard_error: float
    statistic: float
    pvalue: float
    influence_function: pd.Series
    overlap: OverlapDiagnostic
    nobs: int
    n_treated: int
    n_control: int
    covariance_type: InferenceType
    inference_distribution: str
    inference_df: float | None
    n_clusters: int | None
    estimator: str
    estimand: EstimandType
    estimation_index: pd.Index
    clipped_observations: int
    assumptions: tuple[str, ...]
    notes: tuple[str, ...]
    converged: bool = True
    backend: str = "native-influence-function"

    @property
    def params(self) -> pd.Series:
        return pd.Series({self.estimand: self.estimate}, name="coef")

    @property
    def standard_errors(self) -> pd.Series:
        return pd.Series({self.estimand: self.standard_error}, name="std_err")

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series({self.estimand: self.pvalue}, name="p_value")

    @property
    def causal_interpretation(self) -> str:
        return (
            "The estimate is causal only under consistency, no interference, conditional "
            "exchangeability, positivity, and valid nuisance-prediction/inference conditions."
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


def _vector(value: Any, *, name: str, numeric: bool = True) -> tuple[np.ndarray, pd.Index | None]:
    index = value.index.copy() if isinstance(value, pd.Series) else None
    raw = value.to_numpy() if isinstance(value, pd.Series) else np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if numeric:
        try:
            raw = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain only numeric values.") from error
    return raw, index


def _prepare(
    y: Any,
    treatment: Any,
    propensity: Any,
    *,
    outcome_treated: Any | None,
    outcome_control: Any | None,
    clusters: Any | None,
    clip: tuple[float, float] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    pd.Index,
    int,
]:
    y_array, y_index = _vector(y, name="y")
    treatment_array, treatment_index = _vector(treatment, name="treatment")
    propensity_array, propensity_index = _vector(propensity, name="propensity")
    arrays = [y_array, treatment_array, propensity_array]
    indices = [y_index, treatment_index, propensity_index]
    mu1 = mu0 = None
    if outcome_treated is not None:
        mu1, mu1_index = _vector(outcome_treated, name="outcome_treated")
        mu0, mu0_index = _vector(outcome_control, name="outcome_control")
        arrays.extend([mu1, mu0])
        indices.extend([mu1_index, mu0_index])
    cluster_array = None
    if clusters is not None:
        cluster_array, cluster_index = _vector(clusters, name="clusters", numeric=False)
        arrays.append(cluster_array)
        indices.append(cluster_index)
    nobs = len(y_array)
    if any(len(array) != nobs for array in arrays):
        raise ValueError("All inputs must contain the same number of observations.")
    labeled = [index for index in indices if index is not None]
    if labeled and any(not labeled[0].equals(index) for index in labeled[1:]):
        raise ValueError("Pandas indices must match exactly and in the same order.")
    index = labeled[0].copy() if labeled else pd.RangeIndex(nobs)
    if not all(np.isfinite(array).all() for array in arrays[: 3 + (2 if mu1 is not None else 0)]):
        raise ValueError("Numeric inputs must contain only finite values.")
    if cluster_array is not None and pd.isna(cluster_array).any():
        raise ValueError("clusters must not contain missing values.")
    if not np.array_equal(np.unique(treatment_array), np.array([0.0, 1.0])):
        raise ValueError("treatment must contain both arms and be coded exactly 0 and 1.")
    if np.any((propensity_array <= 0) | (propensity_array >= 1)):
        raise ValueError("propensity values must be strictly between zero and one.")
    clipped = 0
    if clip is not None:
        lower, upper = clip
        if not 0 < lower < upper < 1:
            raise ValueError("clip must satisfy 0 < lower < upper < 1.")
        clipped_values = np.clip(propensity_array, lower, upper)
        clipped = int(np.count_nonzero(clipped_values != propensity_array))
        propensity_array = clipped_values
    return y_array, treatment_array, propensity_array, mu1, mu0, cluster_array, index, clipped


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / (weights @ weights))


def _result(
    estimate: float,
    influence: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    *,
    clusters: np.ndarray | None,
    covariance: InferenceType,
    estimator: str,
    estimand: EstimandType,
    index: pd.Index,
    clipped: int,
) -> ObservationalATEResult:
    nobs = len(influence)
    n_clusters = None
    inference_df = None
    if covariance == "robust":
        variance = float(influence @ influence) / (nobs * (nobs - 1))
        distribution = "normal"
    else:
        if clusters is None:
            raise ValueError("clusters are required when covariance='clustered'.")
        codes, labels = pd.factorize(clusters, sort=False)
        n_clusters = len(labels)
        if n_clusters < 2:
            raise ValueError("Clustered inference requires at least two clusters.")
        cluster_scores = np.zeros(n_clusters)
        np.add.at(cluster_scores, codes, influence)
        variance = float(
            n_clusters / (n_clusters - 1) * (cluster_scores @ cluster_scores) / nobs**2
        )
        distribution = "t"
        inference_df = float(n_clusters - 1)
    standard_error = float(np.sqrt(max(variance, 0)))
    statistic = estimate / standard_error
    pvalue = float(
        2
        * (
            norm.sf(abs(statistic))
            if distribution == "normal"
            else t.sf(abs(statistic), inference_df)
        )
    )
    treated_weights = treatment / propensity
    control_weights = (1 - treatment) / (1 - propensity)
    overlap = OverlapDiagnostic(
        propensity_min=float(propensity.min()),
        propensity_max=float(propensity.max()),
        clipped_observations=clipped,
        treated_effective_sample_size=_effective_sample_size(treated_weights[treated_weights > 0]),
        control_effective_sample_size=_effective_sample_size(control_weights[control_weights > 0]),
        warning=bool(propensity.min() < 0.01 or propensity.max() > 0.99),
    )
    notes = []
    if clipped:
        notes.append(
            f"Clipped {clipped} propensity prediction(s); report unclipped sensitivity results."
        )
    if overlap.warning:
        notes.append(
            "Estimated propensity support is extreme; positivity and weight stability require scrutiny."
        )
    return ObservationalATEResult(
        estimate,
        standard_error,
        statistic,
        pvalue,
        pd.Series(influence, index=index, name="influence"),
        overlap,
        nobs,
        int(treatment.sum()),
        int((1 - treatment).sum()),
        covariance,
        distribution,
        inference_df,
        n_clusters,
        estimator,
        estimand,
        index.copy(),
        clipped,
        (
            "Treatment consistency",
            "No interference",
            "Conditional exchangeability given measured pre-treatment covariates",
            "Positivity",
            "Nuisance predictions are valid for the analysis sample",
            "Cross-fitting or justified complexity control when nuisance models are adaptive",
        ),
        tuple(notes),
    )


class IPWATE:
    """IPW ATE, ATT, or ATC from supplied propensity predictions."""

    def __init__(
        self,
        *,
        estimand: EstimandType = "ate",
        covariance: InferenceType = "robust",
        clip: tuple[float, float] | None = None,
    ) -> None:
        if covariance not in {"robust", "clustered"}:
            raise ValueError("covariance must be 'robust' or 'clustered'.")
        if estimand not in {"ate", "att", "atc"}:
            raise ValueError("estimand must be 'ate', 'att', or 'atc'.")
        self.estimand, self.covariance, self.clip = estimand, covariance, clip

    def fit(
        self, y: Any, *, treatment: Any, propensity: Any, clusters: Any | None = None
    ) -> ObservationalATEResult:
        y_array, assigned, ps, _, _, cluster_array, index, clipped = _prepare(
            y,
            treatment,
            propensity,
            outcome_treated=None,
            outcome_control=None,
            clusters=clusters,
            clip=self.clip,
        )
        if self.estimand == "ate":
            scores = assigned * y_array / ps - (1 - assigned) * y_array / (1 - ps)
            estimate = float(np.mean(scores))
            influence = scores - estimate
        elif self.estimand == "att":
            treated_weight = assigned
            control_weight = (1 - assigned) * ps / (1 - ps)
            treated_mean = float(treated_weight @ y_array / treated_weight.sum())
            control_mean = float(control_weight @ y_array / control_weight.sum())
            estimate = treated_mean - control_mean
            influence = (
                treated_weight * (y_array - treated_mean) / treated_weight.mean()
                - control_weight * (y_array - control_mean) / control_weight.mean()
            )
        else:
            treated_weight = assigned * (1 - ps) / ps
            control_weight = 1 - assigned
            treated_mean = float(treated_weight @ y_array / treated_weight.sum())
            control_mean = float(control_weight @ y_array / control_weight.sum())
            estimate = treated_mean - control_mean
            influence = (
                treated_weight * (y_array - treated_mean) / treated_weight.mean()
                - control_weight * (y_array - control_mean) / control_weight.mean()
            )
        return _result(
            estimate,
            influence,
            assigned,
            ps,
            clusters=cluster_array,
            covariance=self.covariance,
            estimator=f"ipw_{self.estimand}",
            estimand=self.estimand,
            index=index,
            clipped=clipped,
        )


class AIPWATE:
    """Augmented IPW ATE, ATT, or ATC from supplied nuisance predictions."""

    def __init__(
        self,
        *,
        estimand: EstimandType = "ate",
        covariance: InferenceType = "robust",
        clip: tuple[float, float] | None = None,
    ) -> None:
        if covariance not in {"robust", "clustered"}:
            raise ValueError("covariance must be 'robust' or 'clustered'.")
        if estimand not in {"ate", "att", "atc"}:
            raise ValueError("estimand must be 'ate', 'att', or 'atc'.")
        self.estimand, self.covariance, self.clip = estimand, covariance, clip

    def fit(
        self,
        y: Any,
        *,
        treatment: Any,
        propensity: Any,
        outcome_treated: Any,
        outcome_control: Any,
        clusters: Any | None = None,
    ) -> ObservationalATEResult:
        y_array, assigned, ps, mu1, mu0, cluster_array, index, clipped = _prepare(
            y,
            treatment,
            propensity,
            outcome_treated=outcome_treated,
            outcome_control=outcome_control,
            clusters=clusters,
            clip=self.clip,
        )
        assert mu1 is not None and mu0 is not None
        if self.estimand == "ate":
            scores = (
                mu1
                - mu0
                + assigned * (y_array - mu1) / ps
                - (1 - assigned) * (y_array - mu0) / (1 - ps)
            )
            estimate = float(np.mean(scores))
            influence = scores - estimate
        elif self.estimand == "att":
            numerator = assigned * (y_array - mu0) - (1 - assigned) * ps / (1 - ps) * (
                y_array - mu0
            )
            denominator = assigned
            estimate = float(numerator.mean() / denominator.mean())
            influence = (numerator - estimate * denominator) / denominator.mean()
        else:
            numerator = assigned * (1 - ps) / ps * (y_array - mu1) - (1 - assigned) * (
                y_array - mu1
            )
            denominator = 1 - assigned
            estimate = float(numerator.mean() / denominator.mean())
            influence = (numerator - estimate * denominator) / denominator.mean()
        return _result(
            estimate,
            influence,
            assigned,
            ps,
            clusters=cluster_array,
            covariance=self.covariance,
            estimator=f"aipw_{self.estimand}",
            estimand=self.estimand,
            index=index,
            clipped=clipped,
        )


__all__ = [
    "AIPWATE",
    "EstimandType",
    "IPWATE",
    "ObservationalATEResult",
    "OverlapDiagnostic",
]

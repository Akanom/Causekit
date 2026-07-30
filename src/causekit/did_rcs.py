"""Conventional difference-in-differences for repeated cross sections.

This module is deliberately separate from :mod:`causekit.did`.  Rows are sampled
observations, not balanced-panel entities, so cell means, influence functions, and
cluster aggregation are all constructed at the observation/PSU level.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import norm
from scipy.stats import t as student_t

from .crossfit import CrossFitTask, CrossFitter
from .did import (
    ControlGroup,
    DiDCovariance,
    _joint_wald,
)

RCSComposition = Literal["stationary"]
RCSInference = Literal["analytic"]


@dataclass(frozen=True)
class RepeatedCrossSectionPretrendDiagnostic:
    """Adjacent repeated-cross-section placebo effects and their joint test."""

    available: bool
    reason: str | None
    placebo_effects: pd.DataFrame
    influence: pd.DataFrame
    covariance: pd.DataFrame
    statistic: float | None
    pvalue: float | None
    n_restrictions: int
    df_num: int
    df_denom: float | None
    distribution: str | None
    covariance_type: DiDCovariance
    n_clusters: int | None
    null_hypothesis: str
    notes: tuple[str, ...]
    conditional: bool = False
    covariates: tuple[str, ...] = ()
    cross_fitted: bool = False


@dataclass(frozen=True)
class RepeatedCrossSectionDiDResult:
    """Auditable conventional DiD result for independent repeated samples."""

    estimate: float
    standard_error: float
    statistic: float
    pvalue: float
    group_time: pd.DataFrame
    event_study: pd.DataFrame
    calendar_time: pd.DataFrame
    group_time_influence: pd.DataFrame
    event_study_influence: pd.DataFrame
    calendar_time_influence: pd.DataFrame
    overall_influence: pd.Series
    pretrend: RepeatedCrossSectionPretrendDiagnostic
    cell_counts: pd.DataFrame
    cohort_sizes: pd.Series
    n_observations: int
    n_periods: int
    n_clusters: int | None
    covariance_type: DiDCovariance
    inference_distribution: str
    inference_df: float | None
    inference_method: RCSInference
    method: str
    parallel_trends: str
    control_group: str
    composition: str
    anticipation: int
    pre_periods: int
    sampling_unit: str
    time_name: str
    outcome_name: str
    treatment_time_name: str
    cluster_name: str | None
    covariates: tuple[str, ...]
    cross_fitted: bool
    estimation_index: pd.Index
    inference_clusters: pd.Series
    times: tuple[float, ...]
    design_fingerprint: str
    assumptions: tuple[str, ...]
    notes: tuple[str, ...]
    converged: bool = True
    backend: str = "native-repeated-cross-section-did"
    nuisance_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    nuisance_fold: pd.Series = field(default_factory=lambda: pd.Series(dtype="int64", name="fold"))
    nuisance_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_splits: int | None = None
    nuisance_probability_floor: float | None = None

    @property
    def nobs(self) -> int:
        return self.n_observations

    @property
    def params(self) -> pd.Series:
        return pd.Series({"esavg": self.estimate}, name="coef")

    @property
    def all_params(self) -> pd.Series:
        return self.params.copy()

    @property
    def coefficients(self) -> pd.Series:
        return self.params.copy()

    @property
    def covariance(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[self.standard_error**2]],
            index=pd.Index(["esavg"], dtype="object"),
            columns=pd.Index(["esavg"], dtype="object"),
        )

    @property
    def standard_errors(self) -> pd.Series:
        return pd.Series({"esavg": self.standard_error}, name="std_err")

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series({"esavg": self.pvalue}, name="p_value")

    @property
    def causal_interpretation(self) -> str:
        return (
            "The estimates are causal only under consistency, no interference, overlap, "
            "the declared no-anticipation window, repeated-cross-section parallel trends, "
            "and stationary composition of the declared cohort populations."
        )

    def conf_int(self, level: float = 0.95) -> pd.Series:
        """Return a pointwise confidence interval for the ESavg summary."""

        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        probability = 0.5 + level / 2.0
        critical = (
            float(norm.ppf(probability))
            if self.inference_distribution == "normal"
            else float(student_t.ppf(probability, self.inference_df))
        )
        return pd.Series(
            {
                "lower": self.estimate - critical * self.standard_error,
                "upper": self.estimate + critical * self.standard_error,
            },
            name="esavg",
        )

    def summary_frame(self, level: float = 0.95) -> pd.DataFrame:
        """Return the scalar ESavg summary in CauseKit's standard schema."""

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
            index=pd.Index(["esavg"], dtype="object"),
        )


@dataclass(frozen=True)
class _RepeatedSample:
    outcome: np.ndarray
    time_positions: np.ndarray
    cohorts: np.ndarray
    treated_cohorts: tuple[float, ...]
    original_positions: dict[float, int]
    effective_positions: dict[float, int]
    row_effective_positions: np.ndarray
    never_mask: np.ndarray
    clusters: np.ndarray | None
    cluster_codes: np.ndarray | None
    n_clusters: int | None
    index: pd.Index
    times: np.ndarray
    cell_counts: pd.DataFrame
    design_fingerprint: str


@dataclass(frozen=True)
class _RepeatedEffectCell:
    cohort: float
    time: float
    event_time: int
    base_period: float
    att: float
    influence: np.ndarray
    n_treated_target: int
    n_treated_base: int
    n_comparison_target: int
    n_comparison_base: int
    comparison_cohorts: tuple[float, ...]
    cohort_share: float


@dataclass(frozen=True)
class _CovariateComparison:
    cohort: float
    target_position: int
    base_position: int
    stage: Literal["group_time", "conditional_pretrend"]
    comparison_mask: np.ndarray
    comparison_cohorts: tuple[float, ...]
    task_names: dict[str, str]


def _numeric_role(data: pd.DataFrame, name: str, *, role: str) -> np.ndarray:
    series = data[name]
    if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        raise ValueError(f"{role} must contain only numeric values.")
    if series.isna().any():
        raise ValueError(f"{role} must not contain missing values.")
    values = series.to_numpy(dtype=float)
    return values


def _prepare_repeated_sample(
    data: pd.DataFrame,
    *,
    outcome: str,
    time: str,
    treatment_time: str,
    never_treated: float,
    anticipation: int,
    covariance: DiDCovariance,
    cluster: str | None,
) -> _RepeatedSample:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must contain observations.")
    if not data.columns.is_unique:
        raise ValueError("data column names must be unique.")
    roles = (outcome, time, treatment_time)
    if any(not isinstance(name, str) or not name for name in roles):
        raise TypeError("outcome, time, and treatment_time must be non-empty column names.")
    if len(set(roles)) != len(roles):
        raise ValueError("outcome, time, and treatment_time must name distinct columns.")
    missing = [name for name in roles if name not in data.columns]
    if missing:
        raise ValueError(f"data is missing required role column(s): {missing}.")
    if not data.index.is_unique:
        raise ValueError("data index labels must be unique for observation-level audit records.")
    try:
        index_missing = bool(np.asarray(pd.isna(data.index)).any())
    except (TypeError, ValueError):
        index_missing = False
    if index_missing:
        raise ValueError("data index labels must not be missing.")

    if covariance == "robust" and cluster is not None:
        raise ValueError("cluster may be supplied only when covariance='clustered'.")
    if covariance == "clustered" and cluster is None:
        raise ValueError("a cluster column is required when covariance='clustered'.")
    if cluster is not None:
        if not isinstance(cluster, str) or not cluster:
            raise TypeError("cluster must be a non-empty column name.")
        if cluster in roles:
            raise ValueError("cluster must be distinct from outcome and timing role columns.")
        if cluster not in data.columns:
            raise ValueError(f"data is missing cluster column {cluster!r}.")

    outcomes = _numeric_role(data, outcome, role="outcome")
    if not np.isfinite(outcomes).all():
        raise ValueError("outcome must contain only finite values.")
    time_values = _numeric_role(data, time, role="time")
    if not np.isfinite(time_values).all():
        raise ValueError("time must contain only finite values.")
    cohort_values = _numeric_role(data, treatment_time, role="treatment_time")

    if not isinstance(never_treated, Real) or isinstance(never_treated, (bool, np.bool_)):
        raise TypeError("never_treated must be a numeric sentinel.")
    never_value = float(never_treated)
    if np.isnan(never_value):
        raise ValueError("never_treated must not be NaN.")
    never_mask = cohort_values == never_value
    if not never_mask.any():
        raise ValueError("the sample must contain the declared never-treated cohort.")
    treated_values = cohort_values[~never_mask]
    if not np.isfinite(treated_values).all():
        raise ValueError(
            "finite treatment cohorts are required; non-finite values must equal never_treated."
        )

    times = np.unique(time_values)
    if len(times) < 2:
        raise ValueError("repeated-cross-section DiD requires at least two observed time periods.")
    if np.any(times == never_value):
        raise ValueError("never_treated must not equal an observed time value.")
    time_positions = np.searchsorted(times, time_values)
    treated_cohorts = tuple(float(value) for value in np.unique(treated_values))
    if not treated_cohorts:
        raise ValueError("the sample must contain at least one treated cohort.")
    observed_times = set(float(value) for value in times)
    if any(cohort not in observed_times for cohort in treated_cohorts):
        raise ValueError("every finite treatment time must coincide with an observed time label.")
    original_positions = {cohort: int(np.searchsorted(times, cohort)) for cohort in treated_cohorts}
    effective_positions = {
        cohort: position - anticipation for cohort, position in original_positions.items()
    }
    if any(position <= 0 for position in effective_positions.values()):
        raise ValueError(
            "every treated cohort must retain a clean observed baseline before the declared "
            "anticipation boundary."
        )
    row_effective_positions = np.full(len(data), len(times) + 1, dtype=int)
    for cohort, position in effective_positions.items():
        row_effective_positions[cohort_values == cohort] = position

    clusters: np.ndarray | None = None
    cluster_codes: np.ndarray | None = None
    n_clusters: int | None = None
    if cluster is not None:
        cluster_series = data[cluster]
        if cluster_series.isna().any():
            raise ValueError("cluster labels must not contain missing values.")
        try:
            codes, labels = pd.factorize(cluster_series, sort=False)
        except TypeError as error:
            raise ValueError("cluster labels must be scalar and hashable.") from error
        if np.any(codes < 0):
            raise ValueError("cluster labels must not contain missing values.")
        if len(labels) < 2:
            raise ValueError("clustered inference requires at least two clusters.")
        clusters = cluster_series.to_numpy(copy=True)
        cluster_codes = codes.astype(int, copy=False)
        n_clusters = len(labels)

    count_frame = pd.DataFrame(
        {
            "cohort": cohort_values,
            "time": time_values,
            "cluster": np.arange(len(data)) if clusters is None else clusters,
        },
        index=data.index,
    )
    grouped = count_frame.groupby(["cohort", "time"], sort=True, dropna=False)
    cell_counts = grouped.size().rename("nobs").to_frame()
    if clusters is not None:
        cell_counts["n_clusters"] = grouped["cluster"].nunique()
    cell_counts["pooled_share"] = cell_counts["nobs"] / len(data)

    fingerprint = hashlib.sha256()
    fingerprint.update(
        repr(
            (
                outcome,
                time,
                treatment_time,
                cluster,
                never_value,
                anticipation,
                covariance,
            )
        ).encode()
    )
    fingerprint_frame = pd.DataFrame(
        {
            "outcome": outcomes,
            "time": time_values,
            "cohort": cohort_values,
            "cluster": np.zeros(len(data), dtype=np.int8) if clusters is None else clusters,
        },
        index=data.index,
    )
    try:
        row_hashes = pd.util.hash_pandas_object(fingerprint_frame, index=True).to_numpy(
            dtype=np.uint64
        )
    except TypeError as error:
        raise ValueError("data index and cluster labels must be scalar and hashable.") from error
    fingerprint.update(np.sort(row_hashes).tobytes())

    return _RepeatedSample(
        outcome=outcomes,
        time_positions=time_positions,
        cohorts=cohort_values,
        treated_cohorts=treated_cohorts,
        original_positions=original_positions,
        effective_positions=effective_positions,
        row_effective_positions=row_effective_positions,
        never_mask=never_mask,
        clusters=clusters,
        cluster_codes=cluster_codes,
        n_clusters=n_clusters,
        index=data.index.copy(),
        times=times,
        cell_counts=cell_counts,
        design_fingerprint=fingerprint.hexdigest(),
    )


def _prepare_repeated_covariates(
    data: pd.DataFrame,
    sample: _RepeatedSample,
    *,
    covariates: Sequence[str],
    protected_names: Sequence[str | None],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if isinstance(covariates, (str, bytes)):
        raise TypeError("covariates must be a non-empty sequence of column names.")
    names = tuple(covariates)
    if not names:
        raise ValueError("covariates must contain at least one column name.")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("covariates must contain only non-empty column names.")
    if len(set(names)) != len(names):
        raise ValueError("covariates must name distinct columns.")
    protected = {name for name in protected_names if name is not None}
    overlap = sorted(set(names) & protected)
    if overlap:
        raise ValueError(
            "covariates must be distinct from outcome, timing, and cluster role columns: "
            f"{overlap}."
        )
    missing = [name for name in names if name not in data.columns]
    if missing:
        raise ValueError(f"data is missing covariate column(s): {missing}.")
    frame = data.loc[:, list(names)].copy()
    if frame.isna().any().any():
        raise ValueError("covariates must not contain missing values.")
    if any(
        not is_numeric_dtype(frame[name].dtype) or is_bool_dtype(frame[name].dtype)
        for name in names
    ):
        raise ValueError("covariates must contain only numeric values.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("covariates must contain only finite values.")
    if not frame.index.equals(sample.index):
        raise RuntimeError("covariate rows must retain the validated estimation index.")
    return frame, names


def _covariate_design_fingerprint(
    sample: _RepeatedSample,
    covariates: pd.DataFrame,
    *,
    n_splits: int,
    probability_floor: float,
) -> str:
    fingerprint = hashlib.sha256()
    fingerprint.update(sample.design_fingerprint.encode())
    fingerprint.update(repr((tuple(covariates.columns), n_splits, probability_floor)).encode())
    row_hashes = pd.util.hash_pandas_object(covariates, index=True).to_numpy(dtype=np.uint64)
    fingerprint.update(np.sort(row_hashes).tobytes())
    return fingerprint.hexdigest()


def _comparison_membership(
    sample: _RepeatedSample,
    *,
    cohort: float,
    target_position: int,
    control_group: ControlGroup,
) -> np.ndarray:
    if control_group == "never_treated":
        membership = sample.never_mask.copy()
    else:
        membership = sample.never_mask | (sample.row_effective_positions > target_position)
    membership &= sample.cohorts != cohort
    return membership


def _cell_mean_and_influence(
    sample: _RepeatedSample,
    mask: np.ndarray,
    *,
    label: str,
) -> tuple[float, np.ndarray, int]:
    count = int(np.count_nonzero(mask))
    if count < 2:
        raise ValueError(
            f"{label} must contain at least two observations for analytical inference."
        )
    if sample.cluster_codes is not None and len(np.unique(sample.cluster_codes[mask])) < 2:
        raise ValueError(
            f"{label} must contain observations from at least two clusters for clustered inference."
        )
    mean = float(sample.outcome[mask].mean())
    probability = count / len(sample.outcome)
    influence = mask.astype(float) / probability * (sample.outcome - mean)
    return mean, influence, count


def _score_statistics_rcs(
    sample: _RepeatedSample,
    estimate: float,
    influence: np.ndarray,
    *,
    covariance: DiDCovariance,
) -> tuple[float, float, float, str, float | None, int | None]:
    n = len(influence)
    inference_df: float | None = None
    if covariance == "robust":
        variance = float(influence @ influence) / (n * (n - 1))
        distribution = "normal"
        n_clusters = None
    else:
        if sample.cluster_codes is None or sample.n_clusters is None:
            raise ValueError("cluster codes are required for clustered inference.")
        cluster_scores = np.zeros(sample.n_clusters)
        np.add.at(cluster_scores, sample.cluster_codes, influence)
        variance = float(
            sample.n_clusters / (sample.n_clusters - 1) * (cluster_scores @ cluster_scores) / n**2
        )
        distribution = "t"
        inference_df = float(sample.n_clusters - 1)
        n_clusters = sample.n_clusters
    standard_error = float(np.sqrt(max(variance, 0.0)))
    if standard_error == 0.0:
        statistic = float("nan") if estimate == 0.0 else float(np.sign(estimate) * np.inf)
        pvalue = float("nan") if estimate == 0.0 else 0.0
    else:
        statistic = estimate / standard_error
        pvalue = float(
            2
            * (
                norm.sf(abs(statistic))
                if distribution == "normal"
                else student_t.sf(abs(statistic), inference_df)
            )
        )
    return standard_error, statistic, pvalue, distribution, inference_df, n_clusters


def _summary_row_rcs(
    sample: _RepeatedSample,
    estimate: float,
    influence: np.ndarray,
    *,
    covariance: DiDCovariance,
) -> dict[str, float]:
    standard_error, statistic, pvalue, _, _, _ = _score_statistics_rcs(
        sample, estimate, influence, covariance=covariance
    )
    return {
        "att": estimate,
        "std_err": standard_error,
        "stat": statistic,
        "p_value": pvalue,
    }


def _score_cross_covariance_rcs(
    sample: _RepeatedSample,
    left: np.ndarray,
    right: np.ndarray,
    *,
    covariance: DiDCovariance,
) -> tuple[np.ndarray, int | None]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Influence matrices must be two-dimensional and row aligned.")
    n = left.shape[0]
    if covariance == "robust":
        return left.T @ right / (n * (n - 1)), None
    if sample.cluster_codes is None or sample.n_clusters is None:
        raise ValueError("cluster codes are required for clustered joint inference.")
    cluster_left = np.zeros((sample.n_clusters, left.shape[1]))
    cluster_right = np.zeros((sample.n_clusters, right.shape[1]))
    np.add.at(cluster_left, sample.cluster_codes, left)
    np.add.at(cluster_right, sample.cluster_codes, right)
    scale = sample.n_clusters / (sample.n_clusters - 1) / n**2
    return scale * cluster_left.T @ cluster_right, sample.n_clusters


def _effect_from_four_cells(
    sample: _RepeatedSample,
    *,
    cohort: float,
    target_position: int,
    base_position: int,
    control_group: ControlGroup,
) -> tuple[float, np.ndarray, tuple[int, int, int, int], tuple[float, ...]]:
    treated_membership = sample.cohorts == cohort
    comparison_membership = _comparison_membership(
        sample,
        cohort=cohort,
        target_position=target_position,
        control_group=control_group,
    )
    treated_target = treated_membership & (sample.time_positions == target_position)
    treated_base = treated_membership & (sample.time_positions == base_position)
    comparison_target = comparison_membership & (sample.time_positions == target_position)
    comparison_base = comparison_membership & (sample.time_positions == base_position)
    treated_target_mean, treated_target_score, n_treated_target = _cell_mean_and_influence(
        sample, treated_target, label="the treated target-period cell"
    )
    treated_base_mean, treated_base_score, n_treated_base = _cell_mean_and_influence(
        sample, treated_base, label="the treated baseline-period cell"
    )
    comparison_target_mean, comparison_target_score, n_comparison_target = _cell_mean_and_influence(
        sample, comparison_target, label="the comparison target-period cell"
    )
    comparison_base_mean, comparison_base_score, n_comparison_base = _cell_mean_and_influence(
        sample, comparison_base, label="the comparison baseline-period cell"
    )
    estimate = (treated_target_mean - treated_base_mean) - (
        comparison_target_mean - comparison_base_mean
    )
    influence = (
        treated_target_score - treated_base_score - comparison_target_score + comparison_base_score
    )
    comparison_cohorts = tuple(
        float(value) for value in np.unique(sample.cohorts[comparison_membership])
    )
    return (
        estimate,
        influence,
        (n_treated_target, n_treated_base, n_comparison_target, n_comparison_base),
        comparison_cohorts,
    )


def _treated_probability_prediction(result: Any, X: pd.DataFrame) -> Any:
    if not hasattr(result, "predict_proba"):
        raise TypeError(
            "The fitted repeated-cross-section propensity result must provide "
            "predict_proba(X), or CrossFitter must supply propensity_predict=."
        )
    values = result.predict_proba(X)
    if isinstance(values, pd.DataFrame):
        if not values.index.equals(X.index):
            raise ValueError("Propensity prediction index must match the held-out index.")
        if 1 in values.columns:
            return values[1]
        if values.shape[1] == 2:
            return values.iloc[:, 1]
        raise ValueError("predict_proba must expose the treated-class probability.")
    raw = np.asarray(values, dtype=float)
    if raw.ndim == 2 and raw.shape[1] == 2:
        return raw[:, 1]
    return raw


def _normalized_ratio(
    weight: np.ndarray,
    value: np.ndarray,
    *,
    label: str,
) -> tuple[float, np.ndarray]:
    denominator = float(weight.mean())
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        raise ValueError(
            f"The {label} normalization has no usable overlap; no clipping, trimming, "
            "or row deletion was applied."
        )
    eta = weight * value / denominator
    mean = float(eta.mean())
    influence = eta - weight * mean / denominator
    if not np.isfinite(influence).all():
        raise ValueError(f"The {label} score is non-finite.")
    return mean, influence


def _effect_from_cross_fitted_score(
    sample: _RepeatedSample,
    comparison: _CovariateComparison,
    predictions: pd.DataFrame,
    *,
    probability_floor: float,
) -> tuple[float, np.ndarray, tuple[int, int, int, int]]:
    cohort_mask = sample.cohorts == comparison.cohort
    target_mask = sample.time_positions == comparison.target_position
    base_mask = sample.time_positions == comparison.base_position
    pair_mask = (cohort_mask | comparison.comparison_mask) & (target_mask | base_mask)
    treated_target = cohort_mask & target_mask
    treated_base = cohort_mask & base_mask
    comparison_target = comparison.comparison_mask & target_mask
    comparison_base = comparison.comparison_mask & base_mask
    cell_masks = (treated_target, treated_base, comparison_target, comparison_base)
    cell_labels = (
        "treated target cell",
        "treated baseline cell",
        "comparison target cell",
        "comparison baseline cell",
    )
    counts: list[int] = []
    for mask, label in zip(cell_masks, cell_labels, strict=True):
        _, _, count = _cell_mean_and_influence(sample, mask, label=label)
        counts.append(count)

    used = np.flatnonzero(pair_mask)
    pair_outcome = sample.outcome[used]
    treated = cohort_mask[used].astype(float)
    post = target_mask[used].astype(float)
    names = comparison.task_names
    propensity = predictions[names["propensity"]].to_numpy(dtype=float)[used]
    if np.any(propensity <= probability_floor) or np.any(propensity >= 1.0 - probability_floor):
        raise ValueError(
            "Cross-fitted repeated-cross-section propensities violate "
            "nuisance_probability_floor; no clipping, trimming, or row deletion was applied."
        )
    m0_pre = predictions[names["outcome_control_pre"]].to_numpy(dtype=float)[used]
    m0_post = predictions[names["outcome_control_post"]].to_numpy(dtype=float)[used]
    m1_pre = predictions[names["outcome_treated_pre"]].to_numpy(dtype=float)[used]
    m1_post = predictions[names["outcome_treated_post"]].to_numpy(dtype=float)[used]
    m0 = post * m0_post + (1.0 - post) * m0_pre

    odds = propensity / (1.0 - propensity)
    ratio_inputs = (
        (treated * post, pair_outcome - m0, 1.0, "treated target residual"),
        (treated * (1.0 - post), pair_outcome - m0, -1.0, "treated baseline residual"),
        (
            odds * (1.0 - treated) * post,
            pair_outcome - m0,
            -1.0,
            "comparison target residual",
        ),
        (
            odds * (1.0 - treated) * (1.0 - post),
            pair_outcome - m0,
            1.0,
            "comparison baseline residual",
        ),
        (treated, m1_post - m0_post, 1.0, "treated pooled post regression"),
        (treated * post, m1_post - m0_post, -1.0, "treated target post regression"),
        (treated, m1_pre - m0_pre, -1.0, "treated pooled baseline regression"),
        (
            treated * (1.0 - post),
            m1_pre - m0_pre,
            1.0,
            "treated baseline regression",
        ),
    )
    estimate = 0.0
    pair_influence = np.zeros(len(used))
    for weight, value, sign, label in ratio_inputs:
        component, component_influence = _normalized_ratio(weight, value, label=label)
        estimate += sign * component
        pair_influence += sign * component_influence

    influence = np.zeros(len(sample.outcome))
    influence[used] = len(sample.outcome) / len(used) * pair_influence
    if abs(float(influence.sum())) > 1e-8 * max(1.0, float(np.abs(influence).sum())):
        raise RuntimeError("The repeated-cross-section efficient influence score is not centered.")
    count_tuple = (counts[0], counts[1], counts[2], counts[3])
    return float(estimate), influence, count_tuple


def _aggregate_influence(
    cells: list[_RepeatedEffectCell], sample: _RepeatedSample
) -> tuple[float, np.ndarray]:
    n = len(sample.outcome)
    shares = np.array([cell.cohort_share for cell in cells], dtype=float)
    denominator = float(shares.sum())
    weights = shares / denominator
    influence = np.zeros(n)
    centered_membership = np.zeros(n)
    masks: list[np.ndarray] = []
    for cell in cells:
        mask = sample.cohorts == cell.cohort
        masks.append(mask)
        centered_membership += mask.astype(float) - cell.cohort_share
    estimate = float(sum(weight * cell.att for weight, cell in zip(weights, cells, strict=True)))
    for weight, share, mask, cell in zip(weights, shares, masks, cells, strict=True):
        share_influence = (
            mask.astype(float) - share
        ) / denominator - share * centered_membership / denominator**2
        influence += weight * cell.influence + cell.att * share_influence
    return estimate, influence


def _empty_pretrend(
    sample: _RepeatedSample,
    *,
    covariance: DiDCovariance,
    reason: str,
    conditional: bool = False,
    covariates: tuple[str, ...] = (),
    cross_fitted: bool = False,
) -> RepeatedCrossSectionPretrendDiagnostic:
    columns = pd.MultiIndex.from_arrays([[], []], names=["cohort", "time"])
    placebo_effects = pd.DataFrame(
        columns=[
            "event_time",
            "base_period",
            "att",
            "std_err",
            "stat",
            "p_value",
            "n_treated_target",
            "n_treated_base",
            "n_comparison_target",
            "n_comparison_base",
        ],
        index=columns,
    )
    influence = pd.DataFrame(index=sample.index.copy(), columns=columns, dtype=float)
    covariance_frame = pd.DataFrame(index=columns, columns=columns, dtype=float)
    n_clusters = None if sample.clusters is None else len(pd.unique(sample.clusters))
    return RepeatedCrossSectionPretrendDiagnostic(
        available=False,
        reason=reason,
        placebo_effects=placebo_effects,
        influence=influence,
        covariance=covariance_frame,
        statistic=None,
        pvalue=None,
        n_restrictions=0,
        df_num=0,
        df_denom=None,
        distribution=None,
        covariance_type=covariance,
        n_clusters=n_clusters,
        null_hypothesis="Every retained repeated-cross-section placebo effect equals zero.",
        notes=(
            "Failure to reject is not evidence that repeated-cross-section parallel trends is true.",
            "Stationary composition is an identifying assumption, not established by this diagnostic.",
        ),
        conditional=conditional,
        covariates=covariates,
        cross_fitted=cross_fitted,
    )


def _pretrend_diagnostic(
    sample: _RepeatedSample,
    *,
    covariance: DiDCovariance,
    control_group: ControlGroup,
) -> RepeatedCrossSectionPretrendDiagnostic:
    cells: list[_RepeatedEffectCell] = []
    for cohort in sample.treated_cohorts:
        effective_position = sample.effective_positions[cohort]
        for target_position in range(1, effective_position):
            base_position = target_position - 1
            estimate, influence, counts, comparison_cohorts = _effect_from_four_cells(
                sample,
                cohort=cohort,
                target_position=target_position,
                base_position=base_position,
                control_group=control_group,
            )
            cells.append(
                _RepeatedEffectCell(
                    cohort=cohort,
                    time=float(sample.times[target_position]),
                    event_time=target_position - sample.original_positions[cohort],
                    base_period=float(sample.times[base_position]),
                    att=estimate,
                    influence=influence,
                    n_treated_target=counts[0],
                    n_treated_base=counts[1],
                    n_comparison_target=counts[2],
                    n_comparison_base=counts[3],
                    comparison_cohorts=comparison_cohorts,
                    cohort_share=float(np.mean(sample.cohorts == cohort)),
                )
            )
    return _pretrend_diagnostic_from_cells(
        sample,
        cells,
        covariance=covariance,
        conditional=False,
        covariates=(),
        cross_fitted=False,
    )


def _pretrend_diagnostic_from_cells(
    sample: _RepeatedSample,
    cells: list[_RepeatedEffectCell],
    *,
    covariance: DiDCovariance,
    conditional: bool,
    covariates: tuple[str, ...],
    cross_fitted: bool,
) -> RepeatedCrossSectionPretrendDiagnostic:
    if not cells:
        return _empty_pretrend(
            sample,
            covariance=covariance,
            reason=(
                "No uncontaminated adjacent repeated-cross-section changes remain before "
                "the declared treatment or anticipation boundary."
            ),
            conditional=conditional,
            covariates=covariates,
            cross_fitted=cross_fitted,
        )

    rows: list[dict[str, Any]] = []
    keys: list[tuple[float, float]] = []
    influences: list[np.ndarray] = []
    for cell in cells:
        keys.append((cell.cohort, cell.time))
        influences.append(cell.influence)
        rows.append(
            {
                "cohort": cell.cohort,
                "time": cell.time,
                "event_time": cell.event_time,
                "base_period": cell.base_period,
                **_summary_row_rcs(
                    sample,
                    cell.att,
                    cell.influence,
                    covariance=covariance,
                ),
                "n_treated_target": cell.n_treated_target,
                "n_treated_base": cell.n_treated_base,
                "n_comparison_target": cell.n_comparison_target,
                "n_comparison_base": cell.n_comparison_base,
                "comparison_cohorts": cell.comparison_cohorts,
            }
        )
    columns = pd.MultiIndex.from_tuples(keys, names=["cohort", "time"])
    influence_matrix = np.column_stack(influences)
    influence_frame = pd.DataFrame(influence_matrix, index=sample.index.copy(), columns=columns)
    covariance_matrix, n_clusters = _score_cross_covariance_rcs(
        sample,
        influence_matrix,
        influence_matrix,
        covariance=covariance,
    )
    covariance_frame = pd.DataFrame(covariance_matrix, index=columns.copy(), columns=columns.copy())
    placebo_effects = pd.DataFrame(rows).set_index(["cohort", "time"])
    estimates = placebo_effects["att"].to_numpy(dtype=float)
    try:
        statistic, pvalue, distribution, denominator_df = _joint_wald(
            estimates,
            covariance_matrix,
            covariance=covariance,
            n_clusters=n_clusters,
        )
    except ValueError as error:
        return RepeatedCrossSectionPretrendDiagnostic(
            available=False,
            reason=str(error),
            placebo_effects=placebo_effects,
            influence=influence_frame,
            covariance=covariance_frame,
            statistic=None,
            pvalue=None,
            n_restrictions=len(keys),
            df_num=len(keys),
            df_denom=None,
            distribution=None,
            covariance_type=covariance,
            n_clusters=n_clusters,
            null_hypothesis="Every retained repeated-cross-section placebo effect equals zero.",
            notes=(
                "Placebo effects remain available, but their joint covariance was singular.",
                "No ridge, dimension dropping, or pseudoinverse was applied.",
            ),
            conditional=conditional,
            covariates=covariates,
            cross_fitted=cross_fitted,
        )
    return RepeatedCrossSectionPretrendDiagnostic(
        available=True,
        reason=None,
        placebo_effects=placebo_effects,
        influence=influence_frame,
        covariance=covariance_frame,
        statistic=statistic,
        pvalue=pvalue,
        n_restrictions=len(keys),
        df_num=len(keys),
        df_denom=denominator_df,
        distribution=distribution,
        covariance_type=covariance,
        n_clusters=n_clusters,
        null_hypothesis="Every retained repeated-cross-section placebo effect equals zero.",
        notes=(
            (
                "Placebos use the same cross-fitted locally efficient conditional score as "
                "reported group-time effects."
                if conditional
                else "Placebos use four independent adjacent cohort-period cell means."
            ),
            "Failure to reject is not evidence that repeated-cross-section parallel trends is true.",
            "Stationary composition is an identifying assumption, not established by this diagnostic.",
        ),
        conditional=conditional,
        covariates=covariates,
        cross_fitted=cross_fitted,
    )


def _assemble_result(
    cells: list[_RepeatedEffectCell],
    sample: _RepeatedSample,
    *,
    covariance: DiDCovariance,
    control_group: ControlGroup,
    composition: RCSComposition,
    anticipation: int,
    outcome_name: str,
    time_name: str,
    treatment_time_name: str,
    cluster_name: str | None,
    pretrend_override: RepeatedCrossSectionPretrendDiagnostic | None = None,
    covariate_names: tuple[str, ...] = (),
    nuisance_predictions: pd.DataFrame | None = None,
    nuisance_fold: pd.Series | None = None,
    nuisance_diagnostics: pd.DataFrame | None = None,
    n_splits: int | None = None,
    nuisance_probability_floor: float | None = None,
    design_fingerprint: str | None = None,
) -> RepeatedCrossSectionDiDResult:
    group_rows: list[dict[str, Any]] = []
    group_influences: dict[tuple[float, float], np.ndarray] = {}
    for cell in cells:
        key = (cell.cohort, cell.time)
        group_influences[key] = cell.influence
        group_rows.append(
            {
                "cohort": cell.cohort,
                "time": cell.time,
                "event_time": cell.event_time,
                "base_period": cell.base_period,
                **_summary_row_rcs(
                    sample,
                    cell.att,
                    cell.influence,
                    covariance=covariance,
                ),
                "n_treated_target": cell.n_treated_target,
                "n_treated_base": cell.n_treated_base,
                "n_comparison_target": cell.n_comparison_target,
                "n_comparison_base": cell.n_comparison_base,
                "comparison_cohorts": cell.comparison_cohorts,
                "cohort_share": cell.cohort_share,
            }
        )
    group_time = pd.DataFrame(group_rows).set_index(["cohort", "time"]).sort_index()
    group_time_influence = pd.DataFrame(group_influences, index=sample.index.copy()).sort_index(
        axis=1
    )
    group_time_influence.columns.names = ["cohort", "time"]

    event_rows: list[dict[str, Any]] = []
    event_influences: dict[int, np.ndarray] = {}
    for event_time in sorted({cell.event_time for cell in cells}):
        selected = [cell for cell in cells if cell.event_time == event_time]
        estimate, influence = _aggregate_influence(selected, sample)
        event_influences[event_time] = influence
        event_rows.append(
            {
                "event_time": event_time,
                **_summary_row_rcs(sample, estimate, influence, covariance=covariance),
                "n_cohorts": len(selected),
            }
        )
    event_study = pd.DataFrame(event_rows).set_index("event_time").sort_index()
    event_study_influence = pd.DataFrame(event_influences, index=sample.index.copy()).sort_index(
        axis=1
    )
    event_study_influence.columns.name = "event_time"

    calendar_rows: list[dict[str, Any]] = []
    calendar_influences: dict[float, np.ndarray] = {}
    for calendar_value in sorted({cell.time for cell in cells if cell.event_time >= 0}):
        selected = [cell for cell in cells if cell.time == calendar_value and cell.event_time >= 0]
        estimate, influence = _aggregate_influence(selected, sample)
        calendar_influences[calendar_value] = influence
        calendar_rows.append(
            {
                "time": calendar_value,
                **_summary_row_rcs(sample, estimate, influence, covariance=covariance),
                "n_cohorts": len(selected),
            }
        )
    calendar_time = pd.DataFrame(calendar_rows).set_index("time").sort_index()
    calendar_time_influence = pd.DataFrame(
        calendar_influences, index=sample.index.copy()
    ).sort_index(axis=1)
    calendar_time_influence.columns.name = "time"

    post_events = [value for value in event_study.index if value >= 0]
    estimate = float(event_study.loc[post_events, "att"].mean())
    overall_array = event_study_influence.loc[:, post_events].mean(axis=1).to_numpy(dtype=float)
    standard_error, statistic, pvalue, distribution, inference_df, n_clusters = (
        _score_statistics_rcs(sample, estimate, overall_array, covariance=covariance)
    )
    overall_influence = pd.Series(overall_array, index=sample.index.copy(), name="esavg_influence")
    cohort_sizes = pd.Series(
        {
            cohort: int(np.count_nonzero(sample.cohorts == cohort))
            for cohort in sample.treated_cohorts
        },
        name="n_observations",
    )
    cohort_sizes.index.name = "cohort"
    pretrend = (
        _pretrend_diagnostic(sample, covariance=covariance, control_group=control_group)
        if pretrend_override is None
        else pretrend_override
    )
    inference_clusters = (
        pd.Series(dtype="object", name="cluster")
        if sample.clusters is None
        else pd.Series(sample.clusters.copy(), index=sample.index.copy(), name="cluster")
    )
    if covariate_names:
        parallel_trends = (
            "conditional_repeated_cross_section_post_stationary_composition"
            if control_group == "never_treated"
            else "conditional_repeated_cross_section_post_not_yet_treated_stationary_composition"
        )
        method = "repeated_cross_section_doubly_robust_group_time"
    else:
        parallel_trends = (
            "repeated_cross_section_post_stationary_composition"
            if control_group == "never_treated"
            else "repeated_cross_section_post_not_yet_treated_stationary_composition"
        )
        method = "repeated_cross_section_group_time"
    return RepeatedCrossSectionDiDResult(
        estimate=estimate,
        standard_error=standard_error,
        statistic=statistic,
        pvalue=pvalue,
        group_time=group_time,
        event_study=event_study,
        calendar_time=calendar_time,
        group_time_influence=group_time_influence,
        event_study_influence=event_study_influence,
        calendar_time_influence=calendar_time_influence,
        overall_influence=overall_influence,
        pretrend=pretrend,
        cell_counts=sample.cell_counts.copy(),
        cohort_sizes=cohort_sizes,
        n_observations=len(sample.outcome),
        n_periods=len(sample.times),
        n_clusters=n_clusters,
        covariance_type=covariance,
        inference_distribution=distribution,
        inference_df=inference_df,
        inference_method="analytic",
        method=method,
        parallel_trends=parallel_trends,
        control_group=control_group,
        composition=composition,
        anticipation=anticipation,
        pre_periods=1,
        sampling_unit="observation" if sample.clusters is None else "cluster",
        time_name=time_name,
        outcome_name=outcome_name,
        treatment_time_name=treatment_time_name,
        cluster_name=cluster_name,
        covariates=covariate_names,
        cross_fitted=bool(covariate_names),
        estimation_index=sample.index.copy(),
        inference_clusters=inference_clusters,
        times=tuple(float(value) for value in sample.times),
        design_fingerprint=design_fingerprint or sample.design_fingerprint,
        assumptions=(
            "Rows are independent repeated-cross-section observations unless PSUs are declared",
            "Stationary composition of the relevant treatment-cohort populations across periods",
            "Absorbing conceptual treatment at the declared first-treatment time",
            "No anticipation outside the declared anticipation window",
            "Consistency and no interference",
            (
                "Strict conditional propensity overlap in every reported fixed-comparison pair"
                if covariate_names
                else "Overlap in every reported treated and fixed-comparison group-period cell"
            ),
            f"Parallel trends contract: {parallel_trends}",
            *(
                (
                    "Cross-fitted nuisance consistency and product-rate conditions for the "
                    "locally efficient repeated-cross-section score",
                )
                if covariate_names
                else ()
            ),
        ),
        notes=(
            (
                "Each group-time effect uses the cross-fitted locally efficient, doubly "
                "robust repeated-cross-section score."
                if covariate_names
                else "Each group-time effect uses four repeated-cross-section cell means."
            ),
            "The comparison cohort-membership rule is held fixed at target and baseline.",
            "Cohort-share aggregation includes estimated pooled-share influence terms.",
            "Stationary composition is declared, not tested or verified.",
            *(
                (
                    "All reported propensity and four group-period outcome-regression "
                    "predictions are out of fold; overlap violations are refused without clipping.",
                )
                if covariate_names
                else ()
            ),
            "Reported confidence intervals are pointwise; simultaneous bands are not implemented on this surface.",
        ),
        nuisance_predictions=(
            pd.DataFrame() if nuisance_predictions is None else nuisance_predictions.copy()
        ),
        nuisance_fold=(
            pd.Series(dtype="int64", name="fold") if nuisance_fold is None else nuisance_fold.copy()
        ),
        nuisance_diagnostics=(
            pd.DataFrame() if nuisance_diagnostics is None else nuisance_diagnostics.copy()
        ),
        n_splits=n_splits,
        nuisance_probability_floor=nuisance_probability_floor,
    )


def _fit_covariate_repeated_cross_section(
    data: pd.DataFrame,
    sample: _RepeatedSample,
    *,
    outcome: str,
    time: str,
    treatment_time: str,
    cluster: str | None,
    covariates: Sequence[str],
    cross_fitter: CrossFitter,
    control_group: ControlGroup,
    composition: RCSComposition,
    anticipation: int,
    covariance: DiDCovariance,
    probability_floor: float,
) -> RepeatedCrossSectionDiDResult:
    if cross_fitter.propensity_factory is None:
        raise ValueError(
            "cross_fitter must provide a propensity_factory for repeated-cross-section DiD."
        )
    if cross_fitter.outcome_factory is None:
        raise ValueError(
            "cross_fitter must provide an outcome_factory for repeated-cross-section DiD."
        )
    X, covariate_names = _prepare_repeated_covariates(
        data,
        sample,
        covariates=covariates,
        protected_names=(outcome, time, treatment_time, cluster),
    )
    nuisance_order = (
        "propensity",
        "outcome_control_pre",
        "outcome_control_post",
        "outcome_treated_pre",
        "outcome_treated_post",
    )
    comparisons: list[_CovariateComparison] = []
    tasks: list[CrossFitTask] = []
    task_metadata: dict[str, tuple[float, float, str, str]] = {}
    task_relevant_masks: dict[str, np.ndarray] = {}
    outcome_target = pd.Series(sample.outcome, index=sample.index.copy(), name="outcome")
    for cohort in sample.treated_cohorts:
        effective_position = sample.effective_positions[cohort]
        for target_position in range(1, len(sample.times)):
            if target_position < effective_position:
                stage: Literal["group_time", "conditional_pretrend"] = "conditional_pretrend"
                base_position = target_position - 1
            else:
                stage = "group_time"
                base_position = effective_position - 1
            comparison_mask = _comparison_membership(
                sample,
                cohort=cohort,
                target_position=target_position,
                control_group=control_group,
            )
            cohort_mask = sample.cohorts == cohort
            target_mask = sample.time_positions == target_position
            base_mask = sample.time_positions == base_position
            pair_mask = (cohort_mask | comparison_mask) & (target_mask | base_mask)
            public_time = float(sample.times[target_position])
            task_names = {
                nuisance: f"rcs_{len(comparisons)}_{nuisance}" for nuisance in nuisance_order
            }
            comparison_cohorts = tuple(
                float(value) for value in np.unique(sample.cohorts[comparison_mask])
            )
            comparison = _CovariateComparison(
                cohort=cohort,
                target_position=target_position,
                base_position=base_position,
                stage=stage,
                comparison_mask=comparison_mask,
                comparison_cohorts=comparison_cohorts,
                task_names=task_names,
            )
            comparisons.append(comparison)
            propensity_target = pd.Series(
                cohort_mask.astype(float),
                index=sample.index.copy(),
                name=task_names["propensity"],
            )
            propensity_predict = cross_fitter.propensity_predict or _treated_probability_prediction
            tasks.append(
                CrossFitTask(
                    name=task_names["propensity"],
                    target=propensity_target,
                    train_mask=pd.Series(pair_mask, index=sample.index.copy()),
                    factory=cross_fitter.propensity_factory,
                    predict=propensity_predict,
                )
            )
            outcome_masks = {
                "outcome_control_pre": comparison_mask & base_mask,
                "outcome_control_post": comparison_mask & target_mask,
                "outcome_treated_pre": cohort_mask & base_mask,
                "outcome_treated_post": cohort_mask & target_mask,
            }
            for nuisance_name, train_mask in outcome_masks.items():
                tasks.append(
                    CrossFitTask(
                        name=task_names[nuisance_name],
                        target=outcome_target,
                        train_mask=pd.Series(train_mask, index=sample.index.copy()),
                    )
                )
            for nuisance_name, task_name in task_names.items():
                task_metadata[task_name] = (cohort, public_time, nuisance_name, stage)
                task_relevant_masks[task_name] = pair_mask.copy()

    cell_pairs = pd.Series(
        pd.factorize(pd.MultiIndex.from_arrays([sample.cohorts, sample.time_positions]))[0],
        index=sample.index.copy(),
        name="cohort_period_cell",
    )
    clusters = (
        None
        if sample.clusters is None
        else pd.Series(sample.clusters, index=sample.index.copy(), name="cluster")
    )
    nuisance_result = cross_fitter.fit_predict_tasks(
        X,
        tasks=tasks,
        strata=cell_pairs,
        clusters=clusters,
    )

    prediction_columns: list[pd.Series] = []
    prediction_keys: list[tuple[float, float, str]] = []
    for comparison in comparisons:
        public_time = float(sample.times[comparison.target_position])
        pair_mask = ((sample.cohorts == comparison.cohort) | comparison.comparison_mask) & (
            (sample.time_positions == comparison.target_position)
            | (sample.time_positions == comparison.base_position)
        )
        for nuisance_name in nuisance_order:
            prediction_columns.append(
                nuisance_result.predictions[comparison.task_names[nuisance_name]].where(pair_mask)
            )
            prediction_keys.append((comparison.cohort, public_time, nuisance_name))
    nuisance_predictions = pd.concat(prediction_columns, axis=1)
    nuisance_predictions.columns = pd.MultiIndex.from_tuples(
        prediction_keys,
        names=["cohort", "time", "nuisance"],
    )
    diagnostic_frame = nuisance_result.model_diagnostics.copy()
    parsed = diagnostic_frame["task"].map(task_metadata)
    diagnostic_frame.insert(0, "cohort", parsed.map(lambda value: value[0]))
    diagnostic_frame.insert(1, "time", parsed.map(lambda value: value[1]))
    diagnostic_frame.insert(2, "nuisance", parsed.map(lambda value: value[2]))
    diagnostic_frame.insert(3, "comparison_stage", parsed.map(lambda value: value[3]))
    relevant_holdout = [
        int(
            np.count_nonzero(
                task_relevant_masks[str(row.task)]
                & (nuisance_result.fold.to_numpy(dtype=int) == int(row.fold))
            )
        )
        for row in diagnostic_frame.itertuples(index=False)
    ]
    diagnostic_frame.insert(
        diagnostic_frame.columns.get_loc("holdout_nobs") + 1,
        "relevant_holdout_nobs",
        relevant_holdout,
    )

    group_cells: list[_RepeatedEffectCell] = []
    placebo_cells: list[_RepeatedEffectCell] = []
    n = len(sample.outcome)
    for comparison in comparisons:
        estimate, influence, counts = _effect_from_cross_fitted_score(
            sample,
            comparison,
            nuisance_result.predictions,
            probability_floor=probability_floor,
        )
        cell = _RepeatedEffectCell(
            cohort=comparison.cohort,
            time=float(sample.times[comparison.target_position]),
            event_time=(comparison.target_position - sample.original_positions[comparison.cohort]),
            base_period=float(sample.times[comparison.base_position]),
            att=estimate,
            influence=influence,
            n_treated_target=counts[0],
            n_treated_base=counts[1],
            n_comparison_target=counts[2],
            n_comparison_base=counts[3],
            comparison_cohorts=comparison.comparison_cohorts,
            cohort_share=float(np.count_nonzero(sample.cohorts == comparison.cohort) / n),
        )
        (group_cells if comparison.stage == "group_time" else placebo_cells).append(cell)

    pretrend = _pretrend_diagnostic_from_cells(
        sample,
        placebo_cells,
        covariance=covariance,
        conditional=True,
        covariates=covariate_names,
        cross_fitted=True,
    )
    return _assemble_result(
        group_cells,
        sample,
        covariance=covariance,
        control_group=control_group,
        composition=composition,
        anticipation=anticipation,
        outcome_name=outcome,
        time_name=time,
        treatment_time_name=treatment_time,
        cluster_name=cluster,
        pretrend_override=pretrend,
        covariate_names=covariate_names,
        nuisance_predictions=nuisance_predictions,
        nuisance_fold=nuisance_result.fold,
        nuisance_diagnostics=diagnostic_frame,
        n_splits=nuisance_result.n_splits,
        nuisance_probability_floor=probability_floor,
        design_fingerprint=_covariate_design_fingerprint(
            sample,
            X,
            n_splits=nuisance_result.n_splits,
            probability_floor=probability_floor,
        ),
    )


class RepeatedCrossSectionDiD:
    """Cohort-time DiD for stationary repeated cross sections.

    Supplying covariates and a provider-neutral :class:`CrossFitter` promotes the
    estimator to the locally efficient doubly robust repeated-cross-section score.
    CauseKit does not own, copy, or silently select nuisance-model implementations.
    """

    control_group: ControlGroup
    composition: RCSComposition
    anticipation: int
    covariance: DiDCovariance
    inference: RCSInference
    nuisance_probability_floor: float

    def __init__(
        self,
        *,
        control_group: ControlGroup = "never_treated",
        composition: RCSComposition = "stationary",
        anticipation: int = 0,
        covariance: DiDCovariance = "robust",
        inference: RCSInference = "analytic",
        nuisance_probability_floor: float = 1e-6,
    ) -> None:
        if control_group not in {"never_treated", "not_yet_treated"}:
            raise ValueError("control_group must be 'never_treated' or 'not_yet_treated'.")
        if composition != "stationary":
            raise NotImplementedError(
                "composition must be 'stationary'; compositional-change-robust DiD requires "
                "a separately validated score."
            )
        if not isinstance(anticipation, Integral) or isinstance(anticipation, (bool, np.bool_)):
            raise TypeError("anticipation must be a non-negative integer.")
        if anticipation < 0:
            raise ValueError("anticipation must be a non-negative integer.")
        if covariance not in {"robust", "clustered"}:
            raise ValueError("covariance must be 'robust' or 'clustered'.")
        if inference != "analytic":
            raise NotImplementedError(
                "inference must be 'analytic'; observation/PSU multiplier bands require "
                "a separate promotion gate."
            )
        if not isinstance(nuisance_probability_floor, Real) or isinstance(
            nuisance_probability_floor, (bool, np.bool_)
        ):
            raise TypeError("nuisance_probability_floor must be a finite real number.")
        if (
            not np.isfinite(nuisance_probability_floor)
            or not 0.0 < float(nuisance_probability_floor) < 0.5
        ):
            raise ValueError(
                "nuisance_probability_floor must be finite and strictly between zero and 0.5."
            )
        self.control_group = control_group
        self.composition = composition
        self.anticipation = int(anticipation)
        self.covariance = covariance
        self.inference = inference
        self.nuisance_probability_floor = float(nuisance_probability_floor)

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        time: str,
        treatment_time: str,
        never_treated: float = np.inf,
        cluster: str | None = None,
        covariates: Sequence[str] | None = None,
        cross_fitter: CrossFitter | None = None,
        sampling_weights: str | Sequence[float] | None = None,
    ) -> RepeatedCrossSectionDiDResult:
        """Estimate repeated-cross-section cohort-time effects and aggregations."""

        if sampling_weights is not None:
            raise NotImplementedError(
                "sampling weights and survey designs are not implemented on this surface."
            )
        if covariates is not None and cross_fitter is None:
            raise ValueError(
                "cross_fitter is required when covariates are supplied; "
                "RepeatedCrossSectionDiD does not own or fit nuisance-model classes."
            )
        if covariates is None and cross_fitter is not None:
            raise ValueError("cross_fitter is only used when covariates are supplied.")
        sample = _prepare_repeated_sample(
            data,
            outcome=outcome,
            time=time,
            treatment_time=treatment_time,
            never_treated=never_treated,
            anticipation=self.anticipation,
            covariance=self.covariance,
            cluster=cluster,
        )
        if covariates is not None:
            if cross_fitter is None:  # defensive narrowing after the public refusal above
                raise ValueError("cross_fitter is required for covariate adjustment.")
            return _fit_covariate_repeated_cross_section(
                data,
                sample,
                outcome=outcome,
                time=time,
                treatment_time=treatment_time,
                cluster=cluster,
                covariates=covariates,
                cross_fitter=cross_fitter,
                control_group=self.control_group,
                composition=self.composition,
                anticipation=self.anticipation,
                covariance=self.covariance,
                probability_floor=self.nuisance_probability_floor,
            )
        cells: list[_RepeatedEffectCell] = []
        n = len(sample.outcome)
        for cohort in sample.treated_cohorts:
            effective_position = sample.effective_positions[cohort]
            base_position = effective_position - 1
            cohort_share = float(np.count_nonzero(sample.cohorts == cohort) / n)
            for target_position in range(effective_position, len(sample.times)):
                estimate, influence, counts, comparison_cohorts = _effect_from_four_cells(
                    sample,
                    cohort=cohort,
                    target_position=target_position,
                    base_position=base_position,
                    control_group=self.control_group,
                )
                cells.append(
                    _RepeatedEffectCell(
                        cohort=cohort,
                        time=float(sample.times[target_position]),
                        event_time=target_position - sample.original_positions[cohort],
                        base_period=float(sample.times[base_position]),
                        att=estimate,
                        influence=influence,
                        n_treated_target=counts[0],
                        n_treated_base=counts[1],
                        n_comparison_target=counts[2],
                        n_comparison_base=counts[3],
                        comparison_cohorts=comparison_cohorts,
                        cohort_share=cohort_share,
                    )
                )
        return _assemble_result(
            cells,
            sample,
            covariance=self.covariance,
            control_group=self.control_group,
            composition=self.composition,
            anticipation=self.anticipation,
            outcome_name=outcome,
            time_name=time,
            treatment_time_name=treatment_time,
            cluster_name=cluster,
        )


__all__ = [
    "RepeatedCrossSectionDiD",
    "RepeatedCrossSectionDiDResult",
    "RepeatedCrossSectionPretrendDiagnostic",
]

"""Conventional difference-in-differences for repeated cross sections.

This module is deliberately separate from :mod:`causekit.did`.  Rows are sampled
observations, not balanced-panel entities, so cell means, influence functions, and
cluster aggregation are all constructed at the observation/PSU level.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import norm
from scipy.stats import t as student_t

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
    )


def _pretrend_diagnostic(
    sample: _RepeatedSample,
    *,
    covariance: DiDCovariance,
    control_group: ControlGroup,
) -> RepeatedCrossSectionPretrendDiagnostic:
    rows: list[dict[str, Any]] = []
    keys: list[tuple[float, float]] = []
    influences: list[np.ndarray] = []
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
            public_time = float(sample.times[target_position])
            keys.append((cohort, public_time))
            influences.append(influence)
            rows.append(
                {
                    "cohort": cohort,
                    "time": public_time,
                    "event_time": target_position - sample.original_positions[cohort],
                    "base_period": float(sample.times[base_position]),
                    **_summary_row_rcs(
                        sample,
                        estimate,
                        influence,
                        covariance=covariance,
                    ),
                    "n_treated_target": counts[0],
                    "n_treated_base": counts[1],
                    "n_comparison_target": counts[2],
                    "n_comparison_base": counts[3],
                    "comparison_cohorts": comparison_cohorts,
                }
            )
    if not rows:
        return _empty_pretrend(
            sample,
            covariance=covariance,
            reason=(
                "No uncontaminated adjacent repeated-cross-section changes remain before "
                "the declared treatment or anticipation boundary."
            ),
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
            "Placebos use four independent adjacent cohort-period cell means.",
            "Failure to reject is not evidence that repeated-cross-section parallel trends is true.",
            "Stationary composition is an identifying assumption, not established by this diagnostic.",
        ),
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
    pretrend = _pretrend_diagnostic(sample, covariance=covariance, control_group=control_group)
    inference_clusters = (
        pd.Series(dtype="object", name="cluster")
        if sample.clusters is None
        else pd.Series(sample.clusters.copy(), index=sample.index.copy(), name="cluster")
    )
    parallel_trends = (
        "repeated_cross_section_post_stationary_composition"
        if control_group == "never_treated"
        else "repeated_cross_section_post_not_yet_treated_stationary_composition"
    )
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
        method="repeated_cross_section_group_time",
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
        covariates=(),
        cross_fitted=False,
        estimation_index=sample.index.copy(),
        inference_clusters=inference_clusters,
        times=tuple(float(value) for value in sample.times),
        design_fingerprint=sample.design_fingerprint,
        assumptions=(
            "Rows are independent repeated-cross-section observations unless PSUs are declared",
            "Stationary composition of the relevant treatment-cohort populations across periods",
            "Absorbing conceptual treatment at the declared first-treatment time",
            "No anticipation outside the declared anticipation window",
            "Consistency and no interference",
            "Overlap in every reported treated and fixed-comparison group-period cell",
            f"Parallel trends contract: {parallel_trends}",
        ),
        notes=(
            "Each group-time effect uses four repeated-cross-section cell means.",
            "The comparison cohort-membership rule is held fixed at target and baseline.",
            "Cohort-share aggregation includes estimated pooled-share influence terms.",
            "Stationary composition is declared, not tested or verified.",
            "Reported confidence intervals are pointwise; simultaneous bands are not implemented on this surface.",
        ),
    )


class RepeatedCrossSectionDiD:
    """Conventional cohort-time DiD for stationary repeated cross sections."""

    control_group: ControlGroup
    composition: RCSComposition
    anticipation: int
    covariance: DiDCovariance
    inference: RCSInference

    def __init__(
        self,
        *,
        control_group: ControlGroup = "never_treated",
        composition: RCSComposition = "stationary",
        anticipation: int = 0,
        covariance: DiDCovariance = "robust",
        inference: RCSInference = "analytic",
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
        self.control_group = control_group
        self.composition = composition
        self.anticipation = int(anticipation)
        self.covariance = covariance
        self.inference = inference

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
        sampling_weights: str | Sequence[float] | None = None,
    ) -> RepeatedCrossSectionDiDResult:
        """Estimate repeated-cross-section cohort-time effects and aggregations."""

        if sampling_weights is not None:
            raise NotImplementedError(
                "sampling weights and survey designs are not implemented in this first slice."
            )
        if covariates is not None:
            raise NotImplementedError(
                "The covariate-adjusted repeated-cross-section path is not implemented; "
                "no nuisance model will be fitted or copied implicitly."
            )
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

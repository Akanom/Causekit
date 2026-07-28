"""Contract-first scalar nearest-neighbor matching."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._data import _validate_pandas_indices

MatchingEstimand = Literal["ate", "att", "atc"]


@dataclass(frozen=True)
class NearestNeighborMatchResult:
    """Auditable point estimate from scalar nearest-neighbor matching."""

    estimate: float
    standard_error: float
    statistic: float
    pvalue: float
    requested_estimand: MatchingEstimand
    realized_estimand: str
    target_population: str
    match_table: pd.DataFrame
    analysis_weights: pd.Series
    effect_weights: pd.Series
    comparison_reuse_counts: pd.Series
    realized_match_count_distribution: pd.Series
    balance: pd.DataFrame
    estimation_index: pd.Index
    support_eligible_index: pd.Index
    matched_focal_index: pd.Index
    support_excluded_index: pd.Index
    caliper_unmatched_index: pd.Index
    nobs: int
    n_treated: int
    n_control: int
    n_support_eligible_treated: int
    n_support_eligible_control: int
    n_matched_focal: int
    n_matched_focal_treated: int
    n_matched_focal_control: int
    matched_focal_fraction: float
    n_support_excluded_treated: int
    n_support_excluded_control: int
    n_caliper_unmatched: int
    n_caliper_unmatched_treated: int
    n_caliper_unmatched_control: int
    boundary_tie_events: int
    maximum_tie_multiplicity: int
    maximum_reuse_count: int
    unique_comparison_observations: int
    neighbors: int
    metric: str
    replacement: bool
    ties: str
    common_support: str | None
    support_bounds: tuple[float, float] | None
    requested_caliper: str | float | None
    realized_caliper: float | None
    caliper_scale: str
    caliper_standard_deviation_ddof: int | None
    propensity_provenance: str
    bias_correction: str
    inference: str
    inference_distribution: str | None
    inference_df: float | None
    notes: tuple[str, ...]
    assumptions: tuple[str, ...]
    converged: bool = True
    backend: str = "native-sorted-scalar-matching"

    @property
    def params(self) -> pd.Series:
        return pd.Series({self.realized_estimand: self.estimate}, name="coef")

    @property
    def standard_errors(self) -> pd.Series:
        return pd.Series({self.realized_estimand: self.standard_error}, name="std_err")

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series({self.realized_estimand: self.pvalue}, name="p_value")

    @property
    def causal_interpretation(self) -> str:
        return (
            "The estimate is causal only under consistency, no interference, conditional "
            "exchangeability, positivity in the realized target, and a valid supplied "
            "propensity-score design."
        )

    @property
    def balance_summary(self) -> pd.Series:
        before = self.balance["smd_before"].abs().dropna()
        after = self.balance["smd_after"].abs().dropna()
        return pd.Series(
            {
                "max_abs_smd_before": float(before.max()) if len(before) else np.nan,
                "mean_abs_smd_before": float(before.mean()) if len(before) else np.nan,
                "max_abs_smd_after": float(after.max()) if len(after) else np.nan,
                "mean_abs_smd_after": float(after.mean()) if len(after) else np.nan,
            },
            name="balance_summary",
        )

    def conf_int(self, level: float = 0.95) -> pd.Series:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        if not np.isfinite(self.standard_error):
            raise ValueError("No confidence interval is available when inference='none'.")
        raise NotImplementedError("Analytical matching confidence intervals are not implemented.")

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": [self.estimate],
                "std_err": [self.standard_error],
                "stat": [self.statistic],
                "p_value": [self.pvalue],
            },
            index=pd.Index([self.realized_estimand], dtype="object"),
        )


def _series(value: Any, *, name: str) -> tuple[pd.Series, bool]:
    labeled = isinstance(value, pd.Series)
    if labeled:
        series = value.copy()
    else:
        raw = np.asarray(value)
        if raw.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        series = pd.Series(raw, name=name)
    try:
        series = series.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric values.") from error
    return series, labeled


def _prepare(
    y: Any,
    treatment: Any,
    propensity: Any,
    covariates: Any | None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame | None]:
    outcome, outcome_labeled = _series(y, name="y")
    assigned, treatment_labeled = _series(treatment, name="treatment")
    score, propensity_labeled = _series(propensity, name="propensity")
    nobs = len(outcome)
    if len(assigned) != nobs or len(score) != nobs:
        raise ValueError("All inputs must contain the same number of observations.")
    reference = _validate_pandas_indices(
        [
            ("y", outcome.index if outcome_labeled else None),
            ("treatment", assigned.index if treatment_labeled else None),
            ("propensity", score.index if propensity_labeled else None),
        ]
    )

    frame = None
    if covariates is not None:
        if isinstance(covariates, pd.Series):
            frame = covariates.to_frame()
        elif isinstance(covariates, pd.DataFrame):
            frame = covariates.copy()
        else:
            try:
                raw = np.asarray(covariates)
            except (TypeError, ValueError) as error:
                raise ValueError("covariates must contain only numeric values.") from error
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            if raw.ndim != 2:
                raise ValueError("covariates must be one- or two-dimensional numeric data.")
            frame = pd.DataFrame(raw, columns=[f"covariate_{i}" for i in range(raw.shape[1])])
        if len(frame) != nobs:
            raise ValueError("covariates must contain the same number of observations as y.")
        if isinstance(covariates, (pd.Series, pd.DataFrame)):
            reference = _validate_pandas_indices(
                [("estimation inputs", reference), ("covariates", frame.index)]
            )
        if frame.shape[1] == 0:
            raise ValueError("covariates must contain at least one column.")
        names = tuple(
            "covariate" if name is None and frame.shape[1] == 1 else str(name)
            for name in frame.columns
        )
        if len(set(names)) != len(names):
            raise ValueError("covariate column names must be unique after string conversion.")
        frame.columns = names
        try:
            frame = frame.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("covariates must contain only numeric values.") from error

    if reference is None:
        reference = pd.RangeIndex(nobs)
    outcome.index = reference
    assigned.index = reference
    score.index = reference
    if frame is not None:
        frame.index = reference

    if not np.isfinite(outcome.to_numpy()).all():
        raise ValueError("y must contain only finite values.")
    if not np.isfinite(assigned.to_numpy()).all():
        raise ValueError("treatment must contain only finite values.")
    if not np.isfinite(score.to_numpy()).all():
        raise ValueError("propensity must contain only finite values.")
    if frame is not None and not np.isfinite(frame.to_numpy()).all():
        raise ValueError("covariates must contain only finite values.")
    if not np.array_equal(np.unique(assigned), np.array([0.0, 1.0])):
        raise ValueError("treatment must contain both arms and be coded exactly 0 and 1.")
    if np.any((score <= 0) | (score >= 1)):
        raise ValueError("propensity must lie strictly between zero and one.")
    return outcome, assigned, score, frame


def _neighbor_group(
    focal: float,
    sorted_metric: np.ndarray,
    sorted_positions: np.ndarray,
    *,
    neighbors: int,
    caliper: float | None,
) -> tuple[list[tuple[int, float, int, int, float]], bool] | None:
    """Return local neighbors without constructing an arm-by-arm distance matrix."""

    position = int(np.searchsorted(sorted_metric, focal, side="left"))
    left, right = position - 1, position
    selected: list[tuple[int, float, int, int, float]] = []
    rank = 1
    expanded_tie = False
    tolerance = 32 * np.finfo(float).eps * max(1.0, abs(focal))
    while left >= 0 or right < len(sorted_metric):
        left_distance = focal - sorted_metric[left] if left >= 0 else np.inf
        right_distance = sorted_metric[right] - focal if right < len(sorted_metric) else np.inf
        boundary = float(min(left_distance, right_distance))
        if caliper is not None and boundary > caliper + tolerance:
            break
        group: list[tuple[int, float]] = []
        while left >= 0 and abs((focal - sorted_metric[left]) - boundary) <= tolerance:
            group.append((int(sorted_positions[left]), boundary))
            left -= 1
        while (
            right < len(sorted_metric)
            and abs((sorted_metric[right] - focal) - boundary) <= tolerance
        ):
            group.append((int(sorted_positions[right]), boundary))
            right += 1
        remaining = neighbors - len(selected)
        if len(group) >= remaining:
            weight = remaining / (neighbors * len(group))
            expanded_tie = len(group) > remaining
            selected.extend((item, distance, rank, rank, weight) for item, distance in group)
            break
        selected.extend((item, distance, rank, rank, 1.0 / neighbors) for item, distance in group)
        rank += len(group)
    if sum(item[4] for item in selected) < 1.0 - 1e-12:
        return None
    return selected, expanded_tie


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(weights @ values / weights.sum())


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    return float(weights @ ((values - mean) ** 2) / weights.sum())


def _weighted_ecdf_distance(
    treated: np.ndarray,
    control: np.ndarray,
    treated_weights: np.ndarray,
    control_weights: np.ndarray,
) -> float:
    grid = np.unique(np.concatenate([treated, control]))
    treated_order = np.argsort(treated)
    control_order = np.argsort(control)
    treated_cdf = np.cumsum(treated_weights[treated_order]) / treated_weights.sum()
    control_cdf = np.cumsum(control_weights[control_order]) / control_weights.sum()
    treated_positions = np.searchsorted(treated[treated_order], grid, side="right") - 1
    control_positions = np.searchsorted(control[control_order], grid, side="right") - 1
    treated_at = np.zeros(len(grid))
    control_at = np.zeros(len(grid))
    treated_valid = treated_positions >= 0
    control_valid = control_positions >= 0
    treated_at[treated_valid] = treated_cdf[treated_positions[treated_valid]]
    control_at[control_valid] = control_cdf[control_positions[control_valid]]
    return float(np.max(np.abs(treated_at - control_at)))


def _balance(
    covariates: pd.DataFrame | None,
    treatment: np.ndarray,
    post_weights: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "covariate",
        "treated_mean_before",
        "control_mean_before",
        "smd_before",
        "treated_mean_after",
        "control_mean_after",
        "smd_after",
        "variance_ratio_after",
        "ecdf_max_after",
        "status",
    ]
    if covariates is None:
        return pd.DataFrame(columns=columns).set_index("covariate")
    treated_mask = treatment == 1
    control_mask = ~treated_mask
    records = []
    for name in covariates.columns:
        values = covariates[name].to_numpy(dtype=float)
        treated = values[treated_mask]
        control = values[control_mask]
        before_t = float(np.mean(treated))
        before_c = float(np.mean(control))
        treated_sample_variance = float(np.var(treated, ddof=1)) if len(treated) > 1 else np.nan
        control_sample_variance = float(np.var(control, ddof=1)) if len(control) > 1 else np.nan
        pooled = float(np.sqrt((treated_sample_variance + control_sample_variance) / 2))
        wt = post_weights[treated_mask]
        wc = post_weights[control_mask]
        after_t = _weighted_mean(treated, wt)
        after_c = _weighted_mean(control, wc)
        variance_t = _weighted_variance(treated, wt)
        variance_c = _weighted_variance(control, wc)
        insufficient = len(treated) < 2 or len(control) < 2
        constant = not insufficient and pooled <= 0
        records.append(
            {
                "covariate": str(name),
                "treated_mean_before": before_t,
                "control_mean_before": before_c,
                "smd_before": (
                    (before_t - before_c) / pooled if not insufficient and not constant else np.nan
                ),
                "treated_mean_after": after_t,
                "control_mean_after": after_c,
                "smd_after": (
                    (after_t - after_c) / pooled if not insufficient and not constant else np.nan
                ),
                "variance_ratio_after": variance_t / variance_c if variance_c > 0 else np.nan,
                "ecdf_max_after": _weighted_ecdf_distance(treated, control, wt, wc),
                "status": "insufficient" if insufficient else "constant" if constant else "ok",
            }
        )
    return pd.DataFrame.from_records(records).set_index("covariate")


class NearestNeighborMatch:
    """Scalar propensity-logit nearest-neighbor matching with replacement."""

    def __init__(
        self,
        *,
        estimand: MatchingEstimand = "att",
        metric: str = "propensity_logit",
        neighbors: int = 1,
        replacement: bool = True,
        caliper: str | float | None = "auto",
        common_support: str | None = "intersection",
        ties: str = "all",
        bias_correction: str = "none",
        inference: str = "none",
    ) -> None:
        if estimand not in {"ate", "att", "atc"}:
            raise ValueError("estimand must be 'ate', 'att', or 'atc'.")
        if metric != "propensity_logit":
            raise NotImplementedError("Only metric='propensity_logit' is implemented.")
        if (
            isinstance(neighbors, bool)
            or not isinstance(neighbors, (int, np.integer))
            or neighbors < 1
        ):
            raise ValueError("neighbors must be a positive integer.")
        if replacement is not True:
            raise NotImplementedError("replacement=False is not implemented.")
        if ties != "all":
            raise ValueError("ties must be 'all'.")
        if bias_correction != "none":
            raise NotImplementedError("bias_correction other than 'none' is not implemented.")
        if inference not in {"abadie_imbens", "none"}:
            raise ValueError("inference must be 'abadie_imbens' or 'none'.")
        if common_support not in {"intersection", None}:
            raise ValueError("common_support must be 'intersection' or None.")
        if isinstance(caliper, str) and caliper != "auto":
            raise ValueError("caliper must be 'auto', None, or a positive finite number.")
        if (
            caliper is not None
            and not isinstance(caliper, str)
            and (
                isinstance(caliper, bool)
                or not isinstance(caliper, Real)
                or not np.isfinite(float(caliper))
                or caliper <= 0
            )
        ):
            raise ValueError("caliper must be 'auto', None, or a positive finite number.")
        self.estimand = estimand
        self.metric = metric
        self.neighbors = int(neighbors)
        self.replacement = replacement
        self.caliper = (
            float(caliper)
            if isinstance(caliper, Real) and not isinstance(caliper, bool)
            else caliper
        )
        self.common_support = common_support
        self.ties = ties
        self.bias_correction = bias_correction
        self.inference = inference

    def fit(
        self,
        y: Any,
        *,
        treatment: Any,
        propensity: Any,
        covariates: Any | None = None,
        propensity_provenance: str = "supplied_unspecified",
        clusters: Any | None = None,
    ) -> NearestNeighborMatchResult:
        if clusters is not None:
            raise NotImplementedError("clustered matching inference is not implemented.")
        if not isinstance(propensity_provenance, str) or not propensity_provenance.strip():
            raise ValueError("propensity_provenance must be a nonempty string.")
        outcome, assigned, propensity_values, covariate_frame = _prepare(
            y, treatment, propensity, covariates
        )
        index = outcome.index
        treatment_values = assigned.to_numpy(dtype=int)
        metric_values = np.log(propensity_values.to_numpy() / (1 - propensity_values.to_numpy()))
        nobs = len(outcome)
        if self.caliper == "auto":
            realized_caliper = 0.2 * float(np.std(metric_values, ddof=1))
        elif self.caliper is None:
            realized_caliper = None
        else:
            realized_caliper = float(self.caliper)

        support_bounds = None
        support_eligible = np.ones(nobs, dtype=bool)
        if self.common_support == "intersection":
            treated_metric = metric_values[treatment_values == 1]
            control_metric = metric_values[treatment_values == 0]
            lower = float(max(treated_metric.min(), control_metric.min()))
            upper = float(min(treated_metric.max(), control_metric.max()))
            if lower > upper:
                raise ValueError("Treatment arms have no intersecting common support.")
            support_bounds = (lower, upper)
            tolerance = 32 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
            support_eligible = (metric_values >= lower - tolerance) & (
                metric_values <= upper + tolerance
            )
        support_excluded = ~support_eligible

        focal_arms: tuple[int, ...]
        if self.estimand == "att":
            focal_arms = (1,)
        elif self.estimand == "atc":
            focal_arms = (0,)
        else:
            focal_arms = (1, 0)

        rows: list[dict[str, Any]] = []
        matched_focal_positions: list[int] = []
        caliper_unmatched_positions: list[int] = []
        unit_effects: list[float] = []
        boundary_tie_events = 0
        outcome_values = outcome.to_numpy(dtype=float)
        for focal_arm in focal_arms:
            focal_positions = np.flatnonzero(support_eligible & (treatment_values == focal_arm))
            candidate_positions = np.flatnonzero(
                support_eligible & (treatment_values == 1 - focal_arm)
            )
            if len(candidate_positions) < self.neighbors:
                raise ValueError(
                    "neighbors exceeds the available opposite-arm candidates after support restriction."
                )
            order = np.argsort(metric_values[candidate_positions], kind="stable")
            sorted_positions = candidate_positions[order]
            sorted_metric = metric_values[sorted_positions]
            for focal_position in focal_positions:
                found = _neighbor_group(
                    metric_values[focal_position],
                    sorted_metric,
                    sorted_positions,
                    neighbors=self.neighbors,
                    caliper=realized_caliper,
                )
                if found is None:
                    caliper_unmatched_positions.append(int(focal_position))
                    continue
                matches, expanded = found
                matched_focal_positions.append(int(focal_position))
                boundary_tie_events += int(expanded)
                matched_outcome = sum(
                    outcome_values[comparison] * weight for comparison, _, _, _, weight in matches
                )
                unit_effects.append(
                    outcome_values[focal_position] - matched_outcome
                    if focal_arm == 1
                    else matched_outcome - outcome_values[focal_position]
                )
                for comparison, distance, rank, tie_group, weight in matches:
                    rows.append(
                        {
                            "focal_index": index[focal_position],
                            "comparison_index": index[comparison],
                            "focal_treatment": focal_arm,
                            "distance": distance,
                            "rank": rank,
                            "tie_group": tie_group,
                            "match_weight": weight,
                            "focal_position": focal_position,
                            "comparison_position": comparison,
                        }
                    )
        if not unit_effects:
            raise ValueError(
                "No focal observations have the requested number of caliper-eligible matches."
            )

        match_table_internal = pd.DataFrame.from_records(rows)
        estimate = float(np.mean(unit_effects))
        effect_weights = np.zeros(nobs)
        analysis_weights = np.zeros(nobs)
        focal_scale = 1.0 / len(unit_effects)
        for focal in matched_focal_positions:
            focal_arm = treatment_values[focal]
            effect_weights[focal] += focal_scale * (1 if focal_arm == 1 else -1)
            analysis_weights[focal] += focal_scale
        for row in rows:
            comparison = int(row["comparison_position"])
            match_weight = float(row["match_weight"])
            focal_arm = int(row["focal_treatment"])
            effect_weights[comparison] += focal_scale * match_weight * (-1 if focal_arm == 1 else 1)
            analysis_weights[comparison] += focal_scale * match_weight
        for arm in (0, 1):
            arm_mask = treatment_values == arm
            arm_total = analysis_weights[arm_mask].sum()
            if arm_total > 0:
                analysis_weights[arm_mask] /= arm_total

        reuse_counts = match_table_internal.groupby("comparison_position").size()
        match_counts = match_table_internal.groupby("focal_position").size()
        match_count_distribution = (
            match_counts.value_counts().sort_index().rename_axis("realized_matches")
        )
        match_count_distribution.name = "focal_units"
        tie_multiplicities = match_table_internal.groupby(["focal_position", "tie_group"]).size()
        maximum_tie_multiplicity = int(tie_multiplicities.max())
        comparison_reuse = pd.Series(0, index=index, dtype=int, name="reuse_count")
        for position, count in reuse_counts.items():
            comparison_reuse.iloc[int(position)] = int(count)
        if self.inference == "abadie_imbens":
            if boundary_tie_events:
                raise ValueError(
                    "Abadie-Imbens analytical inference is unavailable when boundary ties "
                    "expand the realized neighbor count; use inference='none'."
                )
            raise NotImplementedError(
                "Abadie-Imbens analytical variance and the adjustment for supplied "
                "propensity-score estimation are not implemented; use inference='none' "
                "for the audited point-estimation slice."
            )

        focal_target = np.isin(treatment_values, focal_arms)
        changed_target = bool(
            np.any(support_excluded & focal_target) or caliper_unmatched_positions
        )
        realized_estimand = f"{self.estimand}_matched_support" if changed_target else self.estimand
        if changed_target:
            target_population = (
                f"{self.estimand.upper()} for focal observations retained after the declared "
                "support and caliper rules."
            )
        elif self.estimand == "att":
            target_population = "ATT for all supplied treated observations."
        elif self.estimand == "atc":
            target_population = "ATC for all supplied control observations."
        else:
            target_population = "ATE for all supplied observations."
        notes: list[str] = []
        if self.caliper is None:
            notes.append("No caliper was imposed; arbitrarily distant matches were permitted.")
        if self.common_support is None:
            notes.append("No common-support restriction was imposed.")
        if changed_target:
            notes.append("Support or caliper exclusions changed the realized target population.")
        match_table = match_table_internal.copy()
        matched_focal_mask = np.isin(np.arange(nobs), matched_focal_positions)
        caliper_unmatched_mask = np.isin(np.arange(nobs), caliper_unmatched_positions)
        original_focal_count = int(focal_target.sum())
        return NearestNeighborMatchResult(
            estimate=estimate,
            standard_error=float("nan"),
            statistic=float("nan"),
            pvalue=float("nan"),
            requested_estimand=self.estimand,
            realized_estimand=realized_estimand,
            target_population=target_population,
            match_table=match_table,
            analysis_weights=pd.Series(analysis_weights, index=index, name="analysis_weight"),
            effect_weights=pd.Series(effect_weights, index=index, name="effect_weight"),
            comparison_reuse_counts=comparison_reuse,
            realized_match_count_distribution=match_count_distribution,
            balance=_balance(covariate_frame, treatment_values, analysis_weights),
            estimation_index=index.copy(),
            support_eligible_index=index[support_eligible],
            matched_focal_index=index[matched_focal_mask],
            support_excluded_index=index[support_excluded],
            caliper_unmatched_index=index[caliper_unmatched_mask],
            nobs=nobs,
            n_treated=int(treatment_values.sum()),
            n_control=int((1 - treatment_values).sum()),
            n_support_eligible_treated=int(np.sum(support_eligible & (treatment_values == 1))),
            n_support_eligible_control=int(np.sum(support_eligible & (treatment_values == 0))),
            n_matched_focal=len(unit_effects),
            n_matched_focal_treated=int(np.sum(matched_focal_mask & (treatment_values == 1))),
            n_matched_focal_control=int(np.sum(matched_focal_mask & (treatment_values == 0))),
            matched_focal_fraction=len(unit_effects) / original_focal_count,
            n_support_excluded_treated=int(np.sum(support_excluded & (treatment_values == 1))),
            n_support_excluded_control=int(np.sum(support_excluded & (treatment_values == 0))),
            n_caliper_unmatched=len(caliper_unmatched_positions),
            n_caliper_unmatched_treated=int(
                np.sum(caliper_unmatched_mask & (treatment_values == 1))
            ),
            n_caliper_unmatched_control=int(
                np.sum(caliper_unmatched_mask & (treatment_values == 0))
            ),
            boundary_tie_events=boundary_tie_events,
            maximum_tie_multiplicity=maximum_tie_multiplicity,
            maximum_reuse_count=int(comparison_reuse.max()),
            unique_comparison_observations=int((comparison_reuse > 0).sum()),
            neighbors=self.neighbors,
            metric=self.metric,
            replacement=self.replacement,
            ties=self.ties,
            common_support=self.common_support,
            support_bounds=support_bounds,
            requested_caliper=self.caliper,
            realized_caliper=realized_caliper,
            caliper_scale=self.metric,
            caliper_standard_deviation_ddof=1 if self.caliper == "auto" else None,
            propensity_provenance=propensity_provenance.strip(),
            bias_correction=self.bias_correction,
            inference=self.inference,
            inference_distribution=None,
            inference_df=None,
            notes=tuple(notes),
            assumptions=(
                "Treatment consistency",
                "No interference",
                "Conditional exchangeability given measured pre-treatment covariates",
                "Positivity in the realized target population",
                "The supplied propensity score is suitable for the declared design",
            ),
        )


__all__ = ["MatchingEstimand", "NearestNeighborMatch", "NearestNeighborMatchResult"]

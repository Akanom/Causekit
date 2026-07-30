"""Contract-first scalar nearest-neighbor matching."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.stats import norm

from ._data import _validate_pandas_indices

MatchingEstimand = Literal["ate", "att", "atc"]
PropensityScoreStatus = Literal["estimated", "known"]


@runtime_checkable
class FittedPropensityMLEProtocol(Protocol):
    """Structural contract for a fitted full-sample parametric propensity MLE.

    The matching estimator validates the fitted probabilities, likelihood score,
    convergence flag, feature ordering, and sample size. The boundary is provider-neutral:
    CauseKit neither imports a nuisance-estimation package nor owns a duplicate binary-
    response likelihood.
    """

    params: Any
    converged: bool
    nobs: int
    feature_names: tuple[str, ...]

    def predict_proba(self, X: Any) -> Any: ...


@dataclass(frozen=True)
class NearestNeighborMatchResult:
    """Auditable scalar nearest-neighbor matching estimate and inference."""

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
    propensity_score_status: PropensityScoreStatus
    bias_correction: str
    inference: str
    variance: float
    normalized_variance: float
    known_score_variance: float
    known_score_normalized_variance: float
    first_step_variance_adjustment: float
    first_step_asymptotic_adjustment: float
    conditional_variances: pd.Series
    conditional_variance_component: float
    effect_variance_component: float
    variance_neighbors: int
    first_step_covariance_neighbors: int
    first_step_regression_neighbors: int
    first_step_covariate_neighbors: int
    propensity_scores: pd.Series
    propensity_information: pd.DataFrame
    propensity_adjustment_vector: pd.Series
    propensity_target_derivative: pd.Series
    propensity_model_name: str | None
    propensity_model_score_norm: float
    propensity_link: str | None
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
        critical = float(norm.ppf(0.5 + level / 2.0))
        return pd.Series(
            {
                "lower": self.estimate - critical * self.standard_error,
                "upper": self.estimate + critical * self.standard_error,
            },
            name=self.realized_estimand,
        )

    def summary_frame(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        if np.isfinite(self.standard_error):
            interval = self.conf_int(level)
            lower, upper = interval["lower"], interval["upper"]
        else:
            lower = upper = float("nan")
        return pd.DataFrame(
            {
                "coef": [self.estimate],
                "std_err": [self.standard_error],
                "stat": [self.statistic],
                "p_value": [self.pvalue],
                "ci_lower": [lower],
                "ci_upper": [upper],
            },
            index=pd.Index([self.realized_estimand], dtype="object"),
        )

    def to_markdown(self, digits: int = 4) -> str:
        """Render a dependency-free audit summary without exposing outcomes."""

        if digits < 0:
            raise ValueError("digits must be non-negative.")
        standard_error = (
            f"{self.standard_error:.{digits}f}"
            if np.isfinite(self.standard_error)
            else "not estimated"
        )
        inference = {
            "abadie_imbens": "Abadie-Imbens known-score analytical variance",
            "abadie_imbens_estimated": (
                "Abadie-Imbens estimated-propensity MLE analytical variance"
            ),
            "none": "none",
        }[self.inference]
        requested_focal_count = (
            self.n_treated
            if self.requested_estimand == "att"
            else self.n_control
            if self.requested_estimand == "atc"
            else self.nobs
        )
        lines = [
            "# Nearest-neighbor matching result",
            "",
            f"- Requested estimand: `{self.requested_estimand.upper()}`",
            f"- Realized estimand: `{self.realized_estimand.upper()}`",
            f"- Target: {self.target_population}",
            f"- Matched focal observations: `{self.n_matched_focal}` of "
            f"`{requested_focal_count}` supplied focal observations",
            f"- Score status: `{self.propensity_score_status}`",
            f"- Score provenance: `{self.propensity_provenance}`",
            f"- Inference: `{inference}`",
            "",
            "| estimand | estimate | std_err |",
            "|---|---:|---:|",
            f"| {self.realized_estimand.upper()} | {self.estimate:.{digits}f} | {standard_error} |",
            "",
            f"> {self.causal_interpretation}",
        ]
        if self.notes:
            lines.extend(["", "## Notes", "", *(f"- {note}" for note in self.notes)])
        return "\n".join(lines)


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


def _propensity_design(value: Any) -> tuple[pd.DataFrame, bool]:
    labeled = isinstance(value, (pd.Series, pd.DataFrame))
    if isinstance(value, pd.Series):
        frame = value.to_frame()
    elif isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ValueError("propensity_design must contain only numeric values.") from error
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        if raw.ndim != 2 or raw.shape[1] == 0:
            raise ValueError("propensity_design must be one- or two-dimensional numeric data.")
        frame = pd.DataFrame(raw, columns=[f"x_{position}" for position in range(raw.shape[1])])
    if frame.shape[1] == 0:
        raise ValueError("propensity_design must contain at least one column.")
    names = tuple(
        "propensity_covariate" if name is None and frame.shape[1] == 1 else str(name)
        for name in frame.columns
    )
    if len(set(names)) != len(names):
        raise ValueError("propensity_design column names must be unique after string conversion.")
    frame.columns = names
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("propensity_design must contain only numeric values.") from error
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("propensity_design must contain only finite values.")
    return frame, labeled


def _fitted_propensity_probabilities(
    model: FittedPropensityMLEProtocol,
    design: pd.DataFrame,
) -> np.ndarray:
    raw = model.predict_proba(design)
    if isinstance(raw, (pd.Series, pd.DataFrame)) and not raw.index.equals(design.index):
        raise ValueError(
            "propensity_model.predict_proba() must preserve the propensity_design index."
        )
    if isinstance(raw, pd.DataFrame):
        if 1 in raw.columns:
            values = raw[1].to_numpy(dtype=float)
        elif raw.shape[1] == 2:
            values = raw.iloc[:, 1].to_numpy(dtype=float)
        else:
            raise ValueError(
                "propensity_model.predict_proba() must expose the treated-class probability."
            )
    else:
        values = np.asarray(raw, dtype=float)
        if values.ndim == 2 and values.shape[1] == 2:
            values = values[:, 1]
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(design):
        raise ValueError("propensity_model.predict_proba() must return one probability per row.")
    if not np.isfinite(values).all() or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError(
            "propensity_model probabilities must be finite and strictly between zero and one."
        )
    return values


def _validate_full_sample_logit_mle(
    model: FittedPropensityMLEProtocol,
    design: pd.DataFrame,
    treatment: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Validate the exact regular logit-MLE first step required by AI (2016)."""

    if not isinstance(model.converged, (bool, np.bool_)) or not bool(model.converged):
        raise ValueError("propensity_model must report convergence for analytical inference.")
    if (
        isinstance(model.nobs, (bool, np.bool_))
        or not isinstance(model.nobs, (int, np.integer))
        or int(model.nobs) != len(design)
    ):
        raise ValueError("propensity_model.nobs must equal the matching estimation-sample size.")
    feature_names = tuple(str(name) for name in model.feature_names)
    if feature_names != tuple(design.columns):
        raise ValueError(
            "propensity_model feature_names must match propensity_design columns exactly."
        )
    if isinstance(model.params, pd.Series):
        parameter_names = tuple(str(name) for name in model.params.index)
        if parameter_names != tuple(design.columns):
            raise ValueError(
                "propensity_model.params must align exactly with propensity_design columns."
            )
    parameters = np.asarray(model.params, dtype=float)
    if parameters.ndim != 1 or len(parameters) != design.shape[1]:
        raise ValueError("propensity_model.params has an invalid dimension.")
    if not np.isfinite(parameters).all():
        raise ValueError("propensity_model.params must contain only finite values.")

    design_values = design.to_numpy(dtype=float)
    implied = expit(design_values @ parameters)
    if not np.allclose(probabilities, implied, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "propensity_model probabilities must equal the declared logit linear-index "
            "predictions on propensity_design."
        )
    normalized_score = design_values.T @ (treatment - probabilities) / len(design)
    score_norm = float(np.max(np.abs(normalized_score)))
    if not np.isfinite(score_norm) or score_norm > 1e-7:
        raise ValueError(
            "propensity_model must be a full-sample unpenalized logit MLE with a "
            "near-zero likelihood score on the matching sample."
        )

    weights = probabilities * (1.0 - probabilities)
    information = design_values.T @ (weights[:, None] * design_values) / len(design)
    information = (information + information.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(information)
    maximum = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    if (
        not np.isfinite(eigenvalues).all()
        or maximum <= 0.0
        or float(eigenvalues[0]) <= maximum * 1e-12
    ):
        raise ValueError(
            "The normalized propensity information matrix is singular or ill-conditioned."
        )
    return information, score_norm


def _neighbor_group(
    focal: float,
    sorted_metric: np.ndarray,
    sorted_positions: np.ndarray,
    *,
    neighbors: int,
    caliper: float | None,
    exclude_position: int | None = None,
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
            candidate = int(sorted_positions[left])
            if candidate != exclude_position:
                group.append((candidate, boundary))
            left -= 1
        while (
            right < len(sorted_metric)
            and abs((sorted_metric[right] - focal) - boundary) <= tolerance
        ):
            candidate = int(sorted_positions[right])
            if candidate != exclude_position:
                group.append((candidate, boundary))
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


def _local_score_covariances(
    outcome: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    design: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    """Paper Section 4 local covariances of X and Y given score and arm."""

    estimates = np.empty((len(outcome), 2, design.shape[1]), dtype=float)
    for arm in (0, 1):
        arm_positions = np.flatnonzero(treatment == arm)
        if len(arm_positions) < neighbors:
            raise ValueError(
                "first_step_covariance_neighbors exceeds the observations available "
                "in a treatment arm."
            )
        order = np.argsort(score[arm_positions], kind="stable")
        sorted_positions = arm_positions[order]
        sorted_score = score[sorted_positions]
        for focal_position in range(len(outcome)):
            found = _neighbor_group(
                score[focal_position],
                sorted_score,
                sorted_positions,
                neighbors=neighbors,
                caliper=None,
            )
            if found is None:  # pragma: no cover - guarded by arm size
                raise RuntimeError("First-step covariance neighbor search failed unexpectedly.")
            matches, expanded = found
            if expanded:
                raise ValueError(
                    "Estimated-propensity covariance matching encountered expanded "
                    "boundary ties; analytical inference is unavailable."
                )
            positions = np.asarray([position for position, *_ in matches], dtype=int)
            local_design = design[positions]
            local_outcome = outcome[positions]
            estimates[focal_position, arm] = (
                (local_design - local_design.mean(axis=0))
                * (local_outcome - local_outcome.mean())[:, None]
            ).sum(axis=0) / (neighbors - 1)
    return estimates


def _local_score_means(
    outcome: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    """Leave-own-observation-out local score regressions for the ATT correction."""

    estimates = np.empty((len(outcome), 2), dtype=float)
    for arm in (0, 1):
        arm_positions = np.flatnonzero(treatment == arm)
        if len(arm_positions) <= neighbors:
            raise ValueError(
                "first_step_regression_neighbors requires more observations in each "
                "treatment arm than the requested neighbor count."
            )
        order = np.argsort(score[arm_positions], kind="stable")
        sorted_positions = arm_positions[order]
        sorted_score = score[sorted_positions]
        for focal_position in range(len(outcome)):
            found = _neighbor_group(
                score[focal_position],
                sorted_score,
                sorted_positions,
                neighbors=neighbors,
                caliper=None,
                exclude_position=(focal_position if treatment[focal_position] == arm else None),
            )
            if found is None:  # pragma: no cover - guarded by arm size
                raise RuntimeError("First-step regression neighbor search failed unexpectedly.")
            matches, expanded = found
            if expanded:
                raise ValueError(
                    "Estimated-propensity regression matching encountered expanded "
                    "boundary ties; analytical inference is unavailable."
                )
            estimates[focal_position, arm] = sum(
                outcome[position] * weight for position, _, _, _, weight in matches
            )
    return estimates


def _covariate_matched_effect_residuals(
    outcome: np.ndarray,
    treatment: np.ndarray,
    design: np.ndarray,
    *,
    estimate: float,
    neighbors: int,
) -> np.ndarray:
    """Equation following (9): nearest-opposite-arm matching on X without N^2 storage."""

    residuals = np.empty(len(outcome), dtype=float)
    for arm in (0, 1):
        focal_positions = np.flatnonzero(treatment == arm)
        comparison_positions = np.flatnonzero(treatment == 1 - arm)
        if len(comparison_positions) < neighbors:
            raise ValueError(
                "first_step_covariate_neighbors exceeds the observations available "
                "in a treatment arm."
            )
        query_count = min(neighbors + 1, len(comparison_positions))
        tree = cKDTree(design[comparison_positions])
        distances, locations = tree.query(
            design[focal_positions],
            k=query_count,
            workers=-1,
        )
        distances = np.asarray(distances, dtype=float)
        locations = np.asarray(locations, dtype=int)
        if distances.ndim == 1:
            distances = distances[:, None]
            locations = locations[:, None]
        if len(comparison_positions) > neighbors:
            boundary = distances[:, neighbors - 1]
            next_distance = distances[:, neighbors]
            tolerance = 32 * np.finfo(float).eps * np.maximum(1.0, np.abs(boundary))
            if np.any(np.abs(next_distance - boundary) <= tolerance):
                raise ValueError(
                    "Estimated-propensity covariate matching encountered expanded "
                    "boundary ties; analytical inference is unavailable."
                )
        selected = comparison_positions[locations[:, :neighbors]]
        matched_mean = outcome[selected].mean(axis=1)
        residuals[focal_positions] = (2 * arm - 1) * (
            outcome[focal_positions] - matched_mean
        ) - estimate
    return residuals


def _conditional_variances(
    outcome: np.ndarray,
    treatment: np.ndarray,
    metric: np.ndarray,
    *,
    variance_neighbors: int,
) -> np.ndarray:
    """Equation (14) in Abadie and Imbens (2006), using same-arm matches."""

    estimates = np.empty(len(outcome), dtype=float)
    for arm in (0, 1):
        arm_positions = np.flatnonzero(treatment == arm)
        if len(arm_positions) <= variance_neighbors:
            raise ValueError(
                "variance_neighbors requires more same-arm observations than the requested "
                "conditional-variance neighbor count in both treatment arms."
            )
        order = np.argsort(metric[arm_positions], kind="stable")
        sorted_positions = arm_positions[order]
        sorted_metric = metric[sorted_positions]
        for focal_position in arm_positions:
            found = _neighbor_group(
                metric[focal_position],
                sorted_metric,
                sorted_positions,
                neighbors=variance_neighbors,
                caliper=None,
                exclude_position=int(focal_position),
            )
            if found is None:  # pragma: no cover - guarded by the same-arm size check
                raise RuntimeError("Conditional-variance neighbor search failed unexpectedly.")
            matches, expanded = found
            if expanded:
                raise ValueError(
                    "Abadie-Imbens conditional-variance matching encountered expanded "
                    "boundary ties; use inference='none'."
                )
            matched_mean = sum(
                outcome[comparison] * weight for comparison, _, _, _, weight in matches
            )
            residual = outcome[focal_position] - matched_mean
            estimates[focal_position] = (
                variance_neighbors / (variance_neighbors + 1.0) * residual**2
            )
    return estimates


def _abadie_imbens_variance(
    *,
    estimand: MatchingEstimand,
    estimate: float,
    unit_effects: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    metric: np.ndarray,
    reuse_counts: np.ndarray,
    neighbors: int,
    variance_neighbors: int,
) -> tuple[float, float, np.ndarray, float, float]:
    """Return sampling and normalized variances under a known scalar score."""

    conditional_variances = _conditional_variances(
        outcome,
        treatment,
        metric,
        variance_neighbors=variance_neighbors,
    )
    match_ratio = reuse_counts / neighbors
    centered_effects = (unit_effects - estimate) ** 2
    if estimand == "ate":
        target_size = len(outcome)
        normalized_variance = float(
            np.mean(centered_effects)
            + np.mean(
                reuse_counts
                * (reuse_counts + 2 * neighbors - 1)
                / neighbors**2
                * conditional_variances
            )
        )
        conditional_component = float(np.mean((1.0 + match_ratio) ** 2 * conditional_variances))
    elif estimand == "att":
        target_size = int(treatment.sum())
        normalized_variance = float(
            np.mean(centered_effects)
            + np.sum(
                (1 - treatment)
                * reuse_counts
                * (reuse_counts - 1)
                / neighbors**2
                * conditional_variances
            )
            / target_size
        )
        conditional_component = float(
            np.sum((treatment - (1 - treatment) * match_ratio) ** 2 * conditional_variances)
            / target_size
        )
    else:
        target_size = int((1 - treatment).sum())
        normalized_variance = float(
            np.mean(centered_effects)
            + np.sum(
                treatment * reuse_counts * (reuse_counts - 1) / neighbors**2 * conditional_variances
            )
            / target_size
        )
        conditional_component = float(
            np.sum(((1 - treatment) - treatment * match_ratio) ** 2 * conditional_variances)
            / target_size
        )
    variance = normalized_variance / target_size
    return (
        variance,
        normalized_variance,
        conditional_variances,
        conditional_component,
        normalized_variance - conditional_component,
    )


def _estimated_propensity_adjustment(
    *,
    estimand: MatchingEstimand,
    estimate: float,
    known_score_variance: float,
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    design: np.ndarray,
    information: np.ndarray,
    covariance_neighbors: int,
    regression_neighbors: int,
    covariate_neighbors: int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Abadie-Imbens (2016) full-sample parametric-MLE variance correction."""

    inverse_information = np.linalg.inv(information)
    density = propensity * (1.0 - propensity)
    local_covariances = _local_score_covariances(
        outcome,
        treatment,
        propensity,
        design,
        neighbors=covariance_neighbors,
    )
    if estimand == "ate":
        adjustment_vector = np.mean(
            (
                local_covariances[:, 1] / propensity[:, None]
                + local_covariances[:, 0] / (1.0 - propensity)[:, None]
            )
            * density[:, None],
            axis=0,
        )
        target_derivative = np.zeros(design.shape[1], dtype=float)
    else:
        target_treatment = treatment if estimand == "att" else 1 - treatment
        target_propensity = propensity if estimand == "att" else 1.0 - propensity
        target_estimate = estimate if estimand == "att" else -estimate
        target_density = target_propensity * (1.0 - target_propensity)
        target_size = int(target_treatment.sum())
        target_covariances = (
            local_covariances
            if estimand == "att"
            else _local_score_covariances(
                outcome,
                target_treatment,
                target_propensity,
                design,
                neighbors=covariance_neighbors,
            )
        )
        local_means = _local_score_means(
            outcome,
            target_treatment,
            target_propensity,
            neighbors=regression_neighbors,
        )
        target_covariance_component = (
            np.sum(
                (
                    target_covariances[:, 1]
                    + target_propensity[:, None]
                    / (1.0 - target_propensity)[:, None]
                    * target_covariances[:, 0]
                )
                * target_density[:, None],
                axis=0,
            )
            / target_size
        )
        target_regression_component = (
            np.sum(
                design
                * target_density[:, None]
                * (local_means[:, 1] - local_means[:, 0] - target_estimate)[:, None],
                axis=0,
            )
            / target_size
        )
        adjustment_vector = target_regression_component + target_covariance_component
        covariate_effect_residuals = _covariate_matched_effect_residuals(
            outcome,
            target_treatment,
            design,
            estimate=target_estimate,
            neighbors=covariate_neighbors,
        )
        target_derivative = (
            np.sum(
                design * target_density[:, None] * covariate_effect_residuals[:, None],
                axis=0,
            )
            / target_size
        )

    asymptotic_adjustment = float(
        -adjustment_vector @ inverse_information @ adjustment_vector
        + target_derivative @ inverse_information @ target_derivative
    )
    variance_adjustment = asymptotic_adjustment / len(outcome)
    variance = known_score_variance + variance_adjustment
    numerical_tolerance = 1e-12 * max(1.0, abs(known_score_variance), abs(variance_adjustment))
    if not np.isfinite(variance) or variance <= numerical_tolerance:
        raise ValueError(
            "The estimated-propensity first-step correction produced a non-positive "
            "sampling variance; analytical inference is unavailable for this sample."
        )
    target_size = (
        len(outcome)
        if estimand == "ate"
        else int(treatment.sum())
        if estimand == "att"
        else int((1 - treatment).sum())
    )
    return (
        variance,
        variance * target_size,
        variance_adjustment,
        asymptotic_adjustment,
        adjustment_vector,
        target_derivative,
    )


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
        variance_neighbors: int = 1,
        first_step_covariance_neighbors: int = 2,
        first_step_regression_neighbors: int = 2,
        first_step_covariate_neighbors: int = 1,
    ) -> None:
        if estimand not in {"ate", "att", "atc"}:
            raise ValueError("estimand must be 'ate', 'att', or 'atc'.")
        if metric not in {"propensity", "propensity_logit"}:
            raise NotImplementedError(
                "Only metric='propensity' and metric='propensity_logit' are implemented."
            )
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
        if inference not in {"abadie_imbens", "abadie_imbens_estimated", "none"}:
            raise ValueError(
                "inference must be 'abadie_imbens', 'abadie_imbens_estimated', or 'none'."
            )
        if (
            isinstance(variance_neighbors, bool)
            or not isinstance(variance_neighbors, (int, np.integer))
            or variance_neighbors < 1
        ):
            raise ValueError("variance_neighbors must be a positive integer.")
        if (
            isinstance(first_step_covariance_neighbors, bool)
            or not isinstance(first_step_covariance_neighbors, (int, np.integer))
            or first_step_covariance_neighbors < 2
        ):
            raise ValueError("first_step_covariance_neighbors must be an integer of at least two.")
        for name, value in (
            ("first_step_regression_neighbors", first_step_regression_neighbors),
            ("first_step_covariate_neighbors", first_step_covariate_neighbors),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
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
        self.variance_neighbors = int(variance_neighbors)
        self.first_step_covariance_neighbors = int(first_step_covariance_neighbors)
        self.first_step_regression_neighbors = int(first_step_regression_neighbors)
        self.first_step_covariate_neighbors = int(first_step_covariate_neighbors)

    def fit(
        self,
        y: Any,
        *,
        treatment: Any,
        propensity: Any | None = None,
        covariates: Any | None = None,
        propensity_model: FittedPropensityMLEProtocol | None = None,
        propensity_design: Any | None = None,
        propensity_provenance: str = "supplied_unspecified",
        propensity_score_status: PropensityScoreStatus = "estimated",
        clusters: Any | None = None,
    ) -> NearestNeighborMatchResult:
        if clusters is not None:
            raise NotImplementedError("clustered matching inference is not implemented.")
        if not isinstance(propensity_provenance, str) or not propensity_provenance.strip():
            raise ValueError("propensity_provenance must be a nonempty string.")
        if propensity_score_status not in {"estimated", "known"}:
            raise ValueError("propensity_score_status must be 'estimated' or 'known'.")
        model_design: pd.DataFrame | None = None
        model_design_labeled = False
        model_probabilities: np.ndarray | None = None
        if propensity_model is not None:
            if not isinstance(propensity_model, FittedPropensityMLEProtocol):
                raise TypeError("propensity_model must satisfy FittedPropensityMLEProtocol.")
            if propensity_design is None:
                raise ValueError("propensity_model requires propensity_design.")
            model_design, model_design_labeled = _propensity_design(propensity_design)
            model_probabilities = _fitted_propensity_probabilities(propensity_model, model_design)
            if propensity is None:
                propensity = (
                    pd.Series(
                        model_probabilities,
                        index=model_design.index,
                        name="propensity",
                    )
                    if model_design_labeled
                    else model_probabilities
                )
        else:
            if propensity_design is not None:
                raise ValueError("propensity_design requires propensity_model.")
            if propensity is None:
                raise ValueError(
                    "propensity is required unless propensity_model and propensity_design are supplied."
                )
        outcome, assigned, propensity_values, covariate_frame = _prepare(
            y, treatment, propensity, covariates
        )
        index = outcome.index
        treatment_values = assigned.to_numpy(dtype=int)
        propensity_array = propensity_values.to_numpy(dtype=float)
        metric_values = (
            np.log(propensity_array / (1.0 - propensity_array))
            if self.metric == "propensity_logit"
            else propensity_array.copy()
        )
        nobs = len(outcome)
        propensity_information_values = np.empty((0, 0), dtype=float)
        propensity_model_score_norm = float("nan")
        propensity_model_name: str | None = None
        propensity_feature_names: tuple[str, ...] = ()
        if model_design is not None:
            if len(model_design) != nobs:
                raise ValueError(
                    "propensity_design must contain the same number of observations as y."
                )
            if model_design_labeled and not model_design.index.equals(index):
                raise ValueError("Pandas indices must match exactly and in the same order.")
            model_design.index = index
            if model_probabilities is None:  # pragma: no cover - established above
                raise RuntimeError("Propensity-model prediction state is inconsistent.")
            if not np.allclose(
                propensity_array,
                model_probabilities,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    "Supplied propensity values must equal propensity_model predictions."
                )
            propensity_model_name = type(propensity_model).__name__
            propensity_feature_names = tuple(model_design.columns)
        if self.inference == "abadie_imbens_estimated":
            if propensity_score_status != "estimated":
                raise ValueError(
                    "inference='abadie_imbens_estimated' requires "
                    "propensity_score_status='estimated'."
                )
            if self.metric != "propensity":
                raise NotImplementedError(
                    "Abadie-Imbens estimated-propensity inference requires "
                    "metric='propensity' to match the paper's F(X' theta) contract."
                )
            if propensity_model is None or model_design is None:
                raise ValueError(
                    "Estimated-propensity inference requires both propensity_model and "
                    "propensity_design."
                )
            (
                propensity_information_values,
                propensity_model_score_norm,
            ) = _validate_full_sample_logit_mle(
                propensity_model,
                model_design,
                treatment_values,
                propensity_array,
            )
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
        variance = normalized_variance = float("nan")
        known_score_variance = known_score_normalized_variance = float("nan")
        first_step_variance_adjustment = float("nan")
        first_step_asymptotic_adjustment = float("nan")
        propensity_adjustment_values = np.empty(0, dtype=float)
        propensity_target_derivative_values = np.empty(0, dtype=float)
        conditional_variance_component = effect_variance_component = float("nan")
        conditional_variance_values = np.full(nobs, np.nan)
        standard_error = statistic = pvalue = float("nan")
        inference_distribution = None
        if self.inference in {"abadie_imbens", "abadie_imbens_estimated"}:
            if boundary_tie_events:
                raise ValueError(
                    "Abadie-Imbens analytical inference is unavailable when boundary ties "
                    "expand the realized neighbor count; use inference='none'."
                )
            if self.inference == "abadie_imbens" and propensity_score_status != "known":
                raise NotImplementedError(
                    "Abadie-Imbens inference currently requires a declared known/fixed "
                    "propensity score. Estimated or cross-fitted scores require a "
                    "first-step-specific variance adjustment."
                )
            if self.caliper is not None or self.common_support is not None:
                raise NotImplementedError(
                    "Maintained Abadie-Imbens inference currently requires caliper=None "
                    "and common_support=None so data-dependent selection does not change "
                    "the target or variance contract."
                )
            (
                known_score_variance,
                known_score_normalized_variance,
                conditional_variance_values,
                conditional_variance_component,
                effect_variance_component,
            ) = _abadie_imbens_variance(
                estimand=self.estimand,
                estimate=estimate,
                unit_effects=np.asarray(unit_effects),
                outcome=outcome_values,
                treatment=treatment_values,
                metric=metric_values,
                reuse_counts=comparison_reuse.to_numpy(dtype=float),
                neighbors=self.neighbors,
                variance_neighbors=self.variance_neighbors,
            )
            variance = known_score_variance
            normalized_variance = known_score_normalized_variance
            if self.inference == "abadie_imbens_estimated":
                if model_design is None:  # pragma: no cover - validated above
                    raise RuntimeError("Propensity-model design state is inconsistent.")
                (
                    variance,
                    normalized_variance,
                    first_step_variance_adjustment,
                    first_step_asymptotic_adjustment,
                    propensity_adjustment_values,
                    propensity_target_derivative_values,
                ) = _estimated_propensity_adjustment(
                    estimand=self.estimand,
                    estimate=estimate,
                    known_score_variance=known_score_variance,
                    outcome=outcome_values,
                    treatment=treatment_values,
                    propensity=propensity_array,
                    design=model_design.to_numpy(dtype=float),
                    information=propensity_information_values,
                    covariance_neighbors=self.first_step_covariance_neighbors,
                    regression_neighbors=self.first_step_regression_neighbors,
                    covariate_neighbors=self.first_step_covariate_neighbors,
                )
            standard_error = float(np.sqrt(variance))
            if standard_error > 0.0:
                statistic = estimate / standard_error
                pvalue = float(2.0 * norm.sf(abs(statistic)))
            elif estimate == 0.0:
                statistic = pvalue = float("nan")
            else:
                statistic = float(np.sign(estimate) * np.inf)
                pvalue = 0.0
            inference_distribution = "normal"

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
        if self.inference == "abadie_imbens":
            notes.append(
                "Analytical variance conditions on the supplied score as known/fixed and "
                "assumes independent sampling units."
            )
        elif self.inference == "abadie_imbens_estimated":
            notes.append(
                "Analytical variance applies the Abadie-Imbens first-step correction for "
                "a validated full-sample unpenalized logit propensity MLE."
            )
        assumptions = [
            "Treatment consistency",
            "No interference",
            "Conditional exchangeability given measured pre-treatment covariates",
            "Positivity in the realized target population",
            "The supplied propensity score is suitable for the declared design",
        ]
        if self.inference == "abadie_imbens":
            assumptions.extend(
                [
                    "The supplied scalar score is known/fixed rather than estimated in this sample",
                    "Independent sampling units and fixed-neighbor replacement asymptotics",
                ]
            )
        elif self.inference == "abadie_imbens_estimated":
            assumptions.extend(
                [
                    "The propensity model is a correctly specified regular full-sample unpenalized logit MLE",
                    "Independent sampling units and fixed-neighbor replacement asymptotics",
                    "No caliper, common-support trimming, or expanded boundary ties alter the analytical contract",
                ]
            )
        match_table = match_table_internal.copy()
        matched_focal_mask = np.isin(np.arange(nobs), matched_focal_positions)
        caliper_unmatched_mask = np.isin(np.arange(nobs), caliper_unmatched_positions)
        original_focal_count = int(focal_target.sum())
        return NearestNeighborMatchResult(
            estimate=estimate,
            standard_error=standard_error,
            statistic=statistic,
            pvalue=pvalue,
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
            propensity_score_status=propensity_score_status,
            bias_correction=self.bias_correction,
            inference=self.inference,
            variance=variance,
            normalized_variance=normalized_variance,
            known_score_variance=known_score_variance,
            known_score_normalized_variance=known_score_normalized_variance,
            first_step_variance_adjustment=first_step_variance_adjustment,
            first_step_asymptotic_adjustment=first_step_asymptotic_adjustment,
            conditional_variances=pd.Series(
                conditional_variance_values,
                index=index,
                name="conditional_variance",
            ),
            conditional_variance_component=conditional_variance_component,
            effect_variance_component=effect_variance_component,
            variance_neighbors=self.variance_neighbors,
            first_step_covariance_neighbors=self.first_step_covariance_neighbors,
            first_step_regression_neighbors=self.first_step_regression_neighbors,
            first_step_covariate_neighbors=self.first_step_covariate_neighbors,
            propensity_scores=pd.Series(
                propensity_array,
                index=index,
                name="propensity_score",
            ),
            propensity_information=pd.DataFrame(
                propensity_information_values,
                index=pd.Index(propensity_feature_names, dtype="object"),
                columns=pd.Index(propensity_feature_names, dtype="object"),
            ),
            propensity_adjustment_vector=pd.Series(
                propensity_adjustment_values,
                index=pd.Index(propensity_feature_names, dtype="object"),
                name="propensity_adjustment",
            ),
            propensity_target_derivative=pd.Series(
                propensity_target_derivative_values,
                index=pd.Index(propensity_feature_names, dtype="object"),
                name="target_derivative",
            ),
            propensity_model_name=propensity_model_name,
            propensity_model_score_norm=propensity_model_score_norm,
            propensity_link=("logit" if self.inference == "abadie_imbens_estimated" else None),
            inference_distribution=inference_distribution,
            inference_df=None,
            notes=tuple(notes),
            assumptions=tuple(assumptions),
        )


__all__ = [
    "FittedPropensityMLEProtocol",
    "MatchingEstimand",
    "NearestNeighborMatch",
    "NearestNeighborMatchResult",
    "PropensityScoreStatus",
]

"""Conventional and semiparametrically efficient difference-in-differences."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import chi2, f, norm
from scipy.stats import t as student_t

from .crossfit import CrossFitTask, CrossFitter

ControlGroup = Literal["never_treated", "not_yet_treated"]
DiDCovariance = Literal["robust", "clustered"]
DiDInference = Literal["analytic", "multiplier_bootstrap"]
PrePeriods = Literal["all"] | int


@dataclass(frozen=True)
class DiDPretrendDiagnostic:
    """Uncontaminated cohort-period placebo effects and their joint test."""

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
class DiDHausmanDiagnostic:
    """PT-All versus PT-Post event-study Hausman diagnostic."""

    statistic: float
    pvalue: float
    reject: bool
    level: float
    recommendation: str
    event_study: pd.DataFrame
    influence: pd.DataFrame
    covariance: pd.DataFrame
    pt_post_covariance: pd.DataFrame
    pt_all_covariance: pd.DataFrame
    cross_covariance: pd.DataFrame
    n_restrictions: int
    df_num: int
    df_denom: float | None
    distribution: str
    covariance_type: DiDCovariance
    n_clusters: int | None
    null_hypothesis: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DiDResult:
    """Auditable group-time, event-study, calendar-time, and average DiD result."""

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
    pretrend: DiDPretrendDiagnostic
    efficiency_weights: pd.DataFrame
    conditional_efficiency_weights: pd.DataFrame
    candidate_influence_functions: pd.DataFrame
    simultaneous_event_study: pd.DataFrame
    nuisance_fold: pd.Series
    nuisance_weighting: str
    cohort_probabilities: pd.DataFrame
    cohort_ratios: pd.DataFrame
    cohort_ratio_diagnostics: pd.DataFrame
    cohort_ratio_candidate_uses: pd.DataFrame
    cohort_sizes: pd.Series
    n_entities: int
    n_periods: int
    n_clusters: int | None
    covariance_type: DiDCovariance
    inference_distribution: str
    inference_df: float | None
    inference_method: DiDInference
    simultaneous_level: float | None
    simultaneous_critical_value: float | None
    bootstrap_iterations: int | None
    bootstrap_random_state: int | None
    method: str
    parallel_trends: str
    control_group: str
    anticipation: int
    pre_periods: str | int
    entity_name: str
    time_name: str
    outcome_name: str
    treatment_time_name: str
    covariates: tuple[str, ...]
    cross_fitted: bool
    estimation_entities: pd.Index
    inference_clusters: pd.Series
    times: tuple[float, ...]
    design_fingerprint: str
    assumptions: tuple[str, ...]
    notes: tuple[str, ...]
    converged: bool = True
    backend: str = "native-did"

    @property
    def nobs(self) -> int:
        return self.n_entities

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
            "The estimates are causal only under consistency, no interference, the "
            "declared no-anticipation window, overlap, and the recorded parallel-trends "
            "restriction for the reported comparison cohorts."
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
        """Return the scalar ESavg summary in the package's standard schema."""

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
class _Panel:
    outcomes: np.ndarray
    entities: pd.Index
    times: np.ndarray
    original_cohorts: np.ndarray
    treated_cohorts: tuple[float, ...]
    original_positions: dict[float, int]
    effective_positions: dict[float, int]
    entity_effective_positions: np.ndarray
    never_mask: np.ndarray
    clusters: np.ndarray | None
    design_fingerprint: str


@dataclass(frozen=True)
class _EffectCell:
    cohort: float
    time: float
    event_time: int
    base_period: float
    att: float
    influence: np.ndarray
    n_treated: int
    n_comparison: int
    cohort_share: float
    weight_condition_number: float


@dataclass(frozen=True)
class _Change:
    group: float
    end: int
    start: int

    @property
    def is_zero(self) -> bool:
        return self.end == self.start


@dataclass(frozen=True)
class _CandidateSpec:
    auxiliary_cohort: float
    bridge_position: int
    treated_change: _Change
    never_change: _Change
    auxiliary_change: _Change


def _validate_constructor(
    *,
    anticipation: int,
    covariance: str,
    inference: str,
    bootstrap_iterations: int,
    random_state: int | None,
    simultaneous_level: float,
) -> tuple[int, DiDCovariance, DiDInference, int, int | None, float]:
    if isinstance(anticipation, bool) or not isinstance(anticipation, Integral):
        raise ValueError("anticipation must be a non-negative integer number of periods.")
    anticipation = int(anticipation)
    if anticipation < 0:
        raise ValueError("anticipation must be a non-negative integer number of periods.")
    if covariance not in {"robust", "clustered"}:
        raise ValueError("covariance must be 'robust' or 'clustered'.")
    if inference not in {"analytic", "multiplier_bootstrap"}:
        raise ValueError("inference must be 'analytic' or 'multiplier_bootstrap'.")
    if (
        isinstance(bootstrap_iterations, bool)
        or not isinstance(bootstrap_iterations, Integral)
        or int(bootstrap_iterations) < 99
    ):
        raise ValueError("bootstrap_iterations must be an integer of at least 99.")
    if random_state is not None and not isinstance(random_state, Integral):
        raise TypeError("random_state must be an integer or None.")
    if not np.isfinite(simultaneous_level) or not 0.0 < simultaneous_level < 1.0:
        raise ValueError("simultaneous_level must be strictly between zero and one.")
    return (
        anticipation,
        cast(DiDCovariance, covariance),
        cast(DiDInference, inference),
        int(bootstrap_iterations),
        None if random_state is None else int(random_state),
        float(simultaneous_level),
    )


def _prepare_panel(
    data: pd.DataFrame,
    *,
    outcome: str,
    entity: str,
    time: str,
    treatment_time: str,
    never_treated: float,
    anticipation: int,
    covariance: DiDCovariance,
    cluster: str | None,
) -> _Panel:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame in long panel form.")
    if not data.columns.is_unique:
        raise ValueError("data column names must be unique.")
    role_names = [outcome, entity, time, treatment_time]
    if len(set(role_names)) != len(role_names):
        raise ValueError("outcome, entity, time, and treatment_time must be distinct columns.")
    required = role_names + ([cluster] if cluster is not None else [])
    missing_columns = [name for name in required if name not in data.columns]
    if missing_columns:
        raise ValueError(f"data is missing required column(s): {missing_columns}.")
    if data[entity].isna().any():
        raise ValueError("entity identifiers must not contain missing values.")
    if data.duplicated([entity, time]).any():
        raise ValueError("data must contain exactly one row per entity-time pair.")

    try:
        outcome_values = pd.to_numeric(data[outcome], errors="raise").to_numpy(dtype=float)
        time_values = pd.to_numeric(data[time], errors="raise").to_numpy(dtype=float)
        cohort_values = pd.to_numeric(data[treatment_time], errors="raise").to_numpy(dtype=float)
        never_value = float(never_treated)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "outcome, time, treatment_time, and never_treated must be numeric."
        ) from error
    if not np.isfinite(outcome_values).all():
        raise ValueError("outcome must contain only finite values.")
    if not np.isfinite(time_values).all():
        raise ValueError("time must contain only finite values.")
    if np.isnan(never_value):
        raise ValueError("never_treated must be an explicit non-missing numeric sentinel.")
    if np.isnan(cohort_values).any():
        raise ValueError("treatment_time must not contain missing values.")
    invalid_cohort = ~np.isfinite(cohort_values) & (cohort_values != never_value)
    if invalid_cohort.any():
        raise ValueError("treatment_time contains a non-finite value other than never_treated.")

    working = pd.DataFrame(
        {
            "__entity": data[entity].to_numpy(),
            "__time": time_values,
            "__outcome": outcome_values,
            "__cohort": cohort_values,
        }
    )
    try:
        entities = pd.Index(pd.unique(working["__entity"]), name=entity).sort_values()
    except TypeError as error:
        raise ValueError("entity identifiers must have a deterministic sortable order.") from error
    times = np.sort(pd.unique(time_values).astype(float))
    if times.size < 2:
        raise ValueError("difference-in-differences requires at least two panel periods.")
    if np.any(times == never_value):
        raise ValueError("never_treated must not equal an observed time value.")

    expected_rows = len(entities) * len(times)
    if len(working) != expected_rows:
        raise ValueError("data must be a balanced panel with every entity in every period.")
    outcome_wide = working.pivot(index="__entity", columns="__time", values="__outcome")
    outcome_wide = outcome_wide.reindex(index=entities, columns=times)
    if outcome_wide.isna().any().any():
        raise ValueError("data must be a balanced panel with a finite outcome in every cell.")

    cohort_counts = working.groupby("__entity", sort=False)["__cohort"].nunique(dropna=False)
    if (cohort_counts != 1).any():
        raise ValueError("treatment_time must be constant within entity.")
    cohort_by_entity = working.groupby("__entity", sort=False)["__cohort"].first().reindex(entities)
    entity_cohorts = cohort_by_entity.to_numpy(dtype=float)
    never_mask = entity_cohorts == never_value
    if not never_mask.any():
        raise ValueError("at least one never-treated cohort is required.")

    treated_values = np.unique(entity_cohorts[~never_mask])
    if treated_values.size == 0:
        raise ValueError("at least one treated cohort is required.")
    observed_time_set = set(times.tolist())
    if any(float(value) not in observed_time_set for value in treated_values):
        raise ValueError("every finite treatment_time must equal an observed time value.")
    time_positions = {float(value): position for position, value in enumerate(times)}
    treated_cohorts = tuple(
        sorted(
            (float(value) for value in treated_values),
            key=lambda value: time_positions[value],
        )
    )
    original_positions = {cohort: time_positions[cohort] for cohort in treated_cohorts}
    effective_positions = {
        cohort: original_positions[cohort] - anticipation for cohort in treated_cohorts
    }
    if any(position <= 0 for position in effective_positions.values()):
        raise ValueError(
            "every treated cohort must retain at least one uncontaminated baseline period "
            "after applying anticipation."
        )

    cohort_sizes = pd.Series(entity_cohorts).value_counts()
    if int(never_mask.sum()) < 2:
        raise ValueError("the never-treated cohort requires at least two entities for inference.")
    if any(int(cohort_sizes.loc[cohort]) < 2 for cohort in treated_cohorts):
        raise ValueError("every treated cohort requires at least two entities for inference.")

    entity_effective = np.full(len(entities), -1, dtype=int)
    for cohort, position in effective_positions.items():
        entity_effective[entity_cohorts == cohort] = position

    cluster_values: np.ndarray | None = None
    if covariance == "clustered":
        if cluster is None:
            raise ValueError("cluster is required when covariance='clustered'.")
        if data[cluster].isna().any():
            raise ValueError("cluster must not contain missing values.")
        cluster_working = pd.DataFrame(
            {"__entity": data[entity].to_numpy(), "__cluster": data[cluster].to_numpy()}
        )
        cluster_counts = cluster_working.groupby("__entity", sort=False)["__cluster"].nunique(
            dropna=False
        )
        if (cluster_counts != 1).any():
            raise ValueError("cluster must be constant within entity.")
        cluster_values = (
            cluster_working.groupby("__entity", sort=False)["__cluster"]
            .first()
            .reindex(entities)
            .to_numpy()
        )
        if len(pd.unique(cluster_values)) < 2:
            raise ValueError("clustered inference requires at least two clusters.")
    elif cluster is not None:
        raise ValueError("cluster is only allowed when covariance='clustered'.")

    fingerprint = hashlib.sha256()
    fingerprint.update(
        pd.util.hash_pandas_object(pd.Series(entities, dtype="object"), index=False)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    )
    for values in (times, entity_cohorts, outcome_wide.to_numpy(dtype=float)):
        fingerprint.update(np.ascontiguousarray(values, dtype=float).tobytes())
    if cluster_values is not None:
        fingerprint.update(
            pd.util.hash_pandas_object(pd.Series(cluster_values, dtype="object"), index=False)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        )

    return _Panel(
        outcomes=outcome_wide.to_numpy(dtype=float),
        entities=entities,
        times=times,
        original_cohorts=entity_cohorts,
        treated_cohorts=treated_cohorts,
        original_positions=original_positions,
        effective_positions=effective_positions,
        entity_effective_positions=entity_effective,
        never_mask=never_mask,
        clusters=cluster_values,
        design_fingerprint=fingerprint.hexdigest(),
    )


def _score_statistics(
    estimate: float,
    influence: np.ndarray,
    *,
    covariance: DiDCovariance,
    clusters: np.ndarray | None,
) -> tuple[float, float, float, str, float | None, int | None]:
    n = len(influence)
    inference_df = None
    n_clusters = None
    if covariance == "robust":
        variance = float(influence @ influence) / (n * (n - 1))
        distribution = "normal"
    else:
        if clusters is None:  # defensive; panel validation owns the public refusal
            raise ValueError("cluster labels are required for clustered inference.")
        codes, labels = pd.factorize(clusters, sort=False)
        n_clusters = len(labels)
        cluster_scores = np.zeros(n_clusters)
        np.add.at(cluster_scores, codes, influence)
        variance = float(n_clusters / (n_clusters - 1) * (cluster_scores @ cluster_scores) / n**2)
        distribution = "t"
        inference_df = float(n_clusters - 1)
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


def _prepare_covariates(
    data: pd.DataFrame,
    panel: _Panel,
    *,
    entity: str,
    covariates: Sequence[str],
    protected_names: Sequence[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if isinstance(covariates, (str, bytes)):
        raise TypeError("covariates must be a non-empty sequence of column names.")
    names = tuple(covariates)
    if not names:
        raise ValueError("covariates must contain at least one column name.")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("Every covariate name must be a non-empty string.")
    if len(set(names)) != len(names):
        raise ValueError("covariate names must be unique.")
    if set(names).intersection(protected_names):
        raise ValueError("covariates must be distinct from outcome and panel role columns.")
    missing = [name for name in names if name not in data.columns]
    if missing:
        raise ValueError(f"data is missing covariate column(s): {missing}.")
    if data.loc[:, names].isna().any().any():
        raise ValueError("covariates must not contain missing values.")
    grouped = data.groupby(entity, sort=False)[list(names)]
    if (grouped.nunique(dropna=False) != 1).any().any():
        raise ValueError("every covariate must be constant within entity.")
    frame = grouped.first().reindex(panel.entities)
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("covariates must contain only numeric values.") from error
    if not np.isfinite(values).all():
        raise ValueError("covariates must contain only finite values.")
    return pd.DataFrame(values, index=panel.entities.copy(), columns=names), names


def _candidate_specs(
    panel: _Panel,
    *,
    cohort: float,
    baseline_position: int,
    target_position: int,
    never_treated: float,
) -> list[_CandidateSpec]:
    specs: list[_CandidateSpec] = []
    treated_change = _Change(cohort, target_position, baseline_position)
    for auxiliary_cohort in panel.treated_cohorts:
        auxiliary_position = panel.effective_positions[auxiliary_cohort]
        bridge_start = baseline_position if auxiliary_cohort == cohort else baseline_position + 1
        for bridge_position in range(bridge_start, auxiliary_position):
            specs.append(
                _CandidateSpec(
                    auxiliary_cohort=auxiliary_cohort,
                    bridge_position=bridge_position,
                    treated_change=treated_change,
                    never_change=_Change(never_treated, target_position, bridge_position),
                    auxiliary_change=_Change(auxiliary_cohort, bridge_position, baseline_position),
                )
            )
    return specs


def _multiplier_event_study_band(
    event_study: pd.DataFrame,
    influence: pd.DataFrame,
    *,
    panel: _Panel,
    covariance: DiDCovariance,
    iterations: int,
    random_state: int | None,
    level: float,
) -> tuple[pd.DataFrame, float]:
    scores = influence.to_numpy(dtype=float)
    standard_errors = event_study["std_err"].to_numpy(dtype=float)
    if np.any(~np.isfinite(standard_errors)) or np.any(standard_errors <= 0.0):
        raise ValueError(
            "Simultaneous event-study bands require a positive finite standard error "
            "for every event time."
        )
    n = len(panel.entities)
    if covariance == "robust":
        score_units = scores
        finite_sample_scale = np.sqrt(n / (n - 1))
    else:
        if panel.clusters is None:  # defensive; validation owns the public refusal
            raise ValueError("cluster labels are required for clustered multiplier bands.")
        codes, labels = pd.factorize(panel.clusters, sort=False)
        score_units = np.zeros((len(labels), scores.shape[1]))
        np.add.at(score_units, codes, scores)
        finite_sample_scale = np.sqrt(len(labels) / (len(labels) - 1))
    rng = np.random.default_rng(random_state)
    maximum_statistics = np.empty(iterations, dtype=float)
    completed = 0
    while completed < iterations:
        batch = min(256, iterations - completed)
        multipliers = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(score_units)))
        perturbations = finite_sample_scale * multipliers @ score_units / n
        maximum_statistics[completed : completed + batch] = np.max(
            np.abs(perturbations / standard_errors), axis=1
        )
        completed += batch
    critical = float(np.quantile(maximum_statistics, level, method="higher"))
    estimates = event_study["att"].to_numpy(dtype=float)
    bands = pd.DataFrame(
        {
            "att": estimates,
            "std_err": standard_errors,
            "critical_value": critical,
            "level": level,
            "lower": estimates - critical * standard_errors,
            "upper": estimates + critical * standard_errors,
        },
        index=event_study.index.copy(),
    )
    return bands, critical


def _aggregate_influence(cells: list[_EffectCell], panel: _Panel) -> tuple[float, np.ndarray]:
    n = len(panel.entities)
    shares = np.array([cell.cohort_share for cell in cells])
    denominator = float(shares.sum())
    weights = shares / denominator
    influence = np.zeros(n)
    centered_membership = np.zeros(n)
    masks: list[np.ndarray] = []
    for cell in cells:
        mask = panel.original_cohorts == cell.cohort
        masks.append(mask)
        centered_membership += mask.astype(float) - cell.cohort_share
    estimate = float(sum(weight * cell.att for weight, cell in zip(weights, cells, strict=True)))
    for weight, share, mask, cell in zip(weights, shares, masks, cells, strict=True):
        share_influence = (
            mask.astype(float) - share
        ) / denominator - share * centered_membership / denominator**2
        influence += weight * cell.influence + cell.att * share_influence
    return estimate, influence


def _summary_row(
    estimate: float,
    influence: np.ndarray,
    *,
    covariance: DiDCovariance,
    clusters: np.ndarray | None,
) -> dict[str, float]:
    standard_error, statistic, pvalue, _, _, _ = _score_statistics(
        estimate, influence, covariance=covariance, clusters=clusters
    )
    return {
        "att": estimate,
        "std_err": standard_error,
        "stat": statistic,
        "p_value": pvalue,
    }


def _score_cross_covariance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    covariance: DiDCovariance,
    clusters: np.ndarray | None,
) -> tuple[np.ndarray, int | None]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Influence matrices must be two-dimensional and row aligned.")
    n = left.shape[0]
    if covariance == "robust":
        return left.T @ right / (n * (n - 1)), None
    if clusters is None:
        raise ValueError("cluster labels are required for clustered joint inference.")
    codes, labels = pd.factorize(clusters, sort=False)
    cluster_left = np.zeros((len(labels), left.shape[1]))
    cluster_right = np.zeros((len(labels), right.shape[1]))
    np.add.at(cluster_left, codes, left)
    np.add.at(cluster_right, codes, right)
    scale = len(labels) / (len(labels) - 1) / n**2
    return scale * cluster_left.T @ cluster_right, len(labels)


def _joint_wald(
    estimates: np.ndarray,
    covariance_matrix: np.ndarray,
    *,
    covariance: DiDCovariance,
    n_clusters: int | None,
    singularity_tolerance: float = 1e-12,
) -> tuple[float, float, str, float | None]:
    covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    smallest = float(eigenvalues[0]) if eigenvalues.size else 0.0
    if (
        not np.isfinite(eigenvalues).all()
        or largest <= 0.0
        or smallest <= singularity_tolerance * largest
    ):
        raise ValueError(
            "The joint diagnostic covariance is singular or numerically unidentified; "
            "no ridge or pseudoinverse was applied."
        )
    wald = float(estimates @ np.linalg.solve(covariance_matrix, estimates))
    restrictions = len(estimates)
    if covariance == "robust":
        return wald, float(chi2.sf(wald, restrictions)), "chi2", None
    if n_clusters is None:
        raise ValueError("The cluster count is required for clustered joint inference.")
    statistic = wald / restrictions
    denominator_df = float(n_clusters - 1)
    return (
        statistic,
        float(f.sf(statistic, restrictions, denominator_df)),
        "f",
        denominator_df,
    )


def _empty_pretrend(
    panel: _Panel,
    *,
    covariance: DiDCovariance,
    reason: str,
) -> DiDPretrendDiagnostic:
    columns = pd.MultiIndex.from_arrays([[], []], names=["cohort", "time"])
    placebo_effects = pd.DataFrame(
        columns=[
            "event_time",
            "base_period",
            "att",
            "std_err",
            "stat",
            "p_value",
            "n_treated",
            "n_comparison",
        ],
        index=columns,
    )
    influence = pd.DataFrame(index=panel.entities.copy(), columns=columns, dtype=float)
    covariance_frame = pd.DataFrame(index=columns, columns=columns, dtype=float)
    n_clusters = None if panel.clusters is None else len(pd.unique(panel.clusters))
    return DiDPretrendDiagnostic(
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
        null_hypothesis=("Every retained uncontaminated cohort-period placebo effect equals zero."),
        notes=(
            "Failure to reject is not evidence that parallel trends is true.",
            "Post-estimation choices should not be selected mechanically from this diagnostic.",
        ),
    )


def _pretrend_diagnostic(
    panel: _Panel,
    *,
    covariance: DiDCovariance,
    control_group: ControlGroup | None,
    unavailable_reason: str | None,
) -> DiDPretrendDiagnostic:
    if control_group is None:
        return _empty_pretrend(
            panel,
            covariance=covariance,
            reason=unavailable_reason
            or "The declared estimator does not expose an aligned pre-trend diagnostic.",
        )
    n = len(panel.entities)
    rows: list[dict[str, Any]] = []
    keys: list[tuple[float, float]] = []
    influences: list[np.ndarray] = []
    for cohort in panel.treated_cohorts:
        treated_mask = panel.original_cohorts == cohort
        n_treated = int(treated_mask.sum())
        pi_treated = n_treated / n
        effective_position = panel.effective_positions[cohort]
        for target_position in range(1, effective_position):
            base_position = target_position - 1
            if control_group == "never_treated":
                comparison_mask = panel.never_mask.copy()
            else:
                comparison_mask = panel.never_mask | (
                    panel.entity_effective_positions > target_position
                )
            comparison_mask &= ~treated_mask
            n_comparison = int(comparison_mask.sum())
            if n_comparison < 2:
                raise ValueError(
                    "each pre-trend placebo requires at least two eligible comparison entities."
                )
            changes = panel.outcomes[:, target_position] - panel.outcomes[:, base_position]
            treated_mean = float(changes[treated_mask].mean())
            comparison_mean = float(changes[comparison_mask].mean())
            placebo = treated_mean - comparison_mean
            pi_comparison = n_comparison / n
            influence = treated_mask.astype(float) / pi_treated * (
                changes - treated_mean
            ) - comparison_mask.astype(float) / pi_comparison * (changes - comparison_mean)
            public_time = float(panel.times[target_position])
            keys.append((cohort, public_time))
            influences.append(influence)
            rows.append(
                {
                    "cohort": cohort,
                    "time": public_time,
                    "event_time": target_position - panel.original_positions[cohort],
                    "base_period": float(panel.times[base_position]),
                    **_summary_row(
                        placebo,
                        influence,
                        covariance=covariance,
                        clusters=panel.clusters,
                    ),
                    "n_treated": n_treated,
                    "n_comparison": n_comparison,
                }
            )
    if not rows:
        return _empty_pretrend(
            panel,
            covariance=covariance,
            reason=(
                "No uncontaminated adjacent pre-period changes remain before the declared "
                "treatment or anticipation boundary."
            ),
        )

    columns = pd.MultiIndex.from_tuples(keys, names=["cohort", "time"])
    influence_matrix = np.column_stack(influences)
    influence_frame = pd.DataFrame(
        influence_matrix,
        index=panel.entities.copy(),
        columns=columns,
    )
    covariance_matrix, n_clusters = _score_cross_covariance(
        influence_matrix,
        influence_matrix,
        covariance=covariance,
        clusters=panel.clusters,
    )
    covariance_frame = pd.DataFrame(
        covariance_matrix,
        index=columns.copy(),
        columns=columns.copy(),
    )
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
        return DiDPretrendDiagnostic(
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
            null_hypothesis=(
                "Every retained uncontaminated cohort-period placebo effect equals zero."
            ),
            notes=(
                "The placebo effects remain available, but their joint covariance was not invertible.",
                "No ridge, dimension dropping, or pseudoinverse was applied.",
            ),
        )
    return DiDPretrendDiagnostic(
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
        null_hypothesis=("Every retained uncontaminated cohort-period placebo effect equals zero."),
        notes=(
            "Placebos use adjacent changes ending strictly before the anticipation boundary.",
            "Failure to reject is not evidence that parallel trends is true.",
            "Post-estimation choices should not be selected mechanically from this diagnostic.",
        ),
    )


def _assemble_result(
    cells: list[_EffectCell],
    panel: _Panel,
    *,
    covariance: DiDCovariance,
    method: str,
    parallel_trends: str,
    control_group: str,
    anticipation: int,
    pre_periods: str | int,
    outcome_name: str,
    entity_name: str,
    time_name: str,
    treatment_time_name: str,
    efficiency_weights: pd.DataFrame,
    conditional_efficiency_weights: pd.DataFrame,
    candidate_influence: pd.DataFrame,
    inference: DiDInference,
    bootstrap_iterations: int,
    bootstrap_random_state: int | None,
    simultaneous_level: float,
    nuisance_fold: pd.Series,
    nuisance_weighting: str,
    cohort_probabilities: pd.DataFrame,
    cohort_ratios: pd.DataFrame,
    cohort_ratio_diagnostics: pd.DataFrame,
    cohort_ratio_candidate_uses: pd.DataFrame,
    covariates: tuple[str, ...],
    cross_fitted: bool,
    pretrend_control_group: ControlGroup | None,
    pretrend_unavailable_reason: str | None,
    notes: tuple[str, ...],
) -> DiDResult:
    group_rows: list[dict[str, Any]] = []
    group_keys: list[tuple[float, float]] = []
    for cell in cells:
        row: dict[str, Any] = {
            "cohort": cell.cohort,
            "time": cell.time,
            "event_time": cell.event_time,
            "base_period": cell.base_period,
            **_summary_row(
                cell.att,
                cell.influence,
                covariance=covariance,
                clusters=panel.clusters,
            ),
            "n_treated": cell.n_treated,
            "n_comparison": cell.n_comparison,
            "cohort_share": cell.cohort_share,
            "weight_condition_number": cell.weight_condition_number,
        }
        group_rows.append(row)
        group_keys.append((cell.cohort, cell.time))
    group_time = pd.DataFrame(group_rows).set_index(["cohort", "time"]).sort_index()
    group_columns = pd.MultiIndex.from_tuples(group_keys, names=["cohort", "time"])
    group_time_influence = pd.DataFrame(
        np.column_stack([cell.influence for cell in cells]),
        index=panel.entities.copy(),
        columns=group_columns,
    ).sort_index(axis=1)

    event_rows: list[dict[str, Any]] = []
    event_influences: dict[int, np.ndarray] = {}
    for event_time in sorted({cell.event_time for cell in cells}):
        selected = [cell for cell in cells if cell.event_time == event_time]
        estimate, influence = _aggregate_influence(selected, panel)
        event_influences[event_time] = influence
        event_rows.append(
            {
                "event_time": event_time,
                **_summary_row(estimate, influence, covariance=covariance, clusters=panel.clusters),
                "n_cohorts": len(selected),
            }
        )
    event_study = pd.DataFrame(event_rows).set_index("event_time").sort_index()
    event_study_influence = pd.DataFrame(event_influences, index=panel.entities.copy()).sort_index(
        axis=1
    )
    event_study_influence.columns.name = "event_time"

    calendar_rows: list[dict[str, Any]] = []
    calendar_influences: dict[float, np.ndarray] = {}
    calendar_times = sorted({cell.time for cell in cells if cell.event_time >= 0})
    for calendar_time in calendar_times:
        selected = [cell for cell in cells if cell.time == calendar_time and cell.event_time >= 0]
        estimate, influence = _aggregate_influence(selected, panel)
        calendar_influences[calendar_time] = influence
        calendar_rows.append(
            {
                "time": calendar_time,
                **_summary_row(estimate, influence, covariance=covariance, clusters=panel.clusters),
                "n_cohorts": len(selected),
            }
        )
    calendar_time = pd.DataFrame(calendar_rows).set_index("time").sort_index()
    calendar_time_influence = pd.DataFrame(
        calendar_influences, index=panel.entities.copy()
    ).sort_index(axis=1)
    calendar_time_influence.columns.name = "time"

    post_event_times = [value for value in event_study.index if value >= 0]
    estimate = float(event_study.loc[post_event_times, "att"].mean())
    overall_array = event_study_influence.loc[:, post_event_times].mean(axis=1).to_numpy()
    standard_error, statistic, pvalue, distribution, inference_df, n_clusters = _score_statistics(
        estimate, overall_array, covariance=covariance, clusters=panel.clusters
    )
    overall_influence = pd.Series(
        overall_array, index=panel.entities.copy(), name="esavg_influence"
    )
    cohort_sizes = pd.Series(
        {
            cohort: int(np.count_nonzero(panel.original_cohorts == cohort))
            for cohort in panel.treated_cohorts
        },
        name="n_entities",
    )
    cohort_sizes.index.name = "cohort"

    simultaneous_event_study = pd.DataFrame(
        columns=["att", "std_err", "critical_value", "level", "lower", "upper"]
    )
    simultaneous_critical_value: float | None = None
    if inference == "multiplier_bootstrap":
        simultaneous_event_study, simultaneous_critical_value = _multiplier_event_study_band(
            event_study,
            event_study_influence,
            panel=panel,
            covariance=covariance,
            iterations=bootstrap_iterations,
            random_state=bootstrap_random_state,
            level=simultaneous_level,
        )

    pretrend = _pretrend_diagnostic(
        panel,
        covariance=covariance,
        control_group=pretrend_control_group,
        unavailable_reason=pretrend_unavailable_reason,
    )
    inference_clusters = (
        pd.Series(dtype="object", name="cluster")
        if panel.clusters is None
        else pd.Series(panel.clusters.copy(), index=panel.entities.copy(), name="cluster")
    )

    return DiDResult(
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
        efficiency_weights=efficiency_weights,
        conditional_efficiency_weights=conditional_efficiency_weights,
        candidate_influence_functions=candidate_influence,
        simultaneous_event_study=simultaneous_event_study,
        nuisance_fold=nuisance_fold,
        nuisance_weighting=nuisance_weighting,
        cohort_probabilities=cohort_probabilities,
        cohort_ratios=cohort_ratios,
        cohort_ratio_diagnostics=cohort_ratio_diagnostics,
        cohort_ratio_candidate_uses=cohort_ratio_candidate_uses,
        cohort_sizes=cohort_sizes,
        n_entities=len(panel.entities),
        n_periods=len(panel.times),
        n_clusters=n_clusters,
        covariance_type=covariance,
        inference_distribution=distribution,
        inference_df=inference_df,
        inference_method=inference,
        simultaneous_level=(simultaneous_level if inference == "multiplier_bootstrap" else None),
        simultaneous_critical_value=simultaneous_critical_value,
        bootstrap_iterations=(
            bootstrap_iterations if inference == "multiplier_bootstrap" else None
        ),
        bootstrap_random_state=(
            bootstrap_random_state if inference == "multiplier_bootstrap" else None
        ),
        method=method,
        parallel_trends=parallel_trends,
        control_group=control_group,
        anticipation=anticipation,
        pre_periods=pre_periods,
        entity_name=entity_name,
        time_name=time_name,
        outcome_name=outcome_name,
        treatment_time_name=treatment_time_name,
        covariates=covariates,
        cross_fitted=cross_fitted,
        estimation_entities=panel.entities.copy(),
        inference_clusters=inference_clusters,
        times=tuple(float(value) for value in panel.times),
        design_fingerprint=panel.design_fingerprint,
        assumptions=(
            "Balanced short panel with entities sampled independently unless higher-level clusters are declared",
            "Absorbing treatment at the declared first-treatment time",
            "No anticipation outside the declared anticipation window",
            "Consistency and no interference",
            "Overlap for every reported treated and comparison cohort",
            f"Parallel trends contract: {parallel_trends}",
        ),
        notes=notes,
    )


class DifferenceInDifferences:
    """Conventional no-covariate group-time DiD for balanced short panels."""

    def __init__(
        self,
        *,
        control_group: ControlGroup = "never_treated",
        anticipation: int = 0,
        covariance: DiDCovariance = "robust",
        inference: DiDInference = "analytic",
        bootstrap_iterations: int = 999,
        random_state: int | None = None,
        simultaneous_level: float = 0.95,
    ) -> None:
        if control_group not in {"never_treated", "not_yet_treated"}:
            raise ValueError("control_group must be 'never_treated' or 'not_yet_treated'.")
        (
            anticipation,
            covariance,
            inference,
            bootstrap_iterations,
            random_state,
            simultaneous_level,
        ) = _validate_constructor(
            anticipation=anticipation,
            covariance=covariance,
            inference=inference,
            bootstrap_iterations=bootstrap_iterations,
            random_state=random_state,
            simultaneous_level=simultaneous_level,
        )
        self.control_group = control_group
        self.anticipation = anticipation
        self.covariance = covariance
        self.inference = inference
        self.bootstrap_iterations = bootstrap_iterations
        self.random_state = random_state
        self.simultaneous_level = simultaneous_level

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        entity: str,
        time: str,
        treatment_time: str,
        never_treated: float = np.inf,
        cluster: str | None = None,
        covariates: Sequence[str] | None = None,
    ) -> DiDResult:
        """Estimate conventional cohort-time effects and their aggregations."""

        if covariates is not None:
            raise NotImplementedError(
                "The covariate-adjusted DiD path is not implemented in this alpha; "
                "no nuisance model will be fitted or copied implicitly."
            )
        panel = _prepare_panel(
            data,
            outcome=outcome,
            entity=entity,
            time=time,
            treatment_time=treatment_time,
            never_treated=never_treated,
            anticipation=self.anticipation,
            covariance=self.covariance,
            cluster=cluster,
        )
        n = len(panel.entities)
        cells: list[_EffectCell] = []
        for cohort in panel.treated_cohorts:
            treated_mask = panel.original_cohorts == cohort
            n_treated = int(treated_mask.sum())
            pi_treated = n_treated / n
            effective_position = panel.effective_positions[cohort]
            base_position = effective_position - 1
            for target_position in range(effective_position, len(panel.times)):
                if self.control_group == "never_treated":
                    comparison_mask = panel.never_mask.copy()
                else:
                    comparison_mask = panel.never_mask | (
                        panel.entity_effective_positions > target_position
                    )
                comparison_mask &= ~treated_mask
                n_comparison = int(comparison_mask.sum())
                if n_comparison < 2:
                    raise ValueError(
                        "each group-time effect requires at least two eligible comparison entities."
                    )
                changes = panel.outcomes[:, target_position] - panel.outcomes[:, base_position]
                treated_mean = float(changes[treated_mask].mean())
                comparison_mean = float(changes[comparison_mask].mean())
                att = treated_mean - comparison_mean
                pi_comparison = n_comparison / n
                influence = treated_mask.astype(float) / pi_treated * (
                    changes - treated_mean
                ) - comparison_mask.astype(float) / pi_comparison * (changes - comparison_mean)
                cells.append(
                    _EffectCell(
                        cohort=cohort,
                        time=float(panel.times[target_position]),
                        event_time=target_position - panel.original_positions[cohort],
                        base_period=float(panel.times[base_position]),
                        att=att,
                        influence=influence,
                        n_treated=n_treated,
                        n_comparison=n_comparison,
                        cohort_share=pi_treated,
                        weight_condition_number=float("nan"),
                    )
                )

        empty_weights = pd.DataFrame(
            columns=[
                "cohort",
                "time",
                "auxiliary_cohort",
                "bridge_period",
                "candidate_att",
                "weight",
            ]
        )
        empty_candidate = pd.DataFrame(index=panel.entities.copy())
        parallel_trends = (
            "post"
            if self.control_group == "never_treated"
            else "post_with_not_yet_treated_comparisons"
        )
        return _assemble_result(
            cells,
            panel,
            covariance=self.covariance,
            method="conventional_group_time",
            parallel_trends=parallel_trends,
            control_group=self.control_group,
            anticipation=self.anticipation,
            pre_periods=1,
            outcome_name=outcome,
            entity_name=entity,
            time_name=time,
            treatment_time_name=treatment_time,
            efficiency_weights=empty_weights,
            conditional_efficiency_weights=pd.DataFrame(index=panel.entities.copy()),
            candidate_influence=empty_candidate,
            inference=self.inference,
            bootstrap_iterations=self.bootstrap_iterations,
            bootstrap_random_state=self.random_state,
            simultaneous_level=self.simultaneous_level,
            nuisance_fold=pd.Series(dtype="int64", name="fold"),
            nuisance_weighting="none",
            cohort_probabilities=pd.DataFrame(index=panel.entities.copy()),
            cohort_ratios=pd.DataFrame(index=panel.entities.copy()),
            cohort_ratio_diagnostics=pd.DataFrame(),
            cohort_ratio_candidate_uses=pd.DataFrame(),
            covariates=(),
            cross_fitted=False,
            pretrend_control_group=self.control_group,
            pretrend_unavailable_reason=None,
            notes=(
                "Group-time effects use the period immediately before the effective treatment boundary as baseline.",
                (
                    "Event-study bands use a studentized entity-level multiplier bootstrap."
                    if self.inference == "multiplier_bootstrap"
                    else "Reported confidence intervals are pointwise."
                ),
            ),
        )


def did_hausman_test(
    pt_post: DiDResult,
    pt_all: DiDResult,
    *,
    level: float = 0.95,
) -> DiDHausmanDiagnostic:
    """Compare common post-treatment event-study paths under PT-Post and PT-All.

    The finite-sample covariance uses the difference of the two aligned influence
    functions. This is the positive-semidefinite construction discussed after
    Theorem A.1 of Chen, Sant'Anna, and Xie (2025); no variance subtraction,
    dimension dropping, ridge, or pseudoinverse is used.
    """

    if not isinstance(pt_post, DiDResult) or not isinstance(pt_all, DiDResult):
        raise TypeError("pt_post and pt_all must both be fitted DiDResult objects.")
    if pt_post.method != "conventional_group_time":
        raise ValueError("The first argument must be a conventional PT-Post DiD result.")
    if pt_all.method != "chen_santanna_xie_efficient":
        raise ValueError(
            "The second argument must be the no-covariate Chen-Sant'Anna-Xie PT-All result."
        )
    if pt_post.control_group != "never_treated":
        raise ValueError(
            "The maintained Hausman contract requires the PT-Post never-treated comparison."
        )
    if pt_post.covariates or pt_all.covariates or pt_post.cross_fitted or pt_all.cross_fitted:
        raise ValueError(
            "The maintained Hausman contract is currently limited to the aligned "
            "no-covariate PT-Post and PT-All paths."
        )
    if pt_all.pre_periods != "all":
        raise ValueError("The PT-All result must retain all admissible pre-period moments.")
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one.")
    if pt_post.design_fingerprint != pt_all.design_fingerprint:
        raise ValueError(
            "The Hausman diagnostic requires both results to use the same estimation sample, "
            "outcomes, timing, and cluster design."
        )
    if not pt_post.estimation_entities.equals(pt_all.estimation_entities):
        raise ValueError("The Hausman diagnostic requires identically ordered entities.")
    if (
        pt_post.times != pt_all.times
        or pt_post.anticipation != pt_all.anticipation
        or pt_post.entity_name != pt_all.entity_name
        or pt_post.time_name != pt_all.time_name
        or pt_post.outcome_name != pt_all.outcome_name
        or pt_post.treatment_time_name != pt_all.treatment_time_name
    ):
        raise ValueError("The Hausman diagnostic requires identical DiD role and timing contracts.")
    if pt_post.covariance_type != pt_all.covariance_type:
        raise ValueError("The Hausman diagnostic requires the same covariance declaration.")
    if not pt_post.inference_clusters.equals(pt_all.inference_clusters):
        raise ValueError("The Hausman diagnostic requires identical inference clusters.")

    post_events = [int(value) for value in pt_post.event_study.index if int(value) >= 0]
    all_events = [int(value) for value in pt_all.event_study.index if int(value) >= 0]
    if post_events != all_events or not post_events:
        raise ValueError(
            "The PT-Post and PT-All results must expose the same non-empty post-treatment "
            "event-study support."
        )
    pt_post_estimates = pt_post.event_study.loc[post_events, "att"].to_numpy(dtype=float)
    pt_all_estimates = pt_all.event_study.loc[post_events, "att"].to_numpy(dtype=float)
    difference = pt_all_estimates - pt_post_estimates
    post_influence = pt_post.event_study_influence.loc[:, post_events].to_numpy(dtype=float)
    all_influence = pt_all.event_study_influence.loc[:, post_events].to_numpy(dtype=float)
    difference_influence = all_influence - post_influence
    clusters = (
        None if pt_post.covariance_type == "robust" else pt_post.inference_clusters.to_numpy()
    )
    post_covariance, n_clusters = _score_cross_covariance(
        post_influence,
        post_influence,
        covariance=pt_post.covariance_type,
        clusters=clusters,
    )
    all_covariance, _ = _score_cross_covariance(
        all_influence,
        all_influence,
        covariance=pt_post.covariance_type,
        clusters=clusters,
    )
    cross_covariance, _ = _score_cross_covariance(
        post_influence,
        all_influence,
        covariance=pt_post.covariance_type,
        clusters=clusters,
    )
    difference_covariance, _ = _score_cross_covariance(
        difference_influence,
        difference_influence,
        covariance=pt_post.covariance_type,
        clusters=clusters,
    )
    statistic, pvalue, distribution, denominator_df = _joint_wald(
        difference,
        difference_covariance,
        covariance=pt_post.covariance_type,
        n_clusters=n_clusters,
    )
    event_index = pd.Index(post_events, name="event_time")

    def covariance_frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=event_index.copy(), columns=event_index.copy())

    reject = pvalue < 1.0 - float(level)
    return DiDHausmanDiagnostic(
        statistic=statistic,
        pvalue=pvalue,
        reject=reject,
        level=float(level),
        recommendation="pt_post" if reject else "pt_all_not_rejected",
        event_study=pd.DataFrame(
            {
                "pt_post": pt_post_estimates,
                "pt_all": pt_all_estimates,
                "difference": difference,
            },
            index=event_index,
        ),
        influence=pd.DataFrame(
            difference_influence,
            index=pt_post.estimation_entities.copy(),
            columns=event_index.copy(),
        ),
        covariance=covariance_frame(difference_covariance),
        pt_post_covariance=covariance_frame(post_covariance),
        pt_all_covariance=covariance_frame(all_covariance),
        cross_covariance=covariance_frame(cross_covariance),
        n_restrictions=len(post_events),
        df_num=len(post_events),
        df_denom=denominator_df,
        distribution=distribution,
        covariance_type=pt_post.covariance_type,
        n_clusters=n_clusters,
        null_hypothesis=(
            "The common post-treatment event-study path is equal under PT-All and PT-Post."
        ),
        notes=(
            "Rejection is evidence against the extra PT-All restrictions, not against every possible parallel-trends restriction.",
            "Failure to reject does not prove PT-All and should not be used as an automatic estimator-selection rule.",
            "The covariance is computed from the difference influence function and is refused when singular.",
        ),
    )


def _change_values(panel: _Panel, change: _Change) -> np.ndarray:
    return panel.outcomes[:, change.end] - panel.outcomes[:, change.start]


def _ordered_pair(left: _Change, right: _Change) -> tuple[_Change, _Change]:
    left_key = (left.group, left.end, left.start)
    right_key = (right.group, right.end, right.start)
    return (left, right) if left_key <= right_key else (right, left)


def _fit_covariate_efficient(
    data: pd.DataFrame,
    panel: _Panel,
    *,
    outcome: str,
    entity: str,
    time: str,
    treatment_time: str,
    never_treated: float,
    covariates: Sequence[str],
    cross_fitter: CrossFitter,
    pre_periods: PrePeriods,
    anticipation: int,
    covariance: DiDCovariance,
    inference: DiDInference,
    bootstrap_iterations: int,
    bootstrap_random_state: int | None,
    simultaneous_level: float,
    singularity_tolerance: float,
    nuisance_probability_floor: float,
    nuisance_ratio_floor: float,
    nuisance_ratio_ceiling: float,
    nuisance_ratio_min_effective_n: float,
    nuisance_ratio_max_share: float,
    nuisance_ratio_min_psus: int,
) -> DiDResult:
    if cross_fitter.outcome_factory is None:
        raise ValueError("cross_fitter must provide an outcome_factory for DiD nuisances.")
    weighting_routes = int(cross_fitter.propensity_factory is not None) + int(
        cross_fitter.cohort_ratio_factory is not None
    )
    if weighting_routes != 1:
        raise ValueError(
            "Covariate-adjusted EfficientDiD requires exactly one cohort-weighting route: "
            "propensity_factory or cohort_ratio_factory."
        )
    X, covariate_names = _prepare_covariates(
        data,
        panel,
        entity=entity,
        covariates=covariates,
        protected_names=(outcome, entity, time, treatment_time),
    )
    strata = pd.Series(
        panel.original_cohorts,
        index=panel.entities.copy(),
        name="treatment_cohort",
    )

    cell_definitions: list[tuple[float, int, int, list[_CandidateSpec]]] = []
    changes: dict[_Change, None] = {}
    for cohort in panel.treated_cohorts:
        effective_position = panel.effective_positions[cohort]
        if pre_periods == "all":
            baseline_position = 0
        else:
            baseline_position = effective_position - int(pre_periods)
            if baseline_position < 0:
                raise ValueError(
                    f"pre_periods={pre_periods} exceeds the available clean history "
                    f"for cohort {cohort}."
                )
        for target_position in range(effective_position, len(panel.times)):
            specs = _candidate_specs(
                panel,
                cohort=cohort,
                baseline_position=baseline_position,
                target_position=target_position,
                never_treated=never_treated,
            )
            if not specs:
                raise ValueError(
                    f"No admissible PT-All generated outcomes exist for cohort {cohort}."
                )
            cell_definitions.append((cohort, baseline_position, target_position, specs))
            for spec in specs:
                changes[spec.treated_change] = None
                changes[spec.never_change] = None
                changes[spec.auxiliary_change] = None

    probability: dict[float, np.ndarray] = {}
    probabilities = pd.DataFrame(index=panel.entities.copy())
    ratio_frame = pd.DataFrame(index=panel.entities.copy())
    ratio_diagnostics = pd.DataFrame()
    ratio_use_rows: list[dict[str, Any]] = []
    if cross_fitter.propensity_factory is not None:
        class_result = cross_fitter.fit_predict_class_probabilities(
            X,
            classes=strata,
            clusters=panel.clusters,
        )
        probabilities = class_result.probabilities
        if float(probabilities.to_numpy().min()) <= nuisance_probability_floor:
            raise ValueError(
                "Cross-fitted cohort probabilities violate nuisance_probability_floor; "
                "no clipping was applied."
            )
        nuisance_fold = class_result.fold
        nuisance_weighting = "multiclass_probabilities"
        probability = {
            float(label): probabilities[label].to_numpy(dtype=float)
            for label in probabilities.columns
        }
    else:
        pair_use_count: dict[tuple[float, float], int] = {}
        ordered_pairs: list[tuple[float, float]] = []
        identity_pairs: list[tuple[float, float]] = []
        for cohort, _, target_position, specs in cell_definitions:
            public_time = float(panel.times[target_position])
            for spec in specs:
                for role, pair in (
                    ("never", (cohort, float(never_treated))),
                    ("auxiliary", (cohort, spec.auxiliary_cohort)),
                ):
                    pair_use_count[pair] = pair_use_count.get(pair, 0) + 1
                    if pair[0] == pair[1]:
                        if pair not in identity_pairs:
                            identity_pairs.append(pair)
                    elif pair not in ordered_pairs:
                        ordered_pairs.append(pair)
                    ratio_use_rows.append(
                        {
                            "cohort": cohort,
                            "time": public_time,
                            "auxiliary_cohort": spec.auxiliary_cohort,
                            "bridge_period": float(panel.times[spec.bridge_position]),
                            "role": role,
                            "numerator": pair[0],
                            "denominator": pair[1],
                            "source": "identity" if pair[0] == pair[1] else "fitted",
                        }
                    )
        ratio_result = cross_fitter.fit_predict_cohort_odds_ratios(
            X,
            cohorts=strata,
            pairs=ordered_pairs,
            clusters=panel.clusters,
        )
        ratio_frame = ratio_result.ratios.copy()
        for pair in identity_pairs:
            ratio_frame[pair] = 1.0
        ratio_frame = ratio_frame.loc[
            :,
            pd.MultiIndex.from_tuples(
                ordered_pairs + identity_pairs,
                names=["numerator", "denominator"],
            ),
        ]
        ratio_diagnostics = ratio_result.pair_diagnostics.copy()
        ratio_diagnostics["source"] = "fitted"
        ratio_diagnostics["candidate_use_count"] = [
            pair_use_count[(float(row.numerator), float(row.denominator))]
            for row in ratio_diagnostics.itertuples(index=False)
        ]
        identity_rows: list[dict[str, Any]] = []
        for numerator, denominator in identity_pairs:
            cohort_mask = panel.original_cohorts == numerator
            psus = (
                int(cohort_mask.sum())
                if panel.clusters is None
                else len(pd.unique(panel.clusters[cohort_mask]))
            )
            identity_rows.append(
                {
                    "numerator": numerator,
                    "denominator": denominator,
                    "fold": -1,
                    "ratio_min": 1.0,
                    "ratio_median": 1.0,
                    "ratio_p90": 1.0,
                    "ratio_p95": 1.0,
                    "ratio_p99": 1.0,
                    "ratio_max": 1.0,
                    "log_ratio_min": 0.0,
                    "log_ratio_p01": 0.0,
                    "log_ratio_p05": 0.0,
                    "log_ratio_median": 0.0,
                    "log_ratio_p95": 0.0,
                    "log_ratio_p99": 0.0,
                    "log_ratio_max": 0.0,
                    "train_numerator_nobs": int(cohort_mask.sum()),
                    "train_denominator_nobs": int(cohort_mask.sum()),
                    "train_numerator_psus": psus,
                    "train_denominator_psus": psus,
                    "holdout_numerator_nobs": int(cohort_mask.sum()),
                    "holdout_denominator_nobs": int(cohort_mask.sum()),
                    "holdout_numerator_psus": psus,
                    "holdout_denominator_psus": psus,
                    "denominator_importance_effective_n": float(cohort_mask.sum()),
                    "denominator_importance_max_share": float(1.0 / cohort_mask.sum()),
                    "train_index_hash": "identity",
                    "holdout_index_hash": "identity",
                    "source": "identity",
                    "candidate_use_count": pair_use_count[(numerator, denominator)],
                }
            )
        if identity_rows:
            ratio_diagnostics = pd.concat(
                [ratio_diagnostics, pd.DataFrame(identity_rows)],
                ignore_index=True,
                sort=False,
            )
        fitted_diagnostics = ratio_diagnostics.loc[ratio_diagnostics["source"].eq("fitted")]
        if float(fitted_diagnostics["ratio_min"].min()) < nuisance_ratio_floor:
            raise ValueError(
                "Cross-fitted cohort odds violate nuisance_ratio_floor; no clipping was applied."
            )
        if float(fitted_diagnostics["ratio_max"].max()) > nuisance_ratio_ceiling:
            raise ValueError(
                "Cross-fitted cohort odds violate nuisance_ratio_ceiling; no clipping was applied."
            )
        if (
            fitted_diagnostics["denominator_importance_effective_n"]
            < nuisance_ratio_min_effective_n
        ).any():
            raise ValueError(
                "Direct cohort-odds denominator importance violates nuisance_ratio_min_effective_n."
            )
        if (
            fitted_diagnostics["denominator_importance_max_share"] > nuisance_ratio_max_share
        ).any():
            raise ValueError(
                "Direct cohort-odds denominator importance violates nuisance_ratio_max_share."
            )
        if (
            (
                fitted_diagnostics[["train_numerator_psus", "train_denominator_psus"]]
                < nuisance_ratio_min_psus
            )
            .any()
            .any()
        ):
            raise ValueError("Direct cohort-odds pair support violates nuisance_ratio_min_psus.")
        nuisance_fold = ratio_result.fold
        nuisance_weighting = "direct_cohort_odds"

    nonzero_changes = [change for change in changes if not change.is_zero]
    mean_task_names = {
        change: f"conditional_mean_{position}" for position, change in enumerate(nonzero_changes)
    }
    mean_tasks = [
        CrossFitTask(
            name=mean_task_names[change],
            target=pd.Series(
                _change_values(panel, change),
                index=panel.entities.copy(),
            ),
            train_mask=pd.Series(
                panel.original_cohorts == change.group,
                index=panel.entities.copy(),
            ),
        )
        for change in nonzero_changes
    ]
    mean_result = cross_fitter.fit_predict_tasks(
        X,
        tasks=mean_tasks,
        strata=strata,
        clusters=panel.clusters,
        folds=nuisance_fold,
    )
    if not nuisance_fold.equals(mean_result.fold):
        raise RuntimeError("CrossFitter returned inconsistent folds across DiD nuisance tasks.")

    def conditional_mean(change: _Change) -> np.ndarray:
        if change.is_zero:
            return np.zeros(len(panel.entities))
        return mean_result.predictions[mean_task_names[change]].to_numpy(dtype=float)

    residuals = {
        change: _change_values(panel, change) - conditional_mean(change) for change in changes
    }
    covariance_pairs: dict[tuple[_Change, _Change], None] = {}
    for _, _, _, specs in cell_definitions:
        if len(specs) == 1:
            continue
        treated_change = specs[0].treated_change
        covariance_pairs[_ordered_pair(treated_change, treated_change)] = None
        for left in specs:
            for right in specs:
                covariance_pairs[_ordered_pair(left.never_change, right.never_change)] = None
                if left.auxiliary_cohort == treated_change.group:
                    covariance_pairs[_ordered_pair(treated_change, left.auxiliary_change)] = None
                if right.auxiliary_cohort == treated_change.group:
                    covariance_pairs[_ordered_pair(treated_change, right.auxiliary_change)] = None
                if left.auxiliary_cohort == right.auxiliary_cohort:
                    covariance_pairs[
                        _ordered_pair(left.auxiliary_change, right.auxiliary_change)
                    ] = None
    nonzero_pairs = [
        pair for pair in covariance_pairs if not pair[0].is_zero and not pair[1].is_zero
    ]
    covariance_task_names = {
        pair: f"conditional_covariance_{position}" for position, pair in enumerate(nonzero_pairs)
    }
    covariance_predictions: pd.DataFrame | None = None
    if nonzero_pairs:
        second_moment_factory = cross_fitter.second_moment_factory
        if second_moment_factory is None:
            raise ValueError("cross_fitter must provide second_moment_factory or outcome_factory.")
        covariance_tasks = [
            CrossFitTask(
                name=covariance_task_names[pair],
                target=pd.Series(
                    residuals[pair[0]] * residuals[pair[1]],
                    index=panel.entities.copy(),
                ),
                train_mask=pd.Series(
                    panel.original_cohorts == pair[0].group,
                    index=panel.entities.copy(),
                ),
                factory=second_moment_factory,
            )
            for pair in nonzero_pairs
        ]
        covariance_result = cross_fitter.fit_predict_tasks(
            X,
            tasks=covariance_tasks,
            strata=strata,
            clusters=panel.clusters,
            folds=nuisance_fold,
        )
        if not nuisance_fold.equals(covariance_result.fold):
            raise RuntimeError("CrossFitter returned inconsistent conditional-covariance folds.")
        covariance_predictions = covariance_result.predictions

    def conditional_covariance(left: _Change, right: _Change) -> np.ndarray:
        if left.is_zero or right.is_zero:
            return np.zeros(len(panel.entities))
        pair = _ordered_pair(left, right)
        if covariance_predictions is None or pair not in covariance_task_names:
            raise RuntimeError("A required conditional covariance nuisance is missing.")
        return covariance_predictions[covariance_task_names[pair]].to_numpy(dtype=float)

    n = len(panel.entities)
    cells: list[_EffectCell] = []
    weight_rows: list[dict[str, float]] = []
    candidate_arrays: list[np.ndarray] = []
    candidate_keys: list[tuple[float, float, float, float]] = []
    conditional_weight_arrays: list[np.ndarray] = []

    def odds_ratio(numerator: float, denominator: float) -> np.ndarray:
        if numerator == denominator:
            return np.ones(n)
        if nuisance_weighting == "multiclass_probabilities":
            return probability[numerator] / probability[denominator]
        try:
            return ratio_frame[(numerator, denominator)].to_numpy(dtype=float)
        except KeyError as error:
            raise RuntimeError(
                "A required ordered direct cohort-odds nuisance is missing."
            ) from error

    for cohort, baseline_position, target_position, specs in cell_definitions:
        treated_mask = panel.original_cohorts == cohort
        n_treated = int(treated_mask.sum())
        pi_treated = n_treated / n
        generated: list[np.ndarray] = []
        candidate_atts: list[float] = []
        for spec in specs:
            auxiliary_mask = panel.original_cohorts == spec.auxiliary_cohort
            treated_change_values = _change_values(panel, spec.treated_change)
            never_change_values = _change_values(panel, spec.never_change)
            auxiliary_change_values = _change_values(panel, spec.auxiliary_change)
            mean_never = conditional_mean(spec.never_change)
            mean_auxiliary = conditional_mean(spec.auxiliary_change)
            rho_never = odds_ratio(cohort, float(never_treated))
            rho_auxiliary = odds_ratio(cohort, spec.auxiliary_cohort)
            score = (
                treated_mask.astype(float)
                / pi_treated
                * (treated_change_values - mean_never - mean_auxiliary)
                - rho_never
                * panel.never_mask.astype(float)
                / pi_treated
                * (never_change_values - mean_never)
                - rho_auxiliary
                * auxiliary_mask.astype(float)
                / pi_treated
                * (auxiliary_change_values - mean_auxiliary)
            )
            generated.append(score)
            candidate_atts.append(float(score.mean()))
        generated_matrix = np.column_stack(generated)
        candidate_count = len(specs)
        if candidate_count == 1:
            weights = np.ones((n, 1))
            condition_number = 1.0
        else:
            omega = np.empty((n, candidate_count, candidate_count), dtype=float)
            treated_change = specs[0].treated_change
            treated_variance = conditional_covariance(treated_change, treated_change)
            for left_position, left in enumerate(specs):
                for right_position, right in enumerate(specs):
                    if nuisance_weighting == "multiclass_probabilities":
                        value = treated_variance / probability[cohort]
                        value = (
                            value
                            + conditional_covariance(left.never_change, right.never_change)
                            / probability[float(never_treated)]
                        )
                    else:
                        # Omega_tilde = p_g(X) * Omega. The common positive factor
                        # cancels from the normalized inverse-covariance solve.
                        value = treated_variance + odds_ratio(
                            cohort, float(never_treated)
                        ) * conditional_covariance(left.never_change, right.never_change)
                    if left.auxiliary_cohort == cohort:
                        cross_value = conditional_covariance(treated_change, left.auxiliary_change)
                        value = (
                            value - cross_value / probability[cohort]
                            if nuisance_weighting == "multiclass_probabilities"
                            else value - cross_value
                        )
                    if right.auxiliary_cohort == cohort:
                        cross_value = conditional_covariance(treated_change, right.auxiliary_change)
                        value = (
                            value - cross_value / probability[cohort]
                            if nuisance_weighting == "multiclass_probabilities"
                            else value - cross_value
                        )
                    if left.auxiliary_cohort == right.auxiliary_cohort:
                        covariance_value = conditional_covariance(
                            left.auxiliary_change, right.auxiliary_change
                        )
                        value = (
                            value + covariance_value / probability[left.auxiliary_cohort]
                            if nuisance_weighting == "multiclass_probabilities"
                            else value
                            + odds_ratio(cohort, left.auxiliary_cohort) * covariance_value
                        )
                    omega[:, left_position, right_position] = value
            omega = 0.5 * (omega + np.swapaxes(omega, 1, 2))
            eigenvalues = np.linalg.eigvalsh(omega)
            largest = eigenvalues[:, -1]
            smallest = eigenvalues[:, 0]
            invalid = (
                ~np.isfinite(eigenvalues).all(axis=1)
                | (largest <= 0.0)
                | (smallest <= singularity_tolerance * largest)
            )
            if invalid.any():
                raise ValueError(
                    "The cross-fitted conditional covariance system is singular or "
                    "numerically unidentified for at least one entity; no ridge, clipping, "
                    "or pseudoinverse was applied."
                )
            condition_number = float(np.max(largest / smallest))
            ones = np.ones((n, candidate_count))
            solved = np.linalg.solve(omega, ones[..., None])[..., 0]
            denominators = np.sum(solved, axis=1)
            if np.any(~np.isfinite(denominators)) or np.any(
                np.abs(denominators) <= np.finfo(float).eps
            ):
                raise ValueError(
                    "The conditional efficient weight normalization is singular or non-finite."
                )
            weights = solved / denominators[:, None]
        weighted_score = np.sum(weights * generated_matrix, axis=1)
        att = float(weighted_score.mean())
        efficient_influence = weighted_score - treated_mask.astype(float) / pi_treated * att
        public_time = float(panel.times[target_position])
        for position, (spec, candidate_att) in enumerate(zip(specs, candidate_atts, strict=True)):
            candidate_if = generated_matrix[:, position] - (
                treated_mask.astype(float) / pi_treated * candidate_att
            )
            realized_weights = weights[:, position]
            weight_rows.append(
                {
                    "cohort": cohort,
                    "time": public_time,
                    "auxiliary_cohort": spec.auxiliary_cohort,
                    "bridge_period": float(panel.times[spec.bridge_position]),
                    "candidate_att": candidate_att,
                    "weight": float(realized_weights.mean()),
                    "weight_min": float(realized_weights.min()),
                    "weight_max": float(realized_weights.max()),
                    "weight_std": float(realized_weights.std(ddof=0)),
                }
            )
            candidate_arrays.append(candidate_if)
            candidate_keys.append(
                (
                    cohort,
                    public_time,
                    spec.auxiliary_cohort,
                    float(panel.times[spec.bridge_position]),
                )
            )
            conditional_weight_arrays.append(realized_weights)
        cells.append(
            _EffectCell(
                cohort=cohort,
                time=public_time,
                event_time=target_position - panel.original_positions[cohort],
                base_period=float(panel.times[baseline_position]),
                att=att,
                influence=efficient_influence,
                n_treated=n_treated,
                n_comparison=int(panel.never_mask.sum()),
                cohort_share=pi_treated,
                weight_condition_number=condition_number,
            )
        )

    candidate_columns = pd.MultiIndex.from_tuples(
        candidate_keys,
        names=["cohort", "time", "auxiliary_cohort", "bridge_period"],
    )
    candidate_influence = pd.DataFrame(
        np.column_stack(candidate_arrays),
        index=panel.entities.copy(),
        columns=candidate_columns,
    )
    conditional_weights = pd.DataFrame(
        np.column_stack(conditional_weight_arrays),
        index=panel.entities.copy(),
        columns=candidate_columns.copy(),
    )
    return _assemble_result(
        cells,
        panel,
        covariance=covariance,
        method="chen_santanna_xie_efficient_covariate_adjusted",
        parallel_trends="all_conditional_on_covariates",
        control_group="never_and_admissible_auxiliary_cohorts",
        anticipation=anticipation,
        pre_periods=pre_periods,
        outcome_name=outcome,
        entity_name=entity,
        time_name=time,
        treatment_time_name=treatment_time,
        efficiency_weights=pd.DataFrame(weight_rows),
        conditional_efficiency_weights=conditional_weights,
        candidate_influence=candidate_influence,
        inference=inference,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_random_state=bootstrap_random_state,
        simultaneous_level=simultaneous_level,
        nuisance_fold=nuisance_fold,
        nuisance_weighting=nuisance_weighting,
        cohort_probabilities=probabilities,
        cohort_ratios=ratio_frame,
        cohort_ratio_diagnostics=ratio_diagnostics,
        cohort_ratio_candidate_uses=pd.DataFrame(ratio_use_rows),
        covariates=covariate_names,
        cross_fitted=True,
        pretrend_control_group=None,
        pretrend_unavailable_reason=(
            "The covariate-adjusted PT-All path requires a separately cross-fitted "
            "conditional pre-trend score; an unadjusted placebo test would target a "
            "different restriction."
        ),
        notes=(
            "All reported outcome, cohort-weighting, and conditional-covariance nuisance predictions are out of fold.",
            (
                "Cohort odds are formed from cross-fitted multiclass probabilities and are refused below the declared probability floor rather than clipped."
                if nuisance_weighting == "multiclass_probabilities"
                else "Cohort odds are fitted directly as calibrated ordered posterior odds; support failures refuse without clipping, trimming, rescaling, or multiclass fallback."
            ),
            "The conditional covariance follows Chen-Sant'Anna-Xie equation (3.12), estimated by cross-fitted residual-product regressions.",
            "Semiparametric efficiency requires PT-All and the nuisance consistency, overlap, weighting, and product-rate conditions in the paper's Assumption C.1.",
            "Higher-level clustered covariance does not claim the independent-entity semiparametric efficiency bound.",
            (
                "Event-study bands use a studentized multiplier bootstrap."
                if inference == "multiplier_bootstrap"
                else "Reported confidence intervals are pointwise."
            ),
        ),
    )


class EfficientDiD:
    """Chen-Sant'Anna-Xie efficient DiD under PT-All."""

    def __init__(
        self,
        *,
        pre_periods: PrePeriods = "all",
        anticipation: int = 0,
        covariance: DiDCovariance = "robust",
        inference: DiDInference = "analytic",
        bootstrap_iterations: int = 999,
        random_state: int | None = None,
        simultaneous_level: float = 0.95,
        singularity_tolerance: float = 1e-12,
        nuisance_probability_floor: float = 1e-6,
        nuisance_ratio_floor: float = 1e-6,
        nuisance_ratio_ceiling: float = 1e6,
        nuisance_ratio_min_effective_n: float = 2.0,
        nuisance_ratio_max_share: float = 0.8,
        nuisance_ratio_min_psus: int = 2,
    ) -> None:
        if pre_periods != "all" and (
            isinstance(pre_periods, bool)
            or not isinstance(pre_periods, Integral)
            or int(pre_periods) <= 0
        ):
            raise ValueError("pre_periods must be 'all' or a positive integer.")
        if not np.isfinite(singularity_tolerance) or not 0 < singularity_tolerance < 1:
            raise ValueError("singularity_tolerance must be finite and strictly between 0 and 1.")
        if (
            not np.isfinite(nuisance_probability_floor)
            or not 0.0 < nuisance_probability_floor < 0.5
        ):
            raise ValueError(
                "nuisance_probability_floor must be finite and strictly between zero and 0.5."
            )
        if not np.isfinite(nuisance_ratio_floor) or nuisance_ratio_floor <= 0.0:
            raise ValueError("nuisance_ratio_floor must be finite and strictly positive.")
        if (
            not np.isfinite(nuisance_ratio_ceiling)
            or nuisance_ratio_ceiling <= nuisance_ratio_floor
        ):
            raise ValueError(
                "nuisance_ratio_ceiling must be finite and exceed nuisance_ratio_floor."
            )
        if not np.isfinite(nuisance_ratio_min_effective_n) or nuisance_ratio_min_effective_n < 1.0:
            raise ValueError("nuisance_ratio_min_effective_n must be finite and at least one.")
        if not np.isfinite(nuisance_ratio_max_share) or not 0.0 < nuisance_ratio_max_share <= 1.0:
            raise ValueError("nuisance_ratio_max_share must be finite and in (0, 1].")
        if (
            isinstance(nuisance_ratio_min_psus, bool)
            or not isinstance(nuisance_ratio_min_psus, Integral)
            or int(nuisance_ratio_min_psus) < 2
        ):
            raise ValueError("nuisance_ratio_min_psus must be an integer of at least two.")
        (
            anticipation,
            covariance,
            inference,
            bootstrap_iterations,
            random_state,
            simultaneous_level,
        ) = _validate_constructor(
            anticipation=anticipation,
            covariance=covariance,
            inference=inference,
            bootstrap_iterations=bootstrap_iterations,
            random_state=random_state,
            simultaneous_level=simultaneous_level,
        )
        self.pre_periods: PrePeriods = "all" if pre_periods == "all" else int(pre_periods)
        self.anticipation: int = anticipation
        self.covariance: DiDCovariance = covariance
        self.inference: DiDInference = inference
        self.bootstrap_iterations: int = bootstrap_iterations
        self.random_state: int | None = random_state
        self.simultaneous_level: float = simultaneous_level
        self.singularity_tolerance: float = float(singularity_tolerance)
        self.nuisance_probability_floor: float = float(nuisance_probability_floor)
        self.nuisance_ratio_floor: float = float(nuisance_ratio_floor)
        self.nuisance_ratio_ceiling: float = float(nuisance_ratio_ceiling)
        self.nuisance_ratio_min_effective_n: float = float(nuisance_ratio_min_effective_n)
        self.nuisance_ratio_max_share: float = float(nuisance_ratio_max_share)
        self.nuisance_ratio_min_psus: int = int(nuisance_ratio_min_psus)

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        entity: str,
        time: str,
        treatment_time: str,
        never_treated: float = np.inf,
        cluster: str | None = None,
        covariates: Sequence[str] | None = None,
        cross_fitter: CrossFitter | None = None,
    ) -> DiDResult:
        """Estimate efficient group-time effects and their aggregations."""

        if covariates is not None and cross_fitter is None:
            raise ValueError(
                "cross_fitter is required when covariates are supplied; EfficientDiD does "
                "not own or fit nuisance model classes."
            )
        if covariates is None and cross_fitter is not None:
            raise ValueError("cross_fitter is only used when covariates are supplied.")
        panel = _prepare_panel(
            data,
            outcome=outcome,
            entity=entity,
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
            return _fit_covariate_efficient(
                data,
                panel,
                outcome=outcome,
                entity=entity,
                time=time,
                treatment_time=treatment_time,
                never_treated=float(never_treated),
                covariates=covariates,
                cross_fitter=cross_fitter,
                pre_periods=self.pre_periods,
                anticipation=self.anticipation,
                covariance=self.covariance,
                inference=self.inference,
                bootstrap_iterations=self.bootstrap_iterations,
                bootstrap_random_state=self.random_state,
                simultaneous_level=self.simultaneous_level,
                singularity_tolerance=self.singularity_tolerance,
                nuisance_probability_floor=self.nuisance_probability_floor,
                nuisance_ratio_floor=self.nuisance_ratio_floor,
                nuisance_ratio_ceiling=self.nuisance_ratio_ceiling,
                nuisance_ratio_min_effective_n=self.nuisance_ratio_min_effective_n,
                nuisance_ratio_max_share=self.nuisance_ratio_max_share,
                nuisance_ratio_min_psus=self.nuisance_ratio_min_psus,
            )
        n = len(panel.entities)
        pi_never = float(panel.never_mask.mean())
        cells: list[_EffectCell] = []
        weight_rows: list[dict[str, float]] = []
        candidate_arrays: list[np.ndarray] = []
        candidate_keys: list[tuple[float, float, float, float]] = []
        conditional_weight_arrays: list[np.ndarray] = []

        for cohort in panel.treated_cohorts:
            treated_mask = panel.original_cohorts == cohort
            n_treated = int(treated_mask.sum())
            pi_treated = n_treated / n
            effective_position = panel.effective_positions[cohort]
            if self.pre_periods == "all":
                baseline_position = 0
            else:
                baseline_position = effective_position - int(self.pre_periods)
                if baseline_position < 0:
                    raise ValueError(
                        f"pre_periods={self.pre_periods} exceeds the available clean history "
                        f"for cohort {cohort}."
                    )
            for target_position in range(effective_position, len(panel.times)):
                candidate_atts: list[float] = []
                candidate_influences: list[np.ndarray] = []
                candidate_metadata: list[tuple[float, float]] = []
                treated_long_change = (
                    panel.outcomes[:, target_position] - panel.outcomes[:, baseline_position]
                )
                for auxiliary_cohort in panel.treated_cohorts:
                    auxiliary_position = panel.effective_positions[auxiliary_cohort]
                    bridge_start = (
                        baseline_position if auxiliary_cohort == cohort else baseline_position + 1
                    )
                    for bridge_position in range(bridge_start, auxiliary_position):
                        auxiliary_mask = panel.original_cohorts == auxiliary_cohort
                        pi_auxiliary = float(auxiliary_mask.mean())
                        never_change = (
                            panel.outcomes[:, target_position] - panel.outcomes[:, bridge_position]
                        )
                        auxiliary_change = (
                            panel.outcomes[:, bridge_position]
                            - panel.outcomes[:, baseline_position]
                        )
                        mean_never_change = float(never_change[panel.never_mask].mean())
                        mean_auxiliary_change = float(auxiliary_change[auxiliary_mask].mean())
                        candidate_att = float(
                            treated_long_change[treated_mask].mean()
                            - mean_never_change
                            - mean_auxiliary_change
                        )
                        generated_outcome = (
                            treated_mask.astype(float)
                            / pi_treated
                            * (treated_long_change - mean_never_change - mean_auxiliary_change)
                            - panel.never_mask.astype(float)
                            / pi_never
                            * (never_change - mean_never_change)
                            - auxiliary_mask.astype(float)
                            / pi_auxiliary
                            * (auxiliary_change - mean_auxiliary_change)
                        )
                        influence = (
                            generated_outcome
                            - treated_mask.astype(float) / pi_treated * candidate_att
                        )
                        candidate_atts.append(candidate_att)
                        candidate_influences.append(influence)
                        candidate_metadata.append(
                            (
                                auxiliary_cohort,
                                float(panel.times[bridge_position]),
                            )
                        )
                if not candidate_influences:
                    raise ValueError(
                        f"No admissible PT-All generated outcomes exist for cohort {cohort}."
                    )
                influence_matrix = np.vstack(candidate_influences)
                if len(candidate_influences) == 1:
                    weights = np.ones(1)
                    condition_number = 1.0
                else:
                    covariance_matrix = np.atleast_2d(np.cov(influence_matrix, ddof=1))
                    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
                    largest = float(eigenvalues[-1])
                    smallest = float(eigenvalues[0])
                    if (
                        not np.isfinite(eigenvalues).all()
                        or largest <= 0.0
                        or smallest <= self.singularity_tolerance * largest
                    ):
                        raise ValueError(
                            "The efficient influence-function covariance system is singular "
                            "or numerically unidentified; no ridge or pseudoinverse was applied."
                        )
                    condition_number = largest / smallest
                    ones = np.ones(len(candidate_influences))
                    solved = np.linalg.solve(covariance_matrix, ones)
                    denominator = float(ones @ solved)
                    if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps:
                        raise ValueError(
                            "The efficient weight normalization is singular or non-finite."
                        )
                    weights = solved / denominator
                att = float(weights @ np.asarray(candidate_atts))
                efficient_influence = weights @ influence_matrix
                public_time = float(panel.times[target_position])
                for weight, candidate_att, metadata, candidate_if in zip(
                    weights,
                    candidate_atts,
                    candidate_metadata,
                    candidate_influences,
                    strict=True,
                ):
                    auxiliary_cohort, bridge_period = metadata
                    weight_rows.append(
                        {
                            "cohort": cohort,
                            "time": public_time,
                            "auxiliary_cohort": auxiliary_cohort,
                            "bridge_period": bridge_period,
                            "candidate_att": candidate_att,
                            "weight": float(weight),
                        }
                    )
                    candidate_arrays.append(candidate_if)
                    candidate_keys.append((cohort, public_time, auxiliary_cohort, bridge_period))
                    conditional_weight_arrays.append(np.full(n, float(weight)))
                cells.append(
                    _EffectCell(
                        cohort=cohort,
                        time=public_time,
                        event_time=target_position - panel.original_positions[cohort],
                        base_period=float(panel.times[baseline_position]),
                        att=att,
                        influence=efficient_influence,
                        n_treated=n_treated,
                        n_comparison=int(panel.never_mask.sum()),
                        cohort_share=pi_treated,
                        weight_condition_number=condition_number,
                    )
                )

        efficiency_weights = pd.DataFrame(weight_rows)
        candidate_columns = pd.MultiIndex.from_tuples(
            candidate_keys,
            names=["cohort", "time", "auxiliary_cohort", "bridge_period"],
        )
        candidate_influence = pd.DataFrame(
            np.column_stack(candidate_arrays),
            index=panel.entities.copy(),
            columns=candidate_columns,
        )
        conditional_weights = pd.DataFrame(
            np.column_stack(conditional_weight_arrays),
            index=panel.entities.copy(),
            columns=candidate_columns.copy(),
        )
        return _assemble_result(
            cells,
            panel,
            covariance=self.covariance,
            method="chen_santanna_xie_efficient",
            parallel_trends="all",
            control_group="never_and_admissible_auxiliary_cohorts",
            anticipation=self.anticipation,
            pre_periods=self.pre_periods,
            outcome_name=outcome,
            entity_name=entity,
            time_name=time,
            treatment_time_name=treatment_time,
            efficiency_weights=efficiency_weights,
            conditional_efficiency_weights=conditional_weights,
            candidate_influence=candidate_influence,
            inference=self.inference,
            bootstrap_iterations=self.bootstrap_iterations,
            bootstrap_random_state=self.random_state,
            simultaneous_level=self.simultaneous_level,
            nuisance_fold=pd.Series(dtype="int64", name="fold"),
            nuisance_weighting="none",
            cohort_probabilities=pd.DataFrame(index=panel.entities.copy()),
            cohort_ratios=pd.DataFrame(index=panel.entities.copy()),
            cohort_ratio_diagnostics=pd.DataFrame(),
            cohort_ratio_candidate_uses=pd.DataFrame(),
            covariates=(),
            cross_fitted=False,
            pretrend_control_group="never_treated",
            pretrend_unavailable_reason=None,
            notes=(
                "Efficiency is claimed only for the no-covariate PT-All short-panel model implemented here.",
                "Efficiency weights may be negative because they combine homogeneous identifying moments.",
                "Higher-level clustered covariance does not claim the independent-entity semiparametric efficiency bound.",
                (
                    "Event-study bands use a studentized entity-level multiplier bootstrap."
                    if self.inference == "multiplier_bootstrap"
                    else "Reported confidence intervals are pointwise."
                ),
            ),
        )


__all__ = [
    "DiDHausmanDiagnostic",
    "DiDPretrendDiagnostic",
    "DiDResult",
    "DifferenceInDifferences",
    "EfficientDiD",
    "did_hausman_test",
]

"""Static fixed-effects panel instrumental-variables estimation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2, f, norm, t

from ._covariance import CovarianceEstimate, CovarianceType
from ._data import MissingPolicy
from .diagnostics import FirstStageDiagnostic, SarganTest


@dataclass(frozen=True)
class _PreparedPanelIV:
    outcome: np.ndarray
    structural: np.ndarray
    instruments: np.ndarray
    entity_codes: np.ndarray
    time_codes: np.ndarray
    cluster_codes: np.ndarray | None
    cluster_labels: np.ndarray | None
    panel_index: pd.MultiIndex
    source_index: pd.Index
    outcome_name: str
    endogenous_names: tuple[str, ...]
    exogenous_names: tuple[str, ...]
    instrument_names: tuple[str, ...]
    entity_name: str
    time_name: str
    cluster_name: str | None
    n_entities: int
    n_periods: int
    n_clusters: int | None
    balanced: bool
    dropped_rows: int


@dataclass(frozen=True)
class _WithinResult:
    values: np.ndarray
    iterations: int
    converged: bool
    max_abs_group_mean: float


@dataclass(frozen=True)
class PanelIV2SLSResult:
    """Fitted fixed-effects Panel IV/2SLS result with labelled inference."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    test_statistics: pd.Series
    pvalues: pd.Series
    residuals: pd.Series
    fitted_values: pd.Series
    within_fitted_values: pd.Series
    absorbed_effects: pd.Series
    outcome: pd.Series
    first_stage: dict[str, FirstStageDiagnostic]
    overidentification: SarganTest | None
    instrument_variation: pd.DataFrame
    nobs: int
    n_entities: int
    n_periods: int
    balanced: bool
    rank: int
    absorbed_rank: int
    df_resid: int
    covariance_type: CovarianceType
    inference_distribution: str
    inference_df: float | None
    n_clusters: int | None
    cluster_name: str | None
    effects: tuple[str, ...]
    entity_name: str
    time_name: str
    y_name: str
    endogenous_names: tuple[str, ...]
    exogenous_names: tuple[str, ...]
    instrument_names: tuple[str, ...]
    estimation_index: pd.MultiIndex
    source_index: pd.Index
    dropped_rows: int
    within_r_squared: float
    within_iterations: int
    within_converged: bool
    within_max_abs_group_mean: float
    normal_matrix_condition: float
    notes: tuple[str, ...]
    assumptions: tuple[str, ...]
    converged: bool = True
    backend: str = "native-within-panel-2sls"

    @property
    def all_params(self) -> pd.Series:
        return self.params.copy()

    @property
    def coefficients(self) -> pd.Series:
        return self.params.copy()

    @property
    def zstats(self) -> pd.Series:
        """Compatibility alias; clustered and unadjusted fits use t inference."""

        return self.test_statistics.copy()

    @property
    def tstats(self) -> pd.Series:
        return self.test_statistics.copy()

    @property
    def causal_interpretation(self) -> str:
        return (
            "Panel IV coefficients are causal only when instrument relevance, conditional "
            "independence, exclusion, correct linear specification, and the intended "
            "estimand's assumptions remain credible after the declared fixed effects. "
            "Fixed effects do not repair time-varying confounding or invalid instruments."
        )

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        probability = 0.5 + level / 2.0
        if self.inference_distribution == "normal":
            critical = float(norm.ppf(probability))
        else:
            if self.inference_df is None or self.inference_df <= 0:
                raise ValueError("Positive inference degrees of freedom are required.")
            critical = float(t.ppf(probability, self.inference_df))
        return pd.DataFrame(
            {
                "lower": self.params - critical * self.standard_errors,
                "upper": self.params + critical * self.standard_errors,
            }
        )

    def confint(self, alpha: float = 0.05) -> pd.DataFrame:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one.")
        return self.conf_int(level=1.0 - alpha)

    def summary_frame(self, level: float = 0.95) -> pd.DataFrame:
        intervals = self.conf_int(level=level)
        frame = pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.standard_errors,
                "stat": self.test_statistics,
                "p_value": self.pvalues,
                "ci_lower": intervals["lower"],
                "ci_upper": intervals["upper"],
            }
        )
        frame.attrs["statistic_distribution"] = self.inference_distribution
        frame.attrs["statistic_df"] = self.inference_df
        return frame

    def panel_summary(self) -> pd.Series:
        return pd.Series(
            {
                "observations": self.nobs,
                "entities": self.n_entities,
                "periods": self.n_periods,
                "balanced": self.balanced,
                "effects": "+".join(self.effects),
                "absorbed_rank": self.absorbed_rank,
                "residual_df": self.df_resid,
                "cluster": self.cluster_name,
                "clusters": self.n_clusters,
                "within_iterations": self.within_iterations,
                "within_converged": self.within_converged,
                "within_max_abs_group_mean": self.within_max_abs_group_mean,
            },
            name="panel_iv_design",
        )

    def to_markdown(self, digits: int = 4) -> str:
        if digits < 0:
            raise ValueError("digits must be non-negative.")
        lines = [
            "# Panel IV/2SLS result",
            "",
            f"- Outcome: `{self.y_name}`",
            f"- Effects: `{'+'.join(self.effects)}`",
            f"- Observations: `{self.nobs}`",
            f"- Entities: `{self.n_entities}`",
            f"- Periods: `{self.n_periods}`",
            f"- Covariance: `{self.covariance_type}`",
            f"- Cluster: `{self.cluster_name}`",
            f"- Inference: `{self.inference_distribution}`",
            "",
            "| term | coef | std_err | stat | p_value |",
            "|---|---:|---:|---:|---:|",
        ]
        for term, row in self.summary_frame().iterrows():
            lines.append(
                f"| {term} | {row['coef']:.{digits}f} | {row['std_err']:.{digits}f} | "
                f"{row['stat']:.{digits}f} | {row['p_value']:.{digits}f} |"
            )
        lines.extend(["", f"> {self.causal_interpretation}"])
        return "\n".join(lines)


def _names(value: str | Sequence[str] | None, *, role: str, allow_empty: bool) -> tuple[str, ...]:
    if value is None:
        if allow_empty:
            return ()
        raise ValueError(f"{role} must contain at least one column name.")
    names: tuple[str, ...]
    if isinstance(value, str):
        names = (value,)
    else:
        try:
            names = tuple(value)
        except TypeError as error:
            raise TypeError(f"{role} must be a column name or sequence of column names.") from error
    if not names and not allow_empty:
        raise ValueError(f"{role} must contain at least one column name.")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError(f"{role} names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{role} names must be distinct.")
    return names


def _group_means(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    totals = np.zeros((n_groups, values.shape[1]), dtype=float)
    np.add.at(totals, codes, values)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    return totals / counts[:, None]


def _demean(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    return values - _group_means(values, codes, n_groups)[codes]


def _max_abs_group_mean(
    values: np.ndarray,
    entity_codes: np.ndarray,
    n_entities: int,
    time_codes: np.ndarray,
    n_periods: int,
    *,
    time_effects: bool,
) -> float:
    entity_max = float(np.max(np.abs(_group_means(values, entity_codes, n_entities))))
    if not time_effects:
        return entity_max
    time_max = float(np.max(np.abs(_group_means(values, time_codes, n_periods))))
    return max(entity_max, time_max)


def _within_transform(
    values: np.ndarray,
    entity_codes: np.ndarray,
    n_entities: int,
    time_codes: np.ndarray,
    n_periods: int,
    *,
    time_effects: bool,
    balanced: bool,
    tolerance: float = 1e-12,
    max_iterations: int = 10_000,
) -> _WithinResult:
    if not time_effects:
        transformed = _demean(values, entity_codes, n_entities)
        maximum = _max_abs_group_mean(
            transformed,
            entity_codes,
            n_entities,
            time_codes,
            n_periods,
            time_effects=False,
        )
        return _WithinResult(transformed, 1, True, maximum)

    if balanced:
        transformed = (
            values
            - _group_means(values, entity_codes, n_entities)[entity_codes]
            - _group_means(values, time_codes, n_periods)[time_codes]
            + values.mean(axis=0)
        )
        maximum = _max_abs_group_mean(
            transformed,
            entity_codes,
            n_entities,
            time_codes,
            n_periods,
            time_effects=True,
        )
        return _WithinResult(transformed, 1, True, maximum)

    transformed = values.copy()
    converged = False
    iteration = 0
    for current_iteration in range(1, max_iterations + 1):
        iteration = current_iteration
        previous = transformed
        transformed = _demean(previous, entity_codes, n_entities)
        transformed = _demean(transformed, time_codes, n_periods)
        scale = max(1.0, float(np.max(np.abs(previous))))
        if float(np.max(np.abs(transformed - previous))) <= tolerance * scale:
            converged = True
            break
    maximum = _max_abs_group_mean(
        transformed,
        entity_codes,
        n_entities,
        time_codes,
        n_periods,
        time_effects=True,
    )
    if maximum > 1e-10:
        converged = False
    return _WithinResult(transformed, iteration, converged, maximum)


def _two_way_connected(
    entity_codes: np.ndarray,
    time_codes: np.ndarray,
    n_entities: int,
    n_periods: int,
) -> bool:
    left = entity_codes
    right = n_entities + time_codes
    rows = np.concatenate([left, right])
    columns = np.concatenate([right, left])
    graph = coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, columns)),
        shape=(n_entities + n_periods, n_entities + n_periods),
    ).tocsr()
    components, _ = connected_components(graph, directed=False, return_labels=True)
    return bool(components == 1)


def _prepare_panel_iv(
    data: Any,
    *,
    outcome: str,
    endogenous: str | Sequence[str],
    instruments: str | Sequence[str],
    exogenous: str | Sequence[str] | None,
    entity: str,
    time: str,
    cluster: str | None,
    covariance: CovarianceType,
    missing: MissingPolicy,
    time_effects: bool,
) -> _PreparedPanelIV:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame in long panel form.")
    if not data.columns.is_unique:
        raise ValueError("data column names must be unique.")
    for role, name in (("outcome", outcome), ("entity", entity), ("time", time)):
        if not isinstance(name, str) or not name:
            raise TypeError(f"{role} must be a non-empty column name.")
    endogenous_names = _names(endogenous, role="endogenous", allow_empty=False)
    exogenous_names = _names(exogenous, role="exogenous", allow_empty=True)
    instrument_names = _names(instruments, role="instruments", allow_empty=False)

    model_roles = (outcome, *exogenous_names, *endogenous_names, *instrument_names, entity, time)
    if len(set(model_roles)) != len(model_roles):
        raise ValueError(
            "Outcome, exogenous, endogenous, excluded-instrument, entity, and time roles "
            "must be distinct; instruments contain excluded instruments only."
        )
    if cluster is not None:
        if not isinstance(cluster, str) or not cluster:
            raise TypeError("cluster must be a non-empty column name or None.")
        if covariance != "clustered":
            raise ValueError("cluster may be supplied only when covariance='clustered'.")
        if cluster != entity and cluster in model_roles:
            raise ValueError("A higher-level cluster column must be distinct from model roles.")

    required = list(dict.fromkeys([*model_roles, *([] if cluster is None else [cluster])]))
    missing_columns = [name for name in required if name not in data]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    if data[entity].isna().any():
        raise ValueError("entity identifiers must not contain missing values.")
    if data[time].isna().any():
        raise ValueError("time identifiers must not contain missing values.")
    if data.duplicated([entity, time]).any():
        raise ValueError("data must contain exactly one row per entity-time pair.")

    try:
        working = data.loc[:, required].sort_values([entity, time], kind="mergesort").copy()
    except TypeError as error:
        raise ValueError(
            "entity and time identifiers must have deterministic sortable values."
        ) from error

    numeric_names = (outcome, *exogenous_names, *endogenous_names, *instrument_names)
    for name in numeric_names:
        try:
            working[name] = pd.to_numeric(working[name], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain only numeric values.") from error

    numeric = working.loc[:, numeric_names].to_numpy(dtype=float)
    invalid = ~np.isfinite(numeric).all(axis=1)
    if cluster is not None:
        invalid |= working[cluster].isna().to_numpy()
    dropped_rows = int(invalid.sum())
    if dropped_rows and missing == "raise":
        raise ValueError(
            f"Inputs contain missing or non-finite values in {dropped_rows} row(s); "
            "set missing='drop' to remove them jointly."
        )
    if dropped_rows:
        working = working.loc[~invalid].copy()
    if working.empty:
        raise ValueError("No complete observations remain after applying the missing-value policy.")

    counts = working.groupby(entity, sort=False, observed=True).size()
    n_entities = int(counts.size)
    if n_entities < 2:
        raise ValueError("Panel IV requires at least two entities.")
    if int(counts.min()) < 2:
        raise ValueError("Every entity must have at least two retained observations.")
    n_periods = int(working[time].nunique())
    if time_effects and n_periods < 2:
        raise ValueError("Time fixed effects require at least two observed periods.")

    entity_codes, entity_labels = pd.factorize(working[entity], sort=False)
    time_codes, _ = pd.factorize(working[time], sort=False)
    entity_codes = np.asarray(entity_codes, dtype=int)
    time_codes = np.asarray(time_codes, dtype=int)
    balanced = bool(len(working) == n_entities * n_periods)
    if time_effects and not _two_way_connected(entity_codes, time_codes, n_entities, n_periods):
        raise ValueError("Two-way fixed effects require a connected entity-time incidence graph.")

    cluster_codes: np.ndarray | None = None
    cluster_labels: np.ndarray | None = None
    cluster_name: str | None = None
    n_clusters: int | None = None
    if covariance == "clustered":
        cluster_name = entity if cluster is None else cluster
        if cluster_name != entity:
            nested = working.groupby(entity, sort=False, observed=True)[cluster_name].nunique(
                dropna=False
            )
            if bool((nested != 1).any()):
                raise ValueError("A higher-level cluster must be constant within entity.")
        cluster_codes_raw, cluster_labels_index = pd.factorize(working[cluster_name], sort=False)
        cluster_codes = np.asarray(cluster_codes_raw, dtype=int)
        cluster_labels = np.asarray(cluster_labels_index, dtype=object)
        n_clusters = int(len(cluster_labels))
        if n_clusters < 2:
            raise ValueError("Clustered covariance requires at least two clusters.")

    numeric = working.loc[:, numeric_names].to_numpy(dtype=float)
    outcome_values = numeric[:, 0]
    exogenous_stop = 1 + len(exogenous_names)
    endogenous_stop = exogenous_stop + len(endogenous_names)
    exogenous_values = numeric[:, 1:exogenous_stop]
    endogenous_values = numeric[:, exogenous_stop:endogenous_stop]
    instrument_values = numeric[:, endogenous_stop:]
    structural = np.column_stack([exogenous_values, endogenous_values])
    full_instruments = np.column_stack([exogenous_values, instrument_values])
    panel_index = pd.MultiIndex.from_arrays(
        [working[entity].to_numpy(), working[time].to_numpy()], names=[entity, time]
    )
    if not panel_index.is_unique:  # defensive; checked before sorting/filtering
        raise RuntimeError("Internal panel index is not unique.")
    if len(entity_labels) != n_entities:  # defensive
        raise RuntimeError("Internal entity factorization is inconsistent.")
    return _PreparedPanelIV(
        outcome=outcome_values,
        structural=structural,
        instruments=full_instruments,
        entity_codes=entity_codes,
        time_codes=time_codes,
        cluster_codes=cluster_codes,
        cluster_labels=cluster_labels,
        panel_index=panel_index,
        source_index=working.index.copy(),
        outcome_name=outcome,
        endogenous_names=endogenous_names,
        exogenous_names=exogenous_names,
        instrument_names=instrument_names,
        entity_name=entity,
        time_name=time,
        cluster_name=cluster_name,
        n_entities=n_entities,
        n_periods=n_periods,
        n_clusters=n_clusters,
        balanced=balanced,
        dropped_rows=dropped_rows,
    )


def _inverse(matrix: np.ndarray, *, name: str) -> np.ndarray:
    try:
        inverse = np.linalg.solve(matrix, np.eye(matrix.shape[0], dtype=float))
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} is singular; the model is not numerically identified.") from error
    inverse = 0.5 * (inverse + inverse.T)
    if not np.isfinite(inverse).all():
        raise FloatingPointError(f"{name} inverse produced non-finite values.")
    return inverse


def _score_covariance(
    bread: np.ndarray,
    scores: np.ndarray,
    residuals: np.ndarray,
    *,
    covariance: CovarianceType,
    df_resid: int,
    cluster_codes: np.ndarray | None,
    n_clusters: int | None,
) -> CovarianceEstimate:
    nobs = scores.shape[0]
    if df_resid <= 0:
        raise ValueError("Residual degrees of freedom must be positive after fixed effects.")
    if covariance == "unadjusted":
        matrix = float(residuals @ residuals) / df_resid * bread
        return CovarianceEstimate(0.5 * (matrix + matrix.T), "t", float(df_resid), None)
    if covariance == "robust":
        meat = (nobs / df_resid) * (scores.T @ scores)
        matrix = bread @ meat @ bread
        return CovarianceEstimate(0.5 * (matrix + matrix.T), "normal", None, None)
    if cluster_codes is None or n_clusters is None:  # defensive
        raise RuntimeError("Internal clustered-covariance state is incomplete.")
    grouped = np.zeros((n_clusters, scores.shape[1]), dtype=float)
    np.add.at(grouped, cluster_codes, scores)
    correction = (n_clusters / (n_clusters - 1.0)) * ((nobs - 1.0) / df_resid)
    meat = correction * (grouped.T @ grouped)
    matrix = bread @ meat @ bread
    return CovarianceEstimate(0.5 * (matrix + matrix.T), "t", float(n_clusters - 1), n_clusters)


def _ols_covariance_with_effects(
    design: np.ndarray,
    residuals: np.ndarray,
    *,
    covariance: CovarianceType,
    absorbed_rank: int,
    cluster_codes: np.ndarray | None,
    n_clusters: int | None,
) -> CovarianceEstimate:
    bread = _inverse(design.T @ design, name="first-stage normal matrix")
    df_resid = len(residuals) - absorbed_rank - design.shape[1]
    return _score_covariance(
        bread,
        design * residuals[:, None],
        residuals,
        covariance=covariance,
        df_resid=df_resid,
        cluster_codes=cluster_codes,
        n_clusters=n_clusters,
    )


def _first_stage_diagnostics(
    endogenous: np.ndarray,
    full_instruments: np.ndarray,
    included_exogenous: np.ndarray,
    *,
    endogenous_names: tuple[str, ...],
    excluded_count: int,
    covariance: CovarianceType,
    absorbed_rank: int,
    cluster_codes: np.ndarray | None,
    n_clusters: int | None,
) -> dict[str, FirstStageDiagnostic]:
    nobs = len(endogenous)
    unrestricted_rank = full_instruments.shape[1]
    denominator_df = nobs - absorbed_rank - unrestricted_rank
    if denominator_df <= 0:
        raise ValueError("First-stage residual degrees of freedom must be positive.")
    ztz_inverse = _inverse(full_instruments.T @ full_instruments, name="instrument matrix")
    restricted_inverse = (
        _inverse(included_exogenous.T @ included_exogenous, name="exogenous matrix")
        if included_exogenous.shape[1]
        else None
    )
    diagnostics: dict[str, FirstStageDiagnostic] = {}
    for column, name in enumerate(endogenous_names):
        target = endogenous[:, column]
        coefficients = ztz_inverse @ full_instruments.T @ target
        unrestricted_residuals = target - full_instruments @ coefficients
        unrestricted_sse = float(unrestricted_residuals @ unrestricted_residuals)
        if restricted_inverse is None:
            restricted_sse = float(target @ target)
        else:
            restricted_coefficients = restricted_inverse @ included_exogenous.T @ target
            restricted_residuals = target - included_exogenous @ restricted_coefficients
            restricted_sse = float(restricted_residuals @ restricted_residuals)
        total_squares = float(target @ target)
        improvement = max(restricted_sse - unrestricted_sse, 0.0)
        r_squared = 1.0 - unrestricted_sse / total_squares if total_squares > 0 else float("nan")
        partial_r_squared = improvement / restricted_sse if restricted_sse > 0 else float("nan")
        denominator = unrestricted_sse / denominator_df
        classical_f = (
            (improvement / excluded_count) / denominator if denominator > 0 else float("inf")
        )
        classical_p = float(f.sf(classical_f, excluded_count, denominator_df))

        covariance_estimate = _ols_covariance_with_effects(
            full_instruments,
            unrestricted_residuals,
            covariance=covariance,
            absorbed_rank=absorbed_rank,
            cluster_codes=cluster_codes,
            n_clusters=n_clusters,
        )
        excluded_coefficients = coefficients[-excluded_count:]
        excluded_covariance = covariance_estimate.matrix[-excluded_count:, -excluded_count:]
        if np.linalg.matrix_rank(excluded_covariance) < excluded_count:
            if float(np.max(np.abs(excluded_covariance))) <= np.finfo(float).eps:
                wald = float("inf")
            else:
                raise ValueError(
                    f"The excluded-instrument covariance is singular for first stage {name}."
                )
        else:
            wald = float(
                excluded_coefficients @ np.linalg.solve(excluded_covariance, excluded_coefficients)
            )
        if covariance_estimate.distribution == "normal":
            statistic = wald
            p_value = float(chi2.sf(wald, excluded_count))
            distribution = "chi2"
        else:
            statistic = wald / excluded_count
            assert covariance_estimate.df is not None
            p_value = float(f.sf(statistic, excluded_count, covariance_estimate.df))
            distribution = "F"
        diagnostics[name] = FirstStageDiagnostic(
            endogenous=name,
            r_squared=float(r_squared),
            partial_r_squared=float(partial_r_squared),
            classical_f_statistic=float(classical_f),
            classical_f_df_num=int(excluded_count),
            classical_f_df_denom=int(denominator_df),
            classical_f_p_value=classical_p,
            excluded_instrument_statistic=float(statistic),
            excluded_instrument_df=int(excluded_count),
            excluded_instrument_p_value=p_value,
            excluded_instrument_distribution=distribution,
            weak_instrument_warning=bool(np.isfinite(classical_f) and classical_f < 10.0),
        )
    return diagnostics


def _sargan(residuals: np.ndarray, instruments: np.ndarray, df: int) -> SarganTest:
    sigma2 = float(residuals @ residuals / len(residuals))
    if sigma2 <= 0:
        statistic = 0.0
    else:
        moment = instruments.T @ residuals
        statistic = float(moment @ np.linalg.solve(instruments.T @ instruments, moment) / sigma2)
    return SarganTest(statistic=statistic, df=df, p_value=float(chi2.sf(statistic, df)))


class PanelIV2SLS:
    """Static entity-fixed-effects Panel IV/2SLS.

    Entity fixed effects are mandatory. Time fixed effects are optional and enabled by
    default. Clustered inference defaults to entity clusters. Excluded instruments are
    supplied explicitly and must retain variation after the requested absorption.
    """

    def __init__(
        self,
        *,
        covariance: CovarianceType = "clustered",
        time_effects: bool = True,
        missing: MissingPolicy = "raise",
    ) -> None:
        if covariance not in {"unadjusted", "robust", "clustered"}:
            raise ValueError("covariance must be 'unadjusted', 'robust', or 'clustered'.")
        if not isinstance(time_effects, bool):
            raise TypeError("time_effects must be a boolean.")
        if missing not in {"raise", "drop"}:
            raise ValueError("missing must be 'raise' or 'drop'.")
        self.covariance = covariance
        self.time_effects = time_effects
        self.missing = missing

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        endogenous: str | Sequence[str],
        instruments: str | Sequence[str],
        entity: str,
        time: str,
        exogenous: str | Sequence[str] | None = None,
        cluster: str | None = None,
    ) -> PanelIV2SLSResult:
        """Fit fixed-effects Panel IV from a long-form panel DataFrame."""

        prepared = _prepare_panel_iv(
            data,
            outcome=outcome,
            endogenous=endogenous,
            instruments=instruments,
            exogenous=exogenous,
            entity=entity,
            time=time,
            cluster=cluster,
            covariance=self.covariance,
            missing=self.missing,
            time_effects=self.time_effects,
        )
        all_values = np.column_stack(
            [
                prepared.outcome,
                prepared.structural,
                prepared.instruments[:, -len(prepared.instrument_names) :],
            ]
        )
        within = _within_transform(
            all_values,
            prepared.entity_codes,
            prepared.n_entities,
            prepared.time_codes,
            prepared.n_periods,
            time_effects=self.time_effects,
            balanced=prepared.balanced,
        )
        if not within.converged:
            raise ValueError(
                "Fixed-effect alternating projections did not converge to the declared tolerance."
            )
        transformed = within.values
        y_within = transformed[:, 0]
        n_structural = prepared.structural.shape[1]
        structural = transformed[:, 1 : 1 + n_structural]
        excluded = transformed[:, 1 + n_structural :]
        n_exogenous = len(prepared.exogenous_names)
        exogenous_within = structural[:, :n_exogenous]
        endogenous_within = structural[:, n_exogenous:]
        full_instruments = np.column_stack([exogenous_within, excluded])
        structural_names = (*prepared.exogenous_names, *prepared.endogenous_names)

        variation_rows: list[dict[str, float | str]] = []
        raw_endogenous = prepared.structural[:, n_exogenous:]
        for position, name in enumerate(prepared.endogenous_names):
            raw = raw_endogenous[:, position]
            adjusted = endogenous_within[:, position]
            variation_rows.append(
                {
                    "variable": name,
                    "role": "endogenous",
                    "raw_std": float(np.std(raw)),
                    "within_std": float(np.sqrt(np.mean(adjusted**2))),
                    "within_norm": float(np.linalg.norm(adjusted)),
                }
            )
        raw_excluded = prepared.instruments[:, -len(prepared.instrument_names) :]
        for position, name in enumerate(prepared.instrument_names):
            raw = raw_excluded[:, position]
            adjusted = excluded[:, position]
            variation_rows.append(
                {
                    "variable": name,
                    "role": "excluded_instrument",
                    "raw_std": float(np.std(raw)),
                    "within_std": float(np.sqrt(np.mean(adjusted**2))),
                    "within_norm": float(np.linalg.norm(adjusted)),
                }
            )
        instrument_variation = pd.DataFrame(variation_rows).set_index("variable")
        for row in variation_rows:
            raw_scale = max(1.0, float(row["raw_std"]) * np.sqrt(len(y_within)))
            if float(row["within_norm"]) <= 1e-12 * raw_scale:
                raise ValueError(
                    f"{row['role']} {row['variable']} is absorbed by the requested fixed effects."
                )

        nobs = len(y_within)
        n_parameters = structural.shape[1]
        n_instruments = full_instruments.shape[1]
        n_endogenous = endogenous_within.shape[1]
        n_excluded = excluded.shape[1]
        absorbed_rank = (
            prepared.n_entities + prepared.n_periods - 1
            if self.time_effects
            else prepared.n_entities
        )
        df_resid = nobs - absorbed_rank - n_parameters
        if n_excluded < n_endogenous:
            raise ValueError(
                "The model is underidentified: excluded instruments must be at least as "
                "numerous as endogenous regressors."
            )
        if df_resid <= 0:
            raise ValueError(
                "The number of observations must exceed structural plus absorbed rank."
            )
        if nobs - absorbed_rank - n_instruments <= 0:
            raise ValueError(
                "The number of observations must exceed instrument plus absorbed rank."
            )
        structural_rank = int(np.linalg.matrix_rank(structural))
        if structural_rank < n_parameters:
            raise ValueError(
                "The fixed-effect-adjusted structural design is rank deficient; remove "
                "collinear or absorbed regressors."
            )
        instrument_rank = int(np.linalg.matrix_rank(full_instruments))
        if instrument_rank < n_instruments:
            raise ValueError(
                "The fixed-effect-adjusted instrument design is rank deficient; remove "
                "collinear or absorbed instruments."
            )
        cross_rank = int(np.linalg.matrix_rank(full_instruments.T @ structural))
        if cross_rank < n_parameters:
            raise ValueError(
                "The model is not identified: the fixed-effect-adjusted instruments do not "
                "span every structural regressor."
            )

        ztz_inverse = _inverse(full_instruments.T @ full_instruments, name="instrument matrix")
        ztx = full_instruments.T @ structural
        projected_structural = full_instruments @ (ztz_inverse @ ztx)
        normal_matrix = structural.T @ projected_structural
        normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
        condition_number = float(np.linalg.cond(normal_matrix))
        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise ValueError(
                "The fixed-effect-adjusted IV normal matrix is ill-conditioned; the model "
                "is not numerically identified."
            )
        right_hand_side = (
            structural.T @ full_instruments @ (ztz_inverse @ (full_instruments.T @ y_within))
        )
        try:
            coefficients = np.linalg.solve(normal_matrix, right_hand_side)
        except np.linalg.LinAlgError as error:  # defensive after rank/condition checks
            raise ValueError(
                "The fixed-effect-adjusted IV normal equations are singular."
            ) from error

        within_fitted = structural @ coefficients
        residuals = y_within - within_fitted
        bread = _inverse(normal_matrix, name="IV normal matrix")
        covariance_estimate = _score_covariance(
            bread,
            projected_structural * residuals[:, None],
            residuals,
            covariance=self.covariance,
            df_resid=df_resid,
            cluster_codes=prepared.cluster_codes,
            n_clusters=prepared.n_clusters,
        )
        standard_errors = np.sqrt(np.maximum(np.diag(covariance_estimate.matrix), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            statistics = coefficients / standard_errors
        if covariance_estimate.distribution == "normal":
            pvalues = 2.0 * norm.sf(np.abs(statistics))
        else:
            assert covariance_estimate.df is not None
            pvalues = 2.0 * t.sf(np.abs(statistics), covariance_estimate.df)

        first_stage = _first_stage_diagnostics(
            endogenous_within,
            full_instruments,
            exogenous_within,
            endogenous_names=prepared.endogenous_names,
            excluded_count=n_excluded,
            covariance=self.covariance,
            absorbed_rank=absorbed_rank,
            cluster_codes=prepared.cluster_codes,
            n_clusters=prepared.n_clusters,
        )
        overidentification_df = n_instruments - n_parameters
        overidentification = (
            _sargan(residuals, full_instruments, overidentification_df)
            if overidentification_df > 0 and self.covariance == "unadjusted"
            else None
        )

        raw_residual_without_effects = prepared.outcome - prepared.structural @ coefficients
        absorbed_effects = raw_residual_without_effects - residuals
        fitted_values = prepared.outcome - residuals
        total_within_squares = float(y_within @ y_within)
        within_r_squared = (
            1.0 - float(residuals @ residuals) / total_within_squares
            if total_within_squares > 0
            else float("nan")
        )
        notes: list[str] = [
            "Entity fixed effects removed by a compact within transformation.",
        ]
        if self.time_effects:
            notes.append("Time fixed effects removed by the same within transformation.")
        if not prepared.balanced and self.time_effects:
            notes.append("Unbalanced two-way effects used deterministic alternating projections.")
        if prepared.dropped_rows:
            notes.append(f"Dropped {prepared.dropped_rows} row(s) jointly because missing='drop'.")
        if self.covariance == "robust":
            notes.append("HC1 does not account for within-entity serial dependence.")
        if overidentification_df == 0:
            notes.append(
                "The model is exactly identified; overidentifying restrictions cannot be tested."
            )
        elif self.covariance != "unadjusted":
            notes.append(
                "A homoskedastic Sargan test is not reported with robust or clustered inference."
            )
        weak = [
            name for name, diagnostic in first_stage.items() if diagnostic.weak_instrument_warning
        ]
        if weak:
            notes.append(
                "Classical fixed-effect-adjusted first-stage F < 10 for: " + ", ".join(weak) + "."
            )
        if prepared.n_clusters is not None and prepared.n_clusters < 30:
            notes.append(
                "Fewer than 30 clusters are available; cluster-asymptotic inference may be fragile."
            )

        parameter_index = pd.Index(structural_names, dtype="object")
        covariance_frame = pd.DataFrame(
            covariance_estimate.matrix, index=parameter_index, columns=parameter_index
        )
        panel_index = prepared.panel_index.copy()
        return PanelIV2SLSResult(
            params=pd.Series(coefficients, index=parameter_index, name="coef"),
            covariance=covariance_frame,
            standard_errors=pd.Series(standard_errors, index=parameter_index, name="std_err"),
            test_statistics=pd.Series(statistics, index=parameter_index, name="stat"),
            pvalues=pd.Series(pvalues, index=parameter_index, name="p_value"),
            residuals=pd.Series(residuals, index=panel_index, name="residual"),
            fitted_values=pd.Series(fitted_values, index=panel_index, name="fitted"),
            within_fitted_values=pd.Series(within_fitted, index=panel_index, name="within_fitted"),
            absorbed_effects=pd.Series(absorbed_effects, index=panel_index, name="absorbed_effect"),
            outcome=pd.Series(prepared.outcome, index=panel_index, name=prepared.outcome_name),
            first_stage=first_stage,
            overidentification=overidentification,
            instrument_variation=instrument_variation,
            nobs=nobs,
            n_entities=prepared.n_entities,
            n_periods=prepared.n_periods,
            balanced=prepared.balanced,
            rank=absorbed_rank + structural_rank,
            absorbed_rank=absorbed_rank,
            df_resid=df_resid,
            covariance_type=self.covariance,
            inference_distribution=covariance_estimate.distribution,
            inference_df=covariance_estimate.df,
            n_clusters=covariance_estimate.n_clusters,
            cluster_name=prepared.cluster_name,
            effects=("entity", "time") if self.time_effects else ("entity",),
            entity_name=prepared.entity_name,
            time_name=prepared.time_name,
            y_name=prepared.outcome_name,
            endogenous_names=prepared.endogenous_names,
            exogenous_names=prepared.exogenous_names,
            instrument_names=prepared.instrument_names,
            estimation_index=panel_index,
            source_index=prepared.source_index.copy(),
            dropped_rows=prepared.dropped_rows,
            within_r_squared=float(within_r_squared),
            within_iterations=within.iterations,
            within_converged=within.converged,
            within_max_abs_group_mean=within.max_abs_group_mean,
            normal_matrix_condition=condition_number,
            notes=tuple(notes),
            assumptions=(
                "Instrument relevance after fixed-effect absorption",
                "Instrument independence/exogeneity conditional on covariates and fixed effects",
                "Exclusion restriction",
                "Correctly specified static linear structural equation",
                "Additive entity effects and optional additive common time effects",
                "No unsupported dynamic-panel interpretation",
                "Treatment consistency and no interference as required by the design",
                "Estimand-specific assumptions such as monotonicity when interpreting a LATE",
            ),
        )


__all__ = ["PanelIV2SLS", "PanelIV2SLSResult"]

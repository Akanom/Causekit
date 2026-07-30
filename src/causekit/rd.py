"""Continuity-based sharp and fuzzy regression-discontinuity estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from ._data import MissingPolicy, prepare_rd_data

RDDesign = Literal["sharp", "fuzzy"]
RDKernel = Literal["triangular", "uniform", "epanechnikov"]
RDCovariance = Literal["robust", "clustered"]
RDMassPoints = Literal["check", "raise"]
RDBandwidth = float | tuple[float, float] | Literal["native_mse"]


@dataclass(frozen=True)
class RDBandwidthSelection:
    """Auditable bandwidth decision and its bounded candidate grid."""

    method: str
    bandwidth_left: float
    bandwidth_right: float
    bias_bandwidth_left: float
    bias_bandwidth_right: float
    selected_at_boundary: bool
    boundary_sides: tuple[str, ...]
    candidates: pd.DataFrame


@dataclass(frozen=True)
class RDManipulationDiagnostic:
    """One-sided boundary-kernel density diagnostic at the cutoff."""

    available: bool
    method: str
    bandwidth: float
    density_left: float | None
    density_right: float | None
    log_density_jump: float | None
    standard_error: float | None
    statistic: float | None
    pvalue: float | None
    reject: bool | None
    n_left: int
    n_right: int
    reason: str | None


@dataclass(frozen=True)
class RegressionDiscontinuityResult:
    """Sharp or fuzzy local-polynomial RD result with explicit inference layers."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    test_statistics: pd.Series
    pvalues: pd.Series
    conventional_estimate: float
    bias_corrected_estimate: float
    conventional_standard_error: float
    robust_standard_error: float
    conventional_outcome_jump: float
    outcome_jump: float
    conventional_treatment_jump: float
    treatment_jump: float
    first_stage: pd.Series | None
    design: str
    estimand: str
    primary_inference: str
    cutoff: float
    bandwidth_left: float
    bandwidth_right: float
    bias_bandwidth_left: float
    bias_bandwidth_right: float
    polynomial_order: int
    bias_order: int
    kernel: str
    covariance_type: str
    inference_distribution: str
    inference_df: float | None
    confidence_level: float
    nobs: int
    n_effective: int
    n_left: int
    n_right: int
    unique_left: int
    unique_right: int
    mass_points_detected: int
    n_clusters: int | None
    dropped_rows: int
    outcome_name: str
    running_name: str
    treatment_name: str | None
    bandwidth_selection: RDBandwidthSelection
    manipulation: RDManipulationDiagnostic
    weights: pd.DataFrame
    local_sample: pd.DataFrame
    influence: pd.Series
    condition_numbers: pd.Series
    assumptions: tuple[str, ...]
    converged: bool = True
    method: str = "local_polynomial_regression_discontinuity"
    backend: str = "native"

    def _critical_value(self, level: float) -> float:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        probability = 0.5 + level / 2.0
        if self.inference_distribution == "normal":
            return float(norm.ppf(probability))
        if self.inference_df is None:
            raise ValueError("The fitted result does not expose inference degrees of freedom.")
        return float(t.ppf(probability, self.inference_df))

    def conf_int(self, *, level: float | None = None) -> pd.DataFrame:
        """Return primary robust-bias-corrected confidence intervals."""

        requested = self.confidence_level if level is None else float(level)
        critical = self._critical_value(requested)
        estimate = self.params.to_numpy(dtype=float)
        standard_error = self.standard_errors.to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "lower": estimate - critical * standard_error,
                "upper": estimate + critical * standard_error,
            },
            index=self.params.index.copy(),
        )

    def summary_frame(self, *, level: float | None = None) -> pd.DataFrame:
        """Return a defensive primary coefficient summary."""

        interval = self.conf_int(level=level)
        return pd.DataFrame(
            {
                "estimate": self.params,
                "standard_error": self.standard_errors,
                "statistic": self.test_statistics,
                "pvalue": self.pvalues,
                "lower": interval["lower"],
                "upper": interval["upper"],
            }
        ).copy()

    def plot(self, *, bins: int = 20, ax: Any | None = None) -> Any:
        """Plot side-specific binned outcomes and the fitted bias polynomial."""

        if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 2:
            raise ValueError("bins must be an integer of at least two.")
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Regression-discontinuity plotting requires matplotlib; "
                "install causekit with the 'plot' extra."
            ) from error
        if ax is None:
            _, ax = plt.subplots()
        colors = {"left": "#1f77b4", "right": "#d62728"}
        for side in ("left", "right"):
            side_data = self.local_sample.loc[self.local_sample["side"] == side].copy()
            if side_data.empty:
                continue
            running = side_data[self.running_name].to_numpy(dtype=float)
            outcome = side_data[self.outcome_name].to_numpy(dtype=float)
            edges = np.linspace(float(running.min()), float(running.max()), bins + 1)
            categories = pd.cut(running, edges, include_lowest=True, duplicates="drop")
            binned = (
                pd.DataFrame({"running": running, "outcome": outcome, "bin": categories})
                .groupby("bin", observed=True)[["running", "outcome"]]
                .mean()
            )
            ax.scatter(
                binned["running"],
                binned["outcome"],
                color=colors[side],
                alpha=0.85,
                label=f"{side.title()} binned mean",
            )
            ordered = side_data.sort_values(self.running_name)
            ax.plot(
                ordered[self.running_name],
                ordered["fitted_bias"],
                color=colors[side],
                linewidth=1.5,
                label=f"{side.title()} order-{self.bias_order} fit",
            )
        ax.axvline(self.cutoff, color="black", linestyle="--", linewidth=1.0, label="Cutoff")
        ax.set_xlabel(self.running_name)
        ax.set_ylabel(self.outcome_name)
        ax.set_title(
            f"{self.design.title()} RD: robust estimate {self.bias_corrected_estimate:.4g}"
        )
        ax.legend()
        return ax


@dataclass(frozen=True)
class _LocalDesign:
    side: str
    bandwidth: float
    order: int
    mask: np.ndarray
    scaled_running: np.ndarray
    matrix: np.ndarray
    kernel_weights: np.ndarray
    coefficient_weights: np.ndarray
    condition_number: float

    @property
    def nobs(self) -> int:
        return int(self.mask.sum())

    def fit(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_values = values[self.mask]
        coefficients = self.coefficient_weights[:, self.mask] @ local_values
        fitted_local = self.matrix @ coefficients
        residuals = np.zeros(len(values), dtype=float)
        fitted = np.full(len(values), np.nan, dtype=float)
        residuals[self.mask] = local_values - fitted_local
        fitted[self.mask] = fitted_local
        return coefficients, residuals, fitted


def _kernel_weights(distance: np.ndarray, kernel: RDKernel) -> np.ndarray:
    absolute = np.abs(distance)
    if kernel == "triangular":
        return np.maximum(1.0 - absolute, 0.0)
    if kernel == "uniform":
        return (absolute <= 1.0).astype(float)
    return 0.75 * np.maximum(1.0 - absolute**2, 0.0)


def _local_design(
    centered: np.ndarray,
    *,
    side: Literal["left", "right"],
    bandwidth: float,
    order: int,
    kernel: RDKernel,
    max_condition_number: float,
) -> _LocalDesign:
    side_mask = centered < 0.0 if side == "left" else centered >= 0.0
    scaled = centered / bandwidth
    weights_all = _kernel_weights(scaled, kernel)
    mask = side_mask & (np.abs(scaled) <= 1.0) & (weights_all > 0.0)
    nobs = int(mask.sum())
    if nobs <= order + 1:
        raise ValueError(
            f"The {side} side has {nobs} positive-weight observations; "
            f"order {order} requires at least {order + 2}."
        )
    unique = int(np.unique(centered[mask]).size)
    if unique < order + 1:
        raise ValueError(
            f"The {side} side needs at least {order + 1} unique running-variable "
            f"values for polynomial order {order}; found {unique}."
        )

    local_scaled = scaled[mask]
    matrix = np.vander(local_scaled, N=order + 1, increasing=True)
    weights = weights_all[mask]
    gram = matrix.T @ (weights[:, None] * matrix)
    rank = int(np.linalg.matrix_rank(gram))
    if rank < order + 1:
        raise ValueError(f"The {side} local-polynomial design is rank deficient.")
    condition_number = float(np.linalg.cond(gram))
    if not np.isfinite(condition_number) or condition_number > max_condition_number:
        raise ValueError(
            f"The {side} local-polynomial design is ill-conditioned "
            f"(condition number {condition_number:.6g})."
        )
    inverse = np.linalg.inv(gram)
    local_map = inverse @ (matrix.T * weights)
    coefficient_weights = np.zeros((order + 1, len(centered)), dtype=float)
    coefficient_weights[:, mask] = local_map
    return _LocalDesign(
        side=side,
        bandwidth=bandwidth,
        order=order,
        mask=mask,
        scaled_running=scaled,
        matrix=matrix,
        kernel_weights=weights,
        coefficient_weights=coefficient_weights,
        condition_number=condition_number,
    )


def _bias_adjusted_intercept_weights(
    point: _LocalDesign,
    bias: _LocalDesign,
    *,
    polynomial_order: int,
) -> tuple[np.ndarray, float]:
    local_u = point.scaled_running[point.mask]
    omitted = local_u ** (polynomial_order + 1)
    response = point.coefficient_weights[0, point.mask] @ omitted
    scale = (point.bandwidth / bias.bandwidth) ** (polynomial_order + 1)
    factor = float(response * scale)
    adjusted = (
        point.coefficient_weights[0] - factor * bias.coefficient_weights[polynomial_order + 1]
    )
    return adjusted, factor


def _pair_bandwidth(value: object, *, name: str) -> tuple[float, float]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive number or a left/right pair.")
    if isinstance(value, (int, float, np.integer, np.floating)):
        pair = (float(value), float(value))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            pair = (float(value[0]), float(value[1]))
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must contain positive numeric values.") from error
    else:
        raise TypeError(f"{name} must be a positive number or a left/right pair.")
    if not np.isfinite(pair).all() or pair[0] <= 0.0 or pair[1] <= 0.0:
        raise ValueError(f"{name} values must be finite and strictly positive.")
    return pair


def _candidate_bandwidths(
    centered: np.ndarray,
    *,
    side: Literal["left", "right"],
    minimum: int,
    count: int,
    upper_cap: float,
    pilot_cap: float,
) -> tuple[np.ndarray, float]:
    mask = centered < 0.0 if side == "left" else centered >= 0.0
    distances = np.sort(np.abs(centered[mask]))
    if distances.size < minimum:
        raise ValueError(
            f"The {side} side has {distances.size} observations; native bandwidth "
            f"selection requires at least {minimum}."
        )
    quantile_upper = float(np.quantile(distances, 0.8))
    upper_target = max(float(distances[minimum - 1]), min(quantile_upper, upper_cap))
    upper_index = max(
        minimum - 1,
        int(np.searchsorted(distances, upper_target, side="right") - 1),
    )
    positions = np.unique(np.linspace(minimum - 1, upper_index, count).round().astype(int))
    candidates = np.unique(np.nextafter(distances[positions], np.inf))
    candidates = candidates[candidates > 0.0]
    if candidates.size == 0:
        raise ValueError(f"The {side} side has no positive running-variable distance.")
    quantile_pilot = float(np.quantile(distances, 0.9))
    pilot_target = max(float(distances[upper_index]), min(quantile_pilot, pilot_cap))
    pilot_index = max(
        upper_index,
        int(np.searchsorted(distances, pilot_target, side="right") - 1),
    )
    pilot = float(np.nextafter(distances[pilot_index], np.inf))
    return candidates, pilot


def _normal_test(estimate: float, standard_error: float) -> tuple[float, float]:
    if standard_error > 0.0:
        statistic = estimate / standard_error
        return float(statistic), float(2.0 * norm.sf(abs(statistic)))
    if estimate == 0.0:
        return float("nan"), float("nan")
    return float(np.copysign(np.inf, estimate)), 0.0


def _score_variance(
    score: np.ndarray,
    active: np.ndarray,
    *,
    nparams: int,
    covariance: RDCovariance,
    clusters: np.ndarray | None,
    centered: np.ndarray,
) -> tuple[float, int | None, str, float | None]:
    nlocal = int(active.sum())
    if nlocal <= nparams:
        raise ValueError(
            "RD covariance requires more positive-weight local observations than parameters."
        )
    if covariance == "robust":
        parameters_per_side = nparams // 2
        left = active & (centered < 0.0)
        right = active & (centered >= 0.0)
        nleft = int(left.sum())
        nright = int(right.sum())
        if nleft <= parameters_per_side or nright <= parameters_per_side:
            raise ValueError(
                "RD HC1 covariance requires more local observations than polynomial "
                "parameters on each side."
            )
        variance = (nleft / (nleft - parameters_per_side)) * float(score[left] @ score[left]) + (
            nright / (nright - parameters_per_side)
        ) * float(score[right] @ score[right])
        return variance, None, "normal", None

    if clusters is None:  # pragma: no cover - constructor/fit validation protects this
        raise RuntimeError("Internal clustered RD covariance configuration error.")
    local_clusters = np.asarray(clusters, dtype=object)[active]
    try:
        codes, labels = pd.factorize(local_clusters, sort=False)
    except TypeError as error:
        raise TypeError("cluster labels must be hashable scalar values.") from error
    nclusters = int(len(labels))
    if nclusters < 2:
        raise ValueError("Clustered RD covariance requires at least two local clusters.")
    for side, mask in (("left", centered < 0.0), ("right", centered >= 0.0)):
        side_clusters = pd.unique(np.asarray(clusters, dtype=object)[active & mask])
        if len(side_clusters) < 2:
            raise ValueError(
                f"Clustered RD covariance requires at least two local clusters on the {side} side."
            )
    cluster_scores = np.zeros(nclusters, dtype=float)
    np.add.at(cluster_scores, codes, score[active])
    correction = (nclusters / (nclusters - 1.0)) * ((nlocal - 1.0) / (nlocal - nparams))
    variance = correction * float(cluster_scores @ cluster_scores)
    return variance, nclusters, "t", float(nclusters - 1)


def _manipulation_diagnostic(
    centered: np.ndarray,
    *,
    bandwidth: float,
    level: float,
) -> RDManipulationDiagnostic:
    scaled = centered / bandwidth
    kernel = _kernel_weights(scaled, "triangular")
    left = (centered < 0.0) & (np.abs(scaled) <= 1.0) & (kernel > 0.0)
    right = (centered >= 0.0) & (np.abs(scaled) <= 1.0) & (kernel > 0.0)
    nleft = int(left.sum())
    nright = int(right.sum())
    if nleft < 5 or nright < 5:
        return RDManipulationDiagnostic(
            available=False,
            method="one_sided_boundary_kernel_density",
            bandwidth=bandwidth,
            density_left=None,
            density_right=None,
            log_density_jump=None,
            standard_error=None,
            statistic=None,
            pvalue=None,
            reject=None,
            n_left=nleft,
            n_right=nright,
            reason="At least five positive-weight observations are required on each side.",
        )

    # The triangular kernel integrates to one half on either side of the boundary.
    zleft = np.where(left, 2.0 * kernel / bandwidth, 0.0)
    zright = np.where(right, 2.0 * kernel / bandwidth, 0.0)
    density_left = float(zleft.mean())
    density_right = float(zright.mean())
    if density_left <= 0.0 or density_right <= 0.0:
        return RDManipulationDiagnostic(
            available=False,
            method="one_sided_boundary_kernel_density",
            bandwidth=bandwidth,
            density_left=density_left,
            density_right=density_right,
            log_density_jump=None,
            standard_error=None,
            statistic=None,
            pvalue=None,
            reject=None,
            n_left=nleft,
            n_right=nright,
            reason="One-sided boundary density estimate is nonpositive.",
        )
    log_jump = float(np.log(density_right) - np.log(density_left))
    influence = (zright - density_right) / density_right - (zleft - density_left) / density_left
    standard_error = float(np.sqrt(np.var(influence, ddof=1) / len(centered)))
    statistic, pvalue = _normal_test(log_jump, standard_error)
    return RDManipulationDiagnostic(
        available=True,
        method="one_sided_boundary_kernel_density",
        bandwidth=bandwidth,
        density_left=density_left,
        density_right=density_right,
        log_density_jump=log_jump,
        standard_error=standard_error,
        statistic=statistic,
        pvalue=pvalue,
        reject=bool(pvalue < 1.0 - level),
        n_left=nleft,
        n_right=nright,
        reason=None,
    )


class RegressionDiscontinuity:
    """Estimate sharp or fuzzy effects at a known running-variable cutoff."""

    def __init__(
        self,
        *,
        design: RDDesign = "sharp",
        cutoff: float = 0.0,
        bandwidth: RDBandwidth = "native_mse",
        bias_bandwidth: float | tuple[float, float] | None = None,
        polynomial_order: int = 1,
        bias_order: int | None = None,
        kernel: RDKernel = "triangular",
        covariance: RDCovariance = "robust",
        missing: MissingPolicy = "raise",
        mass_points: RDMassPoints = "check",
        confidence_level: float = 0.95,
        manipulation_bandwidth: float | None = None,
        first_stage_tolerance: float = 1e-8,
        bandwidth_candidates: int = 15,
        max_condition_number: float = 1e12,
    ) -> None:
        if design not in {"sharp", "fuzzy"}:
            raise ValueError("design must be 'sharp' or 'fuzzy'.")
        if isinstance(cutoff, bool) or not np.isfinite(cutoff):
            raise ValueError("cutoff must be a finite real number.")
        if isinstance(bandwidth, str):
            if bandwidth != "native_mse":
                raise ValueError("bandwidth must be 'native_mse' or positive numeric values.")
        else:
            _pair_bandwidth(bandwidth, name="bandwidth")
        if bias_bandwidth is not None:
            _pair_bandwidth(bias_bandwidth, name="bias bandwidth")
        if isinstance(polynomial_order, bool) or not isinstance(
            polynomial_order, (int, np.integer)
        ):
            raise TypeError("polynomial_order must be an integer between zero and three.")
        if not 0 <= int(polynomial_order) <= 3:
            raise ValueError("polynomial_order must be between zero and three.")
        resolved_bias_order = int(polynomial_order) + 1 if bias_order is None else bias_order
        if isinstance(resolved_bias_order, bool) or not isinstance(
            resolved_bias_order, (int, np.integer)
        ):
            raise TypeError("bias_order must be an integer.")
        if not int(polynomial_order) < int(resolved_bias_order) <= 4:
            raise ValueError(
                "bias_order must be greater than polynomial_order and no greater than four."
            )
        if kernel not in {"triangular", "uniform", "epanechnikov"}:
            raise ValueError("kernel must be 'triangular', 'uniform', or 'epanechnikov'.")
        if covariance not in {"robust", "clustered"}:
            raise ValueError("covariance must be 'robust' or 'clustered'.")
        if missing not in {"raise", "drop"}:
            raise ValueError("missing must be 'raise' or 'drop'.")
        if mass_points not in {"check", "raise"}:
            raise ValueError("mass_points must be 'check' or 'raise'.")
        if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between zero and one.")
        if manipulation_bandwidth is not None:
            manipulation_pair = _pair_bandwidth(
                manipulation_bandwidth, name="manipulation_bandwidth"
            )
            if manipulation_pair[0] != manipulation_pair[1]:  # pragma: no cover - scalar API
                raise ValueError("manipulation_bandwidth must be a single positive number.")
        if not np.isfinite(first_stage_tolerance) or first_stage_tolerance <= 0.0:
            raise ValueError("first_stage_tolerance must be finite and strictly positive.")
        if isinstance(bandwidth_candidates, bool) or not isinstance(
            bandwidth_candidates, (int, np.integer)
        ):
            raise TypeError("bandwidth_candidates must be an integer between five and fifty.")
        if not 5 <= int(bandwidth_candidates) <= 50:
            raise ValueError("bandwidth_candidates must be between five and fifty.")
        if not np.isfinite(max_condition_number) or max_condition_number <= 10.0:
            raise ValueError("max_condition_number must be finite and greater than ten.")

        self.design = design
        self.cutoff = float(cutoff)
        self.bandwidth = bandwidth
        self.bias_bandwidth = bias_bandwidth
        self.polynomial_order = int(polynomial_order)
        self.bias_order = int(resolved_bias_order)
        self.kernel = kernel
        self.covariance = covariance
        self.missing = missing
        self.mass_points = mass_points
        self.confidence_level = float(confidence_level)
        self.manipulation_bandwidth = (
            float(manipulation_bandwidth) if manipulation_bandwidth is not None else None
        )
        self.first_stage_tolerance = float(first_stage_tolerance)
        self.bandwidth_candidates = int(bandwidth_candidates)
        self.max_condition_number = float(max_condition_number)

    def _native_bandwidths(
        self,
        centered: np.ndarray,
        outcome: np.ndarray,
        treatment: np.ndarray,
    ) -> RDBandwidthSelection:
        minimum = max(self.bias_order + 2, 8)
        spread = min(
            float(np.std(centered, ddof=1)),
            float((np.quantile(centered, 0.75) - np.quantile(centered, 0.25)) / 1.349),
        )
        if not np.isfinite(spread) or spread <= 0.0:
            spread = float(np.max(centered) - np.min(centered)) / 4.0
        reference_bandwidth = 2.5 * spread * len(centered) ** (-0.2)
        upper_cap = 1.5 * reference_bandwidth
        pilot_cap = 2.5 * reference_bandwidth
        left_candidates, left_pilot = _candidate_bandwidths(
            centered,
            side="left",
            minimum=minimum,
            count=self.bandwidth_candidates,
            upper_cap=upper_cap,
            pilot_cap=pilot_cap,
        )
        right_candidates, right_pilot = _candidate_bandwidths(
            centered,
            side="right",
            minimum=minimum,
            count=self.bandwidth_candidates,
            upper_cap=upper_cap,
            pilot_cap=pilot_cap,
        )
        pilot_left = _local_design(
            centered,
            side="left",
            bandwidth=left_pilot,
            order=self.bias_order,
            kernel=self.kernel,
            max_condition_number=self.max_condition_number,
        )
        pilot_right = _local_design(
            centered,
            side="right",
            bandwidth=right_pilot,
            order=self.bias_order,
            kernel=self.kernel,
            max_condition_number=self.max_condition_number,
        )
        y_left, y_resid_left, _ = pilot_left.fit(outcome)
        y_right, y_resid_right, _ = pilot_right.fit(outcome)
        d_left, d_resid_left, _ = pilot_left.fit(treatment)
        d_right, d_resid_right, _ = pilot_right.fit(treatment)
        pilot_treatment_jump = float(d_right[0] - d_left[0])
        pilot_effect = 0.0
        if self.design == "fuzzy":
            if pilot_treatment_jump <= self.first_stage_tolerance:
                raise ValueError(
                    "Native fuzzy-RD bandwidth selection requires a positive treatment jump "
                    "in the pilot local polynomial."
                )
            pilot_effect = float((y_right[0] - y_left[0]) / pilot_treatment_jump)

        records: list[dict[str, float | bool]] = []
        for left_bandwidth in left_candidates:
            left_design = _local_design(
                centered,
                side="left",
                bandwidth=float(left_bandwidth),
                order=self.polynomial_order,
                kernel=self.kernel,
                max_condition_number=self.max_condition_number,
            )
            left_omitted = left_design.scaled_running[left_design.mask] ** (
                self.polynomial_order + 1
            )
            left_response = float(
                left_design.coefficient_weights[0, left_design.mask] @ left_omitted
            )
            left_scale = (float(left_bandwidth) / left_pilot) ** (self.polynomial_order + 1)
            y_bias_left = left_response * left_scale * y_left[self.polynomial_order + 1]
            d_bias_left = left_response * left_scale * d_left[self.polynomial_order + 1]
            candidate_d_left = float(left_design.coefficient_weights[0] @ treatment)
            for right_bandwidth in right_candidates:
                right_design = _local_design(
                    centered,
                    side="right",
                    bandwidth=float(right_bandwidth),
                    order=self.polynomial_order,
                    kernel=self.kernel,
                    max_condition_number=self.max_condition_number,
                )
                right_omitted = right_design.scaled_running[right_design.mask] ** (
                    self.polynomial_order + 1
                )
                right_response = float(
                    right_design.coefficient_weights[0, right_design.mask] @ right_omitted
                )
                right_scale = (float(right_bandwidth) / right_pilot) ** (self.polynomial_order + 1)
                y_bias_right = right_response * right_scale * y_right[self.polynomial_order + 1]
                d_bias_right = right_response * right_scale * d_right[self.polynomial_order + 1]
                candidate_treatment_jump = float(
                    right_design.coefficient_weights[0] @ treatment - candidate_d_left
                )
                admissible = bool(
                    self.design == "sharp" or candidate_treatment_jump > self.first_stage_tolerance
                )
                outcome_bias = float(y_bias_right - y_bias_left)
                treatment_bias = float(d_bias_right - d_bias_left)
                if self.design == "sharp":
                    leading_bias = outcome_bias
                    left_score = left_design.coefficient_weights[0] * y_resid_left
                    right_score = right_design.coefficient_weights[0] * y_resid_right
                else:
                    leading_bias = (
                        outcome_bias - pilot_effect * treatment_bias
                    ) / pilot_treatment_jump
                    left_score = (
                        left_design.coefficient_weights[0]
                        * (y_resid_left - pilot_effect * d_resid_left)
                        / pilot_treatment_jump
                    )
                    right_score = (
                        right_design.coefficient_weights[0]
                        * (y_resid_right - pilot_effect * d_resid_right)
                        / pilot_treatment_jump
                    )
                variance = float(left_score @ left_score + right_score @ right_score)
                objective_value = float(leading_bias**2 + variance) if admissible else float("inf")
                records.append(
                    {
                        "bandwidth_left": float(left_bandwidth),
                        "bandwidth_right": float(right_bandwidth),
                        "leading_bias_squared": float(leading_bias**2),
                        "variance": variance,
                        "objective": objective_value,
                        "candidate_treatment_jump": (
                            1.0 if self.design == "sharp" else candidate_treatment_jump
                        ),
                        "admissible": admissible,
                        "n_left": float(left_design.nobs),
                        "n_right": float(right_design.nobs),
                        "selected": False,
                    }
                )
        candidates = pd.DataFrame.from_records(records)
        objective = candidates["objective"].to_numpy(dtype=float)
        if not np.isfinite(objective).any():
            raise FloatingPointError("Native RD bandwidth selection produced no finite objective.")
        selected_position = int(np.nanargmin(objective))
        candidates.loc[selected_position, "selected"] = True
        selected = candidates.iloc[selected_position]
        boundary_sides: list[str] = []
        for side in ("left", "right"):
            column = f"bandwidth_{side}"
            selected_bandwidth = float(selected[column])
            if selected_bandwidth in {
                float(candidates[column].min()),
                float(candidates[column].max()),
            }:
                boundary_sides.append(side)
        return RDBandwidthSelection(
            method="native_design_conditional_mse_grid",
            bandwidth_left=float(selected["bandwidth_left"]),
            bandwidth_right=float(selected["bandwidth_right"]),
            bias_bandwidth_left=left_pilot,
            bias_bandwidth_right=right_pilot,
            selected_at_boundary=bool(boundary_sides),
            boundary_sides=tuple(boundary_sides),
            candidates=candidates,
        )

    def _bandwidth_selection(
        self,
        centered: np.ndarray,
        outcome: np.ndarray,
        treatment: np.ndarray,
    ) -> RDBandwidthSelection:
        if self.bandwidth == "native_mse":
            if self.bias_bandwidth is not None:
                raise ValueError(
                    "bias_bandwidth must be None when bandwidth='native_mse'; "
                    "the native selector freezes both bandwidth layers."
                )
            return self._native_bandwidths(centered, outcome, treatment)

        bandwidth_left, bandwidth_right = _pair_bandwidth(self.bandwidth, name="bandwidth")
        if self.bias_bandwidth is None:
            bias_left, bias_right = bandwidth_left, bandwidth_right
        else:
            bias_left, bias_right = _pair_bandwidth(self.bias_bandwidth, name="bias bandwidth")
        if bias_left < bandwidth_left or bias_right < bandwidth_right:
            raise ValueError(
                "Each bias bandwidth must be greater than or equal to its point bandwidth."
            )
        candidates = pd.DataFrame(
            {
                "bandwidth_left": [bandwidth_left],
                "bandwidth_right": [bandwidth_right],
                "leading_bias_squared": [np.nan],
                "variance": [np.nan],
                "objective": [np.nan],
                "candidate_treatment_jump": [1.0 if self.design == "sharp" else np.nan],
                "admissible": [True],
                "n_left": [np.nan],
                "n_right": [np.nan],
                "selected": [True],
            }
        )
        return RDBandwidthSelection(
            method="manual",
            bandwidth_left=bandwidth_left,
            bandwidth_right=bandwidth_right,
            bias_bandwidth_left=bias_left,
            bias_bandwidth_right=bias_right,
            selected_at_boundary=False,
            boundary_sides=(),
            candidates=candidates,
        )

    def fit(
        self,
        y: Any,
        *,
        running: Any,
        treatment: Any | None = None,
        clusters: Any | None = None,
    ) -> RegressionDiscontinuityResult:
        """Fit the declared sharp or fuzzy cutoff estimand."""

        if self.covariance == "clustered" and clusters is None:
            raise ValueError("clusters must be provided when covariance='clustered'.")
        if self.covariance != "clustered" and clusters is not None:
            raise ValueError("clusters may be provided only when covariance='clustered'.")
        prepared = prepare_rd_data(
            y,
            running,
            treatment,
            clusters=clusters,
            missing=self.missing,
        )
        centered = prepared.running - self.cutoff
        if not (centered < 0.0).any() or not (centered >= 0.0).any():
            raise ValueError(
                "The cutoff must have observed running-variable support on both sides."
            )

        assignment = (centered >= 0.0).astype(float)
        if self.design == "sharp":
            if prepared.treatment is not None and not np.array_equal(
                prepared.treatment, assignment
            ):
                raise ValueError(
                    "Supplied treatment does not equal the declared sharp assignment 1[running>=cutoff]."
                )
            treatment_array = assignment
        else:
            if prepared.treatment is None:
                raise ValueError("treatment is required when design='fuzzy'.")
            unique_treatment = np.unique(prepared.treatment)
            if not set(unique_treatment).issubset({0.0, 1.0}):
                raise ValueError("Fuzzy-RD treatment must be binary with values zero and one.")
            treatment_array = prepared.treatment

        selection = self._bandwidth_selection(centered, prepared.y, treatment_array)
        point_left = _local_design(
            centered,
            side="left",
            bandwidth=selection.bandwidth_left,
            order=self.polynomial_order,
            kernel=self.kernel,
            max_condition_number=self.max_condition_number,
        )
        point_right = _local_design(
            centered,
            side="right",
            bandwidth=selection.bandwidth_right,
            order=self.polynomial_order,
            kernel=self.kernel,
            max_condition_number=self.max_condition_number,
        )
        bias_left = _local_design(
            centered,
            side="left",
            bandwidth=selection.bias_bandwidth_left,
            order=self.bias_order,
            kernel=self.kernel,
            max_condition_number=self.max_condition_number,
        )
        bias_right = _local_design(
            centered,
            side="right",
            bandwidth=selection.bias_bandwidth_right,
            order=self.bias_order,
            kernel=self.kernel,
            max_condition_number=self.max_condition_number,
        )

        left_adjusted, _ = _bias_adjusted_intercept_weights(
            point_left, bias_left, polynomial_order=self.polynomial_order
        )
        right_adjusted, _ = _bias_adjusted_intercept_weights(
            point_right, bias_right, polynomial_order=self.polynomial_order
        )
        conventional_weights = (
            point_right.coefficient_weights[0] - point_left.coefficient_weights[0]
        )
        adjusted_weights = right_adjusted - left_adjusted

        _, y_resid_point_left, y_fit_point_left = point_left.fit(prepared.y)
        _, y_resid_point_right, y_fit_point_right = point_right.fit(prepared.y)
        _, y_resid_bias_left, y_fit_bias_left = bias_left.fit(prepared.y)
        _, y_resid_bias_right, y_fit_bias_right = bias_right.fit(prepared.y)
        _, d_resid_point_left, _ = point_left.fit(treatment_array)
        _, d_resid_point_right, _ = point_right.fit(treatment_array)
        _, d_resid_bias_left, _ = bias_left.fit(treatment_array)
        _, d_resid_bias_right, _ = bias_right.fit(treatment_array)

        y_resid_point = y_resid_point_left + y_resid_point_right
        y_resid_bias = y_resid_bias_left + y_resid_bias_right
        d_resid_point = d_resid_point_left + d_resid_point_right
        d_resid_bias = d_resid_bias_left + d_resid_bias_right
        conventional_outcome_jump = float(conventional_weights @ prepared.y)
        outcome_jump = float(adjusted_weights @ prepared.y)
        conventional_treatment_jump = float(conventional_weights @ treatment_array)
        treatment_jump = float(adjusted_weights @ treatment_array)
        if self.design == "sharp":
            conventional_treatment_jump = 1.0
            treatment_jump = 1.0

        conventional_active = point_left.mask | point_right.mask
        robust_active = bias_left.mask | bias_right.mask | conventional_active
        conventional_nparams = 2 * (self.polynomial_order + 1)
        robust_nparams = 2 * (self.bias_order + 1)
        first_stage: pd.Series | None = None
        if self.design == "sharp":
            conventional_estimate = conventional_outcome_jump
            estimate = outcome_jump
            conventional_score = conventional_weights * y_resid_point
            robust_score = adjusted_weights * y_resid_bias
        else:
            if conventional_treatment_jump <= self.first_stage_tolerance:
                raise ValueError(
                    "Fuzzy RD requires a positive treatment jump at the cutoff; "
                    "the conventional local first stage failed."
                )
            if treatment_jump <= self.first_stage_tolerance:
                raise ValueError(
                    "Fuzzy RD requires a positive treatment jump at the cutoff; "
                    "the bias-corrected local first stage failed."
                )
            conventional_estimate = conventional_outcome_jump / conventional_treatment_jump
            outcome_bias = conventional_outcome_jump - outcome_jump
            treatment_bias = conventional_treatment_jump - treatment_jump
            ratio_bias = (
                outcome_bias - conventional_estimate * treatment_bias
            ) / conventional_treatment_jump
            estimate = conventional_estimate - ratio_bias
            conventional_score = (
                conventional_weights
                * (y_resid_point - conventional_estimate * d_resid_point)
                / conventional_treatment_jump
            )
            robust_score = (
                adjusted_weights
                * (y_resid_bias - conventional_estimate * d_resid_bias)
                / conventional_treatment_jump
            )

        conventional_variance, _, _, _ = _score_variance(
            conventional_score,
            conventional_active,
            nparams=conventional_nparams,
            covariance=self.covariance,
            clusters=prepared.clusters,
            centered=centered,
        )
        robust_variance, nclusters, distribution, inference_df = _score_variance(
            robust_score,
            robust_active,
            nparams=robust_nparams,
            covariance=self.covariance,
            clusters=prepared.clusters,
            centered=centered,
        )
        conventional_se = float(np.sqrt(max(conventional_variance, 0.0)))
        robust_se = float(np.sqrt(max(robust_variance, 0.0)))

        if self.design == "fuzzy":
            first_stage_score = adjusted_weights * d_resid_bias
            first_stage_variance, _, first_stage_distribution, first_stage_df = _score_variance(
                first_stage_score,
                robust_active,
                nparams=robust_nparams,
                covariance=self.covariance,
                clusters=prepared.clusters,
                centered=centered,
            )
            first_stage_se = float(np.sqrt(max(first_stage_variance, 0.0)))
            first_stage_stat, first_stage_p = _normal_test(treatment_jump, first_stage_se)
            if (
                first_stage_distribution == "t"
                and first_stage_se > 0.0
                and first_stage_df is not None
            ):
                first_stage_stat = treatment_jump / first_stage_se
                first_stage_p = float(2.0 * t.sf(abs(first_stage_stat), first_stage_df))
            first_stage = pd.Series(
                {
                    "estimate": treatment_jump,
                    "standard_error": first_stage_se,
                    "statistic": first_stage_stat,
                    "pvalue": first_stage_p,
                    "weak_first_stage_warning": bool(
                        treatment_jump < 0.10 or abs(first_stage_stat) < 3.29
                    ),
                },
                name="local_treatment_jump",
            )

        local_mask = robust_active
        unique_left = int(np.unique(prepared.running[local_mask & (centered < 0.0)]).size)
        unique_right = int(np.unique(prepared.running[local_mask & (centered >= 0.0)]).size)
        mass_points = int(local_mask.sum() - unique_left - unique_right)
        if self.mass_points == "raise" and mass_points > 0:
            raise ValueError(
                f"Local running-variable support contains {mass_points} mass points; "
                "mass_points='raise' prohibits estimation."
            )

        manipulation_bandwidth = (
            self.manipulation_bandwidth
            if self.manipulation_bandwidth is not None
            else min(selection.bandwidth_left, selection.bandwidth_right)
        )
        manipulation = _manipulation_diagnostic(
            centered,
            bandwidth=manipulation_bandwidth,
            level=self.confidence_level,
        )

        statistic, pvalue = _normal_test(estimate, robust_se)
        if distribution == "t" and robust_se > 0.0 and inference_df is not None:
            statistic = estimate / robust_se
            pvalue = float(2.0 * t.sf(abs(statistic), inference_df))
        index = pd.Index(["rd_effect"], name="parameter")
        params = pd.Series([estimate], index=index, name="estimate")
        standard_errors = pd.Series([robust_se], index=index, name="standard_error")
        test_statistics = pd.Series([statistic], index=index, name="statistic")
        pvalues = pd.Series([pvalue], index=index, name="pvalue")
        covariance_frame = pd.DataFrame(
            [[robust_variance]], index=index.copy(), columns=index.copy()
        )

        weights = pd.DataFrame(
            {
                "conventional": conventional_weights,
                "bias_corrected": adjusted_weights,
            },
            index=prepared.index.copy(),
        )
        fitted_point = np.where(
            point_left.mask, y_fit_point_left, np.where(point_right.mask, y_fit_point_right, np.nan)
        )
        fitted_bias = np.where(
            bias_left.mask, y_fit_bias_left, np.where(bias_right.mask, y_fit_bias_right, np.nan)
        )
        local_data: dict[str, object] = {
            prepared.y_name: prepared.y[local_mask],
            prepared.running_name: prepared.running[local_mask],
            "side": np.where(centered[local_mask] < 0.0, "left", "right"),
            "point_weight": conventional_weights[local_mask],
            "bias_corrected_weight": adjusted_weights[local_mask],
            "fitted_point": fitted_point[local_mask],
            "fitted_bias": fitted_bias[local_mask],
        }
        if prepared.treatment is not None:
            assert prepared.treatment_name is not None
            local_data[prepared.treatment_name] = prepared.treatment[local_mask]
        if prepared.clusters is not None:
            local_data["cluster"] = prepared.clusters[local_mask]
        local_sample = pd.DataFrame(local_data, index=prepared.index[local_mask].copy())
        influence = pd.Series(
            robust_score,
            index=prepared.index.copy(),
            name="robust_score_contribution",
        )
        condition_numbers = pd.Series(
            {
                "point_left": point_left.condition_number,
                "point_right": point_right.condition_number,
                "bias_left": bias_left.condition_number,
                "bias_right": bias_right.condition_number,
            },
            name="condition_number",
        )
        assumptions: tuple[str, ...] = (
            "The cutoff and assignment rule were fixed before observing outcomes.",
            "Potential-outcome conditional means are sufficiently smooth at the cutoff.",
            "Units cannot precisely sort around the cutoff in a way that changes potential outcomes.",
            "No other intervention changes discontinuously at the same cutoff.",
        )
        if self.design == "fuzzy":
            assumptions += (
                "Crossing the cutoff changes treatment take-up monotonically and only through eligibility.",
                "The reported ratio is local to cutoff compliers with a positive first-stage jump.",
            )

        return RegressionDiscontinuityResult(
            params=params,
            covariance=covariance_frame,
            standard_errors=standard_errors,
            test_statistics=test_statistics,
            pvalues=pvalues,
            conventional_estimate=float(conventional_estimate),
            bias_corrected_estimate=float(estimate),
            conventional_standard_error=conventional_se,
            robust_standard_error=robust_se,
            conventional_outcome_jump=conventional_outcome_jump,
            outcome_jump=outcome_jump,
            conventional_treatment_jump=conventional_treatment_jump,
            treatment_jump=treatment_jump,
            first_stage=first_stage,
            design=self.design,
            estimand=(
                "cutoff_average_treatment_effect"
                if self.design == "sharp"
                else "cutoff_complier_local_average_treatment_effect"
            ),
            primary_inference="robust_bias_corrected",
            cutoff=self.cutoff,
            bandwidth_left=selection.bandwidth_left,
            bandwidth_right=selection.bandwidth_right,
            bias_bandwidth_left=selection.bias_bandwidth_left,
            bias_bandwidth_right=selection.bias_bandwidth_right,
            polynomial_order=self.polynomial_order,
            bias_order=self.bias_order,
            kernel=self.kernel,
            covariance_type=self.covariance,
            inference_distribution=distribution,
            inference_df=inference_df,
            confidence_level=self.confidence_level,
            nobs=len(prepared.y),
            n_effective=int(local_mask.sum()),
            n_left=point_left.nobs,
            n_right=point_right.nobs,
            unique_left=unique_left,
            unique_right=unique_right,
            mass_points_detected=mass_points,
            n_clusters=nclusters,
            dropped_rows=prepared.dropped_rows,
            outcome_name=prepared.y_name,
            running_name=prepared.running_name,
            treatment_name=prepared.treatment_name,
            bandwidth_selection=selection,
            manipulation=manipulation,
            weights=weights,
            local_sample=local_sample,
            influence=influence,
            condition_numbers=condition_numbers,
            assumptions=assumptions,
        )


__all__ = [
    "RDBandwidthSelection",
    "RDManipulationDiagnostic",
    "RegressionDiscontinuity",
    "RegressionDiscontinuityResult",
]

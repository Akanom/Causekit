"""Identification and first-stage diagnostics for instrumental variables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, f

from ._covariance import CovarianceType, ols_covariance


@dataclass(frozen=True)
class FirstStageDiagnostic:
    """Diagnostic summary for one endogenous regressor's first stage.

    The classical partial F statistic is a familiar relevance screen, not a
    universal weak-identification test. Its rule-of-thumb warning is most
    interpretable with one endogenous regressor under homoskedastic errors.
    """

    endogenous: str
    r_squared: float
    partial_r_squared: float
    classical_f_statistic: float
    classical_f_df_num: int
    classical_f_df_denom: int
    classical_f_p_value: float
    excluded_instrument_statistic: float
    excluded_instrument_df: int
    excluded_instrument_p_value: float
    excluded_instrument_distribution: str
    weak_instrument_warning: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "endogenous": self.endogenous,
            "r_squared": self.r_squared,
            "partial_r_squared": self.partial_r_squared,
            "classical_f_statistic": self.classical_f_statistic,
            "classical_f_df_num": self.classical_f_df_num,
            "classical_f_df_denom": self.classical_f_df_denom,
            "classical_f_p_value": self.classical_f_p_value,
            "excluded_instrument_statistic": self.excluded_instrument_statistic,
            "excluded_instrument_df": self.excluded_instrument_df,
            "excluded_instrument_p_value": self.excluded_instrument_p_value,
            "excluded_instrument_distribution": self.excluded_instrument_distribution,
            "weak_instrument_warning": self.weak_instrument_warning,
        }


@dataclass(frozen=True)
class SarganTest:
    """Homoskedastic Sargan test of overidentifying restrictions."""

    statistic: float
    df: int
    p_value: float
    name: str = "Sargan test"
    null: str = "All overidentifying restrictions are valid."
    valid_under: str = "Conditional homoskedasticity"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "statistic": self.statistic,
            "df": self.df,
            "p_value": self.p_value,
            "null": self.null,
            "valid_under": self.valid_under,
        }


def _ols_fit(design: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = design.T @ design
    coefficients = np.linalg.pinv(gram) @ design.T @ outcome
    return coefficients, outcome - design @ coefficients


def first_stage_diagnostics(
    endogenous: np.ndarray,
    full_instruments: np.ndarray,
    included_exogenous: np.ndarray,
    *,
    endogenous_names: tuple[str, ...],
    excluded_count: int,
    covariance: CovarianceType,
    clusters: np.ndarray | None,
) -> dict[str, FirstStageDiagnostic]:
    """Compute transparent first-stage fit and excluded-instrument tests."""

    nobs = full_instruments.shape[0]
    unrestricted_rank = int(np.linalg.matrix_rank(full_instruments))
    denominator_df = nobs - unrestricted_rank
    if denominator_df <= 0:
        raise ValueError("First-stage residual degrees of freedom must be positive.")
    if excluded_count <= 0:
        raise ValueError("At least one excluded instrument is required.")

    diagnostics: dict[str, FirstStageDiagnostic] = {}
    for column, name in enumerate(endogenous_names):
        target = endogenous[:, column]
        coefficients, unrestricted_residuals = _ols_fit(full_instruments, target)
        unrestricted_sse = float(unrestricted_residuals @ unrestricted_residuals)

        if included_exogenous.shape[1]:
            _, restricted_residuals = _ols_fit(included_exogenous, target)
            restricted_sse = float(restricted_residuals @ restricted_residuals)
            centered = bool(np.allclose(included_exogenous[:, 0], 1.0))
        else:
            restricted_sse = float(target @ target)
            centered = False
        total_squares = (
            float(np.sum((target - target.mean()) ** 2)) if centered else float(target @ target)
        )
        r_squared = 1.0 - unrestricted_sse / total_squares if total_squares > 0.0 else float("nan")
        improvement = max(restricted_sse - unrestricted_sse, 0.0)
        partial_r_squared = improvement / restricted_sse if restricted_sse > 0.0 else float("nan")
        denominator = unrestricted_sse / denominator_df
        classical_f = (
            (improvement / excluded_count) / denominator if denominator > 0.0 else float("inf")
        )
        classical_p = float(f.sf(classical_f, excluded_count, denominator_df))

        covariance_estimate = ols_covariance(
            full_instruments,
            unrestricted_residuals,
            covariance=covariance,
            clusters=clusters,
        )
        excluded_coefficients = coefficients[-excluded_count:]
        excluded_covariance = covariance_estimate.matrix[-excluded_count:, -excluded_count:]
        wald = float(
            excluded_coefficients @ np.linalg.pinv(excluded_covariance) @ excluded_coefficients
        )
        if covariance_estimate.distribution == "normal":
            excluded_statistic = wald
            excluded_p = float(chi2.sf(wald, excluded_count))
            excluded_distribution = "chi2"
        else:
            excluded_statistic = wald / excluded_count
            assert covariance_estimate.df is not None
            excluded_p = float(f.sf(excluded_statistic, excluded_count, covariance_estimate.df))
            excluded_distribution = "F"

        diagnostics[name] = FirstStageDiagnostic(
            endogenous=name,
            r_squared=float(r_squared),
            partial_r_squared=float(partial_r_squared),
            classical_f_statistic=float(classical_f),
            classical_f_df_num=int(excluded_count),
            classical_f_df_denom=int(denominator_df),
            classical_f_p_value=classical_p,
            excluded_instrument_statistic=float(excluded_statistic),
            excluded_instrument_df=int(excluded_count),
            excluded_instrument_p_value=excluded_p,
            excluded_instrument_distribution=excluded_distribution,
            weak_instrument_warning=bool(np.isfinite(classical_f) and classical_f < 10.0),
        )
    return diagnostics


def sargan_test(
    residuals: np.ndarray,
    instruments: np.ndarray,
    *,
    overidentification_df: int,
) -> SarganTest:
    """Compute the homoskedastic Sargan statistic for an overidentified model."""

    if overidentification_df <= 0:
        raise ValueError("Sargan testing requires at least one overidentifying restriction.")
    sigma2 = float(residuals @ residuals / len(residuals))
    if sigma2 <= 0.0:
        statistic = 0.0
    else:
        moment = instruments.T @ residuals
        statistic = float(moment @ np.linalg.pinv(instruments.T @ instruments) @ moment / sigma2)
    return SarganTest(
        statistic=statistic,
        df=int(overidentification_df),
        p_value=float(chi2.sf(statistic, overidentification_df)),
    )

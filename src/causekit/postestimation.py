"""Ecosystem-aligned post-estimation for causekit results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, f, norm, t

from .iv import IV2SLSResult
from .panel_iv import PanelIV2SLSResult
from .rd import RegressionDiscontinuityResult

InferenceResult = IV2SLSResult | PanelIV2SLSResult | RegressionDiscontinuityResult


def summary_frame(result: InferenceResult, *, level: float = 0.95) -> pd.DataFrame:
    """Return a defensive coefficient summary."""

    return result.summary_frame(level=level).copy()


def vcov(result: InferenceResult) -> pd.DataFrame:
    """Return a defensive copy of the full fitted covariance matrix."""

    return result.covariance.copy()


def confint(result: InferenceResult, *, level: float = 0.95) -> pd.DataFrame:
    """Return confidence intervals using the fitted reference distribution."""

    return result.conf_int(level=level).copy()


def predict(
    result: IV2SLSResult,
    endogenous: Any,
    exogenous: Any | None = None,
) -> pd.Series:
    """Predict structural outcomes from new regressors."""

    return result.predict(endogenous=endogenous, exogenous=exogenous)


def residuals(result: IV2SLSResult | PanelIV2SLSResult) -> pd.Series:
    """Return fitted structural residuals."""

    return result.residuals.copy()


def fitted_values(result: IV2SLSResult | PanelIV2SLSResult) -> pd.Series:
    """Return fitted structural outcomes."""

    return result.fitted_values.copy()


def _critical_value(result: InferenceResult, level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one.")
    probability = 0.5 + level / 2.0
    if result.inference_distribution == "normal":
        return float(norm.ppf(probability))
    if result.inference_df is None:
        raise ValueError("The fitted result does not expose inference degrees of freedom.")
    return float(t.ppf(probability, result.inference_df))


def lincom(
    result: InferenceResult,
    weights: Mapping[str, float],
    *,
    value: float = 0.0,
    level: float = 0.95,
) -> pd.Series:
    """Estimate and test a named linear combination using the full covariance."""

    if not weights:
        raise ValueError("weights must contain at least one parameter.")
    unknown = set(weights) - set(result.params.index)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}.")
    contrast = pd.Series(0.0, index=result.params.index)
    for name, weight in weights.items():
        contrast[name] = float(weight)
    vector = contrast.to_numpy(dtype=float)
    estimate = float(vector @ result.params.to_numpy(dtype=float))
    variance = float(vector @ result.covariance.to_numpy(dtype=float) @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    statistic = (estimate - float(value)) / standard_error if standard_error > 0.0 else float("nan")
    if result.inference_distribution == "normal":
        p_value = float(2.0 * norm.sf(abs(statistic)))
    else:
        assert result.inference_df is not None
        p_value = float(2.0 * t.sf(abs(statistic), result.inference_df))
    critical = _critical_value(result, level)
    return pd.Series(
        {
            "estimate": estimate,
            "standard_error": standard_error,
            "statistic": statistic,
            "p_value": p_value,
            "lower": estimate - critical * standard_error,
            "upper": estimate + critical * standard_error,
        },
        name="lincom",
    )


def wald_test(
    result: InferenceResult,
    restrictions: Mapping[str, float] | Sequence[Mapping[str, float]],
    *,
    values: float | Sequence[float] = 0.0,
) -> pd.Series:
    """Test one or more named linear restrictions."""

    rows = [restrictions] if isinstance(restrictions, Mapping) else list(restrictions)
    if not rows:
        raise ValueError("restrictions must contain at least one restriction.")
    matrix = np.zeros((len(rows), len(result.params)), dtype=float)
    for row_index, row in enumerate(rows):
        unknown = set(row) - set(result.params.index)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}.")
        for name, weight in row.items():
            matrix[row_index, result.params.index.get_loc(name)] = float(weight)
    if isinstance(values, (int, float, np.integer, np.floating)):
        null_values = np.full(len(rows), float(values))
    else:
        null_values = np.asarray(values, dtype=float)
        if null_values.shape != (len(rows),):
            raise ValueError("values must provide one null value per restriction.")
    difference = matrix @ result.params.to_numpy(dtype=float) - null_values
    restricted_covariance = matrix @ result.covariance.to_numpy(dtype=float) @ matrix.T
    chi_square = float(difference @ np.linalg.pinv(restricted_covariance) @ difference)
    numerator_df = int(np.linalg.matrix_rank(matrix))
    if result.inference_distribution == "normal":
        statistic = chi_square
        p_value = float(chi2.sf(statistic, numerator_df))
        distribution = f"chi2({numerator_df})"
        denominator_df = float("nan")
    else:
        if result.inference_df is None:
            raise ValueError("The fitted result does not expose inference degrees of freedom.")
        statistic = chi_square / numerator_df
        p_value = float(f.sf(statistic, numerator_df, result.inference_df))
        distribution = f"F({numerator_df}, {int(result.inference_df)})"
        denominator_df = float(result.inference_df)
    return pd.Series(
        {
            "statistic": statistic,
            "df_num": numerator_df,
            "df_denom": denominator_df,
            "p_value": p_value,
            "distribution": distribution,
        },
        name="wald_test",
    )


__all__ = [
    "confint",
    "fitted_values",
    "lincom",
    "predict",
    "residuals",
    "summary_frame",
    "vcov",
    "wald_test",
]

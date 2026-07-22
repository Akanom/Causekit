"""Covariance estimators shared by linear and instrumental-variable models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

CovarianceType = Literal["unadjusted", "robust", "clustered"]


@dataclass(frozen=True)
class CovarianceEstimate:
    """A covariance matrix and the reference distribution for inference."""

    matrix: np.ndarray
    distribution: str
    df: float | None
    n_clusters: int | None


def _validate_covariance_request(covariance: CovarianceType, clusters: object | None) -> None:
    if not isinstance(covariance, str) or covariance not in {
        "unadjusted",
        "robust",
        "clustered",
    }:
        raise ValueError("covariance must be 'unadjusted', 'robust', or 'clustered'.")
    if covariance == "clustered" and clusters is None:
        raise ValueError("clusters must be provided when covariance='clustered'.")
    if covariance != "clustered" and clusters is not None:
        raise ValueError("clusters may be provided only when covariance='clustered'.")


def _as_float_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real-valued data.")
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric two-dimensional array.") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _as_residual_vector(residuals: np.ndarray, *, nobs: int) -> np.ndarray:
    raw = np.asarray(residuals)
    if np.iscomplexobj(raw):
        raise ValueError("residuals must contain real-valued data.")
    try:
        vector = np.asarray(residuals, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("residuals must be a numeric one-dimensional array.") from exc
    if vector.ndim != 1:
        raise ValueError("residuals must be one-dimensional.")
    if vector.size != nobs:
        raise ValueError("residuals must contain exactly one value per observation.")
    if not np.isfinite(vector).all():
        raise ValueError("residuals must contain only finite values.")
    return vector


def _validate_degrees_of_freedom(nobs: int, nparams: int) -> None:
    if nobs <= nparams:
        raise ValueError("Covariance estimation requires more observations than parameters.")


def _cluster_codes(clusters: object, *, nobs: int) -> tuple[np.ndarray, int]:
    labels = np.asarray(clusters, dtype=object)
    if labels.ndim != 1 or labels.size != nobs:
        raise ValueError("clusters must contain exactly one label per observation.")

    missing = np.asarray(pd.isna(labels))
    if missing.shape != labels.shape:
        raise TypeError("cluster labels must be scalar values.")
    if missing.any():
        raise ValueError("clusters must not contain missing labels.")

    try:
        codes, unique_labels = pd.factorize(labels, sort=False)
    except TypeError as exc:
        raise TypeError("cluster labels must be hashable scalar values.") from exc
    n_clusters = int(len(unique_labels))
    if n_clusters < 2:
        raise ValueError("Clustered covariance requires at least two clusters.")
    return codes, n_clusters


def _sandwich(bread: np.ndarray, meat: np.ndarray) -> np.ndarray:
    covariance = bread @ meat @ bread
    covariance = 0.5 * (covariance + covariance.T)
    if not np.isfinite(covariance).all():
        raise FloatingPointError("Covariance calculation produced non-finite values.")
    return covariance


def _cluster_meat(scores: np.ndarray, codes: np.ndarray, n_clusters: int) -> np.ndarray:
    cluster_scores = np.zeros((n_clusters, scores.shape[1]), dtype=float)
    np.add.at(cluster_scores, codes, scores)
    return cluster_scores.T @ cluster_scores


def _estimate_covariance(
    bread: np.ndarray,
    scores: np.ndarray,
    residuals: np.ndarray,
    *,
    covariance: CovarianceType,
    clusters: object | None,
) -> CovarianceEstimate:
    nobs, nparams = scores.shape
    if covariance == "unadjusted":
        sigma2 = float(residuals @ residuals) / (nobs - nparams)
        matrix = sigma2 * bread
        matrix = 0.5 * (matrix + matrix.T)
        if not np.isfinite(matrix).all():
            raise FloatingPointError("Covariance calculation produced non-finite values.")
        return CovarianceEstimate(
            matrix=matrix,
            distribution="t",
            df=float(nobs - nparams),
            n_clusters=None,
        )

    if covariance == "robust":
        hc1 = nobs / (nobs - nparams)
        meat = hc1 * (scores.T @ scores)
        return CovarianceEstimate(
            matrix=_sandwich(bread, meat),
            distribution="normal",
            df=None,
            n_clusters=None,
        )

    if clusters is None:  # pragma: no cover - protected by request validation
        raise RuntimeError("Internal clustered-covariance configuration error.")
    codes, n_clusters = _cluster_codes(clusters, nobs=nobs)
    cr1 = (n_clusters / (n_clusters - 1.0)) * ((nobs - 1.0) / (nobs - nparams))
    meat = cr1 * _cluster_meat(scores, codes, n_clusters)
    return CovarianceEstimate(
        matrix=_sandwich(bread, meat),
        distribution="t",
        df=float(n_clusters - 1),
        n_clusters=n_clusters,
    )


def iv_covariance(
    X: np.ndarray,
    Z: np.ndarray,
    residuals: np.ndarray,
    *,
    covariance: CovarianceType,
    clusters: object | None = None,
) -> CovarianceEstimate:
    """Estimate a 2SLS covariance matrix without materializing the projection matrix."""

    _validate_covariance_request(covariance, clusters)
    x = _as_float_matrix(X, name="X")
    z = _as_float_matrix(Z, name="Z")
    if z.shape[0] != x.shape[0]:
        raise ValueError("X and Z must contain the same number of observations.")

    nobs, nparams = x.shape
    _validate_degrees_of_freedom(nobs, nparams)
    u = _as_residual_vector(residuals, nobs=nobs)

    ztz_inverse = np.linalg.pinv(z.T @ z, hermitian=True)
    xz_weighted = (x.T @ z) @ ztz_inverse
    information = xz_weighted @ (z.T @ x)
    bread = np.linalg.pinv(information, hermitian=True)

    # Each row is the 2SLS estimating-equation contribution, u_i * (P_Z X)_i.
    # Computing it from Z and X'Z(Z'Z)^+ avoids constructing the n-by-n P_Z.
    projected_scores = (z @ xz_weighted.T) * u[:, None]
    return _estimate_covariance(
        bread,
        projected_scores,
        u,
        covariance=covariance,
        clusters=clusters,
    )


def ols_covariance(
    X: np.ndarray,
    residuals: np.ndarray,
    *,
    covariance: CovarianceType,
    clusters: object | None = None,
) -> CovarianceEstimate:
    """Estimate an OLS homoskedastic, HC1, or one-way CR1 covariance matrix."""

    _validate_covariance_request(covariance, clusters)
    x = _as_float_matrix(X, name="X")
    nobs, nparams = x.shape
    _validate_degrees_of_freedom(nobs, nparams)
    u = _as_residual_vector(residuals, nobs=nobs)

    bread = np.linalg.pinv(x.T @ x, hermitian=True)
    scores = x * u[:, None]
    return _estimate_covariance(
        bread,
        scores,
        u,
        covariance=covariance,
        clusters=clusters,
    )


__all__ = ["CovarianceEstimate", "CovarianceType", "iv_covariance", "ols_covariance"]

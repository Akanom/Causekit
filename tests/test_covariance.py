from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from causalkit._covariance import CovarianceEstimate, iv_covariance, ols_covariance


@pytest.fixture
def ols_inputs() -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    residuals = np.array([0.4, -0.3, 0.2, -0.1, 0.5, -0.2])
    return x, residuals


@pytest.fixture
def iv_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array(
        [
            [1.0, 0.2],
            [1.0, 0.8],
            [1.0, 1.1],
            [1.0, 1.9],
            [1.0, 2.6],
            [1.0, 3.2],
        ]
    )
    z = np.array(
        [
            [1.0, -1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 9.0],
            [1.0, 4.0, 16.0],
        ]
    )
    residuals = np.array([0.5, -0.4, 0.1, 0.3, -0.2, -0.1])
    return x, z, residuals


def test_covariance_estimate_is_frozen() -> None:
    estimate = CovarianceEstimate(np.eye(1), "normal", None, None)

    with pytest.raises(FrozenInstanceError):
        estimate.df = 1.0  # type: ignore[misc]


def test_ols_unadjusted_matches_homoskedastic_formula(
    ols_inputs: tuple[np.ndarray, np.ndarray],
) -> None:
    x, residuals = ols_inputs
    result = ols_covariance(x, residuals, covariance="unadjusted")
    nobs, nparams = x.shape
    expected = (residuals @ residuals) / (nobs - nparams) * np.linalg.inv(x.T @ x)

    np.testing.assert_allclose(result.matrix, expected, rtol=1e-13, atol=1e-15)
    assert result.distribution == "t"
    assert result.df == float(nobs - nparams)
    assert result.n_clusters is None


def test_ols_hc1_matches_row_score_sandwich(
    ols_inputs: tuple[np.ndarray, np.ndarray],
) -> None:
    x, residuals = ols_inputs
    result = ols_covariance(x, residuals, covariance="robust")
    nobs, nparams = x.shape
    bread = np.linalg.inv(x.T @ x)
    meat = x.T @ ((residuals**2)[:, None] * x)
    expected = nobs / (nobs - nparams) * bread @ meat @ bread

    np.testing.assert_allclose(result.matrix, expected, rtol=1e-13, atol=1e-15)
    assert result.distribution == "normal"
    assert result.df is None
    assert result.n_clusters is None


def test_ols_cr1_matches_cluster_score_sandwich_and_accepts_string_labels(
    ols_inputs: tuple[np.ndarray, np.ndarray],
) -> None:
    x, residuals = ols_inputs
    clusters = np.array(["north", "north", "south", "south", "west", "west"])
    result = ols_covariance(x, residuals, covariance="clustered", clusters=clusters)
    nobs, nparams = x.shape
    bread = np.linalg.inv(x.T @ x)
    meat = np.zeros((nparams, nparams))
    for label in ("north", "south", "west"):
        score = x[clusters == label].T @ residuals[clusters == label]
        meat += np.outer(score, score)
    n_clusters = 3
    correction = (n_clusters / (n_clusters - 1)) * ((nobs - 1) / (nobs - nparams))
    expected = correction * bread @ meat @ bread

    np.testing.assert_allclose(result.matrix, expected, rtol=1e-13, atol=1e-15)
    assert result.distribution == "t"
    assert result.df == 2.0
    assert result.n_clusters == 3


def test_iv_unadjusted_matches_2sls_homoskedastic_formula(
    iv_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x, z, residuals = iv_inputs
    result = iv_covariance(x, z, residuals, covariance="unadjusted")
    nobs, nparams = x.shape
    ztz_inverse = np.linalg.inv(z.T @ z)
    bread = np.linalg.inv(x.T @ z @ ztz_inverse @ z.T @ x)
    expected = (residuals @ residuals) / (nobs - nparams) * bread

    np.testing.assert_allclose(result.matrix, expected, rtol=1e-12, atol=1e-14)
    assert result.distribution == "t"
    assert result.df == 4.0


def test_iv_hc1_matches_projected_score_sandwich(
    iv_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x, z, residuals = iv_inputs
    result = iv_covariance(x, z, residuals, covariance="robust")
    nobs, nparams = x.shape
    ztz_inverse = np.linalg.inv(z.T @ z)
    projected_x = z @ ztz_inverse @ z.T @ x
    bread = np.linalg.inv(x.T @ projected_x)
    meat = projected_x.T @ ((residuals**2)[:, None] * projected_x)
    expected = nobs / (nobs - nparams) * bread @ meat @ bread

    np.testing.assert_allclose(result.matrix, expected, rtol=1e-12, atol=1e-14)
    assert result.distribution == "normal"
    assert result.df is None


def test_iv_cr1_matches_clustered_projected_scores(
    iv_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x, z, residuals = iv_inputs
    clusters = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    result = iv_covariance(
        x,
        z,
        residuals,
        covariance="clustered",
        clusters=clusters,
    )
    nobs, nparams = x.shape
    ztz_inverse = np.linalg.inv(z.T @ z)
    projected_x = z @ ztz_inverse @ z.T @ x
    bread = np.linalg.inv(x.T @ projected_x)
    meat = np.zeros((nparams, nparams))
    for label in ("a", "b", "c"):
        score = projected_x[clusters == label].T @ residuals[clusters == label]
        meat += np.outer(score, score)
    correction = (3 / 2) * ((nobs - 1) / (nobs - nparams))
    expected = correction * bread @ meat @ bread

    np.testing.assert_allclose(result.matrix, expected, rtol=1e-12, atol=1e-14)
    assert result.distribution == "t"
    assert result.df == 2.0
    assert result.n_clusters == 3


def test_singular_designs_use_pseudoinverses() -> None:
    x = np.column_stack([np.ones(5), np.ones(5)])
    z = np.column_stack([np.ones(5), np.arange(5.0), np.arange(5.0)])
    residuals = np.array([0.1, -0.2, 0.3, -0.1, -0.1])

    ols_result = ols_covariance(x, residuals, covariance="robust")
    iv_result = iv_covariance(x, z, residuals, covariance="robust")

    assert np.isfinite(ols_result.matrix).all()
    assert np.isfinite(iv_result.matrix).all()
    np.testing.assert_allclose(ols_result.matrix, ols_result.matrix.T, atol=0.0)
    np.testing.assert_allclose(iv_result.matrix, iv_result.matrix.T, atol=0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"covariance": "invalid"}, "covariance must be"),
        ({"covariance": "clustered"}, "clusters must be provided"),
        (
            {"covariance": "robust", "clusters": ["a", "a", "b", "b", "c", "c"]},
            "clusters may be provided only",
        ),
        (
            {"covariance": "clustered", "clusters": ["a", "a"]},
            "exactly one label",
        ),
        (
            {"covariance": "clustered", "clusters": ["a", "a", None, "b", "b", "b"]},
            "missing labels",
        ),
        (
            {"covariance": "clustered", "clusters": ["a"] * 6},
            "at least two clusters",
        ),
    ],
)
def test_ols_rejects_invalid_covariance_requests(
    ols_inputs: tuple[np.ndarray, np.ndarray],
    kwargs: dict[str, object],
    match: str,
) -> None:
    x, residuals = ols_inputs

    with pytest.raises(ValueError, match=match):
        ols_covariance(x, residuals, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("x", "residuals", "match"),
    [
        (np.ones(5), np.ones(5), "X must be two-dimensional"),
        (np.empty((5, 0)), np.ones(5), "at least one row and one column"),
        (np.ones((3, 3)), np.ones(3), "more observations than parameters"),
        (np.ones((5, 2)), np.ones((5, 1)), "residuals must be one-dimensional"),
        (np.ones((5, 2)), np.ones(4), "exactly one value per observation"),
        (
            np.array([[1.0, 0.0], [1.0, np.nan], [1.0, 2.0]]),
            np.ones(3),
            "X must contain only finite values",
        ),
        (
            np.column_stack([np.ones(5), np.arange(5.0)]),
            np.array([0.0, 0.0, np.inf, 0.0, 0.0]),
            "residuals must contain only finite values",
        ),
    ],
)
def test_ols_rejects_invalid_shapes_and_nonfinite_values(
    x: np.ndarray,
    residuals: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ols_covariance(x, residuals, covariance="robust")


def test_iv_rejects_invalid_instrument_matrix(
    iv_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x, z, residuals = iv_inputs

    with pytest.raises(ValueError, match="same number of observations"):
        iv_covariance(x, z[:-1], residuals, covariance="robust")
    with pytest.raises(ValueError, match="Z must be two-dimensional"):
        iv_covariance(x, z[:, 0], residuals, covariance="robust")
    z[0, 0] = np.inf
    with pytest.raises(ValueError, match="Z must contain only finite values"):
        iv_covariance(x, z, residuals, covariance="robust")

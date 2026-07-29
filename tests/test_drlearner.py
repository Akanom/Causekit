"""Public contracts for the separately identified honest DR learner."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from causekit import CATEEstimatorProtocol, DRLearner, DRLearnerResult


class _ColumnOutcomeResult:
    def __init__(self, column: str) -> None:
        self.column = column

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self.column].copy()


class _ColumnOutcomeEstimator:
    def __init__(self, log: list[tuple[str, tuple[object, ...], tuple[float, ...]]]) -> None:
        self.log = log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnOutcomeResult:
        arm = tuple(np.unique(X["arm_marker"].to_numpy(dtype=float)))
        self.log.append(("outcome", tuple(X.index), arm))
        if arm == (1.0,):
            return _ColumnOutcomeResult("mu1")
        if arm == (0.0,):
            return _ColumnOutcomeResult("mu0")
        raise AssertionError("Each outcome nuisance fit must receive exactly one arm.")


class _ColumnPropensityResult:
    classes_ = np.array([0.0, 1.0])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        propensity = X["propensity"].astype(float)
        return pd.DataFrame({0.0: 1.0 - propensity, 1.0: propensity}, index=X.index)


class _ColumnPropensityEstimator:
    def __init__(self, log: list[tuple[str, tuple[object, ...], tuple[float, ...]]]) -> None:
        self.log = log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnPropensityResult:
        self.log.append(("propensity", tuple(X.index), tuple(np.unique(y))))
        return _ColumnPropensityResult()


class _ColumnCATEResult:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["cate_score"].copy()


class _ColumnCATEEstimator:
    def __init__(self, log: list[tuple[str, tuple[object, ...], tuple[float, ...]]]) -> None:
        self.log = log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnCATEResult:
        self.log.append(("cate", tuple(X.index), tuple()))
        return _ColumnCATEResult()


def _data(nobs: int = 120) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = pd.Index([f"dr-{position}" for position in range(nobs)])
    x = np.linspace(-1.8, 1.8, nobs)
    treatment = pd.Series(np.tile([0.0, 1.0], nobs // 2), index=index, name="treatment")
    propensity = np.where(x < 0.0, 0.35, 0.65)
    mu0 = 0.4 + 0.25 * x
    cate = 0.9 + 0.55 * x
    mu1 = mu0 + cate
    noise = 0.08 * np.sin(4.0 * x) + 0.03 * np.cos(7.0 * x)
    outcome = pd.Series(
        np.where(treatment.to_numpy() == 1.0, mu1, mu0) + noise,
        index=index,
        name="outcome",
    )
    covariates = pd.DataFrame(
        {
            "x": x,
            "mu0": mu0,
            "mu1": mu1,
            "propensity": propensity,
            "cate_score": 0.7 + 0.8 * cate,
            "arm_marker": treatment,
        },
        index=index,
    )
    return covariates, treatment, outcome


def _model(
    log: list[tuple[str, tuple[object, ...], tuple[float, ...]]],
    **options: Any,
) -> DRLearner:
    defaults: dict[str, Any] = {
        "outcome_factory": lambda: _ColumnOutcomeEstimator(log),
        "propensity_factory": lambda: _ColumnPropensityEstimator(log),
        "cate_factory": lambda: _ColumnCATEEstimator(log),
        "n_splits": 3,
        "evaluation_fraction": 0.5,
        "random_state": 811,
        "overlap_floor": 0.05,
        "calibration_groups": 3,
        "simultaneous_level": 0.9,
        "bootstrap_iterations": 199,
        "bootstrap_random_state": 314,
    }
    defaults.update(options)
    return DRLearner(**defaults)


def _dr_score(
    y: pd.Series,
    w: pd.Series,
    mu0: pd.Series,
    mu1: pd.Series,
    propensity: pd.Series,
) -> pd.Series:
    return mu1 - mu0 + w * (y - mu1) / propensity - (1.0 - w) * (y - mu0) / (1.0 - propensity)


def _hc1(design: np.ndarray, residual: np.ndarray) -> np.ndarray:
    nobs, nparams = design.shape
    bread = np.linalg.inv(design.T @ design)
    scores = design * residual[:, None]
    return bread @ (nobs / (nobs - nparams) * scores.T @ scores) @ bread


def _cr1(design: np.ndarray, residual: np.ndarray, clusters: pd.Series) -> np.ndarray:
    nobs, nparams = design.shape
    bread = np.linalg.inv(design.T @ design)
    scores = design * residual[:, None]
    codes, labels = pd.factorize(clusters, sort=False)
    cluster_scores = np.zeros((len(labels), nparams))
    np.add.at(cluster_scores, codes, scores)
    scale = len(labels) / (len(labels) - 1) * (nobs - 1) / (nobs - nparams)
    return bread @ (scale * cluster_scores.T @ cluster_scores) @ bread


def test_hand_computed_dr_score_honest_loss_calibration_and_groups() -> None:
    covariates, treatment, outcome = _data()
    log: list[tuple[str, tuple[object, ...], tuple[float, ...]]] = []
    result = _model(log).fit(outcome, treatment=treatment, covariates=covariates)

    assert isinstance(result, DRLearnerResult)
    construction = result.construction_index
    evaluation = result.evaluation_index
    expected_construction_score = _dr_score(
        outcome.loc[construction],
        treatment.loc[construction],
        covariates.loc[construction, "mu0"],
        covariates.loc[construction, "mu1"],
        covariates.loc[construction, "propensity"],
    )
    expected_evaluation_score = _dr_score(
        outcome.loc[evaluation],
        treatment.loc[evaluation],
        covariates.loc[evaluation, "mu0"],
        covariates.loc[evaluation, "mu1"],
        covariates.loc[evaluation, "propensity"],
    )
    pd.testing.assert_series_equal(
        result.construction_dr_score,
        expected_construction_score.rename("dr_score"),
    )
    pd.testing.assert_series_equal(
        result.evaluation_dr_score,
        expected_evaluation_score.rename("dr_score"),
    )

    constant = float(expected_construction_score.mean())
    score = covariates.loc[evaluation, "cate_score"]
    expected_loss = float(np.mean((expected_evaluation_score - score) ** 2))
    expected_constant_loss = float(np.mean((expected_evaluation_score - constant) ** 2))
    assert result.construction_constant_effect == pytest.approx(constant, abs=1e-14)
    assert result.honest_dr_loss == pytest.approx(expected_loss, abs=1e-14)
    assert result.honest_constant_dr_loss == pytest.approx(expected_constant_loss, abs=1e-14)
    assert result.dr_loss_gain == pytest.approx(1.0 - expected_loss / expected_constant_loss)

    center = float(score.mean())
    design = np.column_stack([np.ones(len(evaluation)), score.to_numpy() - center])
    expected_beta = np.linalg.solve(
        design.T @ design,
        design.T @ expected_evaluation_score.to_numpy(),
    )
    np.testing.assert_allclose(result.calibration_coefficients, expected_beta, atol=1e-14)
    calibration_residual = expected_evaluation_score.to_numpy() - design @ expected_beta
    np.testing.assert_allclose(
        result.calibration_covariance,
        _hc1(design, calibration_residual),
        atol=1e-14,
    )
    assert result.calibration_center == pytest.approx(center, abs=1e-14)
    for group in result.group_effects.index:
        mask = result.evaluation_group.eq(group)
        assert result.group_effects.loc[group, "effect"] == pytest.approx(
            expected_evaluation_score.loc[mask].mean(), abs=1e-14
        )

    assignments = result.evaluation_group.to_numpy(dtype=int) - 1
    group_design = np.zeros((len(evaluation), result.calibration_groups))
    group_design[np.arange(len(evaluation)), assignments] = 1.0
    expected_group_effect = np.linalg.solve(
        group_design.T @ group_design,
        group_design.T @ expected_evaluation_score.to_numpy(),
    )
    group_residual = expected_evaluation_score.to_numpy() - group_design @ expected_group_effect
    expected_group_covariance = _hc1(group_design, group_residual)
    np.testing.assert_allclose(result.group_covariance, expected_group_covariance, atol=1e-14)
    n_eval, n_groups = group_design.shape
    bread = np.linalg.inv(group_design.T @ group_design)
    expected_influence = n_eval * (group_design * group_residual[:, None]) @ bread
    np.testing.assert_allclose(result.group_influence, expected_influence, atol=1e-14)
    multipliers = np.random.default_rng(314).choice(np.array([-1.0, 1.0]), size=(199, n_eval))
    perturbations = (
        np.sqrt(n_eval / (n_eval - n_groups)) * multipliers @ expected_influence / n_eval
    )
    standard_errors = np.sqrt(np.diag(expected_group_covariance))
    maximum = np.max(np.abs(perturbations / standard_errors), axis=1)
    expected_critical = float(np.quantile(maximum, 0.9, method="higher"))
    assert result.simultaneous_critical_value == pytest.approx(expected_critical, abs=1e-14)

    assert all(set(rows).isdisjoint(set(evaluation)) for _, rows, _ in log)
    assert all(arms in {(0.0,), (1.0,)} for task, _, arms in log if task == "outcome")


def test_public_cate_protocol_and_prediction_boundary_are_unweighted() -> None:
    assert isinstance(_ColumnCATEEstimator([]), CATEEstimatorProtocol)
    covariates, treatment, outcome = _data()
    result = _model([]).fit(outcome, treatment=treatment, covariates=covariates)
    new = covariates.iloc[:2].copy()
    new.index = pd.Index(["new-a", "new-b"])
    pd.testing.assert_series_equal(result.predict(new), new["cate_score"].rename("cate"))

    exposed_role = result.sample_role
    exposed_role.iloc[0] = "changed"
    assert not exposed_role.equals(result.sample_role)
    exposed_groups = result.evaluation_group
    exposed_groups.iloc[0] = 99
    assert not exposed_groups.equals(result.evaluation_group)
    calibration_data = result.calibration_plot_data()
    calibration_data.iloc[0, calibration_data.columns.get_loc("effect")] = -999.0
    assert not calibration_data.equals(result.calibration_plot_data())
    assert result.cate_distribution_data()["sample_role"].eq("evaluation").all()


def test_optional_dr_graphs_use_only_retained_honest_surfaces() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    covariates, treatment, outcome = _data()
    result = _model([]).fit(outcome, treatment=treatment, covariates=covariates)
    calibration_ax = result.plot_calibration()
    distribution_ax = result.plot_cate_distribution(bins=7)

    assert calibration_ax.get_ylabel() == "Mean doubly robust group score"
    assert distribution_ax.get_xlabel() == "Honest predicted CATE"
    assert result.graph_metadata()["score"] == "augmented_inverse_probability"
    with pytest.raises(ValueError, match="bins"):
        result.plot_cate_distribution(bins=0)
    plt.close(calibration_ax.figure)
    plt.close(distribution_ax.figure)


def test_drlearner_preserves_whole_cluster_roles_and_clustered_inference() -> None:
    covariates, treatment, outcome = _data()
    clusters = pd.Series(np.repeat(np.arange(30), 4), index=covariates.index)
    result = _model(
        [],
        covariance="clustered",
        n_splits=2,
        calibration_groups=2,
        random_state=225,
    ).fit(outcome, treatment=treatment, covariates=covariates, clusters=clusters)

    assert result.sample_role.groupby(clusters).nunique().eq(1).all()
    construction_clusters = clusters.loc[result.construction_index]
    assert result.construction_fold.groupby(construction_clusters).nunique().eq(1).all()
    assert result.inference_distribution == "t"
    assert result.inference_df == clusters.loc[result.evaluation_index].nunique() - 1
    evaluation = result.evaluation_index
    score = covariates.loc[evaluation, "cate_score"]
    dr_score = _dr_score(
        outcome.loc[evaluation],
        treatment.loc[evaluation],
        covariates.loc[evaluation, "mu0"],
        covariates.loc[evaluation, "mu1"],
        covariates.loc[evaluation, "propensity"],
    )
    design = np.column_stack([np.ones(len(evaluation)), score - score.mean()])
    coefficients = np.linalg.solve(design.T @ design, design.T @ dr_score)
    residual = dr_score.to_numpy() - design @ coefficients
    np.testing.assert_allclose(
        result.calibration_covariance,
        _cr1(design, residual, clusters.loc[evaluation]),
        atol=1e-14,
    )


def test_drlearner_refuses_overlap_leakage_and_wrong_public_contracts() -> None:
    covariates, treatment, outcome = _data()
    covariates["propensity"] = 0.99
    with pytest.raises(ValueError, match="overlap interval"):
        _model([]).fit(outcome, treatment=treatment, covariates=covariates)

    class RecycledFactory:
        def __init__(self) -> None:
            self.estimator = _ColumnOutcomeEstimator([])

        def __call__(self) -> _ColumnOutcomeEstimator:
            return self.estimator

    with pytest.raises(ValueError, match="fresh estimator"):
        DRLearner(
            outcome_factory=RecycledFactory(),
            n_splits=3,
            evaluation_fraction=0.5,
            random_state=8,
            calibration_groups=2,
            bootstrap_iterations=99,
        ).fit(outcome, treatment=treatment, covariates=covariates.assign(propensity=0.5))

    class WeightedOnly:
        def fit(self, X: Any, y: Any, *, sample_weight: Any) -> _ColumnCATEResult:
            return _ColumnCATEResult()

    with pytest.raises(TypeError, match=r"fit\(X, pseudo_outcome\)"):
        _model([], cate_factory=lambda: WeightedOnly()).fit(
            outcome,
            treatment=treatment,
            covariates=covariates.assign(propensity=0.5),
        )


def test_drlearner_refuses_constant_scores_and_invalid_inference_configuration() -> None:
    with pytest.raises(ValueError, match="covariance"):
        DRLearner(covariance="unadjusted")
    with pytest.raises(ValueError, match="calibration_groups"):
        DRLearner(calibration_groups=1)
    covariates, treatment, outcome = _data()
    clusters = pd.Series(np.repeat(np.arange(30), 4), index=covariates.index)
    with pytest.raises(ValueError, match="only when covariance='clustered'"):
        _model([]).fit(
            outcome,
            treatment=treatment,
            covariates=covariates,
            clusters=clusters,
        )
    with pytest.raises(ValueError, match="must be provided"):
        _model([], covariance="clustered").fit(
            outcome,
            treatment=treatment,
            covariates=covariates,
        )
    covariates["cate_score"] = 1.0
    with pytest.raises(ValueError, match="unique CATE"):
        _model([]).fit(outcome, treatment=treatment, covariates=covariates)

"""Public contracts for honest R-learner evaluation and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from causekit import RLearner, RLearnerResult


class _ColumnOutcomeResult:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["outcome_mean"].copy()


class _ColumnOutcomeEstimator:
    def __init__(self, training_log: list[tuple[str, tuple[object, ...]]]) -> None:
        self.training_log = training_log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnOutcomeResult:
        self.training_log.append(("outcome", tuple(X.index)))
        return _ColumnOutcomeResult()


class _ColumnPropensityResult:
    classes_ = np.array([0.0, 1.0])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        treated = X["propensity"].astype(float)
        return pd.DataFrame({0.0: 1.0 - treated, 1.0: treated}, index=X.index)


class _ColumnPropensityEstimator:
    def __init__(self, training_log: list[tuple[str, tuple[object, ...]]]) -> None:
        self.training_log = training_log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnPropensityResult:
        self.training_log.append(("propensity", tuple(X.index)))
        return _ColumnPropensityResult()


class _ColumnCATEResult:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["cate_score"].copy()


class _ColumnCATEEstimator:
    def __init__(self, training_log: list[tuple[str, tuple[object, ...]]]) -> None:
        self.training_log = training_log

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series,
    ) -> _ColumnCATEResult:
        assert len(X) == len(y) == len(sample_weight)
        self.training_log.append(("cate", tuple(X.index)))
        return _ColumnCATEResult()


def _data(nobs: int = 96) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = pd.Index([f"unit-{position}" for position in range(nobs)])
    x = np.linspace(-1.8, 1.8, nobs)
    treatment = pd.Series(np.tile([0.0, 1.0], nobs // 2), index=index, name="treatment")
    propensity = np.where(x < 0.0, 0.38, 0.62)
    outcome_mean = 0.8 + 0.35 * x
    cate_score = 0.6 + 0.7 * x
    v = treatment.to_numpy(dtype=float) - propensity
    noise = 0.12 * np.sin(4.0 * x) + 0.04 * np.cos(7.0 * x)
    outcome = pd.Series(
        outcome_mean + (1.1 + 0.75 * cate_score) * v + noise,
        index=index,
        name="outcome",
    )
    covariates = pd.DataFrame(
        {
            "x": x,
            "outcome_mean": outcome_mean,
            "propensity": propensity,
            "cate_score": cate_score,
        },
        index=index,
    )
    return covariates, treatment, outcome


def _model(
    training_log: list[tuple[str, tuple[object, ...]]],
    **options: Any,
) -> RLearner:
    defaults: dict[str, Any] = {
        "outcome_factory": lambda: _ColumnOutcomeEstimator(training_log),
        "propensity_factory": lambda: _ColumnPropensityEstimator(training_log),
        "cate_factory": lambda: _ColumnCATEEstimator(training_log),
        "n_splits": 3,
        "evaluation_fraction": 0.5,
        "random_state": 627,
        "overlap_floor": 0.05,
        "calibration_groups": 3,
        "simultaneous_level": 0.9,
        "bootstrap_iterations": 199,
        "bootstrap_random_state": 918,
    }
    defaults.update(options)
    return RLearner(**defaults)


def _sandwich(
    design: np.ndarray,
    residual: np.ndarray,
    *,
    clusters: pd.Series | None = None,
) -> np.ndarray:
    nobs, nparams = design.shape
    bread = np.linalg.inv(design.T @ design)
    scores = design * residual[:, None]
    if clusters is None:
        meat = nobs / (nobs - nparams) * scores.T @ scores
    else:
        codes, labels = pd.factorize(clusters, sort=False)
        cluster_scores = np.zeros((len(labels), nparams))
        np.add.at(cluster_scores, codes, scores)
        cr1 = len(labels) / (len(labels) - 1) * (nobs - 1) / (nobs - nparams)
        meat = cr1 * cluster_scores.T @ cluster_scores
    return bread @ meat @ bread


def test_hand_computed_honest_r_loss_calibration_groups_and_max_t_band() -> None:
    covariates, treatment, outcome = _data()
    training_log: list[tuple[str, tuple[object, ...]]] = []
    result = _model(training_log).fit(
        outcome,
        treatment=treatment,
        covariates=covariates,
    )

    assert isinstance(result, RLearnerResult)
    construction = result.construction_index
    evaluation = result.evaluation_index
    v_construction = treatment.loc[construction] - covariates.loc[construction, "propensity"]
    u_construction = outcome.loc[construction] - covariates.loc[construction, "outcome_mean"]
    expected_constant = float(np.sum(v_construction * u_construction) / np.sum(v_construction**2))
    v = treatment.loc[evaluation] - covariates.loc[evaluation, "propensity"]
    u = outcome.loc[evaluation] - covariates.loc[evaluation, "outcome_mean"]
    score = covariates.loc[evaluation, "cate_score"]
    expected_r_loss = float(np.mean((u - v * score) ** 2))
    expected_constant_loss = float(np.mean((u - v * expected_constant) ** 2))

    assert result.construction_constant_effect == pytest.approx(expected_constant, abs=1e-14)
    assert result.honest_r_loss == pytest.approx(expected_r_loss, abs=1e-14)
    assert result.honest_constant_r_loss == pytest.approx(expected_constant_loss, abs=1e-14)
    assert result.r_loss_gain == pytest.approx(
        1.0 - expected_r_loss / expected_constant_loss, abs=1e-14
    )

    center = float(np.sum(v**2 * score) / np.sum(v**2))
    calibration_design = np.column_stack([v, v * (score - center)])
    expected_beta = np.linalg.solve(
        calibration_design.T @ calibration_design, calibration_design.T @ u
    )
    calibration_residual = u.to_numpy() - calibration_design @ expected_beta
    expected_covariance = _sandwich(calibration_design, calibration_residual)
    np.testing.assert_allclose(result.calibration_coefficients, expected_beta, atol=1e-14)
    np.testing.assert_allclose(result.calibration_covariance, expected_covariance, atol=1e-14)
    assert result.calibration_center == pytest.approx(center, abs=1e-14)
    heterogeneity_se = np.sqrt(expected_covariance[1, 1])
    expected_one_statistic = (expected_beta[1] - 1.0) / heterogeneity_se
    assert result.calibration_tests.loc["heterogeneity=1", "statistic"] == pytest.approx(
        expected_one_statistic, abs=1e-14
    )

    assignments = result.evaluation_group.to_numpy(dtype=int) - 1
    group_design = np.zeros((len(evaluation), result.calibration_groups))
    group_design[np.arange(len(evaluation)), assignments] = v
    expected_group_effect = np.linalg.solve(group_design.T @ group_design, group_design.T @ u)
    group_residual = u.to_numpy() - group_design @ expected_group_effect
    expected_group_covariance = _sandwich(group_design, group_residual)
    np.testing.assert_allclose(result.group_effects["effect"], expected_group_effect, atol=1e-14)
    np.testing.assert_allclose(
        result.group_effects["std_err"],
        np.sqrt(np.diag(expected_group_covariance)),
        atol=1e-14,
    )

    nobs, nparams = group_design.shape
    bread = np.linalg.inv(group_design.T @ group_design)
    influence = nobs * (group_design * group_residual[:, None]) @ bread
    np.testing.assert_allclose(result.group_influence, influence, atol=1e-14)
    multipliers = np.random.default_rng(918).choice(np.array([-1.0, 1.0]), size=(199, nobs))
    perturbations = np.sqrt(nobs / (nobs - nparams)) * multipliers @ influence / nobs
    max_statistics = np.max(
        np.abs(perturbations / np.sqrt(np.diag(expected_group_covariance))), axis=1
    )
    expected_critical = float(np.quantile(max_statistics, 0.9, method="higher"))
    assert result.simultaneous_critical_value == pytest.approx(expected_critical, abs=1e-14)
    np.testing.assert_allclose(
        result.group_effects["simultaneous_lower"],
        expected_group_effect - expected_critical * np.sqrt(np.diag(expected_group_covariance)),
        atol=1e-14,
    )
    assert all(set(rows).isdisjoint(set(evaluation)) for _, rows in training_log)


def test_clustered_calibration_and_group_covariance_use_evaluation_cluster_sums() -> None:
    rng = np.random.default_rng(93)
    n_clusters, rows_per_cluster = 24, 4
    nobs = n_clusters * rows_per_cluster
    index = pd.Index([f"clustered-{position}" for position in range(nobs)])
    cluster = pd.Series(
        np.repeat([f"g-{i}" for i in range(n_clusters)], rows_per_cluster), index=index
    )
    treatment = pd.Series(np.tile([0.0, 1.0, 0.0, 1.0], n_clusters), index=index)
    x = np.repeat(np.linspace(-1.5, 1.5, n_clusters), rows_per_cluster) + np.tile(
        [-0.03, -0.01, 0.01, 0.03], n_clusters
    )
    propensity = np.full(nobs, 0.5)
    outcome_mean = 0.2 + 0.3 * x
    cate_score = 0.7 + 0.6 * x
    v = treatment.to_numpy() - propensity
    cluster_noise = np.repeat(rng.normal(scale=0.08, size=n_clusters), rows_per_cluster)
    outcome = pd.Series(outcome_mean + v * (1.0 + 0.8 * cate_score) + cluster_noise, index=index)
    covariates = pd.DataFrame(
        {
            "x": x,
            "outcome_mean": outcome_mean,
            "propensity": propensity,
            "cate_score": cate_score,
        },
        index=index,
    )
    result = _model(
        [],
        covariance="clustered",
        calibration_groups=2,
        n_splits=2,
        random_state=331,
        bootstrap_random_state=442,
    ).fit(outcome, treatment=treatment, covariates=covariates, clusters=cluster)

    evaluation = result.evaluation_index
    evaluation_cluster = cluster.loc[evaluation]
    v_eval = treatment.loc[evaluation] - covariates.loc[evaluation, "propensity"]
    u_eval = outcome.loc[evaluation] - covariates.loc[evaluation, "outcome_mean"]
    score = covariates.loc[evaluation, "cate_score"]
    center = float(np.sum(v_eval**2 * score) / np.sum(v_eval**2))
    design = np.column_stack([v_eval, v_eval * (score - center)])
    beta = np.linalg.solve(design.T @ design, design.T @ u_eval)
    residual = u_eval.to_numpy() - design @ beta
    expected = _sandwich(design, residual, clusters=evaluation_cluster)

    np.testing.assert_allclose(result.calibration_covariance, expected, atol=1e-14)
    assert result.sample_role.groupby(cluster).nunique().eq(1).all()
    assert (
        result.construction_fold.groupby(cluster.loc[result.construction_index])
        .nunique()
        .eq(1)
        .all()
    )
    assert result.inference_distribution == "t"
    assert result.inference_df == evaluation_cluster.nunique() - 1
    assert result.n_clusters == evaluation_cluster.nunique()
    assert result.group_effects["n_clusters"].ge(2).all()


def test_result_role_and_graph_data_views_are_copies_and_prediction_preserves_index() -> None:
    covariates, treatment, outcome = _data()
    result = _model([]).fit(outcome, treatment=treatment, covariates=covariates)

    exposed_role = result.sample_role
    exposed_role.iloc[0] = (
        "evaluation" if exposed_role.iloc[0] == "construction" else "construction"
    )
    assert not exposed_role.equals(result.sample_role)
    exposed_groups = result.evaluation_group
    exposed_groups.iloc[0] = 99
    assert not exposed_groups.equals(result.evaluation_group)

    calibration_data = result.calibration_plot_data()
    assert calibration_data.index.equals(result.group_effects.index)
    calibration_data.iloc[0, calibration_data.columns.get_loc("effect")] = -999.0
    assert not calibration_data.equals(result.calibration_plot_data())
    distribution = result.cate_distribution_data()
    assert distribution.index.equals(result.evaluation_index)
    assert distribution["sample_role"].eq("evaluation").all()

    new_index = pd.Index(["new-a", "new-b"])
    new_data = covariates.iloc[:2].copy()
    new_data.index = new_index
    prediction = result.predict(new_data)
    pd.testing.assert_series_equal(
        prediction,
        new_data["cate_score"].rename("cate"),
    )


def test_optional_graph_methods_render_from_retained_honest_data() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    covariates, treatment, outcome = _data()
    result = _model([]).fit(outcome, treatment=treatment, covariates=covariates)
    calibration_ax = result.plot_calibration()
    distribution_ax = result.plot_cate_distribution(bins=7)

    assert calibration_ax.get_xlabel() == "Honest mean predicted CATE"
    assert calibration_ax.get_ylabel() == "Overlap-weighted group effect"
    assert distribution_ax.get_xlabel() == "Honest predicted CATE"
    with pytest.raises(ValueError, match="bins"):
        result.plot_cate_distribution(bins=0)
    plt.close(calibration_ax.figure)
    plt.close(distribution_ax.figure)


def test_calibration_summary_and_hypothesis_probabilities_use_declared_reference() -> None:
    covariates, treatment, outcome = _data()
    robust = _model([]).fit(outcome, treatment=treatment, covariates=covariates)
    summary = robust.summary_frame(level=0.9)
    expected_critical = norm.ppf(0.95)
    np.testing.assert_allclose(
        summary["ci_lower"],
        robust.calibration_coefficients - expected_critical * robust.standard_errors,
    )
    assert robust.params.index.tolist() == ["level", "heterogeneity"]
    assert robust.estimand == "cate_calibration"

    n_clusters = 24
    base = np.arange(n_clusters * 4)
    index = pd.Index([f"r-{i}" for i in base])
    cluster = pd.Series(np.repeat(np.arange(n_clusters), 4), index=index)
    treatment_clustered = pd.Series(np.tile([0.0, 1.0, 0.0, 1.0], n_clusters), index=index)
    x = np.linspace(-1.0, 1.0, len(index))
    clustered_covariates = pd.DataFrame(
        {"x": x, "outcome_mean": 0.2 * x, "propensity": 0.5, "cate_score": 1.0 + x},
        index=index,
    )
    clustered_outcome = pd.Series(
        clustered_covariates["outcome_mean"]
        + (treatment_clustered - 0.5) * (1.0 + 0.5 * x)
        + 0.03 * np.sin(base),
        index=index,
    )
    clustered = _model([], covariance="clustered", calibration_groups=2, n_splits=2).fit(
        clustered_outcome,
        treatment=treatment_clustered,
        covariates=clustered_covariates,
        clusters=cluster,
    )
    test_row = clustered.calibration_tests.loc["heterogeneity=1"]
    expected_pvalue = 2.0 * t.sf(abs(test_row["statistic"]), clustered.inference_df)
    assert test_row["p_value"] == pytest.approx(expected_pvalue, abs=1e-14)


def test_public_rlearner_refuses_invalid_inference_and_group_contracts() -> None:
    with pytest.raises(ValueError, match="covariance"):
        RLearner(covariance="unadjusted")
    with pytest.raises(ValueError, match="calibration_groups"):
        RLearner(calibration_groups=1)
    with pytest.raises(ValueError, match="bootstrap_iterations"):
        RLearner(bootstrap_iterations=20)
    with pytest.raises(ValueError, match="simultaneous_level"):
        RLearner(simultaneous_level=1.0)

    covariates, treatment, outcome = _data()
    clusters = pd.Series(np.repeat(np.arange(24), 4), index=covariates.index)
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


def test_public_rlearner_refuses_constant_scores_and_group_arm_failure() -> None:
    covariates, treatment, outcome = _data()
    covariates["cate_score"] = 1.0
    with pytest.raises(ValueError, match="unique CATE"):
        _model([]).fit(outcome, treatment=treatment, covariates=covariates)

    treatment = pd.Series(np.r_[np.zeros(48), np.ones(48)], index=covariates.index)
    covariates["cate_score"] = np.linspace(-2.0, 2.0, len(covariates))
    with pytest.raises(ValueError, match="both treatment arms"):
        _model([], random_state=9).fit(
            outcome,
            treatment=treatment,
            covariates=covariates,
        )

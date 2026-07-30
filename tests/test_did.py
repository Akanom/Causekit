"""Contract-first tests for conventional and efficient difference-in-differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import (
    CrossFitter,
    DiDHausmanDiagnostic,
    DiDPretrendDiagnostic,
    DiDResult,
    DifferenceInDifferences,
    EfficientDiD,
    did_hausman_test,
)

FIT_COLUMNS = {
    "outcome": "outcome",
    "entity": "entity",
    "time": "time",
    "treatment_time": "treatment_time",
}


def _staggered_panel() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    specifications = (
        ("g2_a", 2.0, 10.0, 1.0, {2: 2.0, 3: 4.0}),
        ("g2_b", 2.0, 12.0, -1.0, {2: 2.0, 3: 4.0}),
        ("g3_a", 3.0, 20.0, 1.0, {3: 6.0}),
        ("g3_b", 3.0, 22.0, -1.0, {3: 6.0}),
        ("never_a", np.inf, 0.0, 1.0, {}),
        ("never_b", np.inf, 2.0, -1.0, {}),
    )
    for entity, cohort, level, sign, effects in specifications:
        scale = {2.0: 0.2, 3.0: 0.3, np.inf: 0.4}[cohort]
        for period in (1, 2, 3):
            rows.append(
                {
                    "entity": entity,
                    "time": period,
                    "treatment_time": cohort,
                    "outcome": level
                    + (period - 1)
                    + sign * scale * period
                    + effects.get(period, 0.0),
                }
            )
    return pd.DataFrame(rows)


def _efficient_hand_panel() -> pd.DataFrame:
    h1 = np.array([1.0, 1.0, -1.0, -1.0])
    h2 = np.array([1.0, -1.0, 1.0, -1.0])
    h3 = np.array([1.0, -1.0, -1.0, 1.0])
    rows: list[dict[str, float | str]] = []
    for position in range(4):
        path = (-(10.0 + h1[position]), -(10.0 + 2 * h2[position]), -(10.0 + 4 * h3[position]), 0.0)
        for period, outcome in enumerate(path, start=1):
            rows.append(
                {
                    "entity": f"treated_{position}",
                    "time": period,
                    "treatment_time": 4.0,
                    "outcome": outcome,
                }
            )
    for position in range(4):
        for period in range(1, 5):
            rows.append(
                {
                    "entity": f"never_{position}",
                    "time": period,
                    "treatment_time": np.inf,
                    "outcome": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _fit(model, data: pd.DataFrame, **kwargs):
    return model.fit(data, **FIT_COLUMNS, **kwargs)


class _ClassProbabilityResult:
    def __init__(self, classes: np.ndarray, probabilities: np.ndarray) -> None:
        self.classes_ = classes
        self.probabilities = probabilities

    def predict_proba(self, X):
        return np.tile(self.probabilities, (len(X), 1))


class _ClassProbability:
    def fit(self, X, y):
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        return _ClassProbabilityResult(classes, counts / counts.sum())


class _LinearResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        return design @ self.coefficients


class _LinearRegression:
    def fit(self, X, y):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        return _LinearResult(np.linalg.lstsq(design, np.asarray(y), rcond=None)[0])


class _ZeroResult:
    def predict(self, X):
        return np.zeros(len(X))


class _ZeroRegression:
    def fit(self, X, y):
        return _ZeroResult()


def _covariate_panel() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    covariate = np.linspace(-2.0, 2.0, 30)
    for cohort, prefix, effect in ((2.0, "treated", 2.0), (np.inf, "never", 0.0)):
        for position, x in enumerate(covariate):
            for period in (1, 2):
                rows.append(
                    {
                        "entity": f"{prefix}_{position}",
                        "time": period,
                        "treatment_time": cohort,
                        "x": x,
                        "outcome": 3.0 + x + (period - 1) * (0.5 * x + effect),
                    }
                )
    return pd.DataFrame(rows)


def _did_cross_fitter(*, seed: int = 37) -> CrossFitter:
    return CrossFitter(
        propensity_factory=_ClassProbability,
        outcome_factory=_LinearRegression,
        n_splits=3,
        random_state=seed,
    )


def _multi_moment_covariate_panel(repetitions: int = 12) -> pd.DataFrame:
    base = _efficient_hand_panel().assign(x=0.0)
    return pd.concat(
        [
            base.assign(entity=lambda frame, repeat=repeat: frame["entity"] + f"_{repeat}")
            for repeat in range(repetitions)
        ],
        ignore_index=True,
    )


def test_conventional_hand_computed_group_time_and_aggregations() -> None:
    result = _fit(DifferenceInDifferences(), _staggered_panel())

    assert isinstance(result, DiDResult)
    assert result.method == "conventional_group_time"
    assert result.parallel_trends == "post"
    assert result.group_time.loc[(2.0, 2), "att"] == pytest.approx(2.0, abs=1e-14)
    assert result.group_time.loc[(2.0, 3), "att"] == pytest.approx(4.0, abs=1e-14)
    assert result.group_time.loc[(3.0, 3), "att"] == pytest.approx(6.0, abs=1e-14)
    assert result.event_study.loc[0, "att"] == pytest.approx(4.0, abs=1e-14)
    assert result.event_study.loc[1, "att"] == pytest.approx(4.0, abs=1e-14)
    assert result.calendar_time.loc[2, "att"] == pytest.approx(2.0, abs=1e-14)
    assert result.calendar_time.loc[3, "att"] == pytest.approx(5.0, abs=1e-14)
    assert result.estimate == pytest.approx(4.0, abs=1e-14)
    assert result.overall_influence.mean() == pytest.approx(0.0, abs=1e-14)
    assert np.isfinite(result.standard_error)
    assert result.efficiency_weights.empty


def test_not_yet_treated_controls_are_used_only_before_their_effective_treatment() -> None:
    result = _fit(
        DifferenceInDifferences(control_group="not_yet_treated"),
        _staggered_panel(),
    )

    assert result.group_time.loc[(2.0, 2), "att"] == pytest.approx(2.0, abs=1e-14)
    assert result.group_time.loc[(2.0, 2), "n_comparison"] == 4
    assert result.group_time.loc[(2.0, 3), "n_comparison"] == 2
    assert result.group_time.loc[(3.0, 3), "n_comparison"] == 2


def test_conventional_group_time_influence_reconstructs_hc1_standard_errors() -> None:
    result = _fit(DifferenceInDifferences(), _staggered_panel())
    n = result.n_entities

    for key in result.group_time.index:
        influence = result.group_time_influence.loc[:, key].to_numpy()
        expected = np.sqrt((influence @ influence) / (n * (n - 1)))
        assert influence.mean() == pytest.approx(0.0, abs=1e-14)
        assert result.group_time.loc[key, "std_err"] == pytest.approx(expected, abs=1e-14)


def test_event_study_influence_includes_estimated_cohort_share_terms() -> None:
    result = _fit(DifferenceInDifferences(), _staggered_panel())
    entities = result.estimation_entities
    membership_g2 = entities.to_series().str.startswith("g2_").to_numpy(dtype=float)
    membership_g3 = entities.to_series().str.startswith("g3_").to_numpy(dtype=float)
    pi_g2 = pi_g3 = 2 / 6
    denominator = pi_g2 + pi_g3
    centered_total = (membership_g2 - pi_g2) + (membership_g3 - pi_g3)
    share_if_g2 = (membership_g2 - pi_g2) / denominator - pi_g2 * centered_total / denominator**2
    share_if_g3 = (membership_g3 - pi_g3) / denominator - pi_g3 * centered_total / denominator**2
    expected = (
        0.5 * result.group_time_influence.loc[:, (2.0, 2)].to_numpy()
        + 0.5 * result.group_time_influence.loc[:, (3.0, 3)].to_numpy()
        + 2.0 * share_if_g2
        + 6.0 * share_if_g3
    )

    np.testing.assert_allclose(result.event_study_influence.loc[:, 0], expected, rtol=0, atol=2e-14)


def test_efficient_hand_computed_weights_att_and_precision_gain() -> None:
    data = _efficient_hand_panel()
    efficient = _fit(EfficientDiD(pre_periods="all"), data)
    conventional = _fit(DifferenceInDifferences(), data)

    weights = efficient.efficiency_weights.sort_values("bridge_period")
    np.testing.assert_allclose(
        weights["weight"].to_numpy(),
        np.array([16 / 21, 4 / 21, 1 / 21]),
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(weights["candidate_att"], 10.0, rtol=0, atol=1e-14)
    assert weights["weight"].sum() == pytest.approx(1.0, abs=1e-14)
    assert efficient.group_time.loc[(4.0, 4), "att"] == pytest.approx(10.0, abs=1e-14)
    assert efficient.estimate == pytest.approx(10.0, abs=1e-14)
    assert efficient.standard_error == pytest.approx(np.sqrt(32 / 147), abs=2e-14)
    assert efficient.standard_error < conventional.standard_error
    assert efficient.parallel_trends == "all"
    assert efficient.method == "chen_santanna_xie_efficient"


def test_pretrend_diagnostic_recovers_hand_computed_placebos_and_joint_wald() -> None:
    result = _fit(DifferenceInDifferences(), _efficient_hand_panel())

    assert isinstance(result.pretrend, DiDPretrendDiagnostic)
    assert result.pretrend.available
    assert result.pretrend.reason is None
    assert result.pretrend.n_restrictions == 2
    assert result.pretrend.distribution == "chi2"
    assert result.pretrend.df_num == 2
    assert result.pretrend.df_denom is None
    assert result.pretrend.statistic == pytest.approx(0.0, abs=1e-14)
    assert result.pretrend.pvalue == pytest.approx(1.0, abs=1e-14)
    assert result.pretrend.placebo_effects.index.tolist() == [(4.0, 2.0), (4.0, 3.0)]
    np.testing.assert_allclose(result.pretrend.placebo_effects["att"], 0.0, atol=1e-14)
    assert result.pretrend.placebo_effects["event_time"].tolist() == [-2, -1]

    h1 = np.array([1.0, 1.0, -1.0, -1.0])
    h2 = np.array([1.0, -1.0, 1.0, -1.0])
    h3 = np.array([1.0, -1.0, -1.0, 1.0])
    expected_influence = np.column_stack(
        [
            np.r_[np.zeros(4), 2 * (h1 - 2 * h2)],
            np.r_[np.zeros(4), 2 * (2 * h2 - 4 * h3)],
        ]
    )
    np.testing.assert_allclose(result.pretrend.influence, expected_influence, atol=1e-14)
    expected_covariance = expected_influence.T @ expected_influence / (8 * 7)
    np.testing.assert_allclose(result.pretrend.covariance, expected_covariance, atol=1e-14)


def test_pretrend_excludes_the_declared_anticipation_window() -> None:
    result = _fit(DifferenceInDifferences(anticipation=1), _efficient_hand_panel())

    assert result.pretrend.placebo_effects.index.tolist() == [(4.0, 2.0)]
    assert result.pretrend.placebo_effects["event_time"].tolist() == [-2]


def test_clustered_pretrend_covariance_uses_one_score_sum_per_cluster() -> None:
    data = _efficient_hand_panel()
    data["cluster"] = data["entity"].str.rsplit("_", n=1).str[-1]
    result = _fit(
        DifferenceInDifferences(covariance="clustered"),
        data,
        cluster="cluster",
    )

    scores = result.pretrend.influence.to_numpy()
    cluster_labels = result.inference_clusters
    cluster_sums = np.vstack(
        [
            scores[cluster_labels.to_numpy() == label].sum(axis=0)
            for label in pd.unique(cluster_labels)
        ]
    )
    expected = 4 / 3 * cluster_sums.T @ cluster_sums / result.n_entities**2
    np.testing.assert_allclose(result.pretrend.covariance, expected, atol=1e-14)
    assert result.pretrend.distribution == "f"
    assert result.pretrend.df_num == 2
    assert result.pretrend.df_denom == 3


def test_pretrend_is_explicitly_unavailable_without_clean_placebo_periods() -> None:
    result = _fit(DifferenceInDifferences(), _covariate_panel().drop(columns="x"))

    assert not result.pretrend.available
    assert result.pretrend.n_restrictions == 0
    assert result.pretrend.placebo_effects.empty
    assert "uncontaminated" in str(result.pretrend.reason)


def test_pretrend_retains_placebos_but_refuses_a_singular_joint_test() -> None:
    data = _efficient_hand_panel()
    patterns = {
        f"treated_{position}": value for position, value in enumerate((1.0, -1.0, 1.0, -1.0))
    }
    for entity, value in patterns.items():
        mask = data["entity"].eq(entity)
        data.loc[mask, "outcome"] = np.array([0.0, value, 2 * value, 3.0 + 2 * value])
    result = _fit(DifferenceInDifferences(), data)

    assert not result.pretrend.available
    assert result.pretrend.n_restrictions == 2
    assert len(result.pretrend.placebo_effects) == 2
    assert "no ridge or pseudoinverse" in str(result.pretrend.reason)


def test_covariate_efficient_pretrend_refuses_to_mislabel_an_unadjusted_test() -> None:
    result = _fit(
        EfficientDiD(),
        _covariate_panel(),
        covariates=["x"],
        cross_fitter=_did_cross_fitter(),
    )

    assert not result.pretrend.available
    assert result.pretrend.placebo_effects.empty
    assert "conditional" in str(result.pretrend.reason)


def test_hausman_diagnostic_uses_common_event_path_and_difference_influence() -> None:
    data = _efficient_hand_panel()
    conventional = _fit(DifferenceInDifferences(), data)
    efficient = _fit(EfficientDiD(pre_periods="all"), data)

    diagnostic = did_hausman_test(conventional, efficient)

    assert isinstance(diagnostic, DiDHausmanDiagnostic)
    assert diagnostic.n_restrictions == 1
    assert diagnostic.distribution == "chi2"
    assert diagnostic.df_num == 1
    assert diagnostic.df_denom is None
    assert diagnostic.statistic == pytest.approx(0.0, abs=1e-14)
    assert diagnostic.pvalue == pytest.approx(1.0, abs=1e-14)
    assert not diagnostic.reject
    assert diagnostic.recommendation == "pt_all_not_rejected"
    assert diagnostic.event_study.index.tolist() == [0]
    assert diagnostic.event_study.loc[0, "pt_post"] == pytest.approx(10.0)
    assert diagnostic.event_study.loc[0, "pt_all"] == pytest.approx(10.0)

    expected_influence = (
        efficient.event_study_influence.loc[:, [0]].to_numpy()
        - conventional.event_study_influence.loc[:, [0]].to_numpy()
    )
    np.testing.assert_allclose(diagnostic.influence, expected_influence, atol=1e-14)
    expected_covariance = expected_influence.T @ expected_influence / (8 * 7)
    np.testing.assert_allclose(diagnostic.covariance, expected_covariance, atol=1e-14)


def test_hausman_diagnostic_refuses_incompatible_or_unsupported_results() -> None:
    data = _efficient_hand_panel()
    conventional = _fit(DifferenceInDifferences(), data)
    efficient = _fit(EfficientDiD(), data)

    with pytest.raises(ValueError, match="first argument"):
        did_hausman_test(efficient, conventional)
    with pytest.raises(ValueError, match="never-treated"):
        did_hausman_test(
            _fit(DifferenceInDifferences(control_group="not_yet_treated"), data),
            efficient,
        )
    altered = data.copy()
    altered.loc[altered["entity"] == "treated_0", "outcome"] += 0.25
    with pytest.raises(ValueError, match="same estimation sample"):
        did_hausman_test(conventional, _fit(EfficientDiD(), altered))


def test_hausman_diagnostic_refuses_singular_difference_covariance() -> None:
    data = _covariate_panel().drop(columns="x")
    conventional = _fit(DifferenceInDifferences(), data)
    efficient = _fit(EfficientDiD(), data)

    with pytest.raises(ValueError, match="no ridge or pseudoinverse"):
        did_hausman_test(conventional, efficient)


def test_efficient_result_exposes_candidate_and_aggregate_influence_identities() -> None:
    result = _fit(EfficientDiD(), _efficient_hand_panel())
    weights = result.efficiency_weights.sort_values("bridge_period")["weight"].to_numpy()
    candidate = result.candidate_influence_functions.to_numpy().T
    reconstructed = weights @ candidate

    np.testing.assert_allclose(
        reconstructed,
        result.group_time_influence.loc[:, (4.0, 4)].to_numpy(),
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        result.overall_influence,
        result.event_study_influence.loc[:, 0],
        rtol=0,
        atol=2e-14,
    )


def test_covariate_adjusted_efficient_path_uses_cross_fitted_nuisances() -> None:
    result = _fit(
        EfficientDiD(),
        _covariate_panel(),
        covariates=["x"],
        cross_fitter=_did_cross_fitter(),
    )

    assert result.estimate == pytest.approx(2.0, abs=2e-12)
    assert result.group_time.loc[(2.0, 2.0), "att"] == pytest.approx(2.0, abs=2e-12)
    assert result.cross_fitted
    assert result.covariates == ("x",)
    assert set(result.nuisance_fold.unique()) == {0, 1, 2}
    assert result.nuisance_fold.index.equals(result.estimation_entities)
    assert result.cohort_probabilities.index.equals(result.estimation_entities)
    np.testing.assert_allclose(result.cohort_probabilities.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(result.conditional_efficiency_weights.sum(axis=1), 1.0, atol=1e-14)


def test_covariate_path_is_deterministic_for_a_fixed_crossfit_seed() -> None:
    data = _covariate_panel()
    first = _fit(EfficientDiD(), data, covariates=["x"], cross_fitter=_did_cross_fitter(seed=8))
    second = _fit(EfficientDiD(), data, covariates=["x"], cross_fitter=_did_cross_fitter(seed=8))

    pd.testing.assert_series_equal(first.nuisance_fold, second.nuisance_fold)
    pd.testing.assert_frame_equal(first.cohort_probabilities, second.cohort_probabilities)
    pd.testing.assert_frame_equal(
        first.conditional_efficiency_weights, second.conditional_efficiency_weights
    )
    pd.testing.assert_frame_equal(first.group_time, second.group_time)


def test_covariate_path_estimates_and_inverts_conditional_multi_moment_covariance() -> None:
    result = _fit(
        EfficientDiD(),
        _multi_moment_covariate_panel(),
        covariates=["x"],
        cross_fitter=_did_cross_fitter(seed=7),
    )

    assert result.estimate == pytest.approx(10.0, abs=0.03)
    assert len(result.efficiency_weights) == 3
    assert result.group_time.loc[(4.0, 4.0), "weight_condition_number"] > 1.0
    assert result.efficiency_weights["weight_std"].gt(0.0).all()
    np.testing.assert_allclose(
        result.conditional_efficiency_weights.sum(axis=1), 1.0, rtol=0, atol=2e-14
    )
    assert np.isfinite(result.candidate_influence_functions).all().all()


def test_anticipation_shifts_effective_boundary_without_relabeling_adoption_cohort() -> None:
    data = _staggered_panel()
    data.loc[np.isfinite(data["treatment_time"]), "treatment_time"] = 3.0
    result = _fit(DifferenceInDifferences(anticipation=1), data)
    first = result.group_time.reset_index().iloc[0]

    assert first["cohort"] == 3.0
    assert first["time"] == 2
    assert first["event_time"] == -1
    assert first["base_period"] == 1


def test_row_permutation_does_not_change_results() -> None:
    data = _staggered_panel()
    original = _fit(DifferenceInDifferences(control_group="not_yet_treated"), data)
    permuted = _fit(
        DifferenceInDifferences(control_group="not_yet_treated"),
        data.sample(frac=1.0, random_state=20260728),
    )

    pd.testing.assert_frame_equal(original.group_time, permuted.group_time)
    pd.testing.assert_frame_equal(original.event_study, permuted.event_study)
    pd.testing.assert_frame_equal(
        original.pretrend.placebo_effects,
        permuted.pretrend.placebo_effects,
    )
    assert original.design_fingerprint == permuted.design_fingerprint


def test_clustered_overall_variance_uses_cluster_summed_entity_scores() -> None:
    data = _staggered_panel()
    cluster_by_entity = {
        "g2_a": "a",
        "g2_b": "a",
        "g3_a": "b",
        "g3_b": "b",
        "never_a": "c",
        "never_b": "c",
    }
    data["cluster"] = data["entity"].map(cluster_by_entity)
    result = _fit(
        DifferenceInDifferences(covariance="clustered"),
        data,
        cluster="cluster",
    )
    scores = result.overall_influence.groupby(pd.Series(cluster_by_entity, name="cluster")).sum()
    expected_variance = 3 / 2 * float(scores @ scores) / result.n_entities**2

    assert result.n_clusters == 3
    assert result.inference_distribution == "t"
    assert result.inference_df == 2
    assert result.standard_error**2 == pytest.approx(expected_variance, abs=1e-14)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: pd.concat([data, data.iloc[[0]]], ignore_index=True), "entity-time"),
        (
            lambda data: data.drop(data.index[(data["entity"] == "g2_a") & (data["time"] == 1)]),
            "balanced panel",
        ),
        (
            lambda data: data.assign(outcome=lambda frame: frame["outcome"].mask(frame.index == 0)),
            "finite",
        ),
        (
            lambda data: data.assign(
                treatment_time=lambda frame: frame["treatment_time"].mask(
                    (frame["entity"] == "g2_a") & (frame["time"] == 1), 3.0
                )
            ),
            "constant within entity",
        ),
        (lambda data: data.assign(treatment_time=2.0), "never-treated"),
        (
            lambda data: data.assign(
                treatment_time=lambda frame: frame["treatment_time"].mask(
                    frame["entity"].eq("g2_a"), 99.0
                )
            ),
            "observed time",
        ),
    ],
)
def test_panel_contract_refusals(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _fit(DifferenceInDifferences(), mutation(_staggered_panel()))


def test_treatment_at_first_effective_period_is_refused() -> None:
    data = _staggered_panel()
    data.loc[data["treatment_time"] == 2.0, "treatment_time"] = 1.0
    with pytest.raises(ValueError, match="baseline"):
        _fit(DifferenceInDifferences(), data)


def test_covariate_path_requires_cross_fitter_and_constant_entity_covariates() -> None:
    data = _staggered_panel().assign(x=1.0)
    with pytest.raises(ValueError, match="cross_fitter"):
        _fit(EfficientDiD(), data, covariates=["x"])
    data.loc[(data["entity"] == "g2_a") & (data["time"] == 1), "x"] = 2.0
    with pytest.raises(ValueError, match="constant within entity"):
        _fit(
            EfficientDiD(),
            data,
            covariates=["x"],
            cross_fitter=_did_cross_fitter(),
        )


def test_multiplier_event_study_band_matches_hand_computed_rademacher_max_t() -> None:
    iterations = 199
    seed = 20260728
    level = 0.90
    result = _fit(
        DifferenceInDifferences(
            inference="multiplier_bootstrap",
            bootstrap_iterations=iterations,
            random_state=seed,
            simultaneous_level=level,
        ),
        _staggered_panel(),
    )
    scores = result.event_study_influence.to_numpy(dtype=float)
    standard_errors = result.event_study["std_err"].to_numpy(dtype=float)
    multipliers = np.random.default_rng(seed).choice(
        np.array([-1.0, 1.0]), size=(iterations, result.n_entities)
    )
    draws = np.sqrt(result.n_entities / (result.n_entities - 1)) * multipliers @ scores
    draws /= result.n_entities
    max_statistics = np.max(np.abs(draws / standard_errors), axis=1)
    expected_critical = np.quantile(max_statistics, level, method="higher")

    assert result.inference_method == "multiplier_bootstrap"
    assert result.simultaneous_critical_value == pytest.approx(expected_critical, abs=1e-14)
    assert result.bootstrap_iterations == iterations
    assert result.bootstrap_random_state == seed
    assert result.simultaneous_level == level
    np.testing.assert_allclose(
        result.simultaneous_event_study["lower"],
        result.event_study["att"] - expected_critical * result.event_study["std_err"],
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.simultaneous_event_study["upper"],
        result.event_study["att"] + expected_critical * result.event_study["std_err"],
        rtol=0,
        atol=1e-14,
    )


def test_multiplier_band_is_reproducible_and_analytic_path_has_no_band() -> None:
    model = EfficientDiD(
        inference="multiplier_bootstrap", bootstrap_iterations=101, random_state=11
    )
    first = _fit(model, _efficient_hand_panel())
    second = _fit(model, _efficient_hand_panel())
    pd.testing.assert_frame_equal(first.simultaneous_event_study, second.simultaneous_event_study)
    assert (
        DifferenceInDifferences()
        .fit(_staggered_panel(), **FIT_COLUMNS)
        .simultaneous_event_study.empty
    )


def test_cluster_multiplier_band_uses_cluster_summed_scores() -> None:
    data = _staggered_panel()
    data["cluster"] = data["entity"]
    iterations = 101
    seed = 18
    result = _fit(
        DifferenceInDifferences(
            covariance="clustered",
            inference="multiplier_bootstrap",
            bootstrap_iterations=iterations,
            random_state=seed,
        ),
        data,
        cluster="cluster",
    )
    scores = result.event_study_influence.to_numpy(dtype=float)
    multipliers = np.random.default_rng(seed).choice(
        np.array([-1.0, 1.0]), size=(iterations, result.n_clusters)
    )
    draws = np.sqrt(result.n_clusters / (result.n_clusters - 1)) * multipliers @ scores
    draws /= result.n_entities
    expected = np.quantile(
        np.max(np.abs(draws / result.event_study["std_err"].to_numpy()), axis=1),
        0.95,
        method="higher",
    )

    assert result.simultaneous_critical_value == pytest.approx(expected, abs=1e-14)


@pytest.mark.simulation
def test_covariate_adjusted_multiplier_band_has_seeded_coverage_smoke() -> None:
    covered = 0
    estimates: list[float] = []
    repetitions = 20
    for seed in range(repetitions):
        rng = np.random.default_rng(seed)
        rows: list[dict[str, float | str]] = []
        for cohort, prefix, effect in ((2.0, "treated", 2.0), (np.inf, "never", 0.0)):
            x = rng.normal(size=80)
            baseline_error = rng.normal(size=80)
            change_error = rng.normal(size=80)
            for position in range(80):
                baseline = 3.0 + x[position] + baseline_error[position]
                for period, value in (
                    (1, baseline),
                    (
                        2,
                        baseline + 0.5 * x[position] + effect + change_error[position],
                    ),
                ):
                    rows.append(
                        {
                            "entity": f"{prefix}_{position}",
                            "time": period,
                            "treatment_time": cohort,
                            "x": x[position],
                            "outcome": value,
                        }
                    )
        result = _fit(
            EfficientDiD(
                inference="multiplier_bootstrap",
                bootstrap_iterations=99,
                random_state=seed,
            ),
            pd.DataFrame(rows),
            covariates=["x"],
            cross_fitter=_did_cross_fitter(seed=seed),
        )
        band = result.simultaneous_event_study.iloc[0]
        covered += int(band["lower"] <= 2.0 <= band["upper"])
        estimates.append(result.estimate)

    assert np.mean(estimates) == pytest.approx(2.0, abs=0.12)
    assert covered >= 16


def test_invalid_multiplier_band_configuration_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 99"):
        DifferenceInDifferences(inference="multiplier_bootstrap", bootstrap_iterations=20)
    with pytest.raises(ValueError, match="strictly between"):
        DifferenceInDifferences(
            inference="multiplier_bootstrap", bootstrap_iterations=99, simultaneous_level=1.0
        )


def test_efficient_singular_weight_system_is_refused_without_regularization() -> None:
    data = _efficient_hand_panel()
    data.loc[data["treatment_time"] == 4.0, "outcome"] = np.tile([-10.0, -10.0, -10.0, 0.0], 4)
    with pytest.raises(ValueError, match="singular"):
        _fit(EfficientDiD(), data)


def test_singular_cross_fitted_conditional_covariance_is_refused_without_repair() -> None:
    fitter = CrossFitter(
        propensity_factory=_ClassProbability,
        outcome_factory=_LinearRegression,
        second_moment_factory=_ZeroRegression,
        n_splits=3,
        random_state=7,
    )
    with pytest.raises(ValueError, match="conditional covariance system is singular"):
        _fit(
            EfficientDiD(),
            _multi_moment_covariate_panel(),
            covariates=["x"],
            cross_fitter=fitter,
        )


def test_invalid_constructor_and_cluster_contracts_are_refused() -> None:
    with pytest.raises(ValueError, match="control_group"):
        DifferenceInDifferences(control_group="already_treated")
    with pytest.raises(ValueError, match="anticipation"):
        DifferenceInDifferences(anticipation=-1)
    with pytest.raises(ValueError, match="pre_periods"):
        EfficientDiD(pre_periods=0)

    data = _staggered_panel().assign(cluster="one")
    with pytest.raises(ValueError, match="at least two clusters"):
        _fit(
            DifferenceInDifferences(covariance="clustered"),
            data,
            cluster="cluster",
        )


def test_cluster_must_be_constant_within_entity() -> None:
    data = _staggered_panel().assign(cluster="a")
    data.loc[(data["entity"] == "g2_a") & (data["time"] == 1), "cluster"] = "b"
    with pytest.raises(ValueError, match="constant within entity"):
        _fit(
            DifferenceInDifferences(covariance="clustered"),
            data,
            cluster="cluster",
        )


def test_result_confidence_interval_validates_level() -> None:
    result = _fit(DifferenceInDifferences(), _staggered_panel())
    interval = result.conf_int()
    assert interval["lower"] < result.estimate < interval["upper"]
    with pytest.raises(ValueError, match="strictly between"):
        result.conf_int(level=1.0)

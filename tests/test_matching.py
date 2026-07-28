"""Contract-first tests for nearest-neighbor matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalkit import (
    FittedPropensityMLEProtocol,
    NearestNeighborMatch,
    NearestNeighborMatchResult,
)


def _three_unit_example():
    index = pd.Index(["control", "treated_near", "treated_far"])
    return (
        pd.Series([1.0, 4.0, 10.0], index=index, name="outcome"),
        pd.Series([0, 1, 1], index=index, name="treatment"),
        pd.Series([0.20, 0.25, 0.75], index=index, name="propensity"),
        pd.DataFrame({"baseline": [0.0, 0.2, 1.0]}, index=index),
    )


def _six_unit_variance_example():
    """No-tie scalar design with hand-computable Abadie-Imbens variance."""

    index = pd.Index(["c0", "c1", "c2", "t0", "t1", "t2"])
    logits = np.array([0.0, 4.0, 10.0, 1.0, 6.0, 9.0])
    return (
        pd.Series([0.0, 2.0, 5.0, 3.0, 8.0, 12.0], index=index, name="outcome"),
        pd.Series([0, 0, 0, 1, 1, 1], index=index, name="treatment"),
        pd.Series(1 / (1 + np.exp(-logits)), index=index, name="propensity"),
    )


class _HandFittedLogit:
    """Minimal fitted-result protocol for a hand-audited finite logit MLE."""

    def __init__(self, params: pd.Series, nobs: int) -> None:
        self.params = params
        self.nobs = nobs
        self.converged = True
        self.feature_names = tuple(params.index)

    def predict_proba(self, X):
        design = np.asarray(X, dtype=float)
        probability = 1 / (1 + np.exp(-(design @ self.params.to_numpy())))
        return pd.DataFrame({0: 1 - probability, 1: probability}, index=X.index)


def _estimated_score_example():
    """Irregular no-tie design with a hand-recorded full-sample logit MLE."""

    index = pd.Index([f"unit_{position}" for position in range(12)])
    x = np.array([-2.60, -2.00, -1.35, -0.95, -0.62, -0.18, 0.13, 0.49, 0.91, 1.38, 1.92, 2.57])
    design = pd.DataFrame({"const": 1.0, "x": x}, index=index)
    treatment = pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], index=index)
    outcome = pd.Series(
        [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 11.0, 10.0, 14.0],
        index=index,
    )
    model = _HandFittedLogit(
        pd.Series(
            [0.01321428027973149, 0.5124700999154841],
            index=["const", "x"],
            name="estimate",
        ),
        len(index),
    )
    return outcome, treatment, design, model


@pytest.mark.parametrize(
    ("estimand", "expected"),
    [("att", 6.0), ("atc", 3.0), ("ate", 5.0)],
)
def test_hand_computed_att_atc_and_bidirectional_ate(estimand: str, expected: float) -> None:
    y, treatment, propensity, covariates = _three_unit_example()
    result = NearestNeighborMatch(
        estimand=estimand,
        caliper=None,
        common_support=None,
        inference="none",
    ).fit(y, treatment=treatment, propensity=propensity, covariates=covariates)

    assert isinstance(result, NearestNeighborMatchResult)
    assert result.estimate == pytest.approx(expected, abs=1e-14)
    assert result.requested_estimand == estimand
    assert result.realized_estimand == estimand
    assert np.isnan(result.standard_error)
    assert np.isnan(result.statistic)
    assert np.isnan(result.pvalue)
    assert (
        result.match_table["match_weight"]
        .groupby(result.match_table["focal_index"])
        .sum()
        .eq(1)
        .all()
    )
    assert "arbitrarily distant" in " ".join(result.notes)


def test_all_boundary_ties_receive_fractional_weight_and_are_order_invariant() -> None:
    logits = np.array([-1.0, 1.0, 0.0])
    propensity = pd.Series(1 / (1 + np.exp(-logits)), index=["c0", "c1", "t0"])
    y = pd.Series([0.0, 2.0, 10.0], index=propensity.index)
    treatment = pd.Series([0, 0, 1], index=propensity.index)
    model = NearestNeighborMatch(caliper=None, common_support=None, inference="none")

    result = model.fit(y, treatment=treatment, propensity=propensity)
    permutation = ["t0", "c1", "c0"]
    permuted = model.fit(
        y.loc[permutation],
        treatment=treatment.loc[permutation],
        propensity=propensity.loc[permutation],
    )

    assert result.estimate == pytest.approx(9.0, abs=1e-14)
    assert sorted(result.match_table["match_weight"]) == [0.5, 0.5]
    assert result.boundary_tie_events == 1
    assert result.maximum_tie_multiplicity == 2
    assert result.realized_match_count_distribution.to_dict() == {2: 1}
    assert permuted.estimate == pytest.approx(result.estimate, abs=1e-14)
    assert set(permuted.match_table["comparison_index"]) == {"c0", "c1"}
    assert result.effect_weights @ y == pytest.approx(result.estimate, abs=1e-14)
    assert result.effect_weights.loc["t0"] == pytest.approx(1.0, abs=1e-14)
    assert result.effect_weights.loc[["c0", "c1"]].sum() == pytest.approx(-1.0, abs=1e-14)


def test_multiple_neighbors_are_equally_weighted_without_false_tie_metadata() -> None:
    logits = np.array([-2.0, 1.0, 0.0])
    propensity = 1 / (1 + np.exp(-logits))
    result = NearestNeighborMatch(
        neighbors=2,
        caliper=None,
        common_support=None,
    ).fit([0.0, 2.0, 10.0], treatment=[0, 0, 1], propensity=propensity)

    assert result.estimate == pytest.approx(9.0, abs=1e-14)
    assert result.match_table["match_weight"].tolist() == [0.5, 0.5]
    assert result.boundary_tie_events == 0
    assert result.maximum_tie_multiplicity == 1
    assert result.realized_match_count_distribution.to_dict() == {2: 1}


def test_common_support_exclusions_change_realized_target_label() -> None:
    index = pd.Index(["c0", "c1", "t0", "t1"])
    y = pd.Series([1.0, 2.0, 4.0, 9.0], index=index)
    treatment = pd.Series([0, 0, 1, 1], index=index)
    propensity = pd.Series([0.20, 0.40, 0.30, 0.80], index=index)

    result = NearestNeighborMatch(inference="none", caliper=None).fit(
        y, treatment=treatment, propensity=propensity
    )

    assert result.realized_estimand == "att_matched_support"
    assert result.support_excluded_index.tolist() == ["c0", "t1"]
    assert result.n_matched_focal == 1
    assert result.n_support_excluded_treated == 1
    assert result.n_support_excluded_control == 1


def test_excluding_only_comparison_units_does_not_relabel_full_att() -> None:
    index = pd.Index(["c_low", "c_mid", "c_high", "t0", "t1"])
    y = pd.Series([0.0, 1.0, 2.0, 4.0, 5.0], index=index)
    treatment = pd.Series([0, 0, 0, 1, 1], index=index)
    propensity = pd.Series([0.10, 0.35, 0.60, 0.30, 0.40], index=index)

    result = NearestNeighborMatch(caliper=None, inference="none").fit(
        y, treatment=treatment, propensity=propensity
    )

    assert result.realized_estimand == "att"
    assert result.support_excluded_index.tolist() == ["c_low", "c_high"]
    assert result.matched_focal_index.tolist() == ["t0", "t1"]


def test_caliper_is_inclusive_and_strictly_records_unmatched_focal_units() -> None:
    index = pd.Index(["c0", "t0", "t1"])
    y = pd.Series([0.0, 2.0, 5.0], index=index)
    treatment = pd.Series([0, 1, 1], index=index)
    propensity = pd.Series([0.5, 1 / (1 + np.exp(-0.1)), 0.9], index=index)
    result = NearestNeighborMatch(
        caliper=0.1,
        common_support=None,
        inference="none",
    ).fit(y, treatment=treatment, propensity=propensity)
    assert result.matched_focal_index.tolist() == ["t0"]
    assert result.caliper_unmatched_index.tolist() == ["t1"]
    assert result.realized_estimand == "att_matched_support"
    assert result.n_caliper_unmatched_treated == 1
    assert result.matched_focal_fraction == pytest.approx(0.5, abs=1e-14)


def test_automatic_caliper_uses_full_sample_logit_standard_deviation() -> None:
    logits = np.array([-1.0, 1.0, -0.9, 0.9])
    propensity = 1 / (1 + np.exp(-logits))
    result = NearestNeighborMatch(common_support=None).fit(
        [0.0, 1.0, 2.0, 3.0], treatment=[0, 0, 1, 1], propensity=propensity
    )

    assert result.realized_caliper == pytest.approx(0.2 * np.std(logits, ddof=1), abs=1e-14)
    assert result.caliper_scale == "propensity_logit"
    assert result.caliper_standard_deviation_ddof == 1


def test_balance_diagnostics_use_the_effect_design_weights_without_pvalues() -> None:
    logits = np.array([-2.0, 2.0, -1.9, -1.8])
    propensity = 1 / (1 + np.exp(-logits))
    covariates = pd.DataFrame({"baseline": [0.0, 10.0, 0.0, 0.0]})
    result = NearestNeighborMatch(caliper=None, common_support=None).fit(
        [0.0, 10.0, 2.0, 3.0],
        treatment=[0, 0, 1, 1],
        propensity=propensity,
        covariates=covariates,
        propensity_provenance="cross_fitted:test-model",
    )

    diagnostic = result.balance.loc["baseline"]
    assert diagnostic["treated_mean_before"] == pytest.approx(0.0, abs=1e-14)
    assert diagnostic["control_mean_before"] == pytest.approx(5.0, abs=1e-14)
    assert diagnostic["smd_before"] == pytest.approx(-1.0, abs=1e-14)
    assert diagnostic["treated_mean_after"] == pytest.approx(0.0, abs=1e-14)
    assert diagnostic["control_mean_after"] == pytest.approx(0.0, abs=1e-14)
    assert diagnostic["smd_after"] == pytest.approx(0.0, abs=1e-14)
    assert diagnostic["ecdf_max_after"] == pytest.approx(0.0, abs=1e-14)
    assert not any("p" in column.lower() for column in result.balance.columns)
    assert result.balance_summary["max_abs_smd_before"] == pytest.approx(1.0, abs=1e-14)
    assert result.balance_summary["max_abs_smd_after"] == pytest.approx(0.0, abs=1e-14)
    assert result.propensity_provenance == "cross_fitted:test-model"
    with pytest.raises(ValueError, match="inference='none'"):
        result.conf_int()


@pytest.mark.simulation
@pytest.mark.parametrize(("estimand", "expected"), [("att", 1.0), ("atc", 1.0), ("ate", 1.0)])
def test_exact_paired_design_recovers_heterogeneous_effect_average(
    estimand: str, expected: float
) -> None:
    score = np.linspace(-1.0, 1.0, 101)
    propensity_by_pair = 1 / (1 + np.exp(-score))
    baseline = 0.4 + 2.0 * score
    effect = 1.0 + 0.5 * score
    propensity = np.concatenate([propensity_by_pair, propensity_by_pair])
    treatment = np.concatenate([np.zeros(len(score)), np.ones(len(score))])
    outcome = np.concatenate([baseline, baseline + effect])

    result = NearestNeighborMatch(estimand=estimand).fit(
        outcome,
        treatment=treatment,
        propensity=propensity,
        propensity_provenance="known_simulation_score",
    )

    assert result.estimate == pytest.approx(expected, abs=1e-13)
    assert result.realized_estimand == estimand
    assert result.matched_focal_fraction == pytest.approx(1.0, abs=1e-14)


def test_outcomes_do_not_change_the_selected_design() -> None:
    logits = np.array([-0.8, 0.4, -0.7, 0.5])
    propensity = 1 / (1 + np.exp(-logits))
    treatment = np.array([0, 0, 1, 1])
    model = NearestNeighborMatch(caliper=None, common_support=None)

    first = model.fit([0.0, 1.0, 2.0, 3.0], treatment=treatment, propensity=propensity)
    second = model.fit([900.0, -50.0, -3.0, 700.0], treatment=treatment, propensity=propensity)

    pd.testing.assert_frame_equal(first.match_table, second.match_table)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"replacement": False}, NotImplementedError, "replacement"),
        ({"metric": "mahalanobis"}, NotImplementedError, "metric"),
        ({"ties": "first"}, ValueError, "ties"),
        ({"bias_correction": "linear"}, NotImplementedError, "bias_correction"),
        ({"inference": "bootstrap"}, ValueError, "inference"),
        ({"neighbors": 0}, ValueError, "neighbors"),
        ({"variance_neighbors": 0}, ValueError, "variance_neighbors"),
        (
            {"first_step_covariance_neighbors": 1},
            ValueError,
            "first_step_covariance_neighbors",
        ),
        (
            {"first_step_regression_neighbors": 0},
            ValueError,
            "first_step_regression_neighbors",
        ),
        (
            {"first_step_covariate_neighbors": False},
            ValueError,
            "first_step_covariate_neighbors",
        ),
        ({"caliper": 0.0}, ValueError, "caliper"),
        ({"caliper": []}, ValueError, "caliper"),
        ({"common_support": "trim"}, ValueError, "common_support"),
        ({"estimand": "ATC"}, ValueError, "estimand"),
    ],
)
def test_constructor_refuses_unsupported_or_invalid_contracts(kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        NearestNeighborMatch(**kwargs)


def test_analytical_inference_refuses_expanded_boundary_ties() -> None:
    logits = np.array([-1.0, 1.0, 0.0])
    propensity = 1 / (1 + np.exp(-logits))
    with pytest.raises(ValueError, match="boundary ties"):
        NearestNeighborMatch(caliper=None, common_support=None, inference="abadie_imbens").fit(
            [0.0, 2.0, 10.0], treatment=[0, 0, 1], propensity=propensity
        )


def test_analytical_inference_is_not_approximated_without_propensity_provenance() -> None:
    y, treatment, propensity, _ = _three_unit_example()
    with pytest.raises(NotImplementedError, match="known/fixed"):
        NearestNeighborMatch(
            caliper=None,
            common_support=None,
            inference="abadie_imbens",
        ).fit(
            y,
            treatment=treatment,
            propensity=propensity,
            propensity_provenance="known is not a machine-readable declaration",
        )


@pytest.mark.parametrize(
    ("estimand", "normalized_variance", "variance_denominator", "conditional_component"),
    [
        ("att", 26.0 / 9.0, 3, 37.0 / 3.0),
        ("atc", 26.0 / 9.0, 3, 37.0 / 3.0),
        ("ate", 137.0 / 9.0, 6, 74.0 / 3.0),
    ],
)
def test_hand_computed_known_score_abadie_imbens_variance(
    estimand: str,
    normalized_variance: float,
    variance_denominator: int,
    conditional_component: float,
) -> None:
    y, treatment, propensity = _six_unit_variance_example()

    result = NearestNeighborMatch(
        estimand=estimand,
        caliper=None,
        common_support=None,
        inference="abadie_imbens",
        variance_neighbors=1,
    ).fit(
        y,
        treatment=treatment,
        propensity=propensity,
        propensity_score_status="known",
        propensity_provenance="fixed_by_simulation_design",
    )

    expected_conditional_variances = pd.Series(
        [2.0, 2.0, 4.5, 12.5, 8.0, 8.0],
        index=y.index,
        name="conditional_variance",
    )
    expected_variance = normalized_variance / variance_denominator
    pd.testing.assert_series_equal(result.conditional_variances, expected_conditional_variances)
    assert result.estimate == pytest.approx(16.0 / 3.0, abs=1e-14)
    assert result.normalized_variance == pytest.approx(normalized_variance, abs=1e-14)
    assert result.variance == pytest.approx(expected_variance, abs=1e-14)
    assert result.known_score_variance == pytest.approx(expected_variance, abs=1e-14)
    assert result.known_score_normalized_variance == pytest.approx(normalized_variance, abs=1e-14)
    assert np.isnan(result.first_step_variance_adjustment)
    assert result.standard_error == pytest.approx(np.sqrt(expected_variance), abs=1e-14)
    assert result.conditional_variance_component == pytest.approx(conditional_component, abs=1e-14)
    assert result.effect_variance_component == pytest.approx(-85.0 / 9.0, abs=1e-14)
    assert result.statistic == pytest.approx(result.estimate / result.standard_error, abs=1e-14)
    assert 0.0 < result.pvalue < 1.0
    assert result.propensity_score_status == "known"
    assert result.variance_neighbors == 1
    assert result.inference_distribution == "normal"
    assert result.inference_df is None
    interval = result.conf_int()
    assert interval.name == estimand
    assert interval["lower"] < result.estimate < interval["upper"]
    assert result.summary_frame().columns.tolist() == [
        "coef",
        "std_err",
        "stat",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]
    rendered = result.to_markdown()
    assert "Abadie-Imbens" in rendered
    assert "known" in rendered
    assert result.realized_estimand.upper() in rendered
    assert "fixed_by_simulation_design" in rendered


@pytest.mark.parametrize(
    ("estimand", "expected_variance"),
    [("att", 26.0 / 27.0), ("atc", 26.0 / 27.0), ("ate", 133.0 / 27.0)],
)
def test_hand_computed_stata_aligned_two_neighbor_variance(
    estimand: str, expected_variance: float
) -> None:
    y, treatment, propensity = _six_unit_variance_example()

    result = NearestNeighborMatch(
        estimand=estimand,
        caliper=None,
        common_support=None,
        inference="abadie_imbens",
        variance_neighbors=2,
    ).fit(
        y,
        treatment=treatment,
        propensity=propensity,
        propensity_score_status="known",
        propensity_provenance="fixed_by_stata_parity_fixture",
    )

    assert result.estimate == pytest.approx(16.0 / 3.0, abs=1e-14)
    assert result.variance == pytest.approx(expected_variance, abs=1e-14)
    assert result.standard_error == pytest.approx(np.sqrt(expected_variance), abs=1e-14)
    assert result.variance_neighbors == 2


@pytest.mark.parametrize(
    (
        "estimand",
        "expected_estimate",
        "expected_known_variance",
        "expected_adjustment",
        "expected_variance",
        "expected_standard_error",
        "expected_adjustment_vector",
        "expected_target_derivative",
    ),
    [
        (
            "ate",
            2.3333333333333335,
            1.0509259259259258,
            -0.2958653270661526,
            0.7550605988597732,
            0.8689422298747904,
            [0.0, 1.2312612242496257],
            [0.0, 0.0],
        ),
        (
            "att",
            2.5,
            1.1435185185185184,
            -0.4568110092302016,
            0.6867075092883168,
            0.8286781699117679,
            [-0.19284514030466185, 1.5160907870638312],
            [-0.08123656658228982, -0.036881407085510076],
        ),
        (
            "atc",
            2.1666666666666665,
            0.46759259259259256,
            -0.17569523227580644,
            0.29189736031678615,
            0.5402752634692672,
            [0.04675893228574632, 0.9504173105133439],
            [-0.0648496414366257, 0.04086705616343389],
        ),
    ],
)
def test_hand_computed_estimated_logit_first_step_adjustment(
    estimand: str,
    expected_estimate: float,
    expected_known_variance: float,
    expected_adjustment: float,
    expected_variance: float,
    expected_standard_error: float,
    expected_adjustment_vector: list[float],
    expected_target_derivative: list[float],
) -> None:
    outcome, treatment, design, model = _estimated_score_example()
    assert isinstance(model, FittedPropensityMLEProtocol)

    result = NearestNeighborMatch(
        estimand=estimand,
        metric="propensity",
        caliper=None,
        common_support=None,
        inference="abadie_imbens_estimated",
        variance_neighbors=2,
        first_step_covariance_neighbors=2,
        first_step_regression_neighbors=1,
        first_step_covariate_neighbors=1,
    ).fit(
        outcome,
        treatment=treatment,
        propensity=None,
        propensity_model=model,
        propensity_design=design,
        propensity_score_status="estimated",
        propensity_provenance="full_sample_unpenalized_logit_mle",
    )

    assert result.estimate == pytest.approx(expected_estimate, abs=2e-14)
    assert result.known_score_variance == pytest.approx(expected_known_variance, abs=2e-14)
    assert result.first_step_variance_adjustment == pytest.approx(expected_adjustment, abs=2e-14)
    assert result.first_step_asymptotic_adjustment == pytest.approx(
        len(outcome) * expected_adjustment, abs=2e-13
    )
    assert result.variance == pytest.approx(expected_variance, abs=2e-14)
    target_size = (
        len(outcome) if estimand == "ate" else int((treatment == (estimand == "att")).sum())
    )
    assert result.normalized_variance == pytest.approx(target_size * expected_variance, abs=2e-13)
    assert result.standard_error == pytest.approx(expected_standard_error, abs=2e-14)
    np.testing.assert_allclose(
        result.propensity_adjustment_vector, expected_adjustment_vector, rtol=0, atol=2e-14
    )
    np.testing.assert_allclose(
        result.propensity_target_derivative, expected_target_derivative, rtol=0, atol=2e-14
    )
    expected_information = (
        design.to_numpy().T
        @ (
            (result.propensity_scores * (1 - result.propensity_scores)).to_numpy()[:, None]
            * design.to_numpy()
        )
        / len(design)
    )
    np.testing.assert_allclose(result.propensity_information, expected_information, atol=2e-14)
    assert result.propensity_model_name == "_HandFittedLogit"
    assert result.propensity_link == "logit"
    assert result.propensity_model_score_norm < 1e-12
    assert "estimated-propensity" in result.to_markdown()


def test_estimated_score_inference_refuses_unverifiable_first_steps() -> None:
    outcome, treatment, design, model = _estimated_score_example()
    estimator = NearestNeighborMatch(
        metric="propensity",
        caliper=None,
        common_support=None,
        inference="abadie_imbens_estimated",
        variance_neighbors=2,
    )

    with pytest.raises(ValueError, match="propensity_model.*propensity_design"):
        estimator.fit(
            outcome,
            treatment=treatment,
            propensity=model.predict_proba(design)[1],
            propensity_score_status="estimated",
        )
    with pytest.raises(TypeError, match="FittedPropensityMLEProtocol"):
        estimator.fit(
            outcome,
            treatment=treatment,
            propensity_model=object(),
            propensity_design=design,
            propensity_score_status="estimated",
        )
    with pytest.raises(ValueError, match="propensity_design requires propensity_model"):
        NearestNeighborMatch().fit(
            outcome,
            treatment=treatment,
            propensity=model.predict_proba(design)[1],
            propensity_design=design,
        )
    with pytest.raises(ValueError, match="full-sample unpenalized logit MLE"):
        estimator.fit(
            outcome,
            treatment=treatment,
            propensity=None,
            propensity_model=_HandFittedLogit(model.params + np.array([0.2, 0.0]), len(design)),
            propensity_design=design,
            propensity_score_status="estimated",
        )
    with pytest.raises(ValueError, match="propensity_score_status='estimated'"):
        estimator.fit(
            outcome,
            treatment=treatment,
            propensity=None,
            propensity_model=model,
            propensity_design=design,
            propensity_score_status="known",
        )
    with pytest.raises(NotImplementedError, match="metric='propensity'"):
        NearestNeighborMatch(
            caliper=None,
            common_support=None,
            inference="abadie_imbens_estimated",
            variance_neighbors=2,
        ).fit(
            outcome,
            treatment=treatment,
            propensity=None,
            propensity_model=model,
            propensity_design=design,
            propensity_score_status="estimated",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("not_converged", "report convergence"),
        ("wrong_nobs", "estimation-sample size"),
        ("wrong_features", "feature_names"),
        ("wrong_param_names", "params must align"),
        ("wrong_probabilities", "logit linear-index predictions"),
    ],
)
def test_estimated_score_inference_refuses_malformed_fitted_results(
    mutation: str, message: str
) -> None:
    outcome, treatment, design, model = _estimated_score_example()
    if mutation == "not_converged":
        model.converged = False
    elif mutation == "wrong_nobs":
        model.nobs = len(design) + 1
    elif mutation == "wrong_features":
        model.feature_names = ("x", "const")
    elif mutation == "wrong_param_names":
        model.params.index = ["intercept", "x"]
    else:
        original_predict = model.predict_proba

        def shifted_probabilities(X):
            probabilities = original_predict(X)
            probabilities[1] += 0.01
            probabilities[0] = 1.0 - probabilities[1]
            return probabilities

        model.predict_proba = shifted_probabilities

    with pytest.raises(ValueError, match=message):
        NearestNeighborMatch(
            metric="propensity",
            caliper=None,
            common_support=None,
            inference="abadie_imbens_estimated",
            variance_neighbors=2,
        ).fit(
            outcome,
            treatment=treatment,
            propensity_model=model,
            propensity_design=design,
            propensity_score_status="estimated",
        )


def test_estimated_score_inference_refuses_singular_information() -> None:
    outcome, treatment, _, _ = _estimated_score_example()
    design = pd.DataFrame(
        {"const": 1.0, "duplicate_const": 1.0},
        index=outcome.index,
    )
    model = _HandFittedLogit(
        pd.Series([0.0, 0.0], index=design.columns, name="estimate"),
        len(design),
    )

    with pytest.raises(ValueError, match="information matrix is singular"):
        NearestNeighborMatch(
            metric="propensity",
            caliper=None,
            common_support=None,
            inference="abadie_imbens_estimated",
            variance_neighbors=2,
        ).fit(
            outcome,
            treatment=treatment,
            propensity_model=model,
            propensity_design=design,
            propensity_score_status="estimated",
        )


def test_estimated_score_inference_refuses_prediction_index_drift() -> None:
    outcome, treatment, design, model = _estimated_score_example()
    original_predict = model.predict_proba

    def misaligned_probabilities(X):
        return original_predict(X).set_axis(pd.RangeIndex(len(X)))

    model.predict_proba = misaligned_probabilities
    with pytest.raises(ValueError, match="preserve.*index"):
        NearestNeighborMatch(
            metric="propensity",
            caliper=None,
            common_support=None,
            inference="abadie_imbens_estimated",
            variance_neighbors=2,
        ).fit(
            outcome,
            treatment=treatment,
            propensity_model=model,
            propensity_design=design,
            propensity_score_status="estimated",
        )


@pytest.mark.parametrize("estimand", ["att", "atc", "ate"])
def test_estimated_score_adjustment_is_invariant_to_consistent_permutation(
    estimand: str,
) -> None:
    outcome, treatment, design, model = _estimated_score_example()
    estimator = NearestNeighborMatch(
        estimand=estimand,
        metric="propensity",
        caliper=None,
        common_support=None,
        inference="abadie_imbens_estimated",
        variance_neighbors=2,
    )
    original = estimator.fit(
        outcome,
        treatment=treatment,
        propensity_model=model,
        propensity_design=design,
        propensity_score_status="estimated",
    )
    order = np.random.default_rng(20_260_728).permutation(len(outcome))
    permuted = estimator.fit(
        outcome.iloc[order],
        treatment=treatment.iloc[order],
        propensity_model=_HandFittedLogit(model.params.copy(), len(design)),
        propensity_design=design.iloc[order],
        propensity_score_status="estimated",
    )

    assert permuted.estimate == pytest.approx(original.estimate, abs=2e-14)
    assert permuted.variance == pytest.approx(original.variance, abs=2e-14)
    pd.testing.assert_series_equal(
        permuted.propensity_adjustment_vector,
        original.propensity_adjustment_vector,
    )
    pd.testing.assert_series_equal(
        permuted.propensity_target_derivative,
        original.propensity_target_derivative,
    )


@pytest.mark.parametrize(
    "selection",
    [
        {"caliper": 1.0, "common_support": None},
        {"caliper": None, "common_support": "intersection"},
    ],
)
def test_estimated_score_inference_refuses_target_selection(selection) -> None:
    outcome, treatment, design, model = _estimated_score_example()
    with pytest.raises(NotImplementedError, match="caliper=None.*common_support=None"):
        NearestNeighborMatch(
            metric="propensity",
            inference="abadie_imbens_estimated",
            variance_neighbors=2,
            **selection,
        ).fit(
            outcome,
            treatment=treatment,
            propensity_model=model,
            propensity_design=design,
            propensity_score_status="estimated",
        )


def test_analytical_inference_refuses_support_or_caliper_target_selection() -> None:
    y, treatment, propensity = _six_unit_variance_example()
    for model in (
        NearestNeighborMatch(
            common_support="intersection",
            caliper=None,
            inference="abadie_imbens",
        ),
        NearestNeighborMatch(
            common_support=None,
            caliper=20.0,
            inference="abadie_imbens",
        ),
    ):
        with pytest.raises(NotImplementedError, match="caliper=None.*common_support=None"):
            model.fit(
                y,
                treatment=treatment,
                propensity=propensity,
                propensity_score_status="known",
            )


def test_analytical_inference_requires_adequate_same_arm_variance_matches() -> None:
    y, treatment, propensity = _six_unit_variance_example()
    with pytest.raises(ValueError, match="variance_neighbors.*same-arm"):
        NearestNeighborMatch(
            caliper=None,
            common_support=None,
            inference="abadie_imbens",
            variance_neighbors=3,
        ).fit(
            y,
            treatment=treatment,
            propensity=propensity,
            propensity_score_status="known",
        )


def test_analytical_inference_refuses_expanded_same_arm_variance_ties() -> None:
    logits = np.array([0.0, 2.0, 4.0, 0.0, 2.0, 4.0])
    propensity = 1 / (1 + np.exp(-logits))
    with pytest.raises(ValueError, match="conditional-variance.*boundary ties"):
        NearestNeighborMatch(
            estimand="att",
            caliper=None,
            common_support=None,
            inference="abadie_imbens",
            variance_neighbors=1,
        ).fit(
            [0.0, 1.0, 2.0, 4.0, 6.0, 9.0],
            treatment=[0, 0, 0, 1, 1, 1],
            propensity=propensity,
            propensity_score_status="known",
        )


@pytest.mark.simulation
def test_known_score_ate_analytical_intervals_have_seeded_coverage_smoke() -> None:
    rng = np.random.default_rng(20_260_728)
    true_effect = 1.5
    estimates = []
    standard_errors = []
    covered = []
    for _ in range(100):
        covariate = rng.uniform(-2.0, 2.0, 600)
        propensity = 1 / (1 + np.exp(-0.8 * covariate))
        treatment = rng.binomial(1, propensity)
        outcome = 0.5 * covariate + true_effect * treatment + rng.normal(size=600)
        result = NearestNeighborMatch(
            estimand="ate",
            caliper=None,
            common_support=None,
            inference="abadie_imbens",
        ).fit(
            outcome,
            treatment=treatment,
            propensity=propensity,
            propensity_score_status="known",
            propensity_provenance="known_simulation_dgp",
        )
        interval = result.conf_int()
        estimates.append(result.estimate)
        standard_errors.append(result.standard_error)
        covered.append(interval["lower"] <= true_effect <= interval["upper"])

    coverage = float(np.mean(covered))
    empirical_standard_deviation = float(np.std(estimates, ddof=1))
    mean_standard_error = float(np.mean(standard_errors))
    assert 0.86 <= coverage <= 0.99
    assert np.mean(estimates) == pytest.approx(true_effect, abs=0.08)
    assert mean_standard_error / empirical_standard_deviation == pytest.approx(1.0, abs=0.3)


def test_clustered_matching_inference_is_not_silently_approximated() -> None:
    y, treatment, propensity, _ = _three_unit_example()
    with pytest.raises(NotImplementedError, match="clustered"):
        NearestNeighborMatch(inference="none").fit(
            y,
            treatment=treatment,
            propensity=propensity,
            clusters=pd.Series([0, 1, 1], index=y.index),
        )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_propensity_must_be_finite_and_strictly_inside_unit_interval(bad: float) -> None:
    y, treatment, propensity, _ = _three_unit_example()
    propensity.iloc[0] = bad
    with pytest.raises(ValueError, match="propensity"):
        NearestNeighborMatch(inference="none").fit(y, treatment=treatment, propensity=propensity)


def test_treatment_alignment_and_missing_outcomes_are_strict() -> None:
    y, treatment, propensity, covariates = _three_unit_example()
    with pytest.raises(ValueError, match="coded exactly"):
        NearestNeighborMatch(inference="none").fit(
            y, treatment=treatment.replace({1: 2}), propensity=propensity
        )
    with pytest.raises(ValueError, match="indices must match"):
        NearestNeighborMatch(inference="none").fit(
            y,
            treatment=treatment.set_axis(pd.RangeIndex(1, 4)),
            propensity=propensity,
        )
    y.iloc[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        NearestNeighborMatch(inference="none").fit(
            y, treatment=treatment, propensity=propensity, covariates=covariates
        )


def test_labeled_inputs_must_align_when_outcome_is_unlabeled() -> None:
    treatment = pd.Series([0, 1, 1], index=["a", "b", "c"])
    propensity = pd.Series([0.2, 0.3, 0.7], index=["c", "b", "a"])
    with pytest.raises(ValueError, match="indices must match"):
        NearestNeighborMatch(inference="none").fit(
            [1.0, 3.0, 5.0], treatment=treatment, propensity=propensity
        )


def test_repeated_aligned_indices_remain_auditable_by_position() -> None:
    index = pd.Index(["repeated", "repeated", "other"])
    result = NearestNeighborMatch(caliper=None, common_support=None).fit(
        pd.Series([1.0, 4.0, 10.0], index=index),
        treatment=pd.Series([0, 1, 1], index=index),
        propensity=pd.Series([0.2, 0.25, 0.75], index=index),
    )

    assert result.estimate == pytest.approx(6.0, abs=1e-14)
    assert result.match_table["focal_position"].tolist() == [1, 2]
    assert result.match_table["comparison_position"].tolist() == [0, 0]
    assert result.match_table.groupby("focal_position")["match_weight"].sum().eq(1).all()


def test_ate_effect_weights_reconstruct_the_estimate_and_reuse_counts() -> None:
    y, treatment, propensity, _ = _three_unit_example()
    result = NearestNeighborMatch(
        estimand="ate", caliper=None, common_support=None, inference="none"
    ).fit(y, treatment=treatment, propensity=propensity)

    assert result.effect_weights @ y == pytest.approx(result.estimate, abs=1e-14)
    assert result.effect_weights[treatment == 1].sum() == pytest.approx(1.0, abs=1e-14)
    assert result.effect_weights[treatment == 0].sum() == pytest.approx(-1.0, abs=1e-14)
    assert result.comparison_reuse_counts.to_dict() == {
        "control": 2,
        "treated_near": 1,
        "treated_far": 0,
    }
    assert result.maximum_reuse_count == 2
    assert result.unique_comparison_observations == 2
    assert "outcome" not in result.match_table.columns


def test_no_eligible_matches_and_too_many_neighbors_fail_actionably() -> None:
    y, treatment, propensity, _ = _three_unit_example()
    with pytest.raises(ValueError, match="No focal observations.*caliper"):
        NearestNeighborMatch(
            caliper=1e-12,
            common_support=None,
            inference="none",
        ).fit(y, treatment=treatment, propensity=propensity)
    with pytest.raises(ValueError, match="neighbors"):
        NearestNeighborMatch(
            neighbors=2,
            caliper=None,
            common_support=None,
            inference="none",
        ).fit(y, treatment=treatment, propensity=propensity)


def test_empty_propensity_provenance_is_refused() -> None:
    y, treatment, propensity, _ = _three_unit_example()
    with pytest.raises(ValueError, match="propensity_provenance"):
        NearestNeighborMatch(caliper=None, common_support=None).fit(
            y,
            treatment=treatment,
            propensity=propensity,
            propensity_provenance="  ",
        )

    with pytest.raises(ValueError, match="propensity_score_status"):
        NearestNeighborMatch(caliper=None, common_support=None).fit(
            y,
            treatment=treatment,
            propensity=propensity,
            propensity_score_status="cross_fitted",
        )


def test_nonoverlapping_support_and_duplicate_covariates_are_refused() -> None:
    with pytest.raises(ValueError, match="no intersecting common support"):
        NearestNeighborMatch().fit(
            [0.0, 1.0, 2.0, 3.0],
            treatment=[0, 0, 1, 1],
            propensity=[0.1, 0.2, 0.8, 0.9],
        )

    covariates = pd.DataFrame(np.ones((4, 2)), columns=["duplicate", "duplicate"])
    with pytest.raises(ValueError, match="column names must be unique"):
        NearestNeighborMatch(common_support=None).fit(
            [0.0, 1.0, 2.0, 3.0],
            treatment=[0, 0, 1, 1],
            propensity=[0.2, 0.4, 0.3, 0.5],
            covariates=covariates,
        )

    string_duplicate_covariates = pd.DataFrame(np.ones((4, 2)), columns=[1, "1"])
    with pytest.raises(ValueError, match="unique after string conversion"):
        NearestNeighborMatch(common_support=None).fit(
            [0.0, 1.0, 2.0, 3.0],
            treatment=[0, 0, 1, 1],
            propensity=[0.2, 0.4, 0.3, 0.5],
            covariates=string_duplicate_covariates,
        )

    with pytest.raises(ValueError, match="at least one column"):
        NearestNeighborMatch(common_support=None).fit(
            [0.0, 1.0, 2.0, 3.0],
            treatment=[0, 0, 1, 1],
            propensity=[0.2, 0.4, 0.3, 0.5],
            covariates=pd.DataFrame(index=pd.RangeIndex(4)),
        )


def test_numpy_integer_neighbor_count_is_accepted() -> None:
    assert NearestNeighborMatch(neighbors=np.int64(1)).neighbors == 1

"""Contract-first tests for nearest-neighbor matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalkit import NearestNeighborMatch, NearestNeighborMatchResult


def _three_unit_example():
    index = pd.Index(["control", "treated_near", "treated_far"])
    return (
        pd.Series([1.0, 4.0, 10.0], index=index, name="outcome"),
        pd.Series([0, 1, 1], index=index, name="treatment"),
        pd.Series([0.20, 0.25, 0.75], index=index, name="propensity"),
        pd.DataFrame({"baseline": [0.0, 0.2, 1.0]}, index=index),
    )


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
    with pytest.raises(NotImplementedError, match="propensity-score estimation"):
        NearestNeighborMatch(
            caliper=None,
            common_support=None,
            inference="abadie_imbens",
        ).fit(y, treatment=treatment, propensity=propensity)


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

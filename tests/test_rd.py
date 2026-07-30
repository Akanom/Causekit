"""Contract tests for continuity-based regression discontinuity estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import (
    RDBandwidthSelection,
    RDManipulationDiagnostic,
    RegressionDiscontinuity,
    RegressionDiscontinuityResult,
)


def _sharp_linear_sample() -> tuple[pd.Series, pd.Series]:
    running = pd.Series(
        [-1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8],
        index=pd.Index(range(100, 110), name="row"),
        name="score",
    )
    outcome = pd.Series(
        1.0 + 2.0 * running + 3.0 * (running >= 0.0),
        index=running.index,
        name="outcome",
    )
    return outcome, running


def test_public_result_and_diagnostic_types_are_frozen_dataclasses() -> None:
    assert RegressionDiscontinuityResult.__dataclass_params__.frozen is True
    assert RDBandwidthSelection.__dataclass_params__.frozen is True
    assert RDManipulationDiagnostic.__dataclass_params__.frozen is True


def test_hand_computed_sharp_local_constant_and_hc1_identity() -> None:
    running = pd.Series([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5], name="score")
    outcome = pd.Series([0.0, 1.0, 2.0, 4.0, 6.0, 8.0], name="outcome")

    result = RegressionDiscontinuity(
        design="sharp",
        bandwidth=2.0,
        bias_bandwidth=2.0,
        polynomial_order=0,
        bias_order=1,
        kernel="uniform",
    ).fit(outcome, running=running)

    # The conventional local-constant effect is mean(Y+) - mean(Y-) = 6 - 1.
    assert result.conventional_estimate == pytest.approx(5.0, abs=1e-14)
    # HC1: 6/(6-2) * sum_i {a_i e_i}^2 = 3/2 * 10/9 = 5/3.
    assert result.conventional_standard_error == pytest.approx(np.sqrt(5.0 / 3.0), abs=1e-14)
    # The order-one bias fit extrapolates the two lines to 2 and 3 at the cutoff.
    assert result.bias_corrected_estimate == pytest.approx(-1.0, abs=1e-13)
    assert result.outcome_jump == pytest.approx(-1.0, abs=1e-13)
    assert result.params.index.tolist() == ["rd_effect"]
    assert result.primary_inference == "robust_bias_corrected"


def test_sharp_piecewise_linear_jump_is_recovered_exactly() -> None:
    outcome, running = _sharp_linear_sample()

    result = RegressionDiscontinuity(
        bandwidth=2.0,
        bias_bandwidth=2.0,
        polynomial_order=1,
        bias_order=2,
    ).fit(outcome, running=running)

    assert result.design == "sharp"
    assert result.estimand == "cutoff_average_treatment_effect"
    assert result.conventional_estimate == pytest.approx(3.0, abs=1e-12)
    assert result.bias_corrected_estimate == pytest.approx(3.0, abs=1e-12)
    assert result.treatment_jump == pytest.approx(1.0, abs=0.0)
    assert result.n_left == 5
    assert result.n_right == 5
    assert result.running_name == "score"
    assert result.outcome_name == "outcome"
    assert result.weights.index.equals(outcome.index)
    assert result.local_sample.index.equals(outcome.index)


def test_fuzzy_local_wald_identity_is_hand_computed() -> None:
    running = pd.Series(
        [-2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0],
        name="score",
    )
    treatment = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="take_up")
    outcome = pd.Series(2.0 * treatment, name="outcome")

    result = RegressionDiscontinuity(
        design="fuzzy",
        bandwidth=2.1,
        bias_bandwidth=2.1,
        polynomial_order=0,
        bias_order=1,
        kernel="uniform",
    ).fit(outcome, running=running, treatment=treatment)

    assert result.estimand == "cutoff_complier_local_average_treatment_effect"
    assert result.conventional_treatment_jump == pytest.approx(0.2, abs=1e-14)
    assert result.conventional_outcome_jump == pytest.approx(0.4, abs=1e-14)
    assert result.conventional_estimate == pytest.approx(2.0, abs=1e-13)
    assert result.bias_corrected_estimate == pytest.approx(2.0, abs=1e-12)
    assert result.first_stage is not None
    assert result.first_stage["estimate"] > 0.0


def test_supplied_sharp_assignment_is_audited_not_used_as_a_fuzzy_first_stage() -> None:
    outcome, running = _sharp_linear_sample()
    treatment = (running >= 0.0).astype(int).rename("eligibility")

    result = RegressionDiscontinuity(design="sharp", bandwidth=2.0).fit(
        outcome,
        running=running,
        treatment=treatment,
    )

    assert result.treatment_name == "eligibility"
    assert result.treatment_jump == pytest.approx(1.0)

    treatment.iloc[0] = 1
    with pytest.raises(ValueError, match="sharp assignment"):
        RegressionDiscontinuity(design="sharp", bandwidth=2.0).fit(
            outcome,
            running=running,
            treatment=treatment,
        )


def test_native_bandwidth_selection_is_deterministic_and_auditable() -> None:
    rng = np.random.default_rng(20_260_730)
    running = pd.Series(rng.uniform(-2.5, 2.5, 900), name="score")
    outcome = pd.Series(
        0.4
        + 0.7 * running
        + 0.25 * running**2
        + 2.0 * (running >= 0.0)
        + rng.normal(scale=0.55, size=len(running)),
        name="outcome",
    )

    first = RegressionDiscontinuity(bandwidth="native_mse", bandwidth_candidates=9).fit(
        outcome, running=running
    )
    second = RegressionDiscontinuity(bandwidth="native_mse", bandwidth_candidates=9).fit(
        outcome, running=running
    )

    assert first.bandwidth_selection.method == "native_design_conditional_mse_grid"
    assert first.bandwidth_left == second.bandwidth_left
    assert first.bandwidth_right == second.bandwidth_right
    pd.testing.assert_frame_equal(
        first.bandwidth_selection.candidates,
        second.bandwidth_selection.candidates,
    )
    assert first.bandwidth_selection.candidates["selected"].sum() == 1
    assert first.bias_bandwidth_left >= first.bandwidth_left
    assert first.bias_bandwidth_right >= first.bandwidth_right
    assert first.bias_corrected_estimate == pytest.approx(2.0, abs=0.35)


def test_native_fuzzy_grid_retains_but_never_selects_nonpositive_first_stages() -> None:
    rng = np.random.default_rng(2_026_103_000)
    running = rng.uniform(-2.0, 2.0, 1_600)
    smooth = 0.5 + 0.55 * running + 0.22 * running**2 + 0.04 * running**3
    probability = 1.0 / (1.0 + np.exp(-(-1.0 + 2.0 * (running >= 0.0) + 0.1 * running)))
    treatment = rng.binomial(1, probability)
    outcome = smooth + 2.2 * treatment + rng.normal(scale=0.7, size=len(running))

    result = RegressionDiscontinuity(
        design="fuzzy",
        bandwidth="native_mse",
        bandwidth_candidates=7,
    ).fit(outcome, running=running, treatment=treatment)
    candidates = result.bandwidth_selection.candidates
    inadmissible = ~candidates["admissible"]

    assert inadmissible.any()
    assert np.isinf(candidates.loc[inadmissible, "objective"]).all()
    assert not candidates.loc[inadmissible, "selected"].any()
    assert candidates.loc[candidates["selected"], "admissible"].all()


def test_asymmetric_manual_bandwidths_and_supported_kernels_are_explicit() -> None:
    outcome, running = _sharp_linear_sample()

    for kernel in ("triangular", "uniform", "epanechnikov"):
        result = RegressionDiscontinuity(
            bandwidth=(1.6, 2.0),
            bias_bandwidth=(1.8, 2.0),
            kernel=kernel,
        ).fit(outcome, running=running)
        assert result.bandwidth_left == 1.6
        assert result.bandwidth_right == 2.0
        assert result.kernel == kernel
        assert result.bias_corrected_estimate == pytest.approx(3.0, abs=1e-11)


def test_density_manipulation_diagnostic_is_separate_from_estimation() -> None:
    running = pd.Series(np.r_[-np.linspace(0.02, 1.0, 80), np.linspace(0.02, 1.0, 80)])
    outcome = pd.Series(1.0 + running + 1.5 * (running >= 0.0))

    result = RegressionDiscontinuity(
        bandwidth=0.9,
        manipulation_bandwidth=0.8,
    ).fit(outcome, running=running)

    diagnostic = result.manipulation
    assert diagnostic.available is True
    assert diagnostic.method == "one_sided_boundary_kernel_density"
    assert diagnostic.log_density_jump == pytest.approx(0.0, abs=1e-14)
    assert diagnostic.pvalue == pytest.approx(1.0, abs=1e-14)
    assert diagnostic.reject is False
    assert result.bias_corrected_estimate == pytest.approx(1.5, abs=1e-11)


def test_mass_points_are_reported_or_refused_by_declared_policy() -> None:
    running = pd.Series([-2, -1, -1, -0.5, 0.5, 1, 1, 2], dtype=float)
    outcome = pd.Series(1 + running + 2 * (running >= 0))

    checked = RegressionDiscontinuity(
        bandwidth=2.1,
        bias_bandwidth=2.1,
        mass_points="check",
    ).fit(outcome, running=running)
    assert checked.mass_points_detected == 2
    assert checked.unique_left == 3
    assert checked.unique_right == 3

    with pytest.raises(ValueError, match="mass points"):
        RegressionDiscontinuity(
            bandwidth=2.1,
            bias_bandwidth=2.1,
            mass_points="raise",
        ).fit(outcome, running=running)


def test_clustered_inference_aggregates_only_local_score_contributions() -> None:
    rng = np.random.default_rng(733)
    clusters = np.repeat(np.arange(40), 4)
    running = pd.Series(rng.uniform(-1.5, 1.5, len(clusters)))
    outcome = pd.Series(
        0.8 * running
        + 1.2 * (running >= 0)
        + rng.normal(size=40)[clusters]
        + rng.normal(scale=0.3, size=len(clusters))
    )

    result = RegressionDiscontinuity(
        bandwidth=1.25,
        bias_bandwidth=1.4,
        covariance="clustered",
    ).fit(outcome, running=running, clusters=clusters)

    assert result.covariance_type == "clustered"
    assert result.n_clusters is not None and result.n_clusters >= 2
    assert result.inference_distribution == "t"
    assert result.inference_df == result.n_clusters - 1
    assert np.isfinite(result.standard_errors.iloc[0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"design": "unknown"}, "design"),
        ({"cutoff": np.nan}, "cutoff"),
        ({"bandwidth": 0.0}, "bandwidth"),
        ({"bandwidth": (1.0, -1.0)}, "bandwidth"),
        ({"bias_bandwidth": 0.0}, "bias bandwidth"),
        ({"polynomial_order": -1}, "polynomial_order"),
        ({"polynomial_order": 4}, "polynomial_order"),
        ({"polynomial_order": 2, "bias_order": 2}, "bias_order"),
        ({"kernel": "gaussian"}, "kernel"),
        ({"covariance": "unadjusted"}, "covariance"),
        ({"missing": "ignore"}, "missing"),
        ({"mass_points": "ignore"}, "mass_points"),
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"manipulation_bandwidth": -1.0}, "manipulation_bandwidth"),
        ({"first_stage_tolerance": 0.0}, "first_stage_tolerance"),
        ({"bandwidth_candidates": 2}, "bandwidth_candidates"),
        ({"max_condition_number": 1.0}, "max_condition_number"),
    ],
)
def test_constructor_refuses_ambiguous_or_unsafe_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RegressionDiscontinuity(**kwargs)


def test_fuzzy_design_refuses_missing_nonbinary_or_nonpositive_first_stage() -> None:
    outcome, running = _sharp_linear_sample()

    with pytest.raises(ValueError, match="treatment is required"):
        RegressionDiscontinuity(design="fuzzy", bandwidth=2.0).fit(outcome, running=running)

    with pytest.raises(ValueError, match="binary"):
        RegressionDiscontinuity(design="fuzzy", bandwidth=2.0).fit(
            outcome,
            running=running,
            treatment=np.linspace(0.0, 1.0, len(running)),
        )

    no_jump = pd.Series([0, 1, 0, 1, 0, 0, 1, 0, 1, 0], index=running.index)
    with pytest.raises(ValueError, match="positive treatment jump"):
        RegressionDiscontinuity(design="fuzzy", bandwidth=2.0).fit(
            outcome,
            running=running,
            treatment=no_jump,
        )


def test_cutoff_support_and_local_rank_refusals_are_explicit() -> None:
    with pytest.raises(ValueError, match="both sides"):
        RegressionDiscontinuity(bandwidth=1.0).fit([1.0, 2.0, 3.0], running=[0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="left side"):
        RegressionDiscontinuity(
            bandwidth=0.2,
            bias_bandwidth=0.2,
            polynomial_order=1,
            bias_order=2,
        ).fit(
            np.arange(8.0),
            running=[-2.0, -1.0, -0.8, -0.4, 0.05, 0.1, 0.15, 0.19],
        )

    repeated = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="unique running"):
        RegressionDiscontinuity(
            bandwidth=2.0,
            bias_bandwidth=2.0,
            polynomial_order=1,
            bias_order=2,
        ).fit(np.arange(8.0), running=repeated)


def test_strict_alignment_missing_and_cluster_contracts() -> None:
    outcome, running = _sharp_linear_sample()

    with pytest.raises(ValueError, match="indices"):
        RegressionDiscontinuity(bandwidth=2.0).fit(
            outcome,
            running=running.iloc[::-1],
        )

    missing_running = running.copy()
    missing_running.iloc[2] = np.nan
    with pytest.raises(ValueError, match="missing"):
        RegressionDiscontinuity(bandwidth=2.0).fit(
            outcome,
            running=missing_running,
        )

    dropped = RegressionDiscontinuity(bandwidth=2.0, missing="drop").fit(
        outcome,
        running=missing_running,
    )
    assert dropped.nobs == len(outcome) - 1
    assert outcome.index[2] not in dropped.local_sample.index

    with pytest.raises(ValueError, match="clusters may be provided"):
        RegressionDiscontinuity(bandwidth=2.0, covariance="robust").fit(
            outcome,
            running=running,
            clusters=np.arange(len(outcome)),
        )
    with pytest.raises(ValueError, match="clusters must be provided"):
        RegressionDiscontinuity(bandwidth=2.0, covariance="clustered").fit(
            outcome,
            running=running,
        )


def test_row_permutation_changes_neither_estimates_nor_labeled_weights() -> None:
    rng = np.random.default_rng(83)
    index = pd.Index(np.arange(200, 440), name="row")
    running = pd.Series(rng.uniform(-2.0, 2.0, len(index)), index=index, name="score")
    outcome = pd.Series(
        0.5 + running + 1.75 * (running >= 0) + rng.normal(scale=0.3, size=len(index)),
        index=index,
        name="outcome",
    )
    order = rng.permutation(len(index))

    original = RegressionDiscontinuity(bandwidth=1.4, bias_bandwidth=1.7).fit(
        outcome, running=running
    )
    permuted = RegressionDiscontinuity(bandwidth=1.4, bias_bandwidth=1.7).fit(
        outcome.iloc[order], running=running.iloc[order]
    )

    assert permuted.bias_corrected_estimate == pytest.approx(
        original.bias_corrected_estimate, abs=1e-12
    )
    assert permuted.robust_standard_error == pytest.approx(
        original.robust_standard_error, abs=1e-12
    )
    pd.testing.assert_frame_equal(
        original.weights.sort_index(),
        permuted.weights.sort_index(),
        check_exact=False,
        atol=1e-14,
        rtol=1e-14,
    )


def test_result_summary_and_confidence_interval_are_defensive() -> None:
    outcome, running = _sharp_linear_sample()
    result = RegressionDiscontinuity(bandwidth=2.0).fit(outcome, running=running)

    summary = result.summary_frame()
    interval = result.conf_int()
    assert summary.index.tolist() == ["rd_effect"]
    assert interval.index.tolist() == ["rd_effect"]
    assert {"estimate", "standard_error", "statistic", "pvalue", "lower", "upper"} <= set(
        summary.columns
    )
    assert interval.loc["rd_effect", "lower"] <= result.params.iloc[0]
    assert interval.loc["rd_effect", "upper"] >= result.params.iloc[0]
    summary.iloc[0, 0] = 999.0
    assert result.params.iloc[0] != 999.0


def test_rd_plot_uses_binned_local_data_and_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    outcome, running = _sharp_linear_sample()
    result = RegressionDiscontinuity(bandwidth=2.0).fit(outcome, running=running)

    axes = result.plot(bins=3)

    assert axes.get_xlabel() == "score"
    assert axes.get_ylabel() == "outcome"
    assert len(axes.lines) == 3
    with pytest.raises(ValueError, match="bins"):
        result.plot(bins=1)

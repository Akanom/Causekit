"""Contract-first tests for conventional and efficient difference-in-differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalkit import DiDResult, DifferenceInDifferences, EfficientDiD

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


def test_covariate_and_multiplier_bootstrap_paths_are_explicitly_refused() -> None:
    data = _staggered_panel().assign(x=1.0)
    with pytest.raises(NotImplementedError, match="covariate-adjusted"):
        _fit(EfficientDiD(), data, covariates=["x"])
    with pytest.raises(NotImplementedError, match="simultaneous"):
        DifferenceInDifferences(inference="multiplier_bootstrap")


def test_efficient_singular_weight_system_is_refused_without_regularization() -> None:
    data = _efficient_hand_panel()
    data.loc[data["treatment_time"] == 4.0, "outcome"] = np.tile([-10.0, -10.0, -10.0, 0.0], 4)
    with pytest.raises(ValueError, match="singular"):
        _fit(EfficientDiD(), data)


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

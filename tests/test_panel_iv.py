"""Contract tests for static fixed-effects panel two-stage least squares."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import PanelIV2SLS, confint, fitted_values, lincom, residuals, vcov, wald_test


def _hand_panel() -> pd.DataFrame:
    z = np.array([-1.5, -0.5, 0.5, 1.5])
    orthogonal = np.array([1.0, -1.0, -1.0, 1.0])
    rows: list[dict[str, float | str]] = []
    for position, entity in enumerate(("a", "b", "c")):
        if position == 0:
            first_stage_error = z
            structural_error = 0.5 * z
        elif position == 1:
            first_stage_error = -z
            structural_error = -0.5 * z
        else:
            first_stage_error = orthogonal
            structural_error = orthogonal
        endogenous = 1.5 * z + first_stage_error + 10.0 * position
        outcome = 2.0 * endogenous + structural_error + 4.0 * position
        for period in range(4):
            rows.append(
                {
                    "entity": entity,
                    "time": float(period),
                    "outcome": float(outcome[period]),
                    "endogenous": float(endogenous[period]),
                    "instrument": float(z[period]),
                }
            )
    return pd.DataFrame(rows)


def _simulated_panel(*, seed: int = 73, n_entities: int = 100, n_periods: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity_effects = rng.normal(size=n_entities)
    time_effects = np.linspace(-0.4, 0.5, n_periods)
    rows: list[dict[str, float | int]] = []
    for entity in range(n_entities):
        for period in range(n_periods):
            instrument = rng.normal()
            control = rng.normal()
            first_error = rng.normal(scale=0.6)
            endogenous = (
                0.9 * instrument + 0.35 * control + 0.4 * entity_effects[entity] + first_error
            )
            error = 0.65 * first_error + rng.normal(scale=0.45)
            outcome = (
                1.8 * endogenous
                - 0.55 * control
                + entity_effects[entity]
                + time_effects[period]
                + error
            )
            rows.append(
                {
                    "entity": entity,
                    "time": period,
                    "outcome": outcome,
                    "endogenous": endogenous,
                    "instrument": instrument,
                    "control": control,
                }
            )
    return pd.DataFrame(rows)


def _fit_hand(data: pd.DataFrame | None = None):
    return PanelIV2SLS(time_effects=False).fit(
        _hand_panel() if data is None else data,
        outcome="outcome",
        endogenous=["endogenous"],
        instruments=["instrument"],
        entity="entity",
        time="time",
    )


def test_hand_computed_entity_fe_point_cr1_and_first_stage_contract() -> None:
    result = _fit_hand()

    assert result.params.index.tolist() == ["endogenous"]
    assert result.params["endogenous"] == pytest.approx(2.0, abs=1e-12)
    assert result.standard_errors["endogenous"] == pytest.approx(np.sqrt(11.0 / 216.0))
    assert result.covariance.iloc[0, 0] == pytest.approx(11.0 / 216.0)
    assert result.inference_distribution == "t"
    assert result.inference_df == 2.0
    assert result.n_clusters == 3
    assert result.nobs == 12
    assert result.n_entities == 3
    assert result.n_periods == 4
    assert result.absorbed_rank == 3
    assert result.rank == 4
    assert result.df_resid == 8
    assert result.effects == ("entity",)
    assert result.cluster_name == "entity"
    assert result.balanced is True
    first = result.first_stage["endogenous"]
    assert first.r_squared == pytest.approx(135.0 / 191.0)
    assert first.partial_r_squared == pytest.approx(135.0 / 191.0)
    assert first.classical_f_statistic == pytest.approx(135.0 / 7.0)
    assert first.classical_f_df_denom == 8
    assert first.excluded_instrument_distribution == "F"
    assert first.weak_instrument_warning is False
    assert result.instrument_variation.loc["instrument", "role"] == "excluded_instrument"
    assert result.instrument_variation.loc["instrument", "within_std"] > 0.0
    np.testing.assert_allclose(
        result.fitted_values.to_numpy() + result.residuals.to_numpy(),
        result.outcome.to_numpy(),
        atol=1e-12,
    )


def test_two_way_fixed_effects_recover_structural_coefficients() -> None:
    data = _simulated_panel()
    result = PanelIV2SLS(time_effects=True).fit(
        data,
        outcome="outcome",
        endogenous=["endogenous"],
        instruments=["instrument"],
        exogenous=["control"],
        entity="entity",
        time="time",
    )

    assert result.effects == ("entity", "time")
    assert result.params["endogenous"] == pytest.approx(1.8, abs=0.09)
    assert result.params["control"] == pytest.approx(-0.55, abs=0.09)
    assert result.absorbed_rank == result.n_entities + result.n_periods - 1
    assert result.within_r_squared > 0.7
    assert result.first_stage["endogenous"].partial_r_squared > 0.4


def test_unbalanced_two_way_panel_is_supported_by_converged_alternating_projections() -> None:
    data = _simulated_panel(n_entities=40, n_periods=5)
    remove = ((data["entity"] + 2 * data["time"]) % 11) == 0
    unbalanced = data.loc[~remove].copy()

    result = PanelIV2SLS(time_effects=True).fit(
        unbalanced,
        outcome="outcome",
        endogenous="endogenous",
        instruments="instrument",
        exogenous="control",
        entity="entity",
        time="time",
    )

    assert result.balanced is False
    assert result.params["endogenous"] == pytest.approx(1.8, abs=0.16)
    assert result.within_iterations > 1
    assert result.within_converged is True
    assert result.within_max_abs_group_mean < 1e-10


def test_row_permutation_preserves_panel_iv_result_and_panel_order() -> None:
    data = _simulated_panel(n_entities=30, n_periods=5)
    first = PanelIV2SLS().fit(
        data,
        outcome="outcome",
        endogenous="endogenous",
        instruments="instrument",
        exogenous="control",
        entity="entity",
        time="time",
    )
    second = PanelIV2SLS().fit(
        data.sample(frac=1.0, random_state=19),
        outcome="outcome",
        endogenous="endogenous",
        instruments="instrument",
        exogenous="control",
        entity="entity",
        time="time",
    )

    pd.testing.assert_series_equal(first.params, second.params)
    pd.testing.assert_frame_equal(first.covariance, second.covariance)
    pd.testing.assert_series_equal(first.residuals, second.residuals)
    assert first.estimation_index.equals(second.estimation_index)


def test_higher_level_clusters_must_be_nested_and_are_used_for_inference() -> None:
    data = _simulated_panel(n_entities=24, n_periods=5)
    data["region"] = data["entity"] // 4
    result = PanelIV2SLS().fit(
        data,
        outcome="outcome",
        endogenous="endogenous",
        instruments="instrument",
        exogenous="control",
        entity="entity",
        time="time",
        cluster="region",
    )

    assert result.cluster_name == "region"
    assert result.n_clusters == 6
    assert result.inference_df == 5.0

    data.loc[data.index[0], "region"] = 99
    with pytest.raises(ValueError, match="constant within entity"):
        PanelIV2SLS().fit(
            data,
            outcome="outcome",
            endogenous="endogenous",
            instruments="instrument",
            exogenous="control",
            entity="entity",
            time="time",
            cluster="region",
        )


def test_missing_drop_uses_one_mask_and_records_unbalanced_panel() -> None:
    data = _hand_panel()
    data.loc[0, "outcome"] = np.nan
    result = PanelIV2SLS(time_effects=False, missing="drop").fit(
        data,
        outcome="outcome",
        endogenous="endogenous",
        instruments="instrument",
        entity="entity",
        time="time",
    )

    assert result.dropped_rows == 1
    assert result.nobs == 11
    assert result.balanced is False
    assert ("a", 0.0) not in result.estimation_index


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"covariance": "bad"}, ValueError, "covariance"),
        ({"missing": "bad"}, ValueError, "missing"),
        ({"time_effects": 1}, TypeError, "time_effects"),
    ],
)
def test_constructor_refusals(kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        PanelIV2SLS(**kwargs)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns="instrument"), "Missing required columns"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "entity-time"),
        (
            lambda frame: frame.assign(entity=lambda value: value["entity"].mask(value.index == 0)),
            "entity identifiers",
        ),
        (
            lambda frame: frame.assign(time=lambda value: value["time"].mask(value.index == 0)),
            "time identifiers",
        ),
        (
            lambda frame: frame.assign(
                outcome=lambda value: value["outcome"].mask(value.index == 0)
            ),
            "missing or non-finite",
        ),
        (lambda frame: frame.assign(instrument="not numeric"), "numeric"),
        (lambda frame: frame.loc[frame["entity"] == "a"].iloc[[0]].copy(), "at least two entities"),
        (
            lambda frame: frame.loc[~((frame["entity"] == "a") & (frame["time"] > 0))].copy(),
            "two retained observations",
        ),
        (
            lambda frame: frame.assign(
                instrument=frame.groupby("entity")["instrument"].transform("mean")
            ),
            "absorbed",
        ),
        (
            lambda frame: frame.assign(
                endogenous=frame.groupby("entity")["endogenous"].transform("mean")
            ),
            "absorbed",
        ),
    ],
)
def test_panel_and_variation_refusals(mutator, message) -> None:
    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        _fit_hand(mutator(_hand_panel()))


def test_role_names_must_be_unique_and_instruments_are_excluded_only() -> None:
    data = _hand_panel()
    with pytest.raises(ValueError, match="distinct"):
        PanelIV2SLS(time_effects=False).fit(
            data,
            outcome="outcome",
            endogenous="endogenous",
            instruments="instrument",
            exogenous="instrument",
            entity="entity",
            time="time",
        )
    with pytest.raises(ValueError, match="distinct"):
        PanelIV2SLS(time_effects=False).fit(
            data,
            outcome="outcome",
            endogenous=["endogenous", "endogenous"],
            instruments=["instrument", "time"],
            entity="entity",
            time="time",
        )


def test_underidentification_collinearity_and_cross_rank_refuse() -> None:
    data = _hand_panel()
    data["endogenous_2"] = data["endogenous"] ** 2
    with pytest.raises(ValueError, match="underidentified"):
        PanelIV2SLS(time_effects=False).fit(
            data,
            outcome="outcome",
            endogenous=["endogenous", "endogenous_2"],
            instruments="instrument",
            entity="entity",
            time="time",
        )

    data["instrument_copy"] = data["instrument"]
    with pytest.raises(ValueError, match="instrument design is rank deficient"):
        PanelIV2SLS(time_effects=False).fit(
            data,
            outcome="outcome",
            endogenous="endogenous",
            instruments=["instrument", "instrument_copy"],
            entity="entity",
            time="time",
        )

    data["irrelevant"] = np.tile([1.0, -1.0, -1.0, 1.0], 3)
    data["endogenous"] = 1.5 * data["instrument"] + data["entity"].map(
        {"a": 0.0, "b": 10.0, "c": 20.0}
    )
    with pytest.raises(ValueError, match="do not span"):
        PanelIV2SLS(time_effects=False).fit(
            data,
            outcome="outcome",
            endogenous="endogenous",
            instruments="irrelevant",
            entity="entity",
            time="time",
        )


def test_cluster_argument_contract_and_cluster_count_refusals() -> None:
    data = _hand_panel()
    data["one_cluster"] = "same"
    with pytest.raises(ValueError, match="only when covariance='clustered'"):
        PanelIV2SLS(covariance="robust", time_effects=False).fit(
            data,
            outcome="outcome",
            endogenous="endogenous",
            instruments="instrument",
            entity="entity",
            time="time",
            cluster="entity",
        )
    with pytest.raises(ValueError, match="at least two clusters"):
        PanelIV2SLS(time_effects=False).fit(
            data,
            outcome="outcome",
            endogenous="endogenous",
            instruments="instrument",
            entity="entity",
            time="time",
            cluster="one_cluster",
        )


def test_disconnected_two_way_panel_refuses_fixed_effect_normalization() -> None:
    data = _simulated_panel(n_entities=4, n_periods=4)
    disconnected = data.loc[
        ((data["entity"] < 2) & (data["time"] < 2)) | ((data["entity"] >= 2) & (data["time"] >= 2))
    ].copy()
    with pytest.raises(ValueError, match="connected"):
        PanelIV2SLS(time_effects=True).fit(
            disconnected,
            outcome="outcome",
            endogenous="endogenous",
            instruments="instrument",
            exogenous="control",
            entity="entity",
            time="time",
        )


def test_result_summary_confidence_and_markdown_are_coherent() -> None:
    result = _fit_hand()
    summary = result.summary_frame()
    interval = result.conf_int()
    assert summary.index.equals(result.params.index)
    assert interval.index.equals(result.params.index)
    assert summary.attrs["statistic_distribution"] == "t"
    assert "Panel IV/2SLS result" in result.to_markdown()
    assert "fixed effects" in result.causal_interpretation.lower()
    pd.testing.assert_frame_equal(vcov(result), result.covariance)
    pd.testing.assert_frame_equal(confint(result), result.conf_int())
    pd.testing.assert_series_equal(residuals(result), result.residuals)
    pd.testing.assert_series_equal(fitted_values(result), result.fitted_values)
    assert lincom(result, {"endogenous": 1.0})["estimate"] == pytest.approx(2.0)
    assert wald_test(result, {"endogenous": 1.0})["df_num"] == 1


def test_overidentification_is_only_reported_for_unadjusted_covariance() -> None:
    data = _simulated_panel(n_entities=30, n_periods=5)
    rng = np.random.default_rng(114)
    data["instrument_2"] = data["instrument"] + rng.normal(size=len(data))
    unadjusted = PanelIV2SLS(covariance="unadjusted").fit(
        data,
        outcome="outcome",
        endogenous="endogenous",
        instruments=["instrument", "instrument_2"],
        exogenous="control",
        entity="entity",
        time="time",
    )
    clustered = PanelIV2SLS().fit(
        data,
        outcome="outcome",
        endogenous="endogenous",
        instruments=["instrument", "instrument_2"],
        exogenous="control",
        entity="entity",
        time="time",
    )

    assert unadjusted.overidentification is not None
    assert unadjusted.overidentification.df == 1
    assert clustered.overidentification is None
    assert any("Sargan" in note and "not reported" in note for note in clustered.notes)


def test_non_dataframe_input_refuses_ambiguous_panel_roles() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        PanelIV2SLS().fit(
            np.ones((10, 5)),
            outcome="outcome",
            endogenous="endogenous",
            instruments="instrument",
            entity="entity",
            time="time",
        )

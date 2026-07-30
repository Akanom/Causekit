"""Contract tests for the optional Universal Output Hub adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import (
    AIPWATE,
    IV2SLS,
    DifferenceInDifferences,
    DRLearner,
    NearestNeighborMatch,
    PanelIV2SLS,
    PartiallyLinearDML,
    RandomizedATE,
    RegressionDiscontinuity,
    RepeatedCrossSectionDiD,
    RepeatedCrossSectionSurveyDesign,
    RLearner,
    add_to_outputhub,
    to_outputhub_model,
)

outputhub = pytest.importorskip("universal_output_hub")


@pytest.fixture(scope="module")
def iv_result():
    rng = np.random.default_rng(20_260_722)
    nobs = 240
    z1 = rng.normal(size=nobs)
    z2 = rng.normal(size=nobs)
    control = rng.normal(size=nobs)
    first_stage_error = rng.normal(size=nobs)
    treatment = z1 - 0.4 * z2 + 0.3 * control + first_stage_error
    outcome = 1.2 + 1.8 * treatment + 0.5 * control + 0.6 * first_stage_error
    index = pd.RangeIndex(nobs)
    return IV2SLS(covariance="unadjusted").fit(
        pd.Series(outcome, index=index, name="outcome"),
        endogenous=pd.DataFrame({"treatment": treatment}, index=index),
        instruments=pd.DataFrame({"z1": z1, "z2": z2}, index=index),
        exogenous=pd.DataFrame({"control": control}, index=index),
    )


def test_result_converts_to_outputhub_without_reestimating(iv_result) -> None:
    model = to_outputhub_model(iv_result, name="Main IV")

    assert isinstance(model, outputhub.RegressionModel)
    assert model.name == "Main IV"
    assert model.depvar == "outcome"
    pd.testing.assert_series_equal(model.params, iv_result.params.rename("coef"))
    pd.testing.assert_series_equal(model.std_errors, iv_result.standard_errors.rename("se"))
    assert model.metadata["estimator"] == "iv_2sls"
    assert model.metadata["causal_interpretation_requires_assumptions"] is True
    assert "First-stage F (treatment)" in model.diagnostics
    assert "Sargan statistic" in model.diagnostics


def test_result_and_first_stage_can_be_added_to_outputhub(iv_result) -> None:
    hub = outputhub.OutputHub("IV analysis")

    model = add_to_outputhub(hub, iv_result, name="Main IV")

    assert model.name == "Main IV"
    assert len(hub.models) == 1
    assert len(hub.tables) == 1
    assert hub.tables[0].name == "Main IV first-stage diagnostics"


def test_panel_iv_exports_design_variation_and_first_stage_without_refitting() -> None:
    rng = np.random.default_rng(891)
    rows = []
    for entity in range(12):
        alpha = rng.normal()
        for period in range(4):
            instrument = rng.normal()
            control = rng.normal()
            first_error = rng.normal(scale=0.5)
            endogenous = instrument + 0.3 * control + first_error
            outcome = 1.6 * endogenous - 0.4 * control + alpha + 0.2 * period + first_error
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
    result = PanelIV2SLS().fit(
        pd.DataFrame(rows),
        outcome="outcome",
        endogenous="endogenous",
        instruments="instrument",
        exogenous="control",
        entity="entity",
        time="time",
    )

    model = to_outputhub_model(result)
    assert model.name == "Panel IV/2SLS"
    assert model.metadata["estimator"] == "panel_iv_2sls"
    assert model.metadata["effects"] == ["entity", "time"]
    assert model.statistics["Entities"] == 12
    hub = outputhub.OutputHub("Panel instrument design")
    add_to_outputhub(hub, result)
    assert [table.name for table in hub.tables] == [
        "Panel IV/2SLS first-stage diagnostics",
        "Panel IV/2SLS instrument variation",
        "Panel IV/2SLS panel design",
    ]


def test_outputhub_adapter_validates_inputs(iv_result) -> None:
    with pytest.raises(TypeError, match="ObservationalATEResult"):
        to_outputhub_model(object())
    with pytest.raises(TypeError, match="add_model"):
        add_to_outputhub(object(), iv_result)


def test_randomized_ate_converts_and_adds_balance_table() -> None:
    rng = np.random.default_rng(73)
    nobs = 200
    treatment = rng.binomial(1, 0.5, nobs)
    baseline = rng.normal(size=nobs)
    outcome = 1.5 * treatment + baseline + rng.normal(size=nobs)
    result = RandomizedATE(adjustment="lin").fit(
        outcome, treatment=treatment, covariates=pd.DataFrame({"baseline": baseline})
    )
    model = to_outputhub_model(result)
    assert model.metadata["estimator"] == "randomized_ate"
    assert model.params.index.tolist() == ["ate"]
    hub = outputhub.OutputHub("Experiment")
    add_to_outputhub(hub, result)
    assert len(hub.models) == 1
    assert len(hub.tables) == 1


def test_observational_ate_converts_without_reestimating_nuisance_models() -> None:
    rng = np.random.default_rng(82)
    treatment = rng.binomial(1, 0.5, 300)
    mu0 = rng.normal(size=300)
    mu1 = mu0 + 1.2
    outcome = np.where(treatment == 1, mu1, mu0) + rng.normal(size=300)
    result = AIPWATE().fit(
        outcome,
        treatment=treatment,
        propensity=np.full(300, 0.5),
        outcome_treated=mu1,
        outcome_control=mu0,
    )
    model = to_outputhub_model(result)
    assert model.metadata["estimator"] == "aipw_ate"
    assert model.metadata["estimand"] == "ate"
    assert model.metadata["nuisance_predictions_supplied"] is True
    assert model.params.index.tolist() == ["ate"]


def test_partially_linear_dml_converts_without_reestimating_nuisance_models() -> None:
    rng = np.random.default_rng(104)
    covariates = pd.DataFrame(rng.normal(size=(180, 3)), columns=["a", "b", "c"])
    treatment = 0.4 * covariates["a"] + rng.normal(size=len(covariates))
    outcome = 1.6 * treatment + 0.5 * covariates["b"] + rng.normal(size=len(covariates))
    result = PartiallyLinearDML(n_splits=3, random_state=17).fit(
        outcome, treatment=treatment, covariates=covariates
    )

    model = to_outputhub_model(result)

    assert model.metadata["estimator"] == "partially_linear_dml2"
    assert model.metadata["native_nuisance"] is True
    assert model.metadata["nuisance_cross_fitted"] is True
    assert model.params.index.tolist() == ["theta"]
    hub = outputhub.OutputHub("Causal ML")
    add_to_outputhub(hub, result)
    assert len(hub.models) == 1
    assert len(hub.tables) == 1
    assert hub.tables[0].name == "Partially linear DML nuisance tuning"
    assert set(hub.tables[0].data["task"]) == {"outcome_mean", "treatment_mean"}


def test_honest_rlearner_exports_loss_calibration_groups_and_tuning_without_refit() -> None:
    rng = np.random.default_rng(731)
    nobs = 320
    covariates = pd.DataFrame(rng.normal(size=(nobs, 4)), columns=list("abcd"))
    propensity = 1.0 / (1.0 + np.exp(-(0.25 * covariates["a"] - 0.15 * covariates["b"])))
    treatment = pd.Series(rng.binomial(1, propensity), index=covariates.index)
    cate = 0.8 + 0.6 * covariates["a"]
    outcome = 0.4 * covariates["b"] + cate * treatment + rng.normal(scale=0.7, size=nobs)
    result = RLearner(
        n_splits=3,
        evaluation_fraction=0.4,
        random_state=73,
        calibration_groups=3,
        bootstrap_iterations=99,
        propensity_tuning_splits=2,
    ).fit(outcome, treatment=treatment, covariates=covariates)

    model = to_outputhub_model(result)

    assert model.metadata["estimator"] == "honest_r_learner"
    assert model.metadata["evaluation_used_for_fitting"] is False
    assert model.metadata["unit_level_intervals"] is False
    assert model.metadata["split_conditional"] is True
    assert model.params.index.tolist() == ["level", "heterogeneity"]
    hub = outputhub.OutputHub("Honest heterogeneous effects")
    add_to_outputhub(hub, result)
    assert len(hub.models) == 1
    assert [table.name for table in hub.tables] == [
        "Honest R-learner honest loss",
        "Honest R-learner calibration tests",
        "Honest R-learner calibration groups",
        "Honest R-learner nuisance tuning",
        "Honest R-learner CATE tuning",
    ]


def test_honest_drlearner_exports_score_loss_groups_and_tuning_without_refit() -> None:
    rng = np.random.default_rng(873)
    nobs = 360
    covariates = pd.DataFrame(rng.normal(size=(nobs, 3)), columns=list("abc"))
    propensity = 1.0 / (1.0 + np.exp(-(0.2 * covariates["a"] - 0.1 * covariates["b"])))
    treatment = pd.Series(rng.binomial(1, propensity), index=covariates.index)
    cate = 0.7 + 0.45 * covariates["a"]
    outcome = 0.3 * covariates["b"] + cate * treatment + rng.normal(scale=0.6, size=nobs)
    result = DRLearner(
        n_splits=3,
        evaluation_fraction=0.4,
        random_state=39,
        calibration_groups=3,
        bootstrap_iterations=99,
        propensity_tuning_splits=2,
    ).fit(outcome, treatment=treatment, covariates=covariates)

    model = to_outputhub_model(result)

    assert model.metadata["estimator"] == "honest_dr_learner"
    assert model.metadata["evaluation_used_for_fitting"] is False
    assert model.metadata["unit_level_intervals"] is False
    assert model.metadata["split_conditional"] is True
    assert model.params.index.tolist() == ["level", "heterogeneity"]
    hub = outputhub.OutputHub("Honest doubly robust effects")
    add_to_outputhub(hub, result)
    assert len(hub.models) == 1
    assert [table.name for table in hub.tables] == [
        "Honest DR learner honest loss",
        "Honest DR learner calibration tests",
        "Honest DR learner calibration groups",
        "Honest DR learner nuisance tuning",
        "Honest DR learner CATE tuning",
    ]


def test_did_converts_and_adds_auditable_effect_tables() -> None:
    rows = []
    for entity, cohort, level, effect in (
        ("t0", 2.0, 0.0, 2.0),
        ("t1", 2.0, 1.0, 2.0),
        ("c0", np.inf, 0.0, 0.0),
        ("c1", np.inf, 1.0, 0.0),
    ):
        rows.extend(
            [
                {
                    "entity": entity,
                    "time": 1,
                    "treatment_time": cohort,
                    "outcome": level,
                },
                {
                    "entity": entity,
                    "time": 2,
                    "treatment_time": cohort,
                    "outcome": level + 1.0 + effect,
                },
            ]
        )
    result = DifferenceInDifferences().fit(
        pd.DataFrame(rows),
        outcome="outcome",
        entity="entity",
        time="time",
        treatment_time="treatment_time",
    )
    model = to_outputhub_model(result)
    assert model.metadata["estimator"] == "conventional_group_time"
    assert model.metadata["parallel_trends"] == "post"
    assert model.metadata["inference_method"] == "analytic"
    assert model.metadata["nuisance_cross_fitted"] is False
    assert model.metadata["pretrend_available"] is False
    assert model.diagnostics["Pre-trend restrictions"] == 0
    assert model.params.index.tolist() == ["esavg"]
    hub = outputhub.OutputHub("DiD analysis")
    add_to_outputhub(hub, result)
    assert len(hub.models) == 1
    assert [table.name for table in hub.tables] == [
        "Difference-in-Differences group-time effects",
        "Difference-in-Differences event study",
        "Difference-in-Differences calendar-time effects",
    ]


def test_did_adds_pretrend_table_when_clean_placebos_exist() -> None:
    rows = []
    treated_paths = (
        (0.0, 1.0, 1.0, 4.0),
        (0.0, -1.0, -1.0, 2.0),
        (0.0, 0.0, 1.0, 4.0),
        (0.0, 0.0, -1.0, 2.0),
    )
    for position, path in enumerate(treated_paths):
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
    result = DifferenceInDifferences().fit(
        pd.DataFrame(rows),
        outcome="outcome",
        entity="entity",
        time="time",
        treatment_time="treatment_time",
    )
    hub = outputhub.OutputHub("DiD pre-trend analysis")
    add_to_outputhub(hub, result)

    assert result.pretrend.available
    assert hub.tables[0].name == "Difference-in-Differences pre-trend placebos"
    assert hub.tables[0].metadata["joint_test_available"] is True


def test_repeated_cross_section_did_exports_cells_and_effect_tables() -> None:
    data = pd.DataFrame(
        {
            "outcome": [1.0, 3.0, 6.0, 8.0, 2.0, 4.0, 3.0, 5.0],
            "time": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
            "treatment_time": [2.0, 2.0, 2.0, 2.0, np.inf, np.inf, np.inf, np.inf],
        }
    )
    result = RepeatedCrossSectionDiD(
        inference="multiplier_bootstrap", bootstrap_iterations=99, random_state=41
    ).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
    )

    model = to_outputhub_model(result)

    assert model.name == "Repeated-cross-section DiD"
    assert model.metadata["estimator"] == "repeated_cross_section_group_time"
    assert model.metadata["sampling_unit"] == "observation"
    assert model.metadata["composition"] == "stationary"
    assert model.metadata["composition_verified"] is False
    assert model.metadata["inference_method"] == "multiplier_bootstrap"
    assert model.metadata["simultaneous_level"] == 0.95
    assert model.metadata["bootstrap_iterations"] == 99
    assert model.metadata["bootstrap_random_state"] == 41
    assert model.statistics["Observations"] == 8
    hub = outputhub.OutputHub("Repeated samples")
    add_to_outputhub(hub, result)
    assert [table.name for table in hub.tables] == [
        "Repeated-cross-section DiD cell counts",
        "Repeated-cross-section DiD group-time effects",
        "Repeated-cross-section DiD event study",
        "Repeated-cross-section DiD simultaneous event-study bands",
        "Repeated-cross-section DiD calendar-time effects",
    ]


def test_survey_repeated_cross_section_exports_design_and_weight_audits() -> None:
    data = pd.DataFrame(
        {
            "outcome": [5, 7, 6, 8, 9, 12, 10, 13, 3, 4, 2, 5, 4, 7, 3, 6],
            "time": np.repeat([1.0, 2.0, 1.0, 2.0], 4),
            "treatment_time": np.repeat([2.0, 2.0, np.inf, np.inf], 4),
            "weight": [1, 2, 1, 3, 2, 1, 3, 1, 1, 2, 2, 1, 2, 1, 1, 2],
            "psu": np.tile(["a1", "a2", "b1", "b2"], 4),
            "stratum": np.tile(["a", "a", "b", "b"], 4),
        }
    )
    result = RepeatedCrossSectionDiD().fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        survey_design=RepeatedCrossSectionSurveyDesign(
            weights="weight", psu="psu", strata="stratum"
        ),
        target_population="survey_population",
    )

    model = to_outputhub_model(result)
    assert model.metadata["population_basis"] == "survey_population"
    assert model.metadata["weight_type"] == "inverse_inclusion"
    hub = outputhub.OutputHub("Survey repeated samples")
    add_to_outputhub(hub, result)
    names = [table.name for table in hub.tables]
    assert names[:4] == [
        "Repeated-cross-section DiD survey-weight diagnostics",
        "Repeated-cross-section DiD survey-design diagnostics",
        "Repeated-cross-section DiD survey-weighted cell counts",
        "Repeated-cross-section DiD cell counts",
    ]


def test_matching_converts_and_adds_design_tables_without_reestimating() -> None:
    logits = np.array([0.0, 4.0, 10.0, 1.0, 6.0, 9.0])
    result = NearestNeighborMatch(
        estimand="att",
        caliper=None,
        common_support=None,
        inference="abadie_imbens",
    ).fit(
        [0.0, 2.0, 5.0, 3.0, 8.0, 12.0],
        treatment=[0, 0, 0, 1, 1, 1],
        propensity=1 / (1 + np.exp(-logits)),
        covariates=pd.DataFrame({"baseline": logits}),
        propensity_score_status="known",
        propensity_provenance="fixed_by_test_design",
    )

    model = to_outputhub_model(result)

    assert model.metadata["estimator"] == "nearest_neighbor_match"
    assert model.metadata["requested_estimand"] == "att"
    assert model.metadata["realized_estimand"] == "att"
    assert model.metadata["propensity_score_status"] == "known"
    assert model.metadata["inference"] == "abadie_imbens"
    assert model.metadata["first_step_covariance_neighbors"] == 2
    assert model.metadata["propensity_model"] is None
    assert model.diagnostics["Known-score variance"] == result.variance
    assert np.isnan(model.diagnostics["First-step variance adjustment"])
    pd.testing.assert_series_equal(model.params, result.params.rename("coef"))
    pd.testing.assert_series_equal(model.std_errors, result.standard_errors.rename("se"))
    hub = outputhub.OutputHub("Matching analysis")
    add_to_outputhub(hub, result)
    assert len(hub.models) == 1
    assert [table.name for table in hub.tables] == [
        "Nearest-neighbor matching matches",
        "Nearest-neighbor matching balance",
    ]


def test_regression_discontinuity_exports_contract_and_diagnostics_without_refit() -> None:
    running = pd.Series(np.r_[-np.linspace(0.05, 1.0, 40), np.linspace(0.05, 1.0, 40)])
    outcome = pd.Series(1.0 + 0.5 * running + 2.0 * (running >= 0.0))
    result = RegressionDiscontinuity(bandwidth=0.9, bias_bandwidth=1.0).fit(
        outcome, running=running
    )

    model = to_outputhub_model(result)

    assert model.name == "Sharp RD"
    assert model.metadata["estimator"] == "local_polynomial_regression_discontinuity"
    assert model.metadata["design"] == "sharp"
    assert model.metadata["estimand"] == "cutoff_average_treatment_effect"
    assert model.metadata["primary_inference"] == "robust_bias_corrected"
    assert model.params.index.tolist() == ["rd_effect"]
    hub = outputhub.OutputHub("Cutoff design")
    add_to_outputhub(hub, result)
    assert [table.name for table in hub.tables] == [
        "Sharp RD bandwidth selection",
        "Sharp RD manipulation diagnostic",
    ]

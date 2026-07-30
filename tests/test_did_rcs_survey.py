"""Contracts for the survey-population repeated-section DiD slice."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from causekit import CrossFitter, RepeatedCrossSectionDiD, RepeatedCrossSectionSurveyDesign


class _WeightedConstantPropensity:
    def fit(self, X, y, *, sample_weight):
        self.probability_ = float(np.average(y, weights=sample_weight))
        return self

    def predict_proba(self, X):
        probability = np.full(len(X), self.probability_)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _WeightedLinearOutcome:
    def fit(self, X, y, *, sample_weight):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        root_weight = np.sqrt(np.asarray(sample_weight, dtype=float))
        self.coefficients_ = np.linalg.lstsq(
            design * root_weight[:, None], np.asarray(y) * root_weight, rcond=None
        )[0]
        return self

    def predict(self, X):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        return pd.Series(design @ self.coefficients_, index=X.index)


class _UnweightedLinearOutcome:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _survey_fixture(*, weight_scale: float = 1.0) -> pd.DataFrame:
    outcomes = {
        "treated_base": (5.0, 7.0, 6.0, 8.0),
        "treated_target": (9.0, 12.0, 10.0, 13.0),
        "control_base": (3.0, 4.0, 2.0, 5.0),
        "control_target": (4.0, 7.0, 3.0, 6.0),
    }
    weights = {
        "treated_base": (1.0, 2.0, 1.0, 3.0),
        "treated_target": (2.0, 1.0, 3.0, 1.0),
        "control_base": (1.0, 2.0, 2.0, 1.0),
        "control_target": (2.0, 1.0, 1.0, 2.0),
    }
    psus = ("a1", "a2", "b1", "b2")
    strata = ("a", "a", "b", "b")
    rows: list[dict[str, object]] = []
    for cell, cell_outcomes in outcomes.items():
        treated = cell.startswith("treated")
        target = cell.endswith("target")
        for position, (psu, stratum) in enumerate(zip(psus, strata, strict=True)):
            rows.append(
                {
                    "row": f"{cell}_{psu}",
                    "outcome": cell_outcomes[position],
                    "time": 2.0 if target else 1.0,
                    "treatment_time": 2.0 if treated else np.inf,
                    "survey_weight": weight_scale * weights[cell][position],
                    "psu": psu,
                    "stratum": stratum,
                }
            )
    frame = pd.DataFrame(rows).set_index("row")
    frame["x"] = np.tile([-1.0, 0.0, 1.0, 2.0], 4)
    return frame


def _design() -> RepeatedCrossSectionSurveyDesign:
    return RepeatedCrossSectionSurveyDesign(
        weights="survey_weight",
        psu="psu",
        strata="stratum",
        weight_type="inverse_inclusion",
        singleton_psu="raise",
    )


def _fit(data: pd.DataFrame):
    return RepeatedCrossSectionDiD().fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        survey_design=_design(),
        target_population="survey_population",
    )


def _hand_cell(
    data: pd.DataFrame,
    mask: pd.Series,
) -> tuple[float, np.ndarray]:
    weights = data["survey_weight"].to_numpy(dtype=float)
    outcome = data["outcome"].to_numpy(dtype=float)
    selected = mask.to_numpy(dtype=bool)
    denominator = float(weights[selected].sum())
    estimate = float(np.dot(weights[selected], outcome[selected]) / denominator)
    linearized = weights * selected * (outcome - estimate) / denominator
    return estimate, linearized


def _hand_att(data: pd.DataFrame) -> tuple[float, np.ndarray]:
    treated = data["treatment_time"].eq(2.0)
    target = data["time"].eq(2.0)
    means_and_scores = (
        _hand_cell(data, treated & target),
        _hand_cell(data, treated & ~target),
        _hand_cell(data, ~treated & target),
        _hand_cell(data, ~treated & ~target),
    )
    signs = (1.0, -1.0, -1.0, 1.0)
    estimate = sum(sign * item[0] for sign, item in zip(signs, means_and_scores, strict=True))
    linearized = sum(
        (sign * item[1] for sign, item in zip(signs, means_and_scores, strict=True)),
        start=np.zeros(len(data)),
    )
    return float(estimate), linearized


def _hand_taylor_covariance(data: pd.DataFrame, linearized: np.ndarray) -> np.ndarray:
    matrix = np.asarray(linearized, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    covariance = np.zeros((matrix.shape[1], matrix.shape[1]))
    frame = pd.DataFrame(matrix, index=data.index)
    frame["stratum"] = data["stratum"]
    frame["psu"] = data["psu"]
    for _, stratum in frame.groupby("stratum", sort=False):
        psu_totals = stratum.groupby("psu", sort=False).sum(numeric_only=True).to_numpy()
        centered = psu_totals - psu_totals.mean(axis=0)
        covariance += len(psu_totals) / (len(psu_totals) - 1.0) * centered.T @ centered
    return covariance


def test_survey_design_is_public_immutable_and_semantically_explicit() -> None:
    design = _design()

    assert design.weight_type == "inverse_inclusion"
    assert design.singleton_psu == "raise"
    with pytest.raises(FrozenInstanceError):
        design.weight_type = "calibrated_analysis"  # type: ignore[misc]


def test_hand_weighted_two_by_two_att_linearization_and_taylor_variance() -> None:
    data = _survey_fixture()
    expected_att, expected_linearized = _hand_att(data)
    expected_covariance = _hand_taylor_covariance(data, expected_linearized)

    result = _fit(data)

    assert result.estimate == pytest.approx(expected_att, abs=1e-14)
    np.testing.assert_allclose(
        result.overall_influence.to_numpy(dtype=float),
        len(data) * expected_linearized,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.survey_linearized["esavg"].to_numpy(dtype=float),
        expected_linearized,
        rtol=0.0,
        atol=1e-14,
    )
    assert result.standard_error**2 == pytest.approx(expected_covariance[0, 0], abs=1e-14)
    assert result.inference_distribution == "t"
    assert result.inference_df == 2.0
    assert result.covariance_type == "survey_taylor"
    assert result.population_basis == "survey_population"
    assert result.target_population == "survey_treated_target_period"
    assert result.sampling_unit == "survey_psu"
    assert result.n_clusters == 4
    assert result.survey_design_diagnostics["n_strata"] == 2
    assert result.survey_design_diagnostics["n_psus"] == 4
    assert result.survey_design_diagnostics["design_df"] == 2
    assert result.weight_normalization == "component_hajek"
    assert result.weight_type == "inverse_inclusion"


def test_positive_global_weight_rescaling_and_row_permutation_are_exact_invariances() -> None:
    data = _survey_fixture()
    reference = _fit(data)
    rescaled = _fit(_survey_fixture(weight_scale=37.5))
    permuted = _fit(data.sample(frac=1.0, random_state=91))

    assert rescaled.estimate == pytest.approx(reference.estimate, abs=1e-14)
    assert rescaled.standard_error == pytest.approx(reference.standard_error, abs=1e-14)
    pd.testing.assert_frame_equal(reference.group_time, rescaled.group_time)
    pd.testing.assert_frame_equal(
        reference.weighted_cell_counts.drop(columns="weight_sum"),
        rescaled.weighted_cell_counts.drop(columns="weight_sum"),
    )
    pd.testing.assert_series_equal(
        reference.overall_influence.sort_index(),
        rescaled.overall_influence.sort_index(),
        atol=1e-14,
    )
    pd.testing.assert_frame_equal(reference.group_time, permuted.group_time)
    pd.testing.assert_series_equal(
        reference.overall_influence.sort_index(),
        permuted.overall_influence.sort_index(),
        atol=1e-14,
    )
    assert reference.design_fingerprint == permuted.design_fingerprint


def test_informative_weights_change_the_population_target_in_expected_direction() -> None:
    data = _survey_fixture()
    survey = _fit(data)
    sample = RepeatedCrossSectionDiD().fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
    )
    expected, _ = _hand_att(data)

    assert survey.estimate == pytest.approx(expected, abs=1e-14)
    assert survey.estimate != pytest.approx(sample.estimate, abs=1e-6)
    assert survey.population_basis == "survey_population"
    assert sample.population_basis == "sample"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.assign(survey_weight=0.0), "strictly positive"),
        (
            lambda frame: frame.assign(
                survey_weight=lambda value: value["survey_weight"].mask(
                    value.index == value.index[0], -1.0
                )
            ),
            "strictly positive",
        ),
        (
            lambda frame: frame.assign(
                survey_weight=lambda value: value["survey_weight"].mask(
                    value.index == value.index[0], np.nan
                )
            ),
            "missing",
        ),
        (lambda frame: frame.assign(survey_weight="bad"), "numeric"),
        (lambda frame: frame.assign(survey_weight=True), "numeric"),
        (lambda frame: frame.assign(survey_weight=1.0 + 1.0j), "numeric"),
        (
            lambda frame: frame.assign(
                survey_weight=lambda value: value["survey_weight"].mask(
                    value.index == value.index[0], np.inf
                )
            ),
            "finite",
        ),
        (
            lambda frame: frame.assign(
                stratum=lambda value: np.where(value["psu"].eq("a1"), "lonely", "other")
            ),
            "at least two PSUs",
        ),
        (
            lambda frame: frame.assign(
                psu=lambda value: np.where(value["psu"].isin(["a1", "b1"]), "shared", value["psu"])
            ),
            "nested within one stratum",
        ),
    ],
)
def test_survey_design_refuses_invalid_weights_and_psu_strata_roles(mutation, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _fit(mutation(_survey_fixture()))


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        (
            {
                "weights": "survey_weight",
                "psu": "psu",
                "strata": "stratum",
                "weight_type": "frequency",
            },
            "weight_type",
        ),
        (
            {
                "weights": "survey_weight",
                "psu": "psu",
                "strata": "stratum",
                "singleton_psu": "adjust",
            },
            "singleton_psu",
        ),
    ],
)
def test_survey_design_refuses_unknown_semantics_and_singleton_adjustment(
    keywords: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RepeatedCrossSectionSurveyDesign(**keywords)


def test_survey_design_refuses_index_drift_and_unsupported_combinations() -> None:
    data = _survey_fixture()
    drifted_weights = data["survey_weight"].sample(frac=1.0, random_state=2)
    design = RepeatedCrossSectionSurveyDesign(
        weights=drifted_weights,
        psu="psu",
        strata="stratum",
    )
    with pytest.raises(ValueError, match="index"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            survey_design=design,
            target_population="survey_population",
        )
    drifted_psu = data["psu"].sample(frac=1.0, random_state=3)
    with pytest.raises(ValueError, match="index"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            survey_design=RepeatedCrossSectionSurveyDesign(
                weights="survey_weight", psu=drifted_psu, strata="stratum"
            ),
            target_population="survey_population",
        )
    with pytest.raises(ValueError, match="target_population"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            survey_design=_design(),
        )
    with pytest.raises(ValueError, match="survey_design"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            target_population="survey_population",
        )
    with pytest.raises(NotImplementedError, match="composition"):
        RepeatedCrossSectionDiD(composition="robust").fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=["survey_weight"],
            cross_fitter=object(),  # type: ignore[arg-type]
            survey_design=_design(),
            target_population="survey_population",
        )
    with pytest.raises(TypeError, match="CrossFitter"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=["survey_weight"],
            cross_fitter=object(),  # type: ignore[arg-type]
            survey_design=_design(),
            target_population="survey_population",
        )
    with pytest.raises(NotImplementedError, match="simultaneous"):
        RepeatedCrossSectionDiD(inference="multiplier_bootstrap").fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            survey_design=_design(),
            target_population="survey_population",
        )
    with pytest.raises(ValueError, match="cluster"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            cluster="psu",
            survey_design=_design(),
            target_population="survey_population",
        )


def test_bare_sampling_weights_continue_to_refuse() -> None:
    with pytest.raises(NotImplementedError, match="bare sampling_weights"):
        RepeatedCrossSectionDiD().fit(
            _survey_fixture(),
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            sampling_weights="survey_weight",
        )


def _hand_weighted_ratio(
    survey_weight: np.ndarray, score_weight: np.ndarray, value: np.ndarray
) -> tuple[float, np.ndarray]:
    denominator = float(np.dot(survey_weight, score_weight))
    estimate = float(np.dot(survey_weight * score_weight, value) / denominator)
    linearized = survey_weight * score_weight * (value - estimate) / denominator
    return estimate, linearized


def test_covariate_survey_score_uses_weighted_crossfitter_and_exact_linearization() -> None:
    data = _survey_fixture()
    result = RepeatedCrossSectionDiD().fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=CrossFitter(
            propensity_factory=_WeightedConstantPropensity,
            outcome_factory=_WeightedLinearOutcome,
            n_splits=2,
            random_state=29,
        ),
        survey_design=_design(),
        target_population="survey_population",
    )

    predictions = result.nuisance_predictions.xs((2.0, 2.0), axis=1, level=(0, 1))
    survey_weight = data["survey_weight"].to_numpy(dtype=float)
    treated = data["treatment_time"].eq(2.0).to_numpy(dtype=float)
    post = data["time"].eq(2.0).to_numpy(dtype=float)
    outcome = data["outcome"].to_numpy(dtype=float)
    propensity = predictions["propensity"].to_numpy(dtype=float)
    m0_pre = predictions["outcome_control_pre"].to_numpy(dtype=float)
    m0_post = predictions["outcome_control_post"].to_numpy(dtype=float)
    m1_pre = predictions["outcome_treated_pre"].to_numpy(dtype=float)
    m1_post = predictions["outcome_treated_post"].to_numpy(dtype=float)
    m0 = post * m0_post + (1.0 - post) * m0_pre
    odds = propensity / (1.0 - propensity)
    components = (
        (treated * post, outcome - m0, 1.0),
        (treated * (1.0 - post), outcome - m0, -1.0),
        (odds * (1.0 - treated) * post, outcome - m0, -1.0),
        (odds * (1.0 - treated) * (1.0 - post), outcome - m0, 1.0),
        (treated, m1_post - m0_post, 1.0),
        (treated * post, m1_post - m0_post, -1.0),
        (treated, m1_pre - m0_pre, -1.0),
        (treated * (1.0 - post), m1_pre - m0_pre, 1.0),
    )
    expected_estimate = 0.0
    expected_linearized = np.zeros(len(data))
    for score_weight, value, sign in components:
        component, linearized = _hand_weighted_ratio(survey_weight, score_weight, value)
        expected_estimate += sign * component
        expected_linearized += sign * linearized

    assert result.estimate == pytest.approx(expected_estimate, abs=1e-12)
    np.testing.assert_allclose(
        result.survey_linearized["esavg"], expected_linearized, rtol=0.0, atol=1e-12
    )
    expected_variance = _hand_taylor_covariance(data, expected_linearized)[0, 0]
    assert result.standard_error**2 == pytest.approx(expected_variance, abs=1e-12)
    assert result.cross_fitted
    assert result.nuisance_diagnostics["weight_hash"].notna().all()
    assert result.nuisance_diagnostics["weight_sum"].gt(0.0).all()
    assert result.nuisance_fold.groupby(data["psu"]).nunique().eq(1).all()


def test_covariate_survey_path_refuses_unweighted_provider_and_concentrated_support() -> None:
    data = _survey_fixture()
    with pytest.raises(TypeError, match="sample_weight"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=["x"],
            cross_fitter=CrossFitter(
                propensity_factory=_WeightedConstantPropensity,
                outcome_factory=_UnweightedLinearOutcome,
                n_splits=2,
                random_state=29,
            ),
            survey_design=_design(),
            target_population="survey_population",
        )
    concentrated = data.copy()
    concentrated.loc["treated_target_a1", "survey_weight"] = 1000.0
    with pytest.raises(ValueError, match="weighted support|weight concentration"):
        _fit(concentrated)

    one_psu_cell = data.copy()
    treated_target = one_psu_cell.index.str.startswith("treated_target")
    one_psu_cell.loc[treated_target, "psu"] = "a1"
    one_psu_cell.loc[treated_target, "stratum"] = "a"
    with pytest.raises(ValueError, match="at least two PSUs"):
        _fit(one_psu_cell)

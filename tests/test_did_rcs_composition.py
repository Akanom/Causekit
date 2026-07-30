"""Failing-first contracts for composition-robust repeated-section DiD."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2
from scipy.stats import f as f_distribution

from causekit import (
    CrossFitter,
    RepeatedCrossSectionCompositionDiagnostic,
    RepeatedCrossSectionDiD,
    add_to_outputhub,
    did_rcs_composition_test,
    to_outputhub_model,
)


class _KnownGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X["x"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                0: 0.40 - 0.03 * x,
                1: 0.20 + 0.02 * x,
                2: 0.25 - 0.01 * x,
                3: 0.15 + 0.02 * x,
            },
            index=X.index,
        )


class _KnownGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _KnownGeneralizedPropensityResult:
        del X, y
        return _KnownGeneralizedPropensityResult()


class _KnownCellOutcomeResult:
    def __init__(self, cell: str) -> None:
        self.cell = cell

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X["x"].to_numpy(dtype=float)
        if self.cell == "d0-s0":
            return 1.0 + 0.50 * x
        if self.cell == "d0-s1":
            return 2.0 + x
        if self.cell == "d1-s0":
            return 3.0 + 0.25 * x
        raise AssertionError(f"Unexpected outcome cell {self.cell!r}.")


class _KnownCellOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _KnownCellOutcomeResult:
        del y
        cells = {str(label).rsplit("-r", 1)[0] for label in X.index}
        if len(cells) != 1:
            raise AssertionError("Each robust outcome nuisance must train within one cell.")
        return _KnownCellOutcomeResult(cells.pop())


class _CompositionShiftGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X["x"].to_numpy(dtype=float)
        low_to_high = 0.125 + 0.25 * x
        high_to_low = 0.375 - 0.25 * x
        return pd.DataFrame(
            {
                0: high_to_low,
                1: low_to_high,
                2: high_to_low,
                3: low_to_high,
            },
            index=X.index,
        )


class _CompositionShiftGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _CompositionShiftGeneralizedPropensityResult:
        del X, y
        return _CompositionShiftGeneralizedPropensityResult()


class _CompositionShiftBinaryPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probability = np.full(len(X), 0.5)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _CompositionShiftBinaryPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _CompositionShiftBinaryPropensityResult:
        del X, y
        return _CompositionShiftBinaryPropensityResult()


class _CompositionShiftOutcomeResult:
    def __init__(self, cell: str) -> None:
        self.cell = cell

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X["x"].to_numpy(dtype=float)
        if self.cell == "d0-s0":
            return 1.0 + x
        if self.cell == "d0-s1":
            return 2.0 + 3.0 * x
        if self.cell == "d1-s0":
            return 4.0 + 2.0 * x
        if self.cell == "d1-s1":
            return 7.0 + 8.0 * x
        raise AssertionError(f"Unexpected outcome cell {self.cell!r}.")


class _CompositionShiftOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _CompositionShiftOutcomeResult:
        del y
        cells = {"-".join(str(label).split("-")[:2]) for label in X.index}
        if len(cells) != 1:
            raise AssertionError("Each outcome nuisance must train within one group-period cell.")
        return _CompositionShiftOutcomeResult(cells.pop())


def _cross_fitter(
    *,
    propensity_factory: Callable[[], object] = _KnownGeneralizedPropensity,
    outcome_factory: Callable[[], object] = _KnownCellOutcome,
    random_state: int = 19,
) -> CrossFitter:
    return CrossFitter(
        propensity_factory=propensity_factory,
        outcome_factory=outcome_factory,
        n_splits=2,
        random_state=random_state,
    )


def _composition_hand_sample() -> pd.DataFrame:
    x = np.tile(np.arange(4.0), 4)
    treated = np.repeat([0.0, 0.0, 1.0, 1.0], 4)
    post = np.repeat([0.0, 1.0, 0.0, 1.0], 4)
    m00 = 1.0 + 0.50 * x
    m01 = 2.0 + x
    m10 = 3.0 + 0.25 * x
    residual = np.array(
        [
            -0.4,
            0.2,
            0.1,
            0.3,
            0.3,
            -0.2,
            0.5,
            -0.1,
            -0.2,
            0.4,
            -0.3,
            0.1,
            0.1,
            -0.2,
            0.3,
            -0.1,
        ]
    )
    treatment_effect = 2.0 + 0.20 * x
    outcome = np.where(
        (treated == 0.0) & (post == 0.0),
        m00 + residual,
        np.where(
            (treated == 0.0) & (post == 1.0),
            m01 + residual,
            np.where(
                (treated == 1.0) & (post == 0.0),
                m10 + residual,
                m10 + m01 - m00 + treatment_effect + residual,
            ),
        ),
    )
    labels = [
        f"d{int(treated[position])}-s{int(post[position])}-r{position % 4}"
        for position in range(len(x))
    ]
    return pd.DataFrame(
        {
            "outcome": outcome,
            "time": post + 1.0,
            "treatment_time": np.where(treated == 1.0, 2.0, np.inf),
            "x": x,
        },
        index=pd.Index(labels, name="row"),
    )


def _composition_shift_sample() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    cell_x_counts = {
        (0, 0): (6, 2),
        (0, 1): (2, 6),
        (1, 0): (6, 2),
        (1, 1): (2, 6),
    }
    for (treated, post), counts in cell_x_counts.items():
        x_values = [0.0] * counts[0] + [1.0] * counts[1]
        for replicate, x in enumerate(x_values):
            m00 = 1.0 + x
            m01 = 2.0 + 3.0 * x
            m10 = 4.0 + 2.0 * x
            treatment_effect = 2.0 + 4.0 * x
            untreated_treated_post = m10 + m01 - m00
            if treated == 0 and post == 0:
                outcome = m00
            elif treated == 0 and post == 1:
                outcome = m01
            elif treated == 1 and post == 0:
                outcome = m10
            else:
                outcome = untreated_treated_post + treatment_effect
            rows.append(
                {
                    "row": f"d{treated}-s{post}-x{int(x)}-r{replicate}",
                    "outcome": outcome,
                    "time": float(post + 1),
                    "treatment_time": 2.0 if treated else np.inf,
                    "x": x,
                    "true_effect": treatment_effect,
                }
            )
    return pd.DataFrame(rows).set_index("row")


def _fit_robust(
    data: pd.DataFrame,
    *,
    cross_fitter: CrossFitter | None = None,
    **constructor: object,
):
    return RepeatedCrossSectionDiD(composition="robust", **constructor).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=cross_fitter or _cross_fitter(),
    )


def test_hand_computed_composition_robust_score_eif_hc1_and_weights() -> None:
    data = _composition_hand_sample()

    result = _fit_robust(data)

    assert result.estimate == pytest.approx(2.295038744545108, abs=2e-14)
    expected_influence = pd.Series(
        [
            -1.15902703606356,
            0.710034580651553,
            0.431794385984466,
            1.5702946940216,
            -1.15307512832478,
            0.792011199253384,
            -2.02855809612692,
            0.413924405039663,
            0.621567268423242,
            -1.4675893837771,
            1.28367153261322,
            -0.494428508973033,
            -0.780154978180434,
            -1.18015497818043,
            1.61984502181957,
            0.819845021819567,
        ],
        index=data.index,
        name="esavg_influence",
    )
    pd.testing.assert_series_equal(result.overall_influence, expected_influence, atol=2e-14)
    expected_se = np.sqrt(
        float(expected_influence @ expected_influence) / (len(data) * (len(data) - 1))
    )
    assert result.standard_error == pytest.approx(expected_se, abs=2e-14)

    expected_weights = pd.DataFrame(
        {
            "w_00": [
                2.89756759015891,
                3.55017290325776,
                4.31794385984465,
                5.23431564673868,
                *([0.0] * 12),
            ],
            "w_01": [
                *([0.0] * 4),
                3.84358376108259,
                3.96005599626692,
                4.05711619225385,
                4.13924405039664,
                *([0.0] * 8),
            ],
            "w_10": [
                *([0.0] * 8),
                3.10783634211621,
                3.66897345944274,
                4.27890510871072,
                4.94428508973033,
                *([0.0] * 4),
            ],
            "w_11": [*([0.0] * 12), 4.0, 4.0, 4.0, 4.0],
        },
        index=data.index,
    )
    pd.testing.assert_frame_equal(result.composition_weights, expected_weights, atol=2e-14)
    assert result.composition_weights.mean().eq(1.0).all()
    assert result.composition == "robust"
    assert result.target_population == "treated_target_period"
    assert result.method == "repeated_cross_section_composition_robust_group_time"
    assert result.parallel_trends == (
        "conditional_repeated_cross_section_post_composition_robust_target_period_treated"
    )
    assert result.pretrend.available is False
    assert result.pretrend.conditional is True
    assert result.nuisance_predictions.columns.get_level_values("nuisance").tolist() == [
        "generalized_propensity_00",
        "generalized_propensity_01",
        "generalized_propensity_10",
        "generalized_propensity_11",
        "outcome_control_pre",
        "outcome_control_post",
        "outcome_treated_pre",
    ]
    assert "outcome_treated_post" not in result.nuisance_predictions.columns.get_level_values(
        "nuisance"
    )
    assert len(result.nuisance_diagnostics) == 8
    assert result.nuisance_fold.index.equals(data.index)


def test_composition_shift_recovers_target_att_and_exposes_stationary_bias() -> None:
    data = _composition_shift_sample()
    robust = _fit_robust(
        data,
        cross_fitter=_cross_fitter(
            propensity_factory=_CompositionShiftGeneralizedPropensity,
            outcome_factory=_CompositionShiftOutcome,
        ),
    )
    stationary = RepeatedCrossSectionDiD(composition="stationary").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_cross_fitter(
            propensity_factory=_CompositionShiftBinaryPropensity,
            outcome_factory=_CompositionShiftOutcome,
        ),
    )

    treated = data["treatment_time"].eq(2.0)
    target = treated & data["time"].eq(2.0)
    baseline = treated & data["time"].eq(1.0)
    target_truth = float(data.loc[target, "true_effect"].mean())
    pooled_treated_truth = float(data.loc[treated, "true_effect"].mean())

    assert data.loc[target, "x"].mean() == pytest.approx(0.75)
    assert data.loc[baseline, "x"].mean() == pytest.approx(0.25)
    assert target_truth == pytest.approx(5.0)
    assert pooled_treated_truth == pytest.approx(4.0)
    assert robust.estimate == pytest.approx(target_truth, abs=1e-14)
    assert stationary.estimate == pytest.approx(pooled_treated_truth, abs=1e-14)
    assert stationary.estimate - target_truth == pytest.approx(-1.0, abs=1e-14)
    assert robust.target_population == "treated_target_period"
    assert stationary.target_population == "pooled_treated_stationary_composition"
    assert robust.overall_influence.mean() == pytest.approx(0.0, abs=1e-14)


def test_robust_composition_refuses_missing_covariates() -> None:
    data = _composition_hand_sample()
    with pytest.raises(ValueError, match="composition='robust'.*covariates"):
        RepeatedCrossSectionDiD(composition="robust").fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
        )
    with pytest.raises(ValueError, match="at least one"):
        RepeatedCrossSectionDiD(composition="robust").fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=[],
            cross_fitter=_cross_fitter(),
        )


class _LongDesignOutcomeResult:
    def __init__(self, period: float) -> None:
        self.period = period

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.period + 0.2 * X["x"].to_numpy(dtype=float)


class _LongDesignOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _LongDesignOutcomeResult:
        del y
        periods = {float(str(label).split("-t")[1].split("-")[0]) for label in X.index}
        if len(periods) != 1:
            raise AssertionError("Each longer-design outcome task must train in one period.")
        return _LongDesignOutcomeResult(periods.pop())


class _EmpiricalClassProbabilityResult:
    def __init__(self, classes: np.ndarray, probabilities: np.ndarray) -> None:
        self.classes_ = classes
        self.probabilities = probabilities

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile(self.probabilities, (len(X), 1))


class _EmpiricalClassProbability:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _EmpiricalClassProbabilityResult:
        del X
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        return _EmpiricalClassProbabilityResult(classes, counts / counts.sum())


def _long_composition_sample(
    *,
    cohorts: tuple[float, ...] = (3.0,),
    periods: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    base_residuals = np.array((-0.30, -0.10, 0.10, 0.30))
    for cohort in (*cohorts, np.inf):
        cohort_label = "never" if np.isinf(cohort) else str(int(cohort))
        for period in periods:
            cohort_offset = 0 if np.isinf(cohort) else int(cohort)
            residuals = np.roll(base_residuals, int(period + cohort_offset) % 4)
            for replicate, residual in enumerate(residuals):
                x = float(replicate)
                untreated = period + 0.2 * x
                effect = (
                    1.0 + 0.5 * (period - cohort) + 0.4 * (cohort - 3.0) + 0.2 * x
                    if np.isfinite(cohort) and period >= cohort
                    else 0.0
                )
                rows.append(
                    {
                        "row": f"g{cohort_label}-t{int(period)}-r{replicate}",
                        "outcome": untreated + effect + residual,
                        "time": period,
                        "treatment_time": cohort,
                        "x": x,
                        "true_effect": effect,
                    }
                )
    return pd.DataFrame(rows).set_index("row")


def _long_cross_fitter(*, random_state: int = 37) -> CrossFitter:
    return CrossFitter(
        propensity_factory=_EmpiricalClassProbability,
        outcome_factory=_LongDesignOutcome,
        n_splits=2,
        random_state=random_state,
    )


def _pairwise_relabelled_fit(
    data: pd.DataFrame,
    *,
    cohort: float,
    base: float,
    target: float,
):
    pair = data.loc[
        data["time"].isin([base, target])
        & (data["treatment_time"].eq(cohort) | np.isinf(data["treatment_time"]))
    ].copy()
    pair.loc[pair["treatment_time"].eq(cohort), "treatment_time"] = target
    return RepeatedCrossSectionDiD(composition="robust").fit(
        pair,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_long_cross_fitter(),
    )


def test_longer_robust_lattice_embeds_pair_influences_and_conditional_placebo() -> None:
    data = _long_composition_sample()

    result = RepeatedCrossSectionDiD(composition="robust").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_long_cross_fitter(),
    )

    assert result.group_time.index.tolist() == [(3.0, 3.0), (3.0, 4.0)]
    assert result.group_time["att"].tolist() == pytest.approx([1.3, 1.8], abs=1e-14)
    assert result.pretrend.placebo_effects.index.tolist() == [(3.0, 2.0)]
    assert result.pretrend.placebo_effects.loc[(3.0, 2.0), "att"] == pytest.approx(0.0, abs=1e-14)
    assert result.pretrend.conditional
    assert result.pretrend.cross_fitted
    assert any("unmeasured composition changes" in note for note in result.pretrend.notes)
    assert all("Stationary composition" not in note for note in result.pretrend.notes)
    assert result.nuisance_fold.index.equals(data.index)
    assert len(result.nuisance_diagnostics) == 24
    assert result.nuisance_diagnostics["comparison_stage"].value_counts().to_dict() == {
        "group_time": 16,
        "conditional_pretrend": 8,
    }
    assert result.pair_ledger.index.tolist() == [(3.0, 2.0), (3.0, 3.0), (3.0, 4.0)]
    assert result.pair_ledger["stage"].tolist() == [
        "conditional_pretrend",
        "group_time",
        "group_time",
    ]
    assert result.pair_ledger["influence_embedding_scale"].eq(2.0).all()
    assert result.pair_ledger["nuisance_task_keys"].map(len).eq(4).all()

    n = len(data)
    for target in (3.0, 4.0):
        pair = _pairwise_relabelled_fit(data, cohort=3.0, base=2.0, target=target)
        relevant = pair.estimation_index
        expected = pd.Series(0.0, index=data.index)
        expected.loc[relevant] = n / len(relevant) * pair.overall_influence
        pd.testing.assert_series_equal(
            result.group_time_influence[(3.0, target)],
            expected.rename((3.0, target)),
            atol=2e-14,
        )
        assert (
            result.group_time_influence.loc[~data.index.isin(relevant), (3.0, target)].eq(0.0).all()
        )

    placebo_pair = _pairwise_relabelled_fit(data, cohort=3.0, base=1.0, target=2.0)
    expected_placebo = pd.Series(0.0, index=data.index)
    expected_placebo.loc[placebo_pair.estimation_index] = (
        n / len(placebo_pair.estimation_index) * placebo_pair.overall_influence
    )
    pd.testing.assert_series_equal(
        result.pretrend.influence[(3.0, 2.0)],
        expected_placebo.rename((3.0, 2.0)),
        atol=2e-14,
    )


@pytest.mark.parametrize("control_group", ["never_treated", "not_yet_treated"])
def test_staggered_robust_lattice_uses_one_global_fold_and_fixed_pair_membership(
    control_group: str,
) -> None:
    data = _long_composition_sample(
        cohorts=(3.0, 4.0),
        periods=(1.0, 2.0, 3.0, 4.0, 5.0),
    )

    result = RepeatedCrossSectionDiD(
        composition="robust",
        control_group=control_group,
    ).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_long_cross_fitter(),
    )

    assert result.group_time.index.tolist() == [
        (3.0, 3.0),
        (3.0, 4.0),
        (3.0, 5.0),
        (4.0, 4.0),
        (4.0, 5.0),
    ]
    assert result.pretrend.placebo_effects.index.tolist() == [
        (3.0, 2.0),
        (4.0, 2.0),
        (4.0, 3.0),
    ]
    assert result.nuisance_fold.index.equals(data.index)
    assert result.nuisance_diagnostics["fold"].nunique() == 2
    assert len(result.nuisance_diagnostics) == 8 * 4 * 2
    first_comparison = result.group_time.loc[(3.0, 3.0), "comparison_cohorts"]
    if control_group == "never_treated":
        assert first_comparison == (np.inf,)
    else:
        assert first_comparison == (4.0, np.inf)
    assert result.group_time.loc[(3.0, 4.0), "comparison_cohorts"] == (np.inf,)


def test_robust_event_and_calendar_aggregation_use_target_shares_with_share_influence() -> None:
    data = _long_composition_sample(
        cohorts=(3.0, 4.0),
        periods=(1.0, 2.0, 3.0, 4.0, 5.0),
    )
    result = RepeatedCrossSectionDiD(composition="robust").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_long_cross_fitter(),
    )

    cells = [(3.0, 3.0), (4.0, 4.0)]
    estimates = result.group_time.loc[cells, "att"].to_numpy(dtype=float)
    assert estimates.tolist() == pytest.approx([1.3, 1.7], abs=1e-14)
    shares = result.group_time.loc[cells, "target_share"].to_numpy(dtype=float)
    assert shares.tolist() == pytest.approx([4 / len(data), 4 / len(data)])
    weights = shares / shares.sum()
    expected_estimate = float(weights @ estimates)
    assert result.event_study.loc[0, "att"] == pytest.approx(expected_estimate, abs=1e-14)
    expected_influence = np.zeros(len(data))
    for weight, share, cell, estimate in zip(weights, shares, cells, estimates, strict=True):
        membership = data["treatment_time"].eq(cell[0]) & data["time"].eq(cell[1])
        expected_influence += weight * result.group_time_influence[cell].to_numpy(dtype=float)
        expected_influence += (
            (estimate - expected_estimate)
            * (membership.to_numpy(dtype=float) - share)
            / shares.sum()
        )
    np.testing.assert_allclose(
        result.event_study_influence[0], expected_influence, rtol=0.0, atol=2e-14
    )
    assert result.group_time_aggregation == "estimated_target_period_treated_cell_shares"
    assert result.esavg_aggregation == "arithmetic_mean_of_nonnegative_event_times"

    calendar_cells = [(3.0, 4.0), (4.0, 4.0)]
    calendar_shares = result.group_time.loc[calendar_cells, "target_share"].to_numpy(dtype=float)
    calendar_estimates = result.group_time.loc[calendar_cells, "att"].to_numpy(dtype=float)
    assert result.calendar_time.loc[4.0, "att"] == pytest.approx(
        float(calendar_shares @ calendar_estimates / calendar_shares.sum()), abs=1e-14
    )


def test_longer_robust_simultaneous_band_reuses_retained_event_influence_with_fixed_seed() -> None:
    data = _long_composition_sample(
        cohorts=(3.0, 4.0),
        periods=(1.0, 2.0, 3.0, 4.0, 5.0),
    )
    iterations = 199
    seed = 202_607_30
    level = 0.90
    result = RepeatedCrossSectionDiD(
        composition="robust",
        inference="multiplier_bootstrap",
        bootstrap_iterations=iterations,
        random_state=seed,
        simultaneous_level=level,
    ).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_long_cross_fitter(),
    )

    scores = result.event_study_influence.to_numpy(dtype=float)
    standard_errors = result.event_study["std_err"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    multipliers = rng.choice(np.array([-1.0, 1.0]), size=(iterations, len(data)))
    perturbations = np.sqrt(len(data) / (len(data) - 1)) * multipliers @ scores / len(data)
    expected = np.quantile(
        np.max(np.abs(perturbations / standard_errors), axis=1),
        level,
        method="higher",
    )
    assert result.simultaneous_critical_value == pytest.approx(expected, abs=1e-14)
    assert result.simultaneous_event_study["critical_value"].eq(expected).all()


def test_longer_composition_diagnostic_uses_complete_sorted_group_time_vector() -> None:
    data = _long_composition_sample(
        cohorts=(3.0, 4.0),
        periods=(1.0, 2.0, 3.0, 4.0, 5.0),
    )
    robust = RepeatedCrossSectionDiD(composition="robust").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_long_cross_fitter(),
    )
    stationary = RepeatedCrossSectionDiD(composition="stationary").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_long_cross_fitter(),
    )
    rng = np.random.default_rng(2_026_0730)
    difference_influence = rng.normal(size=robust.group_time_influence.shape)
    difference_influence -= difference_influence.mean(axis=0)
    stationary_influence = robust.group_time_influence - difference_influence
    differences = np.linspace(0.05, 0.25, len(robust.group_time))
    stationary_group_time = stationary.group_time.copy()
    stationary_group_time["att"] = robust.group_time["att"].to_numpy() - differences
    stationary = replace(
        stationary,
        group_time=stationary_group_time,
        group_time_influence=stationary_influence,
    )

    diagnostic = did_rcs_composition_test(robust, stationary)

    assert diagnostic.n_restrictions == len(robust.group_time)
    assert diagnostic.group_time.index.equals(robust.group_time.index)
    pd.testing.assert_frame_equal(
        diagnostic.influence,
        pd.DataFrame(
            difference_influence,
            index=robust.estimation_index,
            columns=robust.group_time.index,
        ),
        atol=2e-14,
    )
    np.testing.assert_allclose(diagnostic.group_time["difference"], differences, atol=2e-14)


def test_longer_robust_design_is_invariant_to_row_and_psu_permutation() -> None:
    data = _long_composition_sample(
        cohorts=(3.0, 4.0),
        periods=(1.0, 2.0, 3.0, 4.0, 5.0),
    )
    data["psu"] = [str(label).rsplit("-r", 1)[1] for label in data.index]
    expected = RepeatedCrossSectionDiD(
        composition="robust",
        covariance="clustered",
    ).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        cluster="psu",
        covariates=["x"],
        cross_fitter=_long_cross_fitter(),
    )
    permuted_data = data.sample(frac=1.0, random_state=2_026_0730)
    permuted = RepeatedCrossSectionDiD(
        composition="robust",
        covariance="clustered",
    ).fit(
        permuted_data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        cluster="psu",
        covariates=["x"],
        cross_fitter=_long_cross_fitter(),
    )

    pd.testing.assert_frame_equal(permuted.group_time, expected.group_time, atol=2e-14)
    pd.testing.assert_frame_equal(permuted.event_study, expected.event_study, atol=2e-14)
    pd.testing.assert_frame_equal(permuted.calendar_time, expected.calendar_time, atol=2e-14)
    pd.testing.assert_frame_equal(permuted.pair_ledger, expected.pair_ledger, atol=2e-14)
    pd.testing.assert_frame_equal(
        permuted.group_time_influence.sort_index(),
        expected.group_time_influence.sort_index(),
        atol=2e-14,
    )
    assert permuted.nuisance_fold.groupby(permuted_data["psu"]).nunique().eq(1).all()


def test_longer_robust_design_refuses_unsupported_global_and_pair_cells() -> None:
    data = _long_composition_sample()
    sparse = data.drop(index=data.index[data["treatment_time"].eq(3.0) & data["time"].eq(2.0)][1:])
    with pytest.raises(ValueError, match="stratum must contain at least n_splits"):
        RepeatedCrossSectionDiD(composition="robust").fit(
            sparse,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=["x"],
            cross_fitter=_long_cross_fitter(),
        )

    missing_cell = data.loc[~(np.isinf(data["treatment_time"]) & data["time"].eq(4.0))]
    with pytest.raises(ValueError, match="declared class labels.*observed"):
        RepeatedCrossSectionDiD(composition="robust").fit(
            missing_cell,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=["x"],
            cross_fitter=_long_cross_fitter(),
        )


class _NearBoundaryGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                0: np.full(len(X), 0.55),
                1: np.full(len(X), 0.20),
                2: np.full(len(X), 0.24999999),
                3: np.full(len(X), 1e-8),
            },
            index=X.index,
        )


class _NearBoundaryGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _NearBoundaryGeneralizedPropensityResult:
        del X, y
        return _NearBoundaryGeneralizedPropensityResult()


def test_generalized_propensity_overlap_refuses_without_clipping() -> None:
    with pytest.raises(ValueError, match="generalized propensities.*no clipping"):
        _fit_robust(
            _composition_hand_sample(),
            cross_fitter=_cross_fitter(
                propensity_factory=_NearBoundaryGeneralizedPropensity,
            ),
        )


class _ExtraClassGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile([0.25, 0.25, 0.25, 0.25, 0.0], (len(X), 1)),
            index=X.index,
            columns=[0, 1, 2, 3, 4],
        )


class _ExtraClassGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ExtraClassGeneralizedPropensityResult:
        del X, y
        return _ExtraClassGeneralizedPropensityResult()


class _MissingClassGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile([0.40, 0.30, 0.30], (len(X), 1)),
            index=X.index,
            columns=[0, 1, 2],
        )


class _MissingClassGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _MissingClassGeneralizedPropensityResult:
        del X, y
        return _MissingClassGeneralizedPropensityResult()


class _DuplicateClassGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile([0.25, 0.25, 0.25, 0.25, 0.0], (len(X), 1)),
            index=X.index,
            columns=[0, 1, 2, 3, 3],
        )


class _DuplicateClassGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _DuplicateClassGeneralizedPropensityResult:
        del X, y
        return _DuplicateClassGeneralizedPropensityResult()


class _ExtraArrayClassGeneralizedPropensityResult:
    classes_ = np.array([0, 1, 2, 3, 4])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.25, 0.25, 0.25, 0.25, 0.0], (len(X), 1))


class _ExtraArrayClassGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _ExtraArrayClassGeneralizedPropensityResult:
        del X, y
        return _ExtraArrayClassGeneralizedPropensityResult()


class _DuplicateArrayClassGeneralizedPropensityResult:
    classes_ = np.array([0, 1, 2, 3, 3])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.25, 0.25, 0.25, 0.25, 0.0], (len(X), 1))


class _DuplicateArrayClassGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _DuplicateArrayClassGeneralizedPropensityResult:
        del X, y
        return _DuplicateArrayClassGeneralizedPropensityResult()


class _UnlabeledGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.40, 0.20, 0.25, 0.15], (len(X), 1))


class _UnlabeledGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _UnlabeledGeneralizedPropensityResult:
        del X, y
        return _UnlabeledGeneralizedPropensityResult()


class _PermutedGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probabilities = _KnownGeneralizedPropensityResult().predict_proba(X)
        return probabilities.loc[:, [3, 1, 0, 2]]


class _PermutedGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _PermutedGeneralizedPropensityResult:
        del X, y
        return _PermutedGeneralizedPropensityResult()


class _PermutedArrayGeneralizedPropensityResult:
    classes_ = np.array([3, 1, 0, 2])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = _KnownGeneralizedPropensityResult().predict_proba(X)
        return probabilities.loc[:, self.classes_].to_numpy(dtype=float)


class _PermutedArrayGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _PermutedArrayGeneralizedPropensityResult:
        del X, y
        return _PermutedArrayGeneralizedPropensityResult()


@pytest.mark.parametrize(
    ("propensity_factory", "message"),
    [
        (_MissingClassGeneralizedPropensity, "missing class column"),
        (_ExtraClassGeneralizedPropensity, "exactly the observed classes"),
        (_DuplicateClassGeneralizedPropensity, "unique class columns"),
        (_ExtraArrayClassGeneralizedPropensity, "exactly the observed classes"),
        (_DuplicateArrayClassGeneralizedPropensity, "unique class labels"),
        (_UnlabeledGeneralizedPropensity, "requires fitted result.classes_"),
    ],
)
def test_generalized_propensity_refuses_invalid_or_ambiguous_class_schema(
    propensity_factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _fit_robust(
            _composition_hand_sample(),
            cross_fitter=_cross_fitter(propensity_factory=propensity_factory),
        )


@pytest.mark.parametrize(
    "propensity_factory",
    [_PermutedGeneralizedPropensity, _PermutedArrayGeneralizedPropensity],
)
def test_labeled_generalized_propensity_order_is_realigned_without_changing_result(
    propensity_factory: Callable[[], object],
) -> None:
    data = _composition_hand_sample()
    expected = _fit_robust(data)
    permuted = _fit_robust(
        data,
        cross_fitter=_cross_fitter(propensity_factory=propensity_factory),
    )

    assert permuted.estimate == pytest.approx(expected.estimate, abs=1e-14)
    pd.testing.assert_frame_equal(
        permuted.composition_weights,
        expected.composition_weights,
        atol=1e-14,
    )
    pd.testing.assert_series_equal(
        permuted.overall_influence,
        expected.overall_influence,
        atol=1e-14,
    )


def test_composition_robust_result_is_invariant_to_row_permutation() -> None:
    data = _composition_hand_sample()
    permuted_data = data.sample(frac=1.0, random_state=2_026_073)
    expected = _fit_robust(data)
    permuted = _fit_robust(permuted_data)

    assert permuted.estimate == pytest.approx(expected.estimate, abs=1e-14)
    assert permuted.standard_error == pytest.approx(expected.standard_error, abs=1e-14)
    assert permuted.design_fingerprint == expected.design_fingerprint
    pd.testing.assert_frame_equal(permuted.group_time, expected.group_time, atol=1e-14)
    pd.testing.assert_frame_equal(
        permuted.composition_weights.sort_index(),
        expected.composition_weights.sort_index(),
        atol=1e-14,
    )
    pd.testing.assert_series_equal(
        permuted.overall_influence.sort_index(),
        expected.overall_influence.sort_index(),
        atol=1e-14,
    )
    pd.testing.assert_frame_equal(
        permuted.nuisance_predictions.sort_index(),
        expected.nuisance_predictions.sort_index(),
        atol=1e-14,
    )


class _AuditResult:
    def __init__(
        self,
        train_index: pd.Index,
        registry: list[tuple[pd.Index, pd.Index]],
    ) -> None:
        self.train_index = train_index.copy()
        self.registry = registry

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        self.registry.append((self.train_index, X.index.copy()))
        return pd.DataFrame(
            np.full((len(X), 4), 0.25),
            index=X.index,
            columns=[0, 1, 2, 3],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.registry.append((self.train_index, X.index.copy()))
        return np.zeros(len(X))


class _AuditFactory:
    def __init__(self, registry: list[tuple[pd.Index, pd.Index]]) -> None:
        self.registry = registry

    def __call__(self):
        registry = self.registry

        class _AuditModel:
            def fit(self, X: pd.DataFrame, y: pd.Series) -> _AuditResult:
                del y
                return _AuditResult(X.index, registry)

        return _AuditModel()


def test_robust_cross_fitting_preserves_rows_and_indivisible_psus() -> None:
    data = _composition_hand_sample()
    data["psu"] = [f"psu-{position % 4}" for position in range(len(data))]
    registry: list[tuple[pd.Index, pd.Index]] = []
    factory = _AuditFactory(registry)
    result = RepeatedCrossSectionDiD(
        composition="robust",
        covariance="clustered",
    ).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        cluster="psu",
        covariates=["x"],
        cross_fitter=_cross_fitter(
            propensity_factory=factory,
            outcome_factory=factory,
        ),
    )

    assert registry
    for train_index, holdout_index in registry:
        assert set(train_index).isdisjoint(holdout_index)
        assert set(data.loc[train_index, "psu"]).isdisjoint(data.loc[holdout_index, "psu"])
    assert result.nuisance_fold.groupby(data["psu"]).nunique().eq(1).all()


def test_outputhub_preserves_composition_target_nuisances_and_weights() -> None:
    outputhub = pytest.importorskip("universal_output_hub")
    result = _fit_robust(_composition_hand_sample())

    model = to_outputhub_model(result)
    hub = outputhub.OutputHub("Composition-robust repeated sections")
    add_to_outputhub(hub, result)

    assert model.metadata["composition"] == "robust"
    assert model.metadata["target_population"] == "treated_target_period"
    assert model.diagnostics["Nuisance fold fits"] == 8
    assert hub.tables[0].name == "Repeated-cross-section DiD nuisance fitting diagnostics"
    assert hub.tables[1].name == "Repeated-cross-section DiD composition weights"


def _fit_composition_shift_pair(*, clustered: bool = False):
    data = _composition_shift_sample()
    constructor: dict[str, object] = {}
    fit_roles: dict[str, object] = {}
    if clustered:
        data["psu"] = [f"psu-{position % 8}" for position in range(len(data))]
        constructor["covariance"] = "clustered"
        fit_roles["cluster"] = "psu"
    robust = RepeatedCrossSectionDiD(composition="robust", **constructor).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_cross_fitter(
            propensity_factory=_CompositionShiftGeneralizedPropensity,
            outcome_factory=_CompositionShiftOutcome,
        ),
        **fit_roles,
    )
    stationary = RepeatedCrossSectionDiD(composition="stationary", **constructor).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_cross_fitter(
            propensity_factory=_CompositionShiftBinaryPropensity,
            outcome_factory=_CompositionShiftOutcome,
        ),
        **fit_roles,
    )
    return data, robust, stationary


def test_composition_diagnostic_hand_computes_scalar_difference_hc1_and_pvalue() -> None:
    data, robust, stationary = _fit_composition_shift_pair()

    diagnostic = did_rcs_composition_test(robust, stationary)

    assert isinstance(diagnostic, RepeatedCrossSectionCompositionDiagnostic)
    assert diagnostic.group_time.index.tolist() == [(2.0, 2.0)]
    assert diagnostic.group_time.loc[(2.0, 2.0), "robust"] == pytest.approx(5.0)
    assert diagnostic.group_time.loc[(2.0, 2.0), "stationary"] == pytest.approx(4.0)
    assert diagnostic.group_time.loc[(2.0, 2.0), "difference"] == pytest.approx(1.0)
    expected_influence = robust.group_time_influence - stationary.group_time_influence
    pd.testing.assert_frame_equal(diagnostic.influence, expected_influence, atol=1e-14)
    n = len(data)
    expected_variance = float(
        (expected_influence.to_numpy().T @ expected_influence.to_numpy() / (n * (n - 1))).item()
    )
    expected_statistic = 1.0 / expected_variance
    assert diagnostic.covariance.iloc[0, 0] == pytest.approx(expected_variance, abs=1e-14)
    assert diagnostic.statistic == pytest.approx(expected_statistic, abs=1e-14)
    assert diagnostic.pvalue == pytest.approx(chi2.sf(expected_statistic, 1), abs=1e-14)
    assert diagnostic.distribution == "chi2"
    assert diagnostic.df_num == 1
    assert diagnostic.df_denom is None
    assert diagnostic.n_clusters is None
    assert diagnostic.estimation_index.equals(data.index)
    assert diagnostic.inference_clusters.empty
    assert not hasattr(diagnostic, "recommendation")
    assert "selection" in " ".join(diagnostic.notes).lower()


def test_composition_diagnostic_hand_computes_cluster_sums_cr1_and_f_reference() -> None:
    data, robust, stationary = _fit_composition_shift_pair(clustered=True)

    diagnostic = did_rcs_composition_test(robust, stationary, level=0.90)

    expected_influence = robust.group_time_influence - stationary.group_time_influence
    cluster_sums = expected_influence.groupby(data["psu"], sort=False).sum().to_numpy()
    n_clusters = data["psu"].nunique()
    n = len(data)
    expected_variance = float(
        (n_clusters / (n_clusters - 1) * cluster_sums.T @ cluster_sums / n**2).item()
    )
    expected_statistic = 1.0 / expected_variance
    assert diagnostic.covariance.iloc[0, 0] == pytest.approx(expected_variance, abs=1e-14)
    assert diagnostic.statistic == pytest.approx(expected_statistic, abs=1e-14)
    assert diagnostic.pvalue == pytest.approx(
        f_distribution.sf(expected_statistic, 1, n_clusters - 1), abs=1e-14
    )
    assert diagnostic.distribution == "f"
    assert diagnostic.df_num == 1
    assert diagnostic.df_denom == n_clusters - 1
    assert diagnostic.n_clusters == n_clusters
    assert diagnostic.level == pytest.approx(0.90)


def test_composition_diagnostic_exposes_marginal_and_cross_covariance_identity() -> None:
    _, robust, stationary = _fit_composition_shift_pair()

    diagnostic = did_rcs_composition_test(robust, stationary)

    expected = (
        diagnostic.robust_covariance.to_numpy()
        + diagnostic.stationary_covariance.to_numpy()
        - diagnostic.cross_covariance.to_numpy()
        - diagnostic.cross_covariance.to_numpy().T
    )
    np.testing.assert_allclose(diagnostic.covariance, expected, atol=1e-14)
    assert diagnostic.design_fingerprint == robust.design_fingerprint
    assert diagnostic.design_fingerprint == stationary.design_fingerprint


def test_outputhub_composition_diagnostic_reports_both_paths_without_selection() -> None:
    outputhub = pytest.importorskip("universal_output_hub")
    _, robust, stationary = _fit_composition_shift_pair()
    diagnostic = did_rcs_composition_test(robust, stationary)
    hub = outputhub.OutputHub("Composition diagnostic")

    model = add_to_outputhub(hub, diagnostic)

    assert model.metadata["estimator"] == "rcs_composition_equality_diagnostic"
    assert model.metadata["estimator_selection"] is False
    assert model.params.index.tolist() == [(2.0, 2.0)]
    assert hub.tables[0].name == "RCS composition equality diagnostic estimates"
    assert hub.tables[0].data.loc[0, "robust"] == pytest.approx(5.0)
    assert hub.tables[0].data.loc[0, "stationary"] == pytest.approx(4.0)
    assert hub.tables[1].name == "RCS composition equality diagnostic joint test"


def test_composition_diagnostic_refuses_order_sample_fold_roles_and_covariance_mismatch() -> None:
    _, robust, stationary = _fit_composition_shift_pair()

    with pytest.raises(ValueError, match="first argument.*robust"):
        did_rcs_composition_test(stationary, robust)
    with pytest.raises(ValueError, match="same estimation design"):
        did_rcs_composition_test(
            robust,
            replace(stationary, design_fingerprint="different"),
        )
    altered_fold = stationary.nuisance_fold.copy()
    altered_fold.iloc[0] = 1 - int(altered_fold.iloc[0])
    with pytest.raises(ValueError, match="same nuisance-fold"):
        did_rcs_composition_test(robust, replace(stationary, nuisance_fold=altered_fold))
    with pytest.raises(ValueError, match="role and timing"):
        did_rcs_composition_test(robust, replace(stationary, outcome_name="other"))
    with pytest.raises(ValueError, match="same covariance"):
        did_rcs_composition_test(robust, replace(stationary, covariance_type="clustered"))
    with pytest.raises(ValueError, match="same covariates"):
        did_rcs_composition_test(robust, replace(stationary, covariates=("z",)))


def test_composition_diagnostic_refuses_support_cluster_centering_and_singularity_failures() -> (
    None
):
    _, robust, stationary = _fit_composition_shift_pair()
    empty_group_time = stationary.group_time.iloc[0:0].copy()
    empty_influence = stationary.group_time_influence.iloc[:, 0:0].copy()
    with pytest.raises(ValueError, match="same non-empty group-time support"):
        did_rcs_composition_test(
            robust,
            replace(
                stationary,
                group_time=empty_group_time,
                group_time_influence=empty_influence,
            ),
        )
    noncentered = stationary.group_time_influence + 1.0
    with pytest.raises(ValueError, match="centered"):
        did_rcs_composition_test(
            robust,
            replace(stationary, group_time_influence=noncentered),
        )
    identical = replace(
        stationary,
        estimate=robust.estimate,
        group_time=robust.group_time.copy(),
        group_time_influence=robust.group_time_influence.copy(),
    )
    with pytest.raises(ValueError, match="no ridge or pseudoinverse"):
        did_rcs_composition_test(robust, identical)

    _, clustered_robust, clustered_stationary = _fit_composition_shift_pair(clustered=True)
    altered_clusters = clustered_stationary.inference_clusters.copy()
    altered_clusters.iloc[0] = "different-psu"
    with pytest.raises(ValueError, match="identical inference clusters"):
        did_rcs_composition_test(
            clustered_robust,
            replace(clustered_stationary, inference_clusters=altered_clusters),
        )


@pytest.mark.parametrize("level", [0.0, 1.0, np.nan, np.inf, "0.95"])
def test_composition_diagnostic_refuses_invalid_confidence_level(level: object) -> None:
    _, robust, stationary = _fit_composition_shift_pair()

    with pytest.raises((TypeError, ValueError), match="level"):
        did_rcs_composition_test(robust, stationary, level=level)  # type: ignore[arg-type]

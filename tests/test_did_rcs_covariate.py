"""Covariate-adjusted repeated-cross-section DiD contracts."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causekit import CrossFitter, RepeatedCrossSectionDiD, add_to_outputhub, to_outputhub_model


class _KnownPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probability = 0.25 + 0.50 * X["x"].to_numpy(dtype=float)
        return pd.DataFrame(
            {0: 1.0 - probability, 1: probability},
            index=X.index,
        )


class _KnownPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _KnownPropensityResult:
        del X, y
        return _KnownPropensityResult()


class _KnownLogitPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probability = 1.0 / (1.0 + np.exp(-(0.2 + 0.5 * X["x"].to_numpy(dtype=float))))
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _KnownLogitPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _KnownLogitPropensityResult:
        del X, y
        return _KnownLogitPropensityResult()


class _LinearResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        return design @ self.coefficients


class _LinearOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _LinearResult:
        design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        coefficients = np.linalg.solve(design.T @ design, design.T @ y.to_numpy(dtype=float))
        return _LinearResult(coefficients)


def _cross_fitter(
    *,
    n_splits: int = 2,
    random_state: int = 17,
    propensity_factory: Callable[[], object] = _KnownPropensity,
    outcome_factory: Callable[[], object] = _LinearOutcome,
) -> CrossFitter:
    return CrossFitter(
        propensity_factory=propensity_factory,
        outcome_factory=outcome_factory,
        n_splits=n_splits,
        random_state=random_state,
    )


def _hand_sample() -> pd.DataFrame:
    x = np.array([0.0, 0.2, 0.6, 1.0, 0.1, 0.4, 0.7, 0.9, 0.0, 0.3, 0.5, 0.8, 0.2, 0.4, 0.6, 1.0])
    treated = np.repeat([1.0, 0.0], 8)
    post = np.tile(np.repeat([0.0, 1.0], 4), 2)
    m0_pre = 1.0 + x
    m0_post = 2.0 + 2.0 * x
    m1_pre = 3.0 + 0.5 * x
    m1_post = 6.0 + 2.5 * x
    outcome = np.where(
        treated == 1.0,
        np.where(post == 1.0, m1_post, m1_pre),
        np.where(post == 1.0, m0_post, m0_pre),
    )
    return pd.DataFrame(
        {
            "outcome": outcome,
            "time": post + 1.0,
            "treatment_time": np.where(treated == 1.0, 2.0, np.inf),
            "x": x,
        },
        index=pd.Index([f"row-{position}" for position in range(len(x))], name="row"),
    )


def _fit_covariate(
    data: pd.DataFrame,
    *,
    cross_fitter: CrossFitter | None = None,
    **constructor: object,
):
    return RepeatedCrossSectionDiD(**constructor).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=cross_fitter or _cross_fitter(),
    )


def test_hand_computed_locally_efficient_score_and_influence_contract() -> None:
    data = _hand_sample()

    result = _fit_covariate(data)

    # Sant'Anna-Zhao (2020), equation (3.4): four normalized residual terms and
    # four normalized outcome-regression adjustment terms.
    assert result.estimate == pytest.approx(2.4875, abs=2e-13)
    expected_influence = pd.Series(
        [
            -0.975,
            -0.575,
            0.225,
            1.025,
            -0.775,
            -0.175,
            0.425,
            0.825,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        index=data.index,
        name="esavg_influence",
    )
    pd.testing.assert_series_equal(result.overall_influence, expected_influence, atol=2e-12)
    expected_se = np.sqrt(
        float(expected_influence @ expected_influence) / (len(data) * (len(data) - 1))
    )
    assert result.standard_error == pytest.approx(expected_se, abs=2e-14)
    assert result.method == "repeated_cross_section_doubly_robust_group_time"
    assert (
        result.parallel_trends == "conditional_repeated_cross_section_post_stationary_composition"
    )
    assert result.covariates == ("x",)
    assert result.cross_fitted is True
    assert result.nuisance_fold.index.equals(data.index)
    assert set(result.nuisance_fold.unique()) == {0, 1}
    assert result.nuisance_predictions.columns.names == ["cohort", "time", "nuisance"]
    assert result.nuisance_predictions.columns.get_level_values("nuisance").tolist() == [
        "propensity",
        "outcome_control_pre",
        "outcome_control_post",
        "outcome_treated_pre",
        "outcome_treated_post",
    ]
    assert len(result.nuisance_diagnostics) == 10
    assert result.nuisance_diagnostics["fold"].isin([0, 1]).all()
    assert result.nuisance_diagnostics["relevant_holdout_nobs"].eq(8).all()
    assert result.nuisance_diagnostics["comparison_stage"].eq("group_time").all()


def test_covariate_multiplier_band_reuses_retained_oof_event_influence() -> None:
    iterations = 99
    seed = 424
    result = _fit_covariate(
        _hand_sample(),
        inference="multiplier_bootstrap",
        bootstrap_iterations=iterations,
        random_state=seed,
        simultaneous_level=0.95,
    )
    influence = result.event_study_influence.to_numpy(dtype=float)
    multipliers = np.random.default_rng(seed).choice(
        np.array([-1.0, 1.0]), size=(iterations, result.n_observations)
    )
    draws = (
        np.sqrt(result.n_observations / (result.n_observations - 1))
        * multipliers
        @ influence
        / result.n_observations
    )
    expected = np.quantile(
        np.max(np.abs(draws / result.event_study["std_err"].to_numpy(dtype=float)), axis=1),
        0.95,
        method="higher",
    )

    assert result.simultaneous_critical_value == pytest.approx(expected, abs=1e-14)
    assert len(result.nuisance_diagnostics) == 10
    assert result.nuisance_predictions.notna().all().all()


class _AuditResult:
    def __init__(
        self,
        *,
        train_index: pd.Index,
        registry: list[tuple[pd.Index, pd.Index]],
        propensity: bool,
    ) -> None:
        self.train_index = train_index.copy()
        self.registry = registry
        self.propensity = propensity

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.registry.append((self.train_index, X.index.copy()))
        return 1.0 + 0.1 * X["x"].to_numpy(dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        self.registry.append((self.train_index, X.index.copy()))
        probability = np.full(len(X), 0.5)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _AuditFactory:
    def __init__(self, registry: list[tuple[pd.Index, pd.Index]], *, propensity: bool) -> None:
        self.registry = registry
        self.propensity = propensity

    def __call__(self):
        registry = self.registry
        propensity = self.propensity

        class _AuditModel:
            def fit(self, X: pd.DataFrame, y: pd.Series) -> _AuditResult:
                del y
                return _AuditResult(
                    train_index=X.index,
                    registry=registry,
                    propensity=propensity,
                )

        return _AuditModel()


def test_cross_fitting_never_leaks_a_row_or_declared_cluster() -> None:
    data = _hand_sample()
    data["psu"] = np.tile(np.repeat(["a", "b", "c", "d"], 1), 4)
    registry: list[tuple[pd.Index, pd.Index]] = []
    fitter = _cross_fitter(
        propensity_factory=_AuditFactory(registry, propensity=True),
        outcome_factory=_AuditFactory(registry, propensity=False),
    )

    result = RepeatedCrossSectionDiD(covariance="clustered").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        cluster="psu",
        covariates=["x"],
        cross_fitter=fitter,
    )

    assert registry
    for train_index, holdout_index in registry:
        assert set(train_index).isdisjoint(holdout_index)
        assert set(data.loc[train_index, "psu"]).isdisjoint(data.loc[holdout_index, "psu"])
    assert result.nuisance_fold.groupby(data["psu"]).nunique().eq(1).all()


class _BoundaryPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probability = np.where(X["x"].to_numpy(dtype=float) == 0.0, 0.0, 0.5)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _BoundaryPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _BoundaryPropensityResult:
        del X, y
        return _BoundaryPropensityResult()


class _ConstantPropensityResult:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probability = np.full(len(X), self.probability)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _HalfPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ConstantPropensityResult:
        del X, y
        return _ConstantPropensityResult(0.5)


class _EmpiricalPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ConstantPropensityResult:
        del X
        return _ConstantPropensityResult(float(y.mean()))


class _ZeroResult:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class _ZeroOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ZeroResult:
        del X, y
        return _ZeroResult()


def test_overlap_violation_is_refused_without_clipping_or_row_deletion() -> None:
    with pytest.raises(ValueError, match="nuisance_probability_floor.*no clipping"):
        _fit_covariate(
            _hand_sample(),
            cross_fitter=_cross_fitter(propensity_factory=_BoundaryPropensity),
        )


def test_outcome_regression_leg_is_doubly_robust_to_wrong_propensity() -> None:
    result = _fit_covariate(
        _hand_sample(),
        cross_fitter=_cross_fitter(propensity_factory=_HalfPropensity),
    )

    assert result.estimate == pytest.approx(2.4875, abs=2e-13)


def test_propensity_leg_is_doubly_robust_to_wrong_outcome_regressions() -> None:
    data = _hand_sample()
    repeated_x = np.tile(np.array([0.0, 0.2, 0.6, 1.0]), 4)
    data["x"] = repeated_x
    treated = np.repeat([1.0, 0.0], 8)
    post = np.tile(np.repeat([0.0, 1.0], 4), 2)
    data["outcome"] = (
        1.0 + repeated_x + 0.5 * post + treated * (2.0 * post) + treated * (0.4 + 0.3 * repeated_x)
    )
    conventional = RepeatedCrossSectionDiD().fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
    )

    result = _fit_covariate(
        data,
        cross_fitter=_cross_fitter(
            propensity_factory=_HalfPropensity,
            outcome_factory=_ZeroOutcome,
        ),
    )

    assert result.estimate == pytest.approx(2.0, abs=2e-13)
    assert result.estimate == pytest.approx(conventional.estimate, abs=2e-13)
    np.testing.assert_allclose(result.overall_influence, conventional.overall_influence)


def _conditional_pretrend_sample() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    x_by_cell = {
        (1.0, 1.0): (0.0, 0.2, 0.4, 0.6),
        (1.0, 2.0): (0.4, 0.6, 0.8, 1.0),
        (1.0, 3.0): (0.1, 0.3, 0.5, 0.7),
        (0.0, 1.0): (0.4, 0.6, 0.8, 1.0),
        (0.0, 2.0): (0.0, 0.2, 0.4, 0.6),
        (0.0, 3.0): (0.2, 0.4, 0.6, 0.8),
    }
    for treated in (1.0, 0.0):
        for time in (1.0, 2.0, 3.0):
            for replicate, x in enumerate(x_by_cell[(treated, time)]):
                if treated == 1.0:
                    intercept, slope = {
                        1.0: (2.0, 0.5),
                        2.0: (3.0, 1.5),
                        3.0: (6.0, 2.5),
                    }[time]
                else:
                    intercept, slope = {
                        1.0: (0.0, 1.0),
                        2.0: (1.0, 2.0),
                        3.0: (2.0, 3.0),
                    }[time]
                rows.append(
                    {
                        "row": f"d{treated}-t{time}-r{replicate}",
                        "outcome": intercept + slope * x,
                        "time": time,
                        "treatment_time": 3.0 if treated else np.inf,
                        "x": x,
                    }
                )
    return pd.DataFrame(rows).set_index("row")


def _staggered_covariate_sample() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for cohort, level in ((2.0, 10.0), (3.0, 20.0), (np.inf, 30.0)):
        for time in (1.0, 2.0, 3.0):
            effect = 0.0
            if cohort == 2.0 and time == 2.0:
                effect = 2.0
            elif cohort == 2.0 and time == 3.0:
                effect = 4.0
            elif cohort == 3.0 and time == 3.0:
                effect = 6.0
            for replicate, x in enumerate((0.0, 0.3, 0.6, 1.0)):
                rows.append(
                    {
                        "row": f"g{cohort}-t{time}-r{replicate}",
                        "outcome": level + 0.5 * time + x + effect,
                        "time": time,
                        "treatment_time": cohort,
                        "x": x,
                    }
                )
    return pd.DataFrame(rows).set_index("row")


def test_pretrend_uses_the_same_cross_fitted_conditional_score() -> None:
    data = _conditional_pretrend_sample()
    marginal = RepeatedCrossSectionDiD().fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
    )

    result = _fit_covariate(data, cross_fitter=_cross_fitter(random_state=8))

    assert marginal.pretrend.placebo_effects.loc[(3.0, 2.0), "att"] == pytest.approx(1.0)
    assert result.pretrend.available
    assert result.pretrend.conditional is True
    assert result.pretrend.covariates == ("x",)
    assert result.pretrend.cross_fitted is True
    assert result.pretrend.placebo_effects.loc[(3.0, 2.0), "att"] == pytest.approx(0.0, abs=2e-12)
    assert result.pretrend.placebo_effects.loc[(3.0, 2.0), "comparison_cohorts"] == (np.inf,)
    assert result.nuisance_predictions.columns.get_level_values("time").tolist() == [
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
    ]
    assert set(result.nuisance_diagnostics["comparison_stage"]) == {
        "conditional_pretrend",
        "group_time",
    }


@pytest.mark.parametrize("control_group", ["never_treated", "not_yet_treated"])
def test_staggered_cross_fitted_score_preserves_comparison_and_aggregation_contracts(
    control_group: str,
) -> None:
    data = _staggered_covariate_sample()
    conventional = RepeatedCrossSectionDiD(control_group=control_group).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
    )
    result = _fit_covariate(
        data,
        control_group=control_group,
        cross_fitter=_cross_fitter(
            propensity_factory=_EmpiricalPropensity,
            outcome_factory=_ZeroOutcome,
        ),
    )

    pd.testing.assert_frame_equal(result.group_time, conventional.group_time)
    pd.testing.assert_frame_equal(result.event_study, conventional.event_study)
    pd.testing.assert_frame_equal(result.calendar_time, conventional.calendar_time)
    pd.testing.assert_frame_equal(result.group_time_influence, conventional.group_time_influence)
    pd.testing.assert_series_equal(result.overall_influence, conventional.overall_influence)
    assert result.estimate == pytest.approx(4.0, abs=2e-13)
    if control_group == "not_yet_treated":
        assert result.group_time.loc[(2.0, 2.0), "comparison_cohorts"] == (3.0, np.inf)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.assign(x=np.nan), "covariates.*missing"),
        (lambda data: data.assign(x=np.inf), "covariates.*finite"),
        (lambda data: data.assign(x="not numeric"), "covariates.*numeric"),
    ],
)
def test_covariate_role_refusals(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _fit_covariate(mutation(_hand_sample()))


def test_cross_fitter_and_covariate_roles_must_be_supplied_together() -> None:
    data = _hand_sample()
    with pytest.raises(ValueError, match="cross_fitter is required"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=["x"],
        )
    with pytest.raises(ValueError, match="only used when covariates"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            cross_fitter=_cross_fitter(),
        )
    with pytest.raises(ValueError, match="distinct"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=["outcome"],
            cross_fitter=_cross_fitter(),
        )


def test_covariate_path_refuses_missing_factories_and_empty_or_duplicate_roles() -> None:
    data = _hand_sample()
    with pytest.raises(ValueError, match="propensity_factory"):
        _fit_covariate(data, cross_fitter=CrossFitter(outcome_factory=_LinearOutcome))
    with pytest.raises(ValueError, match="outcome_factory"):
        _fit_covariate(data, cross_fitter=CrossFitter(propensity_factory=_KnownPropensity))
    with pytest.raises(ValueError, match="at least one"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=[],
            cross_fitter=_cross_fitter(),
        )
    with pytest.raises(ValueError, match="distinct columns"):
        RepeatedCrossSectionDiD().fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=["x", "x"],
            cross_fitter=_cross_fitter(),
        )


def test_integrated_cross_fitter_refuses_nonfresh_models_and_insufficient_psu_support() -> None:
    data = _hand_sample()
    singleton = _ZeroOutcome()
    with pytest.raises(ValueError, match="fresh estimator"):
        _fit_covariate(
            data,
            cross_fitter=_cross_fitter(outcome_factory=lambda: singleton),
        )

    clustered = data.assign(psu=np.tile(["a", "b"], len(data) // 2))
    with pytest.raises(ValueError, match="at least n_splits clusters"):
        RepeatedCrossSectionDiD(covariance="clustered").fit(
            clustered,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            cluster="psu",
            covariates=["x"],
            cross_fitter=_cross_fitter(n_splits=3),
        )


def test_fixed_seed_reproduces_all_nuisance_and_effect_audits() -> None:
    data = _conditional_pretrend_sample()
    first = _fit_covariate(data, cross_fitter=_cross_fitter(random_state=77))
    second = _fit_covariate(data, cross_fitter=_cross_fitter(random_state=77))

    pd.testing.assert_series_equal(first.nuisance_fold, second.nuisance_fold)
    pd.testing.assert_frame_equal(first.nuisance_predictions, second.nuisance_predictions)
    pd.testing.assert_frame_equal(first.nuisance_diagnostics, second.nuisance_diagnostics)
    pd.testing.assert_frame_equal(first.group_time, second.group_time)
    pd.testing.assert_frame_equal(first.pretrend.placebo_effects, second.pretrend.placebo_effects)


def test_outputhub_exposes_conditional_contract_and_fold_diagnostics_without_refitting() -> None:
    outputhub = pytest.importorskip("universal_output_hub")
    result = _fit_covariate(_hand_sample())

    model = to_outputhub_model(result)
    hub = outputhub.OutputHub("Covariate repeated samples")
    add_to_outputhub(hub, result)

    assert model.metadata["nuisance_cross_fitted"] is True
    assert model.metadata["nuisance_n_splits"] == 2
    assert model.metadata["nuisance_probability_floor"] == pytest.approx(1e-6)
    assert model.metadata["conditional_pretrend"] is True
    assert model.diagnostics["Nuisance fold fits"] == 10
    assert hub.tables[0].name == "Repeated-cross-section DiD nuisance fitting diagnostics"
    assert hub.tables[0].metadata["n_splits"] == 2


@pytest.mark.validation
def test_base_r_reconstructs_fixed_oof_efficient_score_influence_and_hc1() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not available for the cross-language score contract.")
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [rscript, str(repository / "benchmarks" / "validate_did_rcs_covariate_reference.R")],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository,
    )
    fields = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    result = _fit_covariate(_hand_sample())

    assert fields["contract"] == "fixed_oof_santanna_zhao_rc_efficient_hc1"
    assert fields["parity_status"] == "pass"
    assert float(fields["estimate"]) == pytest.approx(result.estimate, abs=1e-12)
    assert float(fields["standard_error_hc1"]) == pytest.approx(result.standard_error, abs=1e-12)
    np.testing.assert_allclose(
        np.fromstring(fields["influence"], sep=","),
        result.overall_influence,
        atol=1e-12,
    )


@pytest.mark.parametrize("value", [0.0, 0.5, np.nan, np.inf, True])
def test_invalid_nuisance_probability_floor_is_refused(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="nuisance_probability_floor"):
        RepeatedCrossSectionDiD(nuisance_probability_floor=value)


@pytest.mark.simulation
def test_cross_fitted_conditional_score_bias_and_hc1_coverage_smoke() -> None:
    estimates: list[float] = []
    covered = 0
    repetitions = 40
    for seed in range(repetitions):
        rng = np.random.default_rng(seed)
        rows: list[dict[str, float]] = []
        for time in (1.0, 2.0):
            x = rng.normal(size=160)
            probability = 1.0 / (1.0 + np.exp(-(0.2 + 0.5 * x)))
            treated = rng.binomial(1, probability)
            error = rng.normal(size=len(x))
            post = float(time == 2.0)
            outcome = 1.0 + x + 1.5 * treated + post * (0.5 + 0.3 * x + 2.0 * treated) + error
            for position in range(len(x)):
                rows.append(
                    {
                        "outcome": float(outcome[position]),
                        "time": time,
                        "treatment_time": 2.0 if treated[position] else np.inf,
                        "x": float(x[position]),
                    }
                )
        result = _fit_covariate(
            pd.DataFrame(rows),
            cross_fitter=_cross_fitter(
                random_state=seed,
                propensity_factory=_KnownLogitPropensity,
            ),
        )
        interval = result.conf_int()
        estimates.append(result.estimate)
        covered += int(interval["lower"] <= 2.0 <= interval["upper"])

    assert float(np.mean(estimates)) == pytest.approx(2.0, abs=0.07)
    assert covered >= 34

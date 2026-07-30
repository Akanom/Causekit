"""Contracts for repeated-cross-section difference-in-differences."""

from __future__ import annotations

import inspect
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causekit import RepeatedCrossSectionDiD, RepeatedCrossSectionDiDResult


def _two_by_two() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome": [1.0, 3.0, 6.0, 8.0, 2.0, 4.0, 3.0, 5.0],
            "time": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
            "treatment_time": [2.0, 2.0, 2.0, 2.0, np.inf, np.inf, np.inf, np.inf],
        },
        index=pd.Index(["tp0", "tp1", "tq0", "tq1", "cp0", "cp1", "cq0", "cq1"]),
    )


def _fit(data: pd.DataFrame, **kwargs: object) -> RepeatedCrossSectionDiDResult:
    return RepeatedCrossSectionDiD(**kwargs).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
    )


def _staggered_sample() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    trends = {1.0: 0.0, 2.0: 1.0, 3.0: 2.0}
    levels = {2.0: 10.0, 3.0: 20.0, np.inf: 30.0}
    for cohort in (2.0, 3.0, np.inf):
        for time in (1.0, 2.0, 3.0):
            effect = 0.0
            if cohort == 2.0 and time == 2.0:
                effect = 2.0
            elif cohort == 2.0 and time == 3.0:
                effect = 4.0
            elif cohort == 3.0 and time == 3.0:
                effect = 6.0
            mean = levels[cohort] + trends[time] + effect
            for replicate, deviation in enumerate((-1.0, 1.0)):
                rows.append(
                    {
                        "row": f"g{cohort}_t{time}_r{replicate}",
                        "outcome": mean + deviation,
                        "time": time,
                        "treatment_time": cohort,
                    }
                )
    return pd.DataFrame(rows).set_index("row")


def test_hand_computed_two_by_two_att_and_observation_influence() -> None:
    data = _two_by_two()

    result = _fit(data)

    assert isinstance(result, RepeatedCrossSectionDiDResult)
    assert result.estimate == pytest.approx((7.0 - 2.0) - (4.0 - 3.0))
    assert result.group_time.loc[(2.0, 2.0), "att"] == pytest.approx(4.0)
    expected = pd.Series(
        [4.0, -4.0, -4.0, 4.0, -4.0, 4.0, 4.0, -4.0],
        index=data.index,
        name="esavg_influence",
    )
    pd.testing.assert_series_equal(result.overall_influence, expected)
    expected_se = np.sqrt(float(expected @ expected) / (len(data) * (len(data) - 1)))
    assert result.standard_error == pytest.approx(expected_se)
    assert result.covariance_type == "robust"
    assert result.inference_distribution == "normal"
    assert result.sampling_unit == "observation"
    assert result.composition == "stationary"
    assert result.cross_fitted is False
    assert result.covariates == ()
    assert result.nuisance_predictions.empty
    assert result.nuisance_fold.empty
    assert result.nuisance_diagnostics.empty
    assert result.n_splits is None
    assert result.nuisance_probability_floor is None
    assert result.pretrend.conditional is False
    assert result.nobs == len(data)
    assert result.n_periods == 2
    assert result.cell_counts["nobs"].to_dict() == {
        (2.0, 1.0): 2,
        (2.0, 2.0): 2,
        (np.inf, 1.0): 2,
        (np.inf, 2.0): 2,
    }


@pytest.mark.parametrize("control_group", ["never_treated", "not_yet_treated"])
def test_staggered_group_event_calendar_and_esavg_identities(control_group: str) -> None:
    result = _fit(_staggered_sample(), control_group=control_group)

    assert result.group_time["att"].to_dict() == pytest.approx(
        {(2.0, 2.0): 2.0, (2.0, 3.0): 4.0, (3.0, 3.0): 6.0}
    )
    assert result.event_study["att"].to_dict() == pytest.approx({0: 4.0, 1: 4.0})
    assert result.calendar_time["att"].to_dict() == pytest.approx({2.0: 2.0, 3.0: 5.0})
    assert result.estimate == pytest.approx(4.0)
    if control_group == "not_yet_treated":
        assert result.group_time.loc[(2.0, 2.0), "comparison_cohorts"] == (3.0, np.inf)
        assert result.group_time.loc[(2.0, 3.0), "comparison_cohorts"] == (np.inf,)


def test_aggregation_includes_estimated_pooled_cohort_share_influence() -> None:
    data = _staggered_sample()
    result = _fit(data)
    group = result.group_time_influence
    cohort = data["treatment_time"].to_numpy()

    expected_event_zero = (
        0.5 * group[(2.0, 2.0)].to_numpy()
        + 0.5 * group[(3.0, 3.0)].to_numpy()
        - 3.0 * (cohort == 2.0)
        + 3.0 * (cohort == 3.0)
    )
    np.testing.assert_allclose(result.event_study_influence[0], expected_event_zero)
    assert float(result.event_study_influence[0].sum()) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        result.overall_influence,
        result.event_study_influence.loc[:, [0, 1]].mean(axis=1),
    )


def test_unequal_period_sizes_and_row_permutation_preserve_point_contract() -> None:
    balanced = _staggered_sample()
    additions = balanced.loc[balanced["time"] == 2.0].copy()
    additions.index = additions.index.map(lambda value: f"extra_{value}")
    unequal = pd.concat([balanced, additions])

    reference = _fit(unequal, control_group="not_yet_treated")
    permuted = _fit(unequal.sample(frac=1.0, random_state=91), control_group="not_yet_treated")

    assert unequal.groupby("time").size().nunique() > 1
    pd.testing.assert_frame_equal(reference.group_time, permuted.group_time)
    pd.testing.assert_frame_equal(reference.event_study, permuted.event_study)
    pd.testing.assert_frame_equal(reference.calendar_time, permuted.calendar_time)
    pd.testing.assert_series_equal(
        reference.overall_influence.sort_index(), permuted.overall_influence.sort_index()
    )
    assert reference.design_fingerprint == permuted.design_fingerprint


def test_anticipation_and_repeated_cross_section_pretrend_use_independent_cells() -> None:
    rows: list[dict[str, float | str]] = []
    for cohort, level in ((4.0, 10.0), (np.inf, 20.0)):
        for time in (1.0, 2.0, 3.0, 4.0):
            effect = 3.0 if cohort == 4.0 and time == 4.0 else 0.0
            for replicate, deviation in enumerate((-1.0, 1.0)):
                rows.append(
                    {
                        "row": f"g{cohort}_t{time}_r{replicate}",
                        "outcome": level + time + effect + deviation,
                        "time": time,
                        "treatment_time": cohort,
                    }
                )
    data = pd.DataFrame(rows).set_index("row")

    result = _fit(data, anticipation=1)

    assert result.group_time["event_time"].to_list() == [-1, 0]
    assert result.group_time["att"].to_list() == pytest.approx([0.0, 3.0])
    assert result.pretrend.n_restrictions == 1
    assert result.pretrend.placebo_effects.index.to_list() == [(4.0, 2.0)]
    assert result.pretrend.placebo_effects.iloc[0]["att"] == pytest.approx(0.0)
    assert result.pretrend.influence.index.equals(data.index)
    assert result.pretrend.covariance.iloc[0, 0] > 0.0


def test_clustered_cr1_covariance_is_reconstructed_from_psu_scores() -> None:
    data = _two_by_two().assign(psu=["a", "b", "a", "b", "a", "b", "b", "a"])

    result = RepeatedCrossSectionDiD(covariance="clustered").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        cluster="psu",
    )

    scores = result.overall_influence.groupby(data["psu"], sort=False).sum().to_numpy()
    expected_variance = 2.0 / (2.0 - 1.0) * float(scores @ scores) / len(data) ** 2
    assert result.standard_error**2 == pytest.approx(expected_variance)
    assert result.inference_distribution == "t"
    assert result.inference_df == 1.0
    assert result.n_clusters == 2
    assert result.sampling_unit == "cluster"


def test_observation_multiplier_band_matches_hand_computed_rademacher_max_t() -> None:
    iterations = 513
    seed = 20_260_730
    level = 0.90
    result = _fit(
        _staggered_sample(),
        inference="multiplier_bootstrap",
        bootstrap_iterations=iterations,
        random_state=seed,
        simultaneous_level=level,
    )
    influence = result.event_study_influence.to_numpy(dtype=float)
    standard_errors = result.event_study["std_err"].to_numpy(dtype=float)
    multipliers = np.random.default_rng(seed).choice(
        np.array([-1.0, 1.0]), size=(iterations, result.n_observations)
    )
    draws = (
        np.sqrt(result.n_observations / (result.n_observations - 1))
        * multipliers
        @ influence
        / result.n_observations
    )
    expected_critical = np.quantile(
        np.max(np.abs(draws / standard_errors), axis=1), level, method="higher"
    )

    assert result.inference_method == "multiplier_bootstrap"
    assert result.simultaneous_level == level
    assert result.simultaneous_critical_value == pytest.approx(expected_critical, abs=1e-14)
    assert result.bootstrap_iterations == iterations
    assert result.bootstrap_random_state == seed
    np.testing.assert_allclose(
        result.simultaneous_event_study["lower"],
        result.event_study["att"] - expected_critical * standard_errors,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.simultaneous_event_study["upper"],
        result.event_study["att"] + expected_critical * standard_errors,
        rtol=0.0,
        atol=1e-14,
    )


def test_multiplier_band_is_seeded_and_analytic_path_retains_no_band() -> None:
    model = RepeatedCrossSectionDiD(
        inference="multiplier_bootstrap",
        bootstrap_iterations=101,
        random_state=733,
    )
    first = model.fit(
        _staggered_sample(), outcome="outcome", time="time", treatment_time="treatment_time"
    )
    second = model.fit(
        _staggered_sample(), outcome="outcome", time="time", treatment_time="treatment_time"
    )
    analytic = _fit(_staggered_sample())

    pd.testing.assert_frame_equal(first.simultaneous_event_study, second.simultaneous_event_study)
    assert first.simultaneous_critical_value == second.simultaneous_critical_value
    assert analytic.simultaneous_event_study.empty
    assert analytic.simultaneous_level is None
    assert analytic.simultaneous_critical_value is None
    assert analytic.bootstrap_iterations is None
    assert analytic.bootstrap_random_state is None


def test_psu_multiplier_band_uses_one_draw_per_indivisible_cluster() -> None:
    data = _staggered_sample()
    replicate_zero = data.index.str.endswith("r0")
    flipped_cells = (data["treatment_time"] == 2.0) & (data["time"] >= 2.0)
    data["psu"] = np.where(replicate_zero ^ flipped_cells, "a", "b")
    iterations = 199
    seed = 91
    level = 0.95
    result = RepeatedCrossSectionDiD(
        covariance="clustered",
        inference="multiplier_bootstrap",
        bootstrap_iterations=iterations,
        random_state=seed,
        simultaneous_level=level,
    ).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        cluster="psu",
    )
    cluster_scores = (
        result.event_study_influence.groupby(data["psu"], sort=False).sum().to_numpy(dtype=float)
    )
    multipliers = np.random.default_rng(seed).choice(
        np.array([-1.0, 1.0]), size=(iterations, result.n_clusters)
    )
    draws = (
        np.sqrt(result.n_clusters / (result.n_clusters - 1))
        * multipliers
        @ cluster_scores
        / result.n_observations
    )
    expected_critical = np.quantile(
        np.max(np.abs(draws / result.event_study["std_err"].to_numpy(dtype=float)), axis=1),
        level,
        method="higher",
    )

    assert result.sampling_unit == "cluster"
    assert result.n_clusters == 2
    assert result.simultaneous_critical_value == pytest.approx(expected_critical, abs=1e-14)


@pytest.mark.parametrize(
    ("constructor", "error", "message"),
    [
        ({"inference": "ordinary_bootstrap"}, ValueError, "inference"),
        (
            {"inference": "multiplier_bootstrap", "bootstrap_iterations": 20},
            ValueError,
            "at least 99",
        ),
        (
            {"inference": "multiplier_bootstrap", "bootstrap_iterations": True},
            ValueError,
            "at least 99",
        ),
        (
            {"inference": "multiplier_bootstrap", "bootstrap_iterations": 99.0},
            ValueError,
            "at least 99",
        ),
        (
            {"inference": "multiplier_bootstrap", "random_state": 1.5},
            TypeError,
            "random_state",
        ),
        (
            {"inference": "multiplier_bootstrap", "random_state": True},
            TypeError,
            "random_state",
        ),
        (
            {"inference": "multiplier_bootstrap", "simultaneous_level": 1.0},
            ValueError,
            "strictly between",
        ),
        (
            {"inference": "multiplier_bootstrap", "simultaneous_level": "0.95"},
            TypeError,
            "simultaneous_level",
        ),
        (
            {"inference": "multiplier_bootstrap", "simultaneous_level": True},
            TypeError,
            "simultaneous_level",
        ),
        (
            {"inference": "multiplier_bootstrap", "simultaneous_level": np.nan},
            ValueError,
            "strictly between",
        ),
    ],
)
def test_multiplier_band_configuration_refuses_invalid_contracts(
    constructor: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        RepeatedCrossSectionDiD(**constructor)


def test_multiplier_band_refuses_nonpositive_event_standard_error() -> None:
    data = _staggered_sample()
    data["outcome"] = (
        data["time"]
        + data["treatment_time"].replace(np.inf, 0.0)
        + 2.0 * ((data["treatment_time"] == 2.0) & (data["time"] >= 2.0)).astype(float)
    )
    with pytest.raises(ValueError, match="positive finite standard error"):
        _fit(
            data,
            inference="multiplier_bootstrap",
            bootstrap_iterations=99,
            random_state=1,
        )


@pytest.mark.simulation
def test_observation_multiplier_band_has_seeded_joint_coverage_smoke() -> None:
    repetitions = 100
    jointly_covered = 0
    truth = np.array([1.5, 1.5])
    for seed in range(repetitions):
        rng = np.random.default_rng(seed)
        rows: list[dict[str, float]] = []
        for cohort, level in ((2.0, 0.0), (3.0, 0.4), (np.inf, -0.2)):
            for period in (1.0, 2.0, 3.0):
                effect = 0.0
                if cohort == 2.0 and period == 2.0:
                    effect = 1.0
                elif cohort == 2.0 and period == 3.0:
                    effect = 1.5
                elif cohort == 3.0 and period == 3.0:
                    effect = 2.0
                for outcome in level + 0.3 * period + effect + rng.normal(size=80):
                    rows.append(
                        {
                            "outcome": float(outcome),
                            "time": period,
                            "treatment_time": cohort,
                        }
                    )
        result = _fit(
            pd.DataFrame(rows),
            inference="multiplier_bootstrap",
            bootstrap_iterations=199,
            random_state=seed + 10_000,
        )
        band = result.simultaneous_event_study
        jointly_covered += int(
            np.all(truth >= band["lower"].to_numpy(dtype=float))
            and np.all(truth <= band["upper"].to_numpy(dtype=float))
        )

    assert jointly_covered >= 88


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        ({"control_group": "already_treated"}, "control_group"),
        ({"composition": "dynamic"}, "composition"),
        ({"anticipation": -1}, "anticipation"),
        ({"anticipation": 0.5}, "anticipation"),
        ({"covariance": "homoskedastic"}, "covariance"),
        ({"inference": "ordinary_bootstrap"}, "inference"),
    ],
)
def test_constructor_refuses_unsupported_contracts(
    constructor: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=message):
        RepeatedCrossSectionDiD(**constructor)


def test_fit_signature_has_no_entity_or_panel_role() -> None:
    parameters = inspect.signature(RepeatedCrossSectionDiD.fit).parameters

    assert "entity" not in parameters
    assert "panel" not in parameters


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.rename(columns={"outcome": "time"}), "unique"),
        (lambda frame: frame.set_axis([0, 0, 1, 2, 3, 4, 5, 6]), "index"),
        (
            lambda frame: frame.assign(
                outcome=lambda value: value["outcome"].mask(value.index == "tp0")
            ),
            "missing",
        ),
        (
            lambda frame: frame.assign(
                outcome=lambda value: value["outcome"].mask(value.index == "tp0", np.inf)
            ),
            "finite",
        ),
        (
            lambda frame: frame.assign(time=lambda value: value["time"].mask(value.index == "tp0")),
            "missing",
        ),
        (
            lambda frame: frame.assign(
                treatment_time=lambda value: value["treatment_time"].mask(value.index == "tp0")
            ),
            "missing",
        ),
        (
            lambda frame: frame.assign(
                treatment_time=lambda value: value["treatment_time"].replace(2.0, 3.0)
            ),
            "observed time",
        ),
        (lambda frame: frame.assign(treatment_time=np.inf), "treated cohort"),
        (lambda frame: frame.loc[frame.index != "tp0"], "at least two observations"),
        (lambda frame: frame.loc[frame.index != "cp0"], "at least two observations"),
    ],
)
def test_fit_refuses_malformed_roles_and_unsupported_cells(mutate, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _fit(mutate(_two_by_two()))


def test_fit_refuses_sampling_weights_and_covariates_without_cross_fitter() -> None:
    data = _two_by_two().assign(weight=1.0, baseline=np.arange(8.0))
    estimator = RepeatedCrossSectionDiD()

    with pytest.raises(NotImplementedError, match="sampling weights"):
        estimator.fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            sampling_weights="weight",
        )
    with pytest.raises(ValueError, match="cross_fitter is required"):
        estimator.fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=["baseline"],
        )


def test_never_treated_sentinel_must_be_distinct_from_time_support() -> None:
    with pytest.raises(ValueError, match="must not equal an observed time"):
        RepeatedCrossSectionDiD().fit(
            _two_by_two().assign(
                treatment_time=lambda frame: frame["treatment_time"].replace(np.inf, 1.0)
            ),
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            never_treated=1.0,
        )


def test_clustered_covariance_requires_a_valid_declared_psu() -> None:
    data = _two_by_two()
    estimator = RepeatedCrossSectionDiD(covariance="clustered")

    with pytest.raises(ValueError, match="cluster column"):
        estimator.fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
        )
    with pytest.raises(ValueError, match="at least two clusters"):
        estimator.fit(
            data.assign(psu="only"),
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            cluster="psu",
        )


def test_robust_covariance_refuses_unused_cluster_role() -> None:
    with pytest.raises(ValueError, match="covariance='clustered'"):
        RepeatedCrossSectionDiD().fit(
            _two_by_two().assign(psu=np.tile(["a", "b"], 4)),
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            cluster="psu",
        )


@pytest.mark.validation
def test_base_r_reconstructs_hand_estimate_influence_and_hc1() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not available for the cross-language hand contract.")
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [rscript, str(repository / "benchmarks" / "validate_did_rcs_reference.R")],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository,
    )
    fields = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    result = _fit(_two_by_two())

    assert fields["parity_status"] == "pass"
    assert float(fields["estimate"]) == pytest.approx(result.estimate, abs=1e-12)
    assert float(fields["standard_error_hc1"]) == pytest.approx(result.standard_error, abs=1e-12)
    np.testing.assert_allclose(
        np.fromstring(fields["influence"], sep=","),
        result.overall_influence,
        atol=1e-12,
    )


@pytest.mark.simulation
def test_two_by_two_bias_and_hc1_coverage_smoke() -> None:
    estimates: list[float] = []
    covered = 0
    true_effect = 2.0
    repetitions = 80
    for seed in range(repetitions):
        rng = np.random.default_rng(seed)
        rows: list[dict[str, float]] = []
        for cohort, treated_group in ((2.0, True), (np.inf, False)):
            for time in (1.0, 2.0):
                mean = 0.5 * time + (true_effect if treated_group and time == 2.0 else 0.0)
                for outcome in mean + rng.normal(size=80):
                    rows.append(
                        {
                            "outcome": float(outcome),
                            "time": time,
                            "treatment_time": cohort,
                        }
                    )
        result = _fit(pd.DataFrame(rows))
        interval = result.conf_int()
        estimates.append(result.estimate)
        covered += int(interval["lower"] <= true_effect <= interval["upper"])

    assert float(np.mean(estimates)) == pytest.approx(true_effect, abs=0.06)
    assert covered >= 70

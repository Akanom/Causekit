"""Failing-first contracts for direct cohort-odds PT-All nuisances."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from causekit import (
    CohortOddsRatioCrossFitResult,
    CohortOddsRatioResultProtocol,
    CrossFitter,
    EfficientDiD,
    add_to_outputhub,
    to_outputhub_model,
)

FIT_COLUMNS = {
    "outcome": "outcome",
    "entity": "entity",
    "time": "time",
    "treatment_time": "treatment_time",
}


class _ClassProbabilityResult:
    def __init__(self, classes: np.ndarray, probabilities: np.ndarray) -> None:
        self.classes_ = classes
        self.probabilities = probabilities

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile(self.probabilities, (len(X), 1))


class _ClassProbability:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ClassProbabilityResult:
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        return _ClassProbabilityResult(classes, counts / counts.sum())


class _ConstantOddsResult:
    odds_ratio_kind_ = "posterior_cohort_odds"
    numerator_class_ = 1
    denominator_class_ = 0

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def predict_odds_ratio(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.ratio, index=X.index, name="odds_ratio")


class _ConstantOdds:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ConstantOddsResult:
        probability = float(np.mean(y))
        return _ConstantOddsResult(probability / (1.0 - probability))


class _LinearResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        return design @ self.coefficients


class _LinearRegression:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _LinearResult:
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        coefficients = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)[0]
        return _LinearResult(coefficients)


class _ZeroResult:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class _ZeroRegression:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ZeroResult:
        return _ZeroResult()


def _efficient_panel(repetitions: int = 12) -> pd.DataFrame:
    h1 = np.array([1.0, 1.0, -1.0, -1.0])
    h2 = np.array([1.0, -1.0, 1.0, -1.0])
    h3 = np.array([1.0, -1.0, -1.0, 1.0])
    rows: list[dict[str, float | str]] = []
    for repeat in range(repetitions):
        for position in range(4):
            path = (
                -(10.0 + h1[position]),
                -(10.0 + 2 * h2[position]),
                -(10.0 + 4 * h3[position]),
                0.0,
            )
            for period, outcome in enumerate(path, start=1):
                rows.append(
                    {
                        "entity": f"treated_{repeat}_{position}",
                        "time": period,
                        "treatment_time": 4.0,
                        "x": float(repeat % 3),
                        "outcome": outcome,
                    }
                )
        for position in range(4):
            for period in range(1, 5):
                rows.append(
                    {
                        "entity": f"never_{repeat}_{position}",
                        "time": period,
                        "treatment_time": np.inf,
                        "x": float(repeat % 3),
                        "outcome": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _three_cohort_panel() -> pd.DataFrame:
    rng = np.random.default_rng(20260730)
    rows: list[dict[str, float | str]] = []
    for cohort, prefix, size, effects in (
        (2.0, "g2", 36, {2: 2.0, 3: 2.5}),
        (3.0, "g3", 30, {3: 3.0}),
        (np.inf, "never", 24, {}),
    ):
        first_change = rng.normal(loc=0.5, scale=0.8, size=size)
        second_change = rng.normal(loc=0.7, scale=0.9, size=size)
        baseline = rng.normal(loc=1.0, scale=0.5, size=size)
        for position in range(size):
            untreated = (
                baseline[position],
                baseline[position] + first_change[position],
                baseline[position] + first_change[position] + second_change[position],
            )
            for period, value in enumerate(untreated, start=1):
                rows.append(
                    {
                        "entity": f"{prefix}_{position:02d}",
                        "time": float(period),
                        "treatment_time": cohort,
                        "x": 0.0,
                        "outcome": value + effects.get(period, 0.0),
                    }
                )
    return pd.DataFrame(rows)


def _multiclass_fitter(*, seed: int = 19) -> CrossFitter:
    return CrossFitter(
        propensity_factory=_ClassProbability,
        outcome_factory=_LinearRegression,
        n_splits=3,
        random_state=seed,
    )


def _direct_fitter(
    *,
    seed: int = 19,
    ratio_factory: Callable[[], object] = _ConstantOdds,
) -> CrossFitter:
    return CrossFitter(
        cohort_ratio_factory=ratio_factory,
        outcome_factory=_LinearRegression,
        n_splits=3,
        random_state=seed,
    )


def _fit(data: pd.DataFrame, fitter: CrossFitter, **model_options):
    return EfficientDiD(**model_options).fit(
        data,
        **FIT_COLUMNS,
        covariates=["x"],
        cross_fitter=fitter,
    )


def test_public_direct_ratio_protocol_and_pairwise_crossfit_contract() -> None:
    X = pd.DataFrame({"x": np.arange(18.0)}, index=pd.Index(range(100, 118)))
    cohorts = pd.Series(np.repeat([2.0, 4.0, np.inf], 6), index=X.index)
    result = CrossFitter(
        cohort_ratio_factory=_ConstantOdds,
        n_splits=3,
        random_state=5,
    ).fit_predict_cohort_odds_ratios(
        X,
        cohorts=cohorts,
        pairs=[(2.0, np.inf), (4.0, 2.0), (np.inf, 2.0)],
    )

    assert isinstance(_ConstantOddsResult(1.0), CohortOddsRatioResultProtocol)
    assert isinstance(result, CohortOddsRatioCrossFitResult)
    assert result.ratios.index.equals(X.index)
    assert result.ratios.columns.names == ["numerator", "denominator"]
    pd.testing.assert_series_equal(result.cohorts, cohorts.rename("cohort"))
    assert result.ratios.columns.tolist() == [
        (2.0, np.inf),
        (4.0, 2.0),
        (np.inf, 2.0),
    ]
    np.testing.assert_allclose(result.ratios, 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        result.ratios[(2.0, np.inf)] * result.ratios[(np.inf, 2.0)],
        1.0,
        rtol=0,
        atol=1e-14,
    )
    assert set(result.fold.unique()) == {0, 1, 2}
    assert len(result.pair_diagnostics) == 9
    assert result.pair_diagnostics["denominator_importance_effective_n"].ge(2.0).all()
    assert result.pair_diagnostics["denominator_importance_max_share"].le(0.5).all()


def test_direct_ratio_reproduces_multiclass_score_influence_and_omega_weights() -> None:
    data = _efficient_panel()
    multiclass = _fit(data, _multiclass_fitter())
    direct = _fit(data, _direct_fitter())

    assert direct.nuisance_weighting == "direct_cohort_odds"
    assert multiclass.nuisance_weighting == "multiclass_probabilities"
    assert direct.cohort_probabilities.empty
    assert not direct.cohort_ratios.empty
    assert direct.cohort_ratios.columns.names == ["numerator", "denominator"]
    assert direct.cohort_ratio_diagnostics["candidate_use_count"].ge(1).all()
    pd.testing.assert_series_equal(direct.nuisance_fold, multiclass.nuisance_fold)
    pd.testing.assert_frame_equal(direct.group_time, multiclass.group_time, atol=2e-12)
    pd.testing.assert_frame_equal(
        direct.candidate_influence_functions,
        multiclass.candidate_influence_functions,
        atol=2e-12,
    )
    pd.testing.assert_frame_equal(
        direct.conditional_efficiency_weights,
        multiclass.conditional_efficiency_weights,
        atol=2e-12,
    )
    pd.testing.assert_frame_equal(
        direct.event_study_influence,
        multiclass.event_study_influence,
        atol=2e-12,
    )
    assert direct.estimate == pytest.approx(multiclass.estimate, abs=2e-12)
    assert direct.standard_error == pytest.approx(multiclass.standard_error, abs=2e-12)


def test_omega_tilde_has_the_same_normalized_solve_as_omega() -> None:
    omega_tilde = np.array(
        [
            [[3.0, 0.4, 0.2], [0.4, 2.0, 0.1], [0.2, 0.1, 1.5]],
            [[1.8, 0.2, 0.1], [0.2, 2.4, 0.3], [0.1, 0.3, 2.0]],
        ]
    )
    target_probability = np.array([0.2, 0.65])
    omega = omega_tilde / target_probability[:, None, None]
    ones = np.ones((2, 3))
    raw = np.linalg.solve(omega, ones[..., None])[..., 0]
    raw_tilde = np.linalg.solve(omega_tilde, ones[..., None])[..., 0]

    np.testing.assert_allclose(
        raw / raw.sum(axis=1, keepdims=True),
        raw_tilde / raw_tilde.sum(axis=1, keepdims=True),
        rtol=0,
        atol=1e-14,
    )


def test_nonidentity_auxiliary_cohort_odds_match_multiclass_omega_and_scores() -> None:
    data = _three_cohort_panel()
    multiclass = _fit(data, _multiclass_fitter(seed=29))
    direct = _fit(data, _direct_fitter(seed=29))

    assert (2.0, 3.0) in direct.cohort_ratios.columns
    assert (2.0, np.inf) in direct.cohort_ratios.columns
    assert (3.0, np.inf) in direct.cohort_ratios.columns
    np.testing.assert_allclose(direct.cohort_ratios[(2.0, 3.0)], 36 / 30, atol=1e-14)
    np.testing.assert_allclose(direct.cohort_ratios[(2.0, np.inf)], 36 / 24, atol=1e-14)
    np.testing.assert_allclose(direct.cohort_ratios[(3.0, np.inf)], 30 / 24, atol=1e-14)
    pd.testing.assert_frame_equal(
        direct.candidate_influence_functions,
        multiclass.candidate_influence_functions,
        atol=2e-12,
    )
    pd.testing.assert_frame_equal(
        direct.conditional_efficiency_weights,
        multiclass.conditional_efficiency_weights,
        atol=2e-12,
    )
    pd.testing.assert_frame_equal(direct.group_time, multiclass.group_time, atol=2e-12)


def test_direct_ratio_clustered_cr1_matches_multiclass_on_immutable_cluster_folds() -> None:
    data = _three_cohort_panel()
    entity_number = data["entity"].str.extract(r"_(\d+)$", expand=False).astype(int)
    data["cluster"] = (
        data["entity"].str.extract(r"^([^_]+)", expand=False)
        + "_"
        + (entity_number // 2).astype(str)
    )
    fit_options = {
        **FIT_COLUMNS,
        "cluster": "cluster",
        "covariates": ["x"],
    }
    multiclass = EfficientDiD(covariance="clustered").fit(
        data,
        **fit_options,
        cross_fitter=_multiclass_fitter(seed=33),
    )
    direct = EfficientDiD(covariance="clustered").fit(
        data,
        **fit_options,
        cross_fitter=_direct_fitter(seed=33),
    )

    assert direct.inference_distribution == "t"
    assert direct.n_clusters == data["cluster"].nunique()
    assert direct.nuisance_fold.groupby(direct.inference_clusters).nunique().eq(1).all()
    pd.testing.assert_frame_equal(direct.group_time, multiclass.group_time, atol=2e-12)
    pd.testing.assert_frame_equal(
        direct.group_time_influence,
        multiclass.group_time_influence,
        atol=2e-12,
    )


def test_calibrated_posterior_odds_retain_the_cohort_prior() -> None:
    X = pd.DataFrame({"x": np.arange(18.0)})
    cohorts = pd.Series(np.repeat([2.0, np.inf], [12, 6]), index=X.index)
    result = CrossFitter(
        cohort_ratio_factory=_ConstantOdds,
        n_splits=3,
        random_state=11,
    ).fit_predict_cohort_odds_ratios(X, cohorts=cohorts, pairs=[(2.0, np.inf)])

    # Equal conditional densities would give a raw density ratio of one. Posterior
    # cohort odds retain the 2:1 prior odds and are scale-sensitive in the PT-All score.
    np.testing.assert_allclose(result.ratios[(2.0, np.inf)], 2.0, rtol=0, atol=1e-14)


def test_omitting_prior_odds_changes_the_scale_sensitive_pt_all_score() -> None:
    rows: list[dict[str, float | str]] = []
    for position in range(12):
        for period, outcome in ((1.0, 0.0), (2.0, 3.0)):
            rows.append(
                {
                    "entity": f"g_{position:02d}",
                    "time": period,
                    "treatment_time": 2.0,
                    "x": 0.0,
                    "outcome": outcome,
                }
            )
    for position in range(6):
        for period, outcome in ((1.0, 0.0), (2.0, float(2 * (position % 2)))):
            rows.append(
                {
                    "entity": f"n_{position:02d}",
                    "time": period,
                    "treatment_time": np.inf,
                    "x": 0.0,
                    "outcome": outcome,
                }
            )
    data = pd.DataFrame(rows)

    class UnitOdds(_ConstantOdds):
        def fit(self, X, y):
            return _ConstantOddsResult(1.0)

    correct = _fit(
        data,
        CrossFitter(
            cohort_ratio_factory=_ConstantOdds,
            outcome_factory=_ZeroRegression,
            n_splits=3,
            random_state=11,
        ),
    )
    omitted_prior = _fit(
        data,
        CrossFitter(
            cohort_ratio_factory=UnitOdds,
            outcome_factory=_ZeroRegression,
            n_splits=3,
            random_state=11,
        ),
    )

    assert correct.estimate == pytest.approx(2.0, abs=1e-14)
    assert omitted_prior.estimate == pytest.approx(2.5, abs=1e-14)
    assert omitted_prior.estimate - correct.estimate == pytest.approx(0.5, abs=1e-14)


class _DeclaredDensityRatioResult(_ConstantOddsResult):
    odds_ratio_kind_ = "group_conditional_density_ratio"


class _DeclaredDensityRatio:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _DeclaredDensityRatioResult:
        return _DeclaredDensityRatioResult(1.0)


def test_declared_uncalibrated_density_ratio_is_refused() -> None:
    X = pd.DataFrame({"x": np.arange(12.0)})
    cohorts = pd.Series(np.repeat([2.0, np.inf], 6), index=X.index)
    with pytest.raises(ValueError, match="posterior cohort odds"):
        CrossFitter(
            cohort_ratio_factory=_DeclaredDensityRatio,
            n_splits=2,
        ).fit_predict_cohort_odds_ratios(X, cohorts=cohorts, pairs=[(2.0, np.inf)])


def test_missing_protocol_and_reversed_binary_orientation_are_refused() -> None:
    class MissingPrediction:
        def fit(self, X, y):
            return object()

    class ReversedResult(_ConstantOddsResult):
        numerator_class_ = 0
        denominator_class_ = 1

    class ReversedOrientation:
        def fit(self, X, y):
            return ReversedResult(1.0)

    X = pd.DataFrame({"x": np.arange(12.0)})
    cohorts = pd.Series(np.repeat([2.0, np.inf], 6), index=X.index)
    with pytest.raises(TypeError, match="predict_odds_ratio"):
        CrossFitter(
            cohort_ratio_factory=MissingPrediction,
            n_splits=2,
        ).fit_predict_cohort_odds_ratios(X, cohorts=cohorts, pairs=[(2.0, np.inf)])
    with pytest.raises(ValueError, match="orientation"):
        CrossFitter(
            cohort_ratio_factory=ReversedOrientation,
            n_splits=2,
        ).fit_predict_cohort_odds_ratios(X, cohorts=cohorts, pairs=[(2.0, np.inf)])


def test_explicit_ratio_prediction_adapter_preserves_labelled_alignment() -> None:
    class AdapterResult:
        odds_ratio_kind_ = "posterior_cohort_odds"

        def __init__(self, ratio: float) -> None:
            self.ratio = ratio

    class AdapterEstimator:
        def fit(self, X, y):
            probability = float(y.mean())
            return AdapterResult(probability / (1.0 - probability))

    def adapter(result, X):
        return pd.Series(result.ratio, index=X.index)

    X = pd.DataFrame({"x": np.arange(18.0)})
    cohorts = pd.Series(np.repeat([2.0, np.inf], [12, 6]), index=X.index)
    result = CrossFitter(
        cohort_ratio_factory=AdapterEstimator,
        cohort_ratio_predict=adapter,
        n_splits=3,
        random_state=11,
    ).fit_predict_cohort_odds_ratios(X, cohorts=cohorts, pairs=[(2.0, np.inf)])

    np.testing.assert_allclose(result.ratios, 2.0, rtol=0, atol=1e-14)


def test_direct_pair_avoids_irrelevant_class_probability_underflow() -> None:
    class UnderflowResult:
        def __init__(self, classes) -> None:
            self.classes_ = np.asarray(classes)

        def predict_proba(self, X):
            values = np.zeros((len(X), 3))
            values[:, 1] = 1.0
            return pd.DataFrame(values, index=X.index, columns=self.classes_)

    class UnderflowMulticlass:
        def fit(self, X, y):
            return UnderflowResult(np.unique(y))

    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 36)})
    cohorts = pd.Series(np.repeat([2.0, 3.0, np.inf], 12), index=X.index)
    multiclass = CrossFitter(
        propensity_factory=UnderflowMulticlass,
        n_splits=3,
        random_state=8,
    ).fit_predict_class_probabilities(X, classes=cohorts)
    direct = CrossFitter(
        cohort_ratio_factory=_ConstantOdds,
        n_splits=3,
        random_state=8,
    ).fit_predict_cohort_odds_ratios(
        X,
        cohorts=cohorts,
        pairs=[(2.0, np.inf), (2.0, 3.0)],
    )

    assert (multiclass.probabilities[[2.0, np.inf]] == 0.0).all().all()
    np.testing.assert_allclose(direct.ratios, 1.0, rtol=0, atol=1e-14)
    assert np.isfinite(direct.ratios).all().all()


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (lambda X: np.full(len(X), np.nan), "finite"),
        (lambda X: np.zeros(len(X)), "strictly positive"),
        (lambda X: -np.ones(len(X)), "strictly positive"),
        (lambda X: np.ones(len(X) - 1), "one value per held-out row"),
        (lambda X: np.full(len(X), 1.0 + 1.0j), "real numeric"),
        (lambda X: pd.Series(1.0, index=X.index[::-1]), "index"),
    ],
)
def test_invalid_direct_ratio_predictions_are_refused(prediction, message: str) -> None:
    class InvalidResult:
        odds_ratio_kind_ = "posterior_cohort_odds"

        def predict_odds_ratio(self, X):
            return prediction(X)

    class InvalidEstimator:
        def fit(self, X, y):
            return InvalidResult()

    X = pd.DataFrame({"x": np.arange(12.0)})
    cohorts = pd.Series(np.repeat([2.0, np.inf], 6), index=X.index)
    with pytest.raises((TypeError, ValueError), match=message):
        CrossFitter(
            cohort_ratio_factory=InvalidEstimator,
            n_splits=2,
        ).fit_predict_cohort_odds_ratios(X, cohorts=cohorts, pairs=[(2.0, np.inf)])


def test_direct_ratio_factories_pairs_and_fold_support_are_strict() -> None:
    X = pd.DataFrame({"x": np.arange(12.0)})
    cohorts = pd.Series(np.repeat([2.0, np.inf], 6), index=X.index)
    singleton = _ConstantOdds()
    with pytest.raises(ValueError, match="fresh estimator"):
        CrossFitter(
            cohort_ratio_factory=lambda: singleton,
            n_splits=2,
        ).fit_predict_cohort_odds_ratios(X, cohorts=cohorts, pairs=[(2.0, np.inf)])
    with pytest.raises(ValueError, match="distinct"):
        CrossFitter(
            cohort_ratio_factory=_ConstantOdds,
            n_splits=2,
        ).fit_predict_cohort_odds_ratios(X, cohorts=cohorts, pairs=[(2.0, 2.0)])
    with pytest.raises(ValueError, match="observed cohort"):
        CrossFitter(
            cohort_ratio_factory=_ConstantOdds,
            n_splits=2,
        ).fit_predict_cohort_odds_ratios(X, cohorts=cohorts, pairs=[(2.0, 99.0)])
    with pytest.raises(ValueError, match="unique"):
        CrossFitter(
            cohort_ratio_factory=_ConstantOdds,
            n_splits=2,
        ).fit_predict_cohort_odds_ratios(
            X,
            cohorts=cohorts,
            pairs=[(2.0, np.inf), (2.0, np.inf)],
        )
    with pytest.raises(ValueError, match="at least n_splits"):
        CrossFitter(
            cohort_ratio_factory=_ConstantOdds,
            n_splits=4,
        ).fit_predict_cohort_odds_ratios(
            X.iloc[:8],
            cohorts=pd.Series([2.0] * 6 + [np.inf] * 2, index=X.index[:8]),
            pairs=[(2.0, np.inf)],
        )


def test_direct_ratio_cluster_roles_are_immutable_and_audited() -> None:
    X = pd.DataFrame({"x": np.arange(24.0)})
    cohorts = pd.Series(np.repeat([2.0, np.inf], 12), index=X.index)
    clusters = pd.Series(
        [f"g-{value}" for value in np.repeat(np.arange(6), 2)]
        + [f"n-{value}" for value in np.repeat(np.arange(6), 2)],
        index=X.index,
    )
    result = CrossFitter(
        cohort_ratio_factory=_ConstantOdds,
        n_splits=3,
        random_state=31,
    ).fit_predict_cohort_odds_ratios(
        X,
        cohorts=cohorts,
        pairs=[(2.0, np.inf)],
        clusters=clusters,
    )

    assert result.fold.groupby(clusters).nunique().eq(1).all()
    pd.testing.assert_series_equal(result.clusters, clusters.rename("cluster"))
    assert result.pair_diagnostics["train_numerator_psus"].ge(4).all()
    assert result.pair_diagnostics["train_denominator_psus"].ge(4).all()
    assert result.pair_diagnostics["holdout_numerator_psus"].ge(2).all()
    assert result.pair_diagnostics["holdout_denominator_psus"].ge(2).all()
    assert result.pair_diagnostics["train_index_hash"].str.len().eq(64).all()
    assert result.pair_diagnostics["holdout_index_hash"].str.len().eq(64).all()


def test_efficient_did_refuses_ambiguous_missing_and_weak_direct_weighting_routes() -> None:
    data = _efficient_panel()
    ambiguous = CrossFitter(
        propensity_factory=_ClassProbability,
        cohort_ratio_factory=_ConstantOdds,
        outcome_factory=_LinearRegression,
        n_splits=3,
    )
    with pytest.raises(ValueError, match="exactly one cohort-weighting route"):
        _fit(data, ambiguous)
    missing = CrossFitter(outcome_factory=_LinearRegression, n_splits=3)
    with pytest.raises(ValueError, match="exactly one cohort-weighting route"):
        _fit(data, missing)

    class TinyOdds(_ConstantOdds):
        def fit(self, X, y):
            return _ConstantOddsResult(1e-12)

    with pytest.raises(ValueError, match="nuisance_ratio_floor"):
        _fit(data, _direct_fitter(ratio_factory=TinyOdds))

    class HugeOdds(_ConstantOdds):
        def fit(self, X, y):
            return _ConstantOddsResult(1e7)

    with pytest.raises(ValueError, match="nuisance_ratio_ceiling"):
        _fit(data, _direct_fitter(ratio_factory=HugeOdds))
    with pytest.raises(ValueError, match="nuisance_ratio_min_effective_n"):
        _fit(
            data,
            _direct_fitter(),
            nuisance_ratio_min_effective_n=20.0,
        )
    with pytest.raises(ValueError, match="nuisance_ratio_max_share"):
        _fit(data, _direct_fitter(), nuisance_ratio_max_share=0.01)
    with pytest.raises(ValueError, match="nuisance_ratio_min_psus"):
        _fit(data, _direct_fitter(), nuisance_ratio_min_psus=40)

    singular_covariance = CrossFitter(
        cohort_ratio_factory=_ConstantOdds,
        outcome_factory=_LinearRegression,
        second_moment_factory=_ZeroRegression,
        n_splits=3,
        random_state=19,
    )
    with pytest.raises(ValueError, match="conditional covariance system is singular"):
        _fit(data, singular_covariance)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"nuisance_ratio_floor": 0.0}, "nuisance_ratio_floor"),
        (
            {"nuisance_ratio_floor": 2.0, "nuisance_ratio_ceiling": 1.0},
            "nuisance_ratio_ceiling",
        ),
        ({"nuisance_ratio_min_effective_n": 0.5}, "min_effective_n"),
        ({"nuisance_ratio_max_share": 0.0}, "max_share"),
        ({"nuisance_ratio_min_psus": 1}, "min_psus"),
    ],
)
def test_invalid_direct_ratio_threshold_contracts_are_refused(options, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EfficientDiD(**options)


def test_direct_ratio_training_never_consumes_holdout_or_third_cohort_rows() -> None:
    records: list[tuple[pd.Index, pd.Series]] = []

    class AuditOdds(_ConstantOdds):
        def fit(self, X, y):
            records.append((X.index.copy(), y.copy()))
            return super().fit(X, y)

    X = pd.DataFrame({"x": np.arange(18.0)}, index=pd.Index(range(100, 118)))
    cohorts = pd.Series(np.repeat([2.0, 4.0, np.inf], 6), index=X.index)
    result = CrossFitter(
        cohort_ratio_factory=AuditOdds,
        n_splits=3,
        random_state=22,
    ).fit_predict_cohort_odds_ratios(
        X,
        cohorts=cohorts,
        pairs=[(2.0, np.inf)],
    )

    assert len(records) == 3
    for fold, (train_index, binary_target) in enumerate(records):
        assert train_index.intersection(result.fold.index[result.fold.eq(fold)]).empty
        assert set(cohorts.loc[train_index]) == {2.0, np.inf}
        assert binary_target.index.equals(train_index)
        np.testing.assert_array_equal(
            binary_target.to_numpy(), cohorts.loc[train_index].eq(2.0).astype(float)
        )


def test_direct_ratio_result_is_invariant_to_input_row_order() -> None:
    data = _efficient_panel()
    original = _fit(data, _direct_fitter(seed=27))
    permuted = _fit(
        data.sample(frac=1.0, random_state=20260730),
        _direct_fitter(seed=27),
    )

    pd.testing.assert_frame_equal(original.group_time, permuted.group_time)
    pd.testing.assert_frame_equal(original.cohort_ratios, permuted.cohort_ratios)
    pd.testing.assert_frame_equal(
        original.conditional_efficiency_weights,
        permuted.conditional_efficiency_weights,
    )
    assert original.design_fingerprint == permuted.design_fingerprint


def test_direct_ratio_reuses_the_first_fold_plan_when_seed_is_none() -> None:
    fitter = CrossFitter(
        cohort_ratio_factory=_ConstantOdds,
        outcome_factory=_LinearRegression,
        n_splits=3,
        random_state=None,
    )
    result = _fit(_efficient_panel(), fitter)

    assert set(result.nuisance_fold.unique()) == {0, 1, 2}
    assert (
        result.cohort_ratio_diagnostics.loc[
            result.cohort_ratio_diagnostics["source"].eq("fitted"), "fold"
        ].nunique()
        == 3
    )


def test_direct_ratio_outputhub_preserves_weighting_route_and_audits() -> None:
    outputhub = pytest.importorskip("universal_output_hub")
    result = _fit(_efficient_panel(), _direct_fitter(seed=15))

    model = to_outputhub_model(result)
    assert model.metadata["nuisance_weighting"] == "direct_cohort_odds"
    hub = outputhub.OutputHub("Direct cohort-odds PT-All")
    add_to_outputhub(hub, result)
    names = [table.name for table in hub.tables]
    assert names[-3:] == [
        "Efficient DiD cohort odds",
        "Efficient DiD cohort-odds diagnostics",
        "Efficient DiD cohort-odds candidate uses",
    ]

"""Tests for reusable nuisance-model cross-fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import (
    AIPWATE,
    ClassProbabilityCrossFitResult,
    ClassProbabilityCrossFitTask,
    ClassProbabilityTaskCrossFitResult,
    CrossFitResult,
    CrossFitTask,
    CrossFitTaskResult,
    CrossFitter,
)


class _ConstantPropensityResult:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X):
        probability = np.full(len(X), self.probability)
        return pd.DataFrame({0: 1 - probability, 1: probability}, index=X.index)


class _ConstantPropensity:
    def fit(self, X, y):
        return _ConstantPropensityResult(float(np.mean(y)))


class _LinearResult:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, X):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        return design @ self.coefficients


class _LinearOutcome:
    def fit(self, X, y):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        return _LinearResult(np.linalg.lstsq(design, np.asarray(y), rcond=None)[0])


class _WeightedAuditOutcome:
    records: list[tuple[pd.Index, pd.Series]] = []

    def fit(
        self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series
    ) -> _WeightedAuditOutcome:
        assert X.index.equals(y.index)
        assert X.index.equals(sample_weight.index)
        self.records.append((X.index.copy(), sample_weight.copy()))
        self.mean_ = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.mean_, index=X.index)


class _DiagnosticLinearResult(_LinearResult):
    def nuisance_diagnostics(self):
        return {"selected_alpha": 0.25, "effective_df": 2.5}


class _DiagnosticLinearOutcome:
    def fit(self, X, y):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        coefficients = np.linalg.lstsq(design, np.asarray(y), rcond=None)[0]
        return _DiagnosticLinearResult(coefficients)


class _PayloadDiagnosticLinearResult(_LinearResult):
    def __init__(self, coefficients, payload) -> None:
        super().__init__(coefficients)
        self.payload = payload

    def nuisance_diagnostics(self):
        return self.payload


class _PayloadDiagnosticLinearOutcome:
    def __init__(self, payload) -> None:
        self.payload = payload

    def fit(self, X, y):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        coefficients = np.linalg.lstsq(design, np.asarray(y), rcond=None)[0]
        return _PayloadDiagnosticLinearResult(coefficients, self.payload)


class _ClassProbabilityResult:
    def __init__(self, classes: np.ndarray, probabilities: np.ndarray) -> None:
        self.classes_ = classes
        self.probabilities = probabilities

    def predict_proba(self, X):
        return np.tile(self.probabilities, (len(X), 1))


class _ClassProbability:
    def fit(self, X, y):
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        return _ClassProbabilityResult(classes, counts / counts.sum())


def _data(nobs: int = 300, seed: int = 902):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=nobs), "x2": rng.normal(size=nobs)})
    treatment = pd.Series(rng.binomial(1, 0.5, nobs), index=X.index)
    outcome = pd.Series(
        0.4 + 1.3 * treatment + 0.7 * X["x1"] - 0.3 * X["x2"] + rng.normal(size=nobs),
        index=X.index,
    )
    return X, treatment, outcome


def test_cross_fitter_returns_complete_aligned_out_of_fold_predictions() -> None:
    X, treatment, outcome = _data()
    result = CrossFitter(
        propensity_factory=_ConstantPropensity,
        outcome_factory=_LinearOutcome,
        n_splits=5,
        random_state=41,
    ).fit_predict(X, treatment=treatment, outcome=outcome)

    assert isinstance(result, CrossFitResult)
    assert result.propensity.index.equals(X.index)
    assert result.outcome_treated.index.equals(X.index)
    assert result.outcome_control.index.equals(X.index)
    assert set(result.fold.unique()) == set(range(5))
    assert np.isfinite(result.propensity).all()
    assert len(result.model_diagnostics) == 15
    assert set(result.model_diagnostics["task"]) == {
        "propensity",
        "outcome_treated",
        "outcome_control",
    }
    assert not result.model_diagnostics["diagnostics_available"].any()
    for fold in range(5):
        held_out = result.fold == fold
        assert treatment[held_out].nunique() == 2


def test_cross_fitting_is_deterministic_for_fixed_seed_and_drives_aipw() -> None:
    X, treatment, outcome = _data(nobs=1000)
    fitter = CrossFitter(
        propensity_factory=_ConstantPropensity,
        outcome_factory=_LinearOutcome,
        n_splits=4,
        random_state=7,
    )
    first = fitter.fit_predict(X, treatment=treatment, outcome=outcome)
    second = fitter.fit_predict(X, treatment=treatment, outcome=outcome)
    pd.testing.assert_series_equal(first.fold, second.fold)
    pd.testing.assert_series_equal(first.propensity, second.propensity)
    effect = AIPWATE().fit(
        outcome,
        treatment=treatment,
        propensity=first.propensity,
        outcome_treated=first.outcome_treated,
        outcome_control=first.outcome_control,
    )
    assert effect.estimate == pytest.approx(1.3, abs=0.14)


def test_factories_must_return_fresh_fit_capable_models() -> None:
    X, treatment, outcome = _data(nobs=80)
    fitter = CrossFitter(
        propensity_factory=lambda: object(),
        outcome_factory=_LinearOutcome,
        n_splits=2,
    )
    with pytest.raises(TypeError, match="fit"):
        fitter.fit_predict(X, treatment=treatment, outcome=outcome)

    singleton = _LinearOutcome()
    with pytest.raises(ValueError, match="fresh estimator"):
        CrossFitter(outcome_factory=lambda: singleton, n_splits=2).fit_predict_tasks(
            X,
            tasks=[CrossFitTask(name="outcome", target=outcome)],
            strata=treatment,
        )

    def shifted_adapter(result, heldout):
        return pd.Series(result.predict(heldout), index=heldout.index[::-1])

    with pytest.raises(ValueError, match="prediction index"):
        CrossFitter(n_splits=2).fit_predict_tasks(
            X,
            tasks=[
                CrossFitTask(
                    name="outcome",
                    target=outcome,
                    factory=_LinearOutcome,
                    predict=shifted_adapter,
                )
            ],
            strata=treatment,
        )


def test_cross_fitting_refuses_small_arms_and_index_drift() -> None:
    X, treatment, outcome = _data(nobs=40)
    fitter = CrossFitter(
        propensity_factory=_ConstantPropensity,
        outcome_factory=_LinearOutcome,
        n_splits=5,
    )
    treatment.iloc[:] = 0
    treatment.iloc[:3] = 1
    with pytest.raises(ValueError, match="at least n_splits"):
        fitter.fit_predict(X, treatment=treatment, outcome=outcome)
    with pytest.raises(ValueError, match="indices must match"):
        fitter.fit_predict(
            X,
            treatment=treatment.set_axis(pd.RangeIndex(1, len(treatment) + 1)),
            outcome=outcome,
        )


def test_multiclass_probabilities_are_aligned_out_of_fold_and_sum_to_one() -> None:
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 18)}, index=pd.Index(range(100, 118)))
    classes = pd.Series(np.repeat([2.0, 4.0, np.inf], 6), index=X.index)
    result = CrossFitter(
        propensity_factory=_ClassProbability,
        outcome_factory=_LinearOutcome,
        n_splits=3,
        random_state=19,
    ).fit_predict_class_probabilities(X, classes=classes)

    assert isinstance(result, ClassProbabilityCrossFitResult)
    assert result.probabilities.index.equals(X.index)
    assert result.probabilities.columns.tolist() == [2.0, 4.0, np.inf]
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    assert result.fold.index.equals(X.index)
    assert len(result.model_diagnostics) == 3
    assert result.model_diagnostics["task"].eq("class_probability").all()
    for fold in range(3):
        assert set(classes[result.fold == fold]) == {2.0, 4.0, np.inf}


def test_masked_regression_tasks_share_folds_and_preserve_task_labels() -> None:
    X = pd.DataFrame({"x": np.tile(np.arange(6, dtype=float), 2)})
    strata = pd.Series(np.repeat([0, 1], 6), index=X.index)
    target = pd.Series(1.0 + 2.0 * X["x"], index=X.index)
    tasks = (
        CrossFitTask(name="all", target=target),
        CrossFitTask(name="stratum_zero", target=target, train_mask=strata.eq(0)),
    )
    result = CrossFitter(
        propensity_factory=_ClassProbability,
        outcome_factory=_LinearOutcome,
        n_splits=2,
        random_state=5,
    ).fit_predict_tasks(X, tasks=tasks, strata=strata)

    assert isinstance(result, CrossFitTaskResult)
    assert result.predictions.columns.tolist() == ["all", "stratum_zero"]
    assert result.predictions.index.equals(X.index)
    np.testing.assert_allclose(result.predictions, np.column_stack([target, target]), atol=1e-12)
    assert set(result.model_names) == {"all", "stratum_zero"}


def test_weighted_tasks_pass_training_only_aligned_weights_and_record_audit() -> None:
    _WeightedAuditOutcome.records.clear()
    index = pd.Index([f"row_{value}" for value in range(12)])
    X = pd.DataFrame({"x": np.arange(12, dtype=float)}, index=index)
    target = pd.Series(np.arange(12, dtype=float), index=index)
    weights = pd.Series(np.arange(1, 13, dtype=float), index=index, name="survey_weight")
    fitter = CrossFitter(
        outcome_factory=_WeightedAuditOutcome,
        n_splits=3,
        random_state=17,
    )

    result = fitter.fit_predict_tasks(
        X,
        tasks=[CrossFitTask(name="weighted", target=target, sample_weight=weights)],
        strata=pd.Series(np.tile([0, 1], 6), index=index),
    )

    assert len(_WeightedAuditOutcome.records) == 3
    for fold, (train_index, consumed_weights) in enumerate(_WeightedAuditOutcome.records):
        expected_index = result.fold.index[result.fold.ne(fold)]
        assert train_index.equals(expected_index)
        pd.testing.assert_series_equal(consumed_weights, weights.loc[expected_index])
        assert train_index.intersection(result.fold.index[result.fold.eq(fold)]).empty
    diagnostics = result.model_diagnostics.set_index("fold")
    for fold in range(3):
        train_weights = weights.loc[result.fold.index[result.fold.ne(fold)]]
        assert diagnostics.loc[fold, "weight_sum"] == pytest.approx(train_weights.sum())
        assert diagnostics.loc[fold, "weight_effective_n"] == pytest.approx(
            train_weights.sum() ** 2 / train_weights.pow(2).sum()
        )
        assert isinstance(diagnostics.loc[fold, "weight_hash"], str)


def test_weighted_tasks_refuse_unweighted_provider_and_weight_index_drift() -> None:
    index = pd.RangeIndex(12)
    X = pd.DataFrame({"x": np.arange(12, dtype=float)}, index=index)
    target = pd.Series(np.arange(12, dtype=float), index=index)
    weights = pd.Series(np.ones(12), index=index)
    fitter = CrossFitter(outcome_factory=_LinearOutcome, n_splits=2, random_state=4)
    with pytest.raises(TypeError, match="sample_weight"):
        fitter.fit_predict_tasks(
            X,
            tasks=[CrossFitTask(name="weighted", target=target, sample_weight=weights)],
        )
    with pytest.raises(ValueError, match="indices"):
        fitter.fit_predict_tasks(
            X,
            tasks=[
                CrossFitTask(
                    name="weighted",
                    target=target,
                    sample_weight=weights.sample(frac=1.0, random_state=2),
                )
            ],
        )


def test_clustered_task_cross_fitting_keeps_clusters_wholly_within_folds() -> None:
    clusters = pd.Series(np.repeat([f"cluster-{i}" for i in range(12)], 2))
    treatment = pd.Series(np.repeat(np.tile([0.0, 1.0], 6), 2))
    covariates = pd.DataFrame({"x": np.arange(len(clusters), dtype=float)})
    target = pd.Series(1.0 + 0.5 * covariates["x"], index=covariates.index)
    result = CrossFitter(n_splits=3, random_state=119).fit_predict_tasks(
        covariates,
        tasks=[CrossFitTask(name="mean", target=target, factory=_LinearOutcome)],
        strata=treatment,
        clusters=clusters,
    )

    assert result.fold.groupby(clusters).nunique().eq(1).all()
    assert (pd.crosstab(result.fold, treatment) > 0).all().all()


def test_task_cross_fitting_exposes_provider_neutral_fold_diagnostics() -> None:
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    target = pd.Series(1.0 + 0.5 * X["x"], index=X.index)
    result = CrossFitter(n_splits=3, random_state=7).fit_predict_tasks(
        X,
        tasks=[
            CrossFitTask(
                name="mean",
                target=target,
                factory=_DiagnosticLinearOutcome,
            )
        ],
    )

    diagnostics = result.model_diagnostics
    assert diagnostics[["task", "fold"]].to_records(index=False).tolist() == [
        ("mean", 0),
        ("mean", 1),
        ("mean", 2),
    ]
    assert diagnostics["model"].eq("_DiagnosticLinearResult").all()
    assert diagnostics["train_nobs"].eq(8).all()
    assert diagnostics["holdout_nobs"].eq(4).all()
    assert diagnostics["diagnostics_available"].all()
    assert diagnostics["selected_alpha"].eq(0.25).all()
    assert diagnostics["effective_df"].eq(2.5).all()


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        (["not", "a", "mapping"], TypeError, "diagnostics must return a mapping"),
        ({"fold": 1}, ValueError, "is reserved"),
        ({"loss": np.inf}, ValueError, "must be finite"),
        ({"path": [1.0, 2.0]}, TypeError, "only scalar values"),
    ],
)
def test_cross_fitting_refuses_malformed_declared_diagnostics(payload, error, message) -> None:
    X = pd.DataFrame({"x": np.arange(8, dtype=float)})
    target = pd.Series(1.0 + X["x"], index=X.index)

    with pytest.raises(error, match=message):
        CrossFitter(n_splits=2, random_state=3).fit_predict_tasks(
            X,
            tasks=[
                CrossFitTask(
                    name="mean",
                    target=target,
                    factory=lambda: _PayloadDiagnosticLinearOutcome(payload),
                )
            ],
        )


def test_generic_cross_fitting_refuses_duplicate_tasks_and_too_small_strata() -> None:
    X = pd.DataFrame({"x": np.arange(6, dtype=float)})
    target = pd.Series(np.arange(6, dtype=float))
    fitter = CrossFitter(outcome_factory=_LinearOutcome, n_splits=3)
    duplicate = CrossFitTask(name="same", target=target)
    with pytest.raises(ValueError, match="task names must be unique"):
        fitter.fit_predict_tasks(X, tasks=[duplicate, duplicate])
    with pytest.raises(ValueError, match="stratum must contain at least n_splits"):
        fitter.fit_predict_tasks(
            X,
            tasks=[duplicate],
            strata=pd.Series([0, 0, 0, 0, 1, 1]),
        )


class _MaskedClassAuditResult:
    def __init__(
        self,
        classes: np.ndarray,
        probabilities: np.ndarray,
        train_index: pd.Index,
        registry: list[tuple[pd.Index, pd.Index]],
    ) -> None:
        self.classes_ = classes
        self.probabilities = probabilities
        self.train_index = train_index.copy()
        self.registry = registry

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.registry.append((self.train_index, X.index.copy()))
        return np.tile(self.probabilities, (len(X), 1))


class _MaskedClassAuditFactory:
    def __init__(self, registry: list[tuple[pd.Index, pd.Index]]) -> None:
        self.registry = registry

    def __call__(self):
        registry = self.registry

        class _Model:
            def fit(self, X: pd.DataFrame, y: pd.Series) -> _MaskedClassAuditResult:
                classes, counts = np.unique(np.asarray(y), return_counts=True)
                return _MaskedClassAuditResult(
                    classes,
                    counts / counts.sum(),
                    X.index,
                    registry,
                )

        return _Model()


def _masked_class_fixture():
    index = pd.Index([f"row-{position}" for position in range(32)], name="row")
    X = pd.DataFrame({"x": np.arange(32, dtype=float)}, index=index)
    global_strata = pd.Series(np.tile(np.arange(8), 4), index=index, name="global_cell")
    first_mask = global_strata.isin([0, 1, 2, 3])
    second_mask = global_strata.isin([4, 5, 6, 7])
    first_classes = pd.Series(global_strata % 4, index=index, name="first_cell")
    second_classes = pd.Series(global_strata % 4, index=index, name="second_cell")
    return X, global_strata, first_mask, second_mask, first_classes, second_classes


def test_masked_multiclass_tasks_share_global_folds_and_predict_only_relevant_rows() -> None:
    X, strata, first_mask, second_mask, first_classes, second_classes = _masked_class_fixture()
    registry: list[tuple[pd.Index, pd.Index]] = []
    factory = _MaskedClassAuditFactory(registry)
    tasks = [
        ClassProbabilityCrossFitTask(
            name="first_pair",
            classes=first_classes,
            train_mask=first_mask,
            predict_mask=first_mask,
            class_labels=(0, 1, 2, 3),
            factory=factory,
        ),
        ClassProbabilityCrossFitTask(
            name="second_pair",
            classes=second_classes,
            train_mask=second_mask,
            predict_mask=second_mask,
            class_labels=(0, 1, 2, 3),
            factory=factory,
        ),
    ]

    result = CrossFitter(n_splits=2, random_state=71).fit_predict_class_probability_tasks(
        X,
        tasks=tasks,
        strata=strata,
    )

    assert isinstance(result, ClassProbabilityTaskCrossFitResult)
    assert result.probabilities.columns.tolist() == [
        ("first_pair", 0),
        ("first_pair", 1),
        ("first_pair", 2),
        ("first_pair", 3),
        ("second_pair", 0),
        ("second_pair", 1),
        ("second_pair", 2),
        ("second_pair", 3),
    ]
    assert result.probabilities.loc[~first_mask, "first_pair"].isna().all().all()
    assert result.probabilities.loc[~second_mask, "second_pair"].isna().all().all()
    np.testing.assert_allclose(
        result.probabilities.loc[first_mask, "first_pair"].sum(axis=1),
        1.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.probabilities.loc[second_mask, "second_pair"].sum(axis=1),
        1.0,
        atol=1e-14,
    )
    assert len(registry) == 4
    for train_index, holdout_index in registry:
        assert set(train_index).isdisjoint(holdout_index)
    assert result.model_diagnostics["relevant_holdout_nobs"].eq(8).all()
    assert result.model_diagnostics["declared_class_count"].eq(4).all()
    scalar_result = CrossFitter(n_splits=2, random_state=71).fit_predict_tasks(
        X,
        tasks=[CrossFitTask(name="identity", target=X["x"], factory=_LinearOutcome)],
        strata=strata,
    )
    pd.testing.assert_series_equal(result.fold, scalar_result.fold)


def test_masked_multiclass_tasks_keep_psus_immutable_across_overlapping_tasks() -> None:
    X, strata, first_mask, second_mask, first_classes, second_classes = _masked_class_fixture()
    clusters = pd.Series(
        [f"psu-{position % 16}" for position in range(len(X))],
        index=X.index,
        name="psu",
    )
    tasks = [
        ClassProbabilityCrossFitTask(
            name="first_pair",
            classes=first_classes,
            train_mask=first_mask,
            class_labels=(0, 1, 2, 3),
            factory=_ClassProbability,
        ),
        ClassProbabilityCrossFitTask(
            name="second_pair",
            classes=second_classes,
            train_mask=second_mask,
            class_labels=(0, 1, 2, 3),
            factory=_ClassProbability,
        ),
    ]

    result = CrossFitter(n_splits=2, random_state=13).fit_predict_class_probability_tasks(
        X,
        tasks=tasks,
        strata=strata,
        clusters=clusters,
    )

    assert result.fold.groupby(clusters).nunique().eq(1).all()
    assert (pd.crosstab(result.fold, strata) > 0).all().all()


def test_masked_multiclass_tasks_refuse_schema_support_masks_and_duplicate_names() -> None:
    X, strata, first_mask, _, first_classes, _ = _masked_class_fixture()
    valid = ClassProbabilityCrossFitTask(
        name="pair",
        classes=first_classes,
        train_mask=first_mask,
        class_labels=(0, 1, 2, 3),
        factory=_ClassProbability,
    )
    fitter = CrossFitter(n_splits=2, random_state=7)

    with pytest.raises(ValueError, match="task names must be unique"):
        fitter.fit_predict_class_probability_tasks(X, tasks=[valid, valid], strata=strata)
    with pytest.raises(ValueError, match="declared class labels.*observed"):
        fitter.fit_predict_class_probability_tasks(
            X,
            tasks=[
                ClassProbabilityCrossFitTask(
                    name="bad_schema",
                    classes=first_classes,
                    train_mask=first_mask,
                    class_labels=(0, 1, 2),
                    factory=_ClassProbability,
                )
            ],
            strata=strata,
        )
    unsupported = first_mask.copy()
    class_three_positions = np.flatnonzero(first_mask & first_classes.eq(3))
    unsupported.iloc[class_three_positions[:-1]] = False
    with pytest.raises(ValueError, match="every declared class.*outside fold"):
        fitter.fit_predict_class_probability_tasks(
            X,
            tasks=[
                ClassProbabilityCrossFitTask(
                    name="unsupported",
                    classes=first_classes,
                    train_mask=unsupported,
                    class_labels=(0, 1, 2, 3),
                    factory=_ClassProbability,
                )
            ],
            strata=strata,
        )
    with pytest.raises(ValueError, match="predict_mask.*subset"):
        fitter.fit_predict_class_probability_tasks(
            X,
            tasks=[
                ClassProbabilityCrossFitTask(
                    name="bad_prediction_role",
                    classes=first_classes,
                    train_mask=first_mask,
                    predict_mask=pd.Series(True, index=X.index),
                    class_labels=(0, 1, 2, 3),
                    factory=_ClassProbability,
                )
            ],
            strata=strata,
        )

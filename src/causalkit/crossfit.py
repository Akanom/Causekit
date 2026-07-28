"""Reusable cross-fitting orchestration for causal nuisance predictions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class NuisanceEstimatorProtocol(Protocol):
    """Minimal fit protocol supported by the cross-fitting orchestrator."""

    def fit(self, X: Any, y: Any) -> Any: ...


@runtime_checkable
class PropensityResultProtocol(Protocol):
    """Default prediction contract for a fitted propensity model."""

    def predict_proba(self, X: Any) -> Any: ...


@runtime_checkable
class OutcomeResultProtocol(Protocol):
    """Default prediction contract for a fitted conditional-outcome model."""

    def predict(self, X: Any) -> Any: ...


NuisanceFactory = Callable[[], NuisanceEstimatorProtocol]
PredictionAdapter = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class CrossFitResult:
    """Out-of-fold nuisance predictions aligned to the original sample."""

    propensity: pd.Series
    outcome_treated: pd.Series
    outcome_control: pd.Series
    fold: pd.Series
    n_splits: int
    random_state: int | None
    propensity_model_name: str
    outcome_model_name: str


@dataclass(frozen=True)
class CrossFitTask:
    """One scalar nuisance regression evaluated on every held-out row.

    ``train_mask`` restricts model fitting without restricting prediction. This is the
    reusable operation needed for group-specific conditional means and second moments.
    A task-specific factory overrides the orchestrator's ``outcome_factory``.
    """

    name: str
    target: Any
    train_mask: Any | None = None
    factory: NuisanceFactory | None = None
    predict: PredictionAdapter | None = None


@dataclass(frozen=True)
class CrossFitTaskResult:
    """Aligned out-of-fold predictions for a collection of scalar tasks."""

    predictions: pd.DataFrame
    fold: pd.Series
    n_splits: int
    random_state: int | None
    model_names: dict[str, str]


@dataclass(frozen=True)
class ClassProbabilityCrossFitResult:
    """Aligned out-of-fold probabilities for a multiclass nuisance model."""

    probabilities: pd.DataFrame
    fold: pd.Series
    n_splits: int
    random_state: int | None
    model_name: str


def _fit(factory: NuisanceFactory, X: Any, y: Any) -> Any:
    estimator = factory()
    if not isinstance(estimator, NuisanceEstimatorProtocol):
        raise TypeError("Each nuisance factory must return an object with fit(X, y).")
    result = estimator.fit(X, y)
    return estimator if result is None else result


def _default_propensity_prediction(result: Any, X: Any) -> np.ndarray:
    if not hasattr(result, "predict_proba"):
        raise TypeError(
            "The fitted propensity result must provide predict_proba(X), or supply "
            "propensity_predict=."
        )
    values = result.predict_proba(X)
    if isinstance(values, pd.DataFrame):
        if 1 in values.columns:
            raw = values[1].to_numpy(dtype=float)
        elif values.shape[1] == 2:
            raw = values.iloc[:, 1].to_numpy(dtype=float)
        else:
            raise ValueError("predict_proba must expose the treated-class probability.")
    else:
        raw = np.asarray(values, dtype=float)
        if raw.ndim == 2 and raw.shape[1] == 2:
            raw = raw[:, 1]
    return np.asarray(raw, dtype=float)


def _default_outcome_prediction(result: Any, X: Any) -> np.ndarray:
    if not hasattr(result, "predict"):
        raise TypeError(
            "The fitted outcome result must provide predict(X), or supply outcome_predict=."
        )
    return np.asarray(result.predict(X), dtype=float)


def _prediction(
    result: Any,
    X: Any,
    *,
    adapter: PredictionAdapter | None,
    propensity: bool,
    expected: int,
) -> np.ndarray:
    raw = (
        adapter(result, X)
        if adapter is not None
        else _default_propensity_prediction(result, X)
        if propensity
        else _default_outcome_prediction(result, X)
    )
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or len(values) != expected:
        raise ValueError("A nuisance prediction must provide one value per held-out row.")
    if not np.isfinite(values).all():
        raise ValueError("Nuisance predictions must contain only finite values.")
    return values


def _frame(X: Any) -> tuple[pd.DataFrame, pd.Index]:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        index = frame.index.copy()
    else:
        raw = np.asarray(X)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        if raw.ndim != 2 or raw.shape[1] == 0:
            raise ValueError("X must be one- or two-dimensional numeric data.")
        frame = pd.DataFrame(raw, columns=[f"x_{i}" for i in range(raw.shape[1])])
        index = frame.index.copy()
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("X must contain only numeric values.") from error
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values.")
    if not frame.columns.is_unique:
        raise ValueError("X column names must be unique.")
    return frame, index


def _series(value: Any, *, name: str, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        if not value.index.equals(index):
            raise ValueError("Pandas indices must match exactly and in the same order.")
        series = value.astype(float).copy()
    else:
        raw = np.asarray(value, dtype=float)
        if raw.ndim != 1 or len(raw) != len(index):
            raise ValueError(f"{name} must contain exactly one value per row of X.")
        series = pd.Series(raw, index=index, name=name)
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values.")
    return series


def _labels(value: Any, *, name: str, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        if not value.index.equals(index):
            raise ValueError("Pandas indices must match exactly and in the same order.")
        series = value.copy()
    else:
        raw = np.asarray(value)
        if raw.ndim != 1 or len(raw) != len(index):
            raise ValueError(f"{name} must contain exactly one value per row of X.")
        series = pd.Series(raw, index=index, name=name)
    if series.isna().any():
        raise ValueError(f"{name} must not contain missing values.")
    return series


def _mask(value: Any, *, name: str, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        if not value.index.equals(index):
            raise ValueError("Pandas indices must match exactly and in the same order.")
        series = value.copy()
    else:
        raw = np.asarray(value)
        if raw.ndim != 1 or len(raw) != len(index):
            raise ValueError(f"{name} must contain exactly one value per row of X.")
        series = pd.Series(raw, index=index, name=name)
    if series.isna().any() or not series.isin([True, False]).all():
        raise ValueError(f"{name} must contain only non-missing booleans.")
    return series.astype(bool)


def _fold_assignments(
    nobs: int,
    *,
    n_splits: int,
    random_state: int | None,
    strata: pd.Series | None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    folds = np.empty(nobs, dtype=int)
    if strata is None:
        positions = np.arange(nobs)
        rng.shuffle(positions)
        folds[positions] = np.arange(nobs) % n_splits
        return folds
    for label in pd.unique(strata):
        positions = np.flatnonzero(strata.to_numpy() == label)
        if len(positions) < n_splits:
            raise ValueError("Each stratum must contain at least n_splits observations.")
        rng.shuffle(positions)
        folds[positions] = np.arange(len(positions)) % n_splits
    return folds


def _class_probability_prediction(
    result: Any,
    X: Any,
    *,
    classes: pd.Index,
    adapter: PredictionAdapter | None,
) -> np.ndarray:
    if adapter is not None:
        raw = adapter(result, X)
    else:
        if not hasattr(result, "predict_proba"):
            raise TypeError(
                "The fitted class-probability result must provide predict_proba(X), or "
                "supply propensity_predict=."
            )
        raw = result.predict_proba(X)
    if isinstance(raw, pd.DataFrame):
        missing = [label for label in classes if label not in raw.columns]
        if missing:
            raise ValueError(f"predict_proba is missing class column(s): {missing}.")
        values = raw.loc[:, list(classes)].to_numpy(dtype=float)
    else:
        values = np.asarray(raw, dtype=float)
        fitted_classes = getattr(result, "classes_", None)
        if fitted_classes is None:
            raise TypeError(
                "Array-valued multiclass predict_proba requires fitted result.classes_."
            )
        fitted_index = pd.Index(np.asarray(fitted_classes))
        positions = fitted_index.get_indexer(classes)
        if np.any(positions < 0):
            raise ValueError("predict_proba does not contain every observed class.")
        if values.ndim != 2 or values.shape[1] != len(fitted_index):
            raise ValueError("Multiclass predict_proba has an invalid column dimension.")
        values = values[:, positions]
    if values.shape != (len(X), len(classes)) or not np.isfinite(values).all():
        raise ValueError("Multiclass nuisance predictions must be a finite row-by-class matrix.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Class-probability predictions must lie between zero and one.")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("Class-probability predictions must sum to one in every row.")
    return values


class CrossFitter:
    """Generate out-of-fold propensity and arm-specific outcome predictions.

    Factories create fresh estimators for every fold. This supports estimators that
    return a fitted result (including ``limiteddepkit``) and estimators that mutate and
    return themselves (including scikit-learn-style models).
    """

    def __init__(
        self,
        *,
        propensity_factory: NuisanceFactory | None = None,
        outcome_factory: NuisanceFactory | None = None,
        second_moment_factory: NuisanceFactory | None = None,
        n_splits: int = 5,
        random_state: int | None = None,
        propensity_predict: PredictionAdapter | None = None,
        outcome_predict: PredictionAdapter | None = None,
    ) -> None:
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least two.")
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError("random_state must be an integer or None.")
        self.propensity_factory = propensity_factory
        self.outcome_factory = outcome_factory
        self.second_moment_factory = second_moment_factory or outcome_factory
        self.n_splits = n_splits
        self.random_state = random_state
        self.propensity_predict = propensity_predict
        self.outcome_predict = outcome_predict

    def fit_predict(self, X: Any, *, treatment: Any, outcome: Any) -> CrossFitResult:
        if self.propensity_factory is None or self.outcome_factory is None:
            raise ValueError("fit_predict requires both propensity_factory and outcome_factory.")
        frame, index = _frame(X)
        assigned = _series(treatment, name="treatment", index=index)
        observed = _series(outcome, name="outcome", index=index)
        if not np.array_equal(np.unique(assigned), np.array([0.0, 1.0])):
            raise ValueError("treatment must contain both arms and be coded exactly 0 and 1.")
        counts = assigned.value_counts()
        if int(counts.min()) < self.n_splits:
            raise ValueError("Each treatment arm must contain at least n_splits observations.")

        folds = _fold_assignments(
            len(frame),
            n_splits=self.n_splits,
            random_state=self.random_state,
            strata=assigned,
        )

        propensity = np.empty(len(frame), dtype=float)
        mu1 = np.empty(len(frame), dtype=float)
        mu0 = np.empty(len(frame), dtype=float)
        propensity_name = outcome_name = "unknown"
        assigned_values = assigned.to_numpy(dtype=float)
        for fold in range(self.n_splits):
            test = folds == fold
            train = ~test
            X_train = frame.iloc[train]
            X_test = frame.iloc[test]
            treatment_train = assigned.iloc[train]
            propensity_result = _fit(self.propensity_factory, X_train, treatment_train)
            propensity_name = type(propensity_result).__name__
            propensity[test] = _prediction(
                propensity_result,
                X_test,
                adapter=self.propensity_predict,
                propensity=True,
                expected=int(test.sum()),
            )
            arm_results: list[Any] = []
            for arm, target in ((1.0, mu1), (0.0, mu0)):
                arm_train = train & (assigned_values == arm)
                result = _fit(self.outcome_factory, frame.iloc[arm_train], observed.iloc[arm_train])
                arm_results.append(result)
                target[test] = _prediction(
                    result,
                    X_test,
                    adapter=self.outcome_predict,
                    propensity=False,
                    expected=int(test.sum()),
                )
            outcome_name = type(arm_results[0]).__name__

        if np.any((propensity <= 0) | (propensity >= 1)):
            raise ValueError(
                "Cross-fitted propensity predictions must lie strictly between zero and one."
            )
        return CrossFitResult(
            propensity=pd.Series(propensity, index=index, name="propensity"),
            outcome_treated=pd.Series(mu1, index=index, name="outcome_treated"),
            outcome_control=pd.Series(mu0, index=index, name="outcome_control"),
            fold=pd.Series(folds, index=index, name="fold"),
            n_splits=self.n_splits,
            random_state=self.random_state,
            propensity_model_name=propensity_name,
            outcome_model_name=outcome_name,
        )

    def fit_predict_class_probabilities(
        self,
        X: Any,
        *,
        classes: Any,
    ) -> ClassProbabilityCrossFitResult:
        """Cross-fit one multiclass model and return all class probabilities."""

        if self.propensity_factory is None:
            raise ValueError("fit_predict_class_probabilities requires propensity_factory.")
        frame, index = _frame(X)
        labels = _labels(classes, name="classes", index=index)
        observed_classes = pd.Index(pd.unique(labels), name="class")
        if len(observed_classes) < 2:
            raise ValueError("classes must contain at least two observed levels.")
        folds = _fold_assignments(
            len(frame),
            n_splits=self.n_splits,
            random_state=self.random_state,
            strata=labels,
        )
        probabilities = np.empty((len(frame), len(observed_classes)), dtype=float)
        model_name = "unknown"
        for fold in range(self.n_splits):
            test = folds == fold
            train = ~test
            result = _fit(self.propensity_factory, frame.iloc[train], labels.iloc[train])
            model_name = type(result).__name__
            probabilities[test] = _class_probability_prediction(
                result,
                frame.iloc[test],
                classes=observed_classes,
                adapter=self.propensity_predict,
            )
        return ClassProbabilityCrossFitResult(
            probabilities=pd.DataFrame(
                probabilities,
                index=index.copy(),
                columns=observed_classes.copy(),
            ),
            fold=pd.Series(folds, index=index.copy(), name="fold"),
            n_splits=self.n_splits,
            random_state=self.random_state,
            model_name=model_name,
        )

    def fit_predict_tasks(
        self,
        X: Any,
        *,
        tasks: Sequence[CrossFitTask],
        strata: Any | None = None,
    ) -> CrossFitTaskResult:
        """Cross-fit arbitrary scalar regressions on shared deterministic folds."""

        frame, index = _frame(X)
        task_list = list(tasks)
        if not task_list:
            raise ValueError("tasks must contain at least one CrossFitTask.")
        if not all(isinstance(task, CrossFitTask) for task in task_list):
            raise TypeError("tasks must contain only CrossFitTask instances.")
        names = [task.name for task in task_list]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Every task name must be a non-empty string.")
        if len(set(names)) != len(names):
            raise ValueError("Cross-fitting task names must be unique.")
        strata_series = None if strata is None else _labels(strata, name="strata", index=index)
        folds = _fold_assignments(
            len(frame),
            n_splits=self.n_splits,
            random_state=self.random_state,
            strata=strata_series,
        )
        predictions = np.empty((len(frame), len(task_list)), dtype=float)
        model_names: dict[str, str] = {}
        for task_position, task in enumerate(task_list):
            target = _series(task.target, name=task.name, index=index)
            train_mask = (
                pd.Series(True, index=index)
                if task.train_mask is None
                else _mask(task.train_mask, name=f"{task.name}.train_mask", index=index)
            )
            factory = task.factory or self.outcome_factory
            if factory is None:
                raise ValueError(f"Task {task.name!r} requires a task factory or outcome_factory.")
            for fold in range(self.n_splits):
                test = folds == fold
                train = (~test) & train_mask.to_numpy()
                if not train.any():
                    raise ValueError(
                        f"Task {task.name!r} has no training observations outside fold {fold}."
                    )
                result = _fit(factory, frame.iloc[train], target.iloc[train])
                model_names[task.name] = type(result).__name__
                predictions[test, task_position] = _prediction(
                    result,
                    frame.iloc[test],
                    adapter=task.predict or self.outcome_predict,
                    propensity=False,
                    expected=int(test.sum()),
                )
        return CrossFitTaskResult(
            predictions=pd.DataFrame(predictions, index=index.copy(), columns=names),
            fold=pd.Series(folds, index=index.copy(), name="fold"),
            n_splits=self.n_splits,
            random_state=self.random_state,
            model_names=model_names,
        )


__all__ = [
    "ClassProbabilityCrossFitResult",
    "CrossFitResult",
    "CrossFitTask",
    "CrossFitTaskResult",
    "CrossFitter",
    "NuisanceEstimatorProtocol",
    "NuisanceFactory",
    "OutcomeResultProtocol",
    "PredictionAdapter",
    "PropensityResultProtocol",
]

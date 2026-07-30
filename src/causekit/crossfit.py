"""Reusable cross-fitting orchestration for causal nuisance predictions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class NuisanceEstimatorProtocol(Protocol):
    """Minimal fit protocol supported by the cross-fitting orchestrator."""

    def fit(self, X: Any, y: Any) -> Any: ...


@runtime_checkable
class WeightedNuisanceEstimatorProtocol(Protocol):
    """Nuisance fit contract that must consume aligned training analysis weights."""

    def fit(self, X: Any, y: Any, *, sample_weight: Any) -> Any: ...


@runtime_checkable
class PropensityResultProtocol(Protocol):
    """Default prediction contract for a fitted propensity model."""

    def predict_proba(self, X: Any) -> Any: ...


@runtime_checkable
class OutcomeResultProtocol(Protocol):
    """Default prediction contract for a fitted conditional-outcome model."""

    def predict(self, X: Any) -> Any: ...


@runtime_checkable
class WeightedCATEEstimatorProtocol(Protocol):
    """Fit contract for a CATE learner receiving the exact R-loss weights."""

    def fit(self, X: Any, y: Any, *, sample_weight: Any) -> Any: ...


@runtime_checkable
class CATEEstimatorProtocol(Protocol):
    """Unweighted fit contract for a DR pseudo-outcome CATE learner."""

    def fit(self, X: Any, y: Any) -> Any: ...


@runtime_checkable
class CATEResultProtocol(Protocol):
    """Prediction contract for a fitted conditional-effect learner."""

    def predict(self, X: Any) -> Any: ...


@runtime_checkable
class NuisanceDiagnosticsProtocol(Protocol):
    """Optional scalar diagnostics exposed by a fitted nuisance result."""

    def nuisance_diagnostics(self) -> Mapping[str, Any]: ...


NuisanceFactory = Callable[[], NuisanceEstimatorProtocol]
CATEFactory = Callable[[], CATEEstimatorProtocol]
WeightedCATEFactory = Callable[[], WeightedCATEEstimatorProtocol]
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
    model_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    sample_weight: Any | None = None


@dataclass(frozen=True)
class CrossFitTaskResult:
    """Aligned out-of-fold predictions for a collection of scalar tasks."""

    predictions: pd.DataFrame
    fold: pd.Series
    n_splits: int
    random_state: int | None
    model_names: dict[str, str]
    model_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class ClassProbabilityCrossFitTask:
    """One masked multiclass nuisance task on a shared global fold plan.

    ``train_mask`` and ``predict_mask`` preserve the task-specific analysis role while
    folds remain global. Predictions outside ``predict_mask`` are retained as missing
    audit placeholders and must never enter a score.
    """

    name: str
    classes: Any
    train_mask: Any
    predict_mask: Any | None = None
    class_labels: Sequence[Any] | None = None
    factory: NuisanceFactory | None = None
    predict: PredictionAdapter | None = None


@dataclass(frozen=True)
class ClassProbabilityCrossFitResult:
    """Aligned out-of-fold probabilities for a multiclass nuisance model."""

    probabilities: pd.DataFrame
    fold: pd.Series
    n_splits: int
    random_state: int | None
    model_name: str
    model_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class ClassProbabilityTaskCrossFitResult:
    """Task-labelled multiclass probabilities on one immutable global fold plan."""

    probabilities: pd.DataFrame
    fold: pd.Series
    n_splits: int
    random_state: int | None
    model_names: dict[str, str]
    class_labels: dict[str, tuple[Any, ...]]
    model_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


def _fit(
    factory: NuisanceFactory,
    X: Any,
    y: Any,
    *,
    seen_estimators: list[Any] | None = None,
) -> Any:
    estimator = factory()
    if not isinstance(estimator, NuisanceEstimatorProtocol):
        raise TypeError("Each nuisance factory must return an object with fit(X, y).")
    if seen_estimators is not None:
        if any(estimator is previous for previous in seen_estimators):
            raise ValueError("Each nuisance factory call must return a fresh estimator instance.")
        seen_estimators.append(estimator)
    result = estimator.fit(X, y)
    return estimator if result is None else result


def _fit_weighted(
    factory: NuisanceFactory,
    X: Any,
    y: Any,
    sample_weight: pd.Series,
    *,
    seen_estimators: list[Any] | None = None,
) -> Any:
    estimator = factory()
    if not isinstance(estimator, WeightedNuisanceEstimatorProtocol):
        raise TypeError(
            "Each weighted nuisance factory must return an object with "
            "fit(X, y, *, sample_weight=...)."
        )
    if seen_estimators is not None:
        if any(estimator is previous for previous in seen_estimators):
            raise ValueError("Each nuisance factory call must return a fresh estimator instance.")
        seen_estimators.append(estimator)
    result = estimator.fit(X, y, sample_weight=sample_weight)
    return estimator if result is None else result


_DIAGNOSTIC_RESERVED_COLUMNS = {
    "task",
    "fold",
    "model",
    "train_nobs",
    "holdout_nobs",
    "diagnostics_available",
    "weight_sum",
    "weight_effective_n",
    "weight_min",
    "weight_max",
    "weight_hash",
}


def _model_diagnostic_row(
    result: Any,
    *,
    task: str,
    fold: int,
    train_nobs: int,
    holdout_nobs: int,
    sample_weight: pd.Series | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task,
        "fold": fold,
        "model": type(result).__name__,
        "train_nobs": train_nobs,
        "holdout_nobs": holdout_nobs,
        "diagnostics_available": False,
    }
    if sample_weight is not None:
        values = sample_weight.to_numpy(dtype=float)
        weight_sum = float(values.sum())
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(sample_weight, index=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        ).hexdigest()
        row.update(
            {
                "weight_sum": weight_sum,
                "weight_effective_n": weight_sum**2 / float(np.dot(values, values)),
                "weight_min": float(values.min()),
                "weight_max": float(values.max()),
                "weight_hash": digest,
            }
        )
    if not isinstance(result, NuisanceDiagnosticsProtocol):
        return row
    supplied = result.nuisance_diagnostics()
    if not isinstance(supplied, Mapping):
        raise TypeError("nuisance diagnostics must return a mapping of scalar values.")
    normalized: dict[str, Any] = {}
    for key, raw_value in supplied.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("nuisance diagnostic names must be non-empty strings.")
        if key in _DIAGNOSTIC_RESERVED_COLUMNS:
            raise ValueError(f"nuisance diagnostic name {key!r} is reserved by CrossFitter.")
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if isinstance(value, bool) or value is None or isinstance(value, str):
            normalized[key] = value
        elif isinstance(value, Integral):
            normalized[key] = int(value)
        elif isinstance(value, Real):
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("numeric nuisance diagnostics must be finite.")
            normalized[key] = numeric
        else:
            raise TypeError("nuisance diagnostics must contain only scalar values.")
    row["diagnostics_available"] = True
    row.update(normalized)
    return row


def _default_propensity_prediction(result: Any, X: Any) -> np.ndarray:
    if not hasattr(result, "predict_proba"):
        raise TypeError(
            "The fitted propensity result must provide predict_proba(X), or supply "
            "propensity_predict=."
        )
    values = result.predict_proba(X)
    if (
        isinstance(values, (pd.Series, pd.DataFrame))
        and isinstance(X, pd.DataFrame)
        and not values.index.equals(X.index)
    ):
        raise ValueError("Propensity prediction index must match the held-out covariate index.")
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
    values = result.predict(X)
    if (
        isinstance(values, (pd.Series, pd.DataFrame))
        and isinstance(X, pd.DataFrame)
        and not values.index.equals(X.index)
    ):
        raise ValueError("Outcome prediction index must match the held-out covariate index.")
    return np.asarray(values, dtype=float)


def _prediction(
    result: Any,
    X: Any,
    *,
    adapter: PredictionAdapter | None,
    propensity: bool,
    expected: int,
) -> np.ndarray:
    if adapter is not None:
        raw = adapter(result, X)
        if (
            isinstance(raw, (pd.Series, pd.DataFrame))
            and isinstance(X, pd.DataFrame)
            and not raw.index.equals(X.index)
        ):
            raise ValueError("Adapted prediction index must match the held-out covariate index.")
    else:
        raw = (
            _default_propensity_prediction(result, X)
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


def _weights(value: Any, *, name: str, index: pd.Index) -> pd.Series:
    raw = value.to_numpy() if isinstance(value, pd.Series) else np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real numeric values.")
    weights = _series(value, name=name, index=index)
    if np.any(weights.to_numpy(dtype=float) <= 0.0):
        raise ValueError(f"{name} must contain only strictly positive values.")
    return weights


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
    clusters: pd.Series | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if clusters is not None:
        cluster_codes, cluster_labels = pd.factorize(clusters, sort=False)
        n_clusters = len(cluster_labels)
        if n_clusters < n_splits:
            raise ValueError("Clustered cross-fitting requires at least n_splits clusters.")
        if strata is None:
            stratum_codes = np.zeros(nobs, dtype=int)
            n_strata = 1
        else:
            stratum_codes, stratum_labels = pd.factorize(strata, sort=False)
            n_strata = len(stratum_labels)
        counts = np.zeros((n_clusters, n_strata), dtype=float)
        np.add.at(counts, (cluster_codes, stratum_codes), 1.0)
        if strata is not None and np.any((counts > 0.0).sum(axis=0) < n_splits):
            raise ValueError(
                "Each stratum must occur in at least n_splits clusters for clustered cross-fitting."
            )
        totals = counts.sum(axis=0)
        targets = totals / n_splits
        cluster_sizes = counts.sum(axis=1)
        randomized = rng.permutation(n_clusters)
        order = randomized[np.argsort(-cluster_sizes[randomized], kind="stable")]
        fold_priority = rng.permutation(n_splits)
        fold_counts = np.zeros((n_splits, n_strata), dtype=float)
        fold_clusters = np.zeros(n_splits, dtype=float)
        cluster_assignment = np.full(n_clusters, -1, dtype=int)
        target_cluster_count = n_clusters / n_splits
        scale = np.maximum(targets, 1.0)
        for cluster_code in order:
            best_fold = -1
            best_score = float("inf")
            for fold in fold_priority:
                proposed_counts = fold_counts.copy()
                proposed_counts[fold] += counts[cluster_code]
                proposed_clusters = fold_clusters.copy()
                proposed_clusters[fold] += 1.0
                score = float(
                    np.sum((proposed_counts - targets) ** 2 / scale)
                    + 0.01 * np.sum((proposed_clusters - target_cluster_count) ** 2)
                )
                if score < best_score:
                    best_score = score
                    best_fold = int(fold)
            cluster_assignment[cluster_code] = best_fold
            fold_counts[best_fold] += counts[cluster_code]
            fold_clusters[best_fold] += 1.0
        if np.any(fold_clusters == 0.0):
            raise ValueError("Clustered cross-fitting produced an empty fold.")
        if strata is not None and np.any(fold_counts == 0.0):
            raise ValueError(
                "Clustered stratification could not retain every stratum in every fold."
            )
        return cluster_assignment[cluster_codes]

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
    if (
        isinstance(raw, (pd.Series, pd.DataFrame))
        and isinstance(X, pd.DataFrame)
        and not raw.index.equals(X.index)
    ):
        raise ValueError(
            "Class-probability prediction index must match the held-out covariate index."
        )
    if isinstance(raw, pd.DataFrame):
        if not raw.columns.is_unique:
            raise ValueError("predict_proba must provide unique class columns.")
        missing = [label for label in classes if label not in raw.columns]
        if missing:
            raise ValueError(f"predict_proba is missing class column(s): {missing}.")
        unexpected = [label for label in raw.columns if label not in classes]
        if unexpected or len(raw.columns) != len(classes):
            raise ValueError(
                "predict_proba must contain exactly the observed classes; "
                f"unexpected class column(s): {unexpected}."
            )
        values = raw.loc[:, list(classes)].to_numpy(dtype=float)
    else:
        values = np.asarray(raw, dtype=float)
        fitted_classes = getattr(result, "classes_", None)
        if fitted_classes is None:
            raise TypeError(
                "Array-valued multiclass predict_proba requires fitted result.classes_."
            )
        fitted_index = pd.Index(np.asarray(fitted_classes))
        if not fitted_index.is_unique:
            raise ValueError("Fitted result.classes_ must contain unique class labels.")
        missing = [label for label in classes if label not in fitted_index]
        if missing:
            raise ValueError(f"Fitted result.classes_ is missing observed class(es): {missing}.")
        unexpected = [label for label in fitted_index if label not in classes]
        if unexpected or len(fitted_index) != len(classes):
            raise ValueError(
                "Fitted result.classes_ must contain exactly the observed classes; "
                f"unexpected class label(s): {unexpected}."
            )
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
    return an immutable fitted result and estimators that mutate and return themselves,
    including scikit-learn-style models.
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

    def fit_predict(
        self,
        X: Any,
        *,
        treatment: Any,
        outcome: Any,
        clusters: Any | None = None,
    ) -> CrossFitResult:
        """Cross-fit propensity and arm-specific outcomes on shared stratified folds.

        When ``clusters`` is supplied, every cluster is assigned wholly to one fold and
        each treatment arm must occur in at least ``n_splits`` clusters.
        """

        if self.propensity_factory is None or self.outcome_factory is None:
            raise ValueError("fit_predict requires both propensity_factory and outcome_factory.")
        frame, index = _frame(X)
        assigned = _series(treatment, name="treatment", index=index)
        observed = _series(outcome, name="outcome", index=index)
        cluster_values = (
            None if clusters is None else _labels(clusters, name="clusters", index=index)
        )
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
            clusters=cluster_values,
        )

        propensity = np.empty(len(frame), dtype=float)
        mu1 = np.empty(len(frame), dtype=float)
        mu0 = np.empty(len(frame), dtype=float)
        propensity_name = outcome_name = "unknown"
        diagnostic_rows: list[dict[str, Any]] = []
        seen_estimators: list[Any] = []
        assigned_values = assigned.to_numpy(dtype=float)
        for fold in range(self.n_splits):
            test = folds == fold
            train = ~test
            X_train = frame.iloc[train]
            X_test = frame.iloc[test]
            treatment_train = assigned.iloc[train]
            propensity_result = _fit(
                self.propensity_factory,
                X_train,
                treatment_train,
                seen_estimators=seen_estimators,
            )
            propensity_name = type(propensity_result).__name__
            propensity[test] = _prediction(
                propensity_result,
                X_test,
                adapter=self.propensity_predict,
                propensity=True,
                expected=int(test.sum()),
            )
            diagnostic_rows.append(
                _model_diagnostic_row(
                    propensity_result,
                    task="propensity",
                    fold=fold,
                    train_nobs=int(train.sum()),
                    holdout_nobs=int(test.sum()),
                )
            )
            arm_results: list[Any] = []
            for arm, target in ((1.0, mu1), (0.0, mu0)):
                arm_train = train & (assigned_values == arm)
                result = _fit(
                    self.outcome_factory,
                    frame.iloc[arm_train],
                    observed.iloc[arm_train],
                    seen_estimators=seen_estimators,
                )
                arm_results.append(result)
                target[test] = _prediction(
                    result,
                    X_test,
                    adapter=self.outcome_predict,
                    propensity=False,
                    expected=int(test.sum()),
                )
                diagnostic_rows.append(
                    _model_diagnostic_row(
                        result,
                        task="outcome_treated" if arm == 1.0 else "outcome_control",
                        fold=fold,
                        train_nobs=int(arm_train.sum()),
                        holdout_nobs=int(test.sum()),
                    )
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
            model_diagnostics=pd.DataFrame(diagnostic_rows),
        )

    def fit_predict_class_probabilities(
        self,
        X: Any,
        *,
        classes: Any,
        clusters: Any | None = None,
    ) -> ClassProbabilityCrossFitResult:
        """Cross-fit one multiclass model and return all class probabilities.

        Optional cluster labels keep each cluster wholly within one outer fold. Labelled
        outputs may permute the observed classes and are realigned; missing, extra,
        duplicate, or unlabeled class schemas refuse.
        """

        if self.propensity_factory is None:
            raise ValueError("fit_predict_class_probabilities requires propensity_factory.")
        frame, index = _frame(X)
        labels = _labels(classes, name="classes", index=index)
        cluster_values = (
            None if clusters is None else _labels(clusters, name="clusters", index=index)
        )
        observed_classes = pd.Index(pd.unique(labels), name="class")
        if len(observed_classes) < 2:
            raise ValueError("classes must contain at least two observed levels.")
        folds = _fold_assignments(
            len(frame),
            n_splits=self.n_splits,
            random_state=self.random_state,
            strata=labels,
            clusters=cluster_values,
        )
        probabilities = np.empty((len(frame), len(observed_classes)), dtype=float)
        model_name = "unknown"
        diagnostic_rows: list[dict[str, Any]] = []
        seen_estimators: list[Any] = []
        for fold in range(self.n_splits):
            test = folds == fold
            train = ~test
            result = _fit(
                self.propensity_factory,
                frame.iloc[train],
                labels.iloc[train],
                seen_estimators=seen_estimators,
            )
            model_name = type(result).__name__
            probabilities[test] = _class_probability_prediction(
                result,
                frame.iloc[test],
                classes=observed_classes,
                adapter=self.propensity_predict,
            )
            diagnostic_rows.append(
                _model_diagnostic_row(
                    result,
                    task="class_probability",
                    fold=fold,
                    train_nobs=int(train.sum()),
                    holdout_nobs=int(test.sum()),
                )
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
            model_diagnostics=pd.DataFrame(diagnostic_rows),
        )

    def fit_predict_tasks(
        self,
        X: Any,
        *,
        tasks: Sequence[CrossFitTask],
        strata: Any | None = None,
        clusters: Any | None = None,
    ) -> CrossFitTaskResult:
        """Cross-fit arbitrary scalar regressions on shared deterministic folds.

        Optional cluster labels keep each cluster wholly within one outer fold.
        """

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
        cluster_values = (
            None if clusters is None else _labels(clusters, name="clusters", index=index)
        )
        folds = _fold_assignments(
            len(frame),
            n_splits=self.n_splits,
            random_state=self.random_state,
            strata=strata_series,
            clusters=cluster_values,
        )
        predictions = np.empty((len(frame), len(task_list)), dtype=float)
        model_names: dict[str, str] = {}
        diagnostic_rows: list[dict[str, Any]] = []
        seen_estimators: list[Any] = []
        for task_position, task in enumerate(task_list):
            target = _series(task.target, name=task.name, index=index)
            sample_weight = (
                None
                if task.sample_weight is None
                else _weights(
                    task.sample_weight,
                    name=f"{task.name}.sample_weight",
                    index=index,
                )
            )
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
                training_weights = None if sample_weight is None else sample_weight.iloc[train]
                result = (
                    _fit(
                        factory,
                        frame.iloc[train],
                        target.iloc[train],
                        seen_estimators=seen_estimators,
                    )
                    if training_weights is None
                    else _fit_weighted(
                        factory,
                        frame.iloc[train],
                        target.iloc[train],
                        training_weights,
                        seen_estimators=seen_estimators,
                    )
                )
                model_names[task.name] = type(result).__name__
                predictions[test, task_position] = _prediction(
                    result,
                    frame.iloc[test],
                    adapter=task.predict or self.outcome_predict,
                    propensity=False,
                    expected=int(test.sum()),
                )
                diagnostic_rows.append(
                    _model_diagnostic_row(
                        result,
                        task=task.name,
                        fold=fold,
                        train_nobs=int(train.sum()),
                        holdout_nobs=int(test.sum()),
                        sample_weight=training_weights,
                    )
                )
        return CrossFitTaskResult(
            predictions=pd.DataFrame(predictions, index=index.copy(), columns=names),
            fold=pd.Series(folds, index=index.copy(), name="fold"),
            n_splits=self.n_splits,
            random_state=self.random_state,
            model_names=model_names,
            model_diagnostics=pd.DataFrame(diagnostic_rows),
        )

    def fit_predict_class_probability_tasks(
        self,
        X: Any,
        *,
        tasks: Sequence[ClassProbabilityCrossFitTask],
        strata: Any | None = None,
        clusters: Any | None = None,
    ) -> ClassProbabilityTaskCrossFitResult:
        """Cross-fit masked multiclass tasks on one deterministic global fold plan.

        Every task/fold must retain all declared classes in both its training and
        relevant holdout support. Models are fresh per task/fold and predictions are
        emitted only for rows in the task's declared prediction role.
        """

        frame, index = _frame(X)
        task_list = list(tasks)
        if not task_list:
            raise ValueError("tasks must contain at least one ClassProbabilityCrossFitTask.")
        if not all(isinstance(task, ClassProbabilityCrossFitTask) for task in task_list):
            raise TypeError("tasks must contain only ClassProbabilityCrossFitTask instances.")
        names = [task.name for task in task_list]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Every task name must be a non-empty string.")
        if len(set(names)) != len(names):
            raise ValueError("Cross-fitting task names must be unique.")
        strata_series = None if strata is None else _labels(strata, name="strata", index=index)
        cluster_values = (
            None if clusters is None else _labels(clusters, name="clusters", index=index)
        )
        folds = _fold_assignments(
            len(frame),
            n_splits=self.n_splits,
            random_state=self.random_state,
            strata=strata_series,
            clusters=cluster_values,
        )

        prepared: list[
            tuple[
                ClassProbabilityCrossFitTask,
                pd.Series,
                pd.Series,
                pd.Series,
                pd.Index,
                NuisanceFactory,
            ]
        ] = []
        class_labels: dict[str, tuple[Any, ...]] = {}
        for task in task_list:
            labels = _labels(task.classes, name=f"{task.name}.classes", index=index)
            train_mask = _mask(task.train_mask, name=f"{task.name}.train_mask", index=index)
            predict_mask = (
                train_mask.copy()
                if task.predict_mask is None
                else _mask(
                    task.predict_mask,
                    name=f"{task.name}.predict_mask",
                    index=index,
                )
            )
            if not predict_mask.any():
                raise ValueError(f"Task {task.name!r} predict_mask must select observations.")
            if (predict_mask & ~train_mask).any():
                raise ValueError(f"Task {task.name!r} predict_mask must be a subset of train_mask.")
            observed = pd.Index(pd.unique(labels[train_mask]), name="class")
            declared = (
                observed
                if task.class_labels is None
                else pd.Index(list(task.class_labels), name="class")
            )
            if declared.empty or not declared.is_unique:
                raise ValueError(
                    f"Task {task.name!r} declared class labels must be non-empty and unique."
                )
            try:
                declared_missing = bool(np.asarray(pd.isna(declared)).any())
            except (TypeError, ValueError):
                declared_missing = False
            if declared_missing:
                raise ValueError(f"Task {task.name!r} declared class labels must not be missing.")
            missing = [label for label in declared if label not in observed]
            unexpected = [label for label in observed if label not in declared]
            if missing or unexpected or len(observed) != len(declared):
                raise ValueError(
                    f"Task {task.name!r} declared class labels must equal the observed "
                    f"training-mask classes; missing={missing}, unexpected={unexpected}."
                )
            if len(declared) < 2:
                raise ValueError(f"Task {task.name!r} must contain at least two classes.")
            factory = task.factory or self.propensity_factory
            if factory is None:
                raise ValueError(
                    f"Task {task.name!r} requires a task factory or propensity_factory."
                )
            class_labels[task.name] = tuple(declared.tolist())
            prepared.append((task, labels, train_mask, predict_mask, declared, factory))

        columns = pd.MultiIndex.from_tuples(
            [(task.name, label) for task, _, _, _, declared, _ in prepared for label in declared],
            names=["task", "class"],
        )
        probabilities = pd.DataFrame(np.nan, index=index.copy(), columns=columns, dtype=float)
        model_names: dict[str, str] = {}
        diagnostic_rows: list[dict[str, Any]] = []
        seen_estimators: list[Any] = []
        for task, labels, train_mask, predict_mask, declared, factory in prepared:
            for fold in range(self.n_splits):
                test = folds == fold
                train = (~test) & train_mask.to_numpy()
                relevant_test = test & predict_mask.to_numpy()
                train_classes = pd.Index(pd.unique(labels.iloc[train]))
                holdout_classes = pd.Index(pd.unique(labels.iloc[relevant_test]))
                if any(label not in train_classes for label in declared):
                    raise ValueError(
                        f"Task {task.name!r} must retain every declared class outside fold {fold}."
                    )
                if any(label not in holdout_classes for label in declared):
                    raise ValueError(
                        f"Task {task.name!r} must retain every declared class in the relevant "
                        f"holdout of fold {fold}."
                    )
                result = _fit(
                    factory,
                    frame.iloc[train],
                    labels.iloc[train],
                    seen_estimators=seen_estimators,
                )
                model_names[task.name] = type(result).__name__
                values = _class_probability_prediction(
                    result,
                    frame.iloc[relevant_test],
                    classes=declared,
                    adapter=task.predict or self.propensity_predict,
                )
                probabilities.loc[index[relevant_test], pd.IndexSlice[task.name, :]] = values
                row = _model_diagnostic_row(
                    result,
                    task=task.name,
                    fold=fold,
                    train_nobs=int(train.sum()),
                    holdout_nobs=int(test.sum()),
                )
                row["relevant_holdout_nobs"] = int(relevant_test.sum())
                row["declared_class_count"] = len(declared)
                diagnostic_rows.append(row)

        return ClassProbabilityTaskCrossFitResult(
            probabilities=probabilities,
            fold=pd.Series(folds, index=index.copy(), name="fold"),
            n_splits=self.n_splits,
            random_state=self.random_state,
            model_names=model_names,
            class_labels=class_labels,
            model_diagnostics=pd.DataFrame(diagnostic_rows),
        )


__all__ = [
    "CATEEstimatorProtocol",
    "CATEFactory",
    "CATEResultProtocol",
    "ClassProbabilityCrossFitTask",
    "ClassProbabilityCrossFitResult",
    "ClassProbabilityTaskCrossFitResult",
    "CrossFitResult",
    "CrossFitTask",
    "CrossFitTaskResult",
    "CrossFitter",
    "NuisanceEstimatorProtocol",
    "WeightedNuisanceEstimatorProtocol",
    "NuisanceDiagnosticsProtocol",
    "NuisanceFactory",
    "OutcomeResultProtocol",
    "PredictionAdapter",
    "PropensityResultProtocol",
    "WeightedCATEEstimatorProtocol",
    "WeightedCATEFactory",
]

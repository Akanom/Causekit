"""Strict data preparation for causal estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

MissingPolicy = Literal["raise", "drop"]


@dataclass(frozen=True)
class PreparedIVData:
    """Numerical IV arrays with their validated labels and estimation index."""

    y: np.ndarray
    endogenous: np.ndarray
    exogenous: np.ndarray
    instruments: np.ndarray
    clusters: np.ndarray | None
    index: pd.Index
    y_name: str
    endogenous_names: tuple[str, ...]
    exogenous_names: tuple[str, ...]
    instrument_names: tuple[str, ...]
    dropped_rows: int


@dataclass(frozen=True)
class PreparedRDData:
    """Numerical RD arrays with their validated labels and estimation index."""

    y: np.ndarray
    running: np.ndarray
    treatment: np.ndarray | None
    clusters: np.ndarray | None
    index: pd.Index
    y_name: str
    running_name: str
    treatment_name: str | None
    dropped_rows: int


def _string_names(columns: Any, *, kind: str) -> tuple[str, ...]:
    names = tuple(str(column) for column in columns)
    if len(set(names)) != len(names):
        raise ValueError(f"{kind} column names must be unique after string conversion.")
    return names


def _as_numeric_frame(
    value: Any,
    *,
    kind: str,
    prefix: str,
) -> tuple[np.ndarray, tuple[str, ...], pd.Index | None]:
    if isinstance(value, pd.Series):
        name = str(value.name) if value.name is not None else prefix
        try:
            array = value.to_numpy(dtype=float).reshape(-1, 1)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{kind} must contain only numeric values.") from error
        return array, (name,), value.index.copy()

    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            raise ValueError(f"{kind} must contain at least one column.")
        names = _string_names(value.columns, kind=kind)
        try:
            array = value.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{kind} must contain only numeric values.") from error
        return array, names, value.index.copy()

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{kind} must contain only numeric values.") from error
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{kind} must be a one- or two-dimensional numeric array.")
    names = tuple(prefix if array.shape[1] == 1 else f"{prefix}_{i}" for i in range(array.shape[1]))
    return array, names, None


def _as_outcome(value: Any) -> tuple[np.ndarray, str, pd.Index | None]:
    index: pd.Index | None = None
    name = "y"
    if isinstance(value, pd.Series):
        index = value.index.copy()
        if value.name is not None:
            name = str(value.name)
        try:
            array = value.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("y must contain only numeric values.") from error
    else:
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("y must contain only numeric values.") from error
    if array.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    return array, name, index


def _as_clusters(value: Any) -> tuple[np.ndarray, pd.Index | None]:
    index: pd.Index | None = None
    if isinstance(value, pd.Series):
        index = value.index.copy()
        array = value.to_numpy()
    elif isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("clusters must contain exactly one column.")
        index = value.index.copy()
        array = value.iloc[:, 0].to_numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("clusters must be one-dimensional.")
    return array, index


def _validate_pandas_indices(indices: list[tuple[str, pd.Index | None]]) -> pd.Index | None:
    labeled = [(kind, index) for kind, index in indices if index is not None]
    if not labeled:
        return None
    reference_kind, reference = labeled[0]
    assert reference is not None
    for kind, index in labeled[1:]:
        assert index is not None
        if not reference.equals(index):
            raise ValueError(
                f"Pandas indices must match exactly and in the same order; "
                f"{kind} does not match {reference_kind}."
            )
    return reference.copy()


def prepare_iv_data(
    y: Any,
    endogenous: Any,
    instruments: Any,
    exogenous: Any | None,
    *,
    clusters: Any | None,
    missing: MissingPolicy,
) -> PreparedIVData:
    """Validate, align, and jointly filter an IV estimation sample."""

    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be 'raise' or 'drop'.")

    y_array, y_name, y_index = _as_outcome(y)
    endog_array, endog_names, endog_index = _as_numeric_frame(
        endogenous, kind="endogenous", prefix="endogenous"
    )
    instrument_array, instrument_names, instrument_index = _as_numeric_frame(
        instruments, kind="instruments", prefix="instrument"
    )
    if exogenous is None:
        exog_array = np.empty((len(y_array), 0), dtype=float)
        exog_names: tuple[str, ...] = ()
        exog_index = None
    else:
        exog_array, exog_names, exog_index = _as_numeric_frame(
            exogenous, kind="exogenous", prefix="exogenous"
        )

    cluster_array: np.ndarray | None = None
    cluster_index: pd.Index | None = None
    if clusters is not None:
        cluster_array, cluster_index = _as_clusters(clusters)

    nobs = len(y_array)
    lengths = {
        "endogenous": len(endog_array),
        "exogenous": len(exog_array),
        "instruments": len(instrument_array),
    }
    if cluster_array is not None:
        lengths["clusters"] = len(cluster_array)
    mismatched = {kind: size for kind, size in lengths.items() if size != nobs}
    if mismatched:
        raise ValueError(f"All inputs must have the same number of rows as y; got {mismatched}.")

    pandas_index = _validate_pandas_indices(
        [
            ("y", y_index),
            ("endogenous", endog_index),
            ("exogenous", exog_index),
            ("instruments", instrument_index),
            ("clusters", cluster_index),
        ]
    )
    index = pandas_index if pandas_index is not None else pd.RangeIndex(nobs)

    structural_overlap = set(endog_names) & set(exog_names)
    if structural_overlap:
        raise ValueError(
            "endogenous and exogenous column names must be distinct; "
            f"overlap: {sorted(structural_overlap)}."
        )
    instrument_overlap = set(instrument_names) & (set(endog_names) | set(exog_names))
    if instrument_overlap:
        raise ValueError(
            "instruments must contain excluded instruments only and use distinct names; "
            f"overlap: {sorted(instrument_overlap)}."
        )

    invalid = ~np.isfinite(y_array)
    invalid |= ~np.isfinite(endog_array).all(axis=1)
    if exog_array.shape[1]:
        invalid |= ~np.isfinite(exog_array).all(axis=1)
    invalid |= ~np.isfinite(instrument_array).all(axis=1)
    if cluster_array is not None:
        invalid |= pd.isna(cluster_array)

    dropped_rows = int(invalid.sum())
    if dropped_rows and missing == "raise":
        raise ValueError(
            f"Inputs contain missing or non-finite values in {dropped_rows} row(s); "
            "set missing='drop' to remove them jointly."
        )
    if dropped_rows:
        keep = ~invalid
        y_array = y_array[keep]
        endog_array = endog_array[keep]
        exog_array = exog_array[keep]
        instrument_array = instrument_array[keep]
        index = index[keep]
        if cluster_array is not None:
            cluster_array = cluster_array[keep]

    if len(y_array) == 0:
        raise ValueError("No complete observations remain after applying the missing-value policy.")

    return PreparedIVData(
        y=np.asarray(y_array, dtype=float),
        endogenous=np.asarray(endog_array, dtype=float),
        exogenous=np.asarray(exog_array, dtype=float),
        instruments=np.asarray(instrument_array, dtype=float),
        clusters=cluster_array,
        index=index,
        y_name=y_name,
        endogenous_names=endog_names,
        exogenous_names=exog_names,
        instrument_names=instrument_names,
        dropped_rows=dropped_rows,
    )


def prepare_rd_data(
    y: Any,
    running: Any,
    treatment: Any | None,
    *,
    clusters: Any | None,
    missing: MissingPolicy,
) -> PreparedRDData:
    """Validate, align, and jointly filter a regression-discontinuity sample."""

    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be 'raise' or 'drop'.")

    y_array, y_name, y_index = _as_outcome(y)
    running_frame, running_names, running_index = _as_numeric_frame(
        running, kind="running", prefix="running"
    )
    if running_frame.shape[1] != 1:
        raise ValueError("running must contain exactly one column.")

    treatment_array: np.ndarray | None = None
    treatment_name: str | None = None
    treatment_index: pd.Index | None = None
    if treatment is not None:
        treatment_frame, treatment_names, treatment_index = _as_numeric_frame(
            treatment, kind="treatment", prefix="treatment"
        )
        if treatment_frame.shape[1] != 1:
            raise ValueError("treatment must contain exactly one column.")
        treatment_array = treatment_frame[:, 0]
        treatment_name = treatment_names[0]

    cluster_array: np.ndarray | None = None
    cluster_index: pd.Index | None = None
    if clusters is not None:
        cluster_array, cluster_index = _as_clusters(clusters)

    nobs = len(y_array)
    lengths = {"running": len(running_frame)}
    if treatment_array is not None:
        lengths["treatment"] = len(treatment_array)
    if cluster_array is not None:
        lengths["clusters"] = len(cluster_array)
    mismatched = {kind: size for kind, size in lengths.items() if size != nobs}
    if mismatched:
        raise ValueError(f"All inputs must have the same number of rows as y; got {mismatched}.")

    pandas_index = _validate_pandas_indices(
        [
            ("y", y_index),
            ("running", running_index),
            ("treatment", treatment_index),
            ("clusters", cluster_index),
        ]
    )
    index = pandas_index if pandas_index is not None else pd.RangeIndex(nobs)

    running_array = running_frame[:, 0]
    invalid = ~np.isfinite(y_array) | ~np.isfinite(running_array)
    if treatment_array is not None:
        invalid |= ~np.isfinite(treatment_array)
    if cluster_array is not None:
        invalid |= pd.isna(cluster_array)

    dropped_rows = int(invalid.sum())
    if dropped_rows and missing == "raise":
        raise ValueError(
            f"Inputs contain missing or non-finite values in {dropped_rows} row(s); "
            "set missing='drop' to remove them jointly."
        )
    if dropped_rows:
        keep = ~invalid
        y_array = y_array[keep]
        running_array = running_array[keep]
        index = index[keep]
        if treatment_array is not None:
            treatment_array = treatment_array[keep]
        if cluster_array is not None:
            cluster_array = cluster_array[keep]

    if len(y_array) == 0:
        raise ValueError("No complete observations remain after applying the missing-value policy.")

    return PreparedRDData(
        y=np.asarray(y_array, dtype=float),
        running=np.asarray(running_array, dtype=float),
        treatment=(
            np.asarray(treatment_array, dtype=float) if treatment_array is not None else None
        ),
        clusters=cluster_array,
        index=index,
        y_name=y_name,
        running_name=running_names[0],
        treatment_name=treatment_name,
        dropped_rows=dropped_rows,
    )


def prepare_prediction_data(
    endogenous: Any,
    exogenous: Any | None,
    *,
    endogenous_names: tuple[str, ...],
    exogenous_names: tuple[str, ...],
    add_constant: bool,
) -> tuple[np.ndarray, pd.Index]:
    """Build a prediction design while enforcing the fitted schema."""

    endog_array, supplied_endog_names, endog_index = _as_numeric_frame(
        endogenous, kind="endogenous", prefix="endogenous"
    )
    if endog_array.shape[1] != len(endogenous_names):
        raise ValueError(
            f"endogenous must contain {len(endogenous_names)} column(s); "
            f"received {endog_array.shape[1]}."
        )
    if (
        isinstance(endogenous, (pd.Series, pd.DataFrame))
        and supplied_endog_names != endogenous_names
    ):
        raise ValueError(
            f"endogenous columns must match the fitted schema and order: {list(endogenous_names)}."
        )

    if exogenous_names:
        if exogenous is None:
            raise ValueError(
                f"exogenous is required and must contain columns {list(exogenous_names)}."
            )
        exog_array, supplied_exog_names, exog_index = _as_numeric_frame(
            exogenous, kind="exogenous", prefix="exogenous"
        )
        if exog_array.shape[1] != len(exogenous_names):
            raise ValueError(
                f"exogenous must contain {len(exogenous_names)} column(s); "
                f"received {exog_array.shape[1]}."
            )
        if (
            isinstance(exogenous, (pd.Series, pd.DataFrame))
            and supplied_exog_names != exogenous_names
        ):
            raise ValueError(
                "exogenous columns must match the fitted schema and order: "
                f"{list(exogenous_names)}."
            )
    else:
        if exogenous is not None:
            raise ValueError(
                "exogenous must be None because the fitted model has no exogenous columns."
            )
        exog_array = np.empty((len(endog_array), 0), dtype=float)
        exog_index = None

    if len(exog_array) != len(endog_array):
        raise ValueError("endogenous and exogenous must have the same number of rows.")
    index = _validate_pandas_indices([("endogenous", endog_index), ("exogenous", exog_index)])
    if index is None:
        index = pd.RangeIndex(len(endog_array))
    if not np.isfinite(endog_array).all() or not np.isfinite(exog_array).all():
        raise ValueError("Prediction inputs contain missing or non-finite values.")

    pieces: list[np.ndarray] = []
    if add_constant:
        pieces.append(np.ones((len(endog_array), 1), dtype=float))
    if exog_array.shape[1]:
        pieces.append(exog_array)
    pieces.append(endog_array)
    return np.column_stack(pieces), index

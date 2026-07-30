"""One-run causal-ML performance comparison on verified real data.

CauseKit's installed package has no external ML dependency. Comparator imports in this
benchmark are optional and never serve as an estimation backend for the native model.

Example:

    python benchmarks/benchmark_ml.py --dataset nsw_mixtape --models all \
        --output ml-benchmark-nsw.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import causekit
from causekit import PartiallyLinearDML
from causekit.datasets import REAL_DATASETS, load_real_dataset

SEED = 20_260_729
DEFAULT_DATASET = "cattaneo2"
MODEL_NAMES = (
    "causekit_native_ridge_gcv",
    "sklearn_ridge_cv",
    "sklearn_hist_gradient_boosting",
    "sklearn_random_forest",
)


@dataclass(frozen=True)
class _DesignSpecification:
    design_id: str
    outcome: str
    treatment: str
    covariates: tuple[str, ...]


_DESIGNS = {
    "cattaneo2": _DesignSpecification(
        design_id="cattaneo2_maternal_smoking_birthweight_v1",
        outcome="bweight",
        treatment="mbsmoke",
        covariates=("mmarried", "mage", "medu", "fbaby"),
    ),
    "nsw_mixtape": _DesignSpecification(
        design_id="nsw_job_training_earnings_v1",
        outcome="re78",
        treatment="treat",
        covariates=(
            "age",
            "educ",
            "black",
            "hisp",
            "marr",
            "nodegree",
            "re74",
            "re75",
        ),
    ),
}
DATASET_NAMES = tuple(_DESIGNS)


def _design(
    dataset: str = DEFAULT_DATASET,
    *,
    data_directory: Path | None,
    download: bool,
) -> dict[str, Any]:
    try:
        specification = _DESIGNS[dataset]
    except KeyError as error:  # pragma: no cover - protected by command-line choices
        available = ", ".join(DATASET_NAMES)
        raise ValueError(
            f"Unknown ML benchmark dataset {dataset!r}; choose one of: {available}."
        ) from error
    data = load_real_dataset(
        dataset,
        data_directory=data_directory,
        download=download,
    )
    covariates = data.loc[:, list(specification.covariates)].astype(float)
    return {
        "design_id": specification.design_id,
        "outcome_name": specification.outcome,
        "treatment_name": specification.treatment,
        "covariate_names": specification.covariates,
        "covariates": covariates,
        "treatment": data[specification.treatment].astype(float),
        "outcome": data[specification.outcome].astype(float),
    }


def _sklearn_factories(name: str) -> tuple[Callable[[], Any], Callable[[], Any], str]:
    try:
        import sklearn
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("scikit-learn is not installed in this benchmark environment") from error

    if name == "sklearn_ridge_cv":

        def factory() -> Any:
            return make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-6, 4, 6)),
            )

    elif name == "sklearn_hist_gradient_boosting":

        def factory() -> Any:
            return HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=31,
                learning_rate=0.08,
                random_state=SEED,
            )

    elif name == "sklearn_random_forest":

        def factory() -> Any:
            return RandomForestRegressor(
                n_estimators=50,
                min_samples_leaf=5,
                max_features=0.8,
                n_jobs=1,
                random_state=SEED,
            )

    else:  # pragma: no cover - protected by command-line choices
        raise ValueError(f"Unknown comparator {name!r}.")
    return factory, factory, sklearn.__version__


def _estimator(name: str) -> tuple[PartiallyLinearDML, str | None]:
    if name == "causekit_native_ridge_gcv":
        return PartiallyLinearDML(n_splits=3, random_state=SEED), None
    outcome_factory, treatment_factory, version = _sklearn_factories(name)
    return (
        PartiallyLinearDML(
            outcome_factory=outcome_factory,
            treatment_factory=treatment_factory,
            n_splits=3,
            random_state=SEED,
        ),
        version,
    )


def _run_once(name: str, design: dict[str, Any]) -> dict[str, Any]:
    try:
        estimator, comparator_version = _estimator(name)
    except RuntimeError as error:
        return {"model": name, "status": "unavailable", "reason": str(error)}

    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    result = estimator.fit(
        design["outcome"],
        treatment=design["treatment"],
        covariates=design["covariates"],
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    outcome_error = (
        result.nuisance_predictions["outcome_mean"].to_numpy() - design["outcome"].to_numpy()
    )
    treatment_error = (
        result.nuisance_predictions["treatment_mean"].to_numpy() - design["treatment"].to_numpy()
    )
    interval = result.conf_int()
    return {
        "model": name,
        "status": "completed",
        "comparator_version": comparator_version,
        "benchmark_repetitions": 1,
        "outer_folds": result.n_splits,
        "estimate": result.estimate,
        "standard_error": result.standard_error,
        "ci_lower": float(interval["lower"]),
        "ci_upper": float(interval["upper"]),
        "outcome_oof_rmse": float(np.sqrt(np.mean(outcome_error**2))),
        "treatment_oof_rmse": float(np.sqrt(np.mean(treatment_error**2))),
        "elapsed_seconds": elapsed,
        "python_peak_memory_mib": peak / 1024**2,
        "outcome_model": result.outcome_model_name,
        "treatment_model": result.treatment_model_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default=DEFAULT_DATASET,
        help="Hash-pinned real-data design to benchmark.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*MODEL_NAMES, "all"],
        default=["all"],
    )
    parser.add_argument("--data-directory", type=Path, default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the absent hash-pinned source over verified HTTPS.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = list(MODEL_NAMES) if "all" in args.models else list(dict.fromkeys(args.models))
    design = _design(
        args.dataset,
        data_directory=args.data_directory,
        download=args.download,
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": design["design_id"],
        "dataset": args.dataset,
        "dataset_source_sha256": REAL_DATASETS[args.dataset].sha256,
        "outcome": design["outcome_name"],
        "treatment": design["treatment_name"],
        "covariates": list(design["covariate_names"]),
        "seed": SEED,
        "nobs": len(design["outcome"]),
        "benchmark_repetitions": 1,
        "comparison_policy": "one identically seeded three-fold fit per model",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "causekit": causekit.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "results": [_run_once(name, design) for name in names],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""One-run honest DR-learner comparison on the hash-verified NSW data.

Every model uses the same estimator contract, native nuisance specifications, immutable
roles, seed, folds, and evaluation metrics. Only the unweighted DR pseudo-outcome CATE
regressor changes. Existing R-learner benchmark artifacts are read only and are not rerun.

Example:

    python benchmarks/benchmark_drlearner.py --models all \
        --output %LOCALAPPDATA%/causekit/benchmarks/causekit-drlearner-nsw-v1.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
import tracemalloc
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import causekit
from causekit import DRLearner
from causekit.datasets import REAL_DATASETS, load_real_dataset

SEED = 20_260_730
DATASET = "nsw_mixtape"
DESIGN_ID = "nsw_job_training_honest_dr_cate_v1"
COVARIATES = ("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75")
MODEL_NAMES = (
    "causekit_native_ridge_gcv",
    "sklearn_ridge_cv",
    "sklearn_hist_gradient_boosting",
    "sklearn_random_forest",
)


def _design(*, data_directory: Path | None, download: bool) -> dict[str, Any]:
    data = load_real_dataset(DATASET, data_directory=data_directory, download=download)
    return {
        "covariates": data.loc[:, list(COVARIATES)].astype(float),
        "treatment": data["treat"].astype(float),
        "outcome": data["re78"].astype(float),
    }


class _StandardizedRidgeCV:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _StandardizedRidgeCV:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        self.columns_ = X.columns.copy()
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = RidgeCV(alphas=np.logspace(-6, 4, 6)).fit(self.scaler_.transform(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not X.columns.equals(self.columns_):
            raise ValueError("Prediction schema drift in benchmark comparator.")
        return np.asarray(self.model_.predict(self.scaler_.transform(X)), dtype=float)


def _external_factory(name: str) -> tuple[Callable[[], Any], str]:
    try:
        import sklearn
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    except ImportError as error:
        raise RuntimeError("scikit-learn is not installed in this benchmark environment") from error

    if name == "sklearn_ridge_cv":
        factory: Callable[[], Any] = _StandardizedRidgeCV
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
                n_estimators=100,
                min_samples_leaf=5,
                max_features=0.8,
                n_jobs=1,
                random_state=SEED,
            )

    else:  # pragma: no cover - command-line choices protect this path
        raise ValueError(f"Unknown DR-learner comparator {name!r}.")
    return factory, sklearn.__version__


def _estimator(name: str) -> tuple[DRLearner, str | None]:
    if name == "causekit_native_ridge_gcv":
        cate_factory = None
        comparator_version = None
    else:
        cate_factory, comparator_version = _external_factory(name)
    return (
        DRLearner(
            cate_factory=cate_factory,
            n_splits=3,
            evaluation_fraction=0.5,
            random_state=SEED,
            calibration_groups=4,
            bootstrap_iterations=999,
            bootstrap_random_state=SEED,
            propensity_tuning_splits=3,
        ),
        comparator_version,
    )


def _index_digest(index: pd.Index) -> str:
    rendered = "\n".join(str(value) for value in index).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _cate_diagnostic(result: Any, name: str) -> Any:
    if result.cate_diagnostics.empty or name not in result.cate_diagnostics:
        return None
    value = result.cate_diagnostics.iloc[0][name]
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


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
    return {
        "model": name,
        "status": "completed",
        "comparator_version": comparator_version,
        "benchmark_repetitions": 1,
        "construction_nobs": result.construction_nobs,
        "evaluation_nobs": result.evaluation_nobs,
        "evaluation_index_sha256": _index_digest(result.evaluation_index),
        "honest_dr_loss": result.honest_dr_loss,
        "honest_constant_dr_loss": result.honest_constant_dr_loss,
        "dr_loss_gain": result.dr_loss_gain,
        "calibration_level": result.calibration_coefficients["level"],
        "calibration_heterogeneity": result.calibration_coefficients["heterogeneity"],
        "heterogeneity_zero_pvalue": result.calibration_tests.loc["heterogeneity=0", "p_value"],
        "heterogeneity_one_pvalue": result.calibration_tests.loc["heterogeneity=1", "p_value"],
        "group_effect_min": result.group_effects["effect"].min(),
        "group_effect_max": result.group_effects["effect"].max(),
        "minimum_propensity": result.minimum_propensity,
        "maximum_propensity": result.maximum_propensity,
        "simultaneous_critical_value": result.simultaneous_critical_value,
        "elapsed_seconds": elapsed,
        "python_peak_memory_mib": peak / 1024**2,
        "outcome_control_model": result.outcome_control_model_name,
        "outcome_treated_model": result.outcome_treated_model_name,
        "propensity_model": result.propensity_model_name,
        "cate_model": result.cate_model_name,
        "cate_selected_alpha": _cate_diagnostic(result, "selected_alpha"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[*MODEL_NAMES, "all"], default=["all"])
    parser.add_argument("--data-directory", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = list(MODEL_NAMES) if "all" in args.models else list(dict.fromkeys(args.models))
    design = _design(data_directory=args.data_directory, download=args.download)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": DESIGN_ID,
        "dataset": DATASET,
        "dataset_source_sha256": REAL_DATASETS[DATASET].sha256,
        "outcome": "re78",
        "treatment": "treat",
        "covariates": list(COVARIATES),
        "seed": SEED,
        "outer_folds": 3,
        "evaluation_fraction": 0.5,
        "calibration_groups": 4,
        "bootstrap_iterations": 999,
        "nobs": len(design["outcome"]),
        "benchmark_repetitions": 1,
        "comparison_policy": (
            "one identical honest split per model; native nuisances fixed in specification; "
            "only the unweighted DR pseudo-outcome CATE learner changes; settled R-learner "
            "artifacts are not rerun"
        ),
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

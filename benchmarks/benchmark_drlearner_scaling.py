"""Reproducible linear-memory clustered smoke for the honest DR learner."""

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

import causekit
from causekit import DRLearner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20_260_730)
    n_clusters = 500
    rows_per_cluster = 4
    nobs = n_clusters * rows_per_cluster
    covariates = pd.DataFrame(rng.normal(size=(nobs, 6)), columns=[f"x{i}" for i in range(6)])
    clusters = pd.Series(np.repeat(np.arange(n_clusters), rows_per_cluster), name="cluster")
    treatment = pd.Series(np.tile([0.0, 1.0, 0.0, 1.0], n_clusters), name="treatment")
    cate = 0.8 + 0.5 * covariates["x0"] - 0.2 * covariates["x2"]
    cluster_noise = np.repeat(rng.normal(scale=0.25, size=n_clusters), rows_per_cluster)
    outcome = (
        0.4 * covariates["x1"] + treatment * cate + cluster_noise + rng.normal(scale=0.5, size=nobs)
    )
    estimator = DRLearner(
        n_splits=3,
        evaluation_fraction=0.4,
        random_state=20_260_730,
        covariance="clustered",
        calibration_groups=4,
        bootstrap_iterations=99,
        propensity_tuning_splits=2,
    )
    tracemalloc.start()
    started = time.perf_counter()
    result = estimator.fit(
        outcome,
        treatment=treatment,
        covariates=covariates,
        clusters=clusters,
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {
        "contract": "honest_drlearner_clustered_linear_memory_smoke_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "causekit": causekit.__version__,
        "seed": 20_260_730,
        "nobs": nobs,
        "n_clusters": n_clusters,
        "construction_nobs": result.construction_nobs,
        "evaluation_nobs": result.evaluation_nobs,
        "n_splits": result.n_splits,
        "elapsed_seconds": elapsed,
        "python_peak_memory_mib": peak / 1024**2,
        "group_influence_shape": list(result.group_influence.shape),
        "nuisance_prediction_columns": list(result.construction_nuisance_predictions.columns),
        "quadratic_distance_matrix": False,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

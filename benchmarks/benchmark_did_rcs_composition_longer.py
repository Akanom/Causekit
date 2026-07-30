"""Fixed-size performance gate for the longer composition-robust pair lattice."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import causekit
from causekit import CrossFitter, RepeatedCrossSectionDiD

try:
    from benchmarks._did_rcs_composition_support import (
        QuadraticOutcome,
        SaturatedClassProbability,
    )
except ModuleNotFoundError:
    from _did_rcs_composition_support import (  # type: ignore[no-redef]
        QuadraticOutcome,
        SaturatedClassProbability,
    )

DEFAULT_N = 120_000
DEFAULT_SPLITS = 2
DEFAULT_SEED = 20_260_730
ELAPSED_GATE_SECONDS = 30.0
PEAK_PYTHON_MIB_GATE = 1_024.0
N_GLOBAL_CELLS = 15
N_PAIRS = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data(n_observations: int, seed: int) -> pd.DataFrame:
    if n_observations < 300 or n_observations % N_GLOBAL_CELLS:
        raise ValueError("n_observations must be at least 300 and divisible by 15.")
    rng = np.random.default_rng(seed)
    row = np.arange(n_observations)
    global_cell = row % N_GLOBAL_CELLS
    group = global_cell // 5
    time_value = global_cell % 5 + 1
    treatment_time = np.choose(group, [3.0, 4.0, np.inf])
    x = (row // N_GLOBAL_CELLS) % 2
    z = rng.uniform(-1.0, 1.0, n_observations)
    untreated = 1.0 + 0.4 * time_value + 0.6 * x + 0.3 * z + 0.2 * z**2
    event_time = time_value - treatment_time
    effect = np.where(
        np.isfinite(treatment_time) & (event_time >= 0),
        1.5 + 0.35 * event_time + 0.8 * x + 0.25 * z**2,
        0.0,
    )
    outcome = untreated + effect + rng.normal(scale=0.8 + 0.2 * x)
    return pd.DataFrame(
        {
            "outcome": outcome,
            "time": time_value.astype(float),
            "treatment_time": treatment_time,
            "x": x,
            "z": z,
            "global_cell": global_cell,
        }
    )


def _probability_bounds(result: Any) -> tuple[float, float]:
    nuisance = result.nuisance_predictions.columns.get_level_values("nuisance").astype(str)
    values = result.nuisance_predictions.loc[:, nuisance.str.contains("propensity")].to_numpy(
        dtype=float
    )
    finite = values[np.isfinite(values)]
    return float(finite.min()), float(finite.max())


def run_benchmark(
    *,
    n_observations: int,
    n_splits: int,
    seed: int,
    measure_memory: bool,
) -> dict[str, Any]:
    data = _data(n_observations, seed)
    cross_fitter = CrossFitter(
        propensity_factory=SaturatedClassProbability,
        outcome_factory=QuadraticOutcome,
        n_splits=n_splits,
        random_state=seed,
    )
    if measure_memory:
        tracemalloc.start()
    started = time.perf_counter()
    result = RepeatedCrossSectionDiD(composition="robust").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x", "z"],
        cross_fitter=cross_fitter,
    )
    elapsed = time.perf_counter() - started
    peak_mib: float | None = None
    if measure_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mib = peak_bytes / (1024.0**2)
    minimum_probability, maximum_probability = _probability_bounds(result)
    return {
        "n_observations": n_observations,
        "n_splits": n_splits,
        "seed": seed,
        "global_cells": int(data["global_cell"].nunique()),
        "pair_count": len(result.pair_ledger),
        "group_time_effects": len(result.group_time),
        "conditional_placebos": len(result.pretrend.placebo_effects),
        "nuisance_task_fits": len(result.nuisance_diagnostics),
        "minimum_probability": minimum_probability,
        "maximum_probability": maximum_probability,
        "elapsed_seconds": elapsed,
        "peak_python_mib": peak_mib,
        "finite": bool(np.isfinite([result.estimate, result.standard_error]).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-observations", type=int, default=DEFAULT_N)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/did_rcs_composition_longer_performance_evidence.json"),
    )
    args = parser.parse_args()
    result = run_benchmark(
        n_observations=args.n_observations,
        n_splits=args.n_splits,
        seed=args.seed,
        measure_memory=True,
    )
    result.update(
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "benchmarks/benchmark_did_rcs_composition_longer.py",
            "generator_sha256": _sha256(Path(__file__)),
            "causekit_version": causekit.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_gate_seconds": ELAPSED_GATE_SECONDS,
            "peak_python_mib_gate": PEAK_PYTHON_MIB_GATE,
        }
    )
    peak = result["peak_python_mib"]
    result["status"] = (
        "pass"
        if result["finite"]
        and result["global_cells"] == N_GLOBAL_CELLS
        and result["pair_count"] == N_PAIRS
        and result["group_time_effects"] == 5
        and result["conditional_placebos"] == 3
        and result["nuisance_task_fits"] == N_PAIRS * 4 * args.n_splits
        and result["minimum_probability"] > 1e-6
        and result["maximum_probability"] < 1.0 - 1e-6
        and result["elapsed_seconds"] <= ELAPSED_GATE_SECONDS
        and peak is not None
        and peak <= PEAK_PYTHON_MIB_GATE
        else "fail"
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if result["status"] != "pass":
        raise SystemExit("longer composition performance gate failed")


if __name__ == "__main__":
    main()

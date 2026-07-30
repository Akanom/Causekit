"""Fixed-size performance gate for composition-robust repeated-section DiD.

The benchmark uses exactly balanced four-cell data and fold-fitted benchmark nuisance
providers. It records wall time, Python allocation peak, overlap, and the complete
nuisance-task count without constructing any pairwise observation matrix.
"""

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
except ModuleNotFoundError:  # Direct ``python benchmarks/...py`` execution.
    from _did_rcs_composition_support import (  # type: ignore[no-redef]
        QuadraticOutcome,
        SaturatedClassProbability,
    )

DEFAULT_N = 100_000
DEFAULT_SPLITS = 2
DEFAULT_SEED = 20_260_730
ELAPSED_GATE_SECONDS = 10.0
PEAK_PYTHON_MIB_GATE = 512.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data(n_observations: int, seed: int) -> pd.DataFrame:
    if isinstance(n_observations, bool) or not isinstance(n_observations, int):
        raise TypeError("n_observations must be an integer.")
    if n_observations < 40 or n_observations % 4:
        raise ValueError("n_observations must be at least 40 and divisible by four.")
    rng = np.random.default_rng(seed)
    row = np.arange(n_observations)
    cell = row % 4
    treated = cell // 2
    post = cell % 2
    x = (row // 4) % 2
    z = rng.uniform(-1.0, 1.0, n_observations)
    m00 = 1.0 + 0.6 * x + 0.3 * z + 0.2 * z**2
    m01 = m00 + 0.5 + 0.4 * x - 0.2 * z + 0.15 * z**2
    m10 = 2.0 + 0.2 * x + 0.5 * z - 0.1 * z**2
    tau = 2.0 + 4.0 * x + 0.5 * z**2
    m11 = m10 + m01 - m00 + tau
    means = np.choose(cell, [m00, m01, m10, m11])
    outcome = means + rng.normal(scale=0.8 + 0.2 * x + 0.1 * post)
    return pd.DataFrame(
        {
            "outcome": outcome,
            "time": post + 1.0,
            "treatment_time": np.where(treated == 1, 2.0, np.inf),
            "x": x,
            "z": z,
            "cell": cell,
        }
    )


def _probability_bounds(result: Any) -> tuple[float, float]:
    nuisance = result.nuisance_predictions.columns.get_level_values("nuisance").astype(str)
    selected = result.nuisance_predictions.loc[:, nuisance.str.contains("propensity")]
    values = selected.to_numpy(dtype=float)
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
    cell_counts = data["cell"].value_counts(sort=False).sort_index().astype(int).tolist()
    finite = np.isfinite([result.estimate, result.standard_error]).all()
    overlap = minimum_probability > 1e-6 and maximum_probability < 1.0 - 1e-6
    return {
        "n_observations": n_observations,
        "n_splits": n_splits,
        "seed": seed,
        "cell_counts": cell_counts,
        "estimate": result.estimate,
        "standard_error": result.standard_error,
        "minimum_probability": minimum_probability,
        "maximum_probability": maximum_probability,
        "nuisance_task_fits": len(result.nuisance_diagnostics),
        "elapsed_seconds": elapsed,
        "peak_python_mib": peak_mib,
        "finite": bool(finite),
        "overlap": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-observations", type=int, default=DEFAULT_N)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/did_rcs_composition_performance_evidence.json"),
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
            "generator": "benchmarks/benchmark_did_rcs_composition.py",
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
        and result["overlap"]
        and result["nuisance_task_fits"] == 4 * args.n_splits
        and result["elapsed_seconds"] <= ELAPSED_GATE_SECONDS
        and peak is not None
        and peak <= PEAK_PYTHON_MIB_GATE
        else "fail"
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if result["status"] != "pass":
        raise SystemExit("composition performance gate failed")


if __name__ == "__main__":
    main()

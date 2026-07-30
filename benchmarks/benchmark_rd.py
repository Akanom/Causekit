"""Large-n fixed-bandwidth smoke benchmark for native RD estimation."""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from datetime import date
from pathlib import Path

import numpy as np

import causekit
from causekit import RegressionDiscontinuity

OUTPUT = Path("benchmarks/rd_performance_evidence.json")


def main() -> None:
    nobs = 200_000
    rng = np.random.default_rng(2_026_073_011)
    running = rng.uniform(-2.0, 2.0, nobs)
    truth = 1.5
    outcome = (
        0.4
        + 0.6 * running
        + 0.2 * running**2
        + truth * (running >= 0.0)
        + rng.normal(scale=0.8, size=nobs)
    )
    estimator = RegressionDiscontinuity(
        bandwidth=1.0,
        bias_bandwidth=1.4,
        polynomial_order=1,
        bias_order=2,
    )

    tracemalloc.start()
    started = time.perf_counter()
    result = estimator.fit(outcome, running=running)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mib = peak / (1024**2)

    gates = {
        "maximum_seconds": 8.0,
        "maximum_peak_mib": 300.0,
        "maximum_absolute_error": 0.05,
        "no_quadratic_matrix": True,
    }
    status = (
        "pass"
        if elapsed <= gates["maximum_seconds"]
        and peak_mib <= gates["maximum_peak_mib"]
        and abs(result.bias_corrected_estimate - truth) <= gates["maximum_absolute_error"]
        else "fail"
    )
    evidence = {
        "artifact": "causekit_regression_discontinuity_performance_v1",
        "generated_on": date.today().isoformat(),
        "generator": "benchmarks/benchmark_rd.py",
        "reproduction_command": "python benchmarks/benchmark_rd.py",
        "causekit_version": causekit.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "design": {
            "nobs": nobs,
            "truth": truth,
            "bandwidth": 1.0,
            "bias_bandwidth": 1.4,
            "polynomial_order": 1,
            "bias_order": 2,
            "kernel": "triangular",
        },
        "result": {
            "elapsed_seconds": elapsed,
            "peak_memory_mib": peak_mib,
            "bias_corrected_estimate": result.bias_corrected_estimate,
            "robust_standard_error": result.robust_standard_error,
            "absolute_error": abs(result.bias_corrected_estimate - truth),
            "n_effective": result.n_effective,
        },
        "gates": gates,
        "status": status,
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"results_file={OUTPUT.as_posix()}")
    print(f"performance_status={status}")
    if status != "pass":
        raise SystemExit(9)


if __name__ == "__main__":
    main()

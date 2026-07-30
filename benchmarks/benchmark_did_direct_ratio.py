"""Fixed-period vectorized-n benchmark for direct cohort-odds PT-All DiD."""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from causekit import CrossFitter, EfficientDiD, __version__

try:
    from benchmarks._did_direct_ratio_support import EmpiricalOdds, MeanRegression
except ModuleNotFoundError:  # direct ``python benchmarks/script.py`` execution
    from _did_direct_ratio_support import EmpiricalOdds, MeanRegression  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "did_direct_ratio_performance_evidence.json"


def _panel(entities_per_cohort: int = 50_000) -> pd.DataFrame:
    position = np.arange(entities_per_cohort, dtype=float)
    x = ((position % 101.0) - 50.0) / 25.0
    trend = 0.4 + 0.25 * x + 0.1 * np.sin(position)
    records: list[pd.DataFrame] = []
    for cohort, prefix, effect in ((2.0, "g", 2.0), (np.inf, "n", 0.0)):
        entity = np.array([f"{prefix}_{value:05d}" for value in range(entities_per_cohort)])
        baseline = 1.0 + 0.5 * x + 0.05 * np.cos(position)
        records.append(
            pd.DataFrame(
                {
                    "entity": np.repeat(entity, 2),
                    "time": np.tile([1.0, 2.0], entities_per_cohort),
                    "treatment_time": cohort,
                    "x": np.repeat(x, 2),
                    "outcome": np.column_stack([baseline, baseline + trend + effect]).ravel(),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def main() -> None:
    panel = _panel()
    tracemalloc.start()
    started = time.perf_counter()
    result = EfficientDiD().fit(
        panel,
        outcome="outcome",
        entity="entity",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=CrossFitter(
            cohort_ratio_factory=EmpiricalOdds,
            outcome_factory=MeanRegression,
            n_splits=5,
            random_state=20260730,
        ),
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    evidence = {
        "schema": "causekit_did_direct_ratio_performance_v1",
        "causekit_version": __version__,
        "generated_on": "2026-07-30",
        "generator": "benchmarks/benchmark_did_direct_ratio.py",
        "reproduction_command": "python benchmarks/benchmark_did_direct_ratio.py",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "panel_rows": len(panel),
        "entities": result.n_entities,
        "periods": result.n_periods,
        "ordered_fitted_pairs": int(
            result.cohort_ratio_diagnostics.loc[
                result.cohort_ratio_diagnostics["source"].eq("fitted"),
                ["numerator", "denominator"],
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "folds": 5,
        "elapsed_seconds": elapsed,
        "python_peak_mib": peak / (1024**2),
        "estimate": result.estimate,
        "absolute_bias": abs(result.estimate - 2.0),
        "minimum_importance_effective_n": float(
            result.cohort_ratio_diagnostics["denominator_importance_effective_n"].min()
        ),
        "maximum_importance_share": float(
            result.cohort_ratio_diagnostics["denominator_importance_max_share"].max()
        ),
        "runtime_gate_seconds": 10.0,
        "memory_gate_mib": 300.0,
        "bias_gate": 1e-4,
    }
    evidence["status"] = (
        "pass"
        if elapsed <= evidence["runtime_gate_seconds"]
        and evidence["python_peak_mib"] <= evidence["memory_gate_mib"]
        and evidence["absolute_bias"] <= evidence["bias_gate"]
        else "fail"
    )
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if evidence["status"] != "pass":
        raise SystemExit("Direct-ratio performance gate failed.")


if __name__ == "__main__":
    main()

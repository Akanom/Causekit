"""Large-panel compact-within performance smoke for Panel IV."""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import causekit
from causekit import PanelIV2SLS

OUTPUT = Path("benchmarks/panel_iv_performance_evidence.json")


def main() -> None:
    n_entities = 20_000
    n_periods = 10
    nobs = n_entities * n_periods
    rng = np.random.default_rng(2_026_073_401)
    entities = np.repeat(np.arange(n_entities), n_periods)
    periods = np.tile(np.arange(n_periods), n_entities)
    entity_effect = np.repeat(rng.normal(size=n_entities), n_periods)
    instrument = rng.normal(size=nobs)
    control = rng.normal(size=nobs)
    first_error = rng.normal(size=nobs)
    endogenous = 0.85 * instrument + 0.3 * control + 0.25 * entity_effect + first_error
    truth = 1.5
    outcome = (
        truth * endogenous
        - 0.4 * control
        + entity_effect
        + 0.15 * periods
        + 0.55 * first_error
        + rng.normal(scale=0.5, size=nobs)
    )
    data = pd.DataFrame(
        {
            "entity": entities,
            "time": periods,
            "outcome": outcome,
            "endogenous": endogenous,
            "instrument": instrument,
            "control": control,
        }
    )

    tracemalloc.start()
    started = time.perf_counter()
    result = PanelIV2SLS().fit(
        data,
        outcome="outcome",
        endogenous="endogenous",
        instruments="instrument",
        exogenous="control",
        entity="entity",
        time="time",
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mib = peak / (1024**2)
    gates = {
        "maximum_seconds": 8.0,
        "maximum_peak_mib": 350.0,
        "maximum_absolute_error": 0.03,
        "no_fixed_effect_dummy_matrix": True,
        "no_observation_projection_matrix": True,
    }
    absolute_error = abs(float(result.params["endogenous"]) - truth)
    status = (
        "pass"
        if elapsed <= gates["maximum_seconds"]
        and peak_mib <= gates["maximum_peak_mib"]
        and absolute_error <= gates["maximum_absolute_error"]
        else "fail"
    )
    evidence = {
        "artifact": "causekit_panel_iv_performance_v1",
        "generated_on": date.today().isoformat(),
        "generator": "benchmarks/benchmark_panel_iv.py",
        "reproduction_command": "python benchmarks/benchmark_panel_iv.py",
        "causekit_version": causekit.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "design": {
            "nobs": nobs,
            "n_entities": n_entities,
            "n_periods": n_periods,
            "effects": ["entity", "time"],
            "covariance": "entity-clustered CR1",
            "truth": truth,
        },
        "result": {
            "elapsed_seconds": elapsed,
            "peak_memory_mib": peak_mib,
            "estimate": float(result.params["endogenous"]),
            "standard_error": float(result.standard_errors["endogenous"]),
            "absolute_error": absolute_error,
            "within_iterations": result.within_iterations,
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

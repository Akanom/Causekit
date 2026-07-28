"""Deterministic balanced-panel performance smoke for causalkit DiD estimators."""

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc

import numpy as np
import pandas as pd

import causalkit
from causalkit import DifferenceInDifferences, EfficientDiD


def _panel(n_entities: int, n_periods: int, seed: int) -> pd.DataFrame:
    if n_entities < 12:
        raise ValueError("n_entities must be at least 12.")
    if n_periods < 5:
        raise ValueError("n_periods must be at least 5.")
    rng = np.random.default_rng(seed)
    cohorts = np.array([3.0, float(n_periods - 1), np.inf])
    assigned = cohorts[np.arange(n_entities) % len(cohorts)]
    entity = np.repeat(np.arange(n_entities), n_periods)
    period = np.tile(np.arange(1, n_periods + 1), n_entities)
    treatment_time = np.repeat(assigned, n_periods)
    levels = np.repeat(rng.normal(scale=2.0, size=n_entities), n_periods)
    slopes = np.repeat(rng.normal(scale=0.15, size=n_entities), n_periods)
    shocks = rng.normal(scale=0.5, size=n_entities * n_periods)
    treated = np.isfinite(treatment_time) & (period >= treatment_time)
    event_time = period - treatment_time
    effect = np.where(treated, 1.0 + 0.25 * event_time, 0.0)
    outcome = levels + 0.4 * period + slopes * period + shocks + effect
    return pd.DataFrame(
        {
            "entity": entity,
            "time": period,
            "treatment_time": treatment_time,
            "outcome": outcome,
        }
    )


def _fit(scenario: str, panel: pd.DataFrame):
    model = (
        DifferenceInDifferences(control_group="not_yet_treated")
        if scenario == "conventional"
        else EfficientDiD(pre_periods="all")
    )
    return model.fit(
        panel,
        outcome="outcome",
        entity="entity",
        time="time",
        treatment_time="treatment_time",
    )


def _run(
    scenario: str,
    panel: pd.DataFrame,
    *,
    measure_memory: bool,
) -> dict[str, object]:
    if measure_memory:
        tracemalloc.start()
    started = time.perf_counter()
    result = _fit(scenario, panel)
    elapsed = time.perf_counter() - started
    peak_mib = None
    if measure_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mib = peak / 1024**2
    if not np.isfinite(result.estimate):
        raise RuntimeError("benchmark produced a non-finite estimate")
    return {
        "scenario": scenario,
        "elapsed_seconds": elapsed,
        "peak_python_mib": peak_mib,
        "estimate": result.estimate,
        "standard_error": result.standard_error,
        "group_time_effects": len(result.group_time),
        "event_time_effects": len(result.event_study),
        "efficiency_candidates": len(result.efficiency_weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("conventional", "efficient", "all"),
        default="all",
    )
    parser.add_argument("--n-entities", type=int, default=20_000)
    parser.add_argument("--periods", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20_260_728)
    parser.add_argument("--measure-memory", action="store_true")
    args = parser.parse_args()

    panel = _panel(args.n_entities, args.periods, args.seed)
    scenarios = ("conventional", "efficient") if args.scenario == "all" else (args.scenario,)
    results = [_run(scenario, panel, measure_memory=args.measure_memory) for scenario in scenarios]
    print(
        json.dumps(
            {
                "causalkit_version": causalkit.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "seed": args.seed,
                "n_entities": args.n_entities,
                "n_periods": args.periods,
                "n_rows": len(panel),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

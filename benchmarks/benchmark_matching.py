"""Reproducible scalar nearest-neighbor matching benchmarks.

Run after installing the checkout, for example:

    python benchmarks/benchmark_matching.py --scenario balanced_ate --n 100000
"""

from __future__ import annotations

import argparse
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

import causalkit
from causalkit import NearestNeighborMatch

SEED = 20_260_728


def _expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _balanced_ate(nobs: int, rng: np.random.Generator) -> dict[str, Any]:
    propensity = rng.uniform(0.05, 0.95, nobs)
    treatment = np.arange(nobs) % 2
    outcome = 1.5 * treatment + rng.normal(size=nobs)
    return {
        "y": outcome,
        "treatment": treatment,
        "propensity": propensity,
        "model": NearestNeighborMatch(estimand="ate", caliper=None, common_support=None),
    }


def _imbalanced_att(nobs: int, rng: np.random.Generator) -> dict[str, Any]:
    logits = rng.normal(loc=-2.2, scale=0.8, size=nobs)
    propensity = _expit(logits)
    treatment = rng.binomial(1, propensity)
    treatment[0], treatment[1] = 0, 1
    outcome = 1.5 * treatment + 0.4 * logits + rng.normal(size=nobs)
    return {
        "y": outcome,
        "treatment": treatment,
        "propensity": propensity,
        "model": NearestNeighborMatch(estimand="att", caliper=None, common_support=None),
    }


def _known_score_ate_inference(nobs: int, rng: np.random.Generator) -> dict[str, Any]:
    logits = rng.normal(scale=0.8, size=nobs)
    propensity = _expit(logits)
    treatment = rng.binomial(1, propensity)
    treatment[0], treatment[1] = 0, 1
    outcome = 1.5 * treatment + 0.4 * logits + rng.normal(size=nobs)
    return {
        "y": outcome,
        "treatment": treatment,
        "propensity": propensity,
        "propensity_score_status": "known",
        "model": NearestNeighborMatch(
            estimand="ate",
            caliper=None,
            common_support=None,
            inference="abadie_imbens",
        ),
    }


def _heavy_ties_att(nobs: int, rng: np.random.Generator) -> dict[str, Any]:
    del rng
    controls = max(2, (nobs + 1) // 2)
    control_logits = np.linspace(-4.0, 4.0, controls)
    treated_logits = (control_logits[:-1] + control_logits[1:]) / 2.0
    logits = np.concatenate([control_logits, treated_logits])
    treatment = np.concatenate([np.zeros(controls), np.ones(controls - 1)])
    outcome = 1.5 * treatment + 0.4 * logits
    return {
        "y": outcome,
        "treatment": treatment,
        "propensity": _expit(logits),
        "model": NearestNeighborMatch(estimand="att", caliper=None, common_support=None),
    }


def _caliper_attrition_att(nobs: int, rng: np.random.Generator) -> dict[str, Any]:
    controls = nobs // 2
    treated = nobs - controls
    control_logits = rng.normal(-1.0, 0.1, controls)
    treated_logits = np.concatenate(
        [
            rng.normal(-1.0, 0.1, treated // 2),
            rng.normal(2.0, 0.1, treated - treated // 2),
        ]
    )
    logits = np.concatenate([control_logits, treated_logits])
    treatment = np.concatenate([np.zeros(controls), np.ones(treated)])
    outcome = 1.5 * treatment + 0.4 * logits + rng.normal(size=nobs)
    return {
        "y": outcome,
        "treatment": treatment,
        "propensity": _expit(logits),
        "model": NearestNeighborMatch(estimand="att", caliper=0.05, common_support=None),
    }


SCENARIOS: dict[str, Callable[[int, np.random.Generator], dict[str, Any]]] = {
    "balanced_ate": _balanced_ate,
    "imbalanced_att": _imbalanced_att,
    "known_score_ate_inference": _known_score_ate_inference,
    "heavy_ties_att": _heavy_ties_att,
    "caliper_attrition_att": _caliper_attrition_att,
}


def _run(name: str, nobs: int, *, measure_memory: bool) -> dict[str, Any]:
    inputs = SCENARIOS[name](nobs, np.random.default_rng(SEED))
    model = inputs.pop("model")
    started = time.perf_counter()
    result = model.fit(**inputs, propensity_provenance="benchmark_generated")
    elapsed = time.perf_counter() - started
    peak = None
    if measure_memory:
        tracemalloc.start()
        model.fit(**inputs, propensity_provenance="benchmark_generated")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {
        "scenario": name,
        "requested_n": nobs,
        "realized_n": result.nobs,
        "estimate": result.estimate,
        "standard_error": result.standard_error,
        "inference": result.inference,
        "matched_focal": result.n_matched_focal,
        "caliper_unmatched": result.n_caliper_unmatched,
        "match_rows": len(result.match_table),
        "boundary_tie_events": result.boundary_tie_events,
        "maximum_reuse_count": result.maximum_reuse_count,
        "elapsed_seconds": elapsed,
        "peak_memory_mib": peak / 1024**2 if peak is not None else None,
        "backend": result.backend,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="balanced_ate")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="run a second tracemalloc-instrumented fit to record Python peak memory",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.n < 4:
        parser.error("--n must be at least four")
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "causalkit": causalkit.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "memory_measurement_second_pass": args.measure_memory,
        "results": [_run(name, args.n, measure_memory=args.measure_memory) for name in names],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

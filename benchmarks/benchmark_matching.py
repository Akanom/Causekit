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

import causekit
from causekit import NearestNeighborMatch

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


class _StatsmodelsLogitResult:
    """Benchmark-only adapter for an independently fitted statsmodels Logit."""

    def __init__(self, result: Any, design: pd.DataFrame) -> None:
        self._result = result
        self.params = pd.Series(result.params, index=design.columns, name="estimate")
        self.converged = bool(result.mle_retvals["converged"])
        self.nobs = len(design)
        self.feature_names = tuple(design.columns)

    def predict_proba(self, X: Any) -> pd.DataFrame:
        probability = np.asarray(self._result.predict(X), dtype=float)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


def _estimated_score_ate_inference(nobs: int, rng: np.random.Generator) -> dict[str, Any]:
    try:
        import statsmodels.api as sm
    except ImportError as error:  # pragma: no cover - benchmark environment contract
        raise RuntimeError(
            "estimated_score_ate_inference requires the causekit validation extra."
        ) from error

    covariate = rng.normal(size=nobs)
    design = pd.DataFrame({"const": 1.0, "x": covariate})
    propensity = _expit(-0.1 + 0.65 * covariate)
    treatment = rng.binomial(1, propensity)
    treatment[0], treatment[1] = 0, 1
    outcome = 1.5 * treatment + 0.4 * covariate + rng.normal(size=nobs)
    fitted = sm.Logit(treatment, design).fit(method="newton", maxiter=200, tol=1e-11, disp=False)
    return {
        "y": pd.Series(outcome),
        "treatment": pd.Series(treatment),
        "propensity_model": _StatsmodelsLogitResult(fitted, design),
        "propensity_design": design,
        "propensity_score_status": "estimated",
        "model": NearestNeighborMatch(
            estimand="ate",
            metric="propensity",
            caliper=None,
            common_support=None,
            inference="abadie_imbens_estimated",
            variance_neighbors=1,
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
    "estimated_score_ate_inference": _estimated_score_ate_inference,
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
        "causekit": causekit.__version__,
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

"""Publication-scale longer-design coverage and composition-diagnostic promotion.

The preregistered design crosses stationary and shifting observed composition with
observation and whole-PSU inference. It evaluates the robust event path, simultaneous
bands, diagnostic size/power, conditional placebo size, overlap, fold counts, and
refusals without estimator selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import causekit
from causekit import CrossFitter, RepeatedCrossSectionDiD, did_rcs_composition_test

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

Design = Literal["stationary", "composition_shift"]
SamplingUnit = Literal["observation", "psu"]

DESIGNS: tuple[Design, ...] = ("stationary", "composition_shift")
SAMPLING_UNITS: tuple[SamplingUnit, ...] = ("observation", "psu")
DEFAULT_REPLICATIONS = 500
DEFAULT_OBSERVATIONS = 2_400
DEFAULT_PSU_COUNT = 600
PSU_SIZE = 4
DEFAULT_SEED = 20_260_730
BOOTSTRAP_ITERATIONS = 199
SIMULTANEOUS_LEVEL = 0.95
COVERAGE_GATE = (0.90, 0.99)
SE_RATIO_GATE = (0.75, 1.25)
ABSOLUTE_BIAS_GATE = 0.08
SIZE_GATE = (0.015, 0.09)
POWER_GATE = 0.80


@dataclass(frozen=True)
class _Job:
    design: Design
    sampling_unit: SamplingUnit
    replications: int
    n_observations: int
    n_psus: int
    seed: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _joint_probabilities(design: Design) -> np.ndarray:
    if design == "stationary":
        probabilities = np.empty((2, 8), dtype=float)
        for x, treated_probability in enumerate((0.30, 0.70)):
            probabilities[x, :4] = (1.0 - treated_probability) / 4.0
            probabilities[x, 4:] = treated_probability / 4.0
        return probabilities
    if design == "composition_shift":
        return np.array(
            [
                [0.08, 0.10, 0.14, 0.18, 0.08, 0.10, 0.14, 0.18],
                [0.18, 0.14, 0.10, 0.08, 0.18, 0.14, 0.10, 0.08],
            ]
        )
    raise ValueError(f"Unknown design {design!r}.")


def _truth(design: Design) -> np.ndarray:
    if design == "stationary":
        x_means = (0.70, 0.70)
    else:
        probabilities = _joint_probabilities(design)
        x_means = tuple(
            probabilities[1, 4 + position]
            / (probabilities[0, 4 + position] + probabilities[1, 4 + position])
            for position in (2, 3)
        )
    return np.array(
        [
            1.5 + 0.4 * event_time + 4.0 * x_mean + 0.25 / 3.0
            for event_time, x_mean in enumerate(x_means)
        ]
    )


def _draw_data(job: _Job, rng: np.random.Generator) -> pd.DataFrame:
    if job.sampling_unit == "observation":
        n = job.n_observations
        psu = None
    else:
        n = job.n_psus * PSU_SIZE
        psu = np.repeat(np.arange(job.n_psus), PSU_SIZE)
    x = rng.integers(0, 2, n)
    z = rng.uniform(-1.0, 1.0, n)
    probabilities = _joint_probabilities(job.design)[x]
    cell = (rng.random(n)[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    treated = cell // 4
    time_position = cell % 4
    time_value = time_position + 1.0
    untreated = 1.0 + 0.3 * time_value + 0.6 * x + 0.2 * z + 0.15 * z**2 + treated * (0.5 + 0.2 * x)
    event_time = time_value - 3.0
    treatment_effect = np.where(
        (treated == 1) & (event_time >= 0),
        1.5 + 0.4 * event_time + 4.0 * x + 0.25 * z**2,
        0.0,
    )
    error = rng.normal(scale=0.8 + 0.2 * x, size=n)
    if psu is not None:
        error += rng.normal(scale=0.55, size=job.n_psus)[psu]
    frame = pd.DataFrame(
        {
            "outcome": untreated + treatment_effect + error,
            "time": time_value,
            "treatment_time": np.where(treated == 1, 3.0, np.inf),
            "x": x,
            "z": z,
        }
    )
    if psu is not None:
        frame["psu"] = psu
    return frame


def _cross_fitter(seed: int) -> CrossFitter:
    return CrossFitter(
        propensity_factory=SaturatedClassProbability,
        outcome_factory=QuadraticOutcome,
        n_splits=2,
        random_state=seed,
    )


def _probability_bounds(result: Any) -> tuple[float, float]:
    nuisance = result.nuisance_predictions.columns.get_level_values("nuisance").astype(str)
    values = result.nuisance_predictions.loc[:, nuisance.str.contains("propensity")].to_numpy(
        dtype=float
    )
    finite = values[np.isfinite(values)]
    return float(finite.min()), float(finite.max())


def _fit_replication(
    data: pd.DataFrame,
    *,
    job: _Job,
    replication: int,
) -> dict[str, Any]:
    clustered = job.sampling_unit == "psu"
    common = {
        "data": data,
        "outcome": "outcome",
        "time": "time",
        "treatment_time": "treatment_time",
        "cluster": "psu" if clustered else None,
        "covariates": ["x", "z"],
    }
    fold_seed = job.seed + replication
    robust = RepeatedCrossSectionDiD(
        composition="robust",
        covariance="clustered" if clustered else "robust",
        inference="multiplier_bootstrap",
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        random_state=job.seed + 100_000 + replication,
        simultaneous_level=SIMULTANEOUS_LEVEL,
    ).fit(**common, cross_fitter=_cross_fitter(fold_seed))
    stationary = RepeatedCrossSectionDiD(
        composition="stationary",
        covariance="clustered" if clustered else "robust",
    ).fit(**common, cross_fitter=_cross_fitter(fold_seed))
    diagnostic = did_rcs_composition_test(robust, stationary)
    if clustered:
        assert robust.nuisance_fold.groupby(data["psu"]).nunique().eq(1).all()
        assert stationary.nuisance_fold.groupby(data["psu"]).nunique().eq(1).all()
    bounds = _probability_bounds(robust)
    event = robust.event_study.loc[[0, 1]]
    bands = robust.simultaneous_event_study.loc[[0, 1]]
    truth = _truth(job.design)
    return {
        "estimate": robust.estimate,
        "standard_error": robust.standard_error,
        "event_estimates": event["att"].to_numpy(dtype=float).tolist(),
        "event_standard_errors": event["std_err"].to_numpy(dtype=float).tolist(),
        "pointwise_covered": (
            np.abs(event["att"].to_numpy(dtype=float) - truth)
            <= 1.96 * event["std_err"].to_numpy(dtype=float)
        ).tolist(),
        "simultaneous_covered": bool(
            np.all(truth >= bands["lower"].to_numpy(dtype=float))
            and np.all(truth <= bands["upper"].to_numpy(dtype=float))
        ),
        "diagnostic_reject": diagnostic.pvalue < 0.05,
        "pretrend_reject": (
            robust.pretrend.pvalue < 0.05 if robust.pretrend.pvalue is not None else False
        ),
        "minimum_probability": bounds[0],
        "maximum_probability": bounds[1],
        "robust_fold_fits": len(robust.nuisance_diagnostics),
        "stationary_fold_fits": len(stationary.nuisance_diagnostics),
    }


def _run_job(job: _Job) -> dict[str, Any]:
    rng = np.random.default_rng(job.seed)
    records: list[dict[str, Any]] = []
    refusals: Counter[str] = Counter()
    for replication in range(job.replications):
        try:
            records.append(
                _fit_replication(
                    _draw_data(job, rng),
                    job=job,
                    replication=replication,
                )
            )
        except (AssertionError, RuntimeError, TypeError, ValueError) as error:
            refusals[f"{type(error).__name__}: {error}"] += 1
    return {
        "design": job.design,
        "sampling_unit": job.sampling_unit,
        "records": records,
        "refusals": dict(refusals),
    }


def _summarize(result: dict[str, Any], attempted: int) -> dict[str, Any]:
    design = result["design"]
    records = result["records"]
    truth = _truth(design)
    estimates = np.array([row["event_estimates"] for row in records], dtype=float)
    standard_errors = np.array([row["event_standard_errors"] for row in records], dtype=float)
    pointwise_coverage = np.mean(
        np.array([row["pointwise_covered"] for row in records], dtype=bool), axis=0
    )
    simultaneous_coverage = float(np.mean([row["simultaneous_covered"] for row in records]))
    bias = estimates.mean(axis=0) - truth
    empirical_sd = estimates.std(axis=0, ddof=1)
    mean_se = standard_errors.mean(axis=0)
    se_ratio = mean_se / empirical_sd
    diagnostic_rejection = float(np.mean([row["diagnostic_reject"] for row in records]))
    pretrend_rejection = float(np.mean([row["pretrend_reject"] for row in records]))
    refusals = attempted - len(records)
    expected_diagnostic = (
        SIZE_GATE[0] <= diagnostic_rejection <= SIZE_GATE[1]
        if design == "stationary"
        else diagnostic_rejection >= POWER_GATE
    )
    status = (
        "pass"
        if refusals == 0
        and np.max(np.abs(bias)) <= ABSOLUTE_BIAS_GATE
        and np.all(
            (pointwise_coverage >= COVERAGE_GATE[0]) & (pointwise_coverage <= COVERAGE_GATE[1])
        )
        and COVERAGE_GATE[0] <= simultaneous_coverage <= COVERAGE_GATE[1]
        and np.all((se_ratio >= SE_RATIO_GATE[0]) & (se_ratio <= SE_RATIO_GATE[1]))
        and SIZE_GATE[0] <= pretrend_rejection <= SIZE_GATE[1]
        and expected_diagnostic
        and min(row["minimum_probability"] for row in records) > 1e-6
        and max(row["maximum_probability"] for row in records) < 1.0 - 1e-6
        and {row["robust_fold_fits"] for row in records} == {24}
        and {row["stationary_fold_fits"] for row in records} == {30}
        else "fail"
    )
    return {
        "design": design,
        "sampling_unit": result["sampling_unit"],
        "replications": len(records),
        "refusals": refusals,
        "truth": truth.tolist(),
        "mean_estimate": estimates.mean(axis=0).tolist(),
        "bias": bias.tolist(),
        "absolute_bias_gate": ABSOLUTE_BIAS_GATE,
        "empirical_standard_deviation": empirical_sd.tolist(),
        "mean_standard_error": mean_se.tolist(),
        "standard_error_ratio": se_ratio.tolist(),
        "standard_error_ratio_gate": list(SE_RATIO_GATE),
        "pointwise_coverage": pointwise_coverage.tolist(),
        "simultaneous_coverage": simultaneous_coverage,
        "coverage_gate": list(COVERAGE_GATE),
        "simultaneous_coverage_mcse": math.sqrt(
            simultaneous_coverage * (1.0 - simultaneous_coverage) / len(records)
        ),
        "diagnostic_rejection_rate": diagnostic_rejection,
        "diagnostic_gate": list(SIZE_GATE) if design == "stationary" else [POWER_GATE, 1.0],
        "conditional_pretrend_rejection_rate": pretrend_rejection,
        "conditional_pretrend_size_gate": list(SIZE_GATE),
        "minimum_probability": min(row["minimum_probability"] for row in records),
        "maximum_probability": max(row["maximum_probability"] for row in records),
        "robust_nuisance_fold_fits": 24,
        "stationary_nuisance_fold_fits": 30,
        "status": status,
    }


def run_simulation(
    *,
    replications: int,
    n_observations: int,
    n_psus: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if replications < 20:
        raise ValueError("replications must be at least 20.")
    jobs = [
        _Job(
            design=design,
            sampling_unit=sampling_unit,
            replications=replications,
            n_observations=n_observations,
            n_psus=n_psus,
            seed=seed + 100_000 * design_position + 10_000 * unit_position,
        )
        for design_position, design in enumerate(DESIGNS)
        for unit_position, sampling_unit in enumerate(SAMPLING_UNITS)
    ]
    if workers == 1:
        raw = [_run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            raw = list(executor.map(_run_job, jobs))
    cells = [_summarize(result, replications) for result in raw]
    refusals: Counter[str] = Counter()
    for result in raw:
        refusals.update(result["refusals"])
    return {
        "replications_per_cell": replications,
        "total_estimator_fits": replications * len(jobs) * 2,
        "total_nuisance_fold_fits": replications * len(jobs) * (24 + 30),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "simultaneous_level": SIMULTANEOUS_LEVEL,
        "cells": cells,
        "refusal_reasons": dict(refusals),
        "estimator_selection": False,
        "status": "pass" if all(cell["status"] == "pass" for cell in cells) else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--n-observations", type=int, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--n-psus", type=int, default=DEFAULT_PSU_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/did_rcs_composition_longer_promotion_evidence.json"),
    )
    args = parser.parse_args()
    simulation = run_simulation(
        replications=args.replications,
        n_observations=args.n_observations,
        n_psus=args.n_psus,
        seed=args.seed,
        workers=args.workers,
    )
    evidence = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/validate_did_rcs_composition_longer_promotion.py",
        "generator_sha256": _sha256(Path(__file__)),
        "causekit_version": causekit.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        "coverage_simulation": simulation,
        "overall_status": simulation["status"],
    }
    rendered = json.dumps(evidence, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if evidence["overall_status"] != "pass":
        raise SystemExit("longer composition promotion gate failed")


if __name__ == "__main__":
    main()

"""Publication-scale coverage for composition-robust repeated-section DiD.

The preregistered design crosses favorable stationary composition and a material
composition shift with observation-level and whole-PSU inference. Both estimators are
reported in every design; no data-dependent estimator selection is performed.
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
from causekit import CrossFitter, RepeatedCrossSectionDiD

try:
    from benchmarks._did_rcs_composition_support import (
        QuadraticOutcome as _QuadraticOutcome,
    )
    from benchmarks._did_rcs_composition_support import (
        SaturatedClassProbability as _SaturatedClassProbability,
    )
except ModuleNotFoundError:  # Direct ``python benchmarks/...py`` execution.
    from _did_rcs_composition_support import (  # type: ignore[no-redef]
        QuadraticOutcome as _QuadraticOutcome,
    )
    from _did_rcs_composition_support import (
        SaturatedClassProbability as _SaturatedClassProbability,
    )

Design = Literal["stationary_unequal", "composition_shift_unequal"]
SamplingUnit = Literal["observation", "psu"]
Composition = Literal["robust", "stationary"]

DESIGNS: tuple[Design, ...] = ("stationary_unequal", "composition_shift_unequal")
SAMPLING_UNITS: tuple[SamplingUnit, ...] = ("observation", "psu")
COMPOSITIONS: tuple[Composition, ...] = ("robust", "stationary")
DEFAULT_REPLICATIONS = 1_000
DEFAULT_OBSERVATIONS = 1_600
DEFAULT_PSU_COUNT = 400
PSU_SIZE = 4
DEFAULT_SEED = 20_260_730
COVERAGE_GATE = (0.90, 0.99)
STANDARD_ERROR_RATIO_GATE = (0.75, 1.25)
ABSOLUTE_BIAS_GATE = 0.08
SHIFT_BIAS_TOLERANCE = 0.08
ROBUST_TRUTH_STATIONARY = 29.0 / 6.0
ROBUST_TRUTH_SHIFT = 11.0 / 2.0
STATIONARY_TRUTH_SHIFT = 143.0 / 30.0
EXPECTED_SHIFT_BIAS = -11.0 / 15.0


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


def _cell_probabilities(x: np.ndarray, design: Design) -> np.ndarray:
    probabilities = np.empty((len(x), 4), dtype=float)
    if design == "stationary_unequal":
        treated_probability = np.where(x == 1, 0.60, 0.30)
        post_probability = 0.35
        probabilities[:, 0] = (1.0 - treated_probability) * (1.0 - post_probability)
        probabilities[:, 1] = (1.0 - treated_probability) * post_probability
        probabilities[:, 2] = treated_probability * (1.0 - post_probability)
        probabilities[:, 3] = treated_probability * post_probability
        return probabilities
    if design != "composition_shift_unequal":
        raise ValueError(f"Unknown coverage design: {design!r}.")
    probabilities[x == 0] = (0.55, 0.10, 0.25, 0.10)
    probabilities[x == 1] = (0.10, 0.25, 0.15, 0.50)
    return probabilities


def _draw_data(job: _Job, rng: np.random.Generator) -> pd.DataFrame:
    if job.sampling_unit == "observation":
        n = job.n_observations
        cluster = None
    else:
        n = job.n_psus * PSU_SIZE
        cluster = np.repeat(np.arange(job.n_psus), PSU_SIZE)
    x = rng.integers(0, 2, n)
    z = rng.uniform(-1.0, 1.0, n)
    probabilities = _cell_probabilities(x, job.design)
    draws = rng.random(n)
    cell = (draws[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    treated = cell // 2
    post = cell % 2

    m00 = 1.0 + 0.6 * x + 0.3 * z + 0.2 * z**2
    trend = 0.5 + 0.4 * x - 0.2 * z + 0.15 * z**2
    m01 = m00 + trend
    m10 = m00 + 0.75
    tau = 2.0 + 4.0 * x + 0.5 * z**2
    m11 = m10 + trend + tau
    mean = np.choose(cell, [m00, m01, m10, m11])
    idiosyncratic = rng.normal(scale=0.8 + 0.2 * x + 0.1 * post)
    if cluster is None:
        outcome = mean + idiosyncratic
    else:
        cluster_shock = rng.normal(scale=0.6, size=job.n_psus)
        outcome = mean + idiosyncratic + cluster_shock[cluster]
    frame = pd.DataFrame(
        {
            "outcome": outcome,
            "time": post + 1.0,
            "treatment_time": np.where(treated == 1, 2.0, np.inf),
            "x": x,
            "z": z,
        }
    )
    if cluster is not None:
        frame["psu"] = cluster
    return frame


def _cross_fitter(seed: int) -> CrossFitter:
    return CrossFitter(
        propensity_factory=_SaturatedClassProbability,
        outcome_factory=_QuadraticOutcome,
        n_splits=2,
        random_state=seed,
    )


def _probability_bounds(result: Any) -> tuple[float, float]:
    nuisance = result.nuisance_predictions.columns.get_level_values("nuisance").astype(str)
    selected = result.nuisance_predictions.loc[:, nuisance.str.contains("propensity")]
    values = selected.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.min()), float(finite.max())


def _fit_one(
    data: pd.DataFrame,
    *,
    composition: Composition,
    sampling_unit: SamplingUnit,
    fold_seed: int,
) -> dict[str, float | int]:
    clustered = sampling_unit == "psu"
    result = RepeatedCrossSectionDiD(
        composition=composition,
        covariance="clustered" if clustered else "robust",
    ).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        cluster="psu" if clustered else None,
        covariates=["x", "z"],
        cross_fitter=_cross_fitter(fold_seed),
    )
    if clustered and not result.nuisance_fold.groupby(data["psu"]).nunique().eq(1).all():
        raise RuntimeError("A PSU crossed nuisance-fold roles.")
    minimum, maximum = _probability_bounds(result)
    return {
        "estimate": result.estimate,
        "standard_error": result.standard_error,
        "minimum_probability": minimum,
        "maximum_probability": maximum,
        "nuisance_fold_fits": len(result.nuisance_diagnostics),
    }


def _run_job(job: _Job) -> dict[str, Any]:
    rng = np.random.default_rng(job.seed)
    records: dict[Composition, list[dict[str, float | int]]] = {
        "robust": [],
        "stationary": [],
    }
    refusal_reasons: Counter[str] = Counter()
    for replication in range(job.replications):
        data = _draw_data(job, rng)
        fold_seed = job.seed + replication
        for composition in COMPOSITIONS:
            try:
                record = _fit_one(
                    data,
                    composition=composition,
                    sampling_unit=job.sampling_unit,
                    fold_seed=fold_seed,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                refusal_reasons[f"{type(exc).__name__}: {exc}"] += 1
            else:
                records[composition].append(record)
    return {
        "design": job.design,
        "sampling_unit": job.sampling_unit,
        "records": records,
        "refusal_reasons": dict(refusal_reasons),
    }


def _truth(design: Design, composition: Composition) -> float:
    if design == "stationary_unequal":
        return ROBUST_TRUTH_STATIONARY
    return ROBUST_TRUTH_SHIFT if composition == "robust" else STATIONARY_TRUTH_SHIFT


def _summarize_cell(
    *,
    design: Design,
    sampling_unit: SamplingUnit,
    composition: Composition,
    records: list[dict[str, float | int]],
    attempted: int,
) -> dict[str, Any]:
    truth = _truth(design, composition)
    estimates = np.array([row["estimate"] for row in records], dtype=float)
    standard_errors = np.array([row["standard_error"] for row in records], dtype=float)
    refusals = attempted - len(records)
    if len(records) < 2:
        return {
            "design": design,
            "sampling_unit": sampling_unit,
            "composition": composition,
            "replications": len(records),
            "refusals": refusals,
            "status": "fail",
        }
    bias = float(estimates.mean() - truth)
    empirical_sd = float(estimates.std(ddof=1))
    mean_se = float(standard_errors.mean())
    se_ratio = mean_se / empirical_sd
    covered = np.abs(estimates - truth) <= 1.96 * standard_errors
    coverage = float(covered.mean())
    coverage_mcse = math.sqrt(coverage * (1.0 - coverage) / len(records))
    minimum_probability = min(float(row["minimum_probability"]) for row in records)
    maximum_probability = max(float(row["maximum_probability"]) for row in records)
    fold_fits = {int(row["nuisance_fold_fits"]) for row in records}
    expected_fold_fits = 8 if composition == "robust" else 10
    status = (
        "pass"
        if refusals == 0
        and abs(bias) <= ABSOLUTE_BIAS_GATE
        and COVERAGE_GATE[0] <= coverage <= COVERAGE_GATE[1]
        and STANDARD_ERROR_RATIO_GATE[0] <= se_ratio <= STANDARD_ERROR_RATIO_GATE[1]
        and minimum_probability > 1e-6
        and maximum_probability < 1.0 - 1e-6
        and fold_fits == {expected_fold_fits}
        else "fail"
    )
    return {
        "design": design,
        "sampling_unit": sampling_unit,
        "composition": composition,
        "target": "treated_target_period" if composition == "robust" else "pooled_treated",
        "truth": truth,
        "replications": len(records),
        "refusals": refusals,
        "mean_estimate": float(estimates.mean()),
        "bias": bias,
        "absolute_bias_gate": ABSOLUTE_BIAS_GATE,
        "empirical_standard_deviation": empirical_sd,
        "mean_standard_error": mean_se,
        "standard_error_ratio": se_ratio,
        "standard_error_ratio_gate": list(STANDARD_ERROR_RATIO_GATE),
        "coverage": coverage,
        "coverage_gate": list(COVERAGE_GATE),
        "coverage_mcse": coverage_mcse,
        "minimum_probability": minimum_probability,
        "maximum_probability": maximum_probability,
        "nuisance_fold_fits_per_replication": expected_fold_fits,
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
    if replications < 2:
        raise ValueError("replications must be at least two.")
    jobs = [
        _Job(
            design=design,
            sampling_unit=sampling_unit,
            replications=replications,
            n_observations=n_observations,
            n_psus=n_psus,
            seed=seed + 100_000 * design_position + 10_000 * sampling_position,
        )
        for design_position, design in enumerate(DESIGNS)
        for sampling_position, sampling_unit in enumerate(SAMPLING_UNITS)
    ]
    if workers == 1:
        job_results = [_run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            job_results = list(executor.map(_run_job, jobs))

    refusal_reasons: Counter[str] = Counter()
    cells: list[dict[str, Any]] = []
    records_by_key: dict[tuple[Design, SamplingUnit, Composition], list[dict[str, Any]]] = {}
    for result in job_results:
        design = result["design"]
        sampling_unit = result["sampling_unit"]
        refusal_reasons.update(result["refusal_reasons"])
        for composition in COMPOSITIONS:
            records = result["records"][composition]
            records_by_key[(design, sampling_unit, composition)] = records
            cells.append(
                _summarize_cell(
                    design=design,
                    sampling_unit=sampling_unit,
                    composition=composition,
                    records=records,
                    attempted=replications,
                )
            )

    cell_by_key = {
        (cell["design"], cell["sampling_unit"], cell["composition"]): cell for cell in cells
    }
    diagnostics: dict[str, Any] = {}
    for sampling_unit in SAMPLING_UNITS:
        favorable_robust = cell_by_key[("stationary_unequal", sampling_unit, "robust")]
        favorable_stationary = cell_by_key[("stationary_unequal", sampling_unit, "stationary")]
        efficiency_ratio = (
            favorable_robust["mean_standard_error"] / favorable_stationary["mean_standard_error"]
        )
        diagnostics[f"stationary_efficiency_{sampling_unit}"] = {
            "robust_to_stationary_mean_se_ratio": efficiency_ratio,
            "interpretation": "values_above_one_favor_stationary_score_under_stationarity",
            "status": "pass" if efficiency_ratio > 1.0 else "fail",
        }
        shifted_robust = cell_by_key[("composition_shift_unequal", sampling_unit, "robust")]
        shifted_stationary = cell_by_key[("composition_shift_unequal", sampling_unit, "stationary")]
        observed_stationary_bias = shifted_stationary["mean_estimate"] - ROBUST_TRUTH_SHIFT
        diagnostics[f"composition_shift_bias_{sampling_unit}"] = {
            "robust_target_truth": ROBUST_TRUTH_SHIFT,
            "stationary_pooled_target_truth": STATIONARY_TRUTH_SHIFT,
            "expected_stationary_bias_for_robust_target": EXPECTED_SHIFT_BIAS,
            "observed_stationary_bias_for_robust_target": observed_stationary_bias,
            "robust_estimator_bias_for_robust_target": shifted_robust["bias"],
            "absolute_tolerance": SHIFT_BIAS_TOLERANCE,
            "status": (
                "pass"
                if abs(observed_stationary_bias - EXPECTED_SHIFT_BIAS) <= SHIFT_BIAS_TOLERANCE
                else "fail"
            ),
        }

    total_estimator_fits = len(jobs) * replications * len(COMPOSITIONS)
    total_nuisance_fold_fits = sum(
        int(row["nuisance_fold_fits"]) for records in records_by_key.values() for row in records
    )
    status = (
        "pass"
        if not refusal_reasons
        and all(cell["status"] == "pass" for cell in cells)
        and all(item["status"] == "pass" for item in diagnostics.values())
        else "fail"
    )
    return {
        "designs": list(DESIGNS),
        "sampling_units": list(SAMPLING_UNITS),
        "replications_per_cell": replications,
        "observation_sample_size": n_observations,
        "psu_count": n_psus,
        "psu_size": PSU_SIZE,
        "seed": seed,
        "total_estimator_fits": total_estimator_fits,
        "total_nuisance_fold_fits": total_nuisance_fold_fits,
        "refusal_reasons": dict(refusal_reasons),
        "cells": cells,
        "target_diagnostics": diagnostics,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--n-observations", type=int, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--n-psus", type=int, default=DEFAULT_PSU_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/did_rcs_composition_promotion_evidence.json"),
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
        "generator": "benchmarks/validate_did_rcs_composition_promotion.py",
        "generator_sha256": _sha256(Path(__file__)),
        "causekit_version": causekit.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "coverage_simulation": simulation,
        "overall_status": simulation["status"],
    }
    rendered = json.dumps(evidence, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if evidence["overall_status"] != "pass":
        raise SystemExit("composition coverage gate failed")


if __name__ == "__main__":
    main()

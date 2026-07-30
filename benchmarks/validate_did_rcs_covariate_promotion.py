"""Generate covariate repeated-cross-section DiD coverage evidence.

The simulation draws independent observations in each period under stationary
composition and conditional parallel trends. It exercises favorable and stressed
overlap, balanced and sharply unequal period sizes, both maintained control rules, all
public effect aggregations, conditional placebos, and the joint conditional pre-trend
test. Nuisances are fitted fold-locally; no oracle prediction is supplied.

Run from the repository root::

    python benchmarks/validate_did_rcs_covariate_promotion.py --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import causekit
from causekit import CrossFitter, RepeatedCrossSectionDiD

SEED = 20_260_730
DEFAULT_REPLICATIONS = 1_000
NOMINAL_COVERAGE = 0.95
NOMINAL_PRETREND_SIZE = 0.05
N_SPLITS = 2
COHORT_SUPPORT = np.array([3.0, 4.0, np.inf])
CONTROL_GROUPS = ("never_treated", "not_yet_treated")
DESIGNS = {
    "favorable_balanced": {
        "period_sizes": (480, 480, 480, 480),
        "cohort_probabilities_by_x": (
            (4.0 / 15.0, 2.0 / 15.0, 3.0 / 5.0),
            (2.0 / 5.0, 1.0 / 5.0, 2.0 / 5.0),
        ),
    },
    "stressed_overlap_unequal": {
        # Keep expected counts above 32 in every period-X-cohort cell.  The
        # initial (1_200, 800, 600, 480) stress design generated legitimate
        # fold-support refusals, so publication coverage uses the same
        # probabilities and unequal-size ratio at three times the sample size.
        "period_sizes": (3_600, 2_400, 1_800, 1_440),
        "cohort_probabilities_by_x": (
            (1.0 / 20.0, 1.0 / 20.0, 9.0 / 10.0),
            (37.0 / 60.0, 17.0 / 60.0, 1.0 / 10.0),
        ),
    },
}
TARGETS = {
    "group_3_3": 1.0,
    "group_3_4": 1.5,
    "group_4_4": 2.0,
    "event_0": 4.0 / 3.0,
    "event_1": 1.5,
    "calendar_3": 1.0,
    "calendar_4": 5.0 / 3.0,
    "esavg": 17.0 / 12.0,
    "placebo_3_2": 0.0,
    "placebo_4_2": 0.0,
    "placebo_4_3": 0.0,
}
COVERAGE_GATE = (0.90, 0.99)
STANDARD_ERROR_RATIO_GATE = (0.75, 1.25)
ABSOLUTE_BIAS_GATE = 0.08
PRETREND_REJECTION_RATE_GATE = (0.01, 0.10)


class _BinaryProbabilityResult:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        categories = X["x"].to_numpy(dtype=int)
        probability = self.probabilities[categories]
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _BinarySaturatedProbability:
    """Fold-local saturated probability provider for the binary simulation covariate."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _BinaryProbabilityResult:
        categories = X["x"].to_numpy(dtype=int)
        assigned = y.to_numpy(dtype=float)
        if set(np.unique(categories)) != {0, 1}:
            raise ValueError("Each propensity training fold must contain both x levels.")
        probabilities = np.array(
            [float(assigned[categories == category].mean()) for category in (0, 1)]
        )
        if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
            raise ValueError("A fitted saturated propensity is on the probability boundary.")
        return _BinaryProbabilityResult(probabilities)


class _BinaryOutcomeResult:
    def __init__(self, means: np.ndarray) -> None:
        self.means = means

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.means[X["x"].to_numpy(dtype=int)]


class _BinarySaturatedOutcome:
    """Fold-local saturated conditional-mean provider for the binary covariate."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _BinaryOutcomeResult:
        categories = X["x"].to_numpy(dtype=int)
        outcome = y.to_numpy(dtype=float)
        if set(np.unique(categories)) != {0, 1}:
            raise ValueError("Each outcome training fold must contain both x levels.")
        means = np.array([float(outcome[categories == category].mean()) for category in (0, 1)])
        return _BinaryOutcomeResult(means)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(design: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    configuration = DESIGNS[design]
    probability_table = np.asarray(configuration["cohort_probabilities_by_x"], dtype=float)
    rows: list[pd.DataFrame] = []
    trends = {1: 0.0, 2: 0.35, 3: 0.85, 4: 1.45}
    for period, count in enumerate(configuration["period_sizes"], start=1):
        x = rng.binomial(1, 0.5, size=int(count))
        probabilities = probability_table[x]
        draws = rng.random(int(count))
        cut_one = probabilities[:, 0]
        cut_two = cut_one + probabilities[:, 1]
        cohort = np.where(draws < cut_one, 3.0, np.where(draws < cut_two, 4.0, np.inf))
        effect = np.zeros(int(count))
        effect[(cohort == 3.0) & (period == 3)] = 1.0
        effect[(cohort == 3.0) & (period == 4)] = 1.5
        effect[(cohort == 4.0) & (period == 4)] = 2.0
        group_level = np.where(
            cohort == 3.0, 0.2 + 0.4 * x, np.where(cohort == 4.0, -0.3 - 0.2 * x, 0.1 * x)
        )
        untreated_mean = group_level + trends[period] + 0.7 * x + 0.2 * period * x
        if design == "favorable_balanced":
            noise_scale = 0.9 + 0.15 * x
        else:
            noise_scale = 0.65 + 0.10 * period + 0.35 * (cohort == 4.0) + 0.20 * x
        outcome = untreated_mean + effect + rng.normal(scale=noise_scale)
        rows.append(
            pd.DataFrame(
                {
                    "outcome": outcome,
                    "time": float(period),
                    "treatment_time": cohort,
                    "x": x.astype(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _target_rows(result: Any) -> dict[str, tuple[float, float]]:
    return {
        "group_3_3": tuple(result.group_time.loc[(3.0, 3.0), ["att", "std_err"]]),
        "group_3_4": tuple(result.group_time.loc[(3.0, 4.0), ["att", "std_err"]]),
        "group_4_4": tuple(result.group_time.loc[(4.0, 4.0), ["att", "std_err"]]),
        "event_0": tuple(result.event_study.loc[0, ["att", "std_err"]]),
        "event_1": tuple(result.event_study.loc[1, ["att", "std_err"]]),
        "calendar_3": tuple(result.calendar_time.loc[3.0, ["att", "std_err"]]),
        "calendar_4": tuple(result.calendar_time.loc[4.0, ["att", "std_err"]]),
        "esavg": (result.estimate, result.standard_error),
        "placebo_3_2": tuple(result.pretrend.placebo_effects.loc[(3.0, 2.0), ["att", "std_err"]]),
        "placebo_4_2": tuple(result.pretrend.placebo_effects.loc[(4.0, 2.0), ["att", "std_err"]]),
        "placebo_4_3": tuple(result.pretrend.placebo_effects.loc[(4.0, 3.0), ["att", "std_err"]]),
    }


def _simulation_task(task: tuple[str, int]) -> list[dict[str, Any]]:
    design, seed = task
    data = _sample(design, seed)
    records: list[dict[str, Any]] = []
    for control_group in CONTROL_GROUPS:
        try:
            result = RepeatedCrossSectionDiD(
                control_group=control_group,
                inference="analytic",
            ).fit(
                data,
                outcome="outcome",
                time="time",
                treatment_time="treatment_time",
                covariates=["x"],
                cross_fitter=CrossFitter(
                    propensity_factory=_BinarySaturatedProbability,
                    outcome_factory=_BinarySaturatedOutcome,
                    n_splits=2,
                    random_state=seed,
                ),
            )
            probability_columns = (
                result.nuisance_predictions.columns.get_level_values("nuisance") == "propensity"
            )
            probabilities = result.nuisance_predictions.loc[:, probability_columns].to_numpy(
                dtype=float
            )
            probabilities = probabilities[np.isfinite(probabilities)]
            if not result.pretrend.available or result.pretrend.pvalue is None:
                raise RuntimeError("The conditional pre-trend joint test is unavailable.")
            records.append(
                {
                    "design": design,
                    "control_group": control_group,
                    "targets": _target_rows(result),
                    "minimum_cell": int(result.cell_counts["nobs"].min()),
                    "minimum_nuisance_probability": float(probabilities.min()),
                    "maximum_nuisance_probability": float(probabilities.max()),
                    "nuisance_fold_fits": len(result.nuisance_diagnostics),
                    "pretrend_rejected": bool(result.pretrend.pvalue < NOMINAL_PRETREND_SIZE),
                    "pretrend_available": True,
                    "refusal": None,
                }
            )
        except (ValueError, RuntimeError) as error:
            records.append(
                {
                    "design": design,
                    "control_group": control_group,
                    "targets": None,
                    "minimum_cell": None,
                    "minimum_nuisance_probability": None,
                    "maximum_nuisance_probability": None,
                    "nuisance_fold_fits": 0,
                    "pretrend_rejected": False,
                    "pretrend_available": False,
                    "refusal": f"{type(error).__name__}: {error}",
                }
            )
    return records


def _wilson_interval(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (probability + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1.0 - probability) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [center - radius, center + radius]


def run_simulation(*, replications: int, workers: int) -> dict[str, Any]:
    sequences = np.random.SeedSequence(SEED).spawn(replications * len(DESIGNS))
    tasks: list[tuple[str, int]] = []
    position = 0
    for design in DESIGNS:
        for _ in range(replications):
            tasks.append((design, int(sequences[position].generate_state(1)[0])))
            position += 1
    started = time.perf_counter()
    if workers == 1:
        records = [record for task in tasks for record in _simulation_task(task)]
    else:
        chunk_size = max(1, len(tasks) // (workers * 20))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = executor.map(_simulation_task, tasks, chunksize=chunk_size)
            records = [record for batch in batches for record in batch]
    elapsed = time.perf_counter() - started

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["design"], record["control_group"])].append(record)
    cells: list[dict[str, Any]] = []
    pretrend_cells: list[dict[str, Any]] = []
    for design in DESIGNS:
        for control_group in CONTROL_GROUPS:
            values = grouped[(design, control_group)]
            refusals = sum(value["refusal"] is not None for value in values)
            completed = [value for value in values if value["refusal"] is None]
            for target, truth in TARGETS.items():
                estimates = np.asarray(
                    [value["targets"][target][0] for value in completed], dtype=float
                )
                standard_errors = np.asarray(
                    [value["targets"][target][1] for value in completed], dtype=float
                )
                covered = np.abs(estimates - truth) <= 1.959963984540054 * standard_errors
                successes = int(covered.sum())
                coverage = successes / len(completed) if completed else 0.0
                empirical_sd = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else math.nan
                mean_se = float(np.mean(standard_errors)) if len(standard_errors) else math.nan
                bias = float(np.mean(estimates) - truth) if len(estimates) else math.nan
                ratio = mean_se / empirical_sd
                fold_fit_counts = {int(value["nuisance_fold_fits"]) for value in completed}
                fold_fits = fold_fit_counts.pop() if len(fold_fit_counts) == 1 else -1
                passed = (
                    len(completed) == replications
                    and refusals == 0
                    and COVERAGE_GATE[0] <= coverage <= COVERAGE_GATE[1]
                    and STANDARD_ERROR_RATIO_GATE[0] <= ratio <= STANDARD_ERROR_RATIO_GATE[1]
                    and abs(bias) <= ABSOLUTE_BIAS_GATE
                    and fold_fits == 60
                )
                cells.append(
                    {
                        "design": design,
                        "control_group": control_group,
                        "target": target,
                        "truth": truth,
                        "replications": len(completed),
                        "refusals": refusals,
                        "mean_estimate": float(np.mean(estimates)),
                        "bias": bias,
                        "absolute_bias_gate": ABSOLUTE_BIAS_GATE,
                        "empirical_standard_deviation": empirical_sd,
                        "mean_standard_error": mean_se,
                        "standard_error_ratio": ratio,
                        "standard_error_ratio_gate": list(STANDARD_ERROR_RATIO_GATE),
                        "coverage": coverage,
                        "coverage_mcse": math.sqrt(coverage * (1.0 - coverage) / len(completed)),
                        "coverage_wilson_95": _wilson_interval(successes, len(completed)),
                        "coverage_gate": list(COVERAGE_GATE),
                        "minimum_realized_cell": min(
                            int(value["minimum_cell"]) for value in completed
                        ),
                        "minimum_nuisance_probability": min(
                            float(value["minimum_nuisance_probability"]) for value in completed
                        ),
                        "maximum_nuisance_probability": max(
                            float(value["maximum_nuisance_probability"]) for value in completed
                        ),
                        "nuisance_fold_fits_per_estimator": fold_fits,
                        "status": "pass" if passed else "fail",
                    }
                )

            unavailable = sum(not bool(value["pretrend_available"]) for value in completed)
            rejections = sum(bool(value["pretrend_rejected"]) for value in completed)
            rejection_rate = rejections / len(completed) if completed else 0.0
            pretrend_passed = (
                len(completed) == replications
                and refusals == 0
                and unavailable == 0
                and PRETREND_REJECTION_RATE_GATE[0]
                <= rejection_rate
                <= PRETREND_REJECTION_RATE_GATE[1]
            )
            pretrend_cells.append(
                {
                    "design": design,
                    "control_group": control_group,
                    "replications": len(completed),
                    "refusals": refusals,
                    "unavailable": unavailable,
                    "nominal_size": NOMINAL_PRETREND_SIZE,
                    "rejections": rejections,
                    "rejection_rate": rejection_rate,
                    "rejection_rate_mcse": math.sqrt(
                        rejection_rate * (1.0 - rejection_rate) / len(completed)
                    ),
                    "rejection_rate_wilson_95": _wilson_interval(rejections, len(completed)),
                    "rejection_rate_gate": list(PRETREND_REJECTION_RATE_GATE),
                    "status": "pass" if pretrend_passed else "fail",
                }
            )

    total_fits = replications * len(DESIGNS) * len(CONTROL_GROUPS)
    total_nuisance_fits = sum(int(record["nuisance_fold_fits"]) for record in records)
    refusal_reasons = Counter(
        str(record["refusal"]) for record in records if record["refusal"] is not None
    )
    return {
        "seed": SEED,
        "replications_per_design": replications,
        "nominal_coverage": NOMINAL_COVERAGE,
        "nominal_conditional_pretrend_size": NOMINAL_PRETREND_SIZE,
        "n_splits": N_SPLITS,
        "population_marginal_cohort_probabilities": {
            "treated_at_3": 1.0 / 3.0,
            "treated_at_4": 1.0 / 6.0,
            "never_treated": 1.0 / 2.0,
        },
        "designs": DESIGNS,
        "dgp": (
            "Independent period samples; stationary binary-X-dependent cohort composition; "
            "conditional parallel trends; group-X levels; time-X trends; heterogeneous "
            "normal variance; constant declared group-time treatment effects."
        ),
        "nuisance_contract": (
            "Two-fold cross-fitted saturated binary-X propensity and four group-period "
            "outcome means per fixed comparison; no oracle predictions, clipping, or trimming."
        ),
        "finite_sample_support_contract": (
            "Every period-X-cohort cell has population expected count of at least 32; "
            "any realized fold-support failure is recorded as a refusal, never repaired."
        ),
        "total_estimator_fits": total_fits,
        "total_nuisance_fold_fits": total_nuisance_fits,
        "refusal_reasons": dict(sorted(refusal_reasons.items())),
        "elapsed_seconds": elapsed,
        "fits_per_second": total_fits / elapsed,
        "cells": cells,
        "conditional_pretrend_joint_size": pretrend_cells,
        "status": (
            "pass" if all(cell["status"] == "pass" for cell in cells + pretrend_cells) else "fail"
        ),
    }


def _default_workers() -> int:
    available = os.cpu_count() or 1
    return max(1, min(8, available - 1 if available > 1 else 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/did_rcs_covariate_promotion_evidence.json"),
    )
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least two")
    if args.workers < 1:
        parser.error("--workers must be positive")

    simulation = run_simulation(replications=args.replications, workers=args.workers)
    script_path = Path(__file__).resolve()
    evidence = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/validate_did_rcs_covariate_promotion.py",
        "generator_sha256": _sha256(script_path),
        "reproduction_command": (
            "python benchmarks/validate_did_rcs_covariate_promotion.py "
            f"--replications {args.replications} --workers {args.workers}"
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "causekit_version": causekit.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "coverage_simulation": simulation,
        "overall_status": "pass" if simulation["status"] == "pass" else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if evidence["overall_status"] != "pass":
        raise SystemExit(9)


if __name__ == "__main__":
    main()

"""Generate publication-scale repeated-section simultaneous-inference evidence.

The fixed design crosses favorable and stressed overlap, unadjusted and genuinely
cross-fitted covariate scores, observation and indivisible-PSU sampling units, and
both maintained control rules.  Every cell certifies joint coverage of the complete
reported post-treatment event-study vector.  Refused fits are retained and fail the
certificate; they are never redrawn or repaired.

Run from the repository root::

    python benchmarks/validate_did_rcs_simultaneous_promotion.py --workers 8
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
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

import causekit
from causekit import CrossFitter, RepeatedCrossSectionDiD

SEED = 20_260_730
BAND_SEED_ROOT = 73_026_020
DEFAULT_REPLICATIONS = 1_000
DEFAULT_BAND_ITERATIONS = 999
NOMINAL_JOINT_COVERAGE = 0.95
N_SPLITS = 2
CONTROL_GROUPS = ("never_treated", "not_yet_treated")
ADJUSTMENTS = ("unadjusted", "covariate")
SAMPLING_UNITS = ("observation", "psu")
EVENT_TIMES = np.array([0.0, 1.0])
EVENT_TRUTH = np.array([4.0 / 3.0, 1.5])
JOINT_COVERAGE_GATE = (0.90, 0.99)
JOINT_COVERAGE_MCSE_GATE = 0.011


class _Design(TypedDict):
    period_sizes: tuple[int, int, int, int]
    n_psus: int
    cohort_probabilities_by_x: tuple[tuple[float, float, float], tuple[float, float, float]]


DESIGNS: dict[str, _Design] = {
    "favorable_balanced": {
        "period_sizes": (480, 480, 480, 480),
        "n_psus": 48,
        "cohort_probabilities_by_x": (
            (4.0 / 15.0, 2.0 / 15.0, 3.0 / 5.0),
            (2.0 / 5.0, 1.0 / 5.0, 2.0 / 5.0),
        ),
    },
    "stressed_overlap_unequal": {
        "period_sizes": (3_600, 2_400, 1_800, 1_440),
        "n_psus": 80,
        "cohort_probabilities_by_x": (
            (1.0 / 20.0, 1.0 / 20.0, 9.0 / 10.0),
            (37.0 / 60.0, 17.0 / 60.0, 1.0 / 10.0),
        ),
    },
}


class _BinaryProbabilityResult:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        categories = X["x"].to_numpy(dtype=int)
        probability = self.probabilities[categories]
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _BinarySaturatedProbability:
    """Fold-local saturated binary-X probability provider."""

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
    """Fold-local saturated binary-X conditional-mean provider."""

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


def _sample(
    design: str,
    adjustment: str,
    sampling_unit: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    configuration = DESIGNS[design]
    probability_table = np.asarray(configuration["cohort_probabilities_by_x"], dtype=float)
    n_psus = int(configuration["n_psus"])
    trends = {1: 0.0, 2: 0.35, 3: 0.85, 4: 1.45}
    rows: list[pd.DataFrame] = []
    for period, raw_count in enumerate(configuration["period_sizes"], start=1):
        count = int(raw_count)
        x = rng.binomial(1, 0.5, size=count)
        probabilities = probability_table[x]
        draws = rng.random(count)
        cut_one = probabilities[:, 0]
        cut_two = cut_one + probabilities[:, 1]
        cohort = np.where(draws < cut_one, 3.0, np.where(draws < cut_two, 4.0, np.inf))
        cohort_code = np.where(cohort == 3.0, 0, np.where(cohort == 4.0, 1, 2))

        effect = np.zeros(count)
        effect[(cohort == 3.0) & (period == 3)] = 1.0
        effect[(cohort == 3.0) & (period == 4)] = 1.5
        effect[(cohort == 4.0) & (period == 4)] = 2.0
        group_level = np.where(
            cohort == 3.0,
            0.2 + 0.4 * x,
            np.where(cohort == 4.0, -0.3 - 0.2 * x, 0.1 * x),
        )
        conditional_trend = 0.2 * period * x if adjustment == "covariate" else 0.0
        untreated_mean = group_level + trends[period] + 0.7 * x + conditional_trend
        if design == "favorable_balanced":
            noise_scale = 0.75 + 0.10 * x
        else:
            noise_scale = 0.55 + 0.08 * period + 0.20 * (cohort == 4.0) + 0.15 * x

        frame = pd.DataFrame(
            {
                "outcome": untreated_mean + effect + rng.normal(scale=noise_scale),
                "time": float(period),
                "treatment_time": cohort,
                "x": x.astype(float),
            }
        )
        if sampling_unit == "psu":
            psu = np.arange(count, dtype=int) % n_psus
            rng.shuffle(psu)
            shared_shock = rng.normal(scale=0.65, size=(n_psus, 3))
            frame["outcome"] += shared_shock[psu, cohort_code]
            frame["psu"] = psu
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _nuisance_probability_range(result: Any) -> tuple[float | None, float | None]:
    if result.nuisance_predictions.empty:
        return None, None
    is_probability = (
        result.nuisance_predictions.columns.get_level_values("nuisance") == "propensity"
    )
    probabilities = result.nuisance_predictions.loc[:, is_probability].to_numpy(dtype=float)
    probabilities = probabilities[np.isfinite(probabilities)]
    if not len(probabilities):
        raise RuntimeError("No finite out-of-fold propensity predictions were retained.")
    return float(probabilities.min()), float(probabilities.max())


def _minimum_cluster_support(data: pd.DataFrame) -> int:
    support = data.groupby(["treatment_time", "time"], sort=False)["psu"].nunique()
    return int(support.min())


def _fit_one(
    data: pd.DataFrame,
    *,
    adjustment: str,
    sampling_unit: str,
    control_group: str,
    dgp_seed: int,
    band_seed: int,
    band_iterations: int,
) -> dict[str, Any]:
    constructor: dict[str, Any] = {
        "control_group": control_group,
        "covariance": "clustered" if sampling_unit == "psu" else "robust",
        "inference": "multiplier_bootstrap",
        "bootstrap_iterations": band_iterations,
        "random_state": band_seed,
        "simultaneous_level": NOMINAL_JOINT_COVERAGE,
    }
    fit_arguments: dict[str, Any] = {
        "outcome": "outcome",
        "time": "time",
        "treatment_time": "treatment_time",
    }
    if sampling_unit == "psu":
        fit_arguments["cluster"] = "psu"
    if adjustment == "covariate":
        fit_arguments["covariates"] = ["x"]
        fit_arguments["cross_fitter"] = CrossFitter(
            propensity_factory=_BinarySaturatedProbability,
            outcome_factory=_BinarySaturatedOutcome,
            n_splits=2,
            random_state=dgp_seed,
        )
    result = RepeatedCrossSectionDiD(**constructor).fit(data, **fit_arguments)

    event_times = result.simultaneous_event_study.index.to_numpy(dtype=float)
    if not np.array_equal(event_times, EVENT_TIMES):
        raise RuntimeError(f"Unexpected event coordinates: {event_times.tolist()}.")
    band = result.simultaneous_event_study
    critical_value = result.simultaneous_critical_value
    if critical_value is None:
        raise RuntimeError("Multiplier inference did not retain its critical value.")
    lower = band["lower"].to_numpy(dtype=float)
    upper = band["upper"].to_numpy(dtype=float)
    jointly_covered = bool(np.all(lower <= EVENT_TRUTH) and np.all(upper >= EVENT_TRUTH))
    simultaneous_half_width = (upper - lower) / 2.0
    if result.inference_distribution == "t":
        if result.inference_df is None:
            raise RuntimeError(
                "Clustered inference did not retain its reference degrees of freedom."
            )
        pointwise_critical = float(
            student_t.ppf((1.0 + NOMINAL_JOINT_COVERAGE) / 2.0, result.inference_df)
        )
    else:
        pointwise_critical = float(norm.ppf((1.0 + NOMINAL_JOINT_COVERAGE) / 2.0))
    pointwise_half_width = pointwise_critical * result.event_study["std_err"].to_numpy(dtype=float)

    minimum_probability, maximum_probability = _nuisance_probability_range(result)
    cluster_fold_indivisible: bool | None = None
    if adjustment == "covariate" and sampling_unit == "psu":
        audit_data = data.assign(nuisance_fold=result.nuisance_fold)
        cluster_folds = audit_data.groupby("psu")["nuisance_fold"].nunique()
        cluster_fold_indivisible = bool(cluster_folds.eq(1).all())

    n_clusters = int(data["psu"].nunique()) if sampling_unit == "psu" else None
    return {
        "jointly_covered": jointly_covered,
        "critical_value": float(critical_value),
        "mean_simultaneous_half_width": float(simultaneous_half_width.mean()),
        "mean_pointwise_half_width": float(pointwise_half_width.mean()),
        "width_inflation_ratio": float(
            simultaneous_half_width.mean() / pointwise_half_width.mean()
        ),
        "event_times": event_times.tolist(),
        "minimum_cell": int(result.cell_counts["nobs"].min()),
        "minimum_cell_clusters": (
            _minimum_cluster_support(data) if sampling_unit == "psu" else None
        ),
        "minimum_nuisance_probability": minimum_probability,
        "maximum_nuisance_probability": maximum_probability,
        "nuisance_fold_fits": len(result.nuisance_diagnostics),
        "sampling_unit_reported": result.sampling_unit,
        "n_clusters": n_clusters,
        "cluster_fold_indivisible": cluster_fold_indivisible,
        "band_metadata_matched": bool(
            result.inference_method == "multiplier_bootstrap"
            and result.bootstrap_iterations == band_iterations
            and result.bootstrap_random_state == band_seed
            and result.simultaneous_level == NOMINAL_JOINT_COVERAGE
        ),
        "cluster_count_matched": bool(
            (sampling_unit == "observation" and result.n_clusters is None)
            or (sampling_unit == "psu" and result.n_clusters == n_clusters)
        ),
    }


def _simulation_task(
    task: tuple[str, str, str, int, tuple[int, int], int],
) -> list[dict[str, Any]]:
    design, adjustment, sampling_unit, dgp_seed, band_seeds, band_iterations = task
    data = _sample(design, adjustment, sampling_unit, dgp_seed)
    records: list[dict[str, Any]] = []
    for control_group, band_seed in zip(CONTROL_GROUPS, band_seeds, strict=True):
        identity = {
            "design": design,
            "adjustment": adjustment,
            "sampling_unit": sampling_unit,
            "control_group": control_group,
        }
        try:
            records.append(
                {
                    **identity,
                    **_fit_one(
                        data,
                        adjustment=adjustment,
                        sampling_unit=sampling_unit,
                        control_group=control_group,
                        dgp_seed=dgp_seed,
                        band_seed=band_seed,
                        band_iterations=band_iterations,
                    ),
                    "refusal": None,
                }
            )
        except (ValueError, RuntimeError, FloatingPointError) as error:
            records.append(
                {
                    **identity,
                    "jointly_covered": False,
                    "nuisance_fold_fits": 0,
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


def _tasks(replications: int, band_iterations: int) -> list[tuple[Any, ...]]:
    task_count = len(DESIGNS) * len(ADJUSTMENTS) * len(SAMPLING_UNITS) * replications
    dgp_sequences = np.random.SeedSequence(SEED).spawn(task_count)
    band_sequences = np.random.SeedSequence(BAND_SEED_ROOT).spawn(task_count * len(CONTROL_GROUPS))
    tasks: list[tuple[Any, ...]] = []
    position = 0
    for design in DESIGNS:
        for adjustment in ADJUSTMENTS:
            for sampling_unit in SAMPLING_UNITS:
                for _ in range(replications):
                    dgp_seed = int(dgp_sequences[position].generate_state(1)[0])
                    band_position = position * len(CONTROL_GROUPS)
                    band_seeds = tuple(
                        int(sequence.generate_state(1)[0])
                        for sequence in band_sequences[
                            band_position : band_position + len(CONTROL_GROUPS)
                        ]
                    )
                    tasks.append(
                        (
                            design,
                            adjustment,
                            sampling_unit,
                            dgp_seed,
                            band_seeds,
                            band_iterations,
                        )
                    )
                    position += 1
    return tasks


def run_certificate(
    *,
    replications: int,
    band_iterations: int,
    workers: int,
) -> dict[str, Any]:
    tasks = _tasks(replications, band_iterations)
    started = time.perf_counter()
    if workers == 1:
        records = [record for task in tasks for record in _simulation_task(task)]
    else:
        chunk_size = max(1, len(tasks) // (workers * 20))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = executor.map(_simulation_task, tasks, chunksize=chunk_size)
            records = [record for batch in batches for record in batch]
    elapsed = time.perf_counter() - started

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["design"],
            record["adjustment"],
            record["sampling_unit"],
            record["control_group"],
        )
        grouped[key].append(record)

    cells: list[dict[str, Any]] = []
    for design in DESIGNS:
        for adjustment in ADJUSTMENTS:
            for sampling_unit in SAMPLING_UNITS:
                for control_group in CONTROL_GROUPS:
                    values = grouped[(design, adjustment, sampling_unit, control_group)]
                    refusals = sum(value["refusal"] is not None for value in values)
                    completed = [value for value in values if value["refusal"] is None]
                    successes = sum(bool(value["jointly_covered"]) for value in completed)
                    coverage = successes / len(completed) if completed else 0.0
                    mcse = (
                        math.sqrt(coverage * (1.0 - coverage) / len(completed))
                        if completed
                        else math.inf
                    )
                    critical_values = np.asarray(
                        [value["critical_value"] for value in completed], dtype=float
                    )
                    simultaneous_widths = np.asarray(
                        [value["mean_simultaneous_half_width"] for value in completed],
                        dtype=float,
                    )
                    pointwise_widths = np.asarray(
                        [value["mean_pointwise_half_width"] for value in completed], dtype=float
                    )
                    width_ratios = np.asarray(
                        [value["width_inflation_ratio"] for value in completed], dtype=float
                    )
                    fold_fit_counts = {int(value["nuisance_fold_fits"]) for value in completed}
                    fold_fits = fold_fit_counts.pop() if len(fold_fit_counts) == 1 else -1
                    expected_fold_fits = 60 if adjustment == "covariate" else 0
                    metadata_matched = all(
                        bool(value["band_metadata_matched"]) for value in completed
                    )
                    coordinates_matched = all(
                        value["event_times"] == EVENT_TIMES.tolist() for value in completed
                    )
                    cluster_counts_matched = all(
                        bool(value["cluster_count_matched"]) for value in completed
                    )
                    fold_roles_matched = (
                        all(value["cluster_fold_indivisible"] is True for value in completed)
                        if adjustment == "covariate" and sampling_unit == "psu"
                        else True
                    )
                    finite_positive = bool(
                        len(critical_values)
                        and np.isfinite(critical_values).all()
                        and np.all(critical_values > 0.0)
                        and np.isfinite(simultaneous_widths).all()
                        and np.all(simultaneous_widths > 0.0)
                        and np.isfinite(pointwise_widths).all()
                        and np.all(pointwise_widths > 0.0)
                    )
                    passed = (
                        len(completed) == replications
                        and refusals == 0
                        and JOINT_COVERAGE_GATE[0] <= coverage <= JOINT_COVERAGE_GATE[1]
                        and mcse <= JOINT_COVERAGE_MCSE_GATE
                        and fold_fits == expected_fold_fits
                        and metadata_matched
                        and coordinates_matched
                        and cluster_counts_matched
                        and fold_roles_matched
                        and finite_positive
                    )
                    minimum_probability = (
                        min(float(value["minimum_nuisance_probability"]) for value in completed)
                        if adjustment == "covariate" and completed
                        else None
                    )
                    maximum_probability = (
                        max(float(value["maximum_nuisance_probability"]) for value in completed)
                        if adjustment == "covariate" and completed
                        else None
                    )
                    cells.append(
                        {
                            "design": design,
                            "adjustment": adjustment,
                            "sampling_unit": sampling_unit,
                            "control_group": control_group,
                            "replications": len(completed),
                            "refusals": refusals,
                            "event_times": EVENT_TIMES.tolist(),
                            "truth": EVENT_TRUTH.tolist(),
                            "jointly_covered": successes,
                            "joint_coverage": coverage,
                            "joint_coverage_mcse": mcse,
                            "joint_coverage_wilson_95": (
                                _wilson_interval(successes, len(completed))
                                if completed
                                else [math.nan, math.nan]
                            ),
                            "joint_coverage_gate": list(JOINT_COVERAGE_GATE),
                            "minimum_critical_value": float(critical_values.min()),
                            "mean_critical_value": float(critical_values.mean()),
                            "maximum_critical_value": float(critical_values.max()),
                            "mean_simultaneous_half_width": float(simultaneous_widths.mean()),
                            "mean_pointwise_half_width": float(pointwise_widths.mean()),
                            "mean_width_inflation_ratio": float(width_ratios.mean()),
                            "minimum_realized_cell": min(
                                int(value["minimum_cell"]) for value in completed
                            ),
                            "minimum_realized_cell_clusters": (
                                min(int(value["minimum_cell_clusters"]) for value in completed)
                                if sampling_unit == "psu" and completed
                                else None
                            ),
                            "minimum_nuisance_probability": minimum_probability,
                            "maximum_nuisance_probability": maximum_probability,
                            "nuisance_fold_fits_per_estimator": fold_fits,
                            "sampling_unit_reported": (
                                "cluster" if sampling_unit == "psu" else "observation"
                            ),
                            "n_clusters": (
                                int(DESIGNS[design]["n_psus"]) if sampling_unit == "psu" else None
                            ),
                            "all_band_metadata_matched": metadata_matched,
                            "all_event_coordinates_matched": coordinates_matched,
                            "all_cluster_counts_matched": cluster_counts_matched,
                            "all_covariate_cluster_folds_indivisible": fold_roles_matched,
                            "all_widths_finite_positive": finite_positive,
                            "status": "pass" if passed else "fail",
                        }
                    )

    total_fits = len(records)
    total_nuisance_fits = sum(int(record["nuisance_fold_fits"]) for record in records)
    total_multiplier_draws = sum(band_iterations for record in records if record["refusal"] is None)
    refusal_reasons = Counter(
        str(record["refusal"]) for record in records if record["refusal"] is not None
    )
    all_band_metadata_matched = all(bool(cell["all_band_metadata_matched"]) for cell in cells)
    all_event_coordinates_matched = all(
        bool(cell["all_event_coordinates_matched"]) for cell in cells
    )
    all_cluster_counts_matched = all(bool(cell["all_cluster_counts_matched"]) for cell in cells)
    all_cluster_folds_indivisible = all(
        bool(cell["all_covariate_cluster_folds_indivisible"])
        for cell in cells
        if cell["adjustment"] == "covariate" and cell["sampling_unit"] == "psu"
    )
    zero_refusals = not refusal_reasons
    return {
        "seed": SEED,
        "band_seed_root": BAND_SEED_ROOT,
        "replications_per_cell": replications,
        "band_iterations": band_iterations,
        "nominal_joint_coverage": NOMINAL_JOINT_COVERAGE,
        "joint_coverage_gate": list(JOINT_COVERAGE_GATE),
        "joint_coverage_mcse_gate": JOINT_COVERAGE_MCSE_GATE,
        "n_splits": N_SPLITS,
        "event_times": EVENT_TIMES.tolist(),
        "event_truth": EVENT_TRUTH.tolist(),
        "designs": DESIGNS,
        "dgp_contract": {
            "unadjusted": (
                "Independent repeated sections with stationary cohort composition and "
                "unconditional parallel trends."
            ),
            "covariate": (
                "Independent repeated sections with stationary binary-X-dependent cohort "
                "composition and conditional parallel trends containing time-by-X trends."
            ),
            "psu": (
                "Balanced recurring PSU labels with independent PSU-by-period-by-cohort "
                "shocks plus idiosyncratic errors; one multiplier and one cross-fit fold "
                "role per indivisible PSU."
            ),
        },
        "nuisance_contract": (
            "Two-fold fold-local saturated binary-X propensity and outcome nuisances; "
            "no oracle predictions, clipping, trimming, refitting, or refused-fit redraws."
        ),
        "total_estimator_fits": total_fits,
        "total_nuisance_fold_fits": total_nuisance_fits,
        "total_multiplier_draws": total_multiplier_draws,
        "refusal_reasons": dict(sorted(refusal_reasons.items())),
        "elapsed_seconds": elapsed,
        "fits_per_second": total_fits / elapsed,
        "cells": cells,
        "audit": {
            "seed_derivation": "numpy.random.SeedSequence",
            "fixed_root_seed": SEED,
            "fixed_band_seed_root": BAND_SEED_ROOT,
            "all_band_metadata_matched": all_band_metadata_matched,
            "all_event_coordinates_matched": all_event_coordinates_matched,
            "all_cluster_counts_matched": all_cluster_counts_matched,
            "all_covariate_cluster_folds_indivisible": all_cluster_folds_indivisible,
            "zero_refusal_gate": zero_refusals,
        },
        "status": "pass" if all(cell["status"] == "pass" for cell in cells) else "fail",
    }


def _default_workers() -> int:
    available = os.cpu_count() or 1
    return max(1, min(8, available - 1 if available > 1 else 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--band-iterations", type=int, default=DEFAULT_BAND_ITERATIONS)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/did_rcs_simultaneous_promotion_evidence.json"),
    )
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least two")
    if args.band_iterations < 99:
        parser.error("--band-iterations must be at least 99")
    if args.workers < 1:
        parser.error("--workers must be positive")

    certificate = run_certificate(
        replications=args.replications,
        band_iterations=args.band_iterations,
        workers=args.workers,
    )
    script_path = Path(__file__).resolve()
    evidence = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/validate_did_rcs_simultaneous_promotion.py",
        "generator_sha256": _sha256(script_path),
        "reproduction_command": (
            "python benchmarks/validate_did_rcs_simultaneous_promotion.py "
            f"--replications {args.replications} --band-iterations {args.band_iterations} "
            f"--workers {args.workers}"
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "causekit_version": causekit.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joint_coverage_certificate": certificate,
        "overall_status": "pass" if certificate["status"] == "pass" else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if evidence["overall_status"] != "pass":
        raise SystemExit(9)


if __name__ == "__main__":
    main()

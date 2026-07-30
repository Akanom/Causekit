"""Generate repeated-cross-section DiD publication-promotion evidence.

The simulation holds cohort composition stationary in the population while drawing
independent observations in every period.  It separately certifies never-treated and
not-yet-treated contracts under balanced and sharply unequal period sizes.

Run from the repository root::

    python benchmarks/validate_did_rcs_promotion.py --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import causekit
from causekit import DifferenceInDifferences, EfficientDiD, RepeatedCrossSectionDiD
from causekit.datasets import REAL_DATASETS, load_real_dataset, verified_real_data_path

SEED = 20_260_730
DEFAULT_REPLICATIONS = 1_000
NOMINAL_COVERAGE = 0.95
COHORT_SUPPORT = np.array([3.0, 4.0, np.inf])
COHORT_PROBABILITIES = np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 2.0])
DESIGNS = {
    "balanced": (240, 240, 240, 240),
    "unequal_period_sizes": (600, 300, 144, 96),
}
CONTROL_GROUPS = ("never_treated", "not_yet_treated")
TARGETS = {
    "group_3_3": 1.0,
    "group_3_4": 1.5,
    "group_4_4": 2.0,
    "event_0": 4.0 / 3.0,
    "event_1": 1.5,
    "calendar_3": 1.0,
    "calendar_4": 5.0 / 3.0,
    "esavg": 17.0 / 12.0,
}
COVERAGE_GATE = (0.90, 0.99)
STANDARD_ERROR_RATIO_GATE = (0.75, 1.25)
ABSOLUTE_BIAS_GATE = 0.08


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(design: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    trends = {1: 0.0, 2: 0.35, 3: 0.85, 4: 1.45}
    for period, count in enumerate(DESIGNS[design], start=1):
        cohort = rng.choice(COHORT_SUPPORT, size=count, p=COHORT_PROBABILITIES)
        effect = np.zeros(count)
        effect[(cohort == 3.0) & (period == 3)] = 1.0
        effect[(cohort == 3.0) & (period == 4)] = 1.5
        effect[(cohort == 4.0) & (period == 4)] = 2.0
        level = np.where(cohort == 3.0, 0.2, np.where(cohort == 4.0, -0.3, 0.0))
        if design == "balanced":
            noise_scale = np.ones(count)
        else:
            noise_scale = 0.65 + 0.12 * period + 0.25 * (cohort == 4.0)
        rows.append(
            pd.DataFrame(
                {
                    "outcome": level + trends[period] + effect + rng.normal(scale=noise_scale),
                    "time": float(period),
                    "treatment_time": cohort,
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
    }


def _simulation_task(task: tuple[str, int]) -> list[dict[str, Any]]:
    design, seed = task
    data = _sample(design, seed)
    rows: list[dict[str, Any]] = []
    for control_group in CONTROL_GROUPS:
        try:
            result = RepeatedCrossSectionDiD(control_group=control_group).fit(
                data,
                outcome="outcome",
                time="time",
                treatment_time="treatment_time",
            )
            estimates = _target_rows(result)
            minimum_cell = int(result.cell_counts["nobs"].min())
            for target, truth in TARGETS.items():
                estimate, standard_error = estimates[target]
                rows.append(
                    {
                        "design": design,
                        "control_group": control_group,
                        "target": target,
                        "truth": truth,
                        "estimate": float(estimate),
                        "standard_error": float(standard_error),
                        "covered": bool(
                            abs(float(estimate) - truth)
                            <= 1.959963984540054 * float(standard_error)
                        ),
                        "minimum_cell": minimum_cell,
                        "refusal": None,
                    }
                )
        except (ValueError, RuntimeError) as error:
            for target, truth in TARGETS.items():
                rows.append(
                    {
                        "design": design,
                        "control_group": control_group,
                        "target": target,
                        "truth": truth,
                        "estimate": None,
                        "standard_error": None,
                        "covered": False,
                        "minimum_cell": None,
                        "refusal": f"{type(error).__name__}: {error}",
                    }
                )
    return rows


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
        rows = [row for task in tasks for row in _simulation_task(task)]
    else:
        chunk_size = max(1, len(tasks) // (workers * 20))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = executor.map(_simulation_task, tasks, chunksize=chunk_size)
            rows = [row for batch in batches for row in batch]
    elapsed = time.perf_counter() - started

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["design"], row["control_group"], row["target"])].append(row)
    cells: list[dict[str, Any]] = []
    for design in DESIGNS:
        for control_group in CONTROL_GROUPS:
            for target, truth in TARGETS.items():
                values = grouped[(design, control_group, target)]
                refusals = sum(value["refusal"] is not None for value in values)
                completed = [value for value in values if value["refusal"] is None]
                estimates = np.asarray([value["estimate"] for value in completed], dtype=float)
                standard_errors = np.asarray(
                    [value["standard_error"] for value in completed], dtype=float
                )
                successes = sum(bool(value["covered"]) for value in completed)
                coverage = successes / len(completed) if completed else 0.0
                empirical_sd = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else math.nan
                mean_se = float(np.mean(standard_errors)) if len(standard_errors) else math.nan
                bias = float(np.mean(estimates) - truth) if len(estimates) else math.nan
                ratio = mean_se / empirical_sd
                passed = (
                    len(completed) == replications
                    and refusals == 0
                    and COVERAGE_GATE[0] <= coverage <= COVERAGE_GATE[1]
                    and STANDARD_ERROR_RATIO_GATE[0] <= ratio <= STANDARD_ERROR_RATIO_GATE[1]
                    and abs(bias) <= ABSOLUTE_BIAS_GATE
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
                        "status": "pass" if passed else "fail",
                    }
                )
    fits = replications * len(DESIGNS) * len(CONTROL_GROUPS)
    return {
        "seed": SEED,
        "replications_per_design": replications,
        "nominal_coverage": NOMINAL_COVERAGE,
        "population_cohort_probabilities": {
            "treated_at_3": 1.0 / 3.0,
            "treated_at_4": 1.0 / 6.0,
            "never_treated": 1.0 / 2.0,
        },
        "period_sample_sizes": {name: list(value) for name, value in DESIGNS.items()},
        "dgp": (
            "Independent period samples; stationary cohort probabilities; common nonlinear "
            "time path; cohort levels; heterogeneous normal variance in unequal-size design; "
            "ATT(3,3)=1, ATT(3,4)=1.5, ATT(4,4)=2."
        ),
        "total_estimator_fits": fits,
        "elapsed_seconds": elapsed,
        "fits_per_second": fits / elapsed,
        "cells": cells,
        "status": "pass" if all(cell["status"] == "pass" for cell in cells) else "fail",
    }


def run_real_data(*, data_directory: Path | None, download: bool) -> dict[str, Any]:
    source = verified_real_data_path("hospdd", data_directory=data_directory, download=download)
    data = load_real_dataset("hospdd", data_directory=data_directory, download=False)
    repeated = data[["hospital", "month", "satis", "procedure"]].copy()
    first_treated = (
        repeated.loc[repeated["procedure"].eq(1)].groupby("hospital", sort=False)["month"].min()
    )
    repeated["treatment_time"] = repeated["hospital"].map(first_treated).fillna(np.inf)
    repeated = repeated.rename(columns={"satis": "outcome"}).reset_index(drop=True)
    rcs = RepeatedCrossSectionDiD(covariance="clustered").fit(
        repeated,
        outcome="outcome",
        time="month",
        treatment_time="treatment_time",
        cluster="hospital",
    )
    panel = (
        data.groupby(["hospital", "month"], as_index=False)
        .agg(outcome=("satis", "mean"), treated=("procedure", "max"))
        .sort_values(["hospital", "month"], kind="stable")
        .reset_index(drop=True)
    )
    panel_first = panel.loc[panel["treated"].eq(1)].groupby("hospital")["month"].min()
    panel["treatment_time"] = panel["hospital"].map(panel_first).fillna(np.inf)
    arguments = {
        "outcome": "outcome",
        "entity": "hospital",
        "time": "month",
        "treatment_time": "treatment_time",
    }
    conventional = DifferenceInDifferences().fit(panel, **arguments)
    efficient = EfficientDiD(pre_periods="all").fit(panel, **arguments)
    rows = {
        "repeated_cross_section_clustered": {
            "estimate": rcs.estimate,
            "standard_error": rcs.standard_error,
            "nobs": rcs.nobs,
            "n_clusters": rcs.n_clusters,
        },
        "panel_conventional": {
            "estimate": conventional.estimate,
            "standard_error": conventional.standard_error,
            "nobs": conventional.nobs,
        },
        "panel_efficient_pt_all": {
            "estimate": efficient.estimate,
            "standard_error": efficient.standard_error,
            "nobs": efficient.nobs,
        },
    }
    complete = all(
        math.isfinite(value[statistic])
        for value in rows.values()
        for statistic in ("estimate", "standard_error")
    )
    return {
        "dataset": "hospdd",
        "source_url": REAL_DATASETS["hospdd"].url,
        "source_sha256": _sha256(source),
        "source_label": "artificial hospital procedure data",
        "interpretation": "execution and clustered-inference smoke, not substantive evidence",
        "rows": rows,
        "status": "pass" if complete else "fail",
    }


def _default_workers() -> int:
    available = os.cpu_count() or 1
    return max(1, min(8, available - 1 if available > 1 else 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--data-directory", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/did_rcs_promotion_evidence.json"),
    )
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least two")
    if args.workers < 1:
        parser.error("--workers must be positive")

    simulation = run_simulation(replications=args.replications, workers=args.workers)
    real_data = run_real_data(data_directory=args.data_directory, download=args.download)
    script_path = Path(__file__).resolve()
    evidence = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/validate_did_rcs_promotion.py",
        "generator_sha256": _sha256(script_path),
        "reproduction_command": (
            "python benchmarks/validate_did_rcs_promotion.py "
            f"--replications {args.replications} --workers {args.workers}"
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "causekit_version": causekit.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "coverage_simulation": simulation,
        "real_data_application": real_data,
        "overall_status": (
            "pass" if simulation["status"] == "pass" and real_data["status"] == "pass" else "fail"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if evidence["overall_status"] != "pass":
        raise SystemExit(9)


if __name__ == "__main__":
    main()

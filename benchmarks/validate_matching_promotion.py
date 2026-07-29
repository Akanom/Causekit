"""Generate matching coverage and real-data sensitivity promotion evidence.

The two sections intentionally target different claims:

* coverage uses only the maintained no-caliper/no-support analytical contracts; and
* Cattaneo sensitivity uses ``inference='none'`` because support/caliper selection can
  change the realized target population.

Default release-certificate command, run from the repository root::

    python benchmarks/validate_matching_promotion.py --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import causekit
from causekit import NearestNeighborMatch
from causekit.datasets import REAL_DATASETS, load_real_dataset, verified_real_data_path

SEED = 20_260_730
TRUE_EFFECT = 1.5
NOMINAL_COVERAGE = 0.95
DEFAULT_REPLICATIONS = 1_000
DEFAULT_NOBS = 500
OVERLAP_DESIGNS = {
    "favorable": 0.5,
    "stressed": 2.2,
}
Estimand = Literal["att", "atc", "ate"]
ESTIMANDS: tuple[Estimand, ...] = ("att", "atc", "ate")
COVERAGE_GATE = (0.90, 0.99)
STANDARD_ERROR_RATIO_GATE = (0.75, 1.25)
ABSOLUTE_BIAS_GATE = 0.08


class _StatsmodelsLogitResult:
    """Provider-neutral adapter around an independently fitted Logit MLE."""

    def __init__(self, result: Any, design: pd.DataFrame) -> None:
        self._result = result
        self.params = pd.Series(result.params, index=design.columns, name="estimate")
        self.converged = bool(result.mle_retvals["converged"])
        self.nobs = len(design)
        self.feature_names = tuple(design.columns)

    def predict_proba(self, X: Any) -> pd.DataFrame:
        probability = np.asarray(self._result.predict(X), dtype=float)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


def _expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_or_none(value: Any) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _coverage_task(task: tuple[str, int, int]) -> list[dict[str, Any]]:
    """Run both maintained contracts on one deterministic simulated sample."""

    try:
        import statsmodels.api as sm  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - benchmark environment contract
        raise RuntimeError(
            "Matching promotion validation requires causekit[validation]."
        ) from error

    overlap, seed, nobs = task
    coefficient = OVERLAP_DESIGNS[overlap]
    rng = np.random.default_rng(seed)
    covariate = rng.uniform(-1.0, 1.0, nobs)
    propensity = _expit(coefficient * covariate)
    treatment = pd.Series(rng.binomial(1, propensity), dtype=int)
    design = pd.DataFrame({"const": 1.0, "x": covariate})
    noise_scale = 0.8 + 0.3 * (covariate + 1.0) / 2.0
    untreated_outcome = 0.5 + 0.7 * covariate + 0.25 * covariate**2
    outcome = pd.Series(
        untreated_outcome
        + TRUE_EFFECT * treatment.to_numpy(dtype=float)
        + rng.normal(scale=noise_scale, size=nobs)
    )
    fitted = sm.Logit(treatment, design).fit(
        method="newton",
        maxiter=200,
        tol=1e-11,
        disp=False,
    )
    fitted_propensity = _StatsmodelsLogitResult(fitted, design)

    rows: list[dict[str, Any]] = []
    for estimand in ESTIMANDS:
        known = NearestNeighborMatch(
            estimand=estimand,
            metric="propensity_logit",
            caliper=None,
            common_support=None,
            inference="abadie_imbens",
            variance_neighbors=1,
        ).fit(
            outcome,
            treatment=treatment,
            propensity=propensity,
            propensity_score_status="known",
            propensity_provenance="known_simulation_dgp",
        )
        estimated = NearestNeighborMatch(
            estimand=estimand,
            metric="propensity",
            caliper=None,
            common_support=None,
            inference="abadie_imbens_estimated",
            variance_neighbors=1,
            first_step_covariance_neighbors=2,
            first_step_regression_neighbors=2,
            first_step_covariate_neighbors=1,
        ).fit(
            outcome,
            treatment=treatment,
            propensity_model=fitted_propensity,
            propensity_design=design,
            propensity_score_status="estimated",
            propensity_provenance="full_sample_unpenalized_logit_mle",
        )
        for contract, result in (("known_score", known), ("estimated_logit", estimated)):
            interval = result.conf_int(level=NOMINAL_COVERAGE)
            rows.append(
                {
                    "overlap": overlap,
                    "contract": contract,
                    "estimand": estimand,
                    "estimate": result.estimate,
                    "standard_error": result.standard_error,
                    "covered": bool(interval["lower"] <= TRUE_EFFECT <= interval["upper"]),
                    "n_treated": result.n_treated,
                    "n_control": result.n_control,
                }
            )
    return rows


def _wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float]:
    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (probability + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1.0 - probability) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [center - radius, center + radius]


def _aggregate_coverage(
    rows: Iterable[dict[str, Any]],
    *,
    replications: int,
    nobs: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["overlap"], row["contract"], row["estimand"])].append(row)

    cells: list[dict[str, Any]] = []
    for overlap in OVERLAP_DESIGNS:
        for contract in ("known_score", "estimated_logit"):
            for estimand in ESTIMANDS:
                values = grouped[(overlap, contract, estimand)]
                estimates = np.asarray([value["estimate"] for value in values], dtype=float)
                standard_errors = np.asarray(
                    [value["standard_error"] for value in values], dtype=float
                )
                successes = sum(bool(value["covered"]) for value in values)
                coverage = successes / len(values)
                empirical_standard_deviation = float(np.std(estimates, ddof=1))
                mean_standard_error = float(np.mean(standard_errors))
                bias = float(np.mean(estimates) - TRUE_EFFECT)
                ratio = mean_standard_error / empirical_standard_deviation
                passed = (
                    len(values) == replications
                    and COVERAGE_GATE[0] <= coverage <= COVERAGE_GATE[1]
                    and STANDARD_ERROR_RATIO_GATE[0] <= ratio <= STANDARD_ERROR_RATIO_GATE[1]
                    and abs(bias) <= ABSOLUTE_BIAS_GATE
                )
                cells.append(
                    {
                        "overlap": overlap,
                        "contract": contract,
                        "estimand": estimand,
                        "replications": len(values),
                        "refusals": 0,
                        "mean_estimate": float(np.mean(estimates)),
                        "bias": bias,
                        "absolute_bias_gate": ABSOLUTE_BIAS_GATE,
                        "empirical_standard_deviation": empirical_standard_deviation,
                        "mean_standard_error": mean_standard_error,
                        "standard_error_ratio": ratio,
                        "standard_error_ratio_gate": list(STANDARD_ERROR_RATIO_GATE),
                        "root_mean_squared_error": float(
                            np.sqrt(np.mean((estimates - TRUE_EFFECT) ** 2))
                        ),
                        "coverage": coverage,
                        "coverage_mcse": math.sqrt(coverage * (1.0 - coverage) / len(values)),
                        "coverage_wilson_95": _wilson_interval(successes, len(values)),
                        "coverage_gate": list(COVERAGE_GATE),
                        "mean_interval_width": float(2.0 * 1.959963984540054 * mean_standard_error),
                        "mean_treated": float(np.mean([value["n_treated"] for value in values])),
                        "mean_control": float(np.mean([value["n_control"] for value in values])),
                        "status": "pass" if passed else "fail",
                    }
                )

    return {
        "seed": SEED,
        "replications_per_overlap": replications,
        "nobs_per_replication": nobs,
        "total_replications": replications * len(OVERLAP_DESIGNS),
        "total_estimator_fits": replications * len(OVERLAP_DESIGNS) * 6,
        "true_effect": TRUE_EFFECT,
        "nominal_coverage": NOMINAL_COVERAGE,
        "dgp": (
            "X~Uniform(-1,1); W~Bernoulli(expit(beta*X)); "
            "Y(0)=0.5+0.7X+0.25X^2+heteroskedastic normal noise; Y(1)-Y(0)=1.5"
        ),
        "overlap_designs": {
            name: {
                "logit_slope": coefficient,
                "theoretical_minimum_propensity": float(_expit(np.array([-coefficient]))[0]),
                "theoretical_maximum_propensity": float(_expit(np.array([coefficient]))[0]),
            }
            for name, coefficient in OVERLAP_DESIGNS.items()
        },
        "elapsed_seconds": elapsed_seconds,
        "cells": cells,
        "status": "pass" if all(cell["status"] == "pass" for cell in cells) else "fail",
    }


def run_coverage(*, replications: int, nobs: int, workers: int) -> dict[str, Any]:
    seed_sequences = np.random.SeedSequence(SEED).spawn(replications * len(OVERLAP_DESIGNS))
    tasks: list[tuple[str, int, int]] = []
    sequence_position = 0
    for overlap in OVERLAP_DESIGNS:
        for _ in range(replications):
            seed = int(seed_sequences[sequence_position].generate_state(1)[0])
            tasks.append((overlap, seed, nobs))
            sequence_position += 1

    started = time.perf_counter()
    if workers == 1:
        rows = [row for task in tasks for row in _coverage_task(task)]
    else:
        chunk_size = max(1, len(tasks) // (workers * 20))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            worker_rows = executor.map(_coverage_task, tasks, chunksize=chunk_size)
            rows = [row for replication in worker_rows for row in replication]
    elapsed = time.perf_counter() - started
    return _aggregate_coverage(
        rows,
        replications=replications,
        nobs=nobs,
        elapsed_seconds=elapsed,
    )


def run_real_data_sensitivity(
    *,
    data_directory: Path | None,
    download: bool,
) -> dict[str, Any]:
    try:
        import statsmodels.api as sm
    except ImportError as error:  # pragma: no cover - benchmark environment contract
        raise RuntimeError(
            "Matching promotion validation requires causekit[validation]."
        ) from error

    source_path = verified_real_data_path(
        "cattaneo2",
        data_directory=data_directory,
        download=download,
    )
    data = load_real_dataset("cattaneo2", data_directory=data_directory, download=False)
    covariate_names = ["mmarried", "mage", "medu", "fbaby"]
    covariates = data[covariate_names].astype(float)
    design = pd.concat(
        [pd.Series(1.0, index=data.index, name="const"), covariates],
        axis=1,
    )
    treatment = data["mbsmoke"].astype(int)
    outcome = data["bweight"].astype(float)
    fitted = sm.Logit(treatment, design).fit(
        method="newton",
        maxiter=200,
        tol=1e-11,
        disp=False,
    )
    propensity = pd.Series(fitted.predict(design), index=data.index, name="propensity")
    logit_score = np.log(propensity / (1.0 - propensity))
    logit_standard_deviation = float(np.std(logit_score, ddof=1))
    specifications: tuple[tuple[str, str | float | None, str | None], ...] = (
        ("no_caliper_no_support", None, None),
        ("no_caliper_intersection", None, "intersection"),
        ("narrow_0.1_sd", 0.1 * logit_standard_deviation, "intersection"),
        ("default_0.2_sd", "auto", "intersection"),
        ("wide_0.3_sd", 0.3 * logit_standard_deviation, "intersection"),
    )

    rows: list[dict[str, Any]] = []
    for specification, caliper, common_support in specifications:
        for estimand in ESTIMANDS:
            result = NearestNeighborMatch(
                estimand=estimand,
                metric="propensity_logit",
                neighbors=1,
                replacement=True,
                caliper=caliper,
                common_support=common_support,
                ties="all",
                inference="none",
            ).fit(
                outcome,
                treatment=treatment,
                propensity=propensity,
                covariates=covariates,
                propensity_score_status="estimated",
                propensity_provenance="full_sample_unpenalized_logit_mle",
            )
            balance = result.balance_summary
            ecdf_after = result.balance["ecdf_max_after"].dropna()
            rows.append(
                {
                    "specification": specification,
                    "requested_estimand": estimand,
                    "realized_estimand": result.realized_estimand,
                    "estimate": result.estimate,
                    "standard_error": None,
                    "inference": result.inference,
                    "realized_caliper": result.realized_caliper,
                    "common_support": result.common_support,
                    "matched_focal": result.n_matched_focal,
                    "matched_focal_fraction": result.matched_focal_fraction,
                    "support_excluded_treated": result.n_support_excluded_treated,
                    "support_excluded_control": result.n_support_excluded_control,
                    "caliper_unmatched": result.n_caliper_unmatched,
                    "boundary_tie_events": result.boundary_tie_events,
                    "maximum_reuse_count": result.maximum_reuse_count,
                    "maximum_absolute_smd_before": _finite_or_none(balance["max_abs_smd_before"]),
                    "maximum_absolute_smd_after": _finite_or_none(balance["max_abs_smd_after"]),
                    "mean_absolute_smd_after": _finite_or_none(balance["mean_abs_smd_after"]),
                    "maximum_ecdf_after": (float(ecdf_after.max()) if len(ecdf_after) else None),
                }
            )

    estimate_ranges = {
        estimand: {
            "minimum": min(
                row["estimate"] for row in rows if row["requested_estimand"] == estimand
            ),
            "maximum": max(
                row["estimate"] for row in rows if row["requested_estimand"] == estimand
            ),
        }
        for estimand in ESTIMANDS
    }
    for values in estimate_ranges.values():
        values["span"] = values["maximum"] - values["minimum"]

    complete = (
        len(rows) == len(specifications) * len(ESTIMANDS)
        and all(row["inference"] == "none" for row in rows)
        and all(row["matched_focal"] > 0 for row in rows)
    )
    return {
        "dataset": "cattaneo2",
        "source_filename": source_path.name,
        "source_url": REAL_DATASETS["cattaneo2"].url,
        "source_sha256": _sha256(source_path),
        "nobs": len(data),
        "n_treated": int(treatment.sum()),
        "n_control": int((1 - treatment).sum()),
        "score_fit": "full_sample_unpenalized_logit_mle",
        "score_formula": "mbsmoke ~ 1 + mmarried + mage + medu + fbaby",
        "score_converged": bool(fitted.mle_retvals["converged"]),
        "score_parameters": {str(name): float(value) for name, value in fitted.params.items()},
        "propensity_minimum": float(propensity.min()),
        "propensity_maximum": float(propensity.max()),
        "logit_standard_deviation_ddof_1": logit_standard_deviation,
        "inference": "none",
        "reason_no_inference": (
            "Support/caliper sensitivity can alter the retained target; the maintained "
            "analytical paths require caliper=None and common_support=None."
        ),
        "estimate_ranges": estimate_ranges,
        "rows": rows,
        "status": "pass" if complete else "fail",
    }


def _default_workers() -> int:
    available = os.cpu_count() or 1
    return max(1, min(8, available - 1 if available > 1 else 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--nobs", type=int, default=DEFAULT_NOBS)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--data-directory", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/matching_promotion_evidence.json"),
    )
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least two")
    if args.nobs < 50:
        parser.error("--nobs must be at least 50")
    if args.workers < 1:
        parser.error("--workers must be positive")

    coverage = run_coverage(
        replications=args.replications,
        nobs=args.nobs,
        workers=args.workers,
    )
    sensitivity = run_real_data_sensitivity(
        data_directory=args.data_directory,
        download=args.download,
    )
    script_path = Path(__file__).resolve()
    evidence = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/validate_matching_promotion.py",
        "generator_sha256": _sha256(script_path),
        "reproduction_command": (
            "python benchmarks/validate_matching_promotion.py "
            f"--replications {args.replications} --nobs {args.nobs} --workers {args.workers}"
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "causekit_version": causekit.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "statsmodels_version": importlib.metadata.version("statsmodels"),
        "coverage_simulation": coverage,
        "real_data_sensitivity": sensitivity,
        "overall_status": (
            "pass" if coverage["status"] == "pass" and sensitivity["status"] == "pass" else "fail"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if evidence["overall_status"] != "pass":
        raise SystemExit(9)


if __name__ == "__main__":
    main()

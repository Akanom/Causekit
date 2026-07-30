"""Fixed-seed nonlinear recovery and pointwise coverage gates for RD promotion."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import norm

import causekit
from causekit import RegressionDiscontinuity

OUTPUT = Path("benchmarks/rd_promotion_evidence.json")


@dataclass(frozen=True)
class Design:
    name: str
    design: str
    nobs: int
    truth: float
    bandwidth: float | str
    bias_bandwidth: float | None
    replications: int
    seed: int
    bandwidth_candidates: int = 15


def _one_replication(specification: Design, replication: int) -> tuple[float, float, bool]:
    rng = np.random.default_rng(specification.seed + replication)
    running = rng.uniform(-2.0, 2.0, specification.nobs)
    smooth = 0.5 + 0.55 * running + 0.22 * running**2 + 0.04 * running**3
    if specification.design == "sharp":
        treatment = None
        outcome = (
            smooth
            + specification.truth * (running >= 0.0)
            + rng.normal(scale=1.0, size=specification.nobs)
        )
    else:
        probability = 1.0 / (1.0 + np.exp(-(-1.0 + 2.0 * (running >= 0.0) + 0.10 * running)))
        treatment = rng.binomial(1, probability)
        outcome = (
            smooth
            + specification.truth * treatment
            + rng.normal(scale=0.70, size=specification.nobs)
        )
    estimator = RegressionDiscontinuity(
        design=specification.design,
        bandwidth=specification.bandwidth,
        bias_bandwidth=specification.bias_bandwidth,
        bandwidth_candidates=specification.bandwidth_candidates,
    )
    result = estimator.fit(outcome, running=running, treatment=treatment)
    return (
        result.bias_corrected_estimate,
        result.robust_standard_error,
        result.bandwidth_selection.selected_at_boundary,
    )


def _evaluate(specification: Design) -> dict[str, object]:
    estimates: list[float] = []
    standard_errors: list[float] = []
    boundary_selections = 0
    refusals = 0
    refusal_messages: dict[str, int] = {}
    critical = float(norm.ppf(0.975))
    for replication in range(specification.replications):
        try:
            estimate, standard_error, at_boundary = _one_replication(specification, replication)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            refusals += 1
            message = str(error)
            refusal_messages[message] = refusal_messages.get(message, 0) + 1
            continue
        estimates.append(estimate)
        standard_errors.append(standard_error)
        boundary_selections += int(at_boundary)
    estimate_array = np.asarray(estimates)
    se_array = np.asarray(standard_errors)
    if estimate_array.size < 2:
        raise RuntimeError(f"{specification.name} produced too few successful replications.")
    empirical_sd = float(np.std(estimate_array, ddof=1))
    mean_se = float(np.mean(se_array))
    coverage = float(np.mean(np.abs(estimate_array - specification.truth) <= critical * se_array))
    return {
        "name": specification.name,
        "design": specification.design,
        "nobs": specification.nobs,
        "truth": specification.truth,
        "bandwidth": specification.bandwidth,
        "bias_bandwidth": specification.bias_bandwidth,
        "bandwidth_candidates": specification.bandwidth_candidates,
        "replications_requested": specification.replications,
        "replications_successful": int(estimate_array.size),
        "seed": specification.seed,
        "refusals": refusals,
        "refusal_messages": refusal_messages,
        "mean_estimate": float(np.mean(estimate_array)),
        "absolute_bias": float(abs(np.mean(estimate_array) - specification.truth)),
        "empirical_standard_deviation": empirical_sd,
        "mean_standard_error": mean_se,
        "standard_error_ratio": mean_se / empirical_sd,
        "pointwise_coverage_95": coverage,
        "boundary_selection_fraction": boundary_selections / estimate_array.size,
    }


def _passes(record: dict[str, object]) -> bool:
    native = record["bandwidth"] == "native_mse"
    bias_limit = 0.10 if native else 0.075
    ratio_lower, ratio_upper = (0.75, 1.25) if native else (0.88, 1.12)
    coverage_lower, coverage_upper = (0.90, 0.99) if native else (0.925, 0.98)
    return bool(
        record["refusals"] == 0
        and record["absolute_bias"] <= bias_limit
        and ratio_lower <= record["standard_error_ratio"] <= ratio_upper
        and coverage_lower <= record["pointwise_coverage_95"] <= coverage_upper
    )


def main() -> None:
    designs = (
        Design(
            name="sharp_fixed_rbc",
            design="sharp",
            nobs=1_200,
            truth=2.0,
            bandwidth=1.3,
            bias_bandwidth=1.7,
            replications=1_000,
            seed=2_026_073_000,
        ),
        Design(
            name="fuzzy_fixed_rbc",
            design="fuzzy",
            nobs=1_600,
            truth=2.2,
            bandwidth=1.2,
            bias_bandwidth=1.6,
            replications=1_000,
            seed=2_026_083_000,
        ),
        Design(
            name="sharp_native_selector",
            design="sharp",
            nobs=1_000,
            truth=2.0,
            bandwidth="native_mse",
            bias_bandwidth=None,
            replications=250,
            seed=2_026_093_000,
            bandwidth_candidates=7,
        ),
        Design(
            name="fuzzy_native_selector",
            design="fuzzy",
            nobs=1_600,
            truth=2.2,
            bandwidth="native_mse",
            bias_bandwidth=None,
            replications=250,
            seed=2_026_103_000,
            bandwidth_candidates=7,
        ),
    )
    records = [_evaluate(design) for design in designs]
    for record in records:
        record["status"] = "pass" if _passes(record) else "fail"
    status = "pass" if all(record["status"] == "pass" for record in records) else "fail"
    evidence = {
        "artifact": "causekit_regression_discontinuity_promotion_v1",
        "generated_on": date.today().isoformat(),
        "generator": "benchmarks/validate_rd_promotion.py",
        "reproduction_command": "python benchmarks/validate_rd_promotion.py",
        "causekit_version": causekit.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "design": (
            "Nonlinear smooth cubic conditional mean, triangular p=1/q=2 RBC, fixed seeds; "
            "fuzzy take-up follows a nonlinear logistic first stage."
        ),
        "gates": {
            "fixed": {
                "replications": 1_000,
                "zero_refusals": True,
                "absolute_bias_max": 0.075,
                "standard_error_ratio": [0.88, 1.12],
                "pointwise_coverage_95": [0.925, 0.98],
            },
            "native_selector": {
                "replications": 250,
                "zero_refusals": True,
                "absolute_bias_max": 0.10,
                "standard_error_ratio": [0.75, 1.25],
                "pointwise_coverage_95": [0.90, 0.99],
                "boundary_fraction_is_diagnostic_not_a_pass_gate": True,
            },
        },
        "results": records,
        "status": status,
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"results_file={OUTPUT.as_posix()}")
    print(f"promotion_status={status}")
    if status != "pass":
        raise SystemExit(9)


if __name__ == "__main__":
    main()

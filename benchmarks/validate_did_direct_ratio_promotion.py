"""Frozen nonlinear/weak-overlap promotion certificate for direct PT-All odds."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from causekit import CrossFitter, EfficientDiD, __version__

try:
    from benchmarks._did_direct_ratio_support import (
        BinaryClassProbability,
        BinaryLogitOdds,
        OracleOdds,
        PolynomialRegression,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/script.py`` execution
    from _did_direct_ratio_support import (  # type: ignore[no-redef]
        BinaryClassProbability,
        BinaryLogitOdds,
        OracleOdds,
        PolynomialRegression,
    )


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "did_direct_ratio_promotion_evidence.json"
TRUE_EFFECT = 2.0
REPLICATIONS = 1_000
NOBS = 600


def _panel(seed: int, *, slope: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=NOBS)
    eta = 0.15 + slope * x + 0.35 * (x**2 - 4.0 / 3.0)
    treated = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(bool)
    if min(int(treated.sum()), int((~treated).sum())) < 60:
        raise ValueError("Frozen simulation unexpectedly lost cohort support.")
    baseline_error = rng.normal(scale=0.8, size=NOBS)
    change_error = rng.normal(scale=1.0, size=NOBS)
    baseline = 1.0 + 0.5 * x - 0.15 * x**2 + baseline_error
    untreated_change = 0.4 + 0.3 * x + 0.2 * x**2 + change_error
    rows: list[dict[str, float | str]] = []
    treated_position = control_position = 0
    for position in range(NOBS):
        if treated[position]:
            entity = f"g_{treated_position:04d}"
            cohort = 2.0
            effect = TRUE_EFFECT
            treated_position += 1
        else:
            entity = f"n_{control_position:04d}"
            cohort = np.inf
            effect = 0.0
            control_position += 1
        rows.extend(
            [
                {
                    "entity": entity,
                    "time": 1.0,
                    "treatment_time": cohort,
                    "x": x[position],
                    "outcome": baseline[position],
                },
                {
                    "entity": entity,
                    "time": 2.0,
                    "treatment_time": cohort,
                    "x": x[position],
                    "outcome": baseline[position] + untreated_change[position] + effect,
                },
            ]
        )
    return pd.DataFrame(rows)


def _odds_function(slope: float) -> Callable[[pd.DataFrame], np.ndarray]:
    def odds(X: pd.DataFrame) -> np.ndarray:
        x = X["x"].to_numpy(dtype=float)
        return np.exp(0.15 + slope * x + 0.35 * (x**2 - 4.0 / 3.0))

    return odds


def _fit(panel: pd.DataFrame, *, route: str, seed: int, slope: float):
    if route == "direct":
        weighting = {"cohort_ratio_factory": lambda: BinaryLogitOdds(degree=2)}
    elif route == "multiclass":
        weighting = {"propensity_factory": lambda: BinaryClassProbability(degree=2)}
    elif route == "oracle":
        weighting = {"cohort_ratio_factory": lambda: OracleOdds(_odds_function(slope))}
    else:  # pragma: no cover - closed benchmark route set
        raise ValueError(route)
    return EfficientDiD(
        inference="multiplier_bootstrap",
        bootstrap_iterations=199,
        random_state=seed + 90_000,
    ).fit(
        panel,
        outcome="outcome",
        entity="entity",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=CrossFitter(
            **weighting,
            outcome_factory=lambda: PolynomialRegression(degree=2),
            n_splits=3,
            random_state=seed + 70_000,
        ),
    )


def _summarize(values: list[tuple[float, float, bool, bool]]) -> dict[str, float | int]:
    estimates = np.array([item[0] for item in values])
    standard_errors = np.array([item[1] for item in values])
    point_coverage = np.mean([item[2] for item in values])
    simultaneous_coverage = np.mean([item[3] for item in values])
    empirical_sd = float(estimates.std(ddof=1))
    return {
        "fits": len(values),
        "mean_estimate": float(estimates.mean()),
        "absolute_bias": float(abs(estimates.mean() - TRUE_EFFECT)),
        "rmse": float(np.sqrt(np.mean((estimates - TRUE_EFFECT) ** 2))),
        "empirical_sd": empirical_sd,
        "mean_standard_error": float(standard_errors.mean()),
        "empirical_sd_to_mean_se": float(empirical_sd / standard_errors.mean()),
        "pointwise_coverage": float(point_coverage),
        "simultaneous_coverage": float(simultaneous_coverage),
    }


def main() -> None:
    started = time.perf_counter()
    designs = {"nonlinear_favorable": 0.45, "nonlinear_weak_overlap": 1.0}
    evidence: dict[str, object] = {
        "schema": "causekit_did_direct_ratio_promotion_v1",
        "causekit_version": __version__,
        "generated_on": "2026-07-30",
        "generator": "benchmarks/validate_did_direct_ratio_promotion.py",
        "reproduction_command": "python benchmarks/validate_did_direct_ratio_promotion.py",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "replications_per_design": REPLICATIONS,
        "entities_per_replication": NOBS,
        "true_effect": TRUE_EFFECT,
        "folds": 3,
        "bootstrap_iterations": 199,
        "fixed_seed_base": 20260730,
        "designs": {},
        "gates": {
            "maximum_absolute_bias": 0.08,
            "coverage_lower": 0.89,
            "coverage_upper": 0.99,
            "se_ratio_lower": 0.78,
            "se_ratio_upper": 1.22,
            "maximum_direct_multiclass_paired_estimate_difference": 1e-7,
            "maximum_direct_multiclass_paired_se_difference": 1e-7,
            "zero_refusals": True,
        },
    }
    all_pass = True
    for design_position, (design_name, slope) in enumerate(designs.items()):
        records = {route: [] for route in ("direct", "multiclass", "oracle")}
        paired_estimate_difference: list[float] = []
        paired_se_difference: list[float] = []
        folds_identical = True
        refusals = 0
        for replication in range(REPLICATIONS):
            seed = 20260730 + design_position * 10_000 + replication
            panel = _panel(seed, slope=slope)
            fitted = {}
            for route in records:
                try:
                    result = _fit(panel, route=route, seed=seed, slope=slope)
                except ValueError:
                    refusals += 1
                    continue
                fitted[route] = result
                critical = float(norm.ppf(0.975))
                records[route].append(
                    (
                        result.estimate,
                        result.standard_error,
                        bool(
                            result.estimate - critical * result.standard_error
                            <= TRUE_EFFECT
                            <= result.estimate + critical * result.standard_error
                        ),
                        bool(
                            result.simultaneous_event_study.iloc[0]["lower"]
                            <= TRUE_EFFECT
                            <= result.simultaneous_event_study.iloc[0]["upper"]
                        ),
                    )
                )
            if len(fitted) == 3:
                folds_identical &= fitted["direct"].nuisance_fold.equals(
                    fitted["multiclass"].nuisance_fold
                ) and fitted["direct"].nuisance_fold.equals(fitted["oracle"].nuisance_fold)
                paired_estimate_difference.append(
                    abs(fitted["direct"].estimate - fitted["multiclass"].estimate)
                )
                paired_se_difference.append(
                    abs(fitted["direct"].standard_error - fitted["multiclass"].standard_error)
                )
        summaries = {route: _summarize(values) for route, values in records.items()}
        maximum_estimate_difference = max(paired_estimate_difference, default=float("inf"))
        maximum_se_difference = max(paired_se_difference, default=float("inf"))
        design_pass = (
            refusals == 0
            and folds_identical
            and maximum_estimate_difference <= 1e-7
            and maximum_se_difference <= 1e-7
            and all(
                summary["absolute_bias"] <= 0.08
                and 0.89 <= summary["pointwise_coverage"] <= 0.99
                and 0.89 <= summary["simultaneous_coverage"] <= 0.99
                and 0.78 <= summary["empirical_sd_to_mean_se"] <= 1.22
                for summary in summaries.values()
            )
        )
        all_pass &= design_pass
        evidence["designs"][design_name] = {
            "slope": slope,
            "refusals": refusals,
            "folds_identical": folds_identical,
            "maximum_direct_multiclass_paired_estimate_difference": maximum_estimate_difference,
            "maximum_direct_multiclass_paired_se_difference": maximum_se_difference,
            "routes": summaries,
            "status": "pass" if design_pass else "fail",
        }
    evidence["elapsed_seconds"] = time.perf_counter() - started
    evidence["numerical_stability_benefit"] = {
        "design": "irrelevant-class softmax underflow",
        "multiclass_issue": (
            "Two score-relevant cohort probabilities underflow to zero when an irrelevant "
            "third-class logit dominates; their finite ratio is no longer recoverable from "
            "the probability matrix."
        ),
        "direct_pair_result": (
            "The binary restricted-cohort posterior odds remain finite because the "
            "irrelevant class cancels before prediction."
        ),
        "benefit_gate": "covered_by_unit_refusal_and_recovery_test",
    }
    evidence["status"] = "pass" if all_pass else "fail"
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not all_pass:
        raise SystemExit("Direct-ratio publication promotion gate failed.")


if __name__ == "__main__":
    main()

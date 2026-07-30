"""Publication evidence for survey-population repeated-section DiD.

Run from the repository root::

    python benchmarks/validate_did_rcs_survey_promotion.py

The fixed-seed certificate covers independent-observation and stratified-PSU Taylor
designs, informative and noninformative analysis weights, balanced and unequal wave
sizes, a hash-pinned YRBS sensitivity, and a 100,000-row performance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

import causekit
from causekit import RepeatedCrossSectionDiD, RepeatedCrossSectionSurveyDesign
from causekit.datasets import REAL_DATASETS, load_real_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "did_rcs_survey_promotion_evidence.json"
SEED = 20_260_730
COVERAGE_GATE = (0.90, 0.99)
SE_RATIO_GATE = (0.75, 1.25)
ABSOLUTE_BIAS_GATE = 0.08


def _simulation_sample(
    rng: np.random.Generator,
    *,
    sampling_unit: str,
    informative: bool,
    unequal_waves: bool,
) -> tuple[pd.DataFrame, float]:
    n_strata = 6
    psus_per_stratum = 8
    cell_sizes = (3, 5, 4, 6) if unequal_waves else (4, 4, 4, 4)
    rows: list[pd.DataFrame] = []
    weight_slope = 0.55 if informative else 0.0
    for stratum in range(n_strata):
        for psu_position in range(psus_per_stratum):
            psu = f"s{stratum}_p{psu_position}"
            cell_shocks = (
                np.zeros(4) if sampling_unit == "observation" else rng.normal(scale=0.45, size=4)
            )
            for cell, cell_n in enumerate(cell_sizes):
                treated = cell < 2
                post = cell % 2 == 1
                x = rng.normal(size=cell_n)
                weight = np.exp(weight_slope * x + 0.04 * stratum)
                treatment_effect = 1.0 + 0.2 * x + 0.1 * x**2
                outcome = (
                    0.4 * x
                    + 0.15 * x**2
                    + 0.3 * treated
                    + post * (0.45 + 0.08 * x)
                    + treated * post * treatment_effect
                    + cell_shocks[cell]
                    + rng.normal(scale=0.9, size=cell_n)
                )
                if sampling_unit == "observation":
                    outcome += rng.normal(scale=0.25, size=cell_n)
                rows.append(
                    pd.DataFrame(
                        {
                            "outcome": outcome,
                            "time": float(post + 1),
                            "treatment_time": 2.0 if treated else np.inf,
                            "weight": weight,
                            "psu": psu,
                            "stratum": stratum,
                        }
                    )
                )
    data = pd.concat(rows, ignore_index=True)
    truth = 1.0 + 0.2 * weight_slope + 0.1 * (1.0 + weight_slope**2)
    return data, truth


def _coverage(replications: int) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    cells: list[dict[str, object]] = []
    total_fits = 0
    for sampling_unit in ("observation", "psu"):
        for informative in (False, True):
            for unequal_waves in (False, True):
                estimates: list[float] = []
                standard_errors: list[float] = []
                covered = 0
                refusals = 0
                min_design_df = np.inf
                for _ in range(replications):
                    data, truth = _simulation_sample(
                        rng,
                        sampling_unit=sampling_unit,
                        informative=informative,
                        unequal_waves=unequal_waves,
                    )
                    try:
                        result = RepeatedCrossSectionDiD().fit(
                            data,
                            outcome="outcome",
                            time="time",
                            treatment_time="treatment_time",
                            survey_design=RepeatedCrossSectionSurveyDesign(
                                weights="weight",
                                psu=None if sampling_unit == "observation" else "psu",
                                strata="stratum",
                            ),
                            target_population="survey_population",
                        )
                    except (TypeError, ValueError, RuntimeError):
                        refusals += 1
                        continue
                    total_fits += 1
                    estimates.append(result.estimate)
                    standard_errors.append(result.standard_error)
                    interval = result.conf_int()
                    covered += int(interval["lower"] <= truth <= interval["upper"])
                    min_design_df = min(min_design_df, float(result.inference_df))
                estimate_array = np.asarray(estimates)
                se_array = np.asarray(standard_errors)
                coverage = covered / len(estimates) if estimates else float("nan")
                empirical_sd = float(estimate_array.std(ddof=1))
                mean_se = float(se_array.mean())
                bias = float(estimate_array.mean() - truth)
                se_ratio = mean_se / empirical_sd
                mcse = float(np.sqrt(coverage * (1.0 - coverage) / len(estimates)))
                status = (
                    "pass"
                    if refusals == 0
                    and COVERAGE_GATE[0] <= coverage <= COVERAGE_GATE[1]
                    and abs(bias) <= ABSOLUTE_BIAS_GATE
                    and SE_RATIO_GATE[0] <= se_ratio <= SE_RATIO_GATE[1]
                    else "fail"
                )
                cells.append(
                    {
                        "sampling_unit": sampling_unit,
                        "weights": "informative" if informative else "noninformative",
                        "wave_sizes": "unequal" if unequal_waves else "balanced",
                        "replications": replications,
                        "successful_fits": len(estimates),
                        "refusals": refusals,
                        "truth": truth,
                        "mean_estimate": float(estimate_array.mean()),
                        "bias": bias,
                        "empirical_sd": empirical_sd,
                        "mean_standard_error": mean_se,
                        "standard_error_ratio": se_ratio,
                        "coverage": coverage,
                        "coverage_mcse": mcse,
                        "minimum_design_df": int(min_design_df),
                        "coverage_gate": list(COVERAGE_GATE),
                        "standard_error_ratio_gate": list(SE_RATIO_GATE),
                        "absolute_bias_gate": ABSOLUTE_BIAS_GATE,
                        "status": status,
                    }
                )
    return {
        "replications_per_cell": replications,
        "total_estimator_fits": total_fits,
        "nominal_coverage": 0.95,
        "cells": cells,
        "status": "pass" if all(cell["status"] == "pass" for cell in cells) else "fail",
    }


def _real_data() -> dict[str, object]:
    specification = REAL_DATASETS["yrbs_beverage_tax"]
    data = load_real_dataset("yrbs_beverage_tax")
    analysis = data.copy()
    analysis["period"] = analysis["comb_group_label"].isin([2, 4]).astype(float) + 1.0
    analysis["treatment_time"] = np.where(analysis["comb_group_label"].isin([1, 2]), 2.0, np.inf)
    survey = RepeatedCrossSectionDiD().fit(
        analysis,
        outcome="soda_usage",
        time="period",
        treatment_time="treatment_time",
        survey_design=RepeatedCrossSectionSurveyDesign(
            weights="weight",
            psu=None,
            strata="stratum",
            weight_type="inverse_inclusion",
        ),
        target_population="survey_population",
    )
    sample = RepeatedCrossSectionDiD().fit(
        analysis,
        outcome="soda_usage",
        time="period",
        treatment_time="treatment_time",
    )
    treated = analysis["treatment_time"].eq(2.0)
    post = analysis["period"].eq(2.0)

    def weighted_mean(mask: pd.Series) -> float:
        return float(
            np.average(analysis.loc[mask, "soda_usage"], weights=analysis.loc[mask, "weight"])
        )

    published_code_mapping = (
        weighted_mean(treated & post)
        - weighted_mean(treated & ~post)
        - weighted_mean(~treated & post)
        + weighted_mean(~treated & ~post)
    )
    return {
        "dataset": "yrbs_beverage_tax",
        "source_commit": "d0e1959d1e0dcc27f30b38a38f31d31fb3959799",
        "source_sha256": specification.sha256,
        "provenance_url": specification.provenance_url,
        "n_observations": len(analysis),
        "n_strata": int(analysis["stratum"].nunique()),
        "psu_mapping": "independent_observation_because_processed_public_file_omits_psu",
        "estimand_mapping": "authors_sampling_weight_only_four_component_hajek_did",
        "survey_estimate": survey.estimate,
        "survey_standard_error": survey.standard_error,
        "published_code_point_mapping": published_code_mapping,
        "point_absolute_difference": abs(survey.estimate - published_code_mapping),
        "sample_estimate": sample.estimate,
        "survey_minus_sample": survey.estimate - sample.estimate,
        "status": "pass" if abs(survey.estimate - published_code_mapping) <= 1e-12 else "fail",
        "interpretation": (
            "Execution sensitivity only: the public processed file omits PSU identifiers, so "
            "the standard error uses the explicitly declared independent-observation-within-"
            "stratum design and is not claimed to reproduce the paper's bootstrap inference."
        ),
    }


def _performance(n_observations: int) -> dict[str, object]:
    if n_observations < 10_000 or n_observations % 4:
        raise ValueError("performance n_observations must be at least 10,000 and divisible by four")
    rng = np.random.default_rng(SEED + 1)
    row = np.arange(n_observations)
    cell = row % 4
    n_psus = max(50, n_observations // 40)
    psu = (row // 4) % n_psus
    stratum = psu % 20
    x = rng.normal(size=n_observations)
    post = cell % 2 == 1
    treated = cell < 2
    weight = np.exp(0.35 * x + 0.02 * stratum)
    outcome = (
        0.5 * x
        + 0.1 * x**2
        + 0.3 * treated
        + 0.4 * post
        + treated * post * (1.0 + 0.2 * x)
        + rng.normal(size=n_observations)
    )
    data = pd.DataFrame(
        {
            "outcome": outcome,
            "time": post.astype(float) + 1.0,
            "treatment_time": np.where(treated, 2.0, np.inf),
            "weight": weight,
            "psu": psu,
            "stratum": stratum,
        }
    )
    tracemalloc.start()
    started = time.perf_counter()
    result = RepeatedCrossSectionDiD().fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        survey_design=RepeatedCrossSectionSurveyDesign(
            weights="weight", psu="psu", strata="stratum"
        ),
        target_population="survey_population",
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mib = peak / 1024**2
    return {
        "n_observations": n_observations,
        "n_psus": result.n_clusters,
        "n_strata": int(result.survey_design_diagnostics["n_strata"]),
        "elapsed_seconds": elapsed,
        "peak_python_mib": peak_mib,
        "estimate": result.estimate,
        "standard_error": result.standard_error,
        "elapsed_gate_seconds": 10.0,
        "memory_gate_mib": 350.0,
        "status": "pass" if elapsed <= 10.0 and peak_mib <= 350.0 else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=1_000)
    parser.add_argument("--n-observations", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.replications < 100:
        raise ValueError("replications must be at least 100")
    coverage = _coverage(args.replications)
    real_data = _real_data()
    performance = _performance(args.n_observations)
    script_path = Path(__file__).resolve()
    evidence = {
        "schema_version": 1,
        "causekit_version": causekit.__version__,
        "generator": "benchmarks/validate_did_rcs_survey_promotion.py",
        "generator_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "estimator_contract": {
            "point": "component_hajek_survey_population_did",
            "variance": "one_stage_with_replacement_stratified_psu_taylor",
            "degrees_of_freedom": "number_of_psus_minus_number_of_strata",
            "weight_types": ["inverse_inclusion", "calibrated_analysis"],
            "singleton_policy": "raise",
            "simultaneous_inference": "unavailable",
        },
        "coverage_simulation": coverage,
        "real_data_sensitivity": real_data,
        "performance": performance,
    }
    evidence["overall_status"] = (
        "pass"
        if coverage["status"] == real_data["status"] == performance["status"] == "pass"
        else "fail"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if evidence["overall_status"] != "pass":
        raise SystemExit("survey repeated-section promotion failed")


if __name__ == "__main__":
    main()

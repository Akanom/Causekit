"""Publication-scale fixed-seed recovery and coverage gates for Panel IV."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

import causekit
from causekit import PanelIV2SLS

OUTPUT = Path("benchmarks/panel_iv_promotion_evidence.json")


@dataclass(frozen=True)
class Design:
    name: str
    n_entities: int
    n_periods: int
    time_effects: bool
    unbalanced: bool
    replications: int
    seed: int
    truth: float = 1.6


def _sample(specification: Design, replication: int) -> pd.DataFrame:
    rng = np.random.default_rng(specification.seed + replication)
    rows: list[dict[str, float | int]] = []
    for entity in range(specification.n_entities):
        alpha = rng.normal()
        first_previous = rng.normal(scale=0.5)
        structural_previous = rng.normal(scale=0.4)
        omitted_period = (
            int(rng.integers(specification.n_periods)) if specification.unbalanced else -1
        )
        for period in range(specification.n_periods):
            first_error = 0.45 * first_previous + rng.normal(scale=0.55)
            structural_noise = 0.35 * structural_previous + rng.normal(scale=0.45)
            instrument = rng.normal()
            control = rng.normal()
            endogenous = 0.9 * instrument + 0.35 * control + 0.3 * alpha + first_error
            structural_error = 0.6 * first_error + structural_noise
            outcome = (
                specification.truth * endogenous
                - 0.45 * control
                + alpha
                + 0.18 * period
                + structural_error
            )
            first_previous = first_error
            structural_previous = structural_noise
            if period == omitted_period:
                continue
            rows.append(
                {
                    "entity": entity,
                    "time": period,
                    "outcome": outcome,
                    "endogenous": endogenous,
                    "instrument": instrument,
                    "control": control,
                }
            )
    return pd.DataFrame(rows)


def _evaluate(specification: Design) -> dict[str, object]:
    estimates: list[float] = []
    standard_errors: list[float] = []
    first_stage_statistics: list[float] = []
    covered: list[bool] = []
    refusals: dict[str, int] = {}
    critical = float(t.ppf(0.975, specification.n_entities - 1))
    for replication in range(specification.replications):
        try:
            result = PanelIV2SLS(time_effects=specification.time_effects).fit(
                _sample(specification, replication),
                outcome="outcome",
                endogenous="endogenous",
                instruments="instrument",
                exogenous="control",
                entity="entity",
                time="time",
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            message = str(error)
            refusals[message] = refusals.get(message, 0) + 1
            continue
        estimate = float(result.params["endogenous"])
        standard_error = float(result.standard_errors["endogenous"])
        estimates.append(estimate)
        standard_errors.append(standard_error)
        first_stage_statistics.append(result.first_stage["endogenous"].classical_f_statistic)
        covered.append(abs(estimate - specification.truth) <= critical * standard_error)
    estimate_array = np.asarray(estimates)
    se_array = np.asarray(standard_errors)
    if estimate_array.size < 2:
        raise RuntimeError(f"{specification.name} produced too few successful replications.")
    empirical_sd = float(np.std(estimate_array, ddof=1))
    mean_se = float(np.mean(se_array))
    return {
        "name": specification.name,
        "n_entities": specification.n_entities,
        "n_periods": specification.n_periods,
        "time_effects": specification.time_effects,
        "unbalanced": specification.unbalanced,
        "truth": specification.truth,
        "replications_requested": specification.replications,
        "replications_successful": int(estimate_array.size),
        "seed": specification.seed,
        "refusals": int(sum(refusals.values())),
        "refusal_messages": refusals,
        "mean_estimate": float(np.mean(estimate_array)),
        "absolute_bias": float(abs(np.mean(estimate_array) - specification.truth)),
        "empirical_standard_deviation": empirical_sd,
        "mean_standard_error": mean_se,
        "standard_error_ratio": mean_se / empirical_sd,
        "pointwise_coverage_95": float(np.mean(covered)),
        "median_first_stage_f": float(np.median(first_stage_statistics)),
    }


def _passes(record: dict[str, object]) -> bool:
    return bool(
        record["refusals"] == 0
        and record["absolute_bias"] <= 0.05
        and 0.82 <= record["standard_error_ratio"] <= 1.18
        and 0.91 <= record["pointwise_coverage_95"] <= 0.985
        and record["median_first_stage_f"] >= 25.0
    )


def main() -> None:
    designs = (
        Design("balanced_entity_fe", 80, 6, False, False, 500, 2_026_073_101),
        Design("balanced_two_way_fe", 80, 6, True, False, 500, 2_026_073_201),
        Design("unbalanced_two_way_fe", 80, 7, True, True, 500, 2_026_073_301),
    )
    records = [_evaluate(design) for design in designs]
    for record in records:
        record["status"] = "pass" if _passes(record) else "fail"
    status = "pass" if all(record["status"] == "pass" for record in records) else "fail"
    evidence = {
        "artifact": "causekit_panel_iv_promotion_v1",
        "generated_on": date.today().isoformat(),
        "generator": "benchmarks/validate_panel_iv_promotion.py",
        "reproduction_command": "python benchmarks/validate_panel_iv_promotion.py",
        "causekit_version": causekit.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "design": (
            "Static valid excluded instrument, endogenous first-stage disturbance, additive "
            "entity/time effects, and serially dependent errors with entity-clustered CR1."
        ),
        "gates": {
            "zero_refusals": True,
            "absolute_bias_max": 0.05,
            "standard_error_ratio": [0.82, 1.18],
            "pointwise_coverage_95": [0.91, 0.985],
            "median_first_stage_f_min": 25.0,
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

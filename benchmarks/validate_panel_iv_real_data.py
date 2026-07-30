"""Hash-pinned wage-panel fixed-effects IV parity and sensitivity record."""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS as ReferenceIV2SLS

import causekit
from causekit import PanelIV2SLS
from causekit.datasets import REAL_DATASETS, load_real_dataset

OUTPUT = Path("benchmarks/panel_iv_real_data_evidence.json")


def _design(*, download: bool) -> pd.DataFrame:
    data = load_real_dataset("wage_panel", download=download).sort_values(["nr", "year"])
    data = data.copy()
    data["union_lag"] = data.groupby("nr", sort=False)["union"].shift()
    data["hours_1000"] = data["hours"] / 1_000.0
    return data


def _reference(data: pd.DataFrame):
    complete = data.dropna(
        subset=["lwage", "union", "union_lag", "hours_1000", "married", "nr", "year"]
    ).copy()
    entity_effects = pd.get_dummies(
        complete["nr"].astype("category"), prefix="entity", drop_first=True, dtype=float
    )
    time_effects = pd.get_dummies(
        complete["year"].astype("category"), prefix="time", drop_first=True, dtype=float
    )
    exogenous = pd.concat(
        [
            pd.Series(1.0, index=complete.index, name="const"),
            complete[["hours_1000", "married"]],
            entity_effects,
            time_effects,
        ],
        axis=1,
    )
    return ReferenceIV2SLS(
        complete["lwage"],
        exogenous,
        complete[["union"]],
        complete[["union_lag"]],
    ).fit(cov_type="clustered", clusters=complete["nr"], debiased=True)


def main() -> None:
    data = _design(download=True)
    tracemalloc.start()
    started = time.perf_counter()
    native = PanelIV2SLS(missing="drop").fit(
        data,
        outcome="lwage",
        endogenous="union",
        instruments="union_lag",
        exogenous=["hours_1000", "married"],
        entity="nr",
        time="year",
    )
    native_elapsed = time.perf_counter() - started
    _, native_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    started = time.perf_counter()
    reference = _reference(data)
    reference_elapsed = time.perf_counter() - started
    terms = ["hours_1000", "married", "union"]
    coefficient_differences = np.abs(
        native.params.loc[terms].to_numpy() - reference.params.loc[terms].to_numpy()
    )
    covariance_differences = np.abs(
        native.covariance.loc[terms, terms].to_numpy() - reference.cov.loc[terms, terms].to_numpy()
    )
    tolerance = 1e-9
    status = (
        "pass"
        if float(coefficient_differences.max()) <= tolerance
        and float(covariance_differences.max()) <= tolerance
        else "fail"
    )
    source = REAL_DATASETS["wage_panel"]
    evidence = {
        "artifact": "causekit_panel_iv_real_data_v1",
        "generated_on": date.today().isoformat(),
        "generator": "benchmarks/validate_panel_iv_real_data.py",
        "reproduction_command": "python benchmarks/validate_panel_iv_real_data.py",
        "causekit_version": causekit.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "comparator": {"package": "linearmodels", "version": version("linearmodels")},
        "source": {
            "name": "Vella-Verbeek wage panel",
            "url": source.url,
            "sha256": source.sha256,
            "rows_raw": int(len(data)),
            "citation": (
                "Vella and Verbeek (1998), Whose Wages Do Unions Raise?, "
                "Journal of Applied Econometrics 13:163-183."
            ),
        },
        "contract": {
            "outcome": "lwage",
            "endogenous": ["union"],
            "excluded_instruments": ["one-period lag of union"],
            "exogenous": ["hours / 1000", "married"],
            "effects": ["nr", "year"],
            "covariance": "entity-clustered CR1",
            "inference": "t(G-1)",
            "missing": "joint drop of first lag row per entity",
        },
        "causekit": {
            "nobs": native.nobs,
            "n_entities": native.n_entities,
            "n_periods": native.n_periods,
            "balanced": native.balanced,
            "params": native.params.to_dict(),
            "standard_errors": native.standard_errors.to_dict(),
            "first_stage": native.first_stage["union"].to_dict(),
            "within_r_squared": native.within_r_squared,
            "elapsed_seconds": native_elapsed,
            "peak_memory_mib": native_peak / (1024**2),
        },
        "linearmodels_explicit_dummy_reference": {
            "params": reference.params.loc[terms].to_dict(),
            "standard_errors": reference.std_errors.loc[terms].to_dict(),
            "elapsed_seconds": reference_elapsed,
        },
        "absolute_differences": {
            "maximum_coefficient": float(coefficient_differences.max()),
            "maximum_covariance": float(covariance_differences.max()),
        },
        "tolerance": tolerance,
        "interpretation": (
            "Numerical parity and specification sensitivity only. Treating current union "
            "status as endogenous and its one-period lag as excluded does not establish "
            "lag exogeneity, exclusion, monotonicity, or a causal union-wage effect."
        ),
        "status": status,
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"results_file={OUTPUT.as_posix()}")
    print(f"parity_status={status}")
    if status != "pass":
        raise SystemExit(9)


if __name__ == "__main__":
    main()

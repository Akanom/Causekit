"""Hash-pinned longer-design composition sensitivity on Stata's hospdd data.

Hospital-month rows are analyzed through the repeated-section score with hospital-level
PSU inference. This is a robustness example, not a claim that recurring hospitals are
independent cross-sectional observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import causekit
from causekit import (
    CrossFitter,
    RepeatedCrossSectionDiD,
    did_rcs_composition_test,
)

try:
    from benchmarks._did_rcs_composition_support import (
        PenalizedSoftmax,
        StandardizedRidgeOutcome,
    )
except ModuleNotFoundError:
    from _did_rcs_composition_support import (  # type: ignore[no-redef]
        PenalizedSoftmax,
        StandardizedRidgeOutcome,
    )

DATA_SHA256 = "db9d3b7182eb8abd2e1c5190305d6edce431b57f690a87550dba27e61df72990"
SOURCE_SHA256 = "e3ae6451e89cb915c546ab772410046726f280ad7d117611376beb4f46a521bb"
SOURCE_URL = "https://www.stata-press.com/data/r19/hospdd.dta"
N_SPLITS = 5
FOLD_SEED = 1_234
BAND_SEED = 20_260_730


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_data_path() -> Path:
    local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return local_root / "causekit" / "parity" / "real_data_v1" / "hospdd_panel.csv"


def _cross_fitter() -> CrossFitter:
    return CrossFitter(
        propensity_factory=lambda: PenalizedSoftmax(alpha=0.5),
        outcome_factory=lambda: StandardizedRidgeOutcome(alpha=1.0),
        n_splits=N_SPLITS,
        random_state=FOLD_SEED,
    )


def _probability_bounds(result: Any) -> tuple[float, float]:
    nuisance = result.nuisance_predictions.columns.get_level_values("nuisance").astype(str)
    values = result.nuisance_predictions.loc[:, nuisance.str.contains("propensity")].to_numpy(
        dtype=float
    )
    finite = values[np.isfinite(values)]
    return float(finite.min()), float(finite.max())


def _table_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.reset_index().copy()
    for column in clean.columns:
        clean[column] = clean[column].map(
            lambda value: list(value) if isinstance(value, tuple) else value
        )
    return clean.to_dict(orient="records")


def run(data_path: Path) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing hash-pinned hospdd CSV: {data_path}. "
            "Run python benchmarks/prepare_real_data.py --download first."
        )
    if _sha256(data_path) != DATA_SHA256:
        raise ValueError("The hospdd analysis CSV does not match its pinned SHA-256 digest.")
    data = pd.read_csv(data_path)
    expected_columns = {"hospital", "month", "outcome", "treated", "treatment_time"}
    if set(data) != expected_columns or len(data) != 322:
        raise ValueError("The hospdd analysis schema or row count changed.")
    data["treatment_time"] = data["treatment_time"].replace(0, np.inf)
    data["hospital_scaled"] = (data["hospital"] - data["hospital"].mean()) / data["hospital"].std()
    common = {
        "data": data,
        "outcome": "outcome",
        "time": "month",
        "treatment_time": "treatment_time",
        "cluster": "hospital",
        "covariates": ["hospital_scaled"],
    }
    robust = RepeatedCrossSectionDiD(
        composition="robust",
        covariance="clustered",
        inference="multiplier_bootstrap",
        bootstrap_iterations=999,
        random_state=BAND_SEED,
        simultaneous_level=0.95,
    ).fit(**common, cross_fitter=_cross_fitter())
    stationary = RepeatedCrossSectionDiD(
        composition="stationary",
        covariance="clustered",
    ).fit(**common, cross_fitter=_cross_fitter())
    diagnostic = did_rcs_composition_test(robust, stationary)
    if not robust.nuisance_fold.groupby(data["hospital"]).nunique().eq(1).all():
        raise RuntimeError("A hospital crossed robust nuisance-fold roles.")
    if not stationary.nuisance_fold.groupby(data["hospital"]).nunique().eq(1).all():
        raise RuntimeError("A hospital crossed stationary nuisance-fold roles.")
    robust_bounds = _probability_bounds(robust)
    stationary_bounds = _probability_bounds(stationary)
    finite = np.isfinite(
        [
            robust.estimate,
            robust.standard_error,
            stationary.estimate,
            stationary.standard_error,
            diagnostic.statistic,
            diagnostic.pvalue,
            *robust_bounds,
            *stationary_bounds,
        ]
    ).all()
    status = (
        "pass"
        if finite
        and robust_bounds[0] > robust.nuisance_probability_floor
        and robust_bounds[1] < 1.0
        and stationary_bounds[0] > stationary.nuisance_probability_floor
        and stationary_bounds[1] < 1.0
        and len(robust.group_time) == 4
        and len(robust.pretrend.placebo_effects) == 2
        and len(robust.pair_ledger) == 6
        and len(robust.nuisance_diagnostics) == 120
        and len(stationary.nuisance_diagnostics) == 150
        else "fail"
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/validate_did_rcs_composition_longer_real_data.py",
        "generator_sha256": _sha256(Path(__file__)),
        "causekit_version": causekit.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source": {
            "name": "Stata hospdd",
            "source_url": SOURCE_URL,
            "source_sha256": SOURCE_SHA256,
            "analysis_csv_sha256": DATA_SHA256,
            "rows": len(data),
            "hospitals": int(data["hospital"].nunique()),
            "periods": int(data["month"].nunique()),
        },
        "design_note": (
            "Recurring hospital-month rows use hospital PSU inference; this sensitivity "
            "does not assert independent repeated cross sections."
        ),
        "external_longer_comparator": (
            "unavailable: official compdid exposes a two-period target, so no staggered "
            "point/influence comparator is manufactured"
        ),
        "robust": {
            "estimate": robust.estimate,
            "standard_error": robust.standard_error,
            "probability_bounds": list(robust_bounds),
            "nuisance_fold_fits": len(robust.nuisance_diagnostics),
            "pair_count": len(robust.pair_ledger),
            "group_time_aggregation": robust.group_time_aggregation,
            "esavg_aggregation": robust.esavg_aggregation,
            "group_time": _table_records(robust.group_time),
            "event_study": _table_records(robust.event_study),
            "calendar_time": _table_records(robust.calendar_time),
            "conditional_placebos": _table_records(robust.pretrend.placebo_effects),
            "simultaneous_event_study": _table_records(robust.simultaneous_event_study),
        },
        "stationary": {
            "estimate": stationary.estimate,
            "standard_error": stationary.standard_error,
            "probability_bounds": list(stationary_bounds),
            "nuisance_fold_fits": len(stationary.nuisance_diagnostics),
            "group_time": _table_records(stationary.group_time),
        },
        "diagnostic": {
            "statistic": diagnostic.statistic,
            "pvalue": diagnostic.pvalue,
            "reject": diagnostic.reject,
            "distribution": diagnostic.distribution,
            "df_num": diagnostic.df_num,
            "df_denom": diagnostic.df_denom,
            "contrast": "robust_minus_stationary",
            "estimator_selection": False,
            "group_time": _table_records(diagnostic.group_time),
        },
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/did_rcs_composition_longer_real_data_evidence.json"),
    )
    args = parser.parse_args()
    evidence = run(args.data)
    rendered = json.dumps(evidence, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if evidence["status"] != "pass":
        raise SystemExit("longer real-data composition gate failed")


if __name__ == "__main__":
    main()

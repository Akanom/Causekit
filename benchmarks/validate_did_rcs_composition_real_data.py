"""Hash-pinned Sequeira sensitivity for robust and stationary composition targets.

Prepare the external analysis CSV first, then run from the repository root::

    Rscript benchmarks/prepare_did_rcs_composition_real_data.R PATH_TO_COMPDID_CHECKOUT
    python benchmarks/validate_did_rcs_composition_real_data.py

The nuisance providers are deliberately small benchmark-owned ridge models. They are
trained inside every declared fold and exercise CauseKit's public provider protocol; this
is a design sensitivity record, not a reproduction of the paper's local-polynomial fit.
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
from causekit import CrossFitter, RepeatedCrossSectionDiD

try:
    from benchmarks._did_rcs_composition_support import (
        PenalizedSoftmax,
        StandardizedRidgeOutcome,
    )
except ModuleNotFoundError:  # Direct ``python benchmarks/...py`` execution.
    from _did_rcs_composition_support import (  # type: ignore[no-redef]
        PenalizedSoftmax,
        StandardizedRidgeOutcome,
    )

REFERENCE_COMMIT = "894bd65a952c30f01a4e0005efba4cb335065eb7"
SOURCE_BLOB = "6a6a8bbe9792bc6385849421a7fd0d76692cd79e"
SOURCE_SHA256 = "37f113f1c706a3b35c325b996884c843beb9a4f773c3928d505ca51b285a7953"
ANALYSIS_CSV_SHA256 = "48e3d0cc1bd2eb757e4ac0ad24a9cc60946c6041cb4c56debaac95e4aaebb492"
SEED = 1_234
N_SPLITS = 5
PROBABILITY_ALPHA = 0.5
OUTCOME_ALPHA = 1.0
DISCRETE_COVARIATES = (
    "differentiated",
    "agri",
    "perishable",
    "dfs",
    "day_w_arrival",
    "monitor",
    "psi",
    "rsa",
    "term",
)
OUTCOME_SPECS = {
    "bp": ("lvalue_tonnage", "tariff2007"),
    "lba": ("lvalue_tonnage", "tariff2007"),
    "lba_value": ("ltonnage", "tariff2007"),
    "lba_tonnage": ("lvalue_shipment_metical", "tariff2007"),
}
PAPER_FINAL = {
    "bp": {"robust": -0.30683023, "stationary": -0.27529705},
    "lba": {"robust": -2.88842169, "stationary": -2.54213811},
    "lba_value": {"robust": -0.02743327, "stationary": -0.01399963},
    "lba_tonnage": {"robust": -1.13074952, "stationary": -0.91795753},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_data_path() -> Path:
    local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return (
        local_root
        / "causekit"
        / "parity"
        / "did_rcs_composition_v1"
        / "sequeira_bribes_analysis.csv"
    )


def _cross_fitter() -> CrossFitter:
    return CrossFitter(
        propensity_factory=lambda: PenalizedSoftmax(alpha=PROBABILITY_ALPHA),
        outcome_factory=lambda: StandardizedRidgeOutcome(alpha=OUTCOME_ALPHA),
        n_splits=5,
        random_state=SEED,
    )


def _probability_bounds(result: Any) -> tuple[float, float]:
    nuisance = result.nuisance_predictions.columns.get_level_values("nuisance").astype(str)
    selected = result.nuisance_predictions.loc[:, nuisance.str.contains("propensity")]
    values = selected.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.min()), float(finite.max())


def _fit_one(data: pd.DataFrame, outcome: str) -> dict[str, Any]:
    covariates = [*OUTCOME_SPECS[outcome], *DISCRETE_COVARIATES]
    common = {
        "data": data,
        "outcome": outcome,
        "time": "time",
        "treatment_time": "treatment_time",
        "cluster": "hc_4digits",
        "covariates": covariates,
    }
    robust = RepeatedCrossSectionDiD(
        composition="robust",
        covariance="clustered",
    ).fit(**common, cross_fitter=_cross_fitter())
    stationary = RepeatedCrossSectionDiD(
        composition="stationary",
        covariance="clustered",
    ).fit(**common, cross_fitter=_cross_fitter())
    robust_bounds = _probability_bounds(robust)
    stationary_bounds = _probability_bounds(stationary)
    if not robust.nuisance_fold.groupby(data["hc_4digits"]).nunique().eq(1).all():
        raise RuntimeError("Robust nuisance folds split an HS-code cluster.")
    if not stationary.nuisance_fold.groupby(data["hc_4digits"]).nunique().eq(1).all():
        raise RuntimeError("Stationary nuisance folds split an HS-code cluster.")

    robust_record = {
        "estimate": robust.estimate,
        "standard_error": robust.standard_error,
        "minimum_probability": robust_bounds[0],
        "maximum_probability": robust_bounds[1],
        "maximum_weight": float(robust.composition_weights.max().max()),
        "nuisance_fold_fits": len(robust.nuisance_diagnostics),
        "target_population": robust.target_population,
        "paper_final_reference": PAPER_FINAL[outcome]["robust"],
    }
    stationary_record = {
        "estimate": stationary.estimate,
        "standard_error": stationary.standard_error,
        "minimum_probability": stationary_bounds[0],
        "maximum_probability": stationary_bounds[1],
        "nuisance_fold_fits": len(stationary.nuisance_diagnostics),
        "target_population": stationary.target_population,
        "paper_final_reference": PAPER_FINAL[outcome]["stationary"],
    }
    values = np.array(
        [
            robust.estimate,
            robust.standard_error,
            stationary.estimate,
            stationary.standard_error,
            *robust_bounds,
            *stationary_bounds,
        ]
    )
    probabilities_are_interior = (
        robust_bounds[0] > 0.0
        and robust_bounds[1] < 1.0
        and stationary_bounds[0] > 0.0
        and stationary_bounds[1] < 1.0
    )
    status = "pass" if np.isfinite(values).all() and probabilities_are_interior else "fail"
    return {
        "outcome": outcome,
        "covariates": covariates,
        "n_splits": N_SPLITS,
        "robust": robust_record,
        "stationary": stationary_record,
        "difference": robust.estimate - stationary.estimate,
        "status": status,
    }


def run(data_path: Path) -> dict[str, Any]:
    actual_hash = _sha256(data_path)
    if actual_hash != ANALYSIS_CSV_SHA256:
        raise RuntimeError(
            "Refusing an unverified Sequeira analysis CSV: "
            f"expected {ANALYSIS_CSV_SHA256}, received {actual_hash}."
        )
    data = pd.read_csv(data_path).set_index("source_row")
    if len(data) != 1_084 or data["hc_4digits"].nunique() != 131:
        raise RuntimeError("Sequeira analysis dimensions do not match the pinned contract.")
    data["time"] = data["post_2008"] + 1.0
    data["treatment_time"] = np.where(data["tariff_change_2008"].eq(1), 2.0, np.inf)
    cell_counts = data.groupby(["tariff_change_2008", "post_2008"], sort=True).size().tolist()
    results = [_fit_one(data, outcome) for outcome in OUTCOME_SPECS]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/validate_did_rcs_composition_real_data.py",
        "generator_sha256": _sha256(Path(__file__)),
        "causekit_version": causekit.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source": {
            "dataset": "Sequeira (2016) tariff liberalization and bribery",
            "reference_repository": "https://github.com/pedrohcgs/comp_did",
            "reference_commit": REFERENCE_COMMIT,
            "source_blob": SOURCE_BLOB,
            "source_sha256": SOURCE_SHA256,
            "analysis_csv_sha256": actual_hash,
            "rows": len(data),
            "clusters": int(data["hc_4digits"].nunique()),
            "cell_counts_00_01_10_11": cell_counts,
        },
        "nuisance_contract": {
            "probability_model": "benchmark_standardized_reference_class_ridge_softmax",
            "probability_alpha": PROBABILITY_ALPHA,
            "outcome_model": "benchmark_standardized_ridge",
            "outcome_alpha": OUTCOME_ALPHA,
            "folds": N_SPLITS,
            "fold_seed": SEED,
            "cluster_split": True,
            "paper_comparison": "descriptive_only_different_nuisance_and_inference_contract",
        },
        "results": results,
        "overall_status": "pass" if all(row["status"] == "pass" for row in results) else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/did_rcs_composition_real_data_evidence.json"),
    )
    args = parser.parse_args()
    report = run(args.data)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if report["overall_status"] != "pass":
        raise SystemExit("real-data composition sensitivity failed")


if __name__ == "__main__":
    main()

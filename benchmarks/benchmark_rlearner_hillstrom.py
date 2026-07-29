"""One-run native linear/nonlinear honest R-learner comparison on Hillstrom RCT data.

The source file is obtained separately from the declared Kaggle mirror. This script never
reads Kaggle credentials and refuses any source whose SHA-256 differs from the reviewed
64,000-row CSV.

Example:

    python benchmarks/benchmark_rlearner_hillstrom.py \
        --data path/to/Kevin_Hillstrom_...csv \
        --output causekit-rlearner-hillstrom-v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import causekit

if __package__:
    from .benchmark_rlearner import _run_once
else:  # pragma: no cover - exercised by the documented direct script workflow
    from benchmark_rlearner import _run_once

SOURCE_REF = "bofulee/kevin-hillstrom-minethatdata-e-mailanalytics"
SOURCE_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
DESIGN_ID = "hillstrom_mens_email_vs_control_honest_cate_v1"
MODEL_NAMES = (
    "causekit_native_weighted_ridge_gcv",
    "causekit_native_spline_ridge_gcv",
)
REQUIRED_COLUMNS = (
    "recency",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_hillstrom(data: pd.DataFrame) -> dict[str, Any]:
    missing = [name for name in REQUIRED_COLUMNS if name not in data.columns]
    if missing:
        raise ValueError(f"Hillstrom source is missing required columns: {missing}.")
    if data.loc[:, list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Hillstrom design columns must not contain missing values.")
    observed_segments = set(data["segment"].astype(str))
    required_segments = {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}
    if observed_segments != required_segments:
        raise ValueError("Hillstrom source must contain the three declared randomized arms.")
    selected = data.loc[data["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
    selected = selected.reset_index(drop=True)
    categorical = pd.get_dummies(
        selected[["zip_code", "channel"]].astype(str),
        prefix=["zip", "channel"],
        drop_first=True,
        dtype=float,
    )
    numeric = selected[["recency", "history", "mens", "womens", "newbie"]].astype(float)
    covariates = pd.concat([numeric, categorical], axis=1)
    return {
        "covariates": covariates,
        "treatment": selected["segment"].eq("Mens E-Mail").astype(float),
        "outcome": selected["visit"].astype(float),
    }


def _design(path: Path) -> dict[str, Any]:
    observed_hash = _sha256(path)
    if observed_hash != SOURCE_SHA256:
        raise ValueError(
            "Hillstrom source SHA-256 mismatch: "
            f"expected {SOURCE_SHA256}, observed {observed_hash}."
        )
    return _prepare_hillstrom(pd.read_csv(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    design = _design(args.data)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": DESIGN_ID,
        "dataset": "hillstrom_email_rct",
        "dataset_source": f"kaggle:{SOURCE_REF}",
        "dataset_source_sha256": SOURCE_SHA256,
        "outcome": "visit",
        "treatment": "Mens E-Mail versus No E-Mail",
        "excluded_arm": "Womens E-Mail",
        "covariates": list(design["covariates"].columns),
        "nobs": len(design["outcome"]),
        "benchmark_repetitions": 1,
        "comparison_policy": (
            "one identical honest split per model; native nuisances fixed in specification; "
            "only the native weighted CATE learner changes"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "causekit": causekit.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "results": [_run_once(name, design) for name in MODEL_NAMES],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

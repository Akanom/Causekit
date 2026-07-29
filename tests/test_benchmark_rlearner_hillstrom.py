"""Contracts for the hash-pinned Hillstrom R-learner benchmark."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from benchmarks import benchmark_rlearner_hillstrom


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recency": [1, 2, 3, 4, 5, 6],
            "history": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "mens": [1, 0, 1, 0, 1, 0],
            "womens": [0, 1, 0, 1, 0, 1],
            "zip_code": ["Urban", "Rural", "Urban", "Rural", "Urban", "Rural"],
            "newbie": [0, 1, 0, 1, 0, 1],
            "channel": ["Web", "Phone", "Web", "Phone", "Web", "Phone"],
            "segment": [
                "Mens E-Mail",
                "No E-Mail",
                "Womens E-Mail",
                "Mens E-Mail",
                "No E-Mail",
                "Womens E-Mail",
            ],
            "visit": [1, 0, 1, 0, 1, 0],
        }
    )


def test_hillstrom_design_keeps_randomized_binary_contrast_and_pretreatment_covariates() -> None:
    design = benchmark_rlearner_hillstrom._prepare_hillstrom(_source())

    assert design["outcome"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert design["treatment"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert "segment" not in design["covariates"]
    assert "visit" not in design["covariates"]
    assert set(design["covariates"].columns) == {
        "recency",
        "history",
        "mens",
        "womens",
        "newbie",
        "zip_Urban",
        "channel_Web",
    }


def test_hillstrom_design_refuses_missing_columns_and_arm_drift() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        benchmark_rlearner_hillstrom._prepare_hillstrom(_source().drop(columns="visit"))
    drifted = _source().copy()
    drifted.loc[drifted["segment"].eq("Womens E-Mail"), "segment"] = "No E-Mail"
    with pytest.raises(ValueError, match="three declared randomized arms"):
        benchmark_rlearner_hillstrom._prepare_hillstrom(drifted)


def test_hillstrom_benchmark_documented_direct_script_path_loads() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "benchmarks/benchmark_rlearner_hillstrom.py", "--help"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--data" in completed.stdout

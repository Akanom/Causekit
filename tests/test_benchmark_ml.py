"""Contracts for the optional causal-ML benchmark harness."""

from __future__ import annotations

import json
import sys

import pandas as pd

from benchmarks import benchmark_ml


def test_nsw_benchmark_design_uses_declared_pre_treatment_covariates(monkeypatch) -> None:
    source = pd.DataFrame(
        {
            "treat": [0, 1],
            "age": [21, 32],
            "educ": [10, 12],
            "black": [1, 0],
            "hisp": [0, 1],
            "marr": [0, 1],
            "nodegree": [1, 0],
            "re74": [0.0, 1200.0],
            "re75": [250.0, 1800.0],
            "re78": [5000.0, 9000.0],
        }
    )
    calls: list[tuple[str, object, bool]] = []

    def load(name: str, *, data_directory=None, download: bool = False) -> pd.DataFrame:
        calls.append((name, data_directory, download))
        return source

    monkeypatch.setattr(benchmark_ml, "load_real_dataset", load)
    design = benchmark_ml._design(
        "nsw_mixtape",
        data_directory=None,
        download=False,
    )

    assert calls == [("nsw_mixtape", None, False)]
    assert design["design_id"] == "nsw_job_training_earnings_v1"
    assert design["outcome_name"] == "re78"
    assert design["treatment_name"] == "treat"
    assert design["covariate_names"] == (
        "age",
        "educ",
        "black",
        "hisp",
        "marr",
        "nodegree",
        "re74",
        "re75",
    )
    assert design["covariates"].columns.tolist() == list(design["covariate_names"])
    assert design["outcome"].tolist() == [5000.0, 9000.0]
    assert design["treatment"].tolist() == [0.0, 1.0]


def test_cattaneo_remains_the_backward_compatible_default_design() -> None:
    assert benchmark_ml.DEFAULT_DATASET == "cattaneo2"
    assert set(benchmark_ml.DATASET_NAMES) == {"cattaneo2", "nsw_mixtape"}


def test_report_persists_the_selected_dataset_contract(tmp_path, monkeypatch) -> None:
    design = {
        "design_id": "nsw_job_training_earnings_v1",
        "outcome_name": "re78",
        "treatment_name": "treat",
        "covariate_names": ("age", "re74"),
        "covariates": pd.DataFrame({"age": [20.0, 30.0], "re74": [0.0, 1.0]}),
        "treatment": pd.Series([0.0, 1.0]),
        "outcome": pd.Series([2.0, 5.0]),
    }
    output = tmp_path / "report.json"
    monkeypatch.setattr(benchmark_ml, "_design", lambda *args, **kwargs: design)
    monkeypatch.setattr(
        benchmark_ml,
        "_run_once",
        lambda name, supplied: {"model": name, "status": "completed"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_ml.py",
            "--dataset",
            "nsw_mixtape",
            "--models",
            "causekit_native_ridge_gcv",
            "--output",
            str(output),
        ],
    )

    benchmark_ml.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["dataset"] == "nsw_mixtape"
    assert report["design"] == "nsw_job_training_earnings_v1"
    assert report["outcome"] == "re78"
    assert report["treatment"] == "treat"
    assert report["covariates"] == ["age", "re74"]
    assert report["nobs"] == 2
    assert report["benchmark_repetitions"] == 1
    assert report["results"] == [{"model": "causekit_native_ridge_gcv", "status": "completed"}]

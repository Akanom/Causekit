"""Contracts for the optional real-data DR-learner benchmark harness."""

from __future__ import annotations

import json
import sys

import pandas as pd

from benchmarks import benchmark_drlearner


def test_drlearner_benchmark_uses_only_declared_pre_treatment_nsw_covariates(
    monkeypatch,
) -> None:
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

    monkeypatch.setattr(benchmark_drlearner, "load_real_dataset", load)
    design = benchmark_drlearner._design(data_directory=None, download=False)

    assert calls == [("nsw_mixtape", None, False)]
    assert design["covariates"].columns.tolist() == list(benchmark_drlearner.COVARIATES)
    assert design["outcome"].tolist() == [5000.0, 9000.0]
    assert design["treatment"].tolist() == [0.0, 1.0]


def test_drlearner_report_records_one_run_cate_only_contract(tmp_path, monkeypatch) -> None:
    design = {
        "covariates": pd.DataFrame({"age": [20.0, 30.0]}),
        "treatment": pd.Series([0.0, 1.0]),
        "outcome": pd.Series([2.0, 5.0]),
    }
    output = tmp_path / "report.json"
    monkeypatch.setattr(benchmark_drlearner, "_design", lambda **kwargs: design)
    monkeypatch.setattr(
        benchmark_drlearner,
        "_run_once",
        lambda name, supplied: {"model": name, "status": "completed"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_drlearner.py",
            "--models",
            "causekit_native_ridge_gcv",
            "--output",
            str(output),
        ],
    )

    benchmark_drlearner.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["design"] == "nsw_job_training_honest_dr_cate_v1"
    assert report["dataset"] == "nsw_mixtape"
    assert report["nobs"] == 2
    assert report["benchmark_repetitions"] == 1
    assert (
        "only the unweighted DR pseudo-outcome CATE learner changes" in report["comparison_policy"]
    )
    assert "settled R-learner artifacts are not rerun" in report["comparison_policy"]
    assert report["results"] == [{"model": "causekit_native_ridge_gcv", "status": "completed"}]


def test_native_dr_benchmark_keeps_all_three_package_owned_stages() -> None:
    estimator, comparator_version = benchmark_drlearner._estimator("causekit_native_ridge_gcv")

    assert estimator.outcome_factory is None
    assert estimator.propensity_factory is None
    assert estimator.cate_factory is None
    assert comparator_version is None

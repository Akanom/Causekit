"""Independent Statsmodels parity for the DML residual-regression stage."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causekit import PartiallyLinearDML


class _ColumnResult:
    def __init__(self, column: str) -> None:
        self.column = column

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(dtype=float)


class _ColumnRegressor:
    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ColumnResult:
        return _ColumnResult(self.column)


def _fixed_contract():
    outcome_residual = np.array(
        [
            -1.5033333333333334,
            1.3033333333333332,
            -3.3066666666666666,
            3.7066666666666666,
            -2.755,
            3.0549999999999997,
            -0.9516666666666667,
            0.45166666666666666,
        ]
    )
    treatment_residual = np.array([-1.0, 1.0, -2.0, 2.0, -1.5, 1.5, -0.5, 0.5])
    index = pd.RangeIndex(len(outcome_residual))
    covariates = pd.DataFrame(
        {"outcome_mean": np.zeros(len(index)), "treatment_mean": np.zeros(len(index))},
        index=index,
    )
    native = PartiallyLinearDML(
        outcome_factory=lambda: _ColumnRegressor("outcome_mean"),
        treatment_factory=lambda: _ColumnRegressor("treatment_mean"),
        n_splits=2,
        random_state=41,
    ).fit(
        pd.Series(outcome_residual, index=index),
        treatment=pd.Series(treatment_residual, index=index),
        covariates=covariates,
    )
    return native, outcome_residual, treatment_residual


@pytest.mark.validation
def test_dml2_matches_statsmodels_no_intercept_hc1_residual_regression() -> None:
    statsmodels = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(8927)
    nobs = 240
    index = pd.Index([f"observation-{position}" for position in range(nobs)])
    outcome_mean = rng.normal(size=nobs)
    treatment_mean = rng.normal(size=nobs)
    treatment_residual = rng.normal(size=nobs)
    outcome_residual = 1.3 * treatment_residual + rng.normal(size=nobs)
    covariates = pd.DataFrame(
        {"outcome_mean": outcome_mean, "treatment_mean": treatment_mean}, index=index
    )
    result = PartiallyLinearDML(
        outcome_factory=lambda: _ColumnRegressor("outcome_mean"),
        treatment_factory=lambda: _ColumnRegressor("treatment_mean"),
        n_splits=4,
        random_state=19,
    ).fit(
        pd.Series(outcome_mean + outcome_residual, index=index),
        treatment=pd.Series(treatment_mean + treatment_residual, index=index),
        covariates=covariates,
    )

    reference = statsmodels.OLS(outcome_residual, treatment_residual[:, None], hasconst=False).fit(
        cov_type="HC1"
    )

    assert result.estimate == pytest.approx(float(reference.params[0]), rel=2e-14, abs=2e-14)
    assert result.standard_error == pytest.approx(float(reference.bse[0]), rel=2e-13, abs=2e-13)


@pytest.mark.validation
def test_dml2_matches_base_r_matrix_hc1_contract() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is required for base-R DML parity")
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [rscript, str(repository_root / "benchmarks" / "validate_dml_reference.R")],
        check=True,
        capture_output=True,
        text=True,
    )
    reference = dict(
        line.split("=", maxsplit=1) for line in completed.stdout.splitlines() if "=" in line
    )
    native, _, _ = _fixed_contract()

    assert reference["artifact"] == "causekit_partially_linear_dml_r_parity"
    assert reference["causekit_version"] == "0.7.0a1"
    assert reference["r_version"] == "4.5.1"
    assert native.estimate == pytest.approx(float(reference["estimate"]), abs=2e-14)
    assert native.standard_error == pytest.approx(float(reference["standard_error"]), abs=2e-14)


def test_dml_stata_harness_persists_results_before_asserting() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (repository_root / "benchmarks" / "validate_dml_stata.do").read_text(encoding="utf-8")
    assert "regress outcome_residual treatment_residual, noconstant vce(robust)" in script
    assert "file close `results_file'" in script
    assert script.index("file close `results_file'") < script.index(
        'if "`parity_status\'" != "pass"'
    )


@pytest.mark.validation
def test_saved_dml_stata_result_passes_when_user_has_run_harness() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output_path = repository_root / "benchmarks" / "validate_dml_stata_output.txt"
    if not output_path.exists():
        pytest.skip("run benchmarks/validate_dml_stata.do manually in Stata 17")
    reference = dict(
        line.split("=", maxsplit=1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    assert reference["artifact"] == "causekit_partially_linear_dml_stata_parity"
    assert reference["causekit_version"] == "0.7.0a1"
    assert reference["contract"] == "fixed_oof_nuisance_residual_regression_hc1"
    assert reference["stata_version"] == "17"
    assert reference["estimate_status"] == "pass"
    assert reference["standard_error_status"] == "pass"
    assert reference["parity_status"] == "pass"

"""Independent base-R parity for the direct cohort-odds PT-All score."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causekit import CrossFitter, EfficientDiD


class _OddsResult:
    odds_ratio_kind_ = "posterior_cohort_odds"

    def __init__(self, value: float) -> None:
        self.value = value

    def predict_odds_ratio(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=X.index)


class _EmpiricalOdds:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _OddsResult:
        probability = float(y.mean())
        return _OddsResult(probability / (1.0 - probability))


class _MeanResult:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=X.index)


class _MeanRegression:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _MeanResult:
        return _MeanResult(float(y.mean()))


def _panel() -> pd.DataFrame:
    entities = [f"g_{value:02d}" for value in range(12)] + [f"n_{value:02d}" for value in range(6)]
    cohorts = [2.0] * 12 + [np.inf] * 6
    changes = [
        3.0,
        3.5,
        2.5,
        4.0,
        2.0,
        3.2,
        3.8,
        2.7,
        3.3,
        2.2,
        4.1,
        2.9,
        1.0,
        1.5,
        0.5,
        2.0,
        0.0,
        1.2,
    ]
    rows: list[dict[str, float | str]] = []
    for entity, cohort, change in zip(entities, cohorts, changes, strict=True):
        rows.extend(
            [
                {
                    "entity": entity,
                    "time": 1.0,
                    "treatment_time": cohort,
                    "x": 0.0,
                    "outcome": 0.0,
                },
                {
                    "entity": entity,
                    "time": 2.0,
                    "treatment_time": cohort,
                    "x": 0.0,
                    "outcome": change,
                },
            ]
        )
    return pd.DataFrame(rows)


@pytest.mark.validation
def test_direct_ratio_score_and_influence_match_independent_base_r() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is required for direct cohort-odds reference parity")
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [rscript, str(repository_root / "benchmarks/validate_did_direct_ratio_reference.R")],
        check=True,
        capture_output=True,
        text=True,
    )
    reference = dict(
        line.split("=", maxsplit=1) for line in completed.stdout.splitlines() if "=" in line
    )
    assert reference["contract"] == "direct_posterior_cohort_odds_two_period_pt_all"
    assert reference["parity_status"] == "reference_completed"

    native = EfficientDiD().fit(
        _panel(),
        outcome="outcome",
        entity="entity",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=CrossFitter(
            cohort_ratio_factory=_EmpiricalOdds,
            outcome_factory=_MeanRegression,
            n_splits=3,
            random_state=17,
        ),
    )
    expected_fold = np.fromstring(reference["fold"], sep=",", dtype=int)
    expected_ratio = np.fromstring(reference["ratio"], sep=",")
    expected_influence = np.fromstring(reference["influence"], sep=",")
    np.testing.assert_array_equal(native.nuisance_fold.to_numpy(), expected_fold)
    np.testing.assert_allclose(
        native.cohort_ratios[(2.0, np.inf)], expected_ratio, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        native.candidate_influence_functions.iloc[:, 0],
        expected_influence,
        rtol=0,
        atol=2e-14,
    )
    assert native.estimate == pytest.approx(float(reference["estimate"]), abs=2e-14)
    assert native.standard_error == pytest.approx(float(reference["standard_error_hc1"]), abs=2e-14)

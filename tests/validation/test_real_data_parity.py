"""Cross-language parity on pinned, hash-verified real datasets."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from benchmarks.prepare_real_data import prepare_real_data
from causekit import (
    AIPWATE,
    IPWATE,
    IV2SLS,
    DifferenceInDifferences,
    EfficientDiD,
    NearestNeighborMatch,
    RandomizedATE,
)


def test_real_data_stata_harness_is_persistent_and_stata_17_safe() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (repository_root / "benchmarks" / "validate_real_data_stata.do").read_text(
        encoding="utf-8"
    )
    assert "ivregress 2sls" in script
    assert script.count("teffects nnmatch") == 3
    assert "quietly logit mbsmoke" in script
    assert "did_overall_i" in script
    assert "unavailable_no_aligned_estimator" in script
    assert "file close `results_file'" in script
    assert script.index("file close `results_file'") < script.index(
        'if "`parity_status\'" != "pass"'
    )
    declarations = re.findall(
        r"(?m)^(?:generate(?:\s+\w+)?|local|matrix|scalar|tempname)\s+([A-Za-z_]\w*)",
        script,
    )
    assert [name for name in declarations if len(name) > 32] == []


def test_saved_real_data_stata_result_records_all_comparable_passes() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output_path = repository_root / "benchmarks" / "validate_real_data_stata_output.txt"
    if not output_path.exists():
        pytest.skip("run benchmarks/validate_real_data_stata.do manually in Stata 17")
    reference = dict(
        line.split("=", maxsplit=1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    assert reference["artifact"] == "causekit_real_data_stata_parity"
    assert reference["causekit_version"] == "0.6.0a4"
    assert reference["stata_version"] == "17"
    assert reference["data_schema"] == "causekit_real_data_parity_v1"
    for family in ("iv", "randomized", "observational", "matching", "did"):
        assert reference[f"{family}_status"] == "pass"
        tolerance_key = (
            "linear_tolerance"
            if family in {"iv", "randomized"}
            else "nuisance_tolerance"
            if family == "observational"
            else "matching_tolerance"
            if family == "matching"
            else "did_tolerance"
        )
        assert float(reference[f"{family}_maximum_absolute_difference"]) <= float(
            reference[tolerance_key]
        )
    assert reference["efficient_did_stata_status"] == "unavailable_no_aligned_estimator"
    assert reference["parity_status"] == "pass"


@pytest.fixture(scope="module")
def real_reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, str], Path]:
    source_root = os.environ.get("CAUSEKIT_REAL_DATA_DIR")
    edid_root = os.environ.get("CAUSEKIT_EDID_REFERENCE")
    matching_root = os.environ.get("CAUSEKIT_MATCHING_REFERENCE")
    rscript = shutil.which("Rscript")
    if None in {source_root, edid_root, matching_root} or rscript is None:
        pytest.skip(
            "set CAUSEKIT_REAL_DATA_DIR, CAUSEKIT_EDID_REFERENCE, and "
            "CAUSEKIT_MATCHING_REFERENCE with Rscript available"
        )
    output_root = tmp_path_factory.mktemp("real-data-parity")
    manifest_path = prepare_real_data(
        source_directory=source_root,
        output_directory=output_root,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "causekit_real_data_parity_v1"
    for entry in manifest["datasets"].values():
        assert entry["source_sha256"] == entry["source_verified_sha256"]

    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            rscript,
            str(repository_root / "benchmarks" / "validate_real_data_reference.R"),
            str(output_root),
            str(edid_root),
            str(matching_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    reference = dict(
        line.split("=", maxsplit=1) for line in completed.stdout.splitlines() if "=" in line
    )
    return reference, output_root


def _values(reference: dict[str, str], key: str) -> np.ndarray:
    return np.fromstring(reference[key], sep=",")


@pytest.mark.validation
def test_real_housing_iv_matches_r_contract(real_reference: tuple[dict[str, str], Path]) -> None:
    reference, root = real_reference
    data = pd.read_csv(root / "hsng.csv")
    region = pd.get_dummies(
        data["region"].astype(int), prefix="region", drop_first=True, dtype=float
    )
    instruments = pd.concat([data[["faminc"]].astype(float), region], axis=1)
    native = IV2SLS(covariance="robust").fit(
        data["rent"].astype(float),
        endogenous=data[["hsngval"]].astype(float),
        instruments=instruments,
        exogenous=data[["pcturban"]].astype(float),
    )

    for name in native.params.index:
        assert native.params[name] == pytest.approx(
            float(reference[f"iv_{name}_estimate"]), abs=1e-7
        )
        assert native.standard_errors[name] == pytest.approx(
            float(reference[f"iv_{name}_standard_error"]), abs=1e-6
        )


@pytest.mark.validation
@pytest.mark.parametrize("adjustment", ["none", "lin"])
def test_real_nsw_randomized_effect_matches_r_hc1(
    real_reference: tuple[dict[str, str], Path], adjustment: str
) -> None:
    reference, root = real_reference
    data = pd.read_csv(root / "nsw_mixtape.csv")
    covariates = data[["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]].astype(
        float
    )
    native = RandomizedATE(adjustment=adjustment).fit(
        data["re78"].astype(float),
        treatment=data["treat"].astype(int),
        covariates=covariates if adjustment == "lin" else None,
    )
    label = "lin" if adjustment == "lin" else "raw"
    assert native.estimate == pytest.approx(
        float(reference[f"randomized_{label}_ate_estimate"]), abs=1e-7
    )
    assert native.standard_error == pytest.approx(
        float(reference[f"randomized_{label}_ate_standard_error"]), abs=1e-6
    )


@lru_cache(maxsize=1)
def _observational_inputs(root: Path):
    data = pd.read_csv(root / "cattaneo2.csv")
    design = sm.add_constant(
        data[["mmarried", "mage", "medu", "fbaby"]].astype(float),
        has_constant="add",
    )
    treatment = data["mbsmoke"].astype(int)
    outcome = data["bweight"].astype(float)
    propensity_fit = sm.Logit(treatment, design).fit(
        method="newton", maxiter=200, tol=1e-12, disp=False
    )
    propensity = pd.Series(propensity_fit.predict(design), index=data.index)
    outcome_predictions: dict[int, pd.Series] = {}
    for arm in (0, 1):
        fitted = sm.OLS(outcome[treatment == arm], design.loc[treatment == arm]).fit()
        outcome_predictions[arm] = pd.Series(fitted.predict(design), index=data.index)
    return data, outcome, treatment, propensity, outcome_predictions


@pytest.mark.validation
@pytest.mark.parametrize("estimator", ["ipw", "aipw"])
@pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
def test_real_observational_effect_matches_r_supplied_nuisance_contract(
    real_reference: tuple[dict[str, str], Path], estimator: str, estimand: str
) -> None:
    reference, root = real_reference
    _, outcome, treatment, propensity, predictions = _observational_inputs(root)
    if estimator == "ipw":
        native = IPWATE(estimand=estimand).fit(
            outcome,
            treatment=treatment,
            propensity=propensity,
        )
    else:
        native = AIPWATE(estimand=estimand).fit(
            outcome,
            treatment=treatment,
            propensity=propensity,
            outcome_treated=predictions[1],
            outcome_control=predictions[0],
        )
    prefix = f"{estimator}_{estimand}"
    # R glm and statsmodels Logit stop at slightly different score tolerances. The
    # The 2e-5-gram absolute bound is two hundredths of one milligram.
    assert native.estimate == pytest.approx(float(reference[f"{prefix}_estimate"]), abs=2e-5)
    assert native.standard_error == pytest.approx(
        float(reference[f"{prefix}_standard_error"]), abs=2e-6
    )


@pytest.mark.validation
@pytest.mark.parametrize("estimand", ["att", "ate", "atc"])
def test_real_matching_point_estimate_matches_pinned_r_matching(
    real_reference: tuple[dict[str, str], Path], estimand: str
) -> None:
    reference, root = real_reference
    data, outcome, treatment, propensity, _ = _observational_inputs(root)
    treated = np.flatnonzero(treatment.to_numpy() == 1)[:200]
    controls = np.flatnonzero(treatment.to_numpy() == 0)[:400]
    selected = np.sort(np.concatenate([treated, controls]))
    assert len(selected) == int(reference["matching_real_subset_rows"])
    native = NearestNeighborMatch(
        estimand=estimand,
        metric="propensity",
        caliper=None,
        common_support=None,
        inference="none",
    ).fit(
        outcome.iloc[selected].reset_index(drop=True),
        treatment=treatment.iloc[selected].reset_index(drop=True),
        propensity=propensity.iloc[selected].reset_index(drop=True),
        propensity_score_status="estimated",
        propensity_provenance="real_cattaneo_full_sample_logit",
    )
    assert data.iloc[selected]["source_row"].is_monotonic_increasing
    assert native.estimate == pytest.approx(
        float(reference[f"matching_{estimand}_estimate"]), abs=1e-9
    )


def _hospital_panel(root: Path) -> pd.DataFrame:
    panel = pd.read_csv(root / "hospdd_panel.csv")
    panel["treatment_time"] = panel["treatment_time"].replace({0: np.inf})
    return panel


@pytest.mark.validation
def test_real_conventional_did_matches_r_group_time_contract(
    real_reference: tuple[dict[str, str], Path],
) -> None:
    reference, root = real_reference
    native = DifferenceInDifferences().fit(
        _hospital_panel(root),
        outcome="outcome",
        entity="hospital",
        time="month",
        treatment_time="treatment_time",
    )
    np.testing.assert_allclose(
        native.group_time["att"].to_numpy(),
        _values(reference, "did_group_time_att"),
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        native.group_time["std_err"].to_numpy(),
        _values(reference, "did_group_time_standard_error"),
        rtol=0,
        atol=2e-14,
    )
    assert native.estimate == pytest.approx(float(reference["did_esavg_estimate"]), abs=2e-14)
    assert native.standard_error == pytest.approx(
        float(reference["did_esavg_standard_error"]), abs=2e-14
    )


@pytest.mark.validation
def test_real_efficient_did_matches_pinned_public_r_edid(
    real_reference: tuple[dict[str, str], Path],
) -> None:
    reference, root = real_reference
    assert reference["artifact"] == "causekit_real_data_r_parity"
    assert reference["r_version"] == "4.5.1"
    assert reference["edid_commit"] == "f55a4a4aba14f0826f59ad7aa4af3bafaeba529b"
    assert reference["matching_commit"] == "1208eaa7bfa888b1fc903481dddfb8c0dffa40d5"
    assert reference["parity_status"] == "reference_completed"
    native = EfficientDiD(pre_periods="all").fit(
        _hospital_panel(root),
        outcome="outcome",
        entity="hospital",
        time="month",
        treatment_time="treatment_time",
    )
    np.testing.assert_allclose(
        native.group_time["att"].to_numpy(),
        _values(reference, "edid_group_time_att"),
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        native.group_time["std_err"].to_numpy(),
        _values(reference, "edid_group_time_standard_error"),
        rtol=0,
        atol=2e-14,
    )
    assert native.estimate == pytest.approx(float(reference["edid_esavg_estimate"]), abs=2e-14)
    assert native.standard_error == pytest.approx(
        float(reference["edid_esavg_standard_error"]), abs=2e-14
    )

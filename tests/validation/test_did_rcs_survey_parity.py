"""R survey and manually reviewed Stata parity for survey-population RCS DiD."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causekit import RepeatedCrossSectionDiD, RepeatedCrossSectionSurveyDesign

ROOT = Path(__file__).resolve().parents[2]
R_SCRIPT = ROOT / "benchmarks" / "validate_did_rcs_survey_reference.R"
R_OUTPUT = ROOT / "benchmarks" / "validate_did_rcs_survey_reference_output.txt"
STATA_SCRIPT = ROOT / "benchmarks" / "validate_did_rcs_survey_stata.do"
STATA_OUTPUT = ROOT / "benchmarks" / "validate_did_rcs_survey_stata_output.txt"
R_OUTPUT_SHA256 = "9a493c94299895704215636fa293b8ef31b805869ea8ffd054e7c63933726d56"
STATA_OUTPUT_SHA256 = "e75e9737567923fa6203c9f6eb0a2f017581479fbf0c3aac61b687e5da0e0a1f"


def _parse(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", maxsplit=1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def _native():
    data = pd.DataFrame(
        {
            "outcome": [5, 7, 6, 8, 9, 12, 10, 13, 3, 4, 2, 5, 4, 7, 3, 6],
            "time": np.repeat([1.0, 2.0, 1.0, 2.0], 4),
            "treatment_time": np.repeat([2.0, 2.0, np.inf, np.inf], 4),
            "weight": [1, 2, 1, 3, 2, 1, 3, 1, 1, 2, 2, 1, 2, 1, 1, 2],
            "psu": np.tile(["a1", "a2", "b1", "b2"], 4),
            "stratum": np.tile(["a", "a", "b", "b"], 4),
        }
    )
    return RepeatedCrossSectionDiD().fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        survey_design=RepeatedCrossSectionSurveyDesign(
            weights="weight", psu="psu", strata="stratum"
        ),
        target_population="survey_population",
    )


def _assert_r_parity(reference: dict[str, str]) -> None:
    result = _native()
    assert reference["contract"] == "survey_rcs_component_hajek_stratified_psu_taylor"
    assert reference["r_version"] == "4.5.1"
    assert reference["survey_version"] == "4.5"
    assert reference["parity_status"] == "pass"
    assert float(reference["estimate"]) == pytest.approx(result.estimate, abs=1e-14)
    assert float(reference["standard_error"]) == pytest.approx(result.standard_error, abs=1e-14)
    assert float(reference["survey_estimate"]) == pytest.approx(result.estimate, abs=1e-14)
    assert float(reference["survey_standard_error"]) == pytest.approx(
        result.standard_error, abs=3e-14
    )
    assert int(reference["design_df"]) == result.inference_df
    np.testing.assert_allclose(
        np.fromstring(reference["linearized"], sep=","),
        result.survey_linearized["esavg"],
        rtol=0.0,
        atol=1e-14,
    )


def test_r_survey_reference_is_hash_pinned_and_passes() -> None:
    assert hashlib.sha256(R_OUTPUT.read_bytes()).hexdigest() == R_OUTPUT_SHA256
    source = R_SCRIPT.read_text(encoding="utf-8")
    assert "survey::svydesign" in source
    assert "survey::svytotal" in source
    assert "survey::svycontrast" in source
    assert "survey::degf" in source
    _assert_r_parity(_parse(R_OUTPUT))


@pytest.mark.validation
def test_live_r_survey_reference_when_requested() -> None:
    if os.environ.get("CAUSEKIT_LIVE_R_SURVEY") != "1" or shutil.which("Rscript") is None:
        pytest.skip("set CAUSEKIT_LIVE_R_SURVEY=1 with R survey 4.5 installed")
    subprocess.run(["Rscript", str(R_SCRIPT)], cwd=ROOT, check=True)
    _assert_r_parity(_parse(R_OUTPUT))


def test_stata_survey_harness_writes_before_asserting_and_uses_svy() -> None:
    source = STATA_SCRIPT.read_text(encoding="utf-8")
    assert "svyset psu_id [pweight=survey_weight], strata(stratum_id)" in source
    assert "quietly svy: total" in source
    assert "quietly nlcom" in source
    assert source.index("scalar design_df = e(df_r)") < source.index("quietly nlcom")
    assert "scalar nlcom_se_tolerance = 1e-9" in source
    assert "collapse (sum) psu_total=linearized, by(stratum_id psu_id)" in source
    assert source.index("file close `results_file'") < source.index(
        'if "`parity_status\'" != "pass"'
    )
    declarations = re.findall(
        r"(?m)^(?:generate(?:\s+\w+)?|local|matrix|scalar|tempname)\s+([A-Za-z_]\w*)",
        source,
    )
    assert [name for name in declarations if len(name) > 32] == []


@pytest.mark.validation
def test_saved_stata_survey_parity_when_reviewed() -> None:
    if not STATA_OUTPUT.exists():
        pytest.skip("run benchmarks/validate_did_rcs_survey_stata.do manually in Stata 17")
    assert hashlib.sha256(STATA_OUTPUT.read_bytes()).hexdigest() == STATA_OUTPUT_SHA256
    reference = _parse(STATA_OUTPUT)
    result = _native()
    assert reference["contract"] == "survey_rcs_component_hajek_stratified_psu_taylor"
    assert reference["stata_version"] == "17"
    assert float(reference["estimate"]) == pytest.approx(result.estimate, abs=1e-12)
    assert float(reference["standard_error"]) == pytest.approx(result.standard_error, abs=1e-12)
    assert float(reference["survey_estimate"]) == pytest.approx(result.estimate, abs=1e-12)
    assert float(reference["survey_standard_error"]) == pytest.approx(
        result.standard_error, abs=1e-9
    )
    assert float(reference["exact_tolerance"]) == 1e-12
    assert float(reference["nlcom_se_tolerance"]) == 1e-9
    assert float(reference["survey_standard_error_absolute_difference"]) <= 1e-9
    assert float(reference["design_df"]) == result.inference_df
    assert reference["parity_status"] == "pass"

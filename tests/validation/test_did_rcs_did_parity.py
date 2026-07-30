"""Estimator-level repeated-cross-section parity against pinned R ``did`` 2.5.0."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causekit import RepeatedCrossSectionDiD

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPOSITORY_ROOT / "benchmarks" / "did_rcs_parity_input.csv"
OUTPUT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_did_output.txt"
SCRIPT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_did_reference.R"
REFERENCE_COMMIT = "c449b8ce72029855d2de94b377f131be0e53e53a"


def _parse_reference(text: str) -> dict[str, list[list[str]]]:
    parsed: dict[str, list[list[str]]] = defaultdict(list)
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        parsed[key].append(value.split(","))
    return parsed


def _native_results() -> dict[str, object]:
    source = pd.read_csv(INPUT_PATH)
    data = source.loc[source.index.repeat(3)].reset_index(drop=True)
    data["treatment_time"] = data["treatment_time"].replace(0.0, np.inf)
    return {
        control: RepeatedCrossSectionDiD(control_group=control).fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
        )
        for control in ("never_treated", "not_yet_treated")
    }


def _assert_parity(reference: dict[str, list[list[str]]]) -> None:
    assert reference["reference_commit"] == [[REFERENCE_COMMIT]]
    assert reference["did_version"] == [["2.5.0"]]
    assert reference["source_observations"] == [["18"]]
    assert reference["deterministic_replications"] == [["3"]]
    assert reference["n_observations"] == [["54"]]
    assert reference["parity_status"] == [["reference_complete"]]

    native = _native_results()
    for control, cohort, time, estimate, _raw_se, adjusted_se in reference["group_time"]:
        result = native[control]
        row = result.group_time.loc[(float(cohort), float(time))]
        assert row["att"] == pytest.approx(float(estimate), abs=2e-14)
        assert row["std_err"] == pytest.approx(float(adjusted_se), rel=2e-13, abs=2e-14)
    for control, event, estimate, _raw_se, adjusted_se in reference["event_time"]:
        result = native[control]
        row = result.event_study.loc[int(float(event))]
        assert row["att"] == pytest.approx(float(estimate), abs=2e-14)
        assert row["std_err"] == pytest.approx(float(adjusted_se), rel=2e-13, abs=2e-14)
    for control, time, estimate, _raw_se, adjusted_se in reference["calendar_time"]:
        result = native[control]
        row = result.calendar_time.loc[float(time)]
        assert row["att"] == pytest.approx(float(estimate), abs=2e-14)
        assert row["std_err"] == pytest.approx(float(adjusted_se), rel=2e-13, abs=2e-14)
    for control, estimate, _raw_se, adjusted_se in reference["esavg"]:
        result = native[control]
        assert result.estimate == pytest.approx(float(estimate), abs=2e-14)
        assert result.standard_error == pytest.approx(float(adjusted_se), rel=2e-13, abs=2e-14)


def test_r_did_comparator_pins_source_and_records_covariance_mapping() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert REFERENCE_COMMIT in script
    assert 'as.character(utils::packageVersion("did")) != "2.5.0"' in script
    assert "panel = FALSE" in script
    assert 'est_method = "reg"' in script
    assert "sqrt(n / (n - 1))" in script
    assert "each = 3" in script


@pytest.mark.validation
def test_saved_r_did_estimator_level_parity_passes() -> None:
    _assert_parity(_parse_reference(OUTPUT_PATH.read_text(encoding="utf-8")))


@pytest.mark.validation
def test_live_r_did_estimator_level_parity_when_configured() -> None:
    reference_root = os.environ.get("CAUSEKIT_DID_REFERENCE")
    rscript = shutil.which("Rscript")
    if reference_root is None or rscript is None:
        pytest.skip("set CAUSEKIT_DID_REFERENCE and install Rscript for live R did parity")
    completed = subprocess.run(
        [rscript, str(SCRIPT_PATH), reference_root],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    _assert_parity(_parse_reference(completed.stdout))

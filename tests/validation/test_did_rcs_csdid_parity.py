"""Static and saved-output gates for manual Stata ``csdid`` parity."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_csdid.do"
OUTPUT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_csdid_output.txt"


def test_csdid_harness_is_repeated_sample_pinned_and_writes_before_asserting() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "time(time) gvar(treatment_time) method(reg) long2" in script
    csdid_calls = re.findall(r"(?m)^quietly csdid .+$", script)
    assert len(csdid_calls) == 2
    assert all("ivar(" not in call for call in csdid_calls)
    assert "notyet" in script
    assert "738defceb5413face7051754b259c82798c67e52" in script
    assert 'aggregate_se_status "non_comparable_period_specific_cell_share_influence"' in script
    assert "file close `results_file'" in script
    assert script.rindex("file close `results_file'") < script.index(
        'if "`parity_status\'" != "pass"'
    )
    declarations = re.findall(
        r"(?m)^(?:generate(?:\s+\w+)?|local|matrix|scalar|tempname)\s+([A-Za-z_]\w*)",
        script,
    )
    assert [name for name in declarations if len(name) > 32] == []


@pytest.mark.validation
def test_saved_csdid_estimator_level_result_passes() -> None:
    if not OUTPUT_PATH.exists():
        pytest.skip("run benchmarks/validate_did_rcs_csdid.do manually in Stata 17+")
    content = OUTPUT_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        "ad250fa9cdfd042e1989460ebcec6b1bac57ed301fae390f2c69331ccf47fd57"
    )
    reference = dict(
        line.split("=", maxsplit=1) for line in content.decode("utf-8").splitlines() if "=" in line
    )

    assert reference["contract"] == "csdid_repeated_cross_section_reg_long2"
    assert reference["method"] == "reg"
    assert reference["panel"] == "false"
    assert reference["base_period"] == "long2"
    assert reference["csdid_available"] == "true"
    assert reference["drdid_available"] == "true"
    assert int(reference["n_observations"]) == 54
    assert float(reference["max_estimate_difference"]) <= float(reference["tolerance"])
    assert reference["estimate_status"] == "pass"
    assert reference["se_convention"] in {
        "did_2.5_analytic",
        "causekit_observation_hc1",
    }
    assert reference["group_time_se_status"] == "pass"
    assert reference["aggregate_se_status"] == (
        "non_comparable_period_specific_cell_share_influence"
    )
    assert reference["parity_scope"] == (
        "group_time_estimates_and_standard_errors_plus_aggregate_points"
    )
    assert reference["parity_status"] == "pass"

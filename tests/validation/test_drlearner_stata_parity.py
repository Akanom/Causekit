"""Static and saved-output gates for manual Stata DR-evaluation parity."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest


def test_drlearner_stata_harness_persists_results_and_respects_name_limits() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (repository_root / "benchmarks" / "validate_drlearner_stata.do").read_text(
        encoding="utf-8"
    )
    assert 'import delimited using "`input_path\'", clear varnames(1) asdouble' in script
    assert "regress dr_score cal_level_x cal_hetero_x, noconstant vce(robust)" in script
    assert "regress dr_score group_x1 group_x2 group_x3, noconstant vce(robust)" in script
    assert "file close `results_file'" in script
    assert script.index("file close `results_file'") < script.index(
        'if "`parity_status\'" != "pass"'
    )
    declarations = re.findall(
        r"(?m)^(?:generate(?:\s+\w+)?|local|matrix|scalar|tempname)\s+([A-Za-z_]\w*)",
        script,
    )
    assert [name for name in declarations if len(name) > 32] == []


@pytest.mark.validation
def test_saved_drlearner_stata_result_passes_when_user_has_run_harness() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output_path = repository_root / "benchmarks" / "validate_drlearner_stata_output.txt"
    if not output_path.exists():
        pytest.skip("run benchmarks/validate_drlearner_stata.do manually in Stata 17")
    reference = dict(
        line.split("=", maxsplit=1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    assert reference["contract"] == "fixed_honest_drlearner_evaluation_hc1"
    assert reference["stata_version"] == "17"
    assert reference["parity_status"] == "pass"
    fixture = pd.read_csv(repository_root / "benchmarks" / "drlearner_stata_input.csv")
    first = fixture.iloc[0]
    expected = {
        "honest_dr_loss": first["exp_dr_loss"],
        "honest_constant_dr_loss": first["exp_const_loss"],
        "dr_loss_gain": first["exp_gain"],
        "calibration_center": first["exp_cal_center"],
        "calibration_level": first["exp_cal_level"],
        "calibration_heterogeneity": first["exp_cal_hetero"],
        "calibration_level_variance": first["exp_cal_v_level"],
        "calibration_heterogeneity_variance": first["exp_cal_v_hetero"],
        "calibration_covariance": first["exp_cal_cov"],
    }
    for group in range(1, 4):
        expected[f"group_{group}_effect"] = first[f"exp_g{group}"]
        expected[f"group_{group}_variance"] = first[f"exp_g{group}_var"]
    tolerance = float(reference["tolerance"])
    for name, target in expected.items():
        assert float(reference[name]) == pytest.approx(float(target), abs=tolerance)

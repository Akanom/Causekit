"""Static and saved-output gates for manual Stata repeated-sample DiD parity."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest


def test_repeated_cross_section_stata_harness_writes_before_asserting() -> None:
    repository = Path(__file__).resolve().parents[2]
    script = (repository / "benchmarks" / "validate_did_rcs_stata.do").read_text(encoding="utf-8")

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
def test_saved_repeated_cross_section_stata_result_passes() -> None:
    repository = Path(__file__).resolve().parents[2]
    output_path = repository / "benchmarks" / "validate_did_rcs_stata_output.txt"
    if not output_path.exists():
        pytest.skip("run benchmarks/validate_did_rcs_stata.do manually in Stata 17")
    content = output_path.read_bytes()
    reference = dict(
        line.split("=", maxsplit=1) for line in content.decode("utf-8").splitlines() if "=" in line
    )

    assert hashlib.sha256(content).hexdigest() == (
        "67cd686b409379a7dbcc58b8172d1defa6a132bb716458dfd0b0217d47288d95"
    )
    assert reference["contract"] == "repeated_cross_section_2x2_observation_hc1"
    assert reference["stata_version"] == "17"
    assert float(reference["estimate"]) == pytest.approx(4.0, abs=1e-12)
    assert float(reference["standard_error_hc1"]) == pytest.approx(
        float(np.sqrt(128.0 / 56.0)), abs=1e-12
    )
    assert reference["estimate_status"] == "pass"
    assert reference["standard_error_status"] == "pass"
    assert reference["parity_status"] == "pass"

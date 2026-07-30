"""Publication-scale repeated-cross-section DiD promotion certificate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_promotion.py"
OUTPUT_PATH = REPOSITORY_ROOT / "benchmarks" / "did_rcs_promotion_evidence.json"


def _evidence() -> dict[str, object]:
    return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))


def test_promotion_runner_keeps_stationary_composition_and_control_contracts_explicit() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert '"never_treated", "not_yet_treated"' in source
    assert "COHORT_PROBABILITIES" in source
    assert 'covariance="clustered"' in source
    assert 'cluster="hospital"' in source
    assert "ordinary_bootstrap" not in source
    assert "recursive-include benchmarks *.py *.R *.do *.txt *.csv *.json" in manifest


@pytest.mark.validation
def test_saved_coverage_certificate_passes_every_declared_gate() -> None:
    evidence = _evidence()
    simulation = evidence["coverage_simulation"]

    assert evidence["schema_version"] == 1
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["generator"] == "benchmarks/validate_did_rcs_promotion.py"
    assert evidence["generator_sha256"] == hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
    assert evidence["overall_status"] == "pass"
    assert simulation["replications_per_design"] == 1_000
    assert simulation["total_estimator_fits"] == 4_000
    assert simulation["nominal_coverage"] == pytest.approx(0.95, abs=0.0)
    assert simulation["status"] == "pass"

    cells = simulation["cells"]
    expected = {
        (design, control, target)
        for design in ("balanced", "unequal_period_sizes")
        for control in ("never_treated", "not_yet_treated")
        for target in (
            "group_3_3",
            "group_3_4",
            "group_4_4",
            "event_0",
            "event_1",
            "calendar_3",
            "calendar_4",
            "esavg",
        )
    }
    assert {(cell["design"], cell["control_group"], cell["target"]) for cell in cells} == expected
    for cell in cells:
        assert cell["replications"] == 1_000
        assert cell["refusals"] == 0
        assert cell["coverage_mcse"] <= 0.011
        assert cell["coverage_gate"] == [0.90, 0.99]
        assert cell["standard_error_ratio_gate"] == [0.75, 1.25]
        assert abs(cell["bias"]) <= cell["absolute_bias_gate"]
        assert cell["status"] == "pass"


@pytest.mark.validation
def test_saved_hospdd_application_is_hash_pinned_and_clustered() -> None:
    real_data = _evidence()["real_data_application"]
    repeated = real_data["rows"]["repeated_cross_section_clustered"]

    assert real_data["status"] == "pass"
    assert real_data["dataset"] == "hospdd"
    assert real_data["source_label"] == "artificial hospital procedure data"
    assert real_data["source_sha256"] == (
        "e3ae6451e89cb915c546ab772410046726f280ad7d117611376beb4f46a521bb"
    )
    assert repeated["nobs"] == 7_368
    assert repeated["n_clusters"] == 46
    assert repeated["estimate"] == pytest.approx(0.867369404129359, abs=1e-14)
    assert repeated["standard_error"] == pytest.approx(0.04233682935122793, abs=1e-14)

"""Publication-scale covariate repeated-cross-section DiD certificate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_covariate_promotion.py"
OUTPUT_PATH = REPOSITORY_ROOT / "benchmarks" / "did_rcs_covariate_promotion_evidence.json"


def _evidence() -> dict[str, object]:
    return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))


def test_runner_freezes_cross_fitting_overlap_and_analytic_inference_contracts() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert '"never_treated", "not_yet_treated"' in source
    assert 'covariates=["x"]' in source
    assert "CrossFitter(" in source
    assert "_BinarySaturatedProbability" in source
    assert "_BinarySaturatedOutcome" in source
    assert 'inference="analytic"' in source
    assert "n_splits=2" in source
    assert "ordinary_bootstrap" not in source
    assert "recursive-include benchmarks *.py *.R *.do *.txt *.csv *.json" in manifest


@pytest.mark.validation
def test_saved_covariate_coverage_certificate_passes_all_44_effect_cells() -> None:
    evidence = _evidence()
    simulation = evidence["coverage_simulation"]

    assert evidence["schema_version"] == 1
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["generator"] == "benchmarks/validate_did_rcs_covariate_promotion.py"
    assert evidence["generator_sha256"] == hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
    assert evidence["overall_status"] == "pass"
    assert simulation["replications_per_design"] == 1_000
    assert simulation["total_estimator_fits"] == 4_000
    assert simulation["total_nuisance_fold_fits"] == 240_000
    assert simulation["nominal_coverage"] == pytest.approx(0.95, abs=0.0)
    assert simulation["designs"]["stressed_overlap_unequal"]["period_sizes"] == [
        3_600,
        2_400,
        1_800,
        1_440,
    ]
    assert simulation["refusal_reasons"] == {}
    assert simulation["status"] == "pass"

    targets = (
        "group_3_3",
        "group_3_4",
        "group_4_4",
        "event_0",
        "event_1",
        "calendar_3",
        "calendar_4",
        "esavg",
        "placebo_3_2",
        "placebo_4_2",
        "placebo_4_3",
    )
    expected = {
        (design, control, target)
        for design in ("favorable_balanced", "stressed_overlap_unequal")
        for control in ("never_treated", "not_yet_treated")
        for target in targets
    }
    cells = simulation["cells"]
    assert len(cells) == 44
    assert {(cell["design"], cell["control_group"], cell["target"]) for cell in cells} == expected
    for cell in cells:
        assert cell["replications"] == 1_000
        assert cell["refusals"] == 0
        assert cell["coverage_mcse"] <= 0.011
        assert cell["coverage_gate"] == [0.90, 0.99]
        assert cell["standard_error_ratio_gate"] == [0.75, 1.25]
        assert abs(cell["bias"]) <= cell["absolute_bias_gate"]
        assert cell["minimum_nuisance_probability"] > 1e-6
        assert cell["maximum_nuisance_probability"] < 1.0 - 1e-6
        assert cell["nuisance_fold_fits_per_estimator"] == 60
        assert cell["status"] == "pass"


@pytest.mark.validation
def test_saved_conditional_pretrend_joint_size_certificate_passes() -> None:
    simulation = _evidence()["coverage_simulation"]
    cells = simulation["conditional_pretrend_joint_size"]

    assert len(cells) == 4
    for cell in cells:
        assert cell["replications"] == 1_000
        assert cell["unavailable"] == 0
        assert cell["refusals"] == 0
        assert cell["nominal_size"] == pytest.approx(0.05, abs=0.0)
        assert cell["rejection_rate_gate"] == [0.01, 0.10]
        assert cell["rejection_rate_mcse"] <= 0.011
        assert cell["status"] == "pass"

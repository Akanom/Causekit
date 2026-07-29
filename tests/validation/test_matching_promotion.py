"""Publication-scale matching inference and real-data sensitivity certificate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_matching_promotion.py"
OUTPUT_PATH = REPOSITORY_ROOT / "benchmarks" / "matching_promotion_evidence.json"


def _evidence() -> dict[str, object]:
    return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))


def test_matching_promotion_runner_keeps_inference_and_sensitivity_contracts_separate() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert 'inference="abadie_imbens"' in source
    assert 'inference="abadie_imbens_estimated"' in source
    assert 'inference="none"' in source
    assert "ordinary_bootstrap" not in source
    assert "common_support=None" in source
    assert 'propensity_score_status="known"' in source
    assert 'propensity_score_status="estimated"' in source
    assert "recursive-include benchmarks *.py *.R *.do *.txt *.csv *.json" in manifest


@pytest.mark.validation
def test_saved_matching_promotion_coverage_certificate_passes_declared_gates() -> None:
    evidence = _evidence()
    simulation = evidence["coverage_simulation"]

    assert evidence["schema_version"] == 1
    assert evidence["causekit_version"] == "0.7.0a4"
    assert evidence["generator"] == "benchmarks/validate_matching_promotion.py"
    assert evidence["generator_sha256"] == hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
    assert evidence["overall_status"] == "pass"
    assert simulation["replications_per_overlap"] == 1_000
    assert simulation["nobs_per_replication"] == 500
    assert simulation["nominal_coverage"] == pytest.approx(0.95, abs=0.0)
    assert simulation["status"] == "pass"

    cells = simulation["cells"]
    expected = {
        (overlap, contract, estimand)
        for overlap in ("favorable", "stressed")
        for contract in ("known_score", "estimated_logit")
        for estimand in ("att", "atc", "ate")
    }
    observed = {(cell["overlap"], cell["contract"], cell["estimand"]) for cell in cells}
    assert observed == expected
    for cell in cells:
        assert cell["replications"] == 1_000
        assert cell["refusals"] == 0
        assert cell["coverage_mcse"] <= 0.011
        assert cell["status"] == "pass"
        assert cell["coverage_gate"] == [0.90, 0.99]
        assert cell["standard_error_ratio_gate"] == [0.75, 1.25]
        assert abs(cell["bias"]) <= cell["absolute_bias_gate"]


@pytest.mark.validation
def test_saved_real_data_sensitivity_certificate_tracks_target_and_balance() -> None:
    evidence = _evidence()
    sensitivity = evidence["real_data_sensitivity"]

    assert sensitivity["status"] == "pass"
    assert sensitivity["dataset"] == "cattaneo2"
    assert (
        sensitivity["source_sha256"]
        == "631e926eb9981828ba2e542b32c16ae08f336b9efa10621651a8a185405e0577"
    )
    assert sensitivity["nobs"] == 4_642
    assert sensitivity["score_fit"] == "full_sample_unpenalized_logit_mle"
    assert sensitivity["inference"] == "none"

    rows = sensitivity["rows"]
    expected = {
        (specification, estimand)
        for specification in (
            "no_caliper_no_support",
            "no_caliper_intersection",
            "narrow_0.1_sd",
            "default_0.2_sd",
            "wide_0.3_sd",
        )
        for estimand in ("att", "atc", "ate")
    }
    observed = {(row["specification"], row["requested_estimand"]) for row in rows}
    assert observed == expected
    assert all(row["inference"] == "none" for row in rows)
    assert all(0.0 < row["matched_focal_fraction"] <= 1.0 for row in rows)
    assert all(row["maximum_absolute_smd_after"] is not None for row in rows)
    assert all(row["mean_absolute_smd_after"] is not None for row in rows)

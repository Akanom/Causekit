"""Saved publication, real-data, and performance gates for survey RCS DiD."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "validate_did_rcs_survey_promotion.py"
OUTPUT = ROOT / "benchmarks" / "did_rcs_survey_promotion_evidence.json"


def _evidence() -> dict[str, object]:
    return json.loads(OUTPUT.read_text(encoding="utf-8"))


def test_survey_promotion_runner_freezes_design_targets_and_real_source() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '("observation", "psu")' in source
    assert "informative=informative" in source
    assert 'psu=None if sampling_unit == "observation" else "psu"' in source
    assert 'strata="stratum"' in source
    assert 'target_population="survey_population"' in source
    assert "d0e1959d1e0dcc27f30b38a38f31d31fb3959799" in source
    assert "ordinary_bootstrap" not in source


@pytest.mark.validation
def test_saved_survey_publication_certificate_passes_all_cells() -> None:
    evidence = _evidence()
    coverage = evidence["coverage_simulation"]
    assert evidence["schema_version"] == 1
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["generator_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert evidence["overall_status"] == "pass"
    assert coverage["replications_per_cell"] == 1_000
    assert coverage["total_estimator_fits"] == 8_000
    assert coverage["status"] == "pass"
    expected = {
        (unit, weights, waves)
        for unit in ("observation", "psu")
        for weights in ("noninformative", "informative")
        for waves in ("balanced", "unequal")
    }
    cells = coverage["cells"]
    assert {
        (cell["sampling_unit"], cell["weights"], cell["wave_sizes"]) for cell in cells
    } == expected
    for cell in cells:
        assert cell["replications"] == 1_000
        assert cell["successful_fits"] == 1_000
        assert cell["refusals"] == 0
        assert cell["coverage_mcse"] <= 0.009
        assert cell["coverage_gate"] == [0.9, 0.99]
        assert cell["standard_error_ratio_gate"] == [0.75, 1.25]
        assert abs(cell["bias"]) <= cell["absolute_bias_gate"]
        assert cell["status"] == "pass"


@pytest.mark.validation
def test_saved_yrbs_mapping_is_hash_pinned_and_estimand_aligned() -> None:
    real = _evidence()["real_data_sensitivity"]
    assert real["dataset"] == "yrbs_beverage_tax"
    assert real["source_commit"] == "d0e1959d1e0dcc27f30b38a38f31d31fb3959799"
    assert real["source_sha256"] == (
        "7aa7b27284d3a1108bd4d7334470b2825bd68e72470f7347e8dac0d9ea3bd58f"
    )
    assert real["n_observations"] == 60_084
    assert real["n_strata"] == 57
    assert real["estimand_mapping"] == "authors_sampling_weight_only_four_component_hajek_did"
    assert real["point_absolute_difference"] <= 1e-12
    assert real["psu_mapping"] == (
        "independent_observation_because_processed_public_file_omits_psu"
    )
    assert "not claimed" in real["interpretation"]
    assert real["status"] == "pass"


@pytest.mark.validation
def test_saved_survey_performance_gate_is_linear_memory_scale() -> None:
    performance = _evidence()["performance"]
    assert performance["n_observations"] == 100_000
    assert performance["n_psus"] == 2_500
    assert performance["n_strata"] == 20
    assert performance["elapsed_seconds"] <= performance["elapsed_gate_seconds"]
    assert performance["peak_python_mib"] <= performance["memory_gate_mib"]
    assert performance["status"] == "pass"

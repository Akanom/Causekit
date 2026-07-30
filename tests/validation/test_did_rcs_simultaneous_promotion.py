"""Publication-scale repeated-section simultaneous-inference certificate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_simultaneous_promotion.py"
OUTPUT_PATH = REPOSITORY_ROOT / "benchmarks" / "did_rcs_simultaneous_promotion_evidence.json"


def _evidence() -> dict[str, Any]:
    return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))


def test_runner_freezes_joint_observation_psu_and_cross_fitting_contracts() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert 'CONTROL_GROUPS = ("never_treated", "not_yet_treated")' in source
    assert 'ADJUSTMENTS = ("unadjusted", "covariate")' in source
    assert 'SAMPLING_UNITS = ("observation", "psu")' in source
    assert '"inference": "multiplier_bootstrap"' in source
    assert "DEFAULT_BAND_ITERATIONS = 999" in source
    assert "SeedSequence" in source
    assert "CrossFitter(" in source
    assert 'groupby("psu")["nuisance_fold"]' in source
    assert "_BinarySaturatedProbability" in source
    assert "_BinarySaturatedOutcome" in source
    assert 'fit_arguments["covariates"] = ["x"]' in source
    assert 'fit_arguments["cluster"] = "psu"' in source
    assert "ordinary_bootstrap" not in source
    assert "recursive-include benchmarks *.py *.R *.do *.txt *.csv *.json" in manifest


@pytest.mark.validation
def test_saved_joint_coverage_certificate_passes_all_16_cells_without_refusals() -> None:
    evidence = _evidence()
    certificate = evidence["joint_coverage_certificate"]

    assert evidence["schema_version"] == 1
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["generator"] == "benchmarks/validate_did_rcs_simultaneous_promotion.py"
    assert evidence["generator_sha256"] == hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
    assert evidence["overall_status"] == "pass"
    assert certificate["replications_per_cell"] == 1_000
    assert certificate["band_iterations"] == 999
    assert certificate["nominal_joint_coverage"] == pytest.approx(0.95, abs=0.0)
    assert certificate["total_estimator_fits"] == 16_000
    assert certificate["total_nuisance_fold_fits"] == 480_000
    assert certificate["total_multiplier_draws"] == 15_984_000
    assert certificate["refusal_reasons"] == {}
    assert certificate["status"] == "pass"

    expected = {
        (design, adjustment, sampling_unit, control_group)
        for design in ("favorable_balanced", "stressed_overlap_unequal")
        for adjustment in ("unadjusted", "covariate")
        for sampling_unit in ("observation", "psu")
        for control_group in ("never_treated", "not_yet_treated")
    }
    cells = certificate["cells"]
    assert len(cells) == 16
    assert {
        (
            cell["design"],
            cell["adjustment"],
            cell["sampling_unit"],
            cell["control_group"],
        )
        for cell in cells
    } == expected

    for cell in cells:
        assert cell["replications"] == 1_000
        assert cell["refusals"] == 0
        assert cell["event_times"] == [0.0, 1.0]
        assert cell["joint_coverage_gate"] == [0.90, 0.99]
        assert 0.90 <= cell["joint_coverage"] <= 0.99
        assert cell["joint_coverage_mcse"] <= 0.011
        assert cell["minimum_critical_value"] > 0.0
        assert cell["minimum_critical_value"] <= cell["mean_critical_value"]
        assert cell["mean_critical_value"] <= cell["maximum_critical_value"]
        assert cell["mean_simultaneous_half_width"] > cell["mean_pointwise_half_width"] > 0.0
        assert cell["mean_width_inflation_ratio"] > 1.0
        assert cell["minimum_realized_cell"] > 0
        assert cell["status"] == "pass"

        if cell["adjustment"] == "covariate":
            assert cell["nuisance_fold_fits_per_estimator"] == 60
            assert cell["minimum_nuisance_probability"] > 1e-6
            assert cell["maximum_nuisance_probability"] < 1.0 - 1e-6
        else:
            assert cell["nuisance_fold_fits_per_estimator"] == 0
            assert cell["minimum_nuisance_probability"] is None
            assert cell["maximum_nuisance_probability"] is None

        if cell["sampling_unit"] == "psu":
            assert cell["minimum_realized_cell_clusters"] >= 24
            assert cell["n_clusters"] in {48, 80}
            assert cell["sampling_unit_reported"] == "cluster"
        else:
            assert cell["minimum_realized_cell_clusters"] is None
            assert cell["n_clusters"] is None
            assert cell["sampling_unit_reported"] == "observation"


@pytest.mark.validation
def test_saved_certificate_retains_fixed_seed_and_indivisible_cluster_audits() -> None:
    certificate = _evidence()["joint_coverage_certificate"]
    audit = certificate["audit"]

    assert audit["seed_derivation"] == "numpy.random.SeedSequence"
    assert audit["fixed_root_seed"] == 20_260_730
    assert audit["fixed_band_seed_root"] == 73_026_020
    assert audit["all_band_metadata_matched"] is True
    assert audit["all_event_coordinates_matched"] is True
    assert audit["all_cluster_counts_matched"] is True
    assert audit["all_covariate_cluster_folds_indivisible"] is True
    assert audit["zero_refusal_gate"] is True

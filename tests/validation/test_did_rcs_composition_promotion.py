"""Promotion gates for pairwise composition-robust repeated-section DiD."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import benchmark_did_rcs_composition

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREPARE_PATH = REPOSITORY_ROOT / "benchmarks" / "prepare_did_rcs_composition_real_data.R"
REAL_RUNNER_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_composition_real_data.py"
REAL_EVIDENCE_PATH = REPOSITORY_ROOT / "benchmarks" / "did_rcs_composition_real_data_evidence.json"
PERFORMANCE_PATH = REPOSITORY_ROOT / "benchmarks" / "did_rcs_composition_performance_evidence.json"
COVERAGE_RUNNER_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_composition_promotion.py"
COVERAGE_PATH = REPOSITORY_ROOT / "benchmarks" / "did_rcs_composition_promotion_evidence.json"
REFERENCE_COMMIT = "894bd65a952c30f01a4e0005efba4cb335065eb7"
SOURCE_BLOB = "6a6a8bbe9792bc6385849421a7fd0d76692cd79e"
SOURCE_SHA256 = "37f113f1c706a3b35c325b996884c843beb9a4f773c3928d505ca51b285a7953"
ANALYSIS_CSV_SHA256 = "48e3d0cc1bd2eb757e4ac0ad24a9cc60946c6041cb4c56debaac95e4aaebb492"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_data_preparation_and_runner_pin_provenance_and_honest_folds() -> None:
    preparation = PREPARE_PATH.read_text(encoding="utf-8")
    runner = REAL_RUNNER_PATH.read_text(encoding="utf-8")
    assert REFERENCE_COMMIT in preparation
    assert SOURCE_BLOB in preparation
    assert SOURCE_SHA256 in preparation
    assert "lba_value[2783L] <- 0" in preparation
    assert "complete.cases" in preparation
    assert 'composition="robust"' in runner
    assert 'composition="stationary"' in runner
    assert "CrossFitter(" in runner
    assert "n_splits=5" in runner
    assert '"cluster": "hc_4digits"' in runner
    assert "saved nuisance" not in runner.lower()


@pytest.mark.validation
def test_hash_pinned_sequeira_sensitivity_certificate_is_complete() -> None:
    evidence = _json(REAL_EVIDENCE_PATH)
    assert evidence["schema_version"] == 1
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["generator"] == "benchmarks/validate_did_rcs_composition_real_data.py"
    assert evidence["generator_sha256"] == hashlib.sha256(REAL_RUNNER_PATH.read_bytes()).hexdigest()
    assert evidence["overall_status"] == "pass"

    source = evidence["source"]
    assert source["reference_commit"] == REFERENCE_COMMIT
    assert source["source_blob"] == SOURCE_BLOB
    assert source["source_sha256"] == SOURCE_SHA256
    assert source["analysis_csv_sha256"] == ANALYSIS_CSV_SHA256
    assert source["rows"] == 1_084
    assert source["clusters"] == 131

    results = evidence["results"]
    assert {result["outcome"] for result in results} == {
        "bp",
        "lba",
        "lba_value",
        "lba_tonnage",
    }
    assert len(results) == 4
    for result in results:
        assert result["status"] == "pass"
        assert result["n_splits"] == 5
        # Five folds times the public nuisance task graph: one probability
        # task plus three/four outcome tasks for robust/stationary scoring.
        assert result["robust"]["nuisance_fold_fits"] == 20
        assert result["stationary"]["nuisance_fold_fits"] == 25
        assert result["robust"]["minimum_probability"] > 1e-6
        assert result["robust"]["maximum_probability"] < 1.0 - 1e-6
        assert result["robust"]["maximum_weight"] > 0.0
        assert result["stationary"]["minimum_probability"] > 1e-6
        assert result["stationary"]["maximum_probability"] < 1.0 - 1e-6
        assert result["difference"] == pytest.approx(
            result["robust"]["estimate"] - result["stationary"]["estimate"],
            abs=1e-14,
        )


def test_composition_benchmark_smoke_is_finite_and_audited() -> None:
    result = benchmark_did_rcs_composition.run_benchmark(
        n_observations=4_000,
        n_splits=2,
        seed=20_260_730,
        measure_memory=False,
    )
    assert result["n_observations"] == 4_000
    assert result["n_splits"] == 2
    assert result["nuisance_task_fits"] == 8
    assert result["cell_counts"] == [1_000, 1_000, 1_000, 1_000]
    assert result["minimum_probability"] > 1e-6
    assert result["maximum_probability"] < 1.0 - 1e-6
    assert result["elapsed_seconds"] > 0.0


@pytest.mark.validation
def test_saved_composition_performance_certificate_passes() -> None:
    evidence = _json(PERFORMANCE_PATH)
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["n_observations"] == 100_000
    assert evidence["n_splits"] == 2
    assert evidence["nuisance_task_fits"] == 8
    assert evidence["elapsed_seconds"] <= evidence["elapsed_gate_seconds"]
    assert evidence["peak_python_mib"] <= evidence["peak_python_mib_gate"]
    assert evidence["status"] == "pass"


def test_coverage_runner_freezes_nonoracle_nonlinear_and_psu_contracts() -> None:
    source = COVERAGE_RUNNER_PATH.read_text(encoding="utf-8")
    assert '"stationary_unequal"' in source
    assert '"composition_shift_unequal"' in source
    assert '"observation", "psu"' in source
    assert "_SaturatedClassProbability" in source
    assert "_QuadraticOutcome" in source
    assert "CrossFitter(" in source
    assert "n_splits=2" in source
    assert "oracle" not in source.lower()
    assert "ordinary_bootstrap" not in source


@pytest.mark.validation
def test_publication_scale_composition_coverage_certificate_passes() -> None:
    evidence = _json(COVERAGE_PATH)
    simulation = evidence["coverage_simulation"]
    assert evidence["schema_version"] == 1
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["generator"] == "benchmarks/validate_did_rcs_composition_promotion.py"
    assert (
        evidence["generator_sha256"]
        == hashlib.sha256(COVERAGE_RUNNER_PATH.read_bytes()).hexdigest()
    )
    assert evidence["overall_status"] == "pass"
    assert simulation["replications_per_cell"] == 1_000
    assert simulation["total_estimator_fits"] == 8_000
    assert simulation["total_nuisance_fold_fits"] == 72_000
    assert simulation["refusal_reasons"] == {}
    assert simulation["status"] == "pass"

    cells = simulation["cells"]
    assert len(cells) == 8
    for cell in cells:
        assert cell["replications"] == 1_000
        assert cell["refusals"] == 0
        assert cell["coverage_gate"] == [0.90, 0.99]
        assert cell["standard_error_ratio_gate"] == [0.75, 1.25]
        assert cell["coverage_mcse"] <= 0.011
        assert abs(cell["bias"]) <= cell["absolute_bias_gate"]
        assert cell["minimum_probability"] > 1e-6
        assert cell["maximum_probability"] < 1.0 - 1e-6
        assert cell["status"] == "pass"

    diagnostics = simulation["target_diagnostics"]
    for sampling_unit in ("observation", "psu"):
        stationary = diagnostics[f"stationary_efficiency_{sampling_unit}"]
        assert stationary["robust_to_stationary_mean_se_ratio"] > 1.0
        shifted = diagnostics[f"composition_shift_bias_{sampling_unit}"]
        assert shifted["expected_stationary_bias_for_robust_target"] == pytest.approx(
            -11.0 / 15.0,
            abs=1e-14,
        )
        assert shifted["status"] == "pass"

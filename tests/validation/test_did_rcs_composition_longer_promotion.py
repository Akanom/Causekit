"""Promotion certificates for the longer composition-robust DiD design."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_RUNNER = REPOSITORY_ROOT / "benchmarks" / ("validate_did_rcs_composition_longer_real_data.py")
REAL_EVIDENCE = (
    REPOSITORY_ROOT / "benchmarks" / ("did_rcs_composition_longer_real_data_evidence.json")
)
PERFORMANCE_RUNNER = REPOSITORY_ROOT / "benchmarks" / ("benchmark_did_rcs_composition_longer.py")
PERFORMANCE_EVIDENCE = (
    REPOSITORY_ROOT / "benchmarks" / ("did_rcs_composition_longer_performance_evidence.json")
)
PROMOTION_RUNNER = (
    REPOSITORY_ROOT / "benchmarks" / ("validate_did_rcs_composition_longer_promotion.py")
)
PROMOTION_EVIDENCE = (
    REPOSITORY_ROOT / "benchmarks" / ("did_rcs_composition_longer_promotion_evidence.json")
)


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load validation module from {path}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


benchmark_did_rcs_composition_longer = _load_module(
    PERFORMANCE_RUNNER,
    "benchmark_did_rcs_composition_longer",
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_longer_runners_freeze_global_folds_targets_bands_and_no_selection() -> None:
    real = REAL_RUNNER.read_text(encoding="utf-8")
    promotion = PROMOTION_RUNNER.read_text(encoding="utf-8")
    assert "db9d3b7182eb8abd2e1c5190305d6edce431b57f690a87550dba27e61df72990" in real
    assert 'composition="robust"' in real
    assert 'composition="stationary"' in real
    assert "did_rcs_composition_test" in real
    assert "n_splits=N_SPLITS" in real
    assert 'groupby(data["hospital"]).nunique().eq(1)' in real
    assert '"stationary", "composition_shift"' in promotion
    assert '"observation", "psu"' in promotion
    assert "SaturatedClassProbability" in promotion
    assert "QuadraticOutcome" in promotion
    assert "bootstrap_iterations=BOOTSTRAP_ITERATIONS" in promotion
    assert '"estimator_selection": False' in promotion
    assert "oracle" not in promotion.lower()


@pytest.mark.validation
def test_hash_pinned_hospdd_longer_sensitivity_passes() -> None:
    evidence = _json(REAL_EVIDENCE)
    assert evidence["schema_version"] == 1
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["generator"] == f"benchmarks/{REAL_RUNNER.name}"
    assert evidence["generator_sha256"] == hashlib.sha256(REAL_RUNNER.read_bytes()).hexdigest()
    assert evidence["status"] == "pass"
    source = evidence["source"]
    assert source["analysis_csv_sha256"] == (
        "db9d3b7182eb8abd2e1c5190305d6edce431b57f690a87550dba27e61df72990"
    )
    assert source["rows"] == 322
    assert source["hospitals"] == 46
    assert source["periods"] == 7
    assert evidence["robust"]["nuisance_fold_fits"] == 120
    assert evidence["stationary"]["nuisance_fold_fits"] == 150
    assert evidence["robust"]["pair_count"] == 6
    assert len(evidence["robust"]["group_time"]) == 4
    assert len(evidence["robust"]["conditional_placebos"]) == 2
    assert len(evidence["robust"]["simultaneous_event_study"]) == 4
    assert evidence["diagnostic"]["estimator_selection"] is False
    assert evidence["external_longer_comparator"].startswith("unavailable")


def test_longer_performance_smoke_is_finite_and_audited() -> None:
    result = benchmark_did_rcs_composition_longer.run_benchmark(
        n_observations=3_000,
        n_splits=2,
        seed=20_260_730,
        measure_memory=False,
    )
    assert result["global_cells"] == 15
    assert result["pair_count"] == 8
    assert result["group_time_effects"] == 5
    assert result["conditional_placebos"] == 3
    assert result["nuisance_task_fits"] == 64
    assert result["minimum_probability"] > 1e-6
    assert result["maximum_probability"] < 1.0 - 1e-6


@pytest.mark.validation
def test_saved_longer_performance_certificate_passes() -> None:
    evidence = _json(PERFORMANCE_EVIDENCE)
    assert (
        evidence["generator_sha256"] == hashlib.sha256(PERFORMANCE_RUNNER.read_bytes()).hexdigest()
    )
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["n_observations"] == 120_000
    assert evidence["nuisance_task_fits"] == 64
    assert evidence["elapsed_seconds"] <= evidence["elapsed_gate_seconds"]
    assert evidence["peak_python_mib"] <= evidence["peak_python_mib_gate"]
    assert evidence["status"] == "pass"


@pytest.mark.validation
def test_publication_scale_longer_coverage_and_diagnostic_certificate_passes() -> None:
    evidence = _json(PROMOTION_EVIDENCE)
    simulation = evidence["coverage_simulation"]
    assert evidence["schema_version"] == 1
    assert evidence["generator_sha256"] == hashlib.sha256(PROMOTION_RUNNER.read_bytes()).hexdigest()
    assert evidence["overall_status"] == "pass"
    assert simulation["replications_per_cell"] == 500
    assert simulation["total_estimator_fits"] == 4_000
    assert simulation["total_nuisance_fold_fits"] == 108_000
    assert simulation["bootstrap_iterations"] == 199
    assert simulation["estimator_selection"] is False
    assert simulation["refusal_reasons"] == {}
    assert simulation["status"] == "pass"
    cells = simulation["cells"]
    assert len(cells) == 4
    for cell in cells:
        assert cell["replications"] == 500
        assert cell["refusals"] == 0
        assert max(abs(value) for value in cell["bias"]) <= cell["absolute_bias_gate"]
        assert all(0.90 <= value <= 0.99 for value in cell["pointwise_coverage"])
        assert 0.90 <= cell["simultaneous_coverage"] <= 0.99
        assert cell["simultaneous_coverage_mcse"] <= 0.012
        assert all(0.75 <= value <= 1.25 for value in cell["standard_error_ratio"])
        assert 0.015 <= cell["conditional_pretrend_rejection_rate"] <= 0.09
        assert cell["minimum_probability"] > 1e-6
        assert cell["maximum_probability"] < 1.0 - 1e-6
        if cell["design"] == "stationary":
            assert 0.015 <= cell["diagnostic_rejection_rate"] <= 0.09
        else:
            assert cell["diagnostic_rejection_rate"] >= 0.80
        assert cell["status"] == "pass"

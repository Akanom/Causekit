"""Recorded promotion, real-data, and performance evidence for direct PT-All odds."""

from __future__ import annotations

import json
from pathlib import Path

from causekit.datasets import REAL_DATASETS

ROOT = Path(__file__).resolve().parents[2]


def _evidence(filename: str) -> dict:
    return json.loads((ROOT / "benchmarks" / filename).read_text(encoding="utf-8"))


def test_direct_ratio_publication_certificate_passes_all_fixed_gates() -> None:
    evidence = _evidence("did_direct_ratio_promotion_evidence.json")

    assert evidence["schema"] == "causekit_did_direct_ratio_promotion_v1"
    assert evidence["causekit_version"] == "0.7.0a6"
    assert evidence["generated_on"] == "2026-07-30"
    assert evidence["replications_per_design"] == 1_000
    assert evidence["status"] == "pass"
    assert set(evidence["designs"]) == {
        "nonlinear_favorable",
        "nonlinear_weak_overlap",
    }
    for design in evidence["designs"].values():
        assert design["status"] == "pass"
        assert design["refusals"] == 0
        assert design["folds_identical"] is True
        assert design["maximum_direct_multiclass_paired_estimate_difference"] <= 1e-7
        assert design["maximum_direct_multiclass_paired_se_difference"] <= 1e-7
        for route in design["routes"].values():
            assert route["fits"] == 1_000
            assert route["absolute_bias"] <= 0.08
            assert 0.89 <= route["pointwise_coverage"] <= 0.99
            assert 0.89 <= route["simultaneous_coverage"] <= 0.99
            assert 0.78 <= route["empirical_sd_to_mean_se"] <= 1.22


def test_direct_ratio_real_data_sensitivity_is_hash_pinned_and_aligned() -> None:
    evidence = _evidence("did_direct_ratio_real_data_evidence.json")

    assert evidence["schema"] == "causekit_did_direct_ratio_real_data_v1"
    assert evidence["generator"] == "benchmarks/validate_did_direct_ratio_real_data.py"
    assert evidence["status"] == "pass"
    assert evidence["source_sha256"] == REAL_DATASETS["hospdd"].sha256
    assert evidence["rows"] == 322
    assert evidence["entities"] == 46
    assert evidence["folds_identical"] is True
    assert evidence["estimate_absolute_difference"] <= 2e-15
    assert evidence["standard_error_absolute_difference"] <= 2e-15
    assert evidence["candidate_influence_max_absolute_difference"] <= 2e-14
    assert evidence["conditional_weight_max_absolute_difference"] <= 2e-14
    assert evidence["direct_min_importance_effective_n"] >= 2.0
    assert evidence["direct_max_importance_share"] <= 0.8
    assert evidence["stata_parity"].startswith("unavailable")


def test_direct_ratio_fixed_period_performance_certificate_passes() -> None:
    evidence = _evidence("did_direct_ratio_performance_evidence.json")

    assert evidence["schema"] == "causekit_did_direct_ratio_performance_v1"
    assert evidence["reproduction_command"] == "python benchmarks/benchmark_did_direct_ratio.py"
    assert evidence["status"] == "pass"
    assert evidence["panel_rows"] == 200_000
    assert evidence["entities"] == 100_000
    assert evidence["periods"] == 2
    assert evidence["ordered_fitted_pairs"] == 1
    assert evidence["elapsed_seconds"] <= evidence["runtime_gate_seconds"]
    assert evidence["python_peak_mib"] <= evidence["memory_gate_mib"]
    assert evidence["absolute_bias"] <= evidence["bias_gate"]

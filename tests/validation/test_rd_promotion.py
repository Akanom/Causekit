"""Promotion gates for committed RD validation evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.validation


def _json(name: str) -> dict[str, object]:
    return json.loads((ROOT / "benchmarks" / name).read_text(encoding="utf-8"))


def _key_values(name: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in (ROOT / "benchmarks" / name).read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        rows[key] = value
    return rows


def test_official_r_fixed_bandwidth_reference_passes() -> None:
    evidence = _key_values("validate_rd_r_output.txt")

    assert evidence["contract"] == "fixed_bandwidth_triangular_p1_q2_hc1"
    assert evidence["rdrobust_version"] == "4.0.0"
    assert evidence["parity_status"] == "pass"
    assert float(evidence["sharp_bias_corrected_estimate_absolute_difference"]) <= 5e-12
    assert float(evidence["sharp_robust_standard_error_absolute_difference"]) <= 5e-12
    assert float(evidence["fuzzy_bias_corrected_estimate_absolute_difference"]) <= 5e-12
    assert float(evidence["fuzzy_robust_standard_error_absolute_difference"]) <= 5e-12


def test_hash_pinned_head_start_parity_and_native_sensitivity_are_recorded() -> None:
    evidence = _json("rd_real_data_evidence.json")

    assert evidence["status"] == "pass"
    assert evidence["source"]["sha256"] == (
        "27e18a6ec3c15aa3a53aaa96c83024ce07841e9a6f21a7d850264d46711fd70b"
    )
    fixed = evidence["fixed_contract"]
    assert fixed["parity_status"] == "pass"
    assert max(fixed["absolute_differences"].values()) <= fixed["tolerance"]
    native = evidence["native_sensitivity"]
    assert native["method"] == "native_design_conditional_mse_grid"
    assert native["n_effective"] > 0
    assert native["selected_grid_boundary"] is True
    assert native["selected_grid_boundary_sides"] == ["left"]


def test_nonlinear_fixed_and_native_coverage_certificates_pass_without_refusals() -> None:
    evidence = _json("rd_promotion_evidence.json")

    assert evidence["status"] == "pass"
    assert len(evidence["results"]) == 4
    assert sum(record["replications_requested"] for record in evidence["results"]) == 2_500
    for record in evidence["results"]:
        assert record["status"] == "pass"
        assert record["refusals"] == 0
        assert record["replications_successful"] == record["replications_requested"]
        assert record["absolute_bias"] <= (0.10 if record["bandwidth"] == "native_mse" else 0.075)
        assert 0.90 <= record["pointwise_coverage_95"] <= 0.99


def test_large_n_performance_certificate_passes() -> None:
    evidence = _json("rd_performance_evidence.json")

    assert evidence["status"] == "pass"
    assert evidence["design"]["nobs"] == 200_000
    assert evidence["result"]["elapsed_seconds"] <= evidence["gates"]["maximum_seconds"]
    assert evidence["result"]["peak_memory_mib"] <= evidence["gates"]["maximum_peak_mib"]
    assert evidence["result"]["absolute_error"] <= evidence["gates"]["maximum_absolute_error"]


def test_reviewed_stata_parity_passes_every_aligned_field() -> None:
    path = ROOT / "benchmarks" / "validate_rd_stata_output.txt"
    assert path.exists(), "The reviewed Stata RD artifact is required."
    evidence = _key_values(path.name)

    assert evidence["contract"] == "fixed_bandwidth_triangular_p1_q2_hc1"
    assert evidence["stata_version"] == "17"
    assert evidence["stata_flavor"] == "IC"
    assert evidence["rdrobust_version_line"] == "*! version 11.1.0 22may2026"
    assert evidence["parity_status"] == "pass"
    expected = {
        "sharp_conventional_estimate": 2.012628078790254,
        "sharp_bias_corrected_estimate": 2.149509125727007,
        "sharp_robust_standard_error": 0.06729310580994985,
        "fuzzy_conventional_estimate": 2.7836636780089483,
        "fuzzy_bias_corrected_estimate": 3.0987154201530087,
        "fuzzy_robust_standard_error": 0.19880345063046992,
        "fuzzy_bias_corrected_treatment_jump": 0.4360805850270855,
    }
    tolerance = float(evidence["tolerance"])
    for field, expected_value in expected.items():
        assert evidence[f"{field}_status"] == "pass"
        assert abs(float(evidence[field]) - expected_value) <= tolerance

"""Saved-evidence gates for Panel IV real data, promotion, performance, R, and Stata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.validation

ROOT = Path(__file__).resolve().parents[2]


def _key_values(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def test_panel_iv_generated_evidence_passes_frozen_gates() -> None:
    real = json.loads((ROOT / "benchmarks/panel_iv_real_data_evidence.json").read_text())
    promotion = json.loads((ROOT / "benchmarks/panel_iv_promotion_evidence.json").read_text())
    performance = json.loads((ROOT / "benchmarks/panel_iv_performance_evidence.json").read_text())
    assert real["status"] == "pass"
    assert real["absolute_differences"]["maximum_coefficient"] <= real["tolerance"]
    assert real["absolute_differences"]["maximum_covariance"] <= real["tolerance"]
    assert promotion["status"] == "pass"
    assert sum(item["refusals"] for item in promotion["results"]) == 0
    assert all(item["status"] == "pass" for item in promotion["results"])
    assert performance["status"] == "pass"


def test_official_r_panel_iv_saved_parity_passes() -> None:
    values = _key_values(ROOT / "benchmarks/validate_panel_iv_r_output.txt")
    assert values["contract"] == "wage_panel_two_way_fe_entity_clustered_cr1"
    assert values["parity_status"] == "pass"
    assert values["source_sha256"] == (
        "ee4f36706491d6348614f06bfcd5b2eef411603590a996ebb061078ae1024e78"
    )
    assert values["nobs"] == "3815"
    assert values["n_entities"] == "545"
    assert values["n_periods"] == "7"
    status_keys = [key for key in values if key.endswith("_status") and key != "parity_status"]
    assert len(status_keys) == 6
    assert all(values[key] == "pass" for key in status_keys)


def test_stata_harness_persists_all_results_before_asserting() -> None:
    script = (ROOT / "benchmarks/validate_panel_iv_stata.do").read_text(encoding="utf-8")
    assert "ivregress 2sls" in script
    assert "vce(cluster nr) small" in script
    assert "set maxvar 2000" not in script
    assert script.index("file write `results_file' \"parity_status=") < script.index(
        'if "`parity_status\'" != "pass"'
    )
    assert "expected_u_se" in script
    assert "expected_h_se" in script
    assert "expected_m_se" in script


def test_official_stata_panel_iv_saved_parity_passes_after_manual_run() -> None:
    path = ROOT / "benchmarks/validate_panel_iv_stata_output.txt"
    if not path.exists():
        pytest.skip("run benchmarks/validate_panel_iv_stata.do manually in Stata 17")
    values = _key_values(path)
    assert values["contract"] == "wage_panel_two_way_fe_entity_clustered_cr1"
    assert values["parity_status"] == "pass"
    assert values["nobs"].strip() == "3815"
    assert values["n_entities"] == "545"
    assert values["n_periods"] == "7"
    status_keys = [key for key in values if key.endswith("_status") and key != "parity_status"]
    assert len(status_keys) == 6
    assert all(values[key] == "pass" for key in status_keys)

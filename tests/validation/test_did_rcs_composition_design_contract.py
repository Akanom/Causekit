"""Design-only gates for the next composition-robust repeated-section layer."""

from __future__ import annotations

from pathlib import Path

import causekit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPOSITORY_ROOT / "docs" / "DID_RCS_COMPOSITION_DIAGNOSTIC_ALIGNMENT_CONTRACT.md"
BASE_CONTRACT_PATH = REPOSITORY_ROOT / "docs" / "DID_RCS_COMPOSITION_CHANGE_CONTRACT.md"


def test_diagnostic_and_longer_alignment_contract_freezes_required_invariants() -> None:
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    required_fragments = (
        "did_rcs_composition_test",
        "RepeatedCrossSectionCompositionDiagnostic",
        "One immutable global fold plan",
        "ClassProbabilityCrossFitTask",
        "phi_ij = (n / n_j) * R_ij * phi_ij,pair",
        "Stationary pooled cohort shares are invalid",
        "Phi_delta   = Phi_robust - Phi_stationary",
        "V_delta     = Cov(Phi_delta)",
        "never by subtracting the two marginal covariance matrices",
        "no estimator-selection recommendation",
        "Gate A — pairwise diagnostic",
        "Gate B — longer influence lattice",
        "Gate C — joint post-estimation",
        "Gate D — promotion",
    )
    for fragment in required_fragments:
        assert fragment in contract


def test_base_contract_routes_to_the_next_stage_without_conflicting_formula() -> None:
    contract = BASE_CONTRACT_PATH.read_text(encoding="utf-8")
    assert "DID_RCS_COMPOSITION_DIAGNOSTIC_ALIGNMENT_CONTRACT.md" in contract
    assert "computes covariance\nfrom the aligned difference influence" in contract
    assert "V_delta = E[psi_delta^2]" not in contract


def test_design_contract_exports_no_premature_placeholder() -> None:
    assert not hasattr(causekit, "did_rcs_composition_test")
    assert not hasattr(causekit, "RepeatedCrossSectionCompositionDiagnostic")

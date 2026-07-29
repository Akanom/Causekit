"""Optional parity against the pinned CRAN Matching pure-R reference path."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from causekit import NearestNeighborMatch


def test_stata_harness_uses_stata_supported_variance_neighbor_count() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (repository_root / "benchmarks" / "validate_matching_stata.do").read_text(
        encoding="utf-8"
    )

    assert "vce(robust, nn(1))" not in script
    assert script.count("vce(robust, nn(2))") == 3
    assert 'display as result "variance_neighbors=2"' in script
    assert "scalar expected_ate_se = sqrt(133 / 27)" in script


@pytest.mark.parametrize("estimand", ["att", "atc", "ate"])
def test_saved_stata_17_result_matches_native_fixed_score_contract(estimand: str) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output = (repository_root / "benchmarks" / "validate_matching_stata_17_output.txt").read_text(
        encoding="utf-8"
    )
    reference = dict(line.split("=", maxsplit=1) for line in output.splitlines() if "=" in line)

    logits = np.array([0.0, 4.0, 10.0, 1.0, 6.0, 9.0])
    native = NearestNeighborMatch(
        estimand=estimand,
        caliper=None,
        common_support=None,
        inference="abadie_imbens",
        variance_neighbors=int(reference["variance_neighbors"]),
    ).fit(
        [0.0, 2.0, 5.0, 3.0, 8.0, 12.0],
        treatment=[0, 0, 0, 1, 1, 1],
        propensity=1 / (1 + np.exp(-logits)),
        propensity_score_status="known",
        propensity_provenance="fixed_by_stata_parity_fixture",
    )

    tolerance = float(reference["tolerance"])
    assert reference["stata_version"] == "17"
    assert reference["parity_status"] == "pass"
    assert native.estimate == pytest.approx(float(reference[f"{estimand}_estimate"]), abs=tolerance)
    assert native.standard_error == pytest.approx(
        float(reference[f"{estimand}_standard_error"]), abs=tolerance
    )


@pytest.fixture(scope="module")
def matching_reference() -> dict[str, str]:
    reference_root = os.environ.get("CAUSEKIT_MATCHING_REFERENCE")
    rscript = shutil.which("Rscript")
    if reference_root is None or rscript is None:
        pytest.skip("set CAUSEKIT_MATCHING_REFERENCE and install Rscript for matching parity")

    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            rscript,
            str(repository_root / "benchmarks" / "validate_matching_reference.R"),
            reference_root,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(
        line.split("=", maxsplit=1) for line in completed.stdout.splitlines() if "=" in line
    )


@pytest.mark.validation
@pytest.mark.parametrize("estimand", ["att", "atc", "ate"])
def test_known_score_matching_matches_pinned_r_reference(
    estimand: str, matching_reference: dict[str, str]
) -> None:
    reference = matching_reference
    assert reference["reference_commit"] == "1208eaa7bfa888b1fc903481dddfb8c0dffa40d5"
    assert reference["matching_version"] == "4.10-15"

    logits = np.array([0.0, 4.0, 10.0, 1.0, 6.0, 9.0])
    native = NearestNeighborMatch(
        estimand=estimand,
        caliper=None,
        common_support=None,
        inference="abadie_imbens",
        variance_neighbors=1,
    ).fit(
        [0.0, 2.0, 5.0, 3.0, 8.0, 12.0],
        treatment=[0, 0, 0, 1, 1, 1],
        propensity=1 / (1 + np.exp(-logits)),
        propensity_score_status="known",
        propensity_provenance="fixed_by_reference_fixture",
    )

    assert native.estimate == pytest.approx(float(reference[f"{estimand}_estimate"]), abs=2e-14)
    assert native.standard_error == pytest.approx(
        float(reference[f"{estimand}_standard_error"]), rel=2e-14, abs=2e-14
    )

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
ATTRIBUTES = ROOT / ".gitattributes"


def _registered_evidence() -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    for path in sorted(BENCHMARKS.glob("*_evidence.json")):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if "generator" in evidence and "generator_sha256" in evidence:
            records.append((path, evidence))
    return records


@pytest.mark.validation
@pytest.mark.parametrize(("evidence_path", "evidence"), _registered_evidence())
def test_promotion_generator_hash_is_platform_independent(
    evidence_path: Path, evidence: dict
) -> None:
    generator = ROOT / evidence["generator"]
    source = generator.read_bytes()

    assert generator.suffix == ".py", evidence_path.name
    assert b"\r" not in source, f"{generator} must be checked out with canonical LF endings"
    assert hashlib.sha256(source).hexdigest() == evidence["generator_sha256"]


def test_hashed_benchmark_sources_have_explicit_lf_checkout_rules() -> None:
    attributes = ATTRIBUTES.read_text(encoding="utf-8")

    assert "benchmarks/*.py text eol=lf" in attributes
    assert "benchmarks/*.R text eol=lf" in attributes
    assert "benchmarks/*.do text eol=lf" in attributes


def test_parity_register_covers_every_public_estimator_family_without_pending_rows() -> None:
    register = (ROOT / "docs" / "PARITY.md").read_text(encoding="utf-8")
    matrix = register.split("## Current matrix", maxsplit=1)[1].split(
        "The RD parity fixture", maxsplit=1
    )[0]

    for family in (
        "`IV2SLS`",
        "`PanelIV2SLS`",
        "`RandomizedATE`",
        "`IPWATE` / `AIPWATE`",
        "`CrossFitter`",
        "`NearestNeighborMatch`",
        "Conventional staggered DiD",
        "Repeated-cross-section DiD",
        "Efficient DiD",
        "`PartiallyLinearDML`",
        "`RLearner`",
        "`DRLearner`",
        "Sharp/fuzzy fixed-bandwidth RD",
    ):
        assert family in matrix
    assert "| pending |" not in matrix.lower()
